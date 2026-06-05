"""
PostgreSQLBrot - PostgreSQL Mandelbrot Set Computation in Plain SQL

This implementation computes the Mandelbrot set with PostgreSQL recursive CTEs.
Both the benchmark path and each live frame execute one full-image SQL request.
Live mode writes the full query result into an RGB frame buffer and displays it
in a navigation-driven window sized from the requested width and height.

Configure the connection with POSTGRES_DSN or normal libpq PG* environment
variables, for example:

    POSTGRES_DSN=postgresql://user:password@localhost:5432/postgres

Author: Thomas Zeutschler
License: MIT
GitHub: https://github.com/Zeutschler/sql-mandelbrot-benchmark
"""

from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from utils import colorize_mandelbrot, format_rps, save_mandelbrot_image


POSTGRES_MANDELBROT_QUERY = """
WITH RECURSIVE
  pixels AS (
    SELECT
      x::integer AS x,
      y::integer AS y,
      (
        %(x_min)s::double precision
        + (x::double precision * %(x_span)s::double precision / %(width_last)s::double precision)
      )::double precision AS cx,
      (
        %(y_min)s::double precision
        + (y::double precision * %(y_span)s::double precision / %(height_last)s::double precision)
      )::double precision AS cy
    FROM
      generate_series(0, %(width_last)s::integer) AS gx(x)
      CROSS JOIN generate_series(%(y_start)s::integer, %(y_end)s::integer) AS gy(y)
  ),
  mandelbrot_iterations(x, y, cx, cy, zx, zy, iteration) AS (
    SELECT
      x,
      y,
      cx,
      cy,
      0.0::double precision AS zx,
      0.0::double precision AS zy,
      0::integer AS iteration
    FROM pixels

    UNION ALL

    SELECT
      m.x,
      m.y,
      m.cx,
      m.cy,
      (m.zx * m.zx - m.zy * m.zy + m.cx)::double precision AS zx,
      (2.0::double precision * m.zx * m.zy + m.cy)::double precision AS zy,
      m.iteration + 1 AS iteration
    FROM mandelbrot_iterations AS m
    WHERE
      m.iteration < %(max_iterations)s::integer
      AND (m.zx * m.zx + m.zy * m.zy) <= 4.0::double precision
  )
SELECT
  x,
  y,
  MAX(iteration)::integer AS depth
FROM mandelbrot_iterations
GROUP BY x, y
ORDER BY y, x;
"""


@dataclass
class LiveStats:
    """Runtime counters for the live PostgreSQL renderer."""

    sql_requests: int = 0
    buffer_renders: int = 0
    full_frames: int = 0
    elapsed_seconds: float = 0.0
    last_sql_ms: float = 0.0
    total_sql_ms: float = 0.0
    last_render_ms: float = 0.0
    total_render_ms: float = 0.0

    @property
    def sql_rps(self):
        if self.last_sql_ms <= 0:
            return 0.0
        return 1000.0 / self.last_sql_ms

    @property
    def avg_sql_rps(self):
        if self.total_sql_ms <= 0:
            return 0.0
        return self.sql_requests * 1000.0 / self.total_sql_ms

    @property
    def render_rps(self):
        if self.last_render_ms <= 0:
            return 0.0
        return 1000.0 / self.last_render_ms

    @property
    def avg_render_rps(self):
        if self.total_render_ms <= 0:
            return 0.0
        return self.buffer_renders * 1000.0 / self.total_render_ms


@dataclass(frozen=True)
class Viewport:
    """Complex-plane bounds for one Mandelbrot render."""

    x_min: float = -2.5
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0

    @property
    def x_span(self):
        return self.x_max - self.x_min

    @property
    def y_span(self):
        return self.y_max - self.y_min

    @property
    def center_x(self):
        return self.x_min + self.x_span / 2.0

    @property
    def center_y(self):
        return self.y_min + self.y_span / 2.0

    @property
    def zoom_level(self):
        return 3.5 / self.x_span

    def shifted(self, dx, dy):
        return Viewport(
            self.x_min + dx,
            self.x_max + dx,
            self.y_min + dy,
            self.y_max + dy,
        )

    def zoomed(self, factor, anchor_x=None, anchor_y=None):
        if factor <= 0:
            raise ValueError("zoom factor must be greater than 0")

        anchor_x = self.center_x if anchor_x is None else anchor_x
        anchor_y = self.center_y if anchor_y is None else anchor_y

        return Viewport(
            anchor_x - (anchor_x - self.x_min) * factor,
            anchor_x + (self.x_max - anchor_x) * factor,
            anchor_y - (anchor_y - self.y_min) * factor,
            anchor_y + (self.y_max - anchor_y) * factor,
        )


def _load_psycopg():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL benchmark requires psycopg. Install it with "
            '`pip install "psycopg[binary]"` or install the project dependencies.'
        ) from exc

    return psycopg


def _connect(dsn=None):
    psycopg = _load_psycopg()
    conninfo = dsn or os.environ.get("POSTGRES_DSN", "")

    try:
        return psycopg.connect(
            conninfo,
            autocommit=True,
            application_name="sql-mandelbrot-benchmark",
        )
    except psycopg.OperationalError as exc:
        raise RuntimeError(
            "Could not connect to PostgreSQL. Set POSTGRES_DSN or libpq PG* "
            "environment variables such as PGHOST, PGDATABASE, PGUSER, and "
            "PGPASSWORD."
        ) from exc


def _validate_dimensions(width, height, max_iterations):
    if width < 2:
        raise ValueError("width must be at least 2")
    if height < 2:
        raise ValueError("height must be at least 2")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")


def _fetch_mandelbrot_rows(
    conn,
    width,
    height,
    max_iterations,
    y_start,
    y_end,
    viewport=None,
):
    viewport = viewport or Viewport()
    params = {
        "width_last": width - 1,
        "height_last": height - 1,
        "y_start": y_start,
        "y_end": y_end,
        "max_iterations": max_iterations,
        "x_min": viewport.x_min,
        "x_span": viewport.x_span,
        "y_min": viewport.y_min,
        "y_span": viewport.y_span,
    }

    with conn.cursor() as cursor:
        cursor.execute(POSTGRES_MANDELBROT_QUERY, params)
        return cursor.fetchall()


def _write_rows(mandelbrot, rows):
    for x, y, depth in rows:
        mandelbrot[y, x] = depth


def _build_color_lookup(max_iterations):
    from matplotlib import colormaps

    values = np.arange(max_iterations + 1, dtype=np.float64)
    scaled = np.zeros(max_iterations + 1, dtype=np.float64)

    escaped = values < max_iterations
    log_scaled = np.log(values[escaped] + 1.0)
    if log_scaled.size > 0 and log_scaled.max() > 0:
        scaled[escaped] = log_scaled / log_scaled.max()

    scaled[max_iterations] = 0.0
    colored = colormaps.get_cmap("hot")(scaled)
    return (colored[:, :3] * 255).astype(np.uint8)


def _write_rows_to_buffers(mandelbrot, rgb_buffer, rows, color_lookup):
    if not rows:
        return

    row_data = np.asarray(rows, dtype=np.int64)
    xs = row_data[:, 0]
    ys = row_data[:, 1]
    depths = row_data[:, 2].astype(np.uint16)
    mandelbrot[ys, xs] = depths
    rgb_buffer[ys, xs] = color_lookup[depths]


def run_postgresqlbrot(width, height, max_iterations, dsn=None):
    """
    Compute Mandelbrot set using PostgreSQL SQL with recursive CTEs.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        max_iterations: Maximum iterations per pixel
        dsn: Optional PostgreSQL connection string. Defaults to POSTGRES_DSN
            or normal libpq PG* environment variables.

    Returns:
        2D NumPy array of iteration counts
    """
    _validate_dimensions(width, height, max_iterations)

    mandelbrot = np.zeros((height, width), dtype=np.uint16)
    with _connect(dsn) as conn:
        rows = _fetch_mandelbrot_rows(
            conn,
            width,
            height,
            max_iterations,
            y_start=0,
            y_end=height - 1,
        )
        _write_rows(mandelbrot, rows)

    return mandelbrot


def _default_live_viewport(width, height):
    x_min = -2.5
    x_max = 1.0
    x_span = x_max - x_min
    y_span = x_span * height / width
    return Viewport(x_min, x_max, -y_span / 2.0, y_span / 2.0)


def _viewport_extent(viewport):
    return (viewport.x_min, viewport.x_max, viewport.y_min, viewport.y_max)


def _live_figure_layout(width, height):
    dpi = 100
    status_height = 164
    total_height = height + status_height
    image_bottom = status_height / total_height
    image_height = height / total_height
    figure_size = (width / dpi, total_height / dpi)
    image_rect = (0.0, image_bottom, 1.0, image_height)
    status_rect = (0.0, 0.0, 1.0, image_bottom)
    return dpi, figure_size, image_rect, status_rect, total_height


def _navigation_button_layout(width, total_height):
    compact = width < 400
    rows = [
        [
            ("Left" if not compact else "L", "left"),
            ("Right" if not compact else "R", "right"),
            ("Up", "up"),
            ("Down" if not compact else "Dn", "down"),
            ("Reset" if not compact else "Rst", "reset"),
            ("Stop", "stop"),
        ],
        [
            ("Zoom -" if not compact else "Z-", "zoom_out"),
            ("Zoom +" if not compact else "Z+", "zoom_in"),
            ("Iter -" if not compact else "I-", "iter_down"),
            ("Iter +" if not compact else "I+", "iter_up"),
        ],
    ]

    margin_px = 8
    gap_px = 5
    button_height_px = 30
    row_gap_px = 7

    rects = []
    for row_index, buttons in enumerate(rows):
        bottom_px = 8 + row_index * (button_height_px + row_gap_px)
        button_width_px = (
            width - 2 * margin_px - (len(buttons) - 1) * gap_px
        ) / len(buttons)

        for index, (label, action) in enumerate(buttons):
            left_px = margin_px + index * (button_width_px + gap_px)
            rects.append(
                (
                    label,
                    action,
                    (
                        left_px / width,
                        bottom_px / total_height,
                        button_width_px / width,
                        button_height_px / total_height,
                    ),
                )
            )

    return rects


def _disable_default_keymaps(plt):
    for keymap in (
        "keymap.back",
        "keymap.forward",
        "keymap.fullscreen",
        "keymap.grid",
        "keymap.home",
        "keymap.pan",
        "keymap.quit",
        "keymap.quit_all",
        "keymap.save",
        "keymap.xscale",
        "keymap.yscale",
        "keymap.zoom",
    ):
        if keymap in plt.rcParams:
            plt.rcParams[keymap] = []


def _focus_figure_window(figure):
    manager = figure.canvas.manager
    window = getattr(manager, "window", None)

    for method_name in ("activateWindow", "raise_", "focus_force"):
        method = getattr(window, method_name, None)
        if method is not None:
            try:
                method()
            except Exception:
                pass

    set_focus = getattr(figure.canvas, "setFocus", None)
    if set_focus is not None:
        try:
            set_focus()
        except Exception:
            pass


@dataclass
class RenderResult:
    generation: int
    viewport: Viewport
    max_iterations: int
    rows: list | None = None
    started_at: float = 0.0
    elapsed_ms: float = 0.0
    error: Exception | None = None


def run_live_postgresqlbrot(
    width,
    height,
    max_iterations,
    *,
    duration=None,
    dsn=None,
    fullscreen=False,
    continuous=True,
):
    """
    Compute and display PostgreSQL Mandelbrot screens in a live Matplotlib window.

    One SQL request computes the whole current screen. SQL RPS and rendering RPS
    both count accepted whole-frame query results, so they should stay equal
    unless a stale or cancelled query is intentionally ignored during navigation.

    Returns:
        Tuple of (2D NumPy array of iteration counts, LiveStats)
    """
    _validate_dimensions(width, height, max_iterations)

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    _disable_default_keymaps(plt)

    mandelbrot = np.zeros((height, width), dtype=np.uint16)
    rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    color_lookup_cache = {}
    stats = LiveStats()

    plt.ion()
    dpi, figure_size, image_rect, status_rect, total_height = _live_figure_layout(
        width, height
    )
    figure = plt.figure(figsize=figure_size, dpi=dpi, facecolor="black")
    image_axes = figure.add_axes(image_rect, facecolor="black")
    status_axes = figure.add_axes(status_rect, facecolor="#101010")

    current_viewport = _default_live_viewport(width, height)
    image = image_axes.imshow(
        rgb_buffer,
        interpolation="nearest",
        origin="lower",
        extent=_viewport_extent(current_viewport),
        aspect="auto",
    )
    image_axes.set_axis_off()
    status_axes.set_axis_off()

    primary_status = status_axes.text(
        0.012,
        0.82,
        "",
        color="white",
        family="monospace",
        fontsize=10 if width >= 700 else 8,
        ha="left",
        va="center",
        clip_on=False,
        transform=status_axes.transAxes,
    )
    detail_status = status_axes.text(
        0.012,
        0.60,
        "",
        color="#d0d0d0",
        family="monospace",
        fontsize=9 if width >= 700 else 7,
        ha="left",
        va="center",
        clip_on=False,
        transform=status_axes.transAxes,
    )

    figure.show()
    if fullscreen and figure.canvas.manager is not None:
        try:
            figure.canvas.manager.full_screen_toggle()
        except Exception:
            pass
    _focus_figure_window(figure)

    start_time = time.perf_counter()
    last_terminal_update = 0.0
    latest_generation = 0
    latest_query_ms = None
    current_iterations = max_iterations
    render_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue()
    stop_event = threading.Event()
    query_lock = threading.Lock()
    active_conn = None
    fatal_error = None

    def color_lookup_for(iterations):
        if iterations not in color_lookup_cache:
            color_lookup_cache[iterations] = _build_color_lookup(iterations)
        return color_lookup_cache[iterations]

    def update_elapsed():
        stats.elapsed_seconds = time.perf_counter() - start_time

    def status_lines(phase, viewport):
        query_part = "query --"
        if latest_query_ms is not None:
            query_part = f"query {latest_query_ms:.2f} ms"

        primary = (
            f"{phase} | SQL RPS {stats.sql_rps:.2f} | "
            f"render RPS {stats.render_rps:.2f} | "
            f"frames {stats.full_frames} | iter {current_iterations} | {query_part}"
        )
        detail = (
            f"center ({viewport.center_x:.8g}, {viewport.center_y:.8g}) | "
            f"zoom {viewport.zoom_level:.3g}x | "
            f"span {viewport.x_span:.8g} x {viewport.y_span:.8g}"
        )
        return primary, detail

    def set_status(phase, viewport, *, force_terminal=False):
        nonlocal last_terminal_update

        update_elapsed()
        primary, detail = status_lines(phase, viewport)
        primary_status.set_text(primary)
        detail_status.set_text(detail)
        figure.canvas.draw_idle()

        now = time.perf_counter()
        if force_terminal or now - last_terminal_update >= 0.5:
            print(f"\r{primary} | {detail}", end="", flush=True)
            last_terminal_update = now

    def cancel_active_query():
        with query_lock:
            conn = active_conn
        if conn is None:
            return

        cancel = getattr(conn, "cancel_safe", None) or getattr(conn, "cancel", None)
        if cancel is None:
            return

        try:
            cancel()
        except Exception:
            pass

    def queue_render(viewport, *, cancel_active=False):
        nonlocal current_viewport, latest_generation

        current_viewport = viewport
        latest_generation += 1
        render_iterations = current_iterations
        apply_viewport(viewport)
        figure.canvas.draw_idle()

        while True:
            try:
                render_queue.get_nowait()
            except queue.Empty:
                break

        try:
            render_queue.put_nowait((latest_generation, viewport, render_iterations))
        except queue.Full:
            try:
                render_queue.get_nowait()
            except queue.Empty:
                pass
            render_queue.put_nowait((latest_generation, viewport, render_iterations))
        if cancel_active:
            cancel_active_query()
        set_status("rendering", viewport, force_terminal=True)

    def worker_loop():
        nonlocal active_conn

        try:
            with _connect(dsn) as conn:
                while not stop_event.is_set():
                    try:
                        generation, viewport, render_iterations = render_queue.get(
                            timeout=0.1
                        )
                    except queue.Empty:
                        continue

                    with query_lock:
                        active_conn = conn

                    started = time.perf_counter()
                    try:
                        rows = _fetch_mandelbrot_rows(
                            conn,
                            width,
                            height,
                            render_iterations,
                            0,
                            height - 1,
                            viewport,
                        )
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        result_queue.put(
                            RenderResult(
                                generation,
                                viewport,
                                render_iterations,
                                rows,
                                started,
                                elapsed_ms,
                            )
                        )
                    except Exception as exc:
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        result_queue.put(
                            RenderResult(
                                generation,
                                viewport,
                                render_iterations,
                                started_at=started,
                                elapsed_ms=elapsed_ms,
                                error=exc,
                            )
                        )
                    finally:
                        with query_lock:
                            active_conn = None
        except Exception as exc:
            result_queue.put(
                RenderResult(
                    0,
                    current_viewport,
                    current_iterations,
                    started_at=time.perf_counter(),
                    error=exc,
                )
            )

    def apply_viewport(viewport):
        image.set_extent(_viewport_extent(viewport))
        image_axes.set_xlim(viewport.x_min, viewport.x_max)
        image_axes.set_ylim(viewport.y_min, viewport.y_max)

    def accept_result(result):
        nonlocal latest_query_ms

        latest_query_ms = result.elapsed_ms
        color_lookup = color_lookup_for(result.max_iterations)
        _write_rows_to_buffers(mandelbrot, rgb_buffer, result.rows, color_lookup)
        apply_viewport(result.viewport)
        image.set_data(rgb_buffer)
        stats.sql_requests += 1
        stats.buffer_renders += 1
        stats.full_frames += 1
        stats.last_sql_ms = result.elapsed_ms
        stats.total_sql_ms += result.elapsed_ms
        stats.last_render_ms = (time.perf_counter() - result.started_at) * 1000
        stats.total_render_ms += stats.last_render_ms
        set_status("ready", result.viewport, force_terminal=True)

    def request_viewport(viewport):
        queue_render(viewport, cancel_active=True)

    def request_iterations(iterations):
        nonlocal current_iterations

        iterations = max(8, min(65535, int(iterations)))
        if iterations == current_iterations:
            return

        current_iterations = iterations
        request_viewport(current_viewport)

    def navigate(action):
        pan_fraction = 0.18

        if action == "stop":
            stop_event.set()
            plt.close(figure)
            return
        if action == "reset":
            request_viewport(_default_live_viewport(width, height))
            return
        if action == "zoom_in":
            request_viewport(current_viewport.zoomed(0.78))
            return
        if action == "zoom_out":
            request_viewport(current_viewport.zoomed(1.0 / 0.78))
            return
        if action == "iter_up":
            request_iterations(current_iterations * 2)
            return
        if action == "iter_down":
            request_iterations(max(8, current_iterations // 2))
            return
        if action == "left":
            request_viewport(
                current_viewport.shifted(-current_viewport.x_span * pan_fraction, 0)
            )
            return
        if action == "right":
            request_viewport(
                current_viewport.shifted(current_viewport.x_span * pan_fraction, 0)
            )
            return
        if action == "up":
            request_viewport(
                current_viewport.shifted(0, current_viewport.y_span * pan_fraction)
            )
            return
        if action == "down":
            request_viewport(
                current_viewport.shifted(0, -current_viewport.y_span * pan_fraction)
            )

    def event_anchor(event):
        if event.inaxes is not image_axes or event.xdata is None or event.ydata is None:
            return current_viewport.center_x, current_viewport.center_y
        return event.xdata, event.ydata

    def on_scroll(event):
        if event.inaxes is not image_axes:
            return

        step = getattr(event, "step", None)
        if step is None:
            step = 1 if event.button == "up" else -1
        factor = 0.78**step
        anchor_x, anchor_y = event_anchor(event)
        request_viewport(current_viewport.zoomed(factor, anchor_x, anchor_y))

    def normalized_key(raw_key):
        key = (raw_key or "").lower()
        if key == "+" or key.endswith("++"):
            return "+"
        if "+" in key:
            key = key.rsplit("+", 1)[-1] or "+"
        return key

    def is_zoom_in_key(raw_key, key):
        raw_key = (raw_key or "").lower()
        return key in {"+", "=", "i", "add"} or raw_key in {
            "shift+=",
            "shift++",
            "ctrl++",
            "ctrl+=",
            "cmd++",
            "cmd+=",
        }

    def is_zoom_out_key(raw_key, key):
        raw_key = (raw_key or "").lower()
        return key in {"-", "_", "o", "subtract"} or raw_key in {
            "shift+-",
            "ctrl+-",
            "cmd+-",
        }

    def on_key(event):
        raw_key = event.key or ""
        key = normalized_key(raw_key)

        if key in {"escape", "q"}:
            navigate("stop")
            return
        if key in {"r", "home"}:
            navigate("reset")
            return
        if is_zoom_in_key(raw_key, key):
            navigate("zoom_in")
            return
        if is_zoom_out_key(raw_key, key):
            navigate("zoom_out")
            return
        if key in {"]", "pageup"}:
            navigate("iter_up")
            return
        if key in {"[", "pagedown"}:
            navigate("iter_down")
            return
        if key in {"left", "a"}:
            navigate("left")
            return
        if key in {"right", "d"}:
            navigate("right")
            return
        if key in {"up", "w"}:
            navigate("up")
            return
        if key in {"down", "s"}:
            navigate("down")

    drag_state = {"start_x": None, "start_y": None, "viewport": None}

    def on_button_press(event):
        if event.inaxes is not image_axes or event.button != 1:
            return
        drag_state["start_x"] = event.xdata
        drag_state["start_y"] = event.ydata
        drag_state["viewport"] = current_viewport

    def on_button_release(event):
        if (
            event.inaxes is not image_axes
            or event.button != 1
            or drag_state["viewport"] is None
            or drag_state["start_x"] is None
            or event.xdata is None
            or event.ydata is None
        ):
            drag_state["viewport"] = None
            return

        dx = drag_state["start_x"] - event.xdata
        dy = drag_state["start_y"] - event.ydata
        request_viewport(drag_state["viewport"].shifted(dx, dy))
        drag_state["viewport"] = None

    def on_close(_event):
        stop_event.set()
        cancel_active_query()

    navigation_buttons = []

    def make_button_callback(action):
        def callback(_event):
            navigate(action)
            _focus_figure_window(figure)

        return callback

    for label, action, rect in _navigation_button_layout(width, total_height):
        button_axes = figure.add_axes(rect)
        button = Button(
            button_axes,
            label,
            color="#202020",
            hovercolor="#303030",
        )
        button.label.set_color("white")
        button.label.set_fontsize(8 if width < 560 else 9)
        button.on_clicked(make_button_callback(action))
        navigation_buttons.append(button)

    figure.canvas.mpl_connect("scroll_event", on_scroll)
    figure.canvas.mpl_connect("key_press_event", on_key)
    figure.canvas.mpl_connect("button_press_event", on_button_press)
    figure.canvas.mpl_connect("button_release_event", on_button_release)
    figure.canvas.mpl_connect("close_event", on_close)

    worker = threading.Thread(target=worker_loop, daemon=True)
    worker.start()
    queue_render(current_viewport)

    while plt.fignum_exists(figure.number) and not stop_event.is_set():
        update_elapsed()

        while True:
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                break

            if result.error is not None:
                if result.generation == latest_generation or result.generation == 0:
                    fatal_error = result.error
                    set_status(
                        f"failed: {result.error}",
                        result.viewport,
                        force_terminal=True,
                    )
                    stop_event.set()
                continue

            if result.generation != latest_generation:
                continue

            accept_result(result)
            if continuous and not stop_event.is_set():
                queue_render(current_viewport)

        if duration is not None and stats.elapsed_seconds >= duration:
            stop_event.set()
            cancel_active_query()
            break

        figure.canvas.flush_events()
        plt.pause(0.02)

    print()
    stop_event.set()
    cancel_active_query()
    worker.join(timeout=1.0)
    if fatal_error is not None:
        raise fatal_error
    return mandelbrot, stats


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the PostgreSQL Mandelbrot benchmark."
    )
    parser.add_argument("--width", type=int, default=1400, help="image width")
    parser.add_argument("--height", type=int, default=800, help="image height")
    parser.add_argument(
        "--iterations",
        type=int,
        default=256,
        help="maximum Mandelbrot iterations per pixel",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL connection string; defaults to POSTGRES_DSN or PG* env vars",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="show a live windowed render with SQL/render RPS output",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="seconds to run --live mode; omit to run until the window is closed",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="request fullscreen mode in --live mode",
    )
    parser.add_argument(
        "--single-render",
        action="store_true",
        help="render the current live viewport once instead of continuously",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="do not save images/postgresqlbrot.png after completion",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    print(
        "Computing PostgreSQL Mandelbrot set "
        f"({args.width}x{args.height}, max {args.iterations} iterations)..."
    )

    try:
        if args.live:
            result, stats = run_live_postgresqlbrot(
                args.width,
                args.height,
                args.iterations,
                duration=args.duration,
                dsn=args.dsn,
                fullscreen=args.fullscreen,
                continuous=not args.single_render,
            )
            print(
                "Live run completed: "
                f"{stats.full_frames} full render(s), "
                f"{stats.sql_requests} SQL request(s), "
                f"{stats.avg_sql_rps:.2f} avg SQL RPS, "
                f"{stats.avg_render_rps:.2f} avg render RPS"
            )
        else:
            start_time = time.perf_counter()
            result = run_postgresqlbrot(
                args.width,
                args.height,
                args.iterations,
                dsn=args.dsn,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"Completed in {elapsed_ms:.2f} ms ({format_rps(elapsed_ms)} RPS)")

        if not args.no_save:
            save_mandelbrot_image(result, args.iterations, "postgresqlbrot.png")

    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    except Exception as exc:
        print(f"Failed to run PostgreSQL benchmark: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
