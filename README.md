# sql-mandelbrot-benchmark
**Because why benchmark sql engines with boring aggregates when you can generate fractals?**

This project uses recursive Common Table Expressions (CTE) to calculate the Mandelbrot set entirely 
in SQL — no loops, no procedural code, just pure SQL. It serves as a fun and visually appealing benchmark 
for testing recursive query performance, floating-point precision, and computational capabilities of SQL engines.

![Mandelbrot Set](images/duckbrot.png)

## What is This?

A benchmark suite that:
- Computes the famous [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set) using SQL recursive CTEs
- Tests multiple SQL engines, including DuckDB, SQLite, PostgreSQL, and Python implementations for reference.
- Generates beautiful fractal images as proof of correct computation
- Prints RPS (full renders per second) alongside elapsed times
- Reveals which database / SQL engine renders infinity fastest

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/duckbrot.git
cd duckbrot

# Install dependencies
pip install -r requirements.txt

# Run the benchmark suite
python main.py
```

PostgreSQL is optional at runtime, but it needs a reachable database. Configure it
with `POSTGRES_DSN` or normal libpq `PG*` variables:

```bash
export POSTGRES_DSN="postgresql://user:password@localhost:5432/postgres"
python postgresqlbrot.py --width 400 --height 240 --iterations 128
```

For live visual rendering with SQL RPS output:

```bash
python postgresqlbrot.py --live --width 1400 --height 800 --iterations 128
```

In `main.py`, RPS means completed full-image renders per second. In
`postgresqlbrot.py --live`, one SQL query renders the whole current screen into
an RGB frame buffer. Live SQL RPS is calculated from the last accepted query
execution time (`1000 / query_ms`), and render RPS is calculated from the last
accepted query-to-buffer-swap time. Use the on-screen buttons, mouse wheel, or `+`/`-` to zoom; use the
on-screen buttons, arrow keys, or `WASD` to pan; use `Iter +` / `Iter -` or
`]` / `[` to change the SQL iteration depth; use `R` to reset and `Q` or `Esc`
to exit. Add `--fullscreen` if you want the live window to request fullscreen
mode.

If the live image looks coarse around the boundary, increase the iteration
depth. PostgreSQL uses `double precision` for the complex-plane math; low
iteration counts usually cause the visible loss of detail before floating-point
precision does.

## Current Benchmark Results

Current results on 1400x800 pixels, 256 max iterations, Macbook Pro M4 Max:

| 🏆 | Engine/Implementation                      | Time (ms) | Relative Performance |
|----|--------------------------------------------|-----------|----------------------|
| *  | Mac Metal GPU (unfair, but the true limit) | 0.77 ms   | ∞ 😵                 |
| 1  | NumPy (vectorized, unrolled)               | 665 ms    | **0.83x** ⭐          |
| 2  | ArrowDatafusion (SQL)                      | 797 ms    | 1.00x (baseline)     |
| 3  | DuckDB (SQL)                               | 1,364 ms  | 1.71x slower         |
| 4  | FasterPybrot                               | 2,850 ms  | 3.58x slower         |
| 5  | FastPybrot                                 | 3,327 ms  | 4.17x slower         |
| 6  | Pure Python                                | 4,328 ms  | 5.43x slower         |
| 7  | SQLite (SQL)                               | 44,918 ms | 56.36x slower        |
| -  | PostgreSQL (SQL)                           | configure locally | optional |

**Winner overall: NumPy** - Just 17% faster than ArrowDatafusion using loop unrolling and vectorized operations!

**Winner SQL: ArrowDatafusion** - Incredibly fast, nearly matching optimized NumPy performance!

## How It Works

The Mandelbrot set is computed by iterating the formula `z = z² + c` for each pixel in the complex plane:

```sql
WITH RECURSIVE
  -- Generate pixel grid and map to complex plane
  pixels AS (
    SELECT
      x, y,
      -2.5 + (x * 3.5 / width) AS cx,
      -1.0 + (y * 2.0 / height) AS cy
    FROM generate_series(0, width-1) AS x,
         generate_series(0, height-1) AS y
  ),
  -- Recursively iterate z = z² + c
  mandelbrot_iterations AS (
    SELECT x, y, cx, cy, 0.0 AS zx, 0.0 AS zy, 0 AS iteration
    FROM pixels

    UNION ALL

    SELECT
      x, y, cx, cy,
      zx * zx - zy * zy + cx AS zx,
      2.0 * zx * zy + cy AS zy,
      iteration + 1
    FROM mandelbrot_iterations
    WHERE iteration < max_iterations
      AND (zx * zx + zy * zy) <= 4.0
  )
SELECT x, y, MAX(iteration) AS depth
FROM mandelbrot_iterations
GROUP BY x, y;
```

The iteration count determines the color of each pixel, creating the iconic fractal pattern.

## Adding New Benchmarks

Want to test MySQL, MariaDB, Oracle, SQL Server, or another SQL engine? Just:

1. Create a new file (e.g., `mysqlbrot.py`)
2. Implement a `run_mysqlbrot(width, height, max_iterations)` function (the DuckDB implementation is a good starting point)
3. Add one line to `main.py`:
   ```python
   BENCHMARKS = [
       ("DuckDB (SQL)", "duckbrot", "run_duckbrot"),
       ("Pure Python", "pybrot", "run_pybrot"),
       ..., 
       ("MySQL", "mysqlbrot", "run_mysqlbrot"),  # New!
   ]
   ```

The framework handles everything else automatically!

## Configuration

Adjust the benchmark parameters in `main.py`:

```python
WIDTH = 1400           # Image width in pixels
HEIGHT = 800           # Image height in pixels
MAX_ITERATIONS = 256   # Maximum recursion depth
```

Higher values = more detail, longer computation time.

## Known Engine Compatibility

### ✅ Works Great
- **NumPy** - Highly optimized with loop unrolling and vectorized operations (fastest!)
- **DuckDB** - Excellent performance, proper DOUBLE precision
- **Pure Python** - Reference implementation, just to have an idea how fast the database engines are
- **SQLite** - Works but significantly slower due to recursive CTE overhead

### Supported With Setup
- **PostgreSQL** - Uses recursive CTEs and `generate_series`; requires a configured PostgreSQL server

### Should Work (untested, please contribute 🤙)
- others 

### Known Issues
- Some engines might struggle with support for DOUBLE precision and may use DECIMAL (not good for fractals, and lead to pixelated results)
- Watch out for type inference - explicit `::DOUBLE` casts are critical!

## What This Tests

This benchmark evaluates:
1. **Recursive CTE Performance** - How efficiently engines handle deep recursion
2. **Floating-Point Precision** - DOUBLE vs DECIMAL arithmetic accuracy
3. **Query Optimization** - How well engines optimize complex recursive queries
4. **Scalability** - Performance with increasing iterations and resolution

## Contributing

Contributions very welcome! Especially:
- New SQL engine implementations (PostgreSQL, MySQL, SQLite, etc.)
- Performance optimizations
- Better visualization options
- Benchmark result submissions

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Credits

Created by Thomas Zeutschler, Ulrich Ludmann, and Jakub Jirak (the grand master of GPU fractals)

Inspired by the mathematical beauty of the Mandelbrot set and the curiosity about SQL engine performance.

## Learn More

- [Mandelbrot Set (Wikipedia)](https://en.wikipedia.org/wiki/Mandelbrot_set)
- [SQL Recursive CTEs](https://en.wikipedia.org/wiki/Hierarchical_and_recursive_queries_in_SQL)
- [DuckDB](https://duckdb.org/)

---

**Curious which database renders infinity fastest? Clone and find out! 🌀**
