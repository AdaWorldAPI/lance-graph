# D-INTAKE-2: intake as a compiler, measured against DuckDB (2026-11-21)

**Status:** MEASURED, in PR. There are two failed gates, recorded as D-INTAKE-2G
and 2H.

**Built** (28 tests; PostgreSQL 16.15 equal for both the one-shot and the
three-append build):
- text codes in DT `CodeTable`;
- `Table::append`, which advances the data version once;
- kept `SelectionRef`s, replayed with no predicate work and lent in place
  from DR's keep log.

**Measured** (`cargo run --release -p lance-graph-intake --example select_bench`,
avx512f, median of 11, answers asserted):

| | 10^5 rows | 10^6 rows |
|---|---|---|
| P1 `count_where` | 62.7 µs | 878.5 µs |
| P2 `count_where` | 79.1 µs | **FAILED** |
| warm `count_of` | 64 ns | 58 ns |
| `rows_of(4096)` | **FAILED** (272 ns/row) | **FAILED** (357 ns/row) |

DuckDB 2.0.1 via Python measured 920–1 169 µs warm `COUNT(*)`, including about
900 µs of client parse and plan per query, so no throughput comparison is
claimed.

**Failed gates** (no millisecond results):
- `rows_of(4096)`: per-cell `String` decode. Fix: D-INTAKE-2G, a sink
  writer.
- P2 count at 10^6 rows: four validity folds per WHERE. Fix: D-INTAKE-2H,
  validity kept per version.
