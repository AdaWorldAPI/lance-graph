# 2026-10-26 — D-DQM-M1: timed arm in both differential halves; exit gate NOT met

**Status:** MEASURED (both arms ran); M1 exit gate open. No performance claim.

Plan: `.claude/plans/duckdb-dr-quack-machine-room-reuse-v1.md` M1.

## What landed

- `tests/duckdb/timing.py` (DuckDB arm): reuses `oracle.load` and
  `oracle.encode`; parity first, then `t_prepare_ns` (PREPARE alone) and
  `t_execute_ns` (EXECUTE + fetchall), median of `--reps`, `SET threads`.
  `oracle.py` only gained `load(con)`; `cases.tsv` regenerates byte-identical.
- `duckdb_differential.rs` (Quack arm): `QUACK_DIFF_TIMED_REPS` (default 0,
  so the default run is unchanged: METRIC lines diffed identical before and
  after). When set, METRIC gains `t_compile_or_bind_ns`, `t_execute_ns`,
  `timed_reps`; every rep asserts the same answer as the untimed run.

## Run (same machine, 4 cores, load 0.2, x86-64-v3 default config)

```
QUACK_DIFF_TIMED_REPS=101 cargo test --release -p lance-graph-quack \
  --test duckdb_differential -- --nocapture --test-threads=1     # 39 pass, 1 ignored
python -I crates/lance-graph-quack/tests/duckdb/timing.py --threads 1 --reps 101
                                                                 # 38/38 parity=OK, DuckDB 2.0.0
```

- Quack, 34 timed cases: bind 56–1201 ns, execute 440–7899 ns (median of 101).
- DuckDB, 38 cases: prepare 449–1714 µs, execute 437–3251 µs. Reported, not
  gated (D-DQM's reference-arm exception).
- 27 cases pair by name; 7 Quack-only (avg split into sum and count, two
  k-program variants, the ordered-view join); 11 DuckDB-only (the avg and
  having cases go through `run_group_avg` / `run_group_having`, which build
  no CaseMetrics and are untimed).

## Why the exit gate is not met (so no ratio is quoted)

The gate asks for identical measured boundaries. They are not:

1. DuckDB's execute includes the Python call and fetchall conversion; Quack's
   includes clearing its out-buffer and the per-rep answer check.
2. Quack's bind includes lowering plus `Scratch` allocation; DuckDB's prepare
   is parse/bind/plan.
3. `run_scalar`'s bind rep re-runs only `Scratch::for_program`, not the
   caller's lowering; `run_group` times separate scratches.
4. No oracle checksum in the METRIC line yet.

OPEN: a DuckDB execute measured without the Python boundary (C API or a
server-side loop) or a stated FFI floor; time the avg/having helpers;
align the `run_scalar` bind boundary; add the checksum field.
