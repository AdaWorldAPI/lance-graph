# 2026-11-12 — D-DR-SELECT-K-SV-3: three DuckDB SelectionVector C-API functions, compared with DuckDB itself

**Kind:** lab crate `crates/duckdb-selvec-compat` (own workspace). No production caller.
**Grade:** MEASURED (`harness/run.sh`: one C fixture linked against the pinned
`libduckdb.so` v1.5.6 and against the adapter, 9 output lines identical,
valgrind clean; `cargo test`: 1 contract test; 6 of 6 disable runs red;
`cargo clippy --all-targets --all-features -- -D warnings` clean).
**Plan:** `.claude/plans/dr-select-k-v1.md` §16. **Linear (existing):** EXO-253.

- Reference pinned: tag `v1.5.6`, commit `069cc9f9b5be`, asset sha256 in the plan.
- `duckdb_create_selection_vector`, `duckdb_destroy_selection_vector`,
  `duckdb_selection_vector_get_data_ptr` as an owned writable buffer; symbols
  exported only with feature `duckdb-symbols`.
- A native selection reaches a writable pointer only by a counted copy
  (`selection_from_positions`); a buffer carries no frame, scope or permission.
- Handles are not interchangeable with DuckDB's; slice and copy are not provided.
- R4 measured on real DuckDB: a slice is a view that shares the selection's
  storage and keeps it alive after destroy; repeats, order, nesting and NULL
  go through the selection. R7 read: no per-lane validity and no owned
  selection in `Planes`.

**Open:** R4 and R7 stay UNDER REVIEW; x86-64 Linux only; allocation failure
not compared; authorization untouched.
