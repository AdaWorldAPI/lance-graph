# D-GLOBAL-WHERE P0: direct-routed virtual SoA views across selections, plus the DuckDB parity gate

**Status:** MEASURED, lab only. No production change.
**Probe:** `.claude/probes/2026-11-03-d-global-where/` (`REPORT.md`; `probe/` with binaries `falsify` and `bench`; `duckdb/` scripts; `results/`).

**Falsifier: 21/21 checks.** The F4 (DR alignment) and F6 (presence clear) guards were each disabled and turned red.
- Production dt64 DR routing (`LocalMap`, `MapTable::{map, map_at, locate}`, `Frame`) reconciles two selections holding the same global id at different local positions.
- A join on local indices gives 32 against an oracle of 14.
- Deleting an edge leaves the identity resolvable and turns the answer false.
- The XOR consistency check detects a single flipped bit.
- Folds allocate 0 bytes.
- `where_dr3::LocalSelection` projects bit-identically for two methods. Its `SchemaChanged` on a second method is a coupling, not a correctness need; Scope refuses on its own.

**DuckDB 2.0.0 gate:** 336/336 answers agree; 8 workloads × DR64 / DR256 / multi-tile (16×DR256) × sparse / medium / dense / skewed, at threads 1 and 4.
- Warm best arm: 12/0/0 W/P/L for every workload.
- But DuckDB's fixed per-query cost (~600 µs; 703 µs at 65,536 rows against 12,106 µs at 10M) dominates most warm rows. Those are not claimed as algorithmic wins.
- Work-avoided rows: W4 3-hop on `multi_dense` is 131 µs against 1,500,724 µs; on `multi_skewed` 89 µs against 57,066 µs, where DuckDB builds 3,726,779 join rows for a 1,164 answer.
- Losses: cold on every `multi_*` dataset, and W7 per-call routing. Both come from addressing: `MapTable::locate` reads every listed tile per id. Routing once and reusing costs 5 µs against 650–970 µs.
- Plan choice by cost: `csr` wins W3 everywhere. For W4, `csr` wins on small or sparse data and `front` on medium, dense and skewed. Tile-first never won.

**OPEN:**
- Scale ≥10M rows, unmeasured on our side.
- Owner: may one selection serve several MethodRefs under one Scope (EXO-46)?
- Owner: is a global → tile index wanted beside `locate`'s ambiguity scan?
