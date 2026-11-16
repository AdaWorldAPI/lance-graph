# G2 / DR virtual projection: what already works, the first missing seam (2026-11-16)

**Status:** VERIFIED-IN-CODE for every row marked so; one falsifier added
(`lance-graph-xlog/tests/cube_counts.rs`, disable-verified). Lab evidence, no
production admission. Plan context: `.claude/plans/xlog-knowledge-spine-v1.md`
G2. Scope: lance-graph (Cube, Quack, mask-risc, r2il binding, dir-sim,
dir-sim-lance, report), report-rs #33 and #120, an optional Vector/VRL edge.

## A. Evidence and reuse matrix

| Piece | Where | State | What it shows | Verdict |
|---|---|---|---|---|
| Lazy views (Scan/Bind/Join) | `lance-graph-xlog/src/cube.rs` `scan`/`bind`/`join` | VERIFIED-IN-CODE | A view is a node; no row is built | REUSE |
| Join size without tuples | `Cube::join_size` (dot product of two degree vectors) | TEST-PINNED: `join_size_equals_materialized_rows_without_building_any`, new `join_size_grows_quadratically_while_work_grows_with_ids` | 2 programs, 0 rows built; work grows with distinct ids, not pairs | REUSE |
| Bag counts | `Cube::bag_count` (checked i64), `lance-graph-r2il/src/binding.rs:513 run_bag` | VERIFIED-IN-CODE | Multiplicity kept as a count lane | REUSE |
| Projection (Identity / Select-Object) | Quack `Agg::Rows` (lib.rs 1150-1155) -> mask-risc `Terminal::Keep` | VERIFIED-IN-CODE | Result is the kept mask; the caller materializes | REUSE |
| Projection terminal with hydration | `lance-graph-report/src/render.rs` `detail_csv` | VERIFIED-IN-CODE | Mask -> `materialize_rows` -> per-row KV/CAM hydration | REUSE |
| Route / gate checks | `lance-graph-dt64/src/lib.rs:1051 check_route`, `binding.rs:462 run_gated` | VERIFIED-IN-CODE | Typed refusals on a changed route or gate | REUSE |
| Delta log on the cube | `Cube::lanes` | VERIFIED-IN-CODE | Empty log returns the base; otherwise one cached full refold | PROBE (cost is per read after any write) |
| Membership, users -> groups | `lance-graph-dir-sim/src/view.rs` `groups_transitive`, `groups_transitive_through`, `security_identifiers` | VERIFIED-IN-CODE | Exists. Corrects this session's earlier note that no groups-of-user reader exists | REUSE |
| Membership semantics | `snapshot.rs:658-661` sort + dedup | VERIFIED-IN-CODE | Membership is a set. A bag witness must come from the selection, never from dir-sim | REUSE (as SET) |
| Sparse history write | `dir-sim/src/log.rs` `LogState::batch` / `committed`; `dir-sim-lance` `LanceLog::append` | VERIFIED-IN-CODE | Only new records per batch; one batch = one Lance append; stale writer refused | REUSE |
| History replay | `log.rs` `replay` / `replay_prefix`; `LanceLog::restore` | VERIFIED-IN-CODE | Read once, refuse at the first damaged batch | REUSE |
| Simulated versions | `store.rs` `simulate`, `revert` | VERIFIED-IN-CODE | Stored as `delta: Vec<Change>` on a parent | REUSE |
| Re-observation | `store.rs:103-126` `observe` | VERIFIED-IN-CODE | Each observation is a new ROOT (`parent: None`) holding the whole `Observation` | **First missing seam** (see C) |
| Vector / VRL | whole workspace + report-rs | ABSENT in the closed space searched (`vrl`, `vectordotdev`, `Vector Remap`, `VRT`, `LogEvent`, `syslog`: 0 hits) | No surface to reuse | Contract only (see D) |
| Gather across coordinates | — | OPEN upstream (#33 ledger item 9) | — | REFUSE until admitted |

## B. report-rs #33 rescue map

#33 (head `81705d9`, open, "Do not merge yet") already carries its own
admission ledger (EXO-174). This map agrees with it; nothing new is admitted.

| Keep | Why |
|---|---|
| eTag change detection, schema reconciliation (added/reordered/renamed/removed columns; ambiguous rename refused) | Intake facts; eTag says the file changed, never which cell (EXO-172) |
| `ClassBinding` over one DT64 `BindStore` (`bind_lanes`, `check_lanes`, `admit`, `retract`) | Upstream socket (#1967/#1970), consumer under test (P21) |
| Refusals: stale frame, changed route, wrong schema/scope/space, unbound field, unadmitted XVERWEIS | Typed, tested |
| Lab zero-alloc fold (Quack/mask-risc equals scalar oracle) | Test-only owner; keep as a lab arm |

| Quarantine | Reason |
|---|---|
| `Lanes = Vec<Vec<StoredCell>>` | Second owner of cell values; materialization |
| Private `CamLabels`/`MemKv` codebook (`xverweis::Codebook`) | Second codebook beside DT64 `CodeTable` (#1982) |
| `Workbook::select`, `Workbook::xverweis` host loops | Per-row host execution |
| `Dirty::between` | Oracle only |
| f64 decimal path | Lossy (`9007199254740993` -> `...992.0`); Codex thread open. Must refuse or stay quarantined; never called exact |

Open ledger items that block admission: reader-scope witness (3), value
ingress (4), D4 cell writer (5b), text->code authority (6), exact decimal (7),
production XVERWEIS `MethodRef` (8), Gather (9).

report-rs #120: fits this contract (receipt with SHA-256 per input and a build
stamp). One small follow-up: set `REPORT_BUILD_REVISION` in the Railway image,
so the stamp is not empty in production.

## C. dir-sim sparse-history adoption map

observation -> record -> batch -> persistence -> replay -> read:

1. `observe::from_ad` / `from_graph` -> `Observation` (exists).
2. `VersionStore::observe` -> new root with the whole `Observation` (exists).
3. `LogState::batch` -> records for what is new: Value, Root, Version, Tag,
   Verdict (exists; a Root record carries the whole observation, a Version
   record only `Vec<Change>`).
4. `LanceLog::append` -> one Lance append per batch; logical `VersionId` never
   equals the Lance version (exists).
5. `LanceLog::restore` -> `replay` (exists).
6. `History::view` / `History::diff` (exists).

Readings: D1 current membership = `view(tag "observed")` + `groups_transitive`;
D2 changes since V = `History::diff(V, now)`; D3 simulated overlay =
`simulate` on a parent. All three exist.

**First missing seam:** step 2. A second directory read is stored as another
full root, so a one-membership change costs a whole observation in the log.
That is the hard stop "a complete snapshot on every tiny change", for the
re-observation path only. The fix is to store a re-read as a delta against the
previous root (`diff_full` already computes the changes at read time). It is a
dir-sim change and gets its own PR with its own falsifier (a second
observation that differs in one membership must add one Change record, not a
Root). Not done here.

## D. Vector / VRL adapter contract (external only)

- Vector is an ingress or egress adapter at the edge. It is never an internal
  IR; no native path depends on JSON or Vector events.
- Ingress: a Vector event becomes an observation through an existing
  admission (`observe::from_*`), with source, tenant and observed-at kept.
- Egress: a terminal (like `detail_csv`) renders a projection as events. The
  native result is computed first and is the same with or without the
  adapter (T8).
- Round trip: event -> observation -> view -> event is pinned by a test.
  Lossy fields (float decimals, timezone-less timestamps, unknown keys) are
  listed in the adapter and refused or labelled, never silently converted.
- Not built. No code reuses or needs it today.

## E. Non-materialization trace

`Identity | Select-Object Identity, Property` in this stack:

1. `plan.selection.lower(batch)` -> filter over lanes (no rows).
2. `lower(Query{filter, agg: Agg::Rows})` -> program ending in `Terminal::Keep`.
3. `execute_into(.., Out::Mask(&mut kept))`: the first unavoidable
   allocations are the population-sized `kept` mask and `Scratch`
   (`words_for(n)` words; one bit per row).
4. `materialize_rows(&kept, n)`: one `usize` per kept row.
5. Hydration per exported row and field (KV text, CAM label, integer).

No Cartesian intermediate exists. Avoidable: none found in this path; the
cube's delta-log refold (A, PROBE) is the one candidate relation rebuild.

No-explosion falsifier (T9/P6), `join_size_grows_quadratically_while_work_grows_with_ids`:
a hub key joins `k` rows with `k` rows, k in {8, 32, 120}. Oracle: the
arithmetic k². Measured: join size = k² (64 -> 14400, 225x), count vector
length = 2k+1 positions (17 -> 241, 14x), 2 programs each time, 0 rows
built. Disable mutation: `join_size` computed as
`materialize_rows(..).len()` turns this test and
`join_size_equals_materialized_rows_without_building_any` red. These are
counts, not timings. "No tuples" here means no row built; it does not mean
zero allocation (the degree vectors are allocated) or constant time (work is
linear in distinct ids).

## F. Smallest follow-up PRs, in order

1. dir-sim: store a re-observation as a delta against the previous root (C).
2. report-rs: set `REPORT_BUILD_REVISION` in the Railway image (#120).
3. Cube: measure the delta-log refold cost per read after a write (A, PROBE).
4. Vector adapter only when a consumer needs it, under D.

Not done together: dir-sim, report-rs and G2 stay separate PRs. G4
(Mistaber) stays its own gate. Nothing is merged automatically.
