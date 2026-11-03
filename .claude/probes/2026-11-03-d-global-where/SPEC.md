# D-GLOBAL-WHERE P0 probe spec (lab only)

Written by the orchestrator, implemented by workers. Every path is relative to
`/home/user/lg-gw` (worktree on branch `ccr-85acaff0-oszq40-global-where`, cut
from `origin/main` `fb5e2d35b`). No file outside
`.claude/probes/2026-11-03-d-global-where/` may be edited.

## Hard rules

- No production change. Consume `lance-graph-dt64`, `lance-graph-quack`,
  `lance-graph-mask-risc`, `ndarray::simd` and (for checks 12/13 only)
  `dr-math-binding-probe` unchanged.
- No tuple materialization in any fold path. No per-object structs. No new
  identity table. No witness store. Probe-local code is labelled LAB.
- Every timing in µs or ns. Our side: any warm or query value >= 1000 µs prints
  `FAILED-GATE` beside it (workspace rule). Cold preparation is reported in µs,
  and also flagged if >= 1000 µs.
- Deterministic data (SplitMix64, seed 0x9E3779B97F4A7C15).
- Exit status 1 if any falsifier check fails or any answer disagrees with the
  scalar oracle.

## Part 1. `probe/` crate (own `[workspace]`), binary `falsify`

`probe/Cargo.toml`: package `d-global-where-probe`, edition 2021, `[workspace]`
empty table. Deps (path is from `probe/` to repo `crates/`, i.e. `../../../../crates/...`):
lance-graph-contract, lance-graph-dt64, lance-graph-quack, lance-graph-mask-risc,
dr-math-binding-probe; `ndarray = { git = "https://github.com/AdaWorldAPI/ndarray",
branch = "master", default-features = false, features = ["std"] }`.
Patches exactly as in
`.claude/probes/2026-11-03-d-xlog-meta-wire/probe/Cargo.toml` on branch
`meta-wire-local` (read it with `git show meta-wire-local:<path>`):
`[patch."https://github.com/AdaWorldAPI/lance-graph"] lance-graph-contract = path`,
`[patch."https://github.com/AdaWorldAPI/ndarray"] ndarray = { path = "../../../../../ndarray" }`
(adjust the relative depth: the ndarray checkout is `/home/user/ndarray`).
`[profile.release] debug = 0`.

A counting global allocator (pattern: `crates/lance-graph-quack/tests/duckdb_differential.rs`
lines 49-75) counting bytes and allocation calls, plus a peak tracker.

### Fixture

- Universe: 65,536 global ids. Lanes `a: Vec<i32>`, `b: Vec<i32>` of length
  65,536, values uniform in 0..1000.
- Selection A = `a < 50` (WHERE(A)), selection B = `b < 300` (WHERE(B)),
  evaluated by Quack: `Query { filter: Filter::Cmp(Col(0), Cmp::LtI32(50)), agg: Agg::Rows }`
  lowered with `lance_graph_quack::lower`, executed with
  `lance_graph_mask_risc::execute_into` over `Planes { n_rows: 65536, masks: &[&live], lanes: &[LaneRef::I32(&a)] }`;
  the Keep mask is read from `Scratch::slot` or the Plane per the returned
  `Value::Mask(Operand)`. See `crates/lance-graph-xlog/src/lib.rs` `run_lanes`.
- Relation domain D: 512 global ids (two DR256 tiles), chosen pseudo-randomly
  from the universe. Edges E: directed set over D, average out-degree 6, no
  duplicates.
- `MapTable` (`lance_graph_dt64::dr`): tile 0 and tile 1 hold D split in
  halves (relation frame R, `Width::W256`). Tile 10 = selection-A tile:
  the members of D with a<50, in DESCENDING global order. Tile 11 =
  selection-B tile: members of D with b<300, in ASCENDING global order.
  Ensure at least one global id y is in both A and B (force it by setting
  lanes for one chosen id) and that its local positions in tiles 10 and 11
  differ (assert).
- Presence plane (LAB): one row mask per R-coordinate (wide coordinate
  `ordinal*256 + local` as in `xlog::Cube::new_tiled`), 8 words per row, built
  from E via `MapTable::locate(frame, &[0,1], g)`. Also a reverse plane
  (column storage of the same edges) for the XOR checks.

### Checks (print `Fn PASS/FAIL detail`)

1. G->L->G: for every id in tiles 0, 1, 10, 11: `global(local(g)) == g`.
2. L->G->L: for every local l of each tile: `local(global(l)) == l`.
3. the shared y has different locals in tile 10 vs 11.
4. Local-coincidence join (treat B-local bit l as R... no: treat A-local and
   B-local positions as one domain, i.e. intersect the tile-10 local mask with
   the tile-11 local mask bitwise) gives a count != oracle |A ∩ B ∩ D|; the
   DR-aligned path (B-local -> global via tile 11 -> tile 10 local via
   `local()`) gives the oracle. Assert both.
5. Case B: pick x in A, y in B with edge (x,y) in E (force one). Route x via
   R frame, route B's selection into R coordinates (B-local -> global ->
   `locate`), answer `row[x] & selB_R != 0` is true; oracle agrees.
6. Clear that edge's presence bit (LAB plane update, one word): `locate` of y
   still Ok, answer now false (when y was x's only B-neighbour; construct the
   fixture so it is). Oracle over E minus that edge agrees.
7. Clear an unrelated edge (both endpoints outside A and B): answer of a
   fixed query set (count of x in A with a B-neighbour) unchanged.
8. Forward plane XOR transpose(reverse plane) == 0 over all words.
9. Flip one bit in the reverse plane: XOR != 0 and the differing position is
   exactly that edge.
10. Historical frame: seal frame f1, then `MapTable::advance` replacing tile 10
    with a reordered map (a replacement, new generation). `map(&f1, 10)`
    still returns the OLD locals (snapshot replay), `map(&f2, 10)` the new;
    `FrameStore`-style currency: `table.revisions()` coordinates changed.
11. Stale/foreign: a frame sealed by a second MapTable read against the first
    -> `MapRead::Stale(ForeignFrame)`; `map_at(future version)` ->
    `FutureVersion`; `locate` of an id in two listed tiles -> `Ambiguous`
    (list tiles 10 and 11 with the shared y).
12+13. Using `dr_math_binding_probe::where_dr3` with its own 64-slot
    `hhtl16::LocalMap` and `FrameStore`s (read that module and
    `crates/dr-math-binding-probe/tests/` for how a test builds `Under`,
    `Guards`, `BoundDr64` via `bind_dr64`):
    project one global WHERE mask once for method M1 (owner class C,
    ordinal 1) and once for M2 (class C, ordinal 2, same row axis).
    (12) admit with a register of a foreign class on the row axis ->
    `AxisMismatch`; (13a) admit M1's selection against M2's register ->
    `SchemaChanged`; (13b) the two projections' `Admitted::truth()` and
    `slots()` are identical bit for bit (projection does not depend on the
    method); (13c) admit with a different `Scope` -> `ScopeChanged`
    (the authorization guard exists independently of the method check).
14. Fold path allocation: around the check-5 and check-7 folds (after
    preparation), allocated bytes == 0 and no tuple vector exists. Report.
15. Same measurement documents "no per-object allocation".
16. Structural: print the list of types the probe defines; assert none stores
    identities except via `LocalMap`/`MapTable` (manual: print only).
17. Boolean vs bag: on E, `count distinct y reachable from A in one hop`
    (popcount of OR of A rows) differs from bag count of 2-paths from A
    (Σ_y indegA[y]·outdeg[y], `ndarray::simd::checked_binary_i64` or checked
    i64), both equal their oracles; `bind_dr64` with
    `Population::Bag` -> `BagNeedsMultiplicity`.
18. Multi-tile: the R frame spans 2 DR256 tiles; check 5 must have x and y in
    different R tiles (assert ordinals differ).

Print a census line: per check, lookups (`TileMap::lookups`), bytes allocated.

## Part 2. binary `bench` (same crate)

Generate datasets and write them to `$OUT` (arg 1, a directory) as CSV, then
run our side and write `$OUT/ours.json`.

Datasets: for each `size` in {dr64: 64, dr256: 256, multi: 4096 (16 DR256 tiles)}
× `dist` in {sparse: out-degree ~1 uniform, medium: ~8 uniform,
dense: p = 0.25 for dr64/dr256 and p = 0.03 for multi (state this),
skewed: Zipf-like, node i (by a random rank) gets out-degree ∝ 1/(rank+1)
scaled to total 8·n, targets drawn with the same Zipf bias}.
Domain ids: `size` pseudo-random globals from 0..65536. Universe nodes table:
65,536 rows `id,a,b` (a,b uniform 0..1000; same lanes for every dataset is
fine). Edges only inside the domain.

Files per dataset `<size>_<dist>`: `nodes.csv` (shared, written once:
`id,a,b`), `<ds>_edges.csv` (`src,dst`), `<ds>_selA.csv`, `<ds>_selB.csv`
(id lists: domain members with a<50 resp. b<300, written in the order of their
selection tiles), `<ds>_delete.csv` (the 1% edges deleted in W8, at least 1).

Workloads and their exact SQL semantics (the Python side uses identical SQL):
- W1 `SELECT count(*) FROM nodes WHERE a < 50`
- W2 `SELECT count(*) FROM nodes n WHERE n.a < 50 AND EXISTS (SELECT 1 FROM edges e JOIN nodes m ON m.id = e.dst WHERE e.src = n.id AND m.b < 300)`
- W3 `SELECT count(*) FROM nodes n WHERE n.a < 50 AND EXISTS (SELECT 1 FROM edges e1 JOIN edges e2 ON e2.src = e1.dst JOIN nodes m ON m.id = e2.dst WHERE e1.src = n.id AND m.b < 300)`
- W4 `SELECT count(DISTINCT e3.dst) FROM nodes n JOIN edges e1 ON e1.src = n.id JOIN edges e2 ON e2.src = e1.dst JOIN edges e3 ON e3.src = e2.dst JOIN nodes m ON m.id = e3.dst WHERE n.a < 50 AND m.b < 300`
- W5a `SELECT count(DISTINCT e.dst) FROM edges e JOIN nodes n ON n.id = e.src WHERE n.a < 50`
- W6 `SELECT count(*) FROM edges e1 JOIN edges e2 ON e2.src = e1.dst JOIN nodes n ON n.id = e1.src WHERE n.a < 50`
- W7 `SELECT count(*) FROM selA s WHERE EXISTS (SELECT 1 FROM edges e JOIN selB t ON t.id = e.dst WHERE e.src = s.id)` plus W7i `SELECT count(*) FROM selA JOIN selB USING (id)`
- W8: W2 repeated, then delete the rows of `delete.csv`, then W2 again.

Our side per dataset, single thread:
- cold: build `MapTable` tiles over the domain (`Axis::for_survivors(size)`),
  presence plane (forward rows), outdeg (popcount per row); time it.
- WHERE: two arms, timed separately: (W-q) Quack Keep over the 65,536-row lane
  then project to domain coordinates via `TileMap::globals()` (size probes);
  (W-v) virtual: test the predicate only at the domain's globals
  (`a[g] < 50` for g in the map), building the domain mask directly. Assert
  equal masks. W1 uses Quack `Agg::Count` over the full lane.
- Fold per workload, plans as named arms where more than one exists:
  W2: any(row[x] & selB) per x in selA.
  W3: `back` (selection-first backward: Yb = {y : row[y] & selB != 0}, then
  count x in A with row[x] & Yb != 0), `tile` (per x in A: R2 = OR of rows of
  row[x]'s members, test & selB), `csr` (CSR adjacency built at cold time,
  DFS with early exit).
  W4: `front` (frontier F=selA; 3× F = OR rows of F; popcount(F & selB)),
  `csr` (BFS over CSR with a visited bitmap per level).
  W5: support = popcount(OR rows of A).
  W6: indegA[y] via per-row bit iteration over x in A, Σ indegA·outdeg, checked i64.
  W7: route selB (tile-11 local order) -> domain coordinates via global id
  (`global()` then `local()`/`locate`), time routing separately and as warm
  reuse (routing done once, reused); then W2-style fold; W7i = popcount of the
  intersection after routing.
  W8: warm W2 median; update = clear presence bits for delete.csv (time it);
  re-run W2.
- Warm: median and p10/p90 of 201 runs each (std::hint::black_box). Cold:
  first-run time of prep + WHERE + fold. Record allocation bytes per stage.
- Correctness: every answer checked against a scalar oracle over the edge
  list (HashSet-based is fine in the oracle only).

`ours.json`: array of records
`{dataset, size, dist, edges, workload, arm, answer, cold_us, warm_med_us,
warm_p10_us, warm_p90_us, where_q_us, where_v_us, route_us, fold_us,
update_us, alloc_bytes_prep, alloc_bytes_query, peak_bytes, interm_cardinality}`
(null where not applicable).

## Part 3. `duckdb/bench_duckdb.py` (lab script, Python, run with `python3 -I`)

Arg 1: the `$OUT` directory. For each dataset: fresh in-memory connection;
`SET threads=T` for T in {1, 4}; load CSVs into tables `nodes`, `edges`,
`selA`, `selB` with `read_csv` and explicit types (INTEGER). Normal optimizer
(no pragmas disabling anything).

Per workload: `PREPARE q AS <sql>`; cold = wall time of PREPARE + first
`EXECUTE q` (perf_counter_ns); warm = median/p10/p90 of 201 `EXECUTE q`
fetching the scalar. Also with `PRAGMA enable_profiling='json'` +
`PRAGMA profiling_output='<file>'` on ONE extra execution, read `latency`
(seconds) from the JSON for the engine-side time. Measure the Python floor:
median of 201 `EXECUTE` of `PREPARE f AS SELECT 1`. Save
`EXPLAIN ANALYZE <sql>` text for every workload of the `dr256_medium` and
`multi_skewed` datasets at T=1 to `$OUT/explain/`. W8: DELETE via
`DELETE FROM edges WHERE (src,dst) IN (SELECT src,dst FROM del)` timed, then
re-execute. Peak memory: `SELECT sum(memory_usage_bytes) FROM duckdb_memory()`
after each workload.

Write `$OUT/duckdb.json` with records `{dataset, threads, workload, answer,
cold_us, warm_med_us, warm_p10_us, warm_p90_us, engine_us, floor_us,
update_us, mem_bytes}`.

## Part 4. `duckdb/compare.py`

Join ours.json and duckdb.json on (dataset, workload); assert answers equal
(exit 1 otherwise); for each ours arm and DuckDB T in {1,4} print a Markdown
table: warm ours (best arm and each arm), DuckDB warm wall, DuckDB warm minus
floor, DuckDB engine latency, ratio DuckDB/ours (>1 = ours faster), cold both
sides, and a verdict column WIN / PARITY (within ±10%) / LOSS. Write
`$OUT/compare.md`.
