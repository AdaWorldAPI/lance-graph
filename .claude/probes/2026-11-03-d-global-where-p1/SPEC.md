# D-GLOBAL-WHERE P1 probe spec: 64K page amortization, routing arms, cost model (lab only)

> Paths under `probe/`, `duckdb/` and `results/` refer to the research branch [`research/d-global-where-p2`](https://github.com/AdaWorldAPI/lance-graph/tree/research/d-global-where-p2/.claude/probes/2026-11-03-d-global-where-p1) (commit `1422b2b92`); they are not on `main`.

Written by the orchestrator; implemented by workers. Root: `/home/user/lg-p1`, branch
`ccr-85acaff0-oszq40-global-where-p1`, cut from `origin/main` `ebbdb27d9`. Edit only
inside `.claude/probes/2026-11-03-d-global-where-p1/`.

P0 (#1943) is evidence, not a dependency. Its probe is readable with
`git -C /home/user/lg-p1 show ccr-85acaff0-oszq40-global-where:.claude/probes/2026-11-03-d-global-where/probe/src/lib.rs`
(and `.../bin/falsify.rs`, `.../bin/bench.rs`, `.../probe/Cargo.toml`). Copy patterns from it
(counting allocator, SplitMix64, Quack WHERE prepared program, bit helpers); do not depend on it.

## Hard rules

- No production change. Consume `lance-graph-dt64` (`dr` module + `BindStore`/`MethodRef`/`admit`),
  `lance-graph-quack`, `lance-graph-mask-risc`, `ndarray::simd` unchanged.
- No second identity authority. Every global → coordinate answer comes from `MapTable` /
  `TileMap` under a sealed `Frame`. A derived index (arm C) is built FROM `MapTable` views under
  one frame, carries that frame, refuses any other frame, and is never consulted for a frame it
  was not built for.
- No tuple vectors, XYZ expansion, per-object structs or persisted intermediate relations in a
  timed fold. Oracle code may use HashSet/HashMap.
- Units: ns/µs. Any warm or per-query value ≥ 1000 µs prints `FAILED-GATE`; cold/preparation
  values ≥ 1000 µs print `COLD (reported, not gate)`.
- Deterministic data: SplitMix64, seed 0x9E3779B97F4A7C15.
- Every binary exits 1 on any failed check or oracle mismatch.

## Model: three layers

| layer | here |
|---|---|
| global corpus | u64 global ids, NOT contiguous (pseudo-random over 0..2^32), N up to 10M |
| logical 64K page | ONE `MapTable` (its own frame store and clock) holding up to 256 DR256 tiles, TileIds 0..255: exactly the reservation `Extent::DR256_GLOBAL_X64` names. Page coordinate = `tile_ordinal * 256 + local` (u16 range) |
| active tile | a `TileMap` borrowed from the page under a frame |

A page holds only the identities a selection demands. Page membership is chosen by the
selection, never by bits of the global id.

## Routing arms (identical answers, including refusals)

- **A, baseline:** `MapTable::locate(frame, &page_tiles, g)` per endpoint.
- **B, page-grouped batch:** collect the distinct globals of all endpoints, sort them, locate each
  ONCE with `locate`, then resolve endpoints by a merge/binary search over that sorted transient
  array (dropped after the call: not an index). Measure locate calls (should equal distinct ids).
- **C, derived index (optional acceleration):** built from `table.map(&frame, t).globals()` for
  every page tile: a sorted `Vec<(u64 global, u16 coord)>`. Adjacent equal globals → that id is
  `Ambiguous`; absence → `Unmapped`. Stores the `Frame` it was built from; `lookup(frame, g)`
  refuses with `StaleIndex` unless `frame.same_issue(&built)`. Building costs one pass over the
  page's ids plus a sort; lookup is one binary search (no per-tile scan).

## Part 1: `src/bin/falsify_p1.rs`

Fixture: page P of 4 DR256 tiles (1,000 ids), page Q of 2 tiles (400 ids), disjoint except as
stated; edges among P ∪ Q (avg out-degree 4, ~8 % crossing P→Q); lanes `a`,`b` over the ids
(stored as a sparse lookup in the ORACLE only; the folds read lanes indexed by page coordinate
built at page preparation as a borrowed slice of the source lane order: put the fixture's lane
values in an array indexed by the position of the id in a fixture list, and document it).

Checks F1..F20 (print `Fn PASS/FAIL detail`, exit 1 on failure):

1. G→L→G and L→G→L parity for every id of P and Q, all three arms.
2. Arms A, B, C give identical coordinates for every edge endpoint of P.
3. `locate` calls: A = 2·edges(P), B = distinct endpoint ids, C = 0 per query after build.
4. An id inserted in two tiles of P: A → `Ambiguous`, B → `Ambiguous`, C → `Ambiguous`.
5. An id in no tile: all arms `Unmapped`.
6. Foreign frame (sealed by Q's table) read against P: A `Stale(ForeignFrame)`; C refuses.
7. Historical frame: seal f1; `advance` replacing tile 0 with a reordered map → f2. Under f1, A and
   a C built for f1 return the OLD coordinates; C(f1) queried with f2 → `StaleIndex`; C rebuilt
   for f2 == A under f2.
8. Extension: append ids to tile 1 (an extension, `TileMap::extends`); every old id keeps its
   coordinate (presence plane built at f_old is still valid for old ids: the answer of the W2 query
   restricted to old ids is unchanged); a guard compares generations, not digests.
9. Replacement invalidates the binding: reusing the f1 binding (coords + presence) under f2 without
   the frame check gives a WRONG answer (measured); with the check it refuses.
10. Tile removal: an id of a removed tile → `Unmapped` under the new frame, still resolves under
    the old frame.
11. Presence-only update: delete an edge; frame and coordinates unchanged; the binding is reused
    (0 new locate calls for the fold; the update itself charges endpoint routing via C); the
    answer drops by the oracle's amount. An unrelated deletion leaves the answer unchanged.
12. Cross-page edge: x ∈ P, y ∈ Q; B's selection held in Q coordinates; evaluating x's edge via
    Q's frame gives the oracle; joining by coordinate coincidence gives a different (wrong)
    count.
13. The same global id at different coordinates in two selections/pages; DR-aligned equality.
14. Reused binding == freshly prepared binding (coords and all W2/W3/W4 answers), Q = 64 reuses.
15. EXO-46 shared selection (LAB type `SharedSelection { frame, scope, criteria, mask }`):
    a. M1 and M2 (same class, both registered with `BindStore::register`, distinct `MethodRef`s)
       both admit the SAME selection when each holds its own permit (a permit = a `Vec<MethodRef>`
       the probe issues);
    b. M2 presenting only M1's permit → refused (no inheritance);
    c. a MethodRef of another class → refused (`admit` of dt64);
    d. a stale sig rev → refused;
    e. another Scope → refused;
    f. a stale frame (after `advance`) → refused;
    g. a presence update does not change the sealed selection's mask (bit-identical before/after)
       while the fold over it reflects the new presence;
    h. no mask is passed between programs (structural: the selection owns the mask, folds borrow
       it);
    i. zero allocation in admit + fold.
    Read `crates/lance-graph-dt64/src/lib.rs` for `BindStore`, `MethodBinding`, `MethodRef::of`,
    `admit` and its refusal enum; the merged probe
    `.claude/probes/2026-11-03-d-xlog-meta-wire/probe/src/{main.rs,bin/v2_compose.rs}` shows a
    working registration.
16. AND/XOR/FOLD parity with the oracle on W2 (1-hop EXISTS into B), W3 (2-hop EXISTS), W4 (3-hop
    DISTINCT reached in B), W5 (COUNT DISTINCT 1-hop), W6 (Bag 2-path count, checked i64).
17. Folds allocate 0 bytes; no tuple vector.
18. Disable C's duplicate check → F4 fails for C (run as a compiled-in `--disable dupcheck` flag:
    the binary takes optional `--disable <name>` and must then report FAIL on the named check).
19. Disable C's frame check → F7/F9 fail (`--disable framecheck`).
20. Disable binding version guard → F9 gives the wrong answer silently (`--disable bindguard`).

## Part 2: `src/bin/sweep.rs` (amortization law and plan choice)

Measure components per configuration, then compute the law, then verify it end to end.

Configurations: active ids per page n ∈ {64, 256, 4096, 65536} (tiles = ceil(n/256), DR64 when
n ≤ 64 via `Axis::for_survivors`), pages p ∈ {1, 4, 16} (p = 64 only for n ≤ 4096),
distribution ∈ {sparse(deg≈1), medium(deg≈8), dense(deg≈min(64, n/8)), skewed(Zipf, mean 8)},
cross-page edge fraction ∈ {0.01, 0.30}. Skip a config whose edge count would exceed 40M and say
so.

Per config, single thread, median of 31 runs:
- `prep_A` / `prep_B` / `prep_C`: build the page tiles (`MapTable::advance`), route all edge
  endpoints with the arm, build the coordinate CSR (u32 offsets, u16/u32 targets per page;
  cross-page targets carry (page, coord)).
- `route_q_A`, `route_q_B`, `route_q_C`: per-query routing cost if the binding is NOT reused (the
  query routes its selection, |A| ids, from scratch).
- `fold_W2`, `fold_W3_csr`, `fold_W3_front`, `fold_W4_csr`, `fold_W4_front`, `fold_W4_tile` (tile
  only for total active ≤ 4096: dense rows).
- `where_v`: virtual WHERE over the page's ids (predicate at mapped globals).
- `update`: delete k edges (k = 0.1 %, 1 % of edges) through C: clear presence bits in the CSR's
  edge-validity mask; coordinates untouched.
Then C_query(Q) = where_v + route_q(reused? 0 : arm) + fold + prep/Q + maintenance/Q for
Q ∈ {1,4,16,64,256,1024} and change rates {0, 0.1 %, 1 %} per query. Break-even Q for: reuse vs
per-query routing (A, B, C). Verify the law: for Q = 64, run the real loop (prep once, 64 queries
with updates) and compare measured total with the formula (report relative error).

Plan choice: `fn choose(meta: &Meta) -> Arm` reading ONLY pre-execution metadata (|A|, |B|, n,
p, edges, mean out-degree, max degree, cross-page fraction, hops, binding prepared?). Its
constants are fitted on a CALIBRATION set (seed+1 datasets) and evaluated on the TEST set (seed).
Report per test config: chosen arm, measured best arm, regret = t(chosen)/t(best), choose() cost
in ns. The acceptance test must show the preferred arm changing across configs.

Output `$OUT/sweep.json` (one record per config × metric) and `$OUT/law.json` (per config × Q ×
change rate: predicted cost, components, break-evens) and `$OUT/plan.json`.

## Part 3: `src/bin/big.rs` (10M corpus, DuckDB comparison data)

N = 10,000,000 nodes, global ids = a SplitMix64 permutation-free random u32 per node (unique:
regenerate on collision; report) — not 0..N. Lanes a, b uniform 0..1000. Edges: out-degree
Poisson-like mean 2 (skewed variant: 1 % hubs with degree 200) → about 20M edges, global ids.
Write `$OUT/big/nodes.csv` (id,a,b), `edges.csv` (src,dst), `delete.csv`.

Load (charged, reported separately, like DuckDB's load): a global CSR over node index (offsets
u32[N+1], targets u32 node index), and a global-id → node-index lookup used only for loading.

Queries (A = a < 1, ~0.1 %; B parameter t varies per run over {100, 200, 300, 400, 500}):
- G1 `count(*) WHERE a < 1` over 10M rows: Quack `Agg::Count` over the 10M lane.
- G2 1-hop EXISTS into B, G3 2-hop EXISTS, G4 3-hop COUNT DISTINCT reached in B,
  G5 COUNT DISTINCT 1-hop, G6 Bag 2-path count.
Ours, two modes:
- `direct`: no page binding; walk the global CSR from A every query (value reads through node index).
- `paged`: once per selection A: demanded identities (A ∪ 1/2/3-hop frontier) bound into 64K
  pages (MapTable, 256 tiles each, arm C index); page-coordinate CSR; then each query = virtual
  WHERE for B on page globals + folds in page coordinates. Report pages, active ids, cross-page
  edges, prep cost, and the break-even Q against `direct` and against DuckDB's warm.
- Presence update: delete 0.1 % of active edges (validity bits); re-query.
Every answer against a scalar oracle over the edge list.
`$OUT/big/ours_big.json` records: {query, mode, answer_per_t: [..], cold_us, prep_us, warm_med_us,
warm_p10_us, warm_p90_us, peak_bytes, alloc_query_bytes, pages, active_ids, cross_page_edges,
update_us}.

## Part 4: `duckdb/bench_big.py`, `duckdb/compare_big.py`

DuckDB 2.0.0, threads 1 and 4, normal optimizer. Load CSVs (load_us reported). For each query a
prepared statement with parameter $1 = t (G1 has none); cold = PREPARE + first EXECUTE; warm =
median/p10/p90 of 51 EXECUTEs cycling t over {100..500}. Engine time from JSON profiling
`query.total_time` (one execution per t). Do NOT subtract `SELECT 1` as Python overhead: report
it as `trivial_query_wall_us` beside wall and engine. EXPLAIN ANALYZE per query (threads 1) to
`$OUT/big/explain/`. Delete via the delete.csv set, time it, re-run. `duckdb_memory()` totals.
Write `$OUT/big/duckdb_big.json`. `compare_big.py` asserts every answer per t equal (exit 1
otherwise) and prints a Markdown table: ours direct / paged warm, paged prep, break-even Q vs
DuckDB warm (prep / (duck_warm − ours_warm), ∞ when ours is slower), DuckDB wall/engine/cold/load,
both thread counts, verdict WIN / PARITY (±10 %) / LOSS on wall and engine. Write `compare_big.md`.

SQL (one dict, identical semantics):
- G1 `SELECT count(*) FROM nodes WHERE a < 1`
- G2 `SELECT count(*) FROM nodes n WHERE n.a < 1 AND EXISTS (SELECT 1 FROM edges e JOIN nodes m ON m.id = e.dst WHERE e.src = n.id AND m.b < $1)`
- G3 `SELECT count(*) FROM nodes n WHERE n.a < 1 AND EXISTS (SELECT 1 FROM edges e1 JOIN edges e2 ON e2.src = e1.dst JOIN nodes m ON m.id = e2.dst WHERE e1.src = n.id AND m.b < $1)`
- G4 `SELECT count(DISTINCT e3.dst) FROM nodes n JOIN edges e1 ON e1.src = n.id JOIN edges e2 ON e2.src = e1.dst JOIN edges e3 ON e3.src = e2.dst JOIN nodes m ON m.id = e3.dst WHERE n.a < 1 AND m.b < $1`
- G5 `SELECT count(DISTINCT e.dst) FROM edges e JOIN nodes n ON n.id = e.src JOIN nodes m ON m.id = e.dst WHERE n.a < 1 AND m.b < $1`
- G6 `SELECT count(*) FROM edges e1 JOIN edges e2 ON e2.src = e1.dst JOIN nodes n ON n.id = e1.src JOIN nodes m ON m.id = e2.dst WHERE n.a < 1 AND m.b < $1`

---

## CORRECTION (operator, 2026-11-03): working-set construction is P0, paging is secondary

The hypothesis is NOT "partition the corpus into 64K pages". It is: **constructing the right,
query-specific ≤ 64K active working set from an unbounded corpus solves most of the scaling;
the existing DR-local fold algebra then runs unchanged.** The "~98 %" figure is a working
hypothesis, not a measurement.

Consequences for this spec (they override the text above where they conflict):

1. **Naming.** The lib type is `WorkingSet` (one `MapTable`, up to 256 DR256 tiles, ≤ 65,536
   identities chosen by the QUERY, not contiguous, not by id bits). "Page" in Parts 1–3 means
   this working set. A second working set exists only for bounded continuation.
2. **P0 measured stages, reported separately** (every binary that builds one):
   `discover` (global WHERE → candidate ids: scan arm over the corpus lane via Quack Keep, and an
   optional pre-built attribute-index arm whose build is charged as preprocessing),
   `expand` (reachable demanded ids via the global relation index, bounded at 65,536),
   `bind` (DR coordinate binding of exactly those ids: arms A/B/C),
   `project` (relation views restricted to the working set: coordinate CSR + presence bits),
   `fold`, and `reuse` (Q queries over one binding).
3. **Corpus-size independence is the success criterion.** Part 3 runs N ∈ {1M, 10M} (30M if
   disk/RAM allow; say so if not) with the SAME demanded active population. Report each stage per
   N. Stages whose cost tracks N (expected: the scan arm of `discover`) are named as such; the
   claim holds only if `expand + bind + project + fold` do not grow with N.
4. **Incremental maintenance (P1).** Presence-only update: the binding survives, only presence
   bits change. Selection change: B-side parameter changes re-run only `where_v` on the
   working-set ids; an A-side change that adds ids EXTENDS the working set (`MapTable::advance`
   with `TileMap::extends`, old coordinates kept) instead of rebuilding; measure both against a
   full rebuild.
5. **Bounded continuation (P1).** Only when the demanded set exceeds 65,536 (G4 3-hop on the
   skewed 10M corpus may): a second working set, cross-set edges resolved through the global id
   (`locate` in the other set's frame), Boolean DISTINCT and Bag counts exact. Report its cost
   separately. No unbounded intermediate, no second identity authority.
6. **Part 2 (sweep) is reduced** to: active size n ∈ {64, 256, 4096, 65536} × distribution ×
   cross-set fraction {0, 0.3}; routing arms A/B/C; amortization over Q; update rates; plan choice.
   The "pages p" dimension is dropped except p = 2 for the continuation case.
7. **DuckDB (Part 4)** also runs N ∈ {1M, 10M}; report preprocessing on both sides (our global
   relation index + optional attribute index; DuckDB load + ANALYZE, and an optional
   `CREATE INDEX` arm on edges(src), nodes(id) with its build time).

---

## P2 (operator, 2026-11-03): criteria-aware WHERE and multi-domain DR composition

This PR carries P1 and P2 together (title P2). Three independent variables, never merged into one
"dataset size": **K** (criteria per WHERE), **population** (survivors), **D** (DR domains composed).

### Source inventory (orchestrator, read in `crates/lance-graph-quack/src/lib.rs`)
- `Filter::{Cmp, Plane, EqU32Via, Semijoin, And(Vec), Or(Vec), Not}` (`:336`), `Filter::and_by_skip`
  (selectivity-ordered AND, `:752`), `Filter::sql_where` three-valued rewrite over validity planes
  (`:549`), `lower` (`:1383`) and `lower_fused` (ternlog fusion, `:1397`), executed by mask-risc
  `execute_into`. So K-criteria WHERE over resident lanes is PRODUCTION; `EqU32Via`/`Semijoin` are
  EXO-140-gated and not used here.

### Part 5: `src/bin/where_k.rs` + `duckdb/bench_where.py`
Corpus N ∈ {1M, 10M}; columns c0..c15 i32 and nullable c15 (null when `f(id,15) % 10 == 0`).
Values are a closed formula both sides compute identically (no CSV for nodes):
`f(id, j) = (((id * 2654435761 + j * 40503) % 2147483648) * 1103515245 % 2147483648) % 1000`
(all BIGINT, no overflow: operands < 2^31 before each multiply). Correlated column: c1 =
`(c0 + f(id,1) % 20) % 1000`.
Cases: K groups {1, 2–4, 5–16, ≥17 (17 and 24, criteria may reuse columns with different
constants)} × population {tiny ≤ 100, medium ≈ 1 %, large ≈ 50 %} × shape {AND-heavy, OR-heavy,
mixed} plus the adversarial pairs: one criterion ≈ everything; one ≈ nothing; 16 nonselective; 16
collapsing to one survivor; correlated criteria that do not reduce; contradictory → empty; a
nullable-column case through `sql_where`. Thresholds are SEARCHED by the generator to hit each
population target and the achieved count is reported (never assumed from K).
The generator writes `$OUT/where/cases.json`: each case = predicate tree {op: and|or|not|cmp,
col, cmpop, value} + target + achieved count (oracle). Python builds SQL from the same tree.
Ours per case, single thread, median of 31: `lower_us` (plain and fused), `eval_us` for
`Agg::Count`, `Agg::Any`, `Agg::Rows` (Keep → transient mask), plain vs fused vs `and_by_skip`
order, temporary bytes, and the representation used: `virtual` (Count/Any: no mask survives),
`transient` (Keep mask), `persistent` (never here). Report weighted (34/46/14/6 over K groups) and
unweighted tables. DuckDB: same predicate as `SELECT count(*) FROM nodes WHERE <pred>` and an
`EXISTS` variant (`SELECT EXISTS(SELECT 1 FROM nodes WHERE <pred>)`), prepared, threads 1 and 4,
answers must match exactly; EXPLAIN ANALYZE for one case per K group. Output `where_ours.json`,
`where_duckdb.json`, `compare_where.md`.

### Multi-domain composition (Part 1 additions, F22–F29)
F22 one / two / three DR domains give the same answers as one domain over the union (EXISTS,
COUNT, DISTINCT, Bag). F23 an edge crossing a domain boundary; F24 a path crossing three domains;
F25 the same global id in two selections; F26 a duplicate id across incorrectly configured domains
→ refused (`Ambiguous` when listed together; a global-id union check otherwise); F27 a deleted
cross-domain edge; F28 historical replay across domains; F29 aggregate composition laws: EXISTS =
OR of parts, COUNT = checked sum over DISJOINT contributions, DISTINCT = exact union (and a
falsifier: summing per-domain distinct counts over overlapping domains gives the WRONG answer),
Bag = multiplicities preserved.
