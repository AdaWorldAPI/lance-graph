# Crepe as a cold rule optimizer in front of relalg (D-SRAM-2 research spike)

**Status:** MEASURED 2026-10-20 (lab spike, not authorization of D-SRAM-2; D-RA-2 still gates any IR ownership). Probe crate `crates/crepe-plan-probe`; crepe fork branch `AdaWorldAPI/crepe` `ccr-d3ce1263-ied9yh` (`9fd5785`).
**Answers:** can Crepe's compiler front end optimize MeTTa-over-MORK queries before relalg without bringing Crepe's runtime into the path? **PARTIALLY — yes for the admitted subset, by a second macro in the fork; no library reuse without a crate split** (§4).

## Overview

The spike tests one hypothesis: most of the physical computation already exists (D-MPS-8 relationship table, relalg, mask-risc), and the missing piece is a small rule compiler that rewrites the question before relalg sees it.

Pipeline, one stage per module of the probe:

| stage | code | what it does |
|---|---|---|
| 1 | crepe fork, `crepe_plan!` | `Context::new` (names, arity, `@input` goals, negation cycles, strata), then crepe's own left-to-right bound/free analysis as data. No `HashSet` store, `HashMap` index or fixpoint loop is emitted. |
| 2 | `logic.rs` | labels to ids through the label table; flat, positive, non-recursive rules over `@input` relations, or a named refusal |
| 3 | `optimize.rs` | binding propagation; duplicate-atom and duplicate-rule elimination gated on duplicate-insensitive answers; the `semijoin` switch read at preparation |
| 4 | `mork_api.rs` | each rule as a typed `query_multi (pattern, template)` (D-MPS-9 shape); nested, repeated-variable and cyclic shapes refused |
| 5 | `native.rs` | a star on one shared variable laid out by the D-MPS-8 table; joint mask = relalg `And` of presence planes, one mask-risc program; terminals read the mask |
| 6 | `reference.rs` | nested-loop evaluator over the unoptimized rules (bag) |

## Checklist

- [x] Phase 0: sources read (crepe `b26a580`, lance-graph `a7c9c6ef`/`010729b9`, MORK oracle `de9312e3` via mork-pathmap-probe, Soufflé `f53dab8`); claim on EXO-50 (`session_01VncKoyG6JkpsoqhqTGTv3f/009`)
- [x] Phase 1: compiler seam — `bind_fields` extracted from `make_clause`, shared by `crepe!` and `crepe_plan!`
- [x] Phase 2: binding propagation
- [x] Phase 3: MORK `query_multi` mapping and admission
- [x] Phase 4: relalg + mask-risc over the D-MPS-8 table; `relalg::run_with_program` (additive) to inspect the program
- [x] Phase 5: differential tests A–M against `crepe!`, datafrog, MORK, reference; 9 disable runs
- [x] Phase 6 (limited): Soufflé harvest table (§3)
- [x] Semijoin reduction before preparation (`Options::semijoin`, `native::prepare_semijoin`; §2b, tests N)
- [x] Stage 3 over Lance 13 / lancedb 0.40: masked read per side, P1–P4 green (§2d); P5 measured at 50,000 rows only (§2e)
- [x] CI workflow for this probe (`.github/workflows/crepe-plan-probe.yml`)
- [x] Arm B, the addressed semijoin over attested lanes (§2e, tests B1–B9)
- [x] ARM-B-PREPARE: per-side read policy (exact ranges, coalesced, full) with a measured switch (§2f, tests B13–B15 and 5 unit tests)

## 1. Integration point

```
crepe_plan! { … }                      crepe fork  src/lib.rs  crepe_plan → Context::new → make_plan
  └─ bind_fields(fact, vars)           shared with make_clause (the crepe! evaluator)
crepe_plan_probe::logic::resolve       labels → RelId / VarId
crepe_plan_probe::optimize::optimize   Demand → Optimized { rules, post_filter, rewrites }
crepe_plan_probe::mork_api::query_multi + admit          QueryMulti { patterns, template }
crepe_plan_probe::native::star → prepare                 StarPlan → PreparedStar (JoinTable::prepare_star)
crepe_plan_probe::native::joint                          witness + JoinTable::check
  └─ lance_graph_relalg::run_with_program(And[Plane(0..k)], Count|Keep)
       └─ FoldDialect → mask-risc Program (And ops only)
```

Coordinate witness: a `PreparedStar` carries the `SideKey` of every side (relation, key column, restriction, arity) and a fingerprint of the shared domain. `joint` refuses a plan whose sides differ (`NoCoordinateWitness`); `graft` refuses a side laid out over another domain (`DomainMismatch`); `JoinTable::check` refuses a stale generation.

## 2. Measured (release, rustc 1.99.0, avx512f, medians of 31)

Query `Answer(17, z)` over `Answer(x, z) <- Parent(x, y), Parent(y, z), Allowed(y); Answer(5, z) <- Sponsor(z);`, set answers, `n` Parent rows. `cargo run --release --example measure`.

| n | opts | optimize ns | map+admit ns | normalize ns | prepare µs | mask ns | terminal µs | rows prepared | rules | answers | parity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | all | 743 | 902 | 702 | 6.2 | 1,295 | 3.7 | 152 | 1 | 12 | ok |
| 100 | none | 270 | 1,325 | 1,008 | 11.2 | 2,244 | 26.5 | 293 | 2 | 12 | ok |
| 1,000 | all | 501 | 538 | 474 | 34.2 | 872 | 2.4 | 1,161 | 1 | 12 | ok |
| 1,000 | none | 178 | 841 | 708 | 62.5 | 1,266 | 192.2 | 2,427 | 2 | 12 | ok |
| 10,000 | all | 489 | 513 | 449 | 489.0 | 896 | 2.5 | 11,286 | 1 | 12 | ok |
| 10,000 | none | 172 | 857 | 698 | 1,128.0 | 1,447 | 2,095.3 | 23,802 | 2 | 12 | ok |
| 50,000 | all | 480 | 514 | 445 | 3,502.9 | 1,104 | 3.6 | 56,286 | 1 | 12 | ok |
| 50,000 | none | 271 | 867 | 718 | 6,910.3 | 1,863 | 11,813.2 | 118,802 | 2 | 12 | ok |

Plan construction (`crepe_plan()`) 164–261 ns, resolution 273–458 ns, independent of `n`. Each mask-risc program has 2 `And` ops (3 planes). Parity: datafrog at every size, nested-loop reference up to 10,000.

Reading:
- **Break-even reuse count is 0.** The rewrite itself costs ~210–470 ns more, but it removes a rule, so mapping and normalization get cheaper by more than that: summed cold cost (optimize + map + normalize) is lower with the rewrites at every size (2,347 vs 2,603 ns at n = 100; 1,439 vs 1,856 ns at 50,000).
- **The win is in preparation and in the terminal, not the mask.** With the constant pushed into the body, the `Parent(17, y)` side is prepared from 8 rows instead of n, and the sponsor branch is never prepared: rows halve. The terminal reads one restricted side instead of enumerating every `x` and filtering: 3.6 µs vs 11.8 ms at 50,000 (≈3,300×). The joint mask itself is ~1 µs either way.
- **Preparation still dominates** (3.5 ms at 50,000) because `Parent(y, z)` is prepared in full. §2b removes that.

## 2b. Semijoin reduction (2026-10-20, same harness)

When a star side is restricted by a constant, every side keeps only the rows whose center all restricted sides have (sorted `Vec<u32>`, binary search). A center outside that set cannot reach the joint mask, so answers and bag multiplicities are unchanged. `no-semijoin` is `Options::ALL` with only the reduction off.

| n | opts | prepare µs | mask ns | terminal µs | rows prepared | parity |
|---|---|---|---|---|---|---|
| 100 | all | 1.9 | 791 | 2.3 | 36 | ok |
| 100 | no-semijoin | 3.9 | 786 | 2.2 | 152 | ok |
| 1,000 | all | 6.8 | 821 | 2.3 | 36 | ok |
| 1,000 | no-semijoin | 33.6 | 868 | 2.3 | 1,161 | ok |
| 10,000 | all | 76.1 | 1,280 | 3.7 | 36 | ok |
| 10,000 | no-semijoin | 614.0 | 902 | 2.5 | 11,286 | ok |
| 50,000 | all | 277.6 | 819 | 2.4 | 36 | ok |
| 50,000 | no-semijoin | 3,473.2 | 1,070 | 3.3 | 56,286 | ok |

- The laid-out rows no longer depend on `n` (36 at every size); preparation at 50,000 falls 12.5×.
- **Preparation is still O(n):** every row is still scanned once to test its center (1.9 → 6.8 → 76 → 278 µs). What became O(k) is the layout (domain, presence planes, payload), not the scan. Removing the scan needs an index by key column on the stored relation, which the in-memory `Store` does not have.
- Mask and terminal are unchanged (~1 µs, ~2–3 µs): the reduction does not change the program, only the domain it runs over.
- Interaction: a reduced table's domain holds only the reduced centers, so re-sealing a side with rows for other centers is refused as `OutsideDomain` (safe; prepare again).

Tests N (4): same answers for set, bag, count and exists with exact prepared-row counts (11 vs 417); no change without a restricted side; an empty restricted side gives 0 rows and no answer; two restricted sides intersect (5 vs 7 rows, bag multiplicity 2 kept). Disables: no reduction → the three reduction tests red; first restricted side only → the intersection test red; filtering by the wrong column → A, B ×2, C, F and both N tests red.

## 2c. The row filter through ndarray (measured, not wired)

The semijoin filter (and the constant restriction) is a scalar test per row: binary search of the row's key in the sorted center set. `ndarray::simd` has the compare (`eq_u32_strided_to_mask` over the row bytes, `eq_u32_to_mask` over a contiguous column, OR-ed with `mask_or_assign`), one pass per needle. `cargo run --release --example row_filter`, 50,000 rows, target-cpu=native (avx512f/bw), medians of 31; all three agree on the kept rows:

| needles | scalar µs | ndarray over the rows µs | ndarray over a contiguous column µs |
|---|---|---|---|
| 1 | 30.9 | 82.7 | 5.4 |
| 8 | 146.4 | 665.9 | 30.5 |
| 32 | 225.3 | 2,892.2 | 114.9 |
| 64 | 278.2 | 4,988.1 | 238.0 |
| 128 | 518.1 | 10,043.3 | 462.0 |

Re-measured 2026-10-20 after the Codex P2 fix on #1632. The contiguous arm used to return the mask; it now collects the kept rows like the other two arms, and the example asserts that its rows equal the scalar test's. Collecting the rows costs at most 1.8 µs per run.

- Over the store as it is (`[u32; 2]` rows, the D-MPS-8 `RelationView` shape) the strided kernel is slower than the scalar test at every needle count: its field loads are scalar by design, and it pays one pass per needle where the binary search pays one pass in total. It is therefore **not** used in `prepare`.
- Over a contiguous column the same compares are 14–22× faster than the strided ones, and they beat the scalar test up to 64 needles in both runs. At 128 needles the two runs disagree: this run 462 vs 518 µs, the first run 485 vs 375 µs. The disagreement is benchmark noise, not a property of the kernels: the scalar time moved by up to 1.8× between runs (17.4 vs 30.9 µs at one needle). The crossover lies near 64–128 needles; pinning it needs a quieter benchmark (more repetitions, pinned CPU). At 1,000 rows the scalar test wins at 128 (10.0 vs 7.2 µs).
- The one-pass alternative, a needle bitmap read through `mask_gather_u32`, is a gather and stays out (2026-10-16 ruling, test H).
- Consequence for the next step: a column-major relation store (one contiguous `u32` lane per column) is what makes ndarray applicable here, and it is the same change the key-column index needs.

## 2d. Stage 3 over Lance 13 (pre-registered 2026-10-20, before any run)

DECISION (operator, 2026-10-20): "ja, bau Stufe 3 mit Lance 13 und CI-Workflow" / "Lance 13 Lancedb 0.40 is mandatory".

The relations are stored column-major in Lance: one table per relation, columns `c0`, `c1` (`UInt32`), written through lancedb 0.40 and read through lance 13. Preparation becomes H14d's stages 2 and 3 per side:

1. Read the one column the address mask needs (the restriction column, or the key column for the semijoin), with `_rowaddr`.
2. Build the address mask `R` through relalg over that column as a `U32` lane: `eq_u32(col, k)` for a restriction, `Or` of `eq_u32(key, c)` for the semijoin centers (lowered by mask-risc onto `ndarray::simd::eq_u32_to_mask`). This works for any number of centers. Until 2026-10-20 it held only 4: relalg lowered `Or(a1..ak)` left-associatively and every atom and every `OR` took a scratch slot of its own, so `k` centers took `2k - 1` slots against `lance_graph_r2il::MAX_SLOTS = 8` (5 centers failed with `ScratchBufferTooSmall`), and above 4 a scalar membership test built the mask. Now `FoldDialect::call_owned` releases the slots an `OR`/`AND`/`NOT` consumed, and relalg, which consumes each subexpression's value exactly once, lowers through it: an `Or` of any width needs 3 slots, and the scalar fallback is removed. This is not the §2c speed crossover, which is ndarray's row filter.
3. Read the side's columns with `with_row_addr_prefilter(R)`.
4. Hand those rows to `JoinTable::prepare_star`, as the in-memory path does.

A side's generation is its Lance table version.

Predictions, each with its falsifier:

- P1. Answers equal the in-memory path and the nested-loop reference on every existing fixture (A, B, C, F, J, N) for set, bag, count and exists. Falsifier: any difference.
- P2. For every masked read, Lance's `rows_scanned` equals `|R|` exactly, and the rows returned equal `|R|`. Falsifier: any read that scans more rows than `R` admits.
- P3. Rows handed to `prepare_star` equal the in-memory semijoin path's `rows_prepared` exactly (11 and 5 on the N fixtures). Falsifier: any difference.
- P4. A write that adds a table version makes a table prepared under the old version stale (`JoinRefusal::Stale`). Falsifier: the old table is accepted.
- P5 (measured, not predicted as a number): bytes read and wall time per stage at 1,000 / 10,000 / 50,000 rows, against reading every row. No direction is predicted for bytes: H14d found bytes follow page locality, not `|R|`.

Disables: build `R` from the wrong column (P1, P2 red); drop the prefilter (P2 red); skip the version check (P4 red).

## 2e. Arm B: the addressed semijoin (measured 2026-10-20)

DECISION (operator, 2026-10-20): "They are ordered by design. The only issue is the relationship maps proposed earlier pulling in global to local which would make the nodeguid order unusual. Build arm b".

A relation is written in key order: row `(a, b)` carries `FacetCascade::from_semantic_tiles([0, 0, a_hi, a_lo, b_hi, b_lo, 0, 0])`, which compares as `(a, b)`. The probe never orders a lane itself. `addressed::attest` reads the keys once and calls `SealedFacetLane::attest_sorted`, which refuses an unsorted lane. A side then reads only the lane ranges of its demanded keys, located with `seek_from` under the lane's `OrderedLaneWitness`, through `with_row_addr_prefilter`. The arm adds no mask column, index, opcode, gather or global-to-local mapping.

The arm refuses, by name:
- `NotLeadingColumn`: a restriction or demand on the second column (that would need a second lane, ordered by that column);
- `Stale`: a table that moved on since its lane was attested;
- `NotSorted`: an unsorted table, at attest;
- `NotContiguous`: a table split over more than one fragment, at attest. An append adds a fragment, so after one the table must be compacted before it can be addressed again.

Measured with `cargo run --release --example addressed` (lance 13, lancedb 0.40, rustc 1.99.0). The query is `Answer(17, z)` over `family`. There are N = 50,000 background edges with distinct `y`, all `Allowed`, and `k` children of 17 spread over them, so there are exactly `k` centers and `k` answers. Times are medians of 9 runs. "Rows" counts rows Lance scanned, mask columns included.

| k | full ms | Arm A ms | Arm B warm ms | attest ms | full rows | Arm A rows | Arm B rows | break-even |
|---|---|---|---|---|---|---|---|---|
| 0 | 9.81 | 4.46 | 0.37 | 7.28 | 150,000 | 150,000 | 0 | 2 |
| 1 | 9.95 | 5.02 | 2.19 | 6.83 | 150,003 | 150,005 | 3 | 3 |
| 8 | 12.30 | 6.60 | 2.24 | 6.67 | 150,024 | 150,040 | 24 | 2 |
| 500 (1%) | 10.42 | 8.86 | 3.47 | 6.71 | 151,500 | 152,500 | 1,500 | 2 |
| 5,000 (10%) | 11.70 | 15.02 | 12.11 | 7.81 | 165,000 | 175,000 | 15,000 | 3 |
| 25,000 (50%) | 18.55 | 41.39 | 34.54 | 7.74 | 225,000 | 275,000 | 75,000 | 2 |
| 50,000 (all) | 30.71 | 36.15 | 38.98 | 9.84 | 300,000 | 400,000 | 150,000 | never |

Break-even is `ceil(P / S)`, where `P` is the attest time and `S` is Arm A's per-query time minus Arm B warm's.

What this says:
- **Source-row reads do go down, by construction, and exactly.** Arm B scans `3k` rows: one row per demanded key on each of the three sides. Arm A scans at least `150,000`, because each side reads its key column in full to build the mask. The answer to "does demand-aware semijoin reduction decrease source-row reads" is yes for Arm B and no for Arm A. Arm A reduces the rows laid out (§2b), not the rows read.
- **Time follows the rows only at low selectivity.** Up to 1% of centers, Arm B is 2.5–12× faster than Arm A and the attest pays for itself after 2–3 queries. Its floor is about 2 ms for three masked reads, even for 3 rows; with no centers it reads nothing and takes 0.37 ms.
- **From 10% of centers up, every masked read loses to a full scan.** Arm A is already slower than the full read at 10%, and both arms are slower at 50%. With all centers, Arm B never breaks even. Many small ranges cost Lance more per row than one sequential scan. A planner must switch to the full read above a crossover between 1% and 10%; that crossover is not measured more finely here.
- `P` (reading and attesting 100,002 keys over three tables) is 7–10 ms: a cold cost the size of one full query.

P5 is measured at 50,000 rows only; 1,000 and 10,000 were not run.

## 2f. ARM-B-PREPARE: when Arm B stops reading ranges (measured 2026-10-20)

Ledger (d) item ARM-B-PREPARE: Arm B removes the O(n) preparation scan only below about 1% of centers; above that, its many short ranges cost Lance more per row than one sequential read (§2e).

Each side now chooses its read with `addressed::plan_read`, a pure function of the located lane ranges and the table size (it reads no row):
- `Read::Ranges`: the located ranges (ranges that touch are merged and still count as exact);
- `Read::Coalesced`: ranges at most `coalesce_gap` rows apart are merged, and the extra rows are dropped after the read by the row's own leading value (co-addressed: no position is taken from another lane's values);
- `Read::Full`: once the merged ranges cover `full_permille` of the table, the whole table is read without a row filter and the same drop applies.

`ReadPolicy::EXACT` keeps the old behaviour and is still what `evaluate_addressed` uses; `evaluate_addressed_with` takes a policy. `ReadPolicy::DEFAULT` = gap 16, full at 100 permille.

Measured with `examples/addressed_read.rs` (release, rustc 1.99.0; same fixture as §2e: uniform demand, one row per center; every strategy checked against the answer count before timing). Times in ms.

50,000 rows (median of 9; 1,000–3,000 re-run with 31):

| k | exact | gap 16 | full |
|---|---|---|---|
| 0 | 0.40 | 0.36 | 2.98 |
| 500 (1%) | 2.72 | 3.20 | 4.09 |
| 1,000 (2%) | 4.55 | 4.06 | 6.50 |
| 2,000 (4%) | 5.66 | 5.10 | 5.48 |
| 3,000 (6%) | 6.53 | 6.58 | 5.79 |
| 5,000 (10%) | 8.87 | 7.27 | 7.46 |
| 25,000 (50%) | 35.10 | 17.97 | 19.06 |

200,000 rows (`READ_N=200000`, median of 15):

| k | exact | gap 16 | gap 64 | full |
|---|---|---|---|---|
| 1,000 (0.5%) | 5.11 | 5.66 | 5.41 | 12.09 |
| 2,000 (1%) | 6.94 | 7.62 | 11.70 | 18.08 |
| 10,000 (5%) | 19.82 | 18.11 | 20.13 | 22.15 |
| 14,285 (7%) | 22.22 | 21.40 | 20.72 | 23.21 |
| 100,000 (50%) | 128.48 | 74.29 | 75.62 | 79.37 |

What this says:
- The crossover is not a fixed fraction of the table: exact ranges win up to about 3% at 50,000 rows and about 5% at 200,000, and tie with the full read at 4–6% and 7–10% respectively. Between the two, the choice is within run-to-run noise (an identical policy measured 3.9 and 6.1 ms at k = 1,000 in one run).
- Above 10% the full read wins: 1.9× at 50% of 50,000 rows, 1.6–1.7× at 50% of 200,000. DEFAULT switches at 10%, which is never measurably worse on either size.
- A full read has a floor (3 ms at 50,000 rows, 6.7 ms at 200,000), so below the switch it would lose up to 3×.
- Coalescing measured neutral at gap 16 and harmful at gap 64 (11.7 vs 6.9 ms at 1% of 200,000 rows) on uniform demand. Clustered demand, where it should help, is not measured.
- Two table sizes only; the crossover's dependence on size is observed, not modelled.

Tests: `addressed::tests` (5: gap boundary, touching ranges, full-read threshold at the exact row, coalesced rows counted toward it, EXACT/FULL extremes); `arm_b::b13` (every policy answers as EXACT, every kind, hit/sponsor/miss), `b14` (the switch fires on dense demand and stays silent on sparse, per side; a full read scans more rows), `b15` (coalescing hands Lance fewer ranges, answer equal to the memory path). Disable runs, each red: no post-read drop (b13, b14), strict gap (3 unit tests), never full (3 unit tests and b14), never coalesce (3 unit tests and b15), touching ranges counted as coalesced (1 unit test). B15 does not pin the drop: undemanded rows find no join partner there.

## 3. Soufflé harvest (souffle-lang/souffle `f53dab8`, UPL-1.0; algorithms only, no source copied)

| Soufflé | transformation | precondition | in crepe | in lance-graph | taken |
|---|---|---|---|---|---|
| `src/ast/transform/MagicSet.cpp` `AdornDatabaseTransformer::adornClause` (l. 624) | bound head fields mark variables (weakly) bound; adorned rule per binding pattern | query binding pattern; head constants are normalised to variables first (`NormaliseDatabaseTransformer::normaliseArguments`, l. 510) | no (crepe evaluates bottom-up, no demand) | no | **yes, the adornment step only**: `optimize::specialize` unifies the head with the demand, drops a rule on a constant conflict, substitutes bound variables into the body. No magic predicates (no recursion in scope) |
| `MagicSet.cpp` `MagicSetCoreTransformer::transform` (l. 1139) | magic predicates carry the demand through derived relations | recursion or derived body relations | no | no | no — follow-up B |
| `MinimiseProgram.cpp` `reduceClauseBodies` (l. 375) | drop a body literal equal to an earlier one | Soufflé relations are sets | no | relalg drops duplicate planes after lowering | **yes, gated**: only for `Exists` / set answers; under bag semantics a repeated atom multiplies derivations (test C) |
| `MinimiseProgram.cpp` `removeRedundantClauses` (l. 349) + `areBijectivelyEquivalent` (l. 203) | drop a clause equal to another under a variable bijection and body permutation | set semantics | no | no | **partly, gated**: equality after first-occurrence renumbering; body permutations are not searched (§5) |
| `ast2ram/utility/SipsMetric.h` | literal ordering by bound arguments | — | crepe keeps written order | the star lowering ANDs all planes, order-free | no — nothing to order in a star |
| SCC / stratification | evaluation order, negation check | — | `strata.rs` (petgraph Kosaraju) | — | reused through `crepe_plan!` |

## 4. Why PARTIALLY

- crepe is a `proc-macro = true` crate: it can only export macros. `Context`, `Strata` and the binding analysis cannot be called from another crate as a library. The spike therefore adds a second macro to the fork; the plan is fixed at Rust compile time.
- Dynamically arriving MeTTa rules need the analysis as a library: split the fork into `crepe-core` (parse to typed rules, `Context`, `Strata`, `bind_fields`, with `abort!` replaced by `Result`) and the two macros on top. Not needed for the static proof; filed as follow-up A.
- The binding analysis is shared, not copied: disabling bound-variable recognition in `bind_fields` turns both the plan test and `crepe!`'s own transitive-closure tests red.

## 5. Tests and disables

20 tests (`tests/pipeline.rs`: A–M of the brief plus N for the semijoin, §2b), all green. Every guard turned its tests red under a disable run (made after commit, restored after):

| disable | red |
|---|---|
| ignore a head-constant conflict | A, B, C, F |
| dedup without the set gate | C |
| skip the domain fingerprint | D |
| skip the coordinate witness | D |
| AND only two of three planes | A, F, I |
| let a nested pattern through | G |
| foreign-hop detector always false | H |
| prepare without the constant restriction | A, B ×2, C, F |
| cap bag multiplicity at 1 | C, F |

Turning the rewrites off (`Options::NONE`) changes plan shape and cost only; tests A and F compare both settings.

Test M (H14d absorption): `normalize(¬(x ∧ (x ∨ y)))` reduces correctly when `x` is a plane or a comparison and misses when `x` is itself an `And` or `Or` — narrower than the H14d note states. Pinned, not fixed; the spike never emits `Not` or `Or`.

Known limits: rule dedup does not search body permutations; the star must have one shared variable and binary or unary atoms; chains are refused by the native star (MORK's shape admits them); the label table stands in for the label codebook.

## 6. Follow-ups (not built)

- ~~Semijoin reduction before preparation~~ — built, §2b. Layout O(k); the scan stays O(n).
- ~~Addressed read~~ — built, §2e. Reads O(k) rows over attested lanes.
- ~~Selectivity crossover~~ — built, §2f. Measured at 50,000 and 200,000 rows; `ReadPolicy::DEFAULT` switches to the full read at 10% coverage. Clustered demand and a size-aware switch remain open.
- Multi-fragment lanes: map lane positions to row addresses per fragment, so that an append does not force a compaction before Arm B can read the table again.
- **Next (recommended):** a column-major relation store (contiguous `u32` lanes). It lets the restriction and semijoin filters run through `ndarray::simd::eq_u32_to_mask` (§2c: 4–6 µs per needle at 50,000 rows instead of a scalar or strided scan), and it is the layout a key-column index would sit on, which then removes the O(n) scan.
- A: `crepe-core` library split in the fork for dynamic rules.
- B: full magic sets for derived body relations and recursion (waits on D-MTP-3 / D-SRAM-4).
- C: cost-informed ordering once a non-star shape is admitted.
- D–G as in the brief (recursion, 115-strategy corpus, 0+63 thought reuse, richer MORK patterns).
