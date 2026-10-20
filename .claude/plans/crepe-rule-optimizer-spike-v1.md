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
| 3 | `optimize.rs` | binding propagation; duplicate-atom and duplicate-rule elimination gated on duplicate-insensitive answers |
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
- [ ] Follow-up (not built): semijoin reduction of the unrestricted sides before preparation (§6)

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
- **Preparation still dominates** (3.5 ms at 50,000) because `Parent(y, z)` is prepared in full. That is the next rewrite (§6), not a physical gap.

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

16 tests (`tests/pipeline.rs`, A–M of the brief), all green. Every guard turned its tests red under a disable run (made after commit, restored after):

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

- **Next (recommended):** semijoin reduction before preparation — restrict the unrestricted sides of a star to the center values the restricted side has (here: prepare `Parent(y, z)` only for `y ∈ {children of 17}`). Same answers, preparation from O(n) to O(k). Needs a demand-aware layout step, not a new mask op.
- A: `crepe-core` library split in the fork for dynamic rules.
- B: full magic sets for derived body relations and recursion (waits on D-MTP-3 / D-SRAM-4).
- C: cost-informed ordering once a non-star shape is admitted.
- D–G as in the brief (recursion, 115-strategy corpus, 0+63 thought reuse, richer MORK patterns).
