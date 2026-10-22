# eggcc 2026: effect-safe extraction is a scheduling proof obligation

**Primary source:** Oliver Flatt, Anjali Pal, Yihong Zhang, Ryan Tjoa, Kirsten Graham, Alex Fischman, Chandrakana Nandi, Eli Rosenthal, Zachary Tatlock, and Haobin Ni, *Efficient Extraction for Effectful E-graphs*, OOPSLA2 2026, Article 398; [author PDF](https://ztatlock.net/pubs/2026-oopsla-eggcc/2026-oopsla-eggcc.pdf), [DOI 10.1145/3839530](https://doi.org/10.1145/3839530). Read against the 28-page paper on 2026-10-22. Sections cited below refer to that paper. The author-hosted copy identifies its license as CC BY 4.0. This note paraphrases rather than reproduces the paper.

**Status:** PRIMARY-SOURCE RESEARCH HARVEST; conceptual mapping to lance-graph, not a checked LG implementation or performance result.

## Source-grounded idea harvest

1. **An e-graph can represent semantically related computations that cannot all be emitted as imperative programs (§§1–4).** Congruence permits substituting an equivalent child, but an effectful expression may then contain conflicting uses of state. In particular, two writes/updates operating from the same input state may not have one unambiguous legal ordering. Equality of pure returned values alone does not validate the resulting sequence of effects.
2. **Thread explicit state through effects (§§3–4).** The paper's intermediate representation links an effectful operation's input state to its output state, so a valid extracted program imposes a single effect order. Some terms inside the enlarged e-graph remain deliberately unextractable.
3. **Statewalk DP groups orders by what can be safely extracted (§§3, 6).** Two statewalks can be grouped when they enable the same sets of pure e-classes. Instead of enumerating every order, dynamic programming keeps low-cost candidates per extractable-set equivalence class.
4. **Hard general case, small practical parameter (§§5–6).** Finding *any* effect-safe extraction is NP-complete in general, not only finding a cost optimum. The paper gives an O(k n²) fixed-parameter bound for finding an effect-safe extraction, with k the **statewalk width** and n the representative-term count; cost minimization adds a logarithmic factor. It is not an unconditional polynomial bound independent of k.
5. **Control flow is regionalized (§7).** eggcc uses an RVSDG-like scoped representation; a region is extracted separately, treating nested regions as placeholders and recursively filling them. This matters when conditionals/loops otherwise have multiple effectful children.
6. **Results measure extraction, not mask-RISC (§8).** Authors report 520× faster extraction than a Gurobi ILP approach on their evaluated workload and code quality broadly comparable to LLVM. Across 9,610 regionalized e-graphs, statewalk DP averages 0.000426 s per region versus 0.965 s for ILP with 234 ILP timeouts. 94.7% of regions have width 1; 98.4% have width <=5. There are severe high-width exceptions, requiring liveness and associativity/commutativity mitigation. Whole-program optimization can still take substantial compile time (raytracer example); do not confuse extraction-only gains with end-to-end execution speed.
7. **Independent effects are not magically commutative (§9).** A refined model that recognizes independent effect categories is identified as future work, not an already-proved universal reordering license.

## Transfer hypothesis: a cold extractor, not a new executor

The useful pattern for lance-graph is a strict split:

```text
typed relalg / D-CMR / R2IL expression candidates
    -> egglog equivalence exploration (pure, lawful rewrites)
    -> extraction constrained by effect/order witnesses (eggcc lesson)
    -> typed R2IL FoldDialect procedures / Quack selection
    -> mask-risc execute_into (unchanged hot executor)
    -> externally checked result and, if permitted, separate commit
```

**Pure candidates:** associative/commutative/idempotent Boolean set folds only within their lawful domain; typed constant simplifications; shared predicate DAGs; fold fusion under a declared lane/domain/validity version. *None* grants SQL bag-idempotence, NULL=FALSE, numerical reassociation, arbitrary foreign-row co-addressing, or an unproved commutation across source versions.

**Candidate effects to model, if a real compiler IR is introduced:** `BIND` (pins focus/coordinate/source versions), `READ` (source access, including counted same-table seeks), `MATERIALIZE` (observable scratch/lineage when externally consumed), `EMIT` (produces ordered output or multiset), `WRITE`/`COMMIT` (durable state and authorization). This classification is an LG *proposal*, not the eggcc paper's instruction set. Scratch-local operations may be modeled pure *only* when aliasing, visibility and lifetime prove them unobservable.

**Existing owner boundary:** `lance-graph-contract` owns admitted single-table set laws (D-RA-2=A); `lance-graph-relalg` owns planning/lowering; R2IL is the SIMD polyfill; Quack routes to mask-risc `execute_into`. The existing `.claude/plans/d-relalg-unified-v1.md` distinguishes proposed `let`-planes, joins and recursion; do not upgrade those proposals to shipped functionality. `ADMISSION=B` allows counted/version-attested same-table seeks during preparation, not value-directed cross-table positional gathers. Effect-safe extraction does not relax that rule.

**Opportunity:** mathematical rewriting should explore large equivalence classes once in a **cold, bounded planning phase**, then extract small, version-pinned *executables*. For repeated 256-rule/4096-frame use, amortize compilation over snapshots; measure the setup/extraction/replay overhead, not just the number of algebraic instructions eliminated.

## Small falsifier set (hypothetical LG probes)

| ID | Independent oracle and fixture | Invariant / red line |
|---|---|---|
| CC1 state ordering | Three stateful writes, with two aliasing addresses and two non-aliasing ones; standalone sequential interpreter | Rewrites that choose ambiguous or different observable write orders must refuse extraction. |
| CC2 version barrier | A bound coordinate snapshot followed by source reorder, deletion, or generation change | An extracted fold cannot reuse a stale binding. `BIND[v1]; EXEC[v2]` refuses, even if ordinal values coincide. |
| CC3 pure/effect split | Same Boolean predicate repeated under one pinned input and again after mutation | CSE before mutation is lawful; CSE across the mutation is not, unless read independence has been proven. |
| CC4 NULL, K3, multiplicity | Valid FALSE, invalid/NULL, K3 UNKNOWN, duplicate bag rows | No extraction silently collapses three truth situations or changes `COUNT`/Bag into Distinct/Boolean existence. |
| CC5 cold economics | 256 reused rules × 4096 frames plus deliberately unshared controls | Report compiler time, e-graph/effects width, extracted operations, bind reads, scratch planes, execute wall time, peak memory and end-to-end cost. No claimed acceleration without a measured total win. |
| CC6 write barrier | Equivalent computation with and without durable store acknowledgement | An algebraic proof cannot assert `CommittedDirty` or write success without actual authorized commit acknowledgement. |

## Decision

**Harvest, do not integrate yet.** The smallest next step is one bounded, source-versioned *offline extraction* fixture on a pure R2IL/relalg subset and one tiny state-order counterexample. Implementing RVSDG or importing eggcc wholesale is not licensed by the paper. A successful result is a typed, replayable procedure with refusal diagnostics; the current hot mask-RISC executor remains the sole execution path.

**Cross-references:** [egglog primary harvest](2304.04332_egglog_equality_saturation_datalog_excerpt.md); [3LA formal interface harvest](2026-10_3la_formal_interface_flexible_matching_application_validation.md); [relalg plan](../.claude/plans/d-relalg-unified-v1.md); [QR256 falsifiers](2026-10_qr256_gaussian_egglog_rotation_falsifier_paths.md).
