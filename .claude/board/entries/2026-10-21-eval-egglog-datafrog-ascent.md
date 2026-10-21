# egglog, datafrog and Ascent as complements to `lance-graph-relalg` — evaluation (2026-10-21)

**Status:** OPEN. This is an exploration map, not a ratification. Nothing in it is measured, and no code was written. The scope is a complement to `lance-graph-relalg`; replacing it is not on the table.

**Update 2026-10-21 (operator decision on oracle scope).** DECISION: *"It's not smart to compare tuples they are a million times more expensive."* SCOPE: every oracle in this workspace, including reference engines used only in tests. BASIS: the 2026-10-21 P0 rule, which already said "answers are compared by folds", now applied to the reference side too. Consequences for this entry:

- **Role (b) is withdrawn for datafrog and Ascent.** Both hold their relations as owned tuples (§2), so neither can serve as an oracle. P-D1, P-A1 and P-A2 are SUPERSEDED.
- **The replacement oracle is an independent fold.** For D-RFT-3 / D-TGB-10 recursion: a second closure algorithm over the same mask planes (for example Warshall over a ≤256 tile, against iterated frontier masks), compared terminal against terminal. Independence comes from the algorithm, not from a second data model. This is a working model, not yet a probe.
- **P-E1 keeps its egglog half** (offline, counts only), but its Kleene check now runs against the mask oracle: the per-row `set_at` it names was replaced by `set_planes`, which evaluates (T, F) masks.
- Done in the same change: the three Kleene oracles (`k3_pg_parity`, `set_algebra` tests, `set_rules` tests) now evaluate whole TRUE/FALSE masks and count by popcount or fold. Each was disable-checked: a two-valued NOT turns its tests red.

**Evidence grades.** Claims about our own tree are marked `VERIFIED-IN-CODE`, with `file:line`. Claims from outside use these grades:

- **READ-IN-FULL:** read completely.
- **SECTION-READ:** the named sections were read.
- **ABSTRACT-ONLY:** only the abstract was read.
- **SECONDHAND:** taken from a summary or a third party.

Three sources gave the outside evidence:
- the source trees of the three AdaWorldAPI forks;
- the text of the CC'22 Ascent paper;
- the text of the PLDI'23 egglog paper (arXiv 2304.04332).

The OOPSLA'23 BYODS paper was **not read**: the ACM DL returned HTTP 403. Every BYODS claim below therefore rests on the fork's `BYODS.MD` (READ-IN-FULL) and its trait sources.

## 0. What relalg is today (the thing being complemented)

- **The IR.** `Rel` is `All`, `Empty`, `Plane`, `Atom`, `And`, `Or` or `Not` over ONE table. It holds no strings and no rows (`crates/lance-graph-relalg/src/ir.rs:46-62`). `Measure` is `Count`, `Sum`, `Min`, `Max` or `Keep` (`ir.rs:98-110`). VERIFIED-IN-CODE.
- **Normalization.** `normalize` converts to the contract's `SetExpr` and delegates to `set_algebra::normalize` (`src/normalize.rs:23-25`). `symbolic` asks `set_rules::rewrite` under `Semantics::exact(n_rows)` whether the measure folds to a constant (`normalize.rs:40-58`). VERIFIED-IN-CODE.
- **Lowering.** Lowering emits R2IL calls over `FoldDialect`, giving one mask-risc program per fold (`src/lower.rs:104-149`). `lower_rel` never loops over rows (`lower.rs:178-194`). VERIFIED-IN-CODE.
- **`set_rules`.**
  - It has 8 guarded rules (`lance-graph-contract/src/set_rules.rs:236-310`).
  - `Semantics` carries `{universe, arith, truth, sums, population}` (`:74-81`).
  - The driver accepts only a strict decrease in size (`:379`).
  - `SetNormalize` uses `normalize_kleene` under `Truth::ThreeValued` (`:444-447`).
  - The independent oracle is a test-only Kleene evaluator over a materialized table (`:616-768`).

  All VERIFIED-IN-CODE.
- **Not built.** Joins (D-RA-3/4) and recursion (`μ`, D-MTP-3) do not exist. `join_body::is_alpha_acyclic` only classifies the shape of a join body (`src/join_body.rs:21`). VERIFIED-IN-CODE. This is the gap a complement could cover.
- **Prior rulings that bind this evaluation:**
  - **X14 (egglog).** egglog is SKIP as a bind- or query-time planner. It is allowed only for offline normal-form generation (`entries/2026-10-21-coresearch-classview-algebra-polyfill.md:127-130`).
  - **datafrog is a SET oracle.** `entries/2026-10-16-coresearch-wcoj-leapfrog.md:83-84`.
  - **D-CMR-8.** Equality saturation is "justified only if a bounded local rewriter measurably commits to worse forms" (`plans/d-cmr-concept-chain-normalization-v1.md:174-177, 220`).
  - **D-RFT-2.** egglog may be only an offline proof or normal-form oracle (`lg-1667/.claude/plans/d-rft-…-v1.md:18`).
  - **D-RFT-3.** datafrog is the recursion reference (`:19`). NARS revision is not presumed to be a semiring.
  - **D-TGB-10.** The folded rules are checked against datafrog, and recursion stays on datafrog until D-MTP-3 (`plans/thought-git-basics-v1.md:247-248, 168`).
  - **CLAUDE.md P0 (2026-10-21).** No millisecond results. No tuple decoding, *including in oracles*: answers are compared by folds. mask-risc is the reference fold algebra. Use AdaWorldAPI forks only.

## 1. Fork status, licence, build cost

| | fork / checkout | version | licence | form | dependency weight | already used in the tree |
|---|---|---|---|---|---|---|
| **datafrog** | AdaWorldAPI/datafrog (`~/.cargo/git/checkouts/datafrog-…/edd7cfc`) | 2.0.1 | Apache-2.0/MIT (`Cargo.toml`) | plain runtime library, about 2 kLOC, **zero dependencies** (dev: proptest) | negligible | dev-dependency of `lance-graph-metta` (`Cargo.toml:20`) and `crepe-plan-probe` (`Cargo.toml:44`) |
| **Ascent** | AdaWorldAPI/ascent (`~/.cargo/git/checkouts/ascent-…/f3634e3`, fork PR #1, edition 2024, toolchain 1.99.0) | 0.8.1 | MIT | **proc macro** (`ascent!`, `ascent_run!`, `ascent_par!`) plus runtime | runtime: hashbrown, rustc-hash, boxcar, cfg-if; the default feature `par` adds dashmap and rayon. Macro: syn, quote, itertools, **petgraph 0.8**, derive-syn-parse | dev-dependency of `crepe-plan-probe` (`Cargo.toml:46`), the second oracle in `examples/walk_mask.rs:92-151` |
| **egglog** | AdaWorldAPI/egglog (`(AdaWorldAPI fork) egglog`, origin AdaWorldAPI/egglog, HEAD `4b5686b`) | 3.0.0 | MIT | runtime library plus a CLI (default feature `bin`) | **heavy**: 8 workspace crates; core-relations brings crossbeam, dashmap, bumpalo, rand, num, fixedbitset; the main crate brings im-rc, indexmap, csv, serde_json, egraph-serialize, ordered-float. Toolchain pin is **1.91.0** against the workspace's 1.99 | not a dependency anywhere |

Licences: all three are permissive. Code is borrowed from none of them; only concepts are, as below.

**AdaWorldAPI forks: all three exist**, so the forks-only P0 rule can be satisfied. Each must be wired by `git` or `path` with `branch = "master"`, never from crates.io.

## 2. What each engine actually does (first-hand)

### datafrog

**Data model.**
- `Relation<Tuple> { pub elements: Vec<Tuple> }` holds a sorted, deduplicated `Vec` (`relation.rs:15-17`). `from_vec` sorts and then deduplicates (`:78-80`). It is a **set of owned tuples**. READ-IN-FULL.
- `Variable` keeps `stable`, `recent` and `to_add` relations. `changed()` merges these and, for a distinct variable, removes tuples already in `stable` using a gallop (`variable.rs:339-395`). This is semi-naive evaluation driven by the caller's `while iteration.changed()` loop (`iteration.rs`). READ-IN-FULL.
- `variable_indistinct` exists. It is documented as having *no termination guarantee*, so it does not support bag semantics.

**Operators.**
- `from_join` (sort-merge join on key; `join.rs:12-137`, `gallop` at `:137`).
- `from_antijoin`, whose **right input must be a fixed `Relation`** (`variable.rs:194-201`). Negation is therefore expressible only against a stratum the caller has already completed; the caller stratifies by hand.
- `from_map`.
- `from_leapjoin`: one dynamic source plus fixed `Leapers` (`extend_with` / `extend_anti` / `filter_*`). The smallest proposer proposes and the others intersect (`treefrog.rs:6-45`). The result is collected into a `Vec` and turned into a `Relation`.

**Not present:** aggregation, lattices, NULL, multiplicities.

### Ascent

**Evaluation** (CC'22 §4, SECTION-READ):
- the rule dependence graph is split into SCCs, evaluated in topological order;
- each dynamic relation is split into delta, total and new for semi-naive evaluation;
- indices are selected from the join columns;
- rules compile to **nested loops over hash indices, binding one tuple at a time** (the paper's generated-code listing for `tc`);
- clauses may be reordered when grounding allows.

**Storage.** The default provider stores a relation as `Vec<$field_types>`, plus index fields (`ascent/src/rel.rs` macro `rel!`). READ (the macro).

**Lattices** (CC'22 §3.1, SECTION-READ; trait `ascent_base/src/lattice.rs:17-41`):
- In a `lattice` relation, the last column must implement `Lattice` (`meet`/`join`).
- Its meaning is a partial map from the non-lattice columns to the lattice column. Conflicting facts are joined.
- **Correctness presumes a true join-semilattice:** idempotent, commutative and associative.
- `Dual<T>` flips the order, which is how shortest paths are written.

**Negation and aggregation.**
- Stratified negation and aggregation are built in.
- Aggregators are library functions: `sum`, `min`, `max`, `count`, `mean`, plus user-defined ones (README; CC'22 §3.2).
- Aggregation runs over the set of matching tuples.

**BYODS** (`BYODS.MD` READ-IN-FULL; `ascent/src/rel_index_read.rs:11-30`):
- A relation can be backed by a provider: `#[ds(provider)]` macros `rel_ind_common!` / `rel_ind!`.
- A provider implements `RelIndexRead::index_get(key) -> Option<Iterator<Value>>`, `RelIndexReadAll::iter_all`, `RelIndexWrite` and `RelIndexMerge`.
- **Only the storage and the index are replaced.** The rule body is still Ascent's generated nested-loop join, which binds per value and writes head tuples one at a time.

**Parallelism.** `ascent_par!` runs on rayon.

### egglog

Paper: PLDI'23 §1 and §3-4 SECTION-READ. Source tree read in part.

**Model.** Datalog over functions with a functional dependency, plus a union-find, rebuilding (congruence) and extraction.

**`:merge` resolves conflicting outputs.** The source states that it "must be monotonic … must define a lattice … If values are merged in different orders, they should still result in the same output" (`src/ast/mod.rs:742-748`; the paper's lines on `:merge` and lattices).

**Rule bodies have no negation.** `GenericFact` is only `Eq | Fact` (`egglog-ast/src/generic_ast.rs:35-38`). READ.

**Rewrites.** `rewrite` / `birewrite` take `:when` guards and `:subsume` (`src/ast/mod.rs:845-891`).

**Schedules.** Rulesets run under `saturate`, `repeat` or `seq` (`:383-386`).

**Proofs.** There is a `prove` command (`:944`) and a term encoding with proof tracking (`src/proofs/`, `proof_encoding.md` read §1).

**Extraction.** Extraction uses a default or custom cost model (`src/lib.md`, READ-IN-FULL).

**Other sorts.** A `multiset` sort exists as a *value* sort (`src/sort/multiset.rs`, existence only). Relations themselves are sets.

**Semantics.** egglog is defined by an inflationary immediate-consequence operator. Semi-naive evaluation is proven equivalent (Theorem 4.1). Non-monotone `:merge` is allowed but its result is order-dependent (paper §4).

### Shared property that decides role (c)

**All three materialize owned tuples:**
- datafrog keeps `Vec<Tuple>` in every `Relation` and `Variable`;
- Ascent keeps `Vec<tuple>` plus index maps, and emits head tuples per binding even under BYODS;
- egglog keeps tables of rows and e-classes.

Under the P0 rule a tuple violates four things: zero copy, folding, natural order and time. All three engines therefore fall outside the substrate's runtime by construction. That is a property of each engine's data model, not a tuning matter.

## 3. Per-stratum semantics

| stratum | datafrog | Ascent | egglog | the substrate today |
|---|---|---|---|---|
| **Boolean / Distinct** | yes: sets by construction | yes: set union for relations (CC'22 §3.1) | yes: unit-valued functions | yes: masks; `Population::Distinct` |
| **Bag / ℕ** | **no.** `from_vec` dedups; `variable_indistinct` gives up termination. A bag needs an explicit multiplicity or witness column (the X17 proposal) | **not natively.** `count`/`sum` aggregate over a *set* of tuples, so a bag needs a witness column too; Σ multiplicity is then expressible with `agg sum` | **no** for relations. `multiset` is a value sort, not relation semantics | partly: `Population::Bag` refusals (`set_rules.rs:463-466, 580-582, 596-598`); join fan-out is not built |
| **K3 NULL** | no | no native support. It could be encoded with `Option<T>` or a 3-element lattice, but the information order and the truth order differ, which invites error | no | **already has an independent oracle:** the Kleene `set_at` evaluator (`set_rules.rs:633-675`) and `normalize_kleene`. None of the three engines adds anything here |
| **Recursion + stratified negation** | **yes, by hand:** antijoin against a completed `Relation`; the caller orders the strata | **yes, built in:** SCC strata, with negation and aggregation across strata | **no negation** in bodies | not built (D-MTP-3); the `walk_mask` lab shows the mask form `new &= !walk; walk |= new` |
| **Lattice-valued** (min-plus, intervals) | no | **yes:** `lattice` plus `Dual<u32>` gives shortest paths directly (CC'22 §3.1) | **yes:** `:merge (min old new)` (`tests/interval.egg` style) | min-plus: OPEN (probe P2 in the polyfill entry) |
| **NARS truth (revision)** | n/a | **disqualified while revision is not idempotent.** A lattice join requires `x ⊔ x = x`. Revising the same evidence twice raises confidence, which is exactly the "duplicate proof becomes new evidence" error D-RFT-5 forbids | **same:** `:merge` must be order-independent and monotone; otherwise behaviour is undefined (`ast/mod.rs:742-748`) | NARS is a finisher only (D-CMH-1); the semiring claim is OPEN (P6) |

Consequence for NARS: neither engine can be the evaluator or the oracle for revision. Their own documented preconditions are the same question probe P6 already asks (associativity, commutativity, idempotence on a quantized grid). A property test answers it, and no engine is needed for that.

## 4. Role by role

### Role (a) — offline proof / normal-form generator for `set_rules`

**egglog fits this role, and only egglog does.** It has:
- guarded rewrites (`:when`);
- per-stratum rulesets: one ruleset per `Semantics` stratum, e.g. a two-valued Boolean algebra ruleset and a Kleene ruleset that leaves out complement and excluded-middle;
- `saturate` schedules;
- cost-based extraction;
- `prove`.

The prior ruling (X14) and D-CMR-8 fix how it may be used: **offline, against a closed enumeration of small terms, never on a bind or query path, and never as a dependency of `lance-graph-contract` or relalg.** Its output is evidence about the catalogue. That output can be:
- a counterexample showing the local rewriter commits to a larger form;
- a proof that an admitted rule follows from the stratum's axioms;
- a missed rule that is a *candidate* for the catalogue, admitted only through the normal `set_rules` path with its guard and the Kleene oracle.

It is never a rule source in its own right.

- **datafrog: SKIP.** It has no equality, no union-find and no rewriting.
- **Ascent: SKIP.** It has no congruence closure. The BYODS `eqrel` provider gives an equivalence relation but not congruence or extraction. egglog is the purpose-built tool, and duplicating it in Ascent buys nothing.

### Role (b) — independent test oracle for D-RFT-3 recursion and D-TGB-10

- **datafrog stays the SET oracle.**
  - It is already designated (D-TGB-10, D-RFT-3, the WCOJ map) and already a dev-dependency in two crates.
  - It has zero dependencies and fits monotone closure and leapjoin.
  - It covers Boolean/Distinct recursion. Stratified negation is possible against completed strata.
  - The one change the 2026-10-21 P0 rule forces is in **how the comparison is made**, not in the engine. Current probes compare tuples:
    - `lance-graph-metta/examples/join_probe.rs:14` compares sorted pair vectors;
    - `crepe-plan-probe/examples/walk_mask.rs:361-372` compares `BTreeSet`s, decoded from the mask planes;
    - `walk_mask` also prints an "ascent ms" column (`:399`), which is a no-ms violation as written.
  - The rule demands fold against fold. The substrate side produces terminals only (`Count`, plus a fold such as `Sum` over an id-hash lane, or several independent hash lanes). The oracle side computes the same folds over its own result relation. The substrate never decodes a row.
- **Ascent: a second oracle, only for strata datafrog cannot express without hand work.** These are:
  - **aggregation inside recursion**, e.g. "count landed facts per source" (D-TGB-8);
  - **stratified negation with automatic stratification**, e.g. "branches with no anomaly";
  - **min-plus lattices**: `Dual<u32>` shortest path is the natural oracle for P2 (hop distance from iterated Boolean frontier masks).

  datafrog can reach the first two only by hand-ordered antijoins plus external counting, and the third only by enumerating every bounded length.
- **egglog: SKIP.** It has no negation and no aggregation, and its relations are sets. Nothing in D-RFT-3 or D-TGB-10 needs equality saturation. D-RFT-2's substitution step explicitly does not need an e-graph (assessment entry, `:64`).

### Role (c) — runtime evaluator

**SKIP for all three.** Four reasons:
- every one materializes owned tuples (§2);
- every one would be a second executor of set operations, which "Polyfill, never legacy" forbids (`thought-git-basics-v1.md:174`);
- mask-risc is the reference fold algebra (P0);
- the measured oracle runs are in ms (datafrog `join_10000` 0.51–0.62 ms; PeTTa and Hyperon much slower: `thought-git-basics-v1.md:62`), which is a failed gate on any runtime path.

**The one route worth naming, and why it fails: an Ascent BYODS provider over resident mask planes.**
- What works: `index_get` could yield the set bit positions of a plane, read in place. That makes it zero-copy with respect to the storage.
- What does not: the generated join is still a nested loop binding one value at a time, and it writes head tuples through `RelIndexWrite`. That is enumeration where a fold belongs.
- Count, exists and sum would still be per-binding loops, not one compare producing one mask.

BYODS changes how a relation is stored. It does not change the execution model, and the execution model is what the P0 rule rejects. This stays SKIP, with the stated reason, unless the operator re-scopes. It is listed because the open item in `thought-git-basics-v1.md:174` ("whether a polyfill-backed BYODS relation fits ascent's interface is unchecked") now has a first-hand answer: **the interface fits, but the execution model does not.**

### Concepts worth borrowing (no code, no dependency)

- **Ascent's SCC + delta/total/new split** is the shape of relalg `μ` (D-MTP-3) in mask form:
  - `delta` becomes a frontier plane;
  - `total |= delta`;
  - `new &= !total`;
  - the fixpoint is reached when `new` is all zero.

  `walk_mask` already does this in a lab. Strata follow from the rule dependence SCCs, as `compute_dag_topo_order` already does for computed properties.
- **egglog's ruleset-per-schedule** maps onto `set_rules`: one enabled-rule set per `Semantics` stratum, which `rewrite_with(enabled)` already supports (`set_rules.rs:343-364`).

## 5. Probes (pre-registered; nothing run)

Every probe lives in a lab crate with its own workspace root, beside `crepe-plan-probe`. No probe reports a timing as a result; offline tools report counts only.

**P-E1 — egglog offline normal-form search (D-CMR-8 / D-RFT-2).**
- **Setup.**
  - Enumerate every `SetExpr` over the leaves {Sym0, Sym1, Res(lane 0 > 0)} built with `And`, `Or` and `Not`, up to 7 set nodes.
  - Write two egglog rulesets:
    - a two-valued Boolean algebra;
    - a Kleene ruleset: no complement, no excluded middle, absorption kept, De Morgan kept.
  - Saturate each ruleset under a node budget and extract with cost = `set_nodes` (`set_rules.rs:128`).
  - Compare against `set_algebra::normalize_with(e, logic)`.
- **Assertions.**
  - Every egglog-equal pair agrees with the Kleene `set_at` oracle on random NULL tables. This makes the oracle independent of egglog.
  - Report how many terms egglog extracts strictly smaller than the local normalizer does.
- **Kill.** Zero strict improvements over the whole enumeration closes D-CMR-8 and moves egglog to PARK for good.
- **Disable (can-fire).** Drop the absorption rule from the egglog ruleset: known absorption terms must then come out larger than `normalize`'s.
- **Silent twin.** Under the Kleene ruleset, `check (= (Or A (Not A)) All)` must fail.

**P-E2 — egglog proofs of the catalogue (optional, after P-E1).**
- For each `set_rules` rule that rests on a set identity (`RelationalDifference`'s set side, `SetNormalize`), `prove` the identity from the stratum's axioms only.
- **Kill.** A catalogue rule that cannot be proven from the stratum's axioms is a finding against the rule or the axiom encoding. Either way it is reported, not patched in egglog.

**P-D1 — datafrog fold-vs-fold oracle (D-TGB-10 / D-RFT-3).**
- **Fixture.** Nested groups with a cycle and a diamond, 256 groups or fewer; a 257th group must be refused.
- **Substrate side.** Closure by iterated Boolean frontier masks, ending in `Count` plus k independent hash-sum terminals over the kept mask. Which terminal carries the hash is OPEN; `Measure::Sum(lane)` over an `I32` hash lane exists (`ir.rs:103`).
- **Oracle side.** datafrog closure, then the same folds computed over its relation.
- **Assertion.** The folds are equal. No `Vec` or set of pairs exists on the substrate side.
- **Disable.** Drop the cycle edge on the substrate side only: the folds must differ.
- **Silent twin.** The diamond counts once (Distinct), not twice.
- **Kill.** A constructed wrong answer of equal count (swap two members) is not detected with k = 2. Then raise k; if the probe still fails at k = 4, fold comparison is insufficient for this fixture, and that goes to the operator.

**P-A1 — Ascent as the negation and aggregation oracle (D-TGB-8 rules).**
- **Rules.** "Branches with no anomaly" (negation) and "landed facts per source" (`agg count`). Each is written once in Ascent and once in datafrog with hand-ordered antijoins.
- **Comparison.** Both are compared to the substrate's folds, fold against fold.
- **Kill.** If every D-TGB-8 rule is reproduced in datafrog with hand stratification at no greater fixture complexity, Ascent goes to PARK and stays out of the dev-dependency set.
- **Disable.** Put the negated relation in the same stratum (datafrog side: antijoin against a still-growing relation). The answers must diverge, which shows that stratification is load-bearing.

**P-A2 — Ascent lattice as the min-plus oracle (feeds P2).**
- **Oracle.** `lattice sp(u32, u32, Dual<u32>)` gives shortest hop counts.
- **Substrate.** The first level at which iterated Boolean frontier masks reach a node.
- **Assertion.** Equal per (source, target), compared as a fold: Σ distance and Count over reached pairs.
- **Disable.** Replace `Dual<u32>` with `u32` (that is, max): the two must disagree.
- **Kill.** Any disagreement on the dir-sim nest.

**P-N — NARS lattice admissibility (already P6; no engine).**
- Check idempotence, commutativity and associativity of revision on a quantized grid.
- **Prediction.** Idempotence fails. That failure disqualifies both an Ascent `lattice` and an egglog `:merge` for NARS, and the probe records it.

## 6. Recommendation matrix

| role \ engine | egglog | datafrog | Ascent |
|---|---|---|---|
| (a) offline proof / normal-form generator for `set_rules` | **PROBE** (P-E1, then P-E2): lab crate only, never linked into the contract or relalg, counts only | SKIP | SKIP |
| (b) independent test oracle, D-RFT-3 / D-TGB-10 | SKIP (no negation or aggregation; no need for equality) | **ADOPT-NOW** as the SET oracle (already adopted). **ADOPT-NOW** (test infrastructure): switch probe comparisons to fold against fold (P-D1) | **PROBE** (P-A1 negation and aggregation, P-A2 min-plus lattice) only for strata datafrog cannot express without hand work |
| (c) runtime evaluator | SKIP | SKIP | SKIP (BYODS keeps the tuple-at-a-time join; reason recorded) |
| concept borrowing | PARK (rulesets per stratum already exist as `rewrite_with(enabled)`) | — | **PARK** (SCC + delta/total → relalg `μ` in mask form) until D-MTP-3 is scheduled |
| NARS stratum | SKIP (`:merge` precondition) | n/a | SKIP (`Lattice` precondition) unless P-N unexpectedly passes |
| K3 stratum | SKIP | SKIP | SKIP (the in-tree Kleene oracle already covers it) |
| Bag stratum | SKIP | PARK (X17 multiplicity column) | PARK (`agg sum` over a witness column) |

For the Bag stratum, one of datafrog and Ascent should carry it, not both; see Q4.

## 7. Open operator questions

1. **The scope of "oracles compare by folds."** datafrog, Ascent and egglog all hold tuples internally. Is an external, test-only engine acceptable when the *comparison* is fold against fold, so that the substrate never decodes and the oracle computes the same folds over its own relation? Or does the P0 rule exclude tuple-native oracles altogether? If they are excluded, then all of role (b) goes, and the only oracle left is the materialized-table evaluator style already in `set_rules` tests, which also builds a table.
2. **The existing violations.** `walk_mask.rs` prints an "ascent ms" column and compares `BTreeSet` tuples. `join_probe.rs` compares sorted pair vectors. Should these be rewritten under P-D1, or banner-marked as pre-ruling labs?
3. **egglog's footprint.** It is a heavy dependency tree, and its fork pins toolchain 1.91.0 while the workspace uses 1.99. Is an isolated lab crate with its own workspace acceptable for P-E1? Or does D-CMR-8 stay a paper exercise until the local rewriter shows a measured loss?
4. **crepe or Ascent.** `crepe-plan-probe` already depends on both. Which one owns the negation and aggregation oracle slot? crepe also owns the rule-as-data front end (`crepe_plan!`), which Ascent's macro does not expose.
5. **The Bag stratum (D-RFT-1).** Should it be carried by datafrog plus a multiplicity column (X17), or by Ascent `agg sum` over a witness column?
6. **The hash terminal for P-D1.** Is a hash-sum over an `I32` id lane through the existing `Measure::Sum` acceptable as a fold fingerprint? Or should mask-risc gain a dedicated order-independent fingerprint terminal? A new terminal needs an insufficiency proof first.

## Not read / not searched

- The OOPSLA'23 BYODS paper: HTTP 403. A web search attributed it to a different author list, which is unverified, so it is not relied on.
- egglog-experimental, which may add negation.
- The Ascent paper's §5 numbers (they are not needed for roles a–c).
- The egglog paper's §5–6.
- The crepe source.
- Soufflé.
- Whether mask-risc already has a fingerprint-style terminal.

`STATUS: open | OUTCOME: evaluation map; datafrog stays the SET oracle with fold comparison; Ascent gets a narrow oracle probe; egglog gets offline probes only; all three SKIP at runtime | OPEN: Q1–Q6`
