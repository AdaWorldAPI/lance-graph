# relalg × thought-git: algebraic method behavior and amortized frontier (v1)

**Status:** PROPOSAL / DOCS ONLY, 2026-10-21.
**Highest implementation priorities:** 'lance-graph-relalg' and reusable 'lance-graph-thought-git'. Keep the existing D-TGB oracle and D-RA/D-CMR owners. No independent rule interpreter, evaluator, graph store, object model, scheduler or hidden write side effect.

## Thesis

One method-bearing class surface should admit 'think', 'counterfactual' and 'do' (names **conceptual**, not proposed hot ABI). Each route binds a declared ClassId + numeric MethodOrdinal + algebraic law to a witnessed focus. Evaluation returns either a relation/result, a replayable Thought delta, or an authorized-action proposal. The SAME relalg/R2IL/mask-RISC lowerer executes admissible set operations. A thought's execution is **not** itself a commit; Rubicon/MUL/ActionDef authorization and explicit Rust LanceCycleWriter write remain separate.

A method is defined by semantics (e.g. Boolean existence, bag relational algebra, exact integer, K3), operands, guards and a result contract. Physical realisations are routes, not alternative laws or alternative truth engines. Prefer existing 'ClassView::compute_dag' and metadata/action accessor over new descriptor hierarchies; compile topological declaration once per class/schema version; data/codebook/focus witnesses must be separately invalidated.

## Why relalg first

Existing relalg is a single-table experimental algebra with 'Rel', 'normalize', 'symbolic', R2IL lowering. Its 'join_body::is_alpha_acyclic' tests the shape of a join, not its execution. The 'set_rules' guarded catalogue belongs to the contract; relalg owns lawful composition/lowering, not a duplicate semantic registry. Existing Quack already has masks, COUNT, EXISTS, ALL, SUM, MIN/MAX, grouped terminals and constrained FK-keyed 'GroupSumViaI32'; confirm rather than rebuilding it.

1. **R0 correctness before expansion.** A Kleene three-valued oracle on NULL lanes for 'set_algebra::normalize' and the catalogue. Check complement, NOT, negated membership, absorb and SUM empty/null. Red implies selective refusal/guard, no blanket 'correct under SQL' claim.
2. **R1 relationship algebra.** Extend/probe typed shared-domain relation composition without value-directed foreign-table gather (ADMISSION B). Distinct/Bag, K3, external references, supports, key/cardinality witnesses, and cyclic join bodies are separate acceptance strata. Use existing #1659 shared-coordinate/side and #1666 DR256 lab results where admitted. Test INNER/SEMI/ANTI/LEFT, fan-out and duplicates.
3. **R2 law-licensed rule catalogue.** 'set_rules::Semantics' identifies admitted rewrites where expressible. If an equality rule needs more conditions, keep explicit per-rule refusals rather than inventing a universal semiring. egglog, if introduced, is an **offline/compile-time rewrite candidate/proof oracle** with admitted equalities; it never changes the method's declared law, and never becomes an unbounded production query-time planner.
4. **R3 recursion.** datafrog is an independent reference for closure/replay; first match its results in a bounded monotone rule fixture. Recursive driver can apply mask/frontier folds where legal; prevent cycles through negation without strata. NARS truth revision is not presumed a semiring; keep its established finisher and revision/provenance semantics separate.
5. **R4 class method reuse.** One behavior method runs correctly on direct ordinals, sparse runs, DR64 and DR256 when lawful. Prove same result including duplicate semantics and the 257th-slot condition. Do not introduce generic planner types or OGAR as a mandatory layer.

## Thought-git as the FIRST real non-SQL consumer

Reuse '.claude/plans/thought-git-basics-v1.md' and 'thought-git-api-v1.md'. The lab 'crates/thought-git-probe' already has fixed anomaly fixtures, from-scratch next-version oracle, sparse slot experiments, with no production branch/land/rebase pipeline. EXO-98 defines the reusable git verbs and driver independence.

- **Thought 0** is immutable sealed givens for the current version; 1..63 are bounded temporary thought slots. An attention delta *is* the branch overlay; replay receipts and derived fact relations live in a **separate metadata table**, not duplicated node objects.
- Input is a (snapshot, focus, class method ordinal, numeric operands, source/relationship witness). Output is derived cells, provenance, read/write masks, anomaly readings and a deterministic replay receipt.
- In-memory branch, diff, squash, discard and replay execute only admitted folds. Commit/land is one authorized, explicit, coalesced sparse writer-cycle operation. 'Plan' and 'Prune' do not write Lance; error/contradiction preserves evidence rather than fabricating confirmation.
- Thought frontier: changing an input invalidates the affected method outputs only; a typed dependency mask propagates the change, not the whole 64-thought stack. A full recompute remains the oracle. Retracting a support path from a diamond MUST preserve the surviving support; naive Boolean subtraction is unsound.
- **EpistemicState5** (CausalEdge64 bits 59..63) is Topology2 × Certification3, NOT a scratch dirty mask, no implied independent supporting root for duplicate/replayed derivations. 'pearl::hydrate' and 'pearl::revise' have distinct monotone writers. A counterfactual never elevates production evidence.
- Snapshot generation and 'thought_address' witness prevent the ABA problem when a freed slot is reused. Rebase/replay is refused on stale codebooks or proof generations without a legal recompute.

### First falsifier: substitution / Sudoku frontier

Fixture: x = 2y, y = 2z, w = 2x. Two independent substitution routes produce the same result; then mutate one source relation, remove one support path, and reuse a thought slot at a new generation. Derive w = 8z when premises hold. Compare full from-scratch derivation and incremental frontier for each affected class/method, including provenance roots and non-commutative/NULL/Bag counterexamples. No duplicate proof becomes 'new evidence'.

Use an IAM nested group diamond as a second fixture: U ∘ G* ∘ P (Boolean). Verify not only result parity but touched tiles/pages, amortized common bindings, and correct deletion of one among two support paths. Use a source-lane direct read; reveal old millisecond CSR preparation explicitly rather than hiding it in a fold time.

## Phases and gates

| Phase | Owner/where to extend | Deliverable | Falsifier |
| --- | --- | --- | --- |
| T0 | relalg + contract rule tests | SQL/K3 truth safety | normalized = original = row/PG oracle for every tested truth-stratum |
| T1 | relalg, Quack existing execution | typed relation/JoinDomain fold | SQL Bag/Distinct + nullable join parity; no gather |
| T2 | thought-git-probe, existing D-TGB API plan | substitution and group-frontier replay | full recompute parity for add, delete, cycle, diamond, ABA |
| T3 | datafrog oracle + relalg frontiers | bounded recursion and stable fixpoint | closure results and exact termination match datafrog |
| T4 | optional egglog proof path | admitted equalities/optimization | each rewrite law licensed; no adverse K3/Bag rewrite |
| T5 | one ClassView method consumed via flow | think vs counterfactual vs proposal | exact replay and unchanged sealed snapshot until authorized commit |

**Performance:** report all costs: codebook/domain bind, prefetch/tile build, first touch, steady fold, output materialization, invalidation. Measure 32×128 vs 64×64 for membership (both 512 bytes), DR256 and sparse; interpret shape selection as a route under fixed laws. No hard-coded shape in API.

**Explicit exclusions:** Odoo deferred; no persisted Node32/SoA layout decision, no new graph interpreter/ORM, no direct action side effects inside folds, no claim of autonomous learning, no shared cache without version-provenance and invalidation tests.
