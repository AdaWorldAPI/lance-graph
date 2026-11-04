# D-CGMM staged PR charter: compiler facts → lazy Codegraph → FieldViewDef → DT64 native binding

> **2026-11-04 · DOCS-ONLY PRE-REGISTRATION.** Only CGMM-0 is opened now. CGMM-1 through CGMM-8 are proposed *separately reviewed* PRs. No production code, new class codebook, general MeTTa runtime, Soufflé/Crepe/MORK executor, new F### word or unapproved source traversal is authorized. [Master](d-cgmm-codegraph-metta-method-builder-v1.md) · [Lazy Codegraph](d-cgmm-lazy-codegraph-fact-demand-v1.md) · [OGIT/OGAR edge/action](d-cgmm-action-relation-schema-v1.md) · [Falsifiers](d-cgmm-falsifier-matrix-v1.md) · [EXO-155](https://linear.app/exo-red/issue/EXO-155).

## The dependency contract

~~~text
CGMM-0 docs, exact source ownership and refusal matrix
    |
    +-- CGMM-1 source-fact intake census and typed Kythe identity/coverage (LAB)
    |       |
    |       +-- CGMM-2 demand/binding-plan and lazy Codegraph view (LAB)
    |       |       |
    |       |       +-- CGMM-3 FieldViewDef + MeTTa-shaped constraint scratch (LAB)
    |       |                   |
    |       |                   +-- CGMM-4 OGIT M:N/tenant + OGAR ActionDef offers (LAB)
    |       |                   +-- CGMM-5 exact pure RelAlg polyfill (LAB)
    |       |                   |           |
    |       |                   |       CGMM-6 owner-admitted DT64 -> R2IL native
    |       |                   |           |
    |       |                   +------- CGMM-7 prepared plan/memo economics
    |       |                               |
    |       +-- CGMM-8 reachability / provenance / NNUE ranking audit
    |
    +-- EXO-79,84,85,95,101,108,121,140,143,152,153,154
          are named gates, not automatic permissions
~~~

## CGMM-0. Architecture + Linear + board (this PR)

**Deliverables:** [master](d-cgmm-codegraph-metta-method-builder-v1.md), [Action/Relation addendum](d-cgmm-action-relation-schema-v1.md), [Lazy Codegraph addendum](d-cgmm-lazy-codegraph-fact-demand-v1.md), this charter, [falsifier matrix](d-cgmm-falsifier-matrix-v1.md), board entry, board status/integration/entry index, EXO-155 issue and append-only EXO-50 handoff.

**Acceptance:** Codegraph source facts separated from derivation plan and lazy projection; ClassView and OGIT/OGAR/DT64 ownership preserved; exact refusal at current R2IL `GuardNotExecutable`/three-valued boundary; no ambiguity of FieldViewDef with render-only OGAR FieldView; no production code or new ISA. No merge without review.

## CGMM-1. Intact Ruff/Charon/Kythe source fact intake (future lab)

**Goal:** map the actual source artifact to stable source node/edge IDs and independent coverage grades without an eager copied AST object graph. Follow the current `.rows.bin` FUNDECL/PROV reader and Codegraph interner. Produce an explicit FACT-LOSS register: receiver types, field read/write, dynamic dispatch, edge polarity, source span, effect ordering and negative statements. A codegraph position is not a stable Kythe ID.

**Oracles:** independent source facts by producer source ID; single method with writes, reads, possible trait dispatch and unresolved call; source revision changes; falsely confident missing effects. Source producer modification is a separate owner review; do not auto-fill unknown columns.

**Metrics:** intake bytes and allocations, source→binary loss, provenance completeness, duplicate facts, AST materializations, cold/hot refresh. **Stop** if a fact is unavailable or a schema extension would break the existing binary contract.

## CGMM-2. Crepe/Soufflé-style demand artifact, lazy Codegraph (future lab)

**Goal:** expose compiler-like **binding-pattern and demand metadata** to existing Codegraph/XLOG/DR queries, without importing Crepe/Soufflé as a live executor or changing the global map. Compare existing eager six-relation intake and repeated `or_partners` queries with a bounded demand-scoped relation/mask plan.

**Required evidence:** query roots, bound/free columns, source graph generation, grade floor, semiring, source/permission scope, recursion/negation/termination policy and unsupported constructs. Crepe `crepe_plan!` D1/D2 drift and XLOG polarity/magic report loss from the 2026-11-02 archaeology are independent falsifiers, not trusted compile data yet.

**Positive:** one anchored call graph query touches only requested symbol relations after admissible indexing; repeated Q=1/2/16/256 amortization, bounded retained masks. **Negative:** `MAY` upgraded to confirmed; missing dynamic call interpreted as absence; stale source-epoch cache; unsupported recursion silently executed; demanded query slower than eager for all cases (report NO BENEFIT).

## CGMM-3. FieldViewDef / ClassView-attached constraint scratch (future lab)

**Goal:** one immutable **FieldViewDef OFFER** over an existing ClassView field or behavior-lens basis; per-session, disposable MeTTa-shaped constraint scratch returns `OFFER | REFUSE | ABSTAIN`. Roles include scalar, relation subject/object and action operand, with no physical data copy. Do not change the zero-dependency ClassView trait's `FieldMask` or `WideFieldMask` purpose.

**Tests:** same field labels under different source/class version, full behavior-lens ID check, schema revision/position invalidation, partial source effect and incompatible method pair. Compile-fail or negative guard rejects mutable class basis and trait additions; scratch eviction does not modify an admitted method.

## CGMM-4. OGIT LinkSpec + OGAR ActionDef M:N / tenant profile (future lab)

**Goal:** use one scratch vocabulary to propose **separate owner-admitted** ontology edge, OGAR action and DT64 method contracts. Cross-app Rails/Odoo/SAP scenarios are *synthetic* unless independently harvested. No automatic equivalence from label or NNUE.

**Fixture:** A tenant has W1/W2 and P1/P2; B tenant W3/P3. A relation includes duplicate (W1,P2) for Bag and a forged cross-tenant edge (W2,P3). A and B share an "Approve" label but have ordered effects that differ.

**Tests:** cardinality vs multiplicity; relation orientation vs inverse; tenant auth vs ontology context; edge-instance identity vs endpoints; unknown source writes → ABSTAIN; one effect intentionally removed causes equivalence to be rejected. No relation insert or ActionInvocation.

## CGMM-5. RelAlg polyfill for exact eligible field views (future lab)

**Goal:** a finite, typed, pure same-universe read `FieldViewDef` lowers via existing `RelAlg::Rel::{All,Empty,Plane,Atom,And,Or,Not}` / `Measure::{Count,Sum,Min,Max,Keep}` to R2IL/mask-risc. Source field label is resolved cold to typed lane slot, never string inside hot IR.

**Tests:** exact I32/U32 comparisons, support vs Bag counts, missing/zero, SQL UNKNOWN refusal when unsupported, same-frame masks, independently computed oracle. **Refuse** cross-population M:N hop, arbitrary recursion, effects, guarded WHEN, dynamic data-dependent folds. `join_body::is_alpha_acyclic` is a classifier, not an executable join.

## CGMM-6. DT64 MethodBinding owner admission and native execute proof (future, gated)

**Owner decisions:** EXO-121/152, OGAR codebook, DT64 execution owner; source lease and scope; existing `ExecPermit` lab-only. A scratch offer is not a ClassId or automatic MethodRef; owner allocates proper identity and accepts exact signature/effects/guard/truth semantics.

**First runnable consumer:** one guard-free two-valued fold is admitted to lab `r2il::binding_native::prepare_native`/`run_native` and compared against independent source-value oracle. Explicit `GuardNotExecutable` and `ThreeValuedNotExecutable` outcomes for other offers are positive refusals. No new method body interpreter.

## CGMM-7. Compile/plan/route memoization economics (future lab)

**Goal:** reuse immutable normalized pure method recipes under all source/artifact/schema/method/ontology/rule/permission/DR frame witnesses without caching source AST values or copying canonical SoA. Keep **cold harvest/constraint/prepare** and **hot execute** separate. Q=1/2/8/16/64/256, scope/source invalidation, cache eviction and negative-control reroute.

**ALU:** compare existing native Boolean/byte path against shipped Abakus carry words only when semantic arithmetic is required; #1956 measured 1.5–4.8× carry loss for its eight shapes, so no forced Abakus winner.

## CGMM-8. Independent oracle and reachability firewall (future lab)

**Goal:** two independent oracles (scalar source-enumeration and bounded Crepe/datafrog/Soufflé/MORK where pinned and licensed) confirm the graded result and refuse unsupported semantics. NNUE can reorder **candidates**, never grant facts or permits. Link the graduated intake to D-DRIVER-ZC caller-coverage proof, reporting unknown/dynamic call paths explicitly.

**Stop:** missing artifact/producer, same function used for oracle and candidate, copy of source value cache, generic Datalog/MeTTa runtime adoption, caller context not attested or claimed complete without reachability proof.

## PR governance

**One stage per PR**, exact commit SHAs, source-backed grades, a deliberate disable that turns a nonvacuous assertion red, cold/hot/representation metrics, separate board entry and EXO-50 append-only handoff. No auto-merge and no reclassifying a research adapter as a production API. All future work gated on independent owner decisions; CGMM-0 does not authorize CGMM-1–8.

**Explicitly out of scope:** migrated Ladybug/cognitive-shader-driver and P64/StreamDTO/BusDTO/PerturbationDTO consolidation; the operator deferred that study.
