# reliability-checklist-arc-v1 — integration plan as a LIST OF POSSIBILITIES

> **Status:** PROPOSAL / possibility menu (2026-05-30). NOT a committed sequence — a
> candidate set for the council + brutal reviewers to RECALIBRATE (re-rank, prune,
> flag theater-risk). Each item cites the session finding it realizes.
> **Arc it serves:** (f,c)=reliability → reliability=checklist-coverage → checklist=NARS/elixir
> template=Odoo D-Atoms → validity=external oracle → psychometrics=offline audit.

## The possibilities (unordered menu — recalibration assigns order/verdict)

### P1 — Cheap checklist-coverage gate (the headline)
- **What:** domain-agnostic `coverage(required: &[Atom], lit: bitmask) -> CoverageState{Covered|Gap(dark)|Unsatisfiable}`; wire to Rubicon `Evaluation→{Commit|Plan|Prune}`. Reads #433 `OdooStyleRecipe.atoms` (required) vs lit/dark bitmask (known).
- **Realizes:** E-RELIABILITY-IS-CHECKLIST-COVERAGE + E-TEMPLATE-IS-CHECKLIST-IS-DATOMS.
- **Size:** S. **Deps:** #433 DAtom (exists), N3 bitmask, N1 class_id (#439). **Risk:** where does `coverage` live (contract zero-dep? ontology?) — dep-direction question.
- **Threshold-free:** dissolves the 0.2/0.8 iron-rule violation (no calibrate).

### P2 — R-GATE live-trace probe (probe-before-wire)
- **What:** does checklist-coverage (or reliability) state CHANGE a Rubicon Commit/Plan/Prune outcome on a real witness trace vs DAG-only? Pass = ≥1 differing terminal; Fail = cosmetic.
- **Realizes:** the reviewers' probe-first rule; gates P1/P5.
- **Size:** S (a test/probe, not a wire). **Risk:** needs a representative witness corpus; may show the gate is cosmetic (then DON'T build P1's wire).

### P3 — StyleStrategy emit edge (de-stub the passthrough fully)
- **What:** `StyleStrategy::plan()` EMITS — thread the resolved style + reliability into `PlanResult.thinking` (or a KanbanMove when A6 output overhaul lands). Kills the remaining passthrough.
- **Realizes:** fixes the council's "two disjoint islands"; D-MBX-A6-P3 emit edge.
- **Size:** M. **Deps:** D-MBX-A6 planner-output overhaul (KanbanMove output) OR the lighter PlanResult.thinking sink. **Risk:** prior-art says don't mint PlannerDTO; use existing sink.

### P4 — impl MailboxSoaOwner for MailboxSoA<N> (make try_advance_phase live)
- **What:** real owner impl in cognitive-shader-driver so the Rubicon lifecycle is wired to a real type (today only FakeSoa calls it).
- **Realizes:** retires the "shipped-but-dead" finding; D-MBX-A6.
- **Size:** M. **Deps:** mailbox_soa.rs (actively edited by other waves — collision risk). **Risk:** HHTL/SoA-column wave conflict.

### P5 — I4x32→argmax→ThinkingStyle keystone decode
- **What:** the projection collapsing composition→identity (creative-explorer's keystone). Contract-side `atoms::I4x32 -> ThinkingStyle`.
- **Realizes:** the keystone; unifies the 3 style representations; makes recipe.rs deletable.
- **Size:** S-M. **Deps:** atoms::I4x32 (landed), thinking::ThinkingStyle. **Risk:** 32-vs-33 dim (sentinel?); is argmax the right decode.

### P6 — DELETE recipe.rs (not revive)
- **What:** remove the stale/orphaned `StyleRecipe`/`PersonaRecipe` (4 savants: don't revive); fold PersonaRecipe's β/threshold into persona.rs if needed.
- **Realizes:** the council's unanimous don't-revive; removes the 5th-column drift.
- **Size:** S. **Risk:** is anything (even aspirationally) depending on it; deletion vs leave-dead.

### P7 — Reconcile the 4-way recipe surface
- **What:** document/converge `recipes`(tactics) / `recipe_kernels`(Tactic) / `recipe.rs`(StyleRecipe, dead) / `OdooStyleRecipe`(domain checklist) into one map — names + roles, not a merge.
- **Realizes:** the 3-recipe-module finding + the template=checklist=DAtoms correction.
- **Size:** S (doc/rename). **Risk:** rename churn across crates.

### P8 — Psychometric offline audit (the heavier reliability path)
- **What:** thinking-engine `reliability_calibration` (Cronbach α/ICC over a recipe/witness corpus) → emit cited bands → audit whether checklist ITEMS cohere. Offline-float→frozen-const.
- **Realizes:** E-CALIBRATE-RELIABILITY-PSYCHOMETRICALLY. Complementary to P1 (audits the checklist, doesn't gate the hot path).
- **Size:** M-L. **Deps:** thinking-engine cronbach.rs (exists). **Risk:** heavier; only worth it if P2 shows the gate matters AND checklist completeness is in question.

### P9 — D-MBX-11 lance 6.0.0→6.0.1 bump (mechanical)
- **What:** the version bump (5 Cargo.tomls). **Size:** XS. **Risk:** none (off critical path). Quick unblock for stack alignment.

### P10 — Polyglot IR-conformance loop (the GEL hub floor)
- **What:** assert the 4 dialects + 2 IR-routes (in-strategy vs ArenaIR) produce equivalent LogicalOp for equivalent queries; attach to consumer-conformance harness.
- **Realizes:** E-POLYGLOT-TWO-IR-ROUTES; grounds the GEL query slice. **Size:** M. **Risk:** orthogonal to the reliability arc (different thread).

### P11 — class_id→checklist resolver + ontology O(1) index
- **What:** `OntologyRegistry.by_entity_type_id` HashMap (the one perf gap) + class_id→required-checklist(D-Atom set) resolver.
- **Realizes:** the cognitive-risc-classes class→checklist projection; the audit's O(1) fix.
- **Size:** S-M. **Deps:** P1 (the checklist type). **Risk:** ontology↔checklist dep direction.

## Open recalibration questions (for the panel)
1. Sequence: which is the true first slice — P2 (probe) before P1 (gate)? P9 (free) now?
2. Which are THEATER-PRONE (ship green, do nothing — like the #439 passthrough)? Flag them.
3. Which violate an iron rule / dep-direction / zero-dep as scoped?
4. Which are DUPLICATES of existing work (build on, don't rebuild)?
5. What's MISSING from the menu (the reframe/second-order option)?
6. Cascade size per item — which are S that masquerade as L (or vice versa)?
