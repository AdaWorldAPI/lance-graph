# D-FLOW-COG: one cognitive fabric, four thin integration surfaces

**Status:** source-grounded architectural follow-up, **not an implemented new runtime**  
**Base:** fresh branch from LanceGraph `main` at `350d30aa9f06ea3e700fd11e5d6e9c4a0a51b2ea` (2026-10-13).  
**Predecessor:** merged [#1536](https://github.com/AdaWorldAPI/lance-graph/pull/1536), with [#1539](https://github.com/AdaWorldAPI/lance-graph/pull/1539) correction of ONA source pins and the unimplemented supervisor respawn-backoff guarantee.  
**Implementation scope:** follow-up PRs only, each with code + independent oracles. This PR records the binding/ownership decision and a Work implementation handoff.

## Thesis

Unify the best interface capabilities of **Supabase (external/realtime membrane), LangGraph/graph-flow (resumable stateful flows), Rig/LangChain (models and tools), z8run/n8n (triggers, workflow UI), OpenCog/Hyperon (typed compositional symbolic cognition)** with **ONA's NAL truth/inference laws** and **two complementary Rubicon lines**:

- **Heckhausen/Gollwitzer psychological Rubicon:** deliberation -> goal commitment -> implementation intention -> action -> evaluation and disengagement/replanning.
- **O'Reilly/Axon computational Rubicon:** cost-benefit selection, gated persistent goal maintenance, observed reward/outcome and temporal credit assignment.
- **ONA is a separate epistemic foundation:** evidence, truth revision, derivation stamps, inference, temporal projection, anticipation; do not reduce goal/desire/authorization to NARS confidence.

The parts must **compile as methods, not compete as independently executing cognitive operating systems**.

### What already exists (verified on current repositories)

- [`lance-graph-callcenter`](../../crates/lance-graph-callcenter/src/lib.rs) implements the external `ExternalMembrane` shaped by a partial Supabase transcode; `LanceMembrane` separates `ingest/project/subscribe`. The current server remains feature-gated and partly scaffolded.
- [`lance-graph-supervisor`](../../crates/lance-graph-supervisor/src/supervisor.rs) has ractor actor skeleton and lifecycle API; **non-Health `DispatchToG` returns `DispatchNotImplemented`**. `DEFAULT_MAILBOX_CAPACITY = 1024` is declared but not applied to the `Actor::spawn_linked` path. **Actual crash-restart backoff is not correctly applied; delayed respawns reschedule themselves** per #1539. Realtime drain in [`callcenter/drain.rs`](../../crates/lance-graph-callcenter/src/drain.rs) still uses unbounded mpsc. No production admission/backpressure claim.
- [`rs-graph-llm/graph-flow`](https://github.com/AdaWorldAPI/rs-graph-llm/tree/main/graph-flow) implements `Task`, `GraphBuilder`, `FlowRunner`, `SessionStorage` and interrupt/resume; [`graph-flow-kanban`](https://github.com/AdaWorldAPI/rs-graph-llm/tree/main/graph-flow-kanban) already wraps the canonical `KanbanColumn` / `KanbanMove` contract. `FlowRunner::run` executes one graph step and persists its session.
- [`graph-flow-lance`](https://github.com/AdaWorldAPI/rs-graph-llm/tree/main/crates/graph-flow-lance) is an existing handle-only adapter. It deliberately reuses `z8run-lance`'s registry and `lance-abi` envelope, supports pause/resume across z8run, and publishes tests for stale generations, one folded mask-set fan-out and DO fold parity. **Do not reimplement these integrations.**
- [`graph-flow's Rig feature`](https://github.com/AdaWorldAPI/rs-graph-llm/blob/main/graph-flow/Cargo.toml) is already optional; [`graph-flow-lance/src/llm.rs`](https://github.com/AdaWorldAPI/rs-graph-llm/blob/main/crates/graph-flow-lance/src/llm.rs) already defines `Completion` and `LlmTask`. The model receives only explicitly materialized exports. This is the initial Rig seam, not a justification for a second model-agent framework.
- [`z8run-lance`](https://github.com/AdaWorldAPI/z8run/blob/main/crates/z8run-lance/src/lib.rs) already owns the handle registry / `Envelope` / node builder, not row or mask transport. z8run remains the triggers, integration events, approvals and visual workflow UI.
- [`lance-graph-nal`](../../crates/lance-graph-nal/src/nal.rs) has prepared `Question`, `TruthAtLeast` and `Choice` (confidence maximum, not argmax identity). [`lance-graph-r2il`](../../crates/lance-graph-r2il/src/lib.rs) already turns procedure calls into mask-risc folds. Full revision, row-wise truth maps, syllogism and executable temporal inference remain targets, not shipped.
- [`lance-graph-contract::kanban`](../../crates/lance-graph-contract/src/kanban.rs) owns stable lifecycle discriminants. [`contract::action`](../../crates/lance-graph-contract/src/action.rs) names the effect gate; the [SAG alignment](../../.claude/plans/sag-capstone-alignment-v1.md) identifies a legacy `commit` bypass risk versus `commit_via`. Audit and fail closed before wiring effects.
- [The ONA assimilation document](ONA_NAL_R2IL_FOLD_METHOD_ASSIMILATION.md) already defines W3–W9, including evidence-root independence, truth-lane output and temporal inference. [Temporal agenda](EPOCH_TEMPORAL_GRAPH_RESEARCH_AGENDA.md) distinguishes valid time, knowledge time and cognitive cycle.

**Historical caution:** The 2026-10-12 D-CMR method-layer reset removed old `glove-parity` anchors. Inspect current `MethodTable`/`ClassView`/hotplug APIs before wiring; do not resurrect deleted crates or assume old docs describe live implementation.

## Four proposed crate names and non-overlapping ownership

| Name | Exact reason to exist | Dependency direction | Must never own |
|---|---|---|---|
| **`lance-graph-flow`** | **Thin binding and IR contract** between existing `graph-flow` tasks/sessions, canonical `UnifiedStep`, NAL/R2IL method handles, `KanbanMove`, execution receipts. It maps graph-flow input/output to *canonical* handles and method calls. | `lance-graph-contract`, optionally NAL/R2IL; consumer adapters in rs-graph-llm depend on it, **not inverse**. | A second `FlowRunner`, `SessionStorage`, worker pool, Kanban state machine, or independent DAG evaluator. |
| **`lance-graph-rig`** | Optional **model/tool bridge** implementing existing `Completion`/tool interfaces: materialized retrieval, structured method *proposals*, explanation and feedback ingestion after independent validation. | contract + optional Rig; actual `graph-flow` integration remains consumer-side, dependency cycle explicitly ruled out. | A second LLM executor, durable reasoning truth, privileged action capability, or unsanitized belief inserts from model prose. |
| **`lance-graph-z8`** | Tiny adapter **only for missing canonical cognitive method node types** and translation into existing `z8run-lance::Envelope`; register additional nodes from z8run side. | contract / NAL method handles, optional consumer integration. Preserve one registry owned by `z8run-lance`. | New JSON wire kind, handle registry, flow scheduler, workflow serializer, visual editor, or copying SoA into `FlowMessage`. |
| **`lance-graph-cog`** | **Pure cognitive-method vocabulary and binding** over ONA NAL, Hyperon-like typed expression/fold composition, Rubicon benefit/cost/goal maintenance and OpenPsi-inspired motivational *scoring*. Return typed inert results. | contract → NAL → R2IL → mask-risc; planner and adapters invoke it. Must avoid cycle back to planner/flow/UI. | A second OpenCog AtomSpace, ONA C runtime, actor supervisor, motivation loop with autonomous authority, or redefinition of ISA byte lanes. |

**Crate creation gate:** Names represent clear modules/capabilities, **not four crates pre-approved**. Start with a source/dynamic call-path audit and use the fewest crates necessary. If `lance-graph-rig` or `lance-graph-z8` duplicates existing consumer-side adapters, keep it as an integration module or **do not create it**. `lance-graph-cog` should justify itself by importing multiple reusable pure cognitive laws without circular dependencies; otherwise expose `lance-graph-nal` and planner methods directly.

## Proposed dependency graph (logical ownership, not a Cargo.toml assertion)

```text
                       EXTERNAL AUTHORITY / IO
      Supabase-shaped API / realtime / z8run / rs-graph-llm / Rig
                               |
             callcenter membrane + supervisor admission
                               |
                       typed UnifiedStep
                               |
            contract: method / action / kanban / witness
                               |
                 [thin lance-graph-flow binding]
                     /                     \
        NAL methods + evidence          Rubicon goal/plan
          / R2IL fold methods           / Kanban + OGAR gate
                     \                     /
                  one canonical plan/read horizon
                               |
                       Quack / FoldDialect
                               |
                    mask-risc -> ndarray
                               |
                    canonical SoA / SPO-G
                               |
                   versioned handle + receipt
                               |
            existing graph-flow-lance / z8run-lance
```

The *outer API* and *flow engine* can schedule work; that does not make them alternative NAL/mask-risc execution engines. "One execution substrate" means one algebraic/physical evaluator for supported pure bulk methods, not a ban on I/O adapters, actor concurrency, external model calls or human approvals.

## The actual two Rubicon lineages, mapped to one legal lifecycle

| Source / question | Pure method result | Canonical authority | Anti-conflation |
|---|---|---|---|
| **Heckhausen/Gollwitzer:** Which goal is worth pursuing? | `GoalCandidate` with desirability, feasibility, source evidence, utility assessment | planner ranking; current `KanbanColumn` remains canonical, not replaced with a new psychological enum | Selecting a candidate is **not** crossing Rubicon. |
| **Heckhausen/Gollwitzer:** What if-then intention should follow commitment? | `ImplementationIntention` proposal: cue + guarded `ActionDef` + expiration + revocation conditions | explicit goal-commit and authorization, existing action gate | `cue-match` is not permission; stale intent is held/rejected. |
| **O'Reilly/Axon:** Cost/benefit competition and BG-inspired gating | Typed `BenefitCost`, `GoalScore` / `GateRecommendation`, **not a second BG neural network** | existing `GateDecision`, committed goal scope, planner; may issue Hold/Block suggestions only to policy evaluator | Optimizer score is not current permit. |
| **O'Reilly/Axon:** Persistent goal representation + temporal credit assignment | Goal-state **read/view**, outcome prediction and `CreditAssignment` fold based on observed result receipts | sealed-cycle history and observed outcomes, with explicit goal-maintenance horizon | Goal persistence is **not** endless stale authorization. |
| **ONA:** What is supported, how strongly, by which independent observations? | `TruthValue`, witnessed `Revision`, `Derivation`, temporal projection, confidence/expectation | `lance-graph-nal` W3–W9, `R2IL`, canonical evidence roots | f/c is **not** utility/probability/authorization. |
| **OpenCog/Hyperon/OpenPsi:** How to compose questions and allocate attention? | Typed expression tree, mask/fold composition, optional curiosity/expected information gain *with calibrated distribution* | same methods / planner budget; optional proposed `lance-graph-cog` composition | No imported AtomSpace; no fabricated cause from entropy. |

See [Axon's Rubicon model](https://github.com/emer/axon/blob/main/Rubicon.md), [Go implementation](https://github.com/emer/axon/blob/main/axon/rubicon.go), [choose simulator](https://github.com/emer/axon/tree/main/sims/choose), [Rubicon volition plan](../../.claude/plans/rubicon-volition-v1.md), [ONA pinned Truth.c](https://github.com/opennars/OpenNARS-for-Applications/blob/dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a/src/Truth.c) and [SAG wisdom plan](../../.claude/plans/sag-wisdom-frontier-ona-hyperon-v1.md).

**Canonical phases currently are** `Planning -> CognitiveWork -> Evaluation -> {Commit|Plan|Prune}`, not a full Gollwitzer psychological stage enumeration. An implementation intention or persistent goal is an **overlay with a separate witnessed commitment**, never an extra unofficial `KanbanColumn` value (stable ABI).

## Core method contract: one method language with typed, non-scalar results

**Illustrative method descriptors only; not a new ABI:**

```text
CognitiveMethodDescriptor {
  method_key, source, semantic_law_version,
  reading_id, class_scope, schema_seal,
  required_fields: WideFieldMask,
  effect_class: Pure,
  input_types, output_kind,
  proof_obligations, budget, capability_requirements
}

BoundCognitiveCall {
  method, source_generation, coherent_read_horizon,
  security_context, ordered_lane_witness,
  premise_root_refs, temporal_query, args
}

CognitiveResult {
  selection_or_truth_lanes_or_candidate_handles,
  original_evidence_roots, derivation_ids,
  source_revision, rule_version, trace_id,
  proposed_effect: None
}
```

Use existing `MethodKey`/ClassView / `CompiledProgram` / `UnifiedStep` and action types where they still exist. This is a conceptual responsibility map; do **not** copy these three structs literally before auditing the present contracts. Explicit preconditions: schema, typed reading, source generation, policy, row/order witness, original evidence-root independence and temporal knowledge horizon. Maintain the carrier/ISA separation: immutable byte layout is not a NARS/CE64/OpenPsi data schema.

Method calls should prefer **one bind and one compiled graph of pure folds**, eliminating algebraic no-ops, reusing resident masks and preserving non-scalar intermediates. Actual R2IL currently finalizes at scalar fold boundaries; whole-chain normalization and row-wise inference output require separate evidenced work.

## Lifecycle/flow mapping: LangGraph and n8n without a second runtime

```text
Webhook / API / incoming observation
      |
callcenter external intent -> authenticated / bounded / admitted?
      |
source-horizon-pinned observation -> witnessed NAL methods
      |
  pure Infer/Revise/Derive (W3–W7)
      |
  compare goal candidates (ONA expected consequence + Axon cost/benefit)
      |
  propose / deliberate (Heckhausen)
      |
  Rubicon commit? -- no --> Hold / re-evaluate
      | yes, witnessed goal commit
      v
  implementation intention + cue / guard
      |
  current ActionDef / ClassRbac / idempotency / permit
      |
  z8run/graph-flow external execution or human approval
      |
  outcome receipt -> separate observation -> NAL revision
      |
  temporal credit assignment -> update next choice
      |
  bounded follow-up cycle through canonical planner
```

**Zero automatic execution from `NARS f/c`, Axon goal score, entropy or LLM response.** Mark "proposed/validated/committed/dispatched/observed" as separate events with idempotency and versioning. FlowRunner session storage is an orchestration checkpoint; sealed evidence is the cognitive source of truth.

### Example end-to-end falsifiable scenario

Given an observed service degradation and two possible remediation options:
1. Ingest two observed premises with distinct evidence roots and valid/knowledge epochs.
2. NAL W3 revision updates support for hypothesis H (first run must not claim W3 is built).
3. ONA-style deduction/abduction composes candidates; truth/frequency/confidence are kept distinct from utility and costs.
4. Rubicon/H-G computes alternatives and records *one inert candidate*; Axon-style policy weighs benefit, cost, predictability, and desired maintenance horizon. Comparison stays pure.
5. Authorize the chosen implementation intention under fresh action policy. One denied RBAC path must **never** dispatch.
6. `graph-flow` pauses on human approval through the **existing graph-flow-lance ↔ z8run-lance shared handle seam** and resumes with only a version-pinned handle.
7. Simulated action produces observed outcome receipt. The next sealed cycle re-ranks options using that **independent** outcome rather than predicted self-reward.

**Visible trace:** actual NAL law, original roots, f/c, method compilation/version, goal candidate, gate/refusal, `KanbanMove`, workflow/session handle, actual outcome, temporal query viewpoint.

## Critical anti-duplication and dependency-cycle rules

1. `lance-graph-contract` never depends on `flow`, `rig`, `z8`, `cog`, `rs-graph-llm` or z8run; it is the zero/low-dependency authority of shared types.
2. The NAL core must not depend on LangGraph, Rig, z8run, an Axum server or a model provider. Method composition and scheduling stay separate.
3. Neither `lance-graph-flow` nor `lance-graph-z8` owns another state machine, task runner, session database or `lance-abi` registry. Reuse [`graph-flow-lance`](https://github.com/AdaWorldAPI/rs-graph-llm/tree/main/crates/graph-flow-lance) and [`z8run-lance`](https://github.com/AdaWorldAPI/z8run/tree/main/crates/z8run-lance) directly where possible.
4. If a new crate imports `graph-flow-lance` while `graph-flow-lance` is required to depend on the new crate, **stop and invert the trait dependency**, putting a minimal trait in contract or one direction only.
5. `lance-graph-rig` is a model boundary, **not the authority of truth**. A model response is text until an independently validated structured observation/receipt is admitted.
6. Pure NAL methods may produce `ProposedMethod` candidates; that does not grant execution permission. Never route around `commit_via` via the legacy action-commit bypass.
7. Do not reuse the same evidence root as independent support; no derived conclusion gains a fresh original observation root.
8. Only use a global epoch after proving correspondence with a sealed coherent read horizon; no alignment by unrelated Lance dataset version numbers.
9. Hypotheses and forecasts stay typed and can be refused by schema/policy; never auto-upgrade them into observed facts.
10. Disable tests for backpressure and respawn behavior until actual bounds/restart correctness is proven; label the live callcenter surface LAB-ONLY where necessary.

## Staged implementation with measurable acceptance

### F0: dependency and surface inventory
Inspect current cargo workspaces, live call paths and conditional feature gates. Produce **SHIPPED / TEST-ONLY / DORMANT / PROPOSED** status of the four candidate crates. Verify current #1536/#1539 and D-CMR reset. **Do not create any new crate yet.**

**Pass:** graph shows no dependency cycle; inventory demonstrates which existing adapter already fulfills each requirement.

### F1: one pure cognitive method, through existing NAL and R2IL
Select a W3 NARS revision oracle with two independent observation roots. Compile the named NAL operation as an R2IL fold and return typed f/c + witness. Negative control: overlapping evidence stamp would inflate confidence and must be rejected/choice-only.

**Pass:** source-pinned ONA actual output vs independent row oracle vs R2IL/mask-risc parity. Existing W1/W2 stay green.

### F2: cognitive-method binding, without extra framework
Prototype the smallest `lance-graph-cog` or existing-NAL **module** for cost-benefit + goal-score + inert choice. Test that f/c never accidentally becomes policy authority; candidate selection remains deterministic.

**Pass:** one input population through NAL and Rubicon score produces one inert candidate with trace; no external effect.

### F3: workflow bridge reuse
Extend the **existing** `graph-flow-lance` task seam to carry method/result handles where needed. Check same handle round-trip in z8run and graph-flow. Only add `lance-graph-flow` crate if it actually eliminates duplicated typed bindings in at least two consumers.

**Pass:** same source-generation-pinned handle, identical result, exact pause/resume; no rows/SoA in session or `FlowMessage`.

### F4: model and visual adapters
Use existing `graph-flow-lance::Completion` and `z8run-lance` registry. Only create `lance-graph-rig` or `lance-graph-z8` if there is a **tested capability genuinely missing** from the two existing adapters.

**Pass:** model only sees authorized materialized data, proposes but cannot commit; reconfigure/trace from visual workflow without copy of the runner or registry.

### F5: two-lineage Rubicon integration
Add typed precommit goal candidate ranking and postcommit maintained-goal/outcome-credit methods (not a second `KanbanColumn` machine). Enforce witness, freshness, revocation and authorization separately.

**Pass:** temporal outcome changes subsequent ranked choice; no-feedback control does not; permission-denied case produces zero effects; stale intention never executes.

### F6: stronger inference and temporal graph
Integrate W4/W5 value-lane + syllogism results and W7 epoch-aware inference as they land. Bind temporal graph views to the same reader horizon. Keep Hyperon/MeTTa expression equivalence and OpenPsi motivation as separate falsifiable parity programs.

**Pass:** no future-knowledge leak, no duplicate roots, exact output handles/provenance, no extra evaluator.

## Independent falsifier matrix

| Probe | Failure it must detect |
|---|---|
| An input with two repeated ONA evidential roots | confidence inflation without independent evidence |
| High-confidence belief with forbidden action | action executes without `commit_via` or fresh permit |
| Competing goals with identical utility / reordered iteration | unstable tie break or result depending on container iteration |
| Model suggests "commit" | LLM output crosses evidence/authority gate |
| One paused graph-flow session, then source republish | stale handle still resolves |
| One z8run replay/redelivery | duplicate action or duplicate original evidence root |
| Crash of a supervisor child | phantom success / no usable restart / runaway repeated respawn; distinguish currently expected failure from corrected implementation |
| Too many incoming requests | unbounded memory from drain, instead of explicit backpressure |
| Same relation selected at two valid/knowledge epochs | future-arriving evidence leaks backward |
| Causal inference reversed in time | identical causal conclusion as forward-time reference |
| Wrong typed ClassId reading | method accidentally reads unrelated u8 carrier bytes |
| Disengaged or revoked Rubicon goal | stale implementation-intention dispatch |
| Two providers with equal candidate output | non-deterministic mutation from ad hoc scheduler |
| Missing Rig or z8run optional feature | core NAL/cognitive methods fail to build |

## Sources / further reading

- [#1536 outer membrane/ONA/epoch](https://github.com/AdaWorldAPI/lance-graph/pull/1536), [#1539 corrections](https://github.com/AdaWorldAPI/lance-graph/pull/1539).
- [Original ONA folding-IR plan](ONA_NAL_R2IL_FOLD_METHOD_ASSIMILATION.md), [epoch research plan](EPOCH_TEMPORAL_GRAPH_RESEARCH_AGENDA.md), [Rubicon psychological plan](../../.claude/plans/rubicon-volition-v1.md), [Rubicon-Loco-rung convergence](../../.claude/plans/rubicon-loco-rung-cognitive-fabric-v1.md).
- [rs-graph-llm graph-flow-lance README](https://github.com/AdaWorldAPI/rs-graph-llm/blob/main/crates/graph-flow-lance/README.md), [graph-flow-kanban](https://github.com/AdaWorldAPI/rs-graph-llm/tree/main/graph-flow-kanban), [z8run-lance](https://github.com/AdaWorldAPI/z8run/tree/main/crates/z8run-lance), [Rig feature](https://github.com/AdaWorldAPI/rs-graph-llm/blob/main/graph-flow/Cargo.toml).
- [Axon/Rubicon](https://github.com/emer/axon/blob/main/Rubicon.md), [O'Reilly implementation](https://github.com/emer/axon/blob/main/axon/rubicon.go), [ONA pinned source](https://github.com/opennars/OpenNARS-for-Applications/tree/dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a/src), [Hyperon paper](https://arxiv.org/abs/2310.18318), [OpenPsi](https://github.com/iCog-Labs-Dev/hyperon-openpsi).

**Implementation rule:** start with the smallest **measurable coding PR** from current main; separate documentation, pure method parity, external workflow wiring and action effects. If the four proposed crates turn out to be mostly aliases around existing adapters, **the successful outcome is fewer new crates** and one tested interoperable flow.
