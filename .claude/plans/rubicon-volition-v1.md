# rubicon-volition-v1 — falsifiable goal -> plan -> action -> revision overlay

**Date:** 2026-10-12  
**Status:** PROPOSAL ONLY (RV-0..RV-9). No production code, new crate, opcode, actor VM, bit reassignment, wire contract, API or implementation authorization.  
**Literature:** [.arxiv/2026-10_rubicon_gollwitzer_volition_source_excerpts.md](../../.arxiv/2026-10_rubicon_gollwitzer_volition_source_excerpts.md).  
**Constraints:** [CE64 plateau](../knowledge/ce64-reasoning-substrate-plateau.md), [D-HL-0..10](ce64-hypothesis-layers-v1.md), [NAL polyfill](nal-polyfill-v1.md), [prediction != purpose](../../.arxiv/2607.21273_prediction_may_update_belief_not_purpose.md), [CARE commit gate](../../.arxiv/2606.14581_synthesize_programs_gate_commit.md).  
**Related but distinct:** [lance-graph PR #1496](https://github.com/AdaWorldAPI/lance-graph/pull/1496) CE64 plateau / hypotheses; [#1497](https://github.com/AdaWorldAPI/lance-graph/pull/1497) PLN x Shannon/attention scheduling; [#1498](https://github.com/AdaWorldAPI/lance-graph/pull/1498) Hyperon/MeTTa projection/folds. These are references, **not** dependencies to merge or modify. Preserve sibling isolation.

## One falsifiable question

Can a revocable if-then policy formed **after** evidence-sensitive goal selection (the Rubicon) achieve equal/better oracle-verified task outcomes with fewer unnecessary re-deliberations, **without** masking contradictory evidence or creating autonomous stale actions?

A cognitively plausible story is insufficient. Compare against matched budgeted baselines; negative results are acceptable.

## RV-0: preserve four separate contracts

1. **Carrier/ISA**: stable opaque LE bytes / byte-lane ALU, projection and mask contracts. Never conflate register layout with psychological, NARS or actor semantics.
2. **Epistemic + statistical evidence**: declarative ClassView reading; NARS f/c is statistical support/evidence stability when so interpreted; CE64 59..63 is an epistemic/topology/certification reading under the declared class, **not** goal intent; a different source class may interpret the same carrier differently. Keep EvidenceRootId and derivation lineage separate from action receipts; no intent inserted into bits 59..63.
3. **Goal/volition**: separate, versioned records for candidate goals, commitment, active policies, revocation and evaluation. This is purpose/authority, not inferred from confidence or entropy.
4. **Runtime/execution**: R2IL/fold/mask + ActionDef/actor-like dispatch are proposals that must pass semantic capability and permissions gates. An actor is a scheduling/ownership choice, never a claim that a graph edge itself executes.

Invariant: **predicted agreement != truth != desire/utility != authorization**. Moore activation or entropy may recommend examining evidence, never silently elevate a desire into an action commitment.

## RV-1: state machine (proposed, not implemented)

    Observation/Evidence ---------+
                                    v
    OPEN / PREDECISIONAL --evaluate desirability x feasibility--> CANDIDATE
                                    |
                    independent commit gate + authorization
                                    v
                               COMMITTED         <- RUBICON
                                    |
                   plan formation (if cue X, do Y)
                                    v
                               PLAN_READY
                                    |
                   matching cue + permit + live precondition
                                    v
                               DISPATCHED -> ACTING
                                    |
                       actual receipt / outcome observation
                                    v
                               EVALUATING
                                /   |   \
                     REPLAN <---    |    ---> RELEASED
                                COMMITTED
                         (continue if valid)

Transitions must be explicit and reconstructible by replay; no implicit COMMITTED transition from a belief update, LLM output, entropy focus or actor mailbox arrival. An implementation intention is a *conditional action policy*, not the goal itself.

### Transition law and failure paths

| Transition | Required proof input | Must refuse/silence when |
|---|---|---|
| candidate -> committed | selected GoalId, declared objective, resources, authorizer and snapshot | target unsupported, authorization absent, or conflicting active commitment policy |
| committed -> plan_ready | bound cue predicate, permitted ActionDef, rollback/termination rule | free-text wish not bound to declared vocabulary or unavailable capability |
| cue -> dispatch | active commitment, current version/lease, guard true, fresh enough data, effect permission | revoked, duplicate receipt/idempotency, stale snapshot, guard false or capacity exceeded |
| acting -> evaluating | executed action receipt; observed outcome or timeout labelled as such | LLM reports an imaginary tool call as execution |
| evaluating -> continue/replan/release | outcome against independent task oracle, budget, feasibility and counterevidence | merely high prediction accuracy, correlation-only witness, or simple retry counter |
| release -> dispatch | **forbidden**, unless a new independent commitment/plan version exists | stale event queued before cancellation |

Allowed to switch back to deliberative mindset when a critical assumption changes. Goal commitment creates focus but does **not** make evidence immutable.

## RV-2: typed candidate records (conceptual schema, not an ABI)

    GoalCandidate { goal_id, desired_outcome, feasibility_basis,
                    motivation_score?, utility_basis?, evidence_roots[],
                    horizon, constraint_set, proposer_id }
    GoalCommitment { goal_id, commitment_id, chosen_at, snapshot_id,
                     authorized_by, allowed_capabilities, budget,
                     state, version, expiry_or_review_at }
    IfThenPolicy { plan_id, commitment_id, cue_predicate_id,
                   action_def_id, guard_set_id, priority,
                   idempotency_key_scheme, source_plan_hash, version }
    TransitionReceipt { event_id, old_state, new_state, cause_event_id,
                        actor_or_executor_id?, action_receipt_id?,
                        evidence_snapshot_id, observed_outcome_id?,
                        decision_reason, monotonically_ordered_version }

All names are hypothetical **semantic fields**; do not reserve new bytes/tenant slots, alter CE64, introduce a binary wire format or assume existing OGAR APIs have precisely these shapes. A signed authorization token may be needed for real external side effects: resolve permission model separately.

### Edge semantics and causality

- A temporal message sequence yields predecessor/successor **ordering**, not necessarily causal contribution.
- CARE/causal certification requires interventions or identified causal evidence; an action receipt is not automatically a Causes edge.
- Revision of a belief can leave an intentional commitment active; revocation of a commitment need not erase its previously observed evidence.
- EventId, EvidenceRootId, GoalId, PolicyId and action receipt are **distinct identity domains**.
- A no-action result must also be replayable (why matching cue did *not* fire).

## RV-3: mapping into the current AdaWorldAPI landscape

| Cognitive responsibility | Candidate owner / existing substrate | Contract to reuse | Gap to verify |
|---|---|---|---|
| alternatives, feasibility, uncertain belief | NARS/NAL procedures on lance-graph + CE64 lenses | typed ClassView, NAL truth/provenance, quack joins, D-HL | independent utility/goal semantics, not a confidence-as-reward shortcut |
| candidate ranking / focus | existing Moore/Staunen and PR #1497 attention experiments | fixed evidence budget, masks and reproducible scheduler | Shannon H and EIG are separate from reward/commit |
| Rubicon commit | independent versioned decision gate, conceptually aligned to CARE program vs reference policy | candidate + deterministic reference + commit receipt | explicit who/what authorizes irreversible side effects |
| conditional plan | declarative ActionDef vocabulary in OGAR **if it really exists and has needed shape** | naming/binding -> typed op/procedure, not raw string trigger | inspect actual ActionDef/ractor/OGIT/ONA surfaces before selecting an owner |
| cue detection | Quack/R2IL/filter, potentially mask-risc resident match | row-address read mask x thinking mask, no extra row-sized materialization | event arrival order, negative conditions, time predicates, cue version |
| execution | action dispatcher, hypothetical actor/mailbox façade or existing runner | one owner of effects, idempotent receipt, cancellation fence | whether Erlang-style lightweight virtualization truly outperforms ractor; **no second VM by default** |
| observed outcome / validation | CE64/NARS provenance + independent fixture oracle | time-stamped observation, source roots, versioned snapshot | distinguish observation vs hypothesis vs post-hoc generated explanation |
| counterfactual + disengagement | CARE ablation/PEARL reading and revision where verified | independently test missing evidence, contingency, goal feasibility | no claim that deletion ablation establishes causation under overdetermination |
| reusable semantic graph | OGAR/RIG/ONA projection -> lance-graph | reference-stable addresses and edges | semantics of RIG/ONA not independently source-audited here; avoid fictional interface |
| agent orchestration | rs-graph-llm / LangGraph-style state transition, LangChain-style adapter boundaries as **comparison vocabulary** | explicit state/replay/edge bindings | no assumption that either external framework proves Rubicon equivalence |

**Boundary ruling:** do not directly implement Rubicon inside CE64 field layout or mask-risc. Keep motivational and volitional state external to truth/evidence lanes. Reuse the existing fold execution path for matching/aggregation only if a scalar semantic oracle certifies it. OGAR declarations are naming/binding/procedure surfaces, not an authorization channel by themselves.

**Actor virtualization comparison** (hypothesis only): compare existing ractor, an Erlang-inspired virtual actor lease/scheduler, and deterministic in-process event-loop control *using the same goal/policy/event types*. Count enqueue-to-actuation latency, suspended actor footprint, cancellation race leakage, message duplication, replay parity and failure recovery. This is a separate runtime test, **not** a prerequisite to the volition behavioral experiment. Avoid adding a new actor substrate until a measured insufficiency is shown.

## RV-4: controlled experiment matrix

Fixed tasks, evidence stream, candidate universe, action API, reference oracles, random seeds, wall clock/CPU and candidate budget per arm. Count action work **including** planning/rehearsal/monitor. Cost normalization must include IO, hydration, trigger compilation and all rejected work.

| Arm | Deliberation | Goal commitment | Cue if-then plan | Counterfactual and adaptive release |
|---|---|---|---|---|
| A — deliberative-only | per decision | none persisted | no | fixed task-level stops |
| B — Rubicon | on choice + invalidation | yes | no (ordinary subsequent deliberation) | fixed stops |
| C — implementation intentions | on choice + invalidation | yes | yes | fixed stops |
| D — full volition | on choice + invalidation | yes | yes | monitored revision / release |
| E — MCII decomposition (optional) | as D | yes | yes, with explicit obstacle contrast | same monitor; isolate cost of obstacle formation |

**Factorial cautions:** C vs B estimates conditional-plan contribution under matched goal commitments. D vs C estimates adaptive monitor/release contribution but may add compute; count it. E vs D estimates obstacle-contrasting effect if every other feature is held fixed. A vs D alone cannot attribute which mechanism caused the change. Arm B must not get special extra task knowledge during goal selection.

### Adversarial fixture set (at least two paired versions each)

- **RV-E1:** two mutually exclusive goals; new observation changes feasibility post-commit. Positive switch and negative no-switch controls.
- **RV-E2:** exact trigger; lexical near miss; revoked goal; cue arriving after expiry; duplicate delivery; cue replay after restart. Expected **zero** post-revoke actions.
- **RV-E3:** mental contrast selects an actually blocking obstacle vs misleading high-salience obstacle; test plan-only and contrast-only.
- **RV-E4:** impossible prerequisite versus late achievable payoff. Test justified release *and* premature abandonment (precision/recall for disengagement).
- **RV-E5:** narrative instruction vs canonical if-then predicate; once-rehearsed vs cached/repeated; equal preparation budget.
- **RV-E6:** irrelevant entropy/surprise distractors, correlated duplicate evidence and noncausal co-occurrence. Count false focus and erroneous commitment.
- **RV-E7:** underdetermined/conflicting witness: same statistical f/c but distinct epistemic certificate; actor must not infer causal action from a correlation.
- **RV-E8:** two independent actor execution orders, cancellation crossing an already enqueued task, compaction/restart/replay; same semantic outcome or explicit declared nondeterminism.

### Metrics, not claimed measurements

Behavior: independent task success rate, time-to-correct-goal, missed critical cues, false triggers, stale/post-revocation actions, correct releases, premature releases, false claims of causality and oracle-equivalent outcome traces.

Efficiency: per-episode total CPU/ns, wall latency p50/p95, decoded bytes/rows, resident bytes/goal+policy+actor, number of deliberative cycles, number of folds/replays, trigger candidate scans, read mask selectivity, IO/compaction, execution duplicates.

Safety/epistemics: evidence-source duplication inflation, unsupported actions, stale plan revisions, replay determinism, intervention-vs-observation confusion, receipt-without-actual-execution, permission checks bypassed.

Use confidence intervals across independently seeded episodes; report task distribution, environmental drift and effect heterogeneity. Never transplant Cohen d from human psychology into agent success predictions.

## RV-5: falsifiers and kill gates

1. **Rubicon not needed:** B is behaviorally indistinguishable from A and total cost rises or remains unchanged on tasks containing repeated decisions.
2. **Conditional plan not useful:** C misses or falsely fires as often as B at equivalent all-in budget; trigger engineering does not improve actual goal outcomes.
3. **Stale action leak:** any externally effectful action after a verified revoke/lease fence fails the safety gate.
4. **Inflexibility:** D is less accurate than C under concept drift and does not recover through re-deliberation.
5. **False causal promotion:** correlation or action order alone becomes a CE64 causal certificate without the declared test/witness.
6. **Evidence echo:** duplicated derivations inflate independent confidence or commit authority.
7. **Fabrication:** a replay trace cites a tool action not executed or hides a no-op/failed guard.
8. **Runtime overhead:** actor virtualization wins on handler microbenchmarks but loses all-in workload or violates cancellation/replay parity.
9. **Carrier drift:** a new motivational reading mutates CE64 stored bytes, ALU ISA or bits 59..63 without existing declared ClassView/evidence gate.
10. **Scope creep:** a second scheduler/opcode/runtime is proposed without a reproducible oracle counterexample to existing R2IL/mask-risc procedures.

## RV-6..RV-9: staged plan with stop conditions

| Stage | Artifact (proposal only) | Pass criterion |
|---|---|---|
| RV-6 source census | exact OGAR ActionDef / OGIT / ONA / RIG / rs-graph-llm and ractor symbols + SHAs; compare #1496–#1498 | no assumed API, no duplicate path; open questions resolved or typed-refused |
| RV-7 scalar oracle | tiny independent controller with pure state transitions; typed fixtures RV-E1..E8 | every reference transition + negative/no-action twin passes |
| RV-8 procedure lowering | compile read-only cue matching to existing Quack/R2IL/mask-risc, keep commit external | bit-exact guard parity, versions/cancellation invariant, physical decoded work counted |
| RV-9 factorial & ablation | A/B/C/D/E matched seeded episodes; then isolate actor runtime vs execution path | same task/evidence/seed/budget; uncertainty intervals; publish failures; **no merge/production authorization** |

### Explicit questions for later review

1. Is OGAR's ActionDef a declared operation, an executable task, a serializable action schema or an event payload? Is there a stable existing ID type?
2. What exactly are OGIT, ONA and RIG in *current* sources; are they semantics, vocabulary, runtimes or graphs?
3. Which owner performs goal authorization, and can a caller withdraw an externally queued effect?
4. Does rs-graph-llm already have a checkpointer, conditional routing, cancellation and replay, or is LangGraph only an analogy?
5. Are the CE64 epistemic states certified at the relevant class reading, or only probed on a separate projection?
6. Can a normal R2IL fold express all cue guards without a new primitive? Which concrete adversarial case proves a gap?
7. Does sparse overlay snapshot visibility permit durable cancel-before-effect fences under replay?
8. What is the independent task oracle when a goal's value, not its statistical confidence, is disputed?

**Next authorized work from this plan is only document review and source verification.** No implementation, merge, bit reinterpretation or external side effects are authorized by this proposal.
