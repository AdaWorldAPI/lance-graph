# D-SAG-CF-1: Inverted temporal ablation as a cross-reasoner epistemic probe

> **Aligned by [`sag-capstone-alignment-v1.md`](sag-capstone-alignment-v1.md)** — read it first: canonical `ProposedMethod`, the shared status vocabulary, D-id map, and the claims it marks stale (incl. "second hop refused").

**Status: RESEARCH RFC / NOT AN IMPLEMENTED ADAPTER.** Parent: [#1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509), the merged one-hop Software Awareness Graph (SAG). Relevant siblings: [#1496](https://github.com/AdaWorldAPI/lance-graph/pull/1496), [#1497](https://github.com/AdaWorldAPI/lance-graph/pull/1497), [#1498](https://github.com/AdaWorldAPI/lance-graph/pull/1498). Literature: [research map](../../.arxiv/2026-10_inverted_temporal_ablation_reasoner_oracles.md). Existing audit: [Gadamer ablation](../research/2026-10-recovery/GADAMER-ABLATION-RESEARCH.md).

## 0. The question: nine dots, one hidden boundary

What would the current conclusion, confidence, support set, future prediction and selected decision look like if an **observation**, **rule**, **interpretive method**, or **shared assumption** had never contributed? A further self-introspection experiment asks which conclusions fail when the engine suspends one of its *own* inference procedures.

A removal from a recorded history is not an inverse update on a compressed belief. A single-edge cut is not enough to establish actual causality under overdetermination. A counterfactual belief must never be written as observed truth.

Use factual world W0 and detached counterfactual world W_X. For a pinned horizon h, provenance closure P and common execution policy π:

    B0 = Eval(P, history<=h, π)
    BX = Eval(P without contributors/dependents of X, history<=h, π)
    Shadow(X) = Compare(B0, BX)

This is initially **model-relative counterfactual dependency**, not automatically real-world causal identification. Pearl do(X) additionally requires a specified structural causal model and intervention semantics. Jøsang subjective-logic fission is a useful analogy, **not** algebraically identical to NARS revision or an inverse of CE64 F/C bytes.

Three kinds of outputs remain distinct: changed truth/frequency and confidence; changed justification/provenance and certification; changed operational attention/choice. Record each axis separately. No claim of sentience is implied by the word introspection.

## 1. Existing code seams and their actual limits

| Seam | Exists | Limit / required interface |
| --- | --- | --- |
| `crates/lance-graph-planner/src/temporal.rs` | HLC/version-aware local causal trajectory deinterlacing; `QueryReference::at`, `EpistemicMode` | DATA-causal dependency closure is explicitly deferred. Default ref `u64::MAX` is **not** a pinned no-hindsight read; use a concrete head. |
| `crates/lance-graph-planner/src/chain_replay.rs` | Shared factual replay and first divergence | Replay-time truth revision can overcount shared evidence: existing Gadamer audit reports that replay_step lacks evidence-base independence guard. Do not trust it as exact retraction baseline until fixed or bracketed by an independent oracle. |
| `crates/lance-graph-planner/src/chain_counterfactual.rs` | Detached single-step cut, same replay kernel, distinct sequence range, counterfactual -6 tag; EdgeRole and threshold verdict | Only one `CutStep`; a threshold verdict can miss support changes, redundancy and joint cuts. |
| `crates/lance-graph-planner/src/pearl.rs` | Limited reason/reaction and intervention-arm comparisons | Not a general SCM do() for arbitrary graphs. Independent `crates/learning/src/scm.rs` exists but is not wired to CE64. |
| `crates/lance-graph-contract/src/revision.rs` | Gadamer claims/revision and counterfactual admission types | Nobody computes an end-to-end CounterfactualVerdict; TruthOnly is not safely equivalent to dispensable. Claim -> chain-step linkage is absent. |
| `crates/lance-graph-glove-parity/src/awareness.rs` | #1509 SAG projection with typed method/program/provenance tiers queried through Quack | In-memory and one hop; cannot silently pass a population mask between independent Quack programs. Needs a legal relation carrier or factored EqU32Via read. |
| OGAR ClassView / schema seal / MethodKey; CE64 / Epi5; mask-risc/R2IL | Existing typed registries and register/ISA interpretation contracts | Caller issues method identity. Never equate 8-bit ISA lanes with NARS fields; interpretations remain typed readings. No new opcodes or registry. |

Do not describe all these as already connected. This proposal adds the *contract for joining them*, not a second engine.

## 2. Minimal cross-reasoner contract

A reasoner adapter is a **read-only export and deterministic replay boundary**. Its hot-side projection is typed ordinals/IDs; strings, Narsese or MeTTa syntax belong at the adapter membrane.

Required input fields (logical schema; concrete representation deferred to OGAR hotplug):
- `world_id`, `horizon`, `source_version`, `reasoner`, `reasoner_commit`, `config_digest`, `seed`, `cycle_budget`, `allocation_policy`.
- `node_id` + `class_id` and `method_key` + `schema_seal` + `reading_id` where applicable; one stable identity != display label.
- Evidence: `evidence_id`, observed/asserted origin, `evidential_roots` (the actual independence basis), `source_time`, `ingest_time`, temporal event identity; attention/saccade traces **separate**.
- Derivation: `derived_id`, `rule_id@version`, premise IDs, conclusion ID, materialization epoch, bounded-stamp fidelity, admission/selection reason.
- `frequency`, `confidence`, NARS eternal/temporal distinction, and `epi5_class_reading` (not invented CE64 bits).
- Operation: prediction/anticipation, procedure/decision, priority/resource budget and explicit unknown/unsupported status.

Export event kinds: Observed, Assumed, Derived, Revised, Selected, Evicted, Predicted, Decision, Refused. If internal parentage is not exposed, report **partial provenance** and refuse exact descendant deletion. Never infer parentage solely from identical text or from bounded NARS stamps.

Ablation specification:

    { target_kind: Evidence | RuleVersion | Method | Assumption,
      target_ids: typed IDs,
      horizon: pinned version,
      mode: observational_leave_one_out | logical_method_suppression
            | conditional_contingency | structural_do,
      requested_readings: ..., budget: ... }

Result receipt:

    { factual_world, counterfactual_world, owner, horizon,
      target, first_divergence, affected_cone_ids,
      factual_and_counterfactual_support_roots, shared_ancestor_ids,
      delta_f, delta_c, verdict_before_after,
      certification_before_after, predictions_before_after,
      decisions_before_after, coverage_grade, cost, refusal_reasons }

No `structural_do` receipt may claim causal certification if no explicit validated SCM exists. The counterfactual world must be detached and tagged, and all observed rows must be byte-identical to W0.

## 3. Four ablations, plus the hidden fifth

1. **Evidence ablation:** Remove observation A everywhere its root contributes, including copies/transitive consequences. Sibling independent B must remain intact.
2. **Rule ablation:** Remove one versioned inference rule R1. Retract and re-evaluate all learned conclusions and procedures downstream from R1.
3. **Interpretive-method ablation:** Keep observation bytes but suppress the *method/reading* responsible for converting them into claims. SAG methods are addressable by MethodKey + seal + reading.
4. **Assumption ablation:** A and B appear independent but share upstream U; remove U and discover both supports collapse. Never use two apparent pathways as independent confidence evidence if their root provenance intersects.
5. **Attention-policy ablation (control):** Hold factual observations/rules fixed, change scheduler or cognitive budget. If conclusions change due to eviction or cycle selection, label the outcome **resource-path sensitivity**, not a demonstrated logical dependency.

The interesting 'nine-dot' crossing is (3) or (4): the important contributor can lie outside the apparent data graph. The claim is a testable research hypothesis, not novelty established by comparison.

## 4. Oracle matrix: different systems, different roles

| Adapter/oracle | Why it matters | First test | Interface/limits |
| --- | --- | --- | --- |
| **ONA** OpenNARS for Applications (C) | Temporal predictions, learned operations, attentional selection, resources | Replay one fixed Narsese history twice with the same build/config/seed/budget; suppress one input *before ingestion*; compare anticipation and procedural choice | CLI `./NAR shell < fixture.nal` and `python3 evaluation.py` are documented upstream. No claim ONA exposes complete derivation ancestry or has a direct retract API. |
| **OpenNARS** (Java) | Evidential stamps, temporal and eternal inference, non-axiomatic revision | Two independent roots vs duplicated evidential root vs partial overlap; distinguish revision from choice | Pin working upstream release and evidence-stamp semantics; bounded stamps are **not** full causal provenance. |
| **Hyperon/MeTTa** | Rewrite rules and rule-as-data make interpretation/method removal explicit | Same atoms, two rewrite rules; remove rule R1 and compare derived graph / trace | Pin interpreter/rewrite semantics and isolation. A MeTTa result is not NARS F/C without explicit mapping. |
| **Differential Dataflow** (Rust) | Independent incremental signed-update execution oracle | Insert/delete roots or rule facts, compare stabilized closure with scratch recomputation | Test lawful joins/fixed points and full proof multiplicity, not just Boolean membership. |
| **Soufflé Datalog provenance** | Independent explanation/proof oracle | Cross-check which tuples and proof trees support C | Minimal-height proof tree != complete support provenance. Exhaust alternative derivations before calling retraction exact. |
| **lance-graph Quack + R2IL** | Candidate optimized execution, typed mask semantics and SAG | Identical observations, outputs and support certificates vs the independent oracle | A one-hop DAG/query only today. No phantom multi-hop lowering, new opcode, or self-modification. |

Initial engineering order: ONA external black-box first for temporal effects; OpenNARS evidential control; Hyperon method-ablation; Differential Dataflow recomputation oracle; Soufflé provenance audits. An additional **local deterministic truth-table oracle** should be the first smoke test before running any foreign engine.

## 5. Exact execution protocol (external black-box first)

0. Pin Git SHA, toolchain, configuration, available resources, seed, cycle budget, query horizon and expected output parser. Normalize output to typed evidence receipts, **not** belief-string equality.
1. Compile or run each upstream reasoner's native upstream tests. Save version/output fingerprints as artifacts, do not vendor engines in the core crate.
2. In W0 ingest one synthetic event sequence, freeze output at a concrete h; repeat W0 identically and require deterministic normalized receipts or document the exact nondeterminism.
3. Build W_X from the **same pristine initialization** with *only* X excluded or R1 disabled at admission. Re-run the complete sequence, including learned rules. Compare at matched logical cycle and event time; do not mix wall-clock timing with logical version.
4. Use a fixed provenance DAG with proof alternatives, materialized derivations, root sets and rule versions; compute descendants. Verify incremental invalidation **against scratch re-evaluation** before attempting shortcut retractions.
5. Query candidate affected rows via SAG/Quack. The current one-hop restriction may require a legal relation-carrier materialization or repeated factored reads; explicitly quantify that cost.
6. Record two costs: whole-world replay and incremental-invalidation path. Compute speedup only after *identical observable outputs*. Attribute costs to ingestion, indexing, join, re-evaluation and I/O; report cold/warm, small/large-cone, density, fanout, cycles, p50/p95 and memory.
7. Produce detached, non-authoritative shadow receipt with `Coverage = Complete | Partial | Unsupported`. No automatic new truth, no causal-bit promotion, no graph transformation, no Rubicon commit.
8. Only after provenance and parity gates pass: feed `Reaction` and mapped `claim -> step` into Gadamer EncounterEvidence; leave TruthOnly unresolved unless the policy explicitly distinguishes redundancy.

**One-step vs joint cuts:** use bit positions to batch up to 64 *Boolean reachability scenarios* under a proved interpretation. This is **not** 64 CE64 truth replays or 64 SIMD registers. Each non-Boolean NARS revision needs its actual execution or a separately validated vectorization. Rung(0..9), CE64 bit, SIMD lane, scenario ordinal and Alpha overlay owner remain distinct axes.

## 6. Required fixtures and adversarial falsifiers

| ID | Fixture | Required outcome / failure signal |
| --- | --- | --- |
| F0 | Duplicate pristine baseline; same version/seed/budget | Byte-equivalent normalized receipts or documented nondeterministic envelope. |
| F1 | A -> R1 -> C, independent B -> R2 -> C; cut A | C survives through B, but support set loses A. |
| F2 | Above, cut R1 | C survives; no stale derivation attributed to R1. |
| F3 | Remove both R1,R2 | C loses all modeled supports. |
| F4 | Two nominally separate paths A,B both depend on hidden U; cut U | Both paths collapse; confidence cannot treat them as independent sources. |
| F5 | Duplicate input A under two aliases; cut A's source root | Both aliases retract; independent B unaffected. |
| F6 | Unrelated contributor X | No change to C, derivations, cert or decision. |
| F7 | A OR B; cut A then conditionally cut B | No but-for flip on first cut; A becomes pivotal under contingency B=off. Do not conclude A cannot be a cause. |
| F8 | A changes truth but not threshold verdict | TruthOnly / support change remains distinguishable from Inert and Necessary. |
| F9 | Delete interpretation method, not stored observations | Original observations remain identical; all and only rule-dependent descendants re-evaluate. |
| F10 | Hindsight injection after pinned horizon | Strict `QueryReference::at` never leaks future rows; default latest sentinel is not accepted as proof. |
| F11 | Cyclic or negated derivations, bounded evidential stamps, evictions | Conservative refusal, SCC/fixpoint or scratch replay; never fabricate complete provenance. |
| F12 | Factual vs detached counterfactual storage | No counterfactual result is certified/committed as observation; -6 tag where existing CE64 replay applies. |
| F13 | Quack one-hop vs multi-hop dependency | No illegal population intermediate; either use approved relation-carrier/eq-via or mark BLOCKED. |
| F14 | Baseline vs differential oracle under insert/delete/reinsert | Exact closure and support-count parity, including multiple derivations and shared root IDs. |
| F15 | Scheduler policy changes under fixed evidence | Changes in emitted decisions are labeled resource-path sensitivity; cannot claim underlying facts changed. |

The adversarial **disable test** for each gate must actually go red: allow duplicated roots, drop premise dependency, relax horizon pin, or accept a stale method seal; restore afterwards. A green disable run means the guard has no real test.

## 7. Efficiency question and kill conditions

If a dependency cone has size k within n recorded facts, a correct indexed deletion/update may avoid scanning all n; **O(k)** is not guaranteed because closure discovery, joins, recomputation, SCCs and provenance fanout have costs. First measure `T_full(n)`, `T_incremental(k,n)` and `T_index(n)`; include index build/amortization and memory.

Report two performance frontiers: a *small* dependency cone (candidate sparse advantage) and *near-whole-graph* cone (candidate pathological case). Compare to vanilla replay (ONA), Differential Dataflow incremental maintenance, and Quack baseline. KILL incremental optimization at measured regimes where full replay is faster; KILL exact retraction when provenance is incomplete; KILL causal-certification claim without a declared intervention/SCM; KILL method introspection that cannot identify and version the rule actually used.

Use entropy/expected information gain **only to schedule which falsifiable cut to try next**, not to increase belief confidence. For a budgeted probe choose candidate X using cost-aware expected reduction in uncertainty over hypotheses, then measure its realized discrimination; include random, maximum-confidence and highest-graph-degree baselines. Never require new CE64 bits to store importance, responsibility or entropy.

## 8. Implementation waves and acceptance gates

- **W0: Protocol and oracle fixture (this PR).** One pinned spec, references, expected trace schemas and a runnable, dependency-free Boolean/proof-DAG smoke oracle. No external-engine results claimed.
- **W1: ONA shell adapter.** Native fixtures + reproducible external replay; extract stable IDs, evidential origins, temporal predictions and output decisions, mark unknown ancestry Partial. Gate F0,F1,F6,F8,F15.
- **W2: OpenNARS evidential control.** Stamp independence/revision vs choice, duplicate roots and NARS truth-law parity. Gate F4,F5,F11; do not derive exact fission from bytes.
- **W3: Hyperon method-as-data ablation.** Rule/assumption rewrite suppression and inspect alternatives. Gate F2,F3,F9; distinguish model interpretation from source data.
- **W4: Independent incremental oracle.** Differential Dataflow plus Soufflé explanation, full recomputation truth oracle, signed updates and multiset derivations. Gate F7,F11,F14.
- **W5: SAG/Quack projection.** Map to validated OGAR MethodKey, schema seal, typed reading, tier and provenance. Solve the legal multi-hop seam without new engine; gate F10,F12,F13.
- **W6: Gadamer/Rubicon bridge (separate semantic decision).** Claim-to-step identities + distinct Necessary/Redundant/TruthOnly/Unknown admission policy. No automatic writer or intention inference.
- **W7: Bench and active experimental selection.** Bound memory and runtime, ablation selection with entropy/expected information gain; publish all results and negative controls.

**Review questions before production:** What is the minimum complete evidence root key (not stamp alias)? How can a method/rule be versioned across engines? Are learned rules retracted, or must memory be rebuilt? What does TruthOnly mean at the Gadamer gate? How do we represent incompatible MeTTa and NARS truth semantics without lossy conversion? What is the legally allowed multi-hop Quack carrier? What performance regime actually rewards sparse delta maintenance?

## 9. Claude Code / implementer prompt

> Read this RFC, the SAG #1509 plan, Gadamer ablation audit and the research map before writing code. Inventory the exact current commit and the real methods in temporal.rs, chain_counterfactual.rs, chain_replay.rs, revision.rs and awareness.rs. First run the provided synthetic oracle and demonstrate that disabling each guard breaks its named falsifier. Implement **one** external ONA fixture runner at a pinned upstream commit: baseline vs clean-room re-ingestion with one evidence source suppressed; normalize outputs into an explicitly partial receipt. Validate identical baseline executions and do not claim ONA supports retraction without code evidence. Keep the observed history immutable and the counterfactual world detached. Run F0/F1/F6/F8/F15; report the measured output and any inaccessible provenance. Then propose the smallest OGAR/ClassView adapter and an allowed one-hop Quack query. Never silently reinterpret the CE64 register as NARS-specific, invent an opcode, revise factual confidence from a counterfactual, or promote a logical deletion into Pearl causality. Split unresolved semantic decisions into follow-up PRs; no unverified shipped claims.

## 10. References and traceability

- Upstream: [ONA C repository](https://github.com/opennars/OpenNARS-for-Applications), [OpenNARS Java](https://github.com/opennars/opennars), [MeTTa / Hyperon](https://github.com/trueagi-io/hyperon-experimental), [Differential Dataflow](https://github.com/TimelyDataflow/differential-dataflow), [Soufflé provenance](https://souffle-lang.github.io/provenance).
- Papers and paraphrased research excerpts: [research map](../../.arxiv/2026-10_inverted_temporal_ablation_reasoner_oracles.md) and [pre-existing Jøsang/NARS excerpts](../../.arxiv/2026-10_nars_self_epistemic_needs_evidence_primary_excerpts.md).
- Prior work: [#1509 SAG](https://github.com/AdaWorldAPI/lance-graph/pull/1509), [#1496 NARS](https://github.com/AdaWorldAPI/lance-graph/pull/1496), [#1497 PLN/entropy](https://github.com/AdaWorldAPI/lance-graph/pull/1497), [#1498 Hyperon](https://github.com/AdaWorldAPI/lance-graph/pull/1498). Existing [causal-replay plan](dismech-causal-replay-v1.md) and [actual-causality map](../../.arxiv/1608.07879_actual_causality_needs_contingencies_not_just_deletion.md).

**Explicitly not shipped:** ONA/OpenNARS/Hyperon adapters, causal identification, dependency-complete NARS traces, Quack recursive traversal, proof-correct sparse deletion, CE64 confidence fission, self-modification, or any claimed performance improvement.
