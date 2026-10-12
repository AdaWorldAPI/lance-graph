# D-SAG-2 — Wisdom frontier: OpenPsi x Rubikon x Shannon x CARE x ONA/Hyperon

**Status:** RESEARCH / FALSIFIABLE DESIGN ONLY, 2026-10-12. Sibling extension of merged [D-SAG-1 / #1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509). **No implementation, runtime evaluation, opcode, wire change, new graph engine, bit migration, authorization change, or speedup claim.**

**Read alongside:** [D-SAG-1](software-awareness-graph-v1.md), [D-MATURATION3-0](../knowledge/maslov-pyramid-of-cognition.md), [D-RV](rubicon-volition-v1.md), [PLN/Shannon #1497](../../.arxiv/2026-10_opencog_pln_shannon_attention_research_map.md), [CARE paper notes](../../.arxiv/2606.14581_synthesize_programs_gate_commit.md), and [new annotated sources](../../.arxiv/2026-10_openpsi_rubicon_care_wisdom_frontier_source_excerpts.md). A self-contained execution prompt lives at [sag-wisdom-frontier-claude-code.md](../prompts/sag-wisdom-frontier-claude-code.md).

## Thesis and falsifiable question

Can a stage-0 **validated wisdom anchor** reveal an expanding frontier of unresolved, evidence-grounded consequences; can masked popcount find the smallest epistemic holes cheaply; and can OpenPsi-like intrinsic demands, calibrated expected information gain (EIG), CARE-style reference-conditioned program selection and Rubikon commitments improve **useful independently verified outcomes per total unit of work** relative to deterministic schedulers, without self-support, stale action, outcome leakage or a second execution engine?

Three roles **must not be collapsed**:
1. **Epistemic maturation:** where an individual explanation stands (experimental stage 7 -> 0).
2. **Intrinsic motivation:** how strongly an *investigation* should compete for a scarce budget, computed from demands and expected learnability.
3. **Volition/authority:** whether an intention is committed and whether an action is permitted to execute.

Information gain does not prove causality. A program selected by CARE does not acquire execution authority. A stage-0 anchor is not omniscient; it has demonstrated downstream dependence under an isolated ablation.

## Observed integration baseline, not hypothetical infrastructure

- #1509 projects real MethodTable/prepared methods into SoftwareGraph typed nodes/edges with tiers: declared / derived / compiled / verified / hypothesis, then runs **ordinary Quack Draft -> mask-risc**. It reports only **one concept and one hop**. Source revision can be caller-supplied, graph remains in memory, and multi-hop population intermediate is unresolved.
- Its current H1 candidate is selection fusion for billable_hours / MaskedSumI32. Scalar reference times are **not** timings of a fused SIMD implementation. Predicate ternlog and FusedFold terminal gaps remain.
- #1494 experimentally runs pearl::hydrate with provenance roots, feedback, stage transitions, anchor ablation; no production feedback or production maturation band. #1497 separates inference, scheduling and physical execution. #1500 separates goal selection, conditional commitment and permit to act.
- ONA's public C/evaluation.py suite, Hyperon/MeTTa tests, modern PLN/PeTTa, and Hyperon OpenPsi are **distinct oracles**. The ONA v0.9.3 MeTTa-NARS bridge gives a real integration seam, **not** a guaranteed semantic equivalence proof.

## 0. Contract and safety invariants

| Surface | Existing fact / permitted research | Hard prohibition |
| --- | --- | --- |
| ISA / byte lanes | Stable register/ALU bytes; semantics supplied by ClassView + provenance | Do not put OpenPsi or CARE laws in the universal byte layout |
| CE64[50..52] | **Production plasticity**, read by learn, MooreNars16 and V3. #1494 uses a **test-local, explicitly declared** Maturation3 (7..0) only | Do not reinterpret an existing provenance as maturation or overwrite plasticity |
| CE64[59..63] | Declared EpistemicState5 topology/certification reading | No desire, entropy, or commit bit in epistemic state |
| Statistical support | NARS f/c and observation-root ancestry under declared readings | Do not relabel NARS confidence or PLN strength as calibrated probability |
| Evidence | Derived results carry union of **original observation roots** and no new root; exclude self/descendant support | No mutually grounded cycles, false root independence, or certification through hydration alone |
| Intent | Versioned goal/intention/plan/permit/receipt outside CE64 | Never execute effects because confidence/entropy/goal intensity rose |
| SAG | Existing ClassView/MethodTable/SoftwareGraph -> Quack -> mask-risc | No new index, graph engine or shadow methods registry; do not assert multi-hop works |

## 1. Suggested probe-local record shape (NOT a Rust ABI)

    FrontierItem {
        anchor_id, relation_id, reading_id, source_rev, tier,
        required_mask, established_mask, refuted_mask,
        original_root_ids, derivation_parent_ids,
        maturity_test_local, candidate_hypotheses,
        provenance, cost_estimate, authorization_scope
    }
    Proposal {
        program_hash, schema_seal, data_revision,
        evidence_cutoff, ordered_candidates, scores,
        reference_score, challenger_score,
        invariant_verdict, rubicon_state, decision_receipt
    }

Identity is typed (method key / ClassId / reading / schema seal), never just display text. No inferred raw CE64 bit position stands in for an external evidence-root ID, witness ID or decision receipt.

## 2. Stage-0 seed -> popcount -> recursively exposed frontier

At a validated stage-0 anchor, enumerate **only relations reachable through a documented legal one-hop projection** and which have an eligible expected consequence; do not pretend Quack supports arbitrary traversals. If a second hop is required, materialize a verified scalar/relation carrier or stop with an explicit MultiHopBlocked, never silently add a second engine.

For each candidate typed relation, on a declared prerequisite universe U:

    missing = required & ~(established | refuted)
    blocked = required & refuted
    debt = popcount(missing)

- Bit sets are **schema-defined prerequisite masks** with a declared bit universe, not popcount over raw CE64 or arbitrary evidence bytes.
- Distinguish **unknown** from **refuted** and **unobserved** from **false**. A candidate with blocked != 0 cannot be auto-hydrated.
- debt == 0 means **ready for an independent validation check**, *not true*; debt == 1 nominates a cheap discriminating question; larger debt stays possible if EIG/utility justifies cost.
- Candidate source must be a genuine verified/compiled edge at an eligible tier; hypothetical edges stay hypotheses.
- A derived relation may become a premise *only after resolution*, inheriting unique observation roots. No descendant becomes a premise for its own ancestor, even after reopening.
- Iterate to a **bounded fixed point** with seeded work queue, deduplication, hop/row/time budget, policy version and snapshot; log per-iteration delta and root set. Stop at zero new valid resolutions; repeated pass must be byte-idempotent.
- A new observation contradicting the *derived relation* starts a stage-7 episode for that relation; it does not regress an unrelated stage-0 anchor. Direct counterevidence against an anchor reopens that anchor and its descendants as appropriate.

This is **fractal-like recurrence of the same maturation pattern**, not a claim of mathematical fractal dimension or unbounded growth. Measure branching and depth empirically.

## 3. OpenPsi demand and intrinsic-value policy

Treat OpenPsi demand / modulator / goalSelector as **architectural references**, not a ready-to-import component. Research mapping:

| OpenPsi / psychology | Overlay candidate |
| --- | --- |
| curiosity, competence, resource pressure | demand measurements and budget signals (separate from evidence) |
| demand arbitration and contextual modulation | pure, reproducible candidate scheduler |
| Heckhausen predecisional deliberation | compare goal desirability, feasibility and risk |
| Rubikon | explicit versioned goal-intention commit, revocable |
| Gollwitzer implementation intention | cue -> guarded ActionDef candidate |
| Kanban | Proposed -> Deliberating -> Committed -> Planned -> Ready -> Executing -> Evaluating -> Replan / Hold / Disengage |
| outcome-driven demand revision | external observed outcome receipt; do not reward predicted self-confirmation |

OpenPsi [PR #117](https://github.com/iCog-Labs-Dev/hyperon-openpsi/pull/117) documents a current tie-breaking and demand->goal mapping defect; probe tied demands, iteration permutations and genuinely distinct goals. Its status/merge are not assumed.

The Rubikon psychological commitment is **not** the execution permit. Permission, capability, current cue, replay idempotency and revocation must each be checked at effect time. Failed permits produce a refusal receipt, not an action.

## 4. Shannon, EIG and causally useful exploration

For a **pre-registered, calibrated distribution** over rival explanations Theta, define:

    H(Theta | D) = - sum_i p_i log2(p_i)
    EIG(a) = H(Theta | D)
             - E_{outcome ~ predictive(a,D)} H(Theta | D, outcome)

NARS f/c, PLN truth values and CE64 certification are not p_i by default. If calibration cannot be demonstrated, abstain from entropy/EIG claims or use a separate explicitly declared model. Include a high-entropy **irreducible-noise TV** that naïve uncertainty scheduling wastes budget on, and a modest-entropy *discriminating intervention* with real downstream consequences.

Optional **hypothesis** (not a law):

    priority(a) = [EIG(a) * P(independent causal usefulness | D,a)]
                  / [estimated_total_work(a) + epsilon]

Pre-register how causal usefulness is estimated **without peeking at outcomes**. CARE here has two independently named roles:
- **CARE paper (Liu et al., arXiv:2606.14581v5):** persistent executable ranking program as challenger to non-LLM reference; validate program and score every eligible candidate; gated pre-outcome decision; immutable record; post-outcome update.
- **Causal ablation / Pearl-like counterfactual tests:** a separate local falsification oracle, **not** a feature the paper itself proves or a substitute for its program-selection gate.

Never reward entropy for entropy's sake. Penalize useless curiosity through measured outcomes, not circular proxies.

## 5. CARE-style proposal and Rubikon lifecycle

    READ pre-outcome snapshot S_t (seed, population, allowed evidence roots)
      |-> reference: fixed deterministic popcount/FIFO/confidence policy
      |-> challenger: optional versioned (R2IL/Quack where representable)
      |                 ranking program proposed from SAG + demand state
      |-> validate program: bounded, pure, finite score for ALL candidates,
      |                     permutation invariant, no hidden side effects,
      |                     schema+reading+rev compatible, budget counted
      |-> compare using **only pre-outcome evidence** and an independently
      |   specified deterministic gate; on fail choose reference
      |-> record hash, inputs, scores, ranking, reasons, gate, version, time
      |-> RUBIKON: commit *goal*; conditional plan requires live permit
      |-> actual outcome / policy refusal recorded after selection
      |-> update demand / epistemic evidence / disengage or replan
      |-> optional new knowledge can enter next snapshot only

No proposed optimizer H1 is performed automatically. Any optimization is a proposal checked by semantic equivalence, NULL/bag/set, cost tier and authority, and requires an authorized effect separately.

## 6. Falsifier ledger (proposed D-SAG-2-0 .. 15)

| ID | Fixture / comparison | Failure condition |
| --- | --- | --- |
| 0 | Identical snapshot/candidate ids, fixed RNG, budgets | incomparable baseline or post-outcome evidence leak |
| 1 | One-anchor A->B->C; no feedback vs valid feedback | new root manufactured; no expected fixed point |
| 2 | Independent roots vs two paths to same root | dependent evidence counted twice |
| 3 | Ungrounded A->B->C->A; grounded cycle then anchor removal | self-grounding or resurrection after ablation |
| 4 | Source graph edge changed after a certificate | stale evidence accepted as verified |
| 5 | One missing, many missing, zero missing, explicitly refuted | refuted/unknown conflation; 0 missing -> certified |
| 6 | Permute source rows/chunks, ties, vector lanes | ordering changes selected semantic candidate |
| 7 | High-entropy noisy-TV vs learnable low-entropy cause | naive entropy falsely promoted as successful curiosity |
| 8 | Reward delayed but reachable vs goal impossible | perseverance in impossible goal; premature disengagement |
| 9 | Cue, near miss, revoked permit, cancellation race | action executes without live permit |
| 10 | Hold ONA and Hyperon rules fixed, change scheduler only | claimed benefit actually derives from different reasoning |
| 11 | Hold scheduler fixed, vary ONA/PLN native inference engine | semantic differences masked by adapter |
| 12 | ONA/MeTTa-NARS bridge: bindings, f/c, time, truth operations | adapter coerces non-equivalent semantics |
| 13 | CARE challenger rejects nonfinite/partial/order-dependent scores | invalid challenger commits anyway |
| 14 | SAG #1509: tier mismatch, two-hop query, H1 fused vs scalar | invented shipped feature, or unauthorized optimizer |
| 15 | Whole-work accounting and matched seed/oracle | apparent advantage vanishes with compile/IO/rejections |

For *each* passed positive probe, disable its key guard once and demand a RED test, then restore. Negative/silence twins must remain quiet; no downstream action or EIG inflation when input contains no new valid evidence.

## 7. Four controlled axes and reporting

1. **Oracle/inference:** ONA native, ONA through the MeTTa-NARS bridge, Hyperon/MeTTa + modern PLN (and scalar independent oracle). Pin commits/dependencies; first establish semantics, do **not** force identical truth laws.
2. **Scheduler:** FIFO, confidence-only, smallest popcount debt, raw Shannon entropy, calibrated EIG, EIG + stage and OpenPsi demands, optional CARE challenger.
3. **Commit discipline:** re-deliberate every step, Rubikon commit, Rubikon + if-then cue, revocable policy + evidence/receipt gate. Evaluate each against identical environments.
4. **Physical execution:** scalar reference, existing Quack Draft + mask-risc. SIMD vs scalar tested **only after byte/semantic parity**; no new Quack variant or parallel actor VM.

Report **unique oracle-supported discoveries / total work**, intervention success, regret/best-so-far, incorrect commitments, false-causal rate, response to contradicted anchor, original-root inflation count (must be 0), exact candidates/budget, CPU/memory/decoded bytes/allocations, compile+validation+refusal+replay+IO cost, and per-tier timings. Keep zero-result rows.

## 8. Minimal experiments and gates

- **P0 static census:** source SHA of ONA, Hyperon, OpenPsi, PLN and CARE version; source license; current actual code paths for D-SAG-1, masks, hydration, root derivation, Quack one-hop, ActionDef and permit. Mark **UNVERIFIED** rather than inventing API mappings.
- **P1 scalar oracle:** fixture-based deterministic frontier + reference scheduler, no adapters or agent. Prove bitset semantics, ancestry and fixed point; use test-local maturity metadata. Include disable/red test.
- **P2 existing Quack:** map only what legally lowers. Explicitly refuse multi-hop if no legal representation. Compare scalar byte/results, NULL and bag/set.
- **P3 foreign replay:** ONA eval.py, ONA MeTTa-NARS examples, Hyperon/OpenPsi tests and PLN tests in isolated pinned environments. Normalize *observables*, not truth formulas; record non-parity.
- **P4 preference + volition:** train/calibrate EIG only on training split; apply reference-conditioned CARE gate with timestamped receipts, then optional simulated permitted action; verify revocation.
- **P5 SAG self-inspection:** propose H1 or a capability-gap investigation from #1509; compare to scalar, no automatic source mutation. Promote only under separate approval.

### Stop / kill conditions

Terminate or narrow scope if any test silently reinterprets production CE64 50..52, treats 59..63 as motivation, accepts a loop as independent support, allows invalid/stale proposal to commit, peeks after outcome, silently simulates multi-hop, uses different candidate sets or original roots across schedulers, or reports speedup for an unimplemented H1 terminal.

**Decision requested from operator later, not implicit in this PR:** whether and how Maturation3 is promoted beyond test-local provenance without changing the stable byte-ISA contract; and whether population carrier for legal multi-hop already exists. Until then all projections are research overlays.

**Status of this PR:** documentation and experiment specification only. None of P0-P5 is claimed executed merely because this file exists.


## Deferred extension on the same PR: A16..A33

The first-wave D-SAG-2-0..15 falsifiers deliberately focused on establishing the positive, bounded stage-0 frontier and independently validated CARE-style proposal. The following **second-wave research**, added to #1511, covers safety and epistemic failure modes not yet exhaustively specified in that first wave:

- [Hard-problems / incremental truth maintenance](sag-wisdom-frontier-hard-problems-v1.md): **four-state prerequisite debt** (known / refuted / missing / unread); alternative sufficient supports; deletion/retraction against fresh full-recomputation; source/version invalidation; bounded work and starvation; rubicon commitment hysteresis; conditional permits and cancellation races; graph ablation vs intervention; field-by-field ONA/MeTTa/PLN oracle parity.
- [Second-wave .arxiv source excerpts](../../.arxiv/2026-10_wisdom_frontier_active_inference_delta_and_misspecification.md): Friston et al. 2026 active inference and structure learning, Wei 2024 EFE/value, Catanach/Das 2023 misspecified EIG, Pathak 2017 curiosity, Burda 2018 RND, McSherry et al. 2013 differential dataflow, Beckers 2021 actual causation, plus source limitations.
- [Adversarial Claude Code handoff](../prompts/sag-wisdom-frontier-adversarial-expansion.md): implements **only A16-A19 first in a future implementation PR** (subject to audit and scope decision), then stages A20..A33 as unexecuted red tests.

This extension supersedes no proven claim in #1494/#1509. It does not endorse introducing Differential Dataflow, a new physical graph traversal, cryptographic receipt, reassigning CE64 bands or deploying real-world IAM effects.

**Additional stop condition:** Budget exhausted, source unread, stale certificate, model uncalibrated and lack of causal identifiability must each have a distinct explicit result; none may be silently reclassified as fixed point, negative evidence, successful exploration or permission to act.
