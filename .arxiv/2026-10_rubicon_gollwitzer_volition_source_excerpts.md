# Rubikon, implementation intentions and goal disengagement — source excerpts

**Date:** 2026-10-12  
**Status:** HIGH-SIGNAL LITERATURE DISTILLATION / PROPOSAL INPUT ONLY. All excerpts are **paraphrases**, not verbatim quotations. Source claims, conjectured architecture transfers and unresolved questions are deliberately distinct.  
**Read with:** [rubicon-volition-v1 plan](../.claude/plans/rubicon-volition-v1.md); [CE64 plateau](../.claude/knowledge/ce64-reasoning-substrate-plateau.md); [NAL polyfill](../.claude/plans/nal-polyfill-v1.md); [prediction is not purpose](2607.21273_prediction_may_update_belief_not_purpose.md); [CARE synthesis and commit](2606.14581_synthesize_programs_gate_commit.md).  
**Paper identity note:** this directory is a literature shelf, **not** a claim that the sources below are hosted on arXiv. DOI/publisher/PMC links are canonical here. Last source check: 2026-10-12.

## Filter and thesis

Keep a source only when it identifies a distinct causal mechanism, moderator, failure mode or reproducible contrast that could inform a falsifier. Do **not** infer agent efficacy, cognitive equivalence, ISA semantics or throughput from behavioral psychology outcomes. Central, falsifiable translation hypothesis:

> A goal committed after deliberation and represented by a revocable situational if-then plan can reduce recurrent deliberation cost and missed opportunities, while an independent epistemic/causal monitor can prevent goal fixation when observations invalidate the plan.

This is **an engineering hypothesis**, not an empirical conclusion of the sources.

## S1 — Heckhausen & Gollwitzer (1987): model of action phases / Rubicon

**Grounded through:** Brandstätter & Bernecker (2022), *Persistence and Disengagement in Personal Goal Pursuit*, Annual Review of Psychology 73:271–299, https://doi.org/10.1146/annurev-psych-020821-110710 (section "Model of action phases, mental contrasting, and implementation intentions"). The original 1987 treatment is referenced there; **the primary 1987 text was not directly inspected for this note**.

**Exzerpt (secondary synthesis):** predecisional deliberation assesses desirability and feasibility; crossing the Rubicon means forming a binding goal intention; preactional planning determines when/how; actional pursuit executes; postactional evaluation compares outcomes to standards. This is an analytical distinction between motivational and volitional processing, not proof that a human controller consists of four rigid, irreversible states.

**Transfer candidate:** distinct GOAL_CANDIDATE, COMMITTED, PLAN_READY, ACTING, EVALUATING, RELEASED states, with observable transitions and explicit ability to re-enter deliberation. **Falsifier:** identical behavior and compute in an always-deliberative arm versus a stateful controller under an environment with delayed/opportunistic triggers.

## S2 — Gollwitzer (1999): implementation intentions / strategic automaticity

**Primary:** Gollwitzer, P. M. (1999), *Implementation intentions: Strong effects of simple plans*, American Psychologist 54(7), 493–503, https://doi.org/10.1037/0003-066X.54.7.493.

**Exzerpt:** a goal intention ("achieve Z") and an implementation intention ("if situation X occurs, perform Y") are different commitments. A preformed situation-response link can make the relevant cue more accessible and delegate initiation to the situation without re-deciding every action.

**Transfer candidate:** a declared trigger predicate and permitted ActionDef selected *after* goal commitment. Trigger matching does not authorize action by itself: goal still active, permissions, resources, evidence snapshot and cancellation rules must all hold at execution time. **Falsifier:** stale-cue triggering after revoke, or "automaticity" requiring an LLM/long deliberation for every observed cue. **Silence twin:** matching cue with revoked goal must produce NO action.

## S3 — Gollwitzer & Sheeran (2006): effect/process evidence, not an architecture law

**Primary review/meta-analysis:** *Implementation Intentions and Goal Achievement: A Meta-analysis of Effects and Processes*, Advances in Experimental Social Psychology 38:69–119, https://doi.org/10.1016/S0065-2601(06)38002-1.

**Exzerpt:** strong goal intentions alone do not guarantee enactment; pre-specifying when, where and how to act can help with failures to initiate, distractions and unwanted habits. Published average effect often cited as d≈0.65; source represents a heterogeneous body of psychological experiments, not a deterministic guarantee and not a machine-agent effect size.

**Transfer candidate:** separate goal-selection score from execution reliability, measure false positives *and* missed opportunities. A plan that raises actuation count but lowers completed objectives is a failure.

## S4 — Jutta Heckhausen, Wrosch & Schulz (2010): control, life-span opportunities and letting go

**Primary:** *A Motivational Theory of Life-Span Development*, Psychological Review 117(1):32–60, https://doi.org/10.1037/a0017668 ; open manuscript https://pmc.ncbi.nlm.nih.gov/articles/PMC2820305/.

**Exzerpt:** primary and secondary control and opportunities/constraints across life stages affect appropriate goal engagement and disengagement. Feasibility and time windows matter; persistent pursuit is not always adaptive.

**Transfer candidate:** a goal is not permanent when feasibility, available resources or environmental opportunity changes. A time-limited goal lease / review horizon is a **design choice**, not a claimed psychology mechanism. **Falsifier:** agent continues executing a previously promising goal after budget exhaustion or a disproven prerequisite, while an equally funded revoke-and-replan reference succeeds.

## S5 — Wang, Wang & Gai (2021): MCII / mental contrasting has limits

**Meta-analysis:** *A Meta-Analysis of the Effects of Mental Contrasting With Implementation Intentions on Goal Attainment*, Frontiers in Psychology 12:565202, https://doi.org/10.3389/fpsyg.2021.565202 ; open full text https://pmc.ncbi.nlm.nih.gov/articles/PMC8149892/.

**Exzerpt:** 21 studies, 24 effect sizes, N=15,907; pooled Hedges g≈0.336 (95% CI 0.229–0.443) for mental contrasting plus implementation intentions (MCII). Publication bias is a concern; trim-and-fill estimate ~0.242. Intervention delivery matters; underlying parts cannot be assumed synergistic from cross-study comparisons. Oettingen's mental contrasting is an independently developed research line; do not falsely attribute it solely to Gollwitzer.

**Transfer candidate:** compare (i) a plan, (ii) obstacle/counterfactual generation without a plan, (iii) the combined mechanism, on a **common fixed task set**. **Falsifier:** obstacle generation consumes budget but produces no higher recovery success than plan-only; do not simply label obstacle listing "causal inference".

## S6 — Brandstätter & Bernecker (2022): persistence **and** disengagement

**Primary review:** *Persistence and Disengagement in Personal Goal Pursuit*, Annual Review of Psychology 73:271–299, https://doi.org/10.1146/annurev-psych-020821-110710.

**Exzerpt:** terminating or adapting pursuit can be part of successful self-regulation; factors differ for persistence versus abandoning unattainable goals. The review links action phases, mental contrasting and alternative disengagement theories, rather than claiming that goal deactivation automatically follows a single Rubicon model.

**Transfer candidate:** explicit CANCEL / SUSPEND / REOPEN decisions with machine-checkable reasons, not simply exceeding a retry count. **Falsifier:** premature cancellation when delayed rewards are still reachable; also test sunk-cost persistence.

## S7 — Sheeran, Listrom & Gollwitzer (online 2024; journal issue 2025): 642 tests

**Meta-analysis:** *The when and how of planning: Meta-analysis of the scope and components of implementation intentions in 642 tests*, European Review of Social Psychology 36(1):162–194. Published online **27 March 2024**, issue dated 2025. https://doi.org/10.1080/10463283.2024.2334563.

**Exzerpt:** Across **642 independent tests**, reported effects of if-then planning on cognitive, affective and behavioral outcomes span **d=0.27–0.66** across outcome categories; results were stronger for contingent if-then format, high motivation and rehearsal. These are category/moderator results, **not** one universal effect for all tasks or all persons.

**Transfer candidate:** vary trigger form (explicit conditional vs narrative instruction), commitment strength, and rehearsal/compiled reuse independently. **Falsifier:** any claimed benefit disappearing once both arms use an identical cached predicate and equal opportunity exposure. Machine trigger compilation is a *testable analogy* to rehearsal, not an established identity.

## S8 — Gollwitzer & Sheeran (2025): planning beyond initiation

**Review:** *Psychology of Planning*, Annual Review of Psychology 76:303–328, https://doi.org/10.1146/annurev-psych-021524-110536 (open-access CC BY 4.0).

**Exzerpt:** situational if-then plans can affect attention, prospective memory, executive functions, decision-making, emotions, behavioral change and social interactions; the authors discuss moderators and mechanisms supporting flexible yet persistent goal pursuit. Crucially, "automatic" does not mean unconditioned or without context.

**Transfer candidate:** distinguish cue detection, action authorization, monitoring, inhibition and updating. A policy can be compiled for cheap matching without erasing the interpretive monitor. **Falsifier:** if a trigger/action matcher loses contextual exceptions, a cheaper match count does not prove better goal pursuit.

## S9 — Breitwieser & Reinelt (2026): quality of mental plan representation

**Registered report/meta-analysis:** *The effectiveness of implementation intentions in children: A systematic review and meta-analysis*, British Journal of Psychology, first online **5 March 2026**, https://doi.org/10.1111/bjop.70065 ; https://pubmed.ncbi.nlm.nih.gov/41784001/.

**Exzerpt:** 42 studies, 52 effect sizes, N=12,957, mean age 10.67; Hedges g=0.31 (95% CI 0.21–0.41). Heterogeneity **I²=65.2%**. Effect moderators may involve age and self-regulation capacity; authors highlight high-quality self-generated plans and strong mental representations. This is **not** evidence that the more hardcoded predicates an artificial agent has, the better it becomes.

**Transfer candidate:** represent plan identity, confidence in cue recognition, specificity and permitted action explicitly; compare plan quality under adversarial cues. **Falsifier:** vague predicates fire incorrectly more often than precise ones even when both produce equal action volume.

## Synthesis: what is evidenced versus hypothesized

| Claim | Evidentiary status |
|---|---|
| Conditional if-then planning can improve human goal enactment, with moderators and heterogeneity | **Supported** by 1999, 2006, 2024, 2025, 2026 sources |
| Goal persistence is not always adaptive; disengagement sometimes matters | **Supported** by 2010, 2022 sources |
| MCII improves goal attainment in studied interventions but may show publication bias | **Supported, qualified** by 2021 meta-analysis |
| A NARS/CE64 action gate reproduces these human psychological effects | **NOT ESTABLISHED**; proposed computational analogy |
| Compiling implementation intentions into R2IL/mask-risc or an OGAR ActionDef has parity with human strategic automaticity | **NOT ESTABLISHED**; testable only with specified oracle/tasks |
| Direct CE64 f/c or bits 59..63 encode motivation, commitment or an executable actor | **REFUSED**: evidence, epistemic relation and intention are independent contracts |

## Source-to-experiment crosswalk

| Finding | Experiment ID | Required challenge |
|---|---|---|
| action phases and goal commitment (S1) | RV-E1 | goal switch after costly deliberation |
| cue-response plans (S2, S3) | RV-E2 | exact cue, near miss, revoke, late cue |
| MCII obstacles (S5) | RV-E3 | false obstacles, genuine blockers, delayed alternative |
| persistence vs disengagement (S4, S6) | RV-E4 | irrecoverable prerequisite vs merely delayed reward |
| format and rehearsal moderators (S7) | RV-E5 | narrative vs contingent forms, same compiled runtime budget |
| attention and inhibition (S8) | RV-E6 | irrelevant high-salience cues, contextual exceptions |
| plan quality (S9) | RV-E7 | vague vs precise if-then policy, matched opportunities |

No source paper measured AdaWorldAPI performance. There are **zero executed RV experiments** at time of writing. Reproduce behavior with an independent task oracle; report machine outcomes in machine units, never transplant psychological effect sizes as performance targets.
