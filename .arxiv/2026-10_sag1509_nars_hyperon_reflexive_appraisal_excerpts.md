# Source excerpts: NARS/ONA and Hyperon as a grounded self-model, not a homunculus

**Date:** 2026-10-12. **Status:** primary paper extracts / source-grounded questions, not LG runtime parity. **Parent:** #1509. **Companion:** [D-SAG-2](../.claude/plans/sag-reflexive-cognition-from-1509-v1.md). Paraphrases, not large verbatim quotations.

## Wang, Li, Hammer, 2018: Self in NARS

**Primary:** https://doi.org/10.3389/frobt.2018.00020 , §§3.3–3.5 and §4.3.
**Extract:** self-related knowledge can arise from processing *internal* experiences using the same general reasoning machinery as external experiences. Mental operations include considering and doubting. System- and event-level satisfaction affect finite resource allocation and priority. This is a proposed functional emotion/self-control account with incomplete self-knowledge, **not** empirical proof of a experiencing homunculus or consciousness.

**SAG transfer:** separate the observed compiled Program, independent evidence, derivation receipt and *the act of attending to that receipt*. A fact-only graph and a graph with a new episode-significance edge should retain identical historical observation bytes. A reflective trace-only arm that cannot answer "why?" is a useful negative control. A new priority policy must not change the recorded compiler verdict.

**Falsifier:** two derivations yield same final truth but different premises; the explanation should distinguish them. Copy one observation to two hypothesis layers: confidence must not increase. Revisit the event: no additional source ID. Suppress mental/appraisal policy and compare usefulness with identical budgets.

## Hammer and Lofthouse, 2020: OpenNARS for Applications

**Primary source repo:** https://github.com/opennars/OpenNARS-for-Applications at snapshot dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a. README links to their AGI 2020 paper *OpenNARS for Applications: Architecture and Control* and Lofthouse's 2019 ALANN description. First-hand C entry points: src/Cycle.c, PriorityQueue.c, Memory.c, Stamp.c, Truth.c, Decision.c, Inference.c, NAL.c.

**Extract from the implementation description:** ONA was designed as a new C-based platform, not a fork of the research OpenNARS implementation. It applies event-driven attention, finite queues and concept usefulness to select work; procedural reasoning uses anticipations about outcomes. This makes its event queue a promising *control* reference, not an oracle equivalent to PLN or Shannon.

**SAG transfer:** compiler refusal, changed seal or stale test can be a bounded attention event; the proposal chooses what to investigate next, not what is true. Compare urgent rare change vs popular repeated concept. Do not mix queue priority and evidence frequency.

**Falsifier:** with identical candidate budget and evidence, a priority policy that starves high-impact rare compiler regressions loses to FIFO; copying the same event must not become multiple independent stamps.

## Goertzel et al., 2023: OpenCog Hyperon

**Primary:** https://arxiv.org/pdf/2310.18318 , printed §3.1.3 p22, §4.3.4 pp61–63, §4.3.10 pp72–73, §4.4.2 pp74–77, §4.6 pp80–82.

**Extract A, reflective self-understanding (§4.3.10):** AtomSpace can serve as subject and substrate for its own reasoning and mining operations, but the paper explicitly warns this representational possibility **does not by itself confer deep self-understanding**. Feedback between modest self-knowledge and smarter procedure learning is a research aspiration.

**SAG falsifier:** ask method -> program -> primitive -> certificate; an unsupported two-hop transition must be visible as an insufficiency, not implemented via an invisible second engine. Only the actual compiled program may certify a compiled-tier edge.

**Extract B, inference control (§4.3.4, pp61–63):** authors propose learning what reasoning-step sequences work in what contexts and synthesizing new ones. Programmable interception is discussed as research direction. For LG, store a witnessed rule/step and test learned selection vs a frozen control; do not claim deployed synthesis on the basis of prose.

**Extract C, MicroPsi/Psi emotion (§4.4.2):** subsystems' urgency, importance, competence, experienced uncertainty, selection threshold and arousal interact as parameters shaping control. The paper contemplates machine appraisals unlike familiar human emotional categories. Figure 18 (printed p75) illustrates a regulator graph, not an empirical demonstration of our four names.

**SAG ablation:** Emberglow, Steelwind, Woodwarm and Velvetpause can be optional *operational* policy names. Same truth/evidence budget, change only scheduler/appraisal, measure useful discoveries and failure costs. Overpersistence, familiarity bias, noise fixation and endless deferral are deliberate adversaries.

**Extract D, GOLEM self-modification (§4.6):** meta-level improvement should be tested and reversible. Goal evolution and long-term safety are unsettled. This research PR is not permission to apply any candidate.

## Epistemic needs, dignity in disagreement, and significance: provenance-grade hypotheses

The Maslow pyramid analogy is an *operator-originated framing*: an agent with traceable knowledge, a limited uncertainty budget and stable identity may have more room to explore. It is not a mathematical reduction from Maslow to CE64. "Transcendence" is a philosophical possibility, not a benchmark.

Disagreement: a proposition's evidence, source reliability and relational respect are separable. Source A and B may disagree fiercely about P without either person's other claims being discredited. This is a proposed governance invariant, not a psychological finding extracted from Wang's work.

Reappraisal: a thought brushing awareness at k+1 can cause a new significance/attention edge to an episode at k. The historical fact at k remains unchanged. This is a model of *interpretation*, not an independent new observation, and must not cause confidence inflation.

**Countertest:** A's statement P is refuted by a test, and B's opposing claim survives. Keep unrelated independently verified A claims. If a source-wide trust penalty appears automatically, reject this policy. For Velvetpause, a deadline with sufficient evidence must still permit decision; permanent indecision is a failed controller.
