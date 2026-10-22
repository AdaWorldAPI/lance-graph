# Johansson & Hammer (2026) — Machine Psychology with NARS/ONA: from reinforcement to acquired relations

**Publication:** Robert Johansson and Patrick Hammer, *From operant learning to arbitrarily applicable relational responding: a review of Machine Psychology with the non-axiomatic reasoning system*, Frontiers in Artificial Intelligence 9:1775040, published 2026-03-25.
**DOI:** https://doi.org/10.3389/frai.2026.1775040
**Requested PMC record:** https://pmc.ncbi.nlm.nih.gov/articles/PMC13056877/
**Publisher full text:** https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1775040/full
**Publisher PDF:** https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1775040/pdf
**Read:** 2026-10-22. Peer-reviewed open-access REVIEW (CC BY); it summarizes prior experiments rather than introducing all four from scratch.

## Core progression (secs. 6–9)

The review reports four increasingly demanding controlled tasks:
1. Operant reinforcement, contingency reversal and conditional discrimination (sec. 6).
2. Generalized identity matching with transfer to new exemplars (sec. 7).
3. Functional equivalence and transfer of response function after changed contingencies (sec. 8).
4. Arbitrarily applicable relational responding (AARR), including context-sensitive SAME/OPPOSITE frames and untrained relational tests (sec. 9).

Reported 100% performance applies to **specified controlled experimental test phases**, not arbitrary downstream language understanding or a universal AGI performance guarantee.

## Key distinction: built-in NARS versus added mechanisms (secs. 8–10)

The authors explicitly distinguish native NARS/ONA machinery (temporal and procedural inference, priority-bounded attention, variable introduction and relational inference) from **research-program extensions**, notably functional-equivalence detection and *acquired relations*.

A trained relation may be represented by a named relational product such as:

~~~text
<(A1 * B1) --> SAME>
~~~

Under the learned relational frame, derivation may proceed by mutual entailment and combinatorial entailment: SAME(A,B) and SAME(A,C) can support SAME(B,C); OPPOSITE(A,B) plus OPPOSITE(B,C) can support SAME(A,C). These are **context-dependent frame laws with trained applicability**, not universally valid equivalence of arbitrary two graph nodes sharing a neighbor.

Study IV (sec. 9) used pretraining on relational operators, trained AB/AC links, then held-out BC relation queries with no test-phase reinforcement. Performance and hypothesis confidence were reported separately. The paper does not turn confidence into a frequentist n or validate unknown-length Markov pointer chains.

## Hard external-validity limit (sec. 12)

Authors state experiments used *small stimulus sets, 2–6 items per task*, discrete pre-encoded representations, and explicit reinforcement. Unproven areas: continuous sensorimotor scaling, larger relational spaces, autonomy, linguistic interfaces, deictic and temporal relational frames, and theory-of-mind applications. The conclusion is a developmental *research roadmap*, not a deployment guarantee.

## Transfer to lance-graph / Thought-Git (hypotheses, not source findings)

- Model relational frame identity as an explicit, typed contract; SAME, OPPOSITE, inheritance, analogy and temporal implication require different combinators.
- For each derived relation, retain the **training provenance**, relevant context/cue, source evidential roots, proof path and whether the relation was direct, derived or only hypothetical.
- Run DR/R2IL masked symbolic relational folds against independent, tiny behavioral oracles before larger graph claims.
- Keep feedback-based update/revision separate from derivation and action-selection. A derived edge is not a fresh trial.
- A learned frame must have holdout evidence; shared consequences alone must not automatically certify equivalence outside the task's assumptions.

## Reproducible falsifiers

1. SAME(A,B), SAME(B,C) -> test held-out SAME(A,C) under the same learned frame.
2. OPPOSITE(A,B), OPPOSITE(B,C) -> test SAME(A,C) **only** in a frame validated to obey that composition.
3. Reverse reinforcement after initial learning; score adaptation separately from confidence and from raw graph-edge count.
4. Inject ambiguous context cues; verify that inference is gated by context rather than blindly applying global transitivity.
5. Remove the acquired-relation extension; test that an unmodified baseline is **not** credited with extension-only behavior.
6. Scale held-out stimuli beyond 6 and vary continuous/noisy inputs, logging loss and latency rather than extrapolating 100%.

## Evidence boundary

**Established in the review:** staged experimental literature, ONA/NARS mechanisms and stated added mechanisms, controlled-test findings, explicit limitations.
**Not established:** arbitrary-domain ontology folding, CE64 epistemic state, million-history throughput, clinical applicability, generic Markov-hop depth, causal intervention truth or self-awareness.

**Crosslinks:** [NARS 4.1](2000_wang_nars41_truth_revision_resource_control_primary_excerpt.md), [NARS AGI-20 tutorial](2020_isaev_hammer_nars_agi20_weighted_revision_primary_excerpt.md), [NARS evidence research index](2026-10_nars_evidence_nal_metacognition_distillation_index.md).
