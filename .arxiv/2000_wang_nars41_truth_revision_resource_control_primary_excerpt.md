# Wang (2000) — NARS 4.1: relational evidence, revision, and bounded control

**Primary source:** Pei Wang, *Non-Axiomatic Reasoning System (Version 4.1)*, two-page demonstration note (AAAI copyright 2000).
**PDF:** https://cis.temple.edu/~pwang/Publication/NARS-41.pdf
**Read:** 2026-10-22. Research excerpt, not an arXiv-hosted paper; preserved under the repository's .arxiv literature convention.

## Claim actually supported by the primary source

NARS adapts to experience while processing incoming questions and knowledge under insufficient resources. Its basic judgement contains a subject, a copula/relation, a predicate and the two-valued evidential annotation <f,c>. This is **not** a first-order predicate-logic interpretation of an arbitrary SPO tuple, nor a statement that f is the share of triples in a graph.

### Evidence semantics (p. 1, Knowledge Representation)

- f in [0,1] is the positive share of the available evidence *for the relation*.
- c in (0,1) expresses how much evidence supports that assessment, hence how revisable it is.
- A query can seek a truth estimate or a missing term. The best answer is defeasible; new evidence can change the ranking.
- The demonstration distinguishes a specialization/inheritance relation, an instance relation and similarity. The subject-predicate terms plus *meaning of the copula* are not interchangeable with raw foreign-key joins.

### Inference taxonomy (pp. 1–2)

Wang lists revision, deduction, induction, abduction, analogy and comparison. Premises for these examples share a term; each rule has its own truth-value function. The note distinguishes:
1. Deduction and analogy, which may retain relatively substantial confidence.
2. Induction, abduction and comparison, which yield more tentative confidence.
3. Revision, which combines evidence and can increase confidence beyond both premises.

**Crucial:** Revision combines supporting/opposing evidence about *the same claim*. It is not a numerical addition of atanh(cosine) values and does not itself establish independence of repeated derivation paths. The two-page note illustrates rule structure but does not provide a complete probability/sample-size theory.

### Resource-bounded control (p. 2, Control Mechanism)

Tasks are time-shared. Tasks and knowledge compete through adjustable priorities, and reasoning can continue after an initial answer if enough priority remains. This is evidence for attention/scheduling as part of the reasoning model, not a proof of any current Staunen scheduler implementation.

## Repository transfer: hypotheses, not claims from Wang

- A CE64/DR64/DR256 relation may carry an evidential reading only if its copula, observation basis, polarity and inference rule are declared.
- Neither a graph edge nor a shared intermediate term certifies fresh evidence. Count root observations separately from derivation paths to prevent path fan-out from inflating c.
- Selection of task priority is distinct from truth-value computation; keep control policy separate from numerical evidence registers.
- A fast SIMD/R2IL fold can implement a declared law, but speed/zero-copy does not validate its epistemic interpretation.

## Suggested falsifiers

1. Two independently observed positive/negative supports can be merged; the same root copied through two paths must not be counted twice.
2. One negative observation must differ from absence of an observation.
3. Compare revision, deduction and induction at identical input f,c; reject an engine that assigns the same confidence transform to all rules.
4. Reorder tasks by priority without modifying the semantic truth value of stored evidence.
5. Demonstrate an explicit inheritance copula test versus an arbitrary relational SPO edge; results must not be silently equated.

## Boundary and crosslinks

**Established:** classical NARS 4.1 relation/evidence description, inference classes, task-priority mechanism.
**Not established:** independent root identity, sample standard deviation, Spearman/ICC/ANOVA, Fisher-z inference, CE64 ABI, or lance-graph benchmark performance.

See [Wang 2001 on confidence](2001_wang_confidence_is_evidence_stability_not_a_95pct_interval.md), [Wang 2009 on evidence](2009_wang_two_numbers_preserve_direction_and_stability_of_evidence.md), and [the 2020 tutorial](2020_isaev_hammer_nars_agi20_weighted_revision_primary_excerpt.md).
