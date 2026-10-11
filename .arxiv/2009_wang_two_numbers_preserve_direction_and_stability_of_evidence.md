# Wang 2009 — One number loses the ability to revise evidence responsibly

**Paper:** Pei Wang, *Formalization of Evidence: A Comparative Study*. *Journal of Artificial General Intelligence* 1(1), 2009; official publisher DOI https://doi.org/10.2478/v10229-011-0003-7 .  
**Author-hosted PDF:** https://cis.temple.edu/~pwang/Publication/FormalEvidence.pdf  
**Read:** 2026-10-11. Journal article, not an arXiv ID.

## Punchline

> **A belief needs at least two degrees of freedom if it must represent both evidential direction and evidential stability under future revision. Two numbers do not, by themselves, prove that evidence sources are independent.**

## Scope and comparison

The paper compares:
- binary/axiomatic truth (good at conclusive cases; incapable of grading inconclusive support without extensions);
- single scalar degree of belief, including probabilistic approaches (describes an estimate without separately encoding the size of its evidential base);
- two-number formalisms, including belief/plausibility, Bayesian parameter-based accounts, evidence-weight pairs, and NARS frequency/confidence.

The claim is about **general-purpose reasoning under open-ended evidence arrival**, not that all narrowly scoped probability models are invalid. In fully specified stable probabilistic models, posterior distributions and sufficient statistics may encode more than one scalar even when a reported prediction is one number.

### Three equivalent parameterizations under NARS assumptions

Given positive and negative effective evidence weights:
`
(w+, w-)      evidence counts/weights
(f, c)        f=w+/(w++w-); c=(w++w-)/(w++w-+k)
[l, u]        derived frequency interval [f*c, f*c+(1-c)]
`
These transformations require consistent k and valid nonnegative weights, and boundary cases must be handled explicitly.

They are **mathematical readings**. On the substrate, the ALU/ISA register contract remains primary and can support other interpretations only when their own reading is declared.

## Why the term logic matters

Wang's evidential inheritance `S --> P` is not equivalent to arbitrary subject-predicate-object graph storage. The formalism reasons over *extent* and *intension*, the support for substitutability, and appropriate positive/negative evidence. His discussion of Hempel's ravens illustrates why differently worded classical equivalents need not be evidentially identical.

The study distinguishes:
- *negative evidence* for a relation;
- *positive evidence* supporting it;
- *absence of evidence* (ignorance), which is neither negative evidence nor certainty of falsity.

A graph with a missing edge must not automatically mint evidence against that edge. An ontology parent relationship alone is not a measured support count.

## Warning about evidence fusion

If a single observation O yields two deductions D1 and D2, then fusing D1 and D2 as though independently observed is evidence laundering. Effective provenance therefore needs both:
- event identity: exactly which derivation occurred;
- root-support identity: whether the derivation ultimately depends on the same observation.

This distinction is an architectural extrapolation consistent with Wang's additive evidence assumptions, **not** a claim that his 2009 paper specifies our append-log format.

## Testable harvest

1. Two histories with same frequency but weights 2 and 2,000 must have different stability; one scalar cannot preserve both.
2. A false observation must contribute negative evidence, not be encoded as "not seen."
3. Derived hypothesis with no observation must not earn an independent evidential root.
4. Test interval conversion across all u8 codes and document boundary/sentinel behavior.
5. Compare associativity/commutativity of *independent evidence aggregation* with dependency-safe dedup in full provenance.
6. Maintain separate certificates for inferential support and Pearl intervention/purpose; none follow from large c alone.

## Evidence boundary

**Established by source:** argument for two independent numbers, comparison of major uncertainty formalisms, NARS parameterizations, inheritance evidence semantics.  
**Not established by source:** specific CE64 bit positions, ALU opcode implementations, any measured lance-graph speed, successful reasoning over arbitrary OWL ontologies.

**Further reading:** Wang 2001 *Confidence as Higher-Order Uncertainty*; Wang 1993/2013 *Belief Revision in Probability Theory*.
