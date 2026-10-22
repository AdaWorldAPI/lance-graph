# Isaev & Hammer (AGI-20) — NARS evidence weights and revision are explicit arithmetic

**Source:** Peter Isaev and Patrick Hammer, *NARS: an Artificial General Intelligence (AGI) Project*, Temple University, AGI-20 tutorial, 33 slides.
**PDF:** https://www.opennars.org/workshop2020/NARS2019_files/NARS_Tutorial_AGI-20.pdf
**Read:** 2026-10-22. Conference tutorial slides, not an arXiv identifier.

## Primary mathematical anchor (slides 11–13)

The slides define nonnegative positive/negative evidence weights w+ and w-, with w = w+ + w-. For a judgement S -> P:

~~~text
f = w+ / w                         when w > 0
c = w / (w + 1)                    tutorial convention k = 1
w = c / (1 - c)                    0 <= c < 1, inverse of convention above
w+ = f * w
w- = (1 - f) * w
~~~

The unweighted empty case w=0 needs an explicit policy; f is then undefined, not an observed 0%. For a generalized confidence constant k>0, c=w/(w+k) and w=k*c/(1-c); that generalized k belongs to other NARS discussions, **not the specific k=1 formula displayed on slide 13**.

The tutorial's evidence on slide 11 is set-based: extension/intension overlap supplies positive and negative evidence for an inheritance judgement (slide 11). A count of **all graph triplets** or inbound adjacency degree is *not* the tutorial's w, unless a separate model explicitly maps eligible observations to these evidential sets.

## Revision (slide 19)

For two judgements about the **same** S -> P, the slide gives:

~~~text
A = c1*(1-c2)
B = c2*(1-c1)
f_rev = (f1*A + f2*B) / (A+B)
c_rev = (A+B) / (A+B+(1-c1)*(1-c2))
~~~

If each input is backed by distinct, independent effective evidence weights, this is equivalent to:

~~~text
w_total = w1+w2
w_plus_total = w1*f1 + w2*f2
f_rev = w_plus_total / w_total
c_rev = w_total / (w_total+1)
~~~

Proof sketch: w_i=c_i/(1-c_i), multiply numerator and denominator by (1-c1)(1-c2). **Independence/provenance is a precondition for our safe aggregation**, not a license supplied by the algebra to add a repeated source.

The illustrated bird -> swimmer inputs are positive and negative, while the output frequency and confidence are both revised. This shows why revision is an evidence-weighted compromise, not argmax or a simple majority over different predicate labels.

## Different inference operators are different contracts (slides 15–18)

- Deduction example: robin -> bird plus bird -> animal yields robin -> animal, with derived confidence lower than either input (slide 16).
- Induction and abduction get their own frequency and confidence functions, generally more tentative (slides 17–18).
- Do not substitute the revision sum for every multihop inference. It requires a matching conclusion and an appropriately distinct evidence base.
- Memory, attention and control are distinct: tasks/beliefs are clustered into concepts and prioritized; task/belief pairs are selected with priority bias (slides 21 and 25).
- The NAL hierarchy slides are a hierarchy of representational/inferential capabilities, not a specification of a fixed number of Markov hops (slide 26).

## What statistical reports can and cannot be reconstructed

If w+ and w- are authentic *independent Bernoulli observation counts* and w is an integer, f is the binomial sample proportion and w can be n. That is a **special observation model**. NARS inferred/effective weights need not be iid Bernoulli trials.

Consequently:
- SD, covariance, Pearson r, Spearman rank correlation, ICC and ANOVA are not reconstructible in general from a single (f,c) pair.
- Even if n is known, inferential variance needs a specified random process and independence assumptions.
- For relation-specific *observed* counts, we may additionally retain group labels, timestamps, ranks, paired values and dependence structure to support those statistics. Never pretend the two NARS numbers alone contain that information.
- Fisher's z = atanh(r) assumes a correlation statistic r and its sampling model; f and c are not r.

## Independent oracle and falsifiers

1. From weights (w+,w-)=(1,1) versus (500,500), assert equal f=0.5 but distinct c=2/3 versus 1000/1001.
2. Disjoint supports (9+,1-) plus (3+,1-) give (12+,2-), f=6/7, c=14/15.
3. Revisit that same root through another derivation; confidence must not increase through duplicated evidence.
4. For equal weights and conflicting f, compare exact revision to a tanh/atanh-composition alternative and reject accidental equivalence.
5. Validate undefined w=0 and c->1 quantization/saturation boundaries.
6. Test priority changes independently of stored f/c.

**Established by tutorial:** evidence-set construction, frequency/confidence formulae, rule-family examples, explicit revision formula, control/memory separation.
**Unproven here:** a current lance-graph implementation, all statistical inferences from (f,c), or a universal 3-sigma denoiser.

**Related:** [NARS 4.1](2000_wang_nars41_truth_revision_resource_control_primary_excerpt.md), [NARS vs tanh velocity factors](2008.02078_tanh_velocity_factors_are_not_nars_evidence.md), [Wang 2001](2001_wang_confidence_is_evidence_stability_not_a_95pct_interval.md).
