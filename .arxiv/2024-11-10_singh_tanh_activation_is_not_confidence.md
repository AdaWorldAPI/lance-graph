# Deepankar Singh (2024) — Tanh activation is not statistical confidence

**Article:** Deepankar Singh, *Mastering Tanh: A Deep Dive into Balanced Activation for Machine Learning*, AI-Enthusiast / Medium, 10 November 2024.
**Source:** https://medium.com/ai-enthusiast/mastering-tanh-a-deep-dive-into-balanced-activation-for-machine-learning-4734ec147dd9
**Format:** practitioner tutorial / web article, **not an arXiv paper**. Filed in the repository's research-harvest directory for consistency.
**Companion paper:** [Chandra, arXiv:2008.02078, hardware tanh and NARS distinction](2008.02078_tanh_velocity_factors_are_not_nars_evidence.md).
**Harvest status:** source excerpt and test ideas; no implementation authorization.

## Punchline

> Symmetric bounded activation is a useful *numeric transfer function*. It cannot turn cosine similarity into observed frequency, turn atanh into sample count, or supply NARS confidence without an evidence model.

## What Singh's article actually covers

1. **Definition and zero-centering.** Hyperbolic tangent maps real inputs into (-1,1), preserving sign and saturating smoothly.
2. **Derivative and learning dynamics.** The derivative is 1 - tanh(x)^2. It is largest at x=0 and approaches zero in saturated regions. The article discusses improved gradient behavior around zero and the risk of vanishing gradients at the tails.
3. **Applications.** The article discusses RNN/LSTM hidden-state activations, selected classification/MLP settings and examples in native Python, NumPy, PyTorch and TensorFlow.
4. **Cost tradeoff.** Direct transcendental evaluation can be more expensive than simple activations such as ReLU.

These are textbook activation-function properties, not a proof of NARS revision, a Fisher-z statistical test, or the validity of a 3-sigma gate.

### Stable formulas

~~~text
tanh(x) = (exp(x)-exp(-x)) / (exp(x)+exp(-x))
        = (exp(2*x)-1) / (exp(2*x)+1)

d/dx tanh(x) = 1 - tanh(x)^2

atanh(y) = 0.5 * ln((1+y)/(1-y)), for -1 < y < 1

tanh(atanh(y)) = y, for real |y| < 1
~~~

**Numerical note:** naive exp-based Python examples can overflow for large x; an optimized standard-library tanh or a separately certified approximation is preferable. Floating-point tanh may return exactly ±1 for finite arguments, unlike the ideal mathematical range.

## A useful exact identity (interpretation requires a model)

For *defined binary evidence counts* p positive and n negative, w=p+n>0:

~~~text
f = p / (p+n)             NARS-style frequency
b = 2*f - 1 = (p-n)/(p+n)  symmetric signed representation
f = (b+1)/2               reversible normalization
~~~

This mapping is an **affine transform** and works because p and n are actual evidence statistics.

It does **not** follow that:

~~~text
b = cosine(embedding(S), embedding(P))
~~~

unless a separate, testable calibration model explicitly connects the embedding score to the frequency of a *named, observable relation*. A cosine is not a sample count, implication probability, or automatically a Pearson/Spearman correlation estimate.

Similarly, if b is transformed to z=atanh(b), z becomes an **alternative coordinate for the ratio**, not an additional independent statistic. From z alone, the system cannot recover total w and therefore cannot calculate NARS confidence c=w/(w+k).

## Application to lance-graph and R2IL

Keep distinct layers:

~~~text
semantic input:
    bound S-P identity + observed/derived evidence + provenance

revision:
    (w_plus,w_minus), disjoint evidence only
    merge by checked integer or fixed-point addition

numeric view (optional):
    b = (w_plus-w_minus)/(w_plus+w_minus)
    -> palette256 / tanh / atanh / hardware codec

statistical view (optional):
    named sample statistic + actual n + null/variance model
    -> confidence interval / Fisher transform / calibrated gate
~~~

Do not use a value within [-1,1] as proof of correctness. **Boundedness is a data-type invariant, not an epistemic certificate.**

## Testables

**T1 — Symmetry and endpoints.** For a dense signed input sweep and every quantization bucket, measure odd symmetry, monotonicity, errors at zero and near saturation, and f32 endpoint rounding against a reference implementation.

**T2 — Derivative contract.** Compare exact 1 - tanh(x)^2 with numerical derivatives in the unsaturated interior. Document gradient collapse for large |x| rather than claiming guaranteed stability.

**T3 — Meaning separation.** Construct two evidence histories with the same b and different w. Frequency must match while confidence differs. A tanh/atanh-only design must fail this test if it claims to calculate NARS confidence.

**T4 — Calibration.** If embedding cosine is proposed as a frequency estimator, assess out-of-sample calibration, reliability curves, Brier score, false-positive correlations and shuffled S-P bindings. A numerical range agreement alone is insufficient.

**T5 — Execution cost.** Compare libm tanh, table lookup, polynomial/PWL and Chandra-style grouped velocity-factor approximation under one bounded error target. Benchmark complete SIMD lane paths including quantization and cache behavior, not solely the function call.

## Adoption guardrails

- Treat Singh as accessible pedagogical background, not experimental validation of an optimized accelerator.
- Cite [Chandra 2020](https://arxiv.org/abs/2008.02078) for hardware design hypotheses, with its specific error/area precision envelope, not imaginary general CPU speedups.
- Cite [Wang's NARS 4.1 overview](https://cis.temple.edu/~pwang/Publication/NARS-41.pdf) and the [NARS tutorial](https://www.opennars.org/workshop2020/NARS2019_files/NARS_Tutorial_AGI-20.pdf) for frequency, confidence and revision, not the activation tutorial.
- Require evidence source identity and deduplication before revision. Repeated paths must not multiply the count of observations.
- A 3-sigma threshold needs an independently justified null distribution, effective sample size and uncertainty estimate; it neither prevents hallucinations nor proves a claim true.

## One-line harvest

**Tanh is a cheap bounded *view* worth exploring, but truth must be computed from auditable evidence, not inferred from an activation curve.**
