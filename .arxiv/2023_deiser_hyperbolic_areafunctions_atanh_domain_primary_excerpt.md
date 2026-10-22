# Deiser (2023) — Hyperbolic functions, inverse area functions, and the domain of artanh

**Author/teaching source:** Oliver Deiser, *Skript zur Einführung in die Mathematik 1*, lecture 6, “Arkusfunktionen und hyperbolische Funktionen”.
**Lecture page:** https://www.aleph1.info/?call=Puc&permalink=em1s_6.%20Vorlesung%20Arkusfunktionen%20und%20hyperbolische%20Funktionen
**Requested six-page PDF extract:** https://www.aleph1.info/Resource?method=get&obj=Pdf&name=em1s.pdf&pagestart=43&pageend=48
**Course table of contents:** https://www.aleph1.info/?call=Puc&permalink=em1s_Inhalt
**PDF version on linked pages:** 2023-06-04. Read 2026-10-22. Lecture note, not an arXiv paper.

## Primary source excerpts (PDF printed pp. 43–45)

- Sec. 5, parity decomposition: every function on a symmetric domain splits into even and odd parts; the decomposition is algebraic, not a statistical noise model.
- Sec. 6, hyperbolic functions: cosh(t)=(e^t+e^(-t))/2, sinh(t)=(e^t-e^(-t))/2 and tanh(t)=sinh(t)/cosh(t). The identity cosh²(t)-sinh²(t)=1 follows.
- Sec. 7, inverse area functions: the lecture explicitly names artanh as the inverse of tanh, alongside arcosh/arsinh etc. The displayed logarithm theorem in the supplied PDF proves formulas for **arcosh and arsinh**, not an explicit displayed artanh logarithm formula.
- The PDF extract links to the same material as the lecture web page; the contents URL places the lecture in its larger mathematics course. These three requested URLs are different representations/navigation of *one work*, not three independent studies.

## Mathematical consequence, derived here from the cited definition

Starting with x=tanh(t)=(e^(2t)-1)/(e^(2t)+1), solve for t:

~~~text
e^(2t) = (1+x)/(1-x)
artanh(x) = (1/2) * ln((1+x)/(1-x))       -1 < x < 1
tanh(artanh(x)) = x                      -1 < x < 1
tanh(a+b) = (tanh(a)+tanh(b)) /
            (1+tanh(a)*tanh(b))
~~~

The log formula and addition identity are **elementary derived consequences**, not quotations from the supplied PDF excerpt. Domain restrictions are mandatory: at x=+-1 the real inverse diverges; outside (-1,1) no real artanh exists. In finite precision, approximation, clamping and overflow/saturation require explicit handling.

The exact real-valued operation x ⊕ y = (x+y)/(1+xy) corresponds to adding atanh coordinates. It is not ordinary arithmetic addition of bounded scores, and it is not a NARS independent-evidence revision rule.

## Why this matters to the proposed numerical substrate

The lecture supplies rigorous elementary mathematical definitions for a **numeric representation**. It does not define:
- NARS f/c, positive/negative evidence counts or sample size;
- Fisher z-transform of a sample correlation, normality, Gaussian 3σ thresholds or effective n;
- a guaranteed exact palette256 LUT, SIMD speed, a statistical prior, or a graph-inference semantics.

Possible engineering direction: use a saturating, monotone, reversible-with-tolerance LUT mapping for bounded numeric scores, but **separately store real evidence weights and provenance** when a claim has NARS semantics. The signed atanh coordinate can be a compact arithmetic transform; it cannot recover missing supporting sample identity.

## Falsifiers for a palette256 / fixed-point hypothesis

1. Test monotonicity and odd symmetry of table encoding on all representable input codes.
2. Test tanh(atanh(x)) round trip away from endpoints; quantify error versus f32/f64 reference, including sign and boundary cases.
3. Compare scalar and vectorized hyperbolic composition over signed pairs near -1, 0 and +1.
4. Test exact identity under ideal real arithmetic versus bounded quantized arithmetic; claim neither statistical independence nor exact associativity after rounding.
5. Compare two evidence sets with equal f and different counts: any scheme that maps both to identical NARS c solely from atanh(f) fails.
6. If 3σ is proposed, require a **defined random variable and independently justified variance estimate**, not the parity decomposition or boundedness of tanh as a substitute.

**Related:** [NARS versus tanh implementation](2008.02078_tanh_velocity_factors_are_not_nars_evidence.md), [AGI-20 evidence arithmetic](2020_isaev_hammer_nars_agi20_weighted_revision_primary_excerpt.md).
