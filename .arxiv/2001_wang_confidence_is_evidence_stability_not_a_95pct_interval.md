# Wang 2001 — Confidence measures evidential stability, not a 95% interval

**Paper:** Pei Wang, *Confidence as Higher-Order Uncertainty*, ISIPTA 2001.  
**Primary PDF:** https://cis.temple.edu/~pwang/Publication/confidence.pdf  
**Proceedings:** https://isipta01.sipta.org/proceedings/005.html  
**Read:** 2026-10-11. Not an arXiv-hosted work; retained here under the repository's literature-distillation convention.

## Punchline

> **Frequency says which way evidence points; Confidence says how resistant that frequency is to one fixed additional quantity of evidence. Only independent evidential roots can legitimately increase the weight.**

## What Wang defines

For supporting evidence weight w+ and opposing weight w-, with w = w+ + w-:

- `f = w+ / w` (for w > 0).
- `c = w / (w + k)` for an agreed `k > 0`, often `k = 1` in examples.
- Inverting: `w = k*c/(1-c)` for c < 1.
- Frequency is a direction/ratio; confidence represents higher-order uncertainty or *relative stability* of that ratio as new evidence arrives.
- Wang explicitly cautions that effective w does **not** have to be a literal count of equally weighted observations. c=0.9 with k=1 means strength *equivalent to* nine ideal units, not necessarily nine collected samples.
- These are *NARS truth semantics*, **not a hardware definition of a CE64 byte**.

The "near future" is operationally defined as the arrival of a constant additional evidence weight k; this motivates c. Unlike a fixed 95% frequentist confidence interval, it has no coverage guarantee.

### Interval interpretation

For this specific finite-horizon construction, a frequency interval has lower bound `l = f*c` and upper bound `u = f*c + (1-c)`. Width `u-l = 1-c`. This represents the range after the additional standardized weight k under extreme positive/negative new evidence, **not** a Bayesian credible interval or a Neyman confidence interval.

Beware of equating this interval with *all* possible future evidence. After arbitrarily large new contrary evidence, even a high-confidence current estimate can change sharply.

### Revision example (paper §3.1)

Ten crows, nine black: `f1=9/10, c1=10/11` (k=1). Four further crows, three black: `f2=3/4, c2=4/5`. If and only if these data are not counted twice:
`
w+ = 9+3 = 12
w- = 1+1 = 2
f = 12/14 = 6/7
c = 14/15
`
The two truth-value pairs can be combined with Wang's revision formula, but only when sources provide genuinely distinct evidence. Replaying inference along a new route does not automatically meet that condition.

## Revision is not choice

- **Revision:** merge evidence from independent bases; total weight grows, confidence non-decreases, frequency becomes an evidence-weighted compromise.
- **Choice:** select among alternatives if revising would inappropriately double-count or if only one belief is needed for current processing.
- **Deduction/abduction:** derived conclusions are not fresh sensor observations; treat independence/overlap carefully.
- **Forgetting and explaining-away:** Wang separately acknowledges that confidence can decrease through other mechanisms; "revision increases confidence" is **not** a theorem that all future changes to a belief must be monotonic.

## Numeric failure mode in 8-bit registers

When confidence is stored as an unsigned byte, saturation and quantization become severe near c=1.
For k=1:
- w=1 => c=0.5
- w=9 => c=0.9
- w=99 => c=0.99
- w=999 => c≈0.999
After 8-bit rounding, many distinct high evidence weights become indistinguishable. If a full-scale byte represents exactly c=1, naively inverting it implies infinite evidence, a potentially false certification.

**Do not change the ALU/ISA ABI to patch this.** Instead investigate:
1. exact quantization rule and whether 255 is reserved/sentinel;
2. finite accumulator or root-backed evidence weight external to display byte;
3. inversion error on representable bytes;
4. which NARS rules require original weights vs quantized F/C;
5. byte-exact replay under fixed rounding mode.

## Falsifiers for lance-graph

T1. Match Wang's independent revision example exactly before quantization; compare quantized tolerance separately.

T2. Same root O appears in A and B through different deduction paths, then merges. Refuse to mint a second independent evidential weight.

T3. Distinct roots O1/O2 with the same observed content legitimately increase weight.

T4. Invert byte c near saturation; flag underidentification instead of asserting infinite n.

T5. Keep F/C unchanged but mutate CE64 epistemic status or a recipe opcode: confidence must not be abused as a substitute for epistemic/causal certification.

T6. Explicitly compare a 95% binomial confidence interval with Wang's frequency interval on same observations and assert that they are **not equivalent**.

## Separation of contracts

`
register ABI:  bit positions, byte ordering, width and saturation
ISA:           deterministic operations with declared reading
NARS semantics: f,c functions of evidence weights under k
provenance:    independently identifiable evidential roots
CE64 status:   epistemic reading separate from F/C
`

**Source findings:** formulae and qualitative revision/choice distinction (§§2–4).  
**Project hypotheses:** accumulator strategies, byte saturation tests and CE64 separation; verify against current main before treating as implemented.

**Crosslinks:** 2009 evidence comparison; 1303.1517 belief revision; D2 provenance ledger.
