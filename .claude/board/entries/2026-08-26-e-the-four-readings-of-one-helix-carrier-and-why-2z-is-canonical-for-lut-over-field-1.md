## 2026-08-26 — E-THE-FOUR-READINGS-OF-ONE-HELIX-CARRIER-AND-WHY-2Z-IS-CANONICAL-FOR-LUT-OVER-FIELD-1 — measured: for a LUT over a field, `r` distorts a splat kernel 204×, `y` 11.6×, and 1Z/2Z are both EXACTLY uniform — so 2Z is canonical on UNITS, not resolution; and the tilt-read crossover is the saturation tail, not geometry

**Status:** FINDING [MEASURED] — one probe, shipped `helix` public API,
copy outside the repo (no crate mutated), N=4096, 256-bin field sweep.
Operator ruling recorded: *"for LUT over field (which we mostly do) for
gaussian splat the fisher 2Z is better/canonical."* The measurement
**confirms the ruling and corrects its usual justification**, and
**⊘ falsifies a claim I derived last turn** (the tilt crossover).
**Confidence:** High — both tables are exact-arithmetic consequences
that the measurement reproduces to the digit.

### The four readings of ONE carrier

`Signed360` carries `rim` (→ r, via `arctanh`) and `polar` (→ y), bound
by `r² + y² = 1`. Four parameterisations of the same point, and they
are **not interchangeable** — each is right for a different question:

| reading | what it is | `dρ/d·` (ρ = geodesic arc length) |
|---|---|---|
| `r` | raw disk radius = sin θ | `2/(1−r²)` — diverges at the rim |
| `y` | pole distance = cos θ | `−2/(y·r)` — diverges at **both** ends |
| **1Z** | `atanh(r)` — Fisher z, *statistical* | **2** — constant |
| **2Z** | `2·atanh(r)` — Poincaré depth, *spatial* | **1** — constant |

### Table A — LUT over a field (the Gaussian-splat question)

256 uniform bins per parameterisation; measured **geodesic arc length
spanned per bin**. A fixed-bin-width kernel's footprint is proportional
to that span, so max/min **is** the kernel-width distortion across the
field:

| LUT axis | min span | max span | **distortion** |
|---|---|---|---|
| `r` (raw radius) | 0.007805 | 1.591667 | **203.9×** |
| `y` (polar byte) | 0.014926 | 0.173047 | **11.6×** |
| **1Z** | 0.029689 | 0.029689 | **1.0×** |
| **2Z** | 0.029689 | 0.029689 | **1.0×** |

**`r` and `y` are disqualified outright as LUT axes for a field kernel**
— a splat binned on `r` is 204× wider at the rim than at the centre, on
`y` 11.6× and singular at *both* ends. That is not a tuning problem; a
"fixed-σ" splat simply is not fixed-σ in those coordinates.

**And the honest refinement of the ruling: 1Z and 2Z tie EXACTLY
(0.029689 both, 1.0×).** 2Z is not canonical because it resolves
better — it cannot, the two are affinely related. **2Z is canonical
because it carries the correct UNIT: 2Z *is* ρ, the geodesic arc
length, which is the unit a splat's σ is quoted in.** 1Z is ρ/2, so a
σ meant for ρ applied in 1Z yields a kernel **2× too wide** — and that
error is invisible inside one fitted table (σ absorbs it) and only
surfaces when a table is **shared, composed, or quoted across
families**, which is precisely the universality the Fisher-z LUT is
used for (`E-PALETTE256-IS-A-NEEDLE-…-1`). Units are a composability
property, not a cosmetic one.

> **Use 2Z whenever the LUT axis is a distance the consumer reasons
> about; 1Z only where the table is self-contained and its σ is fitted
> in-place. Never `r` or `y`.**

### Table B — recovering TILT (a different question, a different answer)

Same carrier, but now "what is the polar angle?" — measured Δtilt:

| band | via `rim` (8-bit z) | via `polar` (7-bit y) | `rim` w/ full-range floor |
|---|---|---|---|
| POLAR 90–60° | **0.167°** | 0.476° | 0.259° |
| MID 60–30° | **0.122°** | 0.165° | 0.191° |
| LOW 30–10° | **0.067°** | 0.122° | 0.104° |
| EQUATOR 10–0° | 0.422° (max 4.317°) | **0.114°** | **0.029°** |

### ⊘ CORRECTION — the crossover is the SATURATION TAIL, not geometry

Last turn I *derived* that `rim` and `polar` have complementary
conditioning (`dθ/dr` well-behaved at the pole, `dθ/dy` at the equator)
and predicted a **geometric** crossover. **Measured, that is wrong.**
The control column settles it: with a full-range floor the equatorial
`rim` error collapses **0.422° → 0.029°**, a **14.5×** improvement, and
becomes the best cell in the table. So `polar` wins at the equator for
exactly one reason — **`rim` saturates there by design**
(`ResidueEncoder::new` seeds `hi = aligned(0.99·N)`) — not because of
conditioning. Derivation proposed a mechanism; measurement named a
different one.

**And the saturation tail is a measured-GOOD trade, not a wart.**
Spending the floor on the full range costs the bulk ~1.5× (0.167→0.259,
0.122→0.191, 0.067→0.104) to rescue the top 1 %. The shipped default
buys 99 % of the sphere ~1.5× better tilt for one degraded percent.
State it as a deliberate trade with those numbers; do not "fix" it
without re-measuring the bulk.

### When / where

| you are doing… | use | why |
|---|---|---|
| LUT over a field, splat/kernel, shared or composed table | **2Z** | uniform geodesic bins **and** correct unit (σ in ρ) |
| LUT over a field, self-contained table, σ fitted in place | 1Z ok | ties 2Z exactly; only the unit differs |
| any field LUT on `r` or `y` | **never** | 204× / 11.6× kernel-width distortion |
| recovering tilt, bulk of the sphere | `rim` | 1.4–2.8× better than the polar byte |
| recovering tilt, top ~1 % rim | `polar` | `rim` is saturated there **by design** |
| recovering tilt, want uniformity over accuracy | full-range floor | 0.029–0.259°, but ~1.5× worse across the bulk |
| exact self-similarity / a point question | neither | it is **1.0 by definition of the address** |

**Bit-budget caveat, stated:** `rim` is 8 bits and `polar` is 7, so
Table B is not a like-for-like comparison of the two *encodings* — it
is a comparison of the two **readings as shipped**, which is the
decision a consumer actually faces.

Cross-ref: `helix::fisher_z` (`fisher_z` / `hyperbolic_depth` — the
crate already documents the factor 2 as *"geometry keeps the 2 as arc
length ∫2/(1−t²); statistics drops it for variance stabilisation"*,
which is exactly this entry's Table A);
`E-THE-HELIX-POLE-PENALTY-IS-THE-POLAR-BYTE-NOT-THE-CODEC-…-1`
(same PR — the bounded-vs-divergent pole terms);
`E-PALETTE256-IS-A-NEEDLE-…-1` (why a shared table's unit matters).

