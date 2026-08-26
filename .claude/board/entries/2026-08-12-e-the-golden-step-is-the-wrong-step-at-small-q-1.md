## 2026-08-12 — E-THE-GOLDEN-STEP-IS-THE-WRONG-STEP-AT-SMALL-Q-1

**Status:** FINDING `[G]` — arithmetic, fully reproducible, no fetch. Operator
ruling + three measurements run this session.

**The ruling (operator, three messages, paraphrased):** golden-ratio
structure counts only from convergent index ~17–21 upward, below that it is
unusable · the preferred small-q combination is modulus 17 with stride 4 ·
the stride-11 alternative is explicitly not a golden-section step.

**Measured, and it splits into two regimes with two different generators.**

1. **The index floor is real and my specs were below it.** `F(n+1)/F(n)` is a
   RATIONAL of period `F(n)`; convergent error n=10 → **1.5e-4**, n=13 →
   8.2e-6, **n=17 → 1.8e-7**, n=21 → **3.7e-9**. Below the floor the ratio
   *resonates* — the exact moiré the golden construction exists to prevent, so
   a sub-floor lattice demonstrates the OPPOSITE of the property. The emergent
   parastichy pair of an N-point Vogel lattice sits at `≈ √N`, so index ≥17
   needs **N ≳ F(17)² = 2 550 409**. **W5 was specced at N=4096 (emergent pair
   F(10)/F(11)) and its discovery step searched `j ∈ {1..60}` — structurally
   incapable of returning anything above F(10). The sub-floor answer was
   hardcoded and every bar would have been measured on it.** W2s-a likewise at
   N=2048. Both corrected; both now carry an N-sweep so the floor is measured,
   not inherited.
2. **The angle is NOT what the floor binds.** `2π(1−1/φ)` in f64 is irrational
   to ~1e-16 at any N. What the floor binds is the *addressable stride family*.
   Keeping these apart is load-bearing: "fix the angle" would be the wrong
   repair.
3. **Below the floor, do not approximate φ at all — ENUMERATE.** This is the
   payload. `helix/KNOWLEDGE.md:320` labels `(i·11)%17` the **"golden-step"**
   because `17/φ = 10.51 → 11`. Enumerating all 16 strides mod 17, prefix
   star-discrepancy at m = 5/9/13:

   | stride | m=5 | m=9 | m=13 | |
   |---|---|---|---|---|
   | **4** | **0.2000** | **0.1111** | **0.0769** | shipped `CurveRuler`, D-QUANTGATE |
   | 11 | 0.2000 | 0.1503 | 0.0905 | the "golden-step" |

   **Stride 4 is never worse than 11 and strictly better at m=9 and m=13.**
   And `17/11 = 1.5455` is **7.3e-2** from φ — an order of magnitude worse
   than `13/8` (7.0e-3). At q=17 there is no irrational to approximate, and
   the φ-derived selector picks the measurably worse integer. *(Honest
   caveat: under worst-case-over-all-m, strides 10–15 tie at 0.5000 and
   stride 4 reads 0.7647 — that metric is dominated by m=2 where every stride
   is degenerate. The useful-prefix reading is reported as the one that
   matters, and the other is stated rather than hidden.)*

4. **The MECHANISM below the floor is TEMPERAMENT, not approximation**
   (operator framing: the stride-11-region walk behaves like a 5/3
   circle-of-fifths with a distributed Pythagorean-style comma — not a
   genuine golden step, yet it does not collapse; and that last property is
   the whole theory). A coprime stride `s` over `q` is
   structurally a **circle of fifths**: it does not *approximate* the
   irrational target, it **closes the cycle exactly** (coprimality → full
   permutation) and **distributes the incommensurability error — the comma —
   uniformly around the cycle** instead of letting it accumulate at a seam.
   Measured: 12 pure fifths miss closure by the Pythagorean comma
   **+23.46 ct**; 17-TET's fifth (stride 10) closes **exactly by
   construction** with **+3.93 ct/fifth** spread around the circle. That
   distributed comma is precisely the deterministic **anti-moiré dither**
   D-QUANTGATE names. So the survival property below the floor is
   **CLOSURE, not GOLDENNESS** — which is why the shipped integer walk was
   always right even while its "golden-step" rationale was wrong.
   - **The operator's tentative 5/3 identification resolves exactly:** the
     5/3 major sixth (884.36 ct) lands on **stride 13** (917.65 ct, +33.3 ct)
     — and **13 ≡ −4 (mod 17)**: `4·13 ≡ 1`, they are inverses. **Stride 4 IS
     the descending-5/3-sixth circle.** Stride 11 sits at 776.5 ct ≈ 8/5, the
     neighbouring sixth — the hedge was right, and the measured answer is
     13 = −4, i.e. the operator's own stride 4 read backwards.
   - Interval map for the record (17-TET step vs just, cents; SIGN
     CORRECTED 2026-08-12, codex P2 on PR #932 — the first version used
     `just − TET` for four of five entries and `TET − just` for the fifth,
     an internal contradiction against the "+3.93 ct/fifth" line above it;
     the fixed convention is `TET − just` throughout): stride 3 ≈ 9/8
     (+7.9 ct), 7 ≈ 4/3 (−3.9 ct), **10 ≈ 3/2 (+3.9 ct, the fifth — now
     consistent with the "+3.93 ct/fifth" line above)**, **4 ≈ 7/6
     (+15.5 ct)**, **13 ≈ 5/3 (+33.3 ct)**.
   - The two-regime table thus gets its mechanism column: **continuum** =
     equidistribution by irrationality (φ, needs the index floor);
     **quantized** = temperament — exact closure + distributed comma
     (coprime walk, needs no φ at all).

**Why this is the same defect as two others found this week — third instance
of one shape.** A rule true in its asymptotic home, applied where it does not
hold: **R² "structurally near-blind to encoder bias"** (true at +1.59 Pa →
6th decimal, false at +92.76 Pa → 3rd); **"zero removed lines proves a pure
prepend"** (proves *additive*; an end-append also deletes nothing); and now
**"the golden step is the best step"** (true as N→∞, false at q=17). **The
repair is identical each time: measure/enumerate in the regime you are
actually in, instead of inheriting the asymptotic label.** Corollary for this
workspace: a name carrying a theory ("golden-step", "lossless", "pure
prepend") is a claim, and claims get falsifiers — the label is not the
evidence. *(And the temperament frame shows the deeper reason the label was
seductive: the walk really does share φ's PURPOSE — even coverage without
collapse — it just achieves it by closure + distributed comma rather than by
irrationality. Same goal, different mechanism, wrong name.)*

**Consequences.** Report §10.5 carries the two-regime table + the
addressing note (a `u8` rail holds 256 < F(17)=1597, so the continuum side
must read tier-then-member — while the quantized side never has the problem,
since the 17-alphabet fits a byte fifteen times over). `weather-w-probes-v1`
§0 carries the enumerate-don't-approximate rule so every worker inherits it.
`helix/KNOWLEDGE.md:320`'s "golden-step" label is filed in `ISSUES.md` as
misleading-but-not-wrong-code — the walk it names is fine, the *reason given
for choosing it* is not, and a future session picking a stride by that
rationale would pick 11.

