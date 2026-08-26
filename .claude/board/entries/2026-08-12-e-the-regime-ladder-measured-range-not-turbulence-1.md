## 2026-08-12 — E-THE-REGIME-LADDER-MEASURED-RANGE-NOT-TURBULENCE-1

**Status:** FINDING `[G]` — measured, same run, found by an operator
question rather than by any gate. Qualifies
`E-THE-HYPOTHESIS-REFUTED-CLEANLY-AND-REVERSED-1`, written one commit
earlier, whose *stronger* half it withdraws.

**The operator asked: do we have a storm-modelling problem, since we do
not actively model vortices?** Checked, and the answer is yes, twice over.

**First, the plain fact.** `substrate_comfort_d_cz_2_7.py` loads **only**
`mean_sea_level_pressure`. No wind, no vorticity, no rotation enters any
bar. D-CZ-0/1 fetched 10 m winds solely to *report* `spd_sigma`; nothing
consumed them. R4 "STORM" is a **scalar pressure-gradient** regime, and
the hypothesis it was built to test was about *"high velocity differences
/ turbulence"* — a dynamical property that appears nowhere in the
measurement.

**Second, the confound that follows.** Chasing the question produced two
measurements that qualify the previous entry:

| relationship | statistic |
|---|---|
| `L` vs the cell's `saturation` | Pearson **+0.917** (n=8) |
| `L̄[T]` vs the regime's own value range | **Spearman +1.000 — perfect** |

`L` is essentially *how much of the target falls outside the donor's
codebook range*. A regime's own range determines how hard it is to cover
— R4's is ~**18×** R1's. So C3's monotone rise **restates the width
ordering**, which the `|∇p|` ladder itself produced (deeper low ⇒ steeper
gradient ⇒ wider range). Width is not the whole story — `R3 → R2` has a
*wider* donor yet 0.949 saturation, because the boxes sit at different
absolute pressure levels — so the real driver is **coverage** (width AND
offset), which `saturation` captures directly.

C4 is inflated the same way: `Δ` was amended to RMSE **in Pa**, and RMSE
scales with range, so an absolute-Pa margin cannot be compared across
regimes differing 18×. Normalised as a ratio it reads **3.96 / 1.85 /
1.14 / 2.35** — not monotone, and R1 is the extreme, not R4.

**What survives / what is withdrawn:**

- **SURVIVES:** the hypothesis is **NOT SUPPORTED**. No sign flip in
  either measure, normalised or not; `CAL-ABS` wins its own diagonal in
  all four regimes. Unaffected by the confound.
- **WITHDRAWN:** *"cleanly reversed, monotonic"* and *"storms are LESS
  forgiving of bad calibration."* As measured that says **wide-range
  boxes are harder to cover with a foreign codebook** — arithmetic.

**Why no gate caught this.** C1c *did* pass, and it was the right gate to
have — but it measured the structure of the **`|∇p|` field** (decay
length, Gini, tail ratio), not rotational structure, and the confound is
between the ladder's own discriminator and the codebook's coverage, not
between two regimes. A gate cannot catch a confound that lives *inside*
the variable it was told to trust. **The operator's question was the
instrument here; nothing in the apparatus was positioned to ask it.**

**The transferable rule.** *When a regime axis is built from a scalar
derived from the same field the codec quantises, the axis and the codec's
difficulty are not independent.* Before reading a cross-regime encoding
result as physics, check whether the regime discriminator predicts the
encoding's own failure mode — here, `|∇p|` predicts range, and range
predicts saturation, and saturation IS `L`. A dynamical hypothesis needs
a **dynamical** discriminator (ζ = ∂v/∂x − ∂u/∂y, Okubo–Weiss) and a
**range-normalised** transfer metric, or coverage will masquerade as the
finding every time.

**Filed as D-CZ-8**, the pre-condition for re-asking C3/C4 as a turbulence
question rather than a width question.

