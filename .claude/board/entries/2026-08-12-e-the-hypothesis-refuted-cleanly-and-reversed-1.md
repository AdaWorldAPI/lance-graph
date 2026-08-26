## 2026-08-12 — E-THE-HYPOTHESIS-REFUTED-CLEANLY-AND-REVERSED-1

**Status:** FINDING `[G]` — measured, both pre-registered bars, one run.
`probes/weather-p1/substrate_comfort_d_cz_2_7.json`, plan §7.

**The hypothesis this whole plan existed to test is refuted, and refuted
in the SHARPEST available form: not "no effect", but a clean monotonic
reversal.** The operator's hypothesis — *a badly-calibrated substrate
that maps dynamically preserves MORE structure in strong storms than a
well-calibrated absolute one* — was tested two ways, pre-registered, on
an equal-budget cross-swap over four regimes (calm → ocean → active →
storm):

- **C3 (transfer loss)** required `L̄[R4] < L̄[R1]`. Measured:
  `L̄` = **0.011 → 0.309 → 0.671 → 0.690**, R1→R4 — a **62× increase**,
  monotonic across all four tiers. Storms are *less* forgiving of a
  foreign absolute calibration, not more.
- **C4 (the crossover)** required a sign flip against the diagonal —
  absolute winning in calm, dynamic winning in storm. Measured: absolute
  wins its own diagonal in **all four** regimes, and its margin **grows**
  from ~1 Pa (R1/R2) to **10.78 Pa** (R4) — the opposite of a crossover.

Both measures point the same direction. This is not two noisy nulls; it
is one coherent, reversed relationship measured twice by independent
instruments (`ρ` on rank vs RMSE on the diagonal).

**A real construction bug was found and fixed en route, via the gate
designed to catch exactly this class of defect.** C0 — *"both controls
must be WORSE than every real arm in every regime; if either matches a
real arm anywhere, that cell measures nothing"* — failed on its first run
in R1 and R4: `GEO-DEGENERATE` was built from `truth[:len//64]`, the
first slice of an array already shuffled by `rng.choice` for equal
budget. A random subsample of a flat array is not reliably narrower in
range than the whole array; that is a materially different construction
from D-CZ-1's correct one (`p[si,sj][:n_i,:n_i]`, a genuine 2-D corner).
Fixed by carrying each regime's full 2-D box alongside the flat
evaluation sample and building the degenerate donor from a real spatial
corner. **Disable-verified**: reverting to the flat-slice construction
reproduces the exact original failure; the fix reproduces the identical
real-arm numbers (C2–C6 unchanged) while making C0 pass cleanly in all
four regimes. The bug lived entirely in the control; nothing about the
real arms was ever wrong. Recorded because it is this arc's C0 gate doing
precisely the job it was built for — catching a defect in the apparatus
before it could contaminate a verdict — on the very first probe that
tried to use it for something other than a smoke test.

**And it contradicts an earlier exploratory hint, on purpose, stated
plainly rather than averaged away.** §6.6 measured, *within R4 only*, a
single arm's saturation correlating with storm `\|∇p\|` at ρ = +0.444
(p = 0.058, explicitly labelled not significant, not a result). That
correlation's *direction* matched the hypothesis. The properly-powered,
pre-registered, cross-regime test says the opposite. Where an
unpre-registered n=19 single-regime correlation and a pre-registered
cross-swap disagree, **the pre-registered result governs** — and this
entry exists so a later read of the plan does not quietly split the
difference between "p=0.058, direction matches" and "reversed,
monotonic, two independent measures" into a false middle.

**C1c passed and licensed the whole exercise** — the regimes measurably
differ in correlation structure (decay length, Gini, tail ratio of
`|∇p|`), not merely in gradient magnitude, so the refutation above is a
finding about calibration under real structural variation, not an
artifact of four copies of one condition.

**What is NOT concluded.** One box size (16°, 4225 cells), one variable
(MSLP), three timesteps plus 19 storms. C5 — the *other* half of "good
geometry vs badly calibrated" — was never run: the golden index floor
(N ≥ 2,550,409) is three orders of magnitude above what a box this size
can hold, so `GEO-GOLDEN-HI` has no admissible construction at this
scale. The geometry axis of the operator's original framing remains
completely untested.

**Cross-ref:** `E-A-HORSE-RACE-IS-NOT-A-CROSS-SWAP-1` (the instrument
this run finally exercised); `E-A-CONTROL-THAT-CANNOT-LOSE-IS-NO-CONTROL-1`
(the exact defect class C0 caught here); `E-A-FIGURE-CITED-TWICE-IS-NOT-
CONFIRMED-ONCE-1` (why §6.6's direction is not treated as confirmed by
this section merely because it was written down once already).

