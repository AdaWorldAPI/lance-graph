## 2026-08-11 — E-A-JITTER-AMPLITUDE-YOU-CHOSE-IS-NOT-AN-UNCERTAINTY-YOU-MEASURED-1

**Status:** FINDING `[G]` — CT-F4 + CT-F7, `comet_tail_f4_f7.py` / `.json`,
report §5.6–5.7. Amends E-THE-OFFSET-WAS-THE-APPARATUS-... below, which stands
as written (it was correct for the amplitude it tested).

**My apparatus test condemned a number using a sensitivity amplitude I picked
out of the air.** CT-F3 jittered the storm center by ±100 km and found the
alignment error moved 29.4° — so the −40° offset was declared unmeasurable.
But ±100 km was never *measured*; it was a plausible-sounding round number.
The non-circular question is **how far apart independent center definitions
actually land**, and that is the uncertainty. Four definitions across three
physical fields (sub-grid MSLP min; ∇²p centroid; **10m vorticity** centroid;
sub-grid z850 min) agree to **20 km** / **73 km**, and the answers they give
span **2.3°** / **6.5°** — so the real apparatus noise is ≈ ±3–7° and the
offset **is** measurable. F3 was not wrong; its amplitude was unjustified.

**Two structural lessons, both reusable:**
1. *Prefer a measured disagreement to a chosen perturbation.* Independent
   method variants are a free, non-arbitrary uncertainty estimate — and where
   both exist, the isotropic jitter was a 2× **over**estimate, because real
   variants cluster along a preferred axis rather than scattering evenly.
2. *An anti-vacuity guard must be allowed to refuse a PASS you want.* Storm 1
   scored a 2.3° spread — but its four centers agreed to 20 km, below the
   31.9 km grid diagonal, so the pre-registered CT-F4c guard returned
   NO-VERDICT rather than banking a free pass. Storm 2, whose definitions
   genuinely disagreed (73 km = 2.4× the diagonal), is the one that carries
   the result. The guard cost me the tidier of the two numbers, which is
   what tells me it was real.

**Also, the friction candidate was partly MIS-SPECIFIED and the data said so
before I did.** CT-F7 measured cross-isobar inflow over land (blind storm
selection): **+34.2° land vs +20.5° ocean inside the same disk** — a paired
contrast that controls for depth/latitude/curvature, textbook magnitudes, both
bars passed. But F2/F7 bound the rotation of the **wind** relative to the
isobars, whereas the CT-E3 offset is a rotation of the **pressure dipole**;
friction does not rotate the pressure field except at second order. And an
unplanned pairing already in the F2 output points the wrong way for it: storm 1
is 1 % land (+14.7° inflow), storm 2 is 46 % land (+22.0°), yet their offsets
are −42.0° and −40.2° — **the more frictional storm has the smaller offset.**
`[S]`, n=2, undesigned, recorded as suggestive.

Net: candidate 1 (baroclinic tilt) leading and near-unopposed; candidate 3
(center bias) bounded at ≈±5°; candidate 2 re-scoped, with CT-F9 written to
test the mechanism it *should* have been about (Ekman pumping vs land-fraction
**asymmetry** across the disk, not mean land fraction). Binding constraint is
now n ≥ 10 storms, not the apparatus.

