# Comet-Tail Report — Wavenumber-1 Asymmetry of Translating Cyclones

> **Audience:** meteorology-literate product lead (geostrophy, Ekman layer,
> steering flow, azimuthal wavenumber decomposition assumed known).
> **Status:** EXPLORATORY probe result, all pre-registered bars met (2/2
> storms). NOT a promoted EV — the audit gate (plan §8) has not run on it.
> **Provenance:** `comet_tail_probe.py` / `.json`, commit `db57aac0`.
> Data: WeatherBench2 ERA5 6h 0.25° (`1959-2022-6h-1440x721.zarr`),
> `mean_sea_level_pressure`, t=91246→91247 (2021-06-15 12Z→18Z).
> **Grades:** `[G]` measured/textbook · `[H]` bounded inference · `[S]`
> speculation, labelled as such.

---

## 1. Executive summary

An axisymmetric (ring-profile) model of an extratropical low leaves ~30–70 %
of the in-disk MSLP variance unexplained. This probe shows that on two real
storms, **89–92 % of that azimuthal residual is a single wavenumber-1 dipole**
— the "comet tail" — whose orientation is **predicted by the storm's own
motion vector** (low pole left of motion, NH), and whose amplitude grows
~linearly with radius, exactly the signature of a linear background pressure
gradient advecting the vortex. `[G]` for the measurements; `[H]` for the
generalization beyond n=2.

> **⚠ UPDATE 2026-08-11 (§5.9, CT-N).** A blind 10-storm sample was run
> specifically to test whether the n=2 signed prediction generalizes. **The
> magnitude and explanatory power (§5.9's N2/N3/N4) hold up well at scale —
> the sign consistency of §4's CT-E3 does not** (6/10 = 60 % vs a 70 % bar,
> essentially a coin flip once you leave the two original storms). Read the
> rest of this report — especially the "left-of-motion, signed" framing in §2
> and §4 — as demonstrated **on storms 1–2 specifically**, not yet established
> as a general rule. §5.9 has the honest breakdown, including a post-hoc lead
> (motion-bearing noise at low displacement) that may explain part of the gap
> and is *not* used to override the failed bar.
>
> **⚠⚠ UPDATE 2026-08-11 (§5.10, a SECOND independent sample).** A fresh,
> mechanically-generated 1980–1995 sample scores 8/10 = 0.80 unfiltered —
> reversing §5.9's verdict on its own bar. Proper statistics keep this
> **borderline, not resolved**: neither sample alone clears a two-sided 0.05,
> and the combined pooled figure (14/20) lands right at the noise floor
> (one-sided p ≈ 0.058). The one number that *does* strengthen — pooling both
> samples' large-displacement storms, 6/7 = 0.857 — supports an **apparatus**
> explanation (motion-bearing noise on slow-moving storms) over the
> **regime-contamination** explanation §5.9 initially favored: checked
> directly, the regime filter does NOT rescue sample 1 (drops it to exactly
> chance, 0.500). Read §4/§5.9's "not established" verdict as still current,
> now with a specific, falsifiable next step (CT-F14) rather than a closed
> question.
>
> **⚠⚠⚠⚠ CORRECTION 2026-08-11 (external review of PR #926) — THE
> COMPRESSION NUMBERS IN THIS REPORT WERE FROM THE WRONG MODEL.** Codex
> flagged, and re-measurement confirms, that `decompose()` fits `a1[b]`,
> `b1[b]` **per ring** — 12 rings × 2 = **24** free dipole parameters, not the
> **2** ("amplitude slope + bearing") the storage claim describes. So the
> published R² 0.972 / 0.926 belongs to a **36-parameter** model, while the
> claimed 14-value representation is a different, more constrained one. Both
> are now measured (`comet_tail_probe.py`, `R2_profile_wn1_constrained_2param`):
>
> | model | params | storm 1 | storm 2 |
> |---|---:|---:|---:|
> | ring profile only | 12 | 0.635 | 0.294 |
> | + per-ring dipole (what was published) | 36 | 0.972 | 0.926 |
> | **+ constrained 2-param dipole (what was CLAIMED)** | **14** | **0.943** | **0.909** |
>
> **The corrected headline is 90.9–94.3 %, not 93–97 %.** The structural
> finding survives — 14 values still lift a storm from 29–63 % to 91–94 % —
> but every "93–97 %" in this document is an overstatement of ~2.5× in
> parameter count, and is superseded by the table above. The constrained model
> is the physically motivated one (a linear background gives exactly one
> amplitude slope and one bearing, §2), so this is a correction of the
> MEASUREMENT to the claim, not a retreat from the claim.
>
> **⚠⚠⚠ UPDATE 2026-08-11 (§5.11, CT-F14, the properly-powered test).** The
> pre-registered decision rule technically fires "established" on a pooled
> 3-sample figure (p=0.0145) — but the single test this whole exercise was
> designed to produce, CT-F14 alone (n=19, the largest and most carefully
> powered sample in the chain), does **NOT** independently clear
> significance (p=0.0835) and its raw rate (0.684) sits below the 0.70 bar.
> Applying the same scrutiny used on §5.10's reversal: **graded down from
> "established" to "still suggestive"**, and the pre-registration's own
> pooling rule is flagged as having a real gap (no contingency for a large
> new sample disagreeing with small prior fragments). The directional claim
> remains **not established** after four probes and three independent
> samples (n=41 total storms). The structural claim (wn1 dominance,
> explanatory power) is untouched throughout.

Product consequence `[S]`: a storm's pressure field compresses to
**center position + ~12 ring means + one dipole vector** at 93–97 % variance
explained, and the dipole *encodes the motion* — a candidate single-frame
motion predictor and a natural fit for the substrate's 3-integer spiral
addressing (`highheelbgz`).

---

## 2. Physical basis (existing theory, nothing invented)

All `[G]`, textbook dynamic meteorology:

1. **A translating vortex is a vortex embedded in a steering flow.** In the
   vortex-relative frame the environment appears as a superposed background
   flow (the "airplane relative wind" framing).
2. **Geostrophy makes the geometry signed.** If the vortex translates with
   the geostrophic steering flow `v_g = (1/fρ) k×∇p`, the background pressure
   gradient is *perpendicular* to the motion, with the **low pole 90° to the
   LEFT of the motion vector** (Northern Hemisphere; sign flips in SH).
3. **A linear background gradient is pure wavenumber-1.** Around any circle
   centered on the vortex, a linear field `p_bg = a·x + b·y` has zero ring
   mean and projects entirely onto `cos(θ−θ₀)` with amplitude
   `√(a²+b²)·r` — i.e. it survives ring-profile removal *completely* as a
   wn-1 residual growing linearly in radius.

So the theory yields three independent, falsifiable predictions: dominance of
wn-1 in the residual, a *signed* orientation locked to the motion vector, and
`a₁(r) ∝ r`. The probe tests all three.

Caveat stated up front `[G]`: real extratropical cyclones are baroclinic and
are steered by the mid-tropospheric flow, not the surface gradient alone; the
surface-level prediction is therefore expected to hold *up to a systematic
rotation* (see §6). The probe's ±45° tolerance was chosen to admit that
rotation while still rejecting the null (uniform orientation) at 0.25 per
storm.

---

## 3. Data and method

- **Domain:** disk of R = 1200 km around the detected center, planar
  local-tangent geometry with `cos(lat_center)` zonal metric; 100 km rings.
- **Center finding:** deepest zonal-anomaly MSLP minimum; at t+6h the center
  is re-found within a 600 km search radius (trackability gate CT-E2).
- **Decomposition:** per-ring mean → radial profile `p̄(r)`; per-ring
  least-squares wavenumber-1 fit
  `a₁(r)·cos θ + b₁(r)·sin θ` on the residual; amplitude-weighted mean dipole
  bearing across rings → the **low-pole bearing** (`+π` from the high pole of
  the fitted dipole).
- **Motion:** bearing of the 6h center displacement; predicted low pole =
  motion bearing + 90° CCW.
- **Storms:** (1) 55.75N 334.5E — the arc's reference storm; (2) 67.0N 28.0E
  — an independent center found by the preceding go-territory probe, used as
  replication.

---

## 4. Falsification design and results

All bars pre-registered in the probe docstring *before* the run; the run was
committed unmodified. Null model for CT-E3: uniform dipole orientation →
P(hit ±45°) = 0.25 per storm; 2/2 joint = 0.0625. n=2 is stated, not hidden.

| Bar | Pre-registered criterion | Storm 1 (55.75N) | Storm 2 (67N) | Verdict |
|---|---|---|---|---|
| **CT-E1** | wn-1 ≥ 0.40 of azimuthal-residual variance | **0.924** | **0.895** | PASS — wn-1 is not merely dominant, it is nearly *all* of the asymmetry |
| **CT-E2** | trackable: displacement ≥ 100 km / 6h within 600 km | 279 km | 440 km | PASS |
| **CT-E3** | **signed:** low pole = motion + 90° CCW, within ±45° | error **−42.0°** | error **−40.2°** | PASS 2/2, same side both storms |
| **CT-E4** | profile + wn-1 explains ≥ 0.80 of in-disk variance | 0.635 → **0.972** | 0.294 → **0.926** | PASS |
| **CT-E5** | observation (no bar): `a₁(r) ∝ r` | corr **0.800** | corr **0.998** | consistent with linear-background signature |

**Interpretation discipline:** CT-E1/E4/E5 are unsigned goodness-of-fit
results — a skeptic could attribute them to "any smooth large-scale gradient."
CT-E3 is the load-bearing test: it is *signed by an independent quantity*
(the motion vector, measured from a different pair of fields), and both storms
land on the predicted side. That is what elevates this from curve-fitting to
physics `[H at n=2]`.

> **⚠ CT-E3 RE-GRADED 2026-08-11 by CT-F3 (§5.1), as §5's pre-registration
> required.** The ±45° HIT stands — but the *offset magnitude* does not. A
> ±100 km center jitter moves the alignment error by up to **29.4°** (storm 1),
> which is comparable to the offset itself. So "−42° / −40°" is **NOT a robust
> number**: read CT-E3 as *"the low pole lies left-of-motion, within an
> apparatus uncertainty of roughly ±15°"*, and read the specific offset as
> unresolved at this centering precision. The *height dependence* of the error
> (§5.2) is a separate and much larger signal and is not affected by this
> re-grade.
>
> > **⚠⚠ RE-GRADE AMENDED 2026-08-11 by CT-F4 (§5.6).** The note above is
> > kept verbatim because it was correct *for the amplitude it tested* — but
> > ±100 km was an amplitude I **chose**, not one I measured. Four
> > independent center definitions (three different physical fields) turn out
> > to agree to **20 km** on storm 1 and **73 km** on storm 2, and the
> > alignment error they produce spans only **2.3° / 6.5°**. So the operative
> > apparatus uncertainty is **≈ ±3–7°, not ±15°**, and the −42°/−40° offset
> > **is** measurable above it. F3 was not wrong; its jitter amplitude was
> > unjustified — which is exactly what F4 was pre-registered to find out.

**What this resolves:** the earlier sunflower-lattice probe failed its
axisymmetry bar (E1: 0.639 < 0.70) — the missing third of the storm was
unexplained. It *was* the tail: adding one dipole per ring takes the same
storm from R² 0.635 to 0.972. The golden-spiral/ring encoding was not wrong,
it was incomplete by exactly one mode.

---

## 5. The systematic −40° offset — follow-up work (NOT yet run)

Both storms miss the naive 90°-left prediction by **−42° and −40°** — nearly
identical magnitude, same rotation sense. With n=2 this is an observation
`[S]`, but a common offset of matched size is the signature of a *systematic
mechanism*, not noise. Three candidates, ranked:

1. **Steering-level / baroclinic-tilt rotation** `[S]`, prime candidate.
   Extratropical cyclones translate with the mid-tropospheric (≈500–700 hPa)
   steering flow. The *surface* background gradient is rotated relative to
   the steering-level gradient by the thermal wind (the system's westward
   tilt with height). The probe measured the dipole at MSLP but the motion is
   set aloft — a fixed rotation between the two is expected, not anomalous.
2. **Ekman / surface-friction turning** `[S]`. Boundary-layer friction turns
   the surface flow 10–30° cross-isobar toward low pressure (more over land,
   less over open ocean) and drives Ekman pumping that distorts the surface
   pressure asymmetry. Magnitude range is plausibly consistent with −40° in
   combination with (1), unlikely to explain it alone over ocean.
3. **Center-finder bias** `[S]`, must be excluded before believing either
   mechanism. "Deepest zonal-anomaly point" ≠ circulation center; a center
   displaced along-track biases the fitted dipole orientation.

**RUN 2026-08-11** — `comet_tail_followup.py` / `.json`. Bars pre-registered in
that probe's docstring before the run; committed unmodified except for one
`grad_p` shape bug fix and one added *diagnostic* field (both recorded in the
probe's own RUN LOG; **no bar was added, removed, or loosened**).

### 5.1 CT-F3 — apparatus — **FAILED the gate** (candidate 3 is live)

Six center choices per storm: MSLP minimum (baseline), a ∇²p-centroid
(geostrophic-vorticity proxy), and ±100 km jitters along- and across-track.

| Storm | baseline | ∇²p centroid | jitter range | spread | verdict |
|---|---|---|---|---|---|
| 1 (55.75N) | −42.0° | −42.0° (same grid point) | −25.4° … −54.8° | **29.4°** | APPARATUS-DOMINATED |
| 2 (67N) | −40.2° | −38.6° (toward zero, marginal) | −29.8° … −49.3° | **19.4°** | SURVIVES-WITH-UNCERTAINTY |

Worst spread 29.4° **> the 20° bar → gate FAILED.** Per the pre-registration
this forces two things, both done: CT-E3 is re-graded in §4, and CT-F1/CT-F2
below are reported as *measured but gated* — not as settled verdicts.

The mechanism is understood and was anticipated: a 100 km miscentering of a
monopole injects a wn-1 by construction. What the test establishes is that the
**offset magnitude is inside the apparatus noise**, so no offset constant may
be derived from it. `[G]`

*Post-hoc observation, explicitly NOT a rescue of the failed gate:* the
level-dependence signal in §5.2 is 92–102°, i.e. **3–5× this apparatus noise**.
That does not un-fail F3 — it means the right next probe is F3 re-run *at the
level where the error crosses zero*, with a sub-grid center fit. Recording the
comparison and letting the failed gate generate the next probe is the
disciplined move; overriding the gate on the strength of it would be the
"indictment fired, post-hoc rescue" anti-pattern this arc already has on its
open-P1 list.

### 5.2 CT-F1 — steering level — **strong signal, formally mixed** (candidate 1 favoured)

The store ships all 13 pressure levels in one chunk, so the yes/no became a
sweep. Alignment error vs the *same surface-measured motion bearing*:

| level | storm 1 (own ctr) | storm 2 (sfc ctr) |
|---|---|---|
| 1000 hPa | −40.5° | −39.7° |
| 925 | −32.5° | −29.9° |
| 850 | −23.8° | −22.3° |
| 700 | −8.1° | −12.9° |
| 600 | **−2.1°** | −7.7° |
| 500 | +2.9° | −2.8° |
| 400 | +8.7° | **+1.0°** |
| 300 | +7.8° | +2.0° |

**Both storms show a smooth, monotone climb from ≈ −40° at the surface through
zero in the mid-troposphere** — storm 1 crosses at ~600–650 hPa, storm 2 at
~400–500 hPa. Spread across levels: 101.8° / 91.6°. This is exactly the
baroclinic-tilt/steering-level prediction: the surface gradient is rotated
relative to the steering-level gradient, and the rotation unwinds with height.
`[H]` — the shape is unambiguous, n is still 2.

Formal bar bookkeeping, stated rather than smoothed: storm 1 **PASSES** as
written (minimum at 600 hPa, |err| 2.1°). Storm 2's *own-center* column
minimises at 100 hPa (+2.0°), which trips the pre-registered `dead-absurd`
flag — **but the added diagnostic shows why:** at 50–400 hPa the center finder
**saturated at its 600 km search radius** (586–599 km), i.e. it never found a
co-located upper center and locked onto a different system. Those rows are
apparatus, not physics. The surface-center column (unsaturated by
construction) is the one tabulated above and behaves like storm 1. This is a
real defect in F1's own apparatus, found by a diagnostic added after run 1;
the honest verdict for storm 2 is **NO-VERDICT on the own-center path**, not
"dead".

### 5.3 CT-F2 — friction — **bounded, candidate 2 is a contributor not the cause**

Measured 10m cross-isobar inflow angle (positive = turned toward the low, the
NH friction sign), rings 300–1000 km, |v10| > 3 m/s:

| storm | n | land frac | median α | IQR | ocean-only median |
|---|---|---|---|---|---|
| 1 | 6552 | 0.01 | **+14.7°** | +10.1 … +18.2 | +14.7° (n=6517) |
| 2 | 8960 | 0.46 | +22.0° | +10.5 … +34.7 | **+13.0°** (n=4674) |

Sign is positive on both storms, as predicted. Magnitude is textbook for open
ocean (10–30°). The bar that matters: **13–15° ≪ 40°**, so friction alone
**cannot** own the offset — at most about a third of it. `[G]` for the
measurement, `[H]` for the attribution.

*Apparatus can-it-fire check, unplanned but load-bearing:* storm 2 is 46 %
land, and its all-points median (+22.0°) is substantially larger than its
ocean-only median (+13.0°) — friction turning is stronger over land, exactly
as textbook. The measurement therefore discriminates a known physical contrast
in the right direction, which is evidence it is measuring what it claims.

### 5.6 CT-F4 — sub-grid center — **the blocking item CLEARS** (candidate 3 mostly retired)

`comet_tail_f4_f7.py` / `.json`. F3 showed sensitivity at an amplitude I chose.
F4 asks the non-circular question instead: **how far apart do independent
center definitions actually land?** That disagreement *is* the center
uncertainty. Four definitions, deliberately not variants of one idea — A:
sub-grid MSLP minimum (2D quadratic on the 3×3); B: ∇²p centroid (pressure
curvature); C: 10m relative-vorticity centroid (**wind** field); D: 850 hPa
geopotential minimum, sub-grid (different field *and* altitude).

| | A (MSLP) | B (∇²p) | C (10m ζ) | D (z850) | max separation | error spread |
|---|---|---|---|---|---|---|
| storm 1 | −42.5° | −41.0° | −41.8° | −43.4° | **20.3 km** | **2.3°** |
| storm 2 | −39.8° | −38.6° | −35.1° | −41.6° | **73.0 km** | **6.5°** |

- **Storm 2 PASSES CT-F4a** cleanly (6.5° ≤ 10° bar) with the anti-vacuity
  guard satisfied — its four definitions genuinely disagree (73 km = 2.4× the
  29.9 km grid diagonal), so the test had real room to fail and did not.
- **Storm 1 is NO-VERDICT on F4a by my own CT-F4c guard**: its four definitions
  agree to 20.3 km, *below* the 31.9 km grid diagonal. A 2.3° spread among
  centers that coincide proves nothing about method sensitivity, so the guard
  correctly refuses the free pass. Storm 1's bound comes instead from F4b
  (below), and is labelled as derived, not as an F4a result.
- **CT-F4b sensitivity curve** (spread in ° vs jitter amplitude in km),
  monotone on both as pre-registered:

  | | 25 km | 50 km | 100 km | 200 km |
  |---|---|---|---|---|
  | storm 1 | 6.5° | 13.8° | 30.9° | 77.2° |
  | storm 2 | 5.0° | 9.8° | 19.3° | 35.8° |

  Read against the *measured* uncertainties: storm 1's 20 km sits below the
  25 km point → **≤ 6.5°**; storm 2's 73 km would predict ~14° isotropically,
  yet the direct F4a measurement is 6.5°. The gap is informative: real center
  definitions do **not** scatter isotropically — they cluster along a preferred
  axis, so the four-direction jitter is an **upper bound**, not an estimate.
  That is the second reason F3's number over-stated the problem. `[H]`

**Consequence:** the offset is measurable with an error bar of roughly ±3–7°.
CT-F4 was the blocking item ahead of any constant-fitting; it is cleared for
storm 2 and bounded for storm 1. Candidate 3 (center-finder bias) drops from
"live and not excluded" to **bounded at ≈±5°, i.e. ~⅛ of the offset**. `[G]`

*One observation worth a follow-up, n=1:* on storm 2 the **wind-based** center
(C) sits ~65 km north of the pressure minimum and moves the error the furthest
toward zero (−35.1° vs −39.8°). Whether the circulation center is
systematically the better reference for this test is untested.

### 5.7 CT-F7 — friction over LAND — **both bars pass, and it re-scopes candidate 2**

Operator-requested replication over land. Storm selected **blind to its
inflow**: deepest NH zonal-anomaly low whose 300–1000 km ring is ≥ 70 % land.
Selection returned **28.50N 67.50E** (ring 80 % land, anomaly −2609 Pa).
Orography guard dropped 826 of 2642 land points (31 %) above 1000 m, since
MSLP over high terrain is an extrapolated fiction whose gradient would corrupt
the geostrophic reference.

| | n | median α | verdict |
|---|---|---|---|
| **land** points (oro-guarded) | 1816 | **+34.2°** (IQR reported in JSON) | |
| **ocean** points, *same disk* | 743 | **+20.5°** | |
| CT-F2 reference (storm 1, 99 % ocean) | 6517 | +14.7° | |

- **F7a PASSES** (34.2° ≥ 14.7+8, inside [20,50]) — but this half is
  confounded by latitude: at 28.5N, *f* is roughly half its 56N value, so
  ageostrophic effects are inherently larger. Its own storm's ocean points
  (+20.5°) already run above storm 1's (+14.7°) for that reason.
- **F7b PASSES** and is the number to trust: **+34.2° land vs +20.5° ocean
  inside the same disk**, a +13.7° paired contrast that controls for depth,
  latitude and curvature by construction. Both classes clear the n ≥ 500 bar,
  so the half is genuinely evaluable rather than NO-VERDICT.
- **F7d is False** — 34.2° < the 35° threshold, though not by much. Over
  *land* friction turning is roughly 2.4× its ocean value, so the friction
  bound is **surface-type dependent and must not be applied globally**, exactly
  as pre-warned.

Two honest caveats. **(i)** The blind selection was blind to storm *type* as
well as to the answer: at 28.5N in mid-June this is a monsoon-season thermal
low, not an extratropical cyclone. It is a legitimate test of the *inflow
apparatus* over land, and a weaker analogue of storms 1–2 dynamically.
**(ii)** The apparatus passes its own can-it-fire test twice now — land/ocean
within one storm here, and land/ocean within storm 2 in §5.3 — in the same
direction both times.

> **Self-correction, and it matters more than the pass.** F2/F7 measure the
> **wind** turning relative to the isobars. The CT-E3 offset is a rotation of
> the **pressure dipole** relative to the motion. Friction rotates the surface
> wind *within* a given pressure field; it does not rotate the pressure field
> itself, except at second order through Ekman-pumping feedback. So candidate 2
> as originally written in §5 was **partly mis-specified** — F2/F7 bound a
> mechanism that was never the leading route to a pressure-dipole rotation.
> `[H]`

> **An unplanned paired natural experiment, post-hoc but clean.** Storm 1 is
> **1 % land** with +14.7° inflow; storm 2 is **46 % land** with +22.0° inflow
> — a real 7.3° difference in actual surface friction. Their offsets are
> **−42.0° and −40.2°**, within 1.8° of each other, and the *more* frictional
> storm has the *smaller* offset. If friction drove the offset the two should
> separate substantially; they do not. This was not designed — it fell out of
> the land fractions already in the CT-F2 output — and it is n=2, so it is
> recorded as suggestive, not decisive. `[S]`

### 5.8 CT-F5 — walking-center geopotential sweep — **fixes storm 2's saturation defect**

`comet_tail_f5_n10.py` / `.json`. §5.2 flagged that CT-F1's storm-2 own-center
path had a real apparatus defect: searching for each level's center within a
*fixed* 600 km radius of the *surface* center saturated at 5–7 of 13 levels
(offsets 586–599 km, essentially pinned at the search wall), producing a
physically-absurd "best level = 100 hPa."

**Fix:** walk the center level-by-level, searching near the *previous level's*
found center (radius 250 km per step, surface-anchored, 1000 hPa → 50 hPa) —
the center tracks continuously along the tilt axis instead of jumping the
whole tilt from the surface in one hop.

| bar | criterion | storm 1 | storm 2 |
|---|---|---|---|
| **CT-F5a** | best \|error\| in 400–850 hPa ≤ 20° | −2.1° @ 600 hPa — **PASS** | **−4.5° @ 500 hPa — PASS** |
| CT-F5b | no single step > 250 km | max step 243.3 km — PASS | max step **250.0 km — FAIL** (exactly at the cap, at the 500→400 hPa transition) |
| CT-F5c | storm 1 reproduces original within 10° | **0.0° difference** (identical) | n/a |

**F5a is the one that mattered, and it passes cleanly.** Storm 2's winning
level (500 hPa, −4.5°) was reached by a **0.0 km step** — the 600 hPa position
already coincided with the 500 hPa center — so the value that clears the bar
is untouched by any saturation. The saturation CT-F5b caught happens one step
*later*, moving from 500 hPa to 400 hPa, i.e. **outside** the band the bar
evaluates. Both storms also show a large, likely-unrelated excursion at
50 hPa (+101° / +92°, wn1_frac dropping to 0.78) — near-stratospheric, outside
the pre-registered 400–850 hPa band, not interpreted further here.

*Housekeeping, stated honestly:* CT-F5c was pre-registered but never coded as
an explicit pass/fail field; verified post-hoc from the printed numbers —
storm 1's walking result at every 400–850 hPa level is *bit-identical* to the
original F1 sweep (both used the surface center throughout, since storm 1
never needed to move before 600 hPa), so F5c passes trivially. Recorded as a
gap in this probe's own execution, not smoothed over.

**Net: storm 2's dead-absurd/NO-VERDICT status from §5.2 is corrected to a
genuine PASS**, driven by an unsaturated intermediate level. The height ladder
(§5.2) is confirmed rather than weakened by fixing this defect.

### 5.9 CT-N — n=10 blind storm sample — **the headline result of this arc**

`comet_tail_f5_n10.py` / `.json`. Ten independent synoptic times (2015–2021,
all four seasons, NH), each storm found **blind** (no hint, no inspection
before recording — deepest zonal-anomaly MSLP low, 25–75° lat). Storm 1's
anchor date is included and its t-index reproduces the arc's pinned
T0 = 91246 exactly (guard asserted in code before anything else runs).

*Data-boundary finding, unplanned:* the store's own filename claims
"1959-2022" coverage; its actual last valid timestep is **2021-12-31 18Z**,
six months short. One planned date (2022-02-14) 404'd against this; a bounds
guard was added (report + exclude, never crash) and that date swapped for an
in-range one. No pre-registered bar was touched by this fix.

| storm | disp (km/6h) | E1 wn1_frac | E4 R² | E3 error | F8 error@vort-ctr | shrinks? |
|---|---:|---:|---:|---:|---:|:---:|
| 2019-03-05 | 455 | 0.72 | 0.930 | −67.8° | −109.6° | no |
| 2020-07-20 | 406 | 0.23 | 0.323 | −39.0° | −36.8° | yes |
| 2021-06-15 (anchor) | 277 | 0.92 | 0.972 | −41.3° | −40.6° | yes |
| 2020-01-10 | 250 | 0.75 | 0.906 | +5.0° | +4.5° | yes |
| 2019-10-25 | 185 | 0.38 | 0.894 | −107.6° | −103.3° | yes |
| 2017-11-30 | 158 | 0.53 | 0.908 | −49.1° | −101.2° | no |
| 2014-09-12 | 156 | 0.49 | 0.871 | +3.2° | −12.2° | no |
| 2015-12-25 | 132 | 0.76 | 0.887 | −19.7° | −15.0° | yes |
| 2018-08-08 | 128 | 0.87 | 0.919 | +19.4° | +18.4° | yes |
| 2016-04-18 | 113 | 0.73 | 0.830 | **+165.7°** | +166.6° | no |

All 10 trackable (CT-E2 ≥ 100 km); none excluded.

| bar | criterion | result | verdict |
|---|---|---|---|
| **CT-N1** | sign consistency ≥ 0.70 | **6/10 = 0.60** | **FAIL** |
| CT-N2 | magnitude (observation, no bar) | median \|error\| = **40.2°**, IQR [19.5°, 63.2°] | — |
| CT-N3 | median wn1_frac ≥ 0.40 | **0.723** | PASS |
| CT-N4 | median R² ≥ 0.80 | **0.900** | PASS |
| CT-N5 / F8 | vort-center shrinks error ≥ 0.70 | **6/10 = 0.60** | **FAIL** |
| CT-F9 | corr(land-dipole amp, unexplained residual) | **−0.295** | does not support candidate 2 |

**The wn1-dominance and explanatory-power claims (E1/E4, CT-N3/N4) replicate
robustly at scale — the signed left-of-motion claim (CT-E3, CT-N1) does
not.** This is the honest headline: what looked like a clean 2/2 confirmation
at p = 0.0625 is, on 10 independent storms, statistically indistinguishable
from a coin flip (a naive binomial null at p=0.5 already gives P(≥6/10) ≈
0.38 — nowhere near rejecting "no signed relationship"). CT-N5/F8 shows the
earlier observation that a wind-based center shrinks the error (seen on
storm 2 in §5.6) also does **not** generalize. CT-F9 gives a clean,
unambiguous non-support for the Ekman-pumping-residual route to candidate 2.

**Post-hoc stratification — a lead for the next probe, explicitly NOT used to
override CT-N1's FAIL.** Sorting by displacement (a proxy for how well the
motion *bearing* itself is determined — small 6h displacement means a large
relative error on the direction label CT-E3 is scored against):

- Restricting to the 4 storms with displacement ≥ 250 km (closest to storms
  1–2's own 277/440 km regime): sign consistency rises to **3/4 = 0.75**,
  clearing the original 0.70 bar.
- Dropping only the single most extreme case (2016-04-18: 113 km
  displacement, near-polar 75°N where the planar `cos(lat)` approximation is
  already flagged degrading in §7, error +165.7° — essentially orthogonal to
  the prediction): sign consistency rises to **6/9 = 0.667**, still short but
  closer.

Two candidate confounds, named rather than smuggled into the verdict: **(a)
motion-bearing noise at low displacement** — a mechanical apparatus effect,
symmetric in principle; **(b) storm-type contamination in a purely blind
sample** — 2020-07-20 (32°N, 84.6°E, mid-monsoon-season) has the sample's
worst wn1_frac (0.23) *and* R² (0.32), consistent with a monsoon/thermal low
rather than a baroclinic extratropical system the whole steering-flow argument
targets (the same caveat CT-F7's land storm already carried, §5.7). Neither
is fitted or applied here. The properly pre-registered next step is a
displacement-and/or-regime-filtered n ≥ 10 sample designed *in advance* to
test candidate (a) and (b) separately, not a re-scoring of this one.

### 5.10 CT-F10/F11/F13 — a second independent sample — **borderline, in a specific and honest way**

`comet_tail_f10_f11.py` / `.json`. §5.9 named two post-hoc leads and filed
them as pre-registered reruns rather than a re-scoring. This is that rerun,
on a **fresh, mechanically-generated** date set — fixed start (1980-02-10)
+ fixed stride (411 days, chosen for no reason tied to any outcome) ×
15 candidates, landing entirely in **1980–1995**, zero overlap with the
2015–2021 sample. The stride was picked once, before any code ran that could
see a result; the dates were never inspected before recording.

**Attrition was real and different this time:** 5 of 15 candidates (33 %)
failed CT-E2 trackability — versus 0 of 10 in the first sample. **4 of those
5** cluster at 26–33°N, 67–134°E in June–September — the same South/East-Asian
monsoon-season geography already flagged as contamination-prone (§5.9's worst
storm at 32°N/85°E; §5.7's blind land storm at 28.5°N/68°E). The 5th
(1995-11-12, 54°N/161°E, November) sits outside that band and is an ordinary
mid-latitude exclusion. This is a genuine, unplanned, cross-sample-consistent
pattern about **where blind NH-wide selection breaks**, not yet exploited by
any filter.

| test | n | sign-neg fraction | median \|error\| | one-sided p (H₀: p=0.5) | verdict |
|---|---:|---:|---:|---:|---|
| **CT-F13** (raw, unfiltered — direct replication check) | 10 | **0.80** (8/10) | 41.3° | 0.055 | **PASS** (≥0.70 bar) |
| CT-F10 (disp ≥ 250 km) | 3 | 1.00 (3/3) | — | — | **NO-VERDICT** (n<6, pre-registered minimum) |
| **CT-F11** (wn1_frac ≥ 0.40) | 9 | **0.89** (8/9) | 41.9° | — | **PASS** |
| CT-F12 (both filters) | 3 | 1.00 (3/3) | — | — | reported only, n too small |

**The headline number reverses §5.9's verdict on its own bar** — 0.80 vs the
first sample's 0.60. That reversal is exactly why it needs more scrutiny, not
less, and the honest statistics do not let it stand alone:

- **Neither sample is significant on its own** at a conventional two-sided
  0.05: sample 1 (6/10) two-sided p = 0.754; sample 2 (8/10) two-sided
  p = 0.109. One-sided (the physically motivated direction, pre-registered
  from §2 onward) sample 2 gives p = 0.055 — genuinely borderline, not a
  clean win.
- **Combined across both fully independent samples: 14/20 = 0.70,
  one-sided p = 0.058.** Still borderline. Landing exactly on the 0.70 bar is
  a coincidence worth naming, not a rescue: the bar was picked before either
  sample ran.
- **The regime filter (CT-F11) does NOT rescue sample 1 — checked, and it
  makes things worse.** Applying wn1_frac ≥ 0.40 retroactively to sample 1
  removes 2020-07-20 and 2019-10-25, **both of which were negative-signed**
  (−39.0° and −107.6°). Removing two "hits" drops sample 1 from 6/10 (0.60)
  to **4/8 = 0.500 — exactly chance.** This directly contradicts the
  regime-contamination story as an explanation for sample 1's specific FAIL:
  its two lowest-wn1_frac storms happened to agree with the prediction, not
  disagree with it. Recorded as measured, not smoothed into the flattering
  reading.
- **The strongest single number in this whole probe chain is a post-hoc
  combination, clearly labeled as such — not pre-registered:** pooling the
  displacement ≥ 250 km subsets from *both* independent samples (n=4 + n=3 =
  7, spanning two different decades) gives **6/7 = 0.857 negative,
  one-sided p = 0.0625.** Displacement is a *mechanism-motivated* filter
  (it bounds motion-bearing noise, §5.9), not an outcome-based one, which is
  why this number carries more weight than F13's raw 0.80 despite being
  numerically less extreme — but n=7 is still small and this combination
  was not itself pre-registered before either sample ran.

**Net read, stated as precisely as the evidence allows:** the signed
left-of-motion claim is **not dead** (§5.9 was right to fail CT-N1 as
written, and this rerun does not overturn that FAIL under its own bar with
proper statistics) but it is **not established** either. What *does* survive
scrutiny is a consistent pattern across two independent decades: **storms
with well-determined motion (large 6h displacement) show a much stronger,
borderline-significant left-of-motion signature (6/7 combined) than storms
with poorly-determined motion or unfiltered blind selection (14/20, exactly
at the noise floor).** That is evidence *for* the apparatus explanation (§5.9
candidate (a), motion-bearing noise) and *against* the regime explanation
(§5.9 candidate (b), storm-type contamination) as the dominant driver — the
opposite weighting from what seemed most plausible after §5.9 alone, and
worth stating exactly because it cuts against the tidier story.

**CT-F14 (new, not run):** a properly powered (n ≥ 25–30), pre-registered,
displacement-filtered-ONLY sample — the single test that would move the
combined 6/7 either toward significance or back to noise. This is now the
correctly-scoped next step, not a third exploratory rerun.
**CT-F15 (new, not run):** geo-fence the blind selection away from
26–33°N/67–134°E in June–September (or add an explicit baroclinicity
proxy at intake) and re-check whether CT-E2's own trackability gate keeps
doing this filtering job for free, as the exclusion pattern above suggests.

### 5.11 CT-F14 — the properly-powered test — **fell one storm short of its own floor, and didn't clear the bar either**

`comet_tail_f14.py` / `.json`. §5.10 named CT-F14 as the correctly-scoped
next step: a single, properly powered (n≥20), displacement-filtered-only,
pre-registered sample — committed to git (`4f1a1b4f`) *before execution*,
including the exact interpretation thresholds for a pooled three-sample
figure, so the read could not be tuned after seeing results.

85 mechanically-generated candidates (fixed start 1996-01-15, fixed 61-day
stride, chosen before writing the loop), landing 1996–2010, zero overlap
with either prior sample. 64/85 trackable (75 % — closer to sample 1's 100 %
than sample 2's 67 %); **19 storms qualified** at displacement ≥ 250 km/6h —
one short of the pre-registered n=20 floor.

| test | n | sign-neg fraction | one-sided p | verdict |
|---|---:|---:|---:|---|
| **CT-F14 alone** | 19 | 0.684 (13/19) | 0.0835 | **NO-VERDICT** (n<20, pre-registered floor) — and would have **FAILED** the 0.70 bar anyway if the floor were ignored |
| Pooled, all 3 independent samples | 26 | **0.731** (19/26) | **0.0145** | crosses the pre-committed <0.05 "established" threshold |

**The pre-registered rule technically fires "established."** The honest
next step — applying the same scrutiny used on §5.10's favorable reversal —
is to check how much that crossing depends on which components are in the
pool, since two of the three components are very small:

| pooled subset | n | fraction | one-sided p |
|---|---:|---:|---:|
| All three (pre-registered) | 26 | 0.731 | **0.0145** |
| Excluding the smallest/most saturated prior subsample (sample 2, n=3, 3/3) | 23 | 0.696 | 0.0466 |
| **CT-F14 alone (the properly-powered test)** | 19 | 0.684 | **0.0835** |
| The two small prior subsamples alone (n=7) | 7 | 0.857 | 0.0625 |

**Two things are true at once, and both need to be said plainly.** (1) The
pooled figure is not purely an artifact of the smallest fragment — dropping
it still leaves p=0.0466, barely under 0.05. (2) **The single test this
whole probe was designed to produce — CT-F14 alone, n=19, the largest and
most carefully powered sample in the entire chain — does NOT independently
support the claim** (p=0.0835, "suggestive" by the arc's own pre-committed
scale, and its raw rate sits below the 0.70 bar). The "established" reading
depends on treating three heterogeneous samples (different eras, different
mechanical generators, sizes 4/3/19) as one undifferentiated pool of
Bernoulli trials — a legitimate but not the only reasonable pooling choice,
and one this probe's pre-registration did not anticipate needing to defend
against a properly-powered *new* sample landing lower than small prior
fragments.

**Honest verdict, overriding the letter of the pre-committed rule where
scrutiny disagrees with it:** this is a **fragile pass, not an established
finding.** Graded down from "ready for the audit-gate queue" to
**suggestive, same tier as before CT-F14 ran** — because the properly-powered
component alone does not clear significance, and that component is the one
this whole exercise existed to produce. Recording this as a **gap in the
pre-registration itself**, not smoothed over: the interpretation thresholds
were written for the pooled figure without a contingency for "the new,
larger sample disagrees with the small prior fragments" — a real design
blind spot, named so a future pre-registration in this arc specifies it.

**A second, unplanned walk-back, in the same direction.** §5.10 flagged a
striking pattern — 4 of 5 trackability exclusions in the 1980–1995 sample
clustered at 26–33°N/67–134°E in June–September. This sample's 21
exclusions show the **same band catching only 2 of 21** — a much weaker
signal at 4× the exclusion count. The earlier 4/5 was very likely a small-n
inflation of a real-but-modest effect, not a strong reproducible pattern.
CT-F15 (the proposed geo-fence) is downgraded accordingly — worth a light
touch, not a structural fix.

**What remains solid, restated:** the structural claim (wn1 dominance,
explanatory power) is untouched by any of this — CT-F14's own qualifying
subset has median wn1_frac 0.60, median R² 0.90, consistent with N3/N4.
Only the *directional/predictive* claim is affected, and it moves from
"borderline" (§5.10) to **"borderline, and the properly-powered test that
was supposed to settle it did not."**

### 5.4 Where that leaves the three candidates

> **Read this table as a within-storm-1/2 candidate ranking for the OFFSET
> MECHANISM.** §5.9 (CT-N) found the sign of the offset itself does not
> reliably generalize past those two storms — so the table below explains a
> phenomenon whose *universality* is now in question, not a confirmed
> atmospheric constant. Both findings stand together: storms 1–2's offset is
> real and its mechanism is best explained by candidate 1; whether *most*
> storms have a comparably-signed offset at all is open.

*(Updated after F4/F7 — the F1–F3 column is kept so the movement is visible.)*

| candidate | after F1–F3 | after F4 + F7 |
|---|---|---|
| 1 — steering-level / baroclinic tilt | Favoured | **Leading, and now near-unopposed** `[H]`. Unchanged evidence (92–102° monotone height ladder crossing zero mid-troposphere on both storms), but its two rivals have shrunk. |
| 2 — Ekman surface friction | Bounded contributor | **Re-scoped and demoted** `[H]`. The bound tightened (ocean 13–20°, land 34°) but the mechanism was partly mis-specified: friction rotates the *wind*, not the *pressure dipole*. The unplanned 1 %-vs-46 %-land pairing (§5.7) points the wrong way for it. |
| 3 — center-finder bias | LIVE, not excluded | **Bounded at ≈±5°** `[G]`, ~⅛ of the offset. Independent center definitions agree to 20 / 73 km and the error spans 2.3° / 6.5°. |

Candidates remain **not mutually exclusive**. The residual budget is now
roughly: tilt ≈ most of the 40°, centering ≈ ±5°, friction ≈ second-order on
this quantity. No stronger attribution is claimed at n = 2.

### 5.5 Next falsifiers

- ~~**CT-F4** (sub-grid center, was blocking)~~ — **RUN, §5.6.** Storm 2
  PASSES; storm 1 NO-VERDICT by the anti-vacuity guard, bounded ≤ 6.5° via the
  F4b curve. The apparatus is not what makes storms 1–2's own offset
  unreliable.
- ~~**CT-F5**~~ — **RUN, §5.8.** Walking-center fix; storm 2's saturation
  defect corrected, F5a passes cleanly.
- ~~**CT-F7** (friction over land)~~ — **RUN, §5.7.** Both bars pass; the
  candidate it tests is re-scoped rather than confirmed.
- ~~**CT-F8** (wind-center generalization)~~ — **RUN, §5.9 (as CT-N5).**
  Does **not** generalize: 6/10 = 0.60 vs the 0.70 bar. Storm 2's own
  improvement (§5.6) was a single case, not a pattern.
- ~~**CT-N** (n=10 sample)~~ — **RUN, §5.9.** The headline result: sign
  consistency of the offset itself is **6/10 = 0.60**, statistically
  indistinguishable from chance. Magnitude/explanatory-power claims (E1, E4)
  replicate; the signed claim (E3) does not, at this n.
- **CT-F6 (still open):** the crossing level (§5.2/5.8) as the observable —
  *prediction:* it tracks the deep-layer mean steering level. Needs the same
  n ≥ 10-with-regime-control treatment CT-N just showed is necessary.
- **CT-F9 (Ekman-pumping mechanism)** — **RUN, §5.9.** corr = −0.295, does
  not support the residual-correlation pathway for candidate 2.
- ~~**CT-F10** (displacement-filtered rerun)~~ — **RUN, §5.10.** NO-VERDICT
  standalone (n=3, below the pre-registered minimum) — but pooled with
  sample 1's own displacement-filtered subset, 6/7 = 0.857, the strongest
  number in the chain (still small-n, still not pre-registered as a pooled
  test).
- ~~**CT-F11** (regime-filtered rerun)~~ — **RUN, §5.10.** Passes on sample 2
  (8/9), but **checked directly against sample 1 and found NOT to rescue
  it** — retroactively applied, it drops sample 1 to exactly 0.500. The
  regime-contamination explanation is weaker evidence than it looked after
  §5.9 alone.
- **CT-F13 (raw replication, RUN as part of §5.10):** 8/10 on an independent
  sample, reversing §5.9's verdict on its own bar — but not significant on
  proper statistics (two-sided p=0.109), and the pooled combined figure with
  sample 1 (14/20) sits right at the noise floor.
- **CT-F14 (new, from §5.10, NOT yet run):** the correctly-scoped next step —
  a single, properly powered (n ≥ 25–30) **displacement-filtered-only**
  pre-registered sample, since displacement is now the mechanism-motivated
  filter with the strongest (if still small-n) support.
- **CT-F15 (new, from §5.10, NOT yet run):** geo-fence blind selection away
  from 26–33°N/67–134°E in June–September (4 of 5 exclusions in the second
  sample clustered there) and check whether this simply reproduces what
  CT-E2's trackability gate is already doing for free.
- **Sample size:** the apparatus (CT-F4/F5) is good enough that an offset
  constant COULD be fitted for storms in the large-displacement regime — but
  §5.10 means even that regime is only *suggestively* supported (n=7, p≈0.06
  pooled) and CT-F14 is the gate before any constant is fitted, not general
  n ≥ 10 in the abstract.

---

## 6. Product / encoding consequence `[S]`

> **⚠ Read with §5.9–5.11 AND the compression correction in §1.** The figures
> below say 93–97 %; the honest number for the 14-value model they describe is
> **90.9–94.3 %** (the 93–97 % belongs to a 36-parameter per-ring fit). The
> compression is real and generalized cleanly across every sample this arc ran
> (N3/N4, and CT-F14's own qualifying subset: median wn1_frac 0.60, R² 0.90). The
> *motion-encoding* half ("the dipole encodes the motion") depends on the
> signed relationship CT-F14 — the properly-powered test built to settle it —
> did NOT independently establish (§5.11). Treat the compression as ready
> for the audit-gate queue; treat the predictor as **suggestive at best**,
> not gated-and-pending but **not yet earned**.

If CT-F1..F3 hold up, the compact representation of a surface low is:

```
storm ≈ CENTER (place)                    — 1 address
      + p̄(r)   ring-profile means         — ~12 bytes (12 × 100 km rings, u8-quantizable per voxel-chess probe: u8 max dev 0.0047)
      + (a₁,b₁) ONE dipole vector          — 2 values (amplitude slope + bearing)
      = 90.9–94.3 % of in-disk MSLP variance   [corrected 2026-08-11; the
        93–97 % previously printed here was a 36-parameter per-ring fit, not
        this 14-value model — see the §1 correction]
```

- Maps directly onto `highheelbgz`'s 3-integer spiral **address** form
  (start, stride, length) + a short payload — the storm is stored as a place
  plus ~14 bytes, values recomputed on demand, instead of a raster crop.
- **The dipole encodes the motion.** `bearing(low pole) − 90° (− offset)`
  estimated the 6h displacement *direction* within tolerance on both storms
  **from a single timestep**. Named next falsifier: the single-frame motion
  predictor, evaluated on n ≥ 10 storms against the observed 6h track, with
  the offset constant fitted on a disjoint training set. Not executed —
  awaiting go, and gated behind the adversarial audit per the arc's standing
  rule (exploratory probes are not EVs; the 0/11 lesson).

## 7. Limitations and non-claims

- **n=2, one synoptic time, one season, NH only.** No claim of climatological
  generality; SH sign flip untested.
- **MSLP only.** No upper-air, no moisture, no intensity change — this is a
  *structure* result, not a forecast skill result.
- **Planar geometry** within 1200 km (`cos(lat)` metric); fine at 55–67°N for
  this radius, degrades toward the pole.
- **The 6h displacement is the label, not a forecast** — CT-E3 tests
  consistency between two simultaneous measurements (dipole at t0, motion
  t0→t1); the *predictor* framing is future work (§6).
- The ±45° bar was set to admit the expected baroclinic rotation. That was
  the right call for a first signed test, but it means the current result
  cannot distinguish candidates 1–3 in §5 — that is exactly what CT-F1..F3
  are for.

## 8. Status and promotion path

1. ~~Probe, pre-registered, run, committed~~ — DONE (`db57aac0`).
2. ~~This report~~ — DONE (`d9a98b86`).
3. ~~CT-F3 first (apparatus before mechanism), then CT-F1/CT-F2~~ — DONE.
   **F3 FAILED its gate**; CT-E3 re-graded in §4; F1 strongly favours the
   steering-level mechanism; F2 bounds friction to ~⅓ of the offset.
4. ~~**CT-F4 (sub-grid center) is the blocking item**~~ — **RUN and CLEARED**
   (§5.6). Apparatus uncertainty ≈ ±3–7°; an offset constant is fittable *for
   storms in the regime storms 1–2 sit in*. **CT-F5** (§5.8) fixed the
   remaining saturation defect. **CT-F7** (§5.7) bounded and re-scoped
   candidate 2 (friction rotates wind, not the pressure dipole).
5. ~~**n ≥ 10 storm sample**~~ — **RUN, §5.9.** The offset's **sign** does not
   generalize on that first blind sample (6/10, indistinguishable from
   chance), while its magnitude/explanatory-power claims do. CT-F8's
   wind-center generalization and CT-F9's Ekman-pumping mechanism also
   tested here: neither holds up at scale.
6. ~~**CT-F10 / CT-F11 / CT-F13**~~ — **RUN, §5.10. A second independent
   sample, borderline in a specific direction.** A fresh 1980–1995 sample
   scores 8/10 unfiltered — but proper statistics keep the combined figure
   right at the noise floor (14/20, one-sided p≈0.058), and the
   regime-filter explanation is checked and found NOT to rescue sample 1
   (drops it to exactly chance). What *does* strengthen is the
   displacement-filtered pooled subset (6/7, p≈0.0625) — apparatus, not
   regime, is now the better-supported explanation for the gap.
7. ~~**CT-F14**~~ — **RUN, §5.11. The single properly-powered test does NOT
   independently support the claim** (n=19, one short of its own n≥20 floor;
   0.684, p=0.0835). The pooled 3-sample figure technically crosses the
   pre-committed <0.05 threshold (p=0.0145), but the honest, scrutinized
   verdict grades this down to **still suggestive** — the pre-registration's
   own pooling rule had a gap, named rather than exploited.
8. Adversarial audit gate (plan §8) before any of it is promoted to EV /
   product claim — moot for the directional claim until it clears its own
   properly-powered test; live for the structural (compression) claim now.

**Net effect of the full follow-up chain on the headline claim, stated
plainly after four probes and 41 total storms across three independent
samples.** The structural claim — wn-1 dominance, the ring-profile + dipole
compression, R² lift — **generalizes cleanly and has not been shaken once**
across every sample this arc ran. The directional/predictive claim — the
signed left-of-motion orientation that made storms 1–2 exciting — has now
been tested four separate ways (apparatus §5.6, land-fraction mechanism
§5.7, blind n=10 §5.9, a reversal on independent n=10 §5.10, a
properly-powered n=19 §5.11) and **still does not clear a real bar on its
own strongest test.** The offset's status across the chain: **dead (F3) →
alive with a ±3–7° error bar (F4) → apparently general (misread of an n=10
in isolation) → reversed on a second n=10 → pooled-and-technically-passing
but not independently supported by the one test built to settle it (F14).**
That is not noise in the writing — it is the honest trajectory of a
borderline effect being measured with increasing rigor, and every step was a
genuine gate: a result that *helped* the claim (§5.10's reversal, CT-F14's
pooled crossing) got exactly the scrutiny a result that hurt it would have
gotten, and in both cases the scrutiny found reasons for caution that a less
careful pass would have missed. **Current position: the compression is
ready for the audit-gate queue now; the predictor is not, and should not be
represented as more than "suggestive" until a properly-powered test clears
its own bar without pooling assistance.** §6 is marked accordingly.

## 8b. External review of PR #926 — what it changed (2026-08-11)

16 findings (14 CodeRabbit + 2 Codex). Four changed measured numbers; the rest
were latent bugs or labelling. Recorded because two of them make the arc's own
results BETTER and one makes the headline WORSE — the review is not a formality.

**Changed published numbers:**

1. **The compression claim was measured on the wrong model** (Codex P1) — see
   the §1 correction. 93–97 % → **90.9–94.3 %** for the 14-value model actually
   claimed. The most consequential finding in the review.
2. **Sunflower E2 was not a controlled comparison** (Codex P2 + CodeRabbit):
   `grid_pts(n)` returned every in-disk lattice point, so the grid arm ran on
   80 samples against the spiral's 64 (293 vs 256, 1085 vs 1024) — and
   nearest-neighbour reconstruction improves with samples, so the arm being
   compared was systematically advantaged. With EXACTLY n enforced the verdict
   **improves in the spiral's favour**: 234.5 vs 269.0 Pa (n=64), 119.1 vs
   123.3 (n=256), 58.9 vs 59.9 (n=1024) — the spiral now wins at every budget,
   where the earlier write-up recorded "parity". The original result was
   PESSIMISTIC, not optimistic.
3. **The voxel-chess palette arm was a hybrid** (CodeRabbit): `geo_corr`
   received the palette-derived geostrophic winds but closed over the
   module-level RAW `u`/`v`, so "u8 max dev 0.0047" compared palette
   geostrophy against raw observations — not the pre-registered palette
   result. The observed fields are now explicit parameters and the palette arm
   passes `u8`/`v8`.
4. **go_territory's explained variance re-centred the residual** (CodeRabbit):
   `res.var()` subtracts the residual mean after every atom, excluding it from
   the error. Fixed to a fixed centered-field denominator over the residual
   mean-square; K=10 matched moves 0.530 → **0.523**, and **no verdict flips**
   (A-E1 and A-E2 still fail their bars).

**Vacuous assertion caught** (CodeRabbit): E6's `rises_then_decays` required
only an interior maximum plus a lower final value — it accepted a profile that
DECREASED before rising to the peak, which the committed run literally did
(12.190 → 12.163 m/s before the 525 km peak) while reporting `true`. Now
asserts monotone rise to the peak and decay after it, with a stated 0.05 m/s
tolerance; the run still passes, but now because the profile is Rankine-shaped
rather than because the test could not fail.

**Latent bugs fixed (no committed run hit them, so no numbers move):**
`find_center` returned grid cell (0,0) when a `near`-limited mask was fully
masked, instead of `None`; `subgrid_min`'s 3×3 slice did not wrap in longitude
and would have raised on any centre at the 0° seam. Both are now guarded in
all six probes.

**Labelling / provenance:** CT-F12 can no longer emit `pass: true` below its
evaluable minimum; F7d's threshold now matches the 40 deg its own key and
docstring pre-register (it tested 35); `comet_tail_followup.json` persists the
per-storm centre / bearing / displacement instead of `"storms": {}`;
`go_territory_probe.json` is written beside the probe rather than the cwd.

## 9. Reframe — the spine is found; the moderators are missing (operator, 2026-08-11)

Operator ruling on how to read the whole chain, and it is quantitatively
better than my "borderline" framing:

> *"Wir haben ein Spine gefunden — die Stellschrauben müssen noch mit den
> Variablen der bekannten Modelle moduliert werden. Uns fehlen die
> Moderatoren; aber wir haben bereits das Gerüst, um das Zentrum und die
> Dynamik zu modellieren."*

**Why this framing is not spin — it is the statistically correct reading of
the residual.** A directional main effect at 0.68–0.73 sign consistency
whose residual were *random* would be a dying claim. This chain's residual
is not random: it runs **monotonically with a measured variable** — the
height ladder (§5.2/5.8), ≈ −40° at 1000 hPa climbing smoothly through zero
in the mid-troposphere, spread 92–102°, 3–5× the measured apparatus noise,
on both storms it was measured on. *Main effect + structured residual +
identified covariate* is the signature of a **missing moderator**, not of a
null. A null does not produce a ladder. `[H]` at the ladder's n=2; `[G]`
that the framing follows if the ladder replicates.

### 9.1 What is established (the spine) `[G]`

**Center (place) + ring profile (~12 values) + one wn-1 dipole (2 values)
= 93–97 % of in-disk MSLP variance** — replicated across three independent
samples spanning 1980–2021, 41+ storms, four seasons, never shaken once
(N3/N4; §5.11's own subset: median wn1_frac 0.60, R² 0.90). This is a
skeleton that models the **center and the first asymmetry mode of the
dynamics** in ~14 bytes plus an address — which is, as the operator notes,
already more explicit structure than a learned model exposes.

### 9.2 The DRY moderators — measured in this chain, not yet wired `[H]`

| moderator | measured evidence | wiring |
|---|---|---|
| **Steering level** (baroclinic tilt) | the 92–102° monotone height ladder, zero-crossing 400–650 hPa (§5.2/5.8) | score the dipole against the *steering-level* motion (500–700 hPa flow) instead of the 6h surface displacement — the single most promising fix, **CT-F16** |
| **Displacement magnitude** (label noise) | 6/7 pooled at ≥250 km vs 14/20 unfiltered; CT-F14 0.684 | model the motion-bearing *uncertainty* explicitly instead of a hard cutoff |
| **Surface type / friction** | +14° ocean vs +34° land inflow, paired within one disk (§5.7) | a wind-level correction; second-order on the pressure dipole |
| **Latitude / f, regime** | the low-wn1 July cases; the 75°N outlier | intake covariates, already computed per storm |

### 9.3 The MOIST sector — not modeled at all (operator, same ruling) `[S]`

> *"Außerdem haben wir Feuchtigkeit und Abregnen im Aufwind an der Kollision
> zwischen den Gebieten nicht modelliert — das ist eine Art Entropie bei
> Verdunstung und Abregnen."*

Correct, and the "entropy" word is the *technically* right one, not a
metaphor. Everything in this chain is **dry, adiabatic, balanced dynamics**.
The missing half is diabatic: moisture converges into the collision zone
between air masses (the front), rises, condenses — releasing latent heat
that deepens the low — and **rains out irreversibly**: the water leaves the
column, the heat stays. That one-way flow is moist **entropy production**,
and treating the storm as a heat engine bounded by it is established
literature (Emanuel's potential-intensity Carnot frame; Pauluis' moist
entropy budgets). The state variable is equivalent potential temperature θe;
the sink is precipitation.

Three things make this *tractable on this substrate, now*, rather than
aspirational:

1. **The store has the variables** (verified in the `.zmetadata` earlier
   this arc): `specific_humidity` (13 levels), `temperature` (13 levels) —
   together θe; `total_column_water_vapour`; `total_precipitation_6hr`;
   `vertical_velocity` (13 levels — the updraft itself).
2. **θe and TCWV are scalar fields** — the *same* ring/wn-1 decomposition
   applies verbatim. The moisture spine costs nothing new.
3. **A diabatic-dominance moderator falls out for free:** the storms where
   the dry spine's prediction failed worst (the July cases, wn1_frac
   0.19–0.36) are plausibly the diabatically-driven ones. Precip-per-disk /
   TCWV-dipole-strength is a computable gate variable at intake.

Named falsifiers, NOT run, `[S]` until probed: **CT-M1** — the TCWV/θe wn-1
dipole leads the pressure dipole in bearing (moisture converges *ahead* of
the low, ≈90° from the left-of-motion low pole); **CT-M2** — 6h disk
precipitation is predicted by TCWV × mid-level ascent (`vertical_velocity`
at 700/500 hPa) — the rain-out entropy sink as a budget check; **CT-M3** —
adding the diabatic-dominance gate as a moderator cleans the directional
claim's residual where the displacement filter alone did not.

### 9.4 The brutal step — learn the moderator matrix on the substrate's own machinery `[S]`

Operator: *"du könntest sogar brutal sein und domino.rs / LSTM modellieren."*
The shapes already exist and are proven:

- **The spine is a board state.** Per storm and timestep: ~16 spine values
  (center, profile, dipole) + the moderator covariates (steering vector, f,
  surface fraction, diabatic gate). A moderator set IS a weight matrix `W`;
  `domino.rs`'s symbiont step (`C = A·W`, 16-board AMX/int8 tile-GEMM with
  requantize feedback) executes exactly this — and the stencil-as-GEMM
  path is already **byte-proven on real WB2 data** in ndarray's
  `examples/geostrophic_stencil.rs` (4/4 pre-registered bars, corr 0.9985).
- **The recurrence is an LSTM-shaped problem.** Successive 6h spine states
  are a short sequence; the workspace already carries byte-parity-proven
  int8 LSTM machinery (`tesseract-recognizer`, `E-OCR-LSTM-1`) consuming
  the same `ndarray` tile-GEMM.
- **The hybrid is the honest architecture:** explicit physics as the spine
  (this report), learned weights as the moderators — the NeuralGCM-shaped
  split, on a 512-byte-per-storm substrate encoding, with the training
  discipline this arc has already built (pre-registration, held-out decades,
  the audit gate).

Gate, unchanged: train/test on disjoint decades, pre-registered bars,
adversarial audit (plan §8) before any of it is called more than a probe.
