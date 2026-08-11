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

### 5.4 Where that leaves the three candidates

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
  F4b curve. The blocking item is cleared.
- ~~**CT-F7** (friction over land)~~ — **RUN, §5.7.** Both bars pass; the
  candidate it tests is re-scoped rather than confirmed.
- **CT-F5 (still open):** widen F1's per-level center search (or track the
  upper center along the tilt axis) so the 600 km saturation defect in §5.2
  cannot recur; re-run storm 2's own-center path for a real verdict.
- **CT-F6 (still open):** the crossing level itself as the observable —
  *prediction:* it tracks the deep-layer mean steering level, i.e. deeper /
  more mature systems cross higher. Needs n ≥ 10.
- **CT-F8 (new, from §5.6):** is the **wind-based** circulation center
  systematically the better reference? On storm 2 it sat 65 km north of the
  pressure minimum and moved the error furthest toward zero. *Bar:* across
  n ≥ 10 storms, |error| at the ζ-centroid < |error| at the MSLP minimum.
- **CT-F9 (new, from §5.7):** does asymmetric friction rotate the **pressure**
  dipole via Ekman pumping — the mechanism candidate 2 should have been about?
  *Test:* correlate the dipole bearing residual against the land-fraction
  *asymmetry* across the disk (not the mean). *Prediction under candidate 2:*
  storms with a strong land/ocean split across the vortex show a larger
  residual than uniform-surface storms.
- **Sample size, unchanged:** n ≥ 10 storms across seasons and basins before
  any offset constant is baked into a predictor. The apparatus is now good
  enough to fit one — which makes the sample size, not the centering, the
  binding constraint.

---

## 6. Product / encoding consequence `[S]`

If CT-F1..F3 hold up, the compact representation of a surface low is:

```
storm ≈ CENTER (place)                    — 1 address
      + p̄(r)   ring-profile means         — ~12 bytes (12 × 100 km rings, u8-quantizable per voxel-chess probe: u8 max dev 0.0047)
      + (a₁,b₁) ONE dipole vector          — 2 values (amplitude slope + bearing)
      = 93–97 % of in-disk MSLP variance
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
   (§5.6). Apparatus uncertainty ≈ ±3–7°; an offset constant is now fittable.
   **CT-F7** (land friction, §5.7) run in parallel: both bars pass, candidate 2
   re-scoped.
5. **n ≥ 10 storm sample is now the binding constraint** — the apparatus no
   longer is. CT-F5 / CT-F6 / CT-F8 / CT-F9 alongside it.
6. Adversarial audit gate (plan §8) before any of it is promoted to EV /
   product claim.

**Net effect of the follow-ups on the headline claim.** The §1 summary is
unchanged in substance — wn-1 dominance (0.92/0.89), the R² lift to 0.97/0.93,
and left-of-motion on 2/2 all stand untouched, since none of them depends on
the offset. The offset itself went **dead (F3) → alive with a ±3–7° error bar
(F4)**, and that round-trip is the point: the constant is now defensible
*because* it survived a gate that had already killed it once, at a centering
precision that was measured rather than assumed. What remains ungated is
sample size, not apparatus — so §6's encoding and single-frame-predictor work
is unblocked in principle and still owes n ≥ 10 before any constant ships.
