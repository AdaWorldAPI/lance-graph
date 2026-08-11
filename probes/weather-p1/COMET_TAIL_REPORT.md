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

### 5.4 Where that leaves the three candidates

| candidate | status after F1–F3 |
|---|---|
| 1 — steering-level / baroclinic tilt | **Favoured** `[H]`. Monotone 92–102° height ladder crossing zero in the mid-troposphere on both storms, 3–5× the apparatus noise. |
| 2 — Ekman surface friction | **Bounded contributor** `[G]` on magnitude. 13–15° over ocean, right sign, ~⅓ of the offset at most. |
| 3 — center-finder bias | **LIVE and not excluded** `[G]`. ±100 km centering moves the answer up to 29.4° — enough to dominate the offset magnitude by itself. |

Candidates 1–3 are **not mutually exclusive**, and the measured numbers are
roughly additive in the right direction (tilt ≈ most of it, friction ≈ 13–15°,
centering ≈ ±15° of slop). No attribution is claimed beyond that.

### 5.5 Next falsifiers (NOT run)

- **CT-F4 (apparatus, first):** sub-grid center fit — parabolic interpolation
  of the pressure minimum plus a circulation-centroid at the level where the
  error crosses zero — then re-run F3's jitter test. *Bar:* spread ≤ 10°.
  Until this passes, no offset constant exists to be fitted.
- **CT-F5:** widen F1's center search per level (or track the upper center
  along the tilt axis) so the saturation defect in §5.2 cannot recur; re-run
  storm 2's own-center path for a real verdict.
- **CT-F6:** the crossing level itself as the observable — *prediction:* it
  correlates with the deep-layer mean steering level, i.e. deeper/more mature
  systems cross higher. Needs n ≥ 10.
- **Sample size, unchanged:** n ≥ 10 storms across seasons and basins before
  any offset constant is baked into a predictor. 2/2 at p = 0.0625 justified
  the follow-up; the follow-up now justifies fixing the apparatus, not
  shipping a constant.

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
4. **CT-F4 (sub-grid center) is now the blocking item** — the apparatus must
   pass before any offset constant can be fitted, and therefore before the
   single-frame motion predictor is worth running at all.
5. CT-F5 / CT-F6, then n ≥ 10 storm sample.
6. Adversarial audit gate (plan §8) before any of it is promoted to EV /
   product claim.

**Net effect of the follow-up on the headline claim.** The §1 summary is
unchanged in substance — wn-1 dominance (0.92/0.89), the R² lift to
0.97/0.93, and left-of-motion on 2/2 all stand untouched, since none of them
depends on the offset. What the follow-up removed is a number I would
otherwise have been tempted to ship: the "−40° constant" is **not
measurable at this centering precision**, and the encoding/predictor work in
§6 is gated behind fixing that, not behind more storms.
