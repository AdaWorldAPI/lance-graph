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

**Pre-registered follow-up falsifiers (proposed, awaiting go — not run):**

- **CT-F1 (steering level):** repeat the full decomposition on
  `geopotential` at 500 hPa (available in the same WB2 store). *Prediction:*
  the alignment error shrinks toward 0° at the steering level (|error| ≤ 20°)
  while remaining ≈ −40° at MSLP. If the offset persists unchanged at 500 hPa,
  candidate 1 is dead.
- **CT-F2 (friction):** measure the actual cross-isobar inflow angle from
  `10m_u/v_component_of_wind` vs the MSLP isobars in the storm ring.
  *Prediction:* measured inflow angle ≪ 40° over open ocean — bounding how
  much of the offset friction can own.
- **CT-F3 (center robustness):** recompute the dipole with the center jittered
  ±100 km along and across track, and with a vorticity-centroid center.
  *Prediction:* low-pole bearing stable within ±10°; if it swings with the
  center choice, the −40° is apparatus, and §4's CT-E3 verdict must be
  re-graded (the arc's standing rule: a systematic number is a claim about
  the measurement apparatus until proven otherwise).
- **Sample size:** n ≥ 10 storms across seasons/basins before any offset
  constant is baked into a predictor. 2/2 at p=0.0625 justifies the follow-up,
  not a product constant.

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
2. **This report** — the documentation artifact.
3. CT-F3 first (apparatus before mechanism), then CT-F1/CT-F2 — on operator go.
4. n ≥ 10 storm sample; only then the single-frame motion predictor.
5. Adversarial audit gate (plan §8) before any of it is promoted to EV /
   product claim.
