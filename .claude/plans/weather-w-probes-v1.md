# weather-w-probes-v1 — the W series as self-contained Sonnet worker briefs

> **Status:** ACTIVE for exploratory probes (W5, W2s-a, W6, W2s-b, W7).
> **CT-F17 is verdict-tier and MUST NOT run** until its spec passes an
> independent adversarial audit (the 0-of-11 lesson,
> `E-ZERO-FOR-ELEVEN-THE-AUTHOR-CANNOT-AUDIT-HIS-OWN-FALSIFIERS-1`: the
> author is structurally the wrong person to find his spec's vacuous pass
> routes). All other bars here are author-written and unaudited — fine for
> exploratory tier, said out loud.
>
> **Why this file exists:** token economy + stranded rescue (operator,
> 2026-08-12). A worker receives §0 + ONE brief pasted verbatim — never this
> repo's history, never `COMET_TAIL_REPORT.md`. Product-lead context lives in
> the report §10; this file is executable instructions only.
>
> **Provenance of the physics:** report §10.2–§10.5 (dipole vector-sum model,
> Windkanal two-regime α, sunflower/collision facets, spiral-ADI).

---

## §0 SHARED WORKER PREAMBLE — paste this block verbatim into every worker prompt

You are a Sonnet grindworker executing ONE pre-specified measurement probe.
Bounded input, known output shape, no synthesis, no scope growth.

**SCOPE — iron rules:**
1. You create/modify ONLY: your probe file `probes/weather-p1/<name>.py`, its
   outputs `<name>.json` + `<name>.partial.jsonl`, and your tag-file
   `probes/weather-p1/exec-runs/<name>.txt`. NOTHING else. Never touch
   `.claude/board/*`, `COMET_TAIL_REPORT.md`, or another probe.
2. No `cargo` anything. No Rust. Python 3 + numpy + numcodecs only.
3. Commit the probe file with its bars BEFORE executing it (commit message:
   `probes/weather-p1: <ID> pre-registered BEFORE the run`). Then run. Then
   commit results. NEVER adjust a bar after seeing output; a failed bar is
   reported as FAIL with the same prominence as a pass.
4. Do not claim anything you did not run. `NO-VERDICT` ≠ `FAIL`. If a fetch
   or step dies, record it in the tag-file and stop — do not improvise.
5. Every module and every `def` gets a docstring (coverage gate). Docstrings
   state WHAT is measured and WHY the bar can fail (no vacuous assertions).

**DATA ACCESS (copy verbatim):**
```python
import json, pathlib, urllib.request
import numcodecs, numpy as np

B = ("https://storage.googleapis.com/weatherbench2/datasets/era5/"
     "1959-2022-6h-1440x721.zarr")
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
meta = json.loads(op.open(B + "/.zmetadata", timeout=90).read())["metadata"]

def fetch(var, key):
    """Fetch and decode one zarr chunk from the WB2 store."""
    za = meta[f"{var}/.zarray"]
    raw = op.open(f"{B}/{var}/{key}", timeout=900).read()
    dec = numcodecs.get_codec(za["compressor"]).decode(raw)
    return np.frombuffer(dec, dtype=np.dtype(za["dtype"])).reshape(za["chunks"])
```
Chunk keys: surface vars `f"{t}.0.0"`; 13-level vars (`u_component_of_wind`,
`v_component_of_wind`, `geopotential`, …) `f"{t}.0.0.0"` — ALL 13 levels
arrive in one ~42 MB chunk. Levels: `[50,100,150,200,250,300,400,500,600,
700,850,925,1000]` hPa. Grid: 721×1440, 0.25°, lat from `fetch("latitude",
"0")`. Anchor guard (include verbatim, run before any fetch):
```python
import datetime
EPOCH = datetime.datetime(1959, 1, 1)
def t_index(dt):
    """WB2 time index: 6-hourly steps since 1959-01-01."""
    return int(round((dt - EPOCH).total_seconds() / 3600 / 6))
assert t_index(datetime.datetime(2021, 6, 15, 12)) == 91246
```
Store coverage ENDS 2021-12-31 18Z despite the filename; guard any t against
`meta["mean_sea_level_pressure/.zarray"]["shape"][0] - 1`.

**GEOMETRY + SPINE (copy verbatim where needed):** `wrap_deg(d) = (d+180.0) %
360.0 - 180.0` (range `[-180,180)`); `err_deg(lp, ref) =
wrap_deg(rad2deg(lp - (ref + π/2)))`; the disk/ring decomposition is copied
verbatim from `probes/weather-p1/comet_tail_f16.py` (`geom_ll`/`geom`,
`low_pole_bearing`/`spine`) — R_E=6371.0, R_DISK=1200.0, RING=100.0.

**STATISTICS STANDARD (report §10.1 — binding):**
```python
def circular(errs_deg, n):
    """Resultant length R_bar, mean direction mu (deg), Rayleigh p (Zar/Mardia)."""
    th = np.deg2rad(np.asarray(errs_deg))
    c, s = np.cos(th).mean(), np.sin(th).mean()
    r = float(np.hypot(c, s)); mu = float(np.rad2deg(np.arctan2(s, c)))
    z = n * r * r
    p = float(np.exp(-z) * (1 + (2*z - z*z)/(4*n)
              - (24*z - 132*z**2 + 76*z**3 - 9*z**4)/(288*n*n)))
    return r, mu, max(min(p, 1.0), 0.0)
```
Sign fractions may be PRINTED as descriptive context, NEVER used as a
pass/fail bar. Every bearing-related bar ships with TWO controls through the
identical pipeline: a +90°-rotated referent and a deterministically permuted
one (`(i+7) % n`). The controls' scores are reported FIRST; if a control
matches the real referent's score, the probe is VOID regardless of its bars.
Comparative sentences in your output must name BOTH operands ("R̄ identical
to X", never bare "identical") — a relation between two correct numbers can
be false.

**OUTPUT + STRANDED RESCUE:**
- Final JSON beside the script:
  `with open(pathlib.Path(__file__).with_name("<name>.json"), "w") as fh: json.dump(out, fh, indent=2)`
- **Checkpointing (mandatory for any probe with per-unit fetches):** after
  each unit of work (storm / node batch), append ONE line to
  `<name>.partial.jsonl` (`json.dumps(row)` + newline, flush). On startup,
  read the partial file if it exists and SKIP completed keys (`t0` or unit
  id). All RNG via `np.random.default_rng(<fixed seed stated in the brief>)`.
  A stranded run is then rescued by re-invoking the same script — it resumes,
  never re-fetches finished units.
- Tag-file `exec-runs/<name>.txt`: append start line, one line per ~10 units
  (heartbeat), and a final line with every bar verdict. This is YOUR only
  log; the orchestrator consolidates boards.

---

## §1 BRIEF W5 — spiral-ADI: two Fibonacci-stride sweeps ≈ one 2D diffusion? (Sonnet, zero fetch)

**File:** `spiral_adi_probe.py`. **Seed:** 20260812. **No network.**

**Objective.** On a Vogel lattice (`r = c·√k`, `θ = k·2π(1−1/φ)`, N=4096,
c chosen so max radius = 1.0), test whether alternating tridiagonal smoothing
sweeps along the two parastichy stride families approximate an isotropic 2D
diffusion — and that the result *depends on the strides being Fibonacci*.

**Steps.**
1. Build the lattice. For each radius band (8 equal-area annuli), find the
   dominant nearest-neighbor index-difference pair by measuring, for each
   point, `argmin_j |x_{k+j} − x_k|` over `j ∈ {1..60}`; record the two most
   frequent j per band (expect adjacent Fibonacci numbers; REPORT the bands
   where they transition).
2. Per band, measure the crossing angle between the two stride directions at
   each point (angle between `x_{k+j1}−x_k` and `x_{k+j2}−x_k`); report the
   distribution (median, IQR, per band).
3. Sweep operator: for stride j, order points into chains `(start, j)`
   within a band; one sweep = `y_i ← 0.25·y_{prev} + 0.5·y_i + 0.25·y_{next}`
   along each chain (open ends: hold). One ADI iteration = sweep family A
   then family B, band-appropriate strides.
4. Test field: Gaussian bump `exp(−|x−x0|²/2σ²)`, σ=0.08, x0 at radius 0.45.
   Run 8 ADI iterations. Reference: sample the analytic heat-kernel-blurred
   Gaussian (σ_ref² = σ² + 8·s² where s = the measured mean neighbor spacing
   × 0.5 — CALIBRATE σ_ref by least-squares over σ_ref, then judge SHAPE) at
   the lattice points.
5. Anisotropy metric: fit the blurred bump's second-moment tensor; ratio of
   eigenvalues λ_max/λ_min.

**Bars (pre-registered, commit before run):**
- **B1** (descriptive, no pass/fail): stride pairs per band + transition map
  + crossing-angle table.
- **B2 ISO:** after best-σ_ref calibration, relative L2 error between ADI
  result and isotropic reference ≤ **0.15**, AND second-moment anisotropy
  λ_max/λ_min ≤ **1.25**.
- **B3 CONTROL (can-it-fail):** identical run with strides forced to
  **12 and 18** (non-Fibonacci, non-coprime-ish) must give anisotropy ≥
  **1.5×** the Fibonacci run's. If the wrong strides smooth just as
  isotropically, the Fibonacci claim measures nothing — say VOID.

**Output JSON:** `{bands: [...], crossing_angles: {...}, iso_error, aniso_fib,
aniso_control, verdicts: {B2, B3}}`.

---

## §2 BRIEF W2s-a — golden two-lattice pairing on REAL lat/lon geometry (Sonnet, zero fetch)

**File:** `sunflower_pairing_probe.py`. **Seed:** 20260812. **No network.**

**Objective.** The collision-node construction assumes the two sunflower
lattices pair generically (no ties, even pair distances). Prove it survives
the real cos-lat metric — the #921 lesson says disk properties do NOT
automatically transfer.

**Steps.**
1. Two Vogel lattices, N=2048 each, disk radius 1500 km, centers at
   (55.0N, 340.0E) and (55.0N, ~361.9E) → 1400 km apart at that latitude.
   Project to km via the metric `dx = R_E·cos(lat_c)·Δlon_rad`,
   `dy = R_E·Δlat_rad` (R_E=6371.0) — the same `geom_ll` convention.
2. Overlap band: points of lattice H within 900 km of center T and vice
   versa. Nearest-pair map: for each H-point in band, its nearest T-point.
3. Control: TWO axis-aligned square grids of identical point density over
   the same two disks, same metric, same pairing procedure.

**Bars:**
- **G1 TIES:** count of exact-duplicate nearest-pair distances (float64
  equality after rounding to 1e-9 km): golden = **0**; grid control **> 0**
  (if the grid also has zero ties, the tie test is vacuous on this geometry —
  report VOID for G1 and rely on G2).
- **G2 EVENNESS:** coefficient of variation of nearest-pair distances:
  golden CV **< grid CV** (strict).
- **G3** (descriptive): χ² of pair-midpoint density against uniform across
  the corridor band, both constructions, reported not judged.

**Output JSON:** `{n_pairs, ties_golden, ties_grid, cv_golden, cv_grid,
chi2_golden, chi2_grid, verdicts: {G1, G2}}`.

---

## §3 BRIEF W6 — the dipole deconvolution: neighbor + bow, global fit (Sonnet, ~40 chunks)

**File:** `comet_tail_w6.py`. **Seed:** 20260812.
**Inputs:** `comet_tail_f16.json` rows (19 storms; fields per row: `date`,
`t0`, `center_lat`, `center_lon`, `displacement_km`, `err_surface_deg`,
`low_pole_rad`, `steer_bearing_rad`, `steer_speed_ms`).

**Objective.** Test the report-§10.2 model: the dipole vector is a GLOBAL
linear combination of a neighbor-far-field predictor and a bow-wave
predictor. This is a mechanistic test on stored storms — NOT a verdict
(that is CT-F17).

**Per storm (checkpoint one JSONL row each):**
1. Fetch MSLP `f"{t0}.0.0"`. Recompute the CONSTRAINED spine dipole vector
   `D_i = (a1, b1)` via the verbatim `spine()` code from
   `comet_tail_f16.py` (ring means + lstsq on `[r·cosθ, r·sinθ]`) about the
   stored center.
2. **Motion bearing recovered by ALGEBRA, no tracking:**
   `mth_deg = wrap_deg(rad2deg(low_pole_rad) − 90 − err_surface_deg)`.
   (Exact inversion of `err = wrap(lp − (mth+90))`.) Motion speed =
   `displacement_km` per 6 h → m/s.
3. Fetch u/v `f"{t0}.0.0.0"`; disk-mean 850 hPa wind vector `v_env850`
   (1200 km disk). **`v_rel = v_storm − v_env850`** (vector, m/s). Bow
   predictor: `P_bow,i = (0.5·1.2·|v_rel|²) · û(bearing(v_rel) + 180°)`
   [Pa · unit-vector — low pole BEHIND relative motion].
4. Neighbor: zonal-anomaly field `p − mean(p, axis=1)`; within the annulus
   **600–2500 km** of the center and lat 20–80N, find the strongest POSITIVE
   anomaly cell → `A_H` [Pa], distance `d_H` [km], bearing `θ_H`. Neighbor
   predictor: `P_geo,i = (A_H/d_H) · û(θ_H + 180°)` [Pa/km · unit-vector —
   background low pole points AWAY from the H].
5. Store row: `t0, D, P_geo, P_bow, |v_storm|, |v_rel|, A_H, d_H`.

**Global fit:** stack 19×2 = 38 scalar equations
`D = c_geo·P_geo + c_bow·P_bow`; solve `(c_geo, c_bow)` by lstsq (2 free
parameters — units absorb into the c's; state this). Define
`R²_vec(model) = 1 − Σ|D_i − D̂_i|² / Σ|D_i − mean(D)|²`. Fit also the two
single-predictor models.

**Bars (pre-registered):**
- **B0 CONTROLS FIRST:** joint fit with per-storm PERMUTED `P_bow`
  (`(i+7) % 19`) and, separately, with `P_bow` rotated +90°. Either control's
  joint `R²_vec` must stay ≤ single-geo `R²_vec` + **0.03**; otherwise the
  joint gain is degrees-of-freedom, probe VOID.
- **B1 IDENTIFIABILITY:** joint `R²_vec` ≥ best single `R²_vec` + **0.10**.
- **B2 SIGN:** `c_bow > 0` AND `c_geo > 0` (both components in their
  physically-predicted directions).
- **B3** (descriptive): resultant `(R̄, μ, p)` of residual bearings, overall
  AND stratified by `|v_storm|` < / ≥ 8 m/s — the **stranded stratum**: if
  §10.2's stranded-rescue reading is right, the weak-`|v_storm|` residuals
  should NOT be worse than the strong ones once `v_rel` carries the bow.
- **B4** (descriptive): per-storm table `bearing(D)` vs `bearing(D̂)`.

**Cost:** 19 MSLP (~2 MB ea) + 19 u/v chunks (~42 MB ea) ≈ 850 MB, ~3 min.
Checkpoint after every storm; resume skips completed `t0`.

---

## §4 BRIEF W2s-b — the α-field on a real H–T pair (GATED on W2s-a G2 pass)

**File:** `corridor_alpha_probe.py`. Outline — finalize bars at spawn time
using W2s-a's measured pairing stats:
- One timestep (T0=91246), fields: MSLP + 10m u/v. Deepest NH low +
  strongest H within 3000 km (zonal anomaly), per W6 §3.4's finder.
- Sunflower pair per W2s-a geometry; nodes in the corridor band.
- Per node: `|∇p|` (centred differences on the 0.25° grid, cos-lat metric,
  longitude WRAPS via np.roll — the #926 lesson), measured `|v|`,
  `f = 2Ω·sin(lat)`.
- **Core bar (fixed now):** regression of measured `|v|` on geostrophic
  `|∇p|/(ρf)` over open-corridor nodes (|lat| ∈ 30–75, ∇p above its median):
  slope β ∈ **[0.85, 1.15]** — the geostrophic regime must recover itself,
  else the α machinery measures nothing here. Permuted-node control R² <
  **0.1**. Descriptive: per-segment α map; overlay vs the temperature-
  gradient contested band (`go_territory` §-style gradT).

## §5 BRIEF W7 — Gegendruck (GATED on W2s-b)

Outline only: two consecutive timesteps; corridor lanes = aligned-family
spiral bands; per lane, along-lane mass-flux convergence at t0 vs Δp
(t1−t0) in lanes i±1. Bars to be written AFTER W2s-b fixes the lane
geometry; must include rotated-lane and time-shuffled controls.

## §6 CT-F17 — the fresh-sample VERDICT (GATED: W6 result + independent adversarial spec audit)

The only verdict-tier probe. Shape (parameters frozen NOW, bars audited
before run): mechanically generated dates in **1959–1979** (the only unused
window; samples 1–3 used 2015–21 / 1980–95 / 1996–2010), fixed start
1959-02-01 12Z + fixed stride 53 days, N_CANDIDATES=70 targeting ≥20
qualifying at the observed 0.28 rate; displacement ≥ 250 km/6h a priori;
score each storm's dipole bearing against the **W6-fitted model's predicted
bearing** (fallback branch if W6 fails B1: against surface motion, stated in
advance). Statistics per §10.1: V-test toward the training-fixed μ, p <
0.05, R̄ ≥ 0.35, rotated + permuted controls with floors. **The spec text
goes to an independent adversarial audit (5+3 or codex-style) BEFORE
execution — non-negotiable.**

## §7 Execution order + token budget

| wave | probes | parallel? | worker tokens (est.) |
|---|---|---|---|
| 1 | W5, W2s-a, W6 | yes — disjoint files, no shared state | ~15–25k each (brief + code + run logs) |
| 2 | W2s-b | after W2s-a | ~20k |
| 3 | W7 | after W2s-b | ~20k |
| 4 | F17 spec audit → run | after W6 | audit ~30k, run ~25k |

Orchestrator (Opus) does: spawn with §0+brief, consolidate tag-files, write
boards, interpret results, decide F17's branch. Workers never interpret.
