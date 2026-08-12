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
**GOLDEN-RATIO INDEX FLOOR (operator-ruled 2026-08-12, binding on every
lattice/stride/spiral construction in this plan).** Golden-ratio structure is
usable only from convergent index ~17–21 upward; below that it is unusable
for these constructions. A Fibonacci ratio `F(n+1)/F(n)` is a
RATIONAL with period `F(n)`; it only behaves like the irrational φ from
**n ≈ 17–21** onward. Measured convergent error: n=10 → **1.5e-4**, n=13 →
8.2e-6, **n=17 → 1.8e-7**, n=19 → 2.6e-8, n=21 → **3.7e-9**. Below the floor
the ratio RESONATES — which is exactly the moiré/aliasing the golden
construction exists to prevent, so a sub-floor lattice does not weakly
demonstrate the property, it demonstrates the opposite.

Two things this rule does and does NOT touch, keep them apart:
- **The golden ANGLE** `θ = k·2π(1−1/φ)` in f64 is irrational to ~1e-16.
  **Unaffected.** Do not "fix" the angle.
- **The addressable STRIDE families** that EMERGE in a lattice of N points —
  the visible parastichy numbers sit at `≈ √N`. **This is what the floor
  binds.** For the emergent pair to reach index ≥17 you need
  **N ≳ F(17)² = 2 550 409**, not thousands. A probe at N=4096 has its
  natural pair at **F(10)=55 / F(11)=89** — six orders of magnitude below the
  floor, i.e. squarely in the sub-floor resonance zone, no matter how precise
  its angle.

**BELOW THE FLOOR, DO NOT APPROXIMATE φ AT ALL — enumerate.** (Operator,
same ruling: the preferred small-q combination is modulus 17 with stride 4;
the stride-11 alternative is explicitly NOT a golden-section step.) The floor
does not mean "reach
for a slightly better Fibonacci ratio"; it means the φ-approximation
*selector* is the wrong instrument down there. Use the workspace's shipped
exact integer coprime walk — `CurveRuler::index(k) = (start + 4k) mod 17`
(`crates/helix/src/curve_ruler.rs`), D-QUANTGATE-mandated, bit-exact, full
permutation, no float φ. **Measured proof that φ-proximity mis-selects at
q=17:** `helix/KNOWLEDGE.md:320` calls `(i·11)%17` the "golden-step" because
`17/φ = 10.51 → 11`; enumerating all 16 strides, prefix star-discrepancy at
m = 5/9/13 is **stride 4 → 0.2000/0.1111/0.0769** vs **stride 11 →
0.2000/0.1503/0.0905** — stride 4 is never worse and strictly better at m=9
and m=13. The ratio `17/11 = 1.5455` is `7.3e-2` from φ, an order of
magnitude worse than `13/8`. **If a probe needs a small-q walk, enumerate the
candidates and pick by measured discrepancy; never by closeness to φ.**

**Obey it as the design default AND measure it.** Both lattice probes carry
an N-sweep arm that varies N so the EMERGENT pair lands at indices
{8, 10, 12, 14, 17, 19, 21} and reports the metric against index. If a knee
appears near 17 the floor is measured `[G]`; if the metric is flat from
index 10 up, say so plainly — the floor is then a safety margin, not a
mechanism, and that is the operator's call to keep or relax. Never report the
floor as confirmed without the sweep.

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

**Objective.** On a Vogel lattice (`r = c·√k`, `θ = k·2π(1−1/φ)`,
**N = 3·F(17)² = 7 651 227**, c chosen so max radius = 1.0), test whether
alternating tridiagonal smoothing sweeps along the two parastichy stride
families approximate an isotropic 2D diffusion — and that the result
*depends on the strides being Fibonacci*.

> **⚠ N IS NOT A FREE PARAMETER AND THE FIRST DRAFT GOT IT WRONG — TWICE.**
> Draft 1 specified N=4096 (emergent pair F(10)/F(11), sub-floor by six
> orders of magnitude) with a step-1 search capped at `j ∈ {1..60}`, so it
> was structurally incapable of finding anything above F(10). **Draft 2**
> fixed N to exactly `F(17)²` — but the local parastichy index at radius `r`
> in a Vogel lattice is `√(r²·N)`, NOT `√N`, and draft 2's bump sat at
> `r=0.45`: local index `√(0.2025·2 550 409) ≈ 719`, still sub-floor. The
> disk's inner half is *structurally* sub-floor at ANY finite N (index → 0 as
> r → 0) — no N fixes that, only excluding those bands does. **This draft
> raises N with margin (3× the minimum, not 1×) and moves the bump to a
> radius that comfortably qualifies, then excludes the bands that cannot
> qualify rather than pretending they do.**
>
> With N=7 651 227: **`r_floor = 1597/√N ≈ 0.5774`** — only annuli with
> `r ≥ r_floor` have a local index ≥ the floor. Under the 8-equal-area
> annulus scheme (`r_i = √(i/8)`), that is **bands 3–8** (`r ≥ 0.6124`);
> **bands 1–2 are structurally sub-floor and MUST be reported under B1 only,
> never judged against B2.** The bump moves to **r₀ = 0.75** — local index
> `√(0.5625·7 651 227) ≈ 2077`, comfortably clear of 1597, safely interior
> (not at the disk edge, where a bump would have no full neighbourhood).

**Steps.**
1. Build the lattice. For each radius band (8 equal-area annuli), find the
   dominant stride pair **geometrically, with no capped search window** —
   build a KD-tree (`scipy.spatial.cKDTree`) over the band's points, take
   each point's 8 nearest neighbours, and histogram the **index differences
   `|k_neighbour − k|`**; the two most frequent differences per band are the
   stride pair. Report them and the bands where they transition. **The
   capped `argmin` scan of the first draft is forbidden** — it presupposes
   the answer's magnitude, and a discovery step that cannot return a large
   or non-Fibonacci answer is not a discovery step.
2. Per band, measure the crossing angle between the two stride directions at
   each point (angle between `x_{k+j1}−x_k` and `x_{k+j2}−x_k`); report the
   distribution (median, IQR, per band).
3. Sweep operator: for stride j, order points into chains `(start, j)`
   within a band; one sweep = `y_i ← 0.25·y_{prev} + 0.5·y_i + 0.25·y_{next}`
   along each chain (open ends: hold). One ADI iteration = sweep family A
   then family B, band-appropriate strides. **Restrict all chain-building and
   sweeping to the qualifying bands (3–8, `r ≥ 0.6124`)** — bands 1–2 are
   measured (for B1's transition map) but never swept or judged.
4. Test field: Gaussian bump `exp(−|x−x0|²/2σ²)`, σ=0.08, **x0 at radius
   0.75** (qualifying, interior — see the note above). Run 8 ADI iterations.
   Reference: sample the analytic heat-kernel-blurred Gaussian (σ_ref² = σ²
   + 8·s² where s = the measured mean neighbor spacing × 0.5 — CALIBRATE
   σ_ref by least-squares over σ_ref, then judge SHAPE) at the lattice
   points.
5. Anisotropy metric: fit the blurred bump's second-moment tensor; ratio of
   eigenvalues λ_max/λ_min.

**Bars (pre-registered, commit before run):**
- **B1** (descriptive, no pass/fail): stride pairs per band (report ALL 8
  bands, flag 1–2 as `sub_floor: true`) + transition map + crossing-angle
  table.
- **B2 ISO (bands 3–8 ONLY):** after best-σ_ref calibration, relative L2
  error between ADI result and isotropic reference ≤ **0.15**, AND
  second-moment anisotropy λ_max/λ_min ≤ **1.25**.
- **B3 CONTROL (can-it-fail), at the SAME N, DISTANCE-MATCHED not
  magnitude-matched:** for each point, its true Fibonacci partners are its
  neighbours at index offset `±1597`/`±2584`; the control partner is chosen
  as the point among its **8 real nearest lattice neighbours (via
  cKDTree)** whose PHYSICAL distance is closest to the true Fibonacci
  partner's distance, **excluding the true Fibonacci partner itself**. This
  guarantees near-identical step LENGTH by construction (the confound a
  fixed-integer control cannot rule out) while breaking the arithmetic
  coherence — connections are locally distance-matched, not globally
  recurrence-coherent. Must give anisotropy ≥ **1.5×** the Fibonacci run's.
  If the shuffled-neighbour control smooths just as isotropically, the
  Fibonacci claim measures nothing — say VOID. *(Two earlier attempts at
  this control both failed for the wrong reason: 12/18 connects points
  nowhere near each other — wrong SCALE; 1500/2600 looked scale-matched by
  raw magnitude but is not — a Fibonacci-family stride's actual PHYSICAL
  step is governed by its angular residue `(stride·golden_frac) mod 1`,
  which for 1597/2584 is ≈0.00028/0.00017 (near-zero, that is WHY they are
  parastichy numbers) while 1500/2600 sit at ≈0.051/0.112 — two to three
  orders of magnitude larger, i.e. still a wrong-scale control wearing a
  same-magnitude disguise. The distance-matched-neighbour construction
  above cannot make this mistake, because it measures physical distance
  directly instead of inferring it from integer size.)*
- **B4 INDEX-FLOOR SWEEP (the operator's rule, measured not assumed):** repeat
  the whole pipeline at **N = 3·F(n)²** for `n ∈ {8, 10, 12, 14, 17, 19}`
  (keeping the same 3× margin and the same bump-placement/band-exclusion
  logic scaled to each N's own `r_floor`), each time using that N's OWN
  emergent pair (never a forced stride), and report `iso_error` + `aniso`
  against n. **Two-sided and both readings must be stated:** a knee near
  n≈17 promotes the floor to a measured `[G]`; a curve already flat from
  n≈10 means the floor is a **safety margin, not a mechanism** — report that
  plainly rather than burying it, and leave the keep-or-relax call to the
  operator. `n=21` (`N=3·F(21)² ≈ 3.6e8`) is too large: cap the sweep at
  n=19 and record n=21 as NOT RUN — do not silently drop it.

**Cost note (changed by the floor, and again by the 3× margin):** N=7.65M
f64 ≈ 61 MB/field, cKDTree build ~30–60 s; the B4 sweep is dominated by its
largest N (`3·F(19)² ≈ 5.2e7`). Budget **~20–30 min and ~3 GB peak**. Still
zero fetch. Checkpoint per (N, band) row per §0.

**Output JSON:** `{N, bands: [...], crossing_angles: {...}, iso_error,
aniso_fib, aniso_control, sweep: [{n, N, pair, iso_error, aniso}],
verdicts: {B2, B3, B4}}`.

---

## §2 BRIEF W2s-a — golden two-lattice pairing on REAL lat/lon geometry (Sonnet, zero fetch)

**File:** `sunflower_pairing_probe.py`. **Seed:** 20260812. **No network.**

**Objective.** The collision-node construction assumes the two sunflower
lattices pair generically (no ties, even pair distances). Prove it survives
the real cos-lat metric — the #921 lesson says disk properties do NOT
automatically transfer.

**Steps.**
> **⚠ INDEX FLOOR — applies here too, but the bite is NOT obvious and must
> not be assumed either way.** The first draft used **N=2048** per lattice
> (emergent pair ≈ F(9)/F(10)), sub-floor by §0. But this probe's bars measure
> **pairing between two lattices** (ties, CV of nearest-pair distances), not
> stride addressability — and ties/incommensurability follow from the ANGLE
> being irrational, which holds at any N in f64. The evenness bar G2 is the
> one plausibly governed by the convergent index (three-gap structure).
> **Therefore: raise N so the question does not arise, AND measure it.**
> N = **F(17)² = 2 550 409** per lattice for the headline run.

1. Two Vogel lattices, **N = 2 550 409** each, disk radius 1500 km, centers at
   (55.0N, 340.0E) and (55.0N, ~361.9E) → 1400 km apart at that latitude.
   Project to km via the metric `dx = R_E·cos(lat_c)·Δlon_rad`,
   `dy = R_E·Δlat_rad` (R_E=6371.0) — the same `geom_ll` convention.
2. Overlap band: points of lattice H within 900 km of center T and vice
   versa. Nearest-pair map: for each H-point in band, its nearest T-point.
3. Control: TWO axis-aligned square grids of identical point density over
   the same two disks, same metric, same pairing procedure.

> **⚠ G1/G4 TIE DEFINITION CORRECTED.** The first draft counted global
> duplicate rounded distances across the WHOLE pair population — but that
> statistic is blind to the actual claim (does one H-point have TWO
> equally-near T-candidates, i.e. an ambiguous pairing) and, at the
> million-point sizes now in play, unrelated distance PAIRS from DIFFERENT
> source points will collide after 1e-9 km rounding by ordinary float
> density regardless of mechanism — so "duplicates observed" no longer
> implies "irrational-angle uniqueness failed". Redefined per-source below.

**Bars:**
- **G1 TIES (per-source, corrected):** for each H-point in the overlap band,
  find its 1st- and 2nd-nearest T-lattice points (`d1 ≤ d2`, via cKDTree,
  k=2). Define **near-tie** as `d1/d2 > 1 − 1e-6` (the two candidates are
  ambiguously close FOR THAT SOURCE POINT — the actual pairing-quality
  question). Count near-ties: golden = **0** (irrational angle ⇒ generic
  position, no H-point is equidistant between two T-points except by
  measure-zero coincidence); grid control **> 0** (regular lattices produce
  systematic equidistance, e.g. diagonal ties, by symmetry). If the grid also
  reports 0, the tie test is vacuous on this geometry — report VOID for G1
  and rely on G2.
- **G2 EVENNESS:** coefficient of variation of nearest-pair distances:
  golden CV **< grid CV** (strict).
- **G3** (descriptive): χ² of pair-midpoint density against uniform across
  the corridor band, both constructions, reported not judged.
- **G4 INDEX-FLOOR SWEEP (does the floor bite THIS probe?):** rerun the
  corrected G1 + G2 at `N = F(n)²` for `n ∈ {8, 10, 12, 14, 17, 19}` and
  report near-ties + CV against n. **The pre-registered expectation, written
  down before the run so it can be wrong:** near-ties stay **0 at every n**
  (the angle is irrational at any N), while CV improves monotonically and
  may flatten near the floor. **If near-ties appear below the floor, the
  angle reading was wrong and G1's mechanism is not what this brief
  claims** — report it as a correction to §0's "the angle is unaffected"
  split, which is exactly the kind of claim that should be falsifiable
  rather than inherited.

**Cost note:** the N=2.55M headline plus the sweep; budget ~10 min, ~2 GB,
zero fetch. Checkpoint per (n, arm) row per §0.

**Output JSON:** `{N, n_pairs, ties_golden, ties_grid, cv_golden, cv_grid,
chi2_golden, chi2_grid, sweep: [{n, N, ties, cv}], verdicts: {G1, G2, G4}}`.

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
