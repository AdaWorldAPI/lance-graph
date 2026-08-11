# Evaluation Plan — Weather Normalized Substrate: KNOWN vs TO TEST (v1)

> **READ BY:** certification-officer, truth-architect, family-codec-smith,
> integration-lead, measurement-skeptic-analog sessions, and ANY session
> executing an EV-* probe below.
>
> **Status:** DRAFT pending the verify/attack audit pass (§8) → then ACTIVE.
> **Provenance:** the #915/#917/#920/#921/#922 arc + the post-#920 session
> corrections (`weather-normalized-substrate.md` §12.1–§12.17).
> **Grading:** `[G]` verified in committed code/test with file:line ·
> `[G-session]` measured this session, re-runnable from committed artifacts ·
> `[H]` measured this session, NOT independently re-runnable as committed ·
> `[S]` proposal/conjecture — do not build on before its probe runs.

---

## 0. Purpose

Convert the session's findings into a gated evaluation: one ledger of what is
KNOWN (each entry with its evidence and its honest grade), one queue of what is
TO TEST (each with method, pass/fail, and the falsifier discipline the
workspace mandates: can-it-fire + can-it-stay-silent + no inert thresholds +
never a round-trip metric).

Binding process rules for every EV-* execution:
- **The evaluation frame** is bucket-CI vs noise floor — never decoded
  round-trip error (§12.10/§12.11).
- **Significance** on pooled/spatial data uses Jirak 2016 weak-dependence
  rates, never classical IID Berry–Esseen (`I-NOISE-FLOOR-JIRAK`).
- **Before writing a frame, grep the workspace for the frame** (§12.17): the
  `mu + k·σ` calibrate→band→roll frame ships 4× already.
- Fixtures: fetched, never committed; probes re-runnable; results JSONs
  committed; the `/tmp`-fixture time-bomb rule applies.

---

## 1. KNOWN — the verified ledger

### 1a. Helix codec facts

| ID | Claim | Evidence | Grade |
|---|---|---|---|
| K-1 | `Signed360` = 6 B, wire `[rim.start, rim.end, rim.floor_version, polar, az_lo, az_hi]`; ONE complete full-sphere direction | `crates/helix/src/residue.rs:76–116`; doctrine `helix-cartesian-vs-fisher2z.md` §"Signed360 specifics" | [G] |
| K-2 | Sign partition exact at every magnitude: `Pos ⇒ polar ∈ [128,255]`, `Neg ⇒ [0,127]`, incl. the rim `\|y\| ≈ 0` (#498) | `residue.rs:182–204`; tests `signed360_neg_sign_survives_near_rim_at_high_total`, `polar_partitions_are_exactly_the_two_halves` (disable-verified) | [G] |
| K-3 | `azimuth` spans the FULL u16 circle: min 0, max 65535, 256/256 coarse arcs, 54 319 distinct at N=65536 | `crates/helix/tests/signed360_claims.rs::azimuth_spans_the_full_circle_not_merely_varies` (disable-verified: 10-bit truncation → red) | [G] |
| K-4 | Dormant-lane defect: all-zero lane decodes as definite `Sign::Neg` — pinned as a DEFECT test; filed, NOT fixed | `signed360_claims.rs::dormant_all_zero_lane_decodes_as_a_definite_sign_known_defect` | [G] |
| K-5 | `DistanceLut::circular()` = `min(\|a−b\|, 256−\|a−b\|)` is a metric — EXHAUSTIVE 256³ = 16 777 216 triples, 0 violations; wrap falsifier `d(255,0)=1` vs linear 255 | `crates/helix/src/distance.rs::circular` + 3 tests | [G] |
| K-6 | The `[a,b]` amortization: `quantize()` normalizes once at ingest; `from_floor()` folds the SAME normalization into the table → unit-free pure-lookup comparisons | `quantize.rs:99–108`, `distance.rs:39–50` | [G] |
| K-7 | Bearing-encode paths measured, N=65536: nearest-`n` mean 0.972° vs direct `(polar,azimuth)` 0.097° (10×); mechanism = one index couples lat+az, `sin(2·lat)` disk-lattice density | `crates/helix/tests/bearing_encode_paths.rs` (committed, re-runnable) | [G] |
| K-8 | `helix` is root-workspace-EXCLUDED and named in NO CI workflow → all its tests run only by hand | root `Cargo.toml` `exclude`; grep of `.github/workflows/` | [G-absence] |

### 1b. Weather measurements (ERA5, one timestep 2021-06-15 12:00 UTC unless noted)

| ID | Claim | Evidence | Grade |
|---|---|---|---|
| K-9 | Fisher-Z on `2m_temperature` anomalies: at 0.5–1 K floor an address-economy failure only (sat 0.848 % linear / 0.820 % fisher-z, otherwise indistinguishable); at 0.25 K also a validity failure (+95.65 % interior-CI exceedance) | `probes/weather-p1/p1_ci_vs_floor.{py,json}`; internally consistent without fixture (linear uniform to 3.8e-14; monotone floor sweep) | [G-session] |
| K-10 | Standardization, not Fisher-Z, licenses cross-variable comparison: 0.9997 shared palette vs 0.857–0.875 raw cross-unit; T×Td shared = 0.999556 (BELOW a 0.9996 bar — rounding once hid this) | `probes/weather-p1/p2_probe.py`, `p2_results.json` | [G-session] |
| K-11 | Gate-1 reliability ran in `jc` (Pearson/Spearman/Cronbach α/ICC) with a `--shuffle` negative control and header validation | `crates/jc/examples/weather_substrate_reliability.rs` + committed `jc_input.bin` | [G] |
| K-12 | `Signed360` angular error by latitude: equator (0–5°) 0.112° mean / 0.226° max; pole (85–90°) 3.332° / 4.998° (~30× spread); NO equal-budget gain from the sign split (7-bit+sign vs 8-bit: 0.99–1.02× every band) | measured this session; recorded in `LATEST_STATE.md` #920 entry — **scratch test was deleted, NOT committed** | [H] |
| K-13 | u8-palette azimuth under `circular()`: 1.406° step, 0.352° mean — beats nearest-`n` 0.972° AND keeps field ergonomics | derived arithmetically + circular-LUT proof; per-band measurement not committed | [H] |

### 1c. The shipped frame (what the probes must consume, not re-implement)

| ID | Claim | Evidence | Grade |
|---|---|---|---|
| K-14 | `perturbation_sim::rolling_floor::RollingFloor` IS the corrected evaluation frame: `threshold()` = "the confidence-interval floor" `mu+k·σ`; `z()` = "the Jirak-honest noise-floor units; significance via n^(p/2−1), not IID"; `band()` → Stable…Alarm; `preheat()`; `observe()` tests against the floor as it stood | `crates/perturbation-sim/src/rolling_floor.rs` (~:93–145) | [G] |
| K-15 | `splat.rs` = Gaussian-splat MAGNITUDE side; `sketch.rs` = Walsh/XOR SIGN side — the two-algebra rule instantiated; `morton2`, `box_coarsen`, `ewa_coarsen` | `crates/perturbation-sim/src/{splat,sketch}.rs` | [G] |
| K-16 | `cascade_key.rs`: 16-bit-per-tier OGAR-form HHTL address — `from_spectral`, `to_guid_tiers`, `morton48`, `cascade_distance`, `tile` | `crates/perturbation-sim/src/cascade_key.rs` | [G] |
| K-17 | `hhtl.rs` derives `(HEEL,HIP,TWIG)` by recursive Cheeger bisection of the Laplacian | `crates/perturbation-sim/src/hhtl.rs` | [G] |
| K-18 | `ndarray::hpc::cascade::Cascade` = the Belichtungsmesser original: `calibrate`/`expose`→`Band`/`observe`→`ShiftAlert`/`recalibrate` | `ndarray/src/hpc/cascade.rs` | [G] |
| K-19 | `symbiont/src/domino.rs` proves the AMX path: 4×4 Morton BF16 tiles, 16 SoA boards per AMX 16×16 tile GEMM, cascade feedback, real `TDPBF16PS` (Emerald Rapids); ALL SIMD via `ndarray::simd::*`; "ndarray has no Morton primitive" (consumer-side `morton4`) | `crates/symbiont/src/domino.rs` header + `run_poc` | [G] |
| K-20 | `ndarray::hpc::splat3d` is a full 3DGS pipeline (feature-gated DIRECTORY): gaussian/project/raster/tile/spd3/sh/ply/depth_cascade; `TILE_SIZE = 16`; `depth_cascade` is ALREADY HHTL (`HhtlTier`, `HhtlAction`, `heel_reject_mask`) | `ndarray/src/hpc/splat3d/` | [G] |
| K-21 | Morton motion is O(1) in pixels: `dx=7 dy=5 bit_exact=1 motion_bytes=2 sprite_px=576`; interior residual 0; only the disocclusion strip new (`disocc_frac=0.1215`); comma fence holds (`three_gap_distinct=3`, `coprime_full_perm=1`) | `crates/helix/examples/morton_shift_motion_probe.rs` — RUN this session | [G-session] |
| K-22 | The perturbation phase is the DISCRETE Pythagorean comma: stride-4-over-17 coprime walk, integer, bit-exact, aperiodic, **0 stored bits** (address-derived); `17 = 4²+1` IS the comma; D-QUANTGATE rules it mandatory in quantized layers | `fire_forget_replay_probe.rs:70–73`, `probe_hhtl_intake_blindness.rs:410`, OGAR `CLAUDE.md` D-QUANTGATE | [G] |
| K-23 | 64×64 = 4096 cells = 4096 bit = 512 B = 64×u64 = the CANON node stride; `masked_popcount_batch(words, mask)` IS the Stockfish primitive; magic bitboards = the same LUT amortization | arithmetic identity + `ndarray/src/hpc/bitwise.rs` | [G] |
| K-24 | AMX eats `&[u8]` planes directly: `int8_gemm_amx_tiled(a_u8: &[u8], b_i8: &[i8], c: &mut [i32], …)` | `ndarray/src/hpc/int8_tile_gemm.rs` | [G] |
| K-25 | `ValueTenant::HelixResidue` = `U8×6 @ row_offset 112`; **zero writers, zero decoders** in the whole tree; no per-value-lane reading selector exists in the contract | contract `canonical_node.rs`; exhaustive grep (15 hits) | [G-absence] |

---

## 2. Honesty split — session-only evidence that needs committing

- **K-12 / K-13 are the only KNOWN rows without a committed reproducer.** The
  latitude-band table and the palette-azimuth per-band numbers were measured in
  scratch tests that were deliberately deleted. EV-9 commits them.
- K-9/K-10/K-21 are `[G-session]`: the probe SOURCE is committed and re-runnable,
  but the fixture is fetched (container-reset-cleared) and the run happened this
  session. Re-running is one `fetch.py` away — that is by design, not a gap.

---

## 3. TO TEST — the EV queue

Every EV lands as a committed, re-runnable artifact (Rust test/example, or a
probe script + result JSON), with its pass criterion stated BEFORE the run.
Bars marked *(provisional pin)* are set relative to a measured baseline in the
same run — never an absolute magic number; first run pins them, and a pinned
bar must then be two-sided.

### EV-1 — Advection IS a Morton shift (the §12.16 [S] falsifier)
- **Claim under test:** a wind-derived per-tile `(dx,dy)` Morton shift of
  `2m_temperature(t)` predicts `t+1h` better than persistence.
- **Method:** 2 consecutive ERA5 timesteps; per tile (16×16 first, 64×64
  control) `(dx,dy) = round(mean 10m wind · 3600 s / 27.8 km)`; shift; residual
  vs truth. At 0.25°/1 h, 10 m/s ≈ 1.3 cells — measurable, not sub-cell noise.
- **Pass (can-fire):** on tiles with `|wind| ≥ v_min`, shifted residual MAE
  beats persistence MAE *(provisional pin: report the ratio; adopt only if
  < 0.8)*. **Silence:** on calm tiles (`|wind| < v_min/2`) shift ≈ persistence
  (ratio ∈ [0.9, 1.1]) — a shift that "wins" on calm air is an apparatus bug.
- **Fail →** §12.16's advection paragraph regraded; the wind lane stays a field,
  not a transport operator.
- **Inertness:** `v_min` must be non-inert — sweep it; if the pass/fail verdict
  is insensitive to `v_min` over [2, 10] m/s, the tile-mean wind is too smooth
  and the probe must move to 64×64 tiles.

### EV-2 — Wind lane encode at FIELD level (closes §12.13–§12.15)
- **Claim:** u8-palette azimuth + `circular()` preserves the angular-distance
  structure of a REAL wind field; `linear()` measurably corrupts it.
- **Method:** ERA5 10m u/v → bearings → three encodes (nearest-`n`;
  u8-palette-circular; u16-linear). For sampled point pairs: true angular
  difference vs table distance. Spearman ρ per encode (Jirak-cited
  significance), plus a wrap-corruption COUNT for `linear()` (pairs straddling
  0°/360° whose rank inverts).
- **Pass (can-fire):** palette-circular ρ within 0.02 of the u16 ground-truth
  ranking AND `linear()` wrap-corruption count > 0 on real data (the defect
  must actually fire on nature, not just on synthetic 359°/1°).
  **Silence:** restricted to pairs within a 90° sector (no wrap), `linear()` ≈
  `circular()` — proves the corruption is THE WRAP, not table noise.
- **Fail →** the palette-azimuth option is not field-adequate; §12.15's [S]
  adoption question closes NO.

### EV-3 — Floor-sensitivity sweep (feeds operator decision D-1)
- **Claim:** every floor-dependent verdict (the 0.25 K flip) is stable inside
  the plausible obs-error range.
- **Method:** re-run `p1_ci_vs_floor` over a floor grid [0.1 … 2.0] K per
  variable; deliverable = the flip-point (floor at which Fisher-Z's interior-CI
  exceedance crosses 1 %) per variable.
- **Pass:** flip-points are sharp (not straddling the whole range) and reported
  with the sat-% column; the OUTPUT is the decision input for D-1, so the pass
  bar is completeness, not a verdict.
- **Vacuity guard:** the sweep must show BOTH regimes (some floor where
  Fisher-Z fails, some where it doesn't) — a sweep entirely in one regime
  cannot locate a flip-point and must widen its grid.

### EV-4 — Saturation-window widening (feeds D-2)
- **Method:** sweep the percentile window {0.4–99.6, 0.2–99.8, 0.1–99.9,
  0.02–99.98}; per window: sat-%, interior-CI median/max vs the K-9 floors.
- **Pass:** the tradeoff curve is monotone in the expected directions (sat ↓,
  interior CI ↑ as the window widens) — a non-monotone curve means the
  apparatus is wrong, not the physics. Decision itself is D-2.

### EV-5 — U-shaped variables: the shape rule's OTHER half
- **Claim (from `E-THE-TRANSFORM-MUST-MATCH-THE-DISTRIBUTION-SHAPE-1` [S]):**
  Fisher-Z should WIN on bound-massed variables (`total_cloud_cover`,
  `sea_ice_cover`).
- **Method:** identical P1 pipeline on both variables.
- **Pass (two-sided by construction):** Fisher-Z shows better bucket economy /
  interior CI than linear on these — while K-9 already shows the OPPOSITE on
  temperature. Either outcome is informative: win → the shape rule promotes
  [S]→[H]; loss → the rule is falsified and the epiphany regrades.

### EV-6 — Harness re-expression on the shipped frame (retires the Python)
- **Claim:** the P1/P2 pipeline expressed as consumers of the SHIPPED frame
  (`perturbation_sim::RollingFloor` or `helix::RollingFloor` + `jc`
  reliability + `CascadeKey`/`morton48` addressing) reproduces the committed
  numbers.
- **Equivalence gate (exact, not approximate):** 0.848 % / 0.820 % / 95.65 %
  (P1) and 0.9997 / 0.857–0.875 incl. T×Td = 0.999556 (P2) to 1e-4 relative
  from the same fixture. Python stays as provenance (the repo's
  `p1_noise_floor.py` pattern), Rust becomes the measurement of record.
- **Vacuity guard:** the Rust harness must NOT share code with the Python
  (independent implementation is the point); assert on the committed JSON
  values, not on freshly-computed Python output.

### EV-7a — 16k×16k 3DGS top-k scale run (operator-named)
- **Claim:** the shipped `splat3d` pipeline + HHTL depth cascade handles a
  16 384² canvas = 1024×1024 = 1 048 576 tiles with top-k selection.
- **Method:** synthetic-or-ERA5-derived gaussian population; `TileBinning` at
  1M tiles; `cascade_blocks` + `heel_reject_mask` prune rates;
  top-k via `hamming_top_k_raw`/cascade bands.
- **Pass:** completes within a recorded time/memory envelope *(provisional
  pin — first run records, second run gates)*; heel-reject rate is neither 0 %
  nor 100 % (a pruner that fires on everything or nothing is the
  `closed_class_guess` defect).
- **NOTE:** `splat3d` is feature-gated — the run must state the feature flags
  and host (AMX availability changes the path).

### EV-7b — Comma anti-moiré at tile scale (the D-QUANTGATE falsifier)
- **Claim:** comma-phase (stride-4-over-17) perturbation shows no grid-locked
  aliasing where a REGULAR stride does.
- **Method:** perturb a uniform field per tile with (a) regular phase
  (`addr % k`), (b) comma phase; measure grid-alignment energy (spectral peak
  at the tile frequency, or lag-k autocorrelation at tile pitch — NOT plain
  lag-autocorrelation, which `mu_hydration_probe` documents as the WRONG
  instrument).
- **Pass (two-sided, mandatory):** regular stride MUST show the peak
  (can-fire) AND comma must not (silence). If the regular control does not
  alias, the fixture cannot falsify and must be re-scaled.

### EV-8 — Jirak effective-n for the P2 correlations
- **Claim:** the P2 correlations survive weak-dependence-honest significance.
- **Method:** estimate spatial autocorrelation length of the anomaly fields →
  effective n ≪ 1 038 240 → re-state P2 significance at Jirak's `n^(p/2−1)`;
  feeds P5 (`drift_sigma` under autocorrelation).
- **Pass:** stated effective-n with the derivation; correlations re-graded
  against it (expected to survive — but "expected" is not evidence).

### EV-9 — Commit the K-12/K-13 measurements (Wave 0 — no data needed)
- Re-create the latitude-band error test and the palette-azimuth per-band
  measurement as committed `crates/helix/tests/` with tolerance assertions on
  the recorded numbers (0.112°/3.332°/0.99–1.02×/0.352°), disable-verified.
- **Pass:** green + each assertion red under an injected defect.

### EV-10 — Second timestep + season for P1 ([H]→ promotion path)
- **Method:** winter timestep (2021-01-15 12:00 UTC) + one more variable
  through the full P1-CI pipeline.
- **Pass:** same qualitative ordering as K-9 (economy failure at plausible
  floors). **Fail →** `E-THE-TRANSFORM-MUST-MATCH-…-1` regrades; every §12.11
  conclusion becomes timestep-conditional.

---

## 4. Waves

| Wave | EVs | Prerequisite |
|---|---|---|
| 0 (now, no data) | EV-9 | none |
| 1 (one `fetch.py` re-fetch) | EV-1, EV-2, EV-3, EV-4, EV-5, EV-8, EV-10, EV-6 | fixture on disk |
| 2 (scale) | EV-7a, EV-7b | feature flags + host statement |

Per the workspace probe discipline: **if an EV is NOT RUN, the next deliverable
is the probe, not more synthesis.**

## 5. Operator decision register (open calls; tests feed them, never decide them)

| ID | Decision | Fed by |
|---|---|---|
| D-1 | Pin a citable per-variable noise floor | EV-3 |
| D-2 | Saturation-window policy | EV-4 |
| D-3 | `from_bearing` / wind-lane API shape (incl. u8-palette azimuth adoption) | EV-2, K-5, K-7 |
| D-4 | Dormant-lane fix shape (`Option<Sign>` vs sentinel) — the pinned defect test must break deliberately | K-4 |
| D-5 | helix CI wiring (workspace-excluded, no gate anywhere) | K-8 |
| D-6 | Harness language ruling — adopt EV-6's Rust as measurement-of-record | EV-6 |

