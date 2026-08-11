# Evaluation Plan — Weather Normalized Substrate: KNOWN vs TO TEST (v1)

> **READ BY:** certification-officer, truth-architect, family-codec-smith,
> integration-lead, measurement-skeptic-analog sessions, and ANY session
> executing an EV-* probe below.
>
> **Status:** ACTIVE (audited 2026-08-11 — §8). The 13-agent verify/attack pass
> ran: verify 22 CONFIRMED / 2 PARTIAL (K-21, K-23 corrected in place);
> attack **11 of 11 specs NOT SOUND** — every §3 spec below is the **v2,
> post-audit** version; §8 records what each v1 got wrong.
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
| K-21 | Morton motion is O(1) in pixels — **for a SPRITE, not the whole field** (audit correction: legA rigid-translates a 24×24 `Sprite` within a 256×256 field via one `(dx,dy)` address-delta, bit-exact vs ground-truth re-render; "2 bytes moves the whole field" in §12.16 OVERSTATES — 2 bytes moves one sprite/tile); interior residual 0; disocclusion strip only (`disocc_frac=0.1215`); comma fence holds (`three_gap_distinct=3`, `coprime_full_perm=1`). Note the shift is **toroidal** (`wrapping_add` per lane); the clipped variant is a separate helper | `crates/helix/examples/morton_shift_motion_probe.rs` — RUN this session | [G-session] |
| K-22 | The perturbation phase is the DISCRETE Pythagorean comma: stride-4-over-17 coprime walk, integer, bit-exact, aperiodic, **0 stored bits** (address-derived); `17 = 4²+1` IS the comma; D-QUANTGATE rules it mandatory in quantized layers | `fire_forget_replay_probe.rs:70–73`, `probe_hhtl_intake_blindness.rs:410`, OGAR `CLAUDE.md` D-QUANTGATE | [G] |
| K-23 | 64×64 = 4096 cells = 4096 bit = 512 B = 64×u64 = the CANON node stride; `masked_popcount_batch(words, mask)` IS the Stockfish primitive; magic bitboards = the same LUT amortization | arithmetic identity + **`ndarray/src/bitwise.rs`** (audit correction: NOT `src/hpc/` — `pub mod bitwise` at `lib.rs:290`; fns at `:280/:286/:201`) | [G] |
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

## 3. TO TEST — the EV queue (v2, post-audit — every v1 spec failed the attack pass; §8)

Every EV lands as a committed, re-runnable artifact with its pass criterion
stated BEFORE the run. All bars below are pre-registered; a bar may be re-pinned
only via a recorded §8-style amendment, never silently.

### EV-1 — Advection via per-tile Morton translate (v2)
- **Claim:** a wind-derived per-tile `(dx,dy)` translate of `2m_temperature(t)`
  predicts `t+1h` better than persistence, **because of the wind direction**.
- **Method:** 2 consecutive ERA5 timesteps. Per tile:
  `dx = round(u·3600 / (27 800·cos(lat)))` (zonal spacing SHRINKS with
  latitude — the uncorrected v1 formula was wrong by up to 2.6× exactly where
  winds are strong), `dy = round(v·3600 / 27 800)`. Shift semantics:
  **clipped**, never toroidal (the probe's `wrapping_add` lanes would teleport
  the antimeridian); scoring mask **excludes** disoccluded/inflow cells and
  reports the masked fraction.
- **Pass (can-fire):** on tiles with `|wind| ≥ v_min`, wind-shift residual MAE
  `< 1.0×` persistence MAE — one criterion. (The v1 "0.8" was an absolute
  magic number violating §3's own preamble; **0.8 is now only D-register
  adoption input**, not test pass.)
- **Silence (v2 — the v1 calm-tile band was an IDENTITY comparison: the shift
  rounds to 0 for `|v| < 3.86 m/s`, so calm tiles compared the field to
  itself):** on the SAME windy tiles, a **reversed** `(−dx,−dy)` and a
  **90°-rotated** shift of identical magnitude must each score ≥ 0.98× the
  persistence MAE — direction must be load-bearing, not "some translation
  helps". The calm band survives only as a labelled **wiring canary**
  asserting ratio == 1.0 exactly.
- **Inertness:** sweep `v_min ∈ {4, 6, 8, 10}` m/s (all above the rounding
  floor 3.86); the can-fire margin must move with it.
- **Fail →** §12.16's advection paragraph regraded; the wind lane stays a
  field, not a transport operator.

### EV-2 — Wind-lane encode at FIELD level (v2)
- **Reference, pinned (the v1's most dangerous omission):**
  `Δθ_true(P,Q) = min(|θ_P−θ_Q|, 360−|θ_P−θ_Q|)` in f64 from raw u/v, with an
  in-run assertion `Δθ_true(359°, 1°) == 2°`. u16-linear is an ARM, never the
  reference.
- **Arms:** nearest-`n`; u8-palette + `circular()`; u8-palette + `linear()`
  (the defect arm); u16 + `linear()`.
- **Can-fire:** Spearman ρ(table distance, Δθ_true) for palette-`circular()`
  exceeds palette-`linear()` by a margin that must be positive with a
  Jirak-honest CI; additionally count **discordant pairs-of-pairs involving a
  wrap-straddling pair** (the v1 "count > 0" was implied by sampling design —
  the `elimination_rate() > 0` house defect verbatim; the v2 quantity is the
  measured discordance GAP between the two tables, direction pre-registered).
- **Silence (v2 — the v1 "90° sector" half was an ARITHMETIC TAUTOLOGY:
  `circular == linear` identically for `|a−b| ≤ 128`, and a 90° sector spans
  64 indices):** S1 = an exhaustive **unit test in `distance.rs`**, labelled
  as carrying ZERO field evidence: `circular == linear` for all `|a−b| ≤ 128`,
  `≠` for all `|a−b| > 128`. S2 (field): on pairs whose palette indices differ
  by ≤ 64, the two tables' distances are IDENTICAL by arithmetic — assert it
  as the wiring canary it is, not as evidence.
- **Inertness:** an injected 10-bit azimuth truncation must push
  palette-`circular()`'s ρ measurably down (the disable-run).
- **Fail →** the palette-azimuth option is not field-adequate; §12.15's [S]
  adoption question closes NO.

### EV-3 — Floor flip-points, computed EXACTLY (v2)
- **v1 was a grid search whose only guard was pre-satisfied by committed data.**
  The exceedance-vs-floor curve is the occupancy-weighted survival function of
  interior bucket CI half-widths, so the 1 % flip-point **IS the 99th
  percentile of `ci[idx][interior]`** — compute it in closed form per variable
  (sort interior buckets by CI desc, cumulate occupancy/N, report the crossing
  CI + the number of buckets and mass fraction determining it). No grid, no
  step size, no undefined "sharp".
- **Apparatus control (v2):** the LINEAR arm's CI is `(hi−lo)/512`, a constant
  — its flip is a single threshold at 0.09412 K and it can never show two
  regimes. That is expected behaviour of the apparatus, documented as the
  control; the two-regime guard applies to the **Fisher-Z arm only**.
- **Output:** per-variable flip-point table → the D-1 decision input. The
  sweep range is explicitly provisional-until-D-1 (v1 presupposed the range
  D-1 exists to pin — circularity, now named).
- **404 discipline:** fixture availability probed per variable FIRST; a fill
  array is never saved as a fixture.

### EV-4 — Saturation-window sweep (v2)
- **Method:** windows {0.4–99.6, 0.2–99.8, 0.1–99.9, 0.02–99.98}. Report BOTH
  bucket-0/255 occupancy AND true out-of-window mass — the v1 conflated them
  (`sat%` counts legitimately in-range points in the edge buckets).
- **Pass (v2 — v1's "monotone" admitted a DEAD KNOB: a constant curve is
  monotone in both directions, and a window parameter that never reaches the
  code yields exactly that):** (i) linear CI **strictly** increasing AND
  `ci(0.02–99.98)/ci(0.4–99.6)` equals the corresponding `(hi−lo)` ratio to
  1e-12 — the proof the knob reached the code; (ii) true out-of-window mass
  falls ≈ proportionally (0.8 % → 0.04 % nominal); (iii) Fisher-Z arm
  reported, and a non-monotone Fisher-Z curve is a **FINDING to investigate**
  (both `scale` and `zlo/zhi` move with the window) — not auto-assigned to
  "apparatus wrong" (the v1 wording immunized the expectation against
  falsification).

### EV-5 — U-shaped variables (v2 — the v1 fixture could not exist and its pipeline destroyed the phenomenon)
- **Fixture gate:** `total_cloud_cover` is a KNOWN 404 at the pinned timestep
  and `sea_ice_cover` is in NEITHER fetcher. Resolve a timestep where both
  chunks exist, add both to a fetcher, record `time_index` + nonfinite counts
  in a manifest, abort LOUDLY if absent. (v1 placed this in Wave 1 as if one
  re-fetch sufficed.)
- **Pipeline (v2):** quantize the **RAW [0,1] field** — the zonal-anomaly step
  destroys the bound-massed shape the claim is about, and Kelvin floors are
  meaningless for fractions. Floors in native units {0.01, 0.05} fraction.
- **Metric:** interior-CI survival + `effective_buckets` (exp-entropy of
  occupancy) — never MAE (the banned round-trip metric; the v1 "bucket
  economy" had no operational definition and one natural reading was MAE).
- **Premise correction (v1's "two-sided by construction" collapsed):** in the
  CI frame K-9 does NOT show linear "winning" on temperature (sat 0.848 % vs
  0.820 % — nearly equal; only the 0.25 K interior CI discriminates). The
  contrast is re-based: prediction = Fisher-Z interior-CI advantage on raw
  bound-massed fields at fraction-scale floors, with temperature-at-0.25 K as
  the opposing case. Win → shape rule [S]→[H]; loss → the epiphany regrades.

### EV-6 — Rust harness equivalence (v2 — the v1 gate was UNSATISFIABLE)
- **Split deterministic from stochastic (the v1 demanded 3-sig-fig prose
  constants "to 1e-4 relative" — the house rounding incident rebuilt inside
  the gate):**
  - **P1 (deterministic):** assert the FULL-PRECISION committed values
    `0.008476845430728927`, `0.008202342425643397`, `0.956511018646941`.
    The 3-sig-fig figures are prose, never gate constants.
  - **P2 (Monte-Carlo — numpy PCG64 seed 7, 200 000 pairs; a Rust port cannot
    reproduce the stream):** Rust draws its OWN seeded pairs, K ≥ 20
    resamples, gate `|mean_rust − committed| < 3·sd_rust`, sd printed.
- **Frame pinned:** `helix::RollingFloor` (`quantize`/`bucket_center`) for the
  palette; `jc::reliability` for stats. The v1 named "`perturbation_sim` OR
  `helix` `RollingFloor`" — two different types, one of which (the streaming
  meter) structurally cannot express the bucket metric.
- **Participation canary (v2 — nothing in v1 observed that the frame ran):** a
  disable-run injecting a shifted floor must move the P1 numbers; a harness
  that computes the numbers beside an unused `RollingFloor` fails it.
- **Anti-self-comparison:** expected side = committed JSON; actual side =
  recomputed from the fixture (digest recorded) through the Rust path. Both
  sides reading the same JSON is the forbidden route, now named.

### EV-7a — 16k×16k 3DGS top-k (v2)
- **Fixture pin:** seeded generator (fixed RNG seed, gaussian count, scale
  distribution, camera pose), digest recorded; the envelope gate compares runs
  of the SAME digest only, and is **host-conditional** (CPU model + AMX
  availability recorded — the spec's own NOTE said the path changes, so an
  unconditional gate fails for reasons unrelated to code).
- **Top-k correctness (v1 named top-k in the claim and never tested it — the
  primary vacuous route):** pin `k`; assert the cascade top-k equals a
  brute-force full-sort reference on a non-trivial subset; assert it DIFFERS
  from a stubbed/garbage control (can-fire).
- **Heel-reject:** the fixture places pre-stated fractions in/out of frustum;
  the reject rate is asserted within a band DERIVED from the fixture spec
  (two-sided) — v1's "∉ {0 %, 100 %}" was author-satisfiable by construction.
- **Execution proof:** instance counts must equal an independently computed
  tile-touch count on a subsample — `splat3d` is feature-gated, and "the mode
  never reached the code" is this workspace's measured incident.

### EV-7b — Comma anti-moiré (v2)
- **Instrument, pinned to ONE:** 2-D DFT; statistic = energy in the tile-pitch
  bin / median energy over non-DC bins. The v1's "or lag-k autocorrelation"
  alternative is DELETED — the `mu_hydration_probe` objection it cited applies
  to it.
- **Arms, three (v1's two could only prove divisor-vs-coprime arithmetic):**
  (a) regular divisor stride; (b) the comma walk; (c) a DIFFERENT coprime
  stride. Pre-registered: (a) aliases, (b) and (c) do not — the claim under
  test is D-QUANTGATE's *coprime requirement*, explicitly NOT a 17-uniqueness
  claim. **Wording fix:** the comma walk is PERIODIC with period 17 (a
  permutation of Z₁₇); "aperiodic" means incommensurate with the 16-pitch —
  K-22's phrasing is loosened accordingly.
- **Bars, pre-registered:** regular ratio ≥ R_hi, comma/coprime ≤ R_lo, with
  R_hi ≥ 10·R_lo pinned before the run.
- **Silence-side fixture guard (v1 guarded only the can-fire half):** the
  comma arm's total perturbation energy must be shown nonzero — "no peak"
  must not be "no signal".

### EV-8 — Stability of the P2 estimator (v2 — re-scoped from a rubber-stamp)
- **v1 could not fail:** "state effective-n and re-grade" is always
  achievable, at ρ ≈ 0.9996 any n_eff over a few dozen "survives", and the
  named n (1 038 240) was the WRONG n — the P2 estimator drew **200 000
  independent uniform index pairs** (`p2_probe.py:42,48`), not the grid.
- **v2:** K ≥ 20 seed-resamples of the P2 estimator → CI on each ρ.
  Pre-registered: n_eff ≈ n for independent uniform draws is the EXPECTED
  finding (so that outcome is a result, not a wiring alarm); n_eff ≪ n would
  indict the estimator, not the field. State plainly that ρ(d_sh, truth) is a
  **codec-consistency statistic** (deterministic monotone quantizer of the
  same z), so the deliverable is CI width and cross-seed stability — not a
  p-value dressed as inference. Jirak's `p` pinned in the artifact or the
  Jirak citation dropped for the resampling CI.

### EV-9 — Commit the orphan measurements (v2 — two of four v1 assertions were implied by their own subject)
- **Dropped as near-tautologies:** the bare `0.99–1.02×` equal-budget ratio
  (both arms have ≈ equal step size BY CONSTRUCTION — 127 vs 127.5 levels)
  and the bare `0.352°` mean (= step/4 arithmetically).
- **v2 assertions, tolerances pinned from the ANALYTIC law BEFORE the run
  (the original apparatus was deleted; recorded numbers are cross-checks, not
  gospel):**
  (i) anti-vacuity — the 7-bit+sign and 8-bit arms produce DIFFERENT bytes on
  a stated minimum fraction of samples (proves two code paths exist);
  (ii) the equal-budget ratio with an ASYMMETRIC disable-run (perturb only the
  sign-split arm → ratio must leave the band; a symmetric wiring bug yields
  exactly 1.000 forever and the v1 could not see it);
  (iii) the SPREAD, not two points: pole-band mean / equator-band mean ≥ 20×
  (the finding IS the ratio; independent point assertions can drift together);
  (iv) comparative K-13: palette-`circular()` mean < nearest-`n` mean on the
  SAME sample;
  (v) sampling scheme pinned: N, latitude stratification, azimuth sweep.

### EV-10 — Second season (v2 — the v1 could pass on STALE SUMMER BYTES)
- **Fetcher fix first:** both fetchers gain an explicit `--t` argument and
  write `fixture/<t>/<var>.npy`; the skip-if-exists guard keys on the
  timestep; the probe READS the manifest and asserts its `time_index`,
  emitting it into the result JSON. A run whose fixture cannot prove its own
  timestep reports NO-VERDICT. (v1: `fetch_bg.py:10` hardcodes `t`, `:15`
  skips if the file exists — a winter "run" would happily re-measure summer.
  Also protects EV-6, whose equivalence gate depends on the summer fixture a
  shared directory would have overwritten.)
- **One factor at a time:** Run A = SAME variable, winter timestep. The "one
  more variable" is a separate run, named in advance (`10m_u_component`,
  availability already proven). Three-factors-at-once made a v1 fail
  unattributable.
- **Compared quantity, defined:** the EV-3 closed-form flip-point must agree
  across seasons within a pre-registered factor of 2 — v1's "same qualitative
  ordering" named an ordering the pipeline does not produce (at floors ≥ 0.5 K
  both arms read identically 0.0).

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


---

## 8. AUDIT RECORD — the verify/attack pass (2026-08-11, run `wf_99d677e6-b45`)

13 agents (7 Sonnet source-verifiers, 6 Opus falsifier-auditors), 13/13
completed, 0 errors, ~4.4 M subagent tokens, 165 tool calls, ~20 min.

### Verify: 24 claims → 22 CONFIRMED, 2 PARTIAL (both corrected in §1)

- **K-23 PARTIAL** — functions confirmed with exact signatures, but at
  `ndarray/src/bitwise.rs` (`lib.rs:290`), NOT `src/hpc/bitwise.rs`. Path fixed.
- **K-21 PARTIAL** — legB/legC exact; legA is a **SPRITE translate within a
  256×256 field**, not a whole-field shift; the shift is **toroidal**
  (`wrapping_add` per lane) with a separate clipped helper. §12.16's "2 bytes
  moves the whole field" overstates; corrected in §1 and in the knowledge doc.
- Notable confirmations beyond the row text: K-25's absence claim was
  re-proven independently (zero `value_offset(ValueTenant::HelixResidue)`
  callers anywhere; `mailbox_scan.rs:237-240` documents the Signed360 distance
  tier as "named; wired as they land"); K-9/K-10's committed JSON matches to
  full precision.

### Attack: 11 specs → 10 VACUOUS, 1 UNDERSPECIFIED. Zero SOUND.

Every v1 spec failed. The full per-spec findings are in the run output; the
§3 v2 specs above fold in every fix. The recurring defects, by house name:

| defect class | where it appeared |
|---|---|
| identity/tautology as a "silence half" | EV-1 (calm tiles = shift rounds to 0), EV-2 (90° sector: `circular == linear` for `\|a−b\| ≤ 128` by arithmetic) |
| `count > 0` implied by sampling design (`elimination_rate()` verbatim) | EV-2 |
| guard pre-satisfied by committed data | EV-3 (the two-regime guard) |
| monotone-but-dead-knob pass route | EV-4 (constant curve is monotone; a knob that never reaches the code yields one) |
| fixture cannot exist / pipeline destroys the phenomenon | EV-5 (`total_cloud_cover` 404 at pinned t; `sea_ice_cover` in neither fetcher; zonal-anomaly step erases bound-mass; Kelvin floors on fractions) |
| unsatisfiable gate constants (the rounding incident, rebuilt) | EV-6 (3-sig-fig prose "to 1e-4 relative"; Monte-Carlo numbers under an "exact" gate) |
| claim names a capability no criterion touches | EV-7a (top-k) |
| author-satisfiable anti-vacuity | EV-7a (heel-reject ∉ {0,100} %) |
| self-contradicting instrument; missing third arm | EV-7b (the lag-k alternative was the instrument its own citation bans; two arms only prove divisor-vs-coprime) |
| pass criterion that cannot fail + wrong n | EV-8 ("state and re-grade"; n = 200 000 pairs, not 1 038 240 gridpoints) |
| assertion implied by its own subject | EV-9 (equal-budget ratio; 0.352° = step/4) |
| stale-fixture pass route + three factors at once | EV-10 (`fetch_bg.py` hardcodes `t`, skips-if-exists — a winter run re-measures summer bytes) |
| physical-constant error creating a false-fail route | EV-1 (`cos(lat)` zonal spacing, up to 2.6× where winds are strong) |

### The meta-finding

The v1 specs were written by the same author who had just spent one document
(§12.10–§12.17) being corrected for exactly these failure modes, citing the
falsifiability rule throughout — and still went **0 for 11**. Pre-registration
review by independent adversarial readers is load-bearing, not ceremony:
naming a discipline is not applying it, and the author of a spec is
structurally the wrong person to find its vacuous pass routes.
