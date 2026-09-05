# Reader 3 — proprioception, EWA/temporal sandwich, Mexican hat, 16k search, mask surface (verbatim, 2026-09-05)

## `crates/lance-graph-contract/src/proprioception.rs`
- No `StateClassifier` trait in this file — module implements `ProprioceptionAxes`/`AnchorState`/`StateAnchor`
- 7 anchors + rungs (`ANCHOR_REGISTRY`, line 278-300+): `Intake` rung 5, `Focused` rung 4, `Rest` rung 3, `Flow` (coords `[0.7,0.5,0.3,0.6,0.9,0.6,0.7,0.2,0.8,0.7,0.7]`), `Observer`, `Balanced`, `Baseline` (enum line 161-176, `ALL` line 180-188)
- Nearest-anchor vs softmax: doc line 19-23 only; no concrete fn in range read
- Thresholds: `drive_ratio` cutoffs line 133-139 / 247-254 — `phi < 1.0` → `Explore`, `phi < 1.8` → `Exploit`, else `Reflect`. `STATE_DIMS = 11`, `CORE_AXES = 7`, `DRIVE_AXES = 4`

## `ndarray/src/hpc/pillar/temporal_sandwich.rs` (Pillar-8) / `ewa_sandwich_2d.rs` (Pillar-6)
- temporal: SPD preservation of `Σ_{t+1} = M_t·Σ_t·M_tᵀ` across Cardiac/Respiratory/Micro; `sandwich_update_3x3` line 170; `is_spd_3x3` line 206; `PILLAR_8_PSD_THRESHOLD: f64 = 0.0` (line 59, placeholder); `SIGMA_CARDIAC=0.05`, `SIGMA_RESPIRATORY=0.20`, `SIGMA_MICRO=0.001`; `N_PATHS=1000`, `N_SUBSTEPS=30`; TODO line 57-64 `calibrate-pillar-8-σ_temporal` (comment says 0.10, const is `0.0`)
- ewa: `ewa_sandwich_step_2d` line 118; `PILLAR_6_PSD_THRESHOLD = 0.10` (line 53); `SIGMA_STEP = 0.2`; `N_PATHS=1000`, `N_HOPS=10`, `SPD_EPS=1e-9`; TODO line 54-58 `calibrate-pillar-6-σ_step` — lowered from 0.999 to 0.10 "denormal-tolerant placeholder", σ_step drives Σ to denormal in <30 hops

## `ndarray/src/hpc/pillar/mexican_hat.rs` (Pillar-15)
- DEFERRED pending DoG kernel in `ndarray::hpc::dragonfly`
- κ band `σ_s/σ_c ∈ [1.5, 3.0]` (line 32-36); sweep `{1.1,1.3,1.5,1.8,2.0,2.5,3.0,4.0}` (line 54)
- Certifies (when active): DoG unimodality — single positive critical point, `DoG(0)>0`, decay to 0, exactly one root, second-derivative ratio budget
- `prove_pillar_15()` (line 111) returns placeholder `passed=true`, n_paths=0
- Activation gate: public `dog_eval(r, sigma_c, sigma_s) -> f32`

## `crates/holograph/src/width_16k/search.rs`
- Cascade diagram (line 7-26): L0 schema predicate → L1 Belichtungsmesser 7-point (comment only) → L2 block-masked StackedPopcount → L3 exact
- `passes_predicates` (line 251); `masked_distance` (line 393, no early exit); `masked_distance_with_threshold` (line 415) — per-block partial sums, `if total > threshold { return None }` (line 424-438)
- `search()` (line 447): predicate → thresholded distance → `partition_point` insert → tightens `current_threshold` to kth-best (line 483)
- `bloom_accelerated_search` (line 742), `rl_guided_search` (line 829) reuse the pattern
- No stride-sampling implementation in this file

## `ndarray/src/simd_int_ops.rs` mask surface
- `eq_u32_to_mask(values: &[u32], needle: u32, out_words: &mut [u64])` — line 562
- `eq_u32_strided_to_mask(...)` — line 629
- `mask_and(a, b, dst)` — 775; `mask_or` — 807; `mask_and_assign(dst, src)` — 835; `mask_or_assign` — 861; `mask_andnot(a, b, dst)` — 904; `mask_andnot_assign(a, b)` — 932
- `mask_ternlog<const IMM: i32>(a, b, c, dst)` — 983; `mask_ternlog_assign<const IMM: i32>(a, b, c)` — 1015
