# Reader 1 — cascade.rs, rolling_floor.rs, hdr_cascade.rs (verbatim, 2026-09-05)

## `/home/user/ndarray/src/hpc/cascade.rs`
- `Band` enum (line 22-28): `Foveal, Near, Good, Weak, Reject`
- `expose()` (line 162-176): quarters of `threshold` t: `d<=t/4`→Foveal, `<=t/2`→Near, `<=t*3/4`→Good, `<=t`→Weak, else Reject
- `ShiftAlert` struct (line 32-38): `old_mu, new_mu, old_sigma, new_sigma, observations`
- `Cascade` state fields (line 104-109): `threshold: u64, vec_bytes, mu: f64, sigma: f64, observations: usize`
- `calibrate()` (line 137-160): threshold = `mu + 3.0*sigma` (k=3 fixed), from batch mean/var
- `observe()` (line 182-209): Welford online mu/sigma update; drift check line 198: `observations>10 && old_sigma>0.0 && |mu-old_mu| > 2.0*old_sigma` → emits `ShiftAlert`
- `recalibrate()` (line 211-215): sets mu/sigma from alert, `threshold = new_mu + 3.0*new_sigma`
- Stroke/stacking early exit: `query()` (line 218-309) — Stroke 1 uses `s1_bytes = (((vec_bytes/16).max(64)+63)&!63).min(vec_bytes)` prefix, computes `sigma_est` (binomial approx) and `sigma_pop` (empirical from `warmup_n=128.min(num_vectors)` warmup samples), `sigma = sigma_est.max(sigma_pop).max(1.0)`, reject bound `s1_reject = threshold + 3.0*sigma`; survivors carried to Stroke 2 full Hamming (line 291-306); small vectors (`vec_bytes<128`) skip cascade entirely (line 229-244)
- `PackedDatabase::cascade_query()` (line 583-647) is a 3-stroke variant with different scale factors per stroke (`estimate <= threshold + threshold/4` for stroke1, exact `threshold` for stroke2/3)
- No reservoir sampling present — warmup uses first `warmup_n=128` items directly
- Bucket/percentile mechanism: none — threshold+band based. `adaptive_resolution()` (line 527-534) is a match-based band selector by `(query_entropy, corpus_cv)`, not a bucket/percentile mechanism

## `/home/user/lance-graph/crates/perturbation-sim/src/rolling_floor.rs`
- `FloorBand` enum (line 33-44): `Stable, Watch, Concern, Warning, Alarm` — same quarter-of-threshold scheme as `Cascade::Band`
- `weyl_over_fiedler()` (line 52-65): `Δλ.max(0.0) / λ₂.abs()`, guarded by `SPECTRAL_GAP_FLOOR` → returns `FRAGMENTATION_SENTINEL` if `|λ₂| < floor` (never NaN)
- `RollingFloor` struct (line 72-78): fields `mu: f64, sigma: f64, pub k: f64, n: usize`
- `k` default: none hardcoded — caller supplies via `RollingFloor::new(k)` / `TierFloors::new(k)`; doc (line 75) suggests 2.0 ≈ 97.7% one-sided Gaussian; tests use k=2.0 and k=3.0
- `preheat()` (line 93-103): batch-calibrates mu/sigma from `samples: &[f64]`, sets `n = samples.len()` — how a coarser tier's floor warm-starts a colder finer-tier floor, invoked at `stack_early_exit()` line 230-235: `if self.floors[t+1].threshold() <= 0.0 { copy mu/sigma/n from floors[t] into floors[t+1] (n.max(1)) }`
- `observe()` (line 108-122): tests `band(x)==Alarm` BEFORE updating (line 109), then Welford update
- `threshold()` (line 125-127): `mu + k*sigma`
- `z()` (line 131-137): standardized exceedance `(x-mu)/sigma`, 0 if sigma<1e-12
- `band()` (line 140-161): quarters of threshold; uncalibrated (`t<=0.0`): `x>0.0`→Alarm else Stable
- `TierFloors` (line 192-256): `floors: [RollingFloor;4]` for L1..L4 (HEEL/HIP/TWIG/LEAF)
- `stack_early_exit()` (line 224-256): accumulates `stacked += inc` per tier, preheats next-cold floor, computes band+z, calls `observe()`, early-exits at first tier whose band==Alarm (line 239-247), returning `StackResult{exit_tier, stacked, band, early: t<3, z}`
- Jirak note (line 25-27): "the σ here is from a small, weakly-dependent tier sample, so the nominal CI is approximate — significance is the Jirak `n^(p/2−1)` rate, not a clean Gaussian tail. The floor is an operating threshold, not a proof."
- No bucket/percentile (Prozentrang) mechanism present

## `/home/user/lance-graph/crates/holograph/src/hdr_cascade.rs`
- Level architecture (line 8-42): L0 Belichtungsmesser (7-point sample, ~90% filtered) → L1 1-bit scan (~80% filtered) → L2 Stacked Popcount (early exit if impossible) → L3 Mexican Hat discrimination → optional Voyager Deep Field stacking fallback
- Constants: `DEFAULT_EXCITE=2000` (line 52), `DEFAULT_INHIBIT=5000` (line 55), `METER_POINTS: [usize;7] = [0,23,47,78,101,131,155]` (line 59, prime-spaced, `#[allow(dead_code)]`/unwired)
- `MexicanHat` struct (line 83-90): `excite: u32, inhibit: u32, inhibit_strength: f32` (default 0.5)
- `response()` (line 128-141): if `distance<excite`: `1.0 - distance/excite`; elif `distance<inhibit`: `t=(distance-excite)/(inhibit-excite)`, `-inhibit_strength*(1.0-t)`; else `0.0`
- `QualityTracker` (line 162-173): `ema: f32, sd_history: [u8;4], sd_idx: usize, threshold: u16, base_threshold: u16`; default base_threshold 2000
- `calculate_sweet_spot()` (line 200-211): mean bucket match: `0..=1`→base/2, `2..=3`→base*3/4, `4..=5`→base, `6..=7`→base*3/2, else→base; `sd_factor = 1.0 + sd/150.0`; final = `base*sd_factor as u16` — a bucket mechanism, no overflow handling beyond `_` catch-all
- `infer_trajectory()` (line 214-229): needs `sd_idx>=4`; `slope = (h[3]-h[0])/3` over 4-slot ring (`sd_history[self.sd_idx % 4]`); `slope>10`: `threshold = (threshold + slope*20).min(5000)`; `slope<-10`: `(threshold + slope*15).max(500)`
- `should_retreat()` (line 237-239): `current_quality < ema * 0.6`
- `update_quality()` (line 232-234): `ema = 0.85*ema + 0.15*batch_quality`
- `HdrCascade` state (line 286-299): `threshold_l0: f32 (=0.8), threshold_l1: u32 (=130), threshold_l2: u32 (=3000), batch_size (=64)`
- Per-level early exit in `search()` (line 371-399): L0 `meter.definitely_far(threshold_l0)`; L1 `count_differing_words > threshold_l1`; L2 `StackedPopcount::compute_with_threshold(...)` `None` = rejected
- `search_adaptive()` (line 402-452): per batch of 64, meters first item, `dynamic_threshold` via `calculate_sweet_spot`; quality early break (line 428-434) once `batch_count>=8 && i>batch_start+16` if `should_retreat`
- `classify_signal()` (line 625-632): `(mean, sd, distance)` → `Strong/Moderate/WeakButStackable/Noise`; `_ => Noise`
- `voyager_deep_field()` (line 475-525): capped collection up to `stack_size` (not reservoir); `superposition_clean()` (line 564-605) majority vote `n/2`
- `RollingWindow` (line 764-855): ring buffer with running `sum`/`sum_sq`; `is_coherent()` (line 843-845) `cv() < threshold`; eviction subtracts oldest
- No true reservoir sampling (Algorithm R) in any of the three files
