# B3 — HDR / statistical substrate inventory (ndarray master) and its fit to a Moore/Morton spatial experiment

Checkout: `/home/user/_ro/ndarray`, HEAD `8afa92d` (Merge PR #354). Read-only. All paths relative to that checkout.
Grades: **VERIFIED-IN-CODE** (read the source line), **CLAIMED-IN-DOC** (stated in a doc/comment/blackboard, not re-checked by me), **ABSENT** (searched the named space and read the candidates; not found).
I did not run cargo; nothing here is a measurement. "Tests" means the test exists in source, not that I ran it.

Note on location: there is no `src/hpc/bitwise.rs`. `hpc::bitwise` is a re-export: `src/hpc/mod.rs:44 pub use crate::bitwise;` → `src/bitwise.rs` (813 lines). Also, `PowerSums`/`CrossPowerSums`/`checked_affine` live in `src/simd_masking_ops.rs` (9903 lines), not in `hpc/`; `MomentsU32` is in `src/hpc/statistics.rs:378`.

---

## 1. Popcount / Hamming kernels (`src/bitwise.rs`)

| item | where | semantics | exact? | alloc | SIMD | tests |
|---|---|---|---|---|---|---|
| `hamming_distance_raw(a,b)` | bitwise.rs:192 → `dispatch_hamming` :270 | popcount(a XOR b) over common prefix | exact | none | runtime: VPOPCNTDQ+BW → `kernels_avx512::hamming_distance`; AVX-512BW → `hamming_avx512bw` (:100, vpshufb nibble LUT, u8 acc flushed every 30 iters); else `hamming_u64x8` (:77, `U64x8::popcnt` polyfill, VPOPCNTQ/vcntq/i8x16_popcnt/count_ones per realization) | per-tier + cross-tier + 64KB stress vs byte-wise oracle (:520–:690 region; `test_all_tiers_agree`, `test_large_hamming_all_tiers`, `test_hamming_u64x8_matches_reference`) |
| `hamming_distance_within(a,b,max) -> Option<u64>` | bitwise.rs:221 | `Some(d)` iff d ≤ max; checks after every 256-byte block (`HAMMING_WITHIN_BLOCK` :205) | **exact** (monotone running total; reject costs ≤ 1 block past exhaustion) | none | each block through `dispatch_hamming` | `test_hamming_distance_within_is_exact` (on/below/above d, 0, u64::MAX, all lengths across 256-byte boundaries). `..._rejects_on_the_first_block` — note: its own comment admits the early exit is not observable through the return value; it proves the answer, not that bytes were skipped |
| `popcount_raw(a)` | :237 → `dispatch_popcount` :285 | total set bits | exact | none | VPOPCNTDQ → AVX-512BW → **`popcount_scalar` byte loop** (:43) — asymmetric with hamming, which falls back to the U64x8 polyfill | tier tests |
| `popcount_batch_u64(words)` | :328 | Σ popcount over u64 words | exact | none | `U64x8::popcnt` polyfill, chunked by LANES, tail scalar | `test_popcount_batch_u64_matches_reference` |
| `popcount_per_word`, `masked_popcount_batch`, `masked_popcount_total` | :340/:346/:351 | per-word / (w & mask) popcounts | exact | `Vec<u32>` for the per-word forms | **scalar iterator** (no SIMD) | `test_popcount_per_word`, `test_masked_popcount` |
| `hamming_batch_raw`, `hamming_top_k_raw` | :245/:253 | query-vs-rows distances; top-k via `select_nth_unstable` | exact | `Vec<u64>` of num_rows + `Vec<(usize,u64)>` | batch kernel only on VPOPCNTDQ+BW, else per-row dispatch | `test_hamming_batch`, `test_hamming_top_k`, `test_raw_slice_apis` |

Doc claim "zero allocation" on `hamming_batch_raw` (:243) is wrong as written: it returns a `Vec` of `num_rows` (VERIFIED-IN-CODE: :246 returns `dispatch_hamming_batch(...)`, which `collect()`s at :315). Zero allocation *inside the kernel*, one output vector.

## 2. HDR cascade — progressive strokes (`src/hpc/cascade.rs`, 1099 lines)

### 2.1 `Cascade` state (cascade.rs:103)
`threshold: u64`, `vec_bytes`, private `mu, sigma: f64`, `observations`. f64 Welford, not integer.

- `calibrate(distances, vec_bytes)` :137 — two-pass f64 mean / population σ; **threshold = (μ + 3σ) as u64** (:152). Upper-tail: everything below μ+3σ is "accepted". Exact arithmetic on the given sample, but σ-to-threshold mapping assumes normal-ish noise.
- `expose(d) -> Band` :162 — **bucket/band assignment**: Foveal `d ≤ t/4`, Near `≤ t/2`, Good `≤ 3t/4`, Weak `≤ t`, Reject above (:165–:175). Fixed quarter-fractions of the threshold, **not** σ-lattice points. Tested `cascade_expose_bands`.
- `test(a,b)` :178 — `hamming_distance_within(a,b,threshold).is_some()`: exact single-pair accept.
- `observe(d) -> Option<ShiftAlert>` :182 — scalar f64 Welford; alert when n>10, old σ>0, |Δμ| > 2·old σ (:197). `recalibrate(alert)` :262 resets μ,σ, threshold = μ+3σ.
- `observe_batch(&[u32])` :226 — batch reduced by exact `moments_u32`, merged by Chan–Golub–LeVeque parallel variance (:236–:244). Matches sequential `observe` to f64 rounding (tested `observe_batch_matches_sequential_observe`, shards in any order `observe_batch_shards_merge_in_any_order`, alert can-fire/stay-silent `observe_batch_alerts_on_a_shift_only`, singleton gate parity `observe_batch_singleton_matches_observe_alert_gate`).

### 2.2 `Cascade::query` — the three strokes (cascade.rs:279)
- **Small vectors (< 128 bytes)**: no cascade, exact `hamming_distance_within` per row (:290–:306).
- **Preheating / "128-candidate warmup"** (VERIFIED-IN-CODE): `warmup_n = 128.min(num_vectors)` (:287). The first 128 rows' prefix distances are scaled (`estimate = d·scale1`, scale1 = vec_bytes/s1_bytes) into `warmup_dists` (allocates `Vec::with_capacity(128)`) and their **population σ** computed (:314–:337). Also an analytic binomial σ estimate `sigma_est = vec_bytes·sqrt(8 p(1-p)/s1_bytes)`, p = threshold/total_bits clamped [0.001,0.999] (:310–:312). Effective σ = max(sigma_est, sigma_pop, 1.0) (:338). Note the warm-up sample is the **first 128 rows in storage order**, not a random sample — biased if the database is sorted/clustered.
- **Stroke 1** (:340–:350): prefix of `s1_bytes = (((vec_bytes/16).max(64)+63) & !63).min(vec_bytes)` (:285; =128 at 2048) per row; keep iff scaled estimate ≤ `threshold + 3σ`. **APPROXIMATE — recall not guaranteed** (VERIFIED-IN-CODE doc :268–:277; pinned by `stroke1_drops_a_prefix_heavy_true_hit` :813, where a true 400-bit hit with all differences in the prefix is dropped while the same 400 bits at the tail are kept). Survivors allocate `Vec::with_capacity(num_vectors/20)`.
- **Stroke 2** (:352–:376): exact. Budget `threshold.checked_sub(d_prefix)`; rest scanned with `hamming_distance_within(rest, budget)` → **exact threshold-limited early exit**; result carries the true full distance and a `Band`. Pinned by `query_exact_strokes_respect_the_prefix_budget` with `straddling_database` (k = 590…600 around threshold 600; granting the whole threshold to the rest would admit 593/597/600).
- "Stroke 3" in `Cascade` is the optional precision tier (`query_precise` :417 → `apply_precision_tier` :476): Vnni cosine (scalar i8 dot), F32/BF16 dequant cosine (allocates two `Vec<f32>`), DeltaXor blend, BF16Hamming weighted XOR. **All scalar** (`dot_i8_scalar`, `dot_f32_scalar`, `bf16_hamming_scalar` — "no SIMD dependency" header :431). Re-sorts by `precise` descending.
- Return: hits within threshold, **unsorted** in `query` (only `query_candidates` and packed `cascade_query` sort).

Net exactness of `Cascade::query`: **every returned hit is exact (true distance, ≤ threshold); the hit set is a subset of the true set**. It is **not** an exact oracle. For an exact scan the doc says call `hamming_distance_within` per candidate (:276).

### 2.3 `query_candidates` (cascade.rs:385) — CLAM bridge
Trusts a caller-supplied distance when `clam_dist > 0` **without recomputing** (comment says "Verify" but the code does not, :393–:401); recomputes only when `clam_dist == 0`. So it is exact only if the caller's distances are exact. Sorts by hamming.

### 2.4 `adaptive_resolution(entropy, cv) -> Band` (cascade.rs:592)
Pure lookup on two floats with hand constants (0.2/0.5/0.3/0.5). Not calibrated, not derived; doc cites Rényi/CLAM/Berry-Esseen but code is a 4-arm match.

### 2.5 `cascade::PackedDatabase` (cascade.rs:607) — width-derived three-way split
`pack` :619: s1 as above, `s2_end = (vec_bytes/4).max(s1).min(vec_bytes)`, s3 = rest; three contiguous stroke buffers (allocates all three). `cascade_query(query, &Cascade, top_k)` :653:
- Stroke 1: keep iff scaled estimate ≤ `threshold + threshold/4` (:666) — a **different** slack rule than `Cascade::query` (fixed +25%, no σ warm-up). **Approximate.**
- Stroke 2: keep iff `(d1+d2)` scaled to full width ≤ `threshold` (:676–:683) — **also an estimate prune, approximate** (a candidate whose s1+s2 bits are dense and tail is clean is dropped).
- Stroke 3: exact budgeted early exit `threshold − d12` via `hamming_distance_within` (:694–:697); pinned `packed_stroke3_respects_the_prefix_budget`.
- Sort by distance, truncate to top_k.

### 2.6 `packed::PackedDatabase` (`src/hpc/packed.rs`, 427 lines) — the fixed 2048-byte packer
Constants STROKE1/2/3_BYTES = 128/384/1536, FINGERPRINT_BYTES = 2048 (packed.rs:55–61). `pack` (:116) **panics unless row_bytes == 2048** (:117). Holds an `index: Vec<u32>` of original ids. Software prefetch `prefetcht0/t1` via inline asm on x86_64 (:29–:51, SAFETY-commented; no-op elsewhere).
`cascade_query(query, reject_threshold_s1, reject_threshold_s12, k)` (:192):
- Strokes 1 and 2 compare **raw (unscaled) partial distances** against caller thresholds (`d1 ≤ s1`, `d1+d2 ≤ s12`). A partial Hamming distance is a **lower bound** of the full one, so if the caller passes `s1 = s12 = T` this is an **exact prune for the predicate "d ≤ T"** — a genuinely different (exact-capable) contract from both `Cascade::query` and `cascade::PackedDatabase::cascade_query`, which prune on *scaled estimates* (approximate). With tighter caller thresholds it is approximate in the same way (VERIFIED-IN-CODE).
- Stroke 3 uses `hamming_distance_raw` (no early exit, no final-threshold test) — every s12 survivor is fully scanned and returned; then sort + truncate k (:247–:251). For a top-k query it is exact only if the s1/s12 thresholds do not cut a true top-k member.
- `brute_force_query` (:255) is the exact reference. Test `test_cascade_vs_brute_force_consistency` (thresholds u64::MAX) only proves agreement in the degenerate no-prune case.

### 2.7 "PackedDatabase is three types, not one" + the PR #354 stroke-split pin (VERIFIED-IN-CODE + git)
- `hpc::packed::PackedDatabase` (packed.rs:85), `hpc::cascade::PackedDatabase` (cascade.rs:607), `hpc::cam_pq::PackedDatabase` (cam_pq.rs:96, 6-byte CAM codes, shares only the name). Doc cross-links added in both stroke packers (packed.rs:77–83, cascade.rs:600–606).
- Pin: `packed::tests::stroke_split_equals_the_cascade_split_at_2048` (packed.rs:291) asserts cascade's width-derived split at 2048 is exactly (128, 384, 1536) and the bytes agree per candidate. Commit `035d73b`, merged in `8afa92d` (PR #354). Blackboard `.claude/blackboard.md:1–4`.
- The pin proves **layout** agreement only. The two query paths still have different prune contracts (§2.5 vs §2.6) — the pin does not and cannot equalize them.
- Same PR **withdrew** `mask_row_window` (2-D row window, refuse/zero-fill) and `bit_sliced_weighted_count` (Σ_b 2^b(popcnt(w∧pos_b) − popcnt(w∧neg_b))), added in `a386c9e` and reverted in `4a6dd93`/`547b8aa`. Blackboard:4 reason: *"Signed masks are not part of this facade. A 256×256 tile is the palette256 × palette256 address (u8:u8), answered by 256×256 distance/compose tables, not by geometric windows."* **This is load-bearing for the spatial question (§6).**

## 3. Integer moments (exact, mergeable)

### 3.1 `MomentsU32` (`src/hpc/statistics.rs:378`) + `moments_u32` (:470)
`{ n: u64, sum: u128, sum_sq: u128 }`. `observe` (:391, scalar), `merge` (:401, plain `+` — wraps/panics only past u128, not checked), `mean()`/`variance()` → f64 at the end only; variance forms `n·Σx² − (Σx)²` exactly in u128 when it fits, else centres on ⌊mean⌋ (:426–:466). `moments_u32` (:470): **SIMD via `U64x8`** — 8 lanes, Σx and Σx² split into lo/hi 32-bit halves so u64 lanes cannot overflow, drained every 2^28 chunks (:474–:494); tail scalar. Re-exported `ndarray::simd::{moments_u32, MomentsU32}` (simd.rs:671).
Exact, associative, commutative (tests `moments_u32_is_exact`, `merge_is_exact_and_order_independent`, `moments_u32_does_not_overflow_at_u32_max`, `variance_fallback_does_not_cancel`, `mean_and_variance_match_two_pass`). Allocation: none. **Domain: unsigned u32 only.**

### 3.2 `PowerSums` / `CrossPowerSums` (`src/simd_masking_ops.rs:2118`, `:2372`)
- `PowerSums { n: u64, sum: i64, sum_sq: u128 }`, `#[repr(C)]` 32 bytes, layout pinned by const asserts (:2131–:2141). **Signed i32 domain** (unlike MomentsU32).
- Keyed folds `masked_group_power_sums_i32{,_via,_pair}` (:2169/:2204/:2242): for every mask-selected row fold `values[i]` into `out[key]`. Wrapping adds; `sum` exact while < 2^32 rows per group. All go through `group_walk` (:1730) = **scalar** trailing-zeros bit walk over set mask bits ("scalar by necessity: … scatter-reduce", :1717–:1721). Cost ∝ popcount(mask). No allocation (caller owns `out`).
- `checked_merge` (:2294) — exact union, `None` on overflow.
- **`checked_affine(a, b)`** (:2328) — power sums of `a·x + b` from the sums alone: n'=n, Σx'=aΣx+nb, Σx'²=a²Σx²+2abΣx+nb²; i128 checked throughout; `None` instead of wrap. Exact. Caveat in doc (:2316–:2318): valid only if the group key does not depend on the transformed value.
- `CrossPowerSums { n, sum_x, sum_y, sum_x_sq, sum_y_sq, sum_xy: i128 }` (:2372) — degree-≤2 joint sufficient statistics (covariance, Pearson r, OLS line). `x()`/`y()` marginals, `checked_merge`, `checked_affine(m:[i64;4], t:[i64;2])` (:2471) = sums of `A·(x,y)+t` (S' = AS + nt, M' = AMAᵀ + AStᵀ + tSᵀAᵀ + n ttᵀ). Keyed folds `masked_group_cross_power_sums_i32{,_via,_pair}` (:2545/:2569/:2593), same scalar walker.
- **Bounded u8 tiles**: `BOUNDED_TILE_ROWS = 1 << 16 = 65,536` (:2633) — exactly a 64K field. Within one tile, u8 power sums fit u32 words (const-asserted :2635–:2641). `masked_group_bounded_power_sums_u8{,_via,_pair}` (:2730…) and cross variants (:2846…) fold into 16-byte registers `[n, Σx, Σx², reserved]` / rail pair `[Σy, Σy², Σxy, reserved]`; refuse a tile > 65,536 rows before writing (`BoundedFoldError::TileTooLarge`). `widen_bounded_*` (:2959/:2971) lossless; `fold_bounded_{,cross_}power_sums_tiles` (:3030/:3061) tile larger populations and merge with `checked_merge`, refusing a whole tile on overflow (`MergeOverflow`). Scalar (same walker). All re-exported through `ndarray::simd` (simd.rs:829–:944).
- Blackboard 2026-10-08 entry documents `checked_affine` landing (CLAIMED-IN-DOC; code read and matches).

### 3.3 Float statistics (`src/hpc/statistics.rs:21`, trait `Statistics<A: Float>`)
`median` (sort, allocates), `variance` (two-pass), `var_axis`/`std_axis`, **`percentile(p)`** (:171, sort + **linear interpolation** between nearest ranks — a different quantile rule from `quantile_of_sorted`'s integer rank), `sorted`, `argmin/argmax`, `top_k` (full sort, allocates), `cumsum`, `cosine_similarity`, `norm`. All **scalar iterator** code, f32/f64 only. tests in the same file (`test_percentile` etc.).

### 3.4 Reliability (`src/hpc/reliability.rs`, 406 lines)
`pearson` (:40), `spearman` (:98, average ranks with ties), `cronbach_alpha` (:119), `icc_a1` (:164), `FidelityReport::compute` (:256). `&[f64]` scalar. **`pearson` returns 0.0 on n<2 or variance < 1e-12** (:42–:58) — conflates "degenerate" with "uncorrelated"; a caller using it as a spatial-autocorrelation statistic must guard degeneracy itself. 11 tests (:304–:390).

## 4. Rolling floor, reservoir, σ-lattice, quantiles (`src/hpc/rolling_floor.rs`, 1620 lines)

Harvested from lance-graph `graph/blasgraph/hdr.rs` (module doc :4); arithmetic = `MomentsU32`. **Not re-exported through `ndarray::simd`** (grep of simd.rs: no `rolling_floor`); reached as `ndarray::hpc::rolling_floor::*`.

- `isqrt_u32` (:89), `isqrt_u128` (:113) — exact integer Newton floor-sqrt.
- `rank_per_10000(len, per_10000)` (:140) — integer rank `⌊p·len/10000⌋` clamped to len−1. `quantile_of_sorted(sorted, per_10000)` (:159) — **empirical quantile, nearest-rank, no interpolation, integer**. Test `integer_rank_rule_against_the_reference` pins where it diverges from the old f32 rule.
- **`ReservoirU32`** (:185) — Vitter Algorithm R, **deterministic**: replacement index from a splitmix64 hash of the observation count `seen` (:203–:213, :275–:281). Same stream → same reservoir; **no merge of two reservoirs**; order-dependent. `quantile` sorts a clone each call (O(len log len), allocates). `skewness` = Pearson's 2nd skewness 3(μ−median)/σ integer (:257); `kurtosis` ×100 exact in u128 with a 16.16 fixed-point fallback (:267; helpers after :800). Test `reservoir_is_deterministic_and_bounded`.
- **`SigmaLevel(u8)`** (:327) — point `k` quarter-σ **below** the noise floor; `gaussian_tail_per_10000` (:340) from a fixed table Φ(−k/4) for k=0..16 (:291), k>16 → 0 (sample minimum).
- `EmpiricalShape` (:366) — sorted sample + own (μ_s, σ_s); `locate(level, μ, σ) = μ + ⌊(x−μ_s)·σ/σ_s⌋` (:408). Allocates the sorted copy.
- `RollingFloor` (:506): anchor (μ,σ) + `MomentsU32` + reservoir (cap 1000) + `Shape` + skew/kurt. Constants: `RESERVOIR_CAP=1000`, `EVAL_CADENCE=1000`, `MIN_SHAPE_SAMPLES=100`, `FIRST_CUT=SigmaLevel(4)` (1σ), `LAST_CUT=SigmaLevel(12)` (3σ) (:518–:531). These are policy pins carried from the reference, not derived (CLAIMED-IN-DOC :4–:6).
  - `for_width(bits)` binomial prior μ=bits/2, σ=⌊√(bits/4)⌋ (:554); `calibrate` (:567).
  - `observe` (:583) every observation updates moments + reservoir; checkpoint every 1000 after the first 1000 (`at_checkpoint` :629). `observe_batch` (:599) reproduces the scalar stream exactly (moments merged per chunk, reservoir fed in order, stops at the first shift). Tests `scalar_equals_singleton_batch`, `arbitrary_batching_equals_scalar_stream`.
  - Checkpoint (:635): **drift first** — `CutDrift` from motion of the 1σ and 3σ cuts read through the active shape; shift if |Δμ| > σ_a/2 or |Δσ| > σ_a/4 (:646); "cuts decide WHETHER, moments decide WHERE" — the shift carries the exact running coordinates (:640–:656, test `cuts_decide_whether_moments_decide_where`). **Shape** only on a drift-free checkpoint: normality window |skew|<2 and 200<kurt<500 (`shape_is_normal` :674) else `Shape::Empirical`.
  - `recalibrate` (:622) adopts the shift, forgets moments+reservoir, **keeps shape**.
  - `threshold(level)` (:743), `thresholds([..;N])` (:749), **`shade(x, levels)`** (:757) = number of levels whose threshold lies **strictly above** x ("lower distance is stronger"). Thresholds are derived on demand from running moments, never stored. Gaussian: μ_t − ⌊k·σ_t/4⌋, saturating at 0.
  - **Direction**: the whole lattice is **lower-tail only** (μ − kσ/4). There is no upper-tail `SigmaLevel`. An observable whose "interesting" direction is *large* (e.g. a high ring count) must be reflected (`max − x`) before feeding, or the API extended.
  - Domain: u32, unsigned. Exactness: moments exact; thresholds integer-floored; empirical rank resolution bounded by the 1000-sample reservoir (k ≥ 13 all read the sample minimum, doc :317). The reservoir makes the *shape* a deterministic **sample estimate**, not exact.
- Tests: ~40 in :900–:1620 (cut drift translation vs dilation, boundary strictness, full-u32-range no-overflow, kurtosis fallback agreement, variance floor exact vs centred form, legacy Welford parity at checkpoints).
- `examples/hdr_rolling_floor_bench.rs` (148 lines): times (1) 2048-byte Hamming, (2a/2b) MomentsU32 scalar vs batch, (3) reservoir, (4a/4b) RollingFloor scalar vs batch, (5) shape sort+median+kurtosis, (6) empirical locate ×8, (7) Gaussian thresholds ×8, (8) shade ×8. It prints `avx512f/avx2` cfg flags. It reports ns/op only — no committed numbers and no assertions (so it is a timer, not a gate). I did not run it.

## 5. SIMD realization summary (who is actually vectorized)
- Vectorized: `hamming_distance_raw` (runtime VPOPCNTDQ / AVX-512BW, else U64x8 polyfill), `popcount_batch_u64` (U64x8), `moments_u32` (U64x8), `mask_ternlog*` / `mask_and/or/xor/andnot` (U64x8), compare-to-mask families (per-backend; see `.claude/knowledge/agnostic-surface-cpu-matrix.md` §N.13 for the per-arch table), `mask_shift_morton` pass 1 (U64x8; pass 2 carry scalar).
- Scalar by design: everything through `group_walk` (all keyed sums/counts/power sums/bounded folds), `Cascade` precision tier, `Cascade` warm-up/calibrate f64 loops, `RollingFloor`, reservoir, `EmpiricalShape`, `Statistics` trait, `reliability`, `masked_popcount_*`, `popcount_raw` below AVX-512BW.
- Caveat (doc §N.14 #4, CLAIMED-IN-DOC): `U64x8::popcnt` on AVX-512 is real VPOPCNTQ only when `avx512vpopcntdq` is compiled in, which the v4 config does not enable — so "AVX-512 popcount" in `popcount_batch_u64`/`mask_ternlog_popcount` is per-lane `count_ones` on the default v4 tier.
- `.claude/knowledge/agnostic-surface-cpu-matrix.md` §S (:1522) audits the Belichtungsmesser: no raw intrinsics; per-candidate kernels only; **no cross-candidate lane parallelism**; early exit done (#323); rolling floor "not a SIMD candidate".

## 6. Doc drift found
- `.claude/knowledge/cascade_search.md`: Band "Foveal top 5% / Near 5–25% / Good 25–60% / Weak 60–90%" — **those percentages are not in code**; bands are fixed fractions (¼,½,¾,1) of the threshold (cascade.rs:165–:175). "~84% rejection rate" is unmeasured in code. The doc omits that Stroke 1 can drop true hits and omits `observe_batch`. Its PackedDatabase section describes only the 2048 split.
- `agnostic-surface-cpu-matrix.md` §G still lists `hamming_distance_raw, popcount_raw` as "TD-T-? needs audit"; §N.12 has a post-#323 correction table — the doc carries both states.
- `bitwise.rs:243` "zero allocation" on `hamming_batch_raw` (returns a Vec).
- `cascade.rs:396` comment "Verify with full Hamming" on a branch that does not verify.

## 7. Spatial-adjacent facilities that already exist (read because the question needs them)
- **`mask_shift_morton(src, dir, dst)`** (`src/simd_masking_ops.rs:5384`, `MortonDir` :5197 = PosQ/NegQ/PosR/NegR). Morton-keyed 2-D bitmask field; word = 8×8 block; requires `n_words = 4^k` (a 64K-cell field = 1024 words = 4^5 ✓). OR-accumulates `src` shifted one cell along one axis into `dst`; **no wrap** at the far edge (cells shift off and vanish); diagonals are documented as **two calls via a scratch buffer** (:5347–:5351). Pass 1 vectorized (U64x8), pass 2 carry scalar. A sub-span is treated as its own field: boundary carries are dropped (:5352–:5362). Exact (it is a permutation + drop). Re-exported through `ndarray::simd`. Doc-claimed measurement (agnostic-surface §N.11, CLAIMED-IN-DOC): full-field −14% vs modelled ~98%; −66% when restricted to a trie node's word span.
- **`mask_ternlog::<IMM>` family** + `ternlog::{XOR3 = 0x96, MAJ3 = 0xE8, …}` (simd.rs:638–:662): bitwise full-adder sum and carry over whole mask planes, vectorized. These are the gates a bit-sliced (carry-save) per-cell counter is built from — but **no named counter primitive exists** (see §9).
- `mask_ternlog_popcount` / `popcount_batch_u64` — whole-field (or word-range) counts, not per-cell counts.
- Examples (not library, not gates for this crate): `examples/hex_tenant_mq_probe.rs` — a **65,536-row = 256×256 axial hex field, row = Morton(q,r)**, spread step chained through `mask_shift_morton` + `mask_ternlog_assign::<AND3>`, gates "Morton-arm spread == independent row-major axial BFS" and zero hot-path heap bytes (its own doc :1–:48; I did not run it). `examples/morton_cascade_probe.rs` — Morton 4×4-tile quadtree with per-node **min/max** pruning (an **exact** bound-based cascade: prune only when the node's [min,max] cannot meet the band; PASS = cascade count == brute-force count, per its doc). `examples/life.rs` — upstream ndarray Game of Life on `Array2<u8>` with slice-shifted neighbour sums (a Moore count, dense u8, not bitmask, not Morton).
- `src/hpc/pillar/mexican_hat.rs` — Pillar-15 Mexican-hat / DoG centre-surround **DEFERRED**, no kernel ("pending Mexican-hat resonance kernel", pillar/mod.rs:126–:128; blackboard 2026-10-08 `deferred` flag). `src/hpc/styles/lsi.rs:46 mexican_hat(distance)` is a 5-step banding of one scalar distance by Gaussian quantile constants (0.6745/1.6449/2.3263), not a spatial operator.
- Searched for Moran/Geary/autocorrelation/variogram/neighbour-count/bit-sliced counter across `src/` and `examples/` (grep, then read every hit): **ABSENT** (the only bit-sliced item, `bit_sliced_weighted_count`, was reverted in PR #354).

## 8. Fit to a Moore/Morton spatial experiment (64K field; ring counts; centre vs surround; local learning thresholds)

### 8.0 First: what HDR is, and is not
The HDR cascade answers **"which stored vectors are within Hamming distance T of this query"** — nearest-neighbour retrieval in a 16K-bit space. A spatial experiment asks **"how does a cell's value relate to the values of its lattice neighbours"** — spatial autocorrelation / local contrast. The cascade's machinery (prefix strokes, σ warm-up over the first 128 rows, budgeted early exit) is tuned to the first question: prefixes of a *vector*, scaled by `vec_bytes/s1_bytes` under a binomial-independence assumption. A Morton-ordered field's "prefix" is a spatial quadrant, not an unbiased sample of bits, so the scaling assumption is violated outright. **And HDR Stroke 1 is approximate**: `Cascade::query` and `cascade::PackedDatabase::cascade_query` can and do drop true hits (pinned by `stroke1_drops_a_prefix_heavy_true_hit`). **Neither may be used as an exact oracle** for any spatial ground truth; only `hamming_distance_within` / `hamming_distance_raw` / `packed::PackedDatabase::brute_force_query` are exact.

### 8.1 Reusable UNCHANGED (the observable is already an integer the facility accepts)
| facility | why it fits unchanged | caveat |
|---|---|---|
| `popcount_batch_u64`, `mask_ternlog_popcount` | whole-field / word-range live counts of a Morton bitmask (e.g. total centres, total "surround active") | counts the field, not each cell |
| `mask_shift_morton` + `mask_ternlog`/`mask_and/or/xor` | build the 8 Moore neighbour planes (4 axis shifts + 4 diagonals as 2-call compositions) and combine them, exactly, vectorized | no wrap (edge cells see fewer neighbours — the experiment must decide whether that is wanted; there is no torus option); diagonals cost a scratch plane; needs `4^k` words |
| `MomentsU32` / `moments_u32` | exact Σ, Σ² of any per-cell u32 observable (ring counts 0..8k fit trivially); merge across tiles/threads exactly | f64 only at `mean()/variance()` |
| `PowerSums` + `masked_group_power_sums_i32{,_via,_pair}` | per-group (e.g. per trie node, per label) exact n/Σ/Σ² of a signed observable such as `centre − mean(surround)`; `_pair` keys give a 2-D group address `hi*stride+lo` | scalar walker, cost ∝ mask popcount; keys must not depend on the folded value |
| `masked_group_bounded_power_sums_u8*` with `BOUNDED_TILE_ROWS = 65,536` | a 64K field of u8 observables is **exactly one bounded tile**; u32 register words provably cannot overflow | refuses > 65,536 rows; fold larger fields with `fold_bounded_*_tiles` |
| `CrossPowerSums` + `masked_group_cross_power_sums_i32` / bounded cross u8 | the **degree-2 joint statistics of (centre, surround)** pairs per group → covariance, Pearson r, OLS slope = a Moran-style centre–surround association, computed exactly from integer sums | no projection helper (r, slope) is shipped; the caller divides |
| `PowerSums::checked_affine`, `CrossPowerSums::checked_affine` | re-express statistics after `x → a·x + b` (e.g. ring count → 8k − count, or centring by an integer offset; a 2×2 integer mix of (centre, surround)) without re-reading rows; exact, overflow-refusing | integer coefficients only; group key must not depend on the transformed value |
| `quantile_of_sorted`, `rank_per_10000`, `isqrt_*` | exact nearest-rank empirical quantiles of a u32 observable | caller sorts; no interpolation |
| `ReservoirU32` | deterministic bounded sample of a long per-cell stream | order-dependent, no merge — not usable across shard-parallel tiles |
| `hamming_distance_within` | exact threshold-limited early exit when comparing two field *states* (e.g. "did the field change by ≤ T cells between steps") | it measures whole-field difference, not neighbourhood structure |
| `reliability::{pearson, spearman}` | ground-truth check of a centre–surround statistic vs a reference | f64, returns 0.0 on degeneracy |

### 8.2 Reusable only with a DIFFERENT OBSERVABLE plugged in (mechanism generic, semantics not)
- **`RollingFloor` (σ-lattice, shade, drift/shape checkpoints)** — accepts any u32 stream, so a per-cell ring count or a centre–surround contrast can be fed. But: (a) the lattice is **lower-tail only** ("lower distance is stronger", `shade` counts thresholds strictly above x) — a "high count is significant" observable must be reflected (`8k − count`) or offset to unsigned; (b) `for_width` is a **Hamming-specific binomial prior** (μ = bits/2) — for ring counts use `calibrate` or `from_params` with a prior derived for the count (e.g. Binomial(8k, p) for a Bernoulli(p) field); (c) one floor = one global distribution. A **local** learning threshold (per neighbourhood, per trie node) would need one `RollingFloor` per group — feasible but each carries a 1000-sample reservoir `Vec`, so 1024 nodes ≈ 4 MB of reservoir, and every checkpoint sorts. The mergeable part (moments) shards; the reservoir does not.
- **`Cascade::calibrate` / `observe` / `observe_batch` / `ShiftAlert`** — f64 Welford + μ+3σ upper threshold + 2σ drift alert on any u32 stream. Usable as a quick global drift detector for a spatial statistic; but its threshold is "accept below μ+3σ", the Hamming-retrieval polarity.
- **`Cascade::expose` → `Band`** — bucket assignment by fixed fractions (¼,½,¾,1) of one threshold. Reusable as a 5-way bucketing of any u32, but those fractions have no statistical meaning for a ring count; the σ-lattice (`RollingFloor::shade`) is the principled bucketing.
- **Bounded keyed folds with a spatial key**: the `_pair` / `_via` addressing takes any u32 key lane; using the Morton trie-node id (`morton_index >> (2·L)`) as the key gives per-node local statistics in one pass — the key lane must be materialized (or computed per call); there is no built-in "group = trie node of row i" address.

### 8.3 NOT usable (approximate, or a different problem)
- **`Cascade::query`, `cascade::PackedDatabase::cascade_query`** — approximate retrieval (Stroke 1 estimate prune; packed Stroke 2 is also an estimate prune). Wrong problem (vector NN, not lattice adjacency) and not an oracle. Do not use as ground truth or as a spatial neighbour finder.
- **The 128-candidate preheating** — σ from the **first 128 rows in storage order**: on a Morton field that is one 16×8-ish sub-block, a spatially correlated sample, not a population sample. It would estimate local σ of one corner, not the field.
- **Stroke split / `PackedDatabase` (both stroke packers)** — they split the *bytes of one vector*; a spatial field has no "vector prefix" whose popcount scales to the whole. The PR #354 pin is a layout fact about 2048-byte fingerprints and says nothing about spatial data.
- **`query_candidates`** — trusts caller distances unverified when > 0.
- **`adaptive_resolution`** — hand constants, entropy/CV of a query vector.
- **`Statistics::percentile`** for gates — interpolating f64 rule; mixing it with `quantile_of_sorted` would give two quantile definitions in one experiment.

### 8.4 What is MISSING for the spatial case (named precisely)
1. **A per-cell neighbour COUNT (ring count) primitive.** Everything in-tree counts a whole field (`popcount_batch_u64`) or per mask-selected row into keyed buckets (`group_walk`, scalar, cost ∝ selected rows). There is no op that produces, for every cell, the integer number of set Moore neighbours. The obvious realization — a **bit-sliced carry-save adder** over the 8 shifted planes using `mask_ternlog::<XOR3>` (sum) and `::<MAJ3>` (carry), giving ⌈log2(8k+1)⌉ count bit-planes — is constructible from public API today (ABSENT as a named, tested primitive). The closest thing ever written, `bit_sliced_weighted_count`, was a different shape (a weighted popcount of ONE window, signed planes) and was **withdrawn** in PR #354 with an explicit ruling against signed masks/geometric windows in this facade (blackboard.md:4).
2. **Radius-k rings / Chebyshev neighbourhoods beyond 1** — only unit axis shifts exist; radius k = repeated shifts (k·4 axis + diagonal compositions), each a full pass; no ring-of-radius-k op, no wrap/torus mode.
3. **A Moore (8-neighbour) helper** — `MortonDir` is the 4 axis moves; the module doc and the hex probe compose the 2 hex diagonals; the other 2 (Moore) diagonals are equally compositional but not provided.
4. **An upper-tail σ-lattice / two-sided threshold** in `RollingFloor` (or the reflect convention, documented) — needed if "centre significantly above surround" is the event.
5. **Per-group (local) rolling floors** — a mergeable, reservoir-free local threshold. Moments per group already exist (`PowerSums` keyed folds); what is missing is a `SigmaLevel`→threshold read straight from a `PowerSums` (Gaussian frame) so local learning thresholds need no per-group reservoir.
6. **Projection helpers from `CrossPowerSums` to covariance / r / slope** (and a Moran's-I-style normalization) — the sums are exact and present; the division step is left to every caller.
7. **A spatially unbiased sampler** for calibration (Morton-strided or hashed-index sampling) to replace "first 128 rows" warm-up if any warm-up is wanted.
Before building 1–3, note the standing ruling (blackboard.md:4): a 256×256 tile is a *palette256 × palette256 address* answered by 256×256 distance/compose tables, **not** by geometric windows. A Moore/Morton geometric experiment should either justify itself against that ruling or live in an example/probe (as `hex_tenant_mq_probe.rs` does), not in the facade.
