# 2026-09-29 — Grouped cross moments: Pearson / covariance / OLS / R² over masks

**Status:** MEASURED · DONE — ndarray `simd_masking_ops.rs`, `crates/lance-graph-mask-risc`, `crates/jc`

## What landed
- ndarray: `GroupCrossPowerSums { n, sum_x, sum_y: i64, sum_x2, sum_y2, sum_xy: i128 }` + `masked_group_cross_moments_i32{,_via,_pair}` over the same generic `group_walk`. `x_moments()`/`y_moments()` return the exact univariate `GroupPowerSums`; a test pins them equal to the univariate fold. Bounds: squares ∈ [0, 2^62], products ∈ [−2^62+2^31, 2^62]; the `i64` sums bind at 2^32 rows/group; the `i128` sums cannot overflow at any `u64` row count.
- mask-risc: `Terminal::GroupCrossPowerSumsI32 { mask, key, x, y }` → `Out::CrossPowerSums`.
- jc: `pearson_from_cross_power_sums`, `sample_covariance_from_cross_power_sums` (n−1), `simple_regression_from_cross_power_sums` (`SimpleRegression { slope, intercept }`), `r_squared_from_cross_power_sums` (`multiple_r_squared`'s k=1 contract). Shared tails extracted: `pearson_from_centered`, `sample_cov_tail`, `r_squared_tail`. Centred sums formed exactly in `i128`, checked.

## Measured
48 layouts (n ≤ 100k): |Δr| ≤ 1.3e-14, |ΔR²| ≤ 4.1e-14, relative Δslope ≤ 2.8e-14 vs the materialized path. Huge offset + tiny spread (x ≈ ±2·10⁹): fold path within 1e-15 of an independent exact reference. **Unlike ANOVA, the slice path does not fail here** — `pearson`/`multiple_r_squared` are two-pass. A naive f64 projection of the same moments returns NaN; that is what exact `i128` centring prevents (disable run: 2 tests fail).

## Redundant guards (disable runs, kept as explicit contract)
`pearson_from_cross_power_sums`'s `n < 2`, the regression's `cxx == 0`, and R²'s `cyy == 0` are each subsumed by the shared tail's zero/non-finite rejection. The first past-bound refusal fixture was vacuous (wrapped to negative centred squares); replaced with one that wraps to a plausible `r = 1`.

## Open
- Multi-membership in one pass: `lance-graph-report` lowers `CoordSpec::MaskSet` to one `Filter::Plane` pass per member tuple and per fold state. The single-group limit is in `GroupKeyAddr::group_of` (one `Option<usize>` per row), in the IR's `GroupKey`, and in report lowering. Seam: a word-level walker `for each 64-row word: sel & plane_m → fold hits into out[m]` over existing planes — same plane traffic as K passes, value-lane traffic of the union instead of the sum.

**Renamed 2026-09-30 (rebase onto ndarray master):** upstream shipped `PowerSums { n, sum, sum_sq: u128 }` — the same record as `GroupPowerSums`. The duplicate was dropped: consumers use `PowerSums` / `CrossPowerSums` and `masked_group_(cross_)power_sums_i32*`; `checked_merge` moved onto the upstream types. Square sums are now `u128`; jc converts with `i128::try_from`, and a value past `i128::MAX` is refused as past the bound.
