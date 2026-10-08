# 2026-10-08 — Recipe probe on the shipped PhaseLut; all order statistics; where the power-sum fold spends its time (D-ART-2)

**Status:** MEASURED. Probe only; no production code changes. Follows D-ART-1.
- **Probe:** `crates/lance-graph-mask-risc/examples/algebraic_recipe_probe.rs`, sections 3b, 4 (E-LUT arm) and 4b.
- **Uses:** `ndarray::simd::PhaseLut` (ndarray #343) and `CrossPowerSums::checked_affine` (ndarray #344).
- **Host:** Intel Xeon @ 2.8 GHz, 4 vCPU, `avx2` build, release, `debug = 0`. Medians of 5–9 runs. The VM is noisy; ranges below are across three runs.

## 1. Sector permutation on the shipped phase table

Arm **E-LUT** is arm E (two fixed orderings plus one cosine) with every cosine taken from `phase_lut_4096()`.

| m | A generate + select | C profile LUT, 64 buckets | E sector, `f64` cos | **E-LUT** | E-LUT error on R = 100 | its stated bound |
|---|---|---|---|---|---|---|
| 3 (Wankel) | 106–108 ns | 31–34 | 49–56 | **15.6–18.9** | 3.7e-5 | 8.3e-5 |
| 16 | 400–718 | 10.6–20.9 | 42–61 | **15.8–29.0** | 3.5e-5 | 7.8e-5 |
| 256 | 4,918–5,415 | 10.7–14.9 | 41–46 | **14.6–16.0** | 3.5e-5 | 7.8e-5 |

Ranges are over 65,536 and 1M instances. Errors are max |error| over 200,003 phases × every rank.

- The error stays within `(r + e) · lerp_error_bound()`, asserted on every rank. Disable run: `nearest` instead of `lerp` gives 0.077 against the 8.3e-5 bound, and the assertion fires.
- E-LUT needs a 32 KB shared table plus a 24 B–2 KB permutation, and its cost does not depend on m. For the Wankel triangle it beats both the profile LUT (2×) and the analytic arm B (2.7×).

## 2. All m order statistics per instance

65,536 instances; m = 256 uses 16,384.

| m | A generate + sort | E sector | E-LUT | C profile row |
|---|---|---|---|---|
| 3 | 92 ns | 80 | 35 | **18** |
| 16 | 432 | 352 | 133 | **42** |
| 256 | 9,101 | 4,671 | 1,956 | **493** |

Max error: E 3e-13, E-LUT 5e-5, C 2e-2 / 7e-4 / 1e-5 for m = 3 / 16 / 256.

When every rank is wanted, the profile row is 2–4× faster than E-LUT: the row is contiguous and needs no permutation lookup. When one rank is wanted, E-LUT is the choice: it is exact up to the table's bound and needs no per-m table.

## 3. Where the cross-power-sum fold spends its time

1M rows, one group, all rows selected. ns per row, three runs:

| fold | ns/row |
|---|---|
| plain scalar, `i128` accumulators | 4.4–7.5 |
| same, with `i64` accumulators | **0.5–0.9** |
| `i128` + mask-bit walk | 3.9–6.7 |
| `i128` + group key | 5.3–9.6 |
| shipped `masked_group_cross_power_sums_i32` | 5.3–8.1 |

All five produce identical sums.

- **The accumulator width is the cost.** `i64` accumulators are about 8× faster than `i128`, presumably because they vectorise. The mask walk and the group key are within noise.
- **D-ART-1's "fused 3.7 vs shipped 4.8 ns/row" was noise between two `i128` folds.** It is not a defect in the shipped fold's addressing.
- **`i64` is not exact for the full `i32` range.** One `x²` reaches 2^62, so two rows can overflow it. A faster exact fold therefore needs either known-bounded inputs, or a split such as `x = hi·2^16 + lo` with `u64` partial sums combined into `u128` at the end. ndarray's `masked_group_bounded_power_sums_u8` is the bounded case of this idea for `u8`.

## 4. Shipped `checked_affine`

The probe's own `i128` summary transform now agrees bitwise with `CrossPowerSums::checked_affine` (ndarray #344) on every row count.

## Recommendations

| item | verdict |
|---|---|
| Single order statistic of an m-fold family: sector permutation + `PhaseLut` | **ADOPT** as the method (exact to the table's bound, flat in m) |
| All order statistics, m ≥ 16: interpolated profile row, bucket edges on the tie points | **PROBE** |
| Wide-input power-sum fold with split `u64` partial sums | **PROBE** in ndarray, with a parity test against the `i128` fold; measured lever ≈ 8× |

## OPEN

- One host class.
- The split-accumulator fold is not built. The 8× is measured on `i64` accumulators, which are not exact for full-range `i32`.
