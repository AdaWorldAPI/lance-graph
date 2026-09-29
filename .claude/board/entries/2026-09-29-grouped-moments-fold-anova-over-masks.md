# 2026-09-29 — One-way ANOVA over a population mask from grouped moments

**Status:** MEASURED · DONE — ndarray `simd_masking_ops.rs`, `crates/lance-graph-mask-risc`, `crates/jc/src/stats.rs`

## What landed
- ndarray: `GroupMoments { n: u64, sum: i64, sum_sq: i128 }` and `masked_group_moments_i32{,_via,_pair}` — `(n, Σx, Σx²)` per group in one pass over the rows a mask selects, over the existing `group_walk` (now generic over its slot type). `checked_merge` is exact integer addition; exact up to `GROUP_MOMENTS_MAX_ROWS = 2^32` rows per group (the `i64` sum is the binding field).
- mask-risc: `Terminal::GroupMomentsI32 { mask, key: GroupKey, val }` → `Out::Moments`. Own terminal, not a `GroupFold` member (a `GroupFold` slot is one seeded `i64`). Refuses planes past `MASKED_SUM_I32_MAX_ROWS` and partial extents.
- jc: `anova_from_moments` / `eta_squared_from_moments`. F/p/η² policy is shared with `anova_one_way` / `eta_squared` via `anova_from_ss` / `eta_from_ss`. Sums of squares come from exact `i128` quantities; no two large floats are subtracted.

## Measured
Mask → fold → `anova_from_moments` vs materialized → `anova_one_way`, 42 non-degenerate layouts (n ≤ 100k, k ≤ 6, densities 20/128/256 of 256): max relative ΔF 4.2e-14, max |Δp| 4.4e-16, max |Δη²| 6.5e-16. Adversarial layout (group means ±2·10⁹, within spread 1): exact F = 3·10¹⁹; the moments path returns 3·10¹⁹, the slice path returns `None` (its `ss_t − ss_b` cancels the within-group SS to ≤ 0). A naive float `Q − S²/N` inside the moments path fails even the near-`i32::MAX` agreement test (ΔF ≈ 1e-3), so the exact forms are load-bearing.

## Open
- Multi-membership is NOT one pass: `GroupKey` resolves one group per row, and `lance-graph-report`'s `CoordSpec::MaskSet` executes one `Filter::Plane` program per member. Seam: a key address that walks the selected rows once and folds each row into every member plane that holds it.
- Bivariate `(n, Σx, Σy, Σx², Σy², Σxy)`: same walk, a second value lane in the closure, a wider slot type; not built.
- No early exit, and no `CausalEdge64` commit. ANOVA is non-monotone under future rows.
