# 2026-11-25 — D-AED-5: dispatch once per sealed slice

**Status:** MEASURED, partly NOISY. Plan: `.claude/plans/adaptive-exact-decimal-binding-v1.md` D-AED-5.

`slice_count_sum_where` (`crates/exact-decimal-probe`) resolves a decimal
literal once per sealed slice, at that slice's exponent (`threshold`), and
runs `Count` and `MaskedSumI32` over each slice's extent; sums align per
slice. Equal to the per-value exact oracle for all six operators over slices
at three exponents (`a_threshold_dispatched_per_slice_equals_the_per_value_oracle`;
disable run resolving at one exponent for all slices is red).

Command: `cargo run --release --example slice_dispatch` (1 048 576 rows,
every 9th absent, slices alternating scale 2 / scale 3, best of 30, i32
lanes). The example asserts each per-slice answer equals one pass over the
same values at one exponent. Two runs, ns per 64 lanes:

| selection | 1 slice | 16 slices | 256 slices |
|---|---|---|---|
| dense (885 886 kept) | 41.6 / 61.8 | 39.4 / 39.1 | 43.5 / 42.3 |
| sparse (932 kept) | 31.7 / 31.7 | 25.5 / 25.7 | 36.9 / 33.7 |

- Per-slice dispatch is no slower than one pass: 16 slices are faster,
  likely because each slice's two passes stay in cache.
- The dispatch cost per slice is not resolvable here: the dense one-slice
  baseline moved 41.6 → 61.8 between runs, larger than any per-slice
  difference (computed 121–331 ns at 256 slices, negative at 16). NOISY.

OPEN:
- width-specific kernels: every slice runs on its `i32` lane. PM9/PM17
  reductions do not exist (ndarray #438) and no `LaneRef` lends a U8/U16
  slice (D-AED-2 owner seam), so the PM×width columns of the plan's measure
  cannot be filled yet.
- end-to-end bytes moved, measured only once width kernels exist.
