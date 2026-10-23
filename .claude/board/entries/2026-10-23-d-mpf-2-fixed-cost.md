# 2026-10-23 — D-MPF-2: where the native fold's cost goes

**Status:** MEASURED, lab only (`crates/math-polyfill-probe`, PR #1752). Nothing changed outside the probe.

- Per call at 64 rows: scalar 12 ns; ndarray floor 39; mask-risc +57; `Program::new` +20 and 1 allocation; dialect +50. Total ~172 ns.
- Per row at 64 K rows: mask-risc ≈ ndarray floor. The floor's cost is `ndarray::simd::masked_sum_i32`, a scalar set-bit walk (LLVM: `tzcnt`/`blsr`, no vector register) whose time tracks popcount: 1.7 / 25 / 44 µs at 0 / 50 / 100 % selected. The compare is ~0.09 ns/row.
- Correction to D-MPF-1: "1.6× faster than the scalar loop" compared against the `i128` filter-chain oracle. An equal-work `i64` loop takes ~11 µs at 64 K rows and is ~3× faster than the fold.
- Proposal, not made: density-chosen `masked_sum_i32` body (branch-free above ~16 set bits per word, walk below). The branch-free form equals the oracle and is flat in selectivity; the walk is 7–10× faster on empty words.
- OPEN: whether the mask form beats a scalar loop when the mask is reused or combined (not measured here).

Plan: `.claude/plans/d-mpf-math-polyfill-v1.md` §7.
