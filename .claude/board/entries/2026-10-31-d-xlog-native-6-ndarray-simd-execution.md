# D-XLOG-NATIVE-6 — the XLOG probe executes on the ndarray::simd polyfill

**Date 2026-10-31 · Status IN PR · Plan `.claude/plans/d-xlog-native-v1.md` Amendment B**

`crates/xlog-relcomp-probe/src/native.rs` answers the admitted two-atom chain:
- support: `eq_u32_to_mask` for bound ends, `mask_scatter_or_u32` onto Y, then `mask_and`;
- bag count: `masked_group_count_u32` for each side, then `checked_binary_i64(Mul)` (ndarray #407).

The Y axis is DR's `LocalMap`, so the path answers 200 Y where the register's 64-slot tile refuses. `admit_count` admits XLOG's `count` rule. The register path remains as the differential, with every mask compared in Gate B. CubeCL was not needed.

**OPEN:** a many-to-many join keeping both ends, and chains past two atoms (still NATIVE); the overflow refusal has no reachable fixture.
