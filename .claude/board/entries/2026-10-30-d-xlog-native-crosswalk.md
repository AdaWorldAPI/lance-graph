# D-XLOG-NATIVE-0 — XLOG plan vocabulary crosswalked onto the ndarray::simd polyfill

**Date 2026-10-30 · Status PROPOSED / DOCS ONLY · Plan `.claude/plans/d-xlog-native-v1.md` · Precedent D-ONNX-NATIVE**

XLOG's compiled-plan nodes and operators are mapped onto the algebra under `lance-graph-onnx`: `ndarray::simd` functions, in D-ONNX-NATIVE's four categories. Targets are name-checked against the `ndarray::simd` imports of `lance-graph-mask-risc/src/exec.rs`; mask-risc and relalg are cross-checks only. No row is tested yet.

- ALIAS / INLINE: `eq_u32_to_mask`, `ne_u32_to_mask`, the `*_i32_to_mask` comparisons and their `_under` forms, `mask_and/or/not/andnot`, `mask_any/all`, `mask_gather_u32` (many-to-one semi / anti join), `eq_u32_via_to_mask`, `masked_group_count_u32` (and `_via`), `masked_group_sum_i32`, `masked_group_min/max_i32`, `masked_sum_i32`, `popcount_batch_u64`. The mask operators, `i32` comparisons and `masked_sum_i32` are the same symbols `lance-graph-onnx` aliases.
- Projection: a column-subset `Project` is a `ClassView × WideFieldMask` selection over the relation lens (the ERB / Redmine field-view pattern), not a native call.
- NATIVE: the many-to-many join Gate B uses (runs only on the lab `relcomp::compose`), the bag count over it (a per-key product of two counts), ordered `u32` comparison, 64-bit and symbol constants, recursion.
- REFUSE: outer joins, projection arithmetic, stratified negation, `LogSumExp`, probabilistic programs, float constants.

**OPEN:** no `lance-graph-xlog` crate (ndarray only, like `lance-graph-onnx`) until selection rows are tested against the probe's oracles with native signatures checked, and the many-to-many join has a native home.
