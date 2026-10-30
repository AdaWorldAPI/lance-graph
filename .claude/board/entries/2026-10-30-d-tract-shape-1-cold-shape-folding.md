# D-TRACT-SHAPE-1 — cold shape folding and ON × WHEN × DO admission of the shape ops

**Date 2026-10-30 · Status LAB PROBE · Plan [`d-tract-harvest-0-native-algebra-bridge-v1.md`](../../plans/d-tract-harvest-0-native-algebra-bridge-v1.md) §0.2 · Crate `crates/tract-harvest-probe`**

Every value is classed Const / Shape / Symbolic / Data. A node folds at resolution only through its own registration (WHEN TRUE, then its DO) and only when invariant under the source and the resolved input facts; a run with other input dims is refused. Registry v2 adds Shape, Gather (int64 data), Unsqueeze, Concat, Reshape, Cast i64→i64 and Erf (`vserf`); scalar-Y Where uses `blend_i32_scalar`.

A Tract fact is admitted only from the value's source-producer outlet (a matching label is not authority): typed 1,988 of 2,546 admitted, 558 rejected; inference 0 rejected. Tract `TDim` is read as int64. A Tract panic (out-of-range constant Gather index) is a frontend refusal.

**Jina coverage [WHEN-true, not-true, unregistered], pinned [1,10]:** 4,146 / 0 / 3,474 → **6,404 / 56 / 1,160**; 4,440 nodes fold in 7.3 ms, 1,972 folded values retained (~20 KB). The remaining shape blockers are int64 arithmetic (Mul 115, ConstantOfShape 59, Equal 58, Where 58, Div 56, …), which no ndarray word covers.

Fixtures (ORT 1.29.0 `ORT_DISABLE_ALL`): shape chain bit-exact; Erf ≤ 2 ULP vs ORT and f64 over 4,115 points. 40 tests; 9 disable runs red.

**OPEN:** int64 shape-arithmetic semantics (owner, then ndarray ask); axis as DO parameter vs classid per axis; `vserf` ~4x slower than Tract's erf; ConstantOfShape/Expand via ndarray core not yet registered; multi-tile and MatMul gates unchanged.
