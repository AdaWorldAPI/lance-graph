# D-TRACT-SHAPE-1 — cold shape folding and ON × WHEN × DO admission of the shape ops

**Date 2026-10-30 · Status LAB PROBE · Plan [`d-tract-harvest-0-native-algebra-bridge-v1.md`](../../plans/d-tract-harvest-0-native-algebra-bridge-v1.md) §0.2 · Crate `crates/tract-harvest-probe`**

Every value is classed Const / Shape / Symbolic / Data. A node folds at resolution only through its own registration (WHEN TRUE, then its DO) and only when invariant under the source and the resolved input facts; a run with other input dims is refused. Registry v2 adds Shape, Gather (int64 data), Unsqueeze, Concat, Reshape, Cast i64→i64 and Erf (`vserf`); scalar-Y Where uses `blend_i32_scalar`.

A Tract fact is admitted only from the value's source-producer outlet (a matching label is not authority): typed 1,988 of 2,546 admitted, 558 rejected; inference 0 rejected. Tract `TDim` is read as int64. A Tract panic (out-of-range constant Gather index) is a frontend refusal.

**Jina coverage [WHEN-true, not-true, unregistered], pinned [1,10]:** 4,146 / 0 / 3,474 → **6,404 / 56 / 1,160**; 4,440 nodes fold in 7.3 ms, 1,972 folded values retained (~20 KB). The remaining shape blockers are int64 arithmetic (Mul 115, ConstantOfShape 59, Equal 58, Where 58, Div 56, …), which no ndarray word covers.

Fixtures (ORT 1.29.0 `ORT_DISABLE_ALL`): shape chain bit-exact; Erf ≤ 2 ULP vs ORT and f64 over 4,115 points. 40 tests; 9 disable runs red.

**OPEN:** int64 shape-arithmetic semantics (owner, then ndarray ask); axis as DO parameter vs classid per axis; `vserf` ~4x slower than Tract's erf; ConstantOfShape/Expand via ndarray core not yet registered; multi-tile and MatMul gates unchanged.

**Follow-up (after #400 / #1853):** registry v3 adds int64 Equal/LessOrEqual (#400 compares), int64 Where (`blend_i64[_scalar]`), ConstantOfShape (`from_elem`), Expand (`broadcast`). Optional-output fixture (`LSTM ["", "Y_h"]`) pins that `source_producers` follows Tract's compacted outlet numbering and admits `Y_h`'s fact, not the omitted `Y`'s. Jina coverage 6,404/56/1,160 → 6,581/114/925. Remaining int64 arithmetic (Mul/Div/Add/Sub/Range) is refused; exact checked signatures and the `i64_arith` fixture are the ndarray ask (plan §0.3).

**Hardening + census:** explicit ONNX-position → Tract-slot mapping (`tract_slot`), source-bound facts (`merged_for`), collision/stale/tail-bit/ConstantOfShape/Expand tests (55 green, 5 more disable runs red). Jina census: 6,581/114/925, 4,499 folded; the 56 blocked Reshapes all wait on int64 `Mul`; not folded = 2,461 runtime + 660 shape-only blocked.

**Checked int64 (ndarray #401, plan §0.4):** registry v4 adds int64 Add/Sub/Mul/Div on `checked_binary_i64` and Range on `range_i64` (capacity 65,536); fold DO refusals now refuse the plan. The `i64_arith` chain folds and matches ORT exactly; overflow and unadmitted broadcasts refuse (zero divisor and zero Range step already refused by Tract). Initializers without an admitted fact get their protobuf dtype/dims in fold and coverage. Jina 6,871 / 1 / 748, 5,139 folded; all 58 shape-dependent Reshapes fold; 20 shape-only nodes still blocked, nearly all on the rotary-embedding chain (Cos, Sin, MatMul, Transpose, Cast; its `inv_freq` lives in the external weights), plus And and one broadcast LessOrEqual. 60 tests, 7 more disable runs red.

**Rotary chain (plan §0.5):** registry v5 adds Cast int64→f32 and bool→bool, f32 MatMul (`matmul_f32_batched`), Transpose, Cos/Sin (`vscos`/`vssin`), bool And (`mask_and`), and any-broadcast int64 Equal/LessOrEqual. The rotary `inv_freq` is fetched as a SHA-256-pinned range (data gitignored, digests committed) and inlined into its Constant. The real Jina rotary fold matches ORT `ORT_DISABLE_ALL` within 1 ULP; trig ≤ 1 ULP from f64. Jina 7,274 / 0 / 346, 5,159 folded, **0 shape-only nodes blocked**; the 346 unregistered are run-time ops. 70 tests, 6 more disable runs red. OPEN: ndarray's `matmul_f32` doc claims BF16 on AMX but the word is exact f32; `amx_matmul` is x86_64-only.
