# D-NAM-2 — ONNX and TFLite thin DTO compatibility leaves

**2026-10-25 • PROPOSED ENGINEERING SPEC / DOCS ONLY.** Parent [D-NAM capstone](d-nam-native-algebra-dto-capstone-v1.md), [EXO-152](https://linear.app/exo-red/issue/EXO-152). D-NFC #1753 is binding; no new graph IR or executor.

## Verified current baseline
`crates/lance-graph-onnx` exists on main (PR #1754 and #1758 merged), depends normally only on ndarray, and has exact aliases, inline shape/overflow checks, bounded Tanh/Exp aliases. Its own docs explicitly forbid runtime dispatch and graph parsing. At the read, no `crates/lance-graph-tflite` was found on main; AdaWorldAPI/tflite2onnx is a conversion fork, **not** native TFLite facade evidence.

## Work split
1. **ONNX compatibility**: do not rewrite or duplicate #1754. Build a cold, versioned **DTO mapping table/test layer** referencing existing native aliases. Verify against pinned official ONNX IR/opset schema (domain, since_version, attributes, functions, axes-as-input, shape and type). Map node patterns only when pattern recognition and every output are admissible; otherwise explicit refusal. Preserve symbolic shapes, external weights and control-flow effects as *unadmitted* until separately implemented.
2. **TFLite compatibility**: a future, independently authorized `lance-graph-tflite` leaf may expose only verified native aliases/inline conversions to the *same* native methods. No FlatBuffer parser in hot path and no ONNX facade dependency. The schema must pin BuiltinOperator, BuiltinOptions, subgraph signatures, type, data layout and versions.
3. **Boolean boundary**: TFLite/ONNX external `BOOL` serialized values are not literally `[u64]` packed DR/native mask words. Explicit cold/prologue packing/unpacking or a compatible external buffer view; test tail bits, scalar flags, shape and output. Include conversion cost.
4. **Quantization**: TFLite affine `real=scale*(q-zero_point)` is not Palette256/Fisher-Z; preserve zero points, per-channel scales, accumulator width, rounding, bias, saturation, activation range. Exact quantized interpretation and calibrated approximation are separate arms; do not auto-enable palette.
5. **Nonnumeric**: tensors, variable state, IF/WHILE, custom ops, dynamic reshape, gather/scatter or unsupported FLOAT formats typed-refuse under STRICT_NATIVE. External interpreter is differential oracle only, never hidden fallback.

## Positive fixtures (separate row keys, not blanket compatibility)
- rank-1 signed i32 `GREATER`/ONNX `Greater` -> native `gt_i32_to_mask` with explicit internal mask conversion.
- `SELECT_V2` / ONNX `Where` i32 -> native blend, shape and operand order explicit.
- `SUM` / ONNX `ReduceSum`: guarded reductions; compare `Greater/Where/ReduceSum` against `GREATER/SELECT_V2/SUM` and same native fold only for bounded integer sums, axes/keepdims/shape identical.
- `TANH` / ONNX `Tanh`: f32 ULP-bounded ndarray native alias where numeric contract admits; TFLite int8 TANH is **not** a free f32 alias.

## Required negatives and acceptance gates
Wrong operator-version default, BOOL representation mismatch, broadcasting error, reduce overflow and empty axes, signed zero, NaN, Infinity, subnormal, uint8/i8 zero-point mismatch, per-channel scale mismatch, wrong control-flow effect, unrecognized custom op. Independent ONNX Runtime and TFLite interpreter fixture versions named, checked only in offline tests. At least two hostile mutations cause typed refusal. No cross-dependency from native crates to these leaves, no per-call routing, no new tensor AST. Implementation requires an owner-approved follow-up PR with actual code, CI and receipt; this document does not claim it.
