# D-ONNX-NATIVE-0 — ONNX names over existing native functions

**Date:** 2026-10-23 · **Status:** IMPLEMENTED (draft PR, not merged) · **Plan:** `.claude/plans/d-onnx-native-v1.md`

`crates/lance-graph-onnx` exposes ONNX operator names over existing native functions. It has no dispatcher, no graph and no initialization, and its only normal dependency is `ndarray`. No native crate depends on it.

- **ALIAS** (same symbol, pinned by `fn_addr_eq`): `Abs`, `Sqrt` (vml); `Greater`/`GreaterOrEqual`/`Less`/`LessOrEqual`/`Equal` on int32 vs a scalar; `And`/`Or`/`Xor`/`Not` on packed masks; `Where` int32.
- **INLINE**: float `Add`/`Sub`/`Mul`/`Div` (a broadcast check turns ndarray's panic into a refusal); `Neg`; int32 `ReduceSum` (attributes validated once, plus an overflow refusal); `ReduceSum(Where(Greater))`, the #1752 composite.
- **MEASURED** (ns per call, reported not gated): Greater direct = onnx (14.6 / 14.6 at 64 rows, 0 allocs). The composite is 38.6 direct vs 39.3 onnx vs 224 for the #1752 wrapper (3 allocs). In the asm, the composite's extra instructions are the int32 range check only.
- **FOUND:** `vml::vsneg` returns `+0` for `Neg(+0)` when n ≥ 16 (`0.0 - v` in the SIMD body), so `Neg` is INLINE, not ALIAS. `vml::vsexp` saturates +inf. Both are reported to the ndarray owner, not fixed here.
- **REFUSED:** integer arithmetic (overflow undefined), tensor-vs-tensor compares, MatMul, control flow, protobuf import, automatic Palette substitution.
- **OPEN:** independent review; the NATIVE rows in plan §2 (approximate transcendentals, float ReduceSum tolerance, empty-population min/max).
