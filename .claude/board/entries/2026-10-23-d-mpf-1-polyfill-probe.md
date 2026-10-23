# 2026-10-23 — E-D-MPF-1: the Stage 1 polyfill already exists; per row it is sub-nanosecond, per one-word call it is ~600 ns

**Status:** MEASURED, lab only (`crates/math-polyfill-probe`). Plan `.claude/plans/d-mpf-math-polyfill-v1.md`.

- The mathematical polyfill API is the shipped R2IL procedure layer (`lance-graph-r2il::procs`) plus checked scalar arithmetic in `FoldDialect`, over mask-risc, with relalg as the rewriting frontend. No new primitive, IR or executor.
- One expression, `SUM(v | sel) − k·COUNT(sel)` with a NULL presence plane, written only with those procedures, equals an independent `i128` row loop at 64 / 1 000 / 4 096 / 65 537 rows. 9 tests; disable runs D1–D3 each turn named tests red.
- Typed refusals already exist for lane + lane arithmetic, unsigned compare, overflow and division by zero.
- Cost (ns per call, medians, 3 runs): at 65 536 rows the expression costs 0.86–0.91 ns/row and beats the scalar loop 1.6×; at 64 rows one fold is ~120 ns in mask-risc plus ~150 ns in the dialect, and the scalar loop (~80 ns) beats the whole expression (~600 ns).
- Gaps by counterexample: G1 element-wise value arithmetic, G2 unsigned compare, G3 floats, G4 the one-word fixed cost.
- The ONNX fork `AdaWorldAPI/optimum-onnx-rs` is empty; `AdaWorldAPI/onnx` is not accessible.

Next gate: D-MPF-2 (attribute the one-word fixed cost), and the owner decision on G1.
