# D-TFLP-0 — TFLite models on the native polyfill (capstone, docs only)

**Status:** PROPOSAL / DOCUMENTATION ONLY. Plan:
[d-tflp-tflite-palette-polyfill-capstone-v1.md](../plans/d-tflp-tflite-palette-polyfill-capstone-v1.md).
Specializes D-NAM-2 (#1780) and D-NAM-PZL (#1784); no code, no ABI.

- An int8 TFLite activation tensor is a Palette256 with affine decode
  `scale · (q − zero_point)`. The TFLM int8 reference kernels compute each output code
  from input codes and conversion-time parameters only (read in source: `integer_ops/
  tanh.h`, `logistic.h:27`, `add.h:106-124`, `mul.h:36-41`).
- So unary ops become 256-entry tables and pair ops 65,536-entry tables, bit-exact by
  construction; chains are closed because TFLite requantizes to int8 between ops. The
  EXO-129/EXO-132 closure gate stays on the separate cosine/PaletteZ domain.
- Softmax: a table keyed by `x − max_row` plus one row reduction (`reference/softmax.h`).
- FC/conv: weight zero point is 0 for per-channel kernels (`fully_connected.h:31`), so
  `ia · Σw + bias` folds into a per-channel bias at load and the dot product runs on the
  existing `gemm_u8_i8` / `matmul_i8_to_i32`, then `requantize_i32_to_i8` (#375).
- Semantics are read from `AdaWorldAPI/tflite-micro` (@ `8f6596d`) and `AdaWorldAPI/Rust`
  (TensorFlow 2.18.0 op definitions); neither is linked.

**Open:** VNNI accumulator saturation variant, full-runtime vs TFLM kernel parity, target
models. Stages D-TFLP-1..8 NOT RUN.
