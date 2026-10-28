# D-TFLP — TFLite models on the native polyfill: int8 codes as Palette256, exhaustive tables (capstone v1)

> ⊘ **Revised 2026-10-26 by [v2](d-tflp-tflite-palette-polyfill-capstone-v2.md).** §1 items 1–3 (exactness), §3.1 Mish, §3.2, §3.3 softmax, §3.4 padding, §4 identity, §7 D-TFLP-2/-3/-4 and §9 are corrected there. §1–§2 and §5 stand.

**Date:** 2026-10-25. **Status:** PROPOSAL / DOCUMENTATION ONLY. No code, no ABI, no
crate in this PR. Every stage below is NOT RUN and is approved separately.
**Parent arcs:** D-NAM ([EXO-152](https://linear.app/exo-red/issue/EXO-152), #1778–#1783),
D-NAM-PZL (#1784), D-NFC (#1753), D-ONNX-NATIVE-0 (#1754, #1758).
**Coordination:** [EXO-50](https://linear.app/exo-red/issue/EXO-50) (append-only).

## 0. Goal (operator, 2026-10-25)

> "By achieving tflite polyfill we will be able to use any tflite models in our
> substrate against our polyfill."
>
> "We will add native Palette256:palette256 LUT instead of mish (FisherZ softmax
> cosine roundtrip)."
>
> "All tensor quantization is open source so we can easily reverse engineer."

A TFLite model runs on the substrate with no TFLite runtime, no FlatBuffer
interpreter in the hot path and no float round trip between ops. TFLite's
operator and quantization semantics are reverse-engineered from open source
(TFLM reference kernels, TensorFlow op definitions) and reproduced by native
ndarray methods and precomputed code-to-code tables.

This plan specializes D-NAM-2 (`lance-graph-tflite` as a thin leaf in the shape
of `lance-graph-onnx`) and D-NAM-PZL (palette-native LUT execution). It does not
supersede either, and it adds no executor, tensor IR, `Rel` variant or runtime
dispatcher.

## 1. The key fact: an int8 TFLite tensor is a Palette256

In an int8-quantized TFLite model every activation tensor holds codes
`q ∈ {-128..127}` with a per-tensor affine decode `real = scale · (q − zero_point)`.
That is a 256-entry palette whose representatives are fixed by two numbers.
Weights are symmetric (zero point 0), so each output channel is its
own 256-entry palette. ⊘ Was "per-channel symmetric": the int8 spec requires zero
point 0 for per-tensor and per-axis weights alike, and TFLM's FC prepare asserts it
(`fully_connected_common.cc:147`). A nonzero `weights_offset`, which the generic
reference kernel accepts, is outside the admitted profile and is refused at
admission (audit `entries/2026-10-26-d-tflp-parity-audit.md`, Amendment 1 §A4).

The TFLM int8 reference kernels compute each output code from the input codes
and from parameters that are fixed when the model is converted
(`zero_point`, `multiplier`, `shift`, `range_radius`, activation clamp). Read in
source:

- `integer_ops/tanh.h` (118 lines): output = function of `input_data[i]` and
  `input_zero_point, input_range_radius, input_multiplier, input_shift` only.
- `integer_ops/logistic.h:27` (122 lines): same shape.
- `integer_ops/add.h:106-124` (176 lines): output = function of `(x, y)` and
  `input{1,2}_offset, left_shift, input{1,2}_multiplier/shift,
  output_multiplier/shift, output_offset, activation_min/max` only.
- `integer_ops/mul.h:36-41` (126 lines): same, over `(x, y)`.

Consequences, each by construction rather than by measurement:

1. **Unary op → 256-entry table.** Running the ported reference arithmetic over
   all 256 input codes yields a table that reproduces the kernel bit for bit.
2. **Pair op → 65,536-entry table (64 KiB u8).** Running it over all code pairs
   yields a bit-exact Palette256:palette256 table.
3. **Chains stay exact.** TFLite itself requantizes to int8 between ops, so the
   code space is closed under every op by definition. Consecutive unary tables
   fuse into one (`L2[L1[x]]`); a pair op followed by unary ops fuses into its
   pair table. Two pair ops in a row stay two lookups, still exact.
4. **The closure gate of EXO-129/#1731 and EXO-132 does not apply to this
   domain.** It applies to the separate cosine/Helix/PaletteZ domain (§5), where
   the carrier is calibrated rather than defined by the model.

TFLM itself already tabulates: `micro/kernels/internal/common.h:521`
`LUTPopulateInt8(input_scale, input_zero_point, output_scale, output_zero_point,
transform, …)` fills a 256-entry table from a float transform, and
`softmax_common.cc:62,66` uses it (`LUTPopulate<int16_t>`, the int16 softmax path). The int8 LUT is indexed by the code cast
to `uint8_t`, so it is stored `[0..127, -128..-1]` (`common.h:516-519`); the
native table must use the same index order or remap it once at load.

## 2. Existing assets this plan reuses (verified in source, 2026-10-25)

### 2.1 ndarray (`AdaWorldAPI/ndarray`, `master`)

| Asset | Location | Role here |
|---|---|---|
| gemmlowp/TFLite fixed-point requantization, bit-exact (#375, commit `e2e89ed`) | `src/simd_fixedpoint.rs`: `srdhm_i32:35`, `rounding_divide_by_pot_i32:74`, `multiply_by_quantized_multiplier_i32:95`, `…_single_rounding_i32:112`, `…_i64:130`, `quantize_multiplier:196`, `…_single_rounding:210`, `requantize_i32_to_i8:237`; re-exported from `src/simd.rs:1003-1009`; SIMD lanes in `simd_avx.rs:278-293`, `simd_avx2.rs:2752-2801`, `simd_avx512.rs:1051-1119` | building blocks for porting every int8 kernel; output stage of FC/conv |
| u8×i8→i32 GEMM, runtime-dispatched AMX / VNNI / scalar | `src/simd_int_ops.rs:262 gemm_u8_i8`; `src/simd_runtime/matmul.rs:227`; `src/hpc/quantized.rs:618 int8_gemm_i32` (scalar reference); `src/hpc/int8_tile_gemm.rs:47,357` (AMX) | FC and conv accumulator (§3.4) |
| i8×i8→i32 matmul with sign-shift bias | `src/simd_runtime/matmul.rs:80,193 matmul_i8_to_i32`; `src/hpc/amx_matmul.rs:951` | alternative accumulator for signed activations |
| Codec8 compile-time code spaces (#370) | `src/hpc/codec8.rs`: `CodeSpace:57`, `Asym:73`, `Sym:81`, `UnitFull:89`, `Storage:115`, `I8:122`, `U8:130`, `Rounding:205` (`TiesAway`, `TiesEven`, `Floor`), `NanPolicy:277`, `Calibration:334`, `Codec8:403` | type-level name for "TFLite affine int8 tensor" = `Asym` code space; keeps it apart from Helix/PaletteZ codes |
| `vml::vstanh` (1 ULP) | `src/hpc/vml.rs:646` | float Tanh; ONNX bounded alias already uses it |
| `mish_f32` | `src/hpc/linalg/activations_ext.rs:168` | float Mish reference for building the int8 Mish table and for parity |
| Burn `CompiledLinear` / `CompiledAttention` caches | `crates/burn/src/ops/matmul.rs:30,46,87,105` | **negative precedent**: keyed by shape, so same-shape layers share one table. Tables here are keyed by model/subgraph/tensor identity (§4) |

### 2.2 lance-graph (`main`)

| Asset | Location | Role here |
|---|---|---|
| `lance-graph-onnx` facade (#1754, #1758) | `crates/lance-graph-onnx/src/lib.rs`: ALIAS / INLINE / refusal pattern; `bounded::tanh_f32 = vml::vstanh` (`:83`); `sum_where_greater_i32` (`:288`); depends on ndarray only | template for `lance-graph-tflite`: same three verdicts plus a fourth, TABLE |
| Masking kernels used by the D-NAM fixture | ndarray `simd_masking_ops.rs`: `gt_i32_to_mask:344`, `masked_sum_i32:815`, `blend_i32:5176` | the four D-NAM-1 fixture rows |
| Palette/LUT precedents | `bgz-tensor/src/attention.rs:34 AttentionTable`, `:49 ComposeTable`; `bgz-tensor/src/fisher_z.rs:100 FisherZTable`; `bgz-tensor/src/palette.rs:27 WeightPalette`; `helix/src/distance.rs:18 DistanceLut`; `helix/src/quantize.rs:50 RollingFloor` (bucket centre `(b+0.5)/256`, `:243-249`) | 256² table layout, version stamps, indexed hot path. Their Fisher-Z/cosine semantics are **not** reused for the int8 domain |
| RelAlg | `crates/lance-graph-relalg/src/ir.rs`: `Rel {All, Empty, Plane, Atom, And, Or, Not}`, `Measure {Count, Sum, Min, Max, Keep}` | unchanged; numeric ops never enter `Rel` (D-RA-2) |
| Plans | `.claude/plans/d-onnx-native-v1.md`, `d-native-first-dependency-correction-v1.md`, `d-native-repr-contract-v1.md`, `d-onnx-relalg-s4-s5-palette-export-v1.md`, `d-onnx-relalg-cap-v1.md` | D-NFC allows a cold importer outside the hot path; the representation contract names Palette256 as the native carrier |

### 2.3 Open-source semantics, read only (never linked)

| Source | Location | What it gives |
|---|---|---|
| TFLM int8 reference kernels | `AdaWorldAPI/tflite-micro` @ `8f6596d`, `tensorflow/lite/micro/kernels/internal/reference/integer_ops/`: `tanh.h` (118), `logistic.h` (122), `add.h` (176), `mul.h` (126), `fully_connected.h` (240), `conv.h` (243), `depthwise_conv.h` (293), `pooling.h` (266), `mean.h` (18); `reference/softmax.h` (236) | the bit-exact int8 behaviour every table and kernel must reproduce |
| gemmlowp fixed point (vendored in TFLM) | `micro/kernels/internal/fixedpoint.h` (881): `exp_on_interval_between_negative_one_quarter_and_0_excl:711`, `exp_on_negative_values:736`, `neg_tanh_on_negative_values:809`, `tanh:817`, `one_over_one_plus_x_for_x_in_0_1:832`, `logistic_on_positive_values:856`, `logistic:863` | the functions to port next to `simd_fixedpoint.rs` |
| TFLM LUT helper | `micro/kernels/internal/common.h:495-560` (`LUTPopulateInt8`, `LUTPopulateInt16`) | the float-based table builder some kernels use; reproducing it needs IEEE f32 and `TfLiteRound` (round half away from zero) |
| TensorFlow op definitions | `AdaWorldAPI/Rust` (fork of `tensorflow/rust`, untouched since fork; HEAD `1ddf5a3`, 2025-01-12; crate `tensorflow` 0.21.0, TensorFlow 2.18.0). `tensorflow-op-codegen` generates `src/ops/ops_impl.rs` (17,908 public fns): `greater:79804`, `select_v2:179372`, `sum:211879`, `tanh:216194` | op names, inputs, attributes and type constraints for the float ops and for the TF ops TFLite builtins are converted from. Source to read; the crate links `libtensorflow` and is never a dependency |
| NSL | `AdaWorldAPI/neural-structured-learning` PRs #1–#8 (research docs) | NSL trains with graph/adversarial regularization; inference uses the base model, which exports to TFLite and runs on this same polyfill |

## 3. Operator classes and their native strategy

Verdicts extend the `lance-graph-onnx` vocabulary with one new class:

- **ALIAS**: a native function already has the TFLite contract.
- **INLINE**: a small adapter does the semantic work the native function does not.
- **TABLE**: a 256 or 65,536-entry code table, built once at load from the ported
  reference arithmetic, then a lookup per element.
- **NATIVE-TODO**: a native composition still to be built.
- **REFUSE**: typed refusal with a reason; never a silent approximation.

### 3.1 Unary elementwise → TABLE (256 entries)

`TANH`, `LOGISTIC` (kernels read, §1); candidates whose TFLM kernels exist but
are not yet read: `HARD_SWISH` (`hard_swish.cc`), `ELU` (`elu.cc`); also
`RELU`/`RELU6`/`RELU_N1_TO_1` (also expressible as clamp), the
`QUANTIZE`/`DEQUANTIZE` boundary, and the int8 form of Mish. Build: for every code `q`, run the ported reference
(fixed-point for Tanh/Logistic per `tanh.h`/`logistic.h`; `LUTPopulateInt8`
semantics where TFLM builds a LUT). Hot path: `out[i] = T[in[i]]`, 256 bytes per
op.

**Mish.** TFLite has no builtin Mish; a converted model carries
`x · tanh(softplus(x))` as a chain of builtins (or a fused custom op, which is
REFUSE until mapped). Each builtin in the chain is a table and the chain fuses
into one 256-entry table, exact against running the chain op by op in TFLM.
This is the operator's "Palette256 LUT instead of Mish": no float
softplus/tanh at run time, and no Fisher-Z round trip.

### 3.2 Pair elementwise → TABLE (65,536 entries)

`ADD`, `MUL` (kernels read, §1); candidates with TFLM kernels not yet read:
`SUB`, `SQUARED_DIFFERENCE` (`squared_difference.cc`), `MAXIMUM`/`MINIMUM`
(`maximum_minimum.cc`), compares (`comparisons.cc`: `GREATER`, `LESS`,
`EQUAL`, … with int8 inputs and bool output). Build: run
`add.h:106-124` / `mul.h:36-41` (and the matching kernels) over all `(x, y)`.
64 KiB per op, keyed by the op's parameters. Broadcasting is index arithmetic in
the binding, not a table concern. This is the exact Palette256:palette256 path.

### 3.3 Softmax → TABLE + one reduction per row

`reference/softmax.h:73-106`: the int8 path works on `input_diff = x − max_row`,
which for int8 lies in `[-255, 0]`, rescales it with `input_beta_multiplier/
left_shift` and calls `exp_on_negative_values`, skipping diffs below `diff_min`.
So `exp(diff)` is a table keyed by the diff code (≤ 256 entries per op). The row
sum, the reciprocal (`one_over_one_plus_x_for_x_in_0_1`) and the final
per-element scaling stay a reduction in ndarray. Softmax is not a pair table.

### 3.4 Fully connected and convolution → native GEMM + bias fold

`fully_connected.h:31`: for per-channel kernels the weight zero point is always
0; `:64`: `acc += filter_val * (input_val + input_offset)`, then bias, then
`MultiplyByQuantizedMultiplier` per output channel, then `output_offset` and
clamp.

So, with `ia = input_offset` (= −input zero point) and `n` = reduction length:

```
acc[c] = Σ_k w[k,c] · (a[k] + ia) + bias[c]
       = Σ_k w[k,c] · a[k]  +  (ia · Σ_k w[k,c] + bias[c])
```

The second term depends only on weights and parameters: it is folded into a
per-channel bias once at load. The first term is an i8×i8 dot product
(`matmul_i8_to_i32`), or a u8×i8 one after shifting activations by 128
(`a + 128` as u8, then fold `−128 · Σ_k w[k,c]` into the bias as well), which
reaches `gemm_u8_i8`'s AMX/VNNI arms. The output stage is
`requantize_i32_to_i8` (#375). Conv and depthwise conv are the same identity
over their im2col or windowed reduction (`conv.h`, `depthwise_conv.h`).

This closes the "no zero point in ndarray GEMM" gap without a new kernel family:
the zero point lives in the folded bias.

### 3.5 Other reductions and data movement

- `AVERAGE_POOL_2D`, `MAX_POOL_2D` (`pooling.h`), `MEAN` (`mean.h`), `SUM`:
  ndarray reductions with the kernel's exact rounding; NATIVE-TODO per op.
- `RESHAPE`, `SQUEEZE`, `EXPAND_DIMS`, `TRANSPOSE`, `CONCATENATION`,
  `STRIDED_SLICE`, `PAD`, `GATHER` (same-tensor): zero-copy views or index maps
  where possible; never a per-call copy of the whole tensor.
- Float ops in hybrid or float models: ALIAS/INLINE onto the same native float
  methods `lance-graph-onnx` uses (`vstanh`, masked sums, blend); TF op
  semantics read from `AdaWorldAPI/Rust`.

### 3.6 Refused in the first scope

Custom ops, TF Select (Flex) ops, control flow (`WHILE`, `IF`), dynamic shapes,
int16 activations (16×8 mode), sparse/compressed weights. Each refuses with the
op name and reason, so a model reports exactly what it is missing.

## 4. Importer and binding

- **Cold importer** (D-NFC allows it outside the hot path): read the FlatBuffer
  once, enumerate subgraphs, tensors (shape, type, quantization params, buffers)
  and operators (builtin code, options, version). No FlatBuffer access after load.
- **Bind once:** each node gets its verdict and its prepared object (table,
  folded bias, requant parameters, view) at load. ⊘ OPEN: "at load" holds only if
  production load-time preparation is admitted. ndarray Codec8 (plan
  `quantized-codec8-v2.md` D8, §9) generates production calibrations and tables at
  build time from a manifest and keeps run-time calibration behind
  `lab-calibration`. Each TFLM TANH/LOGISTIC table is a pure function of one input
  `Calibration`, so build-time models need no new seam; models known only at run
  time do (audit Amendment 1 §A3). No per-call dispatch, no
  string matching in the hot path.
- **Identity:** every table and folded bias is keyed by model digest, subgraph,
  tensor/operator index and parameter set, never by shape alone (Burn
  `CompiledLinear` precedent, `crates/burn/src/ops/matmul.rs:87`).
- **Coverage report:** the importer returns, per model, the list of ops by
  verdict. A model with any REFUSE is reported as not runnable, with the ops
  named.
- **Carriers:** activations live as int8 lanes in SoA; DR/Quack binding follows
  D-NAM-4 where a graph step needs it. No new tensor type.

## 5. Two code domains, kept apart

| | Domain A: TFLite affine int8 | Domain B: PaletteZ / Helix / cosine |
|---|---|---|
| Codes defined by | the model (scale, zero point) | calibration at migration |
| Codec8 space | `Asym` (activations), `Sym` (weights) | separate, per PZL |
| Exactness | bit-exact by construction against TFLM | measured error against the float source |
| Chains | closed by definition | closure gate EXO-129/EXO-132, one-hop and k-hop |
| Plan | this capstone | D-NAM-PZL (#1784), PZL-3 |

Moving a tensor from domain A into domain B is a calibration step at migration,
measured against the TFLM reference, never automatic. The Fisher-Z / softmax /
cosine round trip the operator wants removed is computed once at migration in
domain B, or does not arise at all in domain A.

## 6. Which reference counts as truth

- Primary oracle: the TFLM reference integer kernels at a pinned commit
  (`8f6596d` of the fork), compiled and run offline in tests only.
- Where TFLM builds a LUT with `LUTPopulateInt8`, the oracle is that function,
  which uses f32 and `TfLiteRound`; the native builder reproduces the same f32
  operations.
- UNVERIFIED: whether TFLite's full (non-micro) runtime optimized kernels
  produce identical int8 results to TFLM reference kernels for every op. The
  TFLite full-runtime source is not in our tree. Until checked, each op's
  parity claim names the reference it was checked against.

## 7. Stages (each a separate, separately approved PR)

| D-id | Scope | Repo | Gate (pre-registered) |
|---|---|---|---|
| **D-TFLP-0** | this capstone, asset map, board rows | lance-graph | docs only |
| **D-TFLP-1** | port gemmlowp `tanh`, `logistic`, `exp_on_negative_values` next to `simd_fixedpoint.rs`; build 256-entry Tanh/Logistic tables | ndarray | bit-identical to TFLM `integer_ops/tanh.h`/`logistic.h` for all 256 codes under ≥ 5 real parameter sets incl. extreme `range_radius`; disable run (drop the saturation branch) turns red |
| **D-TFLP-2** | pair tables for ADD, MUL (then SUB, compares) | ndarray | bit-identical to `add.h`/`mul.h` over all 65,536 pairs, ≥ 5 parameter sets incl. fused activation clamp; disable run (drop `left_shift`) turns red |
| **D-TFLP-3** | FC and conv via bias fold + `gemm_u8_i8`/`matmul_i8_to_i32` + `requantize_i32_to_i8` | ndarray | bit-identical to `fully_connected.h`/`conv.h` on random int8 tensors incl. k not a multiple of 16/64 and per-channel params; accumulator range proven for the model's k |
| **D-TFLP-4** | softmax: diff-keyed exp table + row reduction | ndarray | bit-identical to `reference/softmax.h` int8 path incl. rows below `diff_min` |
| **D-TFLP-5** | `lance-graph-tflite` facade: ALIAS/INLINE/TABLE/REFUSE, cold importer, bind-once | lance-graph | depends on ndarray only (+ a FlatBuffer reader on the cold path); no native crate depends on it; identity-keyed tables (two same-shape layers with different weights must not share) |
| **D-TFLP-6** | op census on pinned real int8 models | lance-graph | per model: ops by verdict, quantization modes; ranks the NATIVE-TODO list |
| **D-TFLP-7** | one whole model end to end | lance-graph | output tensor bit-identical to TFLM reference on fixed inputs; report load cost (table builds, µs), memory per op, hot-path ns per element |
| **D-TFLP-8** | domain-B option: Palette256:palette256 cosine tables (PZL-3) | lance-graph | PZL gates; error vs float source; k-hop closure (EXO-132) |

Order: 1 → 2 → 4 can run in parallel with 3; 5 needs 1–4; 6 can start as soon
as the importer reads op lists; 7 needs 5 and the census to show full coverage
for the chosen model.

## 8. Costs to report (never inferred from design)

- Load: table build time per op (256 or 65,536 reference evaluations), bias
  folds, total per model, in µs (cold cost, reported, not gated).
- Memory: 256 B per unary table, 64 KiB per pair table, folded bias per
  channel; total per model.
- Hot path: ns per element per op class vs the TFLM reference on the same host,
  per backend arm (AVX2, AVX-512, VNNI, AMX, NEON) only where it actually ran.

## 9. Open points

1. Accumulator range in GEMM: i32 accumulation of `k` products must not
   overflow for the model's `k`. ⊘ The instruction half is closed: every arm of
   `gemm_u8_i8` and `matmul_i8_to_i32` uses non-saturating `vpdpbusd` (AMX
   `tdpbusd`, WASM `extmul`+`extadd`), all accumulating in i32 (audit 2026-10-26).
   The range half stays open.
2. Full-runtime vs TFLM kernel parity (§6), op by op.
3. Target models for D-TFLP-6/7: to be named (PZL-5 proposes a speech model on
   ARM64; the Pi model is unverified).
4. Int16 activations and 16×8 kernels: out of first scope.
5. Whether pair tables are worth it for ops that are one fused multiply-add per
   element (ADD may be cheaper computed than looked up on wide SIMD); D-TFLP-2
   measures both and keeps the faster, since both are bit-exact.

## 10. Mapping to the reviewed PRs

- #1779 (D-NAM-1 crosswalk JSON): TFLite rows cite the TFLM kernel file and the
  `AdaWorldAPI/Rust` op definition; a TABLE route joins DIRECT/COMPOSE/REFUSE.
- #1780 (D-NAM-2): `lance-graph-tflite` is D-TFLP-5; the BOOL boundary and the
  affine quantization contract are §3.2 and §1.
- #1782 (D-NAM-4): binding of int8 lanes onto SoA/DR/Quack.
- #1784 (D-NAM-PZL): domain B (§5); PZL-1's unary table is D-TFLP-1 when the
  source is a TFLite int8 model.
- NSL #4 (`04a`): "missing int8 accumulator" is closed by §3.4 (existing GEMMs
  plus bias fold); the remaining work is parity with TFLM, D-TFLP-3.
- NSL #8 (`00`, lines 39 and 97): cite `AdaWorldAPI/Rust` as the op-semantics
  source; no TensorFlow runtime dependency.
