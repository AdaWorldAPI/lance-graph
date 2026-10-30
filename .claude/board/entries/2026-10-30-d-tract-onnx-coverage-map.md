# D-TRACT-HARVEST: ONNX coverage map and the 346 unregistered Jina nodes

**Date 2026-10-30 · D-id D-TRACT-SHAPE-1 · Status FINDING · Crate `crates/tract-harvest-probe`**

The probe folds every Jina node it has a registration for. 346 nodes have none. This entry maps those 346 against ONNX, Tract and the native ndarray words, so the next waves are scoped by evidence. No code changed.

## 1. Coverage across ONNX, Tract and the probe

| Set | Count |
|---|---|
| ONNX 1.23.2 default-domain ops | 203 (plus ai.onnx.ml 19, preview 1, preview.training 4) |
| Tract bf7ca16 registered op names | 182 |
| of which standard ONNX ops | 163 of 203 |
| of which contrib / ML names | 19 |
| Standard ONNX ops Tract does not register | 40 |
| Probe registered op types | 29 (48 registrations) |
| In both ONNX and Tract, not in the probe (harvestable) | 134 |

The 19 contrib/ML names: ArrayFeatureExtractor, BiasGelu, CategoryMapper, ConstantLike, FastGelu, GatherBlockQuantized, GroupQueryAttention, LinearClassifier, LinearRegressor, MatMulNBits, MultiHeadAttention, Normalizer, ParametricSoftplus, QuickGelu, Rsqrt, ScaledTanh, SimplifiedLayerNormalization, SkipLayerNormalization, TreeEnsembleClassifier.

The 40 standard ops Tract lacks, grouped:
- Sequence and optional types: ConcatFromSequence, Optional, OptionalGetElement, OptionalHasElement, SequenceAt, SequenceConstruct, SequenceEmpty, SequenceErase, SequenceInsert, SequenceLength, SequenceMap, SplitToSequence.
- Strings and text: RegexFullMatch, StringConcat, StringNormalizer, StringSplit, TfIdfVectorizer.
- Training losses: NegativeLogLikelihoodLoss, SoftmaxCrossEntropyLoss.
- Control flow: Loop.
- Detection and geometry: AffineGrid, CenterCropPad, Col2Im, DeformConv, ImageDecoder, LpPool, MaxRoiPool, MaxUnpool, RoiAlign, Upsample.
- Misc: Bernoulli, BitCast, CausalConvWithState, CumProd, Det, LinearAttention, ReverseSequence, SwiGLU, TensorScatter, Unique.

The probe's 29 op types: Abs Add And Cast Concat Constant ConstantOfShape Cos Div Equal Erf Expand Gather Greater LessOrEqual MatMul Mul Pow Range ReduceMean ReduceSum Reshape Shape Sin Sqrt Sub Transpose Unsqueeze Where. All 29 are registered by Tract.

Caveats:
- "Registered by name" means Tract's ONNX frontend has a `reg.insert` for it. It does not prove every attribute or dtype is supported.
- Tract fallback: no Cargo.toml in the Tract checkout mentions ort, onnxruntime, candle, burn, libtorch or tch (0 whole-word hits each). Tract is a pure-Rust runtime and does not delegate to ORT, burn or candle.
- ORT is used in this probe only as the test oracle (python, optimizer off), never at run time.

## 2. The 346 Jina nodes (inputs pinned to [1, 10])

Counts agree with `unreg_sigs.txt` (total 346). No discrepancy found.

Per decoder layer, repeated over 28 layers (336 nodes):

| Role | Op, dtype | Per layer | Total |
|---|---|---|---|
| Rotate-half for RoPE | Slice f32 | 4 | 112 |
| Rotate-half for RoPE | Neg f32 | 2 | 56 |
| Causal-mask slice | Slice bool | 1 | 28 |
| Masking | Where (cond bool, values f32) | 2 | 56 |
| NaN guard | IsNaN f32 | 1 | 28 |
| Attention | Softmax f32, axis -1 | 1 | 28 |
| SiLU | Sigmoid f32 | 1 | 28 |

Once per graph (10 nodes):

| Role | Op, dtype | Count |
|---|---|---|
| Token embedding | Gather f32 [151936,1024] by I64[1,10] | 1 |
| Attention-mask preparation | Cast i64 to bool | 1 |
| | Flatten bool, axis 2 | 1 |
| | Gather bool, axis 0 | 1 |
| | Reshape bool | 2 |
| | Expand bool | 1 |
| Last-token pooling | ReduceSum i64, keepdims 0 | 1 |
| | GatherElements f32, axis 1 | 1 |
| | Squeeze f32 | 1 |

Shapes of the six per-layer groups (from `unreg.tsv`; starts, ends, axes, steps of every Slice are I64[1] constants):
- Slice f32: F32[1,16,10,128] to F32[1,16,10,64] (two per layer, 16-head tensor) and F32[1,8,10,128] to F32[1,8,10,64] (two per layer, 8-head tensor). Each takes a last-axis half of 64.
- Neg f32: F32[1,16,10,64] and F32[1,8,10,64], same shape out.
- Slice bool: Bool[1,1,10,10] to Bool[1,1,10,10]: at sequence length 10 it selects the full range.
- Where, variant A (28): cond Bool[1,16,10,10], X F32[1]=c, Y F32[1,16,10,10], out F32[1,16,10,10].
- Where, variant B (28): cond Bool[1,1,10,10], X F32[1]=c, Y F32[1]=c, out F32[1,1,10,10].
- IsNaN: F32[1,16,10,10] to Bool[1,16,10,10]. Softmax: F32[1,16,10,10] in and out. Sigmoid: F32[1,10,3072] in and out.

Edge-node shapes: Gather F32[151936,1024] with I64[1,10] gives F32[1,10,1024]. Cast I64[1,10] to Bool[1,10]. Flatten Bool[1,10] to Bool[10,1]. Gather Bool[10,1] with constant I64[1,1,1,10] to Bool[1,1,1,10,1]. Reshape to Bool[10], then to Bool[1,1,1,10]. Expand Bool[1,1,10,10]. ReduceSum I64[1,10] to I64[1]. GatherElements F32[1,10,1024] with I64[1,1,1024] to F32[1,1,1024]. Squeeze to F32[1,1024].

Census caveat: the signature column carries the first argument's dtype, so Where is listed as `Where(Some(Bool))`. The value dtype is F32.

## 3. Native candidates per op

Paths are in ndarray 5c6d98b (the version the probe consumes) and lance-graph-onnx.

| Op | Candidate | Note |
|---|---|---|
| Neg | `ndarray::hpc::vml::vsneg` (vml.rs:800); `lance_graph_onnx::neg` (lib.rs:222) | |
| Sigmoid | `ndarray::hpc::activations::sigmoid_f32` (activations.rs:94) | |
| Softmax | `ndarray::hpc::activations::softmax_axis_f32` (activations.rs:390) | Takes ArrayView2. A rank-4 axis -1 softmax needs a reshape to [rows, last] first. |
| IsNaN | `ndarray::simd_masking_ops::not_nan_f32_to_mask` (simd_masking_ops.rs:4086) | Gives NOT-NaN. IsNaN is its complement over the valid bits. |
| Where f32 | `blend_f32` (simd_masking_ops.rs:6150), `blend_f32_scalar` (6172) | See below. |
| Slice | `ArrayBase::slice_axis` / `slice_each_axis` (impl_methods.rs:638, 699) | A view. All Jina parameters are constants. |
| Gather f32 rows | `ndarray::hpc::gather::gather_rows_f32` (hpc/gather.rs:72) | Handles ONNX negative indices per its test name. |
| GatherElements | none in ndarray | Read: `hpc/gather.rs` holds only `gather_rows_f32`; impl_methods.rs has `select` (whole rows/columns), not element-wise index picks. |
| Squeeze / Flatten | `remove_axis` (impl_methods.rs:2717), `into_shape_with_order` (1975), `to_shape` (1877) | |
| Cast i64 to bool, Reshape bool, Gather bool, Expand bool | existing DOs are dtype-matched | Expand handles F32/I64 only (`do_expand`, registry.rs:1679). Needs Bool arms or Bool rows, not new algebra. |
| ReduceSum i64 | composes existing words | Read: `hpc/checked_i64.rs` has `checked_i64` (scalar, line 110) and `checked_binary_i64` (element-wise, 158) but no reduction, so a checked fold over `checked_i64(Add, ..)` covers it. The probe's ReduceSum is i32 only (`lance_graph_onnx::ReduceSum::sum_i32`, lib.rs:279). |

Where in detail. The probe registers Where for i32 and i64 only (registry.rs:1963, 1993), keyed on argument 1's dtype. Its `guard_where` (registry.rs:550) admits Y scalar or Y equal to X in shape. Jina's Where has X scalar (F32[1]=c) with Y a tensor, and X and Y both scalar. The guard must grow an X-scalar case, and the DO must swap or invert the mask for it.

## 4. Insights

- The gap is narrow in kind: 6 op types over 28 layers make up 336 of 346 (`unreg_sigs.txt`). Registering those six unlocks every per-layer node.
- For 4 of the 6 a native ndarray word exists (Neg, Sigmoid, IsNaN via complement, Where via `blend_f32`). Softmax exists for 2-D only. Slice is a view.
- 5 of the 10 edge nodes (Cast, Gather bool, Reshape x2, Expand) are dtype gaps on ops the probe already registers for other dtypes: they need Bool rows, not new algebra. Flatten is not among the probe's 29 op types, so it is a missing op (a view). The brief said 4; the table counts 5, and the table's number is used.
- One genuine native gap: GatherElements (no element-wise index word in ndarray, read). An i64 ReduceSum composes over `checked_i64`.
- Numerics: Neg, Slice, Where, IsNaN, Squeeze, Flatten, Reshape, Gather, Expand and Cast are exact. Sigmoid and Softmax are Bounded (exp), so each needs a measured ULP bound against ORT, like the existing Erf, Cos and Sin rows.
- Census caveat: Where shows as Bool in the census because the signature dtype is the first argument's, while the registry keys Where on argument 1.
- Tract is the right harvest source for semantics: 163 standard ops by name, pure Rust, no runtime fallback. The 40 ops it lacks are mostly sequence, optional, string and training types that an inference graph like Jina does not use.

## 5. Next (proposal, not authorised work)

- Wave A: the six per-layer ops (Slice f32 and bool, Neg, Where f32 with the X-scalar guard, IsNaN, Softmax rank-4, Sigmoid).
- Wave B: the ten edge nodes (Bool rows for Cast, Flatten, Gather, Reshape, Expand; f32 Gather; GatherElements; i64 ReduceSum; Squeeze).
- Then re-run the census and the full-graph `plan::resolve`, expecting 0 unregistered.

Reproduce the census:

```
cargo run --release --example unregistered_census -- fixtures/jina/onnx_model.onnx 10
```

## 6. Outcome (same day, plan §0.10)

All 346 are now bound: 6 generated fields (Neg, IsNaN, Sigmoid; Bool rows for Reshape, Expand, Gather) and 11 hand-written ones. Jina at [1, 10] is 7,620 registered with WHEN TRUE, 0 unregistered. The Cast i64→bool hint was dropped from the table: keyed on the input dtype alone, it pointed at the i64→f32 DO.
