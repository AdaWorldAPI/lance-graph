# D-TRACT-HARVEST-0 — Tract as a cold ONNX frontend, admitted onto native methods

**Date 2026-10-30 · Status LAB PROBE IMPLEMENTED (not production) · Owner D-NAM [EXO-152](https://linear.app/exo-red/issue/EXO-152) · Plan [`d-tract-harvest-0-native-algebra-bridge-v1.md`](../../plans/d-tract-harvest-0-native-algebra-bridge-v1.md) · Crate `crates/tract-harvest-probe` (own workspace)**

Runs the Tract arm of D-ONNX-INTAKE S1 and S3 (#1802), nominated by #1810. Burn `onnx-ir` was not run.

**Chain shown:** source-attested ONNX subgraph → Tract `InferenceModel → TypedModel` (cold) → admission (fixed pattern; domain/opset/initializer identity from the raw protobuf) → concrete recipe calling only `lance-graph-onnx` / `ndarray` → ORT with its graph optimizer off.

| fixture | result vs ORT (`ORT_DISABLE_ALL`) |
|---|---|
| tiny (Greater/Where/ReduceSum + Add/Mul/Abs/Sqrt/Div, opset 18) | bit-exact; 256-input div3 sweep bit-exact |
| Jina v5 small retrieval, layer-0 input RMSNorm (10×1024) | 9,259/10,240 exact, max 4 ulp; all deviation is in the one row whose mean is 1 ulp off |

**Findings (source-backed):**
- Tract looks operators up by `op_type` only (`onnx/src/model.rs:135`): `com.example:Abs` imports as `Abs`. Admission refuses from the protobuf domain.
- `InferenceModel::add_const` merges equal-valued initializers (`hir/src/infer/model.rs:202`): `T = 0` and `Z = 0` become one node.
- Declutter rewrites `Div(Y,B)` → `Mul(Y,1/B)`: 161/1,536 sweep elements differ from ORT. It also fuses RMSNorm and Silu (use as pattern evidence, not kernels).
- An impossible broadcast is typed symbolically and fails only at run time; the native broadcast check refuses.
- `Flatten` refuses >1 symbolic dim (`hir/src/ops/array/flatten.rs:11`); the full Jina graph types only with pinned inputs.
- ORT's default optimizer changes the RMSNorm bytes (7,136/10,240 equal to its literal run): the oracle setting is part of the contract.
- Tract's `Reduce<Mean>` is 33 ulp from an f64 reference mean; native and ORT literal are within 2.

**Cost (reported):** Jina slice native 9 µs (9 allocations, every intermediate materialized) vs Tract fused 22.6 µs; full 7,620-node graph cold import + type + declutter 0.4 s with no weight bytes read.

**OPEN:** float reduction tolerance contract (D-ONNX-NATIVE-1); admitted f32 MatMul (first whole-model blocker); Gather reclassification; ORT_DISABLE_ALL as the standing oracle setting.

**Redesign (same day, operator correction "We want the graph"):** the fixed recipes are removed. The ONNX graph is laid onto a DR#3 tile (nodes = Identity, values = Property, `on`/`do_` as 64x64 relations, `dep = do_ ∘ read_by`) and every node resolves through a DT64 ON × WHEN × DO registry: op signature (domain, op_type, dtype, attributes) → one immutable lab classid, WHEN a K3 guard over Tract's facts, DO an existing `lance-graph-onnx` / `ndarray` function. Plans are sealed in a `FrameStore` frame. Tiny fixture bit-exact (incl. div3 sweep; `Where(c, X, 1)` now runs, 17); Jina RMSNorm through the graph 9,259/10,240 exact, max 4 ulp; full model coverage 3,751 WHEN-true / 169 WHEN-not-true (facts Tract drops) / 3,700 unregistered. 20 tests; 7 disable runs each red on the named test. Plan §0.

**Facts from the inference stage (follow-up):** facts now come from Tract's inference stage (every label kept, a shape only when fully determined) merged with the typed stage, with both kept apart for comparison. Jina: 7,620/7,620 values with full facts (typed alone: 2,546), 0 disagreements; coverage 4,146 / 0 / 3,474 (was 3,751 / 169 / 3,700). Typing failure no longer loses the facts. Plan §0.1.
