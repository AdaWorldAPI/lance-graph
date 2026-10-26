# D-ONNX-INTAKE-0 — Cold ONNX frontend bake-off into existing native methods

**Date:** 2026-10-26. **Status:** S0 CAPSTONE / DOCUMENTATION ONLY. No importer, execution graph, production ABI, training, benchmark, or model parity has been implemented by this PR. **Coordinator:** D-NAM / [EXO-152](https://linear.app/exo-red/issue/EXO-152); session handoff [EXO-50](https://linear.app/exo-red/issue/EXO-50).
**Proposal:** Compare Burn `onnx-ir` and Tract as *cold, optional import frontends*. RTen and Candle are selective reference harvests. ONNX Runtime (ORT) is the primary independent execution oracle. Do not make a premature winner decision by advertised operator count.

## 1. The question and the native-first invariant

Can a **pinned, third-party ONNX reader** parse a real, fixed ONNX model, preserve source semantics, admit its typed operators against **existing** AdaWorldAPI native method contracts, and compile at least one complete independent native subgraph, without importing the third-party executor into the normal runtime?

```text
ModelProto (.onnx) + initializer/external weights
         |
         +---- Burn onnx-ir (cold) --------+
         |                                 | source-attested intake
         +---- Tract InferenceModel -------+  shape/type/opset validation
                 -> TypedModel (cold)      |
                                           v
                                  Candidate native-method bindings
                                      D-NAM / EXO-152
                                           |
                 typed ADMIT / COMPOSE / NUMERIC_BOUND / REFUSE
                                           |
                    existing ndarray / lance-graph-onnx / Quack / R2IL
                                           |
                              independently verified outputs

ORT = independent original-graph oracle, not an invisible fallback.
DT64 = source-object semantic binding (separate, optional consumer).
```

- **No new graph interpreter or internal tensor AST.** The imported graph is cold input to preparation, not a new runtime. A minimal diagnostic representation of source nodes/edges/receipts may exist inside the *offline tool only*; it must not be a generic native execution engine.
- **No native dependency reversal.** `ndarray`, `Quack`, R2IL, relalg, mask-risc, and `lance-graph-onnx` must not depend on Burn, Tract, ORT, the importer, or ONNX protobuf.
- **Preserve source semantics.** Model IR/opset/domain, node names and order, operator attributes, dtype/shape/layout, symbolic dimensions, initializers/external tensors, quantization, control-flow/functions and provenance must be represented or explicitly refused. Unsupported means *no* execution, not partial silently guessed success.
- **Independently reusable recipe ≠ semantic authority.** Native method/certificate, model source digest, prepared weight/content digest, source calibration/codebook and consumer binding identity are separate.
- **No automatic conversion to Palette256.** A prepared SourceParity artifact and CertifiedMath represent different numerical contracts. ONNX TensorProto data are not DT64 predicates or ClassView field-presence bits.

This capstone modifies the ONNX S1 **implementation choice**, not the D-RA-2 / D-NFC ownership rules: a custom ONNX wire parser remains a fallback, not the default new code to write.

## 2. Source-backed frontend candidates

| Candidate | Evaluate | Do NOT assume | Reference |
|---|---|---|---|
| Burn `onnx-ir` | Standalone typed ONNX protobuf intake, operator attributes and shape inference, initializer representation, metadata preservation *before* optional simplification | Burn's generated `forward` or entire tensor executor is necessary; all original source annotations survive graph optimization | https://github.com/tracel-ai/burn-onnx (onnx-ir subtree) |
| Tract `tract-onnx` + HIR/TypedModel | `InferenceModel` import -> `into_typed()`, source-to-typed-node correspondence, handling unknown dimensions, decluttered vs raw typed graph | `TypedModel` is a source-faithful serialization; OPL/optimized graph preserves every original node name; execution can be linked out without measuring deps | https://github.com/sonos/tract ; https://github.com/sonos/tract/blob/main/doc/pipeline.md |
| ONNX Runtime / `ort` | External reference outputs and selected intermediate tensors from original graph, on declared CPU execution provider | ORT output makes semantics or proof automatically portable to every backend | https://onnxruntime.ai/ |
| RTen | Rust ONNX execution as a secondary independent oracle or operator-implementation reference | Same accepted operator set or same numerical result for every dtype/opset | https://github.com/robertknight/rten |
| Candle | Weight loading, attention/embedding implementations and optional independent native algorithm comparison | Candle provides a general ONNX importer or must enter our native execution path | https://github.com/huggingface/candle |

**Pin actual dependency commits, versions, and licenses during S1.** Record whether Burn's `onnx-ir` has public, supported export APIs. Record Tract's transitive dependency tree and API-stability risks. No copying large third-party source. Do not silently use `burn::backend::ir` in place of `onnx-ir`; they are different abstractions.

**Decision criteria:** source preservation, admitted node coverage, typed refusal quality, dependency/compile cost, actual cold parse+shape+normalize time, native glue LOC, independent parity. Model execution under Burn or Tract may be an optional *oracle* arm, not the chosen production backend.

## 3. Existing AdaWorldAPI ownership: reuse, do not compete

- [#1753](https://github.com/AdaWorldAPI/lance-graph/pull/1753): native-first; third-party ONNX is an external compatibility leaf.
- [#1754](https://github.com/AdaWorldAPI/lance-graph/pull/1754) and [#1758](https://github.com/AdaWorldAPI/lance-graph/pull/1758): `crates/lance-graph-onnx` exposes narrow existing aliases, inline semantic guards, bounded Exp/Tanh. **Not a ModelProto parser.**
- [#1752](https://github.com/AdaWorldAPI/lance-graph/pull/1752): one existing `Greater→Where→ReduceSum` native composite; compare independent relalg/mask-risc/R2IL/scalar oracles.
- [#1744](https://github.com/AdaWorldAPI/lance-graph/pull/1744), [#1750](https://github.com/AdaWorldAPI/lance-graph/pull/1750), `.claude/plans/d-onnx-relalg-cap-v1.md`: ONNX intake requirements, version/metadata retention, ADMISSION B constraints and typed refusal; revise S1 frontend choice, not the normative constraints.
- [#1779](https://github.com/AdaWorldAPI/lance-graph/pull/1779) through [#1783](https://github.com/AdaWorldAPI/lance-graph/pull/1783): D-NAM source dialect method normalization, ONNX/TFLite leaf, Quack/SoA/DR bindings and eventual export.
- [#1784](https://github.com/AdaWorldAPI/lance-graph/pull/1784): Palette-native LUT and numerical contract with explicit source codebook/calibration semantics.
- [#1792](https://github.com/AdaWorldAPI/lance-graph/pull/1792) (merged): borrowed sources vs escaped stale `Keep`; recipe reuse does not authorize reuse of old data.

- **[#1800](https://github.com/AdaWorldAPI/lance-graph/pull/1800) (MERGED 2026-10-26):** D-TFLP 6.5 *lab-only* source-oracle Softmax fixture replay + independently checked per-calibration reciprocal preparation gate. Includes 529 kernel-confirmed witnesses, bounded-source semantics and receipts; no ONNX model importer or production kernel. Reuse the test patterns/fixtures where numerical operators need falsification, without falsely treating them as transformer model coverage.
- **[#1801](https://github.com/AdaWorldAPI/lance-graph/pull/1801) (MERGED 2026-10-26):** D-TFLP 6.5 *lab-only* Codec8 semantic admission and D-NAM/DT64/Quack handoff, including explicit Asym −128 versus SymNaN refusal. **Not** a shipping TFLite execution backend or general prepared binding store. Cite its `HANDOFF.md` as existing numerical receipt precedent, and do not reimplement it in the importer. Both PRs are now on main; check HEAD ancestry in the future S1 implementation.


### DT64 PR #1794–#1799 crosswalk (read actual draft heads during S1)

| DT64 PR | Relevant constraint for ONNX intake | Scope explicitly **not** inherited |
|---|---|---|
| [#1794](https://github.com/AdaWorldAPI/lance-graph/pull/1794) predicate catalogue | Source-attested 64 semantic slots; never equate slot, ONNX node ID, ClassView presence, DR coordinate, or Palette code | No 64-slot cap on ONNX ops |
| [#1795](https://github.com/AdaWorldAPI/lance-graph/pull/1795) source IAM/Odoo/SAP binding | Tenant/source object/snapshot/ACL/negative authority remain owners of business DTOs | Imported model node names do not authorize source-system data access |
| [#1796](https://github.com/AdaWorldAPI/lance-graph/pull/1796) directed SPO/DR projections | Typed direction, K3, Bag vs Distinct, global/local focus witness; scalar oracle | ONNX `MatMul` is not Boolean relation composition; tensor `Gather` is not an approved cross-table positional join |
| [#1797](https://github.com/AdaWorldAPI/lance-graph/pull/1797) hypothesis/temporal | Valid/observed time and derived/hypothetical evidence remain separate if an ONNX result is later bound to a DTO | A score is not a source-observed fact or durable Thought-Git assertion |
| [#1798](https://github.com/AdaWorldAPI/lance-graph/pull/1798) native method ONNX leaf | **Primary DT64 handoff:** typed method/model digest/opset/dtype/shape/quantization/recipient kind; computed output is `ComputedNumeric` | DT64 never certifies numerical kernels or executes the model |
| [#1799](https://github.com/AdaWorldAPI/lance-graph/pull/1799) capstone | **Canonical D-DT64-Q0** source-attested borrowed Quack binding, not the duplicate QB document | No second global binding store, data replica or mandatory physical DR³ |

**Canonical interface:** [D-DT64-Q0](https://github.com/AdaWorldAPI/lance-graph/blob/research/dt64-predicate-holographic-capstone-20261026/.claude/plans/d-dt64-q0-borrowed-quack-binding-seam-v1.md). Do not edit the Q0 or six independent draft branches from this capstone. Their current docs are proposals, not merged production APIs.

## 4. A falsifiable, staged bake-off, with one fixed model

### S0: This capstone (NOW)
Docs-only PR, crosswalk, acceptance/refusals, source/handoff. No code or tests run.

### S1: Cold importer and source preservation (separate lab PR)
Pin Burn `onnx-ir` and Tract versions. Run a tiny synthetic ONNX graph **plus one actual checked-in or reproducibly fetched ONNX artifact**. Prefer the prior Jina embeddings v2 small (~570 nodes in an earlier inventory), fallback to Xenova BERT-base or another available encoder, recording actual SHA-256 and node count; no assertion of model availability until observed. Avoid using a model so large that fetching it becomes the research result.

Import graph **without optimizer/simplification first**. Compare against raw protobuf/ORT metadata oracle: op_type/domain/opset, overload/function references where present, input/output graph edges, initializer hashes and external-data locations, quantization annotations, attributed node source positions and symbolic dim constraints. Never treat transformed Tract/Burn node IDs as stable original ONNX identities. Count preserved/lost/renamed metadata by class. Refuse or source-sidecar anything that cannot be reconstructed.

**Mandatory adversarial fixtures:** unknown operator/opset/domain; unknown attributes; mismatched symbolic dims; optional inputs; sparse/external initializers; malformed/truncated weights; int8/u8 and signed zero; bool packed mask vs ONNX bool tensor; control flow/local function; dynamic shape or unsupported training metadata.

### S2: Honest node coverage (separate evidence PR, may combine with S1)
For the same **raw** model, report denominators separately:
- original node count; distinct `(domain, op_type, opset, dtype, rank, attributes)` signatures;
- import success for Burn and Tract, rejected nodes and their reasons;
- existing native alias/inline support, composition candidates, bounded/certified alternatives, missing/unsupported;
- static-vs-dynamic shape requirements and initializer bytes read;
- percentage of nodes **covered under exact declared semantic admission**, not merely operator-name similarity;
- reachability of a complete graph from inputs to outputs; a single blocked mandatory node blocks whole-model forward pass.

Categories: `DIRECT_NATIVE`, `COMPOSE_NATIVE`, `BOUNDED_NUMERIC`, `SOURCE_PARITY`, `REFUSE`, `UNVERIFIED`. Each category must carry a source/target anchor and counterexample or independent conformance test. Do not extrapolate upstream Burn/Tract operator coverage to AdaWorldAPI.

### S3: One complete, admitted static native subgraph (separate lab PR)
Start with a small complete graph `Greater(X,T) → Where(C,X,0) → ReduceSum` or a contiguous subgraph actually present in the chosen model. Connect cold typed graph nodes to existing `lance-graph-onnx`/native methods and execute them **without Burn/Tract in the native call path**. Preserve dtype, shape, broadcasting, axis, overflow, layout and source identity, with independent scalar oracle and ORT output checks. Show unsupported nodes refuse, never route implicitly through Tract or Burn.

A native numerical subgraph with a real pinned initializer and nontrivial shape is preferred before claiming any transformer progress.

### S4: Native numeric and quantization parity (conditional later)
Examine MatMul/Gemm, embedding Gather, Reshape/Transpose, LayerNorm and Softmax as actual model blockers. Reuse existing ndarray kernels, D-NAM native method contracts and Palette256 certification. Distinguish `IntegerExact`, bounded FP, `SourceParity` and `CertifiedMath`. Do not automatically substitute prepared LUTs, reorder floating sums, or assume a row-independent Softmax. Quantized artifact identity needs original code-space/scale/zero-point and source/backend where relevant.

### S5: End-to-end forward pass and cost closure (goal, NOT current authorization)
A full BERT/Jina model executed *entirely* through admitted native recipes, compared layer/intermediate/final outputs to ORT under declared numerical tolerance or exact byte equality. Count cold fetch/load, import, simplification, source provenance reconstruction, binding, preparation, weight materialization, memory, and complete inference. This is the first legitimate whole-model claim, **not** S1 or S2.

## 5. Independent falsifiers

1. Same optimized node type but different original op version or quantization → distinct receipt or REFUSE.
2. Original graph node omitted by constant folding → source provenance retained or acknowledged LOST, not secretly reclassified.
3. Equal tensor shape but different model/initializer SHA-256 → no reused numerical binding without valid source identity.
4. Unknown domain/attribute or external data unavailable → REFUSE before execution.
5. `tensor(bool)` source bytes never reinterpreted as native `u64` mask without proved packing.
6. ONNX `Gather` indices are numerical tensor indices and **not** an admissible cross-source positional join under EXO-102.
7. ONNX MatMul cannot lower to Boolean DR composition; semiring and dtype must match.
8. Floating sum reassociation changes results in some cases: do not label as exact.
9. Asym int8 −128 is numerical, SymNaN −128 sentinel; wrong interpretation refused.
10. Safe immutable source borrow vs escaped stale `Keep` must preserve #1792's 1,032-vs-31 refusal/recompute.
11. Model output `ComputedNumeric` cannot silently become DT64 source-observed IAM/Odoo/SAP fact or authorized effect.
12. Tract runnable/oracle works while our native route refuses a missing node: correct status is **NOT NATIVE COVERED**.
13. Independent oracle and a deliberately broken adapter must turn named tests red; inert mutations reported inert.

## 6. Measurements and decision criteria

Measure importer-specific cold dependencies, compiled artifact size, parse/shape/declutter time, memory and source-provenance retention; native prepared bind cost, allocations and end-to-end subgraph execution separately. Record optimizer settings (OFF vs ON), original and transformed node counts, pointer/source correspondence, model SHA-256, importer revisions and CPU/backend. Reproducibility and license/security review are hard gates.

| Dimension | Decision question |
|---|---|
| Source fidelity | Can metadata and initializer identity be verified after transformations? |
| Typed admission | Does the importer expose enough dtype, rank, opset, attributes and symbolic dims to REFUSE correctly? |
| Cold cost | Can importer dependencies stay out of deployed runtime and keep reproducible toolchain/build? |
| Native mapping | How many *actual model nodes* match the existing crosswalk without new kernels? |
| Provenance | Can we provide a source node → typed node → native method/recipe receipt? |
| Correctness | Can a fully admitted subgraph match ORT independent oracle? |
| Economics | Is imported preparation/packing amortized by execution reuse? |

Possible verdicts: `ADOPT_ONNX_IR`, `ADOPT_TRACT_FRONTEND`, `DUAL_FRONTEND` (only if independent value proves it), `KEEP_EXTERNAL_ORACLE_ONLY`, `DEFER`. No "largest supported op count" shortcut.

## 7. Landing sequence and owner decisions

- **This PR S0:** plan + CC handoff + one small board entry. No Cargo/production files altered.
- **S1/S2:** independently gated frontend intake/coverage probe. Prefer one cold-only crate outside deployed workspace if that is least invasive; no burn/tract dependency in production native crates.
- **S3:** one narrow source-attested, independent native subgraph with ORT oracle.
- **S4:** model-blocking numeric primitives in separately owned ndarray/D-NAM PRs; prior Palette256 lab evidence does not authorize new kernels.
- **S5:** full-model attempt only after native coverage and memory budget justify it.

**Owners/open:** OP-ONNX-2 frontend dependency decision; EXO-152 D-NAM numeric methods/receipts; EXO-115 Quack name/method binding; EXO-102/126 positional access and K3/Bag; EXO-127 proof ownership; EXO-46 global/local witness; #1799 Q0 borrowed source authority; EXO-148 later durable hypotheses. Nothing grants permission to write SAP/Odoo/IAM or import live source credentials.

## 8. Reporting contract

Each stage reports `main SHA | model SHA | importer source revisions/flags | raw-node inventory | admitted/blocked counts | typed refusals | provenance losses | ORT comparisons | independent negative tests | cold/hot timings | extra runtime dependencies | NOT RUN | verdict`.

**Governing invariant:** `ModelProto is source evidence. Third-party importers are cold normalization assistants. D-NAM admits a native method. Quack/ndarray execute only the admitted recipe. DT64, if involved, attests business-source semantics separately.`

Do not claim that importing a graph, parsing a Safetensors payload, or running Tract proves that our own native engine can execute that model.
