# D-ONNX-RELALG-CAP S1–S2: versioned ONNX intake and minimal mathematical polyfill

> **⊘ SUPERSESSION — D-NFC (2026-10-23, operator-approved architecture correction; canonical text `.claude/plans/d-native-first-dependency-correction-v1.md` and representation seams `.claude/plans/d-native-repr-contract-v1.md`, both in [#1753](https://github.com/AdaWorldAPI/lance-graph/pull/1753)).**
>
> 1. **Native polyfill API first.** The existing R2IL polyfill (#1752) is native Stage 1 and has no ONNX prerequisite: no importer, no protobuf fork, no target model (root assumption A2).
> 2. **ONNX semantics are external mappings.** This plan's S1/S2 become the ONNX conformance-adapter layer [7] of the canonical diagram: one operator's ONNX definition mapped onto a native call and checked against it. They do not define the native representation (A1).
> 3. **No internal ONNX graph.** `GraphProto` and the operator schemas do not own the internal computational graph or numerical meaning; native typed operations and compiled R2IL procedures do (A1).
> 4. **Native fallback.** The fallback is the exact native method on the uncompressed data. A whole-model delegated ONNX backend is at most an optional external comparison [8], never the canonical fallback (A4). Approximate methods name their error contract; a rank correlation is not an error bound.
> 5. **Identity.** Internal identity is the native binding (Quack resolution under a sealed frame, plus codebook generation, D-NFR S5/S6/S15). ONNX model / node / initializer identity is external provenance carried by the adapter (A7).
>
> Passages below that contradict these points are marked ⊘ in place; nothing else in this plan changes. The #1744 / #1745 / #1750 S0 ownership dispute is unchanged.
>
> **⊘ Layers [1] and [7] as finally drawn (2026-10-23; #1752 §10, #1753, #1754).** Where this note or the plan says "R2IL polyfill (#1752)", read the R2IL **router**: a route resolved once per ClassId onto `ndarray::simd` (through mask-risc where a program is needed), with `FoldDialect` one strategy beneath it (D-MPF-4). Layer [7], the ONNX conformance adapter, is an external **leaf** facade implemented as `lance-graph-onnx` (#1754): ONNX names over existing native functions, no dispatch, and nothing native depends on it.

**Date:** 2026-10-23 · **Status:** PROPOSED / documentation only · **Parent:** PR #1745 (S0)
**Scope:** ONNX semantic preservation and an evidence-driven mapping to current contract + RelAlg; NO universal ONNX import claims or runtime implementation.

## Ground rules
Use official ONNX protobuf, IR, operator schema and version-history definitions for the pinned ONNX release. Opset 22 is a comparison baseline, **not assumed current**; discover current stable release and schema during S1. Operator identity includes domain + versioned opset import + name; functions, opset imports, defaults and custom domains can alter meaning. Preserve external data references securely; prohibit traversal, unsafe symlink resolution or unbounded resource consumption.

Current D-RA-2: single-table set algebra and laws are owned by lance-graph-contract; lance-graph-relalg owns planning and lowering. The existing RelAlg is narrower than tensor algebra. A successful protobuf parse is neither execution nor export support.

## S1 — semantic intake specification
**Haiku inventory:** installed ONNX importer/exporter, Burn/Candle/RTen dependencies, relevant Rust crates, existing ONNX stubs and functions; identify ownership and supported versions from source. Do not add a new dependency based on name recognition.

**Sonnet mapping:** define an importer-to-typed-adapter boundary using existing contracts where viable, preserving:
- graph and subgraph inputs/outputs, node/operator identity, attributes and defaults;
- tensor element types, rank, shape, symbolic/unknown dimensions, broadcasting and axis conventions;
- function definitions and domain/opset imports;
- constant data and secure external-data handling;
- effects/control flow, quantization and numerical interpretation;
- relevant custom metadata; explicit refusals for unknown or unsupported semantics.
Never convert unknown shape to 64 or 256 merely to fit a register.

**Minimal fixtures:** versioned Add/Broadcast, MatMul with a symbolic dimension, a reduction with keepdims and axes semantics, one Tanh, and a deliberately unsupported If/Loop or custom-domain operator. Round-trip semantic-preservation checks should compare the declared operation, shape, attributes and imported values, not just parser success. Unknown/unsupported must give typed refusal. Security tests include unsafe external data paths and resource ceilings.

**Gate S1:** captured official schema version and exact semantic preservation for a small admitted subset; unsupported cases refused with evidence. No claim of full ONNX compatibility.

## S2 — existing RelAlg mathematical polyfill mapping
**Haiku inventory:** inspect RelAlg Rel/Query variants, contract::set_algebra and set_rules, R2IL, mask-risc terminals, ndarray numeric kernels, JC power-sums and Quack. Verify alleged 'four foundational primitives' from code; do not infer a universal IR.

**Sonnet proposal:** map each admitted ONNX family to (1) existing composed semantics, (2) a small explicit polyfill, (3) specialized numeric kernel, (4) authorized external backend, or (5) typed refusal. Any proposed primitive must include a counterexample proving that current typed composition is insufficient.

Core semantics per operator:
- domain/opset/name/version, inputs/outputs and types;
- LogicalShape, FocusWindow and ResultDemand;
- value carrier and algebra/semiring (OR/AND is not numeric Sum/Product);
- reduction axes, multiplicity, NULL/validity, broadcast and overflow;
- effect and state transition;
- independent mathematical oracle and numerical error policy;
- physical strategies permitted without altering meaning.

Use representative operator families: arithmetic, Boolean, reduction, numeric MatMul, one nonlinear function. Gather/scatter, Conv, Attention and Loop are feasibility questions, not promised executors.

**Mandatory counterexamples:**
1. Numeric MatMul cannot be implemented as Boolean reachability by replacing sum/product with OR/AND.
2. K3 TRUE/FALSE/UNKNOWN and SQL Bag multiplicity cannot be erased by 0/1 presence.
3. Dynamic shape or nontrivial broadcasting cannot be repaired by choosing a 64×64 tile.
4. Floating-point reassociation may fail exact bit parity; NaN, signed zero, overflow and infinities require explicit policy.
5. Gather/scatter must not bypass the ban on cross-table value-directed positional gathers.
6. Unsupported control-flow effects or functions must refuse or use an explicitly eligible fallback.

**Gate S2:** a typed and versioned operator mapping matrix, independent oracle fixtures, zero invented primitives, evidence for any new primitive and explicit fallback/refusal. Proposed implementation is separate.

## Matrix schema
| ONNX domain/op/version | Types & shape | RelAlg/contract mapping | Tensor | DR | Hybrid | Focus dependency | Oracle | Numerical policy | Status |
|---|---|---|---|---|---|---|---|---|---|
| populated by scouts from official schemas | | | | | | | | | |

Classification (orthogonal): import, semantic lowering, backend execution, optimization, certification, export. Mark SHIPPED/LAB/PROPOSED/BLOCKED/UNKNOWN independently; no fabricated percentages.

## Agent handoff
Haiku: source census + official ONNX schema inventory. Sonnet: typed matrix and minimal adapter plan. Opus: semantic review of admissions, especially shape, floating-point, effects, value carriers and D-RA-2. No code or board mutation in this docs PR. S3a/b require the contract; S3d may start with selected fixtures once S2 is reviewed.

## References
PR #1745 S0; #1641 D-RELALG-UNIFIED; #1654 D-RA-2; #1682 Quack K3/NULL; #1685 Bag/Distinct; #1707 DRMC associativity; .claude/plans/d-relalg-unified-v1.md.
---

## 2026-10-23 Cross-session review amendment: P0 S0 conflict, value dispatch and interoperability

**Reconciliation gate:** PR #1744 also owns the prospective canonical S0 path `.claude/plans/d-onnx-relalg-cap-v1.md`, colliding with #1745. Do not merge both; obtain operator/EXO-50 ownership decision first. The source-backed #1744 inventory (narrow `Rel`, 14 OP-ONNX open questions, board closeout) remains authoritative *for observed capability*, subject to current-main recheck. This document is a companion proposal, not a replacement S0.

### S2 binding decision candidate — OP-ONNX-1
**Propose:** ONNX `GraphProto` / versioned operator schemas continue to own the computational graph and numerical operator meaning. Existing `lance-graph-relalg::Rel` is only a Boolean/selection/planning subset; do NOT add `MatMul`, `Tanh`, `Conv`, `Gemm` or a general value-expr AST to `Rel`. The *value-math dispatch boundary* is a typed method/backend choice using existing ClassView/method, ndarray numerical kernels and approved external backends; RelAlg may provide a support/filter/demand plan where relevant. A small immutable dispatch descriptor can be proposed only if existing signatures demonstrably lack necessary information; it is not permission for another graph language.
> ⊘ **D-NFC A1: superseded — native typed operations and R2IL procedures own the internal representation; `GraphProto` is external input to the conformance adapter only.**

This is a recommendation for OP-ONNX-1, **not an operator decision**. The semantics contract must survive lowering even if there is no relational involvement. Probe a numeric `Tanh` bypassing DR and one typed support-guided `MatMul` path.

### S1 whole-model policies — OP-ONNX-13
Specify **two modes**, with separate compatibility/evidence claims:
1. `STRICT_NATIVE` (S1 admissible default): refuse the whole model if any node, dtype, shape, attribute, effect or version is outside the admitted native subset. Never partially execute and present an incomplete output.
2. `DELEGATED_GRAPH` (future opt-in research): route unsupported *connected subgraphs* to a specifically authorized ONNX-compatible external backend. Validate graph partition boundaries, execution order, shape/dtype/mutability, value transfer, permissions/scope, dependency closure and full-model oracle parity. External delegation is not a second proprietary runtime or a hidden fallback. No implementation or provider adoption is authorized by this proposal.

An ONNX `Gather` may be unsupported in our *native* binder due to ADMISSION B, but that does not imply its ONNX semantics are intrinsically forbidden for an authorized external tensor backend. Do not conflate local addressing policy with ONNX standard legality. Unresolved effects, state or tensor transfer result in typed refusal even in delegated mode.

### Version and references
#1744 used an explicitly pinned ONNX v1.23.x, IR 14 / ai.onnx opset 28 for its inventory. Treat 1.24.0 development documentation (opset 29) as a distinct unpinned preview until a release is verified; never merge those operator counts silently. Opset 22 is historical comparison. Pin actual schema before using a fixture.

### New falsifiers and stage gate
- VS1: Value-only `Tanh` needlessly requires DR64 or a `Rel` enum extension (design failure).
- VS2: STRICT_NATIVE partially executes unsupported input then returns a plausible complete tensor (refusal failure).
- VS3: Delegated partition drops symbolic dimensions, dtype, state dependencies, execution ordering or safety scope (refusal/fallback required).
- VS4: `Gather` within an independent tensor backend is treated as a forbidden *cross-table local binder gather* without examining the distinct execution authority (classification failure).
- VS5: Claiming compatibility based on parser success or custom-domain serialization (evidence failure).

Gate remains documentation-only: operator chooses value dispatch and delegated-mode governance; implementation is separately authorized, and #1744/#1745 S0 conflict resolved before shared capstone merge.

### Delegation sequencing safeguard (review refinement)
Do **not** assume that splitting one ONNX GraphProto across native and external runtimes can be done without a new scheduler. Evaluate in this order:
1. **Whole-model delegated fallback**: preserve the imported graph unchanged and execute it in an explicitly authorized established ONNX backend when STRICT_NATIVE refuses, with model-level security/version/numerical checks. This avoids inventing a mixed graph executor.
  > ⊘ **D-NFC A4: not the canonical fallback — the exact native method is. A delegated ONNX run is an optional external comparison.**
2. **Mixed subgraph delegation**: only if a source-verified existing scheduling/partition bridge already meets ordering, data transfer, intermediate lifetime, scope and control-flow constraints. Otherwise keep it OPEN, not promised.
3. Measure the overhead of both, including complete model input/output conversions, before treating partial native acceleration as useful.

STRICT_NATIVE remains the only initial native S1 acceptance target. Neither delegation proposal authorizes an ONNX interpreter or scheduler in lance-graph.

---

## 2026-10-23 discovered reuse: ONNX value-math dispatch is not greenfield

The previous "no value math in relalg" is a **boundary** rather than a request to invent a tensor AST. Existing numerical owners already implement pertinent operators:
- `AdaWorldAPI/ndarray/src/hpc/vml.rs::vstanh` is called from Burn `FloatTensorOps::float_tanh` (compiled SIMD hot path with scalar/strided fallback). A version-pinned ONNX Tanh with an admitted dtype/shape may dispatch here after differential numeric policy checks, without passing through DR or adding `Rel::Tanh`.
- `AdaWorldAPI/ndarray/crates/burn/src/ops/matmul.rs` already has `CompiledAttention`, `CompiledLinear` and ordinary `general_mat_mul`. Both compiled methods can be selected based only on d_head or shape; model/layer/weight/codebook authority is absent. This **is not yet admissible general ONNX MatMul execution**.
- `crates/bgz-tensor`, `crates/helix`, `crates/bgz17` already implement different 256-based palettes, FisherZ, quantization, distance and composition tables. They are not numerical semantics substitutes without typed conversion/quality contracts.

**S1 importer must retain sufficient identity to disambiguate:** ModelProto/GraphProto identity, opset domain and node identity, exact weights/initializer digest, layer designation, dtype and shape, immutable codebook/calibration digest, frame, focus, scope and artifact generation. Reconcile with existing signatures before proposing a new type. None of these should be lost when a model is lowered to a physical candidate. A repeated MatMul shape does not prove it is the registered model layer.
> ⊘ **D-NFC A7: these fields are external provenance for the adapter; internal identity is the native binding (frame seal + codebook generation).**

**Per-operator candidate routing has two legality classes:**
1. **EXACT / declared numeric tolerance:** regular `ndarray`/Burn Tanh or GEMM candidate, checked against independent ONNX oracle.
2. **APPROXIMATE OPT-IN:** codebook-compiled attention or centroid linear candidate, only if model/layer/codebook is witnessed and a bounded calibration/error/quality contract is explicitly requested. A correlation result such as bgz-tensor's reported ρ ≥ 0.999 is *not* bit-equivalence or ONNX normative semantics.

**S1 test additions (proposed, not run):** same-shape separate MatMul layers, duplicate d_head across models, stale palette/weight generation, changed Q/K data, NaN/Inf, tanh SIMD versus reference including tails, and no-registration fallback. Reject or fallback on collision. S2 may choose existing numerical dispatch without a new RelAlg value expression variant.

**P0 S0 collision:** #1744, #1745 and #1750 all create the canonical capstone path. Do not merge this stage as a claim of a settled S0 owner. EXO-50 coordinates. Reuse already implemented primitives; new work should only fill verified semantic/binding gaps.

---

## 2026-10-23 independent Optimum offline frontend — #1751

**New separate draft:** [D-OPTIMUM-BGZ-OFFLINE-v1 / PR #1751](https://github.com/AdaWorldAPI/lance-graph/pull/1751). This is **not** a replacement S1 importer, a mandatory runtime dependency, or a competing ONNX graph/AST.

### S1 interoperability refinement
An out-of-tree offline `huggingface/optimum-onnx` adapter may be used to export an operator-approved model and prepare a controlled ONNX reference fixture using existing `main_export`, `BertOnnxConfig` and `validate_models_outputs`. ONNX `ModelProto` remains the graph. On the eventual native path, preserve graph/node/initializer identity, version/opset/IR, dtype/shapes and codebook lineage before promoting any `ndarray` compiled method. Optimum/ONNX Runtime are **not** guaranteed at edge inference.

Optimum `ORTQuantizer` and `ORTCalibrationDataReader` are existing calibration-data and ORT INT8/QDQ *workflow* sources, **not** BGZ WeightPalette codebook constructors or admissible ONNX operator rewrites by themselves. Avoid treating ORT graph optimization/fusion as identity preserving: retain a pinned unoptimized graph and a documented node/weight lineage if using transformed graphs.

### S2 minimal routing test
For a selected compact BERT layer (model revision chosen by operator, not assumed):
1. Optimum-validated original ONNX reference.
2. Existing ndarray tensor/Burn numerical MatMul under the declared dtype and shape semantics.
3. Optional existing BGZ-compiled layer candidate with separate identity and APPROXIMATE policy, **not an exact replacement on shape match**.
4. Typed refusal/fallback for unsupported and identity-conflicting cases.

**Stage gate:** offline conversion evidence does not make full-model native support or exact palette arithmetic true. Preserve strict native-only refusal and the S0 owner blocking conditions (#1744/#1745/#1750). Only #1751 owns offline Optimum adapter exploration. No Python dependency or duplicated importer mandated in lance-graph.
