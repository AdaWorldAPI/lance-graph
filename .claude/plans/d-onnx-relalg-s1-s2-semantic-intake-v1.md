# D-ONNX-RELALG-CAP S1–S2: versioned ONNX intake and minimal mathematical polyfill
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