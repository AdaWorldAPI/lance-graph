# D-ONNX-RELALG-CAP S4–S5 — Palette256 methods, ONNX export and cold codegen
**Date:** 2026-10-23 · **Status:** PROPOSED / documentation only
**Parents:** #1745 S0, #1746 semantics, #1747 register focus, #1748 physical three-arm benchmark.
**Scope:** numeric specializations, method transitions, standards-compatible export and optional native compilation; NOT a production backend release.

## Core ownership constraints
Use existing Palette256, CausalEdge64, ClassView, numerical kernels, R2IL/Quack/mask-risc, mathematical compiler and Thought-Git boundaries. No new causal encoding, no new universal mathematical AST, no additional executor, no claims of general ONNX runtime compatibility or theoremhood from learned state. Keep cold harvest/certify/extract apart from hot execution and binding.

## S4: typed numerical and cognitive-method compatibility
**Inventory first (Haiku):** Palette256 implemented types/codebooks/LUT operations, ndarray SIMD, available NNUE kernels and ONNX-style nonlinear functions, existing learning probes and provenance/Thought-Git integration. Verify what is SHIPPED vs LAB vs PROPOSED.

**Sonnet design proposal:** versioned ONNX function/primitive mapping with explicit typed input/output units. Candidate operations:
- Tanh, Atanh, Mish and quantized activation under declared precision/palette;
- LUT specialization, LUT composition and admissible finite-domain exhaustive checks;
- optional NNUE-style sparse accumulator updates and reset/full-refresh parity;
- Argmin/Argmax with tie, axis, NaN and type policies;
- Hebbian, Bayesian or BNN-style plasticity as separate state transitions, never pure mathematical equivalences;
- cognitive shader dispatch under a declared method/effect contract.

**Strict prohibitions:** Palette256 codes, Fisher-Z, probability, NARS confidence, Hamming distance, signed i4, continuous Tanh and truth evidence are NOT interchangeable merely because their bytes resemble one another. No implicit conversion. Existing CausalEdge64 bit allocation, epistemic declarations and commit rights are not amended by this plan.

A learned candidate can request a state transition. Thought-Git can preserve a receipt/proposal/history but cannot give that candidate mathematical proof. Numeric finite-domain exhaustive parity certifies only the declared finite domain, not real-valued analytic equivalence.

**S4 fixtures/gates:** exhaustive quantized-domain comparison where bounded, independently computed reference values, saturation/overflow/rounding, nonlinear boundary samples, NNUE full refresh vs incremental update, explicit replay and stale-frame negative tests. Classification: exact / bounded approximate / refused. Preserve operator/version and state effects.

## S5: ONNX export, native codegen and certification handoff
**S5a — export round trip:** distinguish ModelProto serialization with a custom domain from execution portability on stock ONNX Runtime. Preserve type, domain/opset imports, attributes, symbolic shapes, metadata and external data contract. Export standard operator fragments first; custom ops must explicitly declare unsupported runtimes, schema/version and fallback policy. Run independent graph semantic comparison for the supported subset; do not treat parse/export success as equivalence.

**S5b — reuse existing native compilation:** compare current direct SIMD/borrowed DR, R2IL/mask-risc and external numerical kernels under the same result demand, cost accounting and admitted mathematical laws. Candidate constant folding, LUT generation, reduction fusion, law-guided re-association and physical-selection recipes must have typed premises and negative counterexamples.

**S5c — optional LLVM AOT/JIT feasibility (not implementation):** only pursue if the #1748 three-arm experiment identifies a substantial uncovered regime. Measure compilation, proof, bind/rebind, code size, dispatch, execution and materialization separately; estimate break-even N for reusable work. Avoid inserting optimizer/egglog/LLVM JIT into hot fold.

**Certification:** #1707/1738 establish bounded Boolean/Distinct associativity issuance; #1740 demonstrates lab fresh-frame certificate reuse for this narrow family. Independent proof checker, broad mathematical rules, target-lowering coverage, runtime wiring and general safe rebinding remain open gates. An egglog internal proof check is not automatically an independent external proof checker. Mathematical certificate ≠ data truth ≠ authority grant.

**Thought-Git:** provenance, replay, evidence/support roots, recipe selection and rejected candidates may be recorded under existing write/version authority. Never promote evidence weights to law certificates; a mathematical identity does not authorize a storage commit.

## Mandatory falsifiers
- F-S4-1 codebook collision / wrong Palette256 metric conversion;
- F-S4-2 LUT quantized parity mistaken for continuous Tanh equivalence;
- F-S4-3 saturation, checked/wrapping overflow or signed-i4 median shifts semantics;
- F-S4-4 NNUE incremental state diverges from full refresh;
- F-S4-5 Bayesian/Hebbian proposal masquerades as pure mathematical rewrite;
- F-S5-1 ONNX export accepted but unavailable custom-op execution hidden;
- F-S5-2 unsupported dynamic shape/symbolic dimension narrowed;
- F-S5-3 cold compilation excluded from break-even when it should be amortized;
- F-S5-4 cert/rebind valid for Boolean/Distinct mistakenly generalized to MatMul;
- F-S5-5 control-flow, memory mutation, authorization and scope bypassed by optimized plan;
- F-S5-6 new executor/AST created without necessity proof.

Record setup, independent oracle, expected behavior, refusal/fallback, and exact repro.

## Stage independence and acceptance
S4 may investigate a small Tanh or Palette256 typed operation immediately after the S1/S2 schema and numeric contracts are approved; it does not require a DR256³ implementation.
S5 native codegen is gated by semantically eligible P1/P2/P3 result and external baseline from #1748; S5 standard ONNX export may be developed independently for the admitted subset.
Separate draft implementation PRs by owner and operator approval; no single giant ONNX/LLVM merge.

**S4 gate:** typed numerical parity/accuracy and replay of explicit updates without promoting learning to proof.
**S5 gate:** versioned, subset-scoped export parity and independently measured specialization benefit, with cold/hot cost split.

## Agent workflow
Haiku evidence inventory, Sonnet plan and comparative matrices, Opus proof/typed metric/effector review plus independent adversarial check; operator/maintainers retain final authorization. Coordinate through EXO-50. GitHub board is canonical for durable statuses. Existing histories remain append-only.

## Crosswalk
#1745, #1746, #1747, #1748, #1700, #1707, #1738, #1740 and existing D-RELALG-UNIFIED / D-DRMC / CE64 architectural docs.
---

## 2026-10-23 Cross-session review amendment: value dispatch, portability, proof authority

**P0:** #1744 and #1745 S0 conflict must be resolved by owner/operator decision. This companion plan is not permission to merge or overwrite either. Preserve #1744's source-backed admission limits, #1742's proof-storage implementation and #1745's three-peer physical strategy as independent evidence.

### S4 value methods
Under OP-ONNX-1, propose retaining ONNX's own node/function semantics as the numerical vocabulary and delegating eligible *typed numerical methods* to existing ndarray/backend kernels without making `lance-graph-relalg::Rel` a generic `ValueExpr` AST. RelAlg supplies relational support/focus where useful. A numerical-only Tanh may bypass DR entirely. All Palette256 metric conversions remain explicit, typed and version-attested; finite LUT parity does not certify real-valued Tanh.

S4 stateful methods and NNUE incremental updates require affected-dependency/invalidation witnesses. TrainingInfo/state mutation cannot enter a supposedly pure ONNX/RelAlg rewrite by accident.

### S5 proof and codegen
PR #1742 has retained egglog's `ProofStore`/`ProofId` in the lab certificate path; egglog internal proof checking is **not an independent checker**. Split law admission, external proof rechecking, input/binding eligibility, numerical approximation and physical cost selection into separate gates. Keep the proof object rather than reducing it to render/hash. Cross-frame reuse from #1740 is narrow, not a general production contract.

ONNX export must make a distinction among standard-operator portable execution, correctly serialized custom-domain graphs and actual runtime support. An authorized delegated-graph research mode (see #1746) may preserve whole-model interoperability via external providers; it is not permission to introduce another proprietary execution runtime.

S5 auto-selection and LLVM AOT/JIT remain research until #1748 demonstrates verified semantic parity and meaningful end-to-end wins. Compilation, bind, proof/replay, cache reuse, data transfer and output materialization all contribute to break-even.
