# D-NAM-5 — Conditional native→DTO export, oracle parity and capstone closeout

**2026-10-25 • PROPOSED ENGINEERING SPEC / DOCS ONLY.** [EXO-152](https://linear.app/exo-red/issue/EXO-152), [EXO-50](https://linear.app/exo-red/issue/EXO-50). Depends on admitted D-NAM-1 methods, D-NAM-2 dialect adapters, D-NAM-3 lawful normalization, D-NAM-4 native prepared routing. Does not implement import/export here.

## Goal
Do not claim that native algebra automatically converts to ONNX/TFLite/MLIR. Export is the *partial inverse* of an admitted semantic crosswalk, conditional on an external representation with exactly the necessary result and error contract.

For each native method `m` and dialect `d`, export records:
- minimal admitted expression and pinned dialect/operator version(s), typed shape/axis/broadcast/layout contract and final DTO buffer representation;
- guards, scalars/attributes/quantization and whether a multi-node expression or function is necessary;
- pre/postcondition `decode_d(encode_d(m,x)) ≈ eval_native(m,x)` for all fixture-domain x under explicit exact or bounded policy;
- named refusal for absent version, unsupported dtype/effect, impossible packing, ambiguous opset, inaccessible external weights or non-closed approximation carrier.

An external file validly parsed or emitted is never sufficient for a conformance claim. Full ONNX `ModelProto` / TFLite FlatBuffer serialization and trained-model graph coverage are separate future work.

## Three levels of evidence
1. **Semantic admission:** typed and versioned contract, with negative controls, approval scope and source citation.
2. **Executable parity:** independent scalar oracle and separate offline ONNX Runtime/TFLite interpreter/MLIR runner/SQL oracle; the native kernel is not its own only oracle.
3. **Cost/portability:** cold setup, transform, bind, layout/quantization conversion, compute, output allocation, repeated execution; x86 (different SIMD widths), aarch64/NEON and WASM only where hardware/CI actually ran. Report all arms; no invented evidence.

## Mandatory test inventory
- Four-dialect worked fixture: signed rank-1 `Greater → Select/Where → Sum`, guarded i32 range, shape and axes, zero-length input, tails 1/15/16/17/63/64/65, varying thresholds; RelAlg takes a same-table resident lane. Compare direct native, relalg, Quack/R2IL and independent reference arms *when all are actually eligible*.
- Negative: SQL 3VL UNKNOWN vs false, Bag duplicates vs Distinct support, `0*Inf` numeric contraction, FP signed zero/NaN/Tanh ULP policy, wrong TFLite quant zero point, symbolic dynamic dims, external control-flow effects, mismatched boolean packing, wrong codebook generation, stale frame.
- Round-trip: import DTO → native method → DTO export → independently execute. Compare results *and* external shape, dtype, attrs, signed-zero policy and outputs.
- Mutation: removing broadcast check, mask, version gate, overflow refusal, or schema identity causes a targeted test failure; do not rely on doctest failing for unrelated reasons.

## Certifier boundaries
Egglog/DialEgg may justify Boolean associative or admitted effect-safe rewrite; no operator is "certified" by running an optimizer or by passing numerical tests. Proof source/ID/digest, shape/carrier preconditions, replay and independent checking are explicit. No universal floating-point or Palette256 error certificate. If carrier closure is conjectural, mark PROPOSED and refuse repeated lookup composition ([EXO-132](https://linear.app/exo-red/issue/EXO-132)).

## Completion definition
Capstone is **complete** only after: on-main commits for every implemented PR, tests genuinely run (commands/results and independent reference versions), STATUS_BOARD/INTEGRATION_PLANS and Linear mirror updated without contradicting merges, no missing dependencies, no outstanding semantic refusal hidden as success. Draft specs remain proposals, not completed implementation. [EXO-136](https://linear.app/exo-red/issue/EXO-136) enforces verifying landed commits, not merely green draft status.

**Stop signal:** a proposed normalization needs a second graph interpreter, violates the D-RA-2 owner, silently changes K3/Bag/math contracts, or fabricates portability/numerical equivalence. Close that path as refused and retain native execution as source of truth.
