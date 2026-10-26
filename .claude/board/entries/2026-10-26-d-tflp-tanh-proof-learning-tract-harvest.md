# D-TFLP-EVIDENCE-1 — TANH proof receipts, learning-time calibration, Tract as next harvest

**2026-10-26. Status: EVIDENCE INDEX / ARCHITECTURE HANDOFF (docs only).** This entry does not implement a TFLite model executor, alter native kernels, certify training, or mark D-TFLP-1 DONE.

## The false equivalence to stop repeating

The `D-TFLP-1 = Queued (NOT RUN)` row in [STATUS_BOARD](../STATUS_BOARD.md) is about **future native ndarray gemmlowp TANH/LOGISTIC integration and its product gate**. It is **not** a verdict that the TANH mathematical/source-parity lab experiments were unrun. Conversely, tested lab LUTs must not be described as an integrated native D-TFLP-1 kernel.

**Canonical surviving source of record:** [research branch `claude/bpe-tokenization-architecture-3xd4eh`](https://github.com/AdaWorldAPI/lance-graph/tree/claude/bpe-tokenization-architecture-3xd4eh), frozen evidence anchor [commit `4082dbb79`](https://github.com/AdaWorldAPI/lance-graph/commit/4082dbb79) and [D-TFLP parity audit, Amendments 3, 6.1, 6.2, 6.3](https://github.com/AdaWorldAPI/lance-graph/blob/4082dbb79/.claude/board/entries/2026-10-26-d-tflp-parity-audit.md). These results remain in research history; do not redo or quietly cherry-pick the whole branch to establish that they exist.

### Executed and independently falsified

| Existing lab result | Scope / actual receipt | Evidence anchor |
|---|---|---|
| Original TFLM int8 TANH oracle | Real TFLM `Register_TANH` through `KernelRunner`, 7 × 256 fixture codes; TFLM source `8f6596d` / `5f64e4ad` matching tree; 7/7 fixture calibrations and original TFLM kernel self-tests checked | [audit Amendment 3 §C1](https://github.com/AdaWorldAPI/lance-graph/blob/4082dbb79/.claude/board/entries/2026-10-26-d-tflp-parity-audit.md) · [tanh sweep instructions](https://github.com/AdaWorldAPI/lance-graph/blob/4082dbb79/.claude/probes/tanh-sweep-v1/README.md) |
| TANH source parity and mathematical certification | **2,224 input calibrations × 256 codes = 569,344 cells**: separate `SourceParity` table matches original TFLM kernel bytes 569,344/569,344; interval-certified `CertifiedMath` table matches independent high-precision reference 569,344/569,344. **The tables differ in 17 cells**. Identical numeric bytes for many calibrations do not merge distinct receipt contracts | [audit §§H4, K1–K2](https://github.com/AdaWorldAPI/lance-graph/blob/4082dbb79/.claude/board/entries/2026-10-26-d-tflp-parity-audit.md) · [results.json](https://github.com/AdaWorldAPI/lance-graph/blob/4082dbb79/.claude/probes/source-behavior-v1/results.json) |
| 17 source/mathematical divergences understood | At near-zero 0/±1 output boundaries `|x|≈1/256`; the earliest stage is TFLM `exp_on_negative_values` polynomial truncation. Source trace reproduces the kernel bytes at each stage. **This is a successful counterexample to universal source=Math equivalence, not a failed mathematical implementation** | [audit §K1](https://github.com/AdaWorldAPI/lance-graph/blob/4082dbb79/.claude/board/entries/2026-10-26-d-tflp-parity-audit.md) |
| Source-preserving unary fusion | Composed `TANH → LOGISTIC` 256-code LUT agrees with two TFLM kernels on **569,344/569,344** tested cells, preserving intermediate int8 requantization. Continuous math with removed intermediate rounding differs on 69,652; independent negative controls and receipt swaps deliberately fail | [audit §K4–K5](https://github.com/AdaWorldAPI/lance-graph/blob/4082dbb79/.claude/board/entries/2026-10-26-d-tflp-parity-audit.md) · [compose.json](https://github.com/AdaWorldAPI/lance-graph/blob/4082dbb79/.claude/probes/source-behavior-v1/compose.json) |
| Float `vstanh` is a different contract | Earlier ndarray `vstanh` had an exhaustive 2³²-f32-input AVX-512 precision/symmetry probe with max **1 ULP relative to rounded f64 TANH** on tested tier, NOT a general claim that all f32 outputs are mathematically correctly rounded or equal to TFLM int8 bytes; stage/backend limits documented | [2026-10-24 TANH precision record](https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/board/entries/2026-10-24-tanh-precision-symmetric-i8.md) |
| Related Softmax, separate from TANH | On the FamilyGamma **z-space** path, TANH algebraically cancels before Softmax; this is **not** independent evidence of TANH numerical correctness. The source-compatible per-calibration Softmax recipe matched TFLM on **337,878 admitted output codes**, with 111 exact refusal/abort correspondences in that source corpus. Later merged [#1800](https://github.com/AdaWorldAPI/lance-graph/pull/1800) adds 529 independent reciprocal witnesses, replay, 22 calibration decisions and failure controls; [#1801](https://github.com/AdaWorldAPI/lance-graph/pull/1801) adds Codec8 typed semantic admission | [audit §§E2–E6, L3–L5](https://github.com/AdaWorldAPI/lance-graph/blob/4082dbb79/.claude/board/entries/2026-10-26-d-tflp-parity-audit.md) · [merged Softmax65](https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/probes/source-behavior-v1/SOFTMAX65.md) |

**Vocabulary:** `TESTED_LAB` (observed executions), `CERTIFIED_MATH_ON_DECLARED_FINITE_DOMAIN` (interval proof with explicit quantizer and domain), `SOURCE_PARITY_PINNED_BACKEND` (byte-equality vs stated TFLM build), `NOT_RUN` (future implementation), `UNKNOWN` (new environment). Never say just "TANH exact" without specifying the contract. No result is a universal proof for CMSIS-NN, xtensa, full TFLite, int16, ONNX, arbitrary learning/backprop or different revisions.

**Review gate against redundant work:** another agent must inspect the linked data and failure controls before saying "no PoC was ever run". New expensive sweeps must name a *new specific falsifier*: changed source/kernel/revision, untested parameters or backend, a reproducible defect in fixture derivation, real held-out codebook generation, etc. Mere stage-label `NOT RUN` is not grounds to repeat the existing 569,344-cell study. Prefer existing fast Rust/fixture regressions.

## Next research gate: dynamic learning is NOT static LUT certification

**Cold preparation is proven useful; adaptation during training or online calibration is NOT certified.** A fixed 256-code unary map is only valid for its original `(operator, source, contract, scale/zp, code-space, rounding, codebook/weight generation, intermediate quantization)`. When learning changes activations, weights, calibration statistics or quantizer generation, the prepared recipe and its certificate must be reassessed. Reusing an old table because shape or data bytes coincide is invalid without full receipt equivalence.

Potential **generation-state machine**, not a production implementation: `OBSERVE → PREHEAT → PROPOSE → VALIDATE → COMMIT`. Prepare candidate tables off-path, verify SourceParity and/or CertifiedMath with the appropriate independent oracle, atomically publish a new **source-issued** generation and retain old snapshots for borrowers. Rollback/refuse on changed owner/schema, numeric receipt, model weight digest, code-space policy, or permission/completeness scope. Do not mutate in-place while a borrowed native recipe or result is in use. Reusing a compiled recipe does not authorize a stale escaped Quack `Keep` ([#1792](https://github.com/AdaWorldAPI/lance-graph/pull/1792)).

### The existing 8-byte FamilyGamma envelope needs its own tests

Actual source: [`crates/bgz-tensor/src/fisher_z.rs`](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/bgz-tensor/src/fisher_z.rs), `FamilyGamma { z_min: f32, z_range: f32 }`, `BYTE_SIZE=8`, explicitly serialized little-endian via `to_le_bytes/from_le_bytes`. It calibrates an i8 Fisher-z palette; **this is not a general self-describing, versioned envelope**. The 8 bytes carry no codec revision, NaN policy, codebook epoch, checksum or tenant/source attestation.

Existing [2026-10-24 wire audit §B4 and real-data manifest §D](https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/board/entries/2026-10-24-tanh-precision-symmetric-i8.md) confirms:
- `−128` is a **valid, finite** saturated code here, not a `SymNaN` sentinel; do **not** change interpretation of stored bytes.
- Encode truncates rather than rounding-to-nearest, so `encode(decode(q))` is not identity for every q.
- Unversioned `gamma` and its i8 table are persisted; real safetensors/GGUF model-family and held-out validation remain **NOT RUN** in that audit.

**Separate test gates before adaptive learning binds this representation:**
1. Endian/byte-for-byte roundtrip of existing 8-byte metadata and backward reading of persisted artifacts; reject malformed/nonfinite/zero-range policy violations where a newer wrapper chooses to enforce them, without silently changing legacy decoding.
2. Fit/generation change: old `gamma` + new code table, new `gamma` + old table, identical-shape/different-generation, stale digest/snapshot must **REFUSE**; no numerical cache key impersonating source receipt.
3. Source and mathematical parity on real held-out model-family distributions, including `−128`, out-of-fit/clamp behavior, ranking, quantization bias and `encode(decode(q))` drift; compare independently and report changed-codes/ranking/error rather than assume fixed calibration accuracy.
4. Cold candidate validation vs hot read lifetime, revision promotion/rollback, deterministic replay and generation invalidation. Measure prepare/rebuild/lookup cost and which changes allow reuse.
5. Training/backprop is **another** contract: gradients, STE/surrogate derivatives and optimizer state are not inferred from forward-byte-parity. No live-learning integration until explicit evidence and ownership.

**Output:** tested static proof receipt plus a *new versioned wrapper/binding*, if needed, around the legacy 8-byte gamma payload. Do not enlarge or reinterpret the 8-byte wire format in place. Owner decisions D-NAM/[EXO-152](https://linear.app/exo-red/issue/EXO-152), bgz-tensor/Helix/Codec8, and D-NFR source snapshot/generation.

## Architectural next harvest: Tract (cold), because algebra and representation are separable

**Prioritize investigating [Tract](https://github.com/sonos/tract) before writing a new ONNX/TFLite graph parser or executor.** Tract has distinct `InferenceModel` → `TypedModel` → declutter/optimization → execution stages and a translate-once portable model path (tract-OPL/NNEF); see [upstream pipeline](https://github.com/sonos/tract/blob/main/doc/pipeline.md) and [intro](https://github.com/sonos/tract/blob/main/doc/intro.md). This is a strong *architectural precedent for separating typed algebraic operator meaning and graph transformations from chosen tensor representation, storage and execution*.

**Harvest only an optional cold typed frontend first:** load original graph, infer types/shapes and source constraints, inspect operator semantics and graph rewrites, and emit typed source→native method candidates with provenance. Then bind through existing D-NAM native-method receipts to Quack/R2IL/ndarray/Palette256 where admitted. Tract executor is **an external oracle/reference**, never an unannounced fallback or a dependency of native hot-path crates. A `TypedModel` is not by itself our native IR or a proof of operator parity. The boundary and retained metadata must be measured, not assumed.

[ONNX cold importer capstone #1802](https://github.com/AdaWorldAPI/lance-graph/pull/1802) already describes the paired Burn `onnx-ir` versus Tract bake-off. **This note nominates Tract for FIRST harvest attention, not an unmeasured winner**. Burn remains a legitimate comparison; RTen/ORT supply independent external oracles. Tract's **legacy TFLite loader**, if available at pinned revision, may also be evaluated as cold intake for D-TFLP-5a. Check supported FlatBuffer versions and availability, do not silently promote it to a full-TFLite interpreter.

**Minimum next Tract test (separately scoped PR):**
- One pinned tiny graph + one real encoder model with raw source node/initializer/opset SHA-256 receipts.
- With optimizer OFF, compare original source graph vs typed `InferenceModel`/`TypedModel`; with optimizer ON, measure provenance lost/retained and representation-independent rewrites.
- Record exact dtype/rank/dimension/broadcast/quantization/attribute differences and refuse unknowns, including `Gather` vs relational key joins, numeric MatMul vs Boolean DR, and old calibration vs new generation.
- Lower just one admitted static subgraph to existing native methods and compare outputs against independent ORT/Tract/scalar oracle. No new tensor runtime or source writeback.

**Ownership:** D-NFC [#1753](https://github.com/AdaWorldAPI/lance-graph/pull/1753), D-NAM/[EXO-152](https://linear.app/exo-red/issue/EXO-152), DT64 [#1798](https://github.com/AdaWorldAPI/lance-graph/pull/1798) native numerical leaf, canonical [#1799 D-DT64-Q0](https://github.com/AdaWorldAPI/lance-graph/pull/1799) source binding, #1802 importer comparison. The 64 DT64 predicates are not tensor values, gamma bytes, ONNX nodes or physical DR³ axes.

## Scope of this PR

**DOCS ONLY:** retain a source-linked record of previously executed proof work, expose the difference between `D-TFLP-1 NOT RUN` and `TANH LAB DONE`, identify dynamic-learning/8-byte-gamma missing gates, and set Tract as the next architecture harvest candidate. **No tests run by this docs change.** No existing research branch or generated manifest rewritten; no universal exactness, model import or training parity claimed.
