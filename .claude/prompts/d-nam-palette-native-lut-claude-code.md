# Claude Code handoff — D-NAM-PZL source audit and gated palette-native LUT capstone

**Status: READ / REVIEW ONLY. Do not implement, merge, edit board ledgers or create follow-up PRs without separate operator approval.**

Read, in order:
1. `.claude/plans/d-nam-palette-native-lut-capstone-v1.md` (this amendment, EXO-152 owner).
2. `.claude/plans/d-native-first-dependency-correction-v1.md` and `d-native-repr-contract-v1.md` (#1753, supersession).
3. `.claude/plans/d-onnx-relalg-s4-s5-palette-export-v1.md`; `d-onnx-native-v1.md`; `crates/lance-graph-onnx/src/lib.rs` (#1754/#1758).
4. `crates/helix/{KNOWLEDGE.md,src/quantize.rs,src/distance.rs,src/residue.rs}`, `crates/bgz-tensor/src/{palette.rs,attention.rs,fisher_z.rs}`, `crates/bgz17/src/palette.rs`.
5. Sibling `AdaWorldAPI/ndarray/src/hpc/{vml.rs,activations.rs,quantized.rs}`, `src/simd_neon.rs`; inspect any existing Mish primitive and signed/unsigned 8-bit code space.
6. Linear [EXO-152](https://linear.app/exo-red/issue/EXO-152), [EXO-129](https://linear.app/exo-red/issue/EXO-129), [EXO-132](https://linear.app/exo-red/issue/EXO-132), [EXO-127](https://linear.app/exo-red/issue/EXO-127), EXO-50 and PRs #1726/#1731/#1753/#1754/#1758/#1777. GitHub remains code/plan truth.

**Task:** audit whether a source-derived normalized u8 palette can execute a lawful **complete** code-domain unary activation without reconstructing float/cosine and requantizing at each step. Import/source cosine calculations are valid at inbound *migration*, if source model uses cosine. The target optimization eliminates unnecessary **runtime double roundtrips**. Original ONNX/TFLite operator annotations must survive: identity, opset, dtypes, shapes, quantizers, rounding, attributes, graph edge order, provenance. Do not import an ONNX/TFLite computational-graph runtime as native IR.

**Minimal first potential PR, pending approval:** PZL-1 = one offline-generated unary 256-entry table, source-attested [a,b]/source-quantizer interpretation, direct ndarray scalar/NEON lookup, exhaustive 256-code verification, independent oracle and F1/F3/F8 negative controls. No new palette framework, no DR64³ implementation, no new build/execution architecture. Report alternate routes rather than invent APIs.

**Do not confuse:** 
- Helix uniform center `(q+0.5)/256` with TFLite affine scale/zero-point;
- Mish `(x*tanh(softplus(x)))` with vector Softmax;
- Euler-Mascheroni gamma with exponential `e` or a learned calibration constant;
- exact in the discrete code domain with exact in unquantized real arithmetic;
- a closed **set** of codes with operation-preserving abstraction / multi-hop homomorphism;
- sound quantized operator execution with a speedup;
- a model-weight digest with a shape-only cache key;
- cosine migration with a mandatory FisherZ/tanh/cosine runtime.
- documentation measurements with tests that actually ran.

**Output before implementation:** (1) source/owner inventory with SHIPPED/LAB/PROPOSED/UNVERIFIED; (2) one pinned source-model/ONNX-or-TFLite op fixture and exact code-space contract; (3) minimal API reuse proposal; (4) exhaustive unary and adversarial composition/rounding/collision test vectors; (5) independent oracle and measurable Pi3/1GiB acceptance plan; (6) precise blockers and separate PR DAG. Return findings to EXO-152 and coordinate via **a new** EXO-50 message with actual full Claude Code session ID. If unable to verify anything, mark UNKNOWN; never silently expand scope.

**Stopping rule:** if an operator is not closed under the chosen representation, does not commute with quantization at the claimed fidelity, loses annotations, or cannot bind to a source-issued codebook generation, REFUSE. Don't slip the old hydrate/decode/roundtrip back in without an explicit exception.
