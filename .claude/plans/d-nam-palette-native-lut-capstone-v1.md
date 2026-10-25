# D-NAM-PZL — Palette-native LUT lowering without redundant hydration (capstone amendment)

> **2026-10-25 · PROPOSAL / DOCUMENTATION ONLY · EXO-152 child trajectory.**
> Parent: [D-NAM / EXO-152](https://linear.app/exo-red/issue/EXO-152/d-nam-capstone-normalized-native-algebra-onnxtflitemlir-dto-method). Coordination: [EXO-50](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id). This amendment is NOT an approval to modify code, add a runtime, merge, retarget active PRs, or claim a complete model importer.
> **Governance:** native-first D-NFC / D-NFR ([#1753](https://github.com/AdaWorldAPI/lance-graph/pull/1753)) governs any contradictions. This plan adds a *particular physical recipe and proof/refusal gate* within D-NAM, not another architecture.

## 0. Operator insight and corrected claim

A source model may require expensive similarity/cosine/normalization mathematics at its **inbound migration / compilation** boundary. That work is not the mistake. The avoidable mistake is to **hydrate an already normalized discrete palette into a floating-point cosine/similarity/intermediate tensor and immediately re-encode it** between compatible operations.

**Native invariant:** after a source-derived value has been admitted into a source-attested normalized PaletteZ code domain, a chain of operations proven to respect that domain executes directly over codes and precompiled LUTs. A hydrate-to-float or repalette step is an **explicit, typed exception**, never an implicit default. ONNX and TFLite annotations are retained as source semantic/provenance witnesses but are never the internal execution graph.

A source-specific cosine evaluation occurs at model migration **only when the source mathematics actually uses cosine**. No mandatory Fisher-Z, cosine, Helix hemisphere, or Euler-gamma roundtrip exists in the runtime operator chain. Do not confuse the mathematical exponential base *e*, Euler–Mascheroni *gamma*, and an ordinary model-specific offset. Any affine constant already absorbed in normalized [a,b] must not be applied again. Any **input-dependent** correction is removable only after an explicit equivalence test, not by calling it an offset.

Do **not** promise "every model can be converted" in one shot: source graph parsing, operator coverage, dynamic control flow, unsupported value indexing, sequence decoding, and streaming state may prevent whole-model admission. A lawful restricted-domain compiler with complete-model refusal is the first deliverable.

## 1. Authority and non-overlap (real source anchors)

| Owner | Exists / role | This amendment may do | Must not do |
|---|---|---|---|
| `.claude/plans/d-native-first-dependency-correction-v1.md`; `d-native-repr-contract-v1.md`; #1753 | Native contract, bind-once ClassId, Quack + DR/SoA, native kernels; representation seams explicitly distinguished | Reference inherited decisions | Replace with ONNX/TFLite AST or runtime dispatcher |
| `crates/lance-graph-onnx/src/lib.rs`; #1754, #1758 | External ONNX vocabulary leaf; native aliases / bounded Tanh and Exp | Keep annotations/opset provenance and differential oracle names | Introduce native dependency on external DTO facade |
| `.claude/plans/d-onnx-relalg-s4-s5-palette-export-v1.md` | Existing BGZ/Helix/ndarray reuse, compiled-artifact identity, closed-carrier and false-positive warnings | Narrow extension for fused palette-domain operator execution | Redevelop quantizer/attention LUT/MatMul/Rel AST |
| `crates/helix/src/quantize.rs`, `distance.rs`, `residue.rs` | RollingFloor [lo,hi] 256 buckets, versioned calibrations, 256² metric LUT, 3-byte endpoints | Reuse **pattern**: offline calibration, version stamps, indexed hot path | Treat Helix distance as neural activation or import its Fisher-Z chain |
| `crates/bgz-tensor/src/{palette.rs,attention.rs,fisher_z.rs}`; `crates/bgz17/src/palette.rs` | Palette and lookup precedent; different metric/representation contracts | Reuse compatible physical codebooks with explicit interpretation | Pretend all palette encodings and pair operators compose |
| `AdaWorldAPI/ndarray/src/hpc/vml.rs`, `src/hpc/activations.rs`, `src/simd_neon.rs` | Native transcendentals, SIMD and ARM64 NEON kernels | Own reference math / table-generation primitive and direct LUT reads | Add hot-path trait-object, per-element calibration or per-call backend selection |
| `crates/lance-graph-relalg`, `mask-risc`, `r2il`, `quack` | Relational support/planning and existing bound native computation | Schedule lawfully composed native methods; preserve typed masks and value lanes | Add `Rel::Mish`, `Rel::MatMul`, universal tensor IR or numeric-matmul-as-Boolean-reachability |
| EXO-129 / #1726 / #1731, EXO-132 | Multi-hop palette composition falsified for non-closed carriers; proposed homomorphism criterion | Make closure/semantic commutation a gate | Infer arbitrary k-hop closure from one LUT or one-hop parity |
| EXO-127 / #1707 / #1740 | D-DRMC Boolean/Distinct associativity and scoped reusable receipts | Cold-law candidates only, under existing proof owner | Call a numeric LUT table or calibration a universal e-graph proof |
| #1777 (D-DTM-1) | Distributed exchange of computational understanding | Optional **downstream consumer** of source annotations and lawful recipes | Make transport/distributed consensus prerequisite to local inference |

Existing documentation: [D-ONNX S0 #1744](https://github.com/AdaWorldAPI/lance-graph/pull/1744), [#1750](https://github.com/AdaWorldAPI/lance-graph/pull/1750), native-first [#1753](https://github.com/AdaWorldAPI/lance-graph/pull/1753), ONNX facade [#1754](https://github.com/AdaWorldAPI/lance-graph/pull/1754), Tanh/Exp [#1758](https://github.com/AdaWorldAPI/lance-graph/pull/1758). Do not re-litigate competing S0 documents: D-NFC is the superseding dependency correction.

## 2. Minimal typed boundary (concept only; no ABI authorized)

**Source witness**: immutable original operator identity (dialect/domain/opset/operator version), model/graph/node identity, input/output edges, dtype, rank/shape and broadcast, scalar attributes, constants, layout/strides, endianness, quantization scales/zero points, rounding/saturation/NaN/signed-zero policy, state/effects, and source digest/producer metadata. Keep annotations unchanged and exportable, not merely translated operator names.

**Native method binding**: existing canonical ClassId/method + prepared direct callable, admitted input/output **palette domains**, calibration generation and source-to-native provenance, owner-issued frame/focus/scope where relevant. The palette label **PaletteZ** is a proposal name for a normalized finite-code carrier, not a claim that a production PaletteZ type/ABI exists on main. A source's asymmetric/symmetric/SymNaN code space is not silently the same carrier as a Helix u8 index or FisherZ signed code.

**Recipe receipt**: source model/layer/weight identity, codebook+calibration digest/version, exact or bounded-approximate declaration, error envelope and observed tests, source op list in fusion order, output quantizer, LUT table identity/hash, backend/ABI version, scope, invalidation rules. A numerical error measurement is not an egglog proof. Source annotations and compiled recipe are related by provenance, **not** by replacing the native model with `GraphProto` or a second AST.

**Hot path**: a prebound `LUT[u8] -> u8/f32` or, for admitted palette-pairs, `LUT[u8,u8] -> u8/f32`, with no per-element `atanh`, `tanh`, `cos`, `softplus`, calibration, materialization to intermediary tensors, runtime operator lookup, backend probing, or implicit decode/re-encode. A f32 external result is allowed as an **explicit output-boundary materialization**.

## 3. Numerical recipes (the source contract decides)

**Helix bucket-center precedent:** `x_q = a + ((q + 0.5)/256)*(b-a)`, not `a + q/255*(b-a)`. That formula represents **Helix's** quantizer only, not the universal TFLite quantizer. For a TFLite quantized tensor, use its source scale, zero-point and per-axis settings exactly. Choose the appropriate independent reference.

For an admitted finite-domain **unary** function f:

`L[q] = Q_out( f( D_in(q) ) )`

Generate once at preparation, enumerate **all input codes**; at runtime use `out[i] = L[in[i]]`. A 256-entry u8 table is 256 bytes, a 256-entry f32 table 1 KiB; output carrier and semantics must be explicit.

For **Mish**: `Mish(x) = x * tanh(softplus(x))`, with `softplus(x)=ln(1+exp(x))`. Generate a fused table by reference evaluation, including stable extremes. Preserve whichever rounding stages the pinned source model observes. *Fused real-math then final quantization* is not automatically identical to *three sequential quantized operations*; if intermediate quantizers matter, either build the composed discrete mapping with their specified rounding at each step or report measured approximation and refuse EXACT.

For a **palette pair** function f, compute `L[i,j] = Q_out(f(Da(i),Db(j)))` using the source model semantics at migration; the cosine option is only correct when f/source explicitly calls for cosine on those representatives. A 256² u8 LUT costs 65,536 bytes before metadata. Static representative pairs do not license exact evaluation on unseen runtime vectors.

**Softmax** is not generally a unary 256-entry table. The vector denominator, max stabilization, axis, temperature/mask, rounding and reduction order remain group-dependent. A LUT may replace `exp` at finite codes, but must still execute/validate the required reduction or prove that an entire vector domain has been encoded. Do not pass Mish off as Softmax or use Euler-gamma as the exponential normalization constant.

**Closure gate:** simple range closure is **necessary but insufficient**. The abstraction h into palette codes must respect the source operation and the required downstream observation: `h(f(x)) = f_palette(h(x))` where claimed, subject to the precise error/rounding contract. Check at one hop **and** repeated compositions, including same-code representatives with different true values, per EXO-129/EXO-132. If h cannot commute, widen/retain required state, call eligible native arithmetic, or refuse. Never silently snap to a palette and claim exactness.

**Structural operations:** not all ONNX/TFLite operators are reducible to LUTs. MatMul/convolution/attention still need reductions and data movement, possibly with palette-assisted kernels; K3/Bag/Distinct/overflow/effects and value-derived addressing do not disappear.

## 4. PR DAG and independent gates (none automatically authorized)

| Gate | Proposed PR scope | Deliverable / kill |
|---|---|---|
| **D-NAM-PZL-0** (this S0) | Documentation and ownership crosswalk | Operator review resolves source-annotation retention and owner seams. No code. |
| **PZL-1** | ndarray offline *single* normalized-domain unary LUT example + direct accessor, no new codec | Exhaustively enumerate 256 codes, positive numerical oracle + disable-run; special code spaces, signed-zero/NaN/saturation as applicable. Reject mismatched input version. |
| **PZL-2** | Fused Mish/Softplus→Tanh→Mul under pinned ONNX/TFLite op contract | Compare fused LUT with source's **actual intermediate rounding** and final outputs over 256 codes and held-out calibration. Contradictory intermediate semantics => bounded-only/refuse; no false EXACT. |
| **PZL-3** | Reuse existing BGZ/Helix-style **pair LUT** with source-cosine *inbound-only* when required | Enumerate 256² pairs; measure distortion and out-of-support assignments. Multi-hop same-code purity/commutation kill from #1731; no automatic value-derived cross-table gather. |
| **PZL-4** | Native binding recipe/admission into Quack/SoA/DR physical view only after owner-issued generation identity is available | Two same-shape model layers with different weights/calibration must refuse wrong table. Rebind on floor generation change, proof/approx receipt retained, no per-call dispatch. |
| **PZL-5** | Real model fixture: ONNX or TFLite speech/voice component, **1 GiB Pi 3 / AArch64 NEON** as test target | Preserve original annotations; independent ORT/TFLite reference transcript/activations; report accuracy/WER, real-time factor, RSS/peak memory, calibration and cold/warm costs, CPU/audio-service coexistence. Fail if unstable audio or incorrect source semantics. |

**First implementation candidate is PZL-1 only**, in its own separately approved branch. PZL-2 and PZL-3 need not block each other after the admission contract. PZL-4 follows identity-owner verification; PZL-5 follows at least one whole-model compatible path. No proposal to build entire speech engine from a Mish LUT. The existing [audio-deck README](https://github.com/AdaWorldAPI/audio-deck/blob/main/README.md) identifies Vosk recognition and Piper synthesis on a Pi **4**, whereas the operator elects a **hypothetical Pi 3 / 1 GiB** performance gate pending hardware inspection. Existing Vosk is not a TFLite source model; pin a real TFLite or ONNX model first. The Pi's 64-bit OS remains to be verified.

## 5. Required negative controls and stop signs

- **F1 double-roundtrip witness:** instrument intermediate hydration/dequantize/quantize counts. For a fully admitted code→code chain, intermediate decode/re-encode events must be exactly **zero**; explicit input/output crossings separately counted. Compare an *intentionally redundant* baseline with direct LUT at identical semantic settings. Disable fusion must restore redundant operations or fail the budget.
- **F2 annotation loss:** change source op version / quantizer axis / zero point / saturation policy / graph shape with identical codebook shape. A stale LUT must fail closed; source model cannot be reconstructed from the reduced recipe alone.
- **F3 rounding:** at least one code where sequential intermediate quantization differs from single final rounding. EXACT claim must fail unless the generator reproduces the source sequence. Signed-zero, NaN, -128-as-NaN and -128-valid cases are not interchangeable.
- **F4 source collision:** different model or weight digest, same dimensions and palette cardinality; cold bind must refuse cache collision (Burn prior issue). No shape-only cache selection.
- **F5 non-homomorphic carrier:** choose two different true inputs mapping to the same code but yielding different output codes, or two hops failing after one hop appears valid. Generic composition must refuse or state bounded loss.
- **F6 softmax:** same element code under two vectors with different denominators must not return an identical falsely-normalized result via unary LUT.
- **F7 out-of-range:** outside [a,b] and extreme cosine inputs must show saturation/error explicitly, not silently impersonate ordinary in-range codes.
- **F8 invalidation:** roll calibration/floor version and/or change model codebook while holding borrowed view. Stale recipe is refused before next access or cannot type-check under owner borrow; no implicit reuse.
- **F9 execution fidelity:** compare against an independent reference (TFLite interpreter / ONNX Runtime only as offline **oracle**) using full model quantization and decode chain. A table's own generator is NOT an independent truth oracle.
- **F10 Pi reality:** report measured AArch64 NEON throughput vs scalar and reference, peak RSS, physical model footprint, audio dropouts, cold compile/transfer and repeated-run cost. "O(1) lookup" excludes preparation, assignment and output decode. If not faster or if audio glitches, report negative finding, do not retrofit a claim.

## 6. Validation/reporting contract

Three independent claims: (A) **semantic correctness** at the declared code/shape/version/reduction boundary, (B) **approximation quality** on true-domain input distribution and downstream task, (C) **economics** after calibration/build/transfer/bind amortization. A wins neither B nor C automatically. Preserve exact vs bounded approximate vs calibrated/experimental vs unsupported. Keep the independent oracle, observed fixtures, source digest, codebook generation and benchmark host visible.

**Scope exclusion:** no new FlatBuffer runtime, GraphProto executor, tensor API, float law certificate, Helix/FisherZ reconstruction path, P64/cognitive-shader-driver dependency, distributed transport, or egglog in hot execution. egglog/DialEgg only when the separate D-DRMC owner supplies an admitted law/recipe; it is not needed to justify a 256-code exhaustive table comparison.

**Repository coordination:** D-NAM [EXO-152](https://linear.app/exo-red/issue/EXO-152/d-nam-capstone-normalized-native-algebra-onnxtflitemlir-dto-method) owns this amendment. Cross-ref [EXO-129](https://linear.app/exo-red/issue/EXO-129/palette256-pair-laws-does-a-calibrated-palettelut-compose-where), [EXO-132](https://linear.app/exo-red/issue/EXO-132/crumb-trail-is-the-carrier-closed-under-the-operation-the-plan), [EXO-127](https://linear.app/exo-red/issue/EXO-127/d-drmc-structural-dr-to-reusable-mathematical-compilation) and [EXO-50](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id). Let board owner add canonical STATUS_BOARD / INTEGRATION_PLANS rows on the next board census rather than racing other owners' huge generated/indexed ledgers.

**Operator decision requested:** Accept/reject this narrower D-NAM-PZL trajectory and nominate the existing codebook/identity owner. Until then all gates are PROPOSED/NOT RUN.
