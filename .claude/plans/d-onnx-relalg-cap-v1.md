# D-ONNX-RELALG-CAP — ONNX as import membrane over the existing substrate (S0)

**Status:** PROPOSAL, documentation only, 2026-10-23. No code in this plan's PR. Read at `origin/main` `59538865`. ONNX facts are read from the upstream clone at commit `0c8af065` and cited as `onnx:<path>:line`. Stages S1..S5 are PLANNED and not authorized.

**Corrected statement (replaces the original proposal wording).** ONNX is a read-only interchange membrane. An imported model is refused unless every op is in an explicit admitted subset. Admitted ops lower, as written, onto existing contract set-algebra forms and R2IL/Quack/mask-risc terminals through relalg's planning and lowering. Value-indexed ops (`Gather*`, `Scatter*`, `NonZero`, `Unique`, `Compress`), training graphs and float reassociation are refused. The only certified law remains Boolean-Distinct reassociation (egglog, #1707). Every other stage is a differential test. DR64³ and DR256³ are unimplemented proposals with no measured figures. (Source: opusC, "Minimal corrected capstone sentence", adjusted to add `Compress`, which opusA §1 lists as unsupported.)

**Operator rule (binding), from the brief:** "There is no proof without egglog / eggcc / datafrog / ascent, or at least MeTTa over MORK over a PathMap oracle. Anything planned past it is a hallucination." Hand-written Rust reference evaluators and onnxruntime differentials are TESTS, not proofs.

**The proposal "lance-graph-relalg is the mathematical polyfill landing zone" holds ONLY for the Boolean/compare/reduce fragment** (opusA §2). Value mathematics (arithmetic, `MatMul`, `Exp`/`Tanh`/`Mish`) has NO existing home. That is an OPEN decision for the operator (OP-ONNX-1), not a gap to fill silently.

D-RA-2, verbatim (`.claude/plans/d-relalg-unified-v1.md`): "D-RA-2 = A: Die kanonische Single-Table-Set-Algebra und ihre Rewrite-Regeln gehören in `lance-graph-contract`. `relalg` behält Lowering und relationale Planung. Keine allgemeine relationale AST für JC oder Quack einführen."

## 1. What exists

| Capability | State | Evidence |
|---|---|---|
| relalg `Rel` = All/Empty/Plane/Atom/And/Or/Not; `Measure` = Count/Sum/Min/Max/Keep over I32 lanes | SHIPPED | `crates/lance-graph-relalg/src/ir.rs` |
| relalg atoms compare a lane to a constant only (`I32{lane,op,t}`, `U32{lane,eq,v}`) | SHIPPED | `crates/lance-graph-relalg/src/ir.rs` |
| relalg normalize delegates to the contract set algebra | SHIPPED | `crates/lance-graph-relalg/src/normalize.rs` |
| relalg lowers through R2IL `FoldDialect` to mask-risc; "recursion (μ) and the hop (preimage) are not built" | SHIPPED / NOT BUILT | `crates/lance-graph-relalg/src/lib.rs` |
| relalg is "Not a third frontend syntax" | CHARTER | `crates/lance-graph-relalg/src/lib.rs` |
| relalg value arithmetic, matmul, nonlinear functions | NOT BUILT | BRIEF verified facts (`join_body.rs` is a GYO alpha-acyclicity check only) |
| mask-risc lanes `I32\|U32\|U64\|Strided`, no float lane | SHIPPED | `crates/lance-graph-mask-risc/src/value.rs` |
| R2IL scalar arithmetic is scalar×scalar; lane-vs-lane refused `LaneVersusLane` | SHIPPED | `crates/lance-graph-r2il/src/lib.rs` |
| R2IL refuses `BLEND`, `SCATTER_OR` (`Unimplemented`) and unsigned `IntLess`/`IntLessEqual` (`NoUnsignedLaneCompare`) | SHIPPED (refusals) | `crates/lance-graph-r2il/src/lib.rs` |
| mask-risc `Terminal::BlendI32` kernel, not wired to relalg | LAB (kernel exists, unwired) | `crates/lance-graph-mask-risc/src/ir.rs` |
| Boolean-Distinct composition associativity certified by egglog (`admit_composition_assoc`, `prove_reassociation`, `certify_reassociation`) | SHIPPED (lab crate) | `crates/dr-math-binding-probe/src/relcomp.rs` |
| Certificate reuse on fresh frames, 0 egglog calls, 7/7 | SHIPPED (lab, #1740) | `crates/dr-math-binding-probe/tests/x8_fresh_frame.rs`; `.claude/board/entries/2026-10-23-drmc-l1-x8-fresh-frame-result.md` |
| DR64 register, `MethodSignature`/`Axis`, `Frame`/`Focus`/`Scope` guards | LAB | `crates/dr-math-binding-probe/src/lib.rs` |
| DR256 register (`SLOTS = 256`, `Row = [u64; 4]`), `Qr`, `Composer::compose` | LAB | `crates/dir-rmap-probe/src/focus.rs`; `src/qr.rs`; `src/compose.rs` |
| Independent proof checker | BLOCKED (none exists) | BRIEF; `.claude/board/entries/2026-10-23-drmc-l1-x8-fresh-frame-result.md` ("no independent proof checker") |
| eggcc | BLOCKED (docs/.arxiv only, no code) | BRIEF |
| datafrog, ascent, crepe, MORK, PathMap | LAB (deps/oracles) | BRIEF (crepe-plan-probe, lance-graph-metta, mork-pathmap-probe) |
| ONNX importer (any ONNX crate, `ModelProto`/`GraphProto`/`opset` in `*.rs`) | NOT BUILT | BRIEF: no crate depends on onnx/ort/tract/burn/rten/onnxruntime/wonnx/inkwell/llvm/cranelift; no `ModelProto`/`GraphProto`/opset in `*.rs` |
| bool-tensor ⇄ mask-plane pack/unpack | NOT FOUND (grep only) | opusA §1 facts, §6 item 5 |
| DR64³ / DR256³ plan or code | NOT FOUND | §6 below; BRIEF |

## 2. Six distinct concerns and their owner surface

| Concern | Owner surface in this repo | Source |
|---|---|---|
| Graph interchange | ONNX `ModelProto`/`GraphProto` read at the membrane only; no new AST | opusC B1; `onnx:docs/IR.md:35` |
| Mathematical semantics | `lance-graph-contract` set algebra (single-table); mathematical truth is the contract's | `crates/lance-graph-contract/src/set_algebra.rs`; `.claude/board/entries/2026-10-20-six-decisions.md` §2 |
| Addressing and binding | DR64 `MethodSignature`/`Axis`/`Frame`/`Focus`/`Scope` (lab); ClassView; ADMISSION = B | `crates/dr-math-binding-probe/src/lib.rs`; `.claude/plans/d-relalg-unified-v1.md` |
| Physical execution | R2IL `FoldDialect` → mask-risc; `ndarray::simd` kernels; Quack | `crates/lance-graph-relalg/src/lib.rs`; `crates/lance-graph-quack/src/lib.rs` |
| Learned transitions | typed methods over declared relationships, never a source of mathematical truth (operator rule in BRIEF); CausalEdge64 plasticity; ONNX `training_info` refused | BRIEF; `crates/causal-edge/src/layout.rs`; `onnx:docs/IR.md:110,515` |
| Proof and certification | egglog certificate for Boolean-Distinct composition only | `crates/dr-math-binding-probe/src/relcomp.rs` |

## 3. ONNX coverage matrix (family level)

Facts: default domain 203 operators (155 ops + 48 functions), `ai.onnx.ml` 19; ai.onnx opset 28 is the highest with a schema; IR 14 (scoutB §2-3; `onnx:docs/Versioning.md:204` row `1.23.0|14|28|5|1`). Classes: **D** directly representable (restricted), **C** compositionally representable, **K** needs a specialized physical kernel, **P** needs a mathematical polyfill (no home), **X** external backend, **R** refused. Default for every op not named here is **R**.

Restriction for every D/C row: rank-1, length = `n_rows`, no broadcasting, int32/uint32/bool only, scalar constant right operand (opusA §1, last paragraph). Broadcasting is declared multidirectional (`onnx:docs/Operators.md:659`).

| Family | Class | Lands on | Semantic difference |
|---|---|---|---|
| `Not`, `And`, `Or` (bool) | D | `Rel::Not/And/Or` → `Measure::Keep` → `INT_NOT/AND/OR` (`crates/lance-graph-relalg/src/lower.rs`) | broadcasting refused; bytes⇄bits pack is new intake code |
| `Xor` | C | `(a∧¬b)∨(¬a∧b)`; contract `SetExpr` has no Xor variant (`crates/lance-graph-contract/src/set_algebra.rs`) | result set identical (two-valued) |
| `Equal`, `Less`, `Greater`, `LessOrEqual`, `GreaterOrEqual`, int32 vs scalar constant | D | `Atom::I32` (`crates/lance-graph-relalg/src/ir.rs`; `lower.rs`) | `LessOrEqual`/`GreaterOrEqual` are ONNX functions (`onnx:docs/Operators.md:183,191`): match by op name, do not expand. Output is a mask, not tensor(bool) bytes |
| `Equal` uint32 vs scalar | D | `Atom::U32{eq}` (`ir.rs`) | — |
| `Less`/`Greater` on uint32; lane-vs-lane; float/fp16/bf16/double; int8/16/64; string | R | refused `NoUnsignedLaneCompare` (`crates/lance-graph-r2il/src/lib.rs`); no float lane (`crates/lance-graph-mask-risc/src/value.rs`) | float NaN is two-valued false, not UNKNOWN: relalg's NNF `Not` push-down is unsound with NaN (opusC N1; `crates/lance-graph-relalg/tests/rewrites.rs`). Float comparisons do not enter `Rel`/`SetExpr` |
| `ReduceSum` int32, all axes | C, only in the non-overflow domain | `Measure::Sum` → `Answer::Sum(i64)` (`lower.rs`) | three overflow semantics: ONNX output dtype = input dtype (`onnx:docs/Operators.md:32053-32054`); reference wraps (`onnx:onnx/reference/ops/op_reduce_sum.py:26`); onnxruntime 1.29.0 observed saturating (mechanism OPEN); relalg widens to i64. Out-of-i32 result → refusal `OverflowDomain` |
| `ReduceSum` partial axes; uint32/int64/float | R | Sum is I32-lane only (`ir.rs`) | — |
| `ReduceMin`/`ReduceMax` int32, all axes | C | `Measure::Min/Max` → `Option<i32>` (`lower.rs`) | empty reduction: ONNX yields dtype max/min (`onnx:docs/Operators.md:31444`; `onnx:onnx/reference/ops/op_reduce_min.py:38-44`); adapter `None → i32::MAX/MIN`. Float NaN propagation not representable |
| `ReduceSum(Cast(cond, INT64))` row-count pattern | C | `Measure::Count` → `POP_COUNT` (`lower.rs`) | exact 2-node pattern; output int64 |
| `Where` | K (blocked) | `Terminal::BlendI32` (`crates/lance-graph-mask-risc/src/ir.rs`); R2IL refuses `BLEND` (`crates/lance-graph-r2il/src/lib.rs`) | I32 lanes, lane-vs-lane, value-producing; outside `Rel` |
| `Gather`, `GatherElements`, `GatherND`, `ScatterElements` and other Scatter*, `NonZero`, `Unique`, `Compress`, `TopK` indices, any index-tensor input | R (ADMISSION B) | none | value-derived addressing: `onnx:docs/Operators.md:15895` (Gather) (ScatterElements) (NonZero) (Unique). ADMISSION = B: "A position derived from another relational column or table is not admissible as a read address" (`.claude/board/entries/2026-10-20-six-decisions.md` §1). `Compress` writes compacted values; `Keep` writes a mask only (`ir.rs`) |
| `Add/Sub/Mul/Div/Mod/Pow/Neg/Abs`, elementwise | P | none (R2IL arithmetic is scalar×scalar; `INT_SUB` unchecked at `crates/lance-graph-r2il/src/lib.rs`, ADD/MULT checked at) | home OPEN (OP-ONNX-1) |
| `MatMul`, `Gemm`, `Einsum` | K / X | `ndarray::simd::gemm_f64_tiled` is f64 only (ndarray `src/simd.rs`); f32 delegates to the external `matrixmultiply` crate | ONNX `MatMul` is a sum-product over numeric dtypes, not Boolean composition; no float law is admitted (§4). Boolean relation composition is `relcomp::compose`, which no ONNX op names |
| `Exp`, `Log`, `Tanh`, `Sigmoid`, `Erf`, trig | P | `simd_exp_f32` is a clamped approximation, `#[allow(dead_code)]` (ndarray `src/simd.rs`); `simd_ln_f32` is a scalar loop | numpy ULP semantics are not identical; no `tanh` found (grep only) |
| `Mish`, `Gelu`, `Softmax`, `Relu`, `LayerNormalization` (48 Function-table rows, `onnx:docs/Operators.md:169-218`) | P | bottoms out in Exp/Tanh/Log/Mul | functions may be inlined (`onnx:docs/IR.md:35`) |
| `Shape`, `Size`, `Reshape`, `Squeeze`, `Unsqueeze`, `Flatten`, `Identity`, `Constant` | OPEN | none | intake-time folding possible; unverified |
| `If`, `Loop`, `Scan`, `Sequence*`, `Optional*`, `String*`, `Random*`, `ImageDecoder`, `Quantize/Dequantize*`, `Conv*`, `*Pool`, RNN/LSTM/GRU, `ai.onnx.ml.*` | R / X | none | control flow and subgraphs are a computational-graph format (`onnx:docs/IR.md:35,181-212`) |

**Per-op admission table (opusC B2).** Each ONNX op is classed CO-ADDRESSED (element-wise, `Not/And/Or`, `Equal` over same-table lanes) / SAME-TABLE-SEEK-UNDER-B (only with an `attest_cost` witness, which has no contract type yet: `.claude/board/ISSUES.md`) / REFUSED. Default is REFUSED.

## 4. Intake: what must survive, and how to word compatibility

Fields that must survive (proto2, `onnx:onnx/onnx.proto:9`). Missing or unknown identity/version → refuse, never default:
- Identity/version: `ModelProto.ir_version`; `opset_import` → `OperatorSetIdProto.domain`, `.version`; `NodeProto.op_type`, `.domain`, `.overload`. The (domain, opset) resolution rule is OPEN (not read).
- Local functions: `ModelProto.functions` → `FunctionProto.name`, `domain`, `overload`, `opset_import`, `attribute`/`attribute_proto`/, `node`, `value_info`.
- Types/shapes: `ValueInfoProto.type` → `TypeProto.Tensor.elem_type`, `shape`; `Dimension.dim_value`/`dim_param`/`denotation`. A symbolic `dim_param` must bind to the table's `n_rows`, or refuse.
- Tensors: `GraphProto.initializer`, `sparse_initializer`; `TensorProto.dims`, `data_type`, `raw_data`, `external_data`, `data_location`.
- Attributes: `AttributeProto.name`, `type`, `ref_attr_name`, values. Unknown attribute on a known op → refuse.
- Annotations round-trip, not dropped: `metadata_props` (Model, Graph, Node, ValueInfo, Tensor, Function); `doc_string` (Model, Graph, Node, ValueInfo, Attribute, Tensor, Function); `producer_name/version`/; `quantization_annotation`.
- `training_info` is refused on import (opusC N5; `onnx:docs/IR.md:110,515`). No initializer value is ever a law premise: initializers are data; laws are `AdmittedLaw` values (`crates/dr-math-binding-probe/src/relcomp.rs`).

Typed refusal shape (proposal modelled on `RelError`, `crates/lance-graph-relalg/src/lower.rs`, and `FoldError`, `crates/lance-graph-r2il/src/lib.rs`): `Refusal { node: u32, domain, op_type, opset: i64, reason }`, `reason` ∈ {UnknownDomain, OpsetUnsupported, UnsupportedOp, UnsupportedDtype(elem_type), Broadcast, RankNotOne, DimUnbound(dim_param), UnknownAttribute(name), OverflowDomain, NoUnsignedCompare, ControlFlow, ExternalData, TrainingInfo, ValueIndexedAddress}. Whether strings may enter the contract is OPEN (`crates/lance-graph-relalg/src/ir.rs` forbids strings in the IR). Never coerce, never partial-run. Whole-graph versus per-node refusal is OPEN.

**Wording rule (opusC N2).** Do not write "ONNX-compatible". The ONNX IR says "An implementation must support all operators in the set or reject the model" (`onnx:docs/IR.md:80`) and lists "operators that all implementations MUST support" (`onnx:docs/IR.md:67`). The claim is: "ONNX-subset reader that refuses everything else" — it imports IR ≤14, ai.onnx opset N, the listed op subset, and rejects any model whose `opset_import` names an op outside it. "Compatible" is unearned without upstream backend tests per claimed op (`onnx:docs/OnnxBackendTest.md`).

**No float law.** No float law is admitted. Float ONNX ops lower only as written (fixed evaluation order, analogue of `run_as_written`, `crates/lance-graph-relalg/src/lower.rs`) or are refused. Reassociation is admitted for Boolean-Distinct relations only (#1707). The egglog rule set is structural (`crates/dr-math-binding-probe/src/relcomp.rs`); its soundness rests on OR/AND associativity, which IEEE-754 `+`/`*` lack. No ONNX statement fixing a float summation order was found (OPEN).

**No ONNX-level rewrite pass (opusC C1).** No ONNX graph optimizer (constant folding, fusion, shape-inference passes) and no ONNX interpreter beside mask-risc. The only rewrites are contract set-algebra and egglog-admitted laws.

**Palette256 (opusC N4).** ONNX ops never consume or produce Palette256 values except as opaque u8 codes passed through. Forbidden conversions: Palette256 → float to feed an ONNX op; a stored Fisher-z or cosine column produced by an ONNX graph; a generic ONNX distance op over u8 codes (`.claude/board/ISSUES.md`). Codebook attestation has no contract type yet (`.claude/board/ISSUES.md`).

## 5. Importer dependency options — OPEN for the operator; no decision

Facts: nothing in the repo parses ONNX. `prost = "0.13"` is optional in `crates/cognitive-shader-driver/Cargo.toml`, enabled only by feature `grpc`, under a "NEVER in production binary" comment. `protoc` 3.21.12 is installed in the container (`/usr/bin/protoc`).

| Option | Cost | Risk |
|---|---|---|
| A. prost(-build) codegen from `onnx.proto` | new build dep, protoc at build time | proto2 handling in prost: OPEN; whether an AdaWorldAPI prost fork exists: OPEN (lance-graph CLAUDE.md fork policy) |
| B. hand-written wire reader for the ~15 fields S1 needs | zero deps, hundreds of LOC | a second partial proto decoder to maintain |
| C. existing ONNX crate (tract-onnx, candle-onnx, onnx-protobuf) | brings its own proto, often a runtime | an executor arrives with the parser (none present today) |
| D. no parser in S1: build the graph in-test from Rust values mirroring the proto subset | none | intake of real files untested in S1 |
| E. Python-side intake, scratch only | none in repo | test-only, cannot ship |

Correction to the brief (opusA §5): onnxruntime 1.29.0 IS installed in the container's Python (not in the Rust tree). `onnx.reference` cannot run in the clone because `import onnx` needs the C++ extension (`onnx:onnx/__init__.py:82`). Either oracle is a differential test, not a proof.

## 6. DR64³ / DR256³

**Original plan: NOT FOUND.** Negative-search evidence (BRIEF): no `DR64³`, `DR256³`, `64×64×64` or `64x64x64` string anywhere in the repo (`grep -rIl`, 0 files). Branch `ccr-e97e5d58-1bkumg-cartesian12` is CartesianAddress12/Quad8 shader work, not the plan. GitHub PR search "cartesian": 23 PRs, none is a DR64³ plan. The 12-stage PR sequence is not found. `.claude/knowledge/delta-card-addressing-integration-map.md` has an "18-bit hot envelope = 262,144 ... CONCURRENT mailbox working set", a working-set size and not a DR64³ identity. 262,144 = 2¹⁸ coincides arithmetically; the coincidence licenses nothing and no 18-bit DR64³ identity is defined.

**What DR64 and DR256 are today:** DR64 `Operand { rows: &[u64; 64], validity }` (`crates/dr-math-binding-probe/src/lib.rs`), composition `relcomp::compose` (`relcomp.rs`), `Chain` guards (`relcomp.rs`). DR256 `Row = [u64; 4]` (`crates/dir-rmap-probe/src/focus.rs`), `Composer::compose` (`compose.rs`), guard by `prep::Witness` (`prep.rs`), no `MethodSignature`/`Axis` typing.

**NEW PROPOSAL (from opusB §2), not recovered history.** "DR64³" is the logical index cube X×Y×Z (64 each) over which a contraction of typed DR64 views ranges. It is never a stored object.
- (R1) Chain contraction `R: X→Y`, `S: Y→Z`, `(R∘S)[i,k] = ∃j R[i,j] ∧ S[j,k]`. This is `relcomp::compose` today (`relcomp.rs`); nothing is new except the name.
- (R2) Triangle `(R∘S) ∧ Q` with `Q: X→Z`. Needs one new check not in `Chain` (adjacency only, `relcomp.rs`): `Q.row == R.row` and `Q.column == S.column`. Whether it belongs in `Chain` or a separate admission type: OPEN.
- Guards applying unchanged: same frame issue and still current (`lib.rs`, `relcomp.rs`), same focus, same scope, domain adjacency, TwoValued + Distinct only (`lib.rs`), only `BooleanCompose` admitted.
- Not allowed: a new identity encoding (no packed `(i,j,k)` id); any CausalEdge64 change (S/P/O bytes are palette indices, `crates/causal-edge/src/edge.rs`); a new executor; a gather-shaped hop (`crates/lance-graph-mask-risc/src/graph_hop.rs`); Bag through the DR64 binder (`lib.rs`); any claim that 64³ is implemented.
- Sizes, arithmetic only, computed not measured: 64³ bits = 262,144 bits = 32 KiB; 256³ bits = 16,777,216 bits = 2 MiB. The cube must not be materialized; no measurement shows it cheaper (OPEN). No throughput figure is given for either.

**OPEN tension (bit-position scan).** `relcomp::compose` scans set bits for positions (`trailing_zeros`, `crates/dr-math-binding-probe/src/relcomp.rs`) and DR256 `compose` does the same (`crates/dir-rmap-probe/src/compose.rs`), while `crates/dir-rmap-probe/src/qr.rs` states "no loop scans set bits for a position (the D-RU-6 address rule)". Whether position-scanning inside one co-addressed frame is the forbidden gather (ADMISSION B, `.claude/board/ISSUES.md`) is not ruled anywhere found.

**DR256³ preconditions (opusB §5), none met:** DR64³ R1/R2 pass the S3 arms with the R2IL arm measured; a DR256 binder with DR64 typing (`prep.rs`); a certificate route at 256 (no egglog dep in `dir-rmap-probe`); materialization stays refused (precedent `crates/cognitive-shader-driver/src/immaterial_needle.rs`); `TooLarge` (`qr.rs`) refuses on every axis; CausalEdge64 S/P/O bytes are not DR256³ coordinates; cross-width transfer (256 → 64) has no code, only DR64→DR64 reuse exists (#1740).

## 7. Stages

"Proof engine" values: egglog-certified (only `LawShape::BooleanCompose`), differential test (oracle named), none. "Certified" may appear only in the first case (opusC B4). The operator rule names egglog, eggcc, datafrog, ascent and MeTTa over MORK over a PathMap oracle as the engines a proof may come from. Today only egglog produces a proof object in this repo. eggcc has no code. datafrog and Ascent are dev-dependency oracles here and emit no provenance; their tuple-comparison oracle role was withdrawn on 2026-10-21 (`.claude/board/entries/2026-10-23-coresearch-julia-egraphs.md`). A stage that wants them as its proof engine must first show what proof object they produce (OPEN, OP-ONNX-14).

| Stage | Deliverable | Gate | Proof engine | Prerequisite | Status |
|---|---|---|---|---|---|
| S0 | this plan + board entry | review | none | none | this PR |
| S1 | ONNX intake probe, subset reader with typed `Refusal`; Boolean/compare/reduce fragment lowered onto relalg | S1 falsifier below | differential test (onnxruntime 1.29.0; numpy reference semantics) | importer decision (OP-ONNX-2) | PLANNED, not authorized |
| S2 | admission table and value-maths decision recorded | operator ruling | none | OP-ONNX-1 | PLANNED, not authorized |
| S3 | DR64³ R1/R2 contraction probe, three arms | S3 falsifier below | egglog-certified for reassociation only; all else differential test | S2; OP-ONNX-4, OP-ONNX-5 | PLANNED, not authorized |
| S4 | learned-transition boundary: ONNX `training_info` refused; plasticity typed as method, never premise; Palette256 forbidden conversions enforced | refusal tests with disable runs | none (refusals are tests) | S1 | PLANNED, not authorized |
| S5 | export/codegen evaluation | review | none | S1..S4 | PLANNED, not authorized |

The 64³ Explore → 256³ Validate → Certified Method → 64³ Reuse progression: the "256³ Validate" step is a differential test and must be named so (opusC B4). It has no proof engine today.

**S1 falsifier (opusA §5).** Fixture: one table, `n_rows ∈ {0, 1, 63, 64, 65, 1000}`, lane `x: int32` (incl. i32::MIN/MAX), lane `k: uint32`, one bool plane `b`, symbolic dim `N`.
- Positive: G1 `Less(x,c)`; G2 `And(Less(x,c), Not(b))`; G3 `ReduceSum(Cast(G2, INT64))`; G4 `ReduceMin(x)`, `ReduceMax(x)` (N=0 → i32::MAX/MIN via adapter); G5 `Equal(k,v)`. relalg output equals onnxruntime output; mask words equal packed bools.
- Negative (named `Refusal`, never an answer): `Tanh(x)`; `Less` on float; `Less` on uint32; `Less(x,x)` lane-vs-lane; broadcast `[N]` vs `[N,1]`; unknown domain; `ReduceSum` whose i64 result leaves i32.
- Silent twin: every positive graph yields zero refusals; in-range `ReduceSum` does not raise `OverflowDomain`; `metadata_props`/`doc_string` still lower and round-trip unchanged.
- Disable runs (commit first, assert the patch applied): `Less→Le`; drop the dtype check; map empty Min `None→0`; delete the overflow guard; skip `Not`. Each must turn a named test red.

**S3 falsifier (opusB §4).** Fixture: three bound DR64 views on one `FrameStore` seal, one focus, one scope, axes X→Y→Z(→W); seeded fill as `crates/dr-math-binding-probe/tests/relcomp.rs`. Arms, bit-exact over all 64 rows: (1) `Chain::eval` over every order (`relcomp.rs`); (2) a co-addressed R2IL program (`exists(intersect(R_row_i, S_col_k))`, needs `S` transposed first, a preparation cost reported separately per D-RU-6, `.claude/plans/d-relalg-unified-v1.md`), which does not exist; (3) an independent scalar pair-enumeration oracle (`tests/relcomp.rs`), a differential test; (4) reassociation equality via the egglog certificate only (`relcomp.rs`). Costs reported separately (bind; transpose; steady contraction; result materialization; cold egglog) in ns/µs only. Refusals needing can-fire and can-stay-silent cases: `DifferentFrame` (`relcomp.rs`), `StaleFrame`, `DomainMismatch`, `BagNeedsMultiplicity` (`lib.rs`), focus/scope moved (`relcomp.rs`), `NotThisChain`, and the R2 closing-view axis mismatch (does not exist; must be added and disable-verified).

## 8. What ONNX reuse eliminates, and what it does not

- ONNX replaces at most an interchange AST. It does not remove the proof-side term language: egglog needs its own datatype (`crates/dr-math-binding-probe/src/relcomp.rs`) and ONNX has no rewrite or proof semantics (opusC N7).
- The #1741 Julia-map conclusions P-J1..P-J4 stay open: typed egglog rules via the Rust API, keep `ProofStore`, a standalone replay checker (entry `.claude/board/entries/2026-10-23-coresearch-julia-egraphs.md`). X4 (rules as typed values) and X12 (keep the proof DAG) are proof-side work ONNX does not touch. Anchor A2 ("internal proof representation chosen by measurement") is not pre-decided as "ONNX + relalg" and remains open (same entry).
- Certificate reuse is shown only for `source` changes with the same focus, scope and ABI (`crates/dr-math-binding-probe/tests/x8_fresh_frame.rs`); refusals at.

## 9. Open-points register

| ID | Question | Owner | Dependency | Evidence | Falsifier | Decision | Status |
|---|---|---|---|---|---|---|---|
| OP-ONNX-1 | Home for value maths (arithmetic, MatMul, Exp/Tanh/Mish): relalg cannot take it without a new IR | operator | none | opusA §2; D-RA-2 `.claude/plans/d-relalg-unified-v1.md`; `crates/lance-graph-relalg/src/lib.rs` | any proposal adding value-producing nodes to `Rel` | none | OPEN |
| OP-ONNX-2 | Importer dependency (options A-E) | operator | OP-ONNX-1 for scope | §5 above | S1 reads real ONNX bytes | none | OPEN |
| OP-ONNX-3 | DR64³/DR256³ original plan missing | operator | none | §6 negative search (BRIEF) | the plan is located | none | OPEN |
| OP-ONNX-4 | Does bit-position scanning in one co-addressed frame violate ADMISSION B / D-RU-6? | operator | none | `relcomp.rs`; `compose.rs`; `qr.rs`; `.claude/board/ISSUES.md` | a ruling | none | OPEN |
| OP-ONNX-5 | Independent proof checker; none exists | plan owner | #1741 P-J4 | `.claude/board/entries/2026-10-23-drmc-l1-x8-fresh-frame-result.md` | a replay checker that rejects a tampered proof | none | OPEN |
| OP-ONNX-6 | Cross-focus/scope/ABI rebinding of certificates beyond #1740 | plan owner | S3 | `tests/x8_fresh_frame.rs`; opusC N3 | reuse across stores with different focus/scope/geometry | none | OPEN |
| OP-ONNX-7 | Prover revision not recorded in the certificate (egglog `branch = "main"`). Floating deps are per repo pin policy, so this is a certificate-identity gap, not a dependency defect | plan owner | none | `crates/dr-math-binding-probe/Cargo.toml`; opusC N3 | certificate carries the egglog revision | none | OPEN |
| OP-ONNX-8 | Why onnxruntime 1.29.0 saturates int32 `ReduceSum` (observed in 2 cases) | S1 | S1 | opusA §5 | a third case or the kernel source | none | OPEN |
| OP-ONNX-9 | `.claude/plans/dr-mathematical-compiler-v1.md` calls egglog's self-check "independent" (wording defect) | owner of that plan | none | `relcomp.rs` ("egglog checks every proof it returns") | the wording is corrected | none | OPEN |
| OP-ONNX-10 | `.claude/plans/dr-mathematical-compiler-v1.md` "RelAlg/typed AST expresses admissible mathematics" versus D-RA-2; whether it predates 2026-10-20 | owner of that plan | OP-ONNX-1 | opusC B1 | the plan line is reconciled | none | OPEN |
| OP-ONNX-11 | R2 closing-view typing check: in `Chain` or a separate admission type | S3 | S3 | opusB §2 | disable-verified refusal | none | OPEN |
| OP-ONNX-12 | `observe_focus` omits a scope refusal; intended? | plan owner | none | `crates/dr-math-binding-probe/src/population.rs` | a ruling | none | OPEN |
| OP-ONNX-14 | Which proof object datafrog, Ascent or MeTTa/MORK/PathMap would produce, so they can serve as a stage's proof engine | operator | none | `.claude/board/entries/2026-10-23-coresearch-julia-egraphs.md` | an exported, replayable derivation | none | OPEN |
| OP-ONNX-13 | Whole-graph versus per-node refusal; strings in the contract | operator | S1 | §4; `ir.rs` | a ruling | none | OPEN |

## 10. Housekeeping

SAFE NOW (done in this PR, commits `7fa35361` and `83945785`): D-TGA-5D status; `IDEAS.md` amendments X8 and the certified-compiler row; PR_ARC catch-up #1726–#1741; LATEST_STATE after #1741; D-TGA-5 file attribution.

SEPARATE PR: none identified.

BLOCKED: D-DRMC rows missing from STATUS_BOARD (owner CC2, ask on EXO-50); the `dr-mathematical-compiler-v1.md` wording fix (plan owner; OP-ONNX-9).

## 11. Next gate

The smallest independently reviewable step is the S1 intake probe, pending the importer decision (OP-ONNX-2) and the value-maths scope ruling (OP-ONNX-1). Nothing past S0 is authorized.
