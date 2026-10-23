# D-ONNX-RELALG-CAP: adapt the ONNX ecosystem onto existing lance-graph surfaces (capstone, S0)

**Status:** PROPOSAL (2026-10-23) — S0 docs-only; nothing implemented.

Basis: `origin/main` at `59538865`; ndarray sibling at `1ecbccb`; ONNX tag v1.23.2 (ai.onnx opset 28, IR 14, ai.onnx.ml 5) and tag v1.17.0 (opset 22, IR 10). Every statement is labelled EXISTS (read at the cited file:line), PROPOSED (this document's suggestion), or UNKNOWN (not established by anything read). Evidence base: scout maps, a Haiku inventory, and three Opus reviews (A: minimal semantic integration; B: Cartesian / DR64 integration; C: adversarial). Reviewer C's corrections are binding on this document.

## 1. Announcement and mission

This plan asks one question: how much of the ONNX ecosystem (its operator vocabulary, its versioning, its model files, its reference semantics) can lance-graph adapt instead of re-inventing? Success means eliminated development. A stage succeeds when it removes work that would otherwise be written in-house. A stage that adds a new IR, a new executor, or a new primitive has failed its own mission, even if it works.

S0 (this document) is documentation only. It fixes the architecture, the routing, the coverage map, the stage specs, the open points, and the claims that may not be made. No code, no dependency, no board file is changed by this document.

## 2. Corrected architecture

The pipeline "ONNX graph → RelAlg polyfill → Cartesian projection → native execution" is **false as stated**. Its centre stage cannot carry what passes through it.

Why (EXISTS):

- `lance-graph-relalg` is a single-table Boolean selection over `i32`/`u32` lanes with one terminal measure. `Rel` is `All/Empty/Plane/Atom/And/Or/Not`; `Atom` compares a lane to a constant; `Measure` is `Count/Sum/Min/Max/Keep` (`crates/lance-graph-relalg/src/ir.rs`, enums `Rel`, `Atom`, `Measure`; opus-c item 1, opus-a Q1.1).
- It has no tensors, no shapes, no floats, no arithmetic between lanes, no matmul, no nonlinear functions (consolidated F3).
- The contract rewrite layer refuses IEEE arithmetic and Bag populations by name: `Refusal::Ieee`, `Refusal::Bag` (`crates/lance-graph-contract/src/set_rules.rs`, enum `Refusal`).
- R2IL refuses lane-versus-lane arithmetic (`LaneVersusLane`, `crates/lance-graph-r2il/src/lib.rs:217-220`).
- Nothing in the DR64/DR256 code is a tensor. ONNX `MatMul` and `Einsum` do not accept `tensor(bool)` (opus-b Q5), so no ONNX operator maps to Boolean relation composition.

Putting ONNX numerics into relalg would need either a new tensor IR inside relalg or a false statement about where the operators run. D-RA-2 forbids the first. Verbatim (`.claude/plans/d-relalg-unified-v1.md:22`):

> D-RA-2 = A: Die kanonische Single-Table-Set-Algebra und ihre Rewrite-Regeln gehören in `lance-graph-contract`. `relalg` behält Lowering und relationale Planung. Keine allgemeine relationale AST für JC oder Quack einführen.

### 2.1 Corrected routing (PROPOSED)

The ONNX `ModelProto` is the only graph. There is no second graph format, no new AST, no "ONNX-to-RelAlg IR", and no new relational executor. An intake classifier walks the decoded `NodeProto` list in its mandated topological order and sends each node to exactly one of four owners:

```
ONNX ModelProto (decoded with ONNX's own schema)        <- the only graph
   |  resolve (domain, op_type, since_version) per node from opset_import
   |  bind each value name to an existing carrier
   +-- integer selection/aggregate island -> relalg::Query{Rel,Measure}
   |        -> contract SetExpr normalize -> R2IL FoldDialect -> one mask-risc program
   +-- integer terminals relalg does not expose -> R2IL / mask-risc directly
   |        (GroupSumI32, BlendI32, power sums)
   +-- numeric (float, matmul, norms, activations) -> existing ndarray kernel / ndarray-core view
   +-- everything else -> typed Refusal { node, domain, op_type, since_version, reason }
```

### 2.2 What relalg may receive (EXISTS constraints, opus-a Q1.3 and opus-c item 2)

relalg is the landing zone for **one family only**: predicate selection over one integer population, ending in an integer reduction. A subgraph is relalg-receivable when:

- every tensor is a rank-1 `int32`/`uint32` tensor bound to a resident lane of one population, a bool tensor of the same length (a mask plane), or a scalar integer initializer used as a comparison constant;
- interior nodes are only `Not`, `And`, `Or`, and `Equal`, `Less`, `Greater`, `LessOrEqual`, `GreaterOrEqual` of a lane against a constant (ordered compares on `int32` only, because `NoUnsignedLaneCompare`, `lib.rs:213-216`);
- it ends in one terminator: count (`ReduceSum(Cast(mask))`), `ReduceSum`/`ReduceMin`/`ReduceMax` over all axes of an `int32` tensor restricted by the mask, or the mask itself (`Keep`, as `Compress`/`NonZero` before compaction).

`Xor` enters relalg only as `(a AND NOT b) OR (NOT a AND b)`; `Rel` has no `Xor`. An integer `Where` is a mask-risc terminal (`Terminal::BlendI32`, `crates/lance-graph-mask-risc/src/ir.rs`, enum `Terminal`), not a relalg node. About 16 of the 226 opset-28 names can reach relalg, only in this restricted domain (opus-a verdict 4).

### 2.3 What does not land in relalg

- **Numeric ops** (float elementwise, `MatMul`, `Gemm`, `Conv`, norms, softmax, quantized gemm) land on existing ndarray kernels (`hpc/vml`, `hpc/activations`, `hpc/linalg/*`, `hpc/blas_level3`, `hpc/quantized`) and ndarray-core views.
- **Gather, GatherElements, GatherND, Scatter, ScatterElements, ScatterND, TensorScatter** are typed refusals on the substrate path. They take indices from tensor values, which are value-derived positional reads and writes. ADMISSION = B keeps cross-table positional gathers forbidden (`d-relalg-unified-v1.md:16,42`); `STATUS_BOARD.md:178-181` marks D-QMB-0..3 "VIOLATION — GATHER. DO NOT BUILD". Two narrow exceptions need written proof per case: (a) indices that are a compile-time constant initializer equal to a static permutation or slice, which lowers to a view; (b) an embedding lookup recast as an attested frozen-codebook lookup under ADMISSION = B. Inside a plain ndarray float kernel Gather is a kernel concern, and its result is never re-labelled a substrate population.
- **Control flow** (`If`, `Loop`, `Scan`) is a typed refusal. R2IL branches only on scalars (`BranchOnPopulation` refused, `lib.rs:229-233`).
- **`ai.onnx.ml`**, sequence, map and optional types, string ops, random ops, and the `ai.onnx.preview.training` domain are typed refusals or unsupported (matrix, section 6).
- **ONNX graph optimizers** (onnx-simplifier, ORT graph transforms, tract `declutter`) are not part of the path. A pre-simplified model is re-checked as input and never trusted as a rewrite. Set rewrites happen only in `lance_graph_contract::set_rules`. egglog stays a cold certifier and gains no runtime role. There is no second executor; candle, ort and tract may appear only as differential oracle arms in a lab crate, with timings in microseconds labelled "reported, not gates" (opus-c item 3; `CLAUDE.md` scoped exception on cold costs).
- Planning goes through ogar-loco/ogar-r2il, never DataFusion (`CLAUDE.md`, ruling of 2026-09-05).

### 2.4 Result of the operator classification

Of the 226 opset-28 names (155 operators, 48 functions, 4 training, 19 ml), opus-a counts: directly representable 1 (`Not`), compositionally representable 93, requires kernel 91, external fallback 4, unsupported/unknown 37 (34 unsupported, 3 unknown), **requires polyfill 0**. Zero new mathematical primitives are required (opus-a Q4). Every gap is a physical kernel, an importer-side shape/dtype decision, or a refusal. "Requires polyfill = 0" is a statement about primitives, not about work: the importer, shape inference, about 60 missing float kernels, and the semantic bridges in section 7 remain to be built.

Reviewer C binds the wording: calling relalg "the polyfill landing zone for ONNX" is false. It is the landing zone for integer predicate selection with a reduction.

## 3. Six-layer separation

| Layer | Lives in (EXISTS unless marked) | Must NOT do |
|---|---|---|
| 1. Interchange | ONNX `ModelProto` / `onnx.proto` at a pinned tag. PROPOSED: a decoder in a standalone probe crate. | Define a second graph format, a persisted IR, or custom domains for triples. |
| 2. Math semantics | ONNX operator specs (155 ops + 48 functions at v1.23.2); `lance_graph_contract::set_algebra`, `set_rules`, `algebra_law` for the integer set algebra. | Certify float rewrites; add tensor, shape, float or dataflow nodes to `Rel` or `SetExpr`. |
| 3. Relational addressing and binding | ClassView, `WideFieldMask`, relalg `Query`, `BoundDr64` with frame/focus/scope guards (`crates/dr-math-binding-probe/src/lib.rs:217-273`). | Carry bag multiplicity, NULL, rank, evidence or a proof in a bit; add a general relational AST. |
| 4. Physical execution | R2IL `FoldDialect`, mask-risc `MaskOp`/`Terminal` (integer and bit lanes), `ndarray::simd` and `ndarray::hpc` kernels. | Grow a second executor; add an LLVM AOT path; add a mask-risc opcode just to make something lower. |
| 5. Learned state transitions | cognitive-shader-driver, thinking-engine, NNUE-style accumulators, learned LUTs (PROPOSED treatment: versioned state). | Act as a rewrite rule, an egglog fact, an `algebra_law` entry, or a certifier. |
| 6. Proof and certification | egglog as cold certifier (`crates/dr-math-binding-probe/src/relcomp.rs` `fn certify_reassociation`; `crates/egglog-r2il-probe`). | Claim independent checking (the checker is egglog's own); certify float, Bag or Ieee anything; carry a proof across shape, codebook, frame version or snapshot. |

Rules are not learned weights. A plastic weight, an NNUE accumulator update or a learned table is state: versioned and replayable. "Learned f approximates ONNX subgraph g" is an empirical claim with a held-out error bound, never an equivalence (opus-c item 9; `dr-mathematical-compiler-v1.md:21`). A method that consumes a learned table produces a proposal; it cannot certify and cannot feed a certifier.

## 4. Crosswalk: what this builds on

| Item | Role here | Status (as read) |
|---|---|---|
| `.claude/plans/d-relalg-unified-v1.md` | Owns D-RA-2 = A and ADMISSION = B; states `Rel` is selection only (`:53`) and "256³ is a paging scheme. It is not a resident address space." (`:231`). | PROPOSAL; D-RA-2 executed (#1654/#1655, per the ledger entry). |
| `.claude/plans/dr-mathematical-compiler-v1.md` (D-DRMC) | Three contracts: semantic (ClassView), binding (frame/focus/scope), mathematical compilation (cold, replayable). "MeTTa is a declaration frontend, not the numeric executor." | PROPOSAL, amended #1708. |
| d-qr-classview-binding-isa-capstone-v1 | Capstone for ClassView binding and the ISA split. Not re-read for S0. | UNKNOWN beyond its name. |
| `.claude/architecture/cartesian-epistemic-self-model.md` | "Sealed Cartesian reference frame"; `:68` "i is the same Cartesian row coordinate for both operands." Names the row-coordinate frame, not tensors. | EXISTS. |
| #1707 (D-DRMC-5) | `certify_reassociation`: egglog-proved associativity of Boolean-Distinct composition over DR64. The only certified law. | Merged. |
| #1738 | Board + dr-math: "D-DRMC-5 already certifies associativity; correct two stale claims" (merge `05b3516e5`). | Merged. |
| #1740 (L1) | Reuse of a certificate on a fresh frame of the same shape, 0 egglog calls (`tests/x8_fresh_frame.rs`). | Merged. |
| #1741 | Julia coresearch: X10 "jlrs: embedding a full foreign runtime conflicts with single Rust binary / ns hot path (A4)"; P-J1 "build egglog rules through egglog's Rust API instead of strings" (open probe). The same argument applies to ONNX runtimes. | Merged. |
| #1742 | Keeps egglog `ProofStore` + `ProofId` in `RecipeCertificate`; adds a D-DRMC section to STATUS_BOARD. | Merged (`801c8b1f4`). |
| #1743 | Pre-registration of the GF(2) Gaussian elimination probe (board entry only, nothing run). | Merged (`f0b046b9f`). |
| D9 (`elegant-herding-rocket-v1.md:947-1105`, `STATUS_BOARD.md:2329`) | `TripletGraph::to_onnx()` "with custom ops for triples", `ArcPredictor`. 0 hits in `crates` for `to_onnx`/`ArcPredictor`. | Backlog. Conflicts with portable export; see open point OP-8. |
| DU-1 (`unified-integration-v1`, `STATUS_BOARD.md:2522`) | ONNX persona classifier (`style_oracle: Option<&OnnxPersonaClassifier>`). | Queued. Narrows to "import and run its graph through the numeric island" if its ops have kernels (opus-a Q5). |
| ScenarioWorld (`docs/ScenarioWorldCounterfactual.md:163`) | "ONNX — branch-local model dispatch" via ModelBinding. | EXISTS as text. |
| `soa-centroid-attention-field-synthesis-v1.md:69-73` | "ONNX = the perceptual encoder into the field", hosted. | EXISTS as text. |

All four prior ONNX mentions treat ONNX as a model runtime or an export format, not as a math IR lowered onto relalg (consolidated F2). This capstone must say whether it supersedes, absorbs or ignores them. PROPOSED: D9's custom-op export is superseded for portable export (a custom domain would not be portable); DU-1 and the ScenarioWorld dispatch are absorbed as consumers of the numeric island; the centroid-attention encoder stays independent. This is an operator decision (OP-8), not a settled one.

No ONNX parser or runtime exists in any `Cargo.toml` (`ort`, `tract`, `rten`, `wonnx`, `onnx`, `onnx-protobuf`: 0 hits; `ModelProto`/`GraphProto`/`NodeProto`/`opset`: 0 hits in `crates/**/*.rs`). `candle` 0.9 is optional in thinking-engine (`calibration` feature, `crates/thinking-engine/Cargo.toml` `[features] calibration`); `prost` 0.13 is optional and LAB-ONLY in cognitive-shader-driver (consolidated F2).

## 5. DR64³ recovery result

**The original DR64³ master plan was NOT FOUND.** Search: `git grep` for `DR64³`, `DR256³`, `DR64^3`, `DR256^3`, `64×64×64`, `64x64x64`, `256×256×256`, `256x256x256` over `origin/main` and over all 833 remote branches returned 0 files. A closed-space grep over `.claude/{plans,architecture,board,knowledge,v3}`, `docs`, `.arxiv`, `crates/{dr-math-binding-probe,dir-rmap-probe}` returned 0 hits for `DR64³`, `DR256³`, "3D focus", "Certified Method", "12-stage", "twelve stages" (harvest-dr.md). This is recorded as an **OPEN documentation dependency** (OP-1). This document does not reconstruct the missing stages. Anything below is derived from current code, not from that plan.

What DOES exist (EXISTS, lab only; opus-b Q1):

- **DR64** = a 64×64 Boolean participation register, `[u64; 64]`, 512 B (`relcomp::Rows`, `relcomp.rs:42`). Borrowed as `Operand`, bound as `BoundDr64<'a>` (`crates/dr-math-binding-probe/src/lib.rs:217`, `bind_dr64` at `:231`) with method signature, frame, focus and scope guards. `lib.rs:1-10`: it "carries neither global ids, bag multiplicity nor provenance". `Focus` is only `{ revision: u64 }` (`:167-170`).
- **DR256** = a 256×256 bit register, `[[u64;4];256]`, 8 KiB, in `crates/dir-rmap-probe` (`focus.rs:55-56`, `qr.rs`, `compose.rs`; #1666, #1685). It uses a different witness type from `BoundDr64` (`lib.rs:85-87` of dr-math says so).
- **The R∘S chain.** `relcomp::compose(a, b)` computes `out[i] = OR over j in a[i] of b[j]` (`relcomp.rs:210-222`). `Chain::new` (`:142-173`) admits operands only when each column axis equals the next row axis, all share one current frame issue, one focus, one scope, and all are TwoValued/Distinct. The smallest defensible "DR64³" already exists under another name: the X×Y×Z iteration space of one Boolean composition with Y contracted. It stores no cube and needs no identity format (opus-b verdict 2).

What does NOT exist: a ternary operand type (`MethodSignature` has two axes, `lib.rs:46-53`); a `BoundDr256`; a per-axis focus window; any measurement of a 64³ or 256³ object.

**"64³ Explore → 256³ Validate → Certified Method → 64³ Reuse" is invalid as a single track** (opus-c item 10, opus-b Q4). A finite-population validation never admits a law (`recipe.rs:10-11`; `dr-mathematical-compiler-v1.md:82`). The honest shape is two tracks that meet only at reuse: (A) explore and validate on finite populations (observational), and (B) admit, prove, lower, certify (algebraic). The Validate → Certified arrow does not exist. Reuse (L1, #1740) EXISTS for one law, on same-shape frames.

Constraints that bind any S3 work: 256³ is a paging scheme, not resident (`d-relalg-unified-v1.md:231`); a bit means participation only (`dr-mathematical-compiler-v1.md:11`); CausalEdge64 gains zero bits (`.claude/v3/soa_layout/le-contract.md`); no performance claim before an ns/µs measurement; a ms figure is a failed gate (`CLAUDE.md`). A dense float tensor is not a Boolean participation relation, and a mapping of tensors onto DR cubes is undefined.

## 6. ONNX coverage matrix

Pinned surface (from `Versioning.md` at tag v1.23.2): ai.onnx opset 28, IR 14, ai.onnx.ml 5; opset 22 = ONNX 1.17.0 (IR 10). v1.23.2 has 155 operators + 48 functions in ai.onnx, 4 in `ai.onnx.preview.training`, 19 in `ai.onnx.ml` (2 deprecated). v1.17.0 has 154 operators + 39 functions. `main` is 1.24.0-dev and is not pinned.

Classes, as defined by opus-a. A class describes the FULL ONNX semantics of the operator; the relalg-receivable subset is named in the gap column.

- **Directly representable**: full opset-28 semantics match an existing carrier.
- **Compositionally representable**: expressible from existing carriers plus importer-side shape/dtype work or inlining a `FunctionProto` body.
- **Requires polyfill**: needs a new mathematical primitive. Count: **0**.
- **Requires kernel**: needs (or has, where marked "exists") a physical ndarray kernel; no new algebra.
- **External fallback**: only a foreign runtime would run it. At S1 these are typed refusals; a runtime may appear only as a lab oracle arm (opus-c item 3, item 11).
- **Unsupported/unknown**: refused by name (sequence, map, optional, string, random, training, codec) or no carrier read (`If`, `Loop`, `Scan`).

Totals: directly 1, compositionally 93, requires polyfill 0, requires kernel 91, external fallback 4, unsupported/unknown 37 (34 unsupported + 3 unknown). Sum 226. The "routing target" column is a PROPOSED routing derived from the class and the gap text; the gap column is opus-a's text verbatim (pipes escaped).

Mapping-breaking ONNX semantics that the importer must handle or refuse (opus-a Q2): multidirectional numpy broadcasting (`Operators.md:425`) against same-shape ndarray hpc kernels (`vml.rs:520-521`); the dtype set (float16, bfloat16, float8, float4, float6, int4, uint4, int2, uint2, string, bool at `onnx.proto:609-661`) against mostly-f32 kernels and i32/u32/u64 lanes; axis attributes (ndarray softmax/argmax 1-D/2-D, layer_norm last axis only); IEEE NaN/inf; integer overflow (R2IL raises `Overflow`; the ONNX spec text read does not state a general rule, UNKNOWN); empty reductions; optional inputs; opset drift between 22 and 28 (what changed per re-version is UNKNOWN because `Changelog.md` is not pinned).

**No claim of "ONNX-compatible" is made** until the pinned opset/IR range, per-domain version resolution, per-op dtype coverage, multidirectional broadcasting, shape inference, FunctionProto expansion, and a published pass rate on the official ONNX backend node tests for the claimed subset exist. Until then the only permitted wording is "ONNX intake (subset, opset N–M, ops {…}); all other nodes refused".

| operator | domain | since-versions | class | routing target | gap / notes (opus-a) |
|---|---|---|---|---|---|
| Abs | ai.onnx | 13/6/1 | Requires kernel | ndarray kernel (exists, gaps) | f32 SIMD in vml.rs (vsabs 484, vsexp 410, vsln 436, vssqrt 460, vssin 570, vscos 583, vstanh 617, vsfloor 630, vsceil 642, vsneg 668). Gaps: other dtypes; accuracy vs ONNX reference UNKNOWN (SIMD polynomial) |
| Acos | ai.onnx | 22/7 | Requires kernel | ndarray kernel (MISSING) | no ndarray hpc kernel found; scalar std libm only |
| Acosh | ai.onnx | 22/9 | Requires kernel | ndarray kernel (MISSING) | no ndarray hpc kernel found; scalar std libm only |
| Add | ai.onnx | 14/13/7/6/1 | Requires kernel | ndarray kernel (exists, gaps) | f32 same-shape: `vsadd`/`vsmul`/`vsdiv` (vml.rs:519/536/553); ndarray core ops co-broadcast (impl_ops.rs:141). Gaps: ints (R2IL scalar-only checked, lib.rs:1057-1064; lane-vs-lane refused), f16/bf16/f64, integer overflow |
| And | ai.onnx | 7/1 | Compositionally representable | relalg island (int32/uint32, restricted) else ndarray | same-shape bool: `Rel::And`/`Rel::Or` (ir.rs:97-99). Gap: multidirectional broadcasting (needs an Expand view; ndarray impl_methods.rs:2322 `broadcast`) |
| ArgMax | ai.onnx | 13/12/11/1 | Requires kernel | ndarray kernel (exists, gaps) | `reductions::argmax_f32/argmin_f32` (reductions.rs:326/429) 1-D f32 only. Gaps: axis, keepdims, select_last_index, NaN |
| ArgMin | ai.onnx | 13/12/11/1 | Requires kernel | ndarray kernel (exists, gaps) | `reductions::argmax_f32/argmin_f32` (reductions.rs:326/429) 1-D f32 only. Gaps: axis, keepdims, select_last_index, NaN |
| Asin | ai.onnx | 22/7 | Requires kernel | ndarray kernel (MISSING) | no ndarray hpc kernel found; scalar std libm only |
| Asinh | ai.onnx | 22/9 | Requires kernel | ndarray kernel (MISSING) | no ndarray hpc kernel found; scalar std libm only |
| Atan | ai.onnx | 22/7 | Requires kernel | ndarray kernel (MISSING) | no ndarray hpc kernel found; scalar std libm only |
| Atanh | ai.onnx | 22/9 | Requires kernel | ndarray kernel (MISSING) | no ndarray hpc kernel found; scalar std libm only |
| AveragePool | ai.onnx | 22/19/11/10/7/1 | Requires kernel | ndarray kernel (MISSING) | no pooling kernel found |
| BatchNormalization | ai.onnx | 15/14/9/7/6/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | elementwise affine + mean/var (`reductions::mean_f32`, `statistics`); training-mode outputs of BatchNormalization refused |
| BitCast | ai.onnx | 26 | Unsupported/unknown | typed refusal | bit reinterpretation across dtypes (v26); refuse until dtype table exists |
| BitShift | ai.onnx | 28/11 | Requires kernel | ndarray kernel (exists, gaps) | ndarray core `BitAnd`/`Shl`… ops on integer arrays (impl_ops.rs:430-446). Not mask-risc: MaskOp And/Or/Xor act on bit-planes, not u32 lanes. Gap: ONNX BitShift wrap within type width (Operators.md:7295) vs Rust shift overflow panic |
| BitwiseAnd | ai.onnx | 18 | Requires kernel | ndarray kernel (exists, gaps) | ndarray core `BitAnd`/`Shl`… ops on integer arrays (impl_ops.rs:430-446). Not mask-risc: MaskOp And/Or/Xor act on bit-planes, not u32 lanes. Gap: ONNX BitShift wrap within type width (Operators.md:7295) vs Rust shift overflow panic |
| BitwiseNot | ai.onnx | 18 | Requires kernel | ndarray kernel (exists, gaps) | ndarray core `BitAnd`/`Shl`… ops on integer arrays (impl_ops.rs:430-446). Not mask-risc: MaskOp And/Or/Xor act on bit-planes, not u32 lanes. Gap: ONNX BitShift wrap within type width (Operators.md:7295) vs Rust shift overflow panic |
| BitwiseOr | ai.onnx | 18 | Requires kernel | ndarray kernel (exists, gaps) | ndarray core `BitAnd`/`Shl`… ops on integer arrays (impl_ops.rs:430-446). Not mask-risc: MaskOp And/Or/Xor act on bit-planes, not u32 lanes. Gap: ONNX BitShift wrap within type width (Operators.md:7295) vs Rust shift overflow panic |
| BitwiseXor | ai.onnx | 18 | Requires kernel | ndarray kernel (exists, gaps) | ndarray core `BitAnd`/`Shl`… ops on integer arrays (impl_ops.rs:430-446). Not mask-risc: MaskOp And/Or/Xor act on bit-planes, not u32 lanes. Gap: ONNX BitShift wrap within type width (Operators.md:7295) vs Rust shift overflow panic |
| Cast | ai.onnx | 28/25/24/23/21/19/13/9/6/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | bool→int / int→bool between mask and lane: UNKNOWN primitive (no mask→lane op read); f32↔bf16/f16: ndarray `quantized::f32_to_bf16_slice`/`f16` fns. Gap: float8/float4/int4/int2/string targets, saturate attr, round_mode (v28) |
| Ceil | ai.onnx | 13/6/1 | Requires kernel | ndarray kernel (exists, gaps) | f32 SIMD in vml.rs (vsabs 484, vsexp 410, vsln 436, vssqrt 460, vssin 570, vscos 583, vstanh 617, vsfloor 630, vsceil 642, vsneg 668). Gaps: other dtypes; accuracy vs ONNX reference UNKNOWN (SIMD polynomial) |
| Col2Im | ai.onnx | 18 | Requires kernel | ndarray kernel (MISSING) | no kernel found |
| Compress | ai.onnx | 28/11/9 | Compositionally representable | relalg island (int32/uint32, restricted) else ndarray | axis=None, condition len==N: selection = `Measure::Keep` mask (relalg ir.rs:149). Gap: ONNX returns COMPACTED values (row materialisation); short condition; axis≠None. Fuses when consumed by ReduceSum/Min/Max |
| Concat | ai.onnx | 13/11/4/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ndarray `concatenate` (stacking.rs:36) — copies |
| ConcatFromSequence | ai.onnx | 11 | Unsupported/unknown | typed refusal | sequence type (TypeProto sequence_type, onnx.proto:918); refuse |
| Constant | ai.onnx | 25/24/23/21/19/13/12/11/9/1 | Compositionally representable | relalg island (int32/uint32, restricted) else ndarray | literal: scalar int32 → relalg `Atom` threshold / contract `Measure::Const(i128)` (set_rules.rs:108); tensor → ndarray array |
| ConstantOfShape | ai.onnx | 25/24/23/21/20/9 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | static shapes folded at import from value_info; dynamic shapes refused |
| Conv | ai.onnx | 22/11/1 | Requires kernel | ndarray kernel (exists, gaps) | `linalg::conv::conv2d_f32`/`conv2d_im2col_f32`/`conv1d_f32` (conv.rs:60-368). Gaps: group, dilations, auto_pad, N-D |
| ConvInteger | ai.onnx | 10 | Requires kernel | ndarray kernel (exists, gaps) | int8 gemm exists; requant/zero-point/conv variants missing |
| ConvTranspose | ai.onnx | 22/11/1 | Requires kernel | ndarray kernel (MISSING) | no kernel found |
| Cos | ai.onnx | 22/7 | Requires kernel | ndarray kernel (exists, gaps) | f32 SIMD in vml.rs (vsabs 484, vsexp 410, vsln 436, vssqrt 460, vssin 570, vscos 583, vstanh 617, vsfloor 630, vsceil 642, vsneg 668). Gaps: other dtypes; accuracy vs ONNX reference UNKNOWN (SIMD polynomial) |
| Cosh | ai.onnx | 22/9 | Requires kernel | ndarray kernel (MISSING) | no ndarray hpc kernel found; scalar std libm only |
| CumProd | ai.onnx | 26 | Requires kernel | ndarray kernel (MISSING) | no kernel found (prefix scan); not a mask-risc terminal |
| CumSum | ai.onnx | 14/11 | Requires kernel | ndarray kernel (MISSING) | no kernel found (prefix scan); not a mask-risc terminal |
| DFT | ai.onnx | 20/17 | Requires kernel | ndarray kernel (exists, gaps) | `fft::fft_f32/rfft_f32` (fft.rs:23-122) radix-2 only; gap non-power-of-2, onesided/inverse attrs |
| DeformConv | ai.onnx | 22/19 | Requires kernel | ndarray kernel (MISSING) | no kernel found |
| DequantizeLinear | ai.onnx | 28/25/24/23/21/19/13/10 | Requires kernel | ndarray kernel (exists, gaps) | `quantize_f32_to_i8/u8`, `dequantize_i8_to_f32` (quantized.rs:517-586). Gaps: per-axis/block_size, saturate, float8/int4 outputs, rounding mode |
| Det | ai.onnx | 22/11 | Requires kernel | ndarray kernel (exists, gaps) | `LapackOps::lu` (lapack.rs:53) → product of diag; `linalg/matrix.rs:233` det fixed-size |
| Div | ai.onnx | 14/13/7/6/1 | Requires kernel | ndarray kernel (exists, gaps) | f32 same-shape: `vsadd`/`vsmul`/`vsdiv` (vml.rs:519/536/553); ndarray core ops co-broadcast (impl_ops.rs:141). Gaps: ints (R2IL scalar-only checked, lib.rs:1057-1064; lane-vs-lane refused), f16/bf16/f64, integer overflow |
| Dropout | ai.onnx | 22/13/12/10/7/6/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | inference (training_mode=false) = Identity; training mode refused (impure) |
| Einsum | ai.onnx | 28/12 | Requires kernel | ndarray kernel (MISSING) | no kernel; decomposable to Transpose+MatMul+ReduceSum by an importer (UNKNOWN effort) |
| Equal | ai.onnx | 19/13/11/7/1 | Compositionally representable | relalg island (int32/uint32, restricted) else ndarray | int32[N] lane vs int32 scalar initializer: `Atom::I32{op:Eq}`; uint32: `Atom::U32{eq:true}` (ir.rs:74-79). Gaps: tensor-vs-tensor (no lane-vs-lane Pred, mask-risc ir.rs:127-200), float/NaN, int64/int8/16, string, bool-vs-bool |
| Erf | ai.onnx | 13/9 | Requires kernel | ndarray kernel (exists, gaps) | `erf_f32` exists but private (linalg/activations_ext.rs:188); needs `pub` exposure |
| Exp | ai.onnx | 13/6/1 | Requires kernel | ndarray kernel (exists, gaps) | f32 SIMD in vml.rs (vsabs 484, vsexp 410, vsln 436, vssqrt 460, vssin 570, vscos 583, vstanh 617, vsfloor 630, vsceil 642, vsneg 668). Gaps: other dtypes; accuracy vs ONNX reference UNKNOWN (SIMD polynomial) |
| Expand | ai.onnx | 13/8 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ndarray `broadcast` view (impl_methods.rs:2322) |
| EyeLike | ai.onnx | 22/9 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | index-generated tensors; ndarray core construction (Range is a function, v27) |
| Flatten | ai.onnx | 25/24/23/21/13/11/9/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ndarray `into_shape_with_order` (impl_methods.rs:1975), `insert_axis`/`remove_axis` (2700/2717) zero-copy on contiguous data. Gap: Reshape allowzero, -1/0 dims resolved at import |
| Floor | ai.onnx | 13/6/1 | Requires kernel | ndarray kernel (exists, gaps) | f32 SIMD in vml.rs (vsabs 484, vsexp 410, vsln 436, vssqrt 460, vssin 570, vscos 583, vstanh 617, vsfloor 630, vsceil 642, vsneg 668). Gaps: other dtypes; accuracy vs ONNX reference UNKNOWN (SIMD polynomial) |
| GRU | ai.onnx | 22/14/7/3/1 | External fallback | typed refusal at S1; oracle arm only | recurrent cells; composable from Gemm+Sigmoid+Tanh per step but need Scan/Loop; candle is optional only in thinking-engine (Cargo.toml:28-37) |
| Gather | ai.onnx | 13/11/1 | Compositionally representable | typed refusal on substrate path; ndarray-internal only | ndarray `select(axis, indices)` (impl_methods.rs:1069). Gap: negative indices, out-of-range = ONNX error vs panic |
| GatherElements | ai.onnx | 13/11 | Requires kernel | typed refusal on substrate path; ndarray-internal only | no kernel; ScatterElements reduction=add resembles `Terminal::GroupSumI32` (ir.rs:399) but that drops out-of-range keys silently (ir.rs:333) where ONNX errors |
| GatherND | ai.onnx | 13/12/11 | Requires kernel | typed refusal on substrate path; ndarray-internal only | no kernel; ScatterElements reduction=add resembles `Terminal::GroupSumI32` (ir.rs:399) but that drops out-of-range keys silently (ir.rs:333) where ONNX errors |
| Gemm | ai.onnx | 13/11/9/7/6/1 | Requires kernel | ndarray kernel (exists, gaps) | `blas_gemm_into` alpha/beta (blas_level3.rs:39); transA/transB via `permuted_axes` view (impl_methods.rs:2471); C broadcasting gap |
| GlobalAveragePool | ai.onnx | 22/1 | Requires kernel | ndarray kernel (MISSING) | no pooling kernel found |
| GlobalLpPool | ai.onnx | 22/2/1 | Requires kernel | ndarray kernel (MISSING) | no pooling kernel found |
| GlobalMaxPool | ai.onnx | 22/1 | Requires kernel | ndarray kernel (MISSING) | no pooling kernel found |
| Greater | ai.onnx | 13/9/7/1 | Compositionally representable | relalg island (int32/uint32, restricted) else ndarray | int32[N] lane vs int32 scalar: `Atom::I32{op:Lt\|Gt}`. Gaps: uint ordered compare refused (`NoUnsignedLaneCompare`, r2il lib.rs:213-216), tensor-vs-tensor, float/NaN, int64 |
| GridSample | ai.onnx | 22/20/16 | Requires kernel | ndarray kernel (MISSING) | no kernel found (AffineGrid is a function) |
| Hardmax | ai.onnx | 13/11/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ArgMax + OneHot |
| Identity | ai.onnx | 25/24/23/21/19/16/14/13/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | value alias resolved at import (no runtime primitive); sequence/optional inputs refused |
| If | ai.onnx | 25/24/23/21/19/16/13/11/1 | Unsupported/unknown | typed refusal | subgraph attribute; R2IL branches only on scalars (`BranchOnPopulation` refused, r2il lib.rs:229-233); ogar_loco control flow not read |
| ImageDecoder | ai.onnx | 20 | Unsupported/unknown | typed refusal | codec, not math; refuse |
| InstanceNormalization | ai.onnx | 22/6/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | elementwise affine + mean/var (`reductions::mean_f32`, `statistics`); training-mode outputs of BatchNormalization refused |
| IsInf | ai.onnx | 20/10 | Requires kernel | ndarray kernel (MISSING) | no hpc kernel found; trivial elementwise |
| IsNaN | ai.onnx | 20/13/9 | Requires kernel | ndarray kernel (MISSING) | no hpc kernel found; trivial elementwise |
| LRN | ai.onnx | 13/1 | Requires kernel | ndarray kernel (MISSING) | no pooling kernel found |
| LSTM | ai.onnx | 22/14/7/1 | External fallback | typed refusal at S1; oracle arm only | recurrent cells; composable from Gemm+Sigmoid+Tanh per step but need Scan/Loop; candle is optional only in thinking-engine (Cargo.toml:28-37) |
| Less | ai.onnx | 13/9/7/1 | Compositionally representable | relalg island (int32/uint32, restricted) else ndarray | int32[N] lane vs int32 scalar: `Atom::I32{op:Lt\|Gt}`. Gaps: uint ordered compare refused (`NoUnsignedLaneCompare`, r2il lib.rs:213-216), tensor-vs-tensor, float/NaN, int64 |
| Log | ai.onnx | 13/6/1 | Requires kernel | ndarray kernel (exists, gaps) | f32 SIMD in vml.rs (vsabs 484, vsexp 410, vsln 436, vssqrt 460, vssin 570, vscos 583, vstanh 617, vsfloor 630, vsceil 642, vsneg 668). Gaps: other dtypes; accuracy vs ONNX reference UNKNOWN (SIMD polynomial) |
| Loop | ai.onnx | 25/24/23/21/19/16/13/11/1 | Unsupported/unknown | typed refusal | subgraph iteration; no tensor-level loop target read |
| LpNormalization | ai.onnx | 22/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | elementwise affine + mean/var (`reductions::mean_f32`, `statistics`); training-mode outputs of BatchNormalization refused |
| LpPool | ai.onnx | 22/18/11/2/1 | Requires kernel | ndarray kernel (MISSING) | no pooling kernel found |
| MatMul | ai.onnx | 13/9/1 | Requires kernel | ndarray kernel (exists, gaps) | `BlasLevel3::blas_gemm` (blas_level3.rs:36) 2-D; `linalg::batched::batched_gemm_4d_f32`; `amx_matmul::matmul_f32`. Gaps: numpy batch broadcasting, 1-D promote rules, int32/uint64 types |
| MatMulInteger | ai.onnx | 10 | Requires kernel | ndarray kernel (exists, gaps) | `quantized::int8_gemm_i32` (quantized.rs:618), `amx_matmul::matmul_i8_to_i32`. Gap: zero-point inputs, uint8 |
| Max | ai.onnx | 13/12/8/6/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | variadic elementwise: fold of ndarray core / vml binary kernels. Gap: NaN propagation rule UNKNOWN in kernels |
| MaxPool | ai.onnx | 22/12/11/10/8/1 | Requires kernel | ndarray kernel (MISSING) | no pooling kernel found |
| MaxRoiPool | ai.onnx | 22/1 | Requires kernel | ndarray kernel (MISSING) | no pooling kernel found |
| MaxUnpool | ai.onnx | 22/11/9 | Requires kernel | ndarray kernel (MISSING) | no pooling kernel found |
| Mean | ai.onnx | 13/8/6/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | variadic elementwise: fold of ndarray core / vml binary kernels. Gap: NaN propagation rule UNKNOWN in kernels |
| MelWeightMatrix | ai.onnx | 17 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | generator bodies (Range/Cos/…); MelWeightMatrix is an op, index arithmetic |
| Min | ai.onnx | 13/12/8/6/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | variadic elementwise: fold of ndarray core / vml binary kernels. Gap: NaN propagation rule UNKNOWN in kernels |
| Mod | ai.onnx | 28/13/10 | Requires kernel | ndarray kernel (MISSING) | no kernel; fmod 0/1 semantics with IEEE special cases (Operators.md Mod) |
| Mul | ai.onnx | 14/13/7/6/1 | Requires kernel | ndarray kernel (exists, gaps) | f32 same-shape: `vsadd`/`vsmul`/`vsdiv` (vml.rs:519/536/553); ndarray core ops co-broadcast (impl_ops.rs:141). Gaps: ints (R2IL scalar-only checked, lib.rs:1057-1064; lane-vs-lane refused), f16/bf16/f64, integer overflow |
| Multinomial | ai.onnx | 22/7 | Unsupported/unknown | typed refusal | impure/random; refuse |
| Neg | ai.onnx | 13/6/1 | Requires kernel | ndarray kernel (exists, gaps) | f32 SIMD in vml.rs (vsabs 484, vsexp 410, vsln 436, vssqrt 460, vssin 570, vscos 583, vstanh 617, vsfloor 630, vsceil 642, vsneg 668). Gaps: other dtypes; accuracy vs ONNX reference UNKNOWN (SIMD polynomial) |
| NonMaxSuppression | ai.onnx | 11/10 | External fallback | typed refusal at S1; oracle arm only | no kernel; detection post-processing, candidate for backend fallback |
| NonZero | ai.onnx | 13/9 | Compositionally representable | relalg island (int32/uint32, restricted) else ndarray | bool[N]: `Keep` mask is the same set. Gap: ONNX returns int64 index tensor [rank,nnz] (materialisation) |
| Not | ai.onnx | 1 | Directly representable | relalg island (int32/uint32, restricted) else ndarray | bool tensor (any shape, flattened row-major) = one mask plane; `Rel::Not` (relalg ir.rs:101) / `MaskOp::Not` (mask-risc ir.rs:239). Not's only type is bool, elementwise, no broadcast, so no gap. EXISTS |
| OneHot | ai.onnx | 28/11/9 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | index-generated tensors; ndarray core construction (Range is a function, v27) |
| Optional | ai.onnx | 28/15 | Unsupported/unknown | typed refusal | optional type (onnx.proto:924); refuse |
| OptionalGetElement | ai.onnx | 28/18/15 | Unsupported/unknown | typed refusal | optional type (onnx.proto:924); refuse |
| OptionalHasElement | ai.onnx | 28/18/15 | Unsupported/unknown | typed refusal | optional type (onnx.proto:924); refuse |
| Or | ai.onnx | 7/1 | Compositionally representable | relalg island (int32/uint32, restricted) else ndarray | same-shape bool: `Rel::And`/`Rel::Or` (ir.rs:97-99). Gap: multidirectional broadcasting (needs an Expand view; ndarray impl_methods.rs:2322 `broadcast`) |
| Pad | ai.onnx | 25/24/23/21/19/18/13/11/2/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | view/copy compositions (DepthToSpace/SpaceToDepth/CenterCropPad are functions); Pad reflect/edge/wrap modes need code |
| Pow | ai.onnx | 15/13/12/7/1 | Requires kernel | ndarray kernel (exists, gaps) | `vspow` (vml.rs:600) f32 same-shape. Gap: broadcasting, int exponent/base type mixing |
| QLinearConv | ai.onnx | 10 | Requires kernel | ndarray kernel (exists, gaps) | int8 gemm exists; requant/zero-point/conv variants missing |
| QLinearMatMul | ai.onnx | 21/10 | Requires kernel | ndarray kernel (exists, gaps) | int8 gemm exists; requant/zero-point/conv variants missing |
| QuantizeLinear | ai.onnx | 28/25/24/23/21/19/13/10 | Requires kernel | ndarray kernel (exists, gaps) | `quantize_f32_to_i8/u8`, `dequantize_i8_to_f32` (quantized.rs:517-586). Gaps: per-axis/block_size, saturate, float8/int4 outputs, rounding mode |
| RNN | ai.onnx | 22/14/7/1 | External fallback | typed refusal at S1; oracle arm only | recurrent cells; composable from Gemm+Sigmoid+Tanh per step but need Scan/Loop; candle is optional only in thinking-engine (Cargo.toml:28-37) |
| RandomNormal | ai.onnx | 22/1 | Unsupported/unknown | typed refusal | impure/random; refuse |
| RandomNormalLike | ai.onnx | 22/1 | Unsupported/unknown | typed refusal | impure/random; refuse |
| RandomUniform | ai.onnx | 22/1 | Unsupported/unknown | typed refusal | impure/random; refuse |
| RandomUniformLike | ai.onnx | 22/1 | Unsupported/unknown | typed refusal | impure/random; refuse |
| Reciprocal | ai.onnx | 13/6/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | Div(1,x) → `vsdiv` |
| ReduceMax | ai.onnx | 20/18/13/12/11/1 | Compositionally representable | relalg island (int32/uint32, restricted) else ndarray | int32 all axes: `Measure::Min/Max` → `MaskedMin/MaxI32`. Gap: empty → relalg `None` (lower.rs:126-137) vs ONNX dtype max/min (Operators.md ReduceMin/ReduceMax); axes; floats → `min_f32/max_f32` |
| ReduceMean | ai.onnx | 18/13/11/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | int: r2il `avg_parts` (procs.rs:423) = (sum,count); float: `reductions::mean_f32`. Gap: ONNX integer mean rounding UNKNOWN; axes |
| ReduceMin | ai.onnx | 20/18/13/12/11/1 | Compositionally representable | relalg island (int32/uint32, restricted) else ndarray | int32 all axes: `Measure::Min/Max` → `MaskedMin/MaxI32`. Gap: empty → relalg `None` (lower.rs:126-137) vs ONNX dtype max/min (Operators.md ReduceMin/ReduceMax); axes; floats → `min_f32/max_f32` |
| ReduceProd | ai.onnx | 18/13/11/1 | Requires kernel | ndarray kernel (MISSING) | no product reduction found |
| ReduceSum | ai.onnx | 13/11/1 | Compositionally representable | relalg island (int32/uint32, restricted) else ndarray | int32, all axes, over a mask: `Measure::Sum` → `MaskedSumI32` (ir.rs:299, widened i64, lower.rs:121-125). Gaps: ONNX keeps int32 output dtype; axes subset needs GroupSumI32 with a key lane; floats → ndarray `reductions::sum_f32` (KERNEL) |
| RegexFullMatch | ai.onnx | 20 | Unsupported/unknown | typed refusal | string tensors; lance-graph IR holds no strings (relalg ir.rs:41-53) |
| Reshape | ai.onnx | 25/24/23/21/19/14/13/5/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ndarray `into_shape_with_order` (impl_methods.rs:1975), `insert_axis`/`remove_axis` (2700/2717) zero-copy on contiguous data. Gap: Reshape allowzero, -1/0 dims resolved at import |
| Resize | ai.onnx | 19/18/13/11/10 | Requires kernel | ndarray kernel (MISSING) | no kernel found (AffineGrid is a function) |
| ReverseSequence | ai.onnx | 28/10 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | view/copy compositions (DepthToSpace/SpaceToDepth/CenterCropPad are functions); Pad reflect/edge/wrap modes need code |
| RoiAlign | ai.onnx | 22/16/10 | Requires kernel | ndarray kernel (MISSING) | no kernel found (AffineGrid is a function) |
| Round | ai.onnx | 22/11 | Requires kernel | ndarray kernel (exists, gaps) | `vsround` ties-to-even (vml.rs:650-661) matches ONNX half-to-even; f32 only |
| STFT | ai.onnx | 17 | Requires kernel | ndarray kernel (exists, gaps) | `fft::fft_f32/rfft_f32` (fft.rs:23-122) radix-2 only; gap non-power-of-2, onesided/inverse attrs |
| Scan | ai.onnx | 25/24/23/21/19/16/11/9/8 | Unsupported/unknown | typed refusal | subgraph iteration; no tensor-level loop target read |
| Scatter | ai.onnx | 11/9 | Requires kernel | typed refusal on substrate path; ndarray-internal only | no kernel; ScatterElements reduction=add resembles `Terminal::GroupSumI32` (ir.rs:399) but that drops out-of-range keys silently (ir.rs:333) where ONNX errors |
| ScatterElements | ai.onnx | 18/16/13/11 | Requires kernel | typed refusal on substrate path; ndarray-internal only | no kernel; ScatterElements reduction=add resembles `Terminal::GroupSumI32` (ir.rs:399) but that drops out-of-range keys silently (ir.rs:333) where ONNX errors |
| ScatterND | ai.onnx | 18/16/13/11 | Requires kernel | typed refusal on substrate path; ndarray-internal only | no kernel; ScatterElements reduction=add resembles `Terminal::GroupSumI32` (ir.rs:399) but that drops out-of-range keys silently (ir.rs:333) where ONNX errors |
| SequenceAt | ai.onnx | 11 | Unsupported/unknown | typed refusal | sequence type (TypeProto sequence_type, onnx.proto:918); refuse |
| SequenceConstruct | ai.onnx | 11 | Unsupported/unknown | typed refusal | sequence type (TypeProto sequence_type, onnx.proto:918); refuse |
| SequenceEmpty | ai.onnx | 11 | Unsupported/unknown | typed refusal | sequence type (TypeProto sequence_type, onnx.proto:918); refuse |
| SequenceErase | ai.onnx | 11 | Unsupported/unknown | typed refusal | sequence type (TypeProto sequence_type, onnx.proto:918); refuse |
| SequenceInsert | ai.onnx | 11 | Unsupported/unknown | typed refusal | sequence type (TypeProto sequence_type, onnx.proto:918); refuse |
| SequenceLength | ai.onnx | 11 | Unsupported/unknown | typed refusal | sequence type (TypeProto sequence_type, onnx.proto:918); refuse |
| Shape | ai.onnx | 25/24/23/21/19/15/13/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | static shapes folded at import from value_info; dynamic shapes refused |
| Sigmoid | ai.onnx | 13/6/1 | Requires kernel | ndarray kernel (exists, gaps) | `activations::sigmoid_f32` (activations.rs:94) |
| Sign | ai.onnx | 13/9 | Requires kernel | ndarray kernel (MISSING) | no hpc kernel found; trivial elementwise |
| Sin | ai.onnx | 22/7 | Requires kernel | ndarray kernel (exists, gaps) | f32 SIMD in vml.rs (vsabs 484, vsexp 410, vsln 436, vssqrt 460, vssin 570, vscos 583, vstanh 617, vsfloor 630, vsceil 642, vsneg 668). Gaps: other dtypes; accuracy vs ONNX reference UNKNOWN (SIMD polynomial) |
| Sinh | ai.onnx | 22/9 | Requires kernel | ndarray kernel (MISSING) | no ndarray hpc kernel found; scalar std libm only |
| Size | ai.onnx | 25/24/23/21/19/13/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | static shapes folded at import from value_info; dynamic shapes refused |
| Slice | ai.onnx | 13/11/10/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ndarray `slice_each_axis` (impl_methods.rs:699); negative steps via `invert_axis` (2620). Gap: clamping of out-of-range starts/ends |
| Split | ai.onnx | 18/13/11/2/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | axis slicing views |
| SplitToSequence | ai.onnx | 24/11 | Unsupported/unknown | typed refusal | sequence type (TypeProto sequence_type, onnx.proto:918); refuse |
| Sqrt | ai.onnx | 13/6/1 | Requires kernel | ndarray kernel (exists, gaps) | f32 SIMD in vml.rs (vsabs 484, vsexp 410, vsln 436, vssqrt 460, vssin 570, vscos 583, vstanh 617, vsfloor 630, vsceil 642, vsneg 668). Gaps: other dtypes; accuracy vs ONNX reference UNKNOWN (SIMD polynomial) |
| Squeeze | ai.onnx | 25/24/23/21/13/11/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ndarray `into_shape_with_order` (impl_methods.rs:1975), `insert_axis`/`remove_axis` (2700/2717) zero-copy on contiguous data. Gap: Reshape allowzero, -1/0 dims resolved at import |
| StringConcat | ai.onnx | 20 | Unsupported/unknown | typed refusal | string tensors; lance-graph IR holds no strings (relalg ir.rs:41-53) |
| StringNormalizer | ai.onnx | 10 | Unsupported/unknown | typed refusal | string tensors; lance-graph IR holds no strings (relalg ir.rs:41-53) |
| StringSplit | ai.onnx | 20 | Unsupported/unknown | typed refusal | string tensors; lance-graph IR holds no strings (relalg ir.rs:41-53) |
| Sub | ai.onnx | 14/13/7/6/1 | Requires kernel | ndarray kernel (MISSING) | no `vssub`; f32 via `vsadd`+`vsneg` or ndarray core `-` (co-broadcast). Same int gaps as Add |
| Sum | ai.onnx | 13/8/6/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | variadic elementwise: fold of ndarray core / vml binary kernels. Gap: NaN propagation rule UNKNOWN in kernels |
| Tan | ai.onnx | 22/7 | Requires kernel | ndarray kernel (MISSING) | no ndarray hpc kernel found; scalar std libm only |
| Tanh | ai.onnx | 13/6/1 | Requires kernel | ndarray kernel (exists, gaps) | f32 SIMD in vml.rs (vsabs 484, vsexp 410, vsln 436, vssqrt 460, vssin 570, vscos 583, vstanh 617, vsfloor 630, vsceil 642, vsneg 668). Gaps: other dtypes; accuracy vs ONNX reference UNKNOWN (SIMD polynomial) |
| TensorScatter | ai.onnx | 24 | Requires kernel | typed refusal on substrate path; ndarray-internal only | no kernel; ScatterElements reduction=add resembles `Terminal::GroupSumI32` (ir.rs:399) but that drops out-of-range keys silently (ir.rs:333) where ONNX errors |
| TfIdfVectorizer | ai.onnx | 9 | Unsupported/unknown | typed refusal | string tensors; lance-graph IR holds no strings (relalg ir.rs:41-53) |
| Tile | ai.onnx | 13/6/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | view/copy compositions (DepthToSpace/SpaceToDepth/CenterCropPad are functions); Pad reflect/edge/wrap modes need code |
| TopK | ai.onnx | 24/11/10/1 | Requires kernel | ndarray kernel (MISSING) | no matching kernel (`prefilter::top_k_rows_by_norm` is a different op) |
| Transpose | ai.onnx | 25/24/23/21/13/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ndarray `permuted_axes` view (impl_methods.rs:2471) |
| Trilu | ai.onnx | 14 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | index-generated tensors; ndarray core construction (Range is a function, v27) |
| Unique | ai.onnx | 28/11 | Requires kernel | ndarray kernel (MISSING) | no kernel; r2il `count_distinct_sorted` (procs.rs:549) counts only, needs sorted input |
| Unsqueeze | ai.onnx | 25/24/23/21/13/11/1 | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ndarray `into_shape_with_order` (impl_methods.rs:1975), `insert_axis`/`remove_axis` (2700/2717) zero-copy on contiguous data. Gap: Reshape allowzero, -1/0 dims resolved at import |
| Upsample | ai.onnx | 10/9/7 | Requires kernel | ndarray kernel (MISSING) | no kernel found (AffineGrid is a function) |
| Where | ai.onnx | 16/9 | Compositionally representable | mask-risc BlendI32 (int32) else ndarray | int32 same-shape: mask-risc `Terminal::BlendI32` (ir.rs:318) into a caller buffer; float: ndarray core `Zip` (no hpc kernel). Gap: broadcasting, non-int32 dtypes |
| Xor | ai.onnx | 7/1 | Compositionally representable | relalg island (int32/uint32, restricted) else ndarray | same-shape bool: `MaskOp::Xor` (ir.rs:235) / R2IL `INT_XOR` (r2il lib.rs:139); in relalg only as (a∧¬b)∨(¬a∧b) (no `Rel::Xor`). Gap: broadcasting |
| AffineGrid | ai.onnx | 20 (fn body 20) | Requires kernel | ndarray kernel (MISSING) | no kernel found (AffineGrid is a function) |
| Attention | ai.onnx | 25/24/23 (fn body 25) | Requires kernel | ndarray kernel (exists, gaps) | `linalg::attention::attention_f32`/`flash_attention_f32` (attention.rs:100/218). Gaps: mask kinds, GQA, softcap, past KV, qk_matmul_output_mode |
| Bernoulli | ai.onnx | 22/15 (fn body 22) | Unsupported/unknown | typed refusal | impure (random); contract has only `Measure::Opaque{pure:false}` as a placeholder (set_rules.rs:119-123), no sampler |
| BlackmanWindow | ai.onnx | 17 (fn body 17) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | generator bodies (Range/Cos/…); MelWeightMatrix is an op, index arithmetic |
| CastLike | ai.onnx | 25/24/23/21/19/15 (fn body 25) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | function = Cast to other input's dtype |
| CausalConvWithState | ai.onnx | 27 (fn body 27) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | function bodies (v27); kernels for state update UNKNOWN |
| Celu | ai.onnx | 28/12 (fn body 28) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ONNX function: inline body (Max/Where/Mul/Exp…) over f32 kernels |
| CenterCropPad | ai.onnx | 18 (fn body 18) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | view/copy compositions (DepthToSpace/SpaceToDepth/CenterCropPad are functions); Pad reflect/edge/wrap modes need code |
| Clip | ai.onnx | 13/12/11/6/1 (fn body 13) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | function: Min(Max(x,lo),hi) |
| DepthToSpace | ai.onnx | 28/13/11/1 (fn body 28) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | view/copy compositions (DepthToSpace/SpaceToDepth/CenterCropPad are functions); Pad reflect/edge/wrap modes need code |
| DynamicQuantizeLinear | ai.onnx | 11 (fn body 11) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | function: ReduceMin/Max + QuantizeLinear |
| Elu | ai.onnx | 22/6/1 (fn body 18) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ONNX function: inline body (Max/Where/Mul/Exp…) over f32 kernels |
| Gelu | ai.onnx | 20 (fn body 20) | Requires kernel | ndarray kernel (exists, gaps) | `gelu_f32` erf (activations_ext.rs:67) / `gelu_tanh_f32` (93) ↔ attr approximate none\|tanh |
| GreaterOrEqual | ai.onnx | 16/12 (fn body 16) | Compositionally representable | relalg island (int32/uint32, restricted) else ndarray | function body Less∨Equal / Greater∨Equal; int32-vs-scalar subset = `Atom::I32{op:Le\|Ge}` directly. Same gaps as Less |
| GroupNormalization | ai.onnx | 21/18 (fn body 21) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | `group_norm_f32` (norm.rs:143) takes per-ELEMENT gamma (len d) vs ONNX per-channel scale → broadcast at import |
| HammingWindow | ai.onnx | 17 (fn body 17) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | generator bodies (Range/Cos/…); MelWeightMatrix is an op, index arithmetic |
| HannWindow | ai.onnx | 17 (fn body 17) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | generator bodies (Range/Cos/…); MelWeightMatrix is an op, index arithmetic |
| HardSigmoid | ai.onnx | 22/6/1 (fn body 18) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ONNX function: inline body (Max/Where/Mul/Exp…) over f32 kernels |
| HardSwish | ai.onnx | 22/14 (fn body 22) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ONNX function: inline body (Max/Where/Mul/Exp…) over f32 kernels |
| LayerNormalization | ai.onnx | 17 (fn body 17/18) | Requires kernel | ndarray kernel (exists, gaps) | `linalg::norm::layer_norm_f32` (norm.rs:54) last axis, 1-D gamma/beta. Gaps: axis≠-1, stash_type, Mean/InvStdDev outputs |
| LeakyRelu | ai.onnx | 16/6/1 (fn body 16) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ONNX function: inline body (Max/Where/Mul/Exp…) over f32 kernels |
| LessOrEqual | ai.onnx | 16/12 (fn body 16) | Compositionally representable | relalg island (int32/uint32, restricted) else ndarray | function body Less∨Equal / Greater∨Equal; int32-vs-scalar subset = `Atom::I32{op:Le\|Ge}` directly. Same gaps as Less |
| LinearAttention | ai.onnx | 27 (fn body 27) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | function bodies (v27); kernels for state update UNKNOWN |
| LogSoftmax | ai.onnx | 13/11/1 (fn body 13/18) | Requires kernel | ndarray kernel (exists, gaps) | `softmax_axis_f32`/`log_softmax_axis_f32` (activations.rs:390/442) 2-D; N-D axis via reshape |
| MeanVarianceNormalization | ai.onnx | 13/9 (fn body 13/18) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | function body (ReduceMean/Sub/Sqrt/Div) |
| Mish | ai.onnx | 22/18 (fn body 22) | Requires kernel | ndarray kernel (exists, gaps) | `linalg::activations_ext::mish_f32` (activations_ext.rs:168) — corrects fact base F6 "NO mish" |
| NegativeLogLikelihoodLoss | ai.onnx | 22/13/12 (fn body 22) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | function bodies; `linalg::loss::cross_entropy_with_logits_f32` (loss.rs:100) partial |
| PRelu | ai.onnx | 16/9/7/6/1 (fn body 16) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ONNX function: inline body (Max/Where/Mul/Exp…) over f32 kernels |
| RMSNormalization | ai.onnx | 23 (fn body 23) | Requires kernel | ndarray kernel (exists, gaps) | `linalg::norm::rms_norm_f32` (norm.rs:100). Gap: axis, stash_type |
| Range | ai.onnx | 27/11 (fn body 27) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | index-generated tensors; ndarray core construction (Range is a function, v27) |
| ReduceL1 | ai.onnx | 18/13/11/1 (fn body 18) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | function bodies over ReduceSum/Abs/Mul/Log/Exp; `reductions::nrm2_f32` (reductions.rs:517) for L2 |
| ReduceL2 | ai.onnx | 18/13/11/1 (fn body 18) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | function bodies over ReduceSum/Abs/Mul/Log/Exp; `reductions::nrm2_f32` (reductions.rs:517) for L2 |
| ReduceLogSum | ai.onnx | 28/18/13/11/1 (fn body 28) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | function bodies over ReduceSum/Abs/Mul/Log/Exp; `reductions::nrm2_f32` (reductions.rs:517) for L2 |
| ReduceLogSumExp | ai.onnx | 28/18/13/11/1 (fn body 28) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | function bodies over ReduceSum/Abs/Mul/Log/Exp; `reductions::nrm2_f32` (reductions.rs:517) for L2 |
| ReduceSumSquare | ai.onnx | 18/13/11/1 (fn body 18) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | function bodies over ReduceSum/Abs/Mul/Log/Exp; `reductions::nrm2_f32` (reductions.rs:517) for L2 |
| Relu | ai.onnx | 14/13/6/1 (fn body 18) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ONNX function: inline body (Max/Where/Mul/Exp…) over f32 kernels |
| RotaryEmbedding | ai.onnx | 23 (fn body 23) | Requires kernel | ndarray kernel (exists, gaps) | `linalg/rope.rs` `apply_qk_f32` (rope.rs:130). Gap: interleaved, position_ids, rotary_embedding_dim |
| Selu | ai.onnx | 22/6/1 (fn body 18) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ONNX function: inline body (Max/Where/Mul/Exp…) over f32 kernels |
| SequenceMap | ai.onnx | 17 (fn body 17) | Unsupported/unknown | typed refusal | sequence type (TypeProto sequence_type, onnx.proto:918); refuse |
| Shrink | ai.onnx | 9 (fn body 18) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ONNX function: inline body (Max/Where/Mul/Exp…) over f32 kernels |
| Softmax | ai.onnx | 13/11/1 (fn body 13/18) | Requires kernel | ndarray kernel (exists, gaps) | `softmax_axis_f32`/`log_softmax_axis_f32` (activations.rs:390/442) 2-D; N-D axis via reshape |
| SoftmaxCrossEntropyLoss | ai.onnx | 13/12 (fn body 13) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | function bodies; `linalg::loss::cross_entropy_with_logits_f32` (loss.rs:100) partial |
| Softplus | ai.onnx | 22/1 (fn body 18) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | function: Log(Exp(x)+1); no named kernel (mish_f32 has a private softplus step, activations_ext.rs:168-175) |
| Softsign | ai.onnx | 22/1 (fn body 18) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ONNX function: inline body (Max/Where/Mul/Exp…) over f32 kernels |
| SpaceToDepth | ai.onnx | 28/13/1 (fn body 28) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | view/copy compositions (DepthToSpace/SpaceToDepth/CenterCropPad are functions); Pad reflect/edge/wrap modes need code |
| SwiGLU | ai.onnx | 28 (fn body 28) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | function: silu (`silu_f32`, activations_ext.rs:117) × gate |
| Swish | ai.onnx | 24 (fn body 24) | Requires kernel | ndarray kernel (exists, gaps) | `swish_f32(x, beta)` (activations_ext.rs:143); ONNX attr alpha |
| ThresholdedRelu | ai.onnx | 22/10 (fn body 18) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | ONNX function: inline body (Max/Where/Mul/Exp…) over f32 kernels |
| ai.onnx.preview.training.Adagrad | ai.onnx.preview.training | 1 | Unsupported/unknown | typed refusal | training domain; refuse (TrainingInfoProto also refused) |
| ai.onnx.preview.training.Adam | ai.onnx.preview.training | 1 | Unsupported/unknown | typed refusal | training domain; refuse (TrainingInfoProto also refused) |
| ai.onnx.preview.training.Gradient | ai.onnx.preview.training | 1 | Unsupported/unknown | typed refusal | training domain; refuse (TrainingInfoProto also refused) |
| ai.onnx.preview.training.Momentum | ai.onnx.preview.training | 1 | Unsupported/unknown | typed refusal | training domain; refuse (TrainingInfoProto also refused) |
| ai.onnx.ml.ArrayFeatureExtractor | ai.onnx.ml | (see Operators-ml.md) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | Gather on last axis |
| ai.onnx.ml.Binarizer | ai.onnx.ml | (see Operators-ml.md) | Requires kernel | ndarray kernel (exists, gaps) | x>threshold → float 0/1 (float compare; not a mask-risc lane type) |
| ai.onnx.ml.CastMap | ai.onnx.ml | (see Operators-ml.md) | Unsupported/unknown | typed refusal | map/sequence types (onnx.proto:918-921) |
| ai.onnx.ml.CategoryMapper | ai.onnx.ml | (see Operators-ml.md) | Unsupported/unknown | typed refusal | string↔int maps; int→int LabelEncoder would be a lookup KERNEL |
| ai.onnx.ml.DictVectorizer | ai.onnx.ml | (see Operators-ml.md) | Unsupported/unknown | typed refusal | map/sequence types (onnx.proto:918-921) |
| ai.onnx.ml.FeatureVectorizer | ai.onnx.ml | (see Operators-ml.md) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | Concat |
| ai.onnx.ml.Imputer | ai.onnx.ml | (see Operators-ml.md) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | Where(IsNaN(x)\|x==v, fill, x) |
| ai.onnx.ml.LabelEncoder | ai.onnx.ml | (see Operators-ml.md) | Unsupported/unknown | typed refusal | string↔int maps; int→int LabelEncoder would be a lookup KERNEL |
| ai.onnx.ml.LinearClassifier | ai.onnx.ml | (see Operators-ml.md) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | Gemm / elementwise affine / norms over f32 kernels; post_transform variants |
| ai.onnx.ml.LinearRegressor | ai.onnx.ml | (see Operators-ml.md) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | Gemm / elementwise affine / norms over f32 kernels; post_transform variants |
| ai.onnx.ml.Normalizer | ai.onnx.ml | (see Operators-ml.md) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | Gemm / elementwise affine / norms over f32 kernels; post_transform variants |
| ai.onnx.ml.OneHotEncoder | ai.onnx.ml | (see Operators-ml.md) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | OneHot (int64 categories); string categories refused |
| ai.onnx.ml.SVMClassifier | ai.onnx.ml | (see Operators-ml.md) | Requires kernel | ndarray kernel (MISSING) | kernel functions (rbf/poly/sigmoid) over Gemm; no SVM kernel found |
| ai.onnx.ml.SVMRegressor | ai.onnx.ml | (see Operators-ml.md) | Requires kernel | ndarray kernel (MISSING) | kernel functions (rbf/poly/sigmoid) over Gemm; no SVM kernel found |
| ai.onnx.ml.Scaler | ai.onnx.ml | (see Operators-ml.md) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | Gemm / elementwise affine / norms over f32 kernels; post_transform variants |
| ai.onnx.ml.TreeEnsemble | ai.onnx.ml | (see Operators-ml.md) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | each leaf = conjunction of threshold atoms = `Rel::And[Atom…]` SHAPE, but thresholds are float/double with NaN missing-value tracks; i32 lanes would change semantics → gap; leaf-value aggregation needs float sums |
| ai.onnx.ml.TreeEnsembleClassifier (deprecated) | ai.onnx.ml | (see Operators-ml.md) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | each leaf = conjunction of threshold atoms = `Rel::And[Atom…]` SHAPE, but thresholds are float/double with NaN missing-value tracks; i32 lanes would change semantics → gap; leaf-value aggregation needs float sums |
| ai.onnx.ml.TreeEnsembleRegressor (deprecated) | ai.onnx.ml | (see Operators-ml.md) | Compositionally representable | ndarray view or kernel composition; inline FunctionProto body | each leaf = conjunction of threshold atoms = `Rel::And[Atom…]` SHAPE, but thresholds are float/double with NaN missing-value tracks; i32 lanes would change semantics → gap; leaf-value aggregation needs float sums |
| ai.onnx.ml.ZipMap | ai.onnx.ml | (see Operators-ml.md) | Unsupported/unknown | typed refusal | map/sequence types (onnx.proto:918-921) |

Corrections to the fact base found while building the matrix (opus-a, re-verified against the sibling checkout): `mish_f32` and `swish_f32` DO exist (`ndarray/src/hpc/linalg/activations_ext.rs:168,143`); consolidated F6 ("no mish") and opus-c item 5 ("Mish would need a new ndarray primitive") are therefore wrong on that point. `erf_f32` exists but is private (`:188`). The sentence "There is no shared relational AST, and none is built" is in `.claude/board/INTEGRATION_PLANS.md` under the entry "d-relalg-unified-v1 — JC/R statistics and Datalog sharing work through the contract set IR", not in `ISSUES.md` (consolidated F3 and opus-c item 1 cite the latter; `ISSUES.md` has no such line). Accuracy of the ndarray SIMD polynomial kernels (`vsexp`, `vstanh`) against ONNX reference outputs is UNKNOWN until a tolerance policy exists.

## 7. Stages S0–S5

Each stage names its scope, the existing code it reuses, its deliverable, a pre-registered gate or falsifier, and its non-goals. Stages are ordered; a later stage does not start until the earlier gate is green. S1 and later are blocked on the two operator inputs in section 8.

### S0 — this document (docs only)

- **Scope:** architecture, routing, coverage map, stage specs, open points, forbidden claims.
- **Reuse:** the fact base and the three Opus reviews.
- **Deliverable:** this plan and one board entry (`2026-10-23-d-onnx-relalg-cap-s0.md`). Nothing else.
- **Gate:** the document carries all 14 corrections of reviewer C and none of its forbidden claims (section 11).
- **Non-goals:** code, dependency changes, board-index edits, STATUS_BOARD rows, claims of any kind about compatibility.

### S1 — classification with explicit typed refusal (standalone probe crate)

- **Scope:** decode an ONNX `ModelProto` at the pinned tag; resolve every node by `(domain, op_type, since_version)` from `opset_import`; emit one routing decision per node: relalg island, R2IL/mask-risc terminal, ndarray kernel, or `Refusal { node, domain, op_type, since_version, reason }`. This is **classification plus explicit refusal. It is not "semantic preservation".**
- **Reuse:** the `relalg`/`lance-graph-contract` precedent of a standalone crate with its own `[workspace]` (`crates/lance-graph-relalg/Cargo.toml:11`). ONNX's own `onnx.proto` as the schema. The matrix in section 6 as the initial routing table.
- **Dependency rule:** no runtime (`ort`, `tract`, `candle-onnx`, onnxruntime) at S1. The minimum is a protobuf decoder for `onnx.proto` at **v1.23.2** (SHA of the `.proto` recorded), generated once (`prost-build`, or a checked-in generated file) inside the standalone probe crate, excluded from the main workspace. **A protobuf dependency must use an AdaWorldAPI fork if one exists.** The fork coordinates are UNKNOWN. This is a **STOP-and-ask item**: the session does not fall back to crates.io (`CLAUDE.md`, P0 "AdaWorldAPI forks ONLY"; the lance family is the only carve-out). See OP-6.
- **Target model:** exactly **one** concrete target model must be chosen to derive the op list. Without it the op list has no falsifier. UNKNOWN; open point OP-7. Candidates are a small MLP or an NNUE-style head, or an existing lab file such as the jina-v5 encoder only if it actually fits the subset.
- **What must survive decoding** (opus-a Q3): `ir_version`, `opset_import` per domain; `NodeProto.domain/op_type/overload`, input and output names with positions (an empty name is an omitted optional); attributes with type including `ref_attr_name`; graph inputs, outputs, `initializer`, `sparse_initializer`, the input/initializer overlap rule, `external_data`; `value_info` element types and shapes with symbolic `dim_param` kept symbolic; `functions`/`FunctionProto` identity `(name, domain, overload)`. **Refused with a typed error:** `training_info`, `ai.onnx.preview.training`, `device_configurations`, `quantization_annotation` when present, non-tensor `TypeProto` variants, dtypes without a kernel, unknown domains, op versions outside the importer's table. **Provenance only, never lowered:** `doc_string`, `metadata_props`, node `name`, dimension `denotation`.
- **Semantic bridge table** (binding; opus-c item 4). (a) ONNX has no NULL: the Kleene/ThreeValued mode and `is_not_null` are never engaged for ONNX input. (b) Empty `Min`/`Max`: relalg returns `None` (`Answer::Min(Option<i32>)`); ONNX yields the dtype maximum/minimum (`Operators.md:31040,30463`); map explicitly and test it. Empty `ReduceSum` is 0 (`:29496`). (c) ONNX reductions are bag reductions over positions; contract rewrites refuse Bag (`Refusal::Bag`). Declare `Population::Bag` and accept the refusal, or prove row-position distinctness per node. (d) Integer overflow: choose `Arith::Wrapping` or `Arith::Checked` per node, record it, never switch silently. (e) `Sum` is widened to `i64` in relalg (`lower.rs:121-125`) while ONNX `ReduceSum` keeps the input dtype: narrow with an explicit overflow refusal.
- **Pre-registered gate:** (1) every one of the 226 names plus the declared domain/version range classifies to exactly one route or one refusal, with a count that matches section 6; (2) a node outside the pinned IR/opset range is refused, not guessed; (3) on the chosen target model, 100% of nodes are routed or refused by name, with 0 silent drops; (4) each semantic-bridge row has a test whose disable run goes red.
- **Falsifier:** any node that is neither routed nor refused; any relalg node type, `SetExpr` variant or tensor type added; any trust placed in a pre-simplified graph.
- **Non-goals:** executing anything; shape inference beyond reading `value_info`; a new IR; a persisted routing format; any "ONNX-compatible" label.

### S2 — route the integer island through existing relalg, plus one nonlinear op

- **Scope:** (a) take the S1-classified integer selection/aggregate islands and run them through the existing path (`relalg::run`: contract normalize, R2IL lower, one mask-risc program). (b) Run exactly **one** nonlinear op on an existing ndarray kernel. PROPOSED choice: **Tanh** (`ndarray/hpc/vml.rs:617` `vstanh`; EXISTS). **Mish** also exists (`activations_ext.rs:168`, correction above) but is a compound whose accuracy is UNKNOWN; Tanh is the lower-risk first choice and satisfies both reviewers.
- **Reuse:** relalg `Query`, contract `set_rules`, R2IL `FoldDialect`, mask-risc terminals; ndarray `hpc/vml`; ndarray-core views for broadcasting (`impl_ops.rs:141`, `impl_methods.rs:2388-2395`).
- **Deliverable:** a lab test crate or example only; no production path.
- **Gate (pre-registered):** the integer island result equals an independent scalar reference on seeded fixtures including empty populations, all-true and all-false masks, `i32::MIN`/`MAX` thresholds; the nonlinear op matches an **independent** reference (a scalar libm implementation or ONNX reference output) within a ULP or tolerance bound that is stated before the run. Float results are **measured, never certified**.
- **Falsifier:** relalg, the contract or egglog evaluating, rewriting or "certifying" any float op; a float fusion or reorder described as certified; a Gather on the substrate path.
- **Non-goals:** broadcasting inside relalg; lane-versus-lane predicates (a KERNEL decision, OP-5, not a primitive); matmul in relalg; control flow; any claim of ONNX backend conformance.

### S3 — DR64³ lab probe

- **Scope (PROPOSED, opus-b Q6):** `EXP-DR64-CUBE-S3`, lab only, inside `crates/dr-math-binding-probe` (tests plus one example). Hypothesis: every DR64³ operation a ClassView method needs is expressible as contractions over bound DR64 views (chain form), equal to an independent oracle, with no cross-table gather and nothing materialized beyond 512 B per result unless a materialized arm is named. If every ternary-slab operation reduces to chain-form operations with no extra typing, the ternary type is PARKED; that counts as a result, not a failure.
- **Precondition (opus-c item 10):** before S3 starts, a written definition states what each axis is, whether 64³ bits (32 KiB) is resident or paged (the existing ruling says paged), what a bit means (participation only), and how identity binds (ClassView and frame guards). That definition does not exist (OP-1).
- **Reuse:** `relcomp::compose`, `Chain::new`, `certify_reassociation`, `execute_certified`; `dir-rmap-probe` for the DR256 comparison.
- **Arms:** A existing `compose` (value-derived reads counted); A′ coordinate-addressed compose, 64 fixed iterations, no data-derived index; B R2IL/mask-risc lowering, with `NOT EXPRESSIBLE` recorded as a result if it needs `MaskOp::Gather` or a new opcode; C Bag reading thresholded to Distinct, with an overflow assertion at 256 paths.
- **Gate:** all arms equal an independent coordinate-loop oracle that does not call `relcomp::compose`, does not use `trailing_zeros` and stores no tuples; all typing negatives refuse (axis mismatch, other frame, stale frame, other focus, other scope, ThreeValued, Bag, foreign-frame slice); silent twins execute. Timings in ns hot / µs cold, each naming its target-cpu/backend, labelled "reported, not gates". Disable runs D1–D5 (opus-b Q6) each go red; a disable that leaves its test green is reported as a non-binding guard.
- **Kill conditions:** any arm differs from the oracle; any typing negative executes; any arm or the oracle materializes tuples; any arm needs a cross-table positional gather; any timing in ms.
- **Non-goals:** no new identity, no 18-bit identity format, **no CausalEdge64 change**, no resident 2 MiB DR256³ buffer, no `[bool;64;64;64]` or tensor type, no new mask-risc opcode, no ternary ClassView signature before S3 shows the chain form is insufficient, no ONNX operator mapped onto DR64. **DR256³ is a separate track** (paged 4×4×4 blocks of DR64 composes). It needs a second certified law (distributivity of compose over OR; L1 lists "L2" as open) and is not part of S3.

### S4 — Palette256 and cognitive methods

- **Scope (PROPOSED):** state how ONNX weights or activations enter the palette substrate, and how cognitive methods consume ONNX-derived values, as **typed methods, not equivalences**.
- **Reuse:** `bgz_tensor::attention::ComposeTable` and `WeightPalette`; the P-S2 probe (`entries/2026-10-23-ps2-compose-chain-preregistration.md`: multi-hop composition through the compose table is UNSTABLE in both test worlds); `.claude/knowledge/identity-temporal-evidence-primer.md` ("the stored form is the code, never the decode").
- **Rules (binding; opus-c items 8 and 9):** the Palette256 code is the resident value. Float weights or activations are quantized once through a named, frozen codebook with a recorded fidelity figure (ρ). Fisher-Z decode is kernel-local only. Palette L1 distance, Fisher-Z alignment, Hamming popcount on Binary16K, and cosine are four different metrics with no implicit conversion. A tensor op on palette codes is not "the same" as on floats; results are approximations with a measured ρ against the float oracle. NNUE-style accumulators and learned LUTs are versioned state and proposals.
- **Gate:** a held-out error bound against the float oracle, stated before the run; no stage output is labelled equivalent or certified.
- **Non-goals:** treating a learned table as a law, an egglog fact or a certifier; mixing the four metrics.

### S5 — export and native codegen evaluation

- **Scope (PROPOSED):** (a) evaluate exporting a standard-op subset as an ONNX `ModelProto`; (b) evaluate native paths already in the stack: R2IL → mask-risc and `ndarray::simd`.
- **Export gate:** a produced model must pass `onnx.checker` and reproduce on a reference runtime. D9's "custom ops for triples" is not used if portability is the claim; a custom domain breaks portability.
- **LLVM and JIT:** documented only. An LLVM AOT or JIT path exists nowhere in the stack and would be a third executor; it is PARKED as a **separate decision** and is not part of this plan.
- **Non-goals:** any runtime in the execution path; any "ONNX round-trip" claim before export gates pass.

## 8. Ownership and dependencies

- **Coordination:** Linear EXO-50 (the shared blackboard issue). The GitHub board and this plan are the evidence ledger; Linear is for coordination only.
- **Owner of this plan:** the operator assigns. UNKNOWN until assigned.
- **Pending acknowledgement:** CC1 ProofRef ACK is pending (EXO-50 thread, messages 07–07e; no ACK posted as of 2026-10-23). S2–S5 inherit no certification until it is resolved (OP-3).
- **Blocking inputs before S1:** (1) the operator chooses one target model (OP-7); (2) the operator provides AdaWorldAPI protobuf fork coordinates, or confirms none exists and decides what then (OP-6).
- **PR dependencies:** none open. #1742 (RecipeCertificate keeps egglog `ProofStore` + `ProofId`) and #1743 (GF(2) elimination pre-registration) merged 2026-10-23.
- **Crate home for the importer:** UNKNOWN. It must not be relalg or the contract. Candidates: a new lab crate, `lance-graph-ogar`, or thinking-engine (which already holds the only candle dependency). Standalone probe crate is the S1 PROPOSAL.

## 9. Open-points register

| ID | Question | Owner | Dependency | Evidence | Falsifier | Decision | Status |
|---|---|---|---|---|---|---|---|
| OP-1 | DR64³: where is the original master plan, and what is DR64³ (axes, resident or paged, bit meaning, identity binding)? | operator | none | 0 files for the DR64³/DR256³ patterns on `main` and 833 remote branches (harvest-dr.md); `d-relalg-unified-v1.md:231` | A found plan; or S3 shows the chain form suffices and the ternary type is parked | none yet; no stages reconstructed | OPEN |
| OP-2 | ONNX coverage beyond the integer island: which ops, dtypes, broadcast and shape rules does any consumer need? | operator | OP-7 | matrix: 1 direct, 93 compositional, 91 kernel, 4 external, 37 unsupported/unknown; ~16 ops can reach relalg | the backend node-test pass rate for the declared subset | none yet | OPEN |
| OP-3 | Independent proof checker: none exists. `check_proof` is `pub(crate)` and egglog's `session.rs:5` says every proof "has already been checked by its own `check_proof`". | operator | CC1 ProofRef ACK; #1742 | consolidated F7; opus-c item 7 | a checker not derived from egglog that accepts and rejects the same proofs | S2–S5 inherit no certification | BLOCKED |
| OP-4 | Safe cross-snapshot rebinding: a certificate proven on one frame, reused on another with a different shape, codebook or frame version. | operator | L2 | L1 (#1740) proves reuse on fresh frames of the same shape only; `Chain::new` refuses `DifferentFrame` (`relcomp.rs:151-153`) | a changed-axis-type fixture where reuse is wrongly accepted | no cross-snapshot claim allowed | OPEN |
| OP-5 | Lane-versus-lane predicate and mask-to-lane cast in mask-risc, so `Less(a,b)` and `ReduceSum(Cast(mask))` stay inside one island. | operator | none | `LaneVersusLane` refusal (`lib.rs:217-220`); opus-a open question 4 | a physical kernel shown not to need a new `Pred` variant | KERNEL decision, not a primitive | OPEN |
| OP-6 | Protobuf dependency: does an AdaWorldAPI fork exist, and what are its coordinates? | operator | none | `CLAUDE.md` P0 "AdaWorldAPI forks ONLY"; consolidated F2 (no ONNX parser in any `Cargo.toml`) | coordinates supplied, or a statement that none exists | STOP and ask; no crates.io fallback | BLOCKED |
| OP-7 | Which one target model drives the op list? | operator | none | opus-c item 13a | the op set derived from the model is fully routed or refused | none yet | BLOCKED |
| OP-8 | D9 `to_onnx` "custom ops for triples", DU-1, ScenarioWorld, centroid-attention encoder: supersede, absorb or independent? | operator | OP-7 | `STATUS_BOARD.md:2329,2522`; `ScenarioWorldCounterfactual.md:163` | an export of a custom domain that a standard runtime accepts | PROPOSED: supersede D9 export; absorb DU-1 | OPEN |
| OP-9 | Native backend compatibility: do the ndarray kernels match ONNX reference outputs to a stated tolerance, per dtype and per op? | operator | OP-7 | accuracy of `vsexp`/`vstanh` polynomials vs ONNX reference UNKNOWN (opus-a open question 6) | a differential test beyond the tolerance | none yet | OPEN |
| OP-10 | Integer overflow mode per node: `Arith::Wrapping` or `Arith::Checked`? | operator | OP-7 | ONNX spec text read has no general rule (opus-a open question 3); R2IL raises `Overflow` | a model whose result differs between the modes | recorded per node | OPEN |
| OP-11 | Importer crate home. | operator | none | opus-a open question 1 | n/a | not relalg, not the contract | OPEN |
| OP-12 | What changed in each opset 23–28 re-version (Cast 28, Compress 28, Mod 28, Unique 28, …)? | operator | pin `Changelog.md` | opus-a open question 5 | a node whose v22 and v28 semantics differ and are keyed wrongly | key on resolved `since_version`, refuse unknown | OPEN |
| OP-13 | Is the EXTERNAL route (candle, candle-onnx) acceptable at all? | operator | OP-6 | opus-c item 3: oracle arm only, never execution; X10 jlrs (#1741) | n/a | oracle only | CLOSED (as oracle-only, per reviewer C) |
| OP-14 | Does the DR64 `compose` lower to mask-risc without `MaskOp::Gather`? | operator | S3 | opus-b open question 3; bit-broadcast not in the `MaskOp` list | arm B result `NOT EXPRESSIBLE` | report, do not add an opcode | OPEN |
| OP-15 | Is the set-bit row read in `relcomp::compose` (`relcomp.rs:214-218`) admissible under the D-RU-6 address rule that `qr.rs:22-23` follows? | operator | S3 | opus-b Q2 | A′ equals A without it | report by origin | OPEN |
| OP-16 | Third-axis typing if a ternary slab is ever needed; per-axis focus window. | operator | S3 | `MethodSignature` has two axes (`lib.rs:46-53`); `Focus` is one revision (`:167-170`) | S3 chain-form result | none before S3 | OPEN |
| OP-17 | Distributivity of Boolean compose over OR (needed for DR256 block paging). | operator | L2 | L1 result lists L2 as open (opus-b Q3) | block result differs from a direct DR256 compose | no DR256³ claim | OPEN |
| OP-18 | How far the certified-compiler boundary goes (scope question for the operator). | operator | none | `IDEAS.md` entry "A certified-compiler boundary for DR64", LEDGER-REVISION-2026-10-23e Q2 | n/a | none yet | OPEN |
| OP-19 | Palette256 quantization: which named frozen codebook and what ρ for ONNX weights? | operator | S1, OP-7 | opus-c item 8 | ρ below the stated bound on held-out data | none yet | OPEN |

## 10. Housekeeping audit

None of these are done by S0. Classes: SAFE NOW (edit with this docs PR), SEPARATE PR, BLOCKED.

| Item | Evidence | Class |
|---|---|---|
| Stale "no certified DR64 law exists" wording. Still true: the single-operand `admit_recipe` path refuses. Stale: `certify_reassociation` (#1707) certifies one law, and #1740 reuses it. | consolidated F7; harvest-dr.md | DONE: code comments in `dr-math-binding-probe` (`recipe.rs` module doc, `tests/contract_handoff.rs`) fixed in #1742; `IDEAS.md` status amended and `FRUITS.md` regenerated in this PR. Remaining limitations kept: no independent checker, same-shape reuse only. |
| X8 measured in #1740 (`tests/x8_fresh_frame.rs`) but the board row "The five unrun D-DRMC probes: X6, X7, X8, X16, X18" still lists it as unrun. The board defines X8 as "two certified bindings for one key (today the first falls back)", which differs from "fresh frame"; whether #1740 closes X8 as defined is UNCLEAR. | quotes-board.md (IDEAS.md/FRUITS.md rows); opus-b Q4 | SEPARATE PR, after the definition is checked |
| `PR_ARC_INVENTORY.md` has no entries for #1726–#1739; its top entry is #1725 (catch-up comment: "#1660–#1725 had no entries"). | worktree read, line 1-3 | SEPARATE PR (board catch-up) |
| `STATUS_BOARD.md` row D-TGA-5D (`:389`) says "In PR" although #1739 merged (merge `9a458cae6`). | worktree read | SEPARATE PR (board) |
| STATUS_BOARD has no D-DRMC / D-QR rows on main; #1742 adds a D-DRMC section. | consolidated F7 | BLOCKED on #1742 |
| `egglog` is a git dependency on `branch = "main"` with no `rev` (`crates/dr-math-binding-probe/Cargo.toml:15`, `crates/egglog-r2il-probe/Cargo.toml:25`). The repo tracks no lockfile, so the dependency floats. The fork rule is satisfied; reproducibility is not. | worktree read | SEPARATE PR; choosing a `rev` is an operator decision |
| D9 `TripletGraph::to_onnx()` "with custom ops for triples" conflicts with portable export (S5). | `STATUS_BOARD.md:2329`; opus-c item 13b | BLOCKED on OP-8 |
| Citation fix: "no shared relational AST" is at `INTEGRATION_PLANS.md:16`, not `ISSUES.md:16`. | worktree read | SAFE NOW (this document uses the correct citation) |
| Fact-base fix: `mish_f32`/`swish_f32` exist in ndarray. | `activations_ext.rs:143,168` | SAFE NOW (stated in section 6) |
| `DU-1` `recent_fingerprints: Tensor[N, 16384]` shape comment needs review against the 16K binary vs 16K-D float substrate split. | `unified-integration-v1.md:86` note in the board | SEPARATE PR |

## 11. Claims this document does NOT make

- It does not claim that `lance-graph-relalg` is the mathematical polyfill or the landing zone for ONNX. It is the landing zone for integer predicate selection with a reduction.
- It does not claim "ONNX-compatible", "ONNX support" or "ONNX round-trip". The only label is "ONNX intake (subset, opset N–M, ops {…}); all other nodes refused", and only after a backend-test pass rate exists.
- It does not claim "semantic preservation" for S1. The S1 claim is classification plus explicit refusal.
- It does not claim that relalg, the contract or egglog evaluates, rewrites or certifies Tanh, Mish, Softmax, MatMul or any float op.
- It does not claim any certified lowering, fusion or equivalence between an ONNX subgraph and a substrate program, nor any proof reuse across shape, codebook, frame version or snapshot.
- It does not claim that egglog proofs are independently checked.
- It does not claim that DR64³ exists, is defined, is resident, is fast, or that tensors project onto it. It gives no performance number for it and proposes no widening of CausalEdge64.
- It does not claim that Gather, GatherND or Scatter* are supported on the substrate path.
- It does not claim that Palette256 results equal ONNX float results, or that Palette L1, Fisher-Z, Hamming and cosine are interchangeable.
- It does not claim that learned or NNUE tables are mathematical truth, laws, or certified.
- It does not claim "no new graph format is created" while routing the whole graph through relalg. The diagram in section 2.1 routes nodes to existing owners; the ONNX `GraphProto` is the only graph.
- It does not claim that a runtime (ort, tract, candle) is part of the execution path, or that an LLVM AOT path exists.
- It does not claim "existing native execution" for any operator with no R2IL procedure, mask-risc terminal or ndarray kernel today (for example softplus as a named kernel, `If`, `Loop`, `Scan`).

