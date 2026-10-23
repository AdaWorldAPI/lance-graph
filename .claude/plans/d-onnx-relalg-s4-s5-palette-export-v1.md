# D-ONNX-RELALG-CAP S4–S5 — reuse existing BGZ / Helix / ndarray numerical execution
**Date:** 2026-10-23 · **Status:** PROPOSED integration and falsification plan; documentation only  
**Parents:** #1744, #1745, #1750 (CONFLICTING S0 ownership, no canonical merge approved), #1746 (intake/value dispatch), #1747 (focus), #1748 (physical execution).  
**Existing implementation anchors:** AdaWorldAPI/ndarray #42 and #100; lance-graph #1726 and #1731.  
**Critical correction:** Do **not** develop a new Palette256 codec, LUT algebra, Fisher-Z quantizer, attention-table framework, SIMD backend, or mathematical AST merely to satisfy ONNX. Those capabilities have existing implementations and different mathematical constraints. What is missing is verified ONNX admission, immutable model/layer/codebook identity, typed method binding and unbiased differential evaluation.

## 0. Existing capability and evidential status

| Existing surface | Source | Verified source facts | What is NOT established |
|---|---|---|---|
| **bgz-tensor** | `crates/bgz-tensor/src/{palette.rs,fisher_z.rs,attention.rs}` | `WeightPalette` with up-to-256 calibrated representatives/radii; `FamilyGamma` and `FisherZTable` (i8, gamma, 256²); `AttentionTable` (u16 distance), `ComposeTable` and typed LUT lookups | arbitrary ONNX MatMul semantics, closed multi-hop composition, globally valid metric-to-score conversion or no saturation |
| **helix** | `crates/helix/src/{lib.rs,distance.rs,quantize.rs}`, `KNOWLEDGE.md` | `RollingFloor`, `DistanceLut::{linear,from_floor,circular}`, 256² distance tables, deterministic place/residue. Circular metric test exhaustively enumerates 256³ triples | a generic neural activation codebook or interchangeable cosine/FisherZ codes |
| **bgz17** | `crates/bgz17/src/lib.rs` and KNOWLEDGE | existing Base17, layered distance and 256-entry palette components; the metric-safety contract distinguishes nonmetric Scent from L1/Hamming tiers | identical numeric semantics across every palette/codebook family |
| **ndarray** | `AdaWorldAPI/ndarray:crates/burn/src/ops/matmul.rs`, `src/hpc/vml.rs`, Burn `tensor.rs` | `CompiledAttention`, `CompiledLinear`, conventional `general_mat_mul` fallback; native `vstanh` f32 SIMD path and scalar/stride fallback; aarch64 NEON lanes and parity harness | transparent per-node ONNX compatibility, comprehensive identity guards, exact approximation, measured end-to-end advantage for arbitrary models |

**Empirical source report, NOT universal certification:** `bgz-tensor/src/fisher_z.rs` documents Spearman ρ ≥ 0.999 over 21 Qwen3-TTS roles with 5,000 sampled pairs each. Retain its exact dataset/role/sampling context and reproduce before treating it as an ONNX admission gate. Correlation is neither pointwise parity nor proof of a general mathematical identity.

**Finite-domain laws are scoped:** `helix::DistanceLut::circular` contains an exhaustive 256³-triple triangle-inequality regression; `linear` has a smaller stride-sampled test and the exact L1 definition. These metric properties are not certification of numerical `MatMul`, `Attention` or multi-hop arbitrary palette composition.

**Known empirical refusals:** #1726 reports `ComposeTable` multi-hop failure after snapping outside palette support. #1731 shows the learned LUT also fails k=2 outside a closed-carrier case. This refutes those *specific composition laws*, not Palette256 as a whole. EXO-129 / EXO-132 remain reference points; never enable multi-hop reuse based on one-hop table parity alone.

**Known quality-gate debt:** EXO-137 records that the main `rust-test.yml` runs only `bgz-tensor --lib`, omitting integration probes. EXO-139 records `FisherZTable::encode` saturating out-of-family values silently; in-range and saturation negative controls are specified. Do not use un-gated integration test reports or source comments as production ONNX certificates. Fix CI and checked encoding in separate owner-approved PRs, not this plan.

## 1. Existing execution selection: important hazards, not new kernel requests

The Burn `matmul` implementation already tries, in order:
1. `try_compiled_linear` on a registered shape;
2. `try_attention_matmul` on registered `d_head`;
3. `ndarray::linalg::general_mat_mul` as conventional fallback.

However its current **cache keys do not prove operand identity**:
- `ATTENTION_CACHE: HashMap<usize, CompiledAttention>` is keyed by inner dimension `k=d_head`. `try_attention_matmul(_lhs,_rhs,...)` ignores the current operand *values*; uses previously registered `q_assignments`/`k_assignments` and `table`, returns a scaled negative distance.
- `LINEAR_CACHE: Vec<CompiledLinear>` chooses first matching `(n_rows,n_cols)` even for distinct layers of identical shape. Despite a `pop_` name, it **does not consume** the matching entry. `try_compiled_linear(_lhs,_rhs,...)` ignores current `lhs` weight values; the numerical path quantizes centroids and runtime input columns. It records a scale-only approximation and comments that zero-point bias correction is absent.
- These paths are *implemented candidate specializations*, not safe generic transparent ONNX `MatMul` substitutions. A palette lookup of distance or centroid approximation does not automatically preserve the exact ONNX operator mathematical value. Existing code comments about 'O(1)' are per output lookup and exclude registration, assignment, table build, preparation and output materialization.
- Multiple models/layers with the same dimensions, different palettes or changed frame are a concrete **collision counterexample**. The currently registered compiled table must not be automatically applied to arbitrary ONNX MatMul nodes by matching shape alone.

**Fail-closed integration rule:** Until an owner-approved implementation verifies model/layer/codebook identity and numerical eligibility, ONNX STRICT_NATIVE must use conventional native tensor kernels for those operations or explicitly refuse. Compiled palette execution is admitted only for a witnessed, intentionally compiled model/layer with a declared approximation/error contract.

### Proposed binding/admission witness (conceptual, NOT a new ABI)

Check existing owners/types before adding fields. At minimum attest:
- Model graph identity, version/IR/opset and operator/node identity; stable layer/weight identity.
- Original tensor or weight digest **and** compiled artifact/codebook digest, revision and calibration generation.
- Exact `k`, `m`, `n`, dtype, strides/layout and relevant broadcast/attention scale/mask semantics.
- q/k assignment provenance or centroid row mapping, expected layer and shape, cache lifetime.
- Source-issued frame identity and distinct focus/scope/permission guards if using DR global-to-local binding.
- Valid numerical policy: EXACT only if established; otherwise `APPROXIMATE` with calibration population, error/quality thresholds and fallback.
- Versioned backend/kernel ABI, dynamic-input applicability, mutation/invalidations, and explicit model unload/rebind/clear behavior.

**Do not mint a new global identity encoding or put arbitrary model strings in a hot register.** Resolve cold at import/bind, then pass a compact method/capability handle through the existing guarded boundary. Reuse source-attested seals. No blind cache hit.

### High-priority negative tests (preregister before code)
- Two models with same head dimension but different Q/K palettes: compiled attention must refuse wrong model.
- Two same-shape linear layers with distinct weights: must refuse shape-only collision.
- Same model/layer after recompilation, codebook change, new assignments or changed scope/frame: old compiled recipe must refuse or safely rebind under proven rules.
- Current Q/K inputs differ from registration assignments: attention must not silently return identical precomputed output as exact ONNX MatMul.
- Out-of-domain `FisherZTable::encode` saturation is detected, not reported as maximal in-family similarity.
- `CompiledLinear` scale/bias error, boundary quantization, sign, NaN, Infinity and overflow are measured against an independent full-precision reference.
- Multi-hop composition across unsupported codebook support remains rejected even when one-hop distances correlate well.
- Regression: with no compiled registration, standard Burn numeric behavior is unchanged.

## 2. Stage S4 — reuse and numerical admission, NOT Palette256 redevelopment

### S4a: source and test inventory
**Haiku:** inventory `bgz-tensor`, `bgz17`, `helix`, `ndarray` and existing numerical performance paths. Record exact source/commit, public APIs, tests and whether they are enforced in CI. Include Burn `float_tanh` → `ndarray::hpc::vml::vstanh` and the 256² compiled cache methods. Compare documented capability to actual source: SHIPPED, LAB, PROPOSED, REFUTED, UNKNOWN. Identify relevant `helix/KNOWLEDGE.md` Overlap & Consolidation plan; do not create competing quantizers.

### S4b: typed ONNX dispatch to existing implementations
**Sonnet:** design the smallest immutable *ONNX operator → existing method* dispatch, not new `Rel::Tanh`, `Rel::MatMul` or a universal numerical expression tree. `GraphProto` remains interchange graph; RelAlg handles admissible Boolean/relational support and result demand; ndarray/Burn/Palette256 methods remain numerical owners. A standalone `Tanh` does not need a DR register.

**Opus:** assess exact vs approximate eligibility, law family and use of cached compiled artifacts. Numerical operator semantics (and any partial model delegation) require separately specified reference and conversion policies. Do not use an ONNX runtime merely as a witness of a mathematical theorem.

### S4c: codebook and functional quality gate
- The direct `vstanh` path is a concrete existing ONNX Tanh kernel candidate; evaluate its SIMD approximation, NaN/Inf/signed zero, tail and strided policies against a reference.
- A compiled palette is intentionally approximate unless an exact restricted-domain proof exists. Require held-out per-model/per-layer quality metrics, max/percentile error and task-level regressions (e.g. audio/attention) where applicable; ρ alone is insufficient.
- Reuse `WeightPalette` distortion radii, FisherZ gamma, Helix `DistanceLut`, existing NEON/AVX and BF16 conversions with typed codebook identity. No implicit FisherZ/Hamming/Palette/Tanh conversion.
- Bayesian/Hebbian/BNN plasticity, NNUE accumulator state and cognitive shaders are *method-level proposals/state transitions*; do not turn them into ONNX mathematical identities or proof certificates. Verify full refresh vs incremental update only for existing supported owners.
- #1726/#1731 multi-hop failures impose scoped restrictions; do not describe all of Palette256 as unimplemented or refuted.

**S4 gate:** existing kernels reused, real model/layer/codebook identity boundary specified, wrong-artifact negative tests preregistered, typed numerical quality/oracle parity for one admitted operator, and any approximations explicitly opt-in. No new codec or runtime.

## 3. Stage S5 — export, cold compilation, reuse and certification

**S5a export:** versioned ONNX subset roundtrip using standard operators where possible; custom domain serialization is not automatically portable. Preserve metadata, shapes, effects and external tensor-data safety. Existing compiled palette methods may require an explicit custom operator or a native binding extension; either must advertise portability limits and declared approximation.

**S5b reuse existing cold recipe compiler:** D-DRMC Boolean/Distinct associativity #1707/#1738 and fresh-frame reuse #1740/#1742 are *scoped*. Egglog ProofStore/ProofId is retained in #1742; egglog's self-check is not an independently authored verifier. Mathematical law ≠ empirical codebook accuracy ≠ typed scope authorization ≠ cost preference. Keep any palette-calibration receipt separate from a theorem certificate.

**S5c AOT/JIT:** optional only after #1748 shows an uncovered measurable regime; no LLVM, e-graph or ONNX graph executor in the runtime fold. Measure compile, calibration, bind, proof, transfer, cache invalidation, execution and output materialization separately. Reuse existing SIMD and Burn numerical fallback by default.

**S5d hardware:** ndarray already provides ARM NEON, AVX and scalar paths, plus parity harnesses. Do not specify a second NEON kernel framework; add only target-specific measured crossover benchmarks when hardware is available.

**Thought-Git:** evidence, provenance, failed calibration candidates, recipe selection and replay under existing write authority. Do not promote correlations, stochastic parameter updates or source comments into a mathematical proof or commit permission.

## 4. Cross-PR implications
- **#1746:** Value-only ONNX Tanh maps to existing `ndarray::hpc::vml::vstanh`/Burn; `Rel` remains narrow, no new value AST. ONNX model/layer/node identity must survive importer.
- **#1747:** A DR focus bound to a model-specific codebook needs the same source-issued identity, seals and snapshot guards as all other operands. Holographic focus is optional, not a universal palette read.
- **#1748:** Existing `CompiledAttention`/`CompiledLinear` are a measured *candidate*, not an automatically equivalent Tensor/DR/Hybrid plan. Compare exact T/D/H arms first; add an explicit palette-compiled arm only under an opt-in APPROXIMATE quality contract (or after proof of exact restricted-domain parity). Measure cold compilation, table generation, weight assignment, codebook calibration, identity-bind/cache overhead and full output.
- **#1744/#1745/#1750:** the same S0 path conflicts. Preserve #1744's source inventory and #1750's board progress; #1745's three-peer physical concepts are an amendment proposal until EXO-50 resolves ownership. Do not merge three competing S0 files.

## 5. Risk register and exact open points
- **OP-ONNX-1:** typed value-math dispatch ownership without a new AST.
- **OP-ONNX-2:** model importer and protobuf fork choice.
- **OP-ONNX-4:** witnessed intra-register bit enumeration vs forbidden cross-table gather.
- **OP-ONNX-5/7/9:** independent checker, prover revision, stale 'independent' wording.
- **OP-ONNX-BIND-1 (NEW PROPOSAL):** immutable model/layer/weight/codebook/assignment identity and safe invalidation for Burn `CompiledAttention`/`CompiledLinear`.
- **OP-ONNX-PAL-2 (NEW PROPOSAL):** quality contract permitting an approximate palette method for each ONNX operator and domain.
- **OP-ONNX-PAL-3 (NEW PROPOSAL):** reconcile FisherZ checked encode (EXO-139), ungated integration tests (EXO-137) and multi-hop failures (#1726/#1731; EXO-129/EXO-132) before production promotion.

All three new IDs are proposals local to this PR, **not** existing canonical board issue IDs. Owner/board must decide their durable identifiers.

### Falsification checklist
F-S4-1: same dimension different model/weight -> wrong compiled cache hit.
F-S4-2: same shape distinct layers -> wrong CompiledLinear reuse.
F-S4-3: changed palette generation and old assignments -> stale result.
F-S4-4: palette score differs from direct ONNX QK numeric contract.
F-S4-5: incomplete zero-point correction creates unacceptable numeric error.
F-S4-6: FisherZ saturation looks like in-family evidence.
F-S4-7: multi-hop ComposeTable approximation erroneously certified.
F-S4-8: a result claimed CI-gated depends on integration tests not run in CI.
F-S5-1: independent proof checker falsely inferred from egglog self-check.
F-S5-2: ONNX custom domain assumed portable without a runtime handler.
F-S5-3: cold compile/calibration/cache costs omitted from benchmark.
F-S5-4: model-specific approximation silently executed in strict exact native mode.

Each falsifier needs: fixture, expected refusal or fallback, independent reference, can-fire/can-stay-silent controls, reproducible run and owner-approved stage gate.

## 6. Agent and ownership protocol
Haiku source/test/CI inventory. Sonnet draft typed integration and plan costs. Independent Opus review of ONNX equivalence, exact-vs-approximation and cache safety. Operator and repository owners authorize code, ABI changes and merges. Coordinate through [Linear EXO-50](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id). Durable evidence stays in GitHub board, with dated append-only amendments; do not rewrite earlier records.

## Verified code and issue references
- BGZ: https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/bgz-tensor/src/fisher_z.rs
- BGZ attention: https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/bgz-tensor/src/attention.rs
- Helix: https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/helix/src/lib.rs
- Helix DistanceLut: https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/helix/src/distance.rs
- bgz17: https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/bgz17/src/lib.rs
- ndarray Burn compiled paths: https://github.com/AdaWorldAPI/ndarray/blob/master/crates/burn/src/ops/matmul.rs
- ndarray tanh: https://github.com/AdaWorldAPI/ndarray/blob/master/src/hpc/vml.rs
- ndarray historical PRs: https://github.com/AdaWorldAPI/ndarray/pull/42 ; https://github.com/AdaWorldAPI/ndarray/pull/100
- Existing method falsifiers: https://github.com/AdaWorldAPI/lance-graph/pull/1726 ; https://github.com/AdaWorldAPI/lance-graph/pull/1731
- Linear CI/encoding debt: https://linear.app/exo-red/issue/EXO-137/bgz-tensor-integration-tests-never-run-in-ci-rust-testyml-runs-lib ; https://linear.app/exo-red/issue/EXO-139/fisherztableencode-clamps-out-of-family-cosines-to-the-edge-code-with

---

## 2026-10-23 separate cold frontend plan: PR #1751

**New independently owned research plan:** [D-OPTIMUM-BGZ-OFFLINE-v1](https://github.com/AdaWorldAPI/lance-graph/pull/1751), a docs-only cold Optimum ONNX export/calibration → existing BGZ/Palette256 sidecar adapter, with an operator-approved compact BERT-layer numeric probe.

**Ownership correction and stage framing:**
- **#1751 O0–O3 owns** source-backed `huggingface/optimum-onnx` API inventories; pinned model export and ORT/source comparison; calibration workflow reuse; layer/weight/codebook fingerprints; an optional *illustrative, non-destructive* compiled sidecar manifest. It **does not** implement a runtime or require an Optimum fork.
- **This #1749 S4 owns** the native `bgz-tensor`/`helix`/`bgz17` and ndarray compiled method *admission contract*: existing `CompiledLinear`/`CompiledAttention` identity, exact vs approximate quality, explicit cache-invalidation and fail-closed fallback. Do not duplicate Optimum exporter or build a second palette compiler.
- **This #1749 S5 owns** versioned output/sidecar compatibility review, native method reuse, cold proof/certification boundaries and measured optional codegen; no general custom ONNX operator portability implied.
- **#1748 owns** physical Tensor/DR/Hybrid and quality-gated compiled palette crossover benchmarks, with Optimum-exported fixtures from #1751 when available.

**Critical seam:** the Optimum-exported original `MatMul/Gemm` remains the semantic reference; the existing BGZ/CompiledLinear representation is *a candidate* with calibration and numerical errors. Model revision, initializer weight digest, node/path, codebook family+generation and assignments must match at runtime before using a compiled artifact. The global Burn dimension-keyed paths must not become silently enabled for arbitrary imported ONNX models. For POC, prohibit unguarded cache registration; use a local harness or separately authorized identity-bound runtime entrypoint.

**Source references:** `huggingface/optimum-onnx/optimum/exporters/onnx/{model_configs.py,convert.py}`, `optimum/onnxruntime/{quantization.py,optimization.py}`; native `AdaWorldAPI/ndarray/crates/burn/src/ops/matmul.rs`; existing `lance-graph/crates/bgz-tensor`. `ORTQuantizer` does not create BGZ codebooks merely by choosing a different INT8 type. Optimum stays in the **offline Python compiler**. No Python/ORT runtime requirement on the target edge CPU.

**Gate:** S4 native adaptation requires a source-verified candidate and model/layer provenance plus owner signoff; O1 model selection and protobuf fork decision are still operator-owned prerequisites per #1750. #1744/#1745/#1750 S0 remains unresolved. Neither doc plan can auto-merge or authorize runtime changes.
