# D-OPTIMUM-BGZ-OFFLINE-v1 — Hugging Face Optimum as a cold Palette-RISC model frontend

> **⊘ SUPERSESSION — D-NFC (2026-10-23, operator-approved architecture correction; canonical text `.claude/plans/d-native-first-dependency-correction-v1.md` and representation seams `.claude/plans/d-native-repr-contract-v1.md`, both in [#1753](https://github.com/AdaWorldAPI/lance-graph/pull/1753)).**
>
> 1. **Optimum is downstream and optional** (layer [8] of the canonical diagram). No native layer depends on it, and it is never in the execution path (A6).
> 2. **The original ONNX graph is an external validation artifact.** The canonical fallback for an approximate compiled candidate is the exact native method on the uncompressed weights (A4).
> 3. **"ONNX remains the graph"** holds only for this offline export; ONNX is not the native representation (A1). The native representation is Quack binding + DR64³ + Palette256³.
> 4. **This plan is not an extension of native stages.** It may supply external fixtures to #1748 and an external reference to the conformance adapter; native work does not wait for it.
> 5. **The protobuf fork question gates only an external ONNX-file adapter** (A2).
>
> Passages below that contradict these points are marked ⊘ in place; nothing else in this plan changes. The #1744 / #1745 / #1750 S0 ownership dispute is unchanged.


**Date:** 2026-10-23
**Status:** PROPOSED / DOCUMENTATION ONLY; no Optimum adapter, compiled-model artifact, test or numerical result is implemented by this PR.
**Scope:** Separate offline compiler/export/calibration integration plan.
**Parent research:** D-ONNX-RELALG-CAP drafts #1744, #1745, #1750 (S0 ownership unresolved), #1746 (S1/S2 semantic intake), #1747 (DR focus), #1748 (physical strategy benchmarks), #1749 (existing BGZ/Helix/ndarray reuse and safe binding).
**Coordination:** Linear EXO-50; GitHub board stays the durable, owner-controlled evidence ledger. Do not modify another session's S0 document.

## 0. Why this is a separate PR

Hugging Face **`huggingface/optimum-onnx` is an already implemented Python OFFLINE export and calibration ecosystem**, not a numerical theorem prover and not our target runtime. Reusing its model-specific ONNX export, model validation, ORT calibration-data machinery and selected optimizer orchestration removes significant model-preparation work.

The work is **NOT** to modify Optimum's INT8 quantizer by renaming a dtype, replace ONNX with BGZ, port Python/ORT to Rust, introduce a new graph AST, or create a general-purpose graph executor.

Instead, investigate a minimal *adapter or out-of-tree plugin* that consumes a version-pinned ONNX model exported through Optimum, builds an existing `bgz-tensor`/Palette256 representation where meaning and quality admit it, and emits an optional sidecar for native ndarray/DR binding. Fork Optimum only if extension points cannot meet the need; record the demonstrated limitation first. Upstream Optimum is Apache-2.0; respect its notices/dependency/model licensing separately.

**Core claim to falsify:** For one frozen model layer, an offline Optimum-exported numerical operation can be represented by an existing BGZ/Palette256 compiled candidate with identity-safe runtime binding, independently measured numerical quality and a fallback to the original tensor execution.

This claim does not imply exact equivalence, general ONNX operator coverage, broad portability or a speedup.

## 1. Verified external source interfaces (inspect pinned release before use)

| Existing Optimum component | Exact source | Intended reuse and constraint |
|---|---|---|
| `main_export` / CLI | `optimum/exporters/onnx/__main__.py`, `optimum/commands/export/onnx.py` | Export declared architecture to ONNX, pin model revision/weights, opset, IR, library versions. Do not silently optimize away node identity. |
| `BertOnnxConfig` etc. | `optimum/exporters/onnx/model_configs.py` | Model-specific inputs, shapes, export configurations. Choose one operator-authorized tiny BERT model for the first POC. |
| `OnnxConfig` | `optimum/exporters/onnx/base.py` | Typed input/output names, dummy samples and model-specific export characteristics, **not** runtime operator admission. |
| `validate_models_outputs` | `optimum/exporters/onnx/convert.py` | Compare PyTorch/model reference against ONNX export; does not validate the new Palette method, so add a separate layer-level differential oracle. |
| `ORTQuantizer`, `ORTCalibrationDataReader` | `optimum/onnxruntime/quantization.py` | Reuse sample/preprocessing/calibration *workflow* where suitable. ORT's INT8/QDQ quantization semantics are **not** BGZ/Palette256 codebook construction. Avoid modifying a quantizer until a concrete API seam is identified. |
| `ORTOptimizer` | `optimum/onnxruntime/optimization.py` | Optional benchmark/optimization reference. ORT fusions and approximations can change operator identity and output tolerances; preserve a pre-optimization control graph. |
| License & package dependencies | `LICENSE`, `pyproject.toml` | Apache-2.0, Python and versioned dependencies. Prefer external offline use without vendoring; model weights carry independent terms. |

Source: https://github.com/huggingface/optimum-onnx
Reference export docs: https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model

**Version lock gate:** pin the exact upstream git SHA/package version, Transformers, PyTorch, ONNX, ONNX Runtime, model revision, tokenizer and exported ONNX domain/opset/IR. Defaults differ by model and Optimum revision. Pin before counting operators or recording fidelity.

**Security:** external ONNX tensor data/path resolution and resource limits; honor `trust_remote_code=false` unless separately authorized. Prefer deterministic local pinned sources, no model weights or calibration data in Git without explicit permission/licensing. Record whether upstream optimizer was disabled for the baseline.

## 2. Existing execution substrate: REUSE, not redevelopment

- `crates/bgz-tensor`: `WeightPalette`, calibrated family scale via `FamilyGamma`, `FisherZTable`, attention `AttentionTable` and `ComposeTable`, existing sampling/quality evidence (source reports Spearman ρ≥0.999 across 21 Qwen3-TTS roles × 5,000 pairs).
- `crates/bgz17`: Base17 / 256-palette and layered metric candidates.
- `crates/helix`: `RollingFloor`, place/residue and `DistanceLut`, with `KNOWLEDGE.md` overlap-and-consolidation plan; exhaustive 256³ circular triangle-inequality check only certifies that table's metric law.
- `AdaWorldAPI/ndarray`: Burn `CompiledAttention` (#42) and `CompiledLinear` (#100), scalar/AVX/VNNI/ARM-NEON paths, `vstanh`, ordinary `general_mat_mul` fallback.
- #1726/#1731 REFUTE particular multi-hop composition laws outside closed codebook support; do NOT generalize failure to all Palette256.
- Linear EXO-137 documents BGZ integration tests excluded by `cargo test --lib`; EXO-139 documents silent out-of-family FisherZ saturation. These are **production promotion blockers**, not instructions to rebuild the codecs.

### Verified current bind hazards

The current `ndarray/crates/burn/src/ops/matmul.rs` selects:
- `CompiledAttention` via a `d_head`-keyed global cache; compiled attention reads previously registered q/k assignments and a distance-derived score, not current LHS/RHS values;
- `CompiledLinear` via first matching matrix dimensions from an enduring `Vec`; it does not prove current LHS weights are those used for centroids, and its quantization omits full zero-point correction;
- conventional numeric MatMul when no registration intercepts.

These optimizations **are existing concrete candidate kernels**, NOT safe generic ONNX MatMul replacements, even on identically shaped tensors.

An offline compiler must NEVER promote a shape-only cache hit into exact model/weight authority. Until admitted identity and numerical semantics are established, retain conventional numeric execution/fallback or typed refusal.

## 3. Target pipeline and ownership

```text
Hugging Face revision / model configuration (offline Python)
                    |
                    v
Optimum export + model-reference validation
(ModelProto/GraphProto, exact opset, original weights)
                    |
                    v
Cold offline adapter (PROPOSED, not new graph IR)
  - choose one eligible node/layer
  - bind model/layer/initializer and version
  - build existing BGZ/Palette256 representation
  - capture codebook/gamma/assignment provenance
  - validate fidelity against independent tensor reference
                    |
                    v
Non-destructive optional sidecar artifact
  original ONNX graph intact + BGZ sidecar + manifest
  reference fixtures + admitted numerical policy
                    |
                    v
Existing runtime dispatcher (owner-controlled, FUTURE)
  identity+witness check before cache lookup or use
  - Tensor: original typed numerical execution
  - Compiled Palette: calibrated, opt-in, model-bound
  - DR / Hybrid: only if existing ClassView/global-to-local
    dependencies, support and numeric contracts permit
                    |
                    v
Reference fidelity, cold/warm cost, replay and fallback
```

**No mandatory DR64³:** one isolated BERT weight MatMul must be testable without cubic relation implementation. DR64 focus is an optional later physical strategy; not required for model export/calibration. `Rel` remains the existing integer relational planner under D-RA-2, not a general numeric AST. ONNX remains the graph.
> ⊘ **D-NFC A1/A5: "ONNX remains the graph" means the external export graph only; DR64³ remains the intended native relational representation.**

## 4. Proposed model artifact (NOT an existing serialized format)

A non-destructive *illustrative* package, subject to repository owner approval:

```text
model/
  model.onnx                      # original exported ONNX, unchanged
  palette/
    weights.bgz                    # compiled existing BGZ representation (proposal)
    codebooks.bin                  # existing codec representation, versioned
    manifest.json                  # proposed typed artifact identity/admission
  validation/
    numerical-results.json        # layer and whole-model oracle + accepted errors
    fixture-metadata.json          # seeds, sampled inputs, reference versions
  execution/
    recipes.json                  # legal physical candidates, not executable AST
```

Do not invent a new binary BGZ ABI if existing serialization is sufficient. `manifest.json` is illustrative, NOT canonical schema, contract or new proof certificate.

**Cold identity manifest candidates**, resolved against existing types first:
- original source model revision/license; SHA/content identity of ONNX graph and relevant initializer tensors; exact graph domain/opset/IR, stable node and model config;
- layer/node identity including model-local path and expected input/output names, dtype and shape/broadcast rules (node name alone need not be stable after export/optimization);
- codebook family and version, centroid table and assignment digest, calibration dataset/seed/version, FamilyGamma or other typed metric metadata;
- compiled backend and method revision, ONNX operator semantics, exact-versus-approximate numerical contract;
- permitted static/dynamic operands and known state effects; scope/frame/focus witness rechecked by runtime if DR participates;
- linkage between the original graph, transformed graph (if any), and calibration/layer samples, provenance for reuse/invalidation.

A digest authenticates identity only when the trusted binder verifies it against the intended model/layer and permitted source; hashing does not prove mathematical correctness or permission. No arbitrary dynamic strings in the hot DR register.

## 5. Staged POC and smallest independently falsifiable step

### O0: Optimum-to-existing-source interface map (Haiku scout → Sonnet draft → Opus falsifier)
- Inventory exact public API/CLI and version/revision; model-specific `BertOnnxConfig` export, validation and ORT calibration seams.
- Inventory existing `WeightPalette`/codebook builders, ndarray Burn compiled-layer registration and known quality gaps.
- Map data from ONNX initializer + chosen layer to existing BGZ inputs, metadata, calibration and native kernel; show actual type/shape boundary; avoid a guessed adapter function.
- Identify whether extension can remain out-of-tree without an upstream Optimum fork.
- Gate: path+symbol+version evidence and one reproducible proposed execution chain. Docs-only; no model downloaded or dependency added in O0.

### O1: One frozen BERT-layer export & reference probe (requires operator-approved model/revision)
- Select a **compact BERT encoder** and ONE numeric MatMul/Gemm layer; avoid Q/K attention-score replacement as the initial demonstration because distance lookup is not ONNX QK dot product by construction.
- Pin full model and dataset/sample revision, deterministic input/seed, export opset, dtype, optimizer setting (initially OFF) and software versions.
- Run model/source to ONNX export/validation; retain graph/initializer digest. Capture the selected layer's numeric outputs with a genuinely independent reference method.
- Gate: stable node/initializer identification, explicit ONNX operator semantics and reproducible numeric reference, including multiple test batches, not just a passing export.

### O2: Offline BGZ calibration and compiled candidate (separate code PR, not authorized here)
- Feed exact frozen layer weight rows to existing codebook/calibration APIs. Identify required layout conversion and avoid silently replacing original MatMul operator semantics.
- Compare compiled `CompiledLinear`/BGZ prototype output against original fp32/declared tensor path. **Do not use unguarded global Burn registration** to claim validity; a local harness or separately authorized identity-guarded binder is required.
- Record calibration build time, trained/held-out sample size, distortions, per-layer max/p50/p95/p99 errors, downstream model/task quality where available, dtype/rounding and in-range/saturation.
- Fail closed if identity, support, numerical quality or allowed approximation is not established. A high Spearman ρ alone is not an ONNX fidelity gate.
- Gate: reproducible layer-level fidelity report, exact identity witness and explicit opt-in approximation policy; exact numeric path remains the reference/fallback.

### O3: Sidecar packaging and safe replay (separate implementation decision)
- Reuse already-versioned BGZ serializer where compatible. Attach content-addressed sidecar without destructively rewriting ONNX. Verify exact source↔compiled binding and failed stale/mismatched-artifact tests.
- No forced schema, graph serializer or model interpreter. Export no custom ONNX domain until portability implications are understood.
- Gate: deterministic build/replay under pinned versions, content integrity, model/layer cross-binding refusal and valid original-ONNX fallback.
  > ⊘ **D-NFC A4: the canonical fallback is the exact native method; an original-ONNX run is an external comparison.**

### O4: Physical crossover (after #1748 contracts, optional)
- T: conventional numeric tensor.
- D: admitted DR typed support/fold.
- H: DR support + numeric microkernel.
- P: compiled Palette under **separate approximate or proven restricted exact quality policy**.
- Distinguish cold export/calibration/caching costs from warm runtime and update invalidation; compare same input samples, hardware, model identity and output demand. Measure CPU AVX/VNNI and ARM NEON only where real hardware or admissible reproducible benchmarks exist.
- A faster but wrong/unadmitted compiled palette result never wins the exact frontier.
- Gate: documented crossover with independent correctness/quality and full all-in cost, including calibration+identity binding.

## 6. Mandatory falsification matrix

| ID | Counterexample / negative control | Expected response |
|---|---|---|
| O-F1 | Two models share d_head but have different attention codebooks | reject wrong cached artifact; use original tensor |
| O-F2 | Two equal-shaped BERT layers have different weights | shape-only linear hit is **not** admitted |
| O-F3 | ONNX optimizer fuses/renames nodes, changes opset/initializer paths | recover witnessed lineage or refuse artifact reuse |
| O-F4 | Dynamic input Q/K differs from prior palette assignments | no exact ONNX MatMul substitution |
| O-F5 | CompiledLinear scale/zero-point bias changes numeric outputs | report error; reject unless explicitly admitted approximate |
| O-F6 | FisherZ saturates values outside calibration family | checked refusal/flag; never claim maximal in-family evidence |
| O-F7 | Multi-hop composition outside closed codebook | do not promote ComposeTable as universal semiring law (#1726/#1731) |
| O-F8 | Held-out layer/model quality regresses despite good cosine correlation | refuse compiled quality candidate |
| O-F9 | Warm tensor kernel has prepacked weights, Palette has cached tables | include both equal reuse assumptions and full lifetime costs |
| O-F10 | BGZ tests pass locally but relevant integration suites are absent in CI | report evidence as ungated; prevent production promotion (EXO-137) |
| O-F11 | BERT export has unsupported op/control flow | strict native typed refusal or approved whole-model external fallback; no partial graph execution without verified existing scheduler |
| O-F12 | Artifact reads modified ONNX weights or stale codebook | mismatch detected; fallback/refusal |
| O-F13 | External tensor paths escape allowed model root or exhaust memory | resource/path validation error |
| O-F14 | Optimum ORT calibration parameters used as BGZ palette without a declared mapping | refuse incompatible quantization semantics |

For each test, pre-register can-fire/can-stay-silent controls and independent reference; report outcome as PASS/FAIL/NOT RUN, not merely `tests included`.

## 7. Cross-PR and board implications (no silent writes to other owners)

- **#1746 S1/S2**: Reuse Optimum as optional **offline** frontend for model-specific export and reference validation. It does NOT make `Rel` numerical or make the ONNX runtime mandatory. Preserve model identity and original graph. Separate native-only strict admission from future authorized fallback.
- **#1747 S3a–c**: No DR64³ requirement for the first BERT-layer POC. For later DR focus, carry exact model/layer/codebook source witnesses through existing frame/focus/scope and ADMISSION B. Semantic dependency closure first.
- **#1748 S3d**: Existing Palette compiled candidate is a quality-gated fourth strategy; Optimum can provide offline calibration and reproducible model fixtures, not a blanket Tensor-equivalent arm. Benchmark cold/warm and prepacked baselines.
- **#1749 S4–S5**: This PR is the upstream offline **source of admitted compiled artifacts**; #1749 remains owner of native compiled-method eligibility, cache identity repair, export and codegen proposals. Avoid duplicate converter/framework.
- **#1744/#1745/#1750**: Competing S0 path and different ownership/board conclusions are still in HOLD. The Optimum plan is a distinct downstream extension and **does not resolve or modify canonical S0**. Communicate through EXO-50 and request owner decision before amending S0 board ledgers.
- **EXO-137/EXO-139**: remain independently owned CI/codec correctness work. Do not close from this documentation plan.
- **EXO-50**: post source-backed findings, stage dependencies, and falsification review request using the append-only bus envelope.

## 8. Decisions required before O1/O2 execution

1. Operator selection: one compact BERT model and pinned revision; code/weight licensing and data provenance.
2. Confirm the AdaWorldAPI ONNX/protobuf fork coordinates or explicit absence, as requested by #1750's S1 gate. Do not add crates.io dependencies contrary to owner rules.
  > ⊘ **D-NFC A2: needed only if an external ONNX-file adapter is built; not a native prerequisite.**
3. Optimum/Transformers/ONNX pinned versions and authorization for Python offline environment.
4. Which existing BGZ compiled artifact serialization/builder API is actually suitable for a BERT row-centroid layer.
5. Independent ONNX/PyTorch reference, held-out calibration and numerical error/quality thresholds.
6. Ownership for any ndarray Burn identity-safe selector changes. No runtime API/ABI change from this plan.
7. Canonical S0 owner across #1744/#1745/#1750 (independent blocking dependency for integrating these plans, not for drafting this source map).

## 9. Agent handoff and closeout discipline

- **Haiku scouts:** pinned Optimum interface/symbol inventory; existing BGZ/Helix/ndarray API, tests/CI and licensing; detect duplicate frontend efforts.
- **Sonnet:** minimal non-destructive artifact/identity mapping and O1–O4 experiment specifications; cross-PR impact notes.
- **Independent Opus:** challenge numerical equivalence, cache provenance, approximation frontier, graph optimization lineage, effect and security/authorization boundaries.
- **Operator/owners:** choose reference model and revision, provider/fork, stage implementation and any merges. No recursive agent teams; two bounded research waves and one independent adversarial review.
- **No auto-merge**, board closure, code changes, new dependencies, Python runtime packaging or unapproved model downloads in this docs PR.

### Key references
- Upstream: https://github.com/huggingface/optimum-onnx ; https://github.com/huggingface/optimum-onnx/blob/main/optimum/exporters/onnx/model_configs.py ; https://github.com/huggingface/optimum-onnx/blob/main/optimum/exporters/onnx/convert.py ; https://github.com/huggingface/optimum-onnx/blob/main/optimum/onnxruntime/quantization.py ; https://github.com/huggingface/optimum-onnx/blob/main/optimum/onnxruntime/optimization.py
- BGZ: https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/bgz-tensor/src/attention.rs ; https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/bgz-tensor/src/fisher_z.rs ; https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/helix/KNOWLEDGE.md
- Native: https://github.com/AdaWorldAPI/ndarray/blob/master/crates/burn/src/ops/matmul.rs ; https://github.com/AdaWorldAPI/ndarray/pull/42 ; https://github.com/AdaWorldAPI/ndarray/pull/100
- Refuted laws: https://github.com/AdaWorldAPI/lance-graph/pull/1726 ; https://github.com/AdaWorldAPI/lance-graph/pull/1731
- Cross-agent: https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id ; EXO-137 ; EXO-139

**Governing principle:** Optimum understands and exports the model; existing BGZ/Palette256 encodes eligible numerical structure; the witnessed compiler artifact binds that structure to the actual model/layer; ndarray executes; DR optionally selects relational support; independent references decide accuracy and measured costs decide usefulness. Never turn a fast approximate LUT into a silent exact ONNX operation.
