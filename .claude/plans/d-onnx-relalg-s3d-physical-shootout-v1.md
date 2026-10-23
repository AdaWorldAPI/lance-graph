# D-ONNX-RELALG-CAP S3d — Tensor × DR × Hybrid physical strategy shootout

> **⊘ SUPERSESSION — D-NFC (2026-10-23, operator-approved architecture correction; canonical text `.claude/plans/d-native-first-dependency-correction-v1.md` and representation seams `.claude/plans/d-native-repr-contract-v1.md`, both in [#1753](https://github.com/AdaWorldAPI/lance-graph/pull/1753)).**
>
> 1. **Native methods first.** Benchmarks compare physical implementations (tensor / DR / hybrid / palette) of one native mathematical contract with an independent oracle. The #1752 layer benchmark is the baseline shape: ndarray → mask-risc → R2IL, with a fair scalar loop alongside (A3).
> 2. **ONNX conformance is a separate test dimension** on the same operations, not the definition of the benchmarked operation (A3).
> 3. **The prerequisite on #1746 typed ONNX intake is removed** (A2). The prerequisite is a native contract with an oracle.
> 4. **The architectural target is not the benchmark winner.** DR64³ + Palette256³ is the intended representation; a benchmark decides only its physical realization (A5).
> 5. **Exact versus approximate.** A compiled palette candidate competes only on a declared approximate frontier with a named error contract; its fallback is the exact native method.
>
> Passages below that contradict these points are marked ⊘ in place; nothing else in this plan changes. The #1744 / #1745 / #1750 S0 ownership dispute is unchanged.

**Date:** 2026-10-23 · **Status:** PROPOSED differential experiment plan (docs only)
**Prerequisites:** #1745 semantic capstone, #1746 typed ONNX intake, appropriate #1747 S3a/S3b verified contracts.
> ⊘ **D-NFC A2/A3: #1746 typed ONNX intake is not a prerequisite; a native contract with an oracle is.**
**Purpose:** physically compare legal recipes for *the same* ONNX mathematical operation, not change the mathematics to make DR win.

## Architecture and scope
T (Tensor/CISC-like): optimized dense GEMM, Conv, Attention, BLAS/GPU/CPU kernels, relevant packing and fused tensor paths.
D (DR/RISC-like): deterministic Boolean support and address selection, typed numerical folds, existing borrowed SoA, DR64/DR256, mask-risc/R2IL and SIMD.
H (Hybrid): DR chooses legal support/tiles; a numerical microkernel evaluates relevant dense regions.

CISC/RISC are high-level physical-planning analogies, not hardware ISA assertions. SIMD may appear in all three. No new optimizer, graph format or physical executor is authorized; existing native procedures and external backends supply candidates.

## Eligibility before cost
For an ONNX function f, each plan must preserve:
- versioned operator identity, type and shape/broadcasting;
- semiring/value carrier and reduction order/numeric error policy;
- result demand; Bag/Distinct and SQL K3 must not be silently collapsed;
- global coordinate identity, support completeness, frame/focus/scope;
- effects and state transitions;
- materialized output contents and shape.

Numeric MatMul: C[i,j]=sum_k A[i,k]*B[k,j], not Boolean reachability. A DR-selected support S_ij is exact algebraically only when omitted contributors are zero under the applicable numeric semantics; floating-point observable parity additionally depends on signed zero, NaNs, infinities, overflow and accumulation order. Zero * infinity can produce NaN: do not treat every multiplication involving a zero as discardable. If correctness needs a global expansion, include expansion in cost. Unsupported paths must refuse or use an explicitly eligible fallback.

## Verified laboratory precedent to reproduce, not generalize
PR #1685, D-RFT-1 (600 users / 3 pages; median 101 runs, host-specific):
- A direct register composition: **12.4 µs**, 6 answer-lane allocations;
- C+ support then AND/POPCNT: **14 µs**, 0 allocations;
- G dense INT8 tile GEMM with bit-to-byte unpacking: **57.5 µs**, 9 allocations;
- C all-coordinate: **147–151 µs**.
Independent PostgreSQL Bag/Distinct parity was documented. Arm G is a dense implementation of the **same integer relational multiplicity calculation**, not proof of general floating-point ONNX GEMM replacement. Costs include relevant conversion, but a separately prepacked/reused dense baseline is still needed in the expanded experiment.

Existing example: crates/dir-rmap-probe/examples/rft1_bench.rs; execution crates remain their current owners.

## Stage P1 — relational reproduction and cost decomposition
Haiku find existing fixture/benchmark paths and actual test invocations. Sonnet specify fair cold, warm, prepacked arms using identical fixture/output demand. Opus inspect parity, alias/NULL/Bag guards and negative cases.

Record:
- bind/seal and per-frame authorization;
- prepare/index/packing/conversion;
- execute and output materialization;
- allocations/bytes and hardware/runtime;
- cold setup versus N repeated executions.

Gate: reference parity; no unsupported architecture claims. Do not modify existing bench without owner review.

## Stage P2 — genuine numeric ONNX MatMul
Pin actual ONNX MatMul opset and shape/type fixture. At least:
- T: optimized available numeric dense baseline (plus separately prepacked/warm baseline);
- D: typed numerical sum-product over legally identified support (not OR/AND in place of arithmetic);
- H: supported tile/region selection followed by numeric microkernel.
For small matrices use an independent scalar oracle; extend with a reference ONNX runtime only if independently accessible. Test K dimensions, densities (0 / sparse / medium / dense), varied spatial locality, repeated reuse, type widths, shape/broadcast constraints, NaN/Inf/signed-zero and overflow policies.

Measure end-to-end from comparable input representation; report format conversion fairly for all arms. Ensure that a zero support mask cannot mask invalid arithmetic. No benchmark winner until all arms pass the declared operator semantic gate.

## Stage P3 — masked reduction and focus completeness
Selected focus intentionally omits a required nonzero contribution. Expected: legal expansion, fallback or typed refusal, never an incomplete global reduction. Also test when mask is a **semantic** exclusion (legitimately changes domain) vs merely a **physical focus window** (does not change operator meaning).

## Stage P4 — Conv and Attention inventory, not premature implementation
Conv candidates: layouts, pad, stride, dilation, groups, halo and tile boundaries; measure whether DR support is usable.
Attention candidates: masks, softmax denominator dependency, causal behavior, scale and cache/state. Missing globally required keys must not silently be dropped. Start with exact admission maps and reference fixtures.

## Cost and plan selector boundary
Candidate conceptual formula:
C_total(P,N) = C_cold(P)/N + C_bind(P) + C_prepare(P) + C_convert(P) + C_execute(P) + C_materialize(P) + C_transfer(P).
Add explicit validation and rebind costs where not subsumed. Count shared preparation once, indicate input layout and repeated-run amortization. Report separate tail latencies, throughput, cache/memory traffic; energy only with reproducible measurement. Optimization based on a target platform must not generalize to Raspberry Pi, AMX host and GPU without per-target data.

COLD compiler may enumerate legal candidates, choose measured physical recipe, cache/propose under D-DRMC law and binding guards. Proof/admission precedes selection. No egglog in hot fold. A faster but non-equivalent plan is disqualified; a proved rewrite is not automatically faster.

## Falsification gates
F-P1: wrong numerical semiring and Bag/Distinct conflation.
F-P2: local focus treated as complete reduction.
F-P3: packing/preparation hidden from 'fast' arm.
F-P4: dense kernel denied prepacking/reuse opportunity.
F-P5: zero × Inf and floating order change results.
F-P6: cross-snapshot scope, source or method mismatch.
F-P7: physical winner reported without full output parity.
F-P8: 'energy saved' inferred from runtime only.
F-P9: hybrid output materializes hidden intermediate tensors.
F-P10: cold optimization is accidentally required in hot runtime.

All failures require a recorded refusal, fallback or contract revision, with reproduced oracle.

## Decision matrix
For each operation × density × hardware × result demand, record T/D/H legality, correctness, cold/warm total cost, memory bytes and regression cases. Outcome = ADOPT / KEEP_DIRECT / FALLBACK / REJECT / DEFER. Negative results are valid.

**Gate S3d:** at least one selected mathematical operation has three eligible plans, independently validated outputs, separately measured end-to-end costs and a falsifiable selector decision. Until then all speed/energy/hardware wins are hypotheses.

## Sources
#1745 S0, #1746 S1/S2, #1747 S3a/b/c, #1685 lab precedent, #1707 bounded law certification, #1740 narrow fresh-frame reuse, .claude/plans/dr-mathematical-compiler-v1.md.
---

## 2026-10-23 Cross-session review amendment: per-tile crossover, reuse and ARM target

**Dependency warning:** #1744 and #1745 have overlapping S0 ownership. Do not promote this plan to implementation until the canonical S0 and OP-ONNX-1 value-dispatch decision are settled. #1744's observed limitations replace any unverified assumption that numeric operators already lower through `Rel`.

### P5: Within-operator tile crossover
The whole-query three-arm comparison is insufficient to justify a hybrid planner. Generate a **single typed MatMul or masked reduce workload containing mixed dense and sparse regions**. Use identical inputs, output shape, accumulation policy and support-completeness evidence. Compare:
- T: whole-operator prepacked dense numerical kernel.
- D: entire operator via admitted support selection and typed numerical fold.
- H: each *tile* chooses DR fold or packed numerical microkernel using a declared threshold, including support construction, tile classification, switch dispatch, packing, output merging and materialization.
- H-off control: same tile partition without heterogeneous switching, to reveal dispatch overhead.

Sweep matrix shape, inner dimension, tile size, sparse/dense distribution, mask clustering, repeated calls and the allocation/source-witness pattern. Report **crossover regions** and regimes where hybrid is strictly worse, not just a winning median. Record physical selection decisions as performance evidence, not law certificates.

### P6: Reuse economics and incremental updates
Record separate regimes:
1. **Cold:** import/bind/pack/compile/cache acquisition included.
2. **Warm:** same frame, unchanged inputs and reused prepared buffers.
3. **Repeated inference:** N calls (e.g. N=1, 8, 64, 256; adjust by runtime) showing amortized costs.
4. **Incremental update:** one tile/weight/edge or registered input changes, requiring exact invalidation of dependent contributions, compared with full recomputation and NNUE-like incremental update where supported.

Cost must report the setup and revalidation of support, optional prepacking and numeric kernels for *all* arms. For incremental operation, track exactly which epoch/frame/scope and dependency witness permits buffer reuse. Do not treat a changed binding as an unchanged hot hit.

Use explicit break-even estimates: selected H beats T or D only when total comparable cost with reuse is lower, and result parity is preserved.

### P7: ARM NEON and hardware-portability target
Add an **independent ARM64/NEON experiment target**, preferably a Raspberry Pi-class machine when actually available, without assuming such hardware exists in CI. If unavailable, mark TARGET BLOCKED / NOT MEASURED rather than simulating throughput. Compare appropriate ARM numerical library (including prepacked path) versus existing or explicitly scoped SIMD/bitfold paths, with cold-start latency, memory budget, peak bytes, energy only if measured. Record CPU model, RAM, compilation flags and NEON availability. Do not extrapolate an AMX result to ARM.

Optional GPU arm is a distinct environment with host↔device transfer and launch overhead measured, not an implicit Tensor winner.

### New falsifiers
F-P11: H chooses tiles on density alone but all-in switch+packing overhead loses to both uniform plans.
F-P12: T receives a cold baseline while DR silently reuses warmed support; fix unfair comparison.
F-P13: DR skips mathematical contributors based on a physical focus rather than a validated dependency mask.
F-P14: incremental method misses one dependency and disagrees with fresh full recomputation.
F-P15: ARM speed or power claims reported without real ARM hardware and measurements.
F-P16: an optimized plan falls back mid-computation without accounting for intermediate representations, data transfer or effects.

**Updated gate S3d:** one genuine ONNX numerical operator has legal T/D/H execution on the same declared semantics, a heterogeneous-density crossover sweep, cold/warm/reuse scenarios, and independent reference parity. Hardware-specific conclusions require measurement on that hardware; an ARM target may remain pending without blocking the x86 proof-of-concept.
> ⊘ **D-NFC A3: read as one native numerical operation; its ONNX conformance is checked separately.**

---

## 2026-10-23 physical-plan expansion: implemented compiled palette candidates

**Important baseline correction:** `AdaWorldAPI/ndarray` already contains alternative physical execution methods, not merely prospective kernel ideas:
- PR #42: registered 256×256 `CompiledAttention` lookup, d_head-keyed, with ordinary Burn MatMul fallback.
- PR #100: `CompiledLinear` centroid/assignment execution, shape-matched, VNNI/scalar and ordinary MatMul fallback.
- Burn `float_tanh` dispatches to `ndarray::hpc::vml::vstanh` on eligible f32 SIMD layouts.
- `bgz-tensor`/ `bgz17` / `helix` already supply codebooks, FisherZ, 256² LUTs and metric tests. Do NOT build replacements in S3d.

### Four *candidate* lanes, two different semantic frontiers
The existing three arms **T** (conventional numeric Tensor), **D** (DR legal deterministic support+typed numerical fold), and **H** (hybrid DR support + numeric microkernel) remain the **exact/numerically-toleranced competition**, subject to the ONNX op's declared semantics.

Introduce **P — existing compiled palette** as an **independently reported fourth candidate**, not silently counted as an exact arm:
- P-attention: 256² `CompiledAttention` distance-index scores; its `try_attention_matmul` ignores incoming lhs/rhs values, uses cached assignments and returns scaled negative distance. This is *not* an exact ONNX `MatMul` result by construction.
- P-linear: 256-centroid `CompiledLinear` with per-row assignments and input quantization; `try_compiled_linear` ignores its lhs weight values and uses an approximate scale-only correction (zero-point bias correction is not implemented). It may differ from standard numeric MatMul.

**Admission:** P competes on the *approximate / explicitly calibrated quality frontier* only after an identity-safe model/layer/weight/codebook binding has been demonstrated. It can join an exact/tolerance-governed frontier only if independently proven compliant with the declared operator tolerance for the admitted domain. Otherwise P is REJECT/REFUSE for STRICT_NATIVE. A cached-hit shape alone is not a witness.

### Fair costs and crossover
Benchmark T, D, H and when admissible P, separately tracking:
- Compile/calibrate codebook, source/weight digest, quantize + assign rows, build palette and distance/compose/FisherZ table.
- Cache registration, model/layer/weight/codebook digest validation, frame rebind and invalidation.
- Per-run input quantization, zero-point/scaling correction, support construction, DR focus expand and per-tile dispatch.
- Kernel, fallback, host/device transfer, output materialization, memory bytes and peak residency.
- N=1/warm/repeated/incremental; cache hit/miss and collided-key negative controls.
- x86 AMX/AVX, existing ndarray ARM NEON and scalar baselines, actual hardware measured only.

For each eligible operator compare end-to-end latency **and** independent numeric/semantic quality, including held-out task metrics if approximations are admitted. No speedup may compensate for an unacknowledged semantic violation.

### Mandatory added falsifiers
F-P17: cache `d_head` collision between two independent models;
F-P18: identical matmul shapes but different linear weight matrices (same-shape cached artifact);
F-P19: stale codebook/assignment generation or mismatched current Q/K values;
F-P20: CompiledLinear missing zero-point correction yields excessive numeric error;
F-P21: FisherZ saturation or unsupported multi-hop composition (#1726/#1731) is treated as a valid score law;
F-P22: claiming CI-gated palette parity when `bgz-tensor` integration tests are excluded from CI (EXO-137).

### Source evidence
`AdaWorldAPI/ndarray/crates/burn/src/ops/matmul.rs` and `src/hpc/vml.rs`; lance-graph `crates/bgz-tensor/src/{attention.rs,fisher_z.rs,palette.rs}`, `crates/helix/src/distance.rs`, `crates/bgz17/src/lib.rs`; ndarray #42/#100; lance-graph #1726/#1731; Linear EXO-137/EXO-139.

**S3d gate revision:** a three-arm exact comparison still tests the Tensor/DR/Hybrid thesis. An additional P-arm is a *reuse benchmark* only under separately admitted model-specific accuracy/quality semantics. No unguarded Burn compiled interception may be promoted as generic ONNX transparent MatMul.

---

## 2026-10-23 offline workload/quality generator (#1751)

[PR #1751](https://github.com/AdaWorldAPI/lance-graph/pull/1751) introduces an **independent, documentation-only** Optimum model export/calibration/sidecar research plan. Its eventual pinned one-layer BERT fixture should supply the first **real numerical MatMul** reference and source/weight/codebook provenance. Do not create another Optimum adapter inside S3d; this PR owns only differential physical-plan benchmarking and crossover.

**Updated benchmark scope:**
- Retain exact/numerically governed T/D/H, with comparable original inputs and full tensor numerical reference, zero-copy/DR admission where legal.
- Optional P is a *separate approximate quality frontier*, for existing BGZ/ndarray `CompiledLinear` or `CompiledAttention` only after #1749 identity-safe admission. A 256² lookup is **not** evidence of exact ONNX MatMul.
- Reuse Optimum's source-to-ONNX reference validation and source model revision, but run an **independent extra compiled-layer oracle**; validate whole-model quality where meaningful, not only per-layer output.
- Count once-only export, codebook build, calibration data prep, embedding/assignment serialization and native bind cost. Report per-call runtime and N-call amortization **separately** so offline work is neither hidden nor charged to every call incorrectly.
- Compare version-pinned optimizer-off versus optimized export separately: fusions may alter graph/node identity and tensor operators.
- S3d no longer promises to develop palette quantizers, compiled linear routines or a new NEON frontend. The experiment tests existing methods and crossover.

**Falsifier:** a physical plan wins on mean execution latency while failing model-layer binding, held-out numerical quality, exact declared ONNX semantics or reproducibility. Such a plan is ineligible regardless of speed. S3d requires #1746 semantic intake + #1749 compiled-method admission; #1751 supplies optional offline fixtures but doesn't remove these gates.
> ⊘ **D-NFC A2: requires a native semantic contract, not ONNX intake.**

No benchmarks run in this documentation amendment.
