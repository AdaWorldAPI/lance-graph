# D-ONNX-RELALG-CAP S3d — Tensor × DR × Hybrid physical strategy shootout
**Date:** 2026-10-23 · **Status:** PROPOSED differential experiment plan (docs only)
**Prerequisites:** #1745 semantic capstone, #1746 typed ONNX intake, appropriate #1747 S3a/S3b verified contracts.
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
