# Codec Invariants & Experiments

> Session-end catalogue of every compression approach tried in PRs #176–#185
> and the lesson each one produced. Nothing is thrown away. Future sessions
> should use this to recognise which approach fits a given tensor shape,
> role, and quality gate — and to mutate from when the immediate path fails.

## Core invariants (must always hold)

These are structural truths about this codebase that every future codec
must respect. Violating any of them silently corrupts downstream state
(see `#183`, `#184`, #185` for each class of violation in the wild).

### I1. Two regimes, opposite needs

| Regime | Where it lives | What it requires | Error shape |
|---|---|---|---|
| **Argmax-decoded** | attention / MLP / logits / codec head | top-1 argmax stability under `hidden @ W.T` | robust to cos ≈ 0.95 |
| **Index-lookup** | `text_embedding`, `lm_head`, `code_embed` | per-row identity | cascading — no argmax downstream rescues |

Empirically measured on Qwen3-TTS-0.6B: 477/478 tensors survived RVQ at cos ≈ 0.95 (argmax regime); one vocab tensor at cos = 0.05 **destroyed the pipeline** (index regime). See PR `#178` passthrough fix.

### I2. Near-orthogonality of weight rows in high dim

Qwen3 weight matrix rows in 1024-d or 2048-d space behave near-orthogonal for random pairs. Any compression that assumes rows cluster tightly in L2 is wrong.

Concretely this refutes:
- `RVQ_K_LADDER_TUNING.md § 3` claim "one L2 centroid per row at ≤3 rows/leaf → cos ≈ 1" (disproven by PR `#177` HCLAM run: cos = 0.0046).
- Any single-centroid tree quantisation without directional residual (`HhtlDTensor::reconstruct_row` without SlotL cannot synthesise direction).

### I3. Direction vs amplitude cannot be merged into one scalar

A scalar residual (like `Slot V`) can only shift magnitude. It cannot describe direction. Any codec that uses one scalar magnitude + direction-less centroid misses high-dim directional information entirely.

This was the unstated assumption baking into `BGZ_HHTL_D.md`'s "cos ≈ 0.95 typical" claim — probably true for *HHTL cascade inference* (table lookup), definitely false for *f32 GEMM reconstruction* (measured cos = 0.04 on real Qwen3 in PR `#183`).

### I4. Wire-format type widths are hard caps, enforce at encode time

`HhtlF32Entry.twig: u8` silently wraps `ci as u8` for `k > 256` (caught in `#185` codex review). Always `assert!(k <= MAX_*)` at encode sites. Widening the index (u8 → u16) is a wire-format change; log-companded bucketing is the alternative.

### I5. "u8 can span u16/u64 effective" requires the right decoder

Per the bgz17 philosophy: u8 × BF16 (amplitude) × gamma (stride) = u24–u64 effective precision at decode time — **if and only if** the decoder evaluates the universal curve parametrised by those values, not a straight-line interpolation or a tile-back.

`Base17::to_f32` is the floor (tile-and-average). The elevator
(`rehydrate_interpolated` with γ+φ weighting) lives in
`highheelbgz::rehydrate` and is **not wired into `HhtlDTensor`** — that's
part of the gap that made PR #183 fail.

### I6. The ticket-for-curve model

The real primitive per the bgz17 design: each row = a ticket on a
universal kurvenlineal (curve).

```
Universal curve:   r(θ) = a · e^(bθ)  or fitted anchor spline
Ticket per row:    (start, stop, stride, polarity)  — as few as 1 signed byte (i8)
Shared per group:  curve anchors (K × 17 × 2 B BF16), gamma profile (28 B)
```

Reconstruction = curve evaluation at the ticket's parameters. Not
`centroid + residual`. Not tile-and-average. Not tree quantisation.
**`highheelbgz::rehydrate::SpiralEncoding` implements this — useful for
token signature retrieval, but signature-only (not dense reconstruction).**

### I7. Vector-as-location vs vector-as-sparse-signal — the regime split again

The session's hardest lesson: two different framings of "compress a vector"
exist and can't share primitives.

**Vector-as-location** (Cartesian coordinates, L2 distance):
- Raw f32 weight rows, each dim independent.
- Codecs: centroid + residual, tree quantization, palette lookup.
- **On near-orthogonal high-dim rows these ALL fail** (see A3, A6, #183, #184).

**Vector-as-sparse-signal** (Phase + magnitude on an orthogonal basis):
- Project onto orthogonal basis (JL = random ±1 signed, SVD = data-fit, Hadamard = structured JL).
- Encode as (sign, log-magnitude) per projected coefficient = **PolarQuant**.
- Inner products preserved by Lindenstrauss concentration-of-measure.
- No centroid, no palette, no SVD needed — **the orthogonal leaf IS the representation.**

**If the leaf is orthogonal (JL/Hadamard/PolarQuant), you do NOT need the other (centroid + residual).** They solve different problems. The centroid framework was wrong for argmax-regime tensors because argmax needs inner-product preservation, which JL gives directly.

Already in the repo:
- `crates/bgz17/src/rabitq_compat.rs` — JL + dot_correction (structured JL via Hadamard rotation)
- `crates/bgz-tensor/src/matryoshka.rs` — PolarQuant gain-shape split
- `crates/thinking-engine/examples/turboquant_correction_probe.rs` — 466-line probe comparing 4 correction methods across 33-layer chain simulation

**`TurboQuant = PolarQuant + JLQ + correction`** is the canonical argmax-regime codec. This session's centroid+residual work (A5, A6, A7) was solving the wrong problem.

## Approaches tried, what each one was, where it fits

### A1. `HhtlDTensor` — Base17 + Slot D + Slot V (PR #173–#174, codebase existing)

- **What**: 4 B/row tree address (HEEL 2b + HIP 4b + TWIG 8b + polarity) + BF16 scalar magnitude
- **Designed for**: HHTL cascade lookup inference (Skip/Attend/Compose/Escalate routing)
- **Measured on**: Qwen3-TTS-0.6B reconstruction path in `#183` — cos = 0.04
- **Verdict**: **Correct codec, wrong application**. Use for cascade inference (`bgz-tensor::hhtl_cache`). Do NOT use for f32 GEMM reconstruction.
- **Mutation hooks**: Slot L residual (PR `#181`) adds direction correction; Slot V is still unused in `reconstruct_row`. If f32 GEMM is the target, ADD a curve-evaluator decode path (`rehydrate_interpolated`) instead of the current Base17 tile-back.

### A2. Progressive residual RVQ with k-ladder (PR #176)

- **What**: Multiple CLAM codebooks per tensor, residual accumulates across levels
- **Measured on**: Qwen3-TTS-0.6B vocab embedding — cos = 0.054
- **Verdict**: **Works on argmax-regime tensors** (477/478 hit cos ≈ 1). **Fails on index-regime vocab tensors** because k=4096 < rows/4 on 151K-row vocab.
- **Mutation hooks**: Extend k-ladder for large-vocab tensors (e.g. `[256, 1024, 4096, 16384]`) OR switch those tensors to passthrough BF16 (what #178 did).

### A3. Hierarchical CLAM 256×256 (PR #177, REFUTED by #178)

- **What**: Tree quantisation: 256 L1 coarse clusters × 256 L2 fine centroids per cluster, one leaf per row, no residual sum
- **Measured on**: vocab embedding — cos = 0.0046 (**worse than RVQ it replaced**)
- **Verdict**: **Structurally incapable of reconstructing near-orthogonal rows.** Single-centroid picks one existing row as the answer; for near-orthogonal distinct rows, cos ≈ 0.
- **Mutation hooks**: Do NOT use for reconstruction. Could work for lookup-grade routing where only nearest-centroid identity matters, not value fidelity. That is what `HhtlDTensor` already is.
- **Refutation notice**: `docs/RVQ_K_LADDER_TUNING.md § 3` must be read with this refutation in mind.

### A4. Passthrough BF16 for `n_rows > 8192` (PR #178, SHIPS)

- **What**: Skip compression entirely on vocab-sized tensors
- **Measured on**: Qwen3-TTS-0.6B — codec token match 225/225 = 100%
- **Verdict**: **Correctness ship-grade.** Storage ratio 1:1.39 (net loss) — not a product.
- **Mutation hooks**: Replace passthrough with any index-regime codec (SpiralEncoding shared-anchor, HhtlDTensor + SlotL properly reconstructed, f32 palette with log-radial CLAM) as soon as that codec hits ρ ≥ 0.98 on real vocab rows.

### A5. SlotL — 8 × i8 directional residual on shared SVD basis (PR #180, #181, #182)

- **What**: 8 i8 coefficients on a palette-shared Matryoshka SVD basis; encoder projects `row − centroid` onto basis, quantises
- **Measured on**: synthetic low-rank — ρ ≥ 0.98; paired with Base17 centroid on real Qwen3 — ρ ≈ 0.04 (ineffective because centroid is direction-less)
- **Verdict**: **Algorithm is correct in isolation.** Fails at integration because it's adding a direction correction to a centroid that has no direction.
- **Mutation hooks**: Keep the module, reuse with a directional centroid (f32 CLAM or curve-eval output). SlotL is a generic residual primitive that composes.

### A6. HhtlF32Tensor — f32/BF16 CLAM centroid palette + SlotL (PR #184)

- **What**: Replaces Base17 palette with CLAM centroids stored as f32 vectors; reuses SlotL residual
- **Measured on**: Qwen3-TTS-0.6B — ρ̄ ≈ 0.2–0.5 (10× better than Base17's 0.04, still short of 0.95 target)
- **Verdict**: **Right direction, insufficient bandwidth.** k=256 + 8 SVD coefficients is not enough for 1024-d near-orthogonal rows.
- **Mutation hooks**: k=512 or 1024 (needs widening twig to u16); per-leaf local SVD basis; log-radial CLAM on unit-normalised rows. Module already has codex-P1 bounds enforcement from #185.

### A7. cascade_attention_probe — HhtlCache + FisherZTable table lookup for attention (PR #184)

- **What**: Replace `Q · K^T → argmax` with `FisherZTable[pal_idx(Q), pal_idx(K)] → argmax`
- **Measured on**: layer-0 k_proj, 512 queries — 3.71% top-1 agreement
- **Verdict**: **Fails because Base17 palette doesn't preserve inner-product neighbourhoods.** Not an argument against codec-space inference; an argument that the palette under it must preserve inner-product structure first.
- **Mutation hooks**: Retry with f32 CLAM palette (Path A under Path B) — cascade inference only works when the palette faithfully partitions by inner product. This is the Path B / Path A dependency that wasn't clear before running the probe.

## Abstractions that ARE the right primitive

### R1. `highheelbgz::rehydrate::SpiralEncoding`

- 6-byte `SpiralAddress` (start, stride) + K anchors × 17 × 2 B BF16 per row
- `GammaProfile` shared per model (28 B: role_gamma[6] + phi_scale)
- `rehydrate_interpolated(target_spd, gamma)`: φ-weighted interpolation `frac.powf(1/GOLDEN_RATIO)` between anchors — **golden-rule reconstruction, not linear interpolation**
- Self-test in module: exact match round-trip ρ = 1 on self; different vectors get ρ < 1; 1000-token vocab < 200 KB

This is the real kurvenlineal codec. Every other "reconstruction-grade" attempt in this session is a less-capable cousin.

**Unproven**: has not been measured against real Qwen3-TTS weight rows end-to-end. That's the missing probe — see § Open probes.

### R2. Per-role stride in `NeuronPrint` (highheelbgz lib.rs)

Six `SpiralAddress` fields, one per role, with fixed strides per the design:

```
q:    stride=3    (attention, must match K)
k:    stride=3    (attention)
v:    stride=5    (content)
gate: stride=8    (thinking style)
up:   stride=2
down: stride=4    (down/up ratio = effective rank)
```

Total 36 bytes per neuron (6 roles × 6 bytes). This is what `should_use_leaf` / `classify_role` in `bgz-tensor::shared_palette` was reaching toward — mapping roles to per-role encoding parameters. **Currently the two schemes aren't integrated.**

### R3. HHTL cascade inference (`bgz-tensor::hhtl_cache`)

RouteAction { Skip, Attend, Compose, Escalate }. `HhtlDTensor` + `FisherZTable` composed at inference time replaces `hidden @ W.T` with table lookups.

**Requires**: a palette that preserves inner-product neighbourhoods (the Base17 palette probably does *not* — see A7 above). The Path A+B dependency.

## Open probes (unproven claims that need experiment before next build)

### P1. SpiralEncoding on real Qwen3 weights

Claim: `SpiralEncoding::rehydrate_interpolated` hits ρ ≥ 0.95 on real Qwen3-TTS-0.6B weight rows at reasonable K (say K=4–16).

**Probe RUN (PR #186, `spiral_reconstruction_probe.rs`).** Clarification: `SpiralEncoding` is a SIGNATURE codec (17 Base17 dims × K anchor samples per row) not a dense reconstructor, so the probe measures neighborhood preservation instead of per-element ρ.

Measured on `talker.model.layers.0.self_attn.k_proj.weight [1024×1024]`, 256 stride-sampled rows, spiral stride=3:

| K | Top-1 NN | Top-5 NN | Pairwise rank-agree | Bytes/row | Self-cos |
|---|---|---|---|---|---|
| 4 | 18.4% | 39.8% | 0.663 | 142 | 1.000000 |
| 8 | 31.6% | 59.8% | 0.747 | 278 | 1.000000 |
| 16 | **44.9%** | **78.9%** | **0.803** | 550 | 1.000000 |

**Status: PARTIAL — monotonic with K, ~12× better than Base17 palette (#184's 3.71% top-1), but does NOT clear the 90% top-1 / 0.85 rank-agree thresholds at K=16.** Codec is directionally right; quality is K-bound.

Mutation hooks for future probe:
- Larger K (K=32 gives ~1 KB/row — ratio degrades but may cross G2/G3)
- Per-role stride sweep (tested stride=3 for k_proj; other roles have 2/4/5/8)
- Signature + small BF16 residual correction on top (hybrid)
- Different spiral parameter (start ≠ 0) — rows may align better at non-zero start offset

### P2. Shared anchors + i8 position per row

Claim: If anchors are shared across a (component, role, shape) group à la `SharedPaletteGroup`, per-row cost collapses from 142 B to ~1 B.

Probe: NOT YET WRITTEN. Depends on P1 passing first.

Pass → real compression story. Projected 200:1 on vocab tensors at shippable ρ.
Fail → shared anchors lose per-row fidelity; each row needs its own curve calibration.

### P3. Palette preserves inner-product neighbourhoods (Path A → B dependency)

Claim: An f32 CLAM palette on Qwen3 weight rows, used as the substrate for `FisherZTable`, gives `lookup_f32(pal(q), pal(k)) ≈ q · k^T`.

Probe: NOT YET WRITTEN. Successor to `cascade_attention_probe.rs` with f32 palette instead of Base17.

Pass → cascade inference is viable, proceed to pipeline rewire.
Fail → codec-space inference needs richer routing (per-family tables, hierarchical route indices).

### P5. TurboQuant (PolarQuant + JLQ + QJL correction) on real Qwen3 — THE HIGH-PRIORITY PROBE

Claim: per I7, argmax-regime tensors compress correctly via
`TurboQuant = PolarQuant + JLQ + correction` at ~20 B/row with
argmax-parity ≥ 90% over 33-layer chain.

**Probe already written**: `crates/thinking-engine/examples/turboquant_correction_probe.rs`
(466 L). Compares:
  a. Direct i8 (no correction)
  b. Fisher z (arctanh + family gamma — scale correction)
  c. QJL corrected (i8 + bias removal)
  d. RaBitQ corrected (binary + dot_correction)

Across 200-row sample, 33-layer chain simulation, measures Spearman
ranking preservation on final layer output.

**Not yet run on Qwen3-TTS** to the best of this session's knowledge.
That's the single probe that decides whether the whole centroid stack
(A5, A6, A7) is obsolete vs worth keeping.

### P4. Log-radial CLAM with magnitude split

Claim: Unit-normalising rows (direction ∈ sphere) + CLAM on unit sphere + BF16 magnitude separately ≫ linear CLAM on raw f32 rows.

Probe: NOT YET WRITTEN. Would replace `clam_furthest_point_f32` in `hhtl_f32.rs`.

Pass → HhtlF32Tensor ρ̄ improves from 0.2–0.5 to ≥ 0.95 at same k=256.
Fail → direction space is too near-uniform to cluster; needs different factorisation.

## Signposts for future sessions

**Déjà vu triggers** — if a future session is tempted to do any of these,
read the referenced PR first:

| Instinct | Read first |
|---|---|
| "Let's reconstruct rows from Base17 centroids" | #183 — the cos = 0.04 measurement |
| "Hierarchical CLAM will fix the vocab tensor" | #177 → #178, HCLAM got cos = 0.0046, worse than RVQ |
| "Widen twig to u16 for k > 256 centroids" | #185 codex; first probe log-companded bucketing |
| "Base17 palette will preserve attention scoring" | #184 cascade_attention_probe 3.71% agreement |
| "Add more layers of residual" (RVQ-style) | A2 — works for argmax regime only |
| "f32 palette fixes reconstruction entirely" | A6 — 10× better than Base17, still not 0.95 |
| "Single scalar residual (Slot V)" | I3 — can only shift amplitude, cannot add direction |

**Structural checklist before shipping any new codec:**

1. What regime does this tensor belong to? (I1)
2. Does the codec encode direction AND amplitude separately? (I3)
3. Is the palette substrate inner-product-preserving? (I2, A7)
4. Does the decoder evaluate the curve, or tile anchors? (I5)
5. Are wire-format widths asserted at encode time? (I4)

## PR timeline (this session)

| PR | Approach | Gate result |
|---|---|---|
| #176 | AVX-512 F32x16 FMA encoder + AMX polyfill | ✓ SIMD correct |
| #177 | HCLAM 256×256 | ✗ REFUTED for vocab (cos 0.0046) |
| #178 | Passthrough BF16 `n_rows > 8192` + Lance roadmap + WAV test | ✓ token match 225/225 |
| #179 | Compression mindset shifts doc | — (doc) |
| #180 | SlotL foundation (8 × i8 on shared SVD) | ✓ unit tests pass |
| #181 | HhtlDTensor × SlotL integration | ✓ tests pass, integration with centroid flawed |
| #182 | SharedPaletteGroup × SlotL group-level | ✓ tests pass |
| #183 | Universal encoder with Base17 centroid reconstruction | ✗ ρ ≈ 0.04 on real Qwen3 |
| #184 | HhtlF32Tensor + Path A/B probes | ◐ Path A ρ̄ 0.2–0.5 (improves on Base17, short of target); Path B 3.71% (fails) |
| #185 | `HhtlF32Tensor` palette bounds (codex P1) | ✓ safety fix |
| #186 | This doc + SpiralEncoding reconstruction probe | ◐ P1 PARTIAL — K=16 hits 45%/79%/0.80 (top-1/top-5/rank-agree), misses 90%/0.85 gate |

## Session finding: SpiralEncoding is directionally right, K-bound

`SpiralEncoding` at K=16 preserves ~12× more neighborhood structure than the Base17 palette tested in #184, but on Qwen3 k_proj weights it tops out at 45% top-1 NN agreement — short of the 90% threshold for cascade-inference viability.

The monotonic K trend (K=4 → K=8 → K=16 gains 18% → 32% → 45% top-1) suggests K=32 or K=64 could cross the threshold but at 1–2 KB/row — no longer competitive on ratio.

This means the session's forward menu narrows to:
1. **Hybrid codec**: SpiralEncoding signature at small K + a compact residual correction (BF16 scalar or 4-component i8 vector) to lift the NN-top-1 threshold without blowing up per-row bytes.
2. **Per-role stride optimisation**: the probe used stride=3 for k_proj (matches NeuronPrint design); other roles use 2/4/5/8. Sweep.
3. **Accept signature-grade cascade ≠ f32 GEMM reconstruction**: wire SpiralEncoding as the palette substrate for the already-failed `cascade_attention_probe` in #184 and measure there directly. The "45% top-1" result is ON a signature distance, but actual attention scoring might still converge when the cascade routes Skip/Attend/Compose with richer rules than raw argmax.

Next session starts here.

https://claude.ai/code/session_01NYGrxVopyszZYgLBxe4hgj
