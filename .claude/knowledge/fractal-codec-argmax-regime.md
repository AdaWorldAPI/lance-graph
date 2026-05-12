# Fractal-Codec / Argmax-Regime (Codec Research Thread)

> **READ BY:** agents working on `bgz-tensor`, `bgz17`, `ndarray/hpc`
> codec work, `cam_pq`, `clam` tree, CODEC_INVARIANTS, TurboQuant /
> PolarQuant / JLQ research, or the k_proj / q_proj / gate_proj
> compression benchmarks.
>
> **Scope note:** this is a **separate research thread** from the
> grammar / Markov / crystal / quantum work in the rest of
> `.claude/knowledge/`. Captured here so the insight doesn't get
> lost between sessions that focus on the grammar side.
>
> **Source:** conversation input 2026-04-19, referencing
> `docs/CODEC_INVARIANTS_AND_EXPERIMENTS.md` in this repo.

---

## The Argmax Wall (what's documented)

Invariants I1, I2, I7 from `CODEC_INVARIANTS_AND_EXPERIMENTS.md`:

**I1 — Two regimes with opposite needs:**

| Regime | Where | Needs | Error tolerance |
|---|---|---|---|
| Argmax-decoded | attention, MLP, logits | top-1 argmax stability under `hidden @ W.T` | cos ≈ 0.95 fine |
| Index-lookup | text_embedding, lm_head, code_embed | per-row identity | cascading; no rescue |

**I2 — Near-orthogonality at high dim.** Qwen3 1024-D and 2048-D
weight rows are near-orthogonal pairwise. Any compression assuming
L2-clustering is wrong at its foundation. This refuted HCLAM
256×256 (cos = 0.0046 on vocab) and single-centroid + scalar
residual (cos = 0.04 on real Qwen3).

**I7 — Vector-as-location vs vector-as-sparse-signal:**

- **Vector-as-location** (Cartesian, L2 distance). Raw f32 weight
  rows, each dim independent. Centroid + residual / tree-quant /
  palette lookup. **On near-orthogonal high-dim rows these ALL fail.**
- **Vector-as-sparse-signal** (phase + magnitude on orthogonal basis).
  Project onto orthogonal basis (JL / SVD / Hadamard). Encode as
  (sign, log-magnitude) per projected coefficient = **PolarQuant**.
  Inner products preserved by Lindenstrauss concentration.

**If the leaf is orthogonal (JL / Hadamard / PolarQuant), you do NOT
need centroid + residual.** They solve different problems.

**Doc conclusion:** `TurboQuant = PolarQuant + JLQ + correction` is
the canonical argmax-regime codec. The centroid+residual work
(A5, A6, A7) was solving the wrong problem.

---

## The Fractal-Descriptor Leaf Proposal

The proposal: **the fractal structure of the residual on an
orthogonal basis is itself compressible** in a way neither
PolarQuant nor the current LFD-gated cascade exploits.

### Step 1 — near-orthogonal ≠ uniformly-distributed-on-sphere

Trained weights after SVD/Hadamard rotation have **heavy tails and
self-similarity across scales**. Documented in intrinsic-dimension
literature (Ansuini NeurIPS 2019; Pope ICLR 2021; Valeriani NeurIPS
2023; Kataiwa 2025). Trained weights concentrate on a low-ID
manifold of ID ≈ 8–100 even at ambient 1024–8192.

**The projected coefficient sequence is fractal, not flat.**

### Step 2 — orthogonal fractal decomposition as a codec

Encode the row as a **self-similar object on an orthogonal basis**
via its fractal parameters, not its coefficient values. In the I8
layered frame:

```
HEEL (2 bit)   BASIN     coarse location (Q/K, V, Gate, FFN)  existing
HIP  (4 bit)   FAMILY    fine location within basin           existing
LEAF (fractal) ???       scale-invariant shape of residual    THE MOVE
```

**LEAF = (fractal dimension D, multifractal width w, scale factor
σ, sign pattern S)** — a fractal descriptor leaf. At decode time
reconstructs a *signed self-similar coefficient sequence* of known
D, w, σ on the basis — statistically equivalent to the original
row for argmax.

### Why this might work where centroid+residual fails

- Centroid+residual fails because residual is high-dim
  near-orthogonal (I2).
- PolarQuant/TurboQuant works for argmax because JL preserves inner
  products; spends ~8–32 bits per projected coefficient.
- Fractal descriptor spends ~4–8 bytes total on *self-similar shape*
  of the projected coefficient sequence. Argmax depends on inner
  product, which for a fractal shape on a fixed basis is determined
  by (D, w, σ, S).

### Why "orthogonal" is doing real work

LFD as currently computed in `bgz17::generative.rs` is measured in
the **raw feature space** (`count_r / count_half_r` around a cluster
center in Cartesian metric). Captures density, not row-self-
similarity.

The move is: **fractal decomposition of the row's coefficients *after*
projection onto an orthogonal basis.** In that basis:
- Coefficients are approximately decorrelated.
- Heavy-tail behavior is visible (hidden by correlations pre-rotation).
- MFDFA-style analysis on the coefficient sequence is well-defined.
- D, w are intrinsic to the row — invariant under orthogonal linear
  transforms for long sequences (real theorem: multifractal spectra
  preserved under Hadamard rotations).

**Orthogonal fractal decomposition = MFDFA (or PH-dim) on the
Hadamard-rotated coefficient sequence.**

---

## Concrete Wire Format (9 B/row, fits existing I8 budget)

```
HEEL (2 bit)      existing BASIN (Q/K, V, Gate, FFN)
HIP  (4 bit)      existing family within basin
POLARITY (2 bit)  sign anchor (existing, 1 bit polarity + 1 reserved)

FRACTAL DESCRIPTOR (7 bytes):
  D_local  (u16, q0.16)  fractal dimension of coefficient sequence
  w_mfs    (u16, q0.16)  multifractal spectrum width α_max − α_min
  σ_energy (BF16)        total L2 energy
  H_hurst  (u8, q0.8)    Hurst exponent (short-range memory)

= 56 bits descriptor + 8 bits address = 64 bits = 8 bytes/row
```

Hadamard basis shared per (component, role, shape) group — same
sharing mechanism BGZ-HHTL-D already uses. No per-row basis storage.

---

## Decoder (Statistical Twin, Argmax-Sufficient)

Can't reconstruct exact row from (D, w, σ, H) — only generate a
**statistical twin** with correct multi-scale structure. For
**argmax-regime** inference this suffices because argmax needs:

- Correct L2 norm (σ preserves exactly).
- Correct dominant-direction alignment (HEEL basin preserves).
- Correct fine-grained ranking (D + w determine inner-product distribution).

For **index-regime** tensors (vocab embeddings, lm_head) this is
NOT sufficient — per-row identity required. I1 rules. Those stay
on passthrough or SpiralEncoding.

Cascade attention probe (PR #184): Base17 palette got 3.71 % top-1
agreement (palette doesn't preserve inner-product neighborhoods). A
fractal descriptor **does** preserve inner-product neighborhoods at
the argmax-relevant granularity, because multi-scale correlation
structure determines inner-product distributions under random
queries.

---

## Cardiac-Trabeculae Analogy (Meyer Nature 2020, McGurk NatCardio 2024)

Trabecular FD predicts cardiovascular disease outcomes at
Mendelian-randomization strength. Analogous prediction:

> Two Qwen3 weight rows with same (HEEL, HIP, D, w, σ) produce the
> same argmax rankings against random queries within the FD's
> confidence bound.

Empirically testable. Add `FractalDescriptorLeaf` to
`codec_rnd_bench.rs` as a candidate. Run against k_proj / q_proj /
gate_proj populations at 9 B/row. Compare vs I8-Hadamard (current
leader at 9 B) and adaptive codec.

**Thresholds:**
- ≥ 95 % argmax-top-1 at 9 B/row on k_proj → argmax wall cracked.
- 70–90 % → useful hybrid layer.
- < 70 % → self-similar-row hypothesis wrong for Qwen3 (valid
  negative result).

---

## Difference from PR #200 LFD-Adaptive (the delta)

- PR #200 `AdaptiveCodecTensor` uses LFD as a **routing signal** —
  "this row is hard, bump precision." Row still encoded by BF16
  passthrough or i8.
- Fractal-leaf uses fractal parameters as the **encoding** itself.
  Row's self-similar shape becomes the data; exact values discarded.

Also differs from Holographic residual (majority-vote collapse,
loses multi-scale structure). Fractal leaf preserves scaling
*without* preserving bits.

---

## Roadmap (Research Thread)

1. **Cheap probe first** (30 min): compute multifractal spectrum
   width `w` on 100 rows of k_proj and 100 rows of gate_proj. If
   `w` is row-dependent with CoV > 0.3, fractal leaf has signal.
   If w is nearly constant across rows, it reduces to storing σ
   alone (= log-magnitude-per-row, which is already known).
2. **Probe passes** → add `compute_mfdfa_descriptor(row, basis)` to
   `bgz-tensor/src/`. ~200 LOC. Reuse existing `wht_f32` from
   `ndarray::hpc::fft`.
3. Add `FractalDescriptorLeaf` candidate to `codec_rnd_bench.rs`.
   Measure against Qwen3-TTS and Gemma 4 populations (67-codec
   sweep framework from PR #198 is free for this).
4. If works on k_proj → wire into `AdaptiveCodecTensor` as a
   fourth `RowPrecision` variant: `Fractal(FractalDescriptor)` for
   the p70–p90 band, passthrough stays for p90+.
5. **Validation:** `cascade_attention_probe` variant with
   fractal-descriptor palette. If inner-product neighborhoods
   preserved, unlocks codec-space attention inference (Path A/B
   dependency blocked since PR #184).
6. **External benchmark:** UK Biobank cardiac MRI with Meyer
   pipeline FD values as ground truth. If fractal-descriptor leaf
   preserves clinical FD signal at 343:1 compression → "Da Vinci
   trabeculae → Qwen3 argmax" paper.

---

## Honest Uncertainty

Cannot tell from reading code whether Qwen3 weight rows after
Hadamard rotation actually exhibit MFDFA-measurable self-similarity.
Intrinsic-dimension literature (Ansuini, Pope, Valeriani, Kataiwa)
strongly predicts they do — trained weights live on low-ID
fractal-like manifolds. `test_sweep_fractal_invariant` in
`lance-graph-codec-research` looks at a related question but not
this exact one.

**The 30-minute probe is the cheap gating test.**

---

## Why This Is Here (Not in Grammar Knowledge)

This is an **orthogonal research thread** to the grammar / Markov /
crystal / quantum work the other `.claude/knowledge/` docs describe.
Captured here to prevent inter-session context loss. Do not entangle
with grammar work; they share no overlapping code paths.

Cross-reference:
- Grammar thread — see `grammar-tiered-routing.md`,
  `linguistic-epiphanies-2026-04-19.md`, `integration-plan-grammar-
  crystal-arigraph.md`, `endgame-holographic-agi.md`.
- Codec thread — this doc, plus `CODEC_INVARIANTS_AND_EXPERIMENTS.md`,
  `encoding-ecosystem.md`.

Both threads converge ONLY at the 10K-VSA substrate (Binary16K
fingerprint, CAM-PQ codec, Base17). Below that level they diverge.
