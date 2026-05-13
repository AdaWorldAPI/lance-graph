# KNOWLEDGE: Two-Basin Architecture + Representation Routing

## READ BY: truth-architect, savant-research, family-codec-smith,
##          palette-engineer, container-architect, integration-lead

## STATUS: DOCTRINE (two-basin split) + ROUTING TABLE (representation selection)
##         Individual encodings carry their own CONJECTURE/FINDING status.

---

## Two-Basin Doctrine

The architecture factors into two primary basins. Each answers a different question.
Mixing them is the root cause of most design confusion.

### Basin 1: Semantic (WHAT IS IT?)

```
Properties: discrete, stable, addressable, time-invariant
Implemented: CAM + COCA 4096 + scientific vocabulary + DeepNSM
Provides:   identity anchors, stable concept addresses, semantic
            persistence across model drift and time
```

**COCA/CAM is a PERMANENT CODEBOOK.** It is versioned, canonicalized,
and extensible — but the core 4096 vocabulary is fixed. This is not a
learned codebook that retrains. It is a coordinate system.

### Basin 2: Distribution (HOW DOES IT BEHAVE?)

```
Properties: continuous, compressible, dynamic, cheap
Implemented: BGZ family, centroids, wave/perturbation models, signed residuals
Provides:   field shape, activation mass, ranking tendency, perturbation dynamics
```

**BGZ is near-lossless WAVE ENCODING** for distribution-shaped data.
It is not a semantic layer. It does not carry meaning. It carries shape.

**Codebook-cache-only design:** under hard constraints (battery, memory,
bandwidth), you don't store full embeddings — you store codebook indices
and reconstruct from cache-resident palettes. Base17 palette (256 atoms
× 34 bytes = 8.5KB, fits L1 cache) + NeuronPrint (6 bytes per neuron
instead of 204) IS this pattern. Validated at hardware level by
NANOMIND (Li et al. 2025, arXiv:2510.05109): modular LMM inference on
battery-powered SoCs, 42.3% energy reduction via accelerator-mapped
bricks. Same principle: the codebook is cache-resident and tiny, the
indices are the payload, reconstruction is O(1) lookup.

### Possible Third Concern: Projection Consistency

Even if both basins are sound, the architecture fails if projection into
them is inconsistent across tokenizers, models, or time.

```
Risk: same input → different semantic anchors or different distribution fields
      depending on which teacher/tokenizer produced the embedding.
Fix:  cross-model calibration (Jina v5 / ModernBERT / Qwen3 alignment).
Status: HYPOTHESIS — requires benchmark.
```

### Strong Formulation

> Meaning should be discrete and stable.
> Behavior should be continuous and compressible.
> Projection into both must be consistent.

---

## Representation Routing Table

### When to use which encoding

| Question | Basin | Encoding | Why |
|---|---|---|---|
| What concept is this? | Semantic | CAM / COCA codebook | Permanent address, model-independent |
| What's the field shape? | Distribution | BGZ (Base17 i16[17]) | Near-lossless wave, 34 bytes/plane |
| What reinforces/cancels? | Distribution | signed i8/i16 residual | Local contrast, inhibition, polarity |
| Which bucket? | Distribution | CLAM tree path (Slot D) | Hierarchical routing, O(1) |
| Same or different? (pairwise) | Distribution | Pairwise cosine (KEEP PAIRWISE) | Fragile — do not aggregate or average |
| How does perturbation propagate? | Distribution | BGZ-CLAM (distribution + bucket) | Perturbation ≈ field deformation |
| What's the learning curve shape? | Distribution | i16-shaped BGZ vs baselines | Compare: i8, i16, BGZ-HHTL-D, highheelbgz, ZeckF64, CausalEdge64 |
| Is this an anomaly? | Both | CHAODA over CLAM tree | Contradiction detection on bucket paths |
| What's the causal structure? | Both | CausalEdge64 (SPO + NARS truth) | 64-bit atomic: subject/predicate/object + truth values |

### The Pairwise Rule (NON-NEGOTIABLE)

```
Pairwise cosine MUST stay pairwise.

DO NOT:
  - Average pairwise distances into centroid distances
  - Replace pairwise with centroid-to-centroid
  - Assume centroid proximity implies pair proximity

The deep asymmetry: centroids survive compression, pairs do not.
Centroid identity is cheap. Pairwise ranking is fragile.
```

### Encoding Comparison Matrix (for Perturbungslernen)

When testing whether a representation preserves perturbation structure,
compare these encodings on the SAME data:

```
Encoding          Bytes/weight  Basin        Preserves              Loses
────────────────  ────────────  ───────────  ─────────────────────  ──────────────────
i8 direct         1             Distribution  sign, coarse rank      fine precision
i16 direct        2             Distribution  sign, rank, precision  compression
BGZ Base17 i16    2 (34B/17dim) Distribution  wave shape, rank       semantic identity
BGZ-HHTL-D 2×BF16 4            Distribution  wave + bucket routing  storage cost
highheelbgz       varies        Distribution  calibration reference  runtime cost
ZeckF64 u64       8             Distribution  progressive edge       fine local rank
CausalEdge64      8             Both          SPO + NARS truth       continuous field
CAM fingerprint   6 (48-bit)    Semantic      concept identity       behavioral detail
```

For each encoding, measure:
1. Spearman ρ (rank fidelity)
2. Perturbation stability (encode → perturb → decode → compare)
3. Distribution shape (KS test or Pearson on CDF)
4. Pairwise cosine preservation (the fragile one)

### BGZ Must Win In Isolation First

```
RULE: Before building more architecture on top of BGZ, it must be
benchmarked in isolation for:

1. Rank fidelity under compression (Spearman ρ vs i16 baseline)
2. Perturbation stability under encode/decode cycle
3. Distribution-shape preservation (KS statistic)
4. Carrier usefulness: signal ≈ BGZ_carrier + signed_residual
   → residual energy fraction ||r|| / ||x|| should be small

If BGZ does not clearly outperform simpler baselines on distribution
fidelity per cost, it should be restricted or dropped rather than
mythologized.
```

### Audio as Brutal Sanity Test

```
BGZ = carrier (smooth field)
signed i8/i16 = residual (local perturbation)

signal ≈ carrier + signed_residual

Measure: SNR, RMSE, spectral distortion, perceptual artifacts,
         residual energy fraction

WHY: Audio doesn't let semantics or rerankers hide representation errors.
     If BGZ fails as carrier for simple continuous signals, it won't
     succeed as wave substrate for embeddings or activation fields.
```

### Orthogonal Superposition Cleaning (HYPOTHESIS)

```
After BGZ carrier extraction:
  residual r = x - carrier
  Decompose: r = Σ αᵢ · vᵢ  (orthogonal basis: FFT, PCA, or block)
  Filter: keep αᵢ if energy above threshold, aligned with carrier
          gradient, or temporally coherent
  Reconstruct: x_clean = carrier + Σ filtered(αᵢ · vᵢ)

STATUS: HYPOTHESIS. Not implemented, not measured.
The idea: BGZ gives the reference frame, orthogonality separates
deviations into signal vs noise DIRECTIONS, not magnitudes.
```

### Nonlinear Behavior Preservation (the blade)

```
The critical question for the entire stack:

> Does this representation preserve the EFFECT of nonlinearities
> (ReLU, SiLU, softmax), or only the raw values before them?

If it only preserves pre-activation similarity but not what
softmax/SiLU/ReLU actually do to ranking and suppression,
the stack is missing the living part.

Benchmark: compare BGZ-encoded weights against ONNX teacher outputs
for post-ReLU, post-SiLU, post-softmax ranking/mass shape.

STATUS: HYPOTHESIS. This is the experiment that decides whether
the layered representation can preserve nonlinear behavior cheaply.
```

---

## Attribution Discipline

```
DO NOT let BGZ inherit blame for errors caused by:
  - bad semantic anchoring (CAM's job)
  - too-coarse signed quantization (i8/i16's job)
  - poor tokenizer alignment (projection's job)
  - weak reconstruction of SiLU/softmax behavior (activation layer's job)

DO NOT let semantic CAM get credit for behavior actually carried by:
  - signed residuals
  - reranker correction
  - ONNX calibration
```

Each layer is responsible for its own invariant. Test each in isolation
BEFORE testing the stack.
