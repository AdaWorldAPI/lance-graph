# KNOWLEDGE: Encoding Ecosystem — Read This Before Touching Any Codec

## READ BY: ALL AGENTS. MANDATORY before any codec, encoding, distance,
##          compression, or representation work.

## P0 RULE: Never assume architecture or write code without reading actual
##          source first. Guessed implementations have caused real damage
##          in production files. This document is your map. The source is
##          your truth.

---

## The Encoding Landscape (what exists and where)

There are 8+ encoding representations in lance-graph. They are NOT
competing alternatives — they serve different purposes at different
layers. Before proposing changes to ANY of them, understand what each
one IS and what invariant it protects.

```
Encoding         Crate/File                          Bytes   Basin         Invariant Protected
─────────────    ──────────────────────────────────   ─────   ───────────   ─────────────────────────
Base17 i16[17]   bgz17/src/base17.rs                 34      Distribution  Wave shape (golden-step
                                                                           octave fold, 11/17 traversal)

BGZ17 palette    bgz17/src/palette.rs                varies  Distribution  Palette-indexed wave
                 bgz17/src/palette_semiring.rs                             (semiring algebra over
                 bgz17/src/palette_matrix.rs                               Base17 atoms)

StackedN BF16    bgz-tensor/src/stacked.rs           varies  Distribution  BF16 centroid stacking
                 bgz-tensor/src/codebook_calibrated                        (calibrated codebook,
                                                                           ONNX teacher reference)

HighHeelBGZ      lance-graph-contract/src/high_heel   ~32+    Distribution  Progressive cascade
                 bgz17/src/clam_bridge.rs                                  (HEEL/HIP/TWIG/LEAF
                                                                           scent→magnitude→family→exact)

ZeckF64 u64      lance-graph/src/graph/neighborhood   8       Distribution  Progressive 8-byte edge
                 /zeckf64.rs                                               (scent byte + 7 quantile
                                                                           bytes, L1 distance)

CausalEdge64     causal-edge/src/edge.rs              8       Both          64-bit atomic: SPO palette
                                                                           indices + NARS truth values
                                                                           + Pearl causal hierarchy

NeuronPrint 6D   bgz-tensor/src/euler_fold.rs         6       Distribution  6 roles → 6 palette indices
                 bgz-tensor/src/neuron_hetero.rs                           (Q/K/V/Gate/Up/Down folded)

γ+φ transform    bgz-tensor/src/gamma_phi.rs          0       Distribution  DEAD as post-rank monotone.
                                                              (no-op)      ALIVE only as pre-rank
                                                                           discrete selector.

BGZ-HHTL-D       PROPOSED (not implemented)           32      Distribution  Slot D (CLAM tree path)
2×BF16                                                                     + Slot V (BF16 value),
                                                                           branching cascade

Thinking Engine  lance-graph-planner/src/thinking/    —       Meta          12 styles, NARS dispatch,
                                                                           sigma chain (Ω→Δ→Φ→Θ→Λ)
```

## How They Connect (the pipeline, not a flat list)

```
Raw weights (BF16/f32 from ONNX/GGUF safetensors)
  │
  ├──→ StackedN BF16 (centroid stacking, calibrated codebook)
  │       │
  │       ├──→ Base17 i16[17] (golden-step fold into 17 dims, 34 bytes)
  │       │       │
  │       │       ├──→ BGZ17 palette (palette-indexed Base17 atoms)
  │       │       │       │
  │       │       │       └──→ HighHeelBGZ (progressive cascade over palette)
  │       │       │               │
  │       │       │               └──→ ZeckF64 u64 (8-byte progressive edge)
  │       │       │
  │       │       └──→ NeuronPrint 6D (6 roles → 6 palette indices)
  │       │
  │       └──→ BGZ-HHTL-D 2×BF16 (PROPOSED: Slot D + Slot V, branching)
  │
  └──→ CausalEdge64 (SPO triples with NARS truth, orthogonal to above)

Semantic layer (ORTHOGONAL — different basin):
  CAM fingerprint (48-bit) → COCA 4096 codebook → DeepNSM addressing
  This is the PERMANENT CODEBOOK. It does not retrain.
  It answers WHAT IS IT, not HOW DOES IT BEHAVE.
```

## What Each Encoding Has Proven (FINDING vs CONJECTURE)

### Base17 — FINDING
```
Source: bgz17/src/base17.rs, bgz17/src/lib.rs
Golden step = 11, gcd(11,17) = 1 → visits all 17 positions. PROVEN.
|17/φ − 11| = 0.4934 → nearest integer to 17/φ. PROVEN.
Fibonacci mod 17 is BROKEN (misses {6,7,10,11}). PROVEN.
i16 fixed-point (×256): 256× finer than BF16. PROVEN.
34 bytes per plane (341:1 from i8[16384]). PROVEN.
121 tests pass in bgz17 crate. PROVEN.
```

### ZeckF64 — FINDING
```
Source: lance-graph/src/graph/neighborhood/zeckf64.rs
8-byte progressive: scent (byte 0) + 7 quantile bytes. PROVEN.
Only 19 of 128 scent patterns are legal. PROVEN.
L1 distance on integer types (no floating-point). PROVEN.
Byte layout is IMMUTABLE (production contract). PROVEN.
ρ = 0.937 for scent byte only (1 byte). MEASURED.
```

### HighHeelBGZ — FINDING (structure), CONJECTURE (cascade rates)
```
Source: lance-graph-contract/src/high_heel.rs
HEEL/HIP/TWIG/LEAF cascade structure. IMPLEMENTED.
SpoBase17: S/P/O as Base17 triples. IMPLEMENTED.
BasinAccumulator: streaming accumulation. IMPLEMENTED.
LensProfile/LensConfig: model-specific sensor. IMPLEMENTED.
95% HEEL/HIP termination claim. CONJECTURE — Probe M4 NOT RUN.
```

### γ+φ — FINDING (dead in current position)
```
Source: bgz-tensor/src/gamma_phi.rs
Post-rank monotone transform: ρ = 1.000 vs CDF. PROVEN DEAD.
Pre-rank discrete selector: theoretically non-trivial. CONJECTURE.
Probe I (4 γ-phase offsets → different ranked output) NOT RUN.
DO NOT move or modify until Probe I runs.
```

### Certified Lane Verdicts (v2.4, 256-centroid Jina v5) — FINDING
```
Source: bgz-tensor/examples/calibrate_from_jina.rs (certify v2.4)
Data: 256 Jina v5 centroids, 32,640 pairwise cosines, Fisher 3σ CIs

Lane 1 u8 CDF:              ρ = 0.999992  [0.999992, 0.999993]  PASS
Lane 2 i8 direct:            r = 0.999250  [0.999225, 0.999274]  PASS
Lane 3 u8 γ+φ:              ρ = 0.999992  [0.999992, 0.999993]  PASS
Lane 4 i8 γ+φ signed:       ρ = 0.999463  [0.999445, 0.999480]  PASS
Lane 6 BF16 RNE:             r = 0.999978  [0.999978, 0.999979]  PASS

Reality anchors (7 named pairs at p0–p100):
  max |Lane 6 − ref| = 8.76e-4 (~0.44 BF16 ULPs). 7/7 PASS.

Corrections locked in (v2.4, non-negotiable):
  k_3sigma = 3.000000 exactly (not 2.967736)
  Lane 2 jackknife 21× wider than Fisher (not "~3×")
  GammaProfile: 36 bytes, role_gamma[8] (Embed/Q/K/V/O/Gate/Up/Down)
  u8→BF16 speedup: ~100-300× cache-dependent (not "1000×")
  Spearman Fisher SE underestimates by sqrt(1.06) ≈ 3% (acceptable at 4-dec)

CRITICAL: Lane 3 (γ+φ u8) = Lane 1 (u8 CDF) at ρ=0.999992.
  γ+φ as post-rank monotone is a PERFECT NO-OP on rank.
  γ+φ can ONLY carry information as PRE-RANK selector.
```

### Certified v2.5 Additions (PR #158) — FINDING
```
Source: bgz-tensor/examples/calibrate_from_jina.rs (certify v2.5)

Step 11e — Efron BCa bootstrap (B=2000, Efron 1987):
  L2 Pearson: BCa 60% wider than Fisher (ratio 1.616) — expected,
    confirms Lane 2 pair residuals are non-Gaussian.
  L4 Spearman: BCa/Fisher ratio 1.050 — agreement.
  L1/L3 Spearman: BCa saturated (ρ too close to 1.0 for B=2000).
  Fisher z remains 3σ authority (B=2000 undersamples 0.135% tail).

Step 11f — CHAODA outlier filter (Ishaq et al. 2021):
  ndarray::hpc::clam::ClamTree::anomaly_scores on 256 Lane-1 rows.
  Top-10% count-based flagging: 26 of 256 centroids flagged.
  26,335 of 32,640 pairs kept after filtering.
  L1/L3/L4/L6: |Δ| < 1e-4 (clean — outlier removal doesn't change ρ).
  L2 Pearson: filter_removed_easy_pairs (Pearson tail sensitivity).
  THIS IS THE FIRST MEASURED CLAM/CHAODA PROBE ON REAL DATA.

Step 11g — Naive u8 ULP floor (BGZ-adjacent baseline):
  Naive u8 quantize: Pearson 0.999860, Spearman 0.999749.
  γ+φ+CDF benefit over naive: +0.000244 Spearman (real but small).
  ENDGAME: (lane_pearson − naive_u8_pearson) = cascade entry tax budget.
  bgz-hhtl-d gate threshold: ≥ 0.9980 Pearson to justify existence.
  Any encoding below naive u8 floor is WORSE THAN DOING NOTHING.
```

### NeuronPrint 6D — FINDING (compression), CONJECTURE (fidelity)
```
Source: bgz-tensor/src/euler_fold.rs, neuron_hetero.rs
204 bytes (6×34) → 6 bytes (6×u8 palette index). PROVEN.
17× compression per neuron. PROVEN.
Fidelity of palette-indexed reconstruction. CONJECTURE — needs Spearman ρ.
```

### CausalEdge64 — FINDING (encoding), CONJECTURE (causal utility)
```
Source: causal-edge/src/edge.rs
64-bit atomic: SPO palette indices + NARS truth + Pearl 2³ mask. PROVEN.
All in one CPU register. PROVEN.
Causal hierarchy filtering utility. CONJECTURE.
```

### BGZ-HHTL-D 2×BF16 — CONJECTURE (entire proposal)
```
Source: .claude/knowledge/bf16-hhtl-terrain.md
Slot D = CLAM tree path (12-bit hierarchical bucket). CONJECTURE.
Slot V = full BF16 value. CONJECTURE.
Branching cascade (not prefix decoding). DESIGN DECISION.
16→256→4096 alignment with Jina/COCA. CONJECTURE — Probe M1 NOT RUN.
Bucketing > resolution (arxiv consensus). EXTERNAL FINDING, not measured locally.
```

### Thinking Engine — FINDING (structure), CONJECTURE (integration)
```
Source: lance-graph-planner/src/thinking/
12 thinking styles, NARS dispatch. IMPLEMENTED.
Sigma chain Ω→Δ→Φ→Θ→Λ. IMPLEMENTED.
MUL (Meta-Uncertainty Layer). IMPLEMENTED.
Integration with codec pipeline. CONJECTURE.
```

## The Synergies (what connects to what)

### BGZ17 ↔ HighHeelBGZ
```
Base17 atoms are the LEAF level of HighHeelBGZ.
HEEL scent byte IS ZeckF64 byte 0.
The palette in BGZ17 feeds the cascade in HighHeelBGZ.
DO NOT redesign one without checking the other.
```

### BGZ17 ↔ ZeckF64
```
zeckf64_from_base() converts Base17 → ZeckF64 u64.
The conversion preserves scent (byte 0) and quantile structure (bytes 1-7).
Base17 is the ENCODING. ZeckF64 is the DISTANCE METRIC.
```

### StackedN ↔ NeuronPrint ↔ Base17
```
StackedN is the BF16 intermediate.
NeuronPrint folds 6 StackedN roles into 6 palette indices.
Each palette index points to a Base17 atom.
```

### CausalEdge64 ↔ Everything
```
CausalEdge64 is ORTHOGONAL to the wave encodings.
It encodes RELATIONSHIPS (SPO), not VALUES (distributions).
It connects to Base17 via palette indices in the S/P/O fields.
```

### BGZ-HHTL-D ↔ CLAM ↔ COCA
```
BGZ-HHTL-D PROPOSES that Slot D is a CLAM tree path.
CLAM tree over 256 Jina centroids would give 16→256→4096 branching.
4096 terminal buckets are conjectured to align with COCA vocabulary.
NONE of this is measured. All depends on Probe M1.
```

### Thinking Engine ↔ Codec Selection
```
Different thinking styles may want different encoding paths.
Analytical → exact (LEAF, full BF16, pairwise cosine)
Creative → approximate (HEEL, scent-level, basin routing)
This mapping is CONJECTURE — no benchmark exists.
```

## Before You Touch Any Encoding

### MANDATORY checklist

```
[ ] I have READ the source file for this encoding (not just docs)
[ ] I know which BASIN this encoding serves (semantic vs distribution)
[ ] I know which INVARIANT this encoding protects
[ ] I know its status: FINDING (measured) or CONJECTURE (proposed)
[ ] I know what CONNECTS to it (upstream producers, downstream consumers)
[ ] I have checked the probe queue in bf16-hhtl-terrain.md
[ ] If my change affects pairwise cosine: I understand the pairwise rule
[ ] If my change affects γ+φ: I know which regime (pre-rank vs post-rank)
[ ] If my change affects golden-step: I will NOT reintroduce Fibonacci mod 17
[ ] If my change adds a layer: I have a probe with pass/fail criteria
```

### The search for insight

When investigating encoding behavior, the correct method is:

```
1. READ the source (P0 rule — never guess architecture)
2. IDENTIFY the invariant this encoding is supposed to protect
3. MEASURE the invariant on real data (not synthetic, not toy)
4. COMPARE against baselines in the Pareto frontier:
   | Encoding        | Bytes | ρ      | Status  |
   | Scent byte      | 1     | 0.937  | PROVEN  |
   | ZeckBF17 plane  | 48    | ?      | MEASURE |
   | ZeckBF17 edge   | 116   | ?      | MEASURE |
   | BitVec 16Kbit   | 2048  | 0.834  | PROVEN  |
   | Full planes     | 6144  | 1.000  | DEFINITION |
5. REPORT the number with its context (which data, which metric)
6. THEN propose changes — not before
```

### What "insight" means here

An insight is a MEASURED relationship between encodings that was not
previously known. Examples:

- "Base17 at stride=4 gives ρ=X on Jina v5 layer 12 Gate weights"
- "NeuronPrint palette index 47 clusters with COCA word 'function'"
- "HEEL terminates 73% of queries on reranker v3 cross-encoder data"

An insight is NOT:
- "BGZ should be able to..."
- "It would be elegant if..."
- "The mathematical proof suggests..."

Lock in truths. Measure conjectures. Label everything.

## The Mathematical Proof (scope)

The φ-spiral reconstruction proof (`.claude/knowledge/zeckendorf-spiral-proof.md`)
provides a theoretical bound ρ ≥ 1 − C·ρ_c²/(N²R²). It is:

- VALID for continuous golden-angle sampling (large stride regime)
- VACUOUS at ZeckF8 (3 orders of magnitude loose)
- SILENT on Fibonacci mod 17, NeuronPrint, γ+φ, HHTL cascade
- USEFUL for confirming that 11/17 is the unique correct discrete step
- NOT USEFUL for justifying any specific encoding's production fidelity

Use it as a theoretical certificate for why φ appears in the architecture.
Do not use it as a substitute for running a probe.
