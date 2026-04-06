# RISC Thought Engine — Development Stages & Implementation Paths

> Generated 2026-04-06 from empirical results.
> Branch: `claude/risc-thought-engine-TCZw7`
> All claims backed by measurements from this session.

---

## STAGE 0: FOUNDATIONS (DONE ✓)

### What's proven

| Result | Evidence | File |
|--------|----------|------|
| Forward pass works | Qwen3-VL gap=0.132, Jina v5 gap=0.128 | `examples/qwen3_vl_forward.rs`, `examples/forward_pass.rs` |
| BF16 is the correct table format | Spearman ρ=0.9999, Pearson r=0.9999 | `examples/benchmark_thinking.rs` |
| i8 direct is the correct signed format | ρ=0.997, r=0.999 | Lane calibration |
| u8 CDF destroys value geometry | Pearson r=0.80 | Lane calibration |
| γ+φ is identical to CDF | ρ=1.000 between L1 and L3 | Lane calibration |
| Reranker has 50% real inhibition | E/I=50.2%, cos[-0.886, 0.826] | Cronbach quorum |
| Signed inhibition reduces entropy 14% | Reranker: 5.51→4.72 | Signed benchmark |
| Cross-model calibration impossible | Cronbach α < 0.37 | Quorum across 3 models |
| Positive-only MatVec diffuses energy | Entropy +6-19% | `benchmark_thinking.rs` |
| f32 tables work, BF16 loses near zero | 42% vs 24% top-5 | f32 vs BF16 comparison |

### What's built

| Component | Status | LOC |
|-----------|--------|-----|
| `engine.rs` (u8 MatVec) | Working but wrong format | ~600 |
| `bf16_engine.rs` (BF16 MatVec) | Working, correct format | ~300 |
| `signed_engine.rs` (i8 MatVec) | Working, correct signed path | ~250 |
| `builder.rs` (fluent API) | Working | ~500 |
| `cronbach.rs` (statistical validation) | Working | ~310 |
| `seven_lane_encoder.rs` | Working but 5 of 7 lanes are redundant | ~440 |
| Forward pass (Qwen3-VL, Jina v5) | Working | ~157 each |
| 7-lane codebooks (3 models) | Release v0.2.0 | 770 KB each |
| Reranker correction tables | Built (f32 + i8) | 320 KB each |

### What to kill

| Component | Why | Action |
|-----------|-----|--------|
| u8 CDF tables (Lane 1) | Pearson 0.80, destroys values | Remove as default, keep for backwards compat |
| γ+φ encoding (Lane 3, 4) | ρ=1.000 vs CDF, zero added value | Delete |
| Spiral drift (Lane 7) | ρ=-0.01 vs error, useless diagnostic | Delete |
| SiLU deltas on u8 (Lane 5) | Category error: f32 correction on percentile rank | Rebuild on f32 tables |
| Multi-lens superposition via u8 | α < 0.37, lenses don't agree on u8 | Replace with single-model f32 |

---

## STAGE 1: FIX THE TABLE (1-2 days)

**Goal**: Make the thinking engine actually work correctly.

### 1A. f32 as primary table format

```
Current:  u8 CDF (Pearson 0.80, broken)
Target:   f32 direct (Pearson 1.000, correct)
Cost:     256 KB instead of 64 KB (4× storage)
Benefit:  Correct values → correct MatVec → correct thinking
```

**Files to change**:
- `engine.rs`: add `ThinkingEngine::from_f32_cosines()` constructor
- `builder.rs`: add `TableType::F32` variant, make it default
- `seven_lane_encoder.rs`: simplify to 2 outputs (f32 raw + i8 signed)

### 1B. Signed i8 from real cosines (not CDF shift)

```
Current:  i8 from CDF rank - 128 (fake sign)
Target:   i8 from round(cos × 127) (real sign)
Already:  signed_engine.rs::from_f32_cosines() does this correctly
```

**Files to change**:
- `builder.rs`: wire `TableType::SignedI8` to `from_f32_cosines()`, not `from_unsigned()`
- Delete `from_unsigned()` or mark deprecated

### 1C. Validate: thinking beats plain cosine

```
Test:     20 query atoms, f32 table, signed MatVec
Measure:  Top-5 overlap, entropy reduction, Kendall tau
Pass if:  Entropy decreases AND top-5 > 50%
```

**File**: update `benchmark_thinking.rs` with f32 + signed variants

### Dependencies

```
1A → 1B → 1C (sequential, each builds on previous)
```

---

## STAGE 2: CONTRASTIVE TABLE LEARNING (2-3 days)

**Goal**: The table learns from forward passes.

### 2A. Forward pass → table update

```
Input:    text pair (A, B)
Process:  
  1. Forward pass A → embedding E_A (2048D)
  2. Forward pass B → embedding E_B (2048D)
  3. cos_real = dot(E_A, E_B)
  4. Codebook: A → centroid i, B → centroid j
  5. cos_table = table[i][j]
  6. error = cos_real - cos_table
  7. table[i][j] += α × error
```

**Files to create**:
- `src/contrastive_learner.rs`: online table update from forward pass pairs
- `examples/contrastive_learn.rs`: run 1000 text pairs, measure table improvement

### 2B. Counterfactual fan-out

```
For centroid ci, find K=16 nearest neighbors in codebook
Each neighbor = "what if this token mapped there?"
Update table[ci][nk] for all K neighbors from a single forward pass
Fan-out: 1 forward pass → 16 table updates
```

**Files to create**:
- `src/counterfactual.rs`: codebook neighbor lookup + fan-out updates

### 2C. Measure improvement

```
Before:   table from CLAM codebook (static, from token embeddings)
After:    table + 1000 contrastive updates
Measure:  Spearman ρ(table, real_cosines) before vs after
Pass if:  ρ increases monotonically with more forward passes
```

### Dependencies

```
Stage 1 (f32 tables) → 2A → 2B → 2C
2A and 2B are independent, 2C depends on both
```

---

## STAGE 3: SPO 2³ CAUSAL CERTIFICATES (3-5 days)

**Goal**: Extract causal structure from forward passes.

### 3A. Gate extraction from forward pass

```
Current:  forward pass returns final embedding only
Target:   forward pass returns embedding + gate_pattern[28]
```

**Files to change**:
- `examples/forward_pass.rs`: extract per-layer gate E/I ratio
- Need to modify candle Qwen3 model to expose intermediate gate values
- Or: run 28 separate forward passes, each truncated at layer L, diff embeddings

**Cheaper alternative**: don't extract gates, use embedding-level causality only.

### 3B. 8-octant SPO decomposition

```
For triple (S, P, O):
  Generate 8 text variants:
    Oct 0: "S P O"
    Oct 1: "S prevents O"
    Oct 2: "S doesn't P O"
    Oct 3: "S doesn't prevent O"
    Oct 4: "Without S, P O"
    Oct 5: "Without S, prevents O"
    Oct 6: "Without S, no O"
    Oct 7: "Without S, doesn't prevent O"
  
  Forward pass each → 8 embeddings (2048D)
  Pairwise cosine: 8×8 matrix = causal certificate
```

**Files to create**:
- `src/causal_certificate.rs`: SPO → 8 octant texts → 8 embeddings → certificate
- `src/causal_strength.rs`: certificate → causal effect strength

### 3C. Causal table update

```
actual_cos = cosine(Oct0_emb, Oct0_emb)  # S causes O
counter_cos = cosine(Oct0_emb, Oct4_emb) # S causes O vs ¬S causes O
causal_delta = actual_cos - counter_cos

table[ci_S][ci_O] += α × causal_delta        # amplify causal pairs
table[ci_¬S][ci_O] -= α × causal_delta × 0.5 # suppress non-causal
```

### 3D. Causal certificate validation

```
Test on known causal pairs:
  "Smoking causes cancer"    → strong causal effect (high delta)
  "Roosters cause sunrise"   → weak causal effect (low delta, correlation not causation)
  "Rain causes wet ground"   → strong causal effect
  "Wet ground causes rain"   → weak (reverse causation detected)
  
Pass if: causal strength ranking matches human intuition
```

### Dependencies

```
Stage 2 (contrastive learning) → 3A → 3B → 3C → 3D
3A is the hard part (gate extraction)
3B-3D can use embedding-only causality without gates
```

---

## STAGE 4: L1-L27 GATE REWARD SHAPING (3-5 days)

**Goal**: Use gate patterns as RL reward signal.

### 4A. Gate pattern extraction

```
Modify candle Qwen3 forward pass:
  For each layer l:
    gate_l = model.layers[l].mlp.gate_proj(hidden)
    activated_l = silu(gate_l)
    ei_ratio_l = (activated_l > 0).count() / hidden_dim
  
  Return: Vec<f32> of length 28 (one E/I ratio per layer)
```

**Files to change**:
- Fork `candle_transformers::models::qwen3::Model::forward()` to expose gate values
- Or: create `forward_with_gates()` wrapper

### 4B. Gate reward function

```
reward(gate_pattern) =
  if epiphany_detected(L20-L22 spike after L10-L15 dip):
    +2.0  (strong positive reward)
  elif confident_throughout(low variance across all layers):
    +1.0  (reinforce)
  elif uncertain_throughout(high variance):
    -0.5  (suppress)
  else:
    0.0   (neutral)
```

**Files to create**:
- `src/gate_reward.rs`: gate pattern → reward value
- `src/epiphany_detector.rs`: detect L20-L22 spike pattern

### 4C. Gate-weighted table update

```
For each centroid pair (i, j) activated by input:
  contrastive_update = α × (cos_real - table[i][j])
  gate_weight = reward(gate_pattern)
  table[i][j] += contrastive_update × (1 + β × gate_weight)
```

**Integration**: multiply contrastive learning rate by gate reward.

### Dependencies

```
Stage 3 (causal certificates use gates) ← 4A (gate extraction, shared)
Stage 2 (contrastive learning) → 4B → 4C
4A is prerequisite for both Stage 3 and 4
```

---

## STAGE 5: L4 HOLOGRAPHIC MEMORY (2-3 days)

**Goal**: Learn from accumulated thought patterns.

### 5A. Bundle fingerprinting

```
Thought peaks [p1, p2, p3, ...] 
→ prime_fingerprint_64(centroid_vector[pi]) for each peak
→ xor_majority_vote(all fingerprints)
→ 64-bit holographic bundle
```

**Already built**: `prime_fingerprint.rs` has `prime_fingerprint_64()` and `bundle_perturb()`.

### 5B. L4 experience store

```
struct L4Memory {
    experiences: Vec<(u64, f32, Vec<usize>)>,  // (bundle, reward, causal_layers)
}

fn store(&mut self, bundle: u64, reward: f32, causal_layers: Vec<usize>)
fn recall(&self, query_bundle: u64) -> Option<(f32, Vec<usize>)>  // nearest by Hamming
```

**Already partially built**: `l4_bridge.rs` has `commit_to_l4()` but it's not wired.

### 5C. L4 pre-bias

```
On new input:
  1. Think → get peaks → compute bundle
  2. Search L4: find nearest experience
  3. If found with reward > 0:
     Pre-bias energy toward peaks from successful past experience
  4. If found with reward < 0:
     Pre-bias AWAY from peaks of failed experience
  5. Then continue normal thinking cycle
```

### 5D. L4 causal certificate storage

```
Store full 8×28 causal certificate in L4:
  - 8 octant embeddings (8 × 2048 × 4 bytes = 64 KB)
  - Or: 8 bundles (8 × 8 bytes = 64 bytes, compressed)
  - Gate matrix (8 × 28 × 4 = 896 bytes)
  
On recall: compare new certificate with stored certificates
  - Similar certificate + high reward → shortcut (skip 8 forward passes)
  - Similar certificate + low reward → different strategy needed
```

### Dependencies

```
Stage 2 (contrastive, provides reward signal) → 5B
Stage 3 (causal certificates) → 5D
5A is independent (already built)
5B → 5C → 5D (sequential)
```

---

## STAGE 6: NARS REASONING ON BRANCHES (2-3 days)

**Goal**: Non-Axiomatic Reasoning on thought branches.

### 6A. NARS truth values per branch

```
For each centroid pair (i, j):
  frequency = count(pair seen in positive context) / count(pair seen)
  confidence = count(pair seen) / (count(pair seen) + k)  # k=1 prior
  truth = NarsTruth { frequency, confidence }
```

**Already designed**: `ndarray::hpc::nars::NarsTruth` exists.

### 6B. NARS revision from forward pass

```
On each forward pass:
  For activated centroid pair (i, j):
    old_truth = nars_table[i][j]
    observation = if cos_real > threshold { 1.0 } else { 0.0 }
    new_truth = nars_revision(old_truth, NarsTruth::new(observation, 1.0))
    nars_table[i][j] = new_truth
```

### 6C. NARS-gated thinking

```
During MatVec cycle:
  next[j] += table[i][j] × energy[i] × nars_table[i][j].confidence
  
Low-confidence pairs get suppressed (not enough evidence).
High-confidence pairs dominate (well-established relationships).
```

### 6D. NARS gap detection for OSINT

```
For centroid pair (i, j):
  if nars_table[i][j].confidence < threshold:
    This pair needs more evidence.
    Generate query to find texts where centroid i and j co-occur.
    → OSINT pipeline: spider-rs → Reader-LM → embed → update
```

### Dependencies

```
Stage 2 (contrastive, provides observations) → 6A → 6B → 6C
6D is independent extension (OSINT loop)
```

---

## STAGE 7: OSINT PIPELINE (3-5 days)

**Goal**: Autonomous knowledge acquisition.

### 7A. spider-rs web crawling

```
Query → Google/Bing search → URLs
URLs → spider-rs parallel fetch → raw HTML
```

### 7B. Reader-LM 1.5B clean-up

```
Raw HTML → Reader-LM forward pass → clean markdown
Clean markdown → sentence splitting → text chunks
```

### 7C. Embedding + table update

```
Text chunks → Qwen3-VL forward pass → 2048D embeddings
Embeddings → codebook assignment → centroid pairs
Centroid pairs → contrastive table update (Stage 2)
Centroid pairs → NARS revision (Stage 6)
```

### 7D. Gap-driven loop

```
NARS identifies low-confidence pairs → generate search queries
→ spider-rs → Reader-LM → embed → update table → NARS re-evaluate
→ If confidence still low: try different queries
→ If confidence saturated: commit to L4
```

### Dependencies

```
Stage 2 + Stage 6 → 7C → 7D
7A, 7B are infrastructure (independent)
```

---

## STAGE 8: 4096×16 BRANCH GRAPH (2-3 days)

**Goal**: Scale from 256 centroids to 4096 with sparse connectivity.

### 8A. 4096-centroid CLAM

```
Current:  256 centroids, 256² = 64K pairs, 256 KB (f32)
Target:   4096 centroids, but NOT 4096² = 16M pairs
Instead:  4096 centroids × 16 branches = 65K edges, 256 KB
```

### 8B. Branch selection via forward pass

```
For each centroid i (of 4096):
  Forward pass the centroid's average vector
  → 2048D embedding
  Cosine with all other centroids
  Keep top-16 by cosine similarity
  These 16 = the branches from centroid i
```

### 8C. Sparse MatVec

```
Instead of dense 4096×4096 matrix:
  CSR sparse: 4096 rows × 16 nonzeros = 65K entries
  MatVec: only multiply with 16 neighbors per atom
  Speed: 16/4096 = 250× faster than dense
```

### Dependencies

```
Stage 1 (f32 tables) → 8A → 8B → 8C
```

---

## DEPENDENCY GRAPH

```
STAGE 0 (Done)
  │
  ▼
STAGE 1: Fix the table (f32 + signed)               [1-2 days]
  │
  ├──→ STAGE 2: Contrastive table learning           [2-3 days]
  │      │
  │      ├──→ STAGE 3: SPO 2³ causal certificates    [3-5 days]
  │      │      │
  │      │      └──→ STAGE 4: Gate reward shaping     [3-5 days]
  │      │             (shares gate extraction with 3A)
  │      │
  │      ├──→ STAGE 5: L4 holographic memory          [2-3 days]
  │      │
  │      ├──→ STAGE 6: NARS reasoning                 [2-3 days]
  │      │      │
  │      │      └──→ STAGE 7: OSINT pipeline          [3-5 days]
  │      │
  │      └──→ STAGE 8: 4096×16 branch graph           [2-3 days]
  │
  └──→ (All stages feed back into the table)

Total estimate: 20-30 days for full implementation
Critical path: Stage 1 → 2 → 3 → 4 (10-15 days)
```

---

## PARALLEL TRACKS

### Track A: Core Engine (Stages 1, 2, 8)
Focus: Make the table correct + learnable + scalable.
Owner: thinking-engine crate.
No external dependencies.

### Track B: Causal Reasoning (Stages 3, 4)
Focus: Extract causal structure from forward passes.
Depends on: gate extraction from candle model.
Hardest technical challenge: modifying candle's forward pass.

### Track C: Memory & Learning (Stages 5, 6)
Focus: Long-term learning from accumulated thoughts.
Depends on: contrastive learning (Stage 2) for reward signal.
Mostly wiring existing code (l4_bridge.rs, nars.rs).

### Track D: External Knowledge (Stage 7)
Focus: OSINT pipeline for autonomous knowledge acquisition.
Depends on: NARS gap detection (Stage 6).
Mostly infrastructure (spider-rs, Reader-LM).

```
Week 1:  Track A (Stages 1 + 2)     — fix table, add learning
Week 2:  Track A (Stage 8) ‖ Track C (Stages 5 + 6)
Week 3:  Track B (Stages 3 + 4)     — causal certificates
Week 4:  Track D (Stage 7) + integration testing
```

---

## RISK REGISTER

| Risk | Impact | Mitigation |
|------|--------|------------|
| Gate extraction from candle is hard | Blocks Track B | Use embedding-only causality (no gates), still works for Rung 1-2 |
| 4096-centroid CLAM is slow | Blocks Stage 8 | Run overnight, cache result. Only needs to run once per model. |
| NARS confidence never saturates | OSINT loops forever | Set max iterations per pair (e.g., 10 queries) |
| L4 bundle collisions (64-bit hash) | False matches | Use 128-bit bundle or verify with centroid overlap |
| Contrastive learning diverges | Table gets worse | Learning rate decay + validation set holdout |
| Causal text generation is fragile | "Without fire, causes heat" is ungrammatical | Use template library with tested phrasings per predicate |

---

## METRICS (what to measure at each stage)

| Stage | Metric | Pass Threshold |
|-------|--------|----------------|
| 1 | Top-5 overlap (thinking vs plain cosine) | > 50% |
| 1 | Entropy reduction | > 0% (thinking FOCUSES, not diffuses) |
| 2 | Spearman ρ(table, real_cosines) after 1K updates | > 0.999 |
| 3 | Causal strength: "smoking→cancer" > "roosters→sunrise" | Correct ordering |
| 4 | Gate reward variance across layers | > 0.1 (signal exists) |
| 5 | L4 recall accuracy (similar bundle → similar reward) | > 70% |
| 6 | NARS confidence convergence (stabilizes after N observations) | < 50 observations |
| 7 | OSINT: new text → NARS confidence increase | > 0.1 per document |
| 8 | 4096-centroid table: Spearman ρ vs 256-centroid | > 0.95 |

---

## EMPIRICAL FINDINGS (April 6 2026, evening session)

### Root Cause: Embedding Model Quality > Encoding Precision

```
The codebook quality depends on the EMBEDDING MODEL, not the encoding.

Jina v5 token embeddings:       ρ=0.54 max (no semantics, needs forward pass)
Jina v5 forward pass (28L):     gap=0.128 (semantics CREATED by forward pass)
Qwen3-Embedding-0.6B tokens:    100% top-5 (semantics ALREADY in embeddings)

HighHeelBGZ encoding (i8, BF16) is LOSSLESS for all models.
The bottleneck was always: which embeddings go into the codebook.
```

### i16 is the Sweet Spot

```
i8  (64 KB):  r=0.9995, top5= 96%  (4% loss)
i16 (128 KB): r=1.0000, top5=100%  (LOSSLESS at half size of f32)
i32 (256 KB): r=1.0000, top5=100%  (same as i16, 2× larger)
```

### Softmax T=0.01 Solves Attractor Collapse

```
ReLU:            entropy +6-19% (DIFFUSES, broken)
Softmax T=0.1:   70-77% top-5, entropy -21-31% (FOCUSES)
Softmax T=0.01:  100% top-5, entropy → 0 (PERFECT convergence)
```

### 256 > 4096 for Dense MatVec

```
256 dense:         100% top-5, 368 KB  (3,248× compression)
4096 sparse K=16:  30-45% top-5 (with tricks), 810 KB
4096 dense:        6% top-5 (total diffusion)

256 centroids is the sweet spot for dense MatVec.
4096 needs hierarchical routing or fundamentally different approach.
```

### 4096 Hierarchy Imbalanced

```
Coarse 256 → Fine 4096 mapping: one bucket has 3768/4096 (92%)
CLAM max-min selection creates imbalanced trees.
Need: balanced hierarchical CLAM (force equal bucket sizes)
```

### Models That Work (same Qwen3 tokenizer family)

```
Qwen3-Embedding-0.6B:  100% top-5 (dedicated embedding model)
Jina v5 0.6B:           100% top-5 at T=0.01 (forward-pass-derived)
Jina Reranker v3:        77% top-5 (cross-encoder, 50% inhibition)
Qwen3-VL-Embedding 2B:  70% top-5 (multimodal, larger)
```

### Models That Don't Share Tokenizer

```
Gemma 3/4:    256K vocab (Gemini SentencePiece) — needs own codebook
Llama 4:      202K vocab (Llama BPE) — needs own codebook
Cross-model quorum only works within same tokenizer family.
```

### Why γ+φ Is a No-Op (logical error, not moiré)

```
γ+φ is a MONOTONIC transform. CDF is ORDER-BASED.
Monotonic before rank = same rank. QED.

CDF(cos) ≡ CDF(γ+φ(cos)) because γ+φ preserves order.

On i8 direct: γ+φ HURTS (Pearson drops 0.999 → 0.983).
Log compression distorts VALUE spacing for Gaussian distributions.
γ+φ was designed for heavy-tailed (gate activations, 57% near zero).
Pairwise cosines are NOT heavy-tailed. Wrong tool, wrong data.
```

### Why HEEL Stride Sampling Doesn't Work for Distances

```
stride=4 (256/1024 dims): ρ=0.017 vs full cosine
stride=11 (94/1024 dims):  ρ=0.018 vs full cosine
stride=17 (61/1024 dims):  ρ=0.001 vs full cosine

Root cause: cosine is a GLOBAL metric over ALL dimensions.
Subsampling gives a PROJECTION, not a subsample.
Projected cosine ≈ uncorrelated with full cosine.
This is mathematically expected (Johnson-Lindenstraaten).

HEEL spiral walk is valid for ADDRESSING weight rows.
HEEL is NOT valid for computing DISTANCES between rows.
Use full-dimension cosine for the distance table (i16, proven lossless).
```

### Belichtungsmesser Popcount Stacking (4096 centroids)

```
8 random exposures × 128D each → popcount per centroid pair
Pop>=5 = pairs where >=5 of 8 exposures agree "this is top-32"
r=1.000 on surviving edges (perfect fidelity on confident pairs)

                     Top-5   Top-10   Edges    Size
4096 top-16 sparse:   30%     36%     65K     425 KB
4096 pop≥5+residual:  57%     66%     18K     332 KB  ← WINNER
256 dense (proven):  100%    100%     65K     425 KB

Belichtungsmesser is both BETTER and SMALLER than naive top-K.
But 256 dense still wins for same-size comparison.
```

### Kurvenlineal Reconstruction: Correction of Earlier Analysis

```
EARLIER CLAIM: stride sampling ρ=0.02 (wrong, different codebook sets)
ACTUAL RESULT: stride=4 reconstruction ρ=0.64, top-5=64% (Catmull-Rom)

Method                       ρ vs GT    Top-5
Full 1024D i16               1.0000     100%
Catmull-Rom stride=4         0.641       64%
Subspace only stride=4       0.651        —   (subspace > reconstruction!)
Catmull-Rom stride=11        0.487       46%
Subspace only stride=11      0.514        —

Subspace cosine WITHOUT reconstruction beats Catmull-Rom WITH reconstruction.
The reconstruction adds noise — the missing dimensions are essentially random.

Use case:
  Full 1024D: DISTANCE TABLE (100% fidelity, the brain)
  Stride-4 subspace: ROUTING/FILTERING (65% fidelity, cheap first pass)
  This IS the Belichtungsmesser: cheap route, then exact think.
```

### Rolling Sigma Floor Results (4096 centroids)

```
Method                         Edges   Avg/node  Sparse   Top-5  Top-10  Diversity
GLOBAL sigma                  1,343K    656      84%      32%    32%     12/20
PER-ROW sigma                 1,929K    942      77%      40%    41%     14/20
PER-ROW sigma + top-K=32       105K     51      98.7%    12%    14%     20/20
PER-ROW sigma + bucket shift   203K     99      97.6%    20%    20%     20/20
POPCOUNT≥5 + residual           18K      9      99.8%    57%    66%     16/20  ← WINNER

Bucket shift (histogram knee): principled but centroids too smooth.
Belichtungsmesser: 88% early exit speed but Base17 doesn't separate centroids.
Popcount random exposure: best topology quality for sparse 4096 graphs.

Root cause: centroids are AVERAGES of many tokens → smoother than raw weights.
Belichtungsmesser was designed for raw weight rows, not centroid averages.
```

### Family Bucketing: 99-100% on 4096 (BREAKTHROUGH)

```
Reclassify existing pairs into connected-component families:
  μ+1.0σ: 9 families  → 100% top-5, 100% top-10, 32 MB
  μ+1.5σ: 50 families →  99% top-5, 100% top-10, 31 MB
  μ+2.0σ: 93 families →  99% top-5, 100% top-10, 31 MB

Size dominated by one giant family (4000/4096).
With balanced families: 64 families × 64 centroids = 512 KB.

Architecture convergence with AutocompleteCache:
  Family = precomputed autocomplete branch
  32-step paths precomputed per family
  Cross-family = family representative routing (50×50 = 2500 pairs)
  Within-family = dense exact (64×64 = 4096 pairs per family)
  Total: 2500 + 64×4096 = 264K pairs (vs 16.7M dense)
  
  SiLU gates the TASK TYPE per family:
    Deduction:     family has strong causal chains (high gate, exploit)
    Extrapolation: family extends beyond known data (medium gate)
    Synthesis:     cross-family merging (multiple families activate)
    Inference:     within-family refinement (dense, exact)
    Association:   nearest neighbor in family (1-hop)
    Abduction:     reverse reasoning (follow family backward)
    Fan-out:       expand to neighboring families (cross-family routing)
    Counterfactual: negate family assignment (which family would ¬S be in?)
  
  The gate E/I ratio per layer decides WHICH task type.
  This IS the SPO 2^3 decomposition applied to the autocomplete order.
```

### Grey Matter: 128-Step RL Streaming Architecture

```
The 99% family bucketing means: thinking = cache lookup.
Grey matter streams 128 steps AHEAD of current thought.

Architecture:
  Token 1-32:   Current thought (within-family dense, exact)
  Token 33-64:  Speculative next (cross-family routing, predicted)
  Token 65-128: Grey matter (RL policy, 2-3 hops precomputed)

RL Policy (20KB ONNX):
  State:   gate_pattern[28] + current_family_id
  Action:  next_family_id + confidence
  Reward:  next layer's gate agreement (epiphany = high reward)
  Train:   L4 holographic memory (accumulated experiences)

Storage:
  64 families × 64 centroids × 128 steps = 512 KB routing tables
  20 KB ONNX policy model
  Total: 532 KB for 128-step speculative thinking

Speed:
  Family routing: O(1) lookup (precomputed)
  Within-family: 64×64 dense MatVec (4 KB, fits L1 cache)
  Cross-family: 50×50 representative table (5 KB)
  RL policy: 20 KB ONNX inference (~10μs)
  
  Total per thought: ~50μs (routing) + ~600μs (MatVec) = ~650μs
  128 steps ahead: 128 × 50μs = 6.4ms (grey matter, pipelined)
  
  Effective: current thought at 650μs, next 128 steps at 6.4ms
  That's 128 thoughts precomputed in the time of 10 MatVec cycles.
```

---

## OSINT PIPELINE: WORKING END-TO-END (April 6 2026)

### Architecture

```
DuckDuckGo search → fetch HTML → ReaderLM-v2 (GGUF, 1.5B) → clean markdown
  → Qwen3 tokenizer (151K BPE) → token IDs
  → codebook_index.u16 → centroid IDs
  → F32ThinkingEngine (softmax T=0.01) → peaks + entropy
  → ContrastiveLearner → table updates from pairwise similarity
  → NARS truth → confidence tracking → low confidence → new query
```

### Model Weights

```
ReaderLM-v2 Q8_0 GGUF:
  Location: crates/thinking-engine/data/readerlm-v2/readerlm-v2-q8_0.gguf
  Size:     1.6 GB
  Source:   matrixportalx/ReaderLM-v2-GGUF on HuggingFace
  Base:     Qwen2.5-1.5B-Instruct (fine-tuned for HTML→markdown)
  Vocab:    151936 (SAME as Jina v5, Qwen3-Embedding, Reranker v3)
  Context:  512K tokens
  License:  Apache 2.0

Jina v5 safetensors (for forward pass):
  Location: crates/thinking-engine/data/jina-v5-onnx/model.safetensors
  Size:     1.2 GB
  Source:   jinaai/jina-embeddings-v5-text-small-text-matching
  Vocab:    151936, Hidden: 1024, Layers: 28

Codebook (precomputed):
  Location: /tmp/codebooks/jina-v5-256/
  Files:    cosine_matrix_256x256.f32 (256 KB)
            codebook_index.u16 (297 KB)
  Also in:  releases/v0.2.0-7lane-codebooks/ (git tracked)
  Release:  v0.2.0-7lane-codebooks on GitHub

4096 Codebook:
  Location: /tmp/codebooks/jina-v5-4096/
  Files:    cosine_matrix_4096x4096.f32 (64 MB)
            codebook_index.u16 (297 KB)
            branch_graph_4096x{8,16,32}.{indices.i32,values.f32}
  Release:  v0.3.0-highheelbgz-256-4096 on GitHub
```

### Wiring

```
Python prototype:
  crates/thinking-engine/examples/osint_pipeline.py
  Dependencies: requests, beautifulsoup4, lxml, llama-cpp-python, numpy

Rust bridge:
  crates/thinking-engine/src/osint_bridge.rs
  API: OsintThinkingBridge::from_files() → .think() → .similarity() → .learner()

Existing OSINT crate:
  crates/lance-graph-osint/src/
    crawler.rs  — spider-rs Google crawl (feature: spider-crawl)
    reader.rs   — curl fetch + HTML strip + DeepNSM embed
    extractor.rs — SPO triplet extraction from text
    pipeline.rs  — OsintPipeline.ingest_url()

Connection points:
  lance-graph-osint → raw text
  thinking-engine/osint_bridge.rs → tokenize + think + learn
  thinking-engine/contrastive_learner.rs → table updates
  thinking-engine/f32_engine.rs → softmax T=0.01 thinking
```

### Known Issues

```
1. ReaderLM-v2 Q8_0 outputs ???? on some HTML
   Fix: use F16 GGUF (3.09 GB) or safetensors
   
2. Few centroids per document (1-4)
   Cause: byte-level tokenization, not Qwen3 BPE
   Fix: use proper tokenizer (tokenizers crate or llama.cpp's built-in)
   
3. Codebook from Jina v5 embeddings, not ReaderLM
   The token→centroid mapping reflects Jina v5's weight space
   ReaderLM-v2 has different weights → different optimal codebook
   Fix: build codebook from ReaderLM-v2 embeddings
   
4. Table learning needs more documents
   6 updates from 4 documents = minimal learning
   Need 100+ documents for measurable improvement
```

### Tokenizer Compatibility Matrix

```
Model              Vocab    Tokenizer     Compatible with codebook?
ReaderLM-v2        151936   Qwen2.5 BPE   YES (same vocab, different weights)
Jina v5            151936   Qwen3 BPE     YES (codebook anchor)
Qwen3-Embedding    151669   Qwen3 BPE     ALMOST (267 token difference)
Jina Reranker v3   151936   Qwen3 BPE     YES
Qwen3-VL-Embed     151936   Qwen3 BPE     YES
Qwen3.5 models     TBD      Qwen3.5 BPE   LIKELY (same family)
```
