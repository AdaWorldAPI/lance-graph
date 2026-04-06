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
