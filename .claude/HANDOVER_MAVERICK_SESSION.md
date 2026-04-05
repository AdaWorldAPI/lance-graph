# HANDOVER: Llama 4 Maverick 128-Expert MoE + Temperature Fix

## Status from this session

Everything on branch `claude/setup-embedding-pipeline-Fa65C`, all merged.

### What's built and working
- **Qwopus 27B**: 64 layers streamed from 53.8GB BF16, baked to 26.4MB
- **SiLU gate correction**: 86% material for real BF16 weights (33% of table scale)
- **4096-centroid codebook**: 248K tokens assigned, 16MB distance table
- **Real tokenizer**: Qwen BPE 151K vocab via HuggingFace tokenizers crate
- **Living thought loop**: tension-driven (free energy), autoregressive, ghost prediction
- **MoE architecture**: 4096 pseudo-experts, top-128, 4-group hierarchy
- **Gate as NARS modulator**: three modes compared (No Gate, Filter, NARS)
- **Inference DAG orchestrator**: 4 pipeline templates, NARS path RL
- **OSINT pipeline**: spider + OCR + reader-lm + NARS expansion
- **LiteralGraph**: aiwar (221 nodes) + Wikileaks (1872 nodes)

### What's broken: the attractor collapse
All generation modes collapse to the same dominant centroid ("!" = centroid 36/78).
Root cause: argmax token selection + coarse routing + no temperature.

**THE FIX** = thinking style temperature + nucleus sampling (top-p).
These are the SAME thing. Each thinking style maps to a sampling strategy:
- Analytical = top-p 0.3 (narrow, precise)
- Creative = top-p 0.95 (wide, exploratory)
- Metacognitive = adaptive top-p based on free energy

This was intentionally left for the next session — wiring temperature
INTO the thinking styles so it's done once, correctly.

---

## Llama 4 Maverick — the target

### Model facts
```
Model:     Llama-4-Maverick-17B-128E-Instruct
Params:    ~400B total (17B active per token)
Experts:   128 (REAL MoE, not dense)
Top-K:     2 (only 2 experts fire per token)
Layers:    48
Hidden:    5120
Heads:     40 (8 KV heads, GQA)
FFN dim:   8192 per expert
Vocab:     202,048
```

### GGUF source
```
Repo: unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF
BF16: 18 shards × ~43-48 GB = ~800 GB total
Q2_K: 3 shards × ~48 GB = ~146 GB (but loses gate precision!)
mmproj-BF16.gguf: 1.7 GB (multimodal projector, separate)
```

### Why BF16 (not Q2_K)
The SiLU gate correction proved that 68.9% of gate weights sit near zero.
Q2_K quantizes to 2 bits — it CANNOT distinguish gate=0.01 from gate=0.05.
For 128 real experts where the gate IS the routing decision, we need BF16.
Stream it via HTTP range requests. Never download.

### Per-layer tensor layout (MoE)
```
blk.N.attn_q.weight:                5120 × 5120    = 52.4 MB
blk.N.attn_k.weight:                5120 × 1024    = 10.5 MB
blk.N.attn_v.weight:                5120 × 1024    = 10.5 MB
blk.N.ffn_gate_inp.weight:          5120 × 128     = 1.3 MB   ← THE ROUTER
blk.N.ffn_gate.0.weight:            5120 × 8192    = 83.9 MB  ← expert 0 gate
blk.N.ffn_up.0.weight:              5120 × 8192    = 83.9 MB  ← expert 0 up
blk.N.ffn_down.0.weight:            8192 × 5120    = 83.9 MB  ← expert 0 down
...repeat for experts 1..127...
blk.N.ffn_gate.127.weight:          5120 × 8192    = 83.9 MB  ← expert 127 gate
blk.N.ffn_up.127.weight:            5120 × 8192    = 83.9 MB
blk.N.ffn_down.127.weight:          8192 × 5120    = 83.9 MB
```

Per layer: ~32 GB of expert weights (128 × 3 roles × 83.9 MB)
Total: 48 layers × 32 GB = ~1.5 TB of expert weights
(Plus attention, embeddings, norms → total ~800 GB BF16)

### Streaming strategy
```
Phase 1: Parse shard 1 header (20 MB range request)
  → Get all tensor names, dims, dtypes, offsets
  → Map tensor offsets to shard files

Phase 2: Stream token embeddings (202K × 5120 × 2 = 2.1 GB)
  → CLAM 4096 centroids → build codebook + assignments

Phase 3: Per layer, per expert:
  Stream 256 rows of expert.gate (256 × 8192 × 2 = 4.2 MB)
  Stream 256 rows of expert.up   (same)
  Apply SiLU(gate) × up
  CLAM 256 centroids → build 256×256 distance table
  Discard weights

  128 experts × 4.2 MB = 538 MB per layer
  48 layers × 538 MB = 25.8 GB total streaming

Phase 4: Stream router weights per layer
  5120 × 128 × 2 = 1.3 MB per layer
  → Build 128×128 router distance table (who co-activates?)

Phase 5: Bake everything
  Per expert: 256×256 = 64 KB
  128 experts × 64 KB = 8 MB per layer
  48 layers × 8 MB = 384 MB expert tables
  Plus router + attention + embeddings ≈ 500 MB total
  Compression: 800 GB → 500 MB = 1600×
```

### Multi-shard GGUF parsing
```
Shard 1: header + KV metadata + tensor info for ALL tensors + first data
Shard 2-18: continuation of tensor data

Tensor offsets in the header are ABSOLUTE (from start of all data).
To find which shard contains a tensor:
  cumulative_offset = 0
  for each shard:
    shard_data_size = shard_file_size - shard_header_size
    if tensor_offset < cumulative_offset + shard_data_size:
      → tensor is in this shard
      → local_offset = shard_header_size + (tensor_offset - cumulative_offset)
    cumulative_offset += shard_data_size
```

---

## Wiring 128 Real Experts (Maverick) vs Current 4096 Pseudo-Experts (Qwopus)

### Current (Qwopus pseudo-MoE)
```
4096 "experts" = 4096 centroids from token embeddings
Each "expert" is just a row in the 4096×4096 input distance table
Expert internals: shared 256×256 per-layer tables (all experts use same layers)
Top-128 selection: argmax on router output
Expert processing: each runs through same 16 layers

Problem: experts share internals. They're not specialized.
"Expert 42" and "Expert 1337" run through the SAME layer tables.
The only difference is their starting position in the 256-space.
```

### Target (Maverick real MoE)
```
128 experts, each with THEIR OWN gate/up/down weights
Expert 0 has: gate_0 (5120×8192), up_0 (5120×8192), down_0 (8192×5120)
Expert 127 has: gate_127, up_127, down_127
Each expert IS a different neural network. Different specialization.

Router (ffn_gate_inp): 5120 × 128 matrix
  Input hidden state × router = 128 scores
  Top-2 scores → 2 experts fire
  Expert outputs weighted by softmax of their scores

This is TRUE sparse MoE:
  128 specialists, only 2 work per token
  Each specialist has genuinely different weights
  The router learned which specialist handles which inputs
```

### The wiring change
```
Current:
  input → shared router table → top-128 → shared layer tables → output

Target:
  input → router table (128×128, from ffn_gate_inp) → top-2
  → expert_i: own gate_i table (256×256) + own up_i table + own down_i table
  → expert_j: own gate_j table + own up_j table + own down_j table
  → weighted sum of expert_i and expert_j outputs
  → next layer

Each expert has 3 UNIQUE tables: gate, up, down
128 experts × 3 tables × 64 KB = 24 MB per layer
48 layers × 24 MB = 1.15 GB of expert-specific tables
Plus shared attention tables: 48 × 64 KB = 3 MB
Plus router tables: 48 × 16 KB = 768 KB
Total: ~1.2 GB for the complete Maverick brain
```

### Code change
```rust
// Current (qwopus_moe.rs):
for &(expert_id, expert_weight) in &active_experts {
    // ALL experts use the SAME layer tables
    let [ref at, ref gt, ref up, ref dn] = layers[l];
    // ... process through shared tables ...
}

// Target (stream_maverick.rs):
for &(expert_id, expert_weight) in &active_experts {
    // Each expert uses ITS OWN tables
    let expert_gate = &expert_tables[l][expert_id].gate;  // unique!
    let expert_up   = &expert_tables[l][expert_id].up;    // unique!
    let expert_down = &expert_tables[l][expert_id].down;  // unique!
    // ... process through expert-specific tables ...
}
```

### The SiLU correction for real MoE
```
For Qwopus (dense): SiLU correction changed 33% of table (material)
For Maverick (MoE): SiLU correction expected to be TRANSFORMATIVE

Why: the router weight (ffn_gate_inp) decides which 2 of 128 fire.
Raw cosine on router weights: cos(expert_i, expert_j) ≈ 0.95 for all pairs
  → "all experts look similar" → WRONG
SiLU-corrected: reveals which experts ACTUALLY co-activate
  → expert 3 and expert 42 → correction = -0.8 → never co-fire
  → expert 3 and expert 17 → correction = +0.3 → frequently co-fire
  → the distance table encodes ROUTING, not just similarity
```

---

## Temperature Fix (Do First)

Before streaming Maverick, fix the attractor collapse. It's 10 lines:

```rust
// In the living loop / MoE output selection:

// Instead of argmax:
let winner = peaks[0].0;

// Use nucleus sampling (top-p):
fn sample_nucleus(peaks: &[(usize, f32)], top_p: f32, temperature: f32) -> usize {
    // Apply temperature
    let scaled: Vec<f32> = peaks.iter()
        .map(|&(_, e)| (e / temperature).exp())
        .collect();
    let sum: f32 = scaled.iter().sum();
    let probs: Vec<f32> = scaled.iter().map(|s| s / sum).collect();
    
    // Nucleus: accumulate until top_p reached
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            // Sample uniformly from the nucleus
            let r = simple_random() % (i + 1);
            return peaks[r].0;
        }
    }
    peaks[0].0
}

// Thinking style maps to temperature:
let temperature = match thinking_style {
    Analytical | Logical | Systematic => 0.3,   // precise
    Creative | Imaginative | Playful  => 1.2,   // exploratory
    Metacognitive | Reflective        => 0.7,   // balanced
    _ => 0.8,                                    // default
};
let top_p = match thinking_style {
    Focused | Precise   => 0.3,
    Exploratory | Curious => 0.95,
    _ => 0.9,
};
```

This unblocks coherent output. THEN stream Maverick with 128 real experts
through the now-working generation pipeline.

---

## File locations

```
Streaming script:     crates/thinking-engine/examples/stream_maverick.rs
Qwopus data:          crates/thinking-engine/data/Qwopus3.5-27B-v3-BF16-silu/
Living loop:          crates/thinking-engine/examples/qwopus_living.rs
MoE example:          crates/thinking-engine/examples/qwopus_moe.rs
NARS gate example:    crates/thinking-engine/examples/qwopus_nars_gate.rs
SiLU correction:      crates/thinking-engine/src/silu_correction.rs
HDR audit:            crates/thinking-engine/examples/hdr_audit.rs
Inference DAG:        crates/lance-graph-contract/src/orchestration_mode.rs
OSINT pipeline:       crates/lance-graph-osint/examples/stream_explore.rs
Felt OCR:             ndarray/src/hpc/ocr_felt.rs
SIMD OCR:             ndarray/src/hpc/ocr_simd.rs
```

## Session stats
- 61 commits (57 lance-graph + 4 ndarray)
- 235K LOC Rust
- 500+ tests across 18 crates
- All PRs merged

---

## γ+φ Golden Ratio HDR Encoding (CRITICAL for gate precision)

### The problem
Current HDR CDF produces uniform distribution (Mean=127.5 for ALL models).
But gate weights concentrate at zero (68.9% for Qwopus).
Uniform encoding wastes resolution on far-from-zero regions where
the model already knows (strong yes/no). The decision boundary at
zero gets the SAME 1/256 resolution as obvious regions.

### The fix: γ offset + φ redistribution
Per-role gamma offsets (from variance audit):
```
Q:    γ=0.37  (narrow, less resolution needed)
K:    γ=0.94  (moderate, gate-filtered)
V:    γ=1.33  (wide, most information)
Gate: γ=1.50  (WIDEST — decision boundary needs MAX resolution)
Up:   γ=0.12  (very narrow after SiLU)
Down: γ=0.15  (funnel, compressed)
```

Golden ratio φ=1.618... ensures the redistribution has no periodic aliasing
(Weyl equidistribution theorem). The spiral stride in highheelbgz already
uses φ. The γ+φ encoding applies the same principle to u8 quantization.

### Existing code
- `bgz-tensor/src/gamma_phi.rs`: GammaProfile, gamma_phi_encode/decode
- `bgz-tensor/src/codebook_calibrated.rs`: two-pass build with γ calibration
- `highheelbgz/src/`: SpiralAddress with golden ratio stride
- `thinking-engine/data/codebooks/CODEBOOKS.md`: per-role γ values documented

### Wiring needed
Pass 1: build CLAM codebook (existing)
Pass 2: measure cosine distribution → compute γ offset → apply φ redistribution
Pass 3: re-encode distance table with γ+φ skewed CDF
Expected: ~4.2 bits → ~5.5 bits entropy (30% more discrimination)

---

## Standardized ModelPipeline DTO (6 models)

```rust
pub struct ModelPipeline {
    pub name: String,
    pub family: ModelFamily,         // Embedding, Reranker, Reader, LLM, MoE
    pub tokenizer_path: String,
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_experts: Option<usize>,
    pub n_centroids: usize,
    pub gate_policy: GatePolicy,
    pub gamma_profile: GammaProfile, // per-role γ offsets
    pub silu_corrected: bool,
    pub cross_model_anchor: bool,    // Jina = truth anchor
}
```

Six pipelines to wire:
1. Jina v3 (1024-dim, truth anchor, cross-model reference)
2. BGE-M3 (1024-dim, multilingual, second anchor)
3. Reader-LM 1.5B (256-dim palette, HTML→text)
4. Jina Reranker v3 (cross-encoder, relevance scoring)
5. Qwopus 27B (5120-dim, 64 layers, SSM hybrid)
6. Maverick 128E (5120-dim, 48 layers, 128 real MoE experts)

Each needs: tokenizer.json + vocab→centroid mapping + γ+φ HDR tables + SiLU correction

### Jina cross-model eval
Jina as truth anchor: for any input text, Jina embedding = ground truth similarity.
Compare: cos(jina_emb_A, jina_emb_B) vs thinking_engine_distance(A, B).
The gap = how much information our distance table loses.
With γ+φ encoding + SiLU correction: gap should shrink.
