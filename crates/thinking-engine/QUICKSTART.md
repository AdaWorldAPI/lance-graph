# Thinking Engine — Quickstart (LM Studio-style)

> Load a model. Type text. See thoughts. No GPU needed.

## 1. Install

```bash
# Clone
git clone https://github.com/AdaWorldAPI/lance-graph.git
cd lance-graph

# Download codebooks (770 KB per model, from GitHub Release)
cd crates/thinking-engine/data
curl -LO https://github.com/AdaWorldAPI/lance-graph/releases/download/v0.2.0-7lane-codebooks/qwen3-vl-embedding-7lane.tar.gz
curl -LO https://github.com/AdaWorldAPI/lance-graph/releases/download/v0.2.0-7lane-codebooks/jina-v5-7lane.tar.gz
curl -LO https://github.com/AdaWorldAPI/lance-graph/releases/download/v0.2.0-7lane-codebooks/jina-reranker-v3-BF16-7lane.tar.gz
tar xzf qwen3-vl-embedding-7lane.tar.gz
tar xzf jina-v5-7lane.tar.gz
tar xzf jina-reranker-v3-BF16-7lane.tar.gz
cd ../../..
```

## 2. Run (interactive, like LM Studio)

```bash
cargo run --release --manifest-path crates/thinking-engine/Cargo.toml --example playground
```

Type any text → see which codebook atoms light up, how energy flows through the distance table, which peaks emerge after 10 cycles of signed softmax thinking.

## 3. Run benchmarks

```bash
# Does thinking beat plain cosine? (honest answer)
cargo run --release --example benchmark_thinking \
  --manifest-path crates/thinking-engine/Cargo.toml

# 7-lane encoding (which formats preserve ground truth?)
cargo run --release --features calibration --example seven_lane_encoder \
  --manifest-path crates/thinking-engine/Cargo.toml -- qwen3-vl-embedding

# Forward pass: real 2048D embeddings from Qwen3-VL
cargo run --release --features calibration --example qwen3_vl_forward \
  --manifest-path crates/thinking-engine/Cargo.toml
```

## 4. Docker

```bash
cd crates/thinking-engine
docker build -t thinking-engine .
docker run -it thinking-engine
```

## 5. Use as library

```rust
use thinking_engine::f32_engine::F32ThinkingEngine;

// Load precomputed f32 cosine table (256 KB)
let table: Vec<f32> = load_f32_table("data/qwen3-vl-embedding-7lane/cosine_matrix_256x256.f32");
let mut engine = F32ThinkingEngine::new(table);

// Perturb with codebook indices from tokenized input
engine.perturb(&[42, 100, 150]);

// Think: 10 cycles of signed softmax (T=0.1)
let result = engine.think(10);

// Get top-5 peaks
let peaks = engine.top_k(5);
for (atom, energy) in &peaks {
    println!("Atom {} energy {:.4}", atom, energy);
}
```

## 6. Online learning (table improves with use)

```rust
use thinking_engine::contrastive_learner::ContrastiveLearner;

let table = load_f32_table("data/qwen3-vl-embedding-7lane/cosine_matrix_256x256.f32");
let mut learner = ContrastiveLearner::new(table, 0.01);

// Each forward pass teaches the table
let real_cosine = forward_pass_cosine("text A", "text B");
let centroid_a = codebook_lookup("text A");
let centroid_b = codebook_lookup("text B");
let error = learner.update_pair(centroid_a, centroid_b, real_cosine);
println!("Table error: {:.4}", error);

// After learning, build engine from improved table
let engine = F32ThinkingEngine::new(learner.table().to_vec());
```

## Architecture

```
Text → Tokenize (Qwen3 BPE) → Codebook lookup (256 centroids)
  → F32ThinkingEngine.perturb() → .think(10)
  → Signed MatVec + Softmax(T=0.1) per cycle
  → Top-K peaks = thought output

The distance table IS the brain.
Each MatVec cycle spreads energy through cosine similarity.
Softmax concentrates on best matches (not ReLU which destroys information).
10 cycles: 70-77% agreement with ground truth embedding cosine.
```

## Key Results

| Metric | Value |
|--------|-------|
| Table format | f32 (Pearson r=0.9999 vs ground truth) |
| Normalization | Softmax T=0.1 (not ReLU) |
| Top-5 agreement | 70% (Qwen3-VL), 77% (Reranker) |
| Entropy reduction | -21% to -31% (focuses, doesn't diffuse) |
| Peak diversity | 100% (no attractor collapse) |
| Speed | ~600μs/query (256 centroids, CPU) |
| Table size | 256 KB (f32) or 64 KB (i8 signed) |
| Models | 3 pretrained codebooks in GitHub Release |

## Models Available

| Model | Params | Dims | Cosine Range | Best For |
|-------|--------|------|--------------|----------|
| Qwen3-VL-Embedding-2B | 2B | 2048D | [-0.85, 0.54] | Multimodal (text+vision) |
| Jina v5 | 0.6B | 1024D | [-0.19, 0.68] | Text embedding |
| Jina Reranker v3 | 0.6B | 1024D | [-0.89, 0.83] | Cross-encoder (50% inhibition) |

## What Failed (honest)

- u8 CDF tables: destroy value geometry (Pearson 0.80)
- γ+φ golden ratio: identical to CDF (zero added value)
- ReLU normalization: attractor collapse (power iteration)
- Multi-lens superposition: Cronbach α < 0.37 (models don't agree)
- Gestalt awareness: zero improvement
- Inhibition leak: zero improvement
