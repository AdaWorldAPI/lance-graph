# SESSION: bgz-tensor — Compile, Benchmark, Beat TurboQuant

## What This Is

`crates/bgz-tensor/` is a metric-algebraic tensor codec that replaces transformer
weight matrix computation with precomputed table lookup. NOT quantization — 
computation compilation.

The core insight from bgz17: you don't compress the operand, you replace the
operation. A 256×256 distance table (128 KB) replaces every attention matmul
in the model. One memory access per attention score instead of O(d) multiply-adds.

## Files

```
crates/bgz-tensor/
├── Cargo.toml           # standalone, zero deps
└── src/
    ├── lib.rs           # crate root, thesis, module declarations
    ├── projection.rs    # f32/f16 weight vector → Base17 (34 bytes via golden-step)
    ├── palette.rs       # CLAM-inspired manifold clustering → 256 archetypes
    ├── attention.rs     # AttentionTable (128KB), ComposeTable (64KB), CompiledHead
    ├── cascade.rs       # HHTL 4-layer cascade: HEEL→HIP→TWIG→LEAF
    └── quality.rs       # Pearson/Spearman ρ, top-K recall, QualityReport
```

## Task List

### P0: Compile and test

```bash
cd crates/bgz-tensor
cargo check
cargo test
```

Add `"crates/bgz-tensor"` to the workspace `exclude` list in root `Cargo.toml`
if not already present.

### P1: Synthetic attention benchmark

Build `tests/synthetic_benchmark.rs`:

```rust
use bgz_tensor::*;

#[test]
fn synthetic_attention_quality() {
    let d_model = 256;
    let d_head = 64;
    let palette_k = 64; // start small

    // Generate random weight matrices (simulating one attention head)
    let q_weights: Vec<f32> = (0..d_head * d_model)
        .map(|i| ((i * 7 + 13) % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let k_weights: Vec<f32> = (0..d_head * d_model)
        .map(|i| ((i * 11 + 29) % 1000) as f32 / 500.0 - 1.0)
        .collect();

    // Ground truth: full dot products
    let gt_dots = quality::ground_truth_dots(&q_weights, &k_weights, d_head, d_model);

    // Compile head
    let v_weights = vec![0.0f32; d_head * d_model];
    let head = CompiledHead::compile(&q_weights, &k_weights, &v_weights, d_model, d_head, palette_k);

    // Get compiled distances (negate for correlation with dot products)
    let q_projected = projection::project_weight_matrix(&q_weights, d_head, d_model);
    let k_projected = projection::project_weight_matrix(&k_weights, d_head, d_model);
    let compiled_dists = projection::pairwise_l1(&q_projected, &k_projected);
    let compiled_f64: Vec<f64> = compiled_dists.iter().map(|&d| -(d as f64)).collect();

    // Measure quality
    let report = QualityReport::compute(
        &gt_dots, &compiled_f64,
        (q_weights.len() + k_weights.len()) * 4, // input bytes
        head.byte_size(),
    );

    println!("{}", report.summary());
    assert!(report.pearson_rho > 0.8, "ρ = {:.4} — should be > 0.8", report.pearson_rho);
}

#[test]
fn cascade_elimination_rate() {
    let n = 64;
    let q: Vec<Base17> = (0..n).map(|i| {
        Base17::from_f32(&(0..256).map(|d| ((i * 97 + d * 31) % 1000) as f32 / 500.0 - 1.0).collect::<Vec<_>>())
    }).collect();
    let k: Vec<Base17> = (0..n).map(|i| {
        Base17::from_f32(&(0..256).map(|d| ((i * 53 + d * 71 + 500) % 1000) as f32 / 500.0 - 1.0).collect::<Vec<_>>())
    }).collect();

    let all: Vec<Base17> = q.iter().chain(k.iter()).cloned().collect();
    let palette = WeightPalette::build(&all, 32);
    let q_idx = palette.assign_all(&q);
    let k_idx = palette.assign_all(&k);
    let table = AttentionTable::build(&palette);

    let config = CascadeConfig {
        heel_min_agreement: 1,
        hip_max_distance: 30000,
        ..Default::default()
    };

    let (_, stats) = cascade::cascade_attention(&q, &k, &q_idx, &k_idx, &table, &config);
    println!("{}", stats.summary());
    // Cascade should eliminate at least some computation
    assert!(stats.elimination_rate() > 0.0);
}
```

### P2: Real weight matrix benchmark

This requires loading actual Llama weights. Options:

**Option A — Extract one head from GGUF:**
Download a small GGUF (e.g., TinyLlama 1.1B Q8_0) and extract one attention head's
Q, K, V weight matrices. Dequantize to f32 for ground truth.

**Option B — Use HuggingFace safetensors:**
Load one layer's `q_proj`, `k_proj`, `v_proj` from a safetensors file.
Split into per-head matrices.

**Option C — Generate realistic synthetic:**
Use statistics from published Llama weight distributions (mean, std, kurtosis per layer)
to generate synthetic weight matrices with realistic structure.

Target: demonstrate ρ > 0.95 on one real attention head.

### P3: Compression ratio analysis

For a 7B model (d_model=4096, n_heads=32, n_layers=32):

```
Standard Q4_K_M per attention layer:
  3 matrices × 4096 × 4096 × 0.5 bytes ≈ 24 MB

bgz-tensor per attention layer:
  3 palettes × 256 entries × 34 bytes ≈ 25 KB (codebooks)
  3 × 4096 palette indices × 1 byte ≈ 12 KB (assignments)
  3 distance tables × 128 KB ≈ 384 KB (shared across heads)
  3 compose tables × 64 KB ≈ 192 KB (shared across heads)
  Total: ≈ 613 KB per layer

Compression: 24 MB / 613 KB ≈ 40× over Q4_K_M
                                ≈ 25,000× over fp16
```

### P4: Integrate with bgz17

The bgz-tensor crate replicates Base17, palette, compose table, etc. from bgz17
because both crates are zero-dependency. Eventually these should share code.

Key integration points:
- bgz17::PaletteSemiring IS the AttentionSemiring — same algebra
- bgz17::SimilarityTable maps distance → calibrated attention score
- bgz17::CrossPlaneMatrices maps to Q×K, Q×V, K×V cross-attention
- bgz17 HHTL cascade IS the inference-time elimination

### P5: Write the paper

Title: "Metric-Algebraic Tensor Codec: Replacing Attention Matmul with
Precomputed Distance Lookup"

Key claims:
1. Projective compression (ρ=0.992) reduces 4096D weight vectors to 17D
2. Palette quantization (256 archetypes) reduces continuous → discrete (1 byte/row)
3. Distance table (128KB L1-resident) replaces O(d) dot product with O(1) lookup
4. Compose table gives multi-hop attention for free (vs stacking layers)
5. HHTL cascade eliminates 95% of attention pairs via metric triangle inequality
6. PCDVQ weighting (20×/3×/1×) preserves direction structure that matters most
7. Runs on CPU L1 cache, no GPU needed

## What NOT to change

- Zero dependencies. No external crates.
- Base17 golden-step folding must use step=11, dim=17 (same as bgz17).
- PCDVQ weights must be 20/3/1 (from arXiv 2506.05432).
- Palette size must be ≤ 256 (8-bit index).
- Distance matrix must be u16 (not u8 — weight distances need more resolution than word distances).
- Compose table must use XOR bind (self-inverse, associative).

## Architecture Cross-Reference

```
bgz17 concept          → bgz-tensor equivalent
─────────────────────────────────────────────────
Base17 (34 bytes)      → Weight vector projection
Palette (256 entries)  → WeightPalette (CLAM-inspired)
DistanceMatrix (u16)   → AttentionTable
PaletteSemiring        → AttentionSemiring
compose_table          → ComposeTable
SimilarityTable        → (TODO: calibrate distance → attention score)
CrossPlaneMatrices     → (TODO: Q×K, Q×V, K×V cross-attention)
HHTL cascade           → cascade::cascade_attention
SpoBase17 (102 bytes)  → CompiledHead (Q+K+V palettes)
Precision enum         → CascadeLevel enum

deepnsm concept        → bgz-tensor equivalent
─────────────────────────────────────────────────
WordDistanceMatrix     → AttentionTable (same idea, different domain)
SimilarityTable        → (same, maps distance → calibrated score)
SPO triple             → Q/K/V weight decomposition
ContextWindow          → (KV-cache replacement via palette indices)
```

## Repo Access

Use the AdaWorldAPI GitHub PAT from secure config.
Branch: create `bgz-tensor-crate` from main.
