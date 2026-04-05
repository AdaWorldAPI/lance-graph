# SESSION HANDOVER: Calibration, Encoding & Cronbach Alpha Validation

## HYPOTHESIS

The thinking engine's distance tables are a MEASUREMENT INSTRUMENT.
Like any instrument, they need validity testing before the measurements
mean anything. The hypothesis chain:

### H1: BF16 truncation causes systematic rank flips
```
TEST: For 10,000 centroid pairs:
  Compute cos from ONNX f32 weights → rank order R_f32
  Compute cos from GGUF BF16 weights → rank order R_bf16
  Count rank disagreements where |cos_f32| < 0.008 (BF16 uncertainty zone)

PREDICT: ~5% of pairs within ±0.008 of bucket boundaries flip rank.
MEASURE: Spearman ρ(R_f32, R_bf16) < 1.0 by exactly this amount.
VALIDATE: If ρ matches prediction → we understand the truncation physics.
```

### H2: γ+φ encoding preserves more rank order than linear CDF
```
TEST: Same 10,000 pairs:
  Encode cos → u8 via linear CDF → rank order R_linear
  Encode cos → u8 via γ+φ redistribution → rank order R_gamma
  
PREDICT: ρ(R_f32, R_gamma) > ρ(R_f32, R_linear)
  because γ concentrates resolution near the gate decision boundary
  where BF16 truncation causes the most rank flips.
VALIDATE: If γ+φ > linear → golden ratio redistribution IS calibration.
```

### H3: i8 signed preserves more information than u8 unsigned for gate-heavy roles
```
TEST: For ffn_gate role specifically:
  Encode cos → u8[0,255] → Spearman vs f32 → ρ_unsigned
  Encode cos → i8[-128,+127] → Spearman vs f32 → ρ_signed
  
PREDICT: ρ_signed > ρ_unsigned for gate (68.9% near zero, sign matters)
  ρ_signed ≈ ρ_unsigned for Q (positive-skewed, sign rarely matters)
VALIDATE: If gate benefits but Q doesn't → sign preservation is role-specific.
```

### H4: ICC profile correction brings ALL encoding paths to ρ > 0.998
```
TEST: After ICC correction on each path:
  ρ(R_f32, R_corrected_linear) > 0.998
  ρ(R_f32, R_corrected_gamma) > 0.998
  ρ(R_f32, R_corrected_signed) > 0.998
  
PREDICT: ICC correction absorbs the residual error regardless of encoding path.
  The CHEAPEST encoding + ICC ≈ the BEST encoding without ICC.
VALIDATE: If all paths reach 0.998 after ICC → encoding choice doesn't matter,
  only the ICC quality matters. Simplify to cheapest encoding + good ICC.
```

### H5: Multi-lens Cronbach alpha shows internal consistency
```
TEST: For N sentence pairs, compute distances via all 6 lenses.
  Each lens = one "item" in the psychometric instrument.
  Cronbach α = internal consistency of the multi-lens measurement.
  
PREDICT: α > 0.90 for similar-pair detection (all lenses agree on "similar")
  α < 0.70 for relevance detection (lenses disagree = DIFFERENT information)
VALIDATE: High α for similarity = lenses are redundant (use one, save compute).
  Low α for relevance = lenses are complementary (use all, superposition helps).
```

---

## TESTING PROTOCOL

### Phase 1: Ground Truth (ONNX f32)

```python
# Load Jina v5 ONNX via rten
import rten  # or: use burn, or: use candle

model = rten.load("model.onnx", "model.onnx_data")
tokenizer = load_tokenizer("tokenizer.json")

# Generate ground truth for 1000 sentence pairs
pairs = load_test_pairs()  # diverse: similar, dissimilar, related, unrelated
f32_cosines = []
for a, b in pairs:
    emb_a = model.forward(tokenizer.encode(a))  # f32 embedding
    emb_b = model.forward(tokenizer.encode(b))
    f32_cosines.append(cosine(emb_a, emb_b))

# This is the RAW. Everything calibrates against this.
```

### Phase 2: BF16 Baseline

```python
# Stream Jina v5 F16 GGUF
bf16_weights = stream_gguf("v5-small-text-matching-F16.gguf", "token_embd.weight")

# Same CLAM centroids, but from BF16
bf16_centroids = clam_sample(bf16_weights, n=256)
bf16_table = build_cosine_table(bf16_centroids)  # f32 cosine on bf16 inputs

# Map same sentences through BF16-derived codebook
bf16_distances = []
for a, b in pairs:
    ca = codebook_lookup(tokenizer.encode(a), bf16_assignments)
    cb = codebook_lookup(tokenizer.encode(b), bf16_assignments)
    bf16_distances.append(bf16_table[ca][cb])
```

### Phase 3: Encoding Paths

```rust
// For each encoding path, produce a u8/i8 distance table:

// Path 1: Linear CDF (current HDR encoding)
let linear_table = hdr_cdf_encode(&raw_cosines, 256);  // u8[0,255]

// Path 2: γ+φ redistributed
let gamma_table = gamma_phi_encode(&raw_cosines, gamma=1.50, 256);

// Path 3: Signed i8
let signed_table = signed_encode(&raw_cosines, 256);  // i8[-128,+127]

// Path 4: γ+φ signed
let gamma_signed_table = gamma_phi_signed_encode(&raw_cosines, gamma=1.50, 256);

// Path 5: highheelbgz spiral
let spiral_table = spiral_encode(&raw_cosines, stride=golden_ratio, 256);
```

### Phase 4: Spearman ρ per path

```rust
fn evaluate_path(f32_cosines: &[f32], encoded_table: &[u8], assignments: &[u16], pairs: &[(Text, Text)]) -> f32 {
    let encoded_distances: Vec<f32> = pairs.iter()
        .map(|(a, b)| {
            let ca = assignments[tokenize(a)];
            let cb = assignments[tokenize(b)];
            encoded_table[ca * N + cb] as f32
        })
        .collect();
    
    spearman(&f32_cosines, &encoded_distances)
}

// Measure BEFORE ICC:
let rho_linear  = evaluate_path(&f32_cosines, &linear_table, ...);
let rho_gamma   = evaluate_path(&f32_cosines, &gamma_table, ...);
let rho_signed  = evaluate_path(&f32_cosines, &signed_table, ...);
let rho_spiral  = evaluate_path(&f32_cosines, &spiral_table, ...);

// Build ICC profiles:
let icc_linear  = LensProfile::build("jina-v5", "token_embd", Linear, &f32_cosines, &linear_table, N);
let icc_gamma   = LensProfile::build("jina-v5", "token_embd", GammaPhi, &f32_cosines, &gamma_table, N);

// Measure AFTER ICC:
let rho_linear_corrected  = evaluate_corrected(&f32_cosines, &linear_table, &icc_linear, ...);
let rho_gamma_corrected   = evaluate_corrected(&f32_cosines, &gamma_table, &icc_gamma, ...);
```

### Phase 5: Cronbach Alpha

```rust
/// Cronbach's alpha for multi-lens internal consistency.
///
/// items[lens][pair] = distance measured by this lens for this pair.
/// Higher α = more agreement between lenses.
fn cronbach_alpha(items: &[Vec<f32>]) -> f32 {
    let k = items.len() as f32;  // number of lenses
    let n = items[0].len();       // number of pairs
    
    // Total score variance
    let totals: Vec<f32> = (0..n)
        .map(|pair| items.iter().map(|lens| lens[pair]).sum::<f32>())
        .collect();
    let var_total = variance(&totals);
    
    // Sum of item variances
    let var_sum: f32 = items.iter()
        .map(|lens| variance(lens))
        .sum();
    
    // α = (k / (k-1)) × (1 - Σvar_item / var_total)
    (k / (k - 1.0)) * (1.0 - var_sum / var_total)
}

fn variance(data: &[f32]) -> f32 {
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n
}

// Measure:
let lens_distances = vec![
    jina_distances,      // lens 0
    bge_distances,       // lens 1
    reranker_distances,  // lens 2
    reader_distances,    // lens 3
    qwopus_distances,    // lens 4
];

let alpha = cronbach_alpha(&lens_distances);
// α > 0.90: lenses are redundant (one is enough for this task)
// α 0.70-0.90: lenses agree mostly (superposition adds a little)
// α < 0.70: lenses see different things (superposition is valuable)
```

---

## SYNTHESIS: What the Results Tell Us

### If H1 confirms (BF16 flips ~5% of ranks):
→ boundary_risk metadata is ESSENTIAL
→ 95/5 split: fast cascade for safe pairs, LEAF validation for boundary pairs
→ γ+φ should reduce the 5% by moving boundaries away from BF16 quant steps

### If H2 confirms (γ+φ > linear):
→ γ+φ becomes the DEFAULT encoding (not optional, mandatory)
→ Per-role γ offsets are critical (Gate=1.50, Q=0.37)
→ The golden ratio IS the calibration (not a cosmetic choice)

### If H3 confirms (i8 > u8 for gate, ≈ for Q):
→ i8 for gate-modulated roles (K, V, Up), u8 for extern roles (Q, Down)
→ Mixed encoding per role within the same layer
→ SiLU-ONNX is definitively unnecessary

### If H4 confirms (ICC brings all to 0.998):
→ Encoding choice is secondary to ICC quality
→ Cheapest encoding + good ICC = optimal
→ Focus engineering effort on ICC, not on better encodings

### If H5 shows high Cronbach α for similarity:
→ Multi-lens superposition is REDUNDANT for similarity tasks
→ Use ONE lens (cheapest) for similarity, save 5× compute
→ Reserve multi-lens for tasks where α < 0.70 (complementary lenses)

### If H5 shows low Cronbach α for relevance:
→ Multi-lens superposition IS VALUABLE for relevance tasks
→ Embedding (Jina) sees different things than reranker
→ The DISAGREEMENT between lenses IS information
→ Keep all lenses, the superposition product captures what no single lens sees

---

## CODE LOCATIONS

```
Contract DTOs:
  crates/lance-graph-contract/src/high_heel.rs
    → LensProfile (ICC profile DTO)
    → LensConfig (6-lane registry)
    → EncodingPath enum
    → LENS_REGISTRY static array

Thinking Engine:
  crates/thinking-engine/src/jina_lens.rs     (Jina v3 lens, 250K vocab)
  crates/thinking-engine/src/bge_m3_lens.rs   (BGE-M3 lens, 250K vocab)
  crates/thinking-engine/src/reranker_lens.rs  (Reranker v3, 151K vocab, NEW)
  crates/thinking-engine/src/silu_correction.rs (may be replaced by i8)
  crates/thinking-engine/src/engine.rs         (MatVec cycle)

Calibration:
  crates/thinking-engine/examples/calibrate_lenses.rs  (Spearman + ICC harness)
  crates/thinking-engine/examples/hdr_audit.rs         (all models compared)
  crates/thinking-engine/examples/silu_crosscheck.rs   (u8 vs corrected)

Codec:
  crates/bgz-tensor/src/gamma_phi.rs          (γ+φ encode/decode)
  crates/bgz-tensor/src/codebook_calibrated.rs (two-pass build)
  crates/highheelbgz/src/                      (spiral stride, golden ratio)

ONNX:
  AdaWorldAPI/rten                             (ONNX runtime, your fork)
  AdaWorldAPI/rten-ndarray-demo                (rten ↔ ndarray bridge)

Ground Truth Models:
  jinaai/jina-embeddings-v5-text-small-text-matching
    → model.onnx + model.onnx_data (2.4 GB, f32 precision)
    → v5-small-text-matching-F16.gguf (1.2 GB, streamable)
    → tokenizer.json (11.4 MB, real BPE)

Data:
  crates/thinking-engine/data/Qwopus3.5-27B-v3-BF16-silu/
    → 64 layers × 5 role tables (305 binary files)
    → 248K token assignments
    → tokenizer.json (Qwen2 BPE)
    → layer_stats.json
  crates/thinking-engine/data/jina-v3-hdr/ (64 KB table + 488 KB index)
  crates/thinking-engine/data/bge-m3-hdr/ (64 KB table + 488 KB index)
  crates/thinking-engine/data/jina-reranker-v3-BF16-hdr/ (64 KB table + 296 KB index)
  crates/thinking-engine/data/codebooks/ (8 models, 64×64 tables)
```

---

## IMPLEMENTATION ORDER

```
1. Download Jina v5 ONNX (2.4 GB) + GGUF (1.2 GB) + tokenizer
2. Load ONNX via rten → generate f32 ground truth for 1000 pairs
3. Stream GGUF → CLAM → build 5 encoding variants
4. Measure Spearman ρ for each (H1-H3)
5. Build ICC profiles for each
6. Measure corrected ρ (H4)
7. Run all 6 lenses on same pairs → Cronbach α (H5)
8. Synthesize: which encoding × which role × ICC or not
9. Encode findings as LensProfile metadata in contract
10. Re-bake tables with winning encoding per role

Estimated: 3-4 hours for complete validation.
```

---

## SESSION CONTEXT (what was built before this)

```
This session (session_01ChLvBfpJS8dQhHxRD4pYNp) delivered:
  67+ commits across lance-graph + ndarray
  235K LOC Rust across 18 crates
  
Key deliverables:
  - Qwopus 27B: 64 layers streamed from 53.8 GB BF16 in 116s
  - SiLU gate correction: 86% material (BUT may be replaced by i8 signed)
  - 4096-centroid codebook: 248K tokens
  - Real Qwen BPE tokenizer
  - Living thought loop (tension-driven autoregressive)
  - MoE architecture (4096 experts, top-128)
  - NARS gate modulator (three modes)
  - Jina Reranker lens (wired, 9 tests)
  - LensProfile ICC DTO
  - LensConfig 6-lane registry
  - Calibration harness (Spearman + ICC builder)
  - OSINT pipeline (spider + OCR + NARS expansion)
  - Wikileaks graph (1,872 nodes)
  - SIMD OCR (10× faster than tesseract)
  - Felt OCR (Base17/polar/palette)

Parallel session doing:
  - i8 signed tables (excitation/inhibition)
  - u8 vs i8 dual-path comparison
  - Started from reranker lens (this session wired it)

This calibration session validates ALL of the above.
Without it, every measurement drifts. GPS without relativity.
```
