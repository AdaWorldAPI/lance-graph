# PLAN: BF16 Distance Tables — From StackedN Cosine to Thinking Engine

## SCOPE

Replace the u8/i8 distance tables in thinking-engine with BF16 (u16) tables
built from the existing bgz-tensor StackedN/ClamCodebook pipeline.

**NOT in scope:** new models, new GGUF streaming, ONNX ground truth, LoRA training.
Those come AFTER the tables are correct.

---

## PREREQUISITES (already exist)

```
✓ bgz-tensor/stacked_n.rs:    StackedN, ClamCodebook, bf16_to_f32, f32_to_bf16
✓ bgz-tensor/gamma_phi.rs:    GammaProfile, gamma_phi_encode/decode
✓ highheelbgz/src/:            SpiralAddress, SpiralWalk, CoarseBand
✓ thinking-engine/engine.rs:   ThinkingEngine (u8 table, MatVec cycle)
✓ thinking-engine/pooling.rs:  ArgMax/Mean/TopK/Nucleus
✓ thinking-engine/builder.rs:  ThinkingEngineBuilder, Temperature, CommitSinks
✓ Qwopus BF16 data:            GitHub Release v0.1.2-qwopus-layers (27 MB)
✓ Reranker BF16 data:          data/jina-reranker-v3-BF16-hdr/ (baked)
```

---

## PHASE 1: BF16 ThinkingEngine (replace u8 engine)

**Owner:** thinking-engine crate
**Depends on:** bgz-tensor (StackedN, bf16_to_f32)
**Estimated:** 2-3 hours

### 1.1 New module: `bf16_engine.rs`

```rust
use bgz_tensor::stacked_n::{bf16_to_f32, f32_to_bf16};

pub struct BF16ThinkingEngine {
    distance_table: Vec<u16>,  // BF16 bit patterns
    pub energy: Vec<f32>,
    pub size: usize,
    pub cycles: u16,
    pub convergence_threshold: f32,
}

impl BF16ThinkingEngine {
    /// Build from ClamCodebook pairwise cosine.
    pub fn from_codebook(codebook: &ClamCodebook) -> Self { ... }
    
    /// Build from raw BF16 cosine values (e.g. streamed from GGUF).
    pub fn from_bf16_cosines(cosines: &[u16], size: usize) -> Self { ... }
    
    /// Load BF16 table from file (256×256 × 2 bytes = 128 KB).
    pub fn load(path: &Path) -> Self { ... }

    /// ONE cycle: BF16 → f32 (lossless shift) → accumulate → normalize.
    /// Sign bit (bit 15) = excitation (+) vs inhibition (-).
    /// No floor needed. Negative cosines naturally inhibit.
    pub fn cycle(&mut self) { ... }
    
    /// Cycle with temperature (softmax/T).
    pub fn cycle_with_temperature(&mut self, temperature: f32) { ... }
    
    /// Think until convergence.
    pub fn think(&mut self, max_cycles: usize) -> ResonanceDto { ... }
    pub fn think_with_temperature(&mut self, max_cycles: usize, t: f32) -> ResonanceDto { ... }
    
    /// Same API as ThinkingEngine: perturb, reset, commit, entropy, active_count.
    pub fn perturb(&mut self, codebook_indices: &[u16]) { ... }
    pub fn reset(&mut self) { ... }
    pub fn commit(&self) -> BusDto { ... }
}
```

### 1.2 Update builder.rs

```rust
pub enum TableType {
    UnsignedU8,    // legacy, CDF encoded (deprecated for new tables)
    SignedI8,      // deprecated (8-bit bottleneck, CDF rank relabeling)
    BF16,          // NEW: from StackedN cosine, sign preserved, full dynamic range
}

pub enum BuiltEngine {
    Unsigned(ThinkingEngine),
    Signed(SignedThinkingEngine),
    BF16(BF16ThinkingEngine),  // NEW
}
```

### 1.3 Tests

```
- bf16_engine_creates (from synthetic BF16 table)
- bf16_cycle_excites_and_inhibits (positive/negative BF16 values)
- bf16_think_converges
- bf16_temperature_differentiates (T=0.1 vs T=1.5 produce different peaks)
- bf16_vs_u8_comparison (same centroids, BF16 should discriminate better)
- bf16_from_codebook (build from real ClamCodebook — needs StackedN data)
```

### 1.4 Quality check

```
[ ] Uses StackedN cosine for pairwise distances
[ ] BF16 precision (u16, bf16_to_f32 lossless)
[ ] Sign preserved (bit 15 = inhibition)
[ ] No CDF encoding (direct cosine → BF16)
[ ] No floor needed (negative cosines = natural inhibition)
[ ] Tests use real BF16 values, not synthetic u8 converted
```

---

## PHASE 2: Gate-Modulated Table for Up Role

**Owner:** thinking-engine + bgz-tensor
**Depends on:** Phase 1, Qwopus GGUF data (Release v0.1.2)
**Estimated:** 2-3 hours

### 2.1 Gate modulation on StackedN level

```rust
// In bgz-tensor or thinking-engine:
pub fn gate_modulate_stacked(
    gate_stacked: &StackedN,   // gate centroid
    role_stacked: &StackedN,   // up/k/v centroid
) -> Vec<f32> {
    let gate_f32 = gate_stacked.hydrate_f32();
    let role_f32 = role_stacked.hydrate_f32();
    gate_f32.iter().zip(&role_f32)
        .map(|(&g, &r)| silu(g) * r)
        .collect()
}

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
```

### 2.2 Build gate-modulated BF16 table

```rust
pub fn build_gate_modulated_table(
    gate_codebook: &ClamCodebook,
    up_codebook: &ClamCodebook,
) -> BF16ThinkingEngine {
    let n = up_codebook.entries.len();
    let mut table = vec![0u16; n * n];
    for i in 0..n {
        let activated_i = gate_modulate_stacked(
            &gate_codebook.entries[i].stacked,
            &up_codebook.entries[i].stacked,
        );
        for j in (i+1)..n {
            let activated_j = gate_modulate_stacked(
                &gate_codebook.entries[j].stacked,
                &up_codebook.entries[j].stacked,
            );
            let cos = cosine_f32(&activated_i, &activated_j);
            let bf16 = f32_to_bf16(cos);
            table[i * n + j] = bf16;
            table[j * n + i] = bf16;
        }
        table[i * n + i] = f32_to_bf16(1.0);
    }
    BF16ThinkingEngine::from_bf16_cosines(&table, n)
}
```

### 2.3 Per-role table set

```rust
pub struct LayerTables {
    pub attn_q: BF16ThinkingEngine,    // raw cosine
    pub attn_k: BF16ThinkingEngine,    // raw cosine (NOT gate-modulated)
    pub attn_v: BF16ThinkingEngine,    // raw cosine (NOT gate-modulated)
    pub ffn_gate: BF16ThinkingEngine,  // raw cosine (topology reference)
    pub ffn_up: BF16ThinkingEngine,    // silu(gate) × up (gate-modulated, 33% Δ)
    pub ffn_down: BF16ThinkingEngine,  // raw cosine
}
```

### 2.4 Quality check

```
[ ] Gate modulation happens on StackedN centroids BEFORE cosine
[ ] Only ffn_up gets silu(gate) (not K, not V)
[ ] Gate table is raw (topology reference for NARS trust)
[ ] BF16 output matches f64 StackedN.cosine() to within 1 ULP
[ ] 33% Δ measured: cos(raw_up) vs cos(silu(gate)×up) table difference
```

---

## PHASE 3: Wire BF16 Engine to Existing Modules

**Owner:** thinking-engine
**Depends on:** Phase 1
**Estimated:** 1-2 hours

### 3.1 Update modules to accept BF16

```
pooling.rs:      works already (operates on f32 energy, not table)
builder.rs:      add TableType::BF16, BuiltEngine::BF16
dual_engine.rs:  compare u8 vs BF16 (not u8 vs i8)
composite_engine.rs: support BF16 engines in multi-lens
cronbach.rs:     variance_agreement_scores() for BF16 tables (convert to f32 for comparison)
l4_bridge.rs:    use StackedN centroids instead of table rows
```

### 3.2 Update examples

```
end_to_end_signed.rs → end_to_end_bf16.rs (BF16 table, real tokenizer)
dual_signed_experiment.rs → dual_bf16_experiment.rs (u8 CDF vs BF16 direct)
stream_signed_lens.rs → stream_bf16_lens.rs (output BF16 table, not i8)
```

### 3.3 Quality check

```
[ ] end_to_end_bf16.rs: Rumi↔Rumi ≠ Rumi↔TCP (discrimination works)
[ ] dual_bf16_experiment.rs: BF16 shows < 50% agreement with u8 CDF
[ ] Spearman ρ(BF16_distances, expert_ground_truth) > u8 CDF ρ
```

---

## PHASE 4: Bake BF16 Lenses (replace u8 HDR lenses)

**Owner:** thinking-engine data/
**Depends on:** Phase 1-3 validated
**Estimated:** 1 hour per model (streaming from HF, blocked in sandbox)

### 4.1 Stream and bake

```bash
# For each model:
cargo run --release --example stream_bf16_lens -- \
  jinaai/jina-reranker-v3-GGUF jina-reranker-v3-BF16.gguf

# Outputs:
#   data/jina-reranker-v3-BF16/distance_table_256x256.bf16  (128 KB)
#   data/jina-reranker-v3-BF16/codebook_index.u16           (304 KB)
```

### 4.2 New lens modules

```rust
// reranker_bf16_lens.rs:
pub static RERANKER_BF16_TABLE: &[u8; 256 * 256 * 2] =
    include_bytes!("../data/jina-reranker-v3-BF16/distance_table_256x256.bf16");

pub fn reranker_bf16_engine() -> BF16ThinkingEngine {
    let table: Vec<u16> = RERANKER_BF16_TABLE.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    BF16ThinkingEngine::from_bf16_cosines(&table, 256)
}
```

### 4.3 Quality check

```
[ ] BF16 table has REAL negative values (not CDF-symmetric)
[ ] Sign distribution matches raw cosine range (e.g. Reranker: ~50/50)
[ ] end_to_end test discriminates (tier 1 > tier 4)
[ ] Cronbach α across BF16 lenses similar to u8 lenses (~71% disagreement)
```

---

## PHASE 5: Calibration with Ground Truth

**Owner:** thinking-engine (calibration feature)
**Depends on:** Phase 4, candle forward pass
**Estimated:** 3-4 hours

### 5.1 Candle ground truth

```
Jina v5 safetensors → candle forward pass → f32 embeddings → cosine
= GROUND TRUTH for calibration
```

### 5.2 Spearman ρ per table type

```
For 1000 text pairs:
  ρ(u8_CDF_distances, ground_truth_cosines)   → expect < 0.50 (CDF destroys magnitude)
  ρ(BF16_distances, ground_truth_cosines)      → expect > 0.90 (preserves magnitude + sign)
  ρ(BF16_gate_Up_distances, ground_truth)      → expect > 0.95 (gate modulation helps)
```

### 5.3 ICC profiles where ρ < 0.998

```
Transfer curve: corrected = f(baked_distance, ground_truth)
Per centroid pair: boundary_risk from variance_agreement_scores()
```

---

## WHAT TO DELETE AFTER PHASE 3

```
signed_engine.rs:    REPLACE with bf16_engine.rs
dual_engine.rs:      REWRITE to compare u8 vs BF16
signed_domino.rs:    DELETE (dead code, never called)
stream_signed_lens.rs: REWRITE as stream_bf16_lens.rs

KEEP:
  engine.rs (u8, legacy compatibility)
  pooling.rs, builder.rs, cronbach.rs, auto_detect.rs
  tensor_bridge.rs, tokenizer_registry.rs, ground_truth.rs
  l4_bridge.rs (rewrite to use StackedN centroids)
  composite_engine.rs (update to support BF16)
  semantic_chunker.rs (update to use BF16 engine)
```

---

## DEPENDENCY GRAPH

```
Phase 1 (bf16_engine.rs)
  ↓
Phase 2 (gate modulation)     Phase 3 (wire existing modules)
  ↓                              ↓
Phase 4 (bake BF16 lenses)  ← needs HF streaming (outside sandbox)
  ↓
Phase 5 (calibration)       ← needs candle forward pass
```

Phase 1-3 can be done NOW. Phase 4 needs HF access. Phase 5 needs candle wiring.

---

## SUCCESS CRITERIA

```
1. end_to_end_bf16.rs: Rumi↔Rumi cos > Rumi↔TCP cos (discrimination)
2. Tier monotonicity: tier1 > tier2 > tier3 > tier4
3. Spearman ρ(BF16, ground_truth) > 0.90
4. Spearman ρ(BF16, ground_truth) > ρ(u8_CDF, ground_truth) by at least 0.20
5. Gate-modulated Up table differs from raw Up table by > 20% of entries
6. All 244+ tests still pass
```
