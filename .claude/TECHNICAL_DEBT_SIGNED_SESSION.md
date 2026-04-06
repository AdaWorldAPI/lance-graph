# TECHNICAL DEBT REVIEW — Signed Session (2026-04-06)

## VERDICT: 13 modules built, ~4100 lines. Half is useful, half bypasses existing architecture.

---

## WHAT BYPASSES THE EXISTING ARCHITECTURE

### The existing pipeline (bgz-tensor + highheelbgz):
```
GGUF BF16 → StackedN::from_bf16(weights, SPD=32)     # golden-step folded BF16
  → ClamCodebook::build_cosine(stacked, k=256)         # CLAM on StackedN cosine
  → CodebookEntry { stacked, population, radius }       # BF16 centroids
  → entry_i.stacked.cosine(&entry_j) → f64              # SIMD cosine (F64x8)
  → f32_to_bf16(cos) → BF16 distance table              # 128 KB for 256×256
  → highheelbgz: i16 HEEL + i16 HIP = i32 CLAM address  # φ-stride addressing
  → CoarseBand: Foveal/Near/Maybe/Reject                 # cascade routing
```

### What I built INSTEAD (ignoring the above):
```
GGUF BF16 → raw f32 vectors (NO StackedN, NO golden-step folding)
  → CLAM on raw vectors (duplicating ClamCodebook::build_cosine)
  → raw cosine → CDF percentile → u8[0,255] (NOT BF16)
  → u8 - 128 → i8 (calling it "signed" but it's CDF rank relabeling)
  → ThinkingEngine MatVec on i8 (cos=1.000 for all inputs = BROKEN)
```

### The damage:
- `signed_engine.rs`: builds a parallel engine that ignores StackedN/BF16/CLAM
- `stream_signed_lens.rs`: duplicates the GGUF streaming + CLAM pipeline from scratch
- `dual_engine.rs`: compares two broken pipelines against each other
- `from_unsigned()`: relabels CDF ranks as signs (WARNING added but fundamental flaw)
- End result: cos=1.000 for ALL text pairs. Zero discrimination. Confirmed broken.

---

## WHAT IS ACTUALLY USEFUL (keep)

| Module | Lines | Why useful |
|--------|-------|-----------|
| `pooling.rs` | 386 | ArgMax/Mean/TopK/Nucleus — clean, correct, needed for any engine |
| `builder.rs` | 507 | Fluent API, Temperature, CommitSinks — good design pattern |
| `cronbach.rs` | 308 | cronbach_alpha() formula correct, variance_agreement_scores() useful |
| `auto_detect.rs` | 265 | Config.json routing — correct, model-agnostic |
| `tensor_bridge.rs` | 209 | F32/I8/U8 conversions — correct, needed for calibration |
| `tokenizer_registry.rs` | 333 | 8 model tokenizers — real BPE, works |
| `ground_truth.rs` | 277 | DTOs for calibration — placeholder but correct design |

**Total useful: ~2,285 lines (56%)**

## WHAT NEEDS REWRITE TO USE EXISTING ARCHITECTURE

| Module | Lines | What's wrong |
|--------|-------|-------------|
| `signed_engine.rs` | 532 | Should use StackedN BF16 cosine, not raw f32. Should output BF16 table, not i8. |
| `dual_engine.rs` | 209 | Compares u8 CDF vs i8 CDF-relabeled. Neither is correct. |
| `composite_engine.rs` | 242 | Should compose StackedN codebooks, not raw ThinkingEngines. |
| `l4_bridge.rs` | 264 | Uses table rows as centroid proxies. Should use StackedN centroids. |
| `signed_domino.rs` | 258 | Dead code. Never called. Cascade should use BF16 tables. |
| `semantic_chunker.rs` | 326 | Untested on real text. Convergence-jump is good IDEA, needs real data. |

**Total needs rewrite: ~1,831 lines (44%)**

## WHAT SHOULD THE SIGNED ENGINE ACTUALLY LOOK LIKE

```rust
// CORRECT: use StackedN BF16 pipeline
use bgz_tensor::stacked_n::{StackedN, ClamCodebook, bf16_to_f32, f32_to_bf16};

pub struct BF16DistanceTable {
    /// BF16 distance values. Sign IS the gate decision.
    /// 256×256 × 2 bytes = 128 KB.
    table: Vec<u16>,  // BF16 bit patterns
    n: usize,
}

impl BF16DistanceTable {
    /// Build from ClamCodebook: pairwise cosine between StackedN centroids.
    pub fn from_codebook(codebook: &ClamCodebook) -> Self {
        let n = codebook.entries.len();
        let mut table = vec![0u16; n * n];
        for i in 0..n {
            table[i * n + i] = f32_to_bf16(1.0); // self = max
            for j in (i+1)..n {
                let cos = codebook.entries[i].stacked.cosine(&codebook.entries[j].stacked);
                let bf16 = f32_to_bf16(cos as f32);
                table[i * n + j] = bf16;
                table[j * n + i] = bf16;
            }
        }
        Self { table, n }
    }

    /// Build GATE-MODULATED table: silu(gate) × role before cosine.
    /// Uses StackedN centroids from BOTH gate and role codebooks.
    pub fn from_gate_modulated(
        gate_codebook: &ClamCodebook,
        role_codebook: &ClamCodebook,
    ) -> Self {
        // silu(gate_centroid) ⊙ role_centroid → cosine → BF16
        todo!("needs gate + role StackedN centroids from same GGUF stream")
    }

    /// Cycle with BF16 precision.
    pub fn cycle(&self, energy: &mut [f32]) {
        let n = self.n;
        let mut next = vec![0.0f32; n];
        for i in 0..n {
            if energy[i].abs() < 1e-10 { continue; }
            let row = &self.table[i * n..(i + 1) * n];
            for j in 0..n {
                // BF16 → f32 is lossless (bit shift)
                let dist = bf16_to_f32(row[j]);
                // SIGNED: positive excites, negative inhibits
                next[j] += dist * energy[i];
            }
        }
        // Clamp negative energy (inhibited atoms die)
        for e in &mut next { *e = e.max(0.0); }
        // Normalize
        let total: f32 = next.iter().sum();
        if total > 1e-10 { for e in &mut next { *e /= total; } }
        energy.copy_from_slice(&next);
    }
}
```

## THE CRITICAL MISUNDERSTANDING

I treated the distance table values as the PRIMARY encoding decision.
The actual architecture treats them as a CONSEQUENCE of the StackedN/CLAM pipeline.

```
WRONG (what I did):
  "Should the table be u8 or i8?" → built two engines, compared them

RIGHT (what the architecture says):
  "The table is BF16 because the centroids are BF16 (StackedN)."
  "The sign comes from the cosine, which comes from the BF16 weights."
  "The gate modulation happens BEFORE CLAM, on the StackedN level."
  "highheelbgz addresses the table. bgz-tensor encodes the centroids."
  "The table precision matches the centroid precision. Always."
```

## WHAT TO DO NEXT

```
1. DO NOT build more i8/u8 alternatives. BF16 is the answer.
2. Wire signed_engine to use StackedN cosine → BF16 table (not raw f32 → i8).
3. Wire l4_bridge to use StackedN centroids (not table rows).
4. Wire stream_signed_lens to use StackedN::from_bf16 + ClamCodebook::build_cosine.
5. Keep: pooling, builder, cronbach, auto_detect, tensor_bridge, tokenizer_registry.
6. Rewrite: signed_engine, dual_engine, composite_engine, l4_bridge.
7. Delete: signed_domino (dead code, never called).
8. Test with REAL data: stream Reranker GGUF → StackedN → BF16 table → engine.
```
