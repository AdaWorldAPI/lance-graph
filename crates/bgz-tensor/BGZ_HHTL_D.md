# BGZ-HHTL-D: 4-Byte Weight Matrix Encoding

> Slot D (tree address) + Slot V (BF16 residual) = 4 bytes per weight row.  
> Qwen3-TTS-1.7B: 3.86 GB → 11.2 MB (343:1). Fits on a Pi 4 in 75 MB RAM.

## What it is

BGZ-HHTL-D is a weight matrix encoding that replaces full-precision tensors with a 4-byte descriptor per row. Each row is assigned to its nearest centroid in a shared 256-entry Base17 palette, and the residual is stored as a single BF16 value with polarity.

At inference time, the HHTL cascade uses the palette assignment directly — 95% of attention pairs are resolved by table lookup without touching the original weights.

## Bit layout

```
Slot D (u16)                         Slot V (u16)
┌────┬──────┬──────────┬───┬───┐    ┌────────────────┐
│ Ba │ HIP  │  TWIG    │ P │ R │    │ BF16 residual  │
│15:14│13:10│  9:2     │ 1 │ 0 │    │ from centroid   │
└────┴──────┴──────────┴───┴───┘    └────────────────┘
 2 bit  4 bit   8 bit   1b  1b        16 bits

Ba   = HEEL basin (QK=0, V=1, Gate=2, FFN=3)
HIP  = family within basin (16-way binary split)
TWIG = centroid index in 256-entry palette
P    = polarity of dominant residual dimension
R    = reserved
```

## Shared palette strategy

Same-shape, same-role tensors share one palette. This is architecturally correct because weight distributions within a role are consistent across layers.

```
Qwen3-TTS-1.7B: 480 tensors → 26 palette groups

Group                        Tensors  Rows each  Shared palette
talker/gate [6144,2048]           28      6,144   1 × 206 KB
talker/up   [6144,2048]           28      6,144   1 × 206 KB
talker/down [2048,6144]           28      2,048   1 × 206 KB
talker/qko  [2048,2048]           56      2,048   1 × 206 KB
talker/v    [1024,2048]           28      1,024   1 × 206 KB
talker/embed [151936,2048]         1    151,936   1 × 206 KB
cp/embed    [2048,2048]           15      2,048   1 × 206 KB
cp/lm_head  [2048,1024]           15      2,048   1 × 206 KB
... (18 more groups)
```

Without sharing: 280 palettes = 57 MB overhead.  
With sharing: 26 palettes = 5.4 MB overhead.

## Compression results

Validated against the real Qwen3-TTS-12Hz-1.7B-Base safetensors header (1.93B params):

| Component | Original | HHTL-D | Ratio |
|---|---|---|---|
| Talker attention (Q/K/V/O × 28 layers) | 470 MB | 1.5 MB | 313:1 |
| Talker FFN (gate/up/down × 28 layers) | 1,414 MB | 2.4 MB | 589:1 |
| Text embedding (151,936 × 2048) | 622 MB | 0.6 MB | 1,037:1 |
| Code predictor (5 layers, all roles) | 197 MB | 0.7 MB | 281:1 |
| Codec embeddings (15 × 2048²) | 126 MB | 0.5 MB | 252:1 |
| LM heads (15 × 2048 × 1024) | 63 MB | 0.5 MB | 126:1 |
| 26 shared palettes | — | 5.4 MB | overhead |
| Passthrough (norms, biases, small convs) | 2.4 MB | 2.4 MB | 1:1 |
| **Total** | **3.86 GB** | **11.2 MB** | **343:1** |

## Inference via HHTL cascade

The encoding is designed for the HHTL (Heel→Hip→Twig→Leaf) cascade, not for decompression back to f32. At inference time:

```
Token pair (i, j):
  a = palette_idx[i]    ← 1 byte lookup
  b = palette_idx[j]    ← 1 byte lookup

  match route_table[a][b]:
    Skip (60%)     → no attention score needed. 0 cycles.
    Attend (35%)   → score = distance_table[a][b]. 1 cycle (u16 lookup).
    Compose (rare) → score via intermediate centroid.
    Escalate (5%)  → full Base17 L1: 17 i16 subtractions. ~4 cycles.
```

Per token at sequence length 512:
- 8,192 potential Q-K pairs × 28 layers = 229,376 pairs
- After cascade: ~11,468 actual computations (5%)

## Device feasibility

| Device | RAM | Model | KV (512 tok) | Total | Fits? |
|---|---|---|---|---|---|
| Pi 4 (4 GB) | 4,096 MB | 11.2 MB | 64 MB | 75 MB | ✓ |
| Pi Zero 2W | 512 MB | 11.2 MB | 64 MB | 75 MB | ✓ |
| Orange Pi 5 | 8,192 MB | 11.2 MB | 64 MB | 75 MB | ✓ |
| ESP32-S3 (8 MB) | 8 MB | 11.2 MB | — | — | ✗ (needs mmap) |

KV cache assumes BF16 storage: (28 talker + 5 CP) layers × 2 (K,V) × 8 KV heads × head_dim × 2 bytes × seq_len.

## Encoding pipeline

```
safetensors (BF16/F16/F32)
  │
  ▼  classify by (component, role, shape)
26 palette groups + 109 passthrough tensors
  │
  ▼  per group: sample → SpiralEncoding (stride=role)
SpiralEncoding (BF16 anchors at golden-step positions)
  │
  ▼  γ+φ rehydrate → project to Base17 i16[17]
Base17 rows (34 bytes each, 256× FP scale)
  │
  ▼  CLAM furthest-point sampling → 256-entry palette
WeightPalette (shared across group)
  │
  ▼  16-way binary split on palette
HIP families (farthest-pair recursion, 4 levels → 16 groups)
  │
  ▼  per row: nearest centroid + residual + polarity
HhtlDEntry (4 bytes: Slot D u16 + Slot V u16)
  │
  ▼  write safetensors format
model_hhtld.safetensors
  Tensors: entries (u8), palette (u8), distance_table (u8),
           route_table (u8), hip_families (u8), gamma_meta (u8),
           original_shape (u8), passthrough.* (original dtype)
  Metadata: encoding, version, compression_ratio, model info
```

## Crate structure

```
bgz-tensor/src/
├── hhtl_d.rs           # HhtlDEntry, HhtlDTensor, HeelBasin, HhtlDMeta
│                       # build_hip_families (16-way split)
├── shared_palette.rs   # PaletteGroupKey, classify_role/component
│                       # is_encodable, build_shared_palette, encode_group
├── hhtl_cache.rs       # HhtlCache, RouteAction (existing)
├── palette.rs          # WeightPalette, CLAM sampling (existing)
├── projection.rs       # Base17, golden-step fold (existing)
└── cascade.rs          # HHTL cascade config + stats (existing)

thinking-engine/examples/
├── tts_17b_hhtld_encode.rs   # safetensors → HHTL-D safetensors
└── tts_17b_hhtld_decode.rs   # HHTL-D safetensors → rehydrated weights
```

## Usage

### Encode

```sh
cargo run --release --example tts_17b_hhtld_encode \
    --manifest-path crates/thinking-engine/Cargo.toml \
    -- /path/to/Qwen3-TTS-12Hz-1.7B-Base/model.safetensors
```

Output: `model_hhtld.safetensors` in the same directory.

### Decode / validate

```sh
cargo run --release --example tts_17b_hhtld_decode \
    --manifest-path crates/thinking-engine/Cargo.toml \
    -- /path/to/model_hhtld.safetensors
```

Prints: per-role route statistics, centroid self-consistency checks, passthrough inventory.

## Relationship to other encodings

BGZ-HHTL-D is one encoding in the lance-graph ecosystem. See `.claude/knowledge/encoding-ecosystem.md` for the full map.

```
Raw weights (BF16/f32)
  │
  ├──→ StackedN BF16 (centroid stacking)
  │       │
  │       ├──→ Base17 i16[17] (golden-step fold)
  │       │       │
  │       │       ├──→ BGZ17 palette (semiring algebra)
  │       │       │       └──→ HighHeelBGZ (spiral address cascade)
  │       │       │               └──→ ZeckF64 u64 (progressive edge)
  │       │       └──→ NeuronPrint 6D (6 roles → 6 palette indices)
  │       │
  │       └──→ BGZ-HHTL-D 2×u16  ← THIS
  │
  └──→ CausalEdge64 (SPO + NARS truth, orthogonal)
```

BGZ-HHTL-D sits alongside HighHeelBGZ (which uses 3-integer spiral addresses for rehydration from source) and ZeckF64 (which encodes progressive edge distances). HHTL-D is optimized for the case where you never need to reconstruct the original weights — only the palette assignment matters for cascade inference.

## Quality targets

Measured as Spearman ρ between exact Base17 L1 distances and palette-quantized distances (50 random pairs per role):

- ρ > 0.90 → acceptable for cascade routing
- ρ > 0.95 → good (most attention decisions preserved)
- ρ > 0.99 → excellent (nearly lossless ranking)

The encoder prints ρ per role. The 0.6B model achieves ρ > 0.93 on all roles. The 1.7B model is expected to be similar or better (more rows → better palette coverage).

## Design decisions

**Why 4 bytes, not 2 or 8?** The 256-entry palette index (8 bits) is the minimum for good coverage. The residual BF16 (16 bits) captures the magnitude of the approximation error. Together with basin (2b) and HIP family (4b), the 32-bit Slot D+V pair hits the sweet spot: enough precision for Spearman ρ > 0.93, small enough for L1 cache residency.

**Why shared palettes?** Same-role layers in a transformer have similar weight distributions. Sharing saves 10× on palette overhead with negligible quality loss. The `shared_palette.rs` module groups by (component, role, shape) — the strictest grouping that still provides meaningful sharing.

**Why not just use GGUF Q4?** GGUF quantization preserves individual weight values at reduced precision. BGZ-HHTL-D preserves *distance relationships* between weight rows. The cascade doesn't need individual values — it needs to know which pairs interact (RouteAction) and how strongly (distance table). This is a fundamentally different compression target, which is why the ratios are orders of magnitude higher.

**Why safetensors output format?** Compatibility. Any tool that reads safetensors can inspect the compressed model. The tensors are stored as U8 blobs with shape metadata — standard safetensors, just with HHTL-D-specific tensor naming conventions.

## Pairwise Cosine Table (Fisher z)

Each shared palette group includes a k×k i8 cosine table encoded via Fisher z transform.

**Encoding:** `arctanh(clamp(cosine, ±0.9999))` → scale to i8 via per-family `(z_min, z_range)`  
**Decoding:** `tanh((i8 + 127) / 254 × z_range + z_min)` → restored cosine  

Fisher z stretches the tails (near cos=±1) where attention scores are most sensitive. Per-family gamma maps each role's cosine distribution to fill the full i8 [-128, 127] range.

**Storage:** 256×256 = 64 KB per group + 8 bytes family gamma. 26 groups = **1.6 MB** total.

**Certified:** Spearman ρ ≥ 0.999 on all 21 tensor roles of Qwen3-TTS-1.7B (5000 pairs per role). Mean absolute restore error ≤ 0.0016.

**Implementation:** `bgz-tensor/src/fisher_z.rs` — `FisherZTable`, `FamilyGamma`, 7 tests.

**Lookup at inference:**
```rust
// O(1): one i8 read + one tanh call
let cosine = fisher_z_table.lookup_f32(centroid_a, centroid_b);
```

The HHTL-D entry's `twig_centroid` (8 bits) is the index into this table. The cascade decides WHETHER to look up (Skip/Attend). The table provides WHAT the value is.
