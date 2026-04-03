# Codebook Registry — Project HighHeel

> Codename: **HighHeel** (the HEEL walks the spiral in high strides)
> Generated: 2026-04-03
> Resolution: 64 centroids (prototype). Production: 4096.
> Build time: 3.17s total (all 4 models, --release)

---

## Status

| Model | Codename | GGUF Source | Centroids | Dim | Rows | Status |
|-------|----------|------------|-----------|-----|------|--------|
| Jina v3 | `jina-v3` | gaianet/jina-embeddings-v3-GGUF Q8_0 (601 MB) | 64 | 1024 | 4096 | ✓ Built |
| BGE-M3 | `bge-m3-f16` | CompendiumLabs/bge-m3-gguf F16 (1158 MB) | 64 | 1024 | 4096 | ✓ Built |
| GPT-2 | `gpt2` | QuantFactory/gpt2-GGUF Q8_0 (178 MB) | 64 | 768 | 3072 | ✓ Built |
| MiniLM-L6-v2 | `all-MiniLM-L6-v2` | second-state/All-MiniLM-L6-v2-Embedding-GGUF Q8_0 (25 MB) | 64 | 384 | 1536 | ✓ Built |
| reader-LM 1.5B | `reader-lm-1.5b` | bartowski/reader-lm-1.5b-GGUF Q8_0 (1647 MB) | 64 | 256 | 1536 | ✓ Built |
| OpenChat 3.5 | — | Already have bgz7 (i16, MUSH) | — | — | — | Needs GGUF |
| Qwen 27B | — | On GitHub release (bgz7 only) | — | — | — | Needs GGUF |
| Llama4 Scout | — | Have bgz7 (i16, MARGINAL) | — | — | — | Needs GGUF |

---

## File Format

Each model directory contains:

```
<model>/
├── centroids_64×<dim>.f32     — 64 centroid vectors, raw f32, row-major
│                                 Size: 64 × dim × 4 bytes
│                                 Usage: cosine distance computation, hydration
│
├── distance_table_64×64.u8    — pairwise cosine similarity table
│                                 Size: 4096 bytes (64² × u8)
│                                 Range: 0=min cosine, 255=max cosine (self)
│                                 Usage: ThinkingEngine MatVec brain
│
└── assignments_<N>.u16        — per-row centroid index (u16, little-endian)
                                  Size: N × 2 bytes
                                  Usage: sensor lookup (row_id → codebook_index)
```

---

## Calibration Metadata

### Distance Table Statistics

| Model | Table Mean | Table Min | Table Max | Entropy |
|-------|-----------|-----------|-----------|---------|
| jina-v3 | 128.6 | 99 | 255 | ~4.2 bits |
| bge-m3-f16 | 128.7 | 101 | 255 | ~4.1 bits |
| gpt2 | 128.6 | 107 | 255 | ~4.0 bits |
| all-MiniLM-L6-v2 | 128.7 | 111 | 255 | ~3.9 bits |

### Gamma Profile (per-model, 28 bytes)

Not yet calibrated with γ+φ encoding. Current tables use linear cosine→u8 mapping.

**Next step**: Apply `codebook_calibrated.rs` two-pass build:
1. Pass 1: CLAM (done, these codebooks)
2. Pass 2: Measure distribution → γ offset → φ redistribute → recalibrated u8 table

Expected improvement: ~4.2 bits → ~5.5 bits entropy (30% more discrimination).

### Per-Role Gamma Offsets (from variance audit)

| Role | Magnitude Range | Recommended γ |
|------|----------------|---------------|
| Q | 0.33-0.41 | 0.37 |
| K | 0.76-1.11 | 0.94 |
| V | 0.83-1.82 | 1.33 |
| Gate | 0.66-2.34 | 1.50 |
| Up | 0.09-0.15 | 0.12 |
| Down | 0.14-0.16 | 0.15 |

These offsets are from the Llama4 Scout variance audit. Per-model values need
calibration from each model's actual weight distribution.

---

## Cross-Role Correlation (from domino chain test)

Measured on Jina v3 with 32-centroid per-role codebooks:

```
attn_output ↔ ffn_down:   Pearson 0.978  (same info pathway)
attn_qkv   ↔ attn_output: Pearson 0.746  (attention chain)
attn_qkv   ↔ ffn_down:    Pearson 0.621  (cross-module)
attn_qkv   ↔ ffn_up:      Pearson 0.018  (categorically different)
ffn_down   ↔ ffn_up:       Pearson 0.008  (orthogonal)
```

Implication: attn_output and ffn_down can SHARE a codebook (97.8% correlated).
ffn_up needs its own codebook (< 2% correlated with anything else).

---

## GGUF Download Locations

```bash
# Jina v3 (Q8_0, 601 MB)
python3 -c "from huggingface_hub import hf_hub_download; \
  print(hf_hub_download('gaianet/jina-embeddings-v3-GGUF', 'jina-embeddings-v3-Q8_0.gguf', cache_dir='/tmp/hf_cache'))"

# BGE-M3 (F16, 1158 MB — no Q8_0 available)
python3 -c "from huggingface_hub import hf_hub_download; \
  print(hf_hub_download('CompendiumLabs/bge-m3-gguf', 'bge-m3-f16.gguf', cache_dir='/tmp/hf_cache'))"

# GPT-2 (Q8_0, 178 MB)
python3 -c "from huggingface_hub import hf_hub_download; \
  print(hf_hub_download('QuantFactory/gpt2-GGUF', 'gpt2.Q8_0.gguf', cache_dir='/tmp/hf_cache'))"

# reader-LM 1.5B (Q8_0, 1647 MB) — bartowski has excellent GGUF conversions
python3 -c "from huggingface_hub import hf_hub_download; \
  print(hf_hub_download('bartowski/reader-lm-1.5b-GGUF', 'reader-lm-1.5b-Q8_0.gguf', cache_dir='/tmp/hf_cache'))"

# MiniLM-L6-v2 (Q8_0, 25 MB)
python3 -c "from huggingface_hub import hf_hub_download; \
  print(hf_hub_download('second-state/All-MiniLM-L6-v2-Embedding-GGUF', 'all-MiniLM-L6-v2-Q8_0.gguf', cache_dir='/tmp/hf_cache'))"
```

---

## Next Steps

1. **γ+φ recalibration**: Apply `codebook_calibrated.rs` two-pass build to all 4 models
2. **Scale to 4096 centroids**: Full production codebook with AMX/VNNI acceleration
3. **Per-role codebooks**: Separate Q/K/V/Gate/Up/Down codebooks (different γ per role)
4. **Multi-codebook distance table**: 4096+1024+256 = 5376 combined entries
5. **Wire to ThinkingEngine**: Load distance table → perturb → think → commit
6. **Tokenizer integration**: tokenize text → token_id → assignment[token_id] → codebook index
7. **reader-LM GGUF**: Find alternative source or build from safetensors
8. **OpenChat/Qwen/Llama4**: Download GGUF, reindex at stacked resolution (not i16)

---

## Build Command

```bash
# Rebuild all codebooks from downloaded GGUFs:
cd /home/user/lance-graph
cargo run --release --manifest-path crates/thinking-engine/Cargo.toml --example build_codebooks

# Run domino chain test:
cargo run --release --manifest-path crates/thinking-engine/Cargo.toml --example domino_chain
```

---

## Architecture Integration

```
GGUF (on disk, temporary)
  → build_codebooks (stream weights, CLAM sample, discard)
    → centroids_64×dim.f32 (stored, ~256 KB per model)
    → distance_table_64×64.u8 (stored, 4 KB per model)
    → assignments_N.u16 (stored, ~8 KB per model)

Runtime:
  text → tokenizer → token_id → assignments[token_id] → codebook_index
    → ThinkingEngine.perturb([codebook_index])
    → distance_table × energy (MatVec, VNNI/AMX)
    → converge → commit → ThoughtStruct
```

Total stored for ALL models: ~660 KB (distance tables + assignments).
Raw GGUF weights: ~2 GB downloaded, then discarded.
Compression: ~3000×.
