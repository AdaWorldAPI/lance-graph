# v0.2.0 — 7-Lane HighHeelBGZ Codebooks

Generated 2026-04-06 from safetensors weights on `claude/risc-thought-engine-TCZw7`.

## Models

| Model | Params | Hidden | Source | Cosine Range |
|-------|--------|--------|--------|--------------|
| qwen3-vl-embedding | 2B | 2048D | `Qwen/Qwen3-VL-Embedding-2B` safetensors (BF16) | [-0.846, 0.539] |
| jina-v5 | 0.6B | 1024D | `jinaai/jina-embeddings-v5` safetensors (BF16) | [-0.187, 0.675] |

## Forward Pass Discrimination (proven)

```
Qwen3-VL: Rumi<>Rumi cos=0.454, Rumi<>TCP cos=0.322, gap=0.132
Jina v5:  Rumi<>Rumi cos=0.512, Rumi<>TCP cos=0.384, gap=0.128
```

## 7 Encoding Lanes

| Lane | File | Type | Size | Purpose |
|------|------|------|------|---------|
| 1 | `distance_table_256x256.u8` | u8 | 64 KB | CDF percentile rank |
| 2 | `distance_table_256x256.i8` | i8 | 64 KB | round(cos*127), signs preserved |
| 3 | `distance_table_256x256.gamma_phi.u8` | u8 | 64 KB | Golden ratio redistribution |
| 4 | `distance_table_256x256.gamma_phi.i8` | i8 | 64 KB | Signed gamma+phi |
| 5 | `silu_deltas_256x256.f32` | f32 | 256 KB | SiLU correction (zeros for embed) |
| 6 | `distance_table_256x256.bf16` | bf16 | 128 KB | StackedN source precision |
| 7 | `spiral_drift_256x256.u8` | u8 | 64 KB | HighHeelBGZ reconstruction drift |

Plus: `cosine_matrix_256x256.f32` (raw), `codebook_index.u16` (assignments), `encoding_metadata.json`

## Usage in Dockerfile

```dockerfile
# Copy codebooks from repo
COPY releases/v0.2.0-7lane-codebooks/ /app/codebooks/

# Or download specific model
COPY releases/v0.2.0-7lane-codebooks/qwen3-vl-embedding-7lane/ /app/data/qwen3-vl-embedding-7lane/
```

## Encoder Command

```bash
cargo run --release --features calibration --example seven_lane_encoder \
  --manifest-path crates/thinking-engine/Cargo.toml \
  -- qwen3-vl-embedding   # or jina-v5
```

## Calibration Metadata (from encoding_metadata.json)

### Qwen3-VL-Embedding-2B
- gamma: 0.1677, phi_scale: 0.8458
- E/I ratio: 99.2% (pos=32368, neg=260, zero=12)
- BF16 max roundtrip error: 0.003827
- Spiral drift: stride=11, avg=0.057, max=0.331

### Jina v5
- gamma: 0.2261, phi_scale: 0.6753
- E/I ratio: 99.3% (pos=32407, neg=222, zero=11)
- BF16 max roundtrip error: 0.003862
- Spiral drift: stride=11, avg=0.096, max=0.434
