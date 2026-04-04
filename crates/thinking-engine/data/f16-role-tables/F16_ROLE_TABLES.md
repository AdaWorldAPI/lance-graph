# F16 1:1 Per-Role Distance Tables

24 tables, 77 MB total. Built from F16 GGUF weights with ndarray SIMD cosine (F64x8 FMA) + rayon parallel.

## Models

| Model | Roles | Widest cosine range | Layer |
|-------|-------|--------------------:|-------|
| BGE-M3 | Q,K,V,O,Up,Down | [-0.785, 0.749] | 23 |
| reader-lm 1.5B | Q,K,V,O,Gate,Up,Down | [-0.885, 0.816] | 27 |
| Jina v3 | QKV,Q,O,Up,Down | [-0.645, 0.530] | 23 |
| MiniLM | Q,K,V,O,Up,Down | [-0.677, 0.752] | 5 |

## Key Findings

- **Q8_0 produces cos[0,0]** — quantization destroys topology. F16/BF16 required.
- **reader-lm attn_output** has widest spread: cos[-0.866, 0.649]
- **reader-lm ffn_down** most asymmetric: cos[-0.885, 0.188] (negative-heavy)
- **Jina QKV fused** — attn_q and attn_qkv produce identical tables (same tensor)
- **ffn_gate/up** near-symmetric around 0: cos[-0.095, 0.336] / [-0.133, 0.131]

## Rebuild

```bash
# Download F16 GGUFs to /tmp/hf_cache/ first
cargo run --release --manifest-path crates/thinking-engine/Cargo.toml --example build_1to1_roles
```

Outputs to `/tmp/codebooks/<model>-roles-f16/<role>/distance_table_NxN.u8`

## Table Format

- Raw u8 row-major: `table[i * N + j]` = quantized cosine between row i and row j
- Encoding: `u8 = ((cosine + 1) / 2) * 255` (linear, 256 levels)
- Diagonal = 255 (self-similarity)
- Each role dir also has `meta.json` with tensor name, dtype, cosine range, SHA256
