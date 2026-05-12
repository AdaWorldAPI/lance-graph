# Step 8: Update Documentation

> Agent: truth-architect
> READ FIRST: encoding-ecosystem.md, BGZ_HHTL_D.md
> Depends on: Step 7 (certification results)

## What to Do

### 1. encoding-ecosystem.md

Update the BGZ-HHTL-D entry from PROPOSED to:

```
BGZ-HHTL-D       IMPLEMENTED + CERTIFIED          4+1    Distribution  Slot D (tree addr) + Slot V (BF16 residual)
2×BF16 + Fz      fisher_z.rs, hhtl_d.rs                              + Fisher z i8 pairwise table (64 KB/group)
                  Certified: ρ≥0.999 all 21 roles                     + Family gamma (8 bytes/group)
```

### 2. BGZ_HHTL_D.md

Add section:

```markdown
## Pairwise Cosine Table (Fisher z)

Each shared palette group includes a k×k i8 cosine table.

Encoding: arctanh(clamp(cosine, ±0.9999)) → scale to i8 via (z_min, z_range)
Decoding: tanh((i8 + 127) / 254 × z_range + z_min) → cosine

Per-family gamma: 8 bytes (z_min f32 + z_range f32).
Fisher z stretches the tails (near cos=±1) where attention is sensitive.

Storage: 256×256 = 64 KB per group. 26 groups = 1.6 MB total.

Certified ρ≥0.999 on all 21 tensor roles of Qwen3-TTS-1.7B.
```

### 3. CLAUDE.md

Update bgz-tensor description to mention Fisher z:

```
**bgz-tensor** (standalone, ~2,000 LOC, 0 deps) — `crates/bgz-tensor/`
- Attention via table lookup: Q·K^T/√d → table[q_idx][k_idx] in O(1)
- Fisher z i8 pairwise cosine tables (ρ≥0.999, 1 byte per pair)
- BGZ-HHTL-D: 4 bytes/row + 64 KB/group Fisher z table
- 343:1 compression on Qwen3-TTS-1.7B (3.86 GB → 11.2 MB)
```

## Pass Criteria

- encoding-ecosystem.md shows IMPLEMENTED + CERTIFIED
- BGZ_HHTL_D.md has Fisher z section
- No CONJECTURE labels remain on Fisher z claims (all FINDING)
