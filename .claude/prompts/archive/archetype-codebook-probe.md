# Archetype Codebook: Weight Rows as Vocabulary

## Core insight

Weight rows are NOT data to compress. They are FUNCTIONS to index.

Audio codecs don't compress waveforms — they look up spectral archetypes 
and transmit indices. Weight codecs should do the same: look up behavioral 
archetypes and transmit indices.

The 1024D weight row has a semantic identity — what it DOES in the 
transformer. Two rows that do similar things share an archetype. The 
archetype codebook IS the vocabulary of behaviors.

## The RVQ parallel

The Qwen3-TTS model already uses this for audio:
```
Codec level 0: 2048 archetypes → pitch/energy (fundamental)
Codec level 1: 2048 archetypes → formant detail (residual of L0)
Codec level 2: 2048 archetypes → texture (residual of L1)
...15 levels: each captures what the previous missed
```

We do the same for weights:
```
L0:  256 archetypes  → HEEL basin (which kind of computation)
L1:  512 archetypes  → HIP family (which sub-function)
L2: 1024 archetypes  → TWIG centroid (which specific pattern)
L3: 4096 archetypes  → LEAF (fine discrimination)
L4: 17K archetypes   → Base17 space (full resolution)
L5: 64K archetypes   → exhaustive
```

Each level operates on the RESIDUAL of the previous.
Row = L0_centroid[idx0] + L1_centroid[idx1] + ... + Ln_centroid[idxN]

Storage per row: 6-12 bytes of indices.
Codebook: shared across all 28 layers of the same role.

## Task

### Step 1: Run the probe

```sh
cargo run --release --example archetype_codebook_probe \
    --manifest-path crates/thinking-engine/Cargo.toml \
    -- ~/models/Qwen3-TTS-12Hz-1.7B-Base/model.safetensors
```

### Step 2: Find the cos=0.999 threshold

The probe measures progressive reconstruction quality. Find where
row cosine first exceeds 0.999 for each role. That's the minimum
codebook depth for surviving 33 transformer layers.

Expected outcomes:
- If L2 (1024) reaches 0.999: codebook ~8 MB, indices 3B/row → 8.5 MB total
- If L3 (4096) reaches 0.999: codebook ~32 MB, indices 6B/row → 32.3 MB total
- If L4 (17K) reaches 0.999: codebook ~140 MB → too large, need different approach
- If L5 (64K) doesn't reach 0.999: RVQ can't represent these weights → fallback to Q8

### Step 3: WAV test at the threshold level

Once you know the minimum level, run TTS inference with codebook-reconstructed
weights at that level. The matryoshka-wav-test.md prompt has the full protocol.

### Step 4: Compare storage

| Approach | Storage | row cos | WAV quality |
|---|---|---|---|
| Raw BF16 | 3.86 GB | 1.000 | reference |
| GGUF Q8 | ~1.9 GB | ~0.999 | excellent |
| GGUF Q4 | ~0.9 GB | ~0.995 | good |
| RVQ L3 (4096) | ~32 MB | 0.999? | ? |
| RVQ L2 (1024) | ~8 MB | 0.99? | ? |
| HHTL-D only | 11 MB | 0.07 | silence |

The archetype codebook could land between GGUF Q4 and HHTL-D:
much smaller than GGUF, actually functional (unlike pure HHTL-D).

## Connection to Jina

For text models: the weight row's receptive field (the input pattern that 
maximizes its activation) can be Jina-embedded. The Jina embedding IS the 
semantic identity. Two rows with similar Jina embeddings do similar things.

For TTS models: the equivalent is spectral — what frequency pattern does 
this row respond to? The Opus band decomposition of the receptive field 
gives the "audio Jina embedding."

In both cases, the codebook entry IS the archetype — the prototypical 
function that a cluster of rows all approximate. The residual levels 
capture how each specific row deviates from its archetype.

## Key files

```
archetype_codebook_probe.rs          ← THE PROBE (run this)
bgz-tensor/src/matryoshka.rs         ← SVD-ordered variable bit allocation
bgz-tensor/src/hhtl_d.rs             ← HHTL-D routing (Skip/Attend)
bgz-tensor/src/shared_palette.rs     ← palette grouping
bgz-tensor/src/fisher_z.rs           ← Fisher z scoring
.claude/prompts/matryoshka-wav-test.md ← WAV end-to-end protocol
```

## Agent routing

- **archaeologist**: reads the probe output, identifies which levels matter
- **truth-architect**: interprets the cos=0.999 threshold, relates to 
  error budget math
- **family-codec-smith**: if L3 doesn't reach 0.999, experiments with 
  codebook construction strategies (weighted CLAM, k-means++, etc.)
- **integration-lead**: if RVQ works, replaces HHTL-D Slot V with RVQ 
  indices in the safetensors output format
