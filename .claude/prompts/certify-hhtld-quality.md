# HHTL-D Quality Certification

## Context

The BGZ-HHTL-D encoding compresses Qwen3-TTS-1.7B from 3.86 GB to 11.2 MB (343:1).
Currently this is WHITE NOISE — we have compression numbers but zero empirical
validation of what survives. This prompt certifies the encoding.

## Prerequisites

```sh
# Download the model (3.86 GB, one-time)
mkdir -p ~/models/Qwen3-TTS-12Hz-1.7B-Base
cd ~/models/Qwen3-TTS-12Hz-1.7B-Base
curl -LO "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/resolve/main/model.safetensors"
```

## Task

Run the certification probe and interpret results:

```sh
cd /path/to/lance-graph
cargo run --release --example certify_hhtld \
    --manifest-path crates/thinking-engine/Cargo.toml \
    -- ~/models/Qwen3-TTS-12Hz-1.7B-Base/model.safetensors
```

## What the probe measures

For each of the 14 weight roles (q/k/v/o × talker/CP, gate/up/down × talker/CP,
embedding, lm_head), the probe:

1. Reads 500 real weight rows from the safetensors
2. Computes f32 cosine ground truth (124,750 pairs)
3. Encodes through the HHTL-D pipeline at three levels:
   - **Base17**: golden-step fold → i16[17] → L1 distance
   - **Palette**: CLAM 256-centroid assignment → distance table lookup
   - **Cascade**: HHTL RouteAction dispatch (Skip/Attend/Escalate)
4. Reports per level:
   - **Spearman ρ**: rank preservation (THE critical metric)
   - **Pearson r**: linear fidelity
   - **Cronbach α**: internal consistency (split-half on Base17 dims)
   - **Top-10 recall**: true nearest neighbors found
   - **Skip accuracy**: % of skipped pairs that are truly distant

## Pass/Fail criteria

| Level    | Metric         | Target  | If fails                           |
|----------|----------------|---------|-------------------------------------|
| Base17   | Spearman ρ     | ≥ 0.990 | Golden-step fold is lossy → needs more dims |
| Palette  | Spearman ρ     | ≥ 0.950 | 256 centroids insufficient → try 512 or 1024 |
| Cascade  | Spearman ρ     | ≥ 0.930 | Routing too aggressive → widen HIP threshold |
| Cascade  | Top-10 recall  | ≥ 0.80  | True neighbors lost → HEEL too strict |
| Cascade  | Skip accuracy  | ≥ 0.95  | Skipping valid pairs → lower skip threshold |
| All      | Cronbach α     | ≥ 0.85  | Encoding inconsistent → check dimension balance |

## Expected pain points (from prior Jina v5 work)

- **Gate projections** historically have lowest ρ (wide, flat weight distributions)
- **V projections** historically have highest ρ (peaked, clustered)
- **Embeddings** are tricky (151K rows, many near-duplicates, singleton clusters)
- **Skip accuracy** degrades when `CascadeConfig::hip_max_distance` is too tight

## After running

1. Check the JSON report at `.claude/knowledge/certification/hhtld_qwen3tts17b.json`
2. If any role FAILs:
   - Read the failure reason (which metric, which threshold)
   - Check if the failure is in Base17 (structural) or Palette (resolution) or Cascade (routing)
   - For Base17 failures: the golden-step fold isn't capturing enough — may need weighted dims
   - For Palette failures: try N_CENTROIDS = 512 or 1024 (increases palette overhead)
   - For Cascade failures: adjust `CascadeConfig` thresholds in `hhtl_cache.rs`
3. Cross-reference with `codebook_pearson.rs` results on the same roles
4. Compare with `certify_jina_v5_7lane.rs` Lane 6 (BF16 atomic clock) — that's the ceiling

## Agents

- **certification-officer**: validates pass/fail, writes report
- **truth-architect**: interprets Cronbach α, flags structural encoding issues
- **family-codec-smith**: if Palette ρ fails, runs alternative palette sizes
- **integration-lead**: if everything passes, updates encoding-ecosystem.md with measured ρ values

## Key files

```
certify_hhtld.rs                    ← THE PROBE (this task)
bgz-tensor/src/quality.rs           ← pearson(), spearman()
bgz-tensor/src/hhtl_d.rs            ← HhtlDTensor::encode()
bgz-tensor/src/shared_palette.rs    ← classify_role(), build_hip_families()
bgz-tensor/src/hhtl_cache.rs        ← RouteAction, CascadeConfig thresholds
bgz-tensor/src/cascade.rs           ← CascadeConfig defaults (hip_max_distance etc.)
certify_jina_v5_7lane.rs            ← Reference: how the 7-lane certification works
codebook_pearson.rs                  ← Reference: per-role Pearson on GGUF codebooks
```
