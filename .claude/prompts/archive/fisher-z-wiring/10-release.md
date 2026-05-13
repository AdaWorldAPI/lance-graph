# Step 10: GitHub Release v0.2.0-hhtld-fisherz

> Agent: integration-lead
> Depends on: Step 9 (baked archive)

## What to Do

Create GitHub release `v0.2.0-hhtld-fisherz` on AdaWorldAPI/lance-graph.

### Release assets:

1. `qwen3-tts-1.7b-hhtld.tar.gz` — full archive (≤15 MB)
2. `model_hhtld.safetensors` — standalone encoded model
3. `manifest.json` — group metadata + certification
4. `certification.json` — full probe results

### Release notes:

```markdown
## BGZ-HHTL-D with Fisher z Pairwise Tables

Qwen3-TTS-12Hz-1.7B-Base compressed to 11.2 MB (343:1).
Pairwise cosine tables encoded as Fisher z i8 (ρ≥0.999 all 21 roles).

### What's included
- 26 shared palettes (256 centroids each)
- 26 Fisher z tables (64 KB each, i8 signed cosine)
- 869,760 HHTL-D entries (4 bytes/row)
- 109 passthrough tensors (norms, biases)
- Certification JSON (Spearman ρ per role)

### Usage
```sh
./scripts/download_codebooks.sh --tag v0.2.0-hhtld-fisherz --dir ./codebooks
cargo run --example tts_17b_hhtld_decode -- ./codebooks/model_hhtld.safetensors
```

### Device Requirements
- Pi 4: 75 MB RAM (11 MB model + 64 MB KV cache)
- Pi Zero 2W: feasible (512 MB - 75 MB = enough)
```

## Pass Criteria

- Release created on GitHub
- Assets downloadable via download_codebooks.sh
- Dockerfile.hhtld builds with new release tag
