# Step 9: Bake Codebooks with Fisher z Tables

> Agent: integration-lead
> READ FIRST: scripts/bake_hhtld_codebooks.sh, 05-encoder.md
> Depends on: Steps 5, 7 (encoder + certifier working)

## What to Do

Update `bake_hhtld_codebooks.sh` to include Fisher z tables in the
baked archive.

### Updated archive contents:

```
qwen3-tts-1.7b-hhtld/
  model_hhtld.safetensors          # HHTL-D entries + palettes + Fisher z tables
  manifest.json                     # group keys, shapes, SHA256, gamma values
  certification.json                # probe results (ρ, err per role)
  families/
    talker_gate.fz_gamma.bin        # 8 bytes per group
    talker_gate.fz_table.bin        # 64 KB per group (k×k i8)
    ... (26 groups)
```

### manifest.json format:

```json
{
  "model": "Qwen3-TTS-12Hz-1.7B-Base",
  "encoding": "BGZ-HHTL-D + Fisher z i8",
  "groups": [
    {
      "key": "talker/gate_proj",
      "shape": [6144, 2048],
      "n_tensors": 28,
      "k": 256,
      "fisher_z": {
        "z_min": -0.532,
        "z_range": 1.064,
        "table_bytes": 65536,
        "gamma_bytes": 8
      },
      "certification": {
        "spearman_rho": 0.9994,
        "restore_err": 0.00088
      }
    }
  ],
  "total_bytes": 11200000,
  "compression_ratio": 343
}
```

### Run command:

```sh
./scripts/bake_hhtld_codebooks.sh \
  --model /path/to/Qwen3-TTS-12Hz-1.7B-Base/model.safetensors \
  --output /tmp/qwen3-tts-1.7b-hhtld \
  --certify \
  --release v0.2.0-hhtld-fisherz
```

## Pass Criteria

- Archive produced with all 26 Fisher z tables
- manifest.json contains certification results per group
- Total archive ≤ 15 MB
- Certification: all 21 roles ρ≥0.995
