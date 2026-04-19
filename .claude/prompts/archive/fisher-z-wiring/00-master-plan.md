# Fisher z Wiring — Master Plan

> Status: fisher_z.rs landed (7 tests). Remaining: 12 tasks.
> Agent routing: each task has a designated agent + knowledge prereqs.
> Certified: ρ≥0.999 on all 21 roles (Qwen3-TTS-1.7B, 5000 pairs each).

## Architecture

```
bgz-tensor/src/fisher_z.rs          ← DONE (FisherZTable, FamilyGamma, 7 tests)
bgz-tensor/src/hhtl_d.rs            ← UPDATE: add FisherZTable field
bgz-tensor/src/shared_palette.rs    ← UPDATE: compute Fisher z per group
ndarray/src/hpc/fisher_z.rs         ← NEW: SIMD-accelerated encode/decode
thinking-engine/examples/           ← UPDATE: encoder, decoder, certifier
scripts/bake_hhtld_codebooks.sh     ← UPDATE: include Fisher z in archive
```

## Task Execution Order

```
Step 1:  [DONE] fisher_z.rs in bgz-tensor
Step 2:  wire into hhtl_d.rs (HhtlDTensor gets FisherZTable)
Step 3:  wire into shared_palette.rs (per-group tables)
Step 4:  ndarray SIMD module (array_window + F32x16)
Step 5:  update encoder example (emit tables to safetensors)
Step 6:  update decoder example (restore via tanh)
Step 7:  update certifier (Fisher z cascade level)
Step 8:  update docs (encoding-ecosystem.md, BGZ_HHTL_D.md)
Step 9:  bake codebooks (run on 1.7B)
Step 10: GitHub release v0.2.0-hhtld-fisherz
Step 11: wire into TTS inference (table lookup replaces matmul)
Step 12: validate WAV output
```

## Files per Task

See numbered task files in this directory:
  02-hhtl-d-integration.md
  03-shared-palette.md
  04-ndarray-simd.md
  05-encoder.md
  06-decoder.md
  07-certifier.md
  08-docs.md
  09-bake.md
  10-release.md
  11-tts-inference.md
  12-validate.md
