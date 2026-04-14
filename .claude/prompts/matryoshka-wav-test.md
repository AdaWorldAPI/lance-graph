# Matryoshka WAV End-to-End Test

## The only metric that matters

Pairwise ρ lies. SVD d=512 gave ρ=1.0 pairwise but 0.4% codec token match.
The transformer is a 33-stage nonlinear amplifier. Per-layer ε compounds as
(1-ε)^33. The ONLY valid test is: compressed weights → 33 layers → WAV → listen.

## What was proven

Session findings from the SVD/Fisher z probes:

| Approach | Pairwise ρ | Codec token match | PCM correlation |
|---|---|---|---|
| Raw BF16 (reference) | 1.000 | 100% | 1.000 |
| SVD d=512 flat i16 | 1.000 | 0.4% | 0.11 |
| SVD d=256 flat i16 | 0.960 | 0.3% | 0.37 |
| GGUF Q4_K_M | ~0.99 | ~95%+ | ~0.98 |

Static metrics are useless. WAV output is the test.

## What matryoshka changes

Instead of flat precision across all SVD components, variable allocation:

```
Components 0-63:    i16  (128 bytes)  ← critical attention routing
Components 64-191:  i8   (128 bytes)  ← head discrimination
Components 192-383: i4   (96 bytes)   ← detail
Components 384-D:   i2   (var bytes)  ← noise floor

Total: ~512 bytes/row (4:1 from BF16)
```

Per-band error budget:
  Band 0 (i16): ε = 0.00003/element. (0.99997)^33 = 0.999
  Band 1 (i8):  ε = 0.004/element.   Weighted by energy: still < 0.001/layer
  Band 3 (i2):  ε = 0.25/element.    But <5% of energy: weighted ε < 0.0005

Theory predicts matryoshka survives where flat SVD dies because the
error is concentrated in low-energy components.

## Task

### Phase 1: Build SVD bases from the real model

```python
# In scripts/matryoshka_wav_test.py (or Rust example)
# Load Qwen3-TTS-12Hz-1.7B-Base/model.safetensors
# For each role group:
#   Sample 4096 rows → compute SVD → store top-D basis vectors
#   D = min(512, n_cols)
```

### Phase 2: Encode/decode with 3 profiles, compare WAV

Run TTS inference with:
1. `raw` — original BF16 (reference WAV)
2. `conservative` — i16×128 + i8×256 + i4×rest (2:1)
3. `standard` — i16×64 + i8×128 + i4×192 + i2×rest (4:1)
4. `aggressive` — i16×32 + i8×96 + i4×128 + i2×rest (8:1)
5. `q8_baseline` — simple i8 per element, no SVD (GGUF Q8 equivalent)

For each profile:
```python
# Compress all weight tensors (Q/K/V/O/gate/up/down per layer)
# Run talker (28 layers) + code_predictor (5 layers) → codec tokens
# Decode tokens → WAV
# Compare:
#   - codec_token_match: % identical to raw
#   - pcm_correlation: Pearson r of PCM waveforms
#   - perceptual: listen to the WAV (human judgment)
```

### Phase 3: Find the threshold

If `standard` (4:1) fails, try intermediate profiles:
- Widen i16 band: components 0-96 at i16 instead of 0-64
- Widen i8 band: components 64-256 at i8 instead of 64-192
- Drop i2 band entirely: everything below i4

The goal: find the minimum bytes/row that produces intelligible speech.

### Phase 4: Per-role profiling

If the WAV fails with one profile but passes with another, identify
WHICH roles are sensitive:
- Compress only Q_proj at standard, rest at conservative → WAV
- Compress only gate_proj at standard, rest at conservative → WAV
- etc.

Gate projections are the predicted pain point (bimodal distribution,
highest nonlinearity). Text embeddings may need conservative treatment
(151K rows, lookup table behavior).

## Key files

```
crates/bgz-tensor/src/matryoshka.rs  ← MatryoshkaCodec (this prompt)
  SvdBasis::build()                  ← power iteration, no LAPACK
  BandProfile::standard/aggressive/conservative
  encode_row() / decode_row()        ← variable-precision per band
  measure_quality()                  ← row cosine + pairwise ρ

scripts/tts_inference.py             ← reference TTS (raw BF16 → WAV)
  Modify to accept compressed weight dict

crates/bgz-tensor/src/hhtl_d.rs      ← HHTL-D routing (Skip/Attend)
crates/bgz-tensor/src/fisher_z.rs    ← Fisher z scoring (i8 cosine)
```

## Agent routing

- **truth-architect**: interprets the WAV results, identifies which 
  layer/role is the failure point
- **family-codec-smith**: if standard profile fails, sweeps band 
  boundaries to find the threshold
- **certification-officer**: writes JSON report with per-profile 
  codec_token_match and pcm_correlation
- **integration-lead**: if a profile passes, wires it into the 
  bake script and Dockerfile

## Error budget math

For the WAV to survive, the per-layer reconstruction error must satisfy:

  (1 - ε_layer)^33 ≥ 0.95

  ε_layer < 0.0015

For matryoshka standard profile (4:1):
  ε_band0 = 64 components × (1/32767)² per element = negligible
  ε_band1 = 128 components × (1/127)² × energy_fraction(64..192) ≈ 0.0003
  ε_band2 = 192 components × (1/7)² × energy_fraction(192..384) ≈ 0.0004
  ε_band3 = (D-384) components × (1/1)² × energy_fraction(384..) ≈ 0.0005

  ε_layer ≈ sqrt(Σ ε_band²) ≈ 0.0007

  (1 - 0.0007)^33 = 0.977 → should survive

This is the prediction. The WAV test validates it.

## Critical constraint

DO NOT evaluate with pairwise ρ, Spearman, Pearson, or any static metric.
The ONLY evaluation is:
1. codec_token_match vs raw (must be >90% for intelligible speech)
2. pcm_correlation vs raw (must be >0.90)
3. Human listening test (must be recognizable as the same sentence)

If codec_token_match < 50%, the compression is too aggressive regardless
of what any static metric says.
