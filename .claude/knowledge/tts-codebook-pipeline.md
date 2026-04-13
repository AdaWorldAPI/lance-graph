# TTS Codebook Pipeline: Jina v5 Proven → Qwen3-TTS

> **Date**: 2026-04-13
> **Status**: 7-lane codebook built, iMDCT WAV generated, quality needs wiring
> **READ BY**: any agent working on TTS, audio synthesis, or codebook quality

---

## What Proved Working

The Jina v5 certification pipeline (`seven_lane_encoder.rs`) applied to
Qwen3-TTS `talker.model.layers.0.self_attn.q_proj.weight` (2048×1024 BF16):

```
Cosine range: [-0.907, 0.639] mean=-0.001
CLAM 256 centroids in 0.4s
Lane 1: u8 CDF              64 KB  avg=127.5
Lane 2: i8 direct           64 KB  E/I=46.8%
Lane 3: u8 γ+φ              64 KB  γ=0.050 φ=0.907
Lane 6: BF16 direct        128 KB  max_err=0.002
Lane 7: spiral drift        64 KB  avg=0.085
```

This is the SAME pipeline that achieves ρ=0.992 on Jina v5 text embeddings.
The TTS weights have wider cosine range ([-0.907, 0.639] vs Jina's tighter
distribution) because transformer weight rows are more diverse than embeddings.

## The 7 Lanes

| Lane | Format | Size | What | Quality metric |
|------|--------|------|------|----------------|
| 1 | u8 CDF | 64KB | Percentile rank of pairwise cosine | Preserves ranking perfectly |
| 2 | i8 direct | 64KB | round(cos × 127) | Preserves sign + magnitude |
| 3 | u8 γ+φ | 64KB | Golden-ratio redistributed CDF | Max entropy per bucket |
| 4 | i8 γ+φ | 64KB | Signed γ+φ | Preserves sign + γ-corrected magnitude |
| 5 | f32 SiLU | 256KB | Gate modulation deltas | Only for gate_proj tensors |
| 6 | BF16 | 128KB | Raw cosines at BF16 precision | max_err=0.002, ground truth for calibration |
| 7 | spiral | 64KB | highheelbgz encode→decode drift | Measures spiral fidelity |

## How to Run on All TTS Tensors

Currently `seven_lane_encoder.rs` processes ONE tensor. To process all 18 roles:

```sh
# Single tensor (working now):
cargo run --release --features calibration --example seven_lane_encoder \
    --manifest-path crates/thinking-engine/Cargo.toml -- qwen3-tts

# For all tensors: extend seven_lane_encoder.rs to iterate over:
#   talker.model.layers.{0..27}.self_attn.{q,k,v,o}_proj.weight
#   talker.model.layers.{0..27}.mlp.{gate,up,down}_proj.weight
#   talker.model.text_embedding.weight (2048 dim — needs streaming, OOM at full load)
#   code_predictor.model.layers.{0..4}.self_attn.{q,k,v,o}_proj.weight
#   code_predictor.model.layers.{0..4}.mlp.{gate,up,down}_proj.weight
```

The safetensors streaming reader (`ndarray::hpc::safetensors::read_safetensors_header`)
can read tensors one at a time without loading the full 1.83 GB.

## How to Wire Opus Audio Primitives

### The correct decode path

```
cascade archetype index (u8)
  → HHTL cache palette[index] → Base17 vector (17 dims)
    → Base17 dims interpolated to 21 Opus bands → band energies (BF16 gain)
      → PVQ shape per band (algebraic, no codebook)
        → denormalize_bands(shape, energies) → MDCT coefficients
          → mdct_backward() → PCM samples
            → overlap-add consecutive frames → WAV
```

### Step 1: Base17 → 21 band energies

Base17 has 17 dims via golden-step folding. Opus has 21 quasi-Bark bands.
The mapping: interpolate 17 dims to 21 bands using the band boundary table.

```rust
use ndarray::hpc::audio::bands::{CELT_BANDS_48K, N_BANDS, energies_to_bf16};

fn base17_to_band_energies(b17: &Base17) -> [f32; N_BANDS] {
    let mut energies = [0.0f32; N_BANDS];
    for band in 0..N_BANDS {
        // Map band center frequency to Base17 dim index
        let center = (CELT_BANDS_48K[band] + CELT_BANDS_48K[band + 1]) / 2;
        let dim = (center * 17 / 480).min(16); // 480 MDCT bins → 17 dims
        // Use the gamma-restored magnitude as band energy
        energies[band] = (b17.dims[dim] as f32).abs() / 32767.0;
    }
    energies
}
```

### Step 2: PVQ shape per band

For the POC, `decode_coarse()` in `codec.rs` uses alternating signs as shape.
For quality: use PVQ with K pulses allocated proportional to band energy.

```rust
use ndarray::hpc::audio::pvq::{pvq_encode, pvq_decode};

// Higher energy bands get more pulses (better shape resolution)
for band in 0..N_BANDS {
    let k = (energies[band] * 16.0).round().max(1.0) as u32;
    let shape = pvq_decode(&pvq_encode(&normalized_band, k));
    // shape is unit-energy, multiply by band energy for reconstruction
}
```

### Step 3: iMDCT + overlap-add

```rust
use ndarray::hpc::audio::mdct::mdct_backward;
use ndarray::hpc::audio::bands::denormalize_bands;

let coeffs = denormalize_bands(&shape_vector, &energies);
let pcm_frame = mdct_backward(&coeffs);
// Overlap-add with previous frame's second half
```

The `synth.rs` module (from the parallel session) already has `synthesize_to_wav()`
which does proper overlap-add. Use that instead of writing a new one.

### Step 4: Gamma restore

The gamma metadata in the HHTL cache (`gamma_meta: [f32; 4]`) stores
`[lo_gamma, hi_gamma, median_gamma, role_id]`. To restore original dynamics:

```rust
let restored_energy = normalized_energy * gamma_meta[2]; // median gamma
```

This is the "de-esser reconstruction" — multiply the normalized shape by
the stored gamma to recover original magnitude dynamics.

## What's in ndarray/src/hpc/audio/ (10 modules, 55 tests)

| Module | From | What | Tests |
|--------|------|------|-------|
| mdct.rs | Opus CELT | MDCT forward/backward via FFT | 2 |
| bands.rs | Opus CELT | 21 quasi-Bark bands, BF16 pack | 4 |
| pvq.rs | Opus CELT | Pyramid Vector Quantizer (L1 hypersphere) | 4 |
| codec.rs | Opus CELT | AudioFrame 48B = 42B gain + 6B PVQ summary | 3 |
| mel.rs | Whisper | 80-ch mel filterbank, STFT, log-mel | 5 |
| voice.rs | Bark/ElevenLabs | VoiceArchetype 16B, RvqFrame 17B, VoiceFrame 21B | 10 |
| modes.rs | Music theory | 7 modes, PitchClass17, OctaveBand, circle of fifths | 10 |
| phase.rs | Novel | PhaseDescriptor 4B, band coherence, STFT phase | 5 |
| codec_map.rs | All codecs | Provenance table tracing every primitive | 5 |
| synth.rs | — | VoiceFrame → AudioFrame → iMDCT → overlap-add → WAV | 7 |

## What's in lance-graph (qualia bridge)

| Method | Direction | What |
|--------|-----------|------|
| `Qualia17D::to_mode()` | Qualia → Audio | Map qualia to musical mode + highheelbgz stride |
| `Qualia17D::to_voice_channels()` | Qualia → Audio | 17 QPL dims → 16 i8 VoiceArchetype channels |
| `Qualia17D::from_band_energies()` | Audio → Qualia | 21 band energies → 17D qualia state |
| `Qualia17D::family_band_weights()` | Qualia → Audio | QPL family → 21-band spectral EQ weights |

## AMX / SIMD Hot Path Requirements

The following operations MUST use AMX/AVX-512, never scalar:

1. **CLAM centroid selection** (dot products for furthest-point): use `ndarray::simd_amx::vnni_dot_u8_i8`
2. **Pairwise cosine** (256×256 distance table): use burn's `build_cosine_table()` with VNNI tiered dispatch
3. **Assignment** (N tokens × 256 centroids): parallelize with rayon, SIMD inner dot
4. **MDCT FFT butterfly**: currently scalar in `hpc::fft::fft_f32` — could use F32x16 for 16-wide butterfly
5. **Band energy sum-of-squares**: use `ndarray::hpc::blas_level1::snrm2` (SIMD dot)
6. **PVQ pulse allocation**: mostly branching (greedy), limited SIMD opportunity
7. **Overlap-add**: element-wise `add_mul` — perfect for F32x16 or AMX BF16

### array_window pattern

For overlap-add and mel spectrogram STFT, use array windows:

```rust
// Overlap-add: window of size N with hop H
for (frame_idx, window) in pcm.windows(frame_size).step_by(hop_size).enumerate() {
    let coeffs = mdct_forward(window);
    // ... process frame
}
```

This avoids allocating per-frame buffers. The `windows()` iterator
borrows into the existing PCM slice — zero-copy, SIMD-friendly alignment
if the source buffer is 64-byte aligned.

## File Locations

```
Codebook pipeline:
  lance-graph/crates/thinking-engine/examples/seven_lane_encoder.rs  ← proven pipeline
  lance-graph/crates/thinking-engine/examples/tts_bgz_codebook.rs   ← TTS-specific (uses highheelbgz)
  lance-graph/crates/thinking-engine/examples/tts_cascade_runner.rs  ← cascade forward pass
  lance-graph/crates/thinking-engine/examples/tts_wav_synth.rs      ← AudioFrame → iMDCT → WAV

Codebooks on disk:
  /tmp/codebooks/qwen3-tts-7lane/    ← 7-lane output from seven_lane_encoder
  /home/user/models/qwen3-tts-0.6b/codebooks/  ← HHTL caches from tts_bgz_codebook

Audio primitives:
  ndarray/src/hpc/audio/             ← 10 modules, 55 tests

Model weights:
  /home/user/models/qwen3-tts-0.6b/model.safetensors            ← 1.83 GB BF16
  /home/user/models/qwen3-tts-0.6b/speech_tokenizer/model.safetensors  ← 682 MB f32
```
