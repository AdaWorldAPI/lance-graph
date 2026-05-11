# Session Handover: TTS Cascade Reality Check

> **Date**: 2026-04-13 (updated after parallel session merged PRs #102, #169)
> **Branch**: `claude/risc-thought-engine-TCZw7` (both repos)
> **READ FIRST**: `.claude/prompts/audio_session1_opus_celt.md`, `audio_session3_bark_voice.md`, `session_bgz_tensor.md`

---

## Current State (post-merge)

### What's working end-to-end
- 55 audio tests passing (ndarray `cargo test --lib -- audio`)
- 3 thinking-engine examples build clean: `tts_bgz_codebook`, `tts_cascade_runner`, `tts_wav_synth`
- Cascade runs at 1.08M tok/s with ComposeTable (was 1.2M without)
- 18 HHTL caches + assignments + audio codes on disk
- AMX SIGILL fixed (`_xgetbv(0)` + `prctl(ARCH_REQ_XCOMP_PERM)`)
- Qualia17D ↔ audio bridge wired (modes, band weights, voice channels)

### What the parallel session (PRs #102, #169) fixed
1. ✅ `tts_bgz_codebook.rs` — now uses `highheelbgz::SpiralEncoding` + `GammaProfile::calibrate()`
2. ✅ `tts_cascade_runner.rs` — now uses `ComposeTable::build()` + `ct.compose(a, b)`
3. ✅ `tts_wav_synth.rs` — now uses `AudioFrame::decode_coarse()` via band energies + iMDCT
4. ✅ 6 new ndarray audio modules: mel, voice, modes, phase, codec_map, synth
5. ✅ AMX SIGILL: proper `_xgetbv` + `prctl` instead of CPUID-only

### What still needs doing (lance-graph side)
See "Summary" in the user's audit: 55 of ~85 deliverables done.
The main gaps are ALL on the lance-graph side:

---

## Hints for Next Session: Lance-Graph Side

### 1. Phoneme graph (Session 2 gap)
- File: `lance-graph/crates/lance-graph/src/graph/audio/phoneme.rs`
- Needs: `PhonemeNode` struct, `PhonemeCodebook`, SPO traversal for phoneme sequences
- **Watch out**: the mel module in ndarray outputs 80-channel features per frame. The phoneme graph needs to classify these into phoneme nodes. Use `mel::mel_spectrogram()` → `PhonemeCodebook::classify()` → `PhonemeNode`.
- **Don't reimplement mel** — it's in `ndarray::hpc::audio::mel` already (5 tests passing).

### 2. Synthesis graph traversal (Session 3 gap)
- File: `lance-graph/crates/lance-graph/src/graph/audio/synthesis.rs`
- Needs: `synthesize_coarse()` and `synthesize_fine()` as graph traversals
- **Watch out**: the VoiceArchetype (16B) is in `ndarray::hpc::audio::voice`. Don't create a second one in lance-graph. Import it.
- `synthesize_coarse()` = SPO graph traversal: (archetype, phoneme) → band energies per frame
- `synthesize_fine()` = PVQ from `ndarray::hpc::audio::pvq` applied per band

### 3. CalibratedCodebook wiring
- `bgz_tensor::codebook_calibrated::CalibratedCodebook` has `centroids_f32` + γ+φ calibrated u8 distance table
- **Never instantiated in any example yet**. The codebook builder uses `WeightPalette::build()` directly.
- For better quality: use `CalibratedCodebook::build()` which does CLAM + γ+φ automatically.
- **Watch out**: CalibratedCodebook operates on `Vec<Vec<f32>>` (full-dim rows), not Base17. It's for the distance table quality, not the palette structure.

### 4. highheelbgz SpiralEncoding details
- `SpiralEncoding::encode(weights, start, stride, k)` — the start=20 (skip degenerate region) is hardcoded in the example. Real value should come from the model's weight distribution.
- `GammaProfile::calibrate()` takes `&[&[f32]]` row refs. The current implementation samples first tensor only — should sample across all tensors for a role.
- `GammaProfile` stores `role_gamma: [f32; 6]` (Q/K/V/Gate/Up/Down) but the TTS model has 8 roles (+ O + Embed). The 6-slot array was extended to 8 in `gamma_phi.rs` but `rehydrate.rs` still uses 6. Check for mismatch.

### 5. ComposeTable correctness
- `ComposeTable::build(&palette)` computes `compose[a][b] = nearest(palette[a].xor_bind(palette[b]))`.
- `xor_bind` on Base17 is element-wise XOR of i16 values. This is the VSA binding operation.
- **Watch out**: XOR on i16 produces valid Base17 only if the dims are in the right range. Large values can overflow. The current implementation clamps but doesn't check — verify with `assert!(result.l1(&Base17::zero()) < 100000)` or similar.

### 6. Disk space: 5.4 GB free
- Models: 1.83 GB safetensors + 682 MB speech tokenizer = 2.5 GB
- Build targets can eat space fast. `rm -rf crates/*/target` in lance-graph if needed.
- ndarray target: ~3 GB after full build.

### 7. The codebook files need regenerating
- The HHTL caches on disk were built with the OLD pipeline (before highheelbgz wiring).
- After the parallel session fixed `tts_bgz_codebook.rs` to use SpiralEncoding, the caches need to be rebuilt:
  ```sh
  cargo run --release --example tts_bgz_codebook \
      --manifest-path crates/thinking-engine/Cargo.toml \
      -- /home/user/models/qwen3-tts-0.6b/model.safetensors
  ```
- Then regenerate cascade audio codes:
  ```sh
  cargo run --release --example tts_cascade_runner \
      --manifest-path crates/thinking-engine/Cargo.toml
  ```
- Then regenerate WAV:
  ```sh
  cargo run --release --example tts_wav_synth \
      --manifest-path crates/thinking-engine/Cargo.toml
  ```

### 8. The round-trip demo is the goal
- Encode: WAV → mel spectrogram → AudioFrame (48B) → HHTL graph node
- Search: query graph for "similar sounding frames"
- Decode: AudioFrame → VoiceFrame (21B) → iMDCT overlap-add → PCM → WAV
- This uses `ndarray::hpc::audio::synth::synthesize_to_wav()` which already works (7 tests)

---

## What Was Built This Session

### lance-graph (7 PRs merged + 1 pending)
- `tts_bgz_codebook.rs` — streams Qwen3-TTS 0.6B safetensors, gamma-corrects, builds CLAM palettes, HHTL caches. 18 roles, 417:1 compression.
- `tts_cascade_runner.rs` — loads 18 HHTL caches, runs talker (28 layers) + code predictor (5 layers) cascade. 1.2M tok/s.
- `tts_wav_synth.rs` — WRONG: treated codebook embeddings as spectral magnitudes. Delete or rewrite using the audio primitives.
- `hhtl_cache.rs` — added `gamma_meta: [f32; 4]`, percentile-based route thresholds, perceptual band weighting.
- `.claude/knowledge/role-enum-complementarity.md` — decision record + holy grail hypothesis.

### ndarray (2 PRs merged + 1 pending)
- `crates/burn/src/ops/matmul.rs` — `CompiledLinear` struct with VNNI centroid matmul.
- `src/hpc/audio/` — NEW: mdct.rs, bands.rs, pvq.rs, codec.rs (13 tests passing). Opus CELT primitives.
- `src/hpc/ocr_felt.rs` — EULER_GAMMA cleanup.

### Models on disk
- `/home/user/models/qwen3-tts-0.6b/model.safetensors` — 1.83 GB BF16
- `/home/user/models/qwen3-tts-0.6b/codebooks/` — 18 HHTL caches + assignment files
- `/home/user/models/qwen3-tts-0.6b/speech_tokenizer/model.safetensors` — 682 MB f32 (RVQ codebooks)

---

## What Was Done Wrong

### 1. Bypassed highheelbgz — went straight to bgz-tensor
The correct pipeline uses **highheelbgz first**:
```
safetensors → highheelbgz::SpiralEncoding (BF16 anchors, stride=role)
  → highheelbgz::GammaProfile (exact restore metadata, 28 bytes)
    → bgz-tensor::WeightPalette (CLAM from spiral-encoded rows)
      → bgz-tensor::HhtlCache (route table + distance table + compose table)
```

What was done instead:
```
safetensors → manual gamma hack (row / mean_magnitude) → bgz-tensor::WeightPalette
```

highheelbgz has: `SpiralEncoding::encode()`, `SpiralAddress` with stride=role (8/5/4/3/2), `GammaProfile::calibrate()`, `SpiralPalette::assign()`, `coarse_band()`. None of these were used.

### 2. Never used ComposeTable
The cascade runner's `RouteAction::Compose` branch does `(a + b) / 2` — meaningless. Should use `ComposeTable::compose(a, b)` which is O(1) multi-hop via XOR bind. The `AttentionSemiring` (distance + compose as unified algebra) was never instantiated.

### 3. Never used CalibratedCodebook
`bgz_tensor::codebook_calibrated::CalibratedCodebook` has `centroids_f32` + γ+φ calibrated u8 distance table + per-role gamma. Was read but never wired.

### 4. WAV synthesis was bits-as-vectors
The `tts_wav_synth.rs` treated RVQ codebook embeddings (latent vectors) as spectral magnitudes and summed cosines. Produced noise, not audio. The correct path is:
- Cascade codes → map to band energies (from highheelbgz spiral) → PVQ shape → iMDCT → PCM
- Or: use the Opus CELT primitives just added to `ndarray/src/hpc/audio/`

### 5. Audio primitives not connected to bgz-tensor
The AudioFrame (48 bytes: 21 BF16 band energies + 6B PVQ summary) maps to HHTL levels:
```
HEEL: PVQ summary bytes 0-1 (sign pattern → spectral category)
HIP:  band energies (BF16 gain → L1 distance)
TWIG: PVQ summary bytes 4-5 (harmonic detail)
LEAF: full iMDCT decode
```
This wiring doesn't exist yet.

---

## What Works Correctly

- **Gamma-corrected CLAM palette**: ρ=0.21-0.97 across roles, entropy 4.7-6.8 bits
- **HHTL cascade at 1.2M tok/s**: route table with perceptual band weighting
- **Skip rates**: norm 54%, gate 4%, embedding 0% (perceptually correct)
- **Speech tokenizer config decoded**: `upsample_rates [8,5,4,3]` = highheelbgz stride→role mapping
- **Burn CompiledLinear + VNNI**: compiles, intercepts matmul, ready for codebook registration
- **Audio primitives**: MDCT, bands, PVQ, AudioFrame — 13 tests, zero deps, uses existing hpc::fft
- **Qwen3-TTS generates speech**: reference.wav exists from original weights

---

## Correct Next Steps

### 1. Wire highheelbgz into the TTS pipeline
```rust
use highheelbgz::{SpiralEncoding, SpiralAddress, GammaProfile, SpiralPalette};

// For each weight tensor:
let addr = SpiralAddress::new(start, stride_for_role, length);
let enc = SpiralEncoding::encode(&f32_row, addr.start, addr.stride, k_anchors);
let gamma = GammaProfile::calibrate("qwen3-tts", &role_rows);
// THEN: build palette from spiral-encoded rows
```

### 2. Wire ComposeTable in cascade runner
```rust
// Build AttentionSemiring (not just AttentionTable)
let semiring = AttentionSemiring::build(&palette);
// In cascade:
RouteAction::Compose => {
    let intermediate = semiring.compose.compose(a, b); // O(1) XOR bind lookup
    current = intermediate;
}
```

### 3. Connect AudioFrame to HHTL cascade
The cascade produces archetype indices. Each archetype maps to an AudioFrame (48 bytes). The AudioFrame's band energies ARE the cascade's distance table entries. The decode path: `archetype → AudioFrame → denormalize_bands → iMDCT → PCM`.

### 4. Fix tts_wav_synth.rs
Replace codebook embedding lookup with:
```rust
use ndarray::hpc::audio::codec::AudioFrame;
use ndarray::hpc::audio::mdct::mdct_backward;
use ndarray::hpc::audio::bands::denormalize_bands;

let frame = AudioFrame::from_bytes(&cached_frame_bytes);
let pcm = frame.decode_coarse(); // band energies → flat shape → iMDCT
```

### 5. Per-family basin gamma from gamma deviations
Currently 2-basin (median split). The basins should emerge FROM gamma deviations — rows with similar gamma belong to same basin. No separate clustering step. The gamma deviation IS the CLAM radius implicitly.

---

## Key Files to Read

### highheelbgz (the correct input format)
- `crates/highheelbgz/src/lib.rs` — SpiralAddress, SpiralWalk, coarse_band, TensorRole, NeuronPrint, SpiralPalette
- `crates/highheelbgz/src/rehydrate.rs` — SpiralEncoding, GammaProfile, BF16 anchors, rehydrate
- `crates/highheelbgz/src/tensor_bridge.rs` — walk_to_stacked, cascade_search

### bgz-tensor (the HHTL cascade engine)  
- `crates/bgz-tensor/src/hhtl_cache.rs` — HhtlCache, RouteAction, build_route_table
- `crates/bgz-tensor/src/attention.rs` — AttentionTable, ComposeTable, AttentionSemiring
- `crates/bgz-tensor/src/codebook_calibrated.rs` — CalibratedCodebook (CLAM + γ+φ)
- `crates/bgz-tensor/src/gamma_phi.rs` — GammaProfile, gamma_phi_encode/decode

### ndarray audio (the output format)
- `src/hpc/audio/mdct.rs` — mdct_forward, mdct_backward
- `src/hpc/audio/bands.rs` — 21 Opus bands, band_energies, normalize/denormalize, BF16
- `src/hpc/audio/pvq.rs` — pvq_encode/decode, pvq_summary
- `src/hpc/audio/codec.rs` — AudioFrame (48 bytes), encode, decode_coarse

### Prompts (the architecture spec)
- `.claude/prompts/audio_codec_meta_codex.md` — master plan, 4 sessions
- `.claude/prompts/audio_session1_opus_celt.md` — MDCT, bands, PVQ (implemented ✓)
- `.claude/prompts/audio_session3_bark_voice.md` — HHTL→TTS mapping, VoiceArchetype
- `.claude/prompts/session_bgz_tensor.md` — bgz-tensor architecture, AttentionSemiring

---

## Measurements from This Session

| Metric | Value |
|--------|-------|
| Model compression | 417:1 (1.83 GB → 4.3 MB) |
| Cascade throughput | 1.2M tok/s (0.8µs/token) |
| Best Spearman ρ | 0.97 (talker/norm) |
| Worst Spearman ρ | 0.21 (talker/v_proj) |
| Skip rate (norm) | 54% (perceptual: noise band) |
| Skip rate (v_proj) | 0% (perceptual: formant-critical) |
| Audio primitives | 13 tests passing |
| Codebook load time | 453ms cold, 130ms warm |
| Gamma range (talker) | 0.0007 – 0.0381 (tight BF16 weights) |
| Gamma range (norm) | 0.0 – 19.75 (wide, different distribution) |
