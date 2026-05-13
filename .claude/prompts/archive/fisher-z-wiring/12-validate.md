# Step 12: Validate TTS Output with Fisher z Tables

> Agent: certification-officer + truth-architect
> READ FIRST: 11-tts-inference.md, certification results
> Depends on: Step 11

## What to Do

Run the Fisher z inference path and the raw inference path on the
same input text. Compare the WAV outputs.

### Test inputs:

1. "Hello, world." (short, basic)
2. "This is a test of text to speech synthesis using compressed weights." (medium)
3. "The quick brown fox jumps over the lazy dog." (all phonemes)

### Metrics to compare:

1. **Spectral similarity**: mel spectrogram correlation between raw and Fisher z WAVs
2. **RMS difference**: |rms_raw - rms_fz| / rms_raw (should be <10%)
3. **Peak frequency**: dominant frequency should match within 50 Hz
4. **Duration**: should match exactly (same number of codec frames)
5. **Temporal envelope**: RMS per 100ms chunk, Pearson r ≥ 0.90

### How to run:

```sh
# Raw inference (ground truth)
python3 scripts/tts_inference.py "Hello, world." 
# Output: data/tts-cascade/tts_real_output.wav

# Fisher z inference
cargo run --release --example tts_fisher_z_inference \
    --manifest-path crates/thinking-engine/Cargo.toml \
    -- "Hello, world." codebooks/model_hhtld.safetensors
# Output: data/tts-cascade/tts_fisherz_output.wav
```

### Success criteria:

The Fisher z WAV doesn't need to sound identical — the attention scores
are quantized to i8, which introduces some noise. But:

- Temporal envelope correlation ≥ 0.90
- Same dominant frequency (±50 Hz)
- Non-silent (RMS > 0.05)
- Duration matches (same frame count)

If the WAV sounds different but has the right temporal structure,
the encoding is working correctly. The quality difference is the
quantization noise of 256 cosine levels — certified at ρ≥0.999.

## Final Deliverable

A commit with:
1. Both WAV files (raw + Fisher z)
2. Comparison metrics in the certification JSON
3. Updated encoding-ecosystem.md: "VALIDATED — end-to-end TTS"

This completes the Fisher z wiring. The system permanently uses
Fisher z i8 pairwise tables for all attention score computation.
