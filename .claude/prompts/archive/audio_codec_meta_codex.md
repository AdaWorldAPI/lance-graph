# AUDIO CODEC TRANSCODING — Meta Codex

> Master prompt for 3 transcoding sessions + integration.
> Read `.claude/knowledge/audio_research.md` FIRST.
> Each session produces ndarray compute primitives consumed by lance-graph.

## Goal

Transcode three open-source audio projects into ndarray/lance-graph
primitives. Not port — TRANSCODE. Extract the computational essence,
discard the framework, reimplement in Rust with SIMD.

## Session Order

```
SESSION 1: Opus (CELT component)
  WHY FIRST: Pure signal processing. No ML. No model weights.
  PRODUCES:  MDCT, band energy, PVQ in ndarray
  VALIDATES: BF16 packing, gain-shape split, psychoacoustic bands

SESSION 2: Whisper (mel spectrogram + decoder structure)
  WHY SECOND: Depends on MDCT from Session 1
  PRODUCES:  Mel filterbank, phoneme graph traversal in lance-graph
  VALIDATES: Speech recognition via graph search instead of transformer

SESSION 3: Bark (3-stage RVQ hierarchy)
  WHY THIRD: Depends on understanding from Sessions 1+2
  PRODUCES:  Semantic → Coarse → Fine pipeline mapped to HHTL
  VALIDATES: Text-to-speech via graph synthesis

SESSION 4: Integration (this prompt, run AFTER 1-3)
  PRODUCES:  lance-graph-audio crate wiring all primitives
  VALIDATES: Round-trip: audio → graph → audio with search capability
```

## Architecture

```
ndarray (compute/codec crate):
  src/audio/
    mdct.rs         — MDCT/iMDCT, SIMD-accelerated (from Opus Session 1)
    mel.rs          — Mel filterbank (from Whisper Session 2)
    bands.rs        — Critical band grouping, BF16 packing
    pvq.rs          — Pyramid Vector Quantization (from Opus Session 1)
    rvq.rs          — Residual VQ hierarchy (from Bark Session 3)
    voice.rs        — Voice archetype encoding (from Bark Session 3)
    codec.rs        — Top-level encode/decode API

lance-graph (memory/execution crate):
  graph/audio/
    encoder.rs      — WAV → MDCT → BF16 bands → graph nodes
    decoder.rs      — graph nodes → BF16 → iMDCT → WAV
    phoneme.rs      — Phoneme graph for speech recognition
    qualia.rs       — Feeling annotation (steelwind/emberglow/etc)
    player.rs       — HHTL traversal as progressive playback
    search.rs       — Content-based audio retrieval
```

## Rules for All Sessions

1. READ the source repo FIRST. Clone via zipball, read actual code.
2. Identify the PIPELINE stages (what transforms to what, in what order).
3. For each stage: what is the INPUT format, OUTPUT format, and OPERATION.
4. Map each operation to: ndarray SIMD kernel OR lance-graph storage.
5. If the operation is a neural network: extract what it COMPUTES,
   not how it's trained. The inference path is what matters.
6. Everything that can run INT8/BF16: mark for NPU path.
7. Everything that needs FP32: mark for CPU hydration path.
8. Write Rust, not Python. Use ndarray types. SIMD dispatch via dispatch! macro.

## Success Criteria

After all 3 sessions:
```
cargo test -p ndarray --features audio    → all audio kernels pass
cargo test -p lance-graph --features audio → encode/decode round-trip

Demo:
  1. Encode a WAV file into lance-graph nodes
  2. Search the graph for "similar sounding frames"
  3. Decode back to WAV
  4. A/B test: original vs round-trip at various HHTL depths
```
