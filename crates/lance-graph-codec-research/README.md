# lance-graph-codec-research

Research crate for comparing three audio encoding strategies and their
combinations within the HHTL knowledge graph framework.

## Three Strategies

### Strategy A: Per-Frame MDCT (classical codec path)
```
frame → MDCT → 24 BF16 bands → 1 graph node per frame
75 nodes/second, 48 bytes/node = 3600 bytes/s = 28.8 kbps
Quality: transparent (same as Opus at equivalent bitrate)
Latency: 13ms (one frame)
Search: ZeckF64 edges between consecutive frames
```

### Strategy B: Streaming Accumulator (VSA bundle path)
```
frames → MDCT → BF16 bands → cyclic shift → bundle into accumulator
1 accumulator per second = 48 bytes/s = 384 bps
Quality: parametric (pitch + formants + envelope, transients lost)
Latency: 1 second (accumulation window)
Search: qualia classification of crystallized spectrum
Diamond Markov: extract crystallized components progressively
```

### Strategy C: Hybrid (the combination nobody has tried)
```
frames → MDCT → BF16 bands → BOTH paths simultaneously:
  Path 1: per-frame ZeckF64 scent → neighborhoods.lance (L1 search)
  Path 2: streaming accumulator → crystallize → cognitive_nodes.lance (identity)

Store: per-frame scent for temporal navigation (28.8 kbps overhead)
       + crystallized identity for content search (384 bps)
       + Diamond Markov epiphanies for structural events (variable)

The scent stream IS the P-frame sequence.
The crystallized identity IS the I-frame.
The epiphanies ARE the scene changes / key changes / transitions.
```

## What We're Measuring

```
QUALITY METRICS:
  PESQ (speech)           — 1.0 (bad) to 4.5 (transparent)
  ViSQOL (general audio)  — similar scale
  Spectral distortion     — dB, lower is better
  Pitch accuracy          — Hz deviation from ground truth
  Formant preservation    — F1/F2/F3 deviation

COMPRESSION METRICS:
  Bitrate (bps)
  Compression ratio vs raw PCM
  Compression ratio vs Opus at equivalent quality

SEARCH METRICS:
  Can we find "the moment the key changes" via HHTL? Y/N + latency
  Can we find "all frames that sound like steelwind"? Y/N + latency
  Can we find "similar songs" from crystallized identity? Y/N + latency

CODEC THEORY METRICS:
  Noise floor vs psychoacoustic masking threshold (correlation)
  Crystallization rate vs signal repetitiveness (correlation)
  Diamond Markov invariant: rebundle epiphanies + residual = original? (Hamming distance)
  R-D curve: quality vs bitrate across threshold settings
```
