# Audio Codec Research: What We Have, What We Want

## Context

This project has established that audio codecs and knowledge graph
encoding are structurally identical at the mathematical level.
The architecture already implements most codec principles for
knowledge compression (RVQ = HHTL cascade, floor/residue =
scent/resolution, I/P/B = encounter/delta/qualia, etc.).

The next step: transcode actual audio codec implementations into
ndarray/lance-graph primitives to validate the theory and build
a CPU-native audio codec that stores audio AS a searchable graph.

## What Already Exists in Our Stack

```
ndarray:
  SIMD kernels (AVX2/AVX-512/NEON dispatch)
  PackedDatabase with 3-takt cascade
  BF16 ↔ FP32 hydration
  vpshufb/vtbl table lookup (no popcount needed)
  jitson JIT compiler (JSON → native function pointers)
  Fibonacci/Zeckendorf encoding
  CLAM/CHAODA/CHESS/panCAKES clustering

lance-graph:
  HHTL cascade search (HEEL→HIP→TWIG→LEAF, 90% rejection/stage)
  ZeckF64 progressive encoding (scent + resolution)
  blasgraph semiring algebra with BF16 BitVec
  SPO store with NARS truth values
  Versioned graph with time-travel (I-frame/P-frame pattern)
  neighborhoods.lance with column pruning (= quantizer dropout)

Already formalized (in session transcripts):
  - Opus PVQ → SPO Pyramid mapping
  - Zeckendorf non-consecutivity = psychoacoustic masking
  - BF16 audio encoding: 24 critical bands × BF16 = 48 bytes/frame
  - 13ms frames at 48kHz = 75 nodes/second
  - 3-minute song = 13,500 nodes (fits in 2 scopes)
  - Harmonic Gestalt: 12 Quintenzirkel projections for qualia classification
  - steelwind/woodwarm/emberglow/velvetpause as feeling meta-layer
  - Compression: Scent 200:1, PVQ 40:1, Full 10:1
```

## Three Target Repos for Transcoding

### 1. Whisper (openai/whisper, MIT)
```
What it does:  Audio → Mel spectrogram → Transformer → Text
Key insight:   The mel spectrogram IS a frequency-band decomposition
               80 mel bands × 16ms frames
               This maps directly to our 24 critical bands × BF16
Transcode:     Extract the mel spectrogram pipeline
               Replace transformer with HHTL graph traversal
               Phoneme graph instead of attention matrix
Goal:          CPU-native speech recognition via graph search
```

### 2. Bark (suno-ai/bark, MIT)
```
What it does:  Text → Semantic tokens → Coarse audio → Fine audio
               Uses EnCodec-style RVQ (three-stage hierarchy)
Key files:     bark/generation.py (the 3-stage pipeline)
               bark/model.py (GPT-2 architecture for each stage)
Key insight:   The 3-stage hierarchy IS our HHTL cascade
               Semantic tokens = HEEL (coarse classification)
               Coarse audio = HIP/TWIG (progressive detail)
               Fine audio = LEAF (full resolution)
Transcode:     Extract the RVQ codebook structure
               Map semantic tokens → qualia archetypes
               Replace GPT-2 with SPO graph traversal
Goal:          Text-to-speech via graph synthesis, no transformer
```

### 3. Opus (xiph/opus, BSD, RFC 6716)
```
What it does:  Audio → SILK (speech) or CELT (music) → Bitstream
Key insight:   CELT uses Pyramid Vector Quantization (PVQ)
               PVQ distributes points on L1 hypersphere
               This IS what golden-angle rotation does in our arch
               Gain-shape quantization = our scent/resolution split
Source:        C reference implementation, ~50K lines
               celt/bands.c (band energy normalization)
               celt/vq.c (pyramid vector quantizer)
               celt/celt_encoder.c / celt_decoder.c
Transcode:     Extract MDCT + band energy + PVQ pipeline
               Implement in ndarray with SIMD
               Store in lance-graph as BF16 graph nodes
Goal:          Opus-quality codec running as lance-graph pipeline
```

### Bonus: EnCodec (facebookresearch/encodec, MIT)
```
What it does:  Audio → CNN encoder → RVQ (8 codebooks) → CNN decoder
Key insight:   RVQ is ALREADY our architecture (proven in transcripts)
               The CNN encoder/decoder could be replaced by
               Fibonacci-encoded MDCT coefficients
Source:        Python/PyTorch, clean codebase
               encodec/quantization.py (the RVQ implementation)
               encodec/model.py (encoder/decoder)
Goal:          Extract the RVQ logic, implement in ndarray
```

## What We Want to Find

For each transcode:
1. The EXACT data format at each pipeline stage
2. Where the psychoacoustic masking happens (→ our alpha channel)
3. Where the codebook is (→ our Fibonacci codebook)
4. Where the hierarchy is (→ our HHTL)
5. What can run on INT8/BF16 without loss (→ our NPU path)
6. What needs FP32 (→ our hydration path)

## Architecture Split

```
ndarray = the compute crate:
  MDCT/inverse MDCT (or rustfft glue)
  BF16 band packing/unpacking
  SIMD-accelerated codebook lookup
  jitson-compiled audio kernels
  Voice archetype encoding

lance-graph = the memory/execution crate:
  Audio frames as graph nodes in neighborhoods.lance
  ZeckF64 edges between consecutive frames (temporal prediction)
  HHTL search for content-based audio retrieval
  Qualia annotation (feeling meta-layer)
  Progressive decode via column pruning
```

## The Breakthrough Claim

Current codecs are opaque pipelines: audio → compress → bitstream → decompress.
You can't search the bitstream. You can't navigate it. You can't query
"find the moment the key changes."

Our codec is a GRAPH: audio → transform → graph nodes → searchable.
HHTL-search for "frames that feel like emberglow" across a million songs in 1ms.
The audio IS the graph. The graph IS searchable. The search IS real-time.
