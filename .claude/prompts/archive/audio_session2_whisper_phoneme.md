# SESSION 2: Whisper → lance-graph Phoneme Graph

> Read `.claude/knowledge/audio_research.md` FIRST.
> Depends on Session 1 (MDCT/bands from Opus).

## Source

```
Repo:    openai/whisper (MIT, Python, 97K★)
Also:    ggerganov/whisper.cpp (MIT, C/C++, native implementation)
Clone:   curl -L https://github.com/openai/whisper/archive/refs/heads/main.zip
Key files:
  whisper/audio.py       — mel spectrogram computation
  whisper/model.py       — Transformer encoder/decoder
  whisper/decoding.py    — beam search, token selection
  whisper/tokenizer.py   — text ↔ token mapping
```

## What to Extract

### Stage 1: Mel Spectrogram (the INPUT representation)
```
Whisper:
  16kHz audio → STFT (400-point window, 160-point hop)
  → 201 frequency bins → 80 mel filterbank channels
  → log scale → normalize

This is NOT mysterious. It's:
  1. Windowed FFT (we have MDCT from Session 1, STFT is similar)
  2. Multiply by triangular filterbank matrix (80×201, precomputed)
  3. Log and normalize

PRODUCE in ndarray:
  pub fn mel_spectrogram(pcm: &ArrayView1<f32>, sr: u32) -> Array2<f32>
    // Returns [n_frames × 80] mel features
    // n_frames = (len - 400) / 160 + 1

  pub fn mel_filterbank(n_fft: usize, n_mels: usize, sr: u32) -> Array2<f32>
    // 80×201 triangular filterbank matrix
    // Precomputed at startup, stored as const

  SIMD: the filterbank multiply is a matrix-vector product per frame.
  ndarray's dot() handles this. AVX2/NEON dispatched automatically.
```

### Stage 2: What the Transformer Actually Computes
```
Whisper's encoder: 
  mel features [1500×80] → 32 transformer layers → encoder states [1500×512]
  Each layer: self-attention + FFN
  
Whisper's decoder:
  encoder states + previous tokens → next token probability
  Beam search selects the most likely sequence

The INSIGHT: the encoder converts spectral features to semantic features.
The decoder converts semantic features to text tokens.

For graph-based STT, we don't need the transformer.
We need:
  1. Spectral features (mel, from Stage 1)
  2. A mapping from spectral patterns to phonemes
  3. A graph that sequences phonemes into words
```

### Stage 3: Phoneme Graph (REPLACES the transformer)
```
Instead of attention over 1500 frames:
  Build a graph where nodes = phoneme candidates per frame
  Edges = transition probabilities between adjacent frames
  Path search = finding the most likely phoneme sequence
  
This IS graph traversal, not matrix multiplication.

PRODUCE in lance-graph:
  pub struct PhonemeNode {
      frame_idx: u32,
      phoneme_id: u16,     // one of ~40 English phonemes
      confidence: bf16,     // mel distance to phoneme template
      mel_fingerprint: [u8; 6],  // bgz17 CAM of the mel frame
  }
  
  Edges: PhonemeNode → PhonemeNode (transition weight)
  
  Recognition = shortest path through phoneme graph
  = HHTL cascade:
    HEEL: which broad phoneme class? (vowel/consonant/silence)
    HIP:  which specific phoneme? (template match)
    TWIG: word boundary detection (prosody features)
    LEAF: exact token selection
```

### Stage 4: Phoneme Templates from Whisper's Weights
```
Whisper's trained weights contain implicit phoneme knowledge.
Extract it WITHOUT running the full transformer:

  1. Run Whisper on a phoneme corpus (CMU Pronouncing Dictionary)
  2. For each phoneme: collect the encoder output vectors
  3. Average per phoneme → 40 template vectors (512D each)
  4. Compress to bgz17 CAM fingerprints (6 bytes each)
  5. Store as phoneme codebook in lance-graph

This is a ONE-TIME extraction. After this, the codebook
is static. No transformer needed at runtime.

PRODUCE:
  pub struct PhonemeCodebook {
      templates: [(PhonemeId, CamFingerprint); 40],
  }
  
  impl PhonemeCodebook {
      pub fn classify(&self, mel_frame: &[bf16; 80]) -> (PhonemeId, f32) {
          // Fibonacci-weighted lookup, not cosine distance
          // Returns best matching phoneme + confidence
      }
  }
```

## Integration Point

After this session:
```
ndarray:
  src/audio/mel.rs      — mel spectrogram, filterbank

lance-graph:
  graph/audio/
    phoneme.rs          — PhonemeNode, PhonemeCodebook, phoneme graph
    recognition.rs      — HHTL-based phoneme sequence search
```

## Tests

```
#[test] fn mel_matches_whisper()    → our mel output ≈ whisper.audio.log_mel_spectrogram()
#[test] fn phoneme_classify()      → codebook assigns correct phoneme to known samples
#[test] fn phoneme_graph_path()    → shortest path produces correct transcription
#[test] fn realtime_budget()       → mel + classify + path < 100ms for 1s audio
```
