# SESSION 3: Bark → lance-graph Voice Synthesis

> Read `.claude/knowledge/audio_research.md` FIRST.
> Depends on Sessions 1 (MDCT/PVQ) and 2 (mel/phoneme).

## Source

```
Repo:    suno-ai/bark (MIT, Python, 39K★)
Also:    facebookresearch/encodec (MIT, the codec Bark uses internally)
Clone:   curl -L https://github.com/suno-ai/bark/archive/refs/heads/main.zip
Key files:
  bark/generation.py     — THE 3-stage pipeline
  bark/model.py          — GPT-2 architecture (semantic → coarse → fine)
  bark/model_fine.py     — Fine-stage model
  bark/api.py            — High-level API
```

## What to Extract

### The 3-Stage Hierarchy (this IS our architecture)
```
Bark pipeline:
  Text → [Stage 1: Semantic GPT] → Semantic tokens (codebook 10K entries)
       → [Stage 2: Coarse GPT]   → Coarse audio tokens (EnCodec Q1-2)
       → [Stage 3: Fine Model]   → Fine audio tokens (EnCodec Q3-8)
       → [EnCodec decoder]       → Waveform

Our HHTL mapping:
  Text → [HEEL: qualia classification]     → Which voice archetype?
       → [HIP: SPO graph traversal]        → Which spectral envelope?
       → [TWIG: PVQ refinement]            → Which harmonic detail?
       → [LEAF: BF16 hydration + iMDCT]    → Waveform
```

### Stage 1: Voice Archetypes (REPLACES semantic GPT)
```
Bark's semantic model: GPT-2 with 10K token vocabulary
  Input: text tokens
  Output: semantic tokens representing "what to say and how"
  
What the semantic tokens encode:
  - WHICH phonemes to produce
  - WHAT prosody (pitch contour, timing, emphasis)
  - WHICH speaker characteristics (timbre, accent)

For us: these are QUALIA.
  A voice archetype = a qualia with ~16 channels:
    Channel 0-3:   fundamental frequency range (pitch)
    Channel 4-7:   formant structure (timbre)
    Channel 8-11:  prosodic pattern (rhythm, emphasis)
    Channel 12-15: spectral tilt (brightness, breathiness)

PRODUCE in ndarray:
  pub struct VoiceArchetype {
      channels: [i8; 16],           // qualia channels
      cam_fingerprint: [u8; 6],     // bgz17 compressed
  }

  pub struct VoiceCodebook {
      archetypes: Vec<VoiceArchetype>,  // extracted from Bark's embeddings
  }

EXTRACTION:
  1. Run Bark with different speaker prompts
  2. Capture semantic token distributions per speaker
  3. Cluster into archetypes (k=256 or 1024)
  4. Compress each archetype to 16 channels × i8
  5. This is a ONE-TIME extraction from Bark's weights
```

### Stage 2: Spectral Envelope from Graph (REPLACES coarse GPT)
```
Bark's coarse model: GPT-2 conditioned on semantic tokens
  Produces EnCodec codebook indices for quantizers 1-2
  These encode the spectral envelope (coarse energy per band)

For us: this is a GRAPH TRAVERSAL.
  Given voice archetype + phoneme sequence (from Session 2):
    1. Look up archetype in voice codebook → spectral template
    2. For each phoneme: modify template by phoneme characteristics
    3. Apply prosodic contour from archetype
    4. Output: 24 band energies as BF16 per frame (= gain from Session 1)

PRODUCE in lance-graph:
  pub fn synthesize_coarse(
      archetype: &VoiceArchetype,
      phonemes: &[PhonemeId],
      prosody: &ProsodyContour,
  ) -> Vec<[bf16; 24]>   // one set of band energies per frame

  This is graph traversal through a voice-phoneme graph:
    Node: (archetype, phoneme) pair
    Edge: transition weight (natural phoneme co-occurrence)
    The path through the graph IS the spectral envelope sequence
```

### Stage 3: Fine Detail from PVQ (REPLACES fine model)
```
Bark's fine model: fills in EnCodec quantizers 3-8
  Adds harmonic detail, consonant texture, noise

For us: PVQ from Session 1.
  Given coarse band energies (gain):
    1. PVQ allocates pulses per band (shape)
    2. Shape is predicted from archetype + phoneme
    3. Residual is the LEAF level — random detail
    4. Bark uses a neural net for this. We use PVQ + noise model.

PRODUCE in ndarray:
  pub fn synthesize_fine(
      coarse: &[[bf16; 24]],
      archetype: &VoiceArchetype,
      phonemes: &[PhonemeId],
  ) -> Vec<AudioFrame>   // full 48-byte frames with PVQ shape
```

### Stage 4: ElevenLabs Archetype Extraction (BONUS)
```
ElevenLabs voices have characteristic "fingerprints":
  - Adam: warm baritone, measured pace
  - Rachel: clear, bright, moderate pitch
  - Domi: expressive, wide dynamic range

These can be encoded as VoiceArchetypes:
  Record a few seconds of each voice
  Run through mel spectrogram (Session 2)
  Average the spectral characteristics
  Compress to 16-channel qualia format
  Store in voice codebook

PRODUCE:
  pub fn extract_archetype(samples: &[f32], sr: u32) -> VoiceArchetype
```

## Integration Point

After this session:
```
ndarray:
  src/audio/voice.rs     — VoiceArchetype, VoiceCodebook, extraction
  src/audio/rvq.rs       — RVQ hierarchy mapped from Bark's 3 stages

lance-graph:
  graph/audio/
    synthesis.rs         — synthesize_coarse(), synthesize_fine()
    archetype.rs         — Voice archetype graph (speaker × phoneme nodes)
```

## Tests

```
#[test] fn archetype_round_trip()  → encode then decode voice preserves timbre
#[test] fn coarse_synthesis()      → band energies match expected spectral shape
#[test] fn fine_pvq_quality()      → A/B test: our PVQ vs Bark's fine model
#[test] fn tts_pipeline()          → text → phonemes → coarse → fine → WAV plays
```
