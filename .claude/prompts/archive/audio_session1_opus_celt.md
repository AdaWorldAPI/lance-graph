# SESSION 1: Opus CELT → ndarray Audio Primitives

> Read `.claude/knowledge/audio_research.md` FIRST.
> This is the foundation session. Everything else depends on it.

## Source

```
Repo:    xiph/opus (BSD, C, RFC 6716)
Focus:   celt/ directory ONLY (not SILK speech codec)
Clone:   curl -L https://github.com/xiph/opus/archive/refs/heads/main.zip -o opus.zip
Key files:
  celt/bands.c           — band energy normalization (gain-shape)
  celt/vq.c              — Pyramid Vector Quantizer
  celt/celt_encoder.c    — encoder pipeline
  celt/celt_decoder.c    — decoder pipeline
  celt/mdct.c            — MDCT transform
  celt/kiss_fft.c        — FFT for MDCT
  celt/modes.c           — sampling rates, frame sizes, band definitions
  include/opus_defines.h — constants
```

## What to Extract

### Stage 1: MDCT
```
INPUT:  960 PCM samples (48kHz, 20ms frame) as f32
OUTPUT: 480 frequency coefficients as f32

Read celt/mdct.c:
  clt_mdct_forward() — the forward transform
  clt_mdct_backward() — the inverse transform
  Uses kiss_fft internally

PRODUCE in ndarray:
  pub fn mdct_forward(pcm: &ArrayView1<f32>, window: &[f32]) -> Array1<f32>
  pub fn mdct_backward(coeffs: &ArrayView1<f32>, window: &[f32]) -> Array1<f32>

  Use rustfft crate for the FFT, not hand-rolled.
  The MDCT is: window → FFT → fold. ~30 lines of Rust.
  SIMD: the windowing multiply is element-wise f32 — use ndarray's * operator.
```

### Stage 2: Band Energy (the "Gain")
```
INPUT:  480 frequency coefficients
OUTPUT: 21 band energies as BF16 (Opus uses 21 bands at 48kHz)

Read celt/bands.c:
  compute_band_energies() — sums energy per quasi-Bark band
  normalise_bands() — normalizes coefficients by band energy
  eBands[] — the band boundary table (in modes.c)

PRODUCE in ndarray:
  pub const CELT_BANDS: [usize; 22] = [...]; // from modes.c eBands48
  pub fn band_energies(coeffs: &ArrayView1<f32>) -> Array1<bf16>
  pub fn normalize_bands(coeffs: &mut ArrayViewMut1<f32>, energies: &[bf16])

  Pack each energy as BF16:
    sign = always positive (energy)
    exponent = ~60dB dynamic range (8 bits = 255 steps, plenty)
    mantissa = fine energy detail (7 bits)

  This IS the gain in gain-shape quantization.
  The shape comes from the normalized coefficients.
```

### Stage 3: PVQ (the "Shape")
```
INPUT:  Normalized band coefficients (unit-energy per band)
OUTPUT: Integer pulses on L1 hypersphere

Read celt/vq.c:
  alg_quant() — quantize one band onto PVQ lattice
  alg_unquant() — dequantize
  cwrsi() / icwrs() — combinatorial enumeration (CWRS encoding)

PVQ distributes K integer pulses across N dimensions
such that sum(|pulse_i|) = K. The number of valid codewords
is C(N+K-1, K) — computed combinatorially, no trained codebook.

PRODUCE in ndarray:
  pub fn pvq_encode(band: &ArrayView1<f32>, k: u32) -> PvqCode
  pub fn pvq_decode(code: &PvqCode, n: usize) -> Array1<f32>
  pub fn cwrs_index(pulses: &[i32]) -> u64  // combinatorial index
  pub fn cwrs_from_index(index: u64, n: usize, k: u32) -> Vec<i32>

  NOTE: PVQ is algebraic. No codebook training. No learned weights.
  The L1 hypersphere IS our Fibonacci encoding of the SPO simplex.
  K pulses across N dimensions = Zeckendorf decomposition of energy.
```

### Stage 4: BF16 Graph Node Packing
```
INPUT:  21 band energies (BF16) + PVQ codes per band
OUTPUT: One graph node (48 bytes) for lance-graph

PRODUCE in ndarray:
  pub struct AudioFrame {
      pub band_energies: [bf16; 21],  // 42 bytes (gain)
      pub pvq_summary: [u8; 6],       // 6 bytes (shape fingerprint)
      // total: 48 bytes — one CAM-compatible fingerprint
  }

  The pvq_summary is a bgz17-style hash of the PVQ codes:
    HEEL (2 bytes): coarse spectral category
    HIP  (2 bytes): temporal pattern
    TWIG (2 bytes): harmonic structure

  This maps directly to SPO:
    Subject = spectral (WHAT frequencies)
    Predicate = temporal (WHEN they happen)
    Object = harmonic (HOW they ring)
```

## Integration Point

After this session, ndarray has:
```
src/audio/
  mdct.rs    — forward/backward MDCT via rustfft
  bands.rs   — Opus band definitions, energy computation, BF16 packing
  pvq.rs     — PVQ encode/decode, CWRS combinatorial indexing
  codec.rs   — AudioFrame type, encode(pcm) → frame, decode(frame) → pcm
```

lance-graph consumes these in Session 4 (integration).

## Tests

```
#[test] fn mdct_round_trip()     → forward then backward = original ± epsilon
#[test] fn band_energy_bf16()    → energy preserved through BF16 quantization
#[test] fn pvq_round_trip()      → encode then decode = original within PVQ precision
#[test] fn frame_48_bytes()      → AudioFrame serializes to exactly 48 bytes
#[test] fn opus_parity()         → compare our MDCT output with Opus reference on same input
```

## Dependencies

```toml
[dependencies]
rustfft = "6"       # FFT for MDCT
half = "2"           # BF16 type
ndarray = "0.16"     # array operations
```
