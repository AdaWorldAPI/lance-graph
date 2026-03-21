//! # lance-graph-codec-research
//!
//! Research crate comparing three audio encoding strategies:
//!
//! - **Strategy A** (per-frame): MDCT → BF16 bands → one graph node per frame
//! - **Strategy B** (accumulator): frames → VSA bundle → crystallize → few nodes
//! - **Strategy C** (hybrid): both paths, per-frame for navigation + accumulator for identity
//!
//! The core hypothesis: VSA streaming accumulation IS lossy compression where
//! the 25% noise floor IS the psychoacoustic masking threshold. What crystallizes
//! is what repeats. What repeats is what humans perceive. What doesn't is noise.

pub mod transform;
pub mod bands;
pub mod perframe;
pub mod accumulator;
pub mod hybrid;
pub mod diamond;
pub mod metrics;
pub mod universal_perception;
pub mod zeckbf17;

/// Sample rate for all experiments (CD-quality mono).
pub const SAMPLE_RATE: u32 = 48000;

/// Samples per frame. 640 samples = 13.3ms at 48kHz.
/// Matches EnCodec/SoundStream frame rate of 75 fps.
pub const SAMPLES_PER_FRAME: usize = 640;

/// Frame rate (frames per second).
pub const FRAME_RATE: u32 = 75;

/// Number of Bark-scale critical bands.
pub const BARK_BANDS: usize = 24;

/// One encoded audio frame (Strategy A output).
#[derive(Clone, Debug)]
pub struct AudioFrame {
    /// Frame index from start of audio.
    pub idx: u64,
    /// 24 BF16 band energies — the spectral snapshot.
    pub bands: [u16; BARK_BANDS],
    /// 24 BF16 temporal deltas (energy change from previous frame).
    pub temporal: [u16; BARK_BANDS],
    /// 24 BF16 harmonic ratios (band[k] / band[fundamental]).
    pub harmonic: [u16; BARK_BANDS],
    /// Psychoacoustic masking threshold per band.
    pub mask: [u16; BARK_BANDS],
}

/// Crystallized spectral component (Strategy B output / Diamond Markov epiphany).
#[derive(Clone, Debug)]
pub struct CrystallizedComponent {
    /// What crystallized: the sign bits of the accumulator above threshold.
    pub spectrum: [u16; BARK_BANDS],
    /// Alpha mask: which bands had sufficient evidence.
    pub alpha: [bool; BARK_BANDS],
    /// How many frames contributed before crystallization.
    pub encounter_count: u32,
    /// Which qualia this component represents.
    pub qualia: AudioQualia,
    /// Frame range during which this component was active.
    pub start_frame: u64,
    pub end_frame: u64,
}

/// Accumulator state: i16 per BF16 bit position (384 accumulators for 24×16 bits).
/// Saturating arithmetic. Sign = belief. Magnitude = confidence.
#[derive(Clone, Debug)]
pub struct SpectralAccumulator {
    /// 24 bands × 16 bits per BF16 = 384 accumulator cells.
    /// Each cell: i16, saturating. Sign = bit belief. |value| = confidence.
    pub cells: [i16; BARK_BANDS * 16],
    /// Number of frames accumulated so far.
    pub frame_count: u32,
}

/// The four audio qualia feelings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AudioQualia {
    /// Bright, cutting, exposed. High spectral centroid.
    Steelwind = 0,
    /// Round, resonant, grounded. Mid centroid, high coherence.
    Woodwarm = 1,
    /// Rich, swelling, intense. Rising energy.
    Emberglow = 2,
    /// Soft, held, suspended. Falling energy, long sustain.
    Velvetpause = 3,
}

/// Results from comparing two encoding strategies on the same audio.
#[derive(Clone, Debug)]
pub struct ComparisonResult {
    pub strategy_name: String,
    /// Bits per second of the encoded representation.
    pub bitrate_bps: f64,
    /// Compression ratio vs raw PCM (48000 samples × 16 bits = 768000 bps).
    pub compression_ratio: f64,
    /// Spectral distortion in dB (lower = better).
    pub spectral_distortion_db: f64,
    /// Correlation between original and reconstructed band energies.
    pub band_energy_correlation: f64,
    /// Number of graph nodes produced.
    pub node_count: usize,
    /// Number of crystallized components (Strategy B/C only).
    pub crystallized_count: usize,
    /// Alpha density: fraction of accumulator cells above threshold.
    pub alpha_density: f64,
    /// Noise floor correlation with masking threshold (THE hypothesis).
    pub noise_mask_correlation: f64,
    /// Diamond Markov invariant: Hamming distance after rebundling.
    pub invariant_hamming: u64,
}
