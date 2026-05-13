//! # Universal Perception Experiment
//!
//! THE experiment: does the VSA accumulator noise floor correlate with
//! human perceptual masking thresholds ACROSS modalities?
//!
//! Three tests:
//!   1. AUDIO: noise floor vs ISO 11172-3 psychoacoustic masking curve
//!   2. TEXT:  noise floor vs cognitive salience (entity importance)
//!   3. VIDEO: noise floor vs psychovisual contrast sensitivity
//!
//! If all three correlations are r > 0.5, the accumulator IS a universal
//! perceptual model. The 25% noise floor IS the masking threshold.
//! Psychoacoustic model design becomes unnecessary — the accumulator
//! discovers it from the statistics of the signal.
//!
//! If any correlation is r < 0.3, the mapping is substrate-specific
//! and doesn't transfer. Still useful, but not universal.

// ═══════════════════════════════════════════════════════════════════════
// SHARED: The Universal Accumulator
// ═══════════════════════════════════════════════════════════════════════

/// A modality-agnostic accumulator.
/// Works on any signal decomposed into N bands of B bits each.
/// The SAME accumulator structure for audio, text, and video.
pub struct UniversalAccumulator {
    /// Accumulator cells: N_bands × B_bits_per_band.
    pub cells: Vec<i16>,
    /// Band count (24 for audio, 16 for text qualia, 32 for video).
    pub n_bands: usize,
    /// Bits per band (16 for BF16, 8 for byte, etc).
    pub bits_per_band: usize,
    /// Frame count accumulated.
    pub frame_count: u32,
}

impl UniversalAccumulator {
    pub fn new(n_bands: usize, bits_per_band: usize) -> Self {
        Self {
            cells: vec![0i16; n_bands * bits_per_band],
            n_bands,
            bits_per_band,
            frame_count: 0,
        }
    }

    /// Accumulate one frame: N_bands values, each B bits.
    /// Direct accumulation — no cyclic shift (decorrelation at encoding level).
    pub fn accumulate(&mut self, band_values: &[u16]) {
        assert_eq!(band_values.len(), self.n_bands);

        for band in 0..self.n_bands {
            for bit in 0..self.bits_per_band {
                let cell_idx = band * self.bits_per_band + bit;
                let bit_val = ((band_values[band] >> bit) & 1) as i16;
                let bipolar = bit_val * 2 - 1;
                self.cells[cell_idx] = self.cells[cell_idx].saturating_add(bipolar);
            }
        }
        self.frame_count += 1;
    }

    /// Alpha density: fraction of cells above threshold.
    pub fn alpha_density(&self, threshold: i16) -> f64 {
        let above = self.cells.iter().filter(|&&c| c.abs() > threshold).count();
        above as f64 / self.cells.len() as f64
    }

    /// Noise floor per band: fraction of bits below threshold in each band.
    /// Returns [0.0, 1.0] per band. 1.0 = all bits below threshold = full noise.
    pub fn noise_floor_per_band(&self, threshold: i16) -> Vec<f64> {
        (0..self.n_bands)
            .map(|band| {
                let below = (0..self.bits_per_band)
                    .filter(|&bit| {
                        let idx = band * self.bits_per_band + bit;
                        self.cells[idx].abs() <= threshold
                    })
                    .count();
                below as f64 / self.bits_per_band as f64
            })
            .collect()
    }

    /// THE MEASUREMENT: Pearson correlation between noise floor and masking curve.
    pub fn noise_mask_correlation(&self, threshold: i16, mask_curve: &[f64]) -> f64 {
        let noise = self.noise_floor_per_band(threshold);
        assert_eq!(noise.len(), mask_curve.len(),
            "Noise bands ({}) != mask bands ({})", noise.len(), mask_curve.len());
        pearson(&noise, mask_curve)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// EXPERIMENT 1: AUDIO — noise floor vs psychoacoustic masking
// ═══════════════════════════════════════════════════════════════════════

/// ISO 11172-3 absolute threshold of hearing (dB SPL) per Bark band.
/// These are the quietest sounds humans can perceive per frequency range.
/// Normalized to [0,1] for correlation (higher = more easily masked).
const AUDIO_MASKING_CURVE: [f64; 24] = [
    0.80, 0.50, 0.30, 0.20, 0.15, 0.12, 0.10, 0.10,  // Bark 1-8 (20-920Hz)
    0.12, 0.13, 0.15, 0.18, 0.22, 0.28, 0.35, 0.42,  // Bark 9-16 (920-3700Hz)
    0.50, 0.58, 0.65, 0.72, 0.78, 0.83, 0.88, 0.95,  // Bark 17-24 (3700-24000Hz)
];
// Low frequencies: moderate threshold (we hear bass OK but not great)
// Mid frequencies: low threshold (most sensitive, speech range)
// High frequencies: high threshold (rapidly losing sensitivity)

/// Generate a synthetic speech-like signal: harmonics at 150Hz fundamental
/// with formant peaks at 500Hz, 1500Hz, 2500Hz.
fn synthetic_speech(n_frames: usize) -> Vec<[u16; 24]> {
    (0..n_frames)
        .map(|frame| {
            let t = frame as f64 / 75.0; // time in seconds
            let mut bands = [0u16; 24];

            // Fundamental + harmonics (150Hz = Bark band 2-3)
            bands[2] = 0x3C00; // high energy
            bands[3] = 0x3800;

            // Formant F1 at ~500Hz (Bark band 4-5)
            bands[4] = 0x3E00;
            bands[5] = 0x3A00;

            // Formant F2 at ~1500Hz (Bark band 11-12)
            bands[11] = 0x3C00;
            bands[12] = 0x3800;

            // Formant F3 at ~2500Hz (Bark band 14-15)
            bands[14] = 0x3400;
            bands[15] = 0x3000;

            // Breath noise (high bands, time-varying)
            let noise_phase = (t * 7.3).sin().abs();
            bands[20] = (0x2000 as f64 * noise_phase) as u16;
            bands[21] = (0x1800 as f64 * noise_phase) as u16;

            bands
        })
        .collect()
}

pub fn experiment_audio() -> ExperimentResult {
    let mut acc = UniversalAccumulator::new(24, 16);
    let frames = synthetic_speech(225); // 3 seconds at 75fps

    for frame in &frames {
        acc.accumulate(frame);
    }

    let threshold = 10i16;
    let r = acc.noise_mask_correlation(threshold, &AUDIO_MASKING_CURVE);
    let alpha = acc.alpha_density(threshold);
    let noise = acc.noise_floor_per_band(threshold);

    ExperimentResult {
        modality: "AUDIO".to_string(),
        correlation: r,
        alpha_density: alpha,
        noise_floor: noise,
        masking_curve: AUDIO_MASKING_CURVE.to_vec(),
        frame_count: 225,
        threshold,
        hypothesis_supported: r > 0.5,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// EXPERIMENT 2: TEXT — noise floor vs cognitive salience
// ═══════════════════════════════════════════════════════════════════════

/// Cognitive salience per "band" for text.
/// Text decomposed into 16 semantic channels (matching our 16 qualia channels).
/// Salience = how important each channel is for human comprehension.
///
/// Based on Kintsch (1998) construction-integration model:
/// core meaning > relations > context > style > redundancy
const TEXT_SALIENCE_CURVE: [f64; 16] = [
    0.10, // channel 0: core entity type (MOST salient — always needed)
    0.12, // channel 1: primary relation
    0.15, // channel 2: secondary relation
    0.20, // channel 3: attribute values
    0.25, // channel 4: temporal context
    0.30, // channel 5: spatial context
    0.35, // channel 6: causal links
    0.40, // channel 7: emotional valence
    0.50, // channel 8: specificity level
    0.55, // channel 9: confidence/hedging
    0.60, // channel 10: source attribution
    0.65, // channel 11: pragmatic implicature
    0.70, // channel 12: stylistic markers
    0.78, // channel 13: redundant elaboration
    0.85, // channel 14: filler/discourse markers
    0.95, // channel 15: noise/formatting (LEAST salient — easily masked)
];

/// Generate synthetic text encounters: repeated entity mentions with
/// varying levels of detail across channels.
fn synthetic_text_encounters(n_encounters: usize) -> Vec<[u16; 16]> {
    (0..n_encounters)
        .map(|enc| {
            let mut channels = [0u16; 16];
            // Core channels repeat consistently (will crystallize)
            channels[0] = 0xABCD; // entity type — same every encounter
            channels[1] = 0x1234; // primary relation — same
            channels[2] = 0x5678; // secondary relation — mostly same

            // Context channels vary moderately
            let phase = enc as u16;
            channels[4] = 0x3000 ^ (phase.wrapping_mul(7));
            channels[5] = 0x4000 ^ (phase.wrapping_mul(13));
            channels[6] = 0x5000 ^ (phase.wrapping_mul(3));

            // Style/noise channels vary heavily (won't crystallize)
            channels[12] = phase.wrapping_mul(97);
            channels[13] = phase.wrapping_mul(151);
            channels[14] = phase.wrapping_mul(211);
            channels[15] = phase.wrapping_mul(257);

            // Intermediate channels: some consistency
            channels[7] = 0x7000 ^ (phase.wrapping_mul(5) & 0x0FFF);
            channels[8] = 0x8000 ^ (phase.wrapping_mul(11) & 0x1FFF);
            channels[9] = 0x9000 ^ (phase.wrapping_mul(17) & 0x3FFF);
            channels[10] = 0xA000 ^ (phase.wrapping_mul(23) & 0x7FFF);
            channels[11] = 0xB000 ^ (phase.wrapping_mul(29) & 0x7FFF);

            // Fill in channel 3
            channels[3] = 0x2000 ^ (phase.wrapping_mul(2) & 0x00FF);

            channels
        })
        .collect()
}

pub fn experiment_text() -> ExperimentResult {
    let mut acc = UniversalAccumulator::new(16, 16);
    let encounters = synthetic_text_encounters(100);

    for enc in &encounters {
        acc.accumulate(enc);
    }

    let threshold = 10i16;
    let r = acc.noise_mask_correlation(threshold, &TEXT_SALIENCE_CURVE);
    let alpha = acc.alpha_density(threshold);
    let noise = acc.noise_floor_per_band(threshold);

    ExperimentResult {
        modality: "TEXT".to_string(),
        correlation: r,
        alpha_density: alpha,
        noise_floor: noise,
        masking_curve: TEXT_SALIENCE_CURVE.to_vec(),
        frame_count: 100,
        threshold,
        hypothesis_supported: r > 0.5,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// EXPERIMENT 3: VIDEO — noise floor vs psychovisual contrast sensitivity
// ═══════════════════════════════════════════════════════════════════════

/// Contrast Sensitivity Function (CSF) per spatial frequency band.
/// 32 bands from DC to Nyquist. The human visual system is most sensitive
/// at ~4 cycles/degree and falls off at both low and high frequencies.
///
/// Based on Mannos & Sakrison (1974) CSF model.
/// Normalized to [0,1] where higher = MORE easily masked (LESS sensitive).
const VIDEO_CSF_CURVE: [f64; 32] = [
    0.70, 0.55, 0.40, 0.28, 0.18, 0.12, 0.08, 0.06, // DC to ~4 cpd: improving sensitivity
    0.05, 0.06, 0.08, 0.10, 0.13, 0.17, 0.22, 0.28, // ~4-16 cpd: peak then declining
    0.35, 0.42, 0.50, 0.57, 0.63, 0.70, 0.76, 0.81, // 16-32 cpd: rapidly losing sensitivity
    0.85, 0.88, 0.91, 0.93, 0.95, 0.97, 0.98, 0.99, // 32-64 cpd: nearly blind
];

/// Generate synthetic video blocks: a mix of edges (low frequency energy)
/// and texture (high frequency energy), with static background and motion.
fn synthetic_video_blocks(n_frames: usize) -> Vec<[u16; 32]> {
    (0..n_frames)
        .map(|frame| {
            let mut bands = [0u16; 32];
            let t = frame as f64 / 30.0; // 30 fps

            // DC component (always present, static background)
            bands[0] = 0x4000;

            // Low-frequency edges (static structure, will crystallize)
            bands[1] = 0x3800;
            bands[2] = 0x3400;
            bands[3] = 0x3000;

            // Mid-frequency texture (slowly varying)
            let texture_phase = (t * 2.0).sin() * 0.3 + 0.7;
            for b in 8..16 {
                bands[b] = (0x2000 as f64 * texture_phase) as u16;
            }

            // High-frequency detail (rapidly varying = noise-like)
            let noise_seed = frame as u16;
            for b in 20..32 {
                bands[b] = noise_seed.wrapping_mul(97 + b as u16) & 0x1FFF;
            }

            // Motion edge (appears/disappears)
            if frame % 30 > 10 && frame % 30 < 20 {
                bands[4] = 0x3C00; // sharp edge during motion
                bands[5] = 0x3800;
            }

            bands
        })
        .collect()
}

pub fn experiment_video() -> ExperimentResult {
    let mut acc = UniversalAccumulator::new(32, 16);
    let blocks = synthetic_video_blocks(150); // 5 seconds at 30fps

    for block in &blocks {
        acc.accumulate(block);
    }

    let threshold = 10i16;
    let r = acc.noise_mask_correlation(threshold, &VIDEO_CSF_CURVE);
    let alpha = acc.alpha_density(threshold);
    let noise = acc.noise_floor_per_band(threshold);

    ExperimentResult {
        modality: "VIDEO".to_string(),
        correlation: r,
        alpha_density: alpha,
        noise_floor: noise,
        masking_curve: VIDEO_CSF_CURVE.to_vec(),
        frame_count: 150,
        threshold,
        hypothesis_supported: r > 0.5,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MEASUREMENT FRAMEWORK
// ═══════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct ExperimentResult {
    pub modality: String,
    pub correlation: f64,
    pub alpha_density: f64,
    pub noise_floor: Vec<f64>,
    pub masking_curve: Vec<f64>,
    pub frame_count: usize,
    pub threshold: i16,
    pub hypothesis_supported: bool,
}

/// Run all three experiments and produce the universal verdict.
pub fn universal_perception_experiment() -> UniversalVerdict {
    let audio = experiment_audio();
    let text = experiment_text();
    let video = experiment_video();

    let avg_correlation = (audio.correlation + text.correlation + video.correlation) / 3.0;
    let all_supported = audio.hypothesis_supported
        && text.hypothesis_supported
        && video.hypothesis_supported;

    let verdict = if all_supported {
        "UNIVERSAL: The accumulator IS the perceptual model across all modalities."
    } else if avg_correlation > 0.3 {
        "PARTIAL: Correlation exists but is substrate-specific, not universal."
    } else {
        "REJECTED: The noise floor does not correlate with perceptual masking."
    };

    UniversalVerdict {
        audio,
        text,
        video,
        average_correlation: avg_correlation,
        all_supported,
        verdict: verdict.to_string(),
    }
}

#[derive(Clone, Debug)]
pub struct UniversalVerdict {
    pub audio: ExperimentResult,
    pub text: ExperimentResult,
    pub video: ExperimentResult,
    pub average_correlation: f64,
    pub all_supported: bool,
    pub verdict: String,
}

impl std::fmt::Display for UniversalVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\n{}", "═".repeat(72))?;
        writeln!(f, "  UNIVERSAL PERCEPTION EXPERIMENT")?;
        writeln!(f, "  Hypothesis: VSA noise floor = perceptual masking threshold")?;
        writeln!(f, "{}", "═".repeat(72))?;
        writeln!(f)?;

        for exp in [&self.audio, &self.text, &self.video] {
            writeln!(f, "  {} ({}  frames, threshold={})",
                exp.modality, exp.frame_count, exp.threshold)?;
            writeln!(f, "    Noise↔Mask correlation:  r = {:.4}", exp.correlation)?;
            writeln!(f, "    Alpha density:           {:.3}", exp.alpha_density)?;
            writeln!(f, "    Hypothesis (r > 0.5):    {}",
                if exp.hypothesis_supported { "✓ SUPPORTED" } else { "✗ NOT SUPPORTED" })?;

            // Show per-band comparison (first 8 bands)
            let n_show = exp.noise_floor.len().min(8);
            write!(f, "    Noise floor: [")?;
            for i in 0..n_show {
                write!(f, "{:.2}", exp.noise_floor[i])?;
                if i < n_show - 1 { write!(f, ", ")?; }
            }
            writeln!(f, ", ...]")?;
            write!(f, "    Mask curve:  [")?;
            for i in 0..n_show {
                write!(f, "{:.2}", exp.masking_curve[i])?;
                if i < n_show - 1 { write!(f, ", ")?; }
            }
            writeln!(f, ", ...]")?;
            writeln!(f)?;
        }

        writeln!(f, "  {}", "─".repeat(60))?;
        writeln!(f, "  AVERAGE CORRELATION: r = {:.4}", self.average_correlation)?;
        writeln!(f, "  VERDICT: {}", self.verdict)?;
        writeln!(f, "{}", "═".repeat(72))
    }
}

/// Pearson correlation coefficient.
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    if n < 2.0 { return 0.0; }

    let mx: f64 = x.iter().sum::<f64>() / n;
    let my: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }

    if vx < 1e-15 || vy < 1e-15 { return 0.0; }
    cov / (vx.sqrt() * vy.sqrt())
}

// ═══════════════════════════════════════════════════════════════════════
// TESTS: Run the experiments
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_universal_perception() {
        let verdict = universal_perception_experiment();
        println!("{}", verdict);

        // The experiment produces a number. We don't assert the hypothesis —
        // we MEASURE it. The test passes if the code runs.
        // The RESULT tells us whether the hypothesis holds.
        assert!(verdict.audio.correlation.is_finite());
        assert!(verdict.text.correlation.is_finite());
        assert!(verdict.video.correlation.is_finite());
    }

    #[test]
    fn test_audio_alone() {
        let result = experiment_audio();
        println!("\nAUDIO: r = {:.4}, α = {:.3}", result.correlation, result.alpha_density);
        println!("Noise: {:?}", &result.noise_floor[..8]);
        println!("Mask:  {:?}", &result.masking_curve[..8]);
    }

    #[test]
    fn test_text_alone() {
        let result = experiment_text();
        println!("\nTEXT: r = {:.4}, α = {:.3}", result.correlation, result.alpha_density);

        // Core channels (0-2) should crystallize (low noise floor)
        assert!(result.noise_floor[0] < result.noise_floor[15],
            "Core entity should crystallize more than noise channel");
    }

    #[test]
    fn test_video_alone() {
        let result = experiment_video();
        println!("\nVIDEO: r = {:.4}, α = {:.3}", result.correlation, result.alpha_density);

        // DC/low frequency should crystallize (static background)
        assert!(result.noise_floor[0] < result.noise_floor[31],
            "DC should crystallize more than highest frequency");
    }

    #[test]
    fn test_threshold_sweep() {
        // How does correlation change with threshold?
        let mut acc = UniversalAccumulator::new(24, 16);
        let frames = synthetic_speech(225);
        for f in &frames { acc.accumulate(f); }

        println!("\nThreshold sweep (audio):");
        println!("{:>10} {:>10} {:>10}", "threshold", "r", "alpha");
        for threshold in [1, 5, 10, 20, 30, 50, 75, 100] {
            let r = acc.noise_mask_correlation(threshold, &AUDIO_MASKING_CURVE);
            let alpha = acc.alpha_density(threshold);
            println!("{:>10} {:>10.4} {:>10.3}", threshold, r, alpha);
        }
    }

    #[test]
    fn test_accumulation_convergence() {
        // How does correlation evolve as frames accumulate?
        println!("\nConvergence test (audio):");
        println!("{:>10} {:>10} {:>10}", "frames", "r", "alpha");

        let all_frames = synthetic_speech(300);
        let threshold = 10i16;

        for n_frames in [10, 25, 50, 75, 100, 150, 225, 300] {
            let mut acc = UniversalAccumulator::new(24, 16);
            for f in &all_frames[..n_frames] { acc.accumulate(f); }
            let r = acc.noise_mask_correlation(threshold, &AUDIO_MASKING_CURVE);
            let alpha = acc.alpha_density(threshold);
            println!("{:>10} {:>10.4} {:>10.3}", n_frames, r, alpha);
        }
    }

    #[test]
    fn test_pearson_sanity() {
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((pearson(&x, &y) - 1.0).abs() < 1e-10);

        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        assert!((pearson(&x, &y_neg) + 1.0).abs() < 1e-10);

        // Zero correlation (orthogonal)
        let y_zero = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        assert!(pearson(&x, &y_zero).abs() < 0.5);
    }
}
