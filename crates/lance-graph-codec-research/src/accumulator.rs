//! Strategy B: Streaming Accumulator Encoding
//!
//! THE core research hypothesis. Frames are bundled into a VSA accumulator
//! via cyclic shift. After N frames, the accumulator's sign bits hold the
//! CRYSTALLIZED spectrum — what persisted across frames. The noise floor
//! (|acc| ≤ threshold) IS the psychoacoustic masking threshold.
//!
//! 1 accumulator per second = 48 bytes/s = 384 bps.
//! Quality: parametric (tonal components survive, transients don't).
//! The Diamond Markov pipeline extracts crystallized components progressively.

use crate::{AudioFrame, AudioQualia, CrystallizedComponent, SpectralAccumulator,
            BARK_BANDS, SAMPLES_PER_FRAME, FRAME_RATE};
use crate::transform::{mdct, coeffs_to_band_energies, psychoacoustic_mask, sine_window};
use crate::bands::{pack_bands, f32_to_bf16, bf16_to_f32};

/// Accumulator cell count: 24 bands × 16 bits per BF16 = 384.
const CELL_COUNT: usize = BARK_BANDS * 16;

impl SpectralAccumulator {
    /// Create a fresh accumulator (all cells zero).
    pub fn new() -> Self {
        Self {
            cells: [0i16; CELL_COUNT],
            frame_count: 0,
        }
    }

    /// Accumulate one frame into the accumulator.
    ///
    /// The frame's BF16 bands are unpacked to 384 individual bits.
    /// Each bit is added to the accumulator via saturating i16 add.
    ///
    /// NOTE: Earlier versions used a cyclic shift per frame for decorrelation.
    /// This caused a mismatch: crystallize() and unbind() read un-shifted cells.
    /// The shift is removed — decorrelation is handled at the encoding level
    /// by ZeckBF17's golden-step traversal, not at the accumulation level.
    pub fn accumulate_frame(&mut self, bands: &[u16; BARK_BANDS]) {
        // Unpack 24 BF16 values into 384 bits
        for band in 0..BARK_BANDS {
            for bit in 0..16 {
                let cell_idx = band * 16 + bit;

                // Extract bit from BF16 value
                let bit_val = ((bands[band] >> bit) & 1) as i16;
                // Map 0→-1, 1→+1 for bipolar accumulation
                let bipolar = bit_val * 2 - 1;

                // Saturating add — NO shift, cells stay at natural position
                self.cells[cell_idx] = self.cells[cell_idx].saturating_add(bipolar);
            }
        }

        self.frame_count += 1;
    }

    /// Accumulate a batch of frames (e.g., one second = 75 frames).
    pub fn accumulate_pcm(&mut self, samples: &[f32]) {
        let window = sine_window(SAMPLES_PER_FRAME * 2);
        let hop = SAMPLES_PER_FRAME;
        let n_frames = samples.len().saturating_sub(SAMPLES_PER_FRAME * 2) / hop;

        for idx in 0..n_frames {
            let offset = idx * hop;
            let chunk = &samples[offset..offset + SAMPLES_PER_FRAME * 2];

            let mut windowed = vec![0.0f32; chunk.len()];
            for i in 0..chunk.len() {
                windowed[i] = chunk[i] * window[i];
            }

            let mut coeffs = vec![0.0f32; SAMPLES_PER_FRAME];
            mdct(&windowed, &mut coeffs);

            let energies = coeffs_to_band_energies(&coeffs);
            let packed = pack_bands(&energies);

            self.accumulate_frame(&packed);
        }
    }

    /// Extract the crystallized spectrum: sign bits where |cell| > threshold.
    ///
    /// The threshold controls the bitrate/quality tradeoff:
    ///   threshold = 1: everything crystallizes → ~3000 bps (high quality)
    ///   threshold = frame_count/4: only strong patterns → ~384 bps (parametric)
    ///   threshold = frame_count/2: only dominant tones → ~100 bps (pitch only)
    pub fn crystallize(&self, threshold: i16) -> CrystallizedComponent {
        let mut spectrum = [0u16; BARK_BANDS];
        let mut alpha = [false; BARK_BANDS];
        let mut confident_count = 0u32;

        for band in 0..BARK_BANDS {
            let mut bf16_val = 0u16;
            let mut band_confident = true;

            for bit in 0..16 {
                let cell_idx = band * 16 + bit;
                if self.cells[cell_idx].abs() > threshold {
                    // Crystallized: sign bit becomes the belief
                    if self.cells[cell_idx] > 0 {
                        bf16_val |= 1 << bit;
                    }
                    confident_count += 1;
                } else {
                    // Below threshold: noise floor. This bit is uncertain.
                    band_confident = false;
                }
            }

            spectrum[band] = bf16_val;
            alpha[band] = band_confident;
        }

        let qualia = classify_accumulated_qualia(&spectrum, &alpha);

        CrystallizedComponent {
            spectrum,
            alpha,
            encounter_count: self.frame_count,
            qualia,
            start_frame: 0,
            end_frame: self.frame_count as u64,
        }
    }

    /// Alpha density: fraction of cells above threshold.
    /// This measures how much of the signal has crystallized.
    /// Hypothesis: this correlates with the psychoacoustic masking ratio.
    pub fn alpha_density(&self, threshold: i16) -> f64 {
        let confident = self.cells.iter().filter(|&&c| c.abs() > threshold).count();
        confident as f64 / CELL_COUNT as f64
    }

    /// Get the noise floor: cells where |cell| ≤ threshold.
    /// Returns the MAGNITUDE of uncertain cells.
    /// Hypothesis: this distribution matches the psychoacoustic masking thresholds.
    pub fn noise_floor(&self, threshold: i16) -> Vec<(usize, i16)> {
        self.cells.iter().enumerate()
            .filter(|(_, &c)| c.abs() <= threshold)
            .map(|(i, &c)| (i, c))
            .collect()
    }

    /// Compare noise floor distribution against psychoacoustic masking thresholds.
    /// Returns Pearson correlation. THE key measurement of the hypothesis.
    pub fn noise_mask_correlation(&self, threshold: i16, mask: &[f32; BARK_BANDS]) -> f64 {
        // For each band: compute fraction of cells below threshold
        let mut noise_fraction = [0.0f64; BARK_BANDS];
        for band in 0..BARK_BANDS {
            let mut below = 0;
            for bit in 0..16 {
                let cell_idx = band * 16 + bit;
                if self.cells[cell_idx].abs() <= threshold {
                    below += 1;
                }
            }
            noise_fraction[band] = below as f64 / 16.0;
        }

        // Normalize mask to [0, 1]
        let mask_max = mask.iter().copied().fold(0.0f32, f32::max);
        let mask_norm: Vec<f64> = mask.iter()
            .map(|&m| if mask_max > 0.0 { m as f64 / mask_max as f64 } else { 0.0 })
            .collect();

        // Pearson correlation between noise_fraction and mask_norm
        pearson_correlation(&noise_fraction, &mask_norm)
    }

    /// Reset the accumulator for a new segment.
    pub fn reset(&mut self) {
        self.cells = [0i16; CELL_COUNT];
        self.frame_count = 0;
    }

    /// XOR-unbind a crystallized component from the accumulator.
    /// This removes the crystallized pattern, leaving the residual.
    /// Diamond Markov invariant: accumulator + component = original.
    pub fn unbind(&mut self, component: &CrystallizedComponent) {
        for band in 0..BARK_BANDS {
            for bit in 0..16 {
                let cell_idx = band * 16 + bit;
                let comp_bit = ((component.spectrum[band] >> bit) & 1) as i16;
                let bipolar = comp_bit * 2 - 1;
                // Subtract the component's contribution
                self.cells[cell_idx] = self.cells[cell_idx].saturating_sub(bipolar);
            }
        }
    }
}

/// Classify accumulated qualia from crystallized spectrum.
fn classify_accumulated_qualia(spectrum: &[u16; BARK_BANDS], alpha: &[bool; BARK_BANDS]) -> AudioQualia {
    let energies: [f32; BARK_BANDS] = core::array::from_fn(|i| bf16_to_f32(spectrum[i]));
    let total: f32 = energies.iter().sum();
    if total < 1e-10 {
        return AudioQualia::Velvetpause;
    }

    let centroid: f32 = energies.iter().enumerate()
        .map(|(i, &e)| i as f32 * e).sum::<f32>() / total;

    let alpha_count = alpha.iter().filter(|&&a| a).count();
    let alpha_ratio = alpha_count as f32 / BARK_BANDS as f32;

    if centroid > 16.0 && alpha_ratio > 0.7 {
        AudioQualia::Steelwind
    } else if centroid < 8.0 && alpha_ratio > 0.5 {
        AudioQualia::Woodwarm
    } else if alpha_ratio < 0.3 {
        AudioQualia::Velvetpause
    } else {
        AudioQualia::Emberglow
    }
}

/// Pearson correlation coefficient between two equal-length slices.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-15 || var_y < 1e-15 {
        return 0.0;
    }
    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Decode a crystallized component back to band energies.
/// Only confident bands (alpha=true) are decoded. Others stay zero.
pub fn decode_crystallized(component: &CrystallizedComponent) -> [f32; BARK_BANDS] {
    let mut energies = [0.0f32; BARK_BANDS];
    for band in 0..BARK_BANDS {
        if component.alpha[band] {
            energies[band] = bf16_to_f32(component.spectrum[band]);
        }
    }
    energies
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_silence() {
        let mut acc = SpectralAccumulator::new();
        let zero_bands = [0u16; BARK_BANDS];
        for _ in 0..75 {
            acc.accumulate_frame(&zero_bands);
        }
        assert_eq!(acc.frame_count, 75);
        // Silence: all cells should be negative (0 bits → bipolar -1)
        for &cell in &acc.cells {
            assert!(cell <= 0, "Silence should produce non-positive accumulator cells");
        }
    }

    #[test]
    fn test_accumulator_constant_signal() {
        let mut acc = SpectralAccumulator::new();
        // Same bands every frame → should crystallize quickly
        let constant_bands: [u16; BARK_BANDS] = core::array::from_fn(|i| 0x3F80 + i as u16);

        for _ in 0..75 {
            acc.accumulate_frame(&constant_bands);
        }

        let crystal = acc.crystallize(5);
        // Constant signal should crystallize most bands
        let alpha_count = crystal.alpha.iter().filter(|&&a| a).count();
        assert!(alpha_count > BARK_BANDS / 2,
            "Constant signal should crystallize most bands, got {}/{}", alpha_count, BARK_BANDS);
    }

    #[test]
    fn test_alpha_density_increases_with_repetition() {
        let constant_bands: [u16; BARK_BANDS] = core::array::from_fn(|i| 0x4000 + i as u16);
        let threshold = 10i16;

        let mut densities = Vec::new();
        for n_frames in [10, 25, 50, 75, 100] {
            let mut acc = SpectralAccumulator::new();
            for _ in 0..n_frames {
                acc.accumulate_frame(&constant_bands);
            }
            densities.push(acc.alpha_density(threshold));
        }

        // Alpha density should increase with more frames (more evidence)
        for i in 1..densities.len() {
            assert!(densities[i] >= densities[i - 1],
                "Alpha density should increase: {} < {} at frames {}",
                densities[i], densities[i - 1], [10, 25, 50, 75, 100][i]);
        }
    }

    #[test]
    fn test_diamond_markov_invariant() {
        let mut acc = SpectralAccumulator::new();
        let signal: [u16; BARK_BANDS] = core::array::from_fn(|i| 0x3F00 + (i * 7) as u16);

        // Accumulate
        for _ in 0..75 {
            acc.accumulate_frame(&signal);
        }

        // Crystallize
        let component = acc.crystallize(5);

        // Unbind
        acc.unbind(&component);

        // Residual should have lower energy than original accumulator
        let residual_energy: i64 = acc.cells.iter().map(|&c| c.abs() as i64).sum();
        let original_energy: i64 = 75 * CELL_COUNT as i64; // upper bound

        assert!(residual_energy < original_energy,
            "Residual after unbind should have lower energy: {} vs {}",
            residual_energy, original_energy);
    }

    #[test]
    fn test_page_curve() {
        let threshold = 10i16;
        let constant_bands: [u16; BARK_BANDS] = core::array::from_fn(|i| 0x3F80 + i as u16);

        println!("\nPAGE CURVE (constant signal)");
        println!("========================");
        println!("{:>10} {:>12} {:>12} {:>10}",
            "encounters", "alpha_before", "alpha_after", "components");

        for n in [5, 10, 20, 30, 50, 75, 100, 150, 200, 300] {
            let mut acc = SpectralAccumulator::new();
            for _ in 0..n {
                acc.accumulate_frame(&constant_bands);
            }
            let alpha_before = acc.alpha_density(threshold);

            // Diamond Markov extraction
            let mut components = 0;
            loop {
                let crystal = acc.crystallize(threshold);
                let significant = crystal.alpha.iter().filter(|&&a| a).count();
                if significant < 3 { break; }
                acc.unbind(&crystal);
                components += 1;
                if components > 20 { break; }
            }
            let alpha_after = acc.alpha_density(threshold);

            println!("{:>10} {:>12.4} {:>12.4} {:>10}",
                n, alpha_before, alpha_after, components);
        }
    }

    #[test]
    fn test_page_curve_structured() {
        let threshold = 10i16;
        let patterns: [[u16; BARK_BANDS]; 5] = core::array::from_fn(|p| {
            core::array::from_fn(|i| 0x3F00 + (p * 100 + i * 7) as u16)
        });

        println!("\nPAGE CURVE (structured: 5 cycling patterns)");
        println!("========================");
        println!("{:>10} {:>12} {:>12} {:>10}",
            "encounters", "alpha_before", "alpha_after", "components");

        for n in [5, 10, 20, 30, 50, 75, 100, 150, 200, 300] {
            let mut acc = SpectralAccumulator::new();
            for frame in 0..n {
                acc.accumulate_frame(&patterns[frame % 5]);
            }
            let alpha_before = acc.alpha_density(threshold);

            let mut components = 0;
            loop {
                let crystal = acc.crystallize(threshold);
                let significant = crystal.alpha.iter().filter(|&&a| a).count();
                if significant < 3 { break; }
                acc.unbind(&crystal);
                components += 1;
                if components > 20 { break; }
            }
            let alpha_after = acc.alpha_density(threshold);

            println!("{:>10} {:>12.4} {:>12.4} {:>10}",
                n, alpha_before, alpha_after, components);
        }
    }

    #[test]
    fn test_page_curve_random() {
        let threshold = 10i16;

        println!("\nPAGE CURVE (random signal)");
        println!("========================");
        println!("{:>10} {:>12} {:>12} {:>10}",
            "encounters", "alpha_before", "alpha_after", "components");

        let mut rng = 12345u64;

        for n in [5, 10, 20, 30, 50, 75, 100, 150, 200, 300] {
            let mut acc = SpectralAccumulator::new();
            rng = 12345u64; // reset per experiment
            for _ in 0..n {
                let bands: [u16; BARK_BANDS] = core::array::from_fn(|i| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (rng >> (16 + i)) as u16
                });
                acc.accumulate_frame(&bands);
            }
            let alpha_before = acc.alpha_density(threshold);

            let mut components = 0;
            loop {
                let crystal = acc.crystallize(threshold);
                let significant = crystal.alpha.iter().filter(|&&a| a).count();
                if significant < 3 { break; }
                acc.unbind(&crystal);
                components += 1;
                if components > 20 { break; }
            }
            let alpha_after = acc.alpha_density(threshold);

            println!("{:>10} {:>12.4} {:>12.4} {:>10}",
                n, alpha_before, alpha_after, components);
        }
    }

    #[test]
    fn test_noise_floor_exists() {
        let mut acc = SpectralAccumulator::new();
        // Mix of signal and noise: alternating bands
        for frame in 0..75u16 {
            let bands: [u16; BARK_BANDS] = core::array::from_fn(|i| {
                if i % 2 == 0 {
                    0x3F80 // constant (will crystallize)
                } else {
                    0x3F80 ^ (frame.wrapping_mul(7 + i as u16)) // varying (noise floor)
                }
            });
            acc.accumulate_frame(&bands);
        }

        let noise = acc.noise_floor(10);
        assert!(!noise.is_empty(), "Should have some cells in the noise floor");

        // Noise floor should be concentrated in odd bands (the varying ones)
        let odd_band_noise: usize = noise.iter()
            .filter(|&&(idx, _)| (idx / 16) % 2 == 1)
            .count();
        let even_band_noise: usize = noise.iter()
            .filter(|&&(idx, _)| (idx / 16) % 2 == 0)
            .count();

        assert!(odd_band_noise > even_band_noise,
            "Varying bands should have more noise floor cells: odd={} even={}",
            odd_band_noise, even_band_noise);
    }
}
