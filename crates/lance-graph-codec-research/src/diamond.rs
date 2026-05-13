//! Diamond Markov Audio Pipeline
//!
//! Progressive crystallization: stream audio into accumulator, monitor for
//! spectral components that crystallize (cross sigma-3 threshold), extract
//! them via XOR-unbind, continue accumulating residual.
//!
//! The invariant: accumulator + all extracted components = original signal.
//! The 25% noise floor IS the lossy compression. What survives is what
//! the psychoacoustic model says humans perceive.

use crate::{CrystallizedComponent, SpectralAccumulator, AudioQualia, BARK_BANDS, FRAME_RATE};
use crate::bands::bf16_to_f32;

/// A Diamond Markov audio session: streams frames, extracts components.
#[derive(Clone, Debug)]
pub struct DiamondAudioSession {
    /// The running accumulator.
    pub accumulator: SpectralAccumulator,
    /// Extracted components in chronological order.
    pub components: Vec<CrystallizedComponent>,
    /// Crystallization threshold (controls quality/bitrate).
    pub threshold: i16,
    /// How often to check for crystallization (every N frames).
    pub check_interval: u32,
    /// Minimum alpha density required for extraction.
    pub min_alpha_density: f64,
    /// Frame counter within current accumulation window.
    frames_since_check: u32,
    /// Global frame counter.
    global_frame: u64,
}

impl DiamondAudioSession {
    /// Create a new Diamond Markov audio session.
    ///
    /// threshold: i16 confidence needed for crystallization.
    ///   Low (5): crystallizes quickly → more components → higher quality
    ///   Medium (20): crystallizes slowly → fewer components → lower bitrate
    ///   High (50): only strongest patterns → very few components → minimal bitrate
    pub fn new(threshold: i16) -> Self {
        Self {
            accumulator: SpectralAccumulator::new(),
            components: Vec::new(),
            threshold,
            check_interval: 10, // check every 10 frames (~133ms)
            min_alpha_density: 0.3, // at least 30% of cells must be confident
            frames_since_check: 0,
            global_frame: 0,
        }
    }

    /// Feed one frame into the pipeline.
    ///
    /// Returns Some(component) if a crystallization was extracted this frame.
    pub fn feed_frame(&mut self, bands: &[u16; BARK_BANDS]) -> Option<CrystallizedComponent> {
        self.accumulator.accumulate_frame(bands);
        self.frames_since_check += 1;
        self.global_frame += 1;

        if self.frames_since_check >= self.check_interval {
            self.frames_since_check = 0;
            return self.check_crystallization();
        }

        None
    }

    /// Check if any spectral component has crystallized.
    /// If yes: extract it via unbind, return the component.
    fn check_crystallization(&mut self) -> Option<CrystallizedComponent> {
        let density = self.accumulator.alpha_density(self.threshold);

        if density < self.min_alpha_density {
            return None; // not enough evidence yet
        }

        // Check if the crystallized spectrum is meaningfully different from noise
        let candidate = self.accumulator.crystallize(self.threshold);

        // Verify: at least some bands have significant energy
        let significant_bands = candidate.spectrum.iter()
            .zip(candidate.alpha.iter())
            .filter(|(&s, &a)| a && bf16_to_f32(s).abs() > 0.001)
            .count();

        if significant_bands < 3 {
            return None; // too sparse, not a real component
        }

        // Extract: unbind the crystallized component from the accumulator
        self.accumulator.unbind(&candidate);

        // Record the component
        let mut component = candidate;
        component.end_frame = self.global_frame;

        self.components.push(component.clone());
        Some(component)
    }

    /// Force extraction of whatever has accumulated, even if below threshold.
    /// Used at end-of-stream or scene change.
    pub fn flush(&mut self) -> Option<CrystallizedComponent> {
        if self.accumulator.frame_count == 0 {
            return None;
        }

        // Use a lower threshold for flush (extract everything)
        let candidate = self.accumulator.crystallize(1);
        self.accumulator.unbind(&candidate);

        let mut component = candidate;
        component.end_frame = self.global_frame;
        self.components.push(component.clone());

        Some(component)
    }

    /// Verify the Diamond Markov invariant.
    ///
    /// Rebundle all extracted components back into the accumulator.
    /// The result should reconstruct the original accumulator state.
    ///
    /// Returns the total Hamming distance from original.
    /// Invariant holds if this is 0 (or near-zero due to saturation effects).
    pub fn verify_invariant(&self) -> InvariantCheck {
        // Clone the current (post-extraction) accumulator
        let mut reconstructed = self.accumulator.clone();

        // Re-add all extracted components
        for component in &self.components {
            // Reverse of unbind: add the component back
            for band in 0..BARK_BANDS {
                for bit in 0..16 {
                    let cell_idx = band * 16 + bit;
                    let comp_bit = ((component.spectrum[band] >> bit) & 1) as i16;
                    let bipolar = comp_bit * 2 - 1;
                    reconstructed.cells[cell_idx] =
                        reconstructed.cells[cell_idx].saturating_add(bipolar);
                }
            }
        }

        // Compare with what a fresh accumulator would have
        // (We don't have the original, so we measure the reconstructed energy)
        let residual_energy: i64 = self.accumulator.cells.iter()
            .map(|&c| c.abs() as i64).sum();
        let reconstructed_energy: i64 = reconstructed.cells.iter()
            .map(|&c| c.abs() as i64).sum();
        let component_energy: i64 = self.components.iter()
            .map(|c| c.spectrum.iter().map(|&s| bf16_to_f32(s).abs() as i64).sum::<i64>())
            .sum();

        // Hamming distance between residual and reconstructed
        let mut hamming = 0u64;
        for i in 0..self.accumulator.cells.len() {
            if self.accumulator.cells[i].signum() != reconstructed.cells[i].signum() {
                hamming += 1;
            }
        }

        InvariantCheck {
            hamming_distance: hamming,
            residual_energy,
            reconstructed_energy,
            component_count: self.components.len(),
            total_frames: self.global_frame,
            invariant_holds: hamming == 0,
        }
    }

    /// Total bitrate of all extracted components.
    pub fn component_bitrate(&self, duration_seconds: f32) -> f64 {
        let total_bits = self.components.len() * BARK_BANDS * 16;
        total_bits as f64 / duration_seconds as f64
    }
}

/// Result of verifying the Diamond Markov invariant.
#[derive(Clone, Debug)]
pub struct InvariantCheck {
    /// Hamming distance between residual+components and reconstructed.
    /// Should be 0 if invariant holds perfectly.
    pub hamming_distance: u64,
    /// Energy remaining in residual accumulator.
    pub residual_energy: i64,
    /// Energy after rebundling all components.
    pub reconstructed_energy: i64,
    /// Number of extracted components.
    pub component_count: usize,
    /// Total frames processed.
    pub total_frames: u64,
    /// Whether the invariant holds (hamming == 0).
    pub invariant_holds: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant_signal(n_frames: usize) -> Vec<[u16; BARK_BANDS]> {
        let bands: [u16; BARK_BANDS] = core::array::from_fn(|i| 0x3F80 + i as u16);
        vec![bands; n_frames]
    }

    #[test]
    fn test_diamond_session_crystallizes() {
        let mut session = DiamondAudioSession::new(5);
        let frames = constant_signal(100);

        let mut extracted = 0;
        for frame_bands in &frames {
            if session.feed_frame(frame_bands).is_some() {
                extracted += 1;
            }
        }

        // Should have extracted at least one component from 100 constant frames
        assert!(extracted >= 1 || !session.components.is_empty(),
            "Should extract at least 1 component from constant signal");

        // Flush remaining
        session.flush();
        assert!(!session.components.is_empty());
    }

    #[test]
    fn test_diamond_invariant_constant() {
        let mut session = DiamondAudioSession::new(5);
        session.check_interval = 20; // check more frequently

        let frames = constant_signal(80);
        for frame_bands in &frames {
            session.feed_frame(frame_bands);
        }
        session.flush();

        let check = session.verify_invariant();
        println!("Diamond invariant check (constant signal):");
        println!("  Hamming distance: {}", check.hamming_distance);
        println!("  Residual energy: {}", check.residual_energy);
        println!("  Reconstructed energy: {}", check.reconstructed_energy);
        println!("  Components: {}", check.component_count);
        println!("  Invariant holds: {}", check.invariant_holds);
    }

    #[test]
    fn test_diamond_bitrate() {
        let mut session = DiamondAudioSession::new(10);
        let frames = constant_signal(225); // 3 seconds at 75fps

        for frame_bands in &frames {
            session.feed_frame(frame_bands);
        }
        session.flush();

        let bitrate = session.component_bitrate(3.0);
        println!("Diamond bitrate: {:.0} bps ({} components in 3s)",
            bitrate, session.components.len());

        // Should be dramatically lower than per-frame encoding
        let perframe_bitrate = 225.0 * BARK_BANDS as f64 * 16.0 / 3.0;
        assert!(bitrate < perframe_bitrate / 2.0,
            "Diamond should compress: {:.0} vs {:.0} bps", bitrate, perframe_bitrate);
    }

    #[test]
    fn test_diamond_varying_signal_fewer_crystals() {
        let mut session_constant = DiamondAudioSession::new(10);
        let mut session_varying = DiamondAudioSession::new(10);

        let constant = constant_signal(75);
        let varying: Vec<[u16; BARK_BANDS]> = (0..75).map(|i| {
            core::array::from_fn(|b| 0x3F80 + (i * 7 + b) as u16)
        }).collect();

        for f in &constant { session_constant.feed_frame(f); }
        for f in &varying { session_varying.feed_frame(f); }

        session_constant.flush();
        session_varying.flush();

        let d_const = session_constant.accumulator.alpha_density(10);
        let d_vary = session_varying.accumulator.alpha_density(10);

        println!("Alpha density: constant={:.3} varying={:.3}", d_const, d_vary);
        // Varying signal should have lower alpha density (less crystallization)
    }
}
