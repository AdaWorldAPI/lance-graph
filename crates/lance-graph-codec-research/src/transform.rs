//! MDCT transform and critical band mapping.
//!
//! The MDCT (Modified Discrete Cosine Transform) is the shared transform
//! used by MP3, AAC, Opus CELT, and Vorbis. We use it as the common
//! frequency-domain representation for all three encoding strategies.

use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

use crate::{BARK_BANDS, SAMPLES_PER_FRAME, SAMPLE_RATE};

/// Bark-scale critical band edges in Hz for 24 bands.
/// These approximate the frequency resolution of the human cochlea.
/// Bands are narrower at low frequencies (high resolution where pitch matters)
/// and wider at high frequencies (low resolution where detail is masked).
pub fn bark_band_edges() -> [f32; BARK_BANDS + 1] {
    // Zwicker critical band boundaries (Hz), adapted for 48kHz
    [
        20.0, 100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0,
        1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0,
        5300.0, 6400.0, 7700.0, 9500.0, 12000.0, 24000.0,
    ]
}

/// Map frequency in Hz to MDCT bin index.
#[inline]
pub fn freq_to_bin(freq_hz: f32, n_bins: usize) -> usize {
    ((freq_hz / SAMPLE_RATE as f32) * 2.0 * n_bins as f32) as usize
}

/// Sine window for MDCT overlap-add. Length = 2N where N = MDCT size.
pub fn sine_window(len: usize) -> Vec<f32> {
    (0..len)
        .map(|n| (PI * (n as f32 + 0.5) / len as f32).sin())
        .collect()
}

/// Forward MDCT: 2N time-domain samples → N frequency coefficients.
///
/// Implementation via N/2-point FFT with pre/post twiddle factors.
/// This is the standard fast MDCT used in all modern audio codecs.
pub fn mdct(input: &[f32], output: &mut [f32]) {
    let n2 = output.len(); // N/2
    let n = n2 * 2; // N
    let n4 = n2 / 2; // N/4
    assert_eq!(input.len(), n * 2, "MDCT input must be 2N samples");

    // Pre-twiddle: fold and rotate into N/2 complex values
    let mut pre = vec![Complex::new(0.0f32, 0.0f32); n2];
    for k in 0..n2 {
        let cos_tw = (PI / n as f32 * (2.0 * k as f32 + 1.0 + n2 as f32) * 0.5).cos();
        let sin_tw = (PI / n as f32 * (2.0 * k as f32 + 1.0 + n2 as f32) * 0.5).sin();

        // Fold: combine symmetric halves
        let idx0 = (n4 + k) % (n * 2);
        let idx1 = (n + n4 - 1 - k) % (n * 2);
        let idx2 = (n + n4 + k) % (n * 2);
        let idx3 = (2 * n + n4 - 1 - k) % (n * 2);

        let s0 = if idx0 < input.len() { input[idx0] } else { 0.0 };
        let s1 = if idx1 < input.len() { input[idx1] } else { 0.0 };
        let s2 = if idx2 < input.len() { -input[idx2] } else { 0.0 };
        let s3 = if idx3 < input.len() { -input[idx3] } else { 0.0 };

        let re = s0 + s1 + s2 + s3;
        pre[k] = Complex::new(re * cos_tw, re * sin_tw);
    }

    // FFT of length N/2
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n2);
    fft.process(&mut pre);

    // Post-twiddle: extract real MDCT coefficients
    for k in 0..n2 {
        let cos_tw = (PI / n as f32 * (2.0 * k as f32 + 1.0 + n2 as f32) * 0.5).cos();
        let sin_tw = (PI / n as f32 * (2.0 * k as f32 + 1.0 + n2 as f32) * 0.5).sin();
        output[k] = pre[k].re * cos_tw + pre[k].im * sin_tw;
    }
}

/// Inverse MDCT: N frequency coefficients → 2N time-domain samples.
///
/// The decoder hot path. Must be fast enough for real-time playback.
pub fn imdct(input: &[f32], output: &mut [f32]) {
    let n2 = input.len(); // N/2
    let n = n2 * 2; // N
    assert_eq!(output.len(), n * 2, "iMDCT output must be 2N samples");

    // Pre-twiddle
    let mut pre = vec![Complex::new(0.0f32, 0.0f32); n2];
    for k in 0..n2 {
        let cos_tw = (PI / n as f32 * (2.0 * k as f32 + 1.0 + n2 as f32) * 0.5).cos();
        let sin_tw = (PI / n as f32 * (2.0 * k as f32 + 1.0 + n2 as f32) * 0.5).sin();
        pre[k] = Complex::new(input[k] * cos_tw, input[k] * sin_tw);
    }

    // Inverse FFT of length N/2
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n2);
    fft.process(&mut pre);

    // Post-twiddle and unfold
    let scale = 2.0 / n as f32;
    for k in 0..n2 {
        let cos_tw = (PI / n as f32 * (2.0 * k as f32 + 1.0 + n2 as f32) * 0.5).cos();
        let sin_tw = (PI / n as f32 * (2.0 * k as f32 + 1.0 + n2 as f32) * 0.5).sin();
        let val = (pre[k].re * cos_tw + pre[k].im * sin_tw) * scale;

        // Unfold into 2N output with proper symmetry
        let n4 = n2 / 2;
        let idx = (n4 + k) % (n * 2);
        if idx < output.len() {
            output[idx] += val;
        }
    }
}

/// Compute band energies from MDCT coefficients.
/// Groups coefficients into 24 Bark-scale critical bands
/// and computes RMS energy per band.
pub fn coeffs_to_band_energies(coeffs: &[f32]) -> [f32; BARK_BANDS] {
    let edges = bark_band_edges();
    let n_bins = coeffs.len();
    let mut energies = [0.0f32; BARK_BANDS];

    for band in 0..BARK_BANDS {
        let lo = freq_to_bin(edges[band], n_bins);
        let hi = freq_to_bin(edges[band + 1], n_bins).min(n_bins);
        let bin_count = (hi - lo).max(1);

        let mut sum_sq = 0.0f32;
        for bin in lo..hi {
            sum_sq += coeffs[bin] * coeffs[bin];
        }
        energies[band] = (sum_sq / bin_count as f32).sqrt();
    }

    energies
}

/// Reconstruct MDCT coefficients from band energies.
/// Inverse of coeffs_to_band_energies: distributes each band's energy
/// uniformly across its frequency bins.
pub fn band_energies_to_coeffs(energies: &[f32; BARK_BANDS], n_bins: usize) -> Vec<f32> {
    let edges = bark_band_edges();
    let mut coeffs = vec![0.0f32; n_bins];

    for band in 0..BARK_BANDS {
        let lo = freq_to_bin(edges[band], n_bins);
        let hi = freq_to_bin(edges[band + 1], n_bins).min(n_bins);
        let bin_count = (hi - lo).max(1);
        let per_bin = energies[band] / (bin_count as f32).sqrt();
        for bin in lo..hi {
            coeffs[bin] = per_bin;
        }
    }

    coeffs
}

/// Simple psychoacoustic masking model (ISO 11172-3 model 2, simplified).
///
/// For each band: if a neighboring band is louder by > 12dB, this band
/// is masked. The threshold is the maximum of absolute threshold of hearing
/// and the spreading function from neighboring bands.
pub fn psychoacoustic_mask(energies: &[f32; BARK_BANDS]) -> [f32; BARK_BANDS] {
    let mut thresholds = [0.0f32; BARK_BANDS];

    // Absolute threshold of hearing (approximate, in energy units)
    let ath: [f32; BARK_BANDS] = [
        0.01, 0.005, 0.003, 0.002, 0.002, 0.002, 0.002, 0.002,
        0.003, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.013,
        0.016, 0.020, 0.025, 0.032, 0.040, 0.050, 0.063, 0.080,
    ];

    for band in 0..BARK_BANDS {
        let mut mask = ath[band];

        // Spreading function: loud neighbors mask this band
        for other in 0..BARK_BANDS {
            if other == band {
                continue;
            }
            let distance = (band as i32 - other as i32).unsigned_abs() as f32;
            // Spreading: -27 dB per Bark on the high side, -12 dB per Bark on the low side
            let spread = if other < band {
                // Lower frequency masking upper: gentler slope
                energies[other] * 10.0f32.powf(-1.2 * distance)
            } else {
                // Upper frequency masking lower: steeper slope
                energies[other] * 10.0f32.powf(-2.7 * distance)
            };
            mask = mask.max(spread);
        }

        thresholds[band] = mask;
    }

    thresholds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bark_band_edges_monotonic() {
        let edges = bark_band_edges();
        for i in 0..BARK_BANDS {
            assert!(edges[i + 1] > edges[i], "Band edges must be monotonically increasing");
        }
        assert_eq!(edges[0], 20.0);
        assert_eq!(edges[BARK_BANDS], 24000.0);
    }

    #[test]
    fn test_sine_window_symmetry() {
        let w = sine_window(128);
        for i in 0..64 {
            assert!((w[i] - w[127 - i]).abs() < 1e-6, "Sine window must be symmetric");
        }
    }

    #[test]
    fn test_band_energy_roundtrip() {
        // Create synthetic MDCT coefficients: energy in band 5 only
        let n = SAMPLES_PER_FRAME;
        let mut coeffs = vec![0.0f32; n];
        let edges = bark_band_edges();
        let lo = freq_to_bin(edges[5], n);
        let hi = freq_to_bin(edges[6], n);
        for bin in lo..hi.min(n) {
            coeffs[bin] = 1.0;
        }

        let energies = coeffs_to_band_energies(&coeffs);
        assert!(energies[5] > 0.5, "Band 5 should have significant energy");
        for band in [0, 1, 2, 3, 4, 10, 15, 20] {
            assert!(energies[band] < 0.01, "Band {} should be near zero", band);
        }

        // Roundtrip
        let reconstructed = band_energies_to_coeffs(&energies, n);
        let re_energies = coeffs_to_band_energies(&reconstructed);
        for band in 0..BARK_BANDS {
            assert!(
                (energies[band] - re_energies[band]).abs() < 0.01,
                "Band {} energy changed in roundtrip: {} → {}",
                band, energies[band], re_energies[band]
            );
        }
    }

    #[test]
    fn test_psychoacoustic_mask_loud_masks_quiet() {
        let mut energies = [0.001f32; BARK_BANDS];
        energies[10] = 10.0; // loud band at 10
        let mask = psychoacoustic_mask(&energies);

        // Bands near 10 should have elevated thresholds
        assert!(mask[9] > mask[0], "Band 9 should be more masked than band 0");
        assert!(mask[11] > mask[0], "Band 11 should be more masked than band 0");
        // Distant bands should be barely affected
        assert!(mask[0] < 0.1, "Band 0 should not be heavily masked by band 10");
    }
}
