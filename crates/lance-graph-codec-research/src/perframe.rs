//! Strategy A: Per-Frame MDCT Encoding
//!
//! Classical codec path: each frame independently transformed and quantized.
//! 75 nodes/second, 48 bytes/node = 3600 bytes/s = 28.8 kbps.
//! Quality: transparent. Latency: 13ms. No temporal integration.

use crate::{AudioFrame, BARK_BANDS, SAMPLES_PER_FRAME};
use crate::transform::{mdct, imdct, coeffs_to_band_energies, band_energies_to_coeffs,
                        psychoacoustic_mask, sine_window};
use crate::bands::{pack_bands, unpack_bands, f32_to_bf16};

/// Encode PCM samples into per-frame AudioFrame sequence.
pub fn encode_perframe(samples: &[f32]) -> Vec<AudioFrame> {
    let window = sine_window(SAMPLES_PER_FRAME * 2);
    let hop = SAMPLES_PER_FRAME;
    let mut frames = Vec::new();
    let mut prev_bands = [0u16; BARK_BANDS];
    let mut prev_energies = [0.0f32; BARK_BANDS];

    let n_frames = samples.len().saturating_sub(SAMPLES_PER_FRAME * 2) / hop;

    for idx in 0..n_frames {
        let offset = idx * hop;
        let chunk = &samples[offset..offset + SAMPLES_PER_FRAME * 2];

        // Window
        let mut windowed = vec![0.0f32; chunk.len()];
        for i in 0..chunk.len() {
            windowed[i] = chunk[i] * window[i];
        }

        // MDCT
        let mut coeffs = vec![0.0f32; SAMPLES_PER_FRAME];
        mdct(&windowed, &mut coeffs);

        // Band energies
        let energies = coeffs_to_band_energies(&coeffs);
        let packed = pack_bands(&energies);

        // Temporal delta (energy change from previous frame)
        let mut temporal = [0.0f32; BARK_BANDS];
        for b in 0..BARK_BANDS {
            temporal[b] = energies[b] - prev_energies[b];
        }
        let temporal_packed = pack_bands(&temporal);

        // Harmonic ratios (each band vs fundamental)
        let fundamental = energies[0].max(1e-10);
        let mut harmonic = [0.0f32; BARK_BANDS];
        for b in 0..BARK_BANDS {
            harmonic[b] = energies[b] / fundamental;
        }
        let harmonic_packed = pack_bands(&harmonic);

        // Masking
        let mask_f32 = psychoacoustic_mask(&energies);
        let mask_packed: [u16; BARK_BANDS] = core::array::from_fn(|i| f32_to_bf16(mask_f32[i]));

        frames.push(AudioFrame {
            idx: idx as u64,
            bands: packed,
            temporal: temporal_packed,
            harmonic: harmonic_packed,
            mask: mask_packed,
        });

        prev_bands = packed;
        prev_energies = energies;
    }

    frames
}

/// Decode per-frame AudioFrames back to PCM samples.
pub fn decode_perframe(frames: &[AudioFrame]) -> Vec<f32> {
    let window = sine_window(SAMPLES_PER_FRAME * 2);
    let hop = SAMPLES_PER_FRAME;
    let total_samples = (frames.len() + 1) * hop;
    let mut output = vec![0.0f32; total_samples];

    for frame in frames {
        let energies = unpack_bands(&frame.bands);
        let coeffs = band_energies_to_coeffs(&energies, SAMPLES_PER_FRAME);

        let mut time_domain = vec![0.0f32; SAMPLES_PER_FRAME * 2];
        imdct(&coeffs, &mut time_domain);

        let offset = frame.idx as usize * hop;
        for i in 0..time_domain.len() {
            if offset + i < output.len() {
                output[offset + i] += time_domain[i] * window[i];
            }
        }
    }

    output
}

/// Compute bitrate for Strategy A.
pub fn bitrate_perframe(duration_seconds: f32, frame_count: usize) -> f64 {
    let total_bits = frame_count * BARK_BANDS * 16; // 24 bands × 16 bits BF16
    total_bits as f64 / duration_seconds as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_silence() {
        let samples = vec![0.0f32; SAMPLES_PER_FRAME * 10];
        let frames = encode_perframe(&samples);
        assert!(frames.len() >= 7, "Should produce multiple frames");

        // All bands should be near zero for silence
        for frame in &frames {
            for &band in &frame.bands {
                let energy = crate::bands::bf16_to_f32(band);
                assert!(energy.abs() < 0.01, "Silent frame should have near-zero bands");
            }
        }
    }

    #[test]
    fn test_encode_sine_wave() {
        // 440Hz sine wave
        let n = SAMPLES_PER_FRAME * 20;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin() * 0.5)
            .collect();

        let frames = encode_perframe(&samples);
        assert!(!frames.is_empty());

        // Band containing 440Hz (Bark band ~4-5) should have energy
        // 440Hz falls in band 4 (400-510 Hz)
        let frame = &frames[5]; // skip first few frames (window ramp-up)
        let energies = unpack_bands(&frame.bands);
        let max_band = energies.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert!(max_band >= 3 && max_band <= 6,
            "440Hz should be in band 3-6, got band {}", max_band);
    }
}
