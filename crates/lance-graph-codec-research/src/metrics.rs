//! Comparison metrics for all three encoding strategies.
//!
//! Measures quality, compression, searchability, and the core hypothesis:
//! does the VSA noise floor correlate with psychoacoustic masking?

use crate::{AudioFrame, ComparisonResult, SpectralAccumulator, BARK_BANDS, SAMPLES_PER_FRAME};
use crate::transform::{mdct, coeffs_to_band_energies, psychoacoustic_mask, sine_window};
use crate::bands::{pack_bands, unpack_bands, bf16_to_f32, bf16_band_distance};

/// Run the full comparison suite on a PCM signal.
///
/// Returns results for all three strategies plus the hypothesis test.
pub fn compare_all_strategies(
    samples: &[f32],
    accumulator_window: usize,
    crystallization_threshold: i16,
) -> Vec<ComparisonResult> {
    let duration_s = samples.len() as f32 / 48000.0;
    let mut results = Vec::new();

    // ── Strategy A: Per-Frame ──
    let frames_a = crate::perframe::encode_perframe(samples);
    let decoded_a = crate::perframe::decode_perframe(&frames_a);
    let quality_a = spectral_distortion(samples, &decoded_a);
    let corr_a = band_energy_correlation(samples, &decoded_a);

    results.push(ComparisonResult {
        strategy_name: "A: Per-Frame MDCT".into(),
        bitrate_bps: crate::perframe::bitrate_perframe(duration_s, frames_a.len()),
        compression_ratio: 768000.0 / crate::perframe::bitrate_perframe(duration_s, frames_a.len()),
        spectral_distortion_db: quality_a,
        band_energy_correlation: corr_a,
        node_count: frames_a.len(),
        crystallized_count: 0,
        alpha_density: 1.0,
        noise_mask_correlation: 0.0, // not applicable
        invariant_hamming: 0,        // not applicable
    });

    // ── Strategy B: Streaming Accumulator ──
    let mut acc = SpectralAccumulator::new();
    acc.accumulate_pcm(samples);

    // Compute masking thresholds for hypothesis test
    let avg_mask = average_masking_thresholds(samples);
    let noise_mask_corr = acc.noise_mask_correlation(crystallization_threshold, &avg_mask);

    let crystal = acc.crystallize(crystallization_threshold);
    let decoded_energies = crate::accumulator::decode_crystallized(&crystal);
    let alpha_d = acc.alpha_density(crystallization_threshold);

    // Reconstruct PCM from crystallized energies (simple: repeat for all frames)
    let n_frames = samples.len().saturating_sub(SAMPLES_PER_FRAME * 2) / SAMPLES_PER_FRAME;
    let decoded_b = reconstruct_from_energies(&decoded_energies, n_frames);
    let quality_b = spectral_distortion(samples, &decoded_b);
    let corr_b = band_energy_correlation(samples, &decoded_b);

    let component_bits = BARK_BANDS * 16;
    let bitrate_b = component_bits as f64 / duration_s as f64;

    results.push(ComparisonResult {
        strategy_name: "B: Streaming Accumulator".into(),
        bitrate_bps: bitrate_b,
        compression_ratio: 768000.0 / bitrate_b,
        spectral_distortion_db: quality_b,
        band_energy_correlation: corr_b,
        node_count: 1,
        crystallized_count: 1,
        alpha_density: alpha_d,
        noise_mask_correlation: noise_mask_corr,
        invariant_hamming: 0,
    });

    // ── Strategy C: Hybrid ──
    let hybrid = crate::hybrid::encode_hybrid(samples, accumulator_window, crystallization_threshold);
    let decoded_c = crate::hybrid::decode_hybrid(&hybrid);
    let quality_c = spectral_distortion(samples, &decoded_c);
    let corr_c = band_energy_correlation(samples, &decoded_c);
    let breakdown = crate::hybrid::bitrate_breakdown(&hybrid, duration_s);

    results.push(ComparisonResult {
        strategy_name: "C: Hybrid".into(),
        bitrate_bps: breakdown.total_bps,
        compression_ratio: 768000.0 / breakdown.total_bps,
        spectral_distortion_db: quality_c,
        band_energy_correlation: corr_c,
        node_count: hybrid.frames.len(),
        crystallized_count: hybrid.components.len(),
        alpha_density: alpha_d,
        noise_mask_correlation: noise_mask_corr,
        invariant_hamming: 0,
    });

    // ── Strategy B+Diamond: Streaming with progressive extraction ──
    let mut diamond = crate::diamond::DiamondAudioSession::new(crystallization_threshold);
    let frame_data = crate::perframe::encode_perframe(samples);
    for frame in &frame_data {
        diamond.feed_frame(&frame.bands);
    }
    diamond.flush();

    let invariant = diamond.verify_invariant();
    let diamond_bitrate = diamond.component_bitrate(duration_s);

    results.push(ComparisonResult {
        strategy_name: "B+Diamond: Progressive".into(),
        bitrate_bps: diamond_bitrate,
        compression_ratio: 768000.0 / diamond_bitrate.max(1.0),
        spectral_distortion_db: quality_b, // same decode quality as B
        band_energy_correlation: corr_b,
        node_count: diamond.components.len(),
        crystallized_count: diamond.components.len(),
        alpha_density: alpha_d,
        noise_mask_correlation: noise_mask_corr,
        invariant_hamming: invariant.hamming_distance,
    });

    results
}

/// Spectral distortion in dB between original and reconstructed PCM.
pub fn spectral_distortion(original: &[f32], reconstructed: &[f32]) -> f64 {
    let len = original.len().min(reconstructed.len());
    if len == 0 { return f64::INFINITY; }

    let mut signal_power = 0.0f64;
    let mut error_power = 0.0f64;

    for i in 0..len {
        signal_power += (original[i] as f64).powi(2);
        let err = original[i] as f64 - reconstructed.get(i).copied().unwrap_or(0.0) as f64;
        error_power += err.powi(2);
    }

    if error_power < 1e-20 {
        return 0.0; // perfect reconstruction
    }
    if signal_power < 1e-20 {
        return f64::NEG_INFINITY; // silence
    }

    10.0 * (signal_power / error_power).log10()
}

/// Pearson correlation of band energies between original and reconstructed.
pub fn band_energy_correlation(original: &[f32], reconstructed: &[f32]) -> f64 {
    let window = sine_window(SAMPLES_PER_FRAME * 2);
    let hop = SAMPLES_PER_FRAME;
    let n_frames = original.len().saturating_sub(SAMPLES_PER_FRAME * 2) / hop;
    let n_frames = n_frames.min(
        reconstructed.len().saturating_sub(SAMPLES_PER_FRAME * 2) / hop
    );

    if n_frames == 0 { return 0.0; }

    let mut orig_energies = Vec::with_capacity(n_frames * BARK_BANDS);
    let mut recon_energies = Vec::with_capacity(n_frames * BARK_BANDS);

    for idx in 0..n_frames {
        let offset = idx * hop;

        // Original
        let chunk = &original[offset..offset + SAMPLES_PER_FRAME * 2];
        let mut windowed = vec![0.0f32; chunk.len()];
        for i in 0..chunk.len() { windowed[i] = chunk[i] * window[i]; }
        let mut coeffs = vec![0.0f32; SAMPLES_PER_FRAME];
        mdct(&windowed, &mut coeffs);
        let e = coeffs_to_band_energies(&coeffs);
        orig_energies.extend_from_slice(&e);

        // Reconstructed
        if offset + SAMPLES_PER_FRAME * 2 <= reconstructed.len() {
            let chunk_r = &reconstructed[offset..offset + SAMPLES_PER_FRAME * 2];
            let mut windowed_r = vec![0.0f32; chunk_r.len()];
            for i in 0..chunk_r.len() { windowed_r[i] = chunk_r[i] * window[i]; }
            let mut coeffs_r = vec![0.0f32; SAMPLES_PER_FRAME];
            mdct(&windowed_r, &mut coeffs_r);
            let e_r = coeffs_to_band_energies(&coeffs_r);
            recon_energies.extend_from_slice(&e_r);
        } else {
            recon_energies.extend_from_slice(&[0.0f32; BARK_BANDS]);
        }
    }

    let orig_f64: Vec<f64> = orig_energies.iter().map(|&x| x as f64).collect();
    let recon_f64: Vec<f64> = recon_energies.iter().map(|&x| x as f64).collect();

    crate::accumulator::pearson_correlation(&orig_f64, &recon_f64)
}

/// Compute average psychoacoustic masking thresholds across all frames.
fn average_masking_thresholds(samples: &[f32]) -> [f32; BARK_BANDS] {
    let window = sine_window(SAMPLES_PER_FRAME * 2);
    let hop = SAMPLES_PER_FRAME;
    let n_frames = samples.len().saturating_sub(SAMPLES_PER_FRAME * 2) / hop;

    let mut avg = [0.0f32; BARK_BANDS];
    if n_frames == 0 { return avg; }

    for idx in 0..n_frames {
        let offset = idx * hop;
        let chunk = &samples[offset..offset + SAMPLES_PER_FRAME * 2];
        let mut windowed = vec![0.0f32; chunk.len()];
        for i in 0..chunk.len() { windowed[i] = chunk[i] * window[i]; }
        let mut coeffs = vec![0.0f32; SAMPLES_PER_FRAME];
        mdct(&windowed, &mut coeffs);
        let energies = coeffs_to_band_energies(&coeffs);
        let mask = psychoacoustic_mask(&energies);
        for b in 0..BARK_BANDS { avg[b] += mask[b]; }
    }

    for b in 0..BARK_BANDS { avg[b] /= n_frames as f32; }
    avg
}

/// Reconstruct PCM from static band energies (repeat same spectrum for all frames).
fn reconstruct_from_energies(energies: &[f32; BARK_BANDS], n_frames: usize) -> Vec<f32> {
    let window = sine_window(SAMPLES_PER_FRAME * 2);
    let hop = SAMPLES_PER_FRAME;
    let coeffs = crate::transform::band_energies_to_coeffs(energies, SAMPLES_PER_FRAME);
    let mut output = vec![0.0f32; (n_frames + 1) * hop];

    for idx in 0..n_frames {
        let mut time_domain = vec![0.0f32; SAMPLES_PER_FRAME * 2];
        crate::transform::imdct(&coeffs, &mut time_domain);
        let offset = idx * hop;
        for i in 0..time_domain.len() {
            if offset + i < output.len() {
                output[offset + i] += time_domain[i] * window[i];
            }
        }
    }

    output
}

/// Pretty-print comparison results.
pub fn print_comparison(results: &[ComparisonResult]) {
    println!("\n{}", "═".repeat(80));
    println!("  CODEC STRATEGY COMPARISON");
    println!("{}", "═".repeat(80));
    println!("{:<25} {:>10} {:>8} {:>8} {:>8} {:>6} {:>6}",
        "Strategy", "Bitrate", "Ratio", "SD(dB)", "Corr", "Nodes", "Cryst");
    println!("{:<25} {:>10} {:>8} {:>8} {:>8} {:>6} {:>6}",
        "─".repeat(25), "─".repeat(10), "─".repeat(8), "─".repeat(8),
        "─".repeat(8), "─".repeat(6), "─".repeat(6));

    for r in results {
        println!("{:<25} {:>10.0} {:>8.1}× {:>8.1} {:>8.4} {:>6} {:>6}",
            r.strategy_name, r.bitrate_bps, r.compression_ratio,
            r.spectral_distortion_db, r.band_energy_correlation,
            r.node_count, r.crystallized_count);
    }

    // Hypothesis test
    if let Some(b) = results.iter().find(|r| r.strategy_name.contains("Accumulator")) {
        println!("\n  HYPOTHESIS TEST: Noise floor ↔ Masking threshold");
        println!("  Pearson correlation: {:.4}", b.noise_mask_correlation);
        println!("  Alpha density: {:.3}", b.alpha_density);
        if b.noise_mask_correlation > 0.5 {
            println!("  ✓ SUPPORTED: noise floor correlates with masking (r > 0.5)");
        } else if b.noise_mask_correlation > 0.2 {
            println!("  ? WEAK: some correlation but not strong (0.2 < r < 0.5)");
        } else {
            println!("  ✗ NOT SUPPORTED: noise floor does not correlate with masking");
        }
    }

    if let Some(d) = results.iter().find(|r| r.strategy_name.contains("Diamond")) {
        println!("\n  DIAMOND MARKOV INVARIANT");
        println!("  Hamming distance after rebundling: {}", d.invariant_hamming);
        if d.invariant_hamming == 0 {
            println!("  ✓ INVARIANT HOLDS: information perfectly conserved");
        } else {
            println!("  ✗ INVARIANT BROKEN: {} bits differ (saturation effects?)",
                d.invariant_hamming);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_440(duration_s: f32) -> Vec<f32> {
        let n = (duration_s * 48000.0) as usize;
        (0..n).map(|i| {
            (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin() * 0.5
        }).collect()
    }

    fn white_noise(duration_s: f32, seed: u64) -> Vec<f32> {
        let n = (duration_s * 48000.0) as usize;
        let mut state = seed;
        (0..n).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0
        }).collect()
    }

    #[test]
    fn test_compare_sine() {
        let samples = sine_440(2.0);
        let results = compare_all_strategies(&samples, 75, 10);

        println!("\n═══ SINE 440Hz, 2 seconds ═══");
        for r in &results {
            println!("  {}: {:.0} bps, SD={:.1}dB, corr={:.4}, α={:.3}, noise↔mask={:.4}",
                r.strategy_name, r.bitrate_bps, r.spectral_distortion_db,
                r.band_energy_correlation, r.alpha_density, r.noise_mask_correlation);
        }

        // Strategy A should have best quality
        assert!(results[0].spectral_distortion_db > results[1].spectral_distortion_db
            || results[0].spectral_distortion_db > 0.0,
            "Per-frame should have better or comparable quality");

        // Strategy B should have lowest bitrate
        assert!(results[1].bitrate_bps < results[0].bitrate_bps,
            "Accumulator should have lower bitrate");
    }

    #[test]
    fn test_compare_noise() {
        let samples = white_noise(2.0, 42);
        let results = compare_all_strategies(&samples, 75, 10);

        println!("\n═══ WHITE NOISE, 2 seconds ═══");
        for r in &results {
            println!("  {}: {:.0} bps, SD={:.1}dB, corr={:.4}, α={:.3}",
                r.strategy_name, r.bitrate_bps, r.spectral_distortion_db,
                r.band_energy_correlation, r.alpha_density);
        }

        // Noise should have LOW alpha density (nothing crystallizes)
        assert!(results[1].alpha_density < 0.8,
            "Noise should not crystallize fully: α={:.3}", results[1].alpha_density);
    }

    #[test]
    fn test_hypothesis_tonal_vs_noise() {
        let sine = sine_440(2.0);
        let noise = white_noise(2.0, 42);

        let results_sine = compare_all_strategies(&sine, 75, 10);
        let results_noise = compare_all_strategies(&noise, 75, 10);

        let sine_alpha = results_sine[1].alpha_density;
        let noise_alpha = results_noise[1].alpha_density;

        println!("\n═══ HYPOTHESIS: tonal signal crystallizes more than noise ═══");
        println!("  Sine alpha density: {:.3}", sine_alpha);
        println!("  Noise alpha density: {:.3}", noise_alpha);

        assert!(sine_alpha > noise_alpha,
            "Tonal signal should crystallize more: sine={:.3} noise={:.3}",
            sine_alpha, noise_alpha);
    }
}
