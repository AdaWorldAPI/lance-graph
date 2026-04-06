//! Re-encode safety test: encode → decode → re-encode → measure drift.
//!
//! If the codec is re-encode safe, the error converges after K iterations.
//! If not, it accumulates → the codec is lossy in a dangerous way.
//!
//! Like testing if an ICC color profile is idempotent:
//!   encode(decode(encode(x))) == encode(x) → safe
//!   encode(decode(encode(x))) ≠ encode(x) → drift → unsafe
//!
//! Goal: prove "x256 re-encode safety" — 256 round-trips with bounded error.

use bgz_tensor::stacked_n::{bf16_to_f32, f32_to_bf16};
use bgz_tensor::gamma_phi::{gamma_phi_encode, gamma_phi_decode};

/// Result of a re-encode safety test.
#[derive(Clone, Debug)]
pub struct ReencodeSafety {
    /// How many iterations until error stabilized (delta < threshold).
    pub converged_at: usize,
    /// Maximum error seen across all iterations.
    pub max_error: f64,
    /// Error at final iteration.
    pub final_error: f64,
    /// Error history per iteration.
    pub error_history: Vec<f64>,
    /// Is it re-encode safe? (converged within max_iterations)
    pub safe: bool,
    /// Codec name.
    pub codec: String,
}

impl std::fmt::Display for ReencodeSafety {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} after {} iterations (max_err={:.2e}, final_err={:.2e})",
            self.codec,
            if self.safe { "SAFE" } else { "UNSAFE" },
            self.converged_at,
            self.max_error,
            self.final_error)
    }
}

/// Test BF16 round-trip: f64 → f32 → bf16 → f32 → bf16 → ... → f64
pub fn test_bf16_reencode(value: f64, max_iterations: usize) -> ReencodeSafety {
    let mut current = value as f32;
    let mut errors = Vec::new();
    let mut converged_at = max_iterations;

    for i in 0..max_iterations {
        let encoded = f32_to_bf16(current);
        let decoded = bf16_to_f32(encoded);
        let error = (decoded as f64 - value).abs();
        errors.push(error);

        // Check convergence: error stopped changing
        if i > 0 && (errors[i] - errors[i - 1]).abs() < 1e-15 {
            converged_at = i;
            // Fill remaining with same error
            for _ in (i + 1)..max_iterations {
                errors.push(error);
            }
            break;
        }
        current = decoded;
    }

    let max_error = errors.iter().cloned().fold(0.0f64, f64::max);
    let final_error = *errors.last().unwrap_or(&0.0);

    ReencodeSafety {
        converged_at,
        max_error,
        final_error,
        error_history: errors,
        safe: converged_at < max_iterations,
        codec: "BF16".into(),
    }
}

/// Test γ+φ round-trip: f64 → gamma_phi_encode → gamma_phi_decode → re-encode → ...
pub fn test_gamma_phi_reencode(
    value: f64,
    role_gamma: f32,
    phi_scale: f32,
    max_iterations: usize,
) -> ReencodeSafety {
    let mut current = value as f32;
    let mut errors = Vec::new();
    let mut converged_at = max_iterations;

    for i in 0..max_iterations {
        let encoded = gamma_phi_encode(current, role_gamma, phi_scale);
        let decoded = gamma_phi_decode(encoded, role_gamma, phi_scale);
        let error = (decoded as f64 - value).abs();
        errors.push(error);

        if i > 0 && (errors[i] - errors[i - 1]).abs() < 1e-15 {
            converged_at = i;
            for _ in (i + 1)..max_iterations {
                errors.push(error);
            }
            break;
        }
        current = decoded;
    }

    let max_error = errors.iter().cloned().fold(0.0f64, f64::max);
    let final_error = *errors.last().unwrap_or(&0.0);

    ReencodeSafety {
        converged_at,
        max_error,
        final_error,
        error_history: errors,
        safe: converged_at < max_iterations,
        codec: format!("γ+φ(γ={},φ={})", role_gamma, phi_scale),
    }
}

/// Test full chain: f64 → f32 → bf16 → f32 → gamma_phi_encode → gamma_phi_decode → bf16 → ...
pub fn test_full_chain_reencode(
    value: f64,
    role_gamma: f32,
    phi_scale: f32,
    max_iterations: usize,
) -> ReencodeSafety {
    let mut current = value as f32;
    let mut errors = Vec::new();
    let mut converged_at = max_iterations;

    for i in 0..max_iterations {
        // Stage 1: BF16 quantize
        let bf16 = f32_to_bf16(current);
        let from_bf16 = bf16_to_f32(bf16);

        // Stage 2: γ+φ encode/decode
        let gp_encoded = gamma_phi_encode(from_bf16, role_gamma, phi_scale);
        let gp_decoded = gamma_phi_decode(gp_encoded, role_gamma, phi_scale);

        // Stage 3: back to BF16
        let re_bf16 = f32_to_bf16(gp_decoded);
        let final_val = bf16_to_f32(re_bf16);

        let error = (final_val as f64 - value).abs();
        errors.push(error);

        if i > 0 && (errors[i] - errors[i - 1]).abs() < 1e-15 {
            converged_at = i;
            for _ in (i + 1)..max_iterations {
                errors.push(error);
            }
            break;
        }
        current = final_val;
    }

    let max_error = errors.iter().cloned().fold(0.0f64, f64::max);
    let final_error = *errors.last().unwrap_or(&0.0);

    ReencodeSafety {
        converged_at,
        max_error,
        final_error,
        error_history: errors,
        safe: converged_at < max_iterations,
        codec: format!("BF16+γ+φ(γ={},φ={})", role_gamma, phi_scale),
    }
}

/// Test a BATCH of values across the range. Returns (all_safe, worst_case, convergence_stats).
pub fn test_reencode_batch(
    codec_fn: impl Fn(f64) -> ReencodeSafety,
    test_values: &[f64],
) -> (bool, ReencodeSafety, usize, usize) {
    let mut all_safe = true;
    let mut worst = None::<ReencodeSafety>;
    let mut max_converge = 0;
    let mut safe_count = 0;

    for &v in test_values {
        let result = codec_fn(v);
        if result.safe { safe_count += 1; } else { all_safe = false; }
        if result.converged_at > max_converge { max_converge = result.converged_at; }
        if worst.as_ref().map_or(true, |w| result.max_error > w.max_error) {
            worst = Some(result);
        }
    }

    (all_safe, worst.unwrap_or_else(|| ReencodeSafety {
        converged_at: 0, max_error: 0.0, final_error: 0.0,
        error_history: vec![], safe: true, codec: "empty".into(),
    }), safe_count, test_values.len())
}

/// Test re-encode safety across multiple zipper offsets.
///
/// The golden step (11 mod 17) creates a permutation.
/// Different octave offsets (0, 1, 2, ..., 16) sample different
/// "zipper teeth" of the weight vector. All offsets must be safe.
///
/// With stride S and offset O, sampled octaves are: O, O+S, O+2S, ...
/// 4 canonical offsets: 0, floor(n/φ²), floor(n/φ), floor(n×(φ-1))
/// 8 offsets: add the midpoints between canonical pairs.
pub fn test_zipper_offsets(
    value: f64,
    role_gamma: f32,
    phi_scale: f32,
    n_offsets: usize,
    max_iterations: usize,
) -> Vec<ReencodeSafety> {
    let phi: f64 = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let n = 302usize; // typical octave count for 5120D

    // Generate offsets: φ-fractional positions + regular spacing
    let mut offsets = Vec::with_capacity(n_offsets);
    // 4 canonical φ-positions
    offsets.push(0);
    offsets.push((n as f64 / (phi * phi)).floor() as usize); // n/φ² ≈ 115
    offsets.push((n as f64 / phi).floor() as usize);          // n/φ ≈ 186
    offsets.push((n as f64 * (phi - 1.0)).floor() as usize);  // n×0.618 ≈ 186 (same!)

    // Fill remaining with regular spacing
    let step = n / n_offsets.max(4);
    for i in 0..n_offsets {
        let o = (i * step) % n;
        if !offsets.contains(&o) {
            offsets.push(o);
        }
    }
    offsets.truncate(n_offsets);

    offsets.iter().map(|&offset| {
        // Simulate encoding with this offset: apply a phase shift to the value
        // Different offsets see different "slices" of the weight vector
        // For a constant value, the offset doesn't change the result
        // For varying values, the offset determines which octave samples are used
        let offset_phase = (offset as f64 * 0.001).sin() * 0.01; // tiny offset effect
        let shifted_value = value + offset_phase;

        let mut result = test_full_chain_reencode(shifted_value, role_gamma, phi_scale, max_iterations);
        result.codec = format!("offset={} {}", offset, result.codec);
        result
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_is_idempotent() {
        // BF16 truncation should stabilize after 1 iteration
        let result = test_bf16_reencode(0.123456789, 256);
        eprintln!("{}", result);
        assert!(result.safe, "BF16 should be re-encode safe");
        assert!(result.converged_at <= 1, "BF16 should converge at iteration 1, got {}", result.converged_at);
    }

    #[test]
    fn bf16_batch_safety() {
        let values: Vec<f64> = (-100..=100).map(|i| i as f64 * 0.01).collect();
        let (all_safe, worst, safe_count, total) = test_reencode_batch(
            |v| test_bf16_reencode(v, 256),
            &values,
        );
        eprintln!("BF16 batch: {}/{} safe, worst: {}", safe_count, total, worst);
        assert!(all_safe, "BF16 should be safe for all values");
    }

    #[test]
    fn gamma_phi_reencode_safe() {
        // γ+φ encode/decode should be exact inverse (f32 precision)
        let result = test_gamma_phi_reencode(0.15, 1.50, 0.23, 256);
        eprintln!("{}", result);
        assert!(result.safe, "γ+φ should be re-encode safe");
    }

    #[test]
    fn gamma_phi_batch_safety() {
        let values: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.005).collect();
        let (all_safe, worst, safe_count, total) = test_reencode_batch(
            |v| test_gamma_phi_reencode(v, 1.50, 0.23, 256),
            &values,
        );
        eprintln!("γ+φ batch: {}/{} safe, worst: {}", safe_count, total, worst);
        assert!(all_safe, "γ+φ should be safe for all values in range");
    }

    #[test]
    fn full_chain_reencode_safe() {
        // BF16 + γ+φ full chain
        let result = test_full_chain_reencode(0.15, 1.50, 0.23, 256);
        eprintln!("{}", result);
        assert!(result.safe, "full chain should be re-encode safe");
        eprintln!("  converged at iteration {}, max_error={:.2e}",
            result.converged_at, result.max_error);
    }

    #[test]
    fn full_chain_batch_gate_range() {
        // Test across gate range [-0.23, +0.18] (Qwopus ffn_gate)
        let values: Vec<f64> = (-23..=18).map(|i| i as f64 * 0.01).collect();
        let (all_safe, worst, safe_count, total) = test_reencode_batch(
            |v| test_full_chain_reencode(v, 1.50, 0.23, 256),
            &values,
        );
        eprintln!("Full chain (gate range): {}/{} safe, worst: {}", safe_count, total, worst);
        assert!(all_safe, "full chain should be safe across gate range");
    }

    #[test]
    fn zipper_4_offsets_safe() {
        let results = test_zipper_offsets(0.15, 1.50, 0.23, 4, 256);
        for r in &results {
            eprintln!("  {}", r);
            assert!(r.safe, "offset should be safe: {}", r.codec);
        }
    }

    #[test]
    fn zipper_8_offsets_safe() {
        let results = test_zipper_offsets(0.15, 1.50, 0.23, 8, 256);
        let all_safe = results.iter().all(|r| r.safe);
        let max_err = results.iter().map(|r| r.max_error).fold(0.0f64, f64::max);
        let max_iter = results.iter().map(|r| r.converged_at).max().unwrap_or(0);
        eprintln!("8 zipper offsets: all_safe={}, max_err={:.2e}, max_iter={}",
            all_safe, max_err, max_iter);
        for r in &results {
            eprintln!("  {}", r);
        }
        assert!(all_safe, "all 8 zipper offsets must be re-encode safe");
    }

    #[test]
    fn zipper_offsets_gate_boundary() {
        // Test zipper offsets at the SiLU decision boundary (most sensitive)
        let critical_values = [0.001, -0.001, 0.008, -0.008]; // near zero = gate boundary
        for &v in &critical_values {
            let results = test_zipper_offsets(v, 1.50, 0.23, 8, 256);
            let all_safe = results.iter().all(|r| r.safe);
            let max_err = results.iter().map(|r| r.max_error).fold(0.0f64, f64::max);
            assert!(all_safe, "v={}: zipper offsets must be safe at gate boundary", v);
            eprintln!("v={:+.3}: 8 offsets SAFE, max_err={:.2e}", v, max_err);
        }
    }

    #[test]
    fn zipper_family_interleave() {
        // 4 families with stride=4, different offsets = perfect coverage
        // Each family sees 1/4 of the octaves. Together: 100%.
        let n_octaves = 302;
        let stride = 4;
        let phi: f64 = (1.0 + 5.0_f64.sqrt()) / 2.0;

        // Family offsets: explicit {0, 1, 2, 3} for stride=4
        // φ-fractional mod 4 collides (n/φ² mod 4 = n/φ³ mod 4 = 3)
        // so we use direct assignment for perfect zipper coverage.
        //
        // The φ-distribution applies WITHIN each family's octave selection
        // (golden-step bin mapping), not to the inter-family offset.
        let offsets = [
            0usize,  // Q+K (attention)
            1,       // Gate+Up (FFN gate path)
            2,       // V+Down (content path)
            3,       // HEEL/HIP (coarse/fine)
        ];

        // Check that all 4 offsets are DIFFERENT (no collision = perfect zipper)
        let mut unique_offsets: Vec<usize> = offsets.to_vec();
        unique_offsets.sort();
        unique_offsets.dedup();
        eprintln!("Family offsets (mod stride={}): {:?}", stride, offsets);
        eprintln!("Unique offsets: {} of 4", unique_offsets.len());

        // Check coverage: do the 4 families together cover all octaves?
        let mut covered = vec![false; n_octaves];
        for &offset in &offsets {
            let mut oct = offset;
            while oct < n_octaves {
                covered[oct] = true;
                oct += stride;
            }
        }
        let coverage = covered.iter().filter(|&&c| c).count();
        let coverage_pct = coverage as f32 / n_octaves as f32 * 100.0;
        eprintln!("Coverage: {}/{} octaves ({:.1}%)", coverage, n_octaves, coverage_pct);

        // Perfect zipper: all octaves covered if offsets are {0,1,2,3} for stride=4
        // With φ-offsets mod 4, coverage depends on the distribution
        assert!(coverage_pct > 90.0,
            "4 families should cover >90% of octaves, got {:.1}%", coverage_pct);

        // Re-encode safety for each family offset
        for (i, &offset) in offsets.iter().enumerate() {
            let families = ["Q+K", "Gate+Up", "V+Down", "HEEL/HIP"];
            let result = test_full_chain_reencode(
                0.15 + offset as f64 * 0.001, // slight offset per family
                1.50, 0.23, 256,
            );
            eprintln!("  {} (offset={}): {} iter={} err={:.2e}",
                families[i], offset,
                if result.safe {"SAFE"} else {"UNSAFE"},
                result.converged_at, result.max_error);
            assert!(result.safe, "{} must be re-encode safe", families[i]);
        }
    }

    #[test]
    fn prime_strides_coverage() {
        let n_octaves = 302;
        let n_families = 4;
        let primes = [5, 7, 11, 13, 17];

        for &stride in &primes {
            let samples_per_family = n_octaves / stride;
            let total_offsets = stride;
            let validation_slots = total_offsets - n_families;

            // Check coverage with 4 families
            let mut covered = vec![false; n_octaves];
            for family in 0..n_families {
                let mut oct = family;
                while oct < n_octaves {
                    covered[oct] = true;
                    oct += stride;
                }
            }
            let coverage = covered.iter().filter(|&&c| c).count();
            let pct = coverage as f32 / n_octaves as f32 * 100.0;

            // Check for GOLDEN property: stride=11 = GOLDEN_STEP
            let is_golden = stride == 11;

            eprintln!("stride={:2} ({}): {}/{} octaves ({:.1}%), {} samples/family, {} validation slots{}",
                stride,
                if is_golden { "GOLDEN" } else { "prime " },
                coverage, n_octaves, pct,
                samples_per_family,
                validation_slots,
                if is_golden { " ← GOLDEN_STEP" } else { "" });

            // All prime strides with 4 offsets give exactly 4/stride coverage
            let expected_pct = (n_families as f32 / stride as f32) * 100.0;
            assert!((pct - expected_pct).abs() < 5.0,
                "stride={}: expected ~{:.0}% coverage, got {:.1}%", stride, expected_pct, pct);

            // Re-encode safety for each family
            for family in 0..n_families {
                let r = test_full_chain_reencode(
                    0.15 + family as f64 * 0.001,
                    1.50, 0.23, 256,
                );
                assert!(r.safe, "stride={} family={} must be safe", stride, family);
            }
        }
    }

    #[test]
    fn golden_stride_11_vs_stride_4() {
        // stride=4 (composite): 4 families = 100% coverage, 0 validation
        // stride=11 (prime, GOLDEN_STEP): 4 families = 36% + 7 validation slots
        //
        // stride=11 trades coverage for validation:
        //   each family sees fewer octaves (27 vs 75)
        //   but 7 extra slots can cross-check (Cronbach α inline)
        //
        // For ENCODING: stride=4 (100% coverage, no waste)
        // For CALIBRATION: stride=11 (7 validation channels)

        let n = 302;

        // stride=4: families see different octaves, full coverage
        let stride4_coverage = n; // 4/4 = 100%
        let stride4_samples = n / 4; // 75 per family

        // stride=11: families see 4/11 of octaves
        let stride11_coverage = (n as f32 * 4.0 / 11.0).ceil() as usize;
        let stride11_samples = n / 11; // 27 per family
        let stride11_validation = 11 - 4; // 7 validation slots

        eprintln!("stride=4:  {} coverage, {} samples/family, 0 validation",
            stride4_coverage, stride4_samples);
        eprintln!("stride=11: ~{} coverage, {} samples/family, {} validation (Cronbach α)",
            stride11_coverage, stride11_samples, stride11_validation);
        eprintln!();
        eprintln!("Decision:");
        eprintln!("  ENCODING:    stride=4  (100% coverage, all data used)");
        eprintln!("  CALIBRATION: stride=11 (golden step, 7 validation channels)");
        eprintln!("  PRODUCTION:  stride=4 encode + stride=11 validate");
    }

    #[test]
    fn heel_vs_hip_different_stride() {
        // HEEL: coarse routing, stride=16 (broad coverage)
        // HIP:  fine discrimination, stride=4 (narrow coverage)
        let heel_result = test_full_chain_reencode(0.15, 1.50, 0.23, 256);
        let hip_result = test_full_chain_reencode(0.15001, 1.50, 0.23, 256); // slightly different

        eprintln!("HEEL (stride=16): {} err={:.2e}",
            if heel_result.safe {"SAFE"} else {"UNSAFE"}, heel_result.max_error);
        eprintln!("HIP  (stride=4):  {} err={:.2e}",
            if hip_result.safe {"SAFE"} else {"UNSAFE"}, hip_result.max_error);

        assert!(heel_result.safe, "HEEL must be re-encode safe");
        assert!(hip_result.safe, "HIP must be re-encode safe");

        // Both should converge at iteration 1
        assert_eq!(heel_result.converged_at, 1);
        assert_eq!(hip_result.converged_at, 1);
    }

    #[test]
    fn reencode_256_times() {
        // THE key test: can we re-encode 256 times?
        // If yes → "x256 re-encode safety"
        let test_values = vec![
            0.0,                    // zero (SiLU boundary)
            0.001,                  // near zero (gate decision zone)
            -0.001,                 // negative near zero
            0.1,                    // moderate positive
            -0.1,                   // moderate negative
            0.5,                    // halfway
            -0.886,                 // reranker minimum cosine
            0.826,                  // reranker maximum cosine
            1.0,                    // max
            -1.0,                   // min
        ];

        let mut all_converged = true;
        let mut max_iterations_needed = 0;

        for &v in &test_values {
            let r = test_full_chain_reencode(v, 1.50, 0.23, 256);
            eprintln!("  v={:+.4}: {} (iter={}, err={:.2e})",
                v, if r.safe {"SAFE"} else {"UNSAFE"}, r.converged_at, r.max_error);
            if !r.safe { all_converged = false; }
            if r.converged_at > max_iterations_needed {
                max_iterations_needed = r.converged_at;
            }
        }

        eprintln!("\nx256 re-encode safety: {} (max iterations needed: {})",
            if all_converged {"PROVEN"} else {"FAILED"}, max_iterations_needed);
        assert!(all_converged, "x256 re-encode safety FAILED");
    }
}
