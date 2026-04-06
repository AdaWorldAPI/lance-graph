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
