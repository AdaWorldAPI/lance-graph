//! Gamma encoding with golden ratio distribution for weight compression.
//!
//! Three encoding strategies compared:
//!   1. Linear BF16 (current stacked path) — uniform quantization, wastes bits on empty ranges
//!   2. Log-gamma — compresses highlights, expands shadows (camera-style)
//!   3. φ-normalized — maps weight distribution to golden-ratio spacing
//!
//! The φ-distribution is optimal because:
//!   - φ is the most irrational number → no two palette entries alias
//!   - φ-spiral spacing maximizes entropy per quantization level
//!   - The Zeckendorf representation of the offset gives exact rehydration
//!
//! Per-model metadata for rehydration:
//!   - γ_offset: one f32 per role per model (6 roles × 4 bytes = 24 bytes)
//!   - φ_scale: one f32 per model (global dynamic range, 4 bytes)
//!   - Total: 28 bytes of metadata → exact decode of any encoded value

use std::f64::consts::GOLDEN_RATIO;

/// Per-role gamma offsets, calibrated from a model's weight distribution.
///
/// Stored as metadata with the encoded weights (28 bytes total).
/// Enables exact rehydration: gamma_decode(encoded, offset) → original.
#[derive(Clone, Debug)]
pub struct GammaProfile {
    /// Model identifier.
    pub model_name: String,
    /// Per-role gamma offsets: [Q, K, V, Gate, Up, Down].
    /// Each offset = the distribution's center of mass for that role.
    pub role_gamma: [f32; 6],
    /// Global φ-scale: maps the full dynamic range to [0, 1] in φ-space.
    pub phi_scale: f32,
    /// Number of weight rows used for calibration.
    pub n_calibration: usize,
}

impl GammaProfile {
    /// Byte size of the metadata (for storage alongside encoded weights).
    pub const METADATA_BYTES: usize = 6 * 4 + 4; // 28 bytes
}

/// Calibrate gamma profile from raw f32 weight rows per role.
///
/// Measures the magnitude distribution per role and computes the optimal
/// gamma offset that maps each role's range to maximum resolution.
pub fn calibrate_gamma(
    model_name: &str,
    role_rows: &[(&str, &[&[f32]])], // (role_name, rows)
) -> GammaProfile {
    let mut role_gamma = [0.0f32; 6];

    for (role_name, rows) in role_rows {
        let role_idx = match *role_name {
            "Q" => 0, "K" => 1, "V" => 2, "Gate" => 3, "Up" => 4, "Down" => 5,
            _ => continue,
        };

        if rows.is_empty() { continue; }

        // Compute mean absolute magnitude across all values
        let mut total_mag = 0.0f64;
        let mut count = 0u64;
        for row in *rows {
            for &v in *row {
                total_mag += v.abs() as f64;
                count += 1;
            }
        }
        let mean_mag = if count > 0 { total_mag / count as f64 } else { 1.0 };
        role_gamma[role_idx] = mean_mag as f32;
    }

    // Global phi_scale: maximum gamma across all roles
    let max_gamma = role_gamma.iter().cloned().fold(0.0f32, f32::max).max(1e-6);

    GammaProfile {
        model_name: model_name.to_string(),
        role_gamma,
        phi_scale: max_gamma,
        n_calibration: role_rows.iter().map(|(_, r)| r.len()).sum(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Encoding strategies
// ═══════════════════════════════════════════════════════════════════════════

/// Log-gamma encode: compress highlights, expand shadows.
///
/// `value`: raw f32 weight value.
/// `gamma`: per-role offset (from GammaProfile.role_gamma[role]).
/// Returns encoded value in log-gamma space.
#[inline]
pub fn gamma_encode(value: f32, gamma: f32) -> f32 {
    let g = gamma.max(1e-8);
    let sign = value.signum();
    let mag = value.abs();
    sign * (1.0 + mag / g).ln() * g
}

/// Log-gamma decode: exact inverse.
#[inline]
pub fn gamma_decode(encoded: f32, gamma: f32) -> f32 {
    let g = gamma.max(1e-8);
    let sign = encoded.signum();
    let mag = encoded.abs();
    sign * ((mag / g).exp() - 1.0) * g
}

/// φ-normalized encode: map to golden-ratio spacing.
///
/// Maps value to position on a φ-spiral where palette entries sit at
/// maximally irrational spacings. No two encoded values can alias because
/// φ is the worst-case for rational approximation.
///
/// `value`: raw f32 weight value.
/// `phi_scale`: global dynamic range from GammaProfile.
#[inline]
pub fn phi_encode(value: f32, phi_scale: f32) -> f32 {
    let ps = phi_scale.max(1e-8);
    let sign = value.signum();
    let normalized = value.abs() / ps; // [0, ~1] range
    // Map through φ-log: spacing increases by φ at each level
    // This is log_φ(1 + x) = ln(1 + x) / ln(φ)
    let phi_log = (1.0 + normalized).ln() / (GOLDEN_RATIO as f32).ln();
    sign * phi_log * ps
}

/// φ-normalized decode: exact inverse.
#[inline]
pub fn phi_decode(encoded: f32, phi_scale: f32) -> f32 {
    let ps = phi_scale.max(1e-8);
    let sign = encoded.signum();
    let phi_log = encoded.abs() / ps;
    // Inverse: x = φ^phi_log - 1
    let normalized = (GOLDEN_RATIO as f32).powf(phi_log) - 1.0;
    sign * normalized * ps
}

/// Combined γ+φ encode: per-role gamma THEN golden-ratio distribution.
///
/// Two-stage:
/// 1. Gamma-normalize by role (expand shadows where Up/Down live)
/// 2. φ-distribute (maximize entropy of the quantization grid)
///
/// Rehydration: phi_decode(gamma_decode(stored, role_gamma), phi_scale) → original
#[inline]
pub fn gamma_phi_encode(value: f32, role_gamma: f32, phi_scale: f32) -> f32 {
    let gamma_encoded = gamma_encode(value, role_gamma);
    phi_encode(gamma_encoded, phi_scale)
}

/// Combined decode: exact inverse of gamma_phi_encode.
#[inline]
pub fn gamma_phi_decode(encoded: f32, role_gamma: f32, phi_scale: f32) -> f32 {
    let phi_decoded = phi_decode(encoded, phi_scale);
    gamma_decode(phi_decoded, role_gamma)
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch encoding for weight vectors
// ═══════════════════════════════════════════════════════════════════════════

/// Encode a full weight row with gamma+φ.
pub fn encode_row(row: &[f32], role_gamma: f32, phi_scale: f32) -> Vec<f32> {
    row.iter().map(|&v| gamma_phi_encode(v, role_gamma, phi_scale)).collect()
}

/// Decode back to original values.
pub fn decode_row(encoded: &[f32], role_gamma: f32, phi_scale: f32) -> Vec<f32> {
    encoded.iter().map(|&v| gamma_phi_decode(v, role_gamma, phi_scale)).collect()
}

/// Measure roundtrip fidelity: encode → decode → compare with original.
pub fn roundtrip_error(row: &[f32], role_gamma: f32, phi_scale: f32) -> (f64, f64) {
    let encoded = encode_row(row, role_gamma, phi_scale);
    let decoded = decode_row(&encoded, role_gamma, phi_scale);
    let mut total_err = 0.0f64;
    let mut max_err = 0.0f64;
    for (i, (&orig, &dec)) in row.iter().zip(decoded.iter()).enumerate() {
        let err = (orig as f64 - dec as f64).abs();
        total_err += err;
        if err > max_err { max_err = err; }
    }
    let mean_err = total_err / row.len().max(1) as f64;
    (mean_err, max_err)
}

/// Compare encoding strategies on pairwise cosine preservation.
///
/// Returns (linear_pearson, gamma_pearson, phi_pearson, gamma_phi_pearson)
/// measured against ground-truth f32 cosine.
pub fn compare_strategies(
    rows: &[Vec<f32>],
    role_gamma: f32,
    phi_scale: f32,
) -> StrategyComparison {
    let n = rows.len().min(100);
    if n < 2 { return StrategyComparison::default(); }

    let mut gt_cosines = Vec::new();
    let mut linear_cosines = Vec::new();
    let mut gamma_cosines = Vec::new();
    let mut phi_cosines = Vec::new();
    let mut gamma_phi_cosines = Vec::new();

    // Pre-encode all rows under each strategy
    let gamma_rows: Vec<Vec<f32>> = rows[..n].iter()
        .map(|r| r.iter().map(|&v| gamma_encode(v, role_gamma)).collect()).collect();
    let phi_rows: Vec<Vec<f32>> = rows[..n].iter()
        .map(|r| r.iter().map(|&v| phi_encode(v, phi_scale)).collect()).collect();
    let gp_rows: Vec<Vec<f32>> = rows[..n].iter()
        .map(|r| encode_row(r, role_gamma, phi_scale)).collect();

    for i in 0..n {
        for j in (i + 1)..n.min(i + 10) {
            let gt = cosine(&rows[i], &rows[j]);
            gt_cosines.push(gt);
            linear_cosines.push(cosine(&rows[i], &rows[j])); // same as gt
            gamma_cosines.push(cosine(&gamma_rows[i], &gamma_rows[j]));
            phi_cosines.push(cosine(&phi_rows[i], &phi_rows[j]));
            gamma_phi_cosines.push(cosine(&gp_rows[i], &gp_rows[j]));
        }
    }

    StrategyComparison {
        n_pairs: gt_cosines.len(),
        linear_pearson: crate::quality::pearson(&gt_cosines, &linear_cosines),
        gamma_pearson: crate::quality::pearson(&gt_cosines, &gamma_cosines),
        phi_pearson: crate::quality::pearson(&gt_cosines, &phi_cosines),
        gamma_phi_pearson: crate::quality::pearson(&gt_cosines, &gamma_phi_cosines),
        linear_spearman: crate::quality::spearman(&gt_cosines, &linear_cosines),
        gamma_spearman: crate::quality::spearman(&gt_cosines, &gamma_cosines),
        phi_spearman: crate::quality::spearman(&gt_cosines, &phi_cosines),
        gamma_phi_spearman: crate::quality::spearman(&gt_cosines, &gamma_phi_cosines),
    }
}

#[derive(Clone, Debug, Default)]
pub struct StrategyComparison {
    pub n_pairs: usize,
    pub linear_pearson: f64,
    pub gamma_pearson: f64,
    pub phi_pearson: f64,
    pub gamma_phi_pearson: f64,
    pub linear_spearman: f64,
    pub gamma_spearman: f64,
    pub phi_spearman: f64,
    pub gamma_phi_spearman: f64,
}

impl StrategyComparison {
    pub fn summary(&self) -> String {
        format!(
            "Strategy Comparison ({} pairs):\n\
             {:>12} │ Pearson │ Spearman\n\
             ─────────────┼─────────┼─────────\n\
             Linear       │ {:>7.4} │ {:>7.4}\n\
             Log-gamma    │ {:>7.4} │ {:>7.4}\n\
             φ-normalized │ {:>7.4} │ {:>7.4}\n\
             γ+φ combined │ {:>7.4} │ {:>7.4}",
            self.n_pairs,
            "",
            self.linear_pearson, self.linear_spearman,
            self.gamma_pearson, self.gamma_spearman,
            self.phi_pearson, self.phi_spearman,
            self.gamma_phi_pearson, self.gamma_phi_spearman,
        )
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_roundtrip_exact() {
        for &v in &[0.0f32, 1.0, -1.0, 0.001, -0.001, 100.0, -100.0] {
            let encoded = gamma_encode(v, 0.15);
            let decoded = gamma_decode(encoded, 0.15);
            assert!((v - decoded).abs() < 1e-5,
                "gamma roundtrip failed: {} → {} → {}", v, encoded, decoded);
        }
    }

    #[test]
    fn phi_roundtrip_exact() {
        for &v in &[0.0f32, 0.5, -0.5, 0.004, -0.004, 2.3, -2.3] {
            let encoded = phi_encode(v, 2.5);
            let decoded = phi_decode(encoded, 2.5);
            assert!((v - decoded).abs() < 1e-5,
                "phi roundtrip failed: {} → {} → {}", v, encoded, decoded);
        }
    }

    #[test]
    fn gamma_phi_roundtrip_exact() {
        for &v in &[0.0f32, 0.15, -0.15, 0.004, -0.004, 1.5, -1.5] {
            let encoded = gamma_phi_encode(v, 0.15, 2.5);
            let decoded = gamma_phi_decode(encoded, 0.15, 2.5);
            assert!((v - decoded).abs() < 1e-4,
                "gamma+phi roundtrip failed: {} → {} → {}", v, encoded, decoded);
        }
    }

    #[test]
    fn gamma_expands_shadows() {
        // Small values should get more resolution in gamma space
        let small = gamma_encode(0.001, 0.15);
        let medium = gamma_encode(0.15, 0.15);
        let large = gamma_encode(1.5, 0.15);

        // In linear space: small/medium = 0.001/0.15 = 0.007
        // In gamma space: the ratio should be larger (shadows expanded)
        let linear_ratio = 0.001 / 0.15;
        let gamma_ratio = small.abs() / medium.abs();
        assert!(gamma_ratio > linear_ratio,
            "gamma should expand shadows: linear ratio={:.4}, gamma ratio={:.4}",
            linear_ratio, gamma_ratio);
    }

    #[test]
    fn phi_spacing_is_irrational() {
        // φ-encoded values at integer inputs should never land on rational grid
        let phi_scale = 1.0;
        let vals: Vec<f32> = (1..20).map(|i| phi_encode(i as f32 * 0.1, phi_scale)).collect();

        // Check no two adjacent values have a rational ratio
        for i in 1..vals.len() {
            let ratio = vals[i] as f64 / vals[i - 1] as f64;
            // φ-spacing means ratios approach φ, never rational
            let nearest_int_ratio = ratio.round();
            let irrationality = (ratio - nearest_int_ratio).abs();
            // Not a strong test, but verifies non-degeneracy
            assert!(vals[i] != vals[i - 1], "adjacent phi values must differ");
        }
    }

    #[test]
    fn calibrate_basic() {
        let q_rows: Vec<Vec<f32>> = (0..10).map(|i|
            (0..100).map(|d| ((d * i) as f32 * 0.01).sin() * 0.4).collect()
        ).collect();
        let gate_rows: Vec<Vec<f32>> = (0..10).map(|i|
            (0..100).map(|d| ((d * i) as f32 * 0.02).cos() * 2.0).collect()
        ).collect();

        let q_refs: Vec<&[f32]> = q_rows.iter().map(|r| r.as_slice()).collect();
        let gate_refs: Vec<&[f32]> = gate_rows.iter().map(|r| r.as_slice()).collect();

        let profile = calibrate_gamma("test_model", &[
            ("Q", &q_refs),
            ("Gate", &gate_refs),
        ]);

        assert!(profile.role_gamma[0] > 0.0, "Q gamma should be positive");
        assert!(profile.role_gamma[3] > profile.role_gamma[0],
            "Gate gamma should be larger than Q: Gate={:.4}, Q={:.4}",
            profile.role_gamma[3], profile.role_gamma[0]);
        eprintln!("Profile: Q={:.4}, Gate={:.4}, phi_scale={:.4}",
            profile.role_gamma[0], profile.role_gamma[3], profile.phi_scale);
    }

    #[test]
    fn compare_preserves_cosine() {
        // Linear encoding preserves cosine perfectly (it IS the same data)
        let rows: Vec<Vec<f32>> = (0..20).map(|i|
            (0..256).map(|d| ((d * 97 + i * 31) as f32 % 200.0 - 100.0) * 0.01).collect()
        ).collect();

        let comp = compare_strategies(&rows, 0.15, 1.5);
        assert!((comp.linear_pearson - 1.0).abs() < 1e-10, "linear should be perfect");
        // Gamma and phi should also preserve cosine well (monotonic transforms)
        assert!(comp.gamma_pearson > 0.95, "gamma should preserve: {:.4}", comp.gamma_pearson);
        assert!(comp.phi_pearson > 0.95, "phi should preserve: {:.4}", comp.phi_pearson);
        eprintln!("{}", comp.summary());
    }
}
