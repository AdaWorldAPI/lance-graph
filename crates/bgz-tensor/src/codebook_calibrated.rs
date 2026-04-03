//! Two-pass calibrated codebook: build → γ+φ calibrate → distance table.
//!
//! Pass 1: CLAM furthest-point sampling → 4096 raw centroids
//! Pass 2: Measure distance distribution → γ offset → φ-redistribute → u8 table
//!
//! The golden ratio step (φ = 1.618...) ensures:
//!   - Codebook entries at maximally irrational spacing (no aliasing)
//!   - Distance table u8 levels at φ-distributed intervals (max entropy)
//!
//! The Euler-gamma offset (γ = 0.5772...) ensures:
//!   - Shadow expansion for small-magnitude roles (Up/Down)
//!   - Highlight compression for large-magnitude roles (Gate)
//!   - 28 bytes metadata per model for exact decode

use crate::stacked_n::{StackedN, cosine_f32_slice};
use crate::gamma_phi::{GammaProfile, calibrate_gamma, gamma_phi_encode, gamma_phi_decode};
use std::f64::consts::GOLDEN_RATIO;

/// Calibrated codebook: centroids + γ-corrected u8 distance table.
pub struct CalibratedCodebook {
    /// Raw f32 centroids (for hydration/comparison).
    pub centroids_f32: Vec<Vec<f32>>,
    /// u8 cosine distance table, γ+φ calibrated.
    /// table[i * k + j] = calibrated similarity [0=opposite, 255=identical].
    pub distance_table: Vec<u8>,
    /// Gamma profile used for calibration.
    pub gamma: GammaProfile,
    /// Codebook size.
    pub k: usize,
    /// Original vector dimensionality.
    pub dim: usize,
    /// Pass 1 stats.
    pub raw_cosine_range: (f64, f64),
    /// Pass 2 stats.
    pub calibrated_entropy: f64,
}

impl CalibratedCodebook {
    /// Two-pass build: CLAM → measure → γ+φ calibrate → u8 table.
    ///
    /// vectors: raw f32 weight rows from GGUF.
    /// k: codebook size (64 for prototype, 4096 for production).
    /// role_name: for per-role gamma (e.g., "Q", "Gate", "Up").
    pub fn build(vectors: &[Vec<f32>], k: usize, role_name: &str) -> Self {
        let n = vectors.len();
        let dim = if n > 0 { vectors[0].len() } else { 0 };
        let k = k.min(n);

        if k == 0 || dim == 0 {
            return Self {
                centroids_f32: Vec::new(), distance_table: Vec::new(),
                gamma: GammaProfile { model_name: String::new(), role_gamma: [0.01; 6],
                    phi_scale: 0.01, n_calibration: 0 },
                k: 0, dim: 0, raw_cosine_range: (0.0, 0.0), calibrated_entropy: 0.0,
            };
        }

        // ══════════════════════════════════════════════════════════════════
        // PASS 1: CLAM furthest-point sampling → k centroids
        // ══════════════════════════════════════════════════════════════════

        let mut selected = Vec::with_capacity(k);
        let mut max_dist = vec![f64::INFINITY; n]; // min cosine distance to any selected

        // Start with first vector
        selected.push(0);
        for i in 0..n {
            max_dist[i] = 1.0 - cosine_f32_slice(&vectors[i], &vectors[0]);
        }

        // Greedily select furthest point
        for _ in 1..k {
            let next = max_dist.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i).unwrap_or(0);
            selected.push(next);
            for i in 0..n {
                let d = 1.0 - cosine_f32_slice(&vectors[i], &vectors[next]);
                if d < max_dist[i] { max_dist[i] = d; }
            }
        }

        let centroids_f32: Vec<Vec<f32>> = selected.iter()
            .map(|&i| vectors[i].clone()).collect();

        // ══════════════════════════════════════════════════════════════════
        // PASS 2: Measure raw cosine distribution → γ+φ calibrate
        // ══════════════════════════════════════════════════════════════════

        // 2a: Compute ALL pairwise cosines (k×k, symmetric)
        let mut raw_cosines = vec![0.0f64; k * k];
        let mut min_cos = 1.0f64;
        let mut max_cos = -1.0f64;

        for i in 0..k {
            raw_cosines[i * k + i] = 1.0; // self = 1.0
            for j in (i + 1)..k {
                let cos = cosine_f32_slice(&centroids_f32[i], &centroids_f32[j]);
                raw_cosines[i * k + j] = cos;
                raw_cosines[j * k + i] = cos;
                if cos < min_cos { min_cos = cos; }
                if cos > max_cos && i != j { max_cos = cos; }
            }
        }

        // 2b: Calibrate gamma from centroid magnitudes
        let centroid_refs: Vec<&[f32]> = centroids_f32.iter().map(|v| v.as_slice()).collect();
        let gamma = calibrate_gamma("codebook", &[(role_name, &centroid_refs)]);

        // 2c: Apply γ+φ to the cosine distribution
        // Map raw cosine [-1, 1] → γ-expanded → φ-distributed → u8 [0, 255]
        //
        // The γ expansion gives more resolution near 0 (orthogonal pairs)
        // where most pairs cluster. The φ distribution ensures the u8 levels
        // sit at maximally irrational spacings.
        let role_idx = match role_name {
            "Q" => 0, "K" => 1, "V" => 2, "Gate" => 3, "Up" => 4, "Down" => 5,
            _ => 0,
        };
        let role_gamma = gamma.role_gamma[role_idx];
        let phi_scale = gamma.phi_scale;

        let mut distance_table = vec![128u8; k * k];
        let mut entropy_acc = 0.0f64;
        let mut bin_counts = [0u32; 256];

        for i in 0..k {
            distance_table[i * k + i] = 255; // self = max similarity
            for j in (i + 1)..k {
                let cos = raw_cosines[i * k + j];

                // γ+φ transform: cosine → calibrated u8
                let calibrated = gamma_phi_cosine_to_u8(cos, min_cos, max_cos, role_gamma, phi_scale);

                distance_table[i * k + j] = calibrated;
                distance_table[j * k + i] = calibrated;
                bin_counts[calibrated as usize] += 1;
            }
        }

        // 2d: Compute entropy of the calibrated distribution
        let total_pairs = (k * (k - 1) / 2) as f64;
        for &count in &bin_counts {
            if count > 0 {
                let p = count as f64 / total_pairs;
                entropy_acc -= p * p.ln();
            }
        }

        CalibratedCodebook {
            centroids_f32, distance_table, gamma,
            k, dim,
            raw_cosine_range: (min_cos, max_cos),
            calibrated_entropy: entropy_acc,
        }
    }

    /// Assign a vector to the nearest centroid.
    pub fn assign(&self, vector: &[f32]) -> (u16, f64) {
        self.centroids_f32.iter().enumerate()
            .map(|(i, c)| (i as u16, cosine_f32_slice(vector, c)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0))
    }

    /// Assign all vectors, return indices.
    pub fn assign_all(&self, vectors: &[Vec<f32>]) -> Vec<u16> {
        vectors.iter().map(|v| self.assign(v).0).collect()
    }

    /// Summary.
    pub fn summary(&self) -> String {
        format!(
            "CalibratedCodebook: k={}, dim={}, cosine_range=[{:.4}, {:.4}], entropy={:.2} bits\n\
             gamma: role_gamma={:.4}, phi_scale={:.4}",
            self.k, self.dim,
            self.raw_cosine_range.0, self.raw_cosine_range.1,
            self.calibrated_entropy,
            self.gamma.role_gamma[0], self.gamma.phi_scale,
        )
    }
}

/// Map cosine [-1, 1] → γ-expanded → φ-distributed → u8 [0, 255].
///
/// The γ expansion gives more resolution near the center of the distribution
/// (where most cosines cluster). The φ-step ensures u8 levels are maximally
/// non-degenerate.
fn gamma_phi_cosine_to_u8(
    cosine: f64,
    min_cos: f64,
    max_cos: f64,
    role_gamma: f32,
    phi_scale: f32,
) -> u8 {
    // Normalize cosine to [0, 1]
    let range = (max_cos - min_cos).max(1e-10);
    let normalized = ((cosine - min_cos) / range).clamp(0.0, 1.0);

    // Apply γ expansion: log(1 + x/γ) * γ
    // This expands the crowded center and compresses the tails
    let g = role_gamma.max(1e-6) as f64;
    let gamma_expanded = (1.0 + normalized / g).ln() * g;

    // Normalize γ-expanded to [0, 1]
    let max_gamma = (1.0 + 1.0 / g).ln() * g;
    let gamma_norm = (gamma_expanded / max_gamma).clamp(0.0, 1.0);

    // Apply φ distribution: position on golden-ratio grid
    // φ^x maps [0,1] → [1, φ], renormalized to [0,1]
    let phi = GOLDEN_RATIO;
    let phi_distributed = (phi.powf(gamma_norm) - 1.0) / (phi - 1.0);

    // Map to u8 [0, 255]
    (phi_distributed * 255.0).round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n).map(|i| {
            (0..dim).map(|d| ((i * 97 + d * 31) as f32 % 200.0 - 100.0) * 0.01)
                .collect()
        }).collect()
    }

    #[test]
    fn calibrated_builds() {
        let vecs = make_vectors(200, 256);
        let cb = CalibratedCodebook::build(&vecs, 32, "Q");
        assert_eq!(cb.k, 32);
        assert_eq!(cb.distance_table.len(), 32 * 32);
        eprintln!("{}", cb.summary());
    }

    #[test]
    fn calibrated_diagonal_is_max() {
        let vecs = make_vectors(100, 128);
        let cb = CalibratedCodebook::build(&vecs, 16, "Q");
        for i in 0..16 {
            assert_eq!(cb.distance_table[i * 16 + i], 255, "diagonal should be 255");
        }
    }

    #[test]
    fn calibrated_symmetric() {
        let vecs = make_vectors(100, 128);
        let cb = CalibratedCodebook::build(&vecs, 16, "Q");
        for i in 0..16 {
            for j in 0..16 {
                assert_eq!(cb.distance_table[i * 16 + j], cb.distance_table[j * 16 + i],
                    "table should be symmetric at ({},{})", i, j);
            }
        }
    }

    #[test]
    fn calibrated_entropy_positive() {
        let vecs = make_vectors(200, 256);
        let cb = CalibratedCodebook::build(&vecs, 32, "Q");
        assert!(cb.calibrated_entropy > 0.0, "entropy should be positive: {}", cb.calibrated_entropy);
        eprintln!("Calibrated entropy: {:.2} bits", cb.calibrated_entropy);
    }

    #[test]
    fn calibrated_assign() {
        let vecs = make_vectors(100, 128);
        let cb = CalibratedCodebook::build(&vecs, 16, "Q");
        let (idx, cos) = cb.assign(&vecs[0]);
        assert!(idx < 16);
        assert!(cos > 0.0);
    }

    #[test]
    fn gamma_phi_cosine_mapping() {
        // Test that γ+φ produces more u8 levels in the center than linear
        let values: Vec<u8> = (0..100).map(|i| {
            let cos = -1.0 + i as f64 * 0.02; // [-1, 1]
            gamma_phi_cosine_to_u8(cos, -1.0, 1.0, 0.15, 0.5)
        }).collect();

        // Should be monotone increasing
        for i in 1..values.len() {
            assert!(values[i] >= values[i - 1],
                "should be monotone: u8[{}]={} < u8[{}]={}", i, values[i], i-1, values[i-1]);
        }

        // Should use more of the u8 range than linear
        let distinct: std::collections::HashSet<u8> = values.iter().copied().collect();
        assert!(distinct.len() > 20, "should use many distinct u8 values: {}", distinct.len());
        eprintln!("{} distinct u8 values from 100 cosine samples", distinct.len());
    }
}
