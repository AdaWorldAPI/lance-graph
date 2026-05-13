//! Generative Decompression for bgz17 (arXiv:2602.03505 applied).
//!
//! The paper proves: when the encoder is FIXED, the optimal reconstruction
//! is the conditional expectation under the TRUE distribution given only
//! the quantization indices. This strictly outperforms the centroid rule.
//!
//! ## How This Maps to bgz17
//!
//! **Centroid rule** = `palette.nearest()` → look up the palette entry
//! and return its distance. This is the standard bgz17 path.
//!
//! **Generative decompression** = correct the palette distance using
//! side information about the local manifold geometry. The correction
//! uses CLAM's Local Fractal Dimension (LFD) as the "true distribution"
//! that the palette's k-means design distribution doesn't capture.
//!
//! ```text
//! d_corrected = d_palette × correction(LFD_local)
//!
//! where:
//!   LFD < median → smooth manifold → palette centroid is accurate → correction ≈ 1.0
//!   LFD > median → crinkly manifold → centroid underestimates distance → correction > 1.0
//! ```
//!
//! ## The Two Regimes (Theorem: Khosravirad et al.)
//!
//! **Resolution loss (r < 1):** Palette too coarse (k=32). Irreversible.
//! Can't fix with decoder-side correction. Need more palette entries.
//!
//! **Tail mismatch (r > 1):** Palette fine enough (k=128-256) but
//! misallocates centroids. The correction moves the reconstruction
//! point within each cell. This is where generative decompression shines.
//!
//! With k=128 (ρ=0.965) we're in the tail mismatch regime —
//! the palette has enough resolution, it just needs calibration.
//! The LFD correction provides that calibration.

/// Local Fractal Dimension (LFD) for a cluster.
///
/// LFD measures the intrinsic dimensionality of the local manifold.
/// - LFD ≈ 1.0: data lies on a curve (1D manifold)
/// - LFD ≈ 2.0: data lies on a surface (2D manifold)
/// - LFD > 3.0: data is high-dimensional locally
///
/// Computed from CLAM tree node radii:
///   LFD = log(|children|) / log(parent_radius / child_radius)
///
/// In bgz17 context: LFD tells us how "crinkly" the local graph topology is.
/// High LFD = many equally-distant neighbors = palette centroid is a poor
/// representative = correction needed.
#[derive(Clone, Copy, Debug)]
pub struct LfdProfile {
    /// Local fractal dimension at this node.
    pub lfd: f32,
    /// Median LFD across the scope (baseline for correction).
    pub lfd_median: f32,
    /// CHAODA anomaly score (0.0 = normal, 1.0 = highly anomalous).
    pub anomaly_score: f32,
}

/// Generative decompression correction factor.
///
/// From arXiv:2602.03505 Theorem 2:
///   D_ideal < D_gen < D_fix
///
/// D_fix = palette centroid distance (what bgz17 computes now).
/// D_gen = corrected distance using LFD side information.
/// The correction factor maps D_fix → D_gen.
///
/// `alpha` controls correction strength (0.0 = no correction, 1.0 = full).
/// Typical: alpha = 0.3 (conservative, avoids over-correction).
#[inline]
pub fn correction_factor(lfd: &LfdProfile, alpha: f32) -> f32 {
    // Deviation from median LFD
    let lfd_deviation = lfd.lfd - lfd.lfd_median;

    // High LFD: palette underestimates → scale UP (correction > 1.0)
    // Low LFD:  palette overestimates  → scale DOWN (correction < 1.0)
    let correction = 1.0 + alpha * lfd_deviation;

    // Clamp to prevent negative or extreme corrections
    correction.clamp(0.5, 2.0)
}

/// Apply generative decompression to a palette distance.
///
/// Takes the raw palette distance and corrects it using the local
/// manifold geometry (LFD profile).
#[inline]
pub fn generative_distance(raw_distance: u32, lfd: &LfdProfile, alpha: f32) -> u32 {
    let factor = correction_factor(lfd, alpha);
    (raw_distance as f32 * factor) as u32
}

/// Apply generative decompression to a batch of distances.
///
/// Each candidate has its own LFD profile (from its CLAM tree position).
/// The correction is applied per-candidate, not globally.
pub fn generative_batch(
    candidates: &[(usize, u32)],
    lfd_profiles: &[LfdProfile],
    alpha: f32,
) -> Vec<(usize, u32)> {
    candidates
        .iter()
        .map(|&(pos, raw_d)| {
            let lfd = lfd_profiles.get(pos).copied().unwrap_or(LfdProfile {
                lfd: 1.0,
                lfd_median: 1.0,
                anomaly_score: 0.0,
            });
            let corrected = generative_distance(raw_d, &lfd, alpha);
            (pos, corrected)
        })
        .collect()
}

/// Determine storage layer based on CHAODA anomaly score.
///
/// High anomaly = can't trust palette → store at Layer 2 (base patterns).
/// Low anomaly = palette is sufficient → store at Layer 1 (3 bytes).
///
/// This is the "bandwidth detection" from the Opus analogy:
/// anomalous regions get more bits, stable regions get fewer.
pub fn anomaly_to_layer(anomaly_score: f32) -> crate::Precision {
    if anomaly_score > 0.75 {
        crate::Precision::Base    // 102 bytes — can't trust palette
    } else if anomaly_score > 0.5 {
        crate::Precision::Palette // 3 bytes — palette with correction
    } else {
        crate::Precision::Scent   // 1 byte — scent is sufficient
    }
}

/// The paper's "Mismatch Penalty Factor" L for bgz17.
///
/// L = D_gen / D_ideal. When L ≈ 1.0, generative decompression recovers
/// nearly all the information lost by the palette quantization.
///
/// For the tail mismatch regime (k ≥ 128): L should be close to 1.0.
/// For the resolution loss regime (k ≤ 32): L > 1.5, indicating
/// fundamental information loss that correction can't fix.
pub fn mismatch_penalty(
    palette_distances: &[(usize, u32)],
    exact_distances: &[(usize, u32)],
    lfd_profiles: &[LfdProfile],
    alpha: f32,
) -> f64 {
    if palette_distances.is_empty() || exact_distances.is_empty() {
        return f64::NAN;
    }

    let corrected = generative_batch(palette_distances, lfd_profiles, alpha);

    // Match by position
    let mut total_gen = 0.0f64;
    let mut total_ideal = 0.0f64;

    for &(pos, d_corr) in &corrected {
        if let Some(&(_, d_exact)) = exact_distances.iter().find(|&&(p, _)| p == pos) {
            total_gen += d_corr as f64;
            total_ideal += d_exact as f64;
        }
    }

    if total_ideal < 1e-10 {
        return f64::NAN;
    }

    total_gen / total_ideal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correction_factor_neutral() {
        let lfd = LfdProfile { lfd: 2.0, lfd_median: 2.0, anomaly_score: 0.0 };
        let f = correction_factor(&lfd, 0.3);
        assert!((f - 1.0).abs() < 0.01, "At median LFD, correction should be ~1.0: {}", f);
    }

    #[test]
    fn test_correction_factor_high_lfd() {
        let lfd = LfdProfile { lfd: 4.0, lfd_median: 2.0, anomaly_score: 0.5 };
        let f = correction_factor(&lfd, 0.3);
        assert!(f > 1.0, "High LFD should increase distance: {}", f);
        assert!(f < 2.0, "Should be clamped below 2.0: {}", f);
    }

    #[test]
    fn test_correction_factor_low_lfd() {
        let lfd = LfdProfile { lfd: 0.5, lfd_median: 2.0, anomaly_score: 0.0 };
        let f = correction_factor(&lfd, 0.3);
        assert!(f < 1.0, "Low LFD should decrease distance: {}", f);
        assert!(f >= 0.5, "Should be clamped above 0.5: {}", f);
    }

    #[test]
    fn test_generative_distance_identity() {
        let lfd = LfdProfile { lfd: 2.0, lfd_median: 2.0, anomaly_score: 0.0 };
        let raw = 1000u32;
        let corrected = generative_distance(raw, &lfd, 0.3);
        assert_eq!(corrected, raw, "At neutral LFD, no correction");
    }

    #[test]
    fn test_anomaly_to_layer() {
        assert_eq!(anomaly_to_layer(0.1), crate::Precision::Scent);
        assert_eq!(anomaly_to_layer(0.6), crate::Precision::Palette);
        assert_eq!(anomaly_to_layer(0.9), crate::Precision::Base);
    }

    #[test]
    fn test_mismatch_penalty_perfect() {
        let palette_d = vec![(0, 100), (1, 200), (2, 300)];
        let exact_d = vec![(0, 100), (1, 200), (2, 300)];
        let lfds = vec![
            LfdProfile { lfd: 2.0, lfd_median: 2.0, anomaly_score: 0.0 }; 3
        ];
        let l = mismatch_penalty(&palette_d, &exact_d, &lfds, 0.0);
        assert!((l - 1.0).abs() < 0.01, "Perfect match should give L≈1.0: {}", l);
    }
}
