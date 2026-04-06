//! Cronbach α for multi-lens internal consistency.
//!
//! From HANDOVER_CALIBRATION_SESSION.md (H5):
//!   For N sentence pairs, compute distances via all K lenses.
//!   Each lens = one "item" in the psychometric instrument.
//!   Cronbach α = internal consistency of the multi-lens measurement.
//!
//! Used in two contexts:
//!
//! 1. CALIBRATION (H5 hypothesis):
//!    α > 0.90 for similarity → lenses are redundant (use one, save compute)
//!    α < 0.70 for relevance → lenses see different things (superposition valuable)
//!
//! 2. ENCODING QUORUM (per centroid pair):
//!    During 5-lane table build, compute α across existing baked tables.
//!    High α → confident encoding. Low α → boundary_risk = HIGH.
//!    The quorum replaces the BF16 ±0.008 heuristic with empirical cross-model test.

/// Cronbach's alpha for K items measured on N subjects.
///
/// `items[k][n]` = measurement by lens k for pair n.
/// Returns α in range (-∞, 1.0]. Typically:
///   α > 0.90: excellent internal consistency (lenses agree)
///   α 0.70-0.90: acceptable (mostly agree)
///   α < 0.70: poor (lenses see different things)
///   α < 0.50: unacceptable (no agreement)
pub fn cronbach_alpha(items: &[&[f32]]) -> f32 {
    let k = items.len();
    if k < 2 { return 0.0; }
    let n = items[0].len();
    if n < 2 { return 0.0; }
    // Verify all items have same length
    for item in items {
        if item.len() != n { return 0.0; }
    }

    // Total score per subject (sum across all items)
    let totals: Vec<f32> = (0..n)
        .map(|pair| items.iter().map(|lens| lens[pair]).sum::<f32>())
        .collect();
    let var_total = variance(&totals);

    if var_total < 1e-10 { return 0.0; } // no variance = undefined

    // Sum of item variances
    let var_sum: f32 = items.iter()
        .map(|lens| variance(lens))
        .sum();

    // α = (k / (k-1)) × (1 - Σvar_item / var_total)
    let kf = k as f32;
    (kf / (kf - 1.0)) * (1.0 - var_sum / var_total)
}

/// Per-pair variance-based agreement score across lens tables.
///
/// NOTE: This is NOT Cronbach α per pair. It's a normalized variance
/// score measuring how much the lenses agree on each centroid pair.
/// Low variance = high agreement. High variance = disagreement.
///
/// Returns a score per pair:
///   255 = low variance (lenses agree)
///   128 = moderate variance
///     0 = high variance (lenses disagree — investigate or LEAF validate)
///
/// For actual Cronbach α use `cronbach_alpha()` on the full corpus.
pub fn variance_agreement_scores(
    tables: &[&[u8]],  // K tables, each N×N u8
    n: usize,           // table dimension (N)
) -> Vec<u8> {
    let k = tables.len();
    if k < 2 || n < 2 {
        return vec![128u8; n * n]; // default medium confidence
    }

    let mut scores = vec![0u8; n * n];

    for i in 0..n {
        scores[i * n + i] = 255; // diagonal = perfect agreement (self-distance)
        for j in (i + 1)..n {
            // Collect this pair's value across all lenses
            let values: Vec<f32> = tables.iter()
                .map(|t| t[i * n + j] as f32)
                .collect();

            // Compute agreement: how much do the lenses agree on this pair?
            let mean: f32 = values.iter().sum::<f32>() / k as f32;
            let var: f32 = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / k as f32;
            let max_var = 255.0f32 * 255.0 / 4.0; // max variance for u8

            // Low variance relative to max = high agreement
            let agreement = 1.0 - (var / max_var).sqrt();
            let score = (agreement * 255.0).round().clamp(0.0, 255.0) as u8;

            scores[i * n + j] = score;
            scores[j * n + i] = score;
        }
    }

    scores
}

/// Interpret a quorum score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuorumLevel {
    /// α > 0.90 — all models agree. Encode confidently.
    High,
    /// α 0.70-0.90 — mostly agree. Encode with boundary_risk metadata.
    Medium,
    /// α < 0.70 — models disagree. Mark for LEAF validation.
    Low,
    /// α < 0.50 — no agreement. This pair is genuinely ambiguous.
    Ambiguous,
}

impl QuorumLevel {
    pub fn from_score(score: u8) -> Self {
        match score {
            230..=255 => QuorumLevel::High,
            179..=229 => QuorumLevel::Medium,
            128..=178 => QuorumLevel::Low,
            _ => QuorumLevel::Ambiguous,
        }
    }

    /// Should this pair skip the fast cascade and go to LEAF validation?
    pub fn needs_leaf_validation(self) -> bool {
        matches!(self, QuorumLevel::Low | QuorumLevel::Ambiguous)
    }
}

/// Compute Cronbach α for the full calibration corpus.
///
/// Each lens provides distances for the same N text pairs.
/// Returns α and per-pair variance (for identifying problematic pairs).
pub struct CronbachResult {
    /// Overall Cronbach α across all pairs.
    pub alpha: f32,
    /// Per-pair variance across lenses (high = lenses disagree on this pair).
    pub pair_variances: Vec<f32>,
    /// Number of pairs where lenses strongly disagree (variance > threshold).
    pub disagreement_count: usize,
    /// Number of lenses (K).
    pub n_lenses: usize,
    /// Number of pairs (N).
    pub n_pairs: usize,
}

impl CronbachResult {
    pub fn summary(&self) -> String {
        let status = if self.alpha > 0.90 { "EXCELLENT (lenses redundant)" }
            else if self.alpha > 0.70 { "ACCEPTABLE (superposition adds a little)" }
            else if self.alpha > 0.50 { "POOR (lenses see different things — superposition valuable)" }
            else { "UNACCEPTABLE (no agreement — investigate)" };
        format!(
            "Cronbach α = {:.3} [{}]\n  {} lenses × {} pairs, {} disagreements ({:.1}%)",
            self.alpha, status,
            self.n_lenses, self.n_pairs, self.disagreement_count,
            self.disagreement_count as f32 / self.n_pairs.max(1) as f32 * 100.0,
        )
    }
}

/// Compute Cronbach α from lens distance vectors.
pub fn cronbach_analysis(lens_distances: &[Vec<f32>]) -> CronbachResult {
    let k = lens_distances.len();
    let n = lens_distances.first().map(|v| v.len()).unwrap_or(0);

    let refs: Vec<&[f32]> = lens_distances.iter().map(|v| v.as_slice()).collect();
    let alpha = cronbach_alpha(&refs);

    // Per-pair variance
    let pair_variances: Vec<f32> = (0..n).map(|pair| {
        let values: Vec<f32> = lens_distances.iter().map(|lens| lens[pair]).collect();
        variance(&values)
    }).collect();

    // Count high-variance pairs (threshold: > 1 std of pair variances)
    let var_mean = pair_variances.iter().sum::<f32>() / n.max(1) as f32;
    let var_std = variance(&pair_variances).sqrt();
    let threshold = var_mean + var_std;
    let disagreement_count = pair_variances.iter().filter(|&&v| v > threshold).count();

    CronbachResult {
        alpha,
        pair_variances,
        disagreement_count,
        n_lenses: k,
        n_pairs: n,
    }
}

fn variance(data: &[f32]) -> f32 {
    let n = data.len() as f32;
    if n < 2.0 { return 0.0; }
    let mean = data.iter().sum::<f32>() / n;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_agreement() {
        // All lenses give identical measurements
        let lens1 = vec![0.9, 0.5, 0.1, 0.8];
        let lens2 = vec![0.9, 0.5, 0.1, 0.8];
        let lens3 = vec![0.9, 0.5, 0.1, 0.8];
        let alpha = cronbach_alpha(&[&lens1, &lens2, &lens3]);
        assert!((alpha - 1.0).abs() < 0.01,
            "identical items should give α≈1.0, got {}", alpha);
    }

    #[test]
    fn no_agreement() {
        // Lenses give uncorrelated measurements
        let lens1 = vec![0.9, 0.1, 0.5, 0.3];
        let lens2 = vec![0.1, 0.9, 0.3, 0.5];
        let lens3 = vec![0.5, 0.3, 0.9, 0.1];
        let alpha = cronbach_alpha(&[&lens1, &lens2, &lens3]);
        assert!(alpha < 0.5, "uncorrelated should give low α, got {}", alpha);
    }

    #[test]
    fn partial_agreement() {
        // Two agree, one disagrees
        let lens1 = vec![0.9, 0.7, 0.3, 0.1];
        let lens2 = vec![0.85, 0.65, 0.25, 0.15];
        let lens3 = vec![0.1, 0.3, 0.7, 0.9]; // inverted
        let alpha = cronbach_alpha(&[&lens1, &lens2, &lens3]);
        assert!(alpha < 0.7, "one inverted should reduce α, got {}", alpha);
    }

    #[test]
    fn quorum_scores_diagonal() {
        let t1 = vec![255u8, 100, 100, 255]; // 2×2
        let t2 = vec![255, 100, 100, 255];
        let scores = variance_agreement_scores(&[&t1, &t2], 2);
        assert_eq!(scores[0], 255); // diagonal
        assert_eq!(scores[3], 255); // diagonal
        assert!(scores[1] > 200); // both agree on 100
    }

    #[test]
    fn quorum_scores_disagreement() {
        let t1 = vec![255, 200, 200, 255]; // 2×2
        let t2 = vec![255, 50, 50, 255]; // disagrees on off-diagonal
        let scores = variance_agreement_scores(&[&t1, &t2], 2);
        assert!(scores[1] < 200, "disagreement should lower score: {}", scores[1]);
    }

    #[test]
    fn quorum_level_classification() {
        assert_eq!(QuorumLevel::from_score(250), QuorumLevel::High);
        assert_eq!(QuorumLevel::from_score(200), QuorumLevel::Medium);
        assert_eq!(QuorumLevel::from_score(150), QuorumLevel::Low);
        assert_eq!(QuorumLevel::from_score(50), QuorumLevel::Ambiguous);
        assert!(!QuorumLevel::High.needs_leaf_validation());
        assert!(QuorumLevel::Low.needs_leaf_validation());
        assert!(QuorumLevel::Ambiguous.needs_leaf_validation());
    }

    #[test]
    fn cronbach_analysis_summary() {
        let lens1 = vec![0.9, 0.7, 0.5, 0.3, 0.1];
        let lens2 = vec![0.85, 0.65, 0.45, 0.25, 0.15];
        let lens3 = vec![0.88, 0.72, 0.48, 0.28, 0.12];
        let result = cronbach_analysis(&[lens1, lens2, lens3]);
        assert!(result.alpha > 0.90, "correlated lenses should give high α: {}", result.alpha);
        assert_eq!(result.n_lenses, 3);
        assert_eq!(result.n_pairs, 5);
        eprintln!("{}", result.summary());
    }

    #[test]
    fn quorum_on_real_tables() {
        // Use the baked Jina, BGE-M3, Reranker tables
        use crate::jina_lens::JINA_HDR_TABLE;
        use crate::bge_m3_lens::BGE_M3_HDR_TABLE;
        use crate::reranker_lens::RERANKER_HDR_TABLE;

        let scores = variance_agreement_scores(
            &[JINA_HDR_TABLE.as_slice(), BGE_M3_HDR_TABLE.as_slice(), RERANKER_HDR_TABLE.as_slice()],
            256,
        );
        assert_eq!(scores.len(), 256 * 256);

        // Count quorum levels
        let mut high = 0; let mut med = 0; let mut low = 0; let mut amb = 0;
        for &s in &scores {
            match QuorumLevel::from_score(s) {
                QuorumLevel::High => high += 1,
                QuorumLevel::Medium => med += 1,
                QuorumLevel::Low => low += 1,
                QuorumLevel::Ambiguous => amb += 1,
            }
        }
        eprintln!("Quorum on 3 baked lenses (256×256):");
        eprintln!("  High: {} ({:.1}%)", high, high as f32 / scores.len() as f32 * 100.0);
        eprintln!("  Medium: {} ({:.1}%)", med, med as f32 / scores.len() as f32 * 100.0);
        eprintln!("  Low: {} ({:.1}%)", low, low as f32 / scores.len() as f32 * 100.0);
        eprintln!("  Ambiguous: {} ({:.1}%)", amb, amb as f32 / scores.len() as f32 * 100.0);

        // Should have some distribution — not all one level
        assert!(high > 0 || med > 0, "should have some agreement");
    }
}
