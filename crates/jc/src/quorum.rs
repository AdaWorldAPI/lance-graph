//! Lens quorum — **per-pair agreement across `k` lens tables** and the
//! **quorum level** it maps to, plus a per-subject Cronbach report.
//!
//! NOT a pillar. Lifted out of the thinking-engine lab crate under D-TEH-3
//! (`thinking-engine-harvest-closure-v1` §1d) per
//! `E-JC-IS-THE-HOME-OF-ALL-CALIBRATED-MATH-1`; the α estimate itself is
//! [`crate::reliability::cronbach_alpha`] — this module never recomputes it.
//!
//! # Two different questions
//!
//! Cronbach α asks about the WHOLE corpus: "do these `k` lenses behave as one
//! scale over all `n` pairs?" The quorum score asks about ONE cell: "how far
//! apart are the `k` lenses on THIS pair?" It is a normalised dispersion,
//! `1 − σ/σ_max`, where `σ_max = 255/2` is the largest population standard
//! deviation a set of `u8` values can have (half at 0, half at 255). It is
//! NOT an α per pair — α is undefined on a single subject — and the lifted
//! source said so; the name here says so too.
//!
//! # Quorum bands
//!
//! [`QuorumLevel::from_score`] cuts the `u8` score at 230 / 179 / 128 — the
//! α bands 0.90 / 0.70 / 0.50 scaled to 255 and rounded up (229.5 → 230,
//! 178.5 → 179, 127.5 → 128). A pair below `Medium` is one the fast cascade
//! should not decide alone.

use crate::reliability::cronbach_alpha;

/// Largest population variance of a set of `u8` values: half at 0, half at
/// 255 gives `σ² = (255/2)²`.
pub const U8_MAX_VARIANCE: f64 = 255.0 * 255.0 / 4.0;

/// Per-pair agreement across `k` square `u8` tables of side `n`.
///
/// `tables[t][i * n + j]` is lens `t`'s value for pair `(i, j)`. Returns an
/// `n × n` score matrix in `0..=255` — `255` = the lenses coincide on that
/// pair, `0` = maximal disagreement — symmetric, with `255` on the diagonal
/// (a table's self-distance is not a measurement to disagree about).
///
/// Returns `None` for fewer than two tables, `n < 2`, or any table whose
/// length is not `n * n`.
///
/// ```
/// use jc::quorum::pairwise_agreement_u8;
/// let a = [255u8, 100, 100, 255];
/// let b = [255u8, 100, 100, 255];
/// let s = pairwise_agreement_u8(&[&a, &b], 2).unwrap();
/// assert_eq!(s, vec![255, 255, 255, 255]);
/// ```
pub fn pairwise_agreement_u8(tables: &[&[u8]], n: usize) -> Option<Vec<u8>> {
    let k = tables.len();
    if k < 2 || n < 2 || tables.iter().any(|t| t.len() != n * n) {
        return None;
    }
    let kf = k as f64;
    let mut scores = vec![0u8; n * n];
    for i in 0..n {
        scores[i * n + i] = 255;
        for j in (i + 1)..n {
            let idx = i * n + j;
            let mean = tables.iter().map(|t| f64::from(t[idx])).sum::<f64>() / kf;
            let var = tables
                .iter()
                .map(|t| {
                    let d = f64::from(t[idx]) - mean;
                    d * d
                })
                .sum::<f64>()
                / kf;
            let agreement = 1.0 - (var / U8_MAX_VARIANCE).sqrt();
            // `agreement` is in [0, 1] by construction (var ≤ U8_MAX_VARIANCE),
            // so the rounded product is in 0..=255 and the cast cannot
            // truncate; the clamp is belt-and-braces against a rounding tick.
            let score = (agreement * 255.0).round().clamp(0.0, 255.0) as u8;
            scores[idx] = score;
            scores[j * n + i] = score;
        }
    }
    Some(scores)
}

/// How much the lenses agree on a pair, in the bands the cascade acts on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuorumLevel {
    /// Score ≥ 230 (α-equivalent > 0.90): the lenses coincide. Encode
    /// confidently.
    High,
    /// Score 179..=229 (0.70–0.90): mostly agree. Encode, carry a
    /// boundary-risk mark.
    Medium,
    /// Score 128..=178 (0.50–0.70): the lenses see different things. Route to
    /// LEAF validation.
    Low,
    /// Score < 128 (< 0.50): no agreement. The pair is genuinely ambiguous.
    Ambiguous,
}

impl QuorumLevel {
    /// Band floors, as `u8` scores.
    pub const HIGH_FLOOR: u8 = 230;
    /// See [`Self::HIGH_FLOOR`].
    pub const MEDIUM_FLOOR: u8 = 179;
    /// See [`Self::HIGH_FLOOR`].
    pub const LOW_FLOOR: u8 = 128;

    /// Classify a [`pairwise_agreement_u8`] score.
    pub fn from_score(score: u8) -> Self {
        if score >= Self::HIGH_FLOOR {
            Self::High
        } else if score >= Self::MEDIUM_FLOOR {
            Self::Medium
        } else if score >= Self::LOW_FLOOR {
            Self::Low
        } else {
            Self::Ambiguous
        }
    }

    /// Should this pair bypass the fast cascade and be validated at the leaf?
    pub fn needs_leaf_validation(self) -> bool {
        matches!(self, Self::Low | Self::Ambiguous)
    }
}

/// Cronbach α over the corpus plus the per-subject dispersion that says
/// WHICH pairs the lenses disagree on.
#[derive(Clone, Debug, PartialEq)]
pub struct CronbachReport {
    /// [`cronbach_alpha`] over all items and subjects.
    pub alpha: f64,
    /// Population variance across the `k` items, per subject.
    pub subject_variances: Vec<f64>,
    /// `mean + 1 σ` of `subject_variances`; a subject above it counts as a
    /// disagreement.
    pub disagreement_threshold: f64,
    /// Number of subjects whose across-item variance exceeds the threshold.
    pub disagreement_count: usize,
    /// `k`.
    pub n_items: usize,
    /// `n`.
    pub n_subjects: usize,
}

/// Compute α and the per-subject disagreement profile.
///
/// Same shape and degeneracy contract as [`cronbach_alpha`] (`items[i][s]` =
/// item `i`, subject `s`; `None` wherever α is undefined).
pub fn cronbach_report(items: &[Vec<f64>]) -> Option<CronbachReport> {
    let alpha = cronbach_alpha(items)?;
    let k = items.len();
    let n = items[0].len();
    let kf = k as f64;
    let subject_variances: Vec<f64> = (0..n)
        .map(|s| {
            let mean = items.iter().map(|it| it[s]).sum::<f64>() / kf;
            items
                .iter()
                .map(|it| {
                    let d = it[s] - mean;
                    d * d
                })
                .sum::<f64>()
                / kf
        })
        .collect();
    let nf = n as f64;
    let var_mean = subject_variances.iter().sum::<f64>() / nf;
    let var_sd = (subject_variances
        .iter()
        .map(|v| (v - var_mean) * (v - var_mean))
        .sum::<f64>()
        / nf)
        .sqrt();
    let disagreement_threshold = var_mean + var_sd;
    let disagreement_count = subject_variances
        .iter()
        .filter(|&&v| v > disagreement_threshold)
        .count();
    Some(CronbachReport {
        alpha,
        subject_variances,
        disagreement_threshold,
        disagreement_count,
        n_items: k,
        n_subjects: n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Anti-vacuity: the off-diagonal cells are asserted, not only the
    /// diagonal the function stamps unconditionally.
    #[test]
    fn coinciding_tables_score_255_everywhere() {
        let t = [255u8, 100, 40, 100, 255, 7, 40, 7, 255];
        let s = pairwise_agreement_u8(&[&t, &t, &t], 3).unwrap();
        assert!(s.iter().all(|&v| v == 255), "{s:?}");
    }

    /// Maximal disagreement (0 vs 255) scores 0; the score is symmetric and
    /// the diagonal stays 255. Disable: drop the `sqrt` → 0-vs-255 still
    /// scores 0 but a 100-vs-150 pair jumps from 205 to 245 (the third
    /// assertion).
    #[test]
    fn maximal_disagreement_scores_zero_and_a_moderate_one_is_scaled_by_sigma() {
        let a = [255u8, 0, 100, 0, 255, 0, 100, 0, 255];
        let b = [255u8, 255, 150, 255, 255, 0, 150, 0, 255];
        let s = pairwise_agreement_u8(&[&a, &b], 3).unwrap();
        assert_eq!(s[1], 0, "0 vs 255 is maximal disagreement");
        assert_eq!(s[1], s[3], "symmetric");
        // 100 vs 150: σ = 25, σ_max = 127.5 → 1 − 25/127.5 = 0.80392 → 205.
        assert_eq!(s[2], 205, "{s:?}");
        assert_eq!(s[5], 255, "0 vs 0 coincide");
        assert!(
            [0usize, 4, 8].iter().all(|&d| s[d] == 255),
            "diagonal must be 255"
        );
    }

    #[test]
    fn degenerate_inputs_return_none() {
        let t = [255u8, 1, 1, 255];
        assert_eq!(pairwise_agreement_u8(&[&t], 2), None); // k < 2
        assert_eq!(pairwise_agreement_u8(&[&t, &t], 1), None); // n < 2
        let short = [255u8, 1, 1];
        assert_eq!(pairwise_agreement_u8(&[&t, &short], 2), None); // ragged
    }

    /// Both sides of every band floor.
    #[test]
    fn quorum_bands_cut_exactly_at_their_floors() {
        use QuorumLevel::*;
        assert_eq!(QuorumLevel::from_score(255), High);
        assert_eq!(QuorumLevel::from_score(230), High);
        assert_eq!(QuorumLevel::from_score(229), Medium);
        assert_eq!(QuorumLevel::from_score(179), Medium);
        assert_eq!(QuorumLevel::from_score(178), Low);
        assert_eq!(QuorumLevel::from_score(128), Low);
        assert_eq!(QuorumLevel::from_score(127), Ambiguous);
        assert_eq!(QuorumLevel::from_score(0), Ambiguous);
        assert!(!High.needs_leaf_validation());
        assert!(!Medium.needs_leaf_validation());
        assert!(Low.needs_leaf_validation());
        assert!(Ambiguous.needs_leaf_validation());
    }

    /// The report's α IS `reliability::cronbach_alpha` (delegation, not a
    /// second formula), and one wild subject is the one flagged. Disable:
    /// compare against `var_mean` alone (drop the `+ σ`) → the mild subject
    /// at index 1 is flagged too and the count reads 2.
    #[test]
    fn report_flags_the_one_subject_the_items_disagree_on() {
        // 3 items × 5 subjects. Subject 3 is where item 2 breaks rank
        // (across-item variance ≈ 5.4); subject 1 is MILDLY disputed
        // (variance 2.0 — values 2 ± √3), sitting above the mean variance
        // (≈ 1.48) but below mean + σ (≈ 3.59); the other three subjects
        // are near-unanimous (≈ 0.002).
        let items = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.1, 3.732, 3.0, 4.1, 5.0],
            vec![1.0, 0.268, 3.1, 9.0, 5.1],
        ];
        let r = cronbach_report(&items).unwrap();
        assert_eq!(r.alpha, cronbach_alpha(&items).unwrap());
        assert_eq!((r.n_items, r.n_subjects), (3, 5));
        assert_eq!(r.subject_variances.len(), 5);
        let (argmax, _) = r
            .subject_variances
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        assert_eq!(argmax, 3);
        assert_eq!(r.disagreement_count, 1, "{:?}", r.subject_variances);
        // Anti-vacuity for the disable: the mild subject sits ABOVE the mean
        // but below mean + σ, so the σ term is what excludes it.
        let var_mean = r.subject_variances.iter().sum::<f64>() / 5.0;
        assert!(r.subject_variances[1] > var_mean);
        assert!(r.subject_variances[1] <= r.disagreement_threshold);
    }

    #[test]
    fn report_is_none_where_alpha_is_undefined() {
        assert_eq!(cronbach_report(&[vec![1.0, 2.0]]), None);
        let flat = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert_eq!(cronbach_report(&flat), None);
    }
}
