//! **Universal thinking atoms** — operator-ruled (2026-08-31, verbatim):
//! *"shannon, Mengenlehre etc sind universale denk atome."*
//!
//! The rule-7 test every entry here passes: it would mean exactly the same
//! in a chess engine. Shannon entropy of a candidate distribution, the NARS
//! expectation, the mean of a confidence set — none of them knows what a
//! candidate IS. What a consumer keeps at home is only the extraction (its
//! domain objects → these plain slices).
//!
//! The Mengenlehre tally (signed agreement/disagreement over inherited
//! properties, the 24×i4 reading) belongs here too and is NOT yet built —
//! it is gated on property edges existing (the causal mint); this module is
//! its landing site, recorded so it is not built elsewhere.

/// NARS expectation — `c·(f−0.5)+0.5`.
#[must_use]
pub fn expectation(frequency: f32, confidence: f32) -> f32 {
    confidence * (frequency - 0.5) + 0.5
}

/// Normalized Shannon entropy of a weight vector, in `[0, 1]`.
///
/// `H(p)/ln(n)` over `p_i = w_i / Σw`. Conventions, each a decision rather
/// than an accident: `None` for an empty vector (there is no distribution
/// to be uncertain about — the caller keeps its default); `Some(0.0)` for a
/// single weight (a degenerate distribution has zero uncertainty);
/// `Some(1.0)` when every weight is zero (nothing prefers anything —
/// indistinguishable from uniform). `0·ln 0 = 0` by the limit convention.
#[must_use]
pub fn normalized_entropy(weights: &[f32]) -> Option<f32> {
    match weights.len() {
        0 => None,
        1 => Some(0.0),
        n => {
            let sum: f32 = weights.iter().sum();
            if sum <= 0.0 {
                return Some(1.0);
            }
            let h: f32 = weights
                .iter()
                .map(|&w| {
                    let p = w / sum;
                    if p > 0.0 {
                        -p * p.ln()
                    } else {
                        0.0
                    }
                })
                .sum();
            Some((h / (n as f32).ln()).clamp(0.0, 1.0))
        }
    }
}

/// Mean of a confidence set; `None` when empty (no set, no mean — the
/// caller keeps its default rather than receiving an invented one).
#[must_use]
pub fn mean_confidence(confidences: &[f32]) -> Option<f32> {
    if confidences.is_empty() {
        return None;
    }
    Some(confidences.iter().sum::<f32>() / confidences.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-math falsifiers — each convention has the test that would catch
    /// its removal. (Migrated with the atom from its consumer-side origin.)
    #[test]
    fn normalized_entropy_hand_math() {
        assert_eq!(normalized_entropy(&[]), None, "no distribution, no value");
        assert_eq!(
            normalized_entropy(&[0.7]),
            Some(0.0),
            "degenerate = decided"
        );
        let flat = normalized_entropy(&[1.0, 1.0, 1.0, 1.0]).unwrap();
        assert!((flat - 1.0).abs() < 1e-6, "uniform is maximal: {flat}");
        let peaked = normalized_entropy(&[0.97, 0.01, 0.01, 0.01]).unwrap();
        assert!(peaked < flat && peaked > 0.0);
        assert_eq!(
            normalized_entropy(&[0.0, 0.0]),
            Some(1.0),
            "zero-sum = uniform"
        );
    }

    #[test]
    fn expectation_is_the_nars_form() {
        assert_eq!(expectation(0.5, 0.9), 0.5, "f=0.5 is the fixed point");
        assert!(expectation(1.0, 1.0) > 0.99);
        assert!(expectation(0.0, 1.0) < 0.01);
        assert_eq!(expectation(1.0, 0.0), 0.5, "no confidence, no movement");
    }

    #[test]
    fn mean_confidence_keeps_the_default_on_empty() {
        assert_eq!(mean_confidence(&[]), None);
        let m = mean_confidence(&[0.2, 0.4]).unwrap();
        assert!((m - 0.3).abs() < 1e-6);
    }
}
