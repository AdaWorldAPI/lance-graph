//! # template-equivalence — replay comparison
//!
//! A template is not promoted until it is **replayed** (§18.2) and its output is
//! judged **equivalent** to the original LLM trace (§18.4). This crate grades a
//! replay into one of five [`EquivalenceClass`]es and produces an
//! [`EquivalenceReport`].
//!
//! The structural comparisons are real: [`EquivalenceClass::Exact`] (identical
//! ranked items, claims, and source spans) and [`EquivalenceClass::RankOrder`]
//! (top item stable within a tolerance), plus the §7 promotion rule "no new
//! uncited claims". [`EquivalenceClass::Semantic`] (embedding cosine) is the
//! deferred piece and currently degrades to [`EquivalenceClass::Failure`] with
//! an explanatory note — never a fabricated pass.
#![forbid(unsafe_code)]

/// The five equivalence categories (§7), strongest to weakest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EquivalenceClass {
    Exact,
    Semantic,
    RankOrder,
    Tolerance,
    Failure,
}

/// What a single run produced, normalized for comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayObservation {
    /// Ordered ranked output (e.g. ranked source ids).
    pub ranked_items: Vec<String>,
    /// Claims asserted by the run.
    pub claims: Vec<String>,
    /// `(source_id, start, end)` provenance for the claims.
    pub source_spans: Vec<(String, usize, usize)>,
    pub confidence: f32,
}

/// Thresholds for promotion (§7).
#[derive(Debug, Clone)]
pub struct EquivalenceConfig {
    /// Max rank distance the top item may move and still count as RankOrder.
    pub rank_tolerance: usize,
    /// Max allowed |Δconfidence|.
    pub confidence_delta: f32,
    /// Reject if the replay introduces a claim the original did not have.
    pub require_no_new_claims: bool,
}

impl Default for EquivalenceConfig {
    fn default() -> Self {
        Self { rank_tolerance: 1, confidence_delta: 0.1, require_no_new_claims: true }
    }
}

/// The graded result.
#[derive(Debug, Clone, PartialEq)]
pub struct EquivalenceReport {
    pub class: EquivalenceClass,
    pub score: f32,
    pub notes: Vec<String>,
}

impl EquivalenceReport {
    /// §18.4: a report clears the merge gate only if it is not `Failure`.
    pub fn passes(&self) -> bool {
        self.class != EquivalenceClass::Failure
    }
}

/// The comparison surface.
pub trait EquivalenceChecker {
    fn compare(&self, original: &ReplayObservation, replay: &ReplayObservation, cfg: &EquivalenceConfig) -> EquivalenceReport;
}

/// The default checker: real Exact + RankOrder + no-new-claims logic; Semantic
/// is deferred (returns Failure with a note rather than guessing).
#[derive(Debug, Default, Clone, Copy)]
pub struct StructuralChecker;

impl EquivalenceChecker for StructuralChecker {
    fn compare(&self, original: &ReplayObservation, replay: &ReplayObservation, cfg: &EquivalenceConfig) -> EquivalenceReport {
        let mut notes = Vec::new();

        // §7 promotion rule: no new uncited claims.
        if cfg.require_no_new_claims {
            for c in &replay.claims {
                if !original.claims.contains(c) {
                    notes.push(format!("replay introduced new claim: {c}"));
                    return EquivalenceReport { class: EquivalenceClass::Failure, score: 0.0, notes };
                }
            }
        }

        if (original.confidence - replay.confidence).abs() > cfg.confidence_delta {
            notes.push("confidence delta exceeds threshold".into());
            return EquivalenceReport { class: EquivalenceClass::Failure, score: 0.0, notes };
        }

        // §18.3 (no source span → no claim): a replay that keeps claims but drops
        // the provenance backing them must not pass — not even as RankOrder. Without
        // this guard a stable top-ranked item would let an uncited replay through the
        // promotion gate (the original had spans; the replay lost them).
        if !replay.claims.is_empty() && replay.source_spans.is_empty() && !original.source_spans.is_empty() {
            notes.push("replay kept claims but dropped all source spans (§18: no source span → no claim)".into());
            return EquivalenceReport { class: EquivalenceClass::Failure, score: 0.0, notes };
        }

        // Exact: identical ranked items, claims, and spans.
        if original.ranked_items == replay.ranked_items
            && original.claims == replay.claims
            && original.source_spans == replay.source_spans
        {
            return EquivalenceReport { class: EquivalenceClass::Exact, score: 1.0, notes };
        }

        // RankOrder: the top item is stable within tolerance.
        if let (Some(top), Some(pos)) = (original.ranked_items.first(), replay.ranked_items.iter().position(|x| Some(x) == original.ranked_items.first())) {
            if pos <= cfg.rank_tolerance {
                notes.push(format!("top item `{top}` moved to rank {pos} (≤ tolerance {})", cfg.rank_tolerance));
                let score = 1.0 - (pos as f32 / (replay.ranked_items.len().max(1) as f32));
                return EquivalenceReport { class: EquivalenceClass::RankOrder, score, notes };
            }
        }

        // Semantic comparison is the deferred piece — never fabricate a pass.
        notes.push("structural mismatch; semantic equivalence not implemented".into());
        EquivalenceReport { class: EquivalenceClass::Failure, score: 0.0, notes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(items: &[&str], claims: &[&str], conf: f32) -> ReplayObservation {
        ReplayObservation {
            ranked_items: items.iter().map(|s| s.to_string()).collect(),
            claims: claims.iter().map(|s| s.to_string()).collect(),
            source_spans: vec![],
            confidence: conf,
        }
    }

    #[test]
    fn identical_runs_are_exact() {
        let a = obs(&["s1", "s2"], &["c1"], 0.9);
        let r = StructuralChecker.compare(&a, &a.clone(), &EquivalenceConfig::default());
        assert_eq!(r.class, EquivalenceClass::Exact);
        assert!(r.passes());
    }

    #[test]
    fn top_item_within_tolerance_is_rankorder() {
        let a = obs(&["s1", "s2", "s3"], &["c1"], 0.9);
        let b = obs(&["s2", "s1", "s3"], &["c1"], 0.9);
        let r = StructuralChecker.compare(&a, &b, &EquivalenceConfig::default());
        assert_eq!(r.class, EquivalenceClass::RankOrder);
    }

    #[test]
    fn new_uncited_claim_fails() {
        let a = obs(&["s1"], &["c1"], 0.9);
        let b = obs(&["s1"], &["c1", "c2"], 0.9);
        let r = StructuralChecker.compare(&a, &b, &EquivalenceConfig::default());
        assert_eq!(r.class, EquivalenceClass::Failure);
        assert!(!r.passes());
    }

    #[test]
    fn rankorder_with_dropped_spans_fails_not_passes() {
        // Same top item (would be RankOrder), but replay lost the source spans
        // that backed its claim — must fail, not pass.
        let mut a = obs(&["s1", "s2"], &["c1"], 0.9);
        a.source_spans = vec![("doc".into(), 0, 5)];
        let b = obs(&["s2", "s1"], &["c1"], 0.9); // empty source_spans
        let r = StructuralChecker.compare(&a, &b, &EquivalenceConfig::default());
        assert_eq!(r.class, EquivalenceClass::Failure);
        assert!(!r.passes());
    }

    #[test]
    fn confidence_drift_fails() {
        let a = obs(&["s1"], &["c1"], 0.9);
        let b = obs(&["s1"], &["c1"], 0.5);
        let r = StructuralChecker.compare(&a, &b, &EquivalenceConfig::default());
        assert_eq!(r.class, EquivalenceClass::Failure);
    }
}
