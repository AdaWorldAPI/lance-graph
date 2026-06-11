// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The proposer output carrier — `CandidateRule` and the `Proposer` trait.
//!
//! **Float-free.** A rule carries integer evidence counts, not `f32`
//! statistics. Support and confidence are rationals (`count / count`); they
//! are compared as integers (parts-per-million, cross-multiplied) so the
//! discovery decision path never touches a float. `f32` appears only at the
//! explicit edge accessors that feed the downstream `f32` contracts
//! (`spo::truth::TruthValue`, `ruff_spo_triplet::Triple`), and the canonical
//! truth wire is the quantized `CausalEdge64` form — see
//! [`crate::translator::arm_to_truth_u8`].
//!
//! This is the shape every Stage-A proposer (the codebook-probe Aerial+
//! backend, the pair-stats trunk) emits. It is the **local mirror** of the
//! planned `lance-graph-contract::CandidateRule` (D-ARM-2) — *not yet
//! field-frozen*; adopt by `pub use` re-export when the contract carrier
//! lands (the determinism firewall forbids depending on `lance-graph`, NOT on
//! the zero-dep `lance-graph-contract`). Until then this is the migration
//! seam, not the canonical type (TD-ARM-CARRIER-FORK).

/// One `(feature, category)` atom — the unit an antecedent / consequent is
/// built from. `feature` indexes a [`crate::FeatureSpec`] column; `category`
/// indexes a value within it. Both are small integers — a codebook code, not
/// a vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Item {
    /// Feature (column) index into the [`crate::FeatureSpec`].
    pub feature: u32,
    /// Category (value) index within the feature.
    pub category: u32,
}

impl Item {
    /// Construct an item.
    #[must_use]
    pub const fn new(feature: u32, category: u32) -> Self {
        Self { feature, category }
    }
}

/// Fixed-point scale for support/confidence comparison (parts per million).
/// A threshold of "1%" is `10_000`; "50%" is `500_000`. Integer throughout.
pub const PPM: u64 = 1_000_000;

/// A mined association rule `antecedent → consequent` with **integer**
/// ARM evidence.
///
/// Mirrors the planned `lance-graph-contract::CandidateRule` (D-ARM-2) but
/// does not yet match it field-for-field (D-ARM-2 pairs the rule with
/// `WindowMetadata`; this carries a bare `window`). The classical ARM
/// quantities (paper §2) are derived from the three counts:
///
/// - `support     = cooccur / window`
/// - `confidence  = cooccur / antecedent_count`  (= `P(consequent | antecedent)`)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateRule {
    /// Left-hand side items (all must hold). Sorted, at most one per feature.
    pub antecedent: Vec<Item>,
    /// Right-hand side items. The codebook probe emits exactly one; the `Vec`
    /// keeps shape-parity with the planned contract.
    pub consequent: Vec<Item>,
    /// `|{rows ⊇ antecedent ∪ consequent}|` — the co-occurrence count, and the
    /// NARS evidential mass `m`.
    pub cooccur: u32,
    /// `|{rows ⊇ antecedent}|` — the denominator of confidence.
    pub antecedent_count: u32,
    /// Window size `n` the counts were measured over.
    pub window: u32,
}

impl CandidateRule {
    /// Evidential mass `m = cooccur` — the count of rows supporting the rule.
    /// Drives the NARS confidence term `c = m / (m + k)`. Integer.
    #[must_use]
    pub fn evidence(&self) -> u32 {
        self.cooccur
    }

    /// Support in parts-per-million: `cooccur · 1e6 / window`. Integer.
    #[must_use]
    pub fn support_ppm(&self) -> u32 {
        if self.window == 0 {
            return 0;
        }
        ((self.cooccur as u64 * PPM) / self.window as u64) as u32
    }

    /// Confidence in parts-per-million: `cooccur · 1e6 / antecedent_count`.
    /// Integer. Returns 0 when the antecedent never occurs.
    #[must_use]
    pub fn confidence_ppm(&self) -> u32 {
        if self.antecedent_count == 0 {
            return 0;
        }
        ((self.cooccur as u64 * PPM) / self.antecedent_count as u64) as u32
    }

    /// The classical ARM gate, compared entirely in integers (ppm).
    /// This alone is NOT sufficient for a SpoStore-bound rule: the Jirak-bound
    /// significance floor (`I-NOISE-FLOOR-JIRAK`, D-ARM-7) is the stricter
    /// Stage-A gate — see [`Self::passes_stage_a`] /
    /// [`crate::jirak::jirak_floor_ppm`]. D-ARM-5 MUST route through the
    /// Stage-A gate, never through this classical gate alone (ISSUE
    /// ARM-JIRAK-FLOOR, resolved by D-ARM-7).
    #[must_use]
    pub fn passes(&self, min_support_ppm: u32, min_confidence_ppm: u32) -> bool {
        self.support_ppm() >= min_support_ppm && self.confidence_ppm() >= min_confidence_ppm
    }

    /// The Stage-A gate: the classical floors AND the Jirak-bound significance
    /// floor derived from this rule's own `window` (D-ARM-7,
    /// `I-NOISE-FLOOR-JIRAK`). A rule survives Stage A only if its observed
    /// support also clears `jirak_floor_ppm(window, p_moment, alpha)` — the
    /// weak-dependence noise floor the classical thresholds know nothing
    /// about. Defaults: [`crate::jirak::DEFAULT_P_MOMENT`] /
    /// [`crate::jirak::DEFAULT_ALPHA`].
    ///
    /// `f32` appears only in the one-time floor derivation (the crate's float
    /// edge); the comparison itself is integer ppm. For batch filtering,
    /// derive the floor once per window instead
    /// ([`crate::aerial::extract_rules_stage_a`]).
    #[must_use]
    pub fn passes_stage_a(
        &self,
        min_support_ppm: u32,
        min_confidence_ppm: u32,
        p_moment: f32,
        confidence_alpha: f32,
    ) -> bool {
        self.passes(min_support_ppm, min_confidence_ppm)
            && self.support_ppm()
                >= crate::jirak::jirak_floor_ppm(self.window, p_moment, confidence_alpha)
    }

    /// `f32` support — **edge convenience only**, for the downstream `f32`
    /// `TruthValue`/`Triple` contracts. Not used in any decision.
    #[must_use]
    pub fn support_f32(&self) -> f32 {
        self.support_ppm() as f32 / PPM as f32
    }

    /// `f32` confidence — edge convenience only (see [`Self::support_f32`]).
    #[must_use]
    pub fn confidence_f32(&self) -> f32 {
        self.confidence_ppm() as f32 / PPM as f32
    }
}

/// A Stage-A proposer: something that emits a batch of candidate rules.
/// Mirrors the planned `lance-graph-contract::Proposer` (D-ARM-2).
pub trait Proposer {
    /// Produce the next batch of candidate rules. An empty `Vec` signals the
    /// proposer is exhausted for now.
    fn next_batch(&mut self) -> Vec<CandidateRule>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(cooccur: u32, antecedent_count: u32, window: u32) -> CandidateRule {
        CandidateRule {
            antecedent: vec![Item::new(0, 1)],
            consequent: vec![Item::new(1, 0)],
            cooccur,
            antecedent_count,
            window,
        }
    }

    #[test]
    fn evidence_is_the_cooccur_count() {
        assert_eq!(rule(250, 300, 1000).evidence(), 250);
    }

    #[test]
    fn support_and_confidence_are_integer_ppm() {
        let r = rule(250, 300, 1000);
        assert_eq!(r.support_ppm(), 250_000); // 25%
        assert_eq!(r.confidence_ppm(), 833_333); // 250/300 ≈ 0.8333
    }

    #[test]
    fn passes_gates_on_both_floors_in_ppm() {
        let r = rule(100, 125, 2000); // support 5%, confidence 80%
        assert!(r.passes(10_000, 500_000));
        assert!(!r.passes(100_000, 500_000), "support floor not met");
        assert!(!r.passes(10_000, 900_000), "confidence floor not met");
    }

    #[test]
    fn zero_antecedent_is_zero_confidence_not_a_panic() {
        let r = rule(0, 0, 1000);
        assert_eq!(r.confidence_ppm(), 0);
        assert!(!r.passes(0, 1));
    }

    #[test]
    fn stage_a_prunes_classically_passing_but_insignificant_support() {
        // 6% support at n = 600: clears the classical 5% floor, but the
        // Jirak floor at n = 600 (p = 3, α = 0.05) is ≈ 6.72% — pruned.
        let weak = rule(36, 40, 600); // support 60_000 ppm, confidence 90%
        assert!(weak.passes(50_000, 700_000), "classical gate admits it");
        assert!(
            !weak.passes_stage_a(50_000, 700_000, 3.0, 0.05),
            "Stage A must prune support below the Jirak floor"
        );

        // The SAME proportions at n = 10_000 (floor ≈ 1.64%) are significant.
        let strong = rule(600, 660, 10_000); // support 60_000 ppm, confidence ≈ 90.9%
        assert!(strong.passes_stage_a(50_000, 700_000, 3.0, 0.05));
    }

    #[test]
    fn stage_a_never_admits_what_the_classical_gate_rejects() {
        // Stage A = classical AND jirak: a classical rejection stays rejected.
        let r = rule(600, 660, 10_000);
        assert!(
            !r.passes_stage_a(100_000, 700_000, 3.0, 0.05),
            "support floor"
        );
        assert!(
            !r.passes_stage_a(50_000, 950_000, 3.0, 0.05),
            "confidence floor"
        );
    }

    #[test]
    fn item_ordering_is_feature_then_category() {
        let mut items = vec![Item::new(1, 0), Item::new(0, 2), Item::new(0, 1)];
        items.sort();
        assert_eq!(
            items,
            vec![Item::new(0, 1), Item::new(0, 2), Item::new(1, 0)]
        );
    }
}
