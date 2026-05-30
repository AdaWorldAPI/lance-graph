// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The proposer output carrier — `CandidateRule` and the `Proposer` trait.
//!
//! This is the shape every Stage-A proposer (Aerial+, the pair-stats trunk,
//! the Python IPC fan-in) emits. It is the **local mirror** of the planned
//! `lance-graph-contract::CandidateRule` (D-ARM-2) — *not yet field-frozen*:
//! the planned carrier pairs the rule with a `WindowMetadata`, whereas this
//! local one carries a bare `n: u32`. When D-ARM-2 lands, adopt it by
//! `pub use` re-export — the determinism firewall forbids depending on
//! `lance-graph`, NOT on the zero-dep `lance-graph-contract`, so this crate
//! can path-dep the contract and stay zero-dep-from-the-spine. Until then
//! treat this as the migration seam, not the canonical type (TD-ARM-CARRIER-FORK).

/// One `(feature, category)` atom — the unit an antecedent / consequent is
/// built from.
///
/// `feature` indexes a column in the [`crate::FeatureSpec`]; `category`
/// indexes a value within that feature's category list. Both are small
/// integers so a rule is cheap to compare, hash, and serialise.
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

/// A mined association rule `antecedent → consequent` with ARM statistics.
///
/// The local proposer carrier — mirrors the planned
/// `lance-graph-contract::CandidateRule` (D-ARM-2) but does **not** yet match
/// it field-for-field (D-ARM-2 pairs the rule with `WindowMetadata`; this
/// carries a bare `n`). `support` and `confidence` are the classical ARM
/// quantities (paper §2):
///
/// - `support`     = |{rows ⊇ antecedent ∪ consequent}| / n
/// - `confidence`  = |{rows ⊇ antecedent ∪ consequent}| / |{rows ⊇ antecedent}|
///
/// `n` is the window size the statistics were measured over — it is what
/// turns `support` into an evidential count for the NARS confidence mapping
/// in [`crate::translator::arm_to_nars`].
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateRule {
    /// Left-hand side items (all must hold). Sorted, at most one per feature.
    pub antecedent: Vec<Item>,
    /// Right-hand side items. Aerial+ emits exactly one (single-consequent
    /// rules); the `Vec` keeps shape-parity with the planned contract.
    pub consequent: Vec<Item>,
    /// `support ∈ [0, 1]` — co-occurrence fraction over the window.
    pub support: f32,
    /// `confidence ∈ [0, 1]` — `P(consequent | antecedent)`.
    pub confidence: f32,
    /// Window size the statistics were measured over.
    pub n: u32,
}

impl CandidateRule {
    /// Evidential mass: `support × n`, the count of rows supporting the rule.
    /// Drives the NARS confidence term `c = m / (m + k)`.
    #[must_use]
    pub fn evidence(&self) -> f32 {
        self.support * self.n as f32
    }

    /// A rule survives the classical ARM gate iff it clears both the minimum
    /// support and minimum confidence floors.
    ///
    /// **Not implemented yet:** the Jirak-bound significance floor
    /// (`I-NOISE-FLOOR-JIRAK`, deliverable D-ARM-7) is the *mandatory* Stage-A
    /// gate per the plan, but it does **not** exist in this crate today
    /// (`jirak` appears nowhere; D-ARM-7 is Queued). Until it lands, `passes()`
    /// is the ONLY gate, and this proposer MUST NOT be wired to a live
    /// `SpoStore`: the classical floor alone leaks thin-but-frequent rules past
    /// the substrate noise floor (plan §11.1 — "the substrate calcifies on
    /// noise"). Tracked as ISSUE ARM-JIRAK-FLOOR.
    #[must_use]
    pub fn passes(&self, min_support: f32, min_confidence: f32) -> bool {
        self.support >= min_support && self.confidence >= min_confidence
    }
}

/// A Stage-A proposer: something that emits a batch of candidate rules.
///
/// Mirrors the planned `lance-graph-contract::Proposer` (D-ARM-2) so the
/// discovery crate is dependency-injectable: the hypothesis-test stage takes
/// `&mut dyn Proposer` and does not care whether the rules came from the
/// deterministic pair-stats trunk or the Aerial+ fan-in.
pub trait Proposer {
    /// Produce the next batch of candidate rules. Returning an empty `Vec`
    /// signals the proposer is exhausted for now (e.g. the window closed and
    /// no more rules cleared the gate).
    fn next_batch(&mut self) -> Vec<CandidateRule>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_is_support_times_n() {
        let r = CandidateRule {
            antecedent: vec![Item::new(0, 1)],
            consequent: vec![Item::new(1, 0)],
            support: 0.25,
            confidence: 0.9,
            n: 1000,
        };
        assert!((r.evidence() - 250.0).abs() < 1e-3);
    }

    #[test]
    fn passes_gates_on_both_floors() {
        let r = CandidateRule {
            antecedent: vec![Item::new(0, 0)],
            consequent: vec![Item::new(1, 1)],
            support: 0.05,
            confidence: 0.8,
            n: 200,
        };
        assert!(r.passes(0.01, 0.5));
        assert!(!r.passes(0.10, 0.5), "support floor not met");
        assert!(!r.passes(0.01, 0.9), "confidence floor not met");
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
