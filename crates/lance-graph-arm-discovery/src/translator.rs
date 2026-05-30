// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stage B — translate an ARM [`CandidateRule`] into NARS truth + an SPO
//! triple carrier.
//!
//! The mapping is verbatim from Aerial+ §2 / §3.3 and the plan's FINDING:
//!
//! - ARM **confidence** = `P(Y|X)` → NARS **frequency** `f`
//! - ARM **support × n** (evidential mass `m`) → NARS **confidence**
//!   `c = m / (m + k)`, NAL-9 personality constant `k` (default `1.0`)
//!
//! The resulting `(f, c)` is the exact pair consumed by
//! `lance_graph::graph::spo::TruthValue::new(f, c)` and carried by
//! `ruff_spo_triplet::Triple { f, c }`. That is the whole point: a rule mined
//! from runtime data and a triple extracted from static source land in the
//! SPO store on the *same* truth scale.

use crate::rule::{CandidateRule, Item};

/// NAL-9 default personality constant for the support → confidence mapping.
/// Larger `k` ⇒ more evidence needed before confidence approaches 1.
pub const NARS_PERSONALITY_K: f32 = 1.0;

/// A NARS truth value `(frequency, confidence)` — the translator's output.
///
/// Deliberately *not* a re-implementation of the SPO store's `TruthValue`:
/// it is the `(f, c)` pair you feed to `TruthValue::new(f, c)`. Keeping it a
/// distinct, tiny carrier honours `E-SOA-IS-THE-ONLY` (no parallel truth
/// store) while keeping this crate zero-dep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NarsTruth {
    /// NARS frequency `f ∈ [0, 1]`.
    pub frequency: f32,
    /// NARS confidence `c ∈ [0, 1)`.
    pub confidence: f32,
}

impl NarsTruth {
    /// NARS expectation `e = c·(f − 0.5) + 0.5` — matches
    /// `lance_graph::graph::spo::TruthValue::expectation` exactly.
    #[must_use]
    pub fn expectation(&self) -> f32 {
        self.confidence * (self.frequency - 0.5) + 0.5
    }
}

/// Translate a candidate rule's ARM statistics into NARS truth.
///
/// `k` is the NARS personality constant ([`NARS_PERSONALITY_K`] by default,
/// per-feed configurable per OQ-ARM-3).
#[must_use]
pub fn arm_to_nars(rule: &CandidateRule, k: f32) -> NarsTruth {
    // ARM confidence is P(Y|X) — directly the NARS frequency.
    let frequency = rule.confidence.clamp(0.0, 1.0);
    // Evidential mass m = support × n; c = m / (m + k) → 1 as evidence grows.
    let m = rule.evidence().max(0.0);
    let confidence = m / (m + k);
    NarsTruth {
        frequency,
        confidence,
    }
}

/// Projects rule items into SPO IRIs for a particular feed/domain.
///
/// An Odoo feed projects `(model, predicate, value)`; an MedCare feed
/// projects differently. The projector is the only domain-specific seam in
/// Stage B — everything else is feed-agnostic.
pub trait FeedProjector {
    /// IRI for the subject built from the antecedent items.
    fn subject(&self, antecedent: &[Item]) -> String;
    /// The predicate IRI for a discovered implication.
    fn predicate(&self) -> String;
    /// IRI for the object built from the consequent items.
    fn object(&self, consequent: &[Item]) -> String;
}

/// One SPO triple with NARS truth — shape-compatible with
/// `ruff_spo_triplet::Triple` and the SPO store's ndjson loader (same
/// `{s,p,o,f,c}` fields; see [`crate::ndjson`] for the predicate-vocabulary
/// caveat that gates the `ruff` loader on D-ARM-SYN-1).
///
/// `(s, p, o)` is the identity; `(f, c)` is the data-derived truth. The
/// `origin` byte records that this triple came from the ARM-discovery
/// proposer (bits per `discovery_origin` in the plan §7) — it is metadata,
/// dropped on the ndjson wire, not part of the triple identity.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateTriple {
    /// Subject IRI.
    pub s: String,
    /// Predicate IRI.
    pub p: String,
    /// Object IRI.
    pub o: String,
    /// NARS frequency.
    pub f: f32,
    /// NARS confidence.
    pub c: f32,
}

impl CandidateTriple {
    /// Build a triple from a candidate rule, a projector, and a NARS `k`.
    #[must_use]
    pub fn from_rule(
        rule: &CandidateRule,
        projector: &dyn FeedProjector,
        k: f32,
    ) -> Self {
        let truth = arm_to_nars(rule, k);
        Self {
            s: projector.subject(&rule.antecedent),
            p: projector.predicate(),
            o: projector.object(&rule.consequent),
            f: truth.frequency,
            c: truth.confidence,
        }
    }
}

/// A minimal projector that renders items as `feat<i>=cat<j>` and joins an
/// antecedent with `&`. Useful for tests and ndjson smoke checks; real feeds
/// supply a domain projector that emits proper namespaced IRIs.
#[derive(Debug, Clone)]
pub struct DebugProjector {
    /// Predicate IRI to stamp on every emitted implication.
    pub predicate: String,
}

impl Default for DebugProjector {
    fn default() -> Self {
        Self {
            // Not in `ruff_spo_triplet`'s *current* closed vocabulary — see the
            // synergy doc: flowing ARM rules through that loader needs an
            // `implies` predicate added there first.
            predicate: "implies".to_string(),
        }
    }
}

fn render_items(items: &[Item]) -> String {
    let mut parts: Vec<Item> = items.to_vec();
    parts.sort();
    parts
        .iter()
        .map(|it| format!("feat{}=cat{}", it.feature, it.category))
        .collect::<Vec<_>>()
        .join("&")
}

impl FeedProjector for DebugProjector {
    fn subject(&self, antecedent: &[Item]) -> String {
        format!("arm:{}", render_items(antecedent))
    }
    fn predicate(&self) -> String {
        self.predicate.clone()
    }
    fn object(&self, consequent: &[Item]) -> String {
        format!("arm:{}", render_items(consequent))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(support: f32, confidence: f32, n: u32) -> CandidateRule {
        CandidateRule {
            antecedent: vec![Item::new(0, 1)],
            consequent: vec![Item::new(1, 0)],
            support,
            confidence,
            n,
        }
    }

    #[test]
    fn frequency_is_arm_confidence() {
        let t = arm_to_nars(&rule(0.2, 0.83, 1000), NARS_PERSONALITY_K);
        assert!((t.frequency - 0.83).abs() < 1e-6);
    }

    #[test]
    fn confidence_grows_with_evidence() {
        // m = support × n = 0.2 × 1000 = 200; c = 200/201 ≈ 0.995
        let strong = arm_to_nars(&rule(0.2, 0.83, 1000), NARS_PERSONALITY_K);
        // m = 0.2 × 10 = 2; c = 2/3 ≈ 0.667
        let weak = arm_to_nars(&rule(0.2, 0.83, 10), NARS_PERSONALITY_K);
        assert!(strong.confidence > weak.confidence);
        assert!((strong.confidence - 200.0 / 201.0).abs() < 1e-4);
        assert!((weak.confidence - 2.0 / 3.0).abs() < 1e-4);
    }

    #[test]
    fn larger_k_demands_more_evidence() {
        let k1 = arm_to_nars(&rule(0.1, 0.7, 50), 1.0).confidence;
        let k10 = arm_to_nars(&rule(0.1, 0.7, 50), 10.0).confidence;
        assert!(k10 < k1, "larger k lowers confidence for the same evidence");
    }

    #[test]
    fn expectation_matches_spo_formula() {
        let t = NarsTruth {
            frequency: 0.9,
            confidence: 0.8,
        };
        // 0.8 * (0.9 - 0.5) + 0.5 = 0.82
        assert!((t.expectation() - 0.82).abs() < 1e-6);
    }

    #[test]
    fn triple_projection_and_truth() {
        let r = rule(0.25, 0.9, 400); // m = 100; c = 100/101
        let t = CandidateTriple::from_rule(&r, &DebugProjector::default(), NARS_PERSONALITY_K);
        assert_eq!(t.s, "arm:feat0=cat1");
        assert_eq!(t.p, "implies");
        assert_eq!(t.o, "arm:feat1=cat0");
        assert!((t.f - 0.9).abs() < 1e-6);
        assert!((t.c - 100.0 / 101.0).abs() < 1e-4);
    }
}
