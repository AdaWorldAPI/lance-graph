// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stage B — translate a [`CandidateRule`]'s integer evidence into NARS truth
//! + an SPO triple carrier.
//!
//! The mapping is verbatim from Aerial+ §2/§3.3, computed in **integers**:
//!
//! - ARM **confidence** = `cooccur / antecedent_count` → NARS **frequency**
//! - ARM **evidence** `m = cooccur` → NARS **confidence** `c = m / (m + k)`
//!
//! The canonical truth is [`TruthU8`] — `frequency`/`confidence` as `u8`,
//! which is exactly the `CausalEdge64` wire (`confidence_u8` + i4 mantissa),
//! float-free. [`NarsTruth`] / [`arm_to_nars`] are a thin `f32` **edge** that
//! exists only because the downstream `spo::truth::TruthValue` and
//! `ruff_spo_triplet::Triple` are themselves `f32`; nothing in the discovery
//! path consumes the `f32` form.

use crate::rule::{CandidateRule, Item};

/// NAL-9 default personality constant (integer). Larger `k` ⇒ more evidence
/// needed before confidence approaches saturation.
pub const NARS_PERSONALITY_K: u32 = 1;

/// Quantised NARS truth — the canonical, float-free wire form (mirrors the
/// `CausalEdge64` `confidence_u8` + i4 mantissa fields). `255` = 1.0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruthU8 {
    /// NARS frequency, `0..=255` (255 = 1.0).
    pub frequency: u8,
    /// NARS confidence, `0..=255`.
    pub confidence: u8,
}

/// Translate a candidate rule's integer evidence into quantised NARS truth.
/// `k` is the NARS personality constant ([`NARS_PERSONALITY_K`] default).
#[must_use]
pub fn arm_to_truth_u8(rule: &CandidateRule, k: u32) -> TruthU8 {
    // frequency = P(Y|X) = cooccur / antecedent_count
    let frequency = if rule.antecedent_count == 0 {
        128 // unknown ≈ 0.5
    } else {
        ((rule.cooccur as u64 * 255) / rule.antecedent_count as u64).min(255) as u8
    };
    // confidence = m / (m + k), m = cooccur (integer evidential mass)
    let m = rule.cooccur as u64;
    let denom = (m + k as u64).max(1);
    let confidence = ((m * 255) / denom) as u8;
    TruthU8 {
        frequency,
        confidence,
    }
}

/// An `f32` NARS truth — **edge convenience only** (see module docs). Derived
/// from [`TruthU8`]; never read inside the discovery path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NarsTruth {
    /// NARS frequency `f ∈ [0, 1]`.
    pub frequency: f32,
    /// NARS confidence `c ∈ [0, 1)`.
    pub confidence: f32,
}

impl NarsTruth {
    /// NARS expectation `e = c·(f − 0.5) + 0.5` — matches
    /// `spo::truth::TruthValue::expectation`.
    #[must_use]
    pub fn expectation(&self) -> f32 {
        self.confidence * (self.frequency - 0.5) + 0.5
    }
}

/// `f32` edge of [`arm_to_truth_u8`], for the downstream `f32` `TruthValue` /
/// `Triple` contracts.
#[must_use]
pub fn arm_to_nars(rule: &CandidateRule, k: u32) -> NarsTruth {
    let t = arm_to_truth_u8(rule, k);
    NarsTruth {
        frequency: t.frequency as f32 / 255.0,
        confidence: t.confidence as f32 / 255.0,
    }
}

/// Projects rule items into SPO IRIs for a particular feed/domain.
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
/// caveat that gates the `ruff` loader on D-ARM-SYN-1). The `f`/`c` floats are
/// the *serialization format* of that downstream contract, derived from the
/// canonical integer [`TruthU8`] at the wire edge.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateTriple {
    /// Subject IRI.
    pub s: String,
    /// Predicate IRI.
    pub p: String,
    /// Object IRI.
    pub o: String,
    /// NARS frequency (serialization edge).
    pub f: f32,
    /// NARS confidence (serialization edge).
    pub c: f32,
}

impl CandidateTriple {
    /// Build a triple from a candidate rule, a projector, and a NARS `k`.
    #[must_use]
    pub fn from_rule(rule: &CandidateRule, projector: &dyn FeedProjector, k: u32) -> Self {
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

/// A minimal projector rendering items as `feat<i>=cat<j>`, joined with `&`.
#[derive(Debug, Clone)]
pub struct DebugProjector {
    /// Predicate IRI to stamp on every emitted implication.
    pub predicate: String,
}

impl Default for DebugProjector {
    fn default() -> Self {
        Self {
            // Not in `ruff_spo_triplet`'s current closed vocabulary — see the
            // synergy doc: the `ruff` loader needs an `implies` predicate added
            // (D-ARM-SYN-1) before ARM rules flow through it.
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
    fn frequency_is_quantised_arm_confidence() {
        // confidence = 249/300 ≈ 0.83 → 0.83×255 ≈ 211
        let t = arm_to_truth_u8(&rule(249, 300, 1000), NARS_PERSONALITY_K);
        assert_eq!(t.frequency, ((249u64 * 255) / 300) as u8);
    }

    #[test]
    fn confidence_grows_with_evidence_integer() {
        // m=200, k=1 → 200·255/201 = 253
        let strong = arm_to_truth_u8(&rule(200, 240, 1000), NARS_PERSONALITY_K);
        // m=2,   k=1 → 2·255/3 = 170
        let weak = arm_to_truth_u8(&rule(2, 3, 10), NARS_PERSONALITY_K);
        assert!(strong.confidence > weak.confidence);
        assert_eq!(strong.confidence, ((200u64 * 255) / 201) as u8);
        assert_eq!(weak.confidence, ((2u64 * 255) / 3) as u8);
    }

    #[test]
    fn larger_k_demands_more_evidence_integer() {
        let k1 = arm_to_truth_u8(&rule(5, 10, 100), 1).confidence;
        let k10 = arm_to_truth_u8(&rule(5, 10, 100), 10).confidence;
        assert!(k10 < k1);
    }

    #[test]
    fn zero_antecedent_is_unknown_frequency() {
        let t = arm_to_truth_u8(&rule(0, 0, 1000), NARS_PERSONALITY_K);
        assert_eq!(t.frequency, 128);
        assert_eq!(t.confidence, 0);
    }

    #[test]
    fn f32_edge_is_derived_from_u8() {
        let r = rule(100, 125, 400); // m=100; conf_u8 = 100·255/101 = 252
        let t8 = arm_to_truth_u8(&r, NARS_PERSONALITY_K);
        let tf = arm_to_nars(&r, NARS_PERSONALITY_K);
        assert!((tf.confidence - t8.confidence as f32 / 255.0).abs() < 1e-6);
        assert!((tf.frequency - t8.frequency as f32 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn triple_projection() {
        let t = CandidateTriple::from_rule(&rule(90, 100, 400), &DebugProjector::default(), NARS_PERSONALITY_K);
        assert_eq!(t.s, "arm:feat0=cat1");
        assert_eq!(t.p, "implies");
        assert_eq!(t.o, "arm:feat1=cat0");
    }
}
