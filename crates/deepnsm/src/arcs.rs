//! Basin/literal arc split + the fact/story temporal router — the
//! `E-ENGLISH-BIFURCATES` seam (first slice).
//!
//! ## What this is
//!
//! The language↔meaning duality, made into typed, tested Rust on the
//! role-indexed `Trajectory` carrier:
//!
//! - **basin arc** — the semantic spine: ONE role-superposed bundle that
//!   points at a single basin (which DOLCE class / which story-arc). The
//!   *declared, exact* side of the duality (the meaning keyframe).
//! - **literal arc** — the language surface: the COCA literal ranks that fed
//!   the bundle. *Multiple, redundant, prunable* once the basin resolves —
//!   the *detected* side (the witnesses that tombstone after resolution; the
//!   prune lifecycle itself lives contract-side in `WitnessTable`, not here).
//!
//! ## The router (`OQ-ROUTER-SIGNAL` resolved: FORK, not switch)
//!
//! English bifurcates by temporality: an atemporal clause lands as a FACT
//! (ontology / DOLCE frozen identity); a temporal clause threads a STORY-ARC
//! (episodic ±5..500). The router signal already exists in the carrier — the
//! `TEMPORAL` role band `[9000..9200)`. A clause with temporal content
//! (tense/aspect projected into that band) reads non-trivial energy there; an
//! atemporal assertion reads ~0.
//!
//! It is a **fork, not a switch**: every SPO relation is a fact-candidate, and
//! temporal content *adds* a story-arc on top — "the dog, which is a mammal,
//! ran" yields both. So `threads_story` is the *discriminating* gate (story is
//! additive); whether the fact is *committed* to the ontology is a downstream
//! type-relation-vs-event-relation policy, not decided at this layer.
//!
//! ## Firewall
//!
//! Both arcs live in deepnsm (the English side, upstream). The basin arc is
//! f32 because the VSA carrier is f32 *upstream* — it is sign-binarized
//! (`disambiguator_glue`) or resolved to an opaque handle before it ever
//! crosses into the agnostic graph. No COCA rank reaches the hot graph as
//! identity; the literal ranks stay here as prunable witnesses.

use crate::markov_bundle::GrammaticalRole;
use crate::trajectory::Trajectory;

/// The semantic spine: the role-superposed bundle that points at ONE basin.
/// The declared/exact side of the language↔meaning duality.
#[derive(Debug, Clone, PartialEq)]
pub struct BasinArc(pub Vec<f32>);

/// The language surface: the COCA literal ranks that fed the bundle.
/// Multiple, redundant, prunable once the basin resolves (the detected side).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiteralArc(pub Vec<u16>);

/// Where an English clause lands (`E-ENGLISH-BIFURCATES`). A FORK, not a
/// switch (`OQ-ROUTER-SIGNAL`): `fact` is the always-present SPO relation;
/// `story` is the additive temporal placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Landing {
    /// The SPO relation is assertable as an (atemporal) ontology fact.
    /// Always true at this layer — whether it is *committed* is a downstream
    /// policy (type-relation vs event-relation), deliberately not decided here.
    pub fact: bool,
    /// The clause carries temporal content, so it ALSO threads a story-arc.
    pub story: bool,
}

impl Trajectory {
    /// Split this trajectory into its **basin arc** (the full role-superposed
    /// spine bundle) and its **literal arc** (the COCA ranks that fed it).
    ///
    /// The basin is the meaning keyframe; the literals are the prunable
    /// surface pointers. This names the duality at the seam where
    /// `disambiguator_glue` already threads the bundle into the contract
    /// `context_chain` — turning the implicit `&[f32]` + untyped candidates
    /// into the explicit basin↔literal pair.
    #[must_use]
    pub fn split_arcs(&self, literal_ranks: &[u16]) -> (BasinArc, LiteralArc) {
        (
            BasinArc(self.fingerprint.clone()),
            LiteralArc(literal_ranks.to_vec()),
        )
    }

    /// Total absolute energy in the `TEMPORAL` role band `[9000..9200)` — the
    /// bifurcation router signal. ~0 for an atemporal assertion; non-trivial
    /// when tense/aspect content is projected into the temporal band.
    #[must_use]
    pub fn temporal_energy(&self) -> f32 {
        self.role_bundle(GrammaticalRole::Temporal)
            .iter()
            .map(|v| v.abs())
            .sum()
    }

    /// Whether this clause threads a story-arc — the discriminating fork gate.
    /// `temporal_threshold` is explicit (no hidden default), matching the
    /// `role_candidates` convention: the temporal floor is style/corpus-tunable.
    #[must_use]
    pub fn threads_story(&self, temporal_threshold: f32) -> bool {
        self.temporal_energy() > temporal_threshold
    }

    /// Route this trajectory to its landing (`E-ENGLISH-BIFURCATES`). The
    /// `fact` leg is universal (every SPO relation is a fact-candidate); the
    /// `story` leg is additive when temporal content clears `temporal_threshold`.
    #[must_use]
    pub fn landing(&self, temporal_threshold: f32) -> Landing {
        Landing {
            fact: true,
            story: self.threads_story(temporal_threshold),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIMS: usize = 16_384;

    /// Build a 16,384-dim fingerprint with `value` written across the given
    /// role band and 0.0 everywhere else.
    fn fingerprint_in_band(role: GrammaticalRole, value: f32) -> Vec<f32> {
        let mut fp = vec![0.0_f32; DIMS];
        let slice = role.slice();
        for v in fp[slice.start..slice.stop].iter_mut() {
            *v = value;
        }
        fp
    }

    #[test]
    fn split_arcs_preserves_basin_and_literals() {
        let t = Trajectory {
            fingerprint: vec![0.25_f32; DIMS],
            radius: 5,
        };
        let ranks = [12_u16, 670, 2942];
        let (basin, literal) = t.split_arcs(&ranks);
        assert_eq!(basin.0, t.fingerprint, "basin arc IS the spine bundle");
        assert_eq!(literal.0, ranks, "literal arc carries the COCA ranks verbatim");
    }

    #[test]
    fn literal_arc_is_independent_of_basin() {
        // Same basin, two different literal sets → distinct literal arcs.
        // The prune target (literals) is separable from the spine (basin).
        let t = Trajectory {
            fingerprint: vec![1.0_f32; DIMS],
            radius: 5,
        };
        let (b1, l1) = t.split_arcs(&[1, 2, 3]);
        let (b2, l2) = t.split_arcs(&[9, 9]);
        assert_eq!(b1, b2, "basin unchanged by literal choice");
        assert_ne!(l1, l2, "literal arcs differ");
    }

    #[test]
    fn atemporal_bundle_has_zero_temporal_energy_and_no_story() {
        // Content only in the SUBJECT band → temporal band is empty → FACT.
        let t = Trajectory {
            fingerprint: fingerprint_in_band(GrammaticalRole::Subject, 1.0),
            radius: 5,
        };
        assert_eq!(t.temporal_energy(), 0.0);
        assert!(!t.threads_story(0.0), "no temporal content → no story-arc");
        let landing = t.landing(0.0);
        assert!(landing.fact, "every SPO is a fact-candidate");
        assert!(!landing.story);
    }

    #[test]
    fn temporal_bundle_has_positive_energy_and_threads_story() {
        // Content in the TEMPORAL band [9000..9200) → STORY (additive to fact).
        let t = Trajectory {
            fingerprint: fingerprint_in_band(GrammaticalRole::Temporal, 1.0),
            radius: 5,
        };
        let band_len = GrammaticalRole::Temporal.slice().len() as f32;
        assert!((t.temporal_energy() - band_len).abs() < 1e-3);
        assert!(t.threads_story(0.5), "temporal content → story-arc threaded");
        let landing = t.landing(0.5);
        assert!(landing.fact && landing.story, "fork: both fact AND story");
    }

    #[test]
    fn threads_story_respects_explicit_threshold() {
        // Small temporal energy: below a high threshold (fact-only), above a
        // low one (also story). No hidden default decides this.
        let mut fp = vec![0.0_f32; DIMS];
        let slice = GrammaticalRole::Temporal.slice();
        // 10 dims of 1.0 → temporal_energy == 10.0
        for v in fp[slice.start..slice.start + 10].iter_mut() {
            *v = 1.0;
        }
        let t = Trajectory {
            fingerprint: fp,
            radius: 5,
        };
        assert!((t.temporal_energy() - 10.0).abs() < 1e-3);
        assert!(!t.threads_story(20.0), "below threshold → fact-only");
        assert!(t.threads_story(5.0), "above threshold → threads story");
    }
}
