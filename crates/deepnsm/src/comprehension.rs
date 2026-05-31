//! Sentence resolution — literal text comprehension + the fact/story router
//! (the **Wernicke** faculty of `E-ENGLISH-BIFURCATES`).
//!
//! Separate from the MarkovBundler **projection** (`arcs.rs` / `markov_bundle.rs`)
//! by design (user, 2026-05-31: *"the sentence resolution is literal text
//! comprehension with ambiguity resolution without tokens"*). This faculty reads
//! the **comprehended** structure — `SentenceStructure`, whose words are COCA
//! ranks, **not** BPE tokens ("without tokens") — and resolves where each triple
//! lands. It never touches the VSA projection band; routing is a comprehension
//! decision, not a projection measurement.
//!
//! ## The three-faculty boundary (Broca / Wernicke / Hippocampus)
//!
//! - **Broca** (projection / syntax) — PoS-FSM → SPO + the MarkovBundler wave
//!   (`parser.rs`, `markov_bundle.rs`, `arcs.rs`). *Assembles structure.*
//! - **Wernicke** (comprehension / resolution) — **this module**: maps the
//!   comprehended SPO to meaning, resolves ambiguity, routes fact vs story.
//!   *Resolves meaning.* (±5 coreference / `context_chain` is the deeper
//!   ambiguity-resolution wire that plugs in here — not yet connected.)
//! - **Hippocampus** (episodic memory) — downstream/agnostic: the story-arc
//!   (`EpisodicEdges64`, ±5→±500), and consolidation-to-semantic via the
//!   `WitnessTable` lifecycle (`spo_fact_ref None→Some→tombstone` = an aged
//!   story crystallising into a DOLCE fact). *Remembers + consolidates.*
//!
//! ## The router is a FORK, not a switch (`OQ-ROUTER-SIGNAL`)
//!
//! Every SPO relation is a fact-candidate; a triple the parser marked temporal
//! ALSO threads a story-arc. Resolved **per triple**, because one sentence can
//! carry both a timeless relation and a dated event ("the dog, which is a
//! mammal, ran"). Whether the fact leg is *committed* to the ontology (vs left
//! as an event-only relation) is a downstream policy, deliberately not here.

use crate::parser::SentenceStructure;

/// Where a single comprehended triple lands (`E-ENGLISH-BIFURCATES`). A FORK:
/// `fact` is the always-present (atemporal) SPO relation; `story` is the
/// additive temporal placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Landing {
    /// The SPO relation is assertable as an (atemporal) ontology fact.
    /// Always true at this layer; commit-policy is downstream.
    pub fact: bool,
    /// The triple carries a temporal marker, so it ALSO threads a story-arc.
    pub story: bool,
}

impl SentenceStructure {
    /// Did the parser comprehend triple `triple_index` as carrying a temporal
    /// marker? Reads the per-triple `temporals` comprehension signal — no
    /// tokens, no projection band.
    #[must_use]
    pub fn is_temporal(&self, triple_index: usize) -> bool {
        self.temporals.iter().any(|&(ti, _)| ti == triple_index)
    }

    /// Resolve the landing of triple `triple_index`: always a fact-candidate;
    /// also a story-arc iff it carries a temporal marker (the fork).
    #[must_use]
    pub fn triple_landing(&self, triple_index: usize) -> Landing {
        Landing {
            fact: true,
            story: self.is_temporal(triple_index),
        }
    }

    /// Landing for every triple in the sentence, in triple order.
    #[must_use]
    pub fn landings(&self) -> Vec<Landing> {
        (0..self.triples.len())
            .map(|i| self.triple_landing(i))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spo::SpoTriple;

    /// A sentence with two triples; triple 0 is marked temporal (e.g. a past-
    /// tense event), triple 1 is atemporal (a timeless relation).
    fn two_triples_first_temporal() -> SentenceStructure {
        SentenceStructure {
            triples: vec![SpoTriple::new(1, 2, 3), SpoTriple::new(4, 5, 6)],
            modifiers: Vec::new(),
            negations: Vec::new(),
            temporals: vec![(0, 99)], // triple 0 has temporal word rank 99
        }
    }

    #[test]
    fn temporal_triple_forks_into_fact_and_story() {
        let s = two_triples_first_temporal();
        assert!(s.is_temporal(0));
        let l = s.triple_landing(0);
        assert!(l.fact && l.story, "temporal triple → BOTH fact and story (fork)");
    }

    #[test]
    fn atemporal_triple_is_fact_only() {
        let s = two_triples_first_temporal();
        assert!(!s.is_temporal(1));
        let l = s.triple_landing(1);
        assert!(l.fact, "every SPO is a fact-candidate");
        assert!(!l.story, "no temporal marker → no story-arc");
    }

    #[test]
    fn landings_are_per_triple_in_order() {
        let s = two_triples_first_temporal();
        let ls = s.landings();
        assert_eq!(ls.len(), 2);
        assert_eq!(ls[0], Landing { fact: true, story: true });
        assert_eq!(ls[1], Landing { fact: true, story: false });
    }

    #[test]
    fn empty_sentence_has_no_landings() {
        let s = SentenceStructure {
            triples: Vec::new(),
            modifiers: Vec::new(),
            negations: Vec::new(),
            temporals: Vec::new(),
        };
        assert!(s.landings().is_empty());
        // Out-of-range index is fact-only (no temporal marker can match).
        assert_eq!(s.triple_landing(7), Landing { fact: true, story: false });
    }
}
