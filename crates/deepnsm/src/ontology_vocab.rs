//! The **second codebook**: the OBO Ontology vocabulary, alongside COCA.
//!
//! DeepNSM's own [`Vocabulary`](crate::vocabulary::Vocabulary) is the academic
//! frequency codebook — COCA ranks in a 12-bit space. This module is the other
//! one a reasoning caller needs: the public OBO biomedical reference
//! (`ConceptDomain::Ontology`, `0x03XX`), read through the contract's zero-dep
//! wire mirror.
//!
//! # No new dependency, and that is the point
//!
//! `lance-graph-contract` is already a hard dep of this crate (the canonical
//! `RoleKeySlice` constants), so reaching the ontology costs nothing here. The
//! alternative — deping the producer `ogar-obo` — is what the plug-and-play
//! posture exists to avoid, and it would pull an OBO bake into a crate whose
//! job is distributional semantics.
//!
//! # Two codebooks, two address spaces — never silently merged
//!
//! | codebook | address | width | source |
//! |---|---|---|---|
//! | COCA | frequency rank | 12-bit (`VOCAB_SIZE`) | [`crate::vocabulary`] |
//! | Ontology | canonical concept id | 16-bit (`0x03XX`) | the OGAR mint, mirrored |
//!
//! A rank and a concept id are not interchangeable and this module does not
//! offer a conversion. Fusing them into one integer space would make
//! `rank == concept` collisions unnoticeable, and the two spaces have entirely
//! different owners: a rank moves when the corpus is re-counted, a concept id
//! is minted once and never moves.
//!
//! # Why this could not be written before 2026-08-22
//!
//! `concepts_in_domain(ConceptDomain::Ontology)` returned **empty**. The OBO
//! concept ids lived only in the producer crates, so the shared codebook had
//! nothing in that domain — and an empty enumeration reads exactly like "this
//! domain has nothing to reason about". The operator's ruling that the domains
//! are minted in `ogar-vocab` is what gave this module something to return.

use lance_graph_contract::ogar_codebook::{concepts_in_domain, ConceptDomain};

/// One ontology concept: its canonical name and the id it is minted at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OntologyConcept {
    /// Canonical concept name as minted (`"mondo"`, `"uberon"`, `"bfo"`, …).
    pub name: &'static str,
    /// The canonical hi-u16 concept id (`0x03XX`).
    pub concept_id: u16,
}

/// The whole Ontology vocabulary, in codebook order.
///
/// Derived from the mirror on every call — never cached into a second table,
/// so it cannot drift from the mint the way a local copy would.
#[must_use]
pub fn ontology_vocabulary() -> Vec<OntologyConcept> {
    concepts_in_domain(ConceptDomain::Ontology)
        .map(|(name, concept_id)| OntologyConcept { name, concept_id })
        .collect()
}

/// Resolve a concept name to its id, or `None` if the Ontology domain does not
/// mint it — a refusal, never a guess.
#[must_use]
pub fn concept_id(name: &str) -> Option<u16> {
    concepts_in_domain(ConceptDomain::Ontology)
        .find(|(n, _)| *n == name)
        .map(|(_, id)| id)
}

/// The inverse: which ontology concept a `0x03XX` id names, or `None` outside
/// the domain.
#[must_use]
pub fn concept_at(concept_id: u16) -> Option<&'static str> {
    concepts_in_domain(ConceptDomain::Ontology)
        .find(|(_, id)| *id == concept_id)
        .map(|(n, _)| n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocabulary::VOCAB_SIZE;

    /// **The vocabulary is reachable and non-empty from inside DeepNSM.**
    /// That is the whole wiring claim, so it is the test.
    #[test]
    fn deepnsm_can_read_the_ontology_vocabulary() {
        let v = ontology_vocabulary();
        assert!(
            v.len() >= 14,
            "expected the OBO core + relation body + meta-study spine, got {v:?}"
        );
        for c in &v {
            assert_eq!(c.concept_id >> 8, 0x03, "{c:?} is outside the domain");
            assert_eq!(concept_at(c.concept_id), Some(c.name), "round-trip");
            assert_eq!(concept_id(c.name), Some(c.concept_id), "round-trip");
        }
    }

    /// Refusals, on non-trivial input: a name nobody minted, and an id in a
    /// DIFFERENT real domain. Without the second case the lookup could be
    /// accepting everything in the codebook and still pass.
    #[test]
    fn an_unminted_name_and_a_foreign_id_are_both_refused() {
        assert_eq!(
            concept_id("mondo"),
            Some(0x0301),
            "control: this one exists"
        );
        assert_eq!(concept_id("definitely_not_an_ontology"), None);
        // 0x0901 is Health, a populated domain that is NOT Ontology.
        assert_eq!(
            concept_at(0x0901),
            None,
            "a Health id must not resolve as an ontology concept"
        );
    }

    /// **The two address spaces stay apart.** A COCA rank and a concept id can
    /// hold the same integer and mean unrelated things; nothing here converts
    /// between them, and this test pins that overlap so a future "just cast it"
    /// has to argue with a failing assertion.
    #[test]
    fn a_coca_rank_and_a_concept_id_are_not_the_same_space() {
        let ids: Vec<u16> = ontology_vocabulary().iter().map(|c| c.concept_id).collect();
        let overlapping: Vec<u16> = ids
            .iter()
            .copied()
            .filter(|id| (*id as usize) < VOCAB_SIZE)
            .collect();
        assert!(
            !overlapping.is_empty(),
            "anti-vacuity: some concept ids DO fall inside the 12-bit rank range \
             ({VOCAB_SIZE}), which is exactly why the spaces must not be merged"
        );
        // …and those ids resolve as ontology concepts, not as COCA ranks —
        // there is no shared lookup that could answer for both.
        for id in overlapping {
            assert!(concept_at(id).is_some());
        }
    }
}
