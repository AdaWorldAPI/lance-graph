//! # DeepNSM v2 — the distributional semantic core on the V3 palette256² architecture
//!
//! A parallel, updated DeepNSM. The existing `deepnsm` crate is **untouched**;
//! this crate keeps the DeepNSM *signature* — a frequency-ranked vocabulary, a
//! part-of-speech FSM, and SPO triples — but rebuilds the substrate on the V3
//! architecture shipped in `lance-graph-contract`, consuming those primitives
//! rather than reimplementing them.
//!
//! ## What changed (v1 → v2)
//!
//! | DeepNSM v1 | DeepNSM v2 | Consumed from `lance-graph-contract` |
//! |---|---|---|
//! | 4,096-word COCA table, 12-bit ids | `256×256` palette tile, 16-bit ids ([`vocab`]) | — |
//! | `4096²` u8 distance matrix (16 MB, stored) | certified palette256² distance ([`space::SemanticSpace`]) | [`recipe_substrate::PairPalette`] |
//! | (no whole-work code) | `6×256` CAM ADC, one tile per work ([`space::AdcSpace`]) | [`cam::ScalarAdc`] |
//! | 512-bit VSA XOR bind + majority bundle | palette `(basin, identity)` addressing ([`spo::Spo`]) | — |
//! | ±5 sentence ring buffer | version-range read ([`TemporalStream`]) | [`temporal_pov::TemporalPov`] |
//! | 6-state PoS FSM → SPO | 6-state PoS FSM → SPO (preserved) ([`fsm`]) | — |
//!
//! [`recipe_substrate::PairPalette`]: lance_graph_contract::recipe_substrate::PairPalette
//! [`cam::ScalarAdc`]: lance_graph_contract::cam::ScalarAdc
//! [`temporal_pov::TemporalPov`]: lance_graph_contract::temporal_pov::TemporalPov
//!
//! ## Honest scope
//!
//! The palette256² and ADC distances carry **real** semantics only with a
//! **trained codebook**. Producing that from real embeddings is the ndarray-side
//! producer named in `TD-CERTIFIED-DISTANCE-TABLE-UNCONSUMED` and does not exist
//! yet, so [`space`] ships deterministic `demo()` codebooks (a placeholder) plus
//! `from_*` constructors for real ones. This crate wires the architecture and is
//! test-proven end to end on the demo codebook; it does **not** read any real
//! corpus (Aesop/Bible/etc.) — that stays the open `ISS-DCSW-REAL-CORPUS-BLOCKED`.

pub mod fsm;
pub mod space;
pub mod spo;
pub mod vocab;

use lance_graph_contract::temporal_pov::{TemporalPov, VersionRange};

pub use fsm::{parse_to_spo, Pos, Tagged};
pub use space::{AdcSpace, SemanticSpace};
pub use spo::Spo;
pub use vocab::{PaletteVocab, WordId};

/// The DeepNSM v2 engine: a palette vocabulary + a semantic-distance space.
///
/// `ingest` runs the PoS FSM ([`fsm::parse_to_spo`]); `word_similarity` reads
/// the certified palette256² table ([`space::SemanticSpace`]).
#[derive(Debug, Clone)]
pub struct Nsm {
    /// The frequency-ranked palette vocabulary.
    pub vocab: PaletteVocab,
    /// The palette256² semantic-distance space.
    pub space: SemanticSpace,
}

impl Nsm {
    /// Build from a vocabulary and a semantic space.
    #[must_use]
    pub fn new(vocab: PaletteVocab, space: SemanticSpace) -> Self {
        Self { vocab, space }
    }

    /// Run the PoS FSM on a tagged token stream → SPO triples.
    #[must_use]
    pub fn ingest(&self, tokens: &[Tagged]) -> Vec<Spo> {
        parse_to_spo(tokens)
    }

    /// Distributional similarity ∈ `[0, 1]` between two words by surface form
    /// (exact vocabulary match; caller normalizes case). `None` if either word
    /// is out of vocabulary.
    #[must_use]
    pub fn word_similarity(&self, a: &str, b: &str) -> Option<f32> {
        let pa = self.vocab.pair(a)?;
        let pb = self.vocab.pair(b)?;
        Some(self.space.similarity(pa, pb))
    }

    /// Slot-wise similarity between two triples: `[S↔S, P↔P, O↔O]`, each read
    /// from the palette256² table. The DeepNSM "same-role comparison" on v2.
    #[must_use]
    pub fn triple_similarity(&self, a: Spo, b: Spo) -> [f32; 3] {
        let (pa, pb) = (a.pairs(), b.pairs());
        [
            self.space.similarity(pa[0], pb[0]),
            self.space.similarity(pa[1], pb[1]),
            self.space.similarity(pa[2], pb[2]),
        ]
    }
}

/// A version-stamped SPO stream, read through a [`TemporalPov`] window.
///
/// Replaces v1's fixed ±5 ring buffer: each ingested triple gets a monotone
/// version (the Lance-version / sentence-commit tick), and `window` admits
/// exactly the triples a reader at `ref_version` can see — any width, per the
/// `temporal_pov` version-range contract (not a fixed ±5).
#[derive(Debug, Clone, Default)]
pub struct TemporalStream {
    /// `(version, triple)` in append order.
    entries: Vec<(u64, Spo)>,
}

impl TemporalStream {
    /// Empty stream.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a triple at `version` (the caller's monotone tick).
    pub fn push(&mut self, version: u64, triple: Spo) {
        self.entries.push((version, triple));
    }

    /// Every triple a reader pinned at `ref_version` can see — the contemporary
    /// window `row_version ≤ ref_version` ([`TemporalPov::at`]), the version-range
    /// generalization of the ±5 ring.
    #[must_use]
    pub fn window_at(&self, ref_version: u64) -> Vec<Spo> {
        let pov = TemporalPov::at(ref_version, 0);
        self.entries
            .iter()
            .filter(|(v, _)| pov.admits(*v))
            .map(|(_, t)| *t)
            .collect()
    }

    /// Every triple in an explicit half-open [`VersionRange`] `[from, to)` — an
    /// arbitrary-width window (any span, not just the contemporary prefix).
    #[must_use]
    pub fn window_range(&self, range: VersionRange) -> Vec<Spo> {
        self.entries
            .iter()
            .filter(|(v, _)| range.contains(*v))
            .map(|(_, t)| *t)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsm::Tagged;

    fn small_nsm() -> Nsm {
        let mut vocab = PaletteVocab::new();
        vocab.from_frequency_ranked(["dog", "bit", "man", "cat", "saw", "mouse"]);
        Nsm::new(vocab, SemanticSpace::demo(4))
    }

    #[test]
    fn end_to_end_ingest_produces_triples() {
        let nsm = small_nsm();
        // "dog bit man" tagged.
        let toks = [
            Tagged::new(nsm.vocab.id("dog").unwrap(), Pos::Noun),
            Tagged::new(nsm.vocab.id("bit").unwrap(), Pos::Verb),
            Tagged::new(nsm.vocab.id("man").unwrap(), Pos::Noun),
        ];
        let spo = nsm.ingest(&toks);
        assert_eq!(spo.len(), 1);
        assert_eq!(nsm.vocab.word(spo[0].subject), Some("dog"));
        assert_eq!(nsm.vocab.word(spo[0].object), Some("man"));
    }

    #[test]
    fn word_similarity_is_reflexive_and_in_vocab_gated() {
        let nsm = small_nsm();
        assert_eq!(nsm.word_similarity("dog", "dog"), Some(1.0));
        assert!(nsm.word_similarity("dog", "cat").unwrap() >= 0.0);
        assert_eq!(nsm.word_similarity("dog", "unicorn"), None); // OOV
    }

    #[test]
    fn triple_similarity_self_is_all_ones() {
        let nsm = small_nsm();
        let t = Spo::new(0, 1, 2);
        assert_eq!(nsm.triple_similarity(t, t), [1.0, 1.0, 1.0]);
    }

    #[test]
    fn temporal_window_generalizes_the_pm5_ring() {
        let mut s = TemporalStream::new();
        for v in 0..10u64 {
            s.push(v, Spo::new(v as u16, 0, 0));
        }
        // Contemporary window at v=4 admits versions 0..=4 (5 triples).
        assert_eq!(s.window_at(4).len(), 5);
        // Arbitrary width [2,7) admits versions 2..=6 (5 triples) — any span.
        assert_eq!(s.window_range(VersionRange::new(2, 7)).len(), 5);
        // A future-frame triple is not admitted at an earlier ref.
        assert!(s.window_at(4).iter().all(|t| t.subject <= 4));
    }
}
