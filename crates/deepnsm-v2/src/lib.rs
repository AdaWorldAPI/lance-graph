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
//! | `4096²` u8 distance matrix (16 MB, stored) | CAM-PQ **96** DISTRIBUTION — `6×256:256` ([`space::Cam96Space`], ρ 0.828 vs Jina) | [`recipe_substrate::PairPalette`] rail algebra |
//! | (no whole-work code) | `6×256` CAM ADC POINT (the 48-bit reference, ρ 0.711) ([`space::AdcSpace`]) | [`cam::ScalarAdc`] |
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
//! A **trained codebook now exists** (`data/`, produced by `probes/` from real
//! Jina-v3 embeddings of the 12,543-word KJV vocabulary; held-out ρ 0.774 vs
//! the 48-bit point's 0.617) and loads via [`codebook`]. `demo()` codebooks
//! remain for standalone tests only. The crate reads a real corpus end to end:
//! `examples/bible_wave.rs` runs the whole KJV (23,145 verses = one 64k tile)
//! through FSM → SPO → `TemporalStream` with the trained codes — measuring that
//! **63.3% of same-subject context lies beyond ±5** (what v1's ring forfeits).
//! Doc-table ρ values above are in-sample-era; current held-out numbers live in
//! `probes/README.md` (0.766 general / 0.774 Bible-vocab vs 0.624/0.617).

pub mod codebook;
pub mod fsm;
pub mod space;
pub mod spo;
pub mod vocab;
pub mod wave;

use lance_graph_contract::temporal_pov::{TemporalPov, VersionRange};

pub use codebook::{load_cam96_codes, load_cam96_space, CodebookError};
pub use fsm::{parse_to_spo, Pos, Tagged};
pub use space::{AdcSpace, Cam96, Cam96Space, SemanticSpace};
pub use spo::Spo;
pub use vocab::{PaletteVocab, WordId};
pub use wave::WitnessStream;

/// The DeepNSM v2 engine: a routing vocabulary + a CAM-PQ 96 meaning space.
///
/// The two axes are **orthogonal** (measured, `probes/`): the **routing**
/// address is the frequency-ranked [`PaletteVocab`] id (⟂ meaning — ρ≈−0.07 vs
/// Jina), and the **meaning** is the per-word 96-bit [`Cam96`] code read through
/// [`Cam96Space`] (`6×cosine²` DISTRIBUTION — ρ 0.828 vs Jina). `word_similarity`
/// reads the **meaning code**, never the routing pair — the fix for conflating
/// the two (the shipped v1-of-this-crate defect used one `256:256` rail).
///
/// Meaning codes come from embeddings: [`Cam96Space::encode`] quantizes a word's
/// vector into its 12-byte code; supply them via [`with_codes`](Self::with_codes).
/// Without codes, `word_similarity` returns `None` (there is no meaning without
/// an embedding — routing + the FSM still work).
#[derive(Debug, Clone)]
pub struct Nsm {
    /// The frequency-ranked palette vocabulary — the ROUTING address.
    pub vocab: PaletteVocab,
    /// The CAM-PQ 96 meaning-distance space — the DISTRIBUTION.
    pub space: Cam96Space,
    /// Per-word-id meaning code (`codes[id]`); empty until fed by embeddings.
    codes: Vec<Cam96>,
}

impl Nsm {
    /// Build with routing + FSM only (no meaning codes yet). `word_similarity`
    /// returns `None` until [`with_codes`](Self::with_codes) supplies them.
    #[must_use]
    pub fn new(vocab: PaletteVocab, space: Cam96Space) -> Self {
        Self {
            vocab,
            space,
            codes: Vec::new(),
        }
    }

    /// Build with per-word-id meaning codes — `codes[id]` is word `id`'s 96-bit
    /// CAM-PQ DISTRIBUTION code (produced by [`Cam96Space::encode`] on its
    /// embedding).
    #[must_use]
    pub fn with_codes(vocab: PaletteVocab, space: Cam96Space, codes: Vec<Cam96>) -> Self {
        Self {
            vocab,
            space,
            codes,
        }
    }

    /// Run the PoS FSM on a tagged token stream → SPO triples.
    #[must_use]
    pub fn ingest(&self, tokens: &[Tagged]) -> Vec<Spo> {
        parse_to_spo(tokens)
    }

    /// The 96-bit meaning code for a word by surface form, if it is in vocab AND
    /// a code has been supplied.
    #[must_use]
    pub fn code(&self, word: &str) -> Option<&Cam96> {
        self.codes.get(self.vocab.id(word)? as usize)
    }

    /// Meaning similarity ∈ `[0, 1]` between two words, read from their 96-bit
    /// CAM-PQ DISTRIBUTION codes (`6×cosine²`). `None` if either is out of vocab
    /// or has no code. This is the **meaning** axis — distinct from routing.
    #[must_use]
    pub fn word_similarity(&self, a: &str, b: &str) -> Option<f32> {
        Some(self.space.similarity(self.code(a)?, self.code(b)?))
    }

    /// Slot-wise meaning similarity between two triples: `[S↔S, P↔P, O↔O]`, each
    /// read from the CAM-PQ 96 code of the corresponding word id. A slot is
    /// `None` when its word has no code.
    #[must_use]
    pub fn triple_similarity(&self, a: Spo, b: Spo) -> [Option<f32>; 3] {
        let sim = |x: WordId, y: WordId| {
            Some(
                self.space
                    .similarity(self.codes.get(x as usize)?, self.codes.get(y as usize)?),
            )
        };
        [
            sim(a.subject, b.subject),
            sim(a.predicate, b.predicate),
            sim(a.object, b.object),
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

    /// Deterministic distinct 96-bit meaning codes (placeholder; real codes come
    /// from [`Cam96Space::encode`] on embeddings).
    fn demo_codes(n: usize) -> Vec<Cam96> {
        (0..n)
            .map(|i| std::array::from_fn(|k| ((i * 7 + k * 13) % 251) as u8))
            .collect()
    }

    fn small_nsm() -> Nsm {
        let mut vocab = PaletteVocab::new();
        vocab.from_frequency_ranked(["dog", "bit", "man", "cat", "saw", "mouse"]);
        let codes = demo_codes(vocab.len());
        Nsm::with_codes(vocab, Cam96Space::demo(4), codes)
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
    fn word_similarity_reads_meaning_code_not_routing() {
        let nsm = small_nsm();
        // same word → same 96-bit code → maximal meaning similarity.
        assert_eq!(nsm.word_similarity("dog", "dog"), Some(1.0));
        assert!(nsm.word_similarity("dog", "cat").unwrap() >= 0.0);
        assert_eq!(nsm.word_similarity("dog", "unicorn"), None); // OOV
                                                                 // no codes → meaning is None even for in-vocab words (routing still works).
        let mut v = PaletteVocab::new();
        v.from_frequency_ranked(["dog", "cat"]);
        let routing_only = Nsm::new(v, Cam96Space::demo(4));
        assert_eq!(routing_only.word_similarity("dog", "cat"), None);
        assert!(routing_only.vocab.id("dog").is_some()); // routing intact
    }

    #[test]
    fn triple_similarity_self_is_all_ones() {
        let nsm = small_nsm();
        let t = Spo::new(0, 1, 2);
        assert_eq!(
            nsm.triple_similarity(t, t),
            [Some(1.0), Some(1.0), Some(1.0)]
        );
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
