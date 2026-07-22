//! `vocab` — the palette256² word vocabulary.
//!
//! ## What changed from DeepNSM v1
//!
//! v1 used a flat 4,096-word COCA table (12-bit ids) with a `4096²` u8 distance
//! matrix. v2 carves the id space as a **256×256 palette tile**: a word id is a
//! `(basin, identity)` byte pair — `basin = id >> 8`, `identity = id & 0xFF` —
//! so the id IS the address into the palette256² distance table
//! ([`crate::space`]). Capacity is `65_536`; the ~20k academic vocabulary (the
//! `80 × 256` carve the plan names) fits with room to spare.
//!
//! ## `vocabulary = frequency × distance`
//!
//! The two bytes are not arbitrary. Per the operator ruling the vocabulary is
//! `frequency × distance`: the **basin** (high byte) is the frequency-rank band
//! (coarse — words in the same 256-wide band share a basin), and the
//! **identity** (low byte) is the within-band position. Assigning ids in
//! descending frequency order therefore puts high-frequency words in low basins
//! and makes basin-adjacency a frequency proxy — while the *semantic* distance
//! between two ids comes from the trained codebook in [`crate::space`], never
//! from the id arithmetic itself.

/// A word id in the `256×256` palette tile.
///
/// Stored as a `u16`; the `(basin, identity)` split is `(id >> 8, id & 0xFF)`.
pub type WordId = u16;

/// Split a [`WordId`] into its `(basin, identity)` palette byte pair.
#[inline]
#[must_use]
pub const fn split(id: WordId) -> (u8, u8) {
    ((id >> 8) as u8, (id & 0xFF) as u8)
}

/// Recombine a `(basin, identity)` byte pair into a [`WordId`].
#[inline]
#[must_use]
pub const fn join(basin: u8, identity: u8) -> WordId {
    ((basin as u16) << 8) | identity as u16
}

/// The frequency-ranked palette vocabulary.
///
/// Words are inserted in **descending frequency order** (most frequent first),
/// so the first 256 words occupy basin 0, the next 256 basin 1, and so on.
#[derive(Debug, Clone, Default)]
pub struct PaletteVocab {
    /// Insertion-ordered words (index == [`WordId`]).
    words: Vec<String>,
    /// Reverse map word → id, for tokenization.
    index: std::collections::HashMap<String, WordId>,
}

impl PaletteVocab {
    /// Maximum representable vocabulary size (the full `256×256` tile).
    pub const CAPACITY: usize = 65_536;

    /// The `80 × 256` academic-vocab carve the plan names (§8): basins `0..80`.
    pub const ACADEMIC_20K: usize = 80 * 256;

    /// Empty vocabulary.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from a frequency-ranked word list (most frequent FIRST). Words past
    /// [`CAPACITY`](Self::CAPACITY) are dropped (the tile is full); duplicates
    /// keep their first id. Returns the number of words admitted.
    pub fn from_frequency_ranked<I, S>(&mut self, words: I) -> usize
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for w in words {
            if self.words.len() >= Self::CAPACITY {
                break;
            }
            let w = w.into();
            if !self.index.contains_key(&w) {
                let id = self.words.len() as WordId;
                self.index.insert(w.clone(), id);
                self.words.push(w);
            }
        }
        self.words.len()
    }

    /// Number of words in the vocabulary.
    #[must_use]
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Whether the vocabulary is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Look up a word's [`WordId`] (exact match; caller lowercases/normalizes).
    #[must_use]
    pub fn id(&self, word: &str) -> Option<WordId> {
        self.index.get(word).copied()
    }

    /// The word for a [`WordId`], if in range.
    #[must_use]
    pub fn word(&self, id: WordId) -> Option<&str> {
        self.words.get(id as usize).map(String::as_str)
    }

    /// The `(basin, identity)` palette pair for a word, via [`split`].
    #[must_use]
    pub fn pair(&self, word: &str) -> Option<(u8, u8)> {
        self.id(word).map(split)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_join_round_trip() {
        for id in [0u16, 1, 255, 256, 257, 20_479, 65_535] {
            let (b, i) = split(id);
            assert_eq!(join(b, i), id);
        }
    }

    #[test]
    fn frequency_rank_fills_basins_in_order() {
        let mut v = PaletteVocab::new();
        // 300 synthetic words, most-frequent first.
        let words: Vec<String> = (0..300).map(|i| format!("w{i}")).collect();
        assert_eq!(v.from_frequency_ranked(words), 300);
        // word 0 (most frequent) → basin 0; word 256 → basin 1.
        assert_eq!(v.pair("w0"), Some((0, 0)));
        assert_eq!(v.pair("w255"), Some((0, 255)));
        assert_eq!(v.pair("w256"), Some((1, 0)));
        assert_eq!(v.pair("w299"), Some((1, 43)));
    }

    #[test]
    fn duplicates_keep_first_id() {
        let mut v = PaletteVocab::new();
        v.from_frequency_ranked(["the", "cat", "the", "dog"]);
        assert_eq!(v.len(), 3);
        assert_eq!(v.id("the"), Some(0));
        assert_eq!(v.id("dog"), Some(2));
    }

    #[test]
    fn academic_20k_carve_spans_80_basins() {
        // The 80×256 academic carve fills basins 0..80 in frequency order.
        let mut v = PaletteVocab::new();
        let words: Vec<String> = (0..PaletteVocab::ACADEMIC_20K)
            .map(|i| format!("w{i}"))
            .collect();
        assert_eq!(v.from_frequency_ranked(words), 20_480);
        assert_eq!(v.pair("w0"), Some((0, 0))); // most frequent → basin 0
        assert_eq!(v.pair("w20479"), Some((79, 255))); // last academic → basin 79
    }
}
