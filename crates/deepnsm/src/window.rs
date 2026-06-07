//! ±5 sentence window for coreference / pronoun resolution.
//!
//! This is **distinct** from `ContextWindow` in `context.rs`:
//! - `ContextWindow` holds VSA projections (the Broca / MarkovBundler
//!   projection band) for distributional disambiguation.
//! - `SentenceWindow` holds **exact entity candidates** (vocabulary ranks
//!   of NP heads) for **Wernicke / coreference resolution** — the auditable
//!   side of the reading state machine.
//!
//! The two windows serve different faculties and must not be fused
//! (cf. `E-ENGLISH-BIFURCATES` in `comprehension.rs`).
//!
//! ## Coreference heuristic (v1)
//!
//! When the reader encounters a pronoun in the subject slot it calls
//! `resolve_pronoun()`, which walks the entity stack from the most recent
//! sentence backward and returns the first matching entity. In v1 "matching"
//! means "any non-pronoun NP head from a prior sentence" — a pure recency
//! heuristic. Richer disambiguation (gender, number, semantic type) is v2.

use crate::spo::NO_ROLE;

/// Maximum number of NP heads tracked per sentence entry.
const MAX_HEADS_PER_ENTRY: usize = 4;

/// Maximum sentences kept in the window (±5 = 11 total, one per slot).
pub const WINDOW_SIZE: usize = 11;

/// One entry in the ±5 sentence window: the entity candidates from a sentence.
#[derive(Clone, Copy, Debug, Default)]
pub struct WindowEntry {
    /// Sentence identifier (monotonically increasing).
    pub sentence_id: u32,
    /// Vocabulary ranks of NP heads mentioned in this sentence
    /// (subject, object, nominal complements). NO_ROLE fills unused slots.
    pub heads: [u16; MAX_HEADS_PER_ENTRY],
    /// How many heads are actually set (0..=MAX_HEADS_PER_ENTRY).
    pub head_count: usize,
    /// Packed SPO triple from the primary triple in this sentence.
    pub primary_spo_packed: u64,
}

impl WindowEntry {
    /// Push an NP head rank into this entry. Silently drops if full.
    pub fn push_head(&mut self, rank: u16) {
        if self.head_count < MAX_HEADS_PER_ENTRY {
            self.heads[self.head_count] = rank;
            self.head_count += 1;
        }
    }

    /// Iterate over the heads actually set.
    pub fn heads(&self) -> &[u16] {
        &self.heads[..self.head_count]
    }

    /// Does this entry contain the given vocabulary rank?
    pub fn contains(&self, rank: u16) -> bool {
        self.heads().iter().any(|&h| h == rank)
    }
}

/// ±5 sentence ring buffer for exact entity candidate tracking.
///
/// The ring buffer always holds at most `WINDOW_SIZE` (11) entries, dropping
/// the oldest when full. The current sentence is conceptually at offset 0;
/// prior sentences are at offsets −1 .. −5.
#[derive(Debug)]
pub struct SentenceWindow {
    /// Fixed-size ring buffer.
    entries: [WindowEntry; WINDOW_SIZE],
    /// Write head (next slot to overwrite).
    head: usize,
    /// Number of valid entries (0..=WINDOW_SIZE).
    count: usize,
}

impl SentenceWindow {
    /// Create an empty window.
    pub fn new() -> Self {
        Self {
            entries: [WindowEntry::default(); WINDOW_SIZE],
            head: 0,
            count: 0,
        }
    }

    /// Push a new sentence entry, overwriting the oldest if full.
    pub fn push(&mut self, entry: WindowEntry) {
        self.entries[self.head] = entry;
        self.head = (self.head + 1) % WINDOW_SIZE;
        if self.count < WINDOW_SIZE {
            self.count += 1;
        }
    }

    /// Iterate entries from most recent to oldest (offset 0 = most recent).
    pub fn iter_recent_first(&self) -> impl Iterator<Item = (i8, &WindowEntry)> {
        let count = self.count;
        let head  = self.head;
        (0..count).map(move |i| {
            let slot = (head + WINDOW_SIZE - 1 - i) % WINDOW_SIZE;
            (-(i as i8), &self.entries[slot])
        })
    }

    /// Resolve a pronoun: return the vocabulary rank of the most recent
    /// non-excluded NP head. `exclude_rank` is typically the pronoun's own rank
    /// (we don't want to resolve a pronoun to itself).
    ///
    /// Returns `NO_ROLE` if no candidate is found.
    pub fn resolve_pronoun(&self, exclude_rank: u16) -> u16 {
        // Within each entry iterate in reverse (last-mentioned = highest index = most recent).
        for (_offset, entry) in self.iter_recent_first() {
            for &head in entry.heads().iter().rev() {
                if head != NO_ROLE && head != exclude_rank {
                    return head;
                }
            }
        }
        NO_ROLE
    }

    /// How many entries are in the window.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Is the window empty?
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Most recent entry, or `None` if the window is empty.
    pub fn most_recent(&self) -> Option<&WindowEntry> {
        if self.count == 0 {
            return None;
        }
        let slot = (self.head + WINDOW_SIZE - 1) % WINDOW_SIZE;
        Some(&self.entries[slot])
    }
}

impl Default for SentenceWindow {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SentenceWindow {
    fn clone(&self) -> Self {
        Self {
            entries: self.entries,
            head: self.head,
            count: self.count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(sentence_id: u32, heads: &[u16]) -> WindowEntry {
        let mut e = WindowEntry { sentence_id, ..Default::default() };
        for &h in heads {
            e.push_head(h);
        }
        e
    }

    #[test]
    fn push_and_most_recent() {
        let mut w = SentenceWindow::new();
        w.push(entry(0, &[10, 20]));
        w.push(entry(1, &[30]));
        let r = w.most_recent().unwrap();
        assert_eq!(r.sentence_id, 1);
        assert_eq!(r.heads(), &[30]);
    }

    #[test]
    fn iter_recent_first_order() {
        let mut w = SentenceWindow::new();
        for i in 0..4u32 {
            w.push(entry(i, &[(i * 10) as u16]));
        }
        let ids: Vec<u32> = w.iter_recent_first().map(|(_, e)| e.sentence_id).collect();
        assert_eq!(ids, vec![3, 2, 1, 0]);
    }

    #[test]
    fn ring_wraps_correctly() {
        let mut w = SentenceWindow::new();
        for i in 0..15u32 {
            w.push(entry(i, &[(i * 5) as u16]));
        }
        assert_eq!(w.len(), WINDOW_SIZE);
        // Most recent should be sentence 14
        assert_eq!(w.most_recent().unwrap().sentence_id, 14);
        // iter_recent_first should give 14 downto 4
        let ids: Vec<u32> = w.iter_recent_first().map(|(_, e)| e.sentence_id).collect();
        let expected: Vec<u32> = (4..=14).rev().collect();
        assert_eq!(ids, expected);
    }

    #[test]
    fn resolve_pronoun_returns_most_recent() {
        let mut w = SentenceWindow::new();
        w.push(entry(0, &[100, 200])); // older
        w.push(entry(1, &[300]));      // newer
        // pronoun rank=5 (not in window), exclude it, expect 300 (most recent)
        assert_eq!(w.resolve_pronoun(5), 300);
    }

    #[test]
    fn resolve_pronoun_skips_excluded() {
        let mut w = SentenceWindow::new();
        w.push(entry(0, &[100]));
        w.push(entry(1, &[200])); // most recent has 200
        // If pronoun rank is 200, it should skip 200 and return 100
        assert_eq!(w.resolve_pronoun(200), 100);
    }

    #[test]
    fn resolve_pronoun_empty_returns_no_role() {
        let w = SentenceWindow::new();
        assert_eq!(w.resolve_pronoun(5), NO_ROLE);
    }

    #[test]
    fn head_capacity_does_not_overflow() {
        let mut e = WindowEntry::default();
        for i in 0..10u16 {
            e.push_head(i); // only first 4 stored
        }
        assert_eq!(e.head_count, MAX_HEADS_PER_ENTRY);
        assert_eq!(e.heads(), &[0, 1, 2, 3]);
    }

    #[test]
    fn contains_check() {
        let e = entry(0, &[10, 20, 30]);
        assert!(e.contains(20));
        assert!(!e.contains(99));
    }
}
