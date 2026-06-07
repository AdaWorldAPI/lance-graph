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
//!
//! ## Future (v1.5) — Tekamolo/Anaphora64 provenance sidecar
//!
//! v1 records *what* resolved (exact NP-head ranks + expected slots) but not
//! *why* it resolved that way. A future `Anaphora64` sidecar (its own module,
//! NOT folded into `Cam64` or `P64`) should encode coreference provenance:
//!
//! ```text
//! bits  0..11  antecedent_rank_bucket / local entity id
//! bits 12..15  sentence_offset_signed4
//! bits 16..19  source_polarity      (HorizonPolarity: confirmed/expected/inferred_right/basin)
//! bits 20..23  expected_reason      (ExpectedReason: relative/anaphora/ellipsis/causal/temporal)
//! bits 24..31  agreement flags      (number/gender/person/semantic-type/role)
//! bits 32..39  grammatical role score
//! bits 40..47  salience score
//! bits 48..55  confidence q8
//! bits 56..63  reserved / version
//! ```
//!
//! It belongs to the **next** PR (coreference ranking/provenance), not the
//! reader-substrate PR. Add it only once agreement/ranking is implemented, and
//! store it as a provenance field on `EpisodicSpoFrame` (`anaphora_tag: Anaphora64`).
//! The boundary stays clean: SentenceWindow resolves, EpisodicSpoFrame witnesses,
//! Cam64 indexes, Anaphora64 (later) *explains* the resolution.

use crate::spo::NO_ROLE;

/// Maximum number of NP heads tracked per sentence entry.
const MAX_HEADS_PER_ENTRY: usize = 4;

/// Maximum sentences kept in the window (±5 = 11 total, one per slot).
pub const WINDOW_SIZE: usize = 11;

/// Maximum expected (forward-predicted) entities tracked at one time.
const MAX_EXPECTED: usize = 4;

/// Why an entity was pushed into the expected slot.
///
/// Carrying the reason prevents "mystery meat" in resolve_pronoun: callers can
/// filter by reason if they only want to consume a specific prediction type.
/// V1 only uses `RelativeClause` and `Anaphora` in practice; the others are
/// reserved for v2 (ellipsis, causal/temporal continuation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpectedReason {
    /// A relative pronoun ("who", "which") was the left-corner trigger —
    /// the active subject is expected to be the antecedent of the clause.
    RelativeClause,
    /// A personal pronoun was the left-corner trigger — the active subject
    /// is expected to be the referent of the anaphoric pronoun.
    Anaphora,
    /// An omitted subject (pro-drop) — prior subject is expected to continue.
    Ellipsis,
    /// A causal connector ("because", "therefore") — causal agent continues.
    CausalContinuation,
    /// A temporal connector ("then", "after") — temporal anchor continues.
    TemporalContinuation,
}

/// An entity predicted by a left-corner trigger before clause closure.
#[derive(Clone, Copy, Debug)]
pub struct ExpectedSlot {
    /// Vocabulary rank of the predicted entity (NO_ROLE if unused).
    pub rank: u16,
    pub reason: ExpectedReason,
}

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
///
/// ## Forward expectation slots
///
/// When a left-corner trigger fires (relative pronoun, anaphora), the caller
/// pushes the predicted referent into `expected` via `push_expected()`.
/// `resolve_pronoun()` checks these slots first — a confirmed expectation
/// beats any recency heuristic from confirmed sentences.
#[derive(Debug)]
pub struct SentenceWindow {
    /// Fixed-size ring buffer.
    entries: [WindowEntry; WINDOW_SIZE],
    /// Write head (next slot to overwrite).
    head: usize,
    /// Number of valid entries (0..=WINDOW_SIZE).
    count: usize,
    /// Forward-predicted entity slots (Pika left-corner expectations).
    expected: [ExpectedSlot; MAX_EXPECTED],
    /// How many expected slots are active (0..=MAX_EXPECTED).
    expected_count: usize,
}

impl SentenceWindow {
    /// Create an empty window.
    pub fn new() -> Self {
        Self {
            entries: [WindowEntry::default(); WINDOW_SIZE],
            head: 0,
            count: 0,
            expected: [ExpectedSlot { rank: NO_ROLE, reason: ExpectedReason::Anaphora }; MAX_EXPECTED],
            expected_count: 0,
        }
    }

    /// Push a forward-predicted entity into the expectation buffer.
    ///
    /// Called by the reader state machine when a left-corner trigger fires
    /// (relative pronoun, anaphoric pronoun) before the clause closes.
    /// `resolve_pronoun()` drains these slots before consulting the confirmed
    /// sentence ring — expectation beats recency.
    ///
    /// Silently drops if the buffer is full (MAX_EXPECTED = 4).
    pub fn push_expected(&mut self, rank: u16, reason: ExpectedReason) {
        if self.expected_count < MAX_EXPECTED {
            self.expected[self.expected_count] = ExpectedSlot { rank, reason };
            self.expected_count += 1;
        }
    }

    /// Clear all expectation slots.
    ///
    /// `ReadingState::step` calls this at the start of each sentence: v1 treats
    /// every expectation as a **single-step** left-corner prediction, so the
    /// buffer can never accumulate stale slots toward `MAX_EXPECTED`.
    ///
    /// Future (v2, bidirectional / Pika right-context passes): this policy
    /// splits — clear one-step expectations as now, but **retain** multi-step
    /// `HorizonPolarity::InferredRight` slots with a TTL so right-context memo
    /// entries survive across sentences until their window elapses.
    pub fn clear_expected(&mut self) {
        self.expected_count = 0;
    }

    /// Iterate the active expectation slots (most-recently-pushed first).
    pub fn iter_expected(&self) -> &[ExpectedSlot] {
        &self.expected[..self.expected_count]
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

    /// Resolve a pronoun: return the vocabulary rank of the predicted or most
    /// recent non-excluded NP head.
    ///
    /// Resolution order (Pika left-corner priority):
    /// 1. Forward expectation slots (most-recently-pushed first) — these were
    ///    pre-populated by a left-corner trigger and are the strongest signal.
    /// 2. Confirmed sentence ring, most-recent entry first, heads in reverse
    ///    within each entry (last-mentioned in text = highest index).
    ///
    /// `exclude_rank` is typically the pronoun's own vocabulary rank.
    /// Returns `NO_ROLE` if no candidate is found.
    pub fn resolve_pronoun(&self, exclude_rank: u16) -> u16 {
        // Phase 1: forward expectations (Pika left-corner slots).
        for slot in self.iter_expected().iter().rev() {
            if slot.rank != NO_ROLE && slot.rank != exclude_rank {
                return slot.rank;
            }
        }
        // Phase 2: confirmed sentence ring, last-mentioned wins.
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
            expected: self.expected,
            expected_count: self.expected_count,
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

    #[test]
    fn push_expected_stores_slot() {
        let mut w = SentenceWindow::new();
        w.push_expected(42, ExpectedReason::RelativeClause);
        assert_eq!(w.expected_count, 1);
        assert_eq!(w.iter_expected()[0].rank, 42);
        assert_eq!(w.iter_expected()[0].reason, ExpectedReason::RelativeClause);
    }

    #[test]
    fn push_expected_capacity_does_not_overflow() {
        let mut w = SentenceWindow::new();
        for i in 0..10u16 {
            w.push_expected(i, ExpectedReason::Anaphora);
        }
        assert_eq!(w.expected_count, MAX_EXPECTED);
    }

    #[test]
    fn resolve_pronoun_prefers_expected_over_confirmed() {
        let mut w = SentenceWindow::new();
        // Confirmed ring has rank 100.
        w.push(entry(0, &[100]));
        // Forward expectation has rank 200.
        w.push_expected(200, ExpectedReason::RelativeClause);
        // Should return 200 (expectation beats confirmed recency).
        assert_eq!(w.resolve_pronoun(5), 200);
    }

    #[test]
    fn resolve_pronoun_falls_back_to_confirmed_when_expected_empty() {
        let mut w = SentenceWindow::new();
        w.push(entry(0, &[100]));
        // No expectations pushed.
        assert_eq!(w.resolve_pronoun(5), 100);
    }

    #[test]
    fn resolve_pronoun_skips_excluded_in_expected() {
        let mut w = SentenceWindow::new();
        w.push(entry(0, &[100]));
        w.push_expected(200, ExpectedReason::Anaphora);
        // Exclude the expected slot — should fall back to confirmed.
        assert_eq!(w.resolve_pronoun(200), 100);
    }

    #[test]
    fn clear_expected_resets_slots() {
        let mut w = SentenceWindow::new();
        w.push_expected(42, ExpectedReason::Ellipsis);
        w.clear_expected();
        assert_eq!(w.expected_count, 0);
        // resolve_pronoun now falls through to NO_ROLE (empty window).
        assert_eq!(w.resolve_pronoun(5), NO_ROLE);
    }
}
