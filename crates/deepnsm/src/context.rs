//! Streaming context window for local disambiguation.
//!
//! Replaces transformer self-attention with a ±5 sentence sliding window.
//! Each sentence is VSA-encoded and pushed into a ring buffer.
//! The running bundle (majority vote) IS the context.
//!
//! XOR-binding a word with the context shifts its representation toward
//! the contextually appropriate meaning:
//! - "bank" + financial context bundle → financial-colored vector
//! - "bank" + river context bundle → river-colored vector
//!
//! O(1) per sentence update, no recomputation of previous sentences.

use crate::encoder::{self, VsaVec, RoleVectors, bundle};

/// Default context window size: ±5 sentences = 11 total.
pub const DEFAULT_WINDOW_SIZE: usize = 11;

/// A streaming context window that maintains a running VSA bundle
/// over the most recent N sentences.
pub struct ContextWindow {
    /// Ring buffer of sentence vectors.
    buffer: Vec<Option<VsaVec>>,
    /// Current write position in the ring buffer.
    head: usize,
    /// Number of sentences stored (up to capacity).
    count: usize,
    /// Cached running bundle (invalidated on push).
    cached_bundle: Option<VsaVec>,
}

impl ContextWindow {
    /// Create a new context window with given capacity.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        ContextWindow {
            buffer: vec![None; capacity],
            head: 0,
            count: 0,
            cached_bundle: None,
        }
    }

    /// Create with default ±5 sentence window (11 slots).
    pub fn default_window() -> Self {
        Self::new(DEFAULT_WINDOW_SIZE)
    }

    /// Push a sentence vector into the window.
    /// Overwrites the oldest sentence if full.
    pub fn push(&mut self, sentence_vec: VsaVec) {
        self.buffer[self.head] = Some(sentence_vec);
        self.head = (self.head + 1) % self.buffer.len();
        if self.count < self.buffer.len() {
            self.count += 1;
        }
        self.cached_bundle = None; // invalidate cache
    }

    /// Push a sentence by encoding its triples and modifiers.
    pub fn push_sentence(
        &mut self,
        structure: &crate::parser::SentenceStructure,
        roles: &RoleVectors,
    ) {
        if structure.is_empty() {
            return;
        }

        let mut components = Vec::new();

        for (i, triple) in structure.triples.iter().enumerate() {
            let is_negated = structure.negations.contains(&i);
            let vec = if is_negated {
                encoder::encode_triple_negated(
                    triple.subject(),
                    triple.predicate(),
                    if triple.has_object() {
                        Some(triple.object())
                    } else {
                        None
                    },
                    roles,
                )
            } else {
                encoder::encode_triple(
                    triple.subject(),
                    triple.predicate(),
                    if triple.has_object() {
                        Some(triple.object())
                    } else {
                        None
                    },
                    roles,
                )
            };
            components.push(vec);
        }

        for modifier in &structure.modifiers {
            components.push(encoder::encode_modifier(
                modifier.modifier,
                modifier.head,
                roles,
            ));
        }

        if !components.is_empty() {
            self.push(bundle(&components));
        }
    }

    /// Get the current context bundle (majority vote over window).
    ///
    /// This IS the context — the superposition of recent sentences.
    /// Returns None if the window is empty.
    pub fn context(&mut self) -> Option<&VsaVec> {
        if self.count == 0 {
            return None;
        }

        if self.cached_bundle.is_none() {
            let active: Vec<VsaVec> = self
                .buffer
                .iter()
                .filter_map(|slot| slot.clone())
                .collect();

            if active.is_empty() {
                return None;
            }

            self.cached_bundle = Some(bundle(&active));
        }

        self.cached_bundle.as_ref()
    }

    /// Disambiguate a word using current context.
    ///
    /// Returns a context-colored word vector:
    /// `word_vec ⊕ context_bundle → contextualized representation`
    ///
    /// The result is closer to the contextually appropriate meaning:
    /// - "bank" in financial context → closer to "money", "account"
    /// - "bank" in river context → closer to "river", "water"
    pub fn disambiguate(&mut self, word_rank: u16) -> VsaVec {
        let word_vec = VsaVec::from_rank(word_rank);

        match self.context() {
            Some(ctx) => word_vec.bind(ctx),
            None => word_vec,
        }
    }

    /// Number of sentences currently in the window.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Is the window empty?
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Is the window full?
    pub fn is_full(&self) -> bool {
        self.count == self.buffer.len()
    }

    /// Window capacity.
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Clear the window.
    pub fn clear(&mut self) {
        for slot in self.buffer.iter_mut() {
            *slot = None;
        }
        self.head = 0;
        self.count = 0;
        self.cached_bundle = None;
    }

    /// Approximate byte size of the window.
    pub fn byte_size(&self) -> usize {
        self.buffer.len() * (encoder::VSA_WORDS * 8 + 8) // VsaVec + Option overhead
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::VsaVec;

    #[test]
    fn empty_window() {
        let mut ctx = ContextWindow::new(5);
        assert!(ctx.is_empty());
        assert!(ctx.context().is_none());
    }

    #[test]
    fn push_and_context() {
        let mut ctx = ContextWindow::new(3);

        ctx.push(VsaVec::random(1));
        assert_eq!(ctx.len(), 1);
        assert!(ctx.context().is_some());

        ctx.push(VsaVec::random(2));
        ctx.push(VsaVec::random(3));
        assert_eq!(ctx.len(), 3);
        assert!(ctx.is_full());
    }

    #[test]
    fn ring_buffer_wraps() {
        let mut ctx = ContextWindow::new(3);

        ctx.push(VsaVec::random(1));
        ctx.push(VsaVec::random(2));
        ctx.push(VsaVec::random(3));
        assert!(ctx.is_full());

        // Push one more — should overwrite oldest
        ctx.push(VsaVec::random(4));
        assert_eq!(ctx.len(), 3);
    }

    #[test]
    fn disambiguation_changes_vector() {
        let mut ctx = ContextWindow::new(5);

        // Add some "financial" context
        ctx.push(VsaVec::from_rank(500));  // "money"
        ctx.push(VsaVec::from_rank(600));  // "account"
        ctx.push(VsaVec::from_rank(700));  // "invest"

        let plain = VsaVec::from_rank(100); // "bank"
        let disambiguated = ctx.disambiguate(100);

        // Disambiguated should be different from plain
        assert!(plain.similarity(&disambiguated) < 0.5);
    }

    #[test]
    fn clear_resets() {
        let mut ctx = ContextWindow::new(5);
        ctx.push(VsaVec::random(1));
        ctx.push(VsaVec::random(2));

        ctx.clear();
        assert!(ctx.is_empty());
        assert!(ctx.context().is_none());
    }
}
