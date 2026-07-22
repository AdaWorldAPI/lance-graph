//! `fsm` — the part-of-speech finite-state machine that turns a tagged token
//! stream into [`Spo`] triples. This is the DeepNSM v1 signature, preserved.
//!
//! The states track a minimal English clause: an optional determiner/modifier
//! run, a **subject** noun, a **verb** (predicate), an optional modifier run,
//! then an **object** noun that closes the triple. It is deliberately small and
//! deterministic — the semantics live in [`crate::space`], not here.
//!
//! ## Scope + known blind spot (paper-grounded, 2026-07-22)
//!
//! Full CFG parsing of natural language is combinatorially hostile (Moore 2000:
//! the Penn Treebank grammar averages **7.2×10²⁷ parses/sentence**); this FSM
//! deliberately commits to ONE coarse SPO reading and streams — determinism is
//! a feature at this scope (no recursion → cannot non-terminate). The
//! constituency the FSM omits is carried by the pointer fabric over the stream
//! ([`crate::wave`]; see
//! `.claude/knowledge/left-corner-grammar-tree-pointer-fabric.md`).
//!
//! **Known blind spot: MOVEMENT constructions** (Liu 2025, JLM 13(2)) — object
//! relatives ("the rat that the cat bit"), topicalization ("Cheeses, the rat
//! ate"), wh-fronting. These invert canonical S/O order, so first-noun=subject
//! emits the wrong triple. Logged fork (not built): promote a
//! Relativizer/Complementizer tag out of [`Pos::Other`] as a clause-boundary
//! marker that does NOT positionally reset S/O. The tie-breaks below
//! ("last verb wins", "re-anchor subject") are RECENCY HEURISTICS, not
//! attachment decisions — they will silently mis-bind on coordination and
//! center-embedding, acceptable because this extracts a coarse skeleton, not
//! a parse.

use crate::spo::Spo;
use crate::vocab::WordId;

/// A coarse part-of-speech tag — the six the FSM distinguishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pos {
    /// Determiner / article (`the`, `a`) — skipped.
    Det,
    /// Adjective / modifier — attaches to the next noun (not part of the core SPO).
    Adj,
    /// Noun — a subject or object slot.
    Noun,
    /// Verb — the predicate slot.
    Verb,
    /// Adverb / other — skipped for the core triple.
    Other,
    /// End-of-sentence punctuation — flushes any partial clause.
    Stop,
}

/// One tagged token: its palette [`WordId`] plus its [`Pos`].
#[derive(Debug, Clone, Copy)]
pub struct Tagged {
    /// Palette word id.
    pub id: WordId,
    /// Part of speech.
    pub pos: Pos,
}

impl Tagged {
    /// New tagged token.
    #[must_use]
    pub const fn new(id: WordId, pos: Pos) -> Self {
        Self { id, pos }
    }
}

/// The clause state as the FSM consumes tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Nothing yet / after a flush — waiting for the subject noun.
    Start,
    /// Subject noun captured — waiting for the verb.
    HaveSubject,
    /// Subject + verb captured — waiting for the object noun.
    HaveVerb,
}

/// Parse a tagged token stream into SPO triples via the PoS FSM.
///
/// A triple is emitted whenever an object noun closes a `subject → verb →
/// object` clause; the object then becomes the subject of the next clause
/// (serial-verb chaining, as v1 did). A `Stop` resets to [`State::Start`].
#[must_use]
pub fn parse_to_spo(tokens: &[Tagged]) -> Vec<Spo> {
    let mut out = Vec::new();
    let mut state = State::Start;
    let mut subject: WordId = 0;
    let mut predicate: WordId = 0;

    for t in tokens {
        match (state, t.pos) {
            // Skip determiners, modifiers, adverbs — they are not core slots.
            (_, Pos::Det | Pos::Adj | Pos::Other) => {}
            // A stop flushes the partial clause.
            (_, Pos::Stop) => state = State::Start,

            (State::Start, Pos::Noun) => {
                subject = t.id;
                state = State::HaveSubject;
            }
            (State::HaveSubject, Pos::Verb) => {
                predicate = t.id;
                state = State::HaveVerb;
            }
            (State::HaveVerb, Pos::Noun) => {
                out.push(Spo::new(subject, predicate, t.id));
                // Serial-verb chain: the object seeds the next subject.
                subject = t.id;
                state = State::HaveSubject;
            }
            // A verb before a subject, or a second verb, restarts cleanly.
            (State::Start, Pos::Verb) => {}
            (State::HaveSubject, Pos::Noun) => subject = t.id, // re-anchor subject
            (State::HaveVerb, Pos::Verb) => predicate = t.id,  // last verb wins
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(id: WordId) -> Tagged {
        Tagged::new(id, Pos::Noun)
    }
    fn v(id: WordId) -> Tagged {
        Tagged::new(id, Pos::Verb)
    }
    fn det() -> Tagged {
        Tagged::new(0, Pos::Det)
    }
    fn adj(id: WordId) -> Tagged {
        Tagged::new(id, Pos::Adj)
    }

    #[test]
    fn the_big_dog_bit_the_old_man() {
        // "the big dog bit the old man" → SPO(dog, bit, man).
        let toks = [det(), adj(11), n(101), v(202), det(), adj(12), n(303)];
        let spo = parse_to_spo(&toks);
        assert_eq!(spo, vec![Spo::new(101, 202, 303)]);
    }

    #[test]
    fn serial_verbs_chain_the_object_into_the_next_subject() {
        // "dog bit man saw cat" → (dog,bit,man) then (man,saw,cat).
        let toks = [n(1), v(2), n(3), v(4), n(5)];
        let spo = parse_to_spo(&toks);
        assert_eq!(spo, vec![Spo::new(1, 2, 3), Spo::new(3, 4, 5)]);
    }

    #[test]
    fn stop_flushes_a_partial_clause() {
        // "dog bit ." then "cat ran mouse" → only the second closes a triple.
        let toks = [n(1), v(2), Tagged::new(0, Pos::Stop), n(10), v(20), n(30)];
        let spo = parse_to_spo(&toks);
        assert_eq!(spo, vec![Spo::new(10, 20, 30)]);
    }
}
