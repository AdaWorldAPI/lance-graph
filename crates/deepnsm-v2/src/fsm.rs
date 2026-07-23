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
//! **MOVEMENT constructions** (Liu 2025, JLM 13(2)) — object relatives ("the rat
//! that the cat bit"), topicalization, wh-fronting — invert canonical S/O order,
//! so a naive first-noun=subject emits the wrong triple. The cheapest
//! mitigation (logged fork, now BUILT 2026-07-23) is here: a
//! [`Pos::Rel`] relativizer/complementizer tag opens a single-level relative
//! clause whose embedded subject does NOT overwrite the matrix subject — the
//! relativizer's antecedent IS the matrix subject, exactly the FSM-side feeder
//! for the ±8 antecedent pointer in [`crate::wave`]. Object-relative
//! ("rat that cat bit ate cheese") and subject-relative ("dog that chased cat
//! barked man") both preserve the matrix subject through the embedded clause;
//! the embedded S-V-O emits its own triple. STILL out of scope (recency
//! heuristics, not attachment): coordination, nested/center-embedded relatives
//! (>1 level), topicalization, wh-fronting — the "last verb wins" / "re-anchor
//! subject" tie-breaks below will silently mis-bind those. This extracts a
//! coarse skeleton + the one commonest movement, not a parse.

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
    /// Relativizer / complementizer (`that`, `which`, `who`, `whom`, `whose`) —
    /// a clause-boundary marker. Promoted out of [`Pos::Other`] (2026-07-23) so
    /// the embedded clause's subject does NOT positionally overwrite the matrix
    /// subject (the movement blind spot). The relativizer's referent is the
    /// matrix subject (its antecedent); see `parse_to_spo`'s single-level
    /// relative-clause handling. This is the FSM-side feeder for the ±8
    /// antecedent pointer in [`crate::wave`] — a cheap positional tag, not the
    /// full gap→filler resolution.
    Rel,
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

/// The relative-clause sub-machine (single level). Opened by a [`Pos::Rel`]
/// relativizer while a matrix subject is held; closed when the embedded clause
/// resolves, restoring the matrix subject.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rel {
    /// No relative clause open.
    None,
    /// Relativizer just seen; the antecedent is the matrix subject. Awaiting an
    /// embedded subject noun (object-relative) or an embedded verb
    /// (subject-relative, antecedent = embedded subject).
    Open,
    /// Object-relative: the embedded subject noun is captured; awaiting the
    /// embedded verb (whose object is the antecedent).
    ObjSubject(WordId),
    /// The embedded verb is captured (subject-relative path); awaiting the
    /// embedded object noun, or the matrix verb which closes an intransitive
    /// embedded clause with no triple.
    HaveVerb { subj: WordId, verb: WordId },
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
    // Single-level relative clause. When a relativizer opens one, `matrix`
    // parks the untouched matrix subject and `antecedent` records what the
    // relativizer refers to (the matrix subject). The embedded clause resolves
    // into its own triple without clobbering `subject`/`state`; on close, the
    // matrix subject resumes in `HaveSubject`.
    let mut rel = Rel::None;
    let mut matrix: WordId = 0;
    let mut antecedent: WordId = 0;

    for t in tokens {
        // A stop flushes everything, matrix and embedded alike.
        if t.pos == Pos::Stop {
            state = State::Start;
            rel = Rel::None;
            continue;
        }

        // While a relative clause is open, its own tiny machine consumes the
        // embedded S-V-O; the matrix subject stays parked in `matrix`.
        if rel != Rel::None {
            match (rel, t.pos) {
                (_, Pos::Stop) => unreachable!("Stop handled before the rel block"),
                (_, Pos::Det | Pos::Adj | Pos::Other) => {}
                // Object-relative: a noun after the relativizer is the embedded
                // subject ("rat that [cat] bit …"); the antecedent is its object.
                (Rel::Open, Pos::Noun) => rel = Rel::ObjSubject(t.id),
                // Subject-relative: a verb right after the relativizer means the
                // antecedent is the embedded subject ("dog that [chased] …").
                (Rel::Open, Pos::Verb) => {
                    rel = Rel::HaveVerb {
                        subj: antecedent,
                        verb: t.id,
                    };
                }
                // Object-relative embedded verb: emit (embedded_subj, verb,
                // antecedent) and close — the antecedent IS the object.
                (Rel::ObjSubject(es), Pos::Verb) => {
                    out.push(Spo::new(es, t.id, antecedent));
                    subject = matrix;
                    state = State::HaveSubject;
                    rel = Rel::None;
                }
                // Object-relative saw a second noun before its verb — treat the
                // newest as the embedded subject (recency), stay open.
                (Rel::ObjSubject(_), Pos::Noun) => rel = Rel::ObjSubject(t.id),
                (Rel::Open, Pos::Rel) => {} // stray relativizer — ignore
                (Rel::ObjSubject(_), Pos::Rel) => {}
                // Subject-relative embedded object closes the embedded triple.
                (Rel::HaveVerb { subj, verb }, Pos::Noun) => {
                    out.push(Spo::new(subj, verb, t.id));
                    subject = matrix;
                    state = State::HaveSubject;
                    rel = Rel::None;
                }
                // A second verb (no embedded object seen) is the MATRIX verb:
                // close the (intransitive → no triple) embedded clause and let
                // the matrix subject take this verb as its predicate.
                (Rel::HaveVerb { .. }, Pos::Verb) => {
                    subject = matrix;
                    predicate = t.id;
                    state = State::HaveVerb;
                    rel = Rel::None;
                }
                (Rel::HaveVerb { .. }, Pos::Rel) => {}
                (Rel::None, _) => unreachable!("guarded by rel != Rel::None"),
            }
            continue;
        }

        match (state, t.pos) {
            // Skip determiners, modifiers, adverbs — they are not core slots.
            (_, Pos::Det | Pos::Adj | Pos::Other) => {}
            (_, Pos::Stop) => unreachable!("handled above"),

            // A relativizer only opens a relative clause when we already have a
            // matrix subject for it to modify; elsewhere it is inert (skipped
            // like Other), never resetting S/O.
            (State::HaveSubject, Pos::Rel) => {
                matrix = subject;
                antecedent = subject;
                rel = Rel::Open;
            }
            (State::Start | State::HaveVerb, Pos::Rel) => {}

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

    fn rel() -> Tagged {
        Tagged::new(0, Pos::Rel)
    }

    #[test]
    fn object_relative_preserves_the_matrix_subject() {
        // "the rat that the cat bit ate the cheese"
        //   rat=1 cat=2 bit=3 ate=4 cheese=5
        // Correct: (cat, bit, rat) embedded, then (rat, ate, cheese) matrix.
        // The BUG this fixes: without Pos::Rel, `cat` re-anchors the subject and
        // the matrix triple becomes the wrong (cat, ate, cheese).
        let toks = [det(), n(1), rel(), det(), n(2), v(3), v(4), det(), n(5)];
        let spo = parse_to_spo(&toks);
        assert_eq!(spo, vec![Spo::new(2, 3, 1), Spo::new(1, 4, 5)]);
    }

    #[test]
    fn subject_relative_preserves_the_matrix_subject() {
        // "the dog that chased the cat barked the man" (at→dropped)
        //   dog=1 chased=2 cat=3 barked=4 man=5
        // Correct: (dog, chased, cat) embedded, then (dog, barked, man) matrix.
        let toks = [det(), n(1), rel(), v(2), det(), n(3), v(4), det(), n(5)];
        let spo = parse_to_spo(&toks);
        assert_eq!(spo, vec![Spo::new(1, 2, 3), Spo::new(1, 4, 5)]);
    }

    #[test]
    fn intransitive_subject_relative_emits_only_the_matrix_triple() {
        // "the man who slept woke the child"
        //   man=1 slept=2 woke=3 child=4
        // "slept" has no object → no embedded triple; matrix (man, woke, child).
        let toks = [det(), n(1), rel(), v(2), v(3), det(), n(4)];
        let spo = parse_to_spo(&toks);
        assert_eq!(spo, vec![Spo::new(1, 3, 4)]);
    }

    #[test]
    fn relativizer_without_a_matrix_subject_is_inert() {
        // A relativizer at Start (no antecedent) must not corrupt the next clause.
        let toks = [rel(), n(1), v(2), n(3)];
        let spo = parse_to_spo(&toks);
        assert_eq!(spo, vec![Spo::new(1, 2, 3)]);
    }
}
