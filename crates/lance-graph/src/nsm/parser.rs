// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! PoS-driven FSM parser for extracting SPO triples from token streams.
//!
//! Implements a simple finite-state machine that recognizes:
//! - Subject NP (article? adjective* noun+)
//! - Verb
//! - Object NP (article? adjective* noun+)
//! - Modifiers (adjectives, adverbs, prepositions)
//! - Negation (not before verb)
//! - Conjunction (and/or forks into multiple triples)

use super::tokenizer::{PoS, Token};

/// FSM parse states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseState {
    /// Initial state, looking for subject NP.
    Start,
    /// Inside subject noun phrase.
    SubjectNP,
    /// Expecting or consuming verb.
    Verb,
    /// Inside object noun phrase.
    ObjectNP,
    /// Consuming modifiers (adjectives, adverbs, prepositions).
    Modifier,
    /// Parsing complete.
    Complete,
}

/// How a modifier relates to its head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModRelation {
    /// Adjective modifying a noun.
    AdjectiveOf,
    /// Adverb modifying a verb.
    AdverbOf,
    /// Prepositional phrase attachment.
    PrepOf,
}

/// An SPO triple packed into a u64 (36 bits used: 12 + 12 + 12).
///
/// Bits [0..12) = subject rank, [12..24) = predicate rank, [24..36) = object rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpoTriple(pub u64);

impl SpoTriple {
    /// Pack three 12-bit ranks into a triple.
    pub fn new(subject: u16, predicate: u16, object: u16) -> Self {
        let s = (subject as u64) & 0xFFF;
        let p = (predicate as u64) & 0xFFF;
        let o = (object as u64) & 0xFFF;
        SpoTriple(s | (p << 12) | (o << 24))
    }

    /// Extract subject rank (bits 0..12).
    pub fn subject(&self) -> u16 {
        (self.0 & 0xFFF) as u16
    }

    /// Extract predicate rank (bits 12..24).
    pub fn predicate(&self) -> u16 {
        ((self.0 >> 12) & 0xFFF) as u16
    }

    /// Extract object rank (bits 24..36).
    pub fn object(&self) -> u16 {
        ((self.0 >> 24) & 0xFFF) as u16
    }
}

/// A modifier attachment: (modifier_rank, head_rank, relation).
#[derive(Debug, Clone, PartialEq)]
pub struct ModifierAttachment {
    /// Rank of the modifier word.
    pub modifier: u16,
    /// Rank of the head word it modifies.
    pub head: u16,
    /// Type of modification.
    pub relation: ModRelation,
}

/// Parsed sentence structure.
#[derive(Debug, Clone)]
pub struct SentenceStructure {
    /// Extracted SPO triples.
    pub triples: Vec<SpoTriple>,
    /// Modifier attachments.
    pub modifiers: Vec<ModifierAttachment>,
    /// Predicate ranks that are negated.
    pub negations: Vec<u16>,
    /// Temporal marker ranks.
    pub temporals: Vec<u16>,
}

impl SentenceStructure {
    /// Create an empty sentence structure.
    fn new() -> Self {
        SentenceStructure {
            triples: Vec::new(),
            modifiers: Vec::new(),
            negations: Vec::new(),
            temporals: Vec::new(),
        }
    }
}

/// Internal NP accumulator.
#[derive(Debug, Clone)]
struct NounPhrase {
    adjectives: Vec<u16>,
    nouns: Vec<u16>,
}

impl NounPhrase {
    fn new() -> Self {
        NounPhrase {
            adjectives: Vec::new(),
            nouns: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.adjectives.clear();
        self.nouns.clear();
    }

    fn is_empty(&self) -> bool {
        self.nouns.is_empty()
    }

    /// Head noun = rightmost noun in the NP.
    fn head(&self) -> Option<u16> {
        self.nouns.last().copied()
    }
}

/// Parse a token stream into a SentenceStructure using a PoS-driven FSM.
///
/// The FSM recognizes patterns of the form:
/// `NP verb NP (and NP verb NP)*`
/// where NP = article? adjective* noun+
pub fn parse(tokens: &[Token]) -> SentenceStructure {
    let mut result = SentenceStructure::new();
    if tokens.is_empty() {
        return result;
    }

    let mut state = ParseState::Start;
    let mut subject_np = NounPhrase::new();
    let mut object_np = NounPhrase::new();
    let mut current_verb: Option<u16> = None;
    let mut current_verb_negated = false;
    let mut last_verb: Option<u16> = None;
    let mut last_subject_head: Option<u16> = None;

    for token in tokens {
        match state {
            ParseState::Start | ParseState::SubjectNP => {
                match token.pos {
                    PoS::D => {
                        // Article/determiner: skip, stay in SubjectNP
                        state = ParseState::SubjectNP;
                    }
                    PoS::J => {
                        // Adjective: accumulate in subject NP
                        subject_np.adjectives.push(token.rank);
                        state = ParseState::SubjectNP;
                    }
                    PoS::N | PoS::P => {
                        // Noun or pronoun: add to subject NP
                        subject_np.nouns.push(token.rank);
                        state = ParseState::SubjectNP;
                    }
                    PoS::V | PoS::M => {
                        // Verb: transition to Verb state
                        if !subject_np.is_empty() {
                            last_subject_head = subject_np.head();
                            // Attach adjectives to head noun
                            if let Some(head) = subject_np.head() {
                                for &adj in &subject_np.adjectives {
                                    result.modifiers.push(ModifierAttachment {
                                        modifier: adj,
                                        head,
                                        relation: ModRelation::AdjectiveOf,
                                    });
                                }
                            }
                        }
                        current_verb = Some(token.rank);
                        current_verb_negated = token.is_negated;
                        if token.is_negated {
                            result.negations.push(token.rank);
                        }
                        state = ParseState::Verb;
                    }
                    PoS::T => {
                        result.temporals.push(token.rank);
                    }
                    PoS::R => {
                        // Adverb before verb: accumulate as modifier
                        // (will attach to verb when we find one)
                    }
                    _ => {
                        // Skip other tokens in start position
                    }
                }
            }

            ParseState::Verb => {
                match token.pos {
                    PoS::R => {
                        // Adverb after verb: attach to verb
                        if let Some(v) = current_verb {
                            result.modifiers.push(ModifierAttachment {
                                modifier: token.rank,
                                head: v,
                                relation: ModRelation::AdverbOf,
                            });
                        }
                    }
                    PoS::D => {
                        // Article: start object NP
                        object_np.clear();
                        state = ParseState::ObjectNP;
                    }
                    PoS::J => {
                        // Adjective: start object NP
                        object_np.clear();
                        object_np.adjectives.push(token.rank);
                        state = ParseState::ObjectNP;
                    }
                    PoS::N | PoS::P => {
                        // Noun: start object NP
                        object_np.clear();
                        object_np.nouns.push(token.rank);
                        state = ParseState::ObjectNP;
                    }
                    PoS::A => {
                        // Preposition after verb: modifier on verb or start prep phrase
                        if let Some(v) = current_verb {
                            result.modifiers.push(ModifierAttachment {
                                modifier: token.rank,
                                head: v,
                                relation: ModRelation::PrepOf,
                            });
                        }
                    }
                    PoS::T => {
                        result.temporals.push(token.rank);
                    }
                    _ => {}
                }
            }

            ParseState::ObjectNP => {
                match token.pos {
                    PoS::D => {
                        // Article inside object NP: skip
                    }
                    PoS::J => {
                        // Adjective: accumulate
                        object_np.adjectives.push(token.rank);
                    }
                    PoS::N | PoS::P => {
                        // Noun: accumulate
                        object_np.nouns.push(token.rank);
                    }
                    PoS::C => {
                        // Conjunction: emit current triple, fork
                        emit_triple(&subject_np, current_verb, &object_np, &mut result);
                        last_verb = current_verb;
                        // Reset for next clause
                        let saved_subject = subject_np.clone();
                        object_np.clear();
                        subject_np.clear();
                        // After conjunction, could be new subject or new object
                        // We speculatively keep the same subject and verb
                        subject_np = saved_subject;
                        state = ParseState::Modifier;
                    }
                    PoS::A => {
                        // Preposition: attach to object head, then look for prep-object
                        if let Some(head) = object_np.head() {
                            result.modifiers.push(ModifierAttachment {
                                modifier: token.rank,
                                head,
                                relation: ModRelation::PrepOf,
                            });
                        }
                        // Emit current triple before prep phrase
                        emit_triple(&subject_np, current_verb, &object_np, &mut result);
                        last_verb = current_verb;
                        current_verb = None;
                        object_np.clear();
                        state = ParseState::Modifier;
                    }
                    PoS::V | PoS::M => {
                        // New verb: emit current triple, start new clause
                        emit_triple(&subject_np, current_verb, &object_np, &mut result);
                        // The object NP becomes the new subject
                        subject_np = object_np.clone();
                        last_subject_head = subject_np.head();
                        object_np.clear();
                        current_verb = Some(token.rank);
                        current_verb_negated = token.is_negated;
                        if token.is_negated {
                            result.negations.push(token.rank);
                        }
                        state = ParseState::Verb;
                    }
                    PoS::R => {
                        // Adverb after object NP: attach to verb
                        if let Some(v) = current_verb {
                            result.modifiers.push(ModifierAttachment {
                                modifier: token.rank,
                                head: v,
                                relation: ModRelation::AdverbOf,
                            });
                        }
                    }
                    PoS::T => {
                        result.temporals.push(token.rank);
                    }
                    _ => {}
                }
            }

            ParseState::Modifier => {
                match token.pos {
                    PoS::D => {
                        // Article: could start new NP
                        // Check if we have a verb pending
                        if current_verb.is_some() {
                            object_np.clear();
                            state = ParseState::ObjectNP;
                        } else {
                            state = ParseState::SubjectNP;
                        }
                    }
                    PoS::J => {
                        if current_verb.is_some() {
                            object_np.adjectives.push(token.rank);
                            state = ParseState::ObjectNP;
                        } else {
                            subject_np.adjectives.push(token.rank);
                            state = ParseState::SubjectNP;
                        }
                    }
                    PoS::N | PoS::P => {
                        if current_verb.is_some() {
                            object_np.nouns.push(token.rank);
                            state = ParseState::ObjectNP;
                        } else {
                            // After conjunction with no new verb, reuse last verb
                            // This handles "cat chases dog and mouse"
                            object_np.nouns.push(token.rank);
                            current_verb = last_verb;
                            state = ParseState::ObjectNP;
                        }
                    }
                    PoS::V | PoS::M => {
                        current_verb = Some(token.rank);
                        current_verb_negated = token.is_negated;
                        if token.is_negated {
                            result.negations.push(token.rank);
                        }
                        state = ParseState::Verb;
                    }
                    PoS::T => {
                        result.temporals.push(token.rank);
                    }
                    _ => {}
                }
            }

            ParseState::Complete => {
                break;
            }
        }
    }

    // Emit final triple if we have accumulated material
    if !object_np.is_empty() || current_verb.is_some() {
        emit_triple(&subject_np, current_verb, &object_np, &mut result);
    }

    // Attach remaining object adjectives
    if let Some(head) = object_np.head() {
        for &adj in &object_np.adjectives {
            result.modifiers.push(ModifierAttachment {
                modifier: adj,
                head,
                relation: ModRelation::AdjectiveOf,
            });
        }
    }

    let _ = current_verb_negated; // suppress unused warning
    let _ = last_subject_head;

    result
}

/// Emit a triple from the current NP/verb/NP accumulation.
fn emit_triple(
    subject_np: &NounPhrase,
    verb: Option<u16>,
    object_np: &NounPhrase,
    result: &mut SentenceStructure,
) {
    let subject = match subject_np.head() {
        Some(s) => s,
        None => return,
    };
    let predicate = match verb {
        Some(v) => v,
        None => return,
    };
    // Object may be empty (intransitive verb)
    let object = object_np.head().unwrap_or(0);

    result
        .triples
        .push(SpoTriple::new(subject, predicate, object));

    // Attach object adjectives to object head
    if let Some(head) = object_np.head() {
        for &adj in &object_np.adjectives {
            result.modifiers.push(ModifierAttachment {
                modifier: adj,
                head,
                relation: ModRelation::AdjectiveOf,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nsm::tokenizer::{test_vocabulary, PoS, Token};

    fn make_token(rank: u16, pos: PoS, position: u16, is_negated: bool) -> Token {
        Token {
            rank,
            pos,
            position,
            is_negated,
        }
    }

    #[test]
    fn test_simple_svo() {
        // "cat chases dog" -> (cat, chases, dog)
        let tokens = vec![
            make_token(50, PoS::N, 0, false), // cat
            make_token(67, PoS::V, 1, false), // chases
            make_token(51, PoS::N, 2, false), // dog
        ];
        let result = parse(&tokens);
        assert_eq!(result.triples.len(), 1);
        assert_eq!(result.triples[0].subject(), 50);
        assert_eq!(result.triples[0].predicate(), 67);
        assert_eq!(result.triples[0].object(), 51);
    }

    #[test]
    fn test_svo_with_article() {
        // "the cat chases the dog"
        let tokens = vec![
            make_token(9, PoS::D, 0, false),  // the
            make_token(50, PoS::N, 1, false), // cat
            make_token(67, PoS::V, 2, false), // chases
            make_token(9, PoS::D, 3, false),  // the
            make_token(51, PoS::N, 4, false), // dog
        ];
        let result = parse(&tokens);
        assert_eq!(result.triples.len(), 1);
        assert_eq!(result.triples[0].subject(), 50);
        assert_eq!(result.triples[0].predicate(), 67);
        assert_eq!(result.triples[0].object(), 51);
    }

    #[test]
    fn test_adjective_modifier() {
        // "big cat chases small dog"
        let tokens = vec![
            make_token(19, PoS::J, 0, false), // big
            make_token(50, PoS::N, 1, false), // cat
            make_token(67, PoS::V, 2, false), // chases
            make_token(20, PoS::J, 3, false), // small
            make_token(51, PoS::N, 4, false), // dog
        ];
        let result = parse(&tokens);
        assert_eq!(result.triples.len(), 1);
        // "big" should be attached to "cat"
        let big_mod = result.modifiers.iter().find(|m| m.modifier == 19).unwrap();
        assert_eq!(big_mod.head, 50); // cat
        assert_eq!(big_mod.relation, ModRelation::AdjectiveOf);
        // "small" should be attached to "dog"
        let small_mod = result.modifiers.iter().find(|m| m.modifier == 20).unwrap();
        assert_eq!(small_mod.head, 51); // dog
        assert_eq!(small_mod.relation, ModRelation::AdjectiveOf);
    }

    #[test]
    fn test_negation() {
        // "cat does not run" -> negated verb
        let tokens = vec![
            make_token(50, PoS::N, 0, false), // cat
            make_token(60, PoS::V, 1, true),  // run (negated)
        ];
        let result = parse(&tokens);
        assert_eq!(result.triples.len(), 1);
        assert!(result.negations.contains(&60));
    }

    #[test]
    fn test_conjunction_forks() {
        // "cat chases dog and mouse" -> two triples
        let tokens = vec![
            make_token(50, PoS::N, 0, false), // cat
            make_token(67, PoS::V, 1, false), // chases
            make_token(51, PoS::N, 2, false), // dog
            make_token(45, PoS::C, 3, false), // and
            make_token(70, PoS::N, 4, false), // mat (standing in for mouse)
        ];
        let result = parse(&tokens);
        assert_eq!(result.triples.len(), 2);
        // First triple: cat chases dog
        assert_eq!(result.triples[0].subject(), 50);
        assert_eq!(result.triples[0].predicate(), 67);
        assert_eq!(result.triples[0].object(), 51);
        // Second triple: cat chases mat
        assert_eq!(result.triples[1].subject(), 50);
        assert_eq!(result.triples[1].predicate(), 67);
        assert_eq!(result.triples[1].object(), 70);
    }

    #[test]
    fn test_adverb_modifier() {
        // "cat runs quickly"
        let tokens = vec![
            make_token(50, PoS::N, 0, false), // cat
            make_token(60, PoS::V, 1, false), // runs
            make_token(66, PoS::R, 2, false), // quickly
        ];
        let result = parse(&tokens);
        assert_eq!(result.triples.len(), 1);
        let adv_mod = result.modifiers.iter().find(|m| m.modifier == 66).unwrap();
        assert_eq!(adv_mod.head, 60); // run
        assert_eq!(adv_mod.relation, ModRelation::AdverbOf);
    }

    #[test]
    fn test_triple_packing() {
        let t = SpoTriple::new(100, 200, 300);
        assert_eq!(t.subject(), 100);
        assert_eq!(t.predicate(), 200);
        assert_eq!(t.object(), 300);
    }

    #[test]
    fn test_triple_12bit_mask() {
        // Values > 4095 should be masked
        let t = SpoTriple::new(0xFFF, 0xFFF, 0xFFF);
        assert_eq!(t.subject(), 4095);
        assert_eq!(t.predicate(), 4095);
        assert_eq!(t.object(), 4095);

        let t2 = SpoTriple::new(0x1FFF, 0x1FFF, 0x1FFF);
        assert_eq!(t2.subject(), 4095);
        assert_eq!(t2.predicate(), 4095);
        assert_eq!(t2.object(), 4095);
    }

    #[test]
    fn test_empty_input() {
        let result = parse(&[]);
        assert!(result.triples.is_empty());
        assert!(result.modifiers.is_empty());
    }

    #[test]
    fn test_intransitive_verb() {
        // "cat runs" -> triple with object = 0
        let tokens = vec![
            make_token(50, PoS::N, 0, false), // cat
            make_token(60, PoS::V, 1, false), // runs
        ];
        let result = parse(&tokens);
        assert_eq!(result.triples.len(), 1);
        assert_eq!(result.triples[0].subject(), 50);
        assert_eq!(result.triples[0].predicate(), 60);
        assert_eq!(result.triples[0].object(), 0);
    }

    #[test]
    fn test_integrated_tokenizer_parser() {
        let v = test_vocabulary();
        let tokens = v.tokenize("the big cat chases the small dog");
        let result = parse(&tokens);
        assert_eq!(result.triples.len(), 1);
        assert_eq!(result.triples[0].subject(), 50); // cat
        assert_eq!(result.triples[0].predicate(), 67); // chases
        assert_eq!(result.triples[0].object(), 51); // dog
    }
}
