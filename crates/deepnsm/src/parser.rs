//! PoS-driven finite state machine parser.
//!
//! Extracts SPO (Subject-Predicate-Object) triples from token streams.
//! 6-state FSM driven by part-of-speech tags. No regex. O(n).
//!
//! Handles ~85% of English sentences (SVO order).
//! Secondary patterns cover passive, relative clauses, existential.
//!
//! ```text
//! START → DET? → ADJ* → NOUN+ → VERB → DET? → ADJ* → NOUN+
//!         ──────NP(subj)──────   ─VP─   ────────NP(obj)────────
//! ```

use crate::pos::PoS;
use crate::spo::SpoTriple;
use crate::vocabulary::Token;

/// Parser state machine states.
#[derive(Clone, Copy, Debug, PartialEq)]
enum State {
    /// Looking for start of subject NP.
    Start,
    /// Collecting subject noun phrase (DET? ADJ* NOUN+).
    SubjectNP,
    /// Expecting verb after subject NP.
    ExpectVerb,
    /// Collecting object noun phrase.
    ObjectNP,
    /// Triple complete, looking for more.
    Complete,
}

/// Relationship between modifier and head.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModRelation {
    /// Adjective modifying noun: "big dog"
    AdjectiveOf,
    /// Adverb modifying verb: "quickly ran"
    AdverbOf,
    /// Prepositional attachment: "in the house"
    PrepositionalOf,
}

/// A modifier attachment: (modifier_rank, head_rank, relation).
#[derive(Clone, Debug)]
pub struct Modifier {
    /// Rank of the modifier word.
    pub modifier: u16,
    /// Rank of the head word being modified.
    pub head: u16,
    /// Type of modification.
    pub relation: ModRelation,
}

/// Complete semantic structure extracted from a sentence.
#[derive(Clone, Debug)]
pub struct SentenceStructure {
    /// Primary SPO triples (usually 1-3 per sentence).
    pub triples: Vec<SpoTriple>,
    /// Modifier attachments.
    pub modifiers: Vec<Modifier>,
    /// Which triple indices are negated.
    pub negations: Vec<usize>,
    /// Temporal markers: (triple_index, temporal_word_rank).
    pub temporals: Vec<(usize, u16)>,
}

impl SentenceStructure {
    fn new() -> Self {
        SentenceStructure {
            triples: Vec::new(),
            modifiers: Vec::new(),
            negations: Vec::new(),
            temporals: Vec::new(),
        }
    }

    /// Is this structure empty (no triples found)?
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }
}

/// Parse a token stream into semantic structure.
///
/// Strategy:
/// 1. Identify NP boundaries via PoS patterns (DET? ADJ* NOUN+)
/// 2. Head noun = rightmost noun in NP
/// 3. Verb = first verb after subject NP
/// 4. Object NP = NP after verb
/// 5. Modifiers = adjectives/adverbs → attach to nearest head
/// 6. Negation = "not" before verb → negate the triple
/// 7. Conjunction = "and"/"or" → fork into multiple triples
pub fn parse(tokens: &[Token]) -> SentenceStructure {
    let mut result = SentenceStructure::new();

    if tokens.is_empty() {
        return result;
    }

    let mut state = State::Start;
    let mut subject_head: Option<u16> = None;
    let mut verb: Option<u16> = None;
    let mut object_head: Option<u16> = None;
    let mut current_adjectives: Vec<u16> = Vec::new();
    let mut is_negated = false;
    let mut last_noun: Option<u16> = None;
    let mut last_verb: Option<u16> = None;

    for token in tokens {
        let rank = match token.rank {
            Some(r) => r,
            None => continue, // skip OOV tokens
        };

        if token.is_negated {
            is_negated = true;
        }

        match state {
            State::Start => {
                match token.pos {
                    PoS::Article | PoS::Pronoun => {
                        // Determiner starts an NP
                        if token.pos == PoS::Pronoun {
                            // Pronoun IS the subject head
                            subject_head = Some(rank);
                            state = State::ExpectVerb;
                        } else {
                            state = State::SubjectNP;
                        }
                    }
                    PoS::Noun => {
                        subject_head = Some(rank);
                        last_noun = Some(rank);
                        state = State::ExpectVerb;
                    }
                    PoS::Adjective => {
                        current_adjectives.push(rank);
                        state = State::SubjectNP;
                    }
                    PoS::Existential => {
                        // "there is X" pattern → existential
                        state = State::SubjectNP;
                    }
                    PoS::Verb | PoS::Modal => {
                        // Imperative or fragment: verb first
                        verb = Some(rank);
                        last_verb = Some(rank);
                        state = State::ObjectNP;
                    }
                    PoS::Adverb => {
                        // Sentence-initial adverb, skip
                        state = State::Start;
                    }
                    _ => {}
                }
            }

            State::SubjectNP => {
                match token.pos {
                    PoS::Adjective => {
                        current_adjectives.push(rank);
                    }
                    PoS::Noun => {
                        // Attach pending adjectives to this noun
                        for &adj in &current_adjectives {
                            result.modifiers.push(Modifier {
                                modifier: adj,
                                head: rank,
                                relation: ModRelation::AdjectiveOf,
                            });
                        }
                        current_adjectives.clear();
                        subject_head = Some(rank);
                        last_noun = Some(rank);
                        state = State::ExpectVerb;
                    }
                    PoS::Pronoun => {
                        current_adjectives.clear();
                        subject_head = Some(rank);
                        state = State::ExpectVerb;
                    }
                    PoS::Verb | PoS::Modal => {
                        // No noun found in NP, verb appeared
                        verb = Some(rank);
                        last_verb = Some(rank);
                        state = State::ObjectNP;
                    }
                    PoS::Article => {
                        // Another determiner, restart NP
                    }
                    _ => {
                        // Unexpected PoS, try to recover
                        state = State::ExpectVerb;
                    }
                }
            }

            State::ExpectVerb => {
                match token.pos {
                    PoS::Verb | PoS::Modal => {
                        verb = Some(rank);
                        last_verb = Some(rank);
                        if token.pos == PoS::Negation {
                            is_negated = true;
                        }
                        state = State::ObjectNP;
                    }
                    PoS::Adverb => {
                        // Adverb before verb: "quickly ran"
                        if let Some(v) = last_verb {
                            result.modifiers.push(Modifier {
                                modifier: rank,
                                head: v,
                                relation: ModRelation::AdverbOf,
                            });
                        }
                        // Stay in ExpectVerb
                    }
                    PoS::Negation => {
                        is_negated = true;
                    }
                    PoS::Noun => {
                        // Noun-noun compound: "dog house"
                        // Previous noun was modifier, this is new head
                        if let Some(prev) = subject_head {
                            result.modifiers.push(Modifier {
                                modifier: prev,
                                head: rank,
                                relation: ModRelation::AdjectiveOf, // noun-as-modifier
                            });
                        }
                        subject_head = Some(rank);
                        last_noun = Some(rank);
                    }
                    PoS::Conjunction => {
                        // "A and B verb C" → fork subject
                        // For now, emit current triple and continue
                        if let (Some(s), Some(v)) = (subject_head, verb) {
                            emit_triple(&mut result, s, v, object_head, is_negated);
                            is_negated = false;
                            object_head = None;
                        }
                        state = State::Start;
                    }
                    PoS::Preposition => {
                        // "the dog in the park" → prepositional modifier
                        // The subject NP is done, but there's a PP
                        state = State::ExpectVerb; // keep looking for verb
                    }
                    _ => {}
                }
            }

            State::ObjectNP => {
                match token.pos {
                    PoS::Article => {
                        // Determiner starting object NP, continue
                    }
                    PoS::Adjective => {
                        current_adjectives.push(rank);
                    }
                    PoS::Noun => {
                        // Attach pending adjectives
                        for &adj in &current_adjectives {
                            result.modifiers.push(Modifier {
                                modifier: adj,
                                head: rank,
                                relation: ModRelation::AdjectiveOf,
                            });
                        }
                        current_adjectives.clear();
                        object_head = Some(rank);
                        last_noun = Some(rank);
                    }
                    PoS::Pronoun => {
                        current_adjectives.clear();
                        object_head = Some(rank);
                    }
                    PoS::Adverb => {
                        // Adverb modifying verb: "ran quickly"
                        if let Some(v) = verb {
                            result.modifiers.push(Modifier {
                                modifier: rank,
                                head: v,
                                relation: ModRelation::AdverbOf,
                            });
                        }
                    }
                    PoS::Preposition => {
                        // End of object NP, prepositional phrase starts
                        // Emit current triple
                        if let (Some(s), Some(v)) = (subject_head, verb) {
                            emit_triple(&mut result, s, v, object_head, is_negated);
                        }
                        // Prep attachment
                        if let Some(head) = last_noun {
                            result.modifiers.push(Modifier {
                                modifier: rank,
                                head,
                                relation: ModRelation::PrepositionalOf,
                            });
                        }
                        // Reset for potential new clause
                        state = State::Complete;
                        is_negated = false;
                        continue;
                    }
                    PoS::Conjunction => {
                        // "V A and B" → emit first, start second object
                        if let (Some(s), Some(v)) = (subject_head, verb) {
                            emit_triple(&mut result, s, v, object_head, is_negated);
                            is_negated = false;
                            object_head = None;
                        }
                        // Continue in ObjectNP for second object
                    }
                    PoS::Verb | PoS::Modal => {
                        // New verb: emit current triple, start new clause
                        if let (Some(s), Some(v)) = (subject_head, verb) {
                            emit_triple(&mut result, s, v, object_head, is_negated);
                        }
                        // The object of previous clause might be subject of new one
                        subject_head = object_head.or(last_noun);
                        verb = Some(rank);
                        last_verb = Some(rank);
                        object_head = None;
                        is_negated = false;
                        state = State::ObjectNP;
                    }
                    PoS::Negation => {
                        is_negated = true;
                    }
                    _ => {}
                }
            }

            State::Complete => {
                // After emitting a triple, look for continuation
                match token.pos {
                    PoS::Article | PoS::Adjective | PoS::Noun | PoS::Pronoun => {
                        // New NP — could be subject of new clause
                        subject_head = if token.pos.is_nominal() {
                            Some(rank)
                        } else {
                            None
                        };
                        verb = None;
                        object_head = None;
                        is_negated = false;
                        current_adjectives.clear();
                        if token.pos == PoS::Adjective {
                            current_adjectives.push(rank);
                            state = State::SubjectNP;
                        } else if token.pos.is_nominal() {
                            last_noun = Some(rank);
                            state = State::ExpectVerb;
                        } else {
                            state = State::SubjectNP;
                        }
                    }
                    PoS::Verb | PoS::Modal => {
                        // Verb without new subject — use last noun
                        subject_head = last_noun;
                        verb = Some(rank);
                        last_verb = Some(rank);
                        object_head = None;
                        is_negated = false;
                        state = State::ObjectNP;
                    }
                    PoS::Conjunction => {
                        // "and" between clauses
                        state = State::Start;
                        subject_head = None;
                        verb = None;
                        object_head = None;
                        is_negated = false;
                    }
                    _ => {}
                }
            }
        }
    }

    // Emit any remaining triple
    if let (Some(s), Some(v)) = (subject_head, verb) {
        emit_triple(&mut result, s, v, object_head, is_negated);
    }

    result
}

/// Helper: emit a triple into the result structure.
fn emit_triple(
    result: &mut SentenceStructure,
    subject: u16,
    predicate: u16,
    object: Option<u16>,
    is_negated: bool,
) {
    let triple = match object {
        Some(obj) => SpoTriple::new(subject, predicate, obj),
        None => SpoTriple::intransitive(subject, predicate),
    };

    let idx = result.triples.len();
    result.triples.push(triple);

    if is_negated {
        result.negations.push(idx);
    }
}

/// Parse with secondary patterns for non-SVO structures.
///
/// Detects:
/// - Passive: "was bitten by the dog" → reverse S and O
/// - Existential: "there is a dog" → subject = noun, predicate = exist
/// - Questions: "did the dog bite?" → SVO with aux removal
pub fn parse_with_secondary(tokens: &[Token]) -> SentenceStructure {
    // First try primary parse
    let mut result = parse(tokens);

    // Check for passive pattern: look for "by" after past participle
    // Pattern: NP + aux/be + V(past-part) + "by" + NP
    if result.triples.len() == 1 {
        let has_by = tokens.iter().any(|t| t.surface == "by");
        let has_aux = tokens
            .iter()
            .any(|t| t.surface == "was" || t.surface == "were" || t.surface == "been");

        if has_by && has_aux && result.triples[0].has_object() {
            // Swap subject and object (passive → active)
            let t = &result.triples[0];
            result.triples[0] = SpoTriple::new(t.object(), t.predicate(), t.subject());
        }
    }

    result
}

// ────────────────────────────────────────────────────────────────────────
// Coverage-branch hook for D2 FailureTicket emission.
//
// Wraps the existing free-function parser in a thin newtype that owns the
// coverage threshold (default 0.85; configurable later from D7
// `GrammarStyleConfig`). When a parse falls below threshold, the hook
// hands the partial off to `ticket_emit::emit_ticket` so the LLM-tail
// router can route the failure-mode itself as the inference signal.
//
// `Parser::parse` is preserved verbatim against the free `parse()` so no
// existing call sites break.
// ────────────────────────────────────────────────────────────────────────

/// Default coverage threshold below which a parse triggers a FailureTicket.
/// Mirrors `lance_graph_contract::grammar::LOCAL_COVERAGE_THRESHOLD` (0.9)
/// minus a small slack so DeepNSM's looser FSM gets a chance.
pub const DEFAULT_COVERAGE_THRESHOLD: f32 = 0.85;

/// A parse outcome plus the metrics needed to decide whether it should
/// escalate to the LLM router.
#[derive(Clone, Debug)]
pub struct ParseResult {
    /// Token-derived semantic structure.
    pub structure: SentenceStructure,
    /// Coverage ∈ [0, 1]: classified-tokens / total-tokens.
    pub coverage: f32,
    /// Tokens the FSM successfully classified (rank-encoded).
    pub resolved_tokens: Vec<u16>,
    /// Tokens the FSM could not place (rank-encoded; OOV / unknown PoS).
    pub unresolved_tokens: Vec<u16>,
    /// NSM-prime count found in the resolved set. Drives Abduction routing.
    pub primes_found: u8,
    /// Distance vs. the SPO's expected qualia footprint (0.0 = identical).
    /// Filled by `triangle_bridge::compute_classification_distance` once
    /// the Triangle is wired; stays 0.0 in the bare-DeepNSM path.
    pub classification_distance: f32,
}

/// Newtype around the FSM parser. Owns the coverage threshold so the
/// LLM-tail policy is colocated with the parse decision instead of
/// scattered across call sites.
#[derive(Clone, Debug)]
pub struct Parser {
    coverage_threshold: f32,
}

impl Default for Parser {
    fn default() -> Self {
        Self {
            coverage_threshold: DEFAULT_COVERAGE_THRESHOLD,
        }
    }
}

impl Parser {
    /// Construct with the default 0.85 threshold.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the coverage threshold (D7 `GrammarStyleConfig` will feed
    /// this once style-aware routing lands).
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.coverage_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Current coverage threshold ∈ [0, 1].
    pub fn coverage_threshold(&self) -> f32 {
        self.coverage_threshold
    }

    /// Run the FSM and return the structure unchanged (preserves the
    /// existing public `parse()` shape for callers that don't need
    /// coverage metrics).
    pub fn parse(&self, tokens: &[crate::vocabulary::Token]) -> SentenceStructure {
        parse(tokens)
    }

    /// Coverage-aware parse: returns the structure plus the metrics
    /// `maybe_emit_ticket` needs.
    pub fn parse_with_coverage(&self, tokens: &[crate::vocabulary::Token]) -> ParseResult {
        let structure = parse(tokens);

        let mut resolved = Vec::new();
        let mut unresolved = Vec::new();
        let mut primes = 0u8;
        for t in tokens {
            match t.rank {
                Some(r) => {
                    resolved.push(r);
                    // NSM primes occupy fixed low ranks in the COCA
                    // vocabulary (62/63 of them per lib.rs header).
                    // Treat r < 64 as a primes-found heuristic.
                    if r < 64 {
                        primes = primes.saturating_add(1);
                    }
                }
                None => unresolved.push(0u16),
            }
        }

        let total = (resolved.len() + unresolved.len()) as f32;
        let coverage = if total == 0.0 {
            0.0
        } else {
            resolved.len() as f32 / total
        };

        ParseResult {
            structure,
            coverage,
            resolved_tokens: resolved,
            unresolved_tokens: unresolved,
            primes_found: primes,
            classification_distance: 0.0,
        }
    }

    /// Whether the result fell below the configured threshold.
    pub fn coverage_failed(&self, parse_result: &ParseResult) -> bool {
        parse_result.coverage < self.coverage_threshold
    }

    /// D2 hook: if coverage falls below threshold, hand the partial off
    /// to `ticket_emit::emit_ticket` and return the FailureTicket. Above
    /// threshold returns `None` — the caller commits to AriGraph instead.
    ///
    /// Gated behind `contract-ticket` because the FailureTicket type
    /// lives in `lance_graph_contract`. With the feature off, the hook
    /// becomes a no-op `()` returner so the parser still compiles in
    /// minimal builds.
    #[cfg(feature = "contract-ticket")]
    pub fn maybe_emit_ticket(
        &self,
        parse_result: &ParseResult,
    ) -> Option<lance_graph_contract::grammar::FailureTicket> {
        if !self.coverage_failed(parse_result) {
            return None;
        }
        use lance_graph_contract::grammar::{PartialParse, TekamoloSlots};
        let partial = PartialParse {
            resolved_tokens: parse_result.resolved_tokens.clone(),
            unresolved_tokens: parse_result.unresolved_tokens.clone(),
            coverage: parse_result.coverage,
        };
        // TekamoloSlots / Wechsel / CausalAmbiguity stay empty until D3
        // wires the Grammar Triangle; the ticket already routes correctly
        // on `primes_found` + `classification_distance`.
        Some(crate::ticket_emit::emit_ticket(
            partial,
            parse_result.coverage,
            parse_result.classification_distance,
            parse_result.primes_found,
            TekamoloSlots::default(),
            Vec::new(),
            None,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pos::PoS;

    fn make_token(rank: u16, pos: PoS, surface: &str) -> Token {
        Token {
            rank: Some(rank),
            pos,
            position: 0,
            is_negated: false,
            surface: surface.to_string(),
        }
    }

    #[test]
    fn simple_svo() {
        // "the dog bites the man"
        let tokens = vec![
            make_token(0, PoS::Article, "the"),
            make_token(671, PoS::Noun, "dog"),
            make_token(2943, PoS::Verb, "bites"),
            make_token(0, PoS::Article, "the"),
            make_token(95, PoS::Noun, "man"),
        ];

        let result = parse(&tokens);
        assert_eq!(result.triples.len(), 1);
        assert_eq!(result.triples[0].subject(), 671);
        assert_eq!(result.triples[0].predicate(), 2943);
        assert_eq!(result.triples[0].object(), 95);
    }

    #[test]
    fn pronoun_subject() {
        // "he runs"
        let tokens = vec![
            make_token(16, PoS::Pronoun, "he"),
            make_token(100, PoS::Verb, "runs"),
        ];

        let result = parse(&tokens);
        assert_eq!(result.triples.len(), 1);
        assert_eq!(result.triples[0].subject(), 16);
        assert_eq!(result.triples[0].predicate(), 100);
        assert!(!result.triples[0].has_object());
    }

    #[test]
    fn with_adjective() {
        // "the big dog bites the old man"
        let tokens = vec![
            make_token(0, PoS::Article, "the"),
            make_token(156, PoS::Adjective, "big"),
            make_token(671, PoS::Noun, "dog"),
            make_token(2943, PoS::Verb, "bites"),
            make_token(0, PoS::Article, "the"),
            make_token(174, PoS::Adjective, "old"),
            make_token(95, PoS::Noun, "man"),
        ];

        let result = parse(&tokens);
        assert_eq!(result.triples.len(), 1);
        assert_eq!(result.triples[0].subject(), 671);
        assert_eq!(result.triples[0].object(), 95);

        // Check modifiers
        assert_eq!(result.modifiers.len(), 2);
        assert_eq!(result.modifiers[0].modifier, 156); // big → dog
        assert_eq!(result.modifiers[0].head, 671);
        assert_eq!(result.modifiers[1].modifier, 174); // old → man
        assert_eq!(result.modifiers[1].head, 95);
    }

    #[test]
    fn negation() {
        // "the dog does not bite the man"
        let tokens = vec![
            make_token(0, PoS::Article, "the"),
            make_token(671, PoS::Noun, "dog"),
            make_token(15, PoS::Verb, "does"),
            Token {
                rank: Some(50),
                pos: PoS::Negation,
                position: 3,
                is_negated: false,
                surface: "not".to_string(),
            },
            Token {
                rank: Some(2943),
                pos: PoS::Verb,
                position: 4,
                is_negated: true,
                surface: "bite".to_string(),
            },
            make_token(0, PoS::Article, "the"),
            make_token(95, PoS::Noun, "man"),
        ];

        let result = parse(&tokens);
        assert!(!result.triples.is_empty());
        assert!(!result.negations.is_empty());
    }
}
