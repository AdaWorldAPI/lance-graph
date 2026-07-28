//! `clause_cues` — function-word cue catalogues for **delayed clause
//! commitment** (right-corner / fronted-argument parsing).
//!
//! Two tiny, exact catalogues that let an extractor recognise an *incomplete*
//! clause instead of prematurely committing a left-corner SVO reading:
//!
//! 1. **Pronoun case** ([`pronoun_case`]) — English lost case on nouns, but
//!    pronouns still carry it, and Early Modern English (the KJV register)
//!    carries *more* of it (`ye`/`you`, `thou`/`thee`). An **accusative pronoun
//!    in clause-initial position is a deterministic fronted-object signal**
//!    (`him shall ye hear` — `him` cannot be a subject). The catalogue is
//!    honest about erosion: `you` / `it` / `her` are [`Ambiguous`] — by Early
//!    Modern English `you` had already spread into subject use, and `her` may
//!    be an object pronoun OR a possessive determiner. Case alone never decides
//!    those; a consumer must fall back to other evidence.
//!
//! 2. **Modal auxiliaries** ([`is_modal_aux`], [`modal_tense`]) — the finite
//!    left bracket of an auxiliary chain (`shall … hear`, the
//!    Satzklammer-shaped frame). Recognising the modal is what lets the scan
//!    *await* the right-corner lexical predicate instead of binding the first
//!    verb-looking token; [`modal_tense`] reads the clause tense off the
//!    auxiliary (`shall`→Future), because the lexical verb at the right corner
//!    surfaces as a bare infinitive that [`classify_verb`] would call Present.
//!
//! Both are catalogues, not algorithms (the delayed-commitment scan itself
//! lives consumer-side, e.g. `lance-graph-planner` examples): zero-dep, exact,
//! and deliberately small per the core-gap doctrine — extend the Core with the
//! missing primitive, never hack the consumer.
//!
//! [`Ambiguous`]: PronounCase::Ambiguous
//! [`classify_verb`]: super::verb_lexicon::classify_verb

use super::role_keys::Tense;

/// Case class of an English personal pronoun — the surviving morphological
/// case system that makes fronted-object clauses mechanically decidable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PronounCase {
    /// Unambiguous subject form (`I`, `he`, `she`, `we`, `they`, `thou`, `ye`).
    Nominative,
    /// Unambiguous object form (`me`, `him`, `us`, `them`, `thee`).
    Accusative,
    /// Case-eroded or multi-role (`you`, `it`, `her`) — case alone never
    /// decides these; the consumer must use other evidence (word order,
    /// parallelism, rails).
    Ambiguous,
}

/// Unambiguous nominative (subject-only) pronoun forms. `thou`/`ye` are the
/// Early Modern English (KJV) nominatives — the KJV largely preserves the
/// `ye`(subject)/`you`(object) contrast, which is exactly what makes its
/// fronted clauses MORE decidable than modern English.
const NOMINATIVE: &[&str] = &["i", "he", "she", "we", "they", "thou", "ye"];

/// Unambiguous accusative (object-only) pronoun forms. `thee` is the KJV
/// accusative of `thou`.
const ACCUSATIVE: &[&str] = &["me", "him", "us", "them", "thee"];

/// Case-eroded / multi-role forms: `you` (already both cases by Early Modern
/// English), `it` (identical in both), `her` (object pronoun OR possessive
/// determiner).
const AMBIGUOUS: &[&str] = &["you", "it", "her"];

/// Classify a (lowercased) word's pronoun case, or `None` if it is not a
/// personal pronoun in the catalogue.
///
/// ```
/// use lance_graph_contract::grammar::clause_cues::{pronoun_case, PronounCase};
/// assert_eq!(pronoun_case("him"), Some(PronounCase::Accusative)); // fronted object signal
/// assert_eq!(pronoun_case("ye"), Some(PronounCase::Nominative)); // KJV subject form
/// assert_eq!(pronoun_case("you"), Some(PronounCase::Ambiguous)); // case eroded — never decide on this
/// assert_eq!(pronoun_case("shepherd"), None);
/// ```
#[must_use]
pub fn pronoun_case(word: &str) -> Option<PronounCase> {
    if NOMINATIVE.contains(&word) {
        Some(PronounCase::Nominative)
    } else if ACCUSATIVE.contains(&word) {
        Some(PronounCase::Accusative)
    } else if AMBIGUOUS.contains(&word) {
        Some(PronounCase::Ambiguous)
    } else {
        None
    }
}

/// Modal auxiliaries opening an auxiliary chain (the finite left bracket).
/// Includes the KJV second-person-singular forms `shalt` / `wilt`.
const MODALS: &[(&str, Tense)] = &[
    // future-projecting
    ("shall", Tense::Future),
    ("will", Tense::Future),
    ("shalt", Tense::Future), // KJV: "thou shalt not …"
    ("wilt", Tense::Future),  // KJV: "wilt thou …"
    // potential / subjunctive-role
    ("should", Tense::Potential),
    ("would", Tense::Potential),
    ("may", Tense::Potential),
    ("might", Tense::Potential),
    ("can", Tense::Potential),
    ("could", Tense::Potential),
    ("must", Tense::Potential),
];

/// Is this (lowercased) word a modal auxiliary — the finite left bracket of an
/// auxiliary chain (`shall … hear`)?
#[must_use]
pub fn is_modal_aux(word: &str) -> bool {
    MODALS.iter().any(|(m, _)| *m == word)
}

/// The clause tense a modal auxiliary projects (`shall`→Future,
/// `might`→Potential). The right-corner lexical verb after a modal is a bare
/// infinitive, so the AUXILIARY carries the clause's tense, not the verb.
///
/// ```
/// use lance_graph_contract::grammar::clause_cues::modal_tense;
/// use lance_graph_contract::grammar::role_keys::Tense;
/// assert_eq!(modal_tense("shall"), Some(Tense::Future));
/// assert_eq!(modal_tense("might"), Some(Tense::Potential));
/// assert_eq!(modal_tense("hear"), None); // lexical verb, not a modal
/// ```
#[must_use]
pub fn modal_tense(word: &str) -> Option<Tense> {
    MODALS.iter().find(|(m, _)| *m == word).map(|(_, t)| *t)
}

/// Negation cues — the words that flip a clause's predicate polarity. An
/// extractor seeing one before its predicate observes the statement with LOW
/// frequency (`f ≈ 0`, an explicit invalidation) instead of high — which is
/// what lets the arena HOLD a negation ("they were **not** ashamed",
/// Gen 2:25; "ye shall **not** surely die", Gen 3:4) so a later positive
/// observation of the same statement produces a genuine NARS revision with
/// preserved contradiction depth. A substrate that cannot hold a negation
/// cannot detect a reversal — the PROBE-EYES-OPENED load-bearing gap.
///
/// Exact catalogue, not morphology: `not`/`no`/`never`/`neither`/`nor`, plus
/// the contracted-erosion form `cannot`. Deliberately excludes `without`
/// (prepositional, scopes a noun phrase not the predicate) and `nothing`/
/// `none` (negative pronouns — they fill a slot rather than flip one).
const NEGATIONS: &[&str] = &["not", "no", "never", "neither", "nor", "cannot"];

/// Is this (lowercased) word a predicate-polarity negation cue?
///
/// ```
/// use lance_graph_contract::grammar::clause_cues::is_negation;
/// assert!(is_negation("not"));   // "were not ashamed" — Gen 2:25
/// assert!(is_negation("never"));
/// assert!(!is_negation("naught"));  // archaic noun, fills a slot
/// assert!(!is_negation("knot"));
/// ```
#[must_use]
pub fn is_negation(word: &str) -> bool {
    NEGATIONS.contains(&word)
}

/// Perception / epistemic verbs — the rung-lift operators. `X knew that P`
/// does not merely observe `P`; it mints a rung-1 statement ABOUT `P`
/// (Tarski ladder: knowledge-about is one rung above the known). The
/// catalogue carries surface forms (incl. KJV irregular pasts) because these
/// verbs are RUNG OPERATORS, not relation families — they deliberately do
/// NOT live in `verb_lexicon::FAMILY_LEXICON` (a `knows-that` edge types the
/// EPISTEMIC level, not the TEKAMOLO slot; conflating them would flatten the
/// ladder into the relation plane).
///
/// The self-referential case (`knower == inner subject`: "they knew that
/// THEY were naked", Gen 3:7) is the awareness signature the probe hunts —
/// distinct from ordinary knows-that ("God saw that it was good", Gen 1:4,
/// rung-1 but not reflexive).
const PERCEPTION_VERBS: &[&str] = &[
    "know",
    "knew",
    "knows",
    "knowing",
    "see",
    "saw",
    "sees",
    "perceive",
    "perceived",
    "understand",
    "understood",
    "realize",
    "realized",
    "recognize",
    "recognized",
];

/// Is this (lowercased) word a perception/epistemic verb — a rung-lift
/// operator that, followed by a complementizer (`that`), lifts the ensuing
/// clause one Tarski rung?
///
/// ```
/// use lance_graph_contract::grammar::clause_cues::is_perception_verb;
/// assert!(is_perception_verb("knew"));  // "they knew that they were naked"
/// assert!(is_perception_verb("saw"));   // "God saw ... that it was good"
/// assert!(!is_perception_verb("sewed")); // action verb, no rung lift
/// assert!(!is_perception_verb("hear"));  // perception, but not in the exact
///                                        // catalogue until a probe needs it
/// ```
#[must_use]
pub fn is_perception_verb(word: &str) -> bool {
    PERCEPTION_VERBS.contains(&word)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kjv_case_pairs_are_exact() {
        // The Early Modern English contrasts the KJV preserves.
        assert_eq!(pronoun_case("thou"), Some(PronounCase::Nominative));
        assert_eq!(pronoun_case("thee"), Some(PronounCase::Accusative));
        assert_eq!(pronoun_case("ye"), Some(PronounCase::Nominative));
        // Modern pairs.
        assert_eq!(pronoun_case("he"), Some(PronounCase::Nominative));
        assert_eq!(pronoun_case("him"), Some(PronounCase::Accusative));
        assert_eq!(pronoun_case("they"), Some(PronounCase::Nominative));
        assert_eq!(pronoun_case("them"), Some(PronounCase::Accusative));
    }

    #[test]
    fn eroded_forms_are_ambiguous_never_decisive() {
        // `you` spread into both cases by Early Modern English; `her` doubles
        // as possessive determiner; `it` is identical in both cases.
        for w in ["you", "it", "her"] {
            assert_eq!(
                pronoun_case(w),
                Some(PronounCase::Ambiguous),
                "{w} must be Ambiguous — case alone never decides it"
            );
        }
    }

    #[test]
    fn non_pronouns_are_none() {
        for w in ["shepherd", "hear", "shall", "the"] {
            assert_eq!(pronoun_case(w), None, "{w} is not a personal pronoun");
        }
    }

    #[test]
    fn modal_spine_recognised_with_kjv_forms() {
        assert!(is_modal_aux("shall"));
        assert!(is_modal_aux("shalt"));
        assert!(is_modal_aux("wilt"));
        assert!(!is_modal_aux("hear"));
        assert_eq!(modal_tense("shall"), Some(Tense::Future));
        assert_eq!(modal_tense("shalt"), Some(Tense::Future));
        assert_eq!(modal_tense("would"), Some(Tense::Potential));
        assert_eq!(modal_tense("carries"), None);
    }

    #[test]
    fn negation_cues_fire_on_polarity_flippers_only() {
        // Fire: the Gen 2:25 / 3:4 forms the reversal blade depends on.
        for w in ["not", "no", "never", "neither", "nor", "cannot"] {
            assert!(is_negation(w), "{w} must flip predicate polarity");
        }
        // Stay silent: scoped/slot-filling negatives and lookalikes are NOT
        // polarity flippers — flagging them would invalidate statements the
        // text never denied.
        for w in ["without", "nothing", "none", "naught", "knot", "note"] {
            assert!(!is_negation(w), "{w} must not flip predicate polarity");
        }
    }

    #[test]
    fn perception_verbs_fire_on_rung_operators_only() {
        // Fire: the epistemic verbs that lift a that-clause one rung.
        for w in ["knew", "know", "saw", "perceived", "understood", "realized"] {
            assert!(is_perception_verb(w), "{w} is a rung operator");
        }
        // Stay silent: action verbs — 3:7's OTHER verbs must not rung-lift,
        // else every clause becomes meta and the ladder flattens (a
        // fires-on-everything guard carries zero information).
        for w in ["sewed", "hid", "eat", "took", "made", "opened"] {
            assert!(!is_perception_verb(w), "{w} must not rung-lift");
        }
    }
}
