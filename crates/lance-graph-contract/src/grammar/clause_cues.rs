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
}
