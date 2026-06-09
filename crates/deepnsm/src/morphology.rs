//! Morphological feature flags — heuristically derived from grammar signals.
//!
//! `MorphFlags` is a 16-bit bitfield. For v1, flags are derived from what
//! `SentenceStructure` already carries: negation, temporal markers, and
//! modal/passive patterns visible in the PoS sequence. Full morphological
//! analysis (number, person, voice) requires a dedicated morphology pass;
//! this module provides the column *shape* for those fields and the
//! heuristic baseline.
//!
//! **Invariant:** these flags describe the *parse frame*, not the reading
//! state. The 64-bit CAM code (`cam64::Cam64`) encodes the reading-state
//! transition; these flags are one of its inputs.

use crate::parser::SentenceStructure;

/// 16-bit morphological feature bitfield.
///
/// Bit layout:
/// ```text
/// bit  0: PAST           (temporal event, heuristic: temporal marker present)
/// bit  1: PRESENT        (atemporal / general)
/// bit  2: FUTURE         (modal present)
/// bit  3: SINGULAR       (number singular)
/// bit  4: PLURAL         (number plural)
/// bit  5: FIRST_PERSON   (I / we)
/// bit  6: SECOND_PERSON  (you)
/// bit  7: THIRD_PERSON   (he / she / it / they, default)
/// bit  8: PASSIVE        (passive construction detected)
/// bit  9: NEGATED        (negation marker in triple)
/// bit 10: INTERROGATIVE  (question marker)
/// bit 11: RELATIVE_CLAUSE (relative pronoun / subordinate clause)
/// bit 12: INFINITIVE     (bare infinitive / to-infinitive)
/// bit 13: SUBORDINATE    (subordinating conjunction)
/// bits 14-15: spare
/// ```
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MorphFlags(u16);

impl MorphFlags {
    pub const PAST: u16 = 1 << 0;
    pub const PRESENT: u16 = 1 << 1;
    pub const FUTURE: u16 = 1 << 2;
    pub const SINGULAR: u16 = 1 << 3;
    pub const PLURAL: u16 = 1 << 4;
    pub const FIRST_PERSON: u16 = 1 << 5;
    pub const SECOND_PERSON: u16 = 1 << 6;
    pub const THIRD_PERSON: u16 = 1 << 7;
    pub const PASSIVE: u16 = 1 << 8;
    pub const NEGATED: u16 = 1 << 9;
    pub const INTERROGATIVE: u16 = 1 << 10;
    pub const RELATIVE_CLAUSE: u16 = 1 << 11;
    pub const INFINITIVE: u16 = 1 << 12;
    pub const SUBORDINATE: u16 = 1 << 13;

    pub fn new(bits: u16) -> Self {
        Self(bits)
    }

    pub fn bits(self) -> u16 {
        self.0
    }

    pub fn has(self, flag: u16) -> bool {
        self.0 & flag != 0
    }
    pub fn set(self, flag: u16) -> Self {
        Self(self.0 | flag)
    }
    pub fn clear(self, flag: u16) -> Self {
        Self(self.0 & !flag)
    }

    pub fn is_past(self) -> bool {
        self.has(Self::PAST)
    }
    pub fn is_present(self) -> bool {
        self.has(Self::PRESENT)
    }
    pub fn is_future(self) -> bool {
        self.has(Self::FUTURE)
    }
    pub fn is_singular(self) -> bool {
        self.has(Self::SINGULAR)
    }
    pub fn is_plural(self) -> bool {
        self.has(Self::PLURAL)
    }
    pub fn is_first_person(self) -> bool {
        self.has(Self::FIRST_PERSON)
    }
    pub fn is_second_person(self) -> bool {
        self.has(Self::SECOND_PERSON)
    }
    pub fn is_third_person(self) -> bool {
        self.has(Self::THIRD_PERSON)
    }
    pub fn is_passive(self) -> bool {
        self.has(Self::PASSIVE)
    }
    pub fn is_negated(self) -> bool {
        self.has(Self::NEGATED)
    }
    pub fn is_interrogative(self) -> bool {
        self.has(Self::INTERROGATIVE)
    }
    pub fn is_relative_clause(self) -> bool {
        self.has(Self::RELATIVE_CLAUSE)
    }
    pub fn is_infinitive(self) -> bool {
        self.has(Self::INFINITIVE)
    }
    pub fn is_subordinate(self) -> bool {
        self.has(Self::SUBORDINATE)
    }

    /// Derive heuristic morph flags from a parsed sentence and triple index.
    ///
    /// V1 heuristics (deterministic, no learned parameters):
    /// - negation list → `NEGATED`
    /// - temporal marker → `PAST` + `THIRD_PERSON` + `SINGULAR`
    /// - neither → `PRESENT` + `THIRD_PERSON` + `SINGULAR` (English default)
    ///
    /// This v1 pass only ever sets the flags above. `FUTURE`, `PLURAL`,
    /// `FIRST_PERSON`, `SECOND_PERSON`, `PASSIVE`, `INTERROGATIVE`,
    /// `RELATIVE_CLAUSE`, `INFINITIVE`, and `SUBORDINATE` are never derived here;
    /// they require a dedicated morphology pass or an explicit `set()` by the caller.
    pub fn from_sentence_structure(s: &SentenceStructure, triple_idx: usize) -> Self {
        let mut flags = Self::default();

        if s.negations.contains(&triple_idx) {
            flags = flags.set(Self::NEGATED);
        }

        let has_temporal = s.temporals.iter().any(|&(ti, _)| ti == triple_idx);

        if has_temporal {
            // V1: temporal marker → treat as past event
            flags = flags
                .set(Self::PAST)
                .set(Self::THIRD_PERSON)
                .set(Self::SINGULAR);
        } else {
            // Default: present atemporal statement
            flags = flags
                .set(Self::PRESENT)
                .set(Self::THIRD_PERSON)
                .set(Self::SINGULAR);
        }

        flags
    }
}

impl core::fmt::Debug for MorphFlags {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut names = Vec::new();
        if self.is_past() {
            names.push("PAST");
        }
        if self.is_present() {
            names.push("PRESENT");
        }
        if self.is_future() {
            names.push("FUTURE");
        }
        if self.is_singular() {
            names.push("SINGULAR");
        }
        if self.is_plural() {
            names.push("PLURAL");
        }
        if self.is_first_person() {
            names.push("1P");
        }
        if self.is_second_person() {
            names.push("2P");
        }
        if self.is_third_person() {
            names.push("3P");
        }
        if self.is_passive() {
            names.push("PASSIVE");
        }
        if self.is_negated() {
            names.push("NEG");
        }
        if self.is_interrogative() {
            names.push("?");
        }
        if self.is_relative_clause() {
            names.push("REL");
        }
        if self.is_infinitive() {
            names.push("INF");
        }
        if self.is_subordinate() {
            names.push("SUB");
        }
        write!(f, "MorphFlags({:?})", names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spo::SpoTriple;

    fn make_sentence(negated: bool, temporal: bool) -> SentenceStructure {
        SentenceStructure {
            triples: vec![SpoTriple::new(1, 2, 3)],
            modifiers: vec![],
            negations: if negated { vec![0] } else { vec![] },
            temporals: if temporal { vec![(0, 42)] } else { vec![] },
        }
    }

    #[test]
    fn negated_triple() {
        let s = make_sentence(true, false);
        let m = MorphFlags::from_sentence_structure(&s, 0);
        assert!(m.is_negated());
        assert!(m.is_present());
    }

    #[test]
    fn temporal_sets_past() {
        let s = make_sentence(false, true);
        let m = MorphFlags::from_sentence_structure(&s, 0);
        assert!(m.is_past());
        assert!(!m.is_present());
    }

    #[test]
    fn default_present_third_singular() {
        let s = make_sentence(false, false);
        let m = MorphFlags::from_sentence_structure(&s, 0);
        assert!(m.is_present());
        assert!(m.is_third_person());
        assert!(m.is_singular());
    }

    #[test]
    fn set_clear_roundtrip() {
        let m = MorphFlags::default()
            .set(MorphFlags::PAST)
            .set(MorphFlags::PLURAL)
            .clear(MorphFlags::PAST);
        assert!(!m.is_past());
        assert!(m.is_plural());
    }

    #[test]
    fn bits_are_distinct() {
        let flags = [
            MorphFlags::PAST,
            MorphFlags::PRESENT,
            MorphFlags::FUTURE,
            MorphFlags::SINGULAR,
            MorphFlags::PLURAL,
            MorphFlags::FIRST_PERSON,
            MorphFlags::SECOND_PERSON,
            MorphFlags::THIRD_PERSON,
            MorphFlags::PASSIVE,
            MorphFlags::NEGATED,
            MorphFlags::INTERROGATIVE,
            MorphFlags::RELATIVE_CLAUSE,
            MorphFlags::INFINITIVE,
            MorphFlags::SUBORDINATE,
        ];
        for i in 0..flags.len() {
            for j in (i + 1)..flags.len() {
                assert_eq!(flags[i] & flags[j], 0, "flags[{i}] and flags[{j}] overlap");
            }
        }
    }
}
