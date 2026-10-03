//! `coca` — the COCA adapter onto the FSM alphabet.
//!
//! [`crate::lexical`] keeps COCA part-of-speech letters exactly as the source
//! wrote them ([`PosCode`]), and [`crate::fsm`] parses over its own
//! language-neutral alphabet ([`Pos`], [`PosSet`]). This module is the one
//! place a COCA letter becomes an FSM tag, so every English consumer folds
//! the same way. A lexicon for another language gets its own adapter beside
//! this one; neither the FSM nor the evidence store changes.
//!
//! The fold is lossy by design — `n` and `p` both become [`Pos::Noun`] —
//! which is why it lives at this boundary and not inside [`crate::lexical`].
//!
//! COCA letters, as used in `lemmas_5k.csv` / `word_forms.csv`: `a` article,
//! `d` determiner (`this`, `which`, `all`, `some`; modals are `v`), `n`
//! noun, `p` pronoun, `v` verb, `j` adjective, anything else (`r` adverb,
//! `i` preposition, `c` conjunction, …) is outside the FSM's core slots.

use crate::fsm::{Pos, PosSet};
use crate::lexical::{LexicalEvidence, PosCode};
use crate::vocab::WordId;

/// The FSM tag for one COCA PoS letter.
#[must_use]
pub const fn fsm_pos(code: PosCode) -> Pos {
    match code.0 {
        b'n' | b'p' => Pos::Noun,
        b'v' => Pos::Verb,
        b'j' => Pos::Adj,
        b'a' | b'd' => Pos::Det,
        _ => Pos::Other,
    }
}

/// [`fsm_pos`] for a letter given as text. Anything that is not exactly one
/// ASCII letter maps to [`Pos::Other`], as an unrecognised letter does.
#[must_use]
pub fn fsm_pos_tag(tag: &str) -> Pos {
    PosCode::from_tag(tag).map_or(Pos::Other, fsm_pos)
}

/// Every FSM reading word `id` has in `evidence`, folded through
/// [`fsm_pos`].
///
/// `None` when the evidence holds no reading for `id`: the word is
/// lexically unknown, which is not the same as a reading observed zero times.
/// A reading whose count is zero or unknown still counts as a reading — the
/// set says which readings exist, never how often.
#[must_use]
pub fn reading_set(evidence: &LexicalEvidence, id: WordId) -> Option<PosSet> {
    let readings = evidence.readings(id);
    if readings.is_empty() {
        return None;
    }
    Some(readings.iter().map(|r| fsm_pos(r.pos)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::{LexicalEvidenceBuilder, LexicalReading};
    use crate::vocab::PaletteVocab;

    #[test]
    fn letters_fold_onto_the_fsm_alphabet() {
        let cases = [
            ("n", Pos::Noun),
            ("p", Pos::Noun),
            ("v", Pos::Verb),
            ("j", Pos::Adj),
            ("a", Pos::Det),
            ("d", Pos::Det),
            ("r", Pos::Other),
            ("i", Pos::Other),
            ("c", Pos::Other),
            ("", Pos::Other),
            ("nn", Pos::Other),
        ];
        for (tag, want) in cases {
            assert_eq!(fsm_pos_tag(tag), want, "letter {tag:?}");
        }
    }

    fn evidence(
        words: &[&str],
        rows: &[(&str, u8, Option<u64>)],
    ) -> (PaletteVocab, LexicalEvidence) {
        let mut vocab = PaletteVocab::new();
        vocab.from_frequency_ranked(words.iter().copied());
        let mut b = LexicalEvidenceBuilder::new(&vocab);
        for &(w, pos, count) in rows {
            b.add_reading(
                vocab.id(w).expect("in vocab"),
                LexicalReading {
                    pos: PosCode(pos),
                    lemma: None,
                    form_count: count,
                },
            )
            .expect("add reading");
        }
        (vocab, b.finish())
    }

    /// T1 at the boundary: a noun/verb homograph yields both readings.
    #[test]
    fn a_homograph_yields_every_reading() {
        let (v, e) = evidence(
            &["record"],
            &[
                ("record", b'n', Some(120_048)),
                ("record", b'v', Some(13_014)),
            ],
        );
        let set = reading_set(&e, v.id("record").unwrap()).unwrap();
        assert_eq!(set, PosSet::single(Pos::Noun).with(Pos::Verb));
    }

    /// Readings that fold together stay one reading: `n` and `p` are both a
    /// noun slot.
    #[test]
    fn folded_readings_collapse_in_the_set() {
        let (v, e) = evidence(&["it"], &[("it", b'p', Some(5)), ("it", b'n', Some(1))]);
        assert_eq!(
            reading_set(&e, v.id("it").unwrap()),
            Some(PosSet::single(Pos::Noun))
        );
    }

    /// T6: no reading is `None`; a reading counted zero, or not counted, is
    /// still a reading.
    #[test]
    fn unknown_is_not_zero() {
        let (v, e) = evidence(
            &["seen", "zero", "uncounted"],
            &[
                ("seen", b'n', Some(3)),
                ("zero", b'v', Some(0)),
                ("uncounted", b'j', None),
            ],
        );
        assert_eq!(
            reading_set(&e, v.id("seen").unwrap()),
            Some(PosSet::single(Pos::Noun))
        );
        assert_eq!(
            reading_set(&e, v.id("zero").unwrap()),
            Some(PosSet::single(Pos::Verb))
        );
        assert_eq!(
            reading_set(&e, v.id("uncounted").unwrap()),
            Some(PosSet::single(Pos::Adj))
        );
        let mut absent_vocab = v.clone();
        absent_vocab.from_frequency_ranked(["absent"]);
        let absent = absent_vocab.id("absent").unwrap();
        assert_eq!(reading_set(&e, absent), None);
    }
}
