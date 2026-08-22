//! One POS lexicon, loaded from the COCA tables — replacing three hand-copied
//! taggers.
//!
//! # Why this module exists
//!
//! `bible_wave`, `genre_shapes` and `toc_hydrate` each carried their own
//! `coca_pos` + `archaic_pos` + `pos_of` map, hand-copied between them. That
//! has already cost real yield twice:
//!
//! 1. A paraphrase of `archaic_pos` in one copy dropped the explicit KJV word
//!    list and silently lost ~6.5k triples (34,277 vs 40,767) before the corpus
//!    caught it.
//! 2. **All three copies read `lemmas_5k.csv` only.** That table is keyed by
//!    LEMMA (`create`, `make`, `call`) while the KJV text is INFLECTED
//!    (`created`, `made`, `called`), so every ordinary past-tense verb fell to
//!    [`Pos::Other`] and the FSM skipped it. `word_forms.csv` — the COCA
//!    form→lemma+PoS table, 11,460 rows, sitting in the same directory —
//!    answers exactly that and was never opened.
//!
//! Measured on the 31,102-verse KJV: consulting the forms table takes verses
//! that yield no triple from **11,080 (35.6%) to 3,803 (12.2%)**. Genesis 1:1
//! — *"In the beginning God created the heaven and the earth"*, every word in
//! vocabulary — was one of the barren ones, because `created` had no tag.
//!
//! # Resolution order, and why it is monotone
//!
//! `lemma table → forms table → archaic → Other`.
//!
//! Lemma-first is deliberate: it means **no word that already had a tag gets a
//! different one**. The forms table can only fill silence, never overrule. So
//! adding it is strictly additive against every prior measurement, which is
//! what makes the 11,080 → 3,803 delta attributable to coverage rather than to
//! a re-tagging. [`tests::adding_forms_never_retags_a_word_the_lemmas_knew`]
//! pins that property.

use crate::fsm::Pos;
use std::collections::HashMap;

/// Map a COCA part-of-speech letter to a [`Pos`].
///
/// `n`/`p` are noun-ish, `v` verbal, `j` adjectival, `a`/`d` determiners.
/// Everything else is [`Pos::Other`] — the FSM skips it.
#[must_use]
pub fn coca_pos(letter: &str) -> Pos {
    match letter {
        "n" | "p" => Pos::Noun,
        "v" => Pos::Verb,
        "j" => Pos::Adj,
        "a" | "d" => Pos::Det,
        _ => Pos::Other,
    }
}

/// Early-modern English forms COCA does not carry.
///
/// The explicit word list is load-bearing and must not be paraphrased down to
/// the `-eth`/`-est` rule: `thou`/`hath`/`shall`/`saith` are among the most
/// frequent tokens in the corpus and none of them matches that suffix.
#[must_use]
pub fn archaic_pos(w: &str) -> Option<Pos> {
    match w {
        "thou" | "thee" | "ye" => Some(Pos::Noun),
        "thy" | "thine" => Some(Pos::Det),
        "shalt" | "hath" | "doth" | "saith" | "spake" | "begat" | "art" | "wilt" | "hast"
        | "shall" | "cometh" | "wast" => Some(Pos::Verb),
        "unto" | "thereof" | "wherefore" | "verily" | "yea" | "lo" => Some(Pos::Other),
        _ => {
            if w.ends_with("eth") || w.ends_with("est") {
                Some(Pos::Verb)
            } else {
                None
            }
        }
    }
}

/// A word → [`Pos`] lexicon over the COCA lemma and form tables.
#[derive(Debug, Clone, Default)]
pub struct Lexicon {
    lemmas: HashMap<String, Pos>,
    forms: HashMap<String, Pos>,
}

impl Lexicon {
    /// Build from the two COCA CSVs, passed as file CONTENTS.
    ///
    /// Contents rather than paths so the crate stays free of I/O and a test can
    /// hand it three lines. Both tables take the FIRST row for a given key —
    /// they are frequency-ranked, so the first is the dominant reading.
    ///
    /// * `lemmas_csv` — `rank,lemma,PoS,…` (`lemmas_5k.csv`)
    /// * `forms_csv` — `lemRank,lemma,PoS,lemFreq,wordFreq,word` (`word_forms.csv`)
    #[must_use]
    pub fn from_coca(lemmas_csv: &str, forms_csv: &str) -> Self {
        Self::from_columns(lemmas_csv, 1, 2).with_forms(forms_csv)
    }

    /// Build the base (lemma-keyed) table from `academic_20k.csv`.
    ///
    /// A different vocabulary, the SAME defect: its `word` column is lemmas
    /// (`be,v`), so it needs [`Self::with_forms`] layered under it exactly as
    /// `lemmas_5k.csv` does.
    #[must_use]
    pub fn from_academic_20k(csv: &str) -> Self {
        // header: ID,band,status,word,Pos,...
        Self::from_columns(csv, 3, 4)
    }

    /// The base table from any frequency-ranked CSV, given the column indices
    /// of the word and its PoS letter. First row per key wins.
    #[must_use]
    pub fn from_columns(csv: &str, word_col: usize, pos_col: usize) -> Self {
        let mut lemmas = HashMap::new();
        for line in csv.lines().skip(1) {
            let f: Vec<&str> = line.split(',').collect();
            let (Some(word), Some(pos)) = (f.get(word_col), f.get(pos_col)) else {
                continue;
            };
            if word.is_empty() {
                continue;
            }
            lemmas
                .entry(word.to_lowercase())
                .or_insert_with(|| coca_pos(pos));
        }
        Self {
            lemmas,
            forms: HashMap::new(),
        }
    }

    /// Layer the COCA form table (`word_forms.csv`) UNDER the base table.
    ///
    /// `lemRank,lemma,PoS,lemFreq,wordFreq,word` — the PoS is the lemma's, at
    /// column 2; the surface form is column 5.
    #[must_use]
    pub fn with_forms(mut self, forms_csv: &str) -> Self {
        for line in forms_csv.lines().skip(1) {
            let f: Vec<&str> = line.split(',').collect();
            let (Some(pos), Some(word)) = (f.get(2), f.get(5)) else {
                continue;
            };
            self.forms
                .entry(word.to_lowercase())
                .or_insert_with(|| coca_pos(pos));
        }
        self
    }

    /// The part of speech for `w`, which must already be lowercased and
    /// stripped to ascii letters.
    ///
    /// See the module doc for why the order is lemma → form → archaic.
    #[must_use]
    pub fn pos(&self, w: &str) -> Pos {
        self.lemmas
            .get(w)
            .or_else(|| self.forms.get(w))
            .copied()
            .or_else(|| archaic_pos(w))
            .unwrap_or(Pos::Other)
    }

    /// Did the lemma table alone know `w`? Used by the monotonicity test and
    /// by any caller reporting where its coverage comes from.
    #[must_use]
    pub fn lemma_knows(&self, w: &str) -> bool {
        self.lemmas.contains_key(w)
    }

    /// `(lemma rows, form rows)`.
    #[must_use]
    pub fn sizes(&self) -> (usize, usize) {
        (self.lemmas.len(), self.forms.len())
    }
}

/// Normalise one raw token: ascii letters only, lowercased. `None` for
/// anything under two letters, which the taggers all skipped.
#[must_use]
pub fn normalise(tok: &str) -> Option<String> {
    let w: String = tok
        .chars()
        .filter(char::is_ascii_alphabetic)
        .collect::<String>()
        .to_lowercase();
    (w.len() >= 2).then_some(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEMMAS: &str = "rank,lemma,PoS,freq\n1,create,v,9\n2,god,n,8\n3,light,n,7\n";
    const FORMS: &str =
        "lemRank,lemma,PoS,lemFreq,wordFreq,word\n1,create,v,9,4,created\n2,be,v,9,4,is\n\
         3,god,n,8,2,god\n";

    #[test]
    fn the_forms_table_tags_what_the_lemma_table_cannot() {
        let lx = Lexicon::from_coca(LEMMAS, FORMS);
        // The measured defect, in miniature: `create` is a lemma, `created` is
        // not, and before the forms table `created` was Pos::Other.
        assert_eq!(lx.pos("create"), Pos::Verb);
        assert!(!lx.lemma_knows("created"));
        assert_eq!(lx.pos("created"), Pos::Verb);
        // Anti-vacuity: without the forms table the same word is untagged, so
        // the assertion above is discriminating.
        let lemmas_only = Lexicon::from_coca(LEMMAS, "");
        assert_eq!(lemmas_only.pos("created"), Pos::Other);
    }

    #[test]
    fn adding_forms_never_retags_a_word_the_lemmas_knew() {
        // The monotonicity property the module doc claims. A forms row that
        // disagrees with the lemma table must lose.
        let conflicting = "lemRank,lemma,PoS,lemFreq,wordFreq,word\n1,x,n,9,4,create\n";
        let lemmas_only = Lexicon::from_coca(LEMMAS, "");
        let both = Lexicon::from_coca(LEMMAS, conflicting);
        assert_eq!(lemmas_only.pos("create"), Pos::Verb);
        assert_eq!(both.pos("create"), Pos::Verb, "the lemma table must win");
        // ...and the conflicting row IS loaded, so the test is not passing
        // merely because the parse dropped it.
        assert_eq!(both.sizes().1, 1);
    }

    #[test]
    fn archaic_still_fires_below_both_tables() {
        let lx = Lexicon::from_coca(LEMMAS, FORMS);
        assert_eq!(lx.pos("hath"), Pos::Verb);
        assert_eq!(lx.pos("thou"), Pos::Noun);
        assert_eq!(lx.pos("thy"), Pos::Det);
        assert_eq!(lx.pos("walketh"), Pos::Verb);
        // ...and it can stay silent: an unknown word is not smuggled to Verb.
        assert_eq!(lx.pos("zzzz"), Pos::Other);
    }

    #[test]
    fn the_academic_table_is_lemma_keyed_too_and_takes_the_same_forms() {
        // ID,band,status,word,Pos,...
        let acad = "ID,band,status,word,Pos,x\n1,1,c,be,v,0\n2,1,c,god,n,0\n";
        let bare = Lexicon::from_academic_20k(acad);
        assert_eq!(bare.pos("be"), Pos::Verb);
        assert_eq!(bare.pos("is"), Pos::Other, "the lemma table alone is blind");
        let layered = Lexicon::from_academic_20k(acad).with_forms(FORMS);
        assert_eq!(layered.pos("is"), Pos::Verb);
    }

    #[test]
    fn normalise_strips_and_drops_the_short() {
        assert_eq!(normalise("God,").as_deref(), Some("god"));
        assert_eq!(normalise("(LORD)").as_deref(), Some("lord"));
        assert_eq!(normalise("a"), None, "one letter was always skipped");
        assert_eq!(normalise("1:1"), None);
    }
}
