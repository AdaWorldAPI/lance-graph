//! Part-of-speech tags from COCA corpus.
//!
//! 13 tags packed into 4 bits (`u8`, upper 4 unused).
//! Matches the PoS column in `word_rank_lookup.csv`.

/// Part of speech tag. 13 values, fits in 4 bits.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum PoS {
    /// `a` — article/determiner ("the", "a", "this")
    Article = 0,
    /// `v` — verb ("be", "have", "do", "go")
    Verb = 1,
    /// `j` — adjective ("big", "old", "new")
    Adjective = 2,
    /// `r` — adverb ("very", "just", "also")
    Adverb = 3,
    /// `i` — preposition ("of", "in", "for", "with")
    Preposition = 4,
    /// `p` — pronoun ("i", "you", "it", "he", "she")
    Pronoun = 5,
    /// `c` — conjunction ("and", "but", "or", "that")
    Conjunction = 6,
    /// `d` — modal/auxiliary ("will", "would", "can")
    Modal = 7,
    /// `n` — noun ("time", "people", "way", "day")
    Noun = 8,
    /// `u` — interjection ("oh", "yes", "well")
    Interjection = 9,
    /// `t` — particle/infinitive marker ("to")
    Particle = 10,
    /// `x` — negation ("not", "n't")
    Negation = 11,
    /// `e` — existential ("there" as in "there is")
    Existential = 12,
}

impl PoS {
    /// Parse from single-character COCA tag.
    #[inline]
    pub fn from_tag(tag: &str) -> Option<PoS> {
        match tag {
            "a" => Some(PoS::Article),
            "v" => Some(PoS::Verb),
            "j" => Some(PoS::Adjective),
            "r" => Some(PoS::Adverb),
            "i" => Some(PoS::Preposition),
            "p" => Some(PoS::Pronoun),
            "c" => Some(PoS::Conjunction),
            "d" => Some(PoS::Modal),
            "n" => Some(PoS::Noun),
            "u" => Some(PoS::Interjection),
            "t" => Some(PoS::Particle),
            "x" => Some(PoS::Negation),
            "e" => Some(PoS::Existential),
            _ => None,
        }
    }

    /// Convert to single-character COCA tag.
    #[inline]
    pub fn as_tag(self) -> &'static str {
        match self {
            PoS::Article => "a",
            PoS::Verb => "v",
            PoS::Adjective => "j",
            PoS::Adverb => "r",
            PoS::Preposition => "i",
            PoS::Pronoun => "p",
            PoS::Conjunction => "c",
            PoS::Modal => "d",
            PoS::Noun => "n",
            PoS::Interjection => "u",
            PoS::Particle => "t",
            PoS::Negation => "x",
            PoS::Existential => "e",
        }
    }

    /// Is this a content word (noun, verb, adjective, adverb)?
    #[inline]
    pub fn is_content(self) -> bool {
        matches!(self, PoS::Noun | PoS::Verb | PoS::Adjective | PoS::Adverb)
    }

    /// Is this a function word (article, preposition, conjunction, etc.)?
    #[inline]
    pub fn is_function(self) -> bool {
        !self.is_content()
    }

    /// Can this PoS be a subject or object head?
    #[inline]
    pub fn is_nominal(self) -> bool {
        matches!(self, PoS::Noun | PoS::Pronoun)
    }

    /// Can this PoS be a predicate?
    #[inline]
    pub fn is_verbal(self) -> bool {
        matches!(self, PoS::Verb | PoS::Modal)
    }

    /// Is this a modifier (adjective or adverb)?
    #[inline]
    pub fn is_modifier(self) -> bool {
        matches!(self, PoS::Adjective | PoS::Adverb)
    }

    /// Is this a determiner-like element that starts an NP?
    #[inline]
    pub fn is_determiner(self) -> bool {
        matches!(self, PoS::Article)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_tags() {
        for tag in &["a", "v", "j", "r", "i", "p", "c", "d", "n", "u", "t", "x", "e"] {
            let pos = PoS::from_tag(tag).unwrap();
            assert_eq!(pos.as_tag(), *tag);
        }
    }

    #[test]
    fn content_vs_function() {
        assert!(PoS::Noun.is_content());
        assert!(PoS::Verb.is_content());
        assert!(PoS::Adjective.is_content());
        assert!(PoS::Adverb.is_content());
        assert!(PoS::Preposition.is_function());
        assert!(PoS::Article.is_function());
        assert!(PoS::Conjunction.is_function());
    }

    #[test]
    fn fits_in_4_bits() {
        assert!((PoS::Existential as u8) < 16);
    }
}
