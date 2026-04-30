// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vocabulary tokenizer for the DeepNSM pipeline.
//!
//! Maps words to ranks (0..4095) with part-of-speech tags.
//! O(1) per word via hash lookup. Handles exact, lowercase, and inflected forms.

use std::collections::HashMap;

/// Part-of-speech tags (13 categories).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoS {
    /// Noun
    N,
    /// Verb
    V,
    /// Adjective
    J,
    /// Adverb
    R,
    /// Interjection / particle
    I,
    /// Pronoun
    P,
    /// Conjunction
    C,
    /// Determiner / article
    D,
    /// Modal / auxiliary
    M,
    /// Unknown / unclassified
    U,
    /// Preposition
    A,
    /// Existential / special
    X,
    /// Temporal marker
    T,
}

impl PoS {
    /// Parse a PoS tag from a single-character string.
    pub fn from_char(c: char) -> Self {
        match c {
            'n' | 'N' => PoS::N,
            'v' | 'V' => PoS::V,
            'j' | 'J' => PoS::J,
            'r' | 'R' => PoS::R,
            'i' | 'I' => PoS::I,
            'p' | 'P' => PoS::P,
            'c' | 'C' => PoS::C,
            'd' | 'D' => PoS::D,
            'm' | 'M' => PoS::M,
            'u' | 'U' => PoS::U,
            'a' | 'A' => PoS::A,
            'x' | 'X' => PoS::X,
            't' | 'T' => PoS::T,
            _ => PoS::U,
        }
    }

    /// Whether this PoS is a noun-like category (noun or pronoun).
    pub fn is_nominal(&self) -> bool {
        matches!(self, PoS::N | PoS::P)
    }

    /// Whether this PoS is a verb-like category (verb or modal).
    pub fn is_verbal(&self) -> bool {
        matches!(self, PoS::V | PoS::M)
    }
}

/// Entry in the vocabulary lookup table.
#[derive(Debug, Clone)]
pub struct WordEntry {
    /// Rank in frequency order (0 = most frequent). 12-bit range: 0..4095.
    pub rank: u16,
    /// Part of speech.
    pub pos: PoS,
    /// Raw frequency count from the corpus.
    pub freq: u32,
}

/// A single token produced by the tokenizer.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// Vocabulary rank (12-bit, 0..4095).
    pub rank: u16,
    /// Part of speech.
    pub pos: PoS,
    /// Position in the sentence (0-indexed).
    pub position: u16,
    /// True if preceded by "not" / negation.
    pub is_negated: bool,
}

/// Vocabulary: the core lookup structure for the NSM tokenizer.
///
/// Holds a forward map (word -> entry), a reverse map (rank -> word),
/// and an inflection map (form -> base rank).
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// Forward lookup: canonical word -> entry.
    words: HashMap<String, WordEntry>,
    /// Reverse lookup: rank -> canonical word. Index = rank.
    reverse: Vec<String>,
    /// Inflected forms: e.g. "running" -> rank of "run".
    forms: HashMap<String, u16>,
}

impl Vocabulary {
    /// Load vocabulary from CSV content strings.
    ///
    /// `rank_csv`: lines of "word,rank,pos,freq"
    /// `forms_csv`: lines of "form,base_rank"
    pub fn load(rank_csv: &str, forms_csv: &str) -> Self {
        let mut words = HashMap::new();
        let mut max_rank: u16 = 0;

        for line in rank_csv.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 4 {
                continue;
            }
            let word = parts[0].trim().to_lowercase();
            let rank: u16 = match parts[1].trim().parse() {
                Ok(r) => r,
                Err(_) => continue,
            };
            let pos = if let Some(c) = parts[2].trim().chars().next() {
                PoS::from_char(c)
            } else {
                PoS::U
            };
            let freq: u32 = parts[3].trim().parse().unwrap_or(0);
            if rank > max_rank {
                max_rank = rank;
            }
            words.insert(word, WordEntry { rank, pos, freq });
        }

        // Build reverse map
        let mut reverse = vec![String::new(); (max_rank as usize) + 1];
        for (word, entry) in &words {
            if (entry.rank as usize) < reverse.len() {
                reverse[entry.rank as usize] = word.clone();
            }
        }

        // Parse forms
        let mut forms = HashMap::new();
        for line in forms_csv.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 2 {
                continue;
            }
            let form = parts[0].trim().to_lowercase();
            let base_rank: u16 = match parts[1].trim().parse() {
                Ok(r) => r,
                Err(_) => continue,
            };
            forms.insert(form, base_rank);
        }

        Vocabulary {
            words,
            reverse,
            forms,
        }
    }

    /// Construct a vocabulary from pre-built entries (for testing / embedding).
    ///
    /// Each entry is (word, rank, pos, freq). Forms is (inflected_form, base_rank).
    pub fn from_entries(entries: &[(&str, u16, PoS, u32)], form_entries: &[(&str, u16)]) -> Self {
        let mut words = HashMap::new();
        let mut max_rank: u16 = 0;

        for &(word, rank, pos, freq) in entries {
            if rank > max_rank {
                max_rank = rank;
            }
            words.insert(word.to_lowercase(), WordEntry { rank, pos, freq });
        }

        let mut reverse = vec![String::new(); (max_rank as usize) + 1];
        for (word, entry) in &words {
            if (entry.rank as usize) < reverse.len() {
                reverse[entry.rank as usize] = word.clone();
            }
        }

        let mut forms = HashMap::new();
        for &(form, base_rank) in form_entries {
            forms.insert(form.to_lowercase(), base_rank);
        }

        Vocabulary {
            words,
            reverse,
            forms,
        }
    }

    /// Tokenize a single word. O(1) hash lookup.
    ///
    /// Resolution order: exact match -> lowercase -> inflected form -> None.
    pub fn tokenize_word(&self, word: &str) -> Option<(u16, PoS)> {
        // Exact match
        if let Some(entry) = self.words.get(word) {
            return Some((entry.rank, entry.pos));
        }
        // Lowercase
        let lower = word.to_lowercase();
        if let Some(entry) = self.words.get(&lower) {
            return Some((entry.rank, entry.pos));
        }
        // Inflected form
        if let Some(&base_rank) = self.forms.get(&lower) {
            if (base_rank as usize) < self.reverse.len() {
                let base_word = &self.reverse[base_rank as usize];
                if let Some(entry) = self.words.get(base_word) {
                    return Some((entry.rank, entry.pos));
                }
            }
        }
        None
    }

    /// Tokenize a sentence into a sequence of tokens.
    ///
    /// Splits on whitespace and punctuation. Sets `is_negated` on verbs
    /// preceded by "not" (or "n't" contracted forms).
    pub fn tokenize(&self, sentence: &str) -> Vec<Token> {
        let raw_words = Self::split_words(sentence);
        let mut tokens = Vec::new();
        let mut negation_pending = false;
        let mut position: u16 = 0;

        for raw in &raw_words {
            // Check for negation markers
            let lower = raw.to_lowercase();
            if lower == "not" || lower == "n't" || lower.ends_with("n't") {
                negation_pending = true;
                // Still tokenize "not" itself if it's in the vocabulary
                if let Some((rank, pos)) = self.tokenize_word(raw) {
                    tokens.push(Token {
                        rank,
                        pos,
                        position,
                        is_negated: false,
                    });
                    position += 1;
                }
                continue;
            }

            if let Some((rank, pos)) = self.tokenize_word(raw) {
                let is_negated = if negation_pending && pos.is_verbal() {
                    negation_pending = false;
                    true
                } else {
                    false
                };

                tokens.push(Token {
                    rank,
                    pos,
                    position,
                    is_negated,
                });
                position += 1;
            }
            // Unknown words are dropped (OOV)
        }

        tokens
    }

    /// Reverse lookup: rank -> word string.
    pub fn word(&self, rank: u16) -> Option<&str> {
        let idx = rank as usize;
        if idx < self.reverse.len() && !self.reverse[idx].is_empty() {
            Some(&self.reverse[idx])
        } else {
            None
        }
    }

    /// Return the number of words in the vocabulary.
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Look up a word entry by string.
    pub fn lookup(&self, word: &str) -> Option<&WordEntry> {
        self.words.get(&word.to_lowercase())
    }

    /// Split a sentence into word tokens, stripping punctuation.
    fn split_words(sentence: &str) -> Vec<String> {
        let mut words = Vec::new();
        let mut current = String::new();

        for ch in sentence.chars() {
            if ch.is_alphanumeric() || ch == '\'' || ch == '-' {
                current.push(ch);
            } else {
                if !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                }
            }
        }
        if !current.is_empty() {
            words.push(current);
        }
        words
    }
}

/// Build the built-in test vocabulary (~50 essential NSM primes + common words).
pub fn test_vocabulary() -> Vocabulary {
    // NSM primes and common words with ranks, PoS, and approximate frequency
    let entries: &[(&str, u16, PoS, u32)] = &[
        // NSM primes (Wierzbicka's semantic primes)
        ("i", 0, PoS::P, 1000000),
        ("you", 1, PoS::P, 900000),
        ("someone", 2, PoS::P, 50000),
        ("something", 3, PoS::P, 60000),
        ("people", 4, PoS::N, 200000),
        ("body", 5, PoS::N, 80000),
        ("kind", 6, PoS::N, 70000),
        ("part", 7, PoS::N, 150000),
        ("this", 8, PoS::D, 500000),
        ("the", 9, PoS::D, 5000000),
        ("same", 10, PoS::J, 120000),
        ("other", 11, PoS::J, 180000),
        ("one", 12, PoS::D, 300000),
        ("two", 13, PoS::D, 100000),
        ("much", 14, PoS::R, 90000),
        ("many", 15, PoS::J, 80000),
        ("little", 16, PoS::J, 70000),
        ("good", 17, PoS::J, 200000),
        ("bad", 18, PoS::J, 100000),
        ("big", 19, PoS::J, 150000),
        ("small", 20, PoS::J, 80000),
        ("think", 21, PoS::V, 200000),
        ("want", 22, PoS::V, 150000),
        ("not", 23, PoS::R, 400000),
        ("know", 24, PoS::V, 300000),
        ("feel", 25, PoS::V, 100000),
        ("see", 26, PoS::V, 200000),
        ("hear", 27, PoS::V, 80000),
        ("say", 28, PoS::V, 250000),
        ("do", 29, PoS::V, 500000),
        ("happen", 30, PoS::V, 60000),
        ("move", 31, PoS::V, 80000),
        ("there", 32, PoS::R, 300000),
        ("is", 33, PoS::V, 2000000),
        ("have", 34, PoS::V, 1000000),
        ("be", 35, PoS::V, 1500000),
        ("live", 36, PoS::V, 100000),
        ("die", 37, PoS::V, 50000),
        ("when", 38, PoS::C, 200000),
        ("where", 39, PoS::R, 150000),
        ("can", 40, PoS::M, 300000),
        ("because", 41, PoS::C, 100000),
        ("if", 42, PoS::C, 200000),
        ("like", 43, PoS::A, 150000),
        ("a", 44, PoS::D, 4000000),
        ("and", 45, PoS::C, 3000000),
        ("of", 46, PoS::A, 2500000),
        ("in", 47, PoS::A, 2000000),
        ("to", 48, PoS::A, 2500000),
        ("with", 49, PoS::A, 800000),
        // Common nouns/verbs for testing
        ("cat", 50, PoS::N, 30000),
        ("dog", 51, PoS::N, 35000),
        ("man", 52, PoS::N, 100000),
        ("woman", 53, PoS::N, 80000),
        ("child", 54, PoS::N, 60000),
        ("house", 55, PoS::N, 70000),
        ("water", 56, PoS::N, 80000),
        ("food", 57, PoS::N, 50000),
        ("time", 58, PoS::N, 200000),
        ("love", 59, PoS::V, 60000),
        ("run", 60, PoS::V, 70000),
        ("eat", 61, PoS::V, 40000),
        ("sleep", 62, PoS::V, 30000),
        ("fast", 63, PoS::J, 50000),
        ("slow", 64, PoS::J, 30000),
        ("very", 65, PoS::R, 200000),
        ("quickly", 66, PoS::R, 30000),
        ("the", 9, PoS::D, 5000000), // dup, ignored by HashMap
        ("chases", 67, PoS::V, 5000),
        ("sits", 68, PoS::V, 4000),
        ("on", 69, PoS::A, 1500000),
        ("mat", 70, PoS::N, 5000),
    ];

    let form_entries: &[(&str, u16)] = &[
        ("thinking", 21),
        ("thinks", 21),
        ("thought", 21),
        ("wanting", 22),
        ("wants", 22),
        ("wanted", 22),
        ("knowing", 24),
        ("knows", 24),
        ("knew", 24),
        ("known", 24),
        ("seeing", 26),
        ("sees", 26),
        ("saw", 26),
        ("seen", 26),
        ("hearing", 27),
        ("hears", 27),
        ("heard", 27),
        ("saying", 28),
        ("says", 28),
        ("said", 28),
        ("doing", 29),
        ("does", 29),
        ("did", 29),
        ("done", 29),
        ("happening", 30),
        ("happens", 30),
        ("happened", 30),
        ("moving", 31),
        ("moves", 31),
        ("moved", 31),
        ("living", 36),
        ("lives", 36),
        ("lived", 36),
        ("dying", 37),
        ("dies", 37),
        ("died", 37),
        ("loving", 59),
        ("loves", 59),
        ("loved", 59),
        ("running", 60),
        ("runs", 60),
        ("ran", 60),
        ("eating", 61),
        ("eats", 61),
        ("ate", 61),
        ("eaten", 61),
        ("sleeping", 62),
        ("sleeps", 62),
        ("slept", 62),
        ("chasing", 67),
        ("chased", 67),
        ("sitting", 68),
        ("sat", 68),
        ("cats", 50),
        ("dogs", 51),
        ("men", 52),
        ("women", 53),
        ("children", 54),
        ("houses", 55),
    ];

    Vocabulary::from_entries(entries, form_entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vocabulary {
        test_vocabulary()
    }

    #[test]
    fn test_basic_lookup() {
        let v = vocab();
        let (rank, pos) = v.tokenize_word("cat").unwrap();
        assert_eq!(rank, 50);
        assert_eq!(pos, PoS::N);
    }

    #[test]
    fn test_case_insensitive_lookup() {
        let v = vocab();
        let (rank, _) = v.tokenize_word("Cat").unwrap();
        assert_eq!(rank, 50);
        let (rank2, _) = v.tokenize_word("CAT").unwrap();
        assert_eq!(rank2, 50);
    }

    #[test]
    fn test_inflected_form_lookup() {
        let v = vocab();
        let (rank, pos) = v.tokenize_word("running").unwrap();
        assert_eq!(rank, 60); // "run"
        assert_eq!(pos, PoS::V);
    }

    #[test]
    fn test_unknown_word() {
        let v = vocab();
        assert!(v.tokenize_word("xyzzyplugh").is_none());
    }

    #[test]
    fn test_reverse_lookup() {
        let v = vocab();
        assert_eq!(v.word(50), Some("cat"));
        assert_eq!(v.word(51), Some("dog"));
        assert_eq!(v.word(9999), None);
    }

    #[test]
    fn test_tokenize_sentence() {
        let v = vocab();
        let tokens = v.tokenize("the cat sits on the mat");
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[0].rank, 9); // the
        assert_eq!(tokens[1].rank, 50); // cat
        assert_eq!(tokens[2].rank, 68); // sits -> sit mapped
        assert_eq!(tokens[3].rank, 69); // on
        assert_eq!(tokens[4].rank, 9); // the
        assert_eq!(tokens[5].rank, 70); // mat
    }

    #[test]
    fn test_negation_on_verb() {
        let v = vocab();
        let tokens = v.tokenize("the cat does not run");
        // "does" at pos 1, "not" at pos 2, "run" at pos 3
        // "not" sets negation_pending, "run" is verb -> negated
        let run_token = tokens.iter().find(|t| t.rank == 60).unwrap();
        assert!(run_token.is_negated);
    }

    #[test]
    fn test_negation_not_on_noun() {
        let v = vocab();
        let tokens = v.tokenize("not cat");
        // "not" sets negation_pending, "cat" is noun -> NOT negated
        let cat_token = tokens.iter().find(|t| t.rank == 50).unwrap();
        assert!(!cat_token.is_negated);
    }

    #[test]
    fn test_punctuation_stripping() {
        let v = vocab();
        let tokens = v.tokenize("the cat, the dog.");
        assert_eq!(tokens.len(), 4);
    }

    #[test]
    fn test_empty_sentence() {
        let v = vocab();
        let tokens = v.tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_vocabulary_len() {
        let v = vocab();
        assert!(v.len() >= 50);
        assert!(!v.is_empty());
    }

    #[test]
    fn test_pos_categories() {
        assert!(PoS::N.is_nominal());
        assert!(PoS::P.is_nominal());
        assert!(!PoS::V.is_nominal());
        assert!(PoS::V.is_verbal());
        assert!(PoS::M.is_verbal());
        assert!(!PoS::N.is_verbal());
    }

    #[test]
    fn test_load_from_csv() {
        let rank_csv = "cat,50,n,30000\ndog,51,n,35000\nrun,60,v,70000\n";
        let forms_csv = "running,60\nruns,60\n";
        let v = Vocabulary::load(rank_csv, forms_csv);
        assert_eq!(v.len(), 3);
        let (rank, pos) = v.tokenize_word("cat").unwrap();
        assert_eq!(rank, 50);
        assert_eq!(pos, PoS::N);
        let (rank2, _) = v.tokenize_word("running").unwrap();
        assert_eq!(rank2, 60);
    }
}
