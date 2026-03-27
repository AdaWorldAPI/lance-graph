//! Vocabulary and tokenizer for DeepNSM.
//!
//! 4,096-word vocabulary from COCA frequency ranking.
//! 98.4% coverage of running English text.
//! O(1) hash lookup per word. No regex.
//!
//! # Loading
//! ```ignore
//! let vocab = Vocabulary::load(Path::new("word_frequency/"));
//! let tokens = vocab.tokenize("the big dog bit the old man");
//! // [(0, Article), (155, Adj), (670, Noun), (2942, Verb), (0, Article), (173, Adj), (94, Noun)]
//! ```

use crate::pos::PoS;
use std::collections::HashMap;
use std::path::Path;

/// Maximum vocabulary size (12-bit addressing).
pub const VOCAB_SIZE: usize = 4096;

/// A single vocabulary entry.
#[derive(Clone, Debug)]
pub struct WordEntry {
    /// Rank in COCA frequency list (0-based index, rank 1 → index 0).
    pub rank: u16,
    /// Part of speech.
    pub pos: PoS,
    /// Raw frequency in 1B-word corpus.
    pub freq: u64,
    /// Canonical lemma form.
    pub lemma: String,
}

/// A token produced by the tokenizer.
#[derive(Clone, Debug)]
pub struct Token {
    /// 12-bit vocabulary index (0-4095). None if OOV.
    pub rank: Option<u16>,
    /// Part of speech (from vocabulary or inferred).
    pub pos: PoS,
    /// Position in sentence (0-based).
    pub position: u16,
    /// Whether preceded by "not" / "n't".
    pub is_negated: bool,
    /// Original surface form.
    pub surface: String,
}

impl Token {
    /// Get the 12-bit rank, defaulting OOV to 0.
    #[inline]
    pub fn rank_or_default(&self) -> u16 {
        self.rank.unwrap_or(0)
    }

    /// Is this token in-vocabulary?
    #[inline]
    pub fn is_known(&self) -> bool {
        self.rank.is_some()
    }
}

/// The vocabulary: 4,096 entries loaded from COCA data.
///
/// Provides O(1) tokenization via hash lookup.
/// Handles inflected forms (e.g., "bit" → "bite", rank 2943).
pub struct Vocabulary {
    /// word string → entry (lowercase canonical)
    lookup: HashMap<String, WordEntry>,
    /// Inflected form → lemma rank ("bit" → rank of "bite")
    forms: HashMap<String, u16>,
    /// rank → word string (reverse lookup)
    reverse: Vec<String>,
    /// rank → PoS
    pos_table: Vec<PoS>,
    /// rank → frequency
    freq_table: Vec<u64>,
    /// Number of entries loaded.
    count: usize,
}

impl Vocabulary {
    /// Load vocabulary from CSV files in a directory.
    ///
    /// Expects:
    /// - `word_rank_lookup.csv`: rank,word,pos,freq
    /// - `word_forms.csv`: lemRank,lemma,PoS,lemFreq,wordFreq,word
    pub fn load(dir: &Path) -> Result<Self, String> {
        let mut lookup = HashMap::with_capacity(VOCAB_SIZE);
        let mut forms = HashMap::with_capacity(12000);
        let mut reverse = vec![String::new(); VOCAB_SIZE];
        let mut pos_table = vec![PoS::Noun; VOCAB_SIZE];
        let mut freq_table = vec![0u64; VOCAB_SIZE];
        let mut count = 0;

        // 1. Load word_rank_lookup.csv
        let rank_path = dir.join("word_rank_lookup.csv");
        let rank_content = std::fs::read_to_string(&rank_path)
            .map_err(|e| format!("Failed to read {}: {}", rank_path.display(), e))?;

        let mut prev_word = String::new();
        for line in rank_content.lines().skip(1) {
            // skip header
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() < 4 {
                continue;
            }

            let rank: u16 = fields[0]
                .parse()
                .map_err(|e| format!("Bad rank '{}': {}", fields[0], e))?;
            let word = fields[1].to_lowercase();
            let pos_tag = fields[2];
            let freq: u64 = fields[3]
                .parse()
                .map_err(|e| format!("Bad freq '{}': {}", fields[3], e))?;

            let pos = PoS::from_tag(pos_tag).unwrap_or(PoS::Noun);

            // COCA has duplicate ranks for homographs (e.g., rank 12 "to" as both
            // preposition and particle). Keep the first occurrence (higher frequency).
            if word == prev_word {
                continue;
            }
            prev_word = word.clone();

            // Convert 1-based rank to 0-based index, cap at VOCAB_SIZE
            let idx = (rank as usize).saturating_sub(1);
            if idx >= VOCAB_SIZE {
                continue;
            }

            let entry = WordEntry {
                rank: idx as u16,
                pos,
                freq,
                lemma: word.clone(),
            };

            lookup.insert(word.clone(), entry);
            reverse[idx] = word;
            pos_table[idx] = pos;
            freq_table[idx] = freq;
            count += 1;
        }

        // 2. Load word_forms.csv for inflection resolution
        let forms_path = dir.join("word_forms.csv");
        if let Ok(forms_content) = std::fs::read_to_string(&forms_path) {
            for line in forms_content.lines().skip(1) {
                let fields: Vec<&str> = line.split(',').collect();
                if fields.len() < 6 {
                    continue;
                }

                let lem_rank: u16 = match fields[0].parse() {
                    Ok(r) => r,
                    Err(_) => continue,
                };

                // Convert to 0-based, skip if out of vocab
                let idx = (lem_rank as usize).saturating_sub(1);
                if idx >= VOCAB_SIZE {
                    continue;
                }

                let surface_form = fields[5].to_lowercase();

                // Only add if the surface form isn't already a primary entry
                if !lookup.contains_key(&surface_form) {
                    forms.insert(surface_form, idx as u16);
                }
            }
        }

        Ok(Vocabulary {
            lookup,
            forms,
            reverse,
            pos_table,
            freq_table,
            count,
        })
    }

    /// Look up a single word. O(1) hash lookup.
    ///
    /// Resolution order:
    /// 1. Exact match (lowercase)
    /// 2. Inflected forms table ("bit" → "bite")
    /// 3. None (out-of-vocabulary)
    pub fn lookup_word(&self, word: &str) -> Option<&WordEntry> {
        let lower = word.to_lowercase();

        // 1. Direct lookup
        if let Some(entry) = self.lookup.get(&lower) {
            return Some(entry);
        }

        // 2. Check inflected forms → get lemma rank → get entry by reverse lookup
        if let Some(&lemma_rank) = self.forms.get(&lower) {
            let lemma = &self.reverse[lemma_rank as usize];
            return self.lookup.get(lemma);
        }

        // 3. Strip common suffixes as fallback
        let stripped = strip_suffix(&lower);
        if stripped != lower {
            if let Some(entry) = self.lookup.get(stripped) {
                return Some(entry);
            }
            if let Some(&lemma_rank) = self.forms.get(stripped) {
                let lemma = &self.reverse[lemma_rank as usize];
                return self.lookup.get(lemma);
            }
        }

        None
    }

    /// Resolve a word to its vocabulary rank. Returns None for OOV.
    #[inline]
    pub fn rank_of(&self, word: &str) -> Option<u16> {
        self.lookup_word(word).map(|e| e.rank)
    }

    /// Tokenize a sentence into a token stream.
    ///
    /// Splits on whitespace and punctuation. O(n) where n = word count.
    /// Handles contractions ("don't" → "do" + negation flag).
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let words = split_words(text);
        let mut tokens = Vec::with_capacity(words.len());
        let mut negation_pending = false;

        for (position, word) in words.iter().enumerate() {
            // Handle contractions
            if word == "n't" || word == "not" {
                negation_pending = true;
                // "not" itself gets tokenized too
                if let Some(entry) = self.lookup_word(word) {
                    tokens.push(Token {
                        rank: Some(entry.rank),
                        pos: PoS::Negation,
                        position: position as u16,
                        is_negated: false,
                        surface: word.to_string(),
                    });
                }
                continue;
            }

            // Handle "'s", "'re", "'m", "'ve", "'ll", "'d"
            if word.starts_with('\'') && word.len() <= 3 {
                if let Some(entry) = self.lookup_word(word) {
                    tokens.push(Token {
                        rank: Some(entry.rank),
                        pos: entry.pos,
                        position: position as u16,
                        is_negated: false,
                        surface: word.to_string(),
                    });
                }
                continue;
            }

            let entry = self.lookup_word(word);
            let token = Token {
                rank: entry.map(|e| e.rank),
                pos: entry.map_or(PoS::Noun, |e| e.pos), // default OOV to noun
                position: position as u16,
                is_negated: negation_pending,
                surface: word.to_string(),
            };
            negation_pending = false;
            tokens.push(token);
        }

        tokens
    }

    /// Reverse lookup: rank → word string.
    #[inline]
    pub fn word(&self, rank: u16) -> &str {
        if (rank as usize) < self.reverse.len() {
            &self.reverse[rank as usize]
        } else {
            "<OOV>"
        }
    }

    /// Get PoS for a rank.
    #[inline]
    pub fn pos(&self, rank: u16) -> PoS {
        if (rank as usize) < self.pos_table.len() {
            self.pos_table[rank as usize]
        } else {
            PoS::Noun
        }
    }

    /// Get frequency for a rank.
    #[inline]
    pub fn freq(&self, rank: u16) -> u64 {
        if (rank as usize) < self.freq_table.len() {
            self.freq_table[rank as usize]
        } else {
            0
        }
    }

    /// Number of entries loaded.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Is the vocabulary empty?
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Number of inflected forms loaded.
    pub fn forms_count(&self) -> usize {
        self.forms.len()
    }
}

// ─── Word splitting ─────────────────────────────────────────────────────────

/// Split text into words. Handles contractions, punctuation, possessives.
/// No regex — pure character-level scanning.
fn split_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let c = chars[i];

        if c.is_alphanumeric() || c == '-' {
            // Continue building word
            current.push(c.to_lowercase().next().unwrap_or(c));
            i += 1;
        } else if c == '\'' || c == '\u{2019}' {
            // Apostrophe: could be contraction
            if !current.is_empty() && i + 1 < len && chars[i + 1].is_alphabetic() {
                // Look ahead for contraction patterns
                let rest: String = chars[i..].iter().take(4).collect();
                let rest_lower = rest.to_lowercase();

                if rest_lower.starts_with("n't") || rest_lower.starts_with("\u{2019}t") {
                    // "don't" → push "do", then "n't"
                    if !current.is_empty() {
                        words.push(current.clone());
                        current.clear();
                    }
                    words.push("n't".to_string());
                    i += 3;
                } else if rest_lower.starts_with("'s")
                    || rest_lower.starts_with("'re")
                    || rest_lower.starts_with("'m")
                    || rest_lower.starts_with("'ve")
                    || rest_lower.starts_with("'ll")
                    || rest_lower.starts_with("'d")
                {
                    // Push current word first
                    if !current.is_empty() {
                        words.push(current.clone());
                        current.clear();
                    }
                    // Find contraction end
                    let mut end = i + 1;
                    while end < len && chars[end].is_alphabetic() {
                        end += 1;
                    }
                    let contraction: String =
                        chars[i..end].iter().map(|c| c.to_lowercase().next().unwrap_or(*c)).collect();
                    words.push(contraction);
                    i = end;
                } else {
                    // Regular apostrophe in word
                    current.push('\'');
                    i += 1;
                }
            } else {
                // Apostrophe at start or isolated
                if !current.is_empty() {
                    words.push(current.clone());
                    current.clear();
                }
                i += 1;
            }
        } else {
            // Whitespace or punctuation: end current word
            if !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
            i += 1;
        }
    }

    if !current.is_empty() {
        words.push(current);
    }

    words
}

/// Strip common English suffixes for fallback resolution.
fn strip_suffix(word: &str) -> &str {
    // Order matters: try longest suffixes first
    for suffix in &["ing", "tion", "sion", "ness", "ment", "ous", "ive", "ful", "less", "ly", "ed", "er", "est", "es", "s"] {
        if word.len() > suffix.len() + 2 && word.ends_with(suffix) {
            return &word[..word.len() - suffix.len()];
        }
    }
    word
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_simple() {
        let words = split_words("the big dog");
        assert_eq!(words, vec!["the", "big", "dog"]);
    }

    #[test]
    fn split_contractions() {
        let words = split_words("don't won't can't");
        assert_eq!(words, vec!["do", "n't", "wo", "n't", "ca", "n't"]);
    }

    #[test]
    fn split_possessive() {
        let words = split_words("he's they're I'm");
        assert_eq!(words, vec!["he", "'s", "they", "'re", "i", "'m"]);
    }

    #[test]
    fn split_punctuation() {
        let words = split_words("Hello, world! How are you?");
        assert_eq!(words, vec!["hello", "world", "how", "are", "you"]);
    }

    #[test]
    fn strip_suffix_basic() {
        assert_eq!(strip_suffix("running"), "runn");
        assert_eq!(strip_suffix("dogs"), "dog");
        assert_eq!(strip_suffix("quickly"), "quick");
    }
}
