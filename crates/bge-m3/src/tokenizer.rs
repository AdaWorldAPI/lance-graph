//! Simple BPE tokenizer stub for BGE-M3 (XLM-RoBERTa).
//!
//! STUB: uses deterministic hashing instead of real SentencePiece.
//! TODO: load sentencepiece.bpe.model for production accuracy.
//! The real tokenizer requires loading the 5MB SentencePiece BPE model
//! and performing proper byte-pair encoding. This stub provides
//! deterministic token IDs for testing and development.

const CLS_TOKEN: u32 = 0;
const SEP_TOKEN: u32 = 2;
#[allow(dead_code)]
const UNK_TOKEN: u32 = 3;

/// Tokenize text into token IDs.
/// Adds \[CLS\] at start and \[SEP\] at end.
///
/// Deterministic: same input always produces the same token sequence.
pub fn tokenize(text: &str) -> Vec<u32> {
    let mut tokens = vec![CLS_TOKEN];
    for word in text.split(|c: char| {
        c.is_whitespace() || c == ',' || c == '.' || c == '!' || c == '?' || c == ';' || c == ':'
    }) {
        let word = word.trim();
        if word.is_empty() {
            continue;
        }
        // Deterministic hash to token ID (within vocab range, skip special tokens 0-3)
        let hash = word
            .bytes()
            .fold(5381u64, |h, b| h.wrapping_mul(33).wrapping_add(b as u64));
        let token_id = (hash % 250000) as u32 + 4;
        tokens.push(token_id);
    }
    tokens.push(SEP_TOKEN);
    tokens
}

/// Tokenize and return token count (for usage stats).
pub fn token_count(text: &str) -> usize {
    tokenize(text).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cls_sep() {
        let tokens = tokenize("hello");
        assert_eq!(tokens[0], CLS_TOKEN);
        assert_eq!(*tokens.last().unwrap(), SEP_TOKEN);
    }

    #[test]
    fn test_deterministic() {
        let a = tokenize("hello world");
        let b = tokenize("hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn test_different_texts() {
        let a = tokenize("hello");
        let b = tokenize("world");
        assert_ne!(a, b);
    }

    #[test]
    fn test_token_range() {
        let tokens = tokenize("The quick brown fox jumps over the lazy dog");
        for &tok in &tokens[1..tokens.len() - 1] {
            assert!(tok >= 4, "non-special tokens must be >= 4");
            assert!(tok < 250004, "tokens must be in vocab range");
        }
    }

    #[test]
    fn test_empty() {
        let tokens = tokenize("");
        assert_eq!(tokens, vec![CLS_TOKEN, SEP_TOKEN]);
    }

    #[test]
    fn test_punctuation_split() {
        let tokens = tokenize("hello,world");
        // Should split on comma: CLS + hello + world + SEP
        assert_eq!(tokens.len(), 4);
    }

    #[test]
    fn test_token_count() {
        assert_eq!(token_count("a b c"), 5); // CLS + a + b + c + SEP
    }
}
