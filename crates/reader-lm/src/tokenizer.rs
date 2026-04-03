//! Simple BPE tokenizer for Reader-LM (Qwen2 style).
//! STUB: deterministic hash. TODO: load vocab.json + merges.txt.

const BOS_TOKEN: u32 = 151643;
const _EOS_TOKEN: u32 = 151645;

pub fn tokenize(text: &str) -> Vec<u32> {
    let mut tokens = vec![BOS_TOKEN];
    for word in text.split(|c: char| c.is_whitespace() || c == '<' || c == '>') {
        let word = word.trim();
        if word.is_empty() { continue; }
        let hash = word.bytes().fold(5381u64, |h, b| h.wrapping_mul(33).wrapping_add(b as u64));
        tokens.push((hash % 151000) as u32);
    }
    tokens
}

pub fn token_count(text: &str) -> usize { tokenize(text).len() }

// --- Real tokenizer support (HuggingFace tokenizers crate) ---

/// Load the real Reader-LM tokenizer from HuggingFace.
#[cfg(feature = "real-tokenizer")]
pub fn load_real_tokenizer() -> Result<tokenizers::Tokenizer, String> {
    tokenizers::Tokenizer::from_pretrained("jinaai/reader-lm-1.5b", None)
        .map_err(|e| format!("Failed to load reader-lm tokenizer: {}", e))
}

/// Tokenize text using the real tokenizer, falling back to the stub on error.
#[cfg(feature = "real-tokenizer")]
pub fn tokenize_real(tokenizer: &tokenizers::Tokenizer, text: &str) -> Vec<u32> {
    tokenizer
        .encode(text, true)
        .map(|enc| enc.get_ids().to_vec())
        .unwrap_or_else(|_| tokenize(text)) // fallback to stub
}

#[cfg(all(test, feature = "real-tokenizer"))]
mod real_tokenizer_tests {
    use super::*;

    #[test]
    fn test_load_real_tokenizer() {
        let tokenizer = load_real_tokenizer().expect("should load tokenizer from HuggingFace");
        let encoding = tokenizer.encode("Hello world", true).expect("should encode");
        let ids = encoding.get_ids();
        assert!(ids.len() > 0, "token count must be > 0");
    }

    #[test]
    fn test_real_tokenizer_deterministic() {
        let tokenizer = load_real_tokenizer().expect("should load tokenizer");
        let ids_a = tokenize_real(&tokenizer, "Hello world");
        let ids_b = tokenize_real(&tokenizer, "Hello world");
        assert_eq!(ids_a, ids_b, "same input must produce same output");
    }
}
