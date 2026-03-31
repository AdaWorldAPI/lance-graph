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
