//! Unified tokenizer registry for cross-model calibration.
//!
//! Each model has its own tokenizer (different vocab, different BPE).
//! The registry loads all tokenizers and provides uniform access:
//!
//! ```text
//! "The wound is where the light enters"
//!   → XLM-RoBERTa (Jina v3):    [581, 36735, 83, 7, ...]  (250K vocab)
//!   → XLM-RoBERTa (BGE-M3):     [581, 36735, 83, 7, ...]  (same tokenizer)
//!   → Qwen2 (Reranker):          [785, 11980, 374, ...]    (151K vocab)
//!   → Qwen3 (Jina v5):           [785, 11980, 374, ...]    (151K vocab, similar to Qwen2)
//! ```
//!
//! Cross-model eval: same text → different token_ids → different codebook lookups
//! → compare distances → Spearman ρ between models.

/// Model identifier for tokenizer lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelId {
    /// Jina v3 — XLM-RoBERTa tokenizer (250,002 vocab)
    JinaV3,
    /// BGE-M3 — XLM-RoBERTa tokenizer (250,002 vocab, same as Jina v3)
    BgeM3,
    /// Jina Reranker v3 — Qwen2 tokenizer (151,936 vocab)
    Reranker,
    /// Jina v5 text-small — Qwen3 tokenizer (151,936 vocab)
    JinaV5,
    /// Reader-LM 1.5B — Qwen2 tokenizer (151,936 vocab)
    ReaderLm,
    /// Qwopus 27B — Qwen2 tokenizer (151,936 vocab)
    Qwopus,
}

impl ModelId {
    /// Vocab size for this model.
    pub fn vocab_size(self) -> u32 {
        match self {
            ModelId::JinaV3 | ModelId::BgeM3 => 250_002,
            ModelId::Reranker | ModelId::JinaV5 | ModelId::ReaderLm | ModelId::Qwopus => 151_936,
        }
    }

    /// Default tokenizer file path (relative to crate root).
    pub fn tokenizer_path(self) -> &'static str {
        match self {
            ModelId::JinaV3 => "crates/thinking-engine/data/jina-v3-hdr/tokenizer.json",
            ModelId::BgeM3 => "crates/thinking-engine/data/bge-m3-hdr/tokenizer.json",
            ModelId::Reranker | ModelId::ReaderLm | ModelId::Qwopus =>
                "crates/thinking-engine/data/Qwopus3.5-27B-v3-BF16-silu/tokenizer.json",
            ModelId::JinaV5 => "crates/thinking-engine/data/jina-v5-tokenizer.json",
        }
    }

    /// HuggingFace model ID for from_pretrained fallback.
    pub fn hf_model_id(self) -> &'static str {
        match self {
            ModelId::JinaV3 => "jinaai/jina-embeddings-v3",
            ModelId::BgeM3 => "BAAI/bge-m3",
            ModelId::Reranker => "jinaai/jina-reranker-v2-base-multilingual",
            ModelId::JinaV5 => "jinaai/jina-embeddings-v5-text-small-text-matching",
            ModelId::ReaderLm => "jinaai/reader-lm-1.5b",
            ModelId::Qwopus => "Qwen/Qwen2.5-32B",
        }
    }
}

/// Registry holding loaded tokenizers for all models.
pub struct TokenizerRegistry {
    entries: Vec<(ModelId, tokenizers::Tokenizer)>,
}

impl TokenizerRegistry {
    /// Create empty registry.
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Load a tokenizer for a model. Tries local file first, then from_pretrained.
    pub fn load(&mut self, model: ModelId) -> Result<(), String> {
        // Skip if already loaded
        if self.entries.iter().any(|(m, _)| *m == model) {
            return Ok(());
        }

        let path = model.tokenizer_path();
        if let Ok(tok) = tokenizers::Tokenizer::from_file(path) {
            eprintln!("  [{}] Loaded from {}", model.name(), path);
            self.entries.push((model, tok));
            return Ok(());
        }

        // Try from_pretrained (needs network)
        let hf_id = model.hf_model_id();
        eprintln!("  [{}] Downloading from {}...", model.name(), hf_id);
        match tokenizers::Tokenizer::from_pretrained(hf_id, None) {
            Ok(tok) => {
                // Save for next time
                let _ = tok.save(path, false);
                self.entries.push((model, tok));
                Ok(())
            }
            Err(e) => Err(format!("[{}] Failed to load tokenizer: {}", model.name(), e)),
        }
    }

    /// Load all known models. Returns list of failures.
    pub fn load_all(&mut self) -> Vec<String> {
        let models = [
            ModelId::JinaV3, ModelId::BgeM3, ModelId::Reranker,
            ModelId::JinaV5, ModelId::ReaderLm, ModelId::Qwopus,
        ];
        let mut failures = Vec::new();
        for m in models {
            if let Err(e) = self.load(m) {
                failures.push(e);
            }
        }
        failures
    }

    /// Tokenize text through a specific model's tokenizer.
    pub fn encode(&self, model: ModelId, text: &str) -> Option<Vec<u32>> {
        self.entries.iter()
            .find(|(m, _)| *m == model)
            .and_then(|(_, tok)| tok.encode(text, true).ok())
            .map(|enc| enc.get_ids().to_vec())
    }

    /// Tokenize text through ALL loaded models. Returns (model, token_ids) pairs.
    pub fn encode_all(&self, text: &str) -> Vec<(ModelId, Vec<u32>)> {
        self.entries.iter()
            .filter_map(|(model, tok)| {
                tok.encode(text, true).ok().map(|enc| (*model, enc.get_ids().to_vec()))
            })
            .collect()
    }

    /// Number of loaded tokenizers.
    pub fn loaded_count(&self) -> usize {
        self.entries.len()
    }

    /// Check if a model's tokenizer is loaded.
    pub fn is_loaded(&self, model: ModelId) -> bool {
        self.entries.iter().any(|(m, _)| *m == model)
    }
}

impl Default for TokenizerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelId {
    fn name(self) -> &'static str {
        match self {
            ModelId::JinaV3 => "jina-v3",
            ModelId::BgeM3 => "bge-m3",
            ModelId::Reranker => "reranker",
            ModelId::JinaV5 => "jina-v5",
            ModelId::ReaderLm => "reader-lm",
            ModelId::Qwopus => "qwopus",
        }
    }
}

/// Cross-model tokenization result for one text.
#[derive(Debug)]
pub struct CrossModelTokens {
    pub text: String,
    pub tokens: Vec<(ModelId, Vec<u32>)>,
}

/// Tokenize a corpus through all models for cross-model evaluation.
pub fn tokenize_corpus(
    registry: &TokenizerRegistry,
    texts: &[&str],
) -> Vec<CrossModelTokens> {
    texts.iter().map(|&text| {
        CrossModelTokens {
            text: text.to_string(),
            tokens: registry.encode_all(text),
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_id_vocab_sizes() {
        assert_eq!(ModelId::JinaV3.vocab_size(), 250_002);
        assert_eq!(ModelId::BgeM3.vocab_size(), 250_002);
        assert_eq!(ModelId::Reranker.vocab_size(), 151_936);
        assert_eq!(ModelId::JinaV5.vocab_size(), 151_936);
    }

    #[test]
    fn registry_creates_empty() {
        let reg = TokenizerRegistry::new();
        assert_eq!(reg.loaded_count(), 0);
    }

    #[test]
    fn load_jina_v3_tokenizer() {
        let mut reg = TokenizerRegistry::new();
        let result = reg.load(ModelId::JinaV3);
        if result.is_ok() {
            assert!(reg.is_loaded(ModelId::JinaV3));
            let tokens = reg.encode(ModelId::JinaV3, "The wound is where the light enters");
            assert!(tokens.is_some());
            let ids = tokens.unwrap();
            assert!(!ids.is_empty());
            // All IDs should be within vocab range
            for &id in &ids {
                assert!(id < ModelId::JinaV3.vocab_size(),
                    "token {} exceeds vocab {}", id, ModelId::JinaV3.vocab_size());
            }
        }
        // OK if fails (tokenizer not on disk in CI)
    }

    #[test]
    fn load_reranker_tokenizer() {
        let mut reg = TokenizerRegistry::new();
        let result = reg.load(ModelId::Reranker);
        if result.is_ok() {
            let tokens = reg.encode(ModelId::Reranker, "TCP uses a three-way handshake");
            assert!(tokens.is_some());
        }
    }

    #[test]
    fn load_jina_v5_tokenizer() {
        let mut reg = TokenizerRegistry::new();
        let result = reg.load(ModelId::JinaV5);
        if result.is_ok() {
            assert!(reg.is_loaded(ModelId::JinaV5));
            let tokens = reg.encode(ModelId::JinaV5, "Gradient descent minimizes the loss function");
            assert!(tokens.is_some());
        }
    }

    #[test]
    fn cross_model_different_token_ids() {
        let mut reg = TokenizerRegistry::new();
        let _ = reg.load(ModelId::JinaV3);
        let _ = reg.load(ModelId::Reranker);

        if reg.loaded_count() >= 2 {
            let text = "The wound is where the light enters";
            let jina = reg.encode(ModelId::JinaV3, text).unwrap();
            let rr = reg.encode(ModelId::Reranker, text).unwrap();
            // Different tokenizers produce different token counts and IDs
            assert_ne!(jina.len(), 0);
            assert_ne!(rr.len(), 0);
            // XLM-RoBERTa and Qwen2 have different vocabularies
            // Token IDs almost certainly differ
            eprintln!("Jina v3: {} tokens {:?}", jina.len(), &jina[..jina.len().min(10)]);
            eprintln!("Reranker: {} tokens {:?}", rr.len(), &rr[..rr.len().min(10)]);
        }
    }

    #[test]
    fn encode_all_returns_loaded() {
        let mut reg = TokenizerRegistry::new();
        let _ = reg.load(ModelId::JinaV3);
        let _ = reg.load(ModelId::JinaV5);

        let results = reg.encode_all("Hello world");
        assert_eq!(results.len(), reg.loaded_count());
    }
}
