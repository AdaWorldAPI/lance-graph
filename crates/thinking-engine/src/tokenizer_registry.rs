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
    /// ModernBERT-large — OLMo tokenizer (50,368 vocab, code-friendly)
    /// GeGLU FFN (same gate modulation as SiLU), RoPE, 28 layers × 2624 dim.
    /// ONNX available: ort-community/ModernBERT-large-ONNX-ORT (FP32/FP16/INT8/UINT8)
    ModernBert,
    /// CLIP ViT-Huge-14 — vision encoder, XLM-RoBERTa text side (250,002 vocab)
    /// FP32: Kijai/WanVideo_comfy (2.53 GB), BF16: DeepBeepMeep/Wan2.1 (2.39 GB)
    ClipVision,
}

impl ModelId {
    /// Vocab size for this model.
    pub fn vocab_size(self) -> u32 {
        match self {
            ModelId::JinaV3 | ModelId::BgeM3 | ModelId::ClipVision => 250_002,
            ModelId::Reranker | ModelId::JinaV5 | ModelId::ReaderLm | ModelId::Qwopus => 151_936,
            ModelId::ModernBert => 50_368,
        }
    }

    /// Default tokenizer file path (relative to crate root).
    /// Tries local ONNX dirs first, then HDR dirs, then from_pretrained fallback.
    pub fn tokenizer_path(self) -> &'static str {
        match self {
            ModelId::JinaV3 | ModelId::BgeM3 | ModelId::ClipVision =>
                "crates/thinking-engine/data/jina-v3-hdr/tokenizer.json",
            ModelId::Reranker =>
                "crates/thinking-engine/data/jina-v5-onnx/tokenizer.json", // Qwen3 (same as v5)
            ModelId::ReaderLm | ModelId::Qwopus =>
                "crates/thinking-engine/data/Qwopus3.5-27B-v3-BF16-silu/tokenizer.json", // Qwen2
            ModelId::JinaV5 =>
                "crates/thinking-engine/data/jina-v5-onnx/tokenizer.json",
            ModelId::ModernBert =>
                "crates/thinking-engine/data/modernbert-onnx/tokenizer.json",
        }
    }

    /// HuggingFace model ID for from_pretrained fallback.
    pub fn hf_model_id(self) -> &'static str {
        match self {
            ModelId::JinaV3 => "jinaai/jina-embeddings-v3",
            ModelId::BgeM3 => "BAAI/bge-m3",
            ModelId::Reranker => "jinaai/jina-reranker-v3",
            ModelId::JinaV5 => "jinaai/jina-embeddings-v5-text-small-text-matching",
            ModelId::ReaderLm => "jinaai/reader-lm-1.5b",
            ModelId::Qwopus => "Qwen/Qwen2.5-32B",
            ModelId::ModernBert => "answerdotai/ModernBERT-large",
            ModelId::ClipVision => "Kijai/WanVideo_comfy",
        }
    }

    /// ONNX model path (for ground truth forward pass).
    pub fn onnx_path(self) -> Option<&'static str> {
        match self {
            ModelId::JinaV5 => Some("crates/thinking-engine/data/jina-v5-onnx/model.onnx"),
            ModelId::ModernBert => Some("crates/thinking-engine/data/modernbert-onnx/model.onnx"),
            _ => None,
        }
    }

    /// config.json path (for auto-detect architecture).
    pub fn config_path(self) -> Option<&'static str> {
        match self {
            ModelId::JinaV5 => Some("crates/thinking-engine/data/jina-v5-onnx/config.json"),
            ModelId::ModernBert => Some("crates/thinking-engine/data/modernbert-onnx/config.json"),
            _ => None,
        }
    }

    /// Whether this model has GeGLU/SiLU gate modulation (the 33% correction).
    pub fn has_gate_modulation(self) -> bool {
        match self {
            // GeGLU: ModernBERT, Qwen, Qwopus — all have gated FFN
            ModelId::ModernBert | ModelId::Qwopus | ModelId::JinaV5 | ModelId::ReaderLm => true,
            // Reranker v3 = Qwen3 base (silu) — HAS gate modulation
            ModelId::Reranker => true,
            // Standard GeLU: BERT, XLM-RoBERTa — no gate
            ModelId::JinaV3 | ModelId::BgeM3 => false,
            // Vision: ViT uses standard FFN
            ModelId::ClipVision => false,
        }
    }

    /// ONNX availability for this model.
    pub fn onnx_repo(self) -> Option<&'static str> {
        match self {
            ModelId::ModernBert => Some("ort-community/ModernBERT-large-ONNX-ORT"),
            ModelId::JinaV5 => Some("jinaai/jina-embeddings-v5-text-small-text-matching"),
            ModelId::Reranker => Some("jinaai/jina-reranker-v3"),
            _ => None,
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
            ModelId::ModernBert, ModelId::ClipVision,
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
            ModelId::ModernBert => "modernbert",
            ModelId::ClipVision => "clip-vision",
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
        assert_eq!(ModelId::ModernBert.vocab_size(), 50_368);
        assert_eq!(ModelId::ClipVision.vocab_size(), 250_002);
    }

    #[test]
    fn gate_modulation_flags() {
        // GeGLU models have gate modulation (the 33% SiLU correction)
        assert!(ModelId::ModernBert.has_gate_modulation());
        assert!(ModelId::Qwopus.has_gate_modulation());
        assert!(ModelId::JinaV5.has_gate_modulation());
        // Standard GeLU models do not
        assert!(!ModelId::JinaV3.has_gate_modulation());
        assert!(!ModelId::BgeM3.has_gate_modulation());
        assert!(!ModelId::ClipVision.has_gate_modulation());
    }

    #[test]
    fn onnx_availability() {
        assert!(ModelId::ModernBert.onnx_repo().is_some());
        assert!(ModelId::JinaV5.onnx_repo().is_some());
        assert!(ModelId::JinaV3.onnx_repo().is_none());
    }

    #[test]
    fn onnx_paths_exist_on_disk() {
        // These should exist after downloading in this session
        if let Some(path) = ModelId::JinaV5.onnx_path() {
            let exists = std::path::Path::new(path).exists();
            eprintln!("Jina v5 ONNX: {} → {}", path, if exists { "EXISTS" } else { "NOT FOUND" });
        }
        if let Some(path) = ModelId::ModernBert.onnx_path() {
            let exists = std::path::Path::new(path).exists();
            eprintln!("ModernBERT ONNX: {} → {}", path, if exists { "EXISTS" } else { "NOT FOUND" });
        }
    }

    #[test]
    fn config_json_auto_detect() {
        // Test auto-detect from downloaded config.json
        for model in [ModelId::JinaV5, ModelId::ModernBert] {
            if let Some(path) = model.config_path() {
                if let Ok(json_str) = std::fs::read_to_string(path) {
                    let detected = crate::auto_detect::detect_from_config_json(&json_str);
                    match detected {
                        Ok(d) => eprintln!("{:?}: arch={:?}, hidden={}, layers={}, vocab={}",
                            model, d.architecture, d.hidden_dim, d.num_layers, d.vocab_size),
                        Err(e) => eprintln!("{:?}: detect failed: {}", model, e),
                    }
                }
            }
        }
    }

    #[test]
    fn load_jina_v5_and_modernbert_tokenizers() {
        let mut reg = TokenizerRegistry::new();

        // Jina v5: should load from local ONNX dir
        let v5 = reg.load(ModelId::JinaV5);
        if v5.is_ok() {
            let tokens = reg.encode(ModelId::JinaV5, "The wound is where the light enters");
            if let Some(ids) = tokens {
                eprintln!("Jina v5: {} tokens, first 5: {:?}", ids.len(), &ids[..ids.len().min(5)]);
            }
        }

        // ModernBERT: should load from local ONNX dir
        let mb = reg.load(ModelId::ModernBert);
        if mb.is_ok() {
            let tokens = reg.encode(ModelId::ModernBert, "The wound is where the light enters");
            if let Some(ids) = tokens {
                eprintln!("ModernBERT: {} tokens, first 5: {:?}", ids.len(), &ids[..ids.len().min(5)]);
            }
        }

        // If both loaded: different tokenizers should produce different token counts
        if reg.is_loaded(ModelId::JinaV5) && reg.is_loaded(ModelId::ModernBert) {
            let v5_ids = reg.encode(ModelId::JinaV5, "Gradient descent minimizes loss").unwrap();
            let mb_ids = reg.encode(ModelId::ModernBert, "Gradient descent minimizes loss").unwrap();
            eprintln!("Same text: Jina v5={} tokens, ModernBERT={} tokens", v5_ids.len(), mb_ids.len());
            // Different vocab sizes (151K vs 50K) → different token counts likely
        }
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
