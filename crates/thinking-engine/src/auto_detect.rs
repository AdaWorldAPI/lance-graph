//! Auto-detect model architecture from config metadata.
//!
//! EmbedAnything reads config.json and routes to the right model.
//! We read GGUF metadata or config.json and route to the right
//! codebook builder + tokenizer + lens configuration.
//!
//! No hardcoded paths per model. One entry point, automatic routing.

use std::collections::HashMap;

/// Known model architectures and their properties.
#[derive(Clone, Debug, PartialEq)]
pub enum Architecture {
    /// XLM-RoBERTa (Jina v3, BGE-M3). SentencePiece BPE, 250K vocab.
    XlmRoberta,
    /// Qwen2/Qwen3 (Reranker, Qwopus, Jina v5). Qwen BPE, 151K vocab.
    Qwen,
    /// ModernBERT (OLMo tokenizer, code-friendly). 50K vocab.
    ModernBert,
    /// BERT base (sentence-transformers, MiniLM). WordPiece, 30K vocab.
    Bert,
    /// Unknown architecture.
    Unknown(String),
}

/// Detected model configuration.
#[derive(Clone, Debug)]
pub struct DetectedModel {
    pub architecture: Architecture,
    pub name: String,
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    /// Recommended lens for this model.
    pub recommended_lens: Option<super::builder::Lens>,
    /// Per-role gate policy.
    pub gate_modulated: bool,
    /// Whether this is an MoE model.
    pub is_moe: bool,
    pub num_experts: Option<usize>,
}

/// Detect architecture from a config.json content string.
pub fn detect_from_config_json(json_str: &str) -> Result<DetectedModel, String> {
    let parsed: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| format!("invalid config.json: {}", e))?;

    // Architecture detection: check "architectures" array or "model_type"
    let arch_str = parsed.get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .or_else(|| parsed.get("model_type").and_then(|v| v.as_str()))
        .unwrap_or("unknown");

    let architecture = match arch_str {
        s if s.contains("XLMRoberta") || s.contains("xlm-roberta") => Architecture::XlmRoberta,
        s if s.contains("Qwen") || s.contains("qwen") => Architecture::Qwen,
        s if s.contains("ModernBert") || s.contains("modernbert") => Architecture::ModernBert,
        s if s.contains("Bert") || s.contains("bert") => Architecture::Bert,
        s if s.contains("JinaBert") => Architecture::XlmRoberta, // Jina BERT = modified XLM-RoBERTa
        other => Architecture::Unknown(other.into()),
    };

    let vocab_size = parsed.get("vocab_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let hidden_dim = parsed.get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let num_layers = parsed.get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let num_heads = parsed.get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let num_experts = parsed.get("num_experts")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);

    let recommended_lens = match &architecture {
        Architecture::XlmRoberta if vocab_size >= 250_000 => {
            Some(super::builder::Lens::Jina) // or BgeM3, both XLM-RoBERTa
        }
        Architecture::Qwen if vocab_size >= 150_000 => {
            Some(super::builder::Lens::Reranker) // Qwen tokenizer = reranker
        }
        _ => None,
    };

    let name = parsed.get("_name_or_path")
        .or_else(|| parsed.get("model_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    Ok(DetectedModel {
        architecture,
        name,
        vocab_size,
        hidden_dim,
        num_layers,
        num_heads,
        recommended_lens,
        gate_modulated: num_experts.is_some() || num_layers > 24,
        is_moe: num_experts.is_some(),
        num_experts,
    })
}

/// Detect architecture from GGUF metadata key-value pairs.
/// GGUF stores metadata as typed KV pairs in the header.
pub fn detect_from_gguf_metadata(metadata: &HashMap<String, String>) -> DetectedModel {
    let arch = metadata.get("general.architecture")
        .or_else(|| metadata.get("general.name"))
        .cloned()
        .unwrap_or_default();

    let architecture = if arch.contains("bert") || arch.contains("roberta") {
        Architecture::XlmRoberta
    } else if arch.contains("qwen") {
        Architecture::Qwen
    } else if arch.contains("modernbert") {
        Architecture::ModernBert
    } else {
        Architecture::Unknown(arch.clone())
    };

    let vocab_size = metadata.get("tokenizer.ggml.tokens")
        .and_then(|v| v.parse().ok())
        .or_else(|| metadata.get("bert.token_count").and_then(|v| v.parse().ok()))
        .unwrap_or(0);

    let hidden_dim = metadata.get("bert.embedding_length")
        .or_else(|| metadata.get("qwen2.embedding_length"))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let num_layers = metadata.get("bert.block_count")
        .or_else(|| metadata.get("qwen2.block_count"))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let num_heads = metadata.get("bert.attention.head_count")
        .or_else(|| metadata.get("qwen2.attention.head_count"))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let num_experts = metadata.get("qwen2.expert_count")
        .and_then(|v| v.parse().ok());

    let name = metadata.get("general.name").cloned().unwrap_or_default();

    let recommended_lens = match &architecture {
        Architecture::XlmRoberta => Some(super::builder::Lens::Jina),
        Architecture::Qwen => Some(super::builder::Lens::Reranker),
        _ => None,
    };

    DetectedModel {
        architecture,
        name,
        vocab_size,
        hidden_dim,
        num_layers,
        num_heads,
        recommended_lens,
        gate_modulated: num_experts.is_some(),
        is_moe: num_experts.is_some(),
        num_experts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_jina_v3() {
        let config = r#"{
            "architectures": ["XLMRobertaForMaskedLM"],
            "vocab_size": 250002,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "_name_or_path": "jinaai/jina-embeddings-v3"
        }"#;
        let model = detect_from_config_json(config).unwrap();
        assert_eq!(model.architecture, Architecture::XlmRoberta);
        assert_eq!(model.vocab_size, 250_002);
        assert_eq!(model.hidden_dim, 1024);
        assert!(model.recommended_lens.is_some());
    }

    #[test]
    fn detect_qwen3() {
        let config = r#"{
            "architectures": ["Qwen3ForCausalLM"],
            "vocab_size": 151936,
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "model_type": "qwen3"
        }"#;
        let model = detect_from_config_json(config).unwrap();
        assert_eq!(model.architecture, Architecture::Qwen);
        assert_eq!(model.vocab_size, 151_936);
    }

    #[test]
    fn detect_modernbert() {
        let config = r#"{
            "architectures": ["ModernBertModel"],
            "vocab_size": 50368,
            "hidden_size": 2624,
            "num_hidden_layers": 28,
            "num_attention_heads": 16
        }"#;
        let model = detect_from_config_json(config).unwrap();
        assert_eq!(model.architecture, Architecture::ModernBert);
        assert_eq!(model.vocab_size, 50_368);
        assert_eq!(model.hidden_dim, 2624);
    }

    #[test]
    fn detect_moe() {
        let config = r#"{
            "architectures": ["Qwen3ForCausalLM"],
            "vocab_size": 202048,
            "hidden_size": 5120,
            "num_hidden_layers": 48,
            "num_attention_heads": 40,
            "num_experts": 128
        }"#;
        let model = detect_from_config_json(config).unwrap();
        assert!(model.is_moe);
        assert_eq!(model.num_experts, Some(128));
        assert!(model.gate_modulated);
    }

    #[test]
    fn detect_from_gguf() {
        let mut meta = HashMap::new();
        meta.insert("general.architecture".into(), "bert".into());
        meta.insert("general.name".into(), "jina-embeddings-v3".into());
        meta.insert("bert.embedding_length".into(), "1024".into());
        meta.insert("bert.block_count".into(), "24".into());

        let model = detect_from_gguf_metadata(&meta);
        assert_eq!(model.architecture, Architecture::XlmRoberta);
        assert_eq!(model.hidden_dim, 1024);
    }

    #[test]
    fn detect_unknown_graceful() {
        let config = r#"{"architectures": ["SomeNewModel"], "vocab_size": 100000}"#;
        let model = detect_from_config_json(config).unwrap();
        assert!(matches!(model.architecture, Architecture::Unknown(_)));
    }
}
