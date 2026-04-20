//! **LAB-ONLY.** Model architecture auto-detection from `config.json`.
//!
//! D0.5 deliverable from the codec-sweep plan — CODING_PRACTICES.md gap 1
//! remediation ("auto-detect model type, don't hardcode model names").
//!
//! Reads the `config.json` sitting next to a safetensors model and returns
//! a [`ModelFingerprint`] with the defaults the codec JIT needs:
//! architecture family, hidden dim, layer count, tokenizer class, vocab
//! size, suggested [`LaneWidth`] and [`Distance`] for the sweep.
//!
//! Consumed by [`WireTokenAgreement`] handler when the client omits
//! `tensor_view.lane_width` — the handler auto-detects and populates
//! the `CodecParams::lane_width` field.
//!
//! Pattern mirrors EmbedAnything's `auto_detect.rs` — 6 tests across
//! `llama`, `qwen3`, `bert`, `modernbert`, `xlm-roberta`, and a generic
//! fallback path.

use lance_graph_contract::cam::{Distance, LaneWidth};
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Auto-detected model properties consumed by the codec-sweep lab surface.
///
/// Produced by [`detect`] from `<model_path>/config.json`. Carries the
/// minimum shape information the JIT kernel needs to compile a decode
/// kernel for this tensor family without requiring the client to specify
/// every parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFingerprint {
    /// Architecture family string from `config.json::model_type` or
    /// the first entry of `config.json::architectures`. Examples:
    /// `"llama"`, `"qwen3"`, `"bert"`, `"modernbert"`, `"xlm-roberta"`.
    pub architecture: String,
    /// `hidden_size` (a.k.a. `d_model`) — embedding / MLP width.
    pub hidden_size: u32,
    /// `num_hidden_layers` (a.k.a. `num_layers` / `n_layer`).
    pub n_layers: u32,
    /// Tokenizer class from `tokenizer_config.json::tokenizer_class`
    /// when available; empty string otherwise.
    pub tokenizer_class: String,
    /// `vocab_size` from `config.json`.
    pub vocab_size: u32,
    /// Suggested JIT lane width. BF16 for architectures that ship
    /// BF16 weights (llama, qwen3); F32x16 as the cautious default.
    pub default_lane_width: LaneWidth,
    /// Suggested ADC variant. AdcU8 by default; AdcI8 when the codec
    /// family expects bipolar cancellation (flagged per-architecture).
    pub default_distance: Distance,
}

/// Errors returned by [`detect`] when `config.json` is missing or
/// malformed. The handler surfaces these verbatim to the REST client;
/// no silent fallbacks.
#[derive(Debug)]
pub enum DetectError {
    /// `config.json` not found next to the safetensors file.
    ConfigMissing { path: String },
    /// IO failure reading `config.json`.
    Io { path: String, source: std::io::Error },
    /// `config.json` failed JSON parse.
    Parse { path: String, source: serde_json::Error },
    /// `config.json` missing a required field (listed in `field`).
    MissingField { path: String, field: &'static str },
}

impl std::fmt::Display for DetectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConfigMissing { path } => write!(f, "config.json missing at {path}"),
            Self::Io { path, source } => write!(f, "io error reading {path}: {source}"),
            Self::Parse { path, source } => write!(f, "parse error in {path}: {source}"),
            Self::MissingField { path, field } => {
                write!(f, "config.json at {path} missing required field `{field}`")
            }
        }
    }
}

impl std::error::Error for DetectError {}

/// Minimal serde shape of `config.json` (Hugging Face convention).
/// Only the fields the codec JIT cares about are captured; extras are
/// ignored silently via `#[serde(other)]`-friendly `Value` catch-all.
#[derive(Debug, Deserialize)]
struct HfConfig {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Option<Vec<String>>,
    hidden_size: Option<u32>,
    #[serde(alias = "d_model")]
    d_model: Option<u32>,
    #[serde(alias = "num_hidden_layers", alias = "n_layer", alias = "num_layers")]
    num_hidden_layers: Option<u32>,
    vocab_size: Option<u32>,
    #[serde(default)]
    torch_dtype: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    tokenizer_class: Option<String>,
}

/// Read `<model_path>/config.json` and infer a [`ModelFingerprint`].
///
/// `model_path` is the directory containing the safetensors files AND
/// `config.json` (standard Hugging Face layout).
pub fn detect(model_path: &Path) -> Result<ModelFingerprint, DetectError> {
    let config_path = model_path.join("config.json");
    let path_str = config_path.display().to_string();

    if !config_path.exists() {
        return Err(DetectError::ConfigMissing { path: path_str });
    }

    let raw = fs::read_to_string(&config_path)
        .map_err(|e| DetectError::Io { path: path_str.clone(), source: e })?;
    let cfg: HfConfig = serde_json::from_str(&raw)
        .map_err(|e| DetectError::Parse { path: path_str.clone(), source: e })?;

    let architecture = cfg
        .model_type
        .clone()
        .or_else(|| cfg.architectures.as_ref().and_then(|a| a.first().cloned()))
        .unwrap_or_else(|| "generic".to_string())
        .to_lowercase();

    let hidden_size = cfg
        .hidden_size
        .or(cfg.d_model)
        .ok_or(DetectError::MissingField { path: path_str.clone(), field: "hidden_size" })?;

    let n_layers = cfg
        .num_hidden_layers
        .ok_or(DetectError::MissingField { path: path_str.clone(), field: "num_hidden_layers" })?;

    let vocab_size = cfg
        .vocab_size
        .ok_or(DetectError::MissingField { path: path_str.clone(), field: "vocab_size" })?;

    let default_lane_width = suggest_lane_width(&architecture, cfg.torch_dtype.as_deref());
    let default_distance = suggest_distance(&architecture);

    // Tokenizer config is best-effort — missing → empty string (not an error).
    let tok_path = model_path.join("tokenizer_config.json");
    let tokenizer_class = if tok_path.exists() {
        fs::read_to_string(&tok_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<TokenizerConfig>(&raw).ok())
            .and_then(|tc| tc.tokenizer_class)
            .unwrap_or_default()
    } else {
        String::new()
    };

    Ok(ModelFingerprint {
        architecture,
        hidden_size,
        n_layers,
        tokenizer_class,
        vocab_size,
        default_lane_width,
        default_distance,
    })
}

/// Per-architecture lane-width suggestion.
///
/// Routes architectures that ship BF16 weights (llama, qwen, mistral) to
/// `BF16x32` (AMX-ready path). Others default to `F32x16` (AVX-512 baseline).
fn suggest_lane_width(architecture: &str, torch_dtype: Option<&str>) -> LaneWidth {
    // Explicit dtype signal wins if present.
    if let Some(dtype) = torch_dtype {
        match dtype.to_lowercase().as_str() {
            "bfloat16" | "bf16" => return LaneWidth::BF16x32,
            "float32" | "fp32" | "f32" => return LaneWidth::F32x16,
            _ => {}
        }
    }
    // Fall back to architecture family heuristic.
    match architecture {
        "llama" | "qwen" | "qwen2" | "qwen3" | "mistral" | "mixtral" => LaneWidth::BF16x32,
        _ => LaneWidth::F32x16,
    }
}

/// Per-architecture distance-variant suggestion.
///
/// All families currently default to AdcU8 (palette-index quantization).
/// Reserved for future bipolar families (zipper codec, 5^5 signed).
fn suggest_distance(_architecture: &str) -> Distance {
    Distance::AdcU8
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Create a temp directory + write `config.json` with the given body.
    /// Returns the directory (as a Drop-guarded TempDir stand-in via raw PathBuf).
    fn fixture(name: &str, config_body: &str, tokenizer_body: Option<&str>) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("jc_auto_detect_{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::File::create(dir.join("config.json"))
            .unwrap()
            .write_all(config_body.as_bytes())
            .unwrap();
        if let Some(tok) = tokenizer_body {
            fs::File::create(dir.join("tokenizer_config.json"))
                .unwrap()
                .write_all(tok.as_bytes())
                .unwrap();
        }
        dir
    }

    #[test]
    fn detects_llama() {
        let dir = fixture(
            "llama",
            r#"{
                "model_type": "llama",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "vocab_size": 128256,
                "torch_dtype": "bfloat16"
            }"#,
            None,
        );
        let fp = detect(&dir).unwrap();
        assert_eq!(fp.architecture, "llama");
        assert_eq!(fp.hidden_size, 4096);
        assert_eq!(fp.n_layers, 32);
        assert_eq!(fp.vocab_size, 128_256);
        assert_eq!(fp.default_lane_width, LaneWidth::BF16x32);
        assert_eq!(fp.default_distance, Distance::AdcU8);
    }

    #[test]
    fn detects_qwen3_with_tokenizer() {
        let dir = fixture(
            "qwen3",
            r#"{
                "model_type": "qwen3",
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "vocab_size": 151936,
                "torch_dtype": "bfloat16"
            }"#,
            Some(r#"{"tokenizer_class": "Qwen2Tokenizer"}"#),
        );
        let fp = detect(&dir).unwrap();
        assert_eq!(fp.architecture, "qwen3");
        assert_eq!(fp.tokenizer_class, "Qwen2Tokenizer");
        assert_eq!(fp.default_lane_width, LaneWidth::BF16x32);
    }

    #[test]
    fn detects_bert_defaults_f32x16() {
        let dir = fixture(
            "bert",
            r#"{
                "model_type": "bert",
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "vocab_size": 30522
            }"#,
            None,
        );
        let fp = detect(&dir).unwrap();
        assert_eq!(fp.architecture, "bert");
        assert_eq!(fp.default_lane_width, LaneWidth::F32x16);
    }

    #[test]
    fn detects_modernbert_via_architectures_fallback() {
        // No `model_type`, only `architectures` — falls back to first entry.
        let dir = fixture(
            "modernbert",
            r#"{
                "architectures": ["ModernBertModel"],
                "hidden_size": 1024,
                "num_hidden_layers": 22,
                "vocab_size": 50368
            }"#,
            None,
        );
        let fp = detect(&dir).unwrap();
        assert_eq!(fp.architecture, "modernbertmodel");
        assert_eq!(fp.default_lane_width, LaneWidth::F32x16);
    }

    #[test]
    fn detects_xlm_roberta_via_d_model_alias() {
        // Some configs use `d_model` instead of `hidden_size`.
        let dir = fixture(
            "xlm-roberta",
            r#"{
                "model_type": "xlm-roberta",
                "d_model": 1024,
                "num_hidden_layers": 24,
                "vocab_size": 250002
            }"#,
            None,
        );
        let fp = detect(&dir).unwrap();
        assert_eq!(fp.architecture, "xlm-roberta");
        assert_eq!(fp.hidden_size, 1024);
    }

    #[test]
    fn generic_fallback_when_model_type_missing() {
        // No `model_type`, no `architectures` — architecture = "generic".
        let dir = fixture(
            "generic",
            r#"{
                "hidden_size": 512,
                "num_hidden_layers": 6,
                "vocab_size": 32000
            }"#,
            None,
        );
        let fp = detect(&dir).unwrap();
        assert_eq!(fp.architecture, "generic");
        assert_eq!(fp.default_lane_width, LaneWidth::F32x16);
    }

    #[test]
    fn missing_config_yields_typed_error() {
        let dir = std::env::temp_dir().join("jc_auto_detect_missing");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let err = detect(&dir).unwrap_err();
        assert!(matches!(err, DetectError::ConfigMissing { .. }));
    }

    #[test]
    fn missing_hidden_size_yields_typed_error() {
        let dir = fixture(
            "no_hidden",
            r#"{"model_type": "bert", "num_hidden_layers": 12, "vocab_size": 30522}"#,
            None,
        );
        let err = detect(&dir).unwrap_err();
        assert!(matches!(err, DetectError::MissingField { field: "hidden_size", .. }));
    }
}
