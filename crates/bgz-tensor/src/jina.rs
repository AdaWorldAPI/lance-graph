//! Jina v3 API client for embedding ground truth collection.
//!
//! Calls the Jina API to get real embeddings, computes cosine similarities,
//! and pairs them with Base17 L1 distances for SimilarityTable calibration.
//!
//! Zero dependencies in the library — this module provides:
//! 1. Request/response types for the Jina API
//! 2. Cosine similarity computation on raw f32 embeddings
//! 3. Base17 projection of Jina embeddings
//! 4. Ground truth pair collection for calibration
//!
//! The actual HTTP call is left to the caller (curl, ureq, etc.) — we just
//! parse the JSON response and compute metrics.

use crate::projection::Base17;

/// A single embedding from the Jina API.
#[derive(Clone, Debug)]
pub struct JinaEmbedding {
    /// The input text.
    pub text: String,
    /// 1024-dimensional f32 embedding vector.
    pub vector: Vec<f32>,
    /// Base17 projection of the embedding.
    pub base17: Base17,
}

impl JinaEmbedding {
    /// Create from raw embedding vector.
    pub fn new(text: String, vector: Vec<f32>) -> Self {
        let base17 = Base17::from_f32(&vector);
        JinaEmbedding { text, vector, base17 }
    }
}

/// Cosine similarity between two f32 vectors.
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

/// A ground truth pair: two texts with their API cosine and Base17 L1 distance.
#[derive(Clone, Debug)]
pub struct GroundTruthPair {
    pub text_a: String,
    pub text_b: String,
    /// Cosine similarity from Jina API f32 embeddings (ground truth).
    pub api_cosine: f64,
    /// L1 distance between Base17 projections.
    pub base17_l1: u32,
    /// Cosine similarity in Base17 space.
    pub base17_cosine: f64,
}

/// Collect ground truth pairs from Jina embeddings.
///
/// For each pair (i, j) where i < j, computes:
/// - API cosine similarity (from raw f32 embeddings)
/// - Base17 L1 distance
/// - Base17 cosine similarity
///
/// Returns pairs sorted by API cosine (descending).
pub fn collect_ground_truth(embeddings: &[JinaEmbedding]) -> Vec<GroundTruthPair> {
    let n = embeddings.len();
    let mut pairs = Vec::with_capacity(n * (n - 1) / 2);

    for i in 0..n {
        for j in (i + 1)..n {
            let api_cosine = cosine_f32(&embeddings[i].vector, &embeddings[j].vector);
            let base17_l1 = embeddings[i].base17.l1(&embeddings[j].base17);
            let base17_cosine = embeddings[i].base17.cosine(&embeddings[j].base17);

            pairs.push(GroundTruthPair {
                text_a: embeddings[i].text.clone(),
                text_b: embeddings[j].text.clone(),
                api_cosine,
                base17_l1,
                base17_cosine,
            });
        }
    }

    pairs.sort_by(|a, b| b.api_cosine.partial_cmp(&a.api_cosine).unwrap_or(std::cmp::Ordering::Equal));
    pairs
}

/// Extract calibration pairs for SimilarityTable from ground truth.
pub fn to_calibration_pairs(pairs: &[GroundTruthPair]) -> Vec<(u32, f64)> {
    pairs.iter().map(|p| (p.base17_l1, p.api_cosine)).collect()
}

/// Summary statistics from ground truth collection.
#[derive(Clone, Debug)]
pub struct GroundTruthSummary {
    pub n_texts: usize,
    pub n_pairs: usize,
    pub mean_api_cosine: f64,
    pub std_api_cosine: f64,
    pub mean_base17_l1: f64,
    pub pearson_l1_vs_cosine: f64,
    pub spearman_l1_vs_cosine: f64,
    pub mean_base17_cosine: f64,
    pub pearson_base17cos_vs_apicos: f64,
}

/// Compute summary statistics from ground truth pairs.
pub fn summarize_ground_truth(pairs: &[GroundTruthPair]) -> GroundTruthSummary {
    let n = pairs.len();
    if n == 0 {
        return GroundTruthSummary {
            n_texts: 0, n_pairs: 0,
            mean_api_cosine: 0.0, std_api_cosine: 0.0,
            mean_base17_l1: 0.0,
            pearson_l1_vs_cosine: 0.0, spearman_l1_vs_cosine: 0.0,
            mean_base17_cosine: 0.0, pearson_base17cos_vs_apicos: 0.0,
        };
    }

    let api_cos: Vec<f64> = pairs.iter().map(|p| p.api_cosine).collect();
    let b17_l1: Vec<f64> = pairs.iter().map(|p| p.base17_l1 as f64).collect();
    let b17_cos: Vec<f64> = pairs.iter().map(|p| p.base17_cosine).collect();
    let neg_l1: Vec<f64> = b17_l1.iter().map(|&d| -d).collect();

    let mean_cos = api_cos.iter().sum::<f64>() / n as f64;
    let var_cos = api_cos.iter().map(|c| (c - mean_cos).powi(2)).sum::<f64>() / n as f64;

    GroundTruthSummary {
        n_texts: 0, // caller fills in
        n_pairs: n,
        mean_api_cosine: mean_cos,
        std_api_cosine: var_cos.sqrt(),
        mean_base17_l1: b17_l1.iter().sum::<f64>() / n as f64,
        pearson_l1_vs_cosine: crate::quality::pearson(&neg_l1, &api_cos),
        spearman_l1_vs_cosine: crate::quality::spearman(&neg_l1, &api_cos),
        mean_base17_cosine: b17_cos.iter().sum::<f64>() / n as f64,
        pearson_base17cos_vs_apicos: crate::quality::pearson(&b17_cos, &api_cos),
    }
}

/// Parse a Jina API JSON response into embeddings.
///
/// Expected format:
/// ```json
/// {
///   "data": [
///     {"embedding": [0.1, 0.2, ...], "index": 0},
///     {"embedding": [-0.3, 0.4, ...], "index": 1}
///   ]
/// }
/// ```
///
/// Minimal JSON parser — no serde dependency. Handles the specific Jina response format.
pub fn parse_jina_response(json: &str, texts: &[&str]) -> Result<Vec<JinaEmbedding>, String> {
    let mut embeddings = Vec::new();

    // Find "data" array
    let data_start = json.find("\"data\"")
        .ok_or("missing 'data' field")?;
    let arr_start = json[data_start..].find('[')
        .ok_or("missing data array")?;
    let json_from_arr = &json[data_start + arr_start..];

    // Parse each embedding object
    let mut pos = 1; // skip opening [
    let mut idx = 0;

    while pos < json_from_arr.len() {
        // Find next "embedding" key
        let emb_start = match json_from_arr[pos..].find("\"embedding\"") {
            Some(p) => pos + p,
            None => break,
        };

        // Find the array start
        let arr_start = match json_from_arr[emb_start..].find('[') {
            Some(p) => emb_start + p,
            None => break,
        };

        // Find the matching ]
        let arr_end = match json_from_arr[arr_start..].find(']') {
            Some(p) => arr_start + p,
            None => break,
        };

        // Parse numbers between [ and ]
        let nums_str = &json_from_arr[arr_start + 1..arr_end];
        let vector: Vec<f32> = nums_str.split(',')
            .filter_map(|s| s.trim().parse::<f32>().ok())
            .collect();

        if !vector.is_empty() {
            let text = if idx < texts.len() { texts[idx].to_string() } else { format!("text_{}", idx) };
            embeddings.push(JinaEmbedding::new(text, vector));
        }

        pos = arr_end + 1;
        idx += 1;
    }

    if embeddings.is_empty() {
        Err("no embeddings parsed from response".into())
    } else {
        Ok(embeddings)
    }
}

/// Standard test texts for calibration — 20 pairs covering diverse similarity ranges.
pub fn calibration_texts() -> Vec<&'static str> {
    vec![
        // Near-identical pairs (high cosine)
        "The cat sat on the mat.",
        "A cat was sitting on the mat.",
        "Machine learning is a subset of artificial intelligence.",
        "ML is a branch of AI.",
        "The stock market crashed today.",
        "Today the stock market experienced a crash.",
        // Related but different (medium cosine)
        "Python is a programming language.",
        "JavaScript is used for web development.",
        "The sun rises in the east.",
        "Sunset paints the western sky.",
        "He ran quickly to the store.",
        "She walked slowly to the market.",
        // Unrelated (low cosine)
        "Quantum mechanics describes subatomic particles.",
        "The recipe calls for two cups of flour.",
        "Mount Everest is the tallest mountain.",
        "Abstract art challenges traditional aesthetics.",
        // Multilingual pairs
        "Hello, how are you?",
        "Hallo, wie geht es dir?",
        "Good morning, world.",
        "Guten Morgen, Welt.",
    ]
}

/// Build the curl command for Jina API embedding request.
pub fn jina_curl_command(texts: &[&str], api_key: &str) -> String {
    let input_json: String = texts.iter()
        .map(|t| format!("\"{}\"", t.replace('\"', "\\\"")))
        .collect::<Vec<_>>()
        .join(",");

    format!(
        r#"curl -s -X POST https://api.jina.ai/v1/embeddings \
  -H "Authorization: Bearer {}" \
  -H "Content-Type: application/json" \
  -d '{{"model":"jina-embeddings-v3","input":[{}],"task":"text-matching"}}'"#,
        api_key, input_json
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical() {
        let a = vec![1.0f32, 2.0, 3.0];
        let c = cosine_f32(&a, &a);
        assert!((c - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_opposite() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![-1.0f32, 0.0, 0.0];
        let c = cosine_f32(&a, &b);
        assert!((c - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let c = cosine_f32(&a, &b);
        assert!(c.abs() < 1e-10);
    }

    #[test]
    fn collect_ground_truth_basic() {
        let embs = vec![
            JinaEmbedding::new("hello".into(), vec![1.0; 1024]),
            JinaEmbedding::new("world".into(), vec![-1.0; 1024]),
            JinaEmbedding::new("hi".into(), vec![0.9; 1024]),
        ];
        let pairs = collect_ground_truth(&embs);
        assert_eq!(pairs.len(), 3); // C(3,2) = 3
        // hello↔hi should have highest cosine
        let best = &pairs[0];
        assert!(best.api_cosine > 0.9);
    }

    #[test]
    fn calibration_texts_count() {
        let texts = calibration_texts();
        assert_eq!(texts.len(), 20);
    }

    #[test]
    fn parse_jina_synthetic() {
        let json = r#"{"data":[{"embedding":[0.1,0.2,0.3],"index":0},{"embedding":[0.4,0.5,0.6],"index":1}]}"#;
        let texts = vec!["hello", "world"];
        let embs = parse_jina_response(json, &texts).unwrap();
        assert_eq!(embs.len(), 2);
        assert_eq!(embs[0].vector.len(), 3);
        assert_eq!(embs[0].text, "hello");
    }

    #[test]
    fn ground_truth_summary_basic() {
        let pairs = vec![
            GroundTruthPair {
                text_a: "a".into(), text_b: "b".into(),
                api_cosine: 0.9, base17_l1: 100, base17_cosine: 0.85,
            },
            GroundTruthPair {
                text_a: "c".into(), text_b: "d".into(),
                api_cosine: 0.1, base17_l1: 5000, base17_cosine: 0.05,
            },
        ];
        let summary = summarize_ground_truth(&pairs);
        assert_eq!(summary.n_pairs, 2);
        assert!((summary.mean_api_cosine - 0.5).abs() < 0.01);
    }
}
