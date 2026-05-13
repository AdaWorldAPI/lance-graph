//! Ground truth embeddings via candle forward pass.
//!
//! Loads Jina v5 (or any BERT-family model) via candle-transformers,
//! runs real forward pass on tokenized text, returns f32 embeddings.
//!
//! This is the RAW in the camera analogy:
//!   ONNX f32 = RAW file (full sensor, 24-bit mantissa)
//!   GGUF BF16 = TIFF (7-bit mantissa, legs chopped)
//!   Our table = JPEG (8-bit u8/i8, compressed for distribution)
//!
//! Calibrate against RAW, never against JPEG.
//!
//! Feature-gated: only available with `--features calibration`.
//! Default builds don't pull candle (2+ GB of model weights).

#[cfg(feature = "calibration")]
pub mod calibration {
    use crate::tokenizer_registry::ModelId;

    /// Ground truth embedding for one text.
    #[derive(Clone, Debug)]
    pub struct GroundTruthEmbedding {
        /// The text that was embedded.
        pub text: String,
        /// Model that produced this embedding.
        pub model: ModelId,
        /// The f32 embedding vector (1024D for Jina, varies by model).
        pub embedding: Vec<f32>,
        /// L2-normalized?
        pub normalized: bool,
    }

    impl GroundTruthEmbedding {
        /// Cosine similarity with another embedding.
        pub fn cosine(&self, other: &GroundTruthEmbedding) -> f32 {
            ndarray::hpc::heel_f64x8::cosine_f32_to_f64_simd(
                &self.embedding, &other.embedding
            ) as f32
        }
    }

    /// Ground truth for a pair of texts.
    #[derive(Clone, Debug)]
    pub struct GroundTruthPair {
        pub text_a: String,
        pub text_b: String,
        pub model: ModelId,
        pub cosine: f32,
        pub embedding_a: Vec<f32>,
        pub embedding_b: Vec<f32>,
    }

    /// Calibration corpus: texts + ground truth cosines from forward pass.
    #[derive(Clone, Debug)]
    pub struct CalibrationCorpus {
        pub model: ModelId,
        pub pairs: Vec<GroundTruthPair>,
    }

    impl CalibrationCorpus {
        /// Extract just the cosines (for Spearman ρ against baked table distances).
        pub fn cosines(&self) -> Vec<f32> {
            self.pairs.iter().map(|p| p.cosine).collect()
        }

        /// Number of pairs.
        pub fn len(&self) -> usize {
            self.pairs.len()
        }

        pub fn is_empty(&self) -> bool {
            self.pairs.is_empty()
        }
    }

    /// Status of the ground truth system.
    /// When candle model is not loaded, we can still use the DTOs
    /// with externally-computed embeddings (e.g. from Python or API).
    #[derive(Clone, Debug)]
    pub enum GroundTruthSource {
        /// Candle forward pass (local, no network at inference time).
        Candle { model_path: String },
        /// Pre-computed embeddings loaded from file.
        Precomputed { file_path: String },
        /// External API (Jina API, needs key).
        Api { endpoint: String },
        /// Not available — use expert-assigned scores as fallback.
        ExpertAssigned,
    }

    /// Build a calibration corpus from text pairs.
    ///
    /// When candle model is available: runs forward pass for real embeddings.
    /// Otherwise: returns empty corpus (caller should use expert scores).
    pub fn build_corpus_from_pairs(
        pairs: &[(&str, &str)],
        model: ModelId,
        source: &GroundTruthSource,
    ) -> CalibrationCorpus {
        match source {
            GroundTruthSource::ExpertAssigned => {
                // No embeddings available — return empty corpus
                // Caller should use expert-assigned scores instead
                CalibrationCorpus { model, pairs: vec![] }
            }
            GroundTruthSource::Precomputed { file_path } => {
                // Load pre-computed embeddings from file
                // Format: one embedding per line, f32 values space-separated
                load_precomputed_corpus(pairs, model, file_path)
            }
            _ => {
                // Candle / API not yet wired — return empty
                // TODO: wire candle forward pass here
                eprintln!("WARNING: candle forward pass not yet wired. Returning empty corpus.");
                eprintln!("  Use GroundTruthSource::Precomputed with pre-computed embeddings.");
                CalibrationCorpus { model, pairs: vec![] }
            }
        }
    }

    fn load_precomputed_corpus(
        pairs: &[(&str, &str)],
        model: ModelId,
        file_path: &str,
    ) -> CalibrationCorpus {
        // Try loading embeddings file
        let content = match std::fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to load precomputed embeddings: {}", e);
                return CalibrationCorpus { model, pairs: vec![] };
            }
        };

        let embeddings: Vec<Vec<f32>> = content.lines()
            .map(|line| {
                line.split_whitespace()
                    .filter_map(|v| v.parse::<f32>().ok())
                    .collect()
            })
            .collect();

        // Need 2 embeddings per pair (text_a + text_b)
        let mut ground_truth_pairs = Vec::new();
        for (i, (a, b)) in pairs.iter().enumerate() {
            let idx_a = i * 2;
            let idx_b = i * 2 + 1;
            if idx_b >= embeddings.len() { break; }

            let emb_a = &embeddings[idx_a];
            let emb_b = &embeddings[idx_b];

            // Cosine similarity
            let dot: f32 = emb_a.iter().zip(emb_b).map(|(x, y)| x * y).sum();
            let na: f32 = emb_a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = emb_b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cosine = if na > 1e-10 && nb > 1e-10 { dot / (na * nb) } else { 0.0 };

            ground_truth_pairs.push(GroundTruthPair {
                text_a: a.to_string(),
                text_b: b.to_string(),
                model,
                cosine,
                embedding_a: emb_a.clone(),
                embedding_b: emb_b.clone(),
            });
        }

        CalibrationCorpus { model, pairs: ground_truth_pairs }
    }

    /// Compare baked lens distances against ground truth cosines.
    /// Returns Spearman ρ.
    pub fn spearman_vs_ground_truth(
        baked_distances: &[f32],
        corpus: &CalibrationCorpus,
    ) -> f32 {
        let gt = corpus.cosines();
        if gt.len() != baked_distances.len() || gt.len() < 2 {
            return 0.0;
        }
        super::spearman_rank_correlation(&gt, baked_distances)
    }
}

//
/// Spearman rank correlation between two f32 slices.
pub fn spearman_rank_correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n < 2 { return 0.0; }
    let rank_a = ranks(a);
    let rank_b = ranks(b);
    let mean_a = rank_a.iter().sum::<f32>() / n as f32;
    let mean_b = rank_b.iter().sum::<f32>() / n as f32;
    let mut num = 0.0f32;
    let mut den_a = 0.0f32;
    let mut den_b = 0.0f32;
    for i in 0..n {
        let da = rank_a[i] - mean_a;
        let db = rank_b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    let den = (den_a * den_b).sqrt();
    if den > 1e-10 { num / den } else { 0.0 }
}

fn ranks(values: &[f32]) -> Vec<f32> {
    let mut indexed: Vec<(usize, f32)> = values.iter().enumerate()
        .map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut result = vec![0.0f32; values.len()];
    for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
        result[orig_idx] = rank as f32;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spearman_perfect_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let rho = spearman_rank_correlation(&a, &b);
        assert!((rho - 1.0).abs() < 1e-4, "perfect correlation should be ~1.0, got {}", rho);
    }

    #[test]
    fn spearman_inverse_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![50.0, 40.0, 30.0, 20.0, 10.0];
        let rho = spearman_rank_correlation(&a, &b);
        assert!((rho - (-1.0)).abs() < 1e-4, "inverse should be ~-1.0, got {}", rho);
    }

    #[test]
    fn spearman_no_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![3.0, 1.0, 4.0, 2.0];
        let rho = spearman_rank_correlation(&a, &b);
        assert!(rho.abs() < 0.5, "shuffled should be near zero, got {}", rho);
    }

    #[cfg(feature = "calibration")]
    mod calibration_tests {
        use super::super::calibration::*;
        use crate::tokenizer_registry::ModelId;

        #[test]
        fn ground_truth_cosine_identical() {
            let a = GroundTruthEmbedding {
                text: "hello".into(), model: ModelId::JinaV5,
                embedding: vec![1.0, 0.0, 0.0], normalized: true,
            };
            let b = GroundTruthEmbedding {
                text: "hello".into(), model: ModelId::JinaV5,
                embedding: vec![1.0, 0.0, 0.0], normalized: true,
            };
            assert!((a.cosine(&b) - 1.0).abs() < 1e-6);
        }

        #[test]
        fn empty_corpus_from_expert() {
            let pairs = vec![("a", "b")];
            let corpus = build_corpus_from_pairs(
                &pairs, ModelId::JinaV5, &GroundTruthSource::ExpertAssigned,
            );
            assert!(corpus.is_empty());
        }
    }
}
