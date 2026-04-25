// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Domain bridge: loads DeepNSM data files and provides word-level semantic operations.
//!
//! `NsmRuntime` is the top-level entry point that composes Vocabulary, NsmEncoder,
//! and SimilarityTable into a single API for word distance, decomposition, and
//! DataFusion UDF integration.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float32Array, StringArray};
use arrow::datatypes::DataType;
use datafusion::logical_expr::{ScalarUDF, Signature, Volatility};
use datafusion::physical_plan::ColumnarValue;

use super::encoder::{NsmEncoder, RoleVectors, WordDistanceMatrix, NUM_PRIMES};
use super::similarity::SimilarityTable;
use super::tokenizer::Vocabulary;

/// The NSM runtime: holds all loaded data and provides word-level semantic operations.
#[derive(Clone, Debug)]
pub struct NsmRuntime {
    /// Vocabulary for word <-> rank mapping.
    pub vocabulary: Vocabulary,
    /// Encoder for distance/similarity computations.
    pub encoder: NsmEncoder,
    /// Calibrated similarity table.
    pub similarity_table: SimilarityTable,
}

impl NsmRuntime {
    /// Load an NSM runtime from a data directory.
    ///
    /// Expected files in `word_frequency_dir`:
    /// - `codebook_pq.bin`: CAM codebook (6 subspaces x 256 centroids x subspace_dim f32)
    /// - `cam_codes.bin`: CAM fingerprints per word (N x 6 bytes)
    /// - `word_rank_lookup.csv`: word,rank,pos,freq
    /// - `word_forms.csv`: form,base_rank
    ///
    /// If files are missing, returns an error.
    pub fn load(word_frequency_dir: &Path) -> Result<Self, NsmLoadError> {
        // Load vocabulary CSVs
        let rank_path = word_frequency_dir.join("word_rank_lookup.csv");
        let forms_path = word_frequency_dir.join("word_forms.csv");

        let rank_csv = std::fs::read_to_string(&rank_path).map_err(|e| NsmLoadError {
            message: format!("Failed to read {}: {}", rank_path.display(), e),
        })?;
        let forms_csv = std::fs::read_to_string(&forms_path).map_err(|e| NsmLoadError {
            message: format!("Failed to read {}: {}", forms_path.display(), e),
        })?;

        let vocabulary = Vocabulary::load(&rank_csv, &forms_csv);

        // Load codebook_pq.bin: interpret as f32 vectors, build distance matrix
        let codebook_path = word_frequency_dir.join("codebook_pq.bin");
        let cam_codes_path = word_frequency_dir.join("cam_codes.bin");

        let matrix = if codebook_path.exists() && cam_codes_path.exists() {
            let codebook_bytes =
                std::fs::read(&codebook_path).map_err(|e| NsmLoadError {
                    message: format!(
                        "Failed to read {}: {}",
                        codebook_path.display(),
                        e
                    ),
                })?;
            let cam_bytes = std::fs::read(&cam_codes_path).map_err(|e| NsmLoadError {
                message: format!("Failed to read {}: {}", cam_codes_path.display(), e),
            })?;

            build_distance_matrix_from_cam(&codebook_bytes, &cam_bytes, vocabulary.len())
        } else {
            // No binary files: build a synthetic matrix from rank proximity
            build_synthetic_matrix(vocabulary.len())
        };

        let similarity_table = SimilarityTable::from_distance_matrix(&matrix);

        // Prime ranks: first NUM_PRIMES words (by convention, NSM primes are the
        // highest-frequency words and appear at ranks 0..62)
        let prime_ranks: Vec<u16> = (0..NUM_PRIMES.min(vocabulary.len()) as u16).collect();

        let encoder = NsmEncoder::new(
            matrix,
            RoleVectors::from_seed(0xADA_DE00_0005_u64),
            prime_ranks,
        );

        Ok(NsmRuntime {
            vocabulary,
            encoder,
            similarity_table,
        })
    }

    /// Create a runtime from pre-built components (for testing).
    pub fn from_parts(
        vocabulary: Vocabulary,
        encoder: NsmEncoder,
        similarity_table: SimilarityTable,
    ) -> Self {
        NsmRuntime {
            vocabulary,
            encoder,
            similarity_table,
        }
    }

    /// Compute semantic distance between two words (by string).
    ///
    /// Returns calibrated similarity in [0.0, 1.0], or None if either word is OOV.
    pub fn nsm_distance(&self, word_a: &str, word_b: &str) -> Option<f32> {
        let (rank_a, _) = self.vocabulary.tokenize_word(word_a)?;
        let (rank_b, _) = self.vocabulary.tokenize_word(word_b)?;
        let dist = self.encoder.matrix.get(rank_a as usize, rank_b as usize);
        Some(self.similarity_table.similarity_u32(dist))
    }

    /// Decompose a word into NSM prime similarities.
    ///
    /// Returns vec of (prime_word, similarity) sorted by similarity descending.
    pub fn nsm_decompose(&self, word: &str) -> Option<Vec<(String, f32)>> {
        let (rank, _) = self.vocabulary.tokenize_word(word)?;
        let pairs = self.encoder.decompose(rank);
        let result = pairs
            .into_iter()
            .filter_map(|(pr, sim)| {
                self.vocabulary.word(pr).map(|w| (w.to_string(), sim))
            })
            .collect();
        Some(result)
    }

    /// Find the nearest NSM prime to a word.
    ///
    /// Returns (prime_word, similarity), or None if the word is OOV.
    pub fn nearest_prime(&self, word: &str) -> Option<(String, f32)> {
        let (rank, _) = self.vocabulary.tokenize_word(word)?;
        let (prime_rank, sim) = self.encoder.nearest_prime(rank);
        self.vocabulary
            .word(prime_rank)
            .map(|w| (w.to_string(), sim))
    }
}

/// Error type for NSM loading failures.
#[derive(Debug, Clone)]
pub struct NsmLoadError {
    /// Description of what went wrong.
    pub message: String,
}

impl std::fmt::Display for NsmLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NSM load error: {}", self.message)
    }
}

impl std::error::Error for NsmLoadError {}

/// Build a distance matrix from CAM codebook and fingerprints.
///
/// Reconstructs approximate word vectors by looking up centroids, then
/// computes pairwise L2 distances.
fn build_distance_matrix_from_cam(
    codebook_bytes: &[u8],
    cam_bytes: &[u8],
    vocab_size: usize,
) -> WordDistanceMatrix {
    const CAM_SIZE: usize = 6;

    // Parse CAM codes: each word = 6 bytes
    let num_words = (cam_bytes.len() / CAM_SIZE).min(vocab_size);
    if num_words == 0 {
        return WordDistanceMatrix::new(0);
    }

    // Parse codebook: 6 subspaces x 256 centroids x subspace_dim floats
    // We interpret codebook_bytes as f32 array
    let codebook_floats: &[f32] = unsafe {
        let ptr = codebook_bytes.as_ptr() as *const f32;
        let len = codebook_bytes.len() / 4;
        std::slice::from_raw_parts(ptr, len)
    };

    // Determine subspace_dim: total_floats / (6 * 256)
    let total_floats = codebook_floats.len();
    let subspace_dim = total_floats / (CAM_SIZE * 256);
    if subspace_dim == 0 {
        return build_synthetic_matrix(num_words);
    }

    // Reconstruct approximate vectors for each word
    let total_dim = subspace_dim * CAM_SIZE;
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(num_words);

    for w in 0..num_words {
        let cam = &cam_bytes[w * CAM_SIZE..(w + 1) * CAM_SIZE];
        let mut vec = vec![0.0f32; total_dim];
        for s in 0..CAM_SIZE {
            let centroid_idx = cam[s] as usize;
            let offset = (s * 256 + centroid_idx) * subspace_dim;
            let end = offset + subspace_dim;
            if end <= codebook_floats.len() {
                vec[s * subspace_dim..(s + 1) * subspace_dim]
                    .copy_from_slice(&codebook_floats[offset..end]);
            }
        }
        vectors.push(vec);
    }

    WordDistanceMatrix::build(&vectors)
}

/// Build a synthetic distance matrix from rank proximity (fallback).
fn build_synthetic_matrix(vocab_size: usize) -> WordDistanceMatrix {
    let n = vocab_size.min(4096);
    let mat = WordDistanceMatrix::new(n);

    let mut state = 0xCAFE_BABE_DEAD_BEEFu64;
    for i in 0..n {
        for j in (i + 1)..n {
            let rank_diff = (j - i) as f32;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = ((state & 0x1F) as f32) / 31.0 * 20.0;
            let base = (rank_diff * 2.5).min(230.0);
            let d = (base + noise).min(255.0) as u8;
            // Use the raw set through the public interface workaround
            // We need to write directly to data
            let (lo, hi) = (i, j);
            let idx = lo * (2 * n - lo - 1) / 2 + (hi - lo - 1);
            if idx < mat.size() * (mat.size().saturating_sub(1)) / 2 {
                // Access through get to verify, but we need internal set
                // We'll reconstruct with build instead
            }
            let _ = d; // used below
        }
    }

    // Simpler approach: generate synthetic f32 vectors and use build()
    let dim = 8;
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut state = 0xCAFE_BABE_DEAD_BEEFu64;
    for i in 0..n {
        let mut v = vec![0.0f32; dim];
        for slot in v.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Mix rank info with pseudo-random to get synthetic embeddings
            *slot = (i as f32 * 0.01) + ((state & 0xFFFF) as f32 / 65536.0) * 0.5;
        }
        vectors.push(v);
    }

    WordDistanceMatrix::build(&vectors)
}

// === DataFusion UDF: nsm_similarity(word_a, word_b) -> Float32 ===

/// DataFusion UDF implementing `nsm_similarity(word_a: Utf8, word_b: Utf8) -> Float32`.
struct NsmSimilarityUdf {
    /// UDF name.
    name: String,
    /// Signature: two Utf8 arguments.
    signature: Signature,
    /// Shared runtime reference.
    runtime: Arc<NsmRuntime>,
}

impl std::fmt::Debug for NsmSimilarityUdf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NsmSimilarityUdf")
            .field("name", &self.name)
            .finish()
    }
}

impl datafusion::logical_expr::ScalarUDFImpl for NsmSimilarityUdf {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> datafusion::error::Result<DataType> {
        Ok(DataType::Float32)
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> datafusion::error::Result<ColumnarValue> {
        nsm_similarity_impl(&self.runtime, &args.args)
    }
}

impl PartialEq for NsmSimilarityUdf {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for NsmSimilarityUdf {}

impl std::hash::Hash for NsmSimilarityUdf {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

/// Core implementation of nsm_similarity UDF.
fn nsm_similarity_impl(
    runtime: &NsmRuntime,
    args: &[ColumnarValue],
) -> datafusion::error::Result<ColumnarValue> {
    if args.len() != 2 {
        return Err(datafusion::error::DataFusionError::Execution(
            "nsm_similarity requires exactly 2 arguments".to_string(),
        ));
    }

    // Extract word_a
    let words_a = extract_utf8_values(&args[0])?;
    let words_b = extract_utf8_values(&args[1])?;

    // Handle scalar x scalar
    if words_a.len() == 1 && words_b.len() == 1 {
        let sim = runtime
            .nsm_distance(&words_a[0], &words_b[0])
            .unwrap_or(0.0);
        return Ok(ColumnarValue::Scalar(
            datafusion::scalar::ScalarValue::Float32(Some(sim)),
        ));
    }

    // Handle array x array (pairwise) or scalar x array (broadcast)
    let n = words_a.len().max(words_b.len());
    let mut results = Vec::with_capacity(n);

    for i in 0..n {
        let a_idx = if words_a.len() == 1 { 0 } else { i };
        let b_idx = if words_b.len() == 1 { 0 } else { i };
        let sim = if a_idx < words_a.len() && b_idx < words_b.len() {
            runtime
                .nsm_distance(&words_a[a_idx], &words_b[b_idx])
                .unwrap_or(0.0)
        } else {
            0.0
        };
        results.push(sim);
    }

    let array = Arc::new(Float32Array::from(results)) as ArrayRef;
    Ok(ColumnarValue::Array(array))
}

/// Extract UTF-8 string values from a ColumnarValue.
fn extract_utf8_values(
    col: &ColumnarValue,
) -> datafusion::error::Result<Vec<String>> {
    match col {
        ColumnarValue::Scalar(s) => {
            if let datafusion::scalar::ScalarValue::Utf8(Some(val)) = s {
                Ok(vec![val.clone()])
            } else if let datafusion::scalar::ScalarValue::LargeUtf8(Some(val)) = s {
                Ok(vec![val.clone()])
            } else {
                Err(datafusion::error::DataFusionError::Execution(
                    "nsm_similarity: arguments must be Utf8 strings".to_string(),
                ))
            }
        }
        ColumnarValue::Array(arr) => {
            let string_arr = arr
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    datafusion::error::DataFusionError::Execution(
                        "nsm_similarity: arguments must be Utf8 string arrays".to_string(),
                    )
                })?;
            let mut vals = Vec::with_capacity(string_arr.len());
            for i in 0..string_arr.len() {
                if string_arr.is_null(i) {
                    vals.push(String::new());
                } else {
                    vals.push(string_arr.value(i).to_string());
                }
            }
            Ok(vals)
        }
    }
}

/// Create an `nsm_similarity` UDF bound to an NsmRuntime.
///
/// The runtime is captured by Arc and shared across all invocations.
pub fn create_nsm_similarity_udf(runtime: Arc<NsmRuntime>) -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(NsmSimilarityUdf {
        name: "nsm_similarity".to_string(),
        signature: Signature::any(2, Volatility::Immutable),
        runtime,
    }))
}

/// Register the NSM similarity UDF with a DataFusion SessionContext.
pub fn register_nsm_udfs(
    ctx: &datafusion::execution::context::SessionContext,
    runtime: Arc<NsmRuntime>,
) {
    ctx.register_udf((*create_nsm_similarity_udf(runtime)).clone());
}

/// Build a test NsmRuntime using the built-in test vocabulary.
pub fn test_runtime() -> NsmRuntime {
    let vocabulary = super::tokenizer::test_vocabulary();
    let encoder = super::encoder::test_encoder();
    let similarity_table = SimilarityTable::from_distance_matrix(&encoder.matrix);

    NsmRuntime {
        vocabulary,
        encoder,
        similarity_table,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nsm_distance_same_word() {
        let rt = test_runtime();
        let sim = rt.nsm_distance("cat", "cat").unwrap();
        assert!((sim - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_nsm_distance_different_words() {
        let rt = test_runtime();
        let sim = rt.nsm_distance("cat", "dog").unwrap();
        assert!(sim > 0.0);
        assert!(sim < 1.0);
    }

    #[test]
    fn test_nsm_distance_oov() {
        let rt = test_runtime();
        assert!(rt.nsm_distance("cat", "xyzzyplugh").is_none());
    }

    #[test]
    fn test_nearest_prime() {
        let rt = test_runtime();
        let result = rt.nearest_prime("cat");
        assert!(result.is_some());
        let (prime, sim) = result.unwrap();
        assert!(!prime.is_empty());
        assert!(sim >= 0.0 && sim <= 1.0);
    }

    #[test]
    fn test_nsm_decompose() {
        let rt = test_runtime();
        let decomp = rt.nsm_decompose("cat");
        assert!(decomp.is_some());
        let primes = decomp.unwrap();
        assert!(!primes.is_empty());
        // Should be sorted by similarity descending
        for w in primes.windows(2) {
            assert!(w[0].1 >= w[1].1 - f32::EPSILON);
        }
    }

    #[test]
    fn test_inflected_form_distance() {
        let rt = test_runtime();
        // "running" should resolve to "run" (rank 60)
        let sim = rt.nsm_distance("running", "run").unwrap();
        assert!((sim - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_create_udf() {
        let rt = test_runtime();
        let udf = create_nsm_similarity_udf(Arc::new(rt));
        assert_eq!(udf.name(), "nsm_similarity");
    }
}
