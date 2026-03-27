//! DeepNSM inference pipeline.
//!
//! The complete semantic processing chain that replaces transformer inference:
//!
//! ```text
//! Raw text
//!   → tokenize (12-bit ranks)
//!     → parse (SPO triples)
//!       → encode (VSA vectors)
//!         → similarity (calibrated f32)
//!           → cognitive verbs (ThinkingGraph)
//! ```
//!
//! Total: < 10μs per sentence. 0 learned parameters. Bit-reproducible.

use std::path::Path;

use crate::codebook::{CamCodes, Codebook};
use crate::context::ContextWindow;
use crate::encoder::{self, RoleVectors, VsaVec};
use crate::parser::{self, SentenceStructure};
use crate::similarity::SimilarityTable;
use crate::spo::{SpoTriple, WordDistanceMatrix};
use crate::vocabulary::Vocabulary;

/// Complete DeepNSM engine. Holds all precomputed data structures.
///
/// Memory budget:
/// - Vocabulary: ~400KB (HashMap + reverse table)
/// - Distance matrix: 16MB (4096² u8)
/// - Similarity table: 1KB (256 × f32)
/// - Context window: ~1KB (11 × 64 bytes)
/// - Role vectors: 384 bytes (6 × 64 bytes)
/// - Codebook: 96KB (optional, only for matrix construction)
///
/// Total: ~16.5 MB (dominated by distance matrix)
pub struct DeepNsmEngine {
    /// Vocabulary for tokenization.
    pub vocab: Vocabulary,
    /// Precomputed word-to-word distance matrix.
    pub distance_matrix: WordDistanceMatrix,
    /// Calibrated similarity lookup.
    pub similarity_table: SimilarityTable,
    /// Fixed role vectors for VSA encoding.
    pub roles: RoleVectors,
    /// Streaming context window.
    pub context: ContextWindow,
}

/// Result of processing a single sentence.
#[derive(Debug)]
pub struct ProcessedSentence {
    /// The parsed semantic structure.
    pub structure: SentenceStructure,
    /// VSA vector for each triple.
    pub triple_vecs: Vec<VsaVec>,
    /// Combined sentence vector (bundle of all triples).
    pub sentence_vec: VsaVec,
    /// Number of tokens (including OOV).
    pub token_count: usize,
    /// Number of in-vocabulary tokens.
    pub known_token_count: usize,
}

/// Similarity result between two sentences.
#[derive(Debug)]
pub struct SentenceSimilarity {
    /// Overall similarity (0.0 - 1.0).
    pub overall: f32,
    /// VSA-space similarity (Hamming-based).
    pub vsa_similarity: f32,
    /// Per-triple similarities (if both have same number of triples).
    pub triple_similarities: Vec<f32>,
}

impl DeepNsmEngine {
    /// Load a complete engine from a data directory.
    ///
    /// The directory should contain:
    /// - `word_rank_lookup.csv`
    /// - `word_forms.csv`
    /// - `cam_codes.bin`
    /// - `codebook_pq.bin`
    ///
    /// The distance matrix is computed at load time from CAM-PQ codes.
    pub fn load(data_dir: &Path) -> Result<Self, String> {
        // 1. Load vocabulary
        let vocab = Vocabulary::load(data_dir)?;
        eprintln!(
            "[deepnsm] Vocabulary loaded: {} words, {} forms",
            vocab.len(),
            vocab.forms_count()
        );

        // 2. Load CAM-PQ codebook and codes
        let codebook = Codebook::load_binary(&data_dir.join("codebook_pq.bin"))?;
        let cam_codes = CamCodes::load(&data_dir.join("cam_codes.bin"))?;
        eprintln!(
            "[deepnsm] Codebook loaded: {} centroids, {} CAM codes",
            codebook.len(),
            cam_codes.len()
        );

        // 3. Build distance matrix from CAM codes
        let cam_array: Vec<[u8; 6]> = (0..cam_codes.len().min(crate::vocabulary::VOCAB_SIZE))
            .filter_map(|i| cam_codes.get(i))
            .collect();

        eprintln!(
            "[deepnsm] Building {}×{} distance matrix...",
            cam_array.len(),
            cam_array.len()
        );
        let distance_matrix = WordDistanceMatrix::build_from_cam(&cam_array, &codebook.centroids);
        eprintln!(
            "[deepnsm] Distance matrix built: {} bytes",
            distance_matrix.byte_size()
        );

        // 4. Build similarity table from exact distribution
        let similarity_table = SimilarityTable::from_distance_matrix(&distance_matrix);
        eprintln!("[deepnsm] Similarity table calibrated: {:?}", similarity_table);

        // 5. Create role vectors and context window
        let roles = RoleVectors::new();
        let context = ContextWindow::default_window();

        Ok(DeepNsmEngine {
            vocab,
            distance_matrix,
            similarity_table,
            roles,
            context,
        })
    }

    /// Load with a precomputed distance matrix (skip CAM-PQ computation).
    pub fn load_with_matrix(
        data_dir: &Path,
        matrix_data: Vec<u8>,
    ) -> Result<Self, String> {
        let vocab = Vocabulary::load(data_dir)?;
        let distance_matrix = WordDistanceMatrix::from_flat(matrix_data);
        let similarity_table = SimilarityTable::from_distance_matrix(&distance_matrix);

        Ok(DeepNsmEngine {
            vocab,
            distance_matrix,
            similarity_table,
            roles: RoleVectors::new(),
            context: ContextWindow::default_window(),
        })
    }

    /// Process a single sentence through the full pipeline.
    ///
    /// ```text
    /// text → tokenize → parse → encode → (push to context)
    /// ```
    pub fn process_sentence(&mut self, text: &str) -> ProcessedSentence {
        // 1. Tokenize
        let tokens = self.vocab.tokenize(text);
        let known_count = tokens.iter().filter(|t| t.is_known()).count();

        // 2. Parse into SPO structure
        let structure = parser::parse(&tokens);

        // 3. Encode each triple as VSA vector
        let mut triple_vecs = Vec::with_capacity(structure.triples.len());
        for (i, triple) in structure.triples.iter().enumerate() {
            let is_negated = structure.negations.contains(&i);
            let vec = if is_negated {
                encoder::encode_triple_negated(
                    triple.subject(),
                    triple.predicate(),
                    if triple.has_object() { Some(triple.object()) } else { None },
                    &self.roles,
                )
            } else {
                encoder::encode_triple(
                    triple.subject(),
                    triple.predicate(),
                    if triple.has_object() { Some(triple.object()) } else { None },
                    &self.roles,
                )
            };
            triple_vecs.push(vec);
        }

        // 4. Bundle all triple vecs into sentence vec
        let sentence_vec = if triple_vecs.is_empty() {
            VsaVec::ZERO
        } else {
            encoder::bundle(&triple_vecs)
        };

        // 5. Push to context window
        if !structure.is_empty() {
            self.context.push(sentence_vec.clone());
        }

        ProcessedSentence {
            structure,
            triple_vecs,
            sentence_vec,
            token_count: tokens.len(),
            known_token_count: known_count,
        }
    }

    /// Compute similarity between two sentences.
    pub fn sentence_similarity(&self, a: &ProcessedSentence, b: &ProcessedSentence) -> SentenceSimilarity {
        // VSA similarity
        let vsa_sim = a.sentence_vec.similarity(&b.sentence_vec);

        // Per-triple similarities via distance matrix
        let mut triple_sims = Vec::new();
        let mut total_sim = 0.0f32;
        let mut pair_count = 0;

        for ta in &a.structure.triples {
            for tb in &b.structure.triples {
                let d = ta.distance(tb, &self.distance_matrix);
                let sim = self.similarity_table.lookup_averaged(d, 3);
                triple_sims.push(sim);
                total_sim += sim;
                pair_count += 1;
            }
        }

        let avg_triple_sim = if pair_count > 0 {
            total_sim / pair_count as f32
        } else {
            0.0
        };

        // Overall: weighted average of VSA and distributional similarity
        let overall = 0.4 * vsa_sim.max(0.0) + 0.6 * avg_triple_sim;

        SentenceSimilarity {
            overall,
            vsa_similarity: vsa_sim,
            triple_similarities: triple_sims,
        }
    }

    /// Find the most similar word to a given word by vocabulary rank.
    /// Returns (rank, distance) pairs sorted by distance.
    pub fn nearest_words(&self, rank: u16, k: usize) -> Vec<(u16, u8)> {
        let mut distances: Vec<(u16, u8)> = (0..crate::vocabulary::VOCAB_SIZE as u16)
            .filter(|&r| r != rank)
            .map(|r| (r, self.distance_matrix.get(rank, r)))
            .collect();

        distances.sort_by_key(|&(_, d)| d);
        distances.truncate(k);
        distances
    }

    /// Look up a word and find its nearest neighbors.
    pub fn word_neighbors(&self, word: &str, k: usize) -> Option<Vec<(String, u8)>> {
        let rank = self.vocab.rank_of(word)?;
        let neighbors = self.nearest_words(rank, k);
        Some(
            neighbors
                .into_iter()
                .map(|(r, d)| (self.vocab.word(r).to_string(), d))
                .collect(),
        )
    }

    /// Get the current context vector (for external use).
    pub fn context_vector(&mut self) -> Option<&VsaVec> {
        self.context.context()
    }

    /// Clear the context window.
    pub fn reset_context(&mut self) {
        self.context.clear();
    }

    /// Disambiguate a word using current context.
    pub fn disambiguate(&mut self, word: &str) -> Option<VsaVec> {
        let rank = self.vocab.rank_of(word)?;
        Some(self.context.disambiguate(rank))
    }

    /// Get human-readable description of a triple.
    pub fn describe_triple(&self, triple: &SpoTriple) -> String {
        let s = self.vocab.word(triple.subject());
        let p = self.vocab.word(triple.predicate());
        if triple.has_object() {
            let o = self.vocab.word(triple.object());
            format!("{} → {} → {}", s, p, o)
        } else {
            format!("{} → {}", s, p)
        }
    }
}

/// Quick-check stats about the engine.
pub fn engine_stats(engine: &DeepNsmEngine) -> String {
    format!(
        "DeepNSM Engine:\n\
         Vocabulary: {} words\n\
         Forms: {} inflections\n\
         Distance matrix: {} bytes ({:.1} MB)\n\
         Similarity table: {} bytes\n\
         Context window: {}/{} sentences\n\
         VSA dimensions: {} bits",
        engine.vocab.len(),
        engine.vocab.forms_count(),
        engine.distance_matrix.byte_size(),
        engine.distance_matrix.byte_size() as f64 / 1_048_576.0,
        SimilarityTable::BYTE_SIZE,
        engine.context.len(),
        engine.context.capacity(),
        encoder::VSA_BITS,
    )
}

#[cfg(test)]
mod tests {
    // Integration tests require the data files, so they're in tests/ directory.
    // Unit tests for pipeline logic:

    use super::*;

    #[test]
    fn processed_sentence_empty() {
        let result = ProcessedSentence {
            structure: parser::SentenceStructure {
                triples: vec![],
                modifiers: vec![],
                negations: vec![],
                temporals: vec![],
            },
            triple_vecs: vec![],
            sentence_vec: VsaVec::ZERO,
            token_count: 0,
            known_token_count: 0,
        };
        assert_eq!(result.token_count, 0);
    }
}
