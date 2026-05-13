//! Bridge: lance-graph-osint → thinking engine.
//!
//! Connects the OSINT crawler output (cleaned text from spider/Reader-LM)
//! to the thinking engine's codebook and contrastive learner.
//!
//! Pipeline:
//! ```text
//! spider crawls Google → raw HTML → Reader-LM cleans → markdown text
//!   → Qwen3 tokenizer → token IDs → codebook_index → centroid IDs
//!   → F32ThinkingEngine (softmax T=0.01) → peaks
//!   → ContrastiveLearner updates table from pairwise cosines
//!   → NARS truth tracks confidence per centroid pair
//! ```

use crate::f32_engine::F32ThinkingEngine;
use crate::contrastive_learner::ContrastiveLearner;

/// Result of processing one document through the thinking engine.
#[derive(Clone, Debug)]
pub struct ThoughtResult {
    /// Codebook centroid IDs activated by this document.
    pub centroids: Vec<u16>,
    /// Top-K peaks after thinking (atom_index, energy).
    pub peaks: Vec<(u16, f32)>,
    /// Shannon entropy of the energy distribution.
    pub entropy: f32,
    /// Number of thinking cycles used.
    pub cycles: u16,
}

/// OSINT → Thinking Engine bridge.
pub struct OsintThinkingBridge {
    /// Token ID → centroid mapping (u16 per token, from CLAM codebook).
    codebook_index: Vec<u16>,
    /// Number of centroids (256 or 4096).
    n_centroids: usize,
    /// F32 cosine distance table.
    table: Vec<f32>,
}

impl OsintThinkingBridge {
    /// Create from precomputed codebook files.
    ///
    /// `codebook_index`: path to codebook_index.u16 (token → centroid mapping)
    /// `cosine_table`: path to cosine_matrix_NxN.f32 (pairwise centroid cosines)
    pub fn from_files(codebook_index_path: &str, cosine_table_path: &str) -> Result<Self, String> {
        let idx_data = std::fs::read(codebook_index_path)
            .map_err(|e| format!("read codebook index: {e}"))?;
        let codebook_index: Vec<u16> = idx_data.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();

        let table_data = std::fs::read(cosine_table_path)
            .map_err(|e| format!("read cosine table: {e}"))?;
        let table: Vec<f32> = table_data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let n = (table.len() as f64).sqrt() as usize;
        assert_eq!(n * n, table.len(), "table not square");

        Ok(Self { codebook_index, n_centroids: n, table })
    }

    /// Map token IDs to codebook centroid IDs.
    /// Deduplicates while preserving order.
    pub fn tokens_to_centroids(&self, token_ids: &[u32]) -> Vec<u16> {
        let mut seen = std::collections::HashSet::new();
        let mut centroids = Vec::new();
        for &tid in token_ids {
            if (tid as usize) < self.codebook_index.len() {
                let c = self.codebook_index[tid as usize];
                if seen.insert(c) {
                    centroids.push(c);
                }
            }
        }
        centroids
    }

    /// Think about a document: tokenize → centroids → softmax thinking.
    pub fn think(&self, token_ids: &[u32], temperature: f32) -> ThoughtResult {
        let centroids = self.tokens_to_centroids(token_ids);

        let mut engine = F32ThinkingEngine::new(self.table.clone());
        engine.perturb(&centroids);
        engine.think_with_temperature(10, temperature);

        let peaks = engine.top_k(5);
        let entropy = engine.entropy();

        ThoughtResult {
            centroids,
            peaks,
            entropy,
            cycles: engine.cycles,
        }
    }

    /// Compute thinking-based similarity between two documents.
    pub fn similarity(&self, tokens_a: &[u32], tokens_b: &[u32], temperature: f32) -> f32 {
        let result_a = self.think(tokens_a, temperature);
        let result_b = self.think(tokens_b, temperature);

        // Cosine between energy distributions
        let e_a = {
            let mut engine = F32ThinkingEngine::new(self.table.clone());
            engine.perturb(&result_a.centroids);
            engine.think_with_temperature(10, temperature);
            engine.energy().to_vec()
        };
        let e_b = {
            let mut engine = F32ThinkingEngine::new(self.table.clone());
            engine.perturb(&result_b.centroids);
            engine.think_with_temperature(10, temperature);
            engine.energy().to_vec()
        };

        let dot: f32 = e_a.iter().zip(&e_b).map(|(a, b)| a * b).sum();
        let norm_a: f32 = e_a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = e_b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 1e-10 && norm_b > 1e-10 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Create a contrastive learner from this bridge's table.
    pub fn learner(&self, alpha: f32) -> ContrastiveLearner {
        ContrastiveLearner::new(self.table.clone(), alpha)
    }

    /// Get the number of centroids.
    pub fn n_centroids(&self) -> usize {
        self.n_centroids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_to_centroids_deduplicates() {
        let bridge = OsintThinkingBridge {
            codebook_index: vec![5, 3, 5, 7, 3, 1],
            n_centroids: 256,
            table: vec![1.0; 256 * 256],
        };
        let cents = bridge.tokens_to_centroids(&[0, 1, 2, 3, 4, 5]);
        // 5, 3, 7, 1 (deduplicated, order preserved)
        assert_eq!(cents, vec![5, 3, 7, 1]);
    }
}
