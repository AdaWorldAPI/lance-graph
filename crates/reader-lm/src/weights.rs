//! Reader LM weight loading from bgz7 and safetensors.

use ndarray::hpc::bgz17_bridge::Base17;

pub const DEFAULT_BGZ7_PATH: &str = "/tmp/reader_lm_1_5b.bgz7";

// Qwen2-1.5B architecture constants
pub const VOCAB_SIZE: usize = 151936;
pub const HIDDEN_DIM: usize = 1536;
pub const NUM_LAYERS: usize = 28;
pub const NUM_HEADS: usize = 12;
pub const NUM_KV_HEADS: usize = 2; // GQA: 12 query heads, 2 KV heads
pub const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 128
pub const MLP_DIM: usize = 8960; // SwiGLU intermediate
pub const MAX_SEQ_LEN: usize = 32768;

/// Reader LM weight index from bgz7.
pub struct ReaderLmWeights {
    pub tensors: Vec<(String, Vec<Base17>)>,
    pub total_rows: usize,
}

impl ReaderLmWeights {
    pub fn load(path: &str) -> Result<Self, String> {
        let compressed = ndarray::hpc::gguf_indexer::read_bgz7_file(path)?;
        let mut tensors = Vec::new();
        let mut total_rows = 0;
        for ct in compressed {
            total_rows += ct.rows.len();
            tensors.push((ct.name, ct.rows));
        }
        Ok(Self { tensors, total_rows })
    }

    pub fn load_default() -> Result<Self, String> {
        Self::load(DEFAULT_BGZ7_PATH)
    }

    pub fn all_rows(&self) -> Vec<&Base17> {
        self.tensors.iter().flat_map(|(_, rows)| rows.iter()).collect()
    }

    /// Get Q-projection rows (subject plane in SPO).
    pub fn q_proj_rows(&self) -> Vec<&Base17> {
        self.rows_matching("q_proj")
    }

    /// Get K-projection rows (predicate plane).
    pub fn k_proj_rows(&self) -> Vec<&Base17> {
        self.rows_matching("k_proj")
    }

    /// Get V-projection rows.
    pub fn v_proj_rows(&self) -> Vec<&Base17> {
        self.rows_matching("v_proj")
    }

    /// Get FFN gate rows (SwiGLU — dominant signal per Qwen diff results).
    pub fn gate_proj_rows(&self) -> Vec<&Base17> {
        self.rows_matching("gate_proj")
    }

    fn rows_matching(&self, pattern: &str) -> Vec<&Base17> {
        self.tensors.iter()
            .filter(|(name, _)| name.contains(pattern))
            .flat_map(|(_, rows)| rows.iter())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_constants() {
        assert_eq!(HEAD_DIM, 128);
        assert_eq!(NUM_HEADS / NUM_KV_HEADS, 6); // GQA ratio
    }

    #[test]
    #[ignore = "requires: /tmp/reader_lm_1_5b.bgz7"]
    fn test_load_weights() {
        let w = ReaderLmWeights::load_default().unwrap();
        assert!(w.total_rows > 0);
        eprintln!("Reader LM: {} tensors, {} rows", w.tensors.len(), w.total_rows);
        for (name, rows) in w.tensors.iter().take(5) {
            eprintln!("  {}: {} rows", name, rows.len());
        }
    }
}
