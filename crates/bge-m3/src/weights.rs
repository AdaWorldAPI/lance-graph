//! BGE-M3 weight loading from bgz7.

use ndarray::hpc::bgz17_bridge::Base17;

pub const DEFAULT_BGZ7_PATH: &str = "/tmp/bge_m3_f16.bgz7";

// XLM-RoBERTa architecture constants (BGE-M3)
pub const VOCAB_SIZE: usize = 250002;
pub const HIDDEN_DIM: usize = 1024;
pub const NUM_LAYERS: usize = 24;
pub const NUM_HEADS: usize = 16;
pub const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 64
pub const MLP_DIM: usize = 4096;
pub const MAX_SEQ_LEN: usize = 8192;

pub struct BgeM3Weights {
    pub tensors: Vec<(String, Vec<Base17>)>,
    pub total_rows: usize,
}

impl BgeM3Weights {
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

    pub fn load_default() -> Result<Self, String> { Self::load(DEFAULT_BGZ7_PATH) }

    pub fn embedding_rows(&self) -> Vec<&Base17> {
        self.tensors.iter()
            .filter(|(n, _)| n.contains("embed") || n.contains("word"))
            .flat_map(|(_, r)| r.iter()).collect()
    }

    pub fn attention_rows(&self) -> Vec<&Base17> {
        self.tensors.iter()
            .filter(|(n, _)| n.contains("attn") || n.contains("self"))
            .flat_map(|(_, r)| r.iter()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() { assert_eq!(HEAD_DIM, 64); }

    #[test]
    #[ignore = "requires: /tmp/bge_m3_f16.bgz7"]
    fn test_load() {
        let w = BgeM3Weights::load_default().unwrap();
        eprintln!("BGE-M3: {} tensors, {} rows", w.tensors.len(), w.total_rows);
    }
}
