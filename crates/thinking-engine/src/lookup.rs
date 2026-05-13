//! # Text-to-Thought Lookup
//!
//! The complete pipeline from text to thought in one struct:
//! ```text
//! text → tokenizer (BPE) → token_ids
//!   → codebook_index (token_id → centroid row)
//!   → perturb(engine) → think(N cycles) → commit()
//! ```
//!
//! No forward pass. No weights. Just lookup + MatVec.

use crate::codebook_index::CodebookIndex;
use crate::dto::BusDto;
use crate::engine::ThinkingEngine;

/// Complete text-to-thought pipeline.
///
/// Owns: tokenizer, codebook index, distance table, engine.
/// One call: `think("some text") → BusDto`.
pub struct TextToThought {
    tokenizer: tokenizers::Tokenizer,
    codebook: CodebookIndex,
    engine: ThinkingEngine,
    /// Table size (N in N×N).
    pub table_size: usize,
    /// Model name.
    pub model: String,
}

/// Stats from a single thought.
pub struct ThoughtResult {
    pub bus: BusDto,
    pub token_count: usize,
    pub unique_atoms: usize,
    pub entropy: f32,
    pub active_count: usize,
    pub think_micros: u64,
}

impl TextToThought {
    /// Load from files on disk.
    ///
    /// - `tokenizer_path`: path to tokenizer.json (e.g. /tmp/bge-m3-tokenizer.json)
    /// - `codebook_path`: path to codebook_index.u16 (e.g. /tmp/codebooks/.../codebook_index.u16)
    /// - `table_path`: path to distance_table_NxN.u8
    /// - `table_size`: N (e.g. 1024 for attn_q, 4096 for ffn_down)
    pub fn load(
        tokenizer_path: &str,
        codebook_path: &str,
        table_path: &str,
        table_size: usize,
        model: &str,
    ) -> Result<Self, String> {
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("tokenizer: {}", e))?;

        let codebook = CodebookIndex::load(
            std::path::Path::new(codebook_path),
            table_size as u16,
            model.into(),
        ).map_err(|e| format!("codebook: {}", e))?;

        let table_data = std::fs::read(table_path)
            .map_err(|e| format!("table: {}", e))?;

        if table_data.len() != table_size * table_size {
            return Err(format!(
                "table size mismatch: file {} bytes, expected {}×{}={}",
                table_data.len(), table_size, table_size, table_size * table_size
            ));
        }

        let engine = ThinkingEngine::new(table_data);

        Ok(Self {
            tokenizer,
            codebook,
            engine,
            table_size,
            model: model.into(),
        })
    }

    /// Load BGE-M3 from default paths.
    pub fn load_bge_m3() -> Result<Self, String> {
        Self::load(
            "/tmp/bge-m3-tokenizer.json",
            "/tmp/codebooks/bge-m3-roles-f16/codebook_index.u16",
            "/tmp/codebooks/bge-m3-roles-f16/attn_q/distance_table_1024x1024.u8",
            1024,
            "bge-m3",
        )
    }

    /// The core pipeline: text → thought.
    ///
    /// ```text
    /// text → tokenize → codebook lookup → perturb → think → commit
    /// ```
    pub fn think(&mut self, text: &str) -> ThoughtResult {
        let start = std::time::Instant::now();

        // Tokenize
        let encoding = self.tokenizer.encode(text, true)
            .expect("tokenization failed");
        let token_ids = encoding.get_ids();

        // Codebook lookup: token_id → centroid row
        let indices = self.codebook.lookup_many(token_ids);

        // Count unique atoms
        let mut unique = indices.clone();
        unique.sort();
        unique.dedup();
        let unique_atoms = unique.len();

        // Perturb + think
        self.engine.reset();
        self.engine.perturb(&indices);
        self.engine.think(10);

        let bus = self.engine.commit();
        let entropy = self.engine.entropy();
        let active = self.engine.active_count(0.001);

        ThoughtResult {
            bus,
            token_count: token_ids.len(),
            unique_atoms,
            entropy,
            active_count: active,
            think_micros: start.elapsed().as_micros() as u64,
        }
    }

    /// Think about two texts and return both results for comparison.
    pub fn compare(&mut self, text_a: &str, text_b: &str) -> (ThoughtResult, ThoughtResult, u8) {
        let a = self.think(text_a);
        let b = self.think(text_b);

        // Distance between dominant atoms in the table
        let idx_a = a.bus.codebook_index as usize;
        let idx_b = b.bus.codebook_index as usize;
        let dist = if idx_a < self.table_size && idx_b < self.table_size {
            // Read from engine's table via the stored table data
            // (engine doesn't expose table, recompute from the index)
            128u8 // placeholder — would need table access
        } else {
            128u8
        };

        (a, b, dist)
    }

    /// Get a reference to the underlying engine (for qualia computation).
    pub fn engine(&self) -> &ThinkingEngine {
        &self.engine
    }

    /// Get the sigma floor the engine is using.
    pub fn floor(&self) -> u8 {
        self.engine.floor
    }

    /// Get the tokenizer vocab size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Get codebook coverage stats.
    pub fn codebook_stats(&self) -> (usize, usize) {
        (self.codebook.len(), self.codebook.unique_centroids())
    }
}

impl std::fmt::Display for ThoughtResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "atom {} (e={:.4}) | {} tokens → {} atoms | H={:.2} | {}μs",
            self.bus.codebook_index,
            self.bus.energy,
            self.token_count,
            self.unique_atoms,
            self.entropy,
            self.think_micros,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thought_result_display() {
        let result = ThoughtResult {
            bus: BusDto {
                codebook_index: 42,
                energy: 0.15,
                top_k: [(42, 0.15), (0, 0.0), (0, 0.0), (0, 0.0),
                         (0, 0.0), (0, 0.0), (0, 0.0), (0, 0.0)],
                cycle_count: 5,
                converged: true,
            },
            token_count: 10,
            unique_atoms: 8,
            entropy: 5.5,
            active_count: 200,
            think_micros: 3500,
        };
        let s = format!("{}", result);
        assert!(s.contains("atom 42"));
        assert!(s.contains("3500μs"));
    }
}
