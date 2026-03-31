//! Qwen2 1.5B forward pass — transcoded for Reader LM inference.
//!
//! Architecture: RoPE + GQA (12 Q heads, 2 KV heads) + SwiGLU FFN.
//! Transcode pattern follows ndarray's GPT-2 and OpenChat engines.
//!
//! For full inference: needs safetensors weights (~3.1 GB).
//! For palette routing: needs bgz7 index (26 MB).

use super::weights::*;

/// Grouped Query Attention state per layer.
pub struct GqaState {
    /// KV cache: [seq_len, num_kv_heads, head_dim]
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub cache_len: usize,
}

impl GqaState {
    pub fn new() -> Self {
        Self {
            k_cache: Vec::with_capacity(MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM),
            v_cache: Vec::with_capacity(MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM),
            cache_len: 0,
        }
    }
}

/// Reader LM inference engine.
///
/// Currently a scaffold — the full forward pass requires loading
/// the actual safetensors weights (3.1 GB f32). The bgz7 weights
/// are Base17 fingerprints (26 MB) for palette routing only.
///
/// TODO: Transcode full Qwen2 forward pass:
/// 1. Token embedding (vocab_size × hidden_dim)
/// 2. RoPE position encoding (cos/sin cache)
/// 3. GQA attention (12 Q heads sharing 2 KV heads)
/// 4. SwiGLU FFN (gate * up → down)
/// 5. RMSNorm (per layer + final)
/// 6. LM head (hidden → vocab logits)
pub struct ReaderLmEngine {
    /// GQA state per layer.
    pub layers: Vec<GqaState>,
    /// Current position in sequence.
    pub position: usize,
}

impl ReaderLmEngine {
    pub fn new() -> Self {
        let layers = (0..NUM_LAYERS).map(|_| GqaState::new()).collect();
        Self { layers, position: 0 }
    }

    pub fn reset(&mut self) {
        self.position = 0;
        for layer in &mut self.layers {
            layer.cache_len = 0;
            layer.k_cache.clear();
            layer.v_cache.clear();
        }
    }

    /// Placeholder for full forward pass.
    /// Returns logits over vocabulary.
    pub fn forward(&mut self, _token_id: u32) -> Vec<f32> {
        self.position += 1;
        // Full Qwen2 forward pass would go here:
        // 1. embed = wte[token_id] (1536 dims)
        // 2. for each of 28 layers:
        //    a. RMSNorm(embed)
        //    b. QKV projection (Q: 12×128, K: 2×128, V: 2×128)
        //    c. RoPE on Q and K
        //    d. GQA: each Q head attends to its KV group (6:1 ratio)
        //    e. O projection
        //    f. Residual add
        //    g. RMSNorm
        //    h. SwiGLU FFN: gate(x) * up(x) → down
        //    i. Residual add
        // 3. Final RMSNorm
        // 4. LM head: hidden → vocab logits
        vec![0.0f32; VOCAB_SIZE]
    }

    /// Generate markdown from HTML tokens.
    /// Placeholder — requires full weights for actual generation.
    pub fn html_to_markdown(&mut self, _html_tokens: &[u32], max_tokens: usize) -> Vec<u32> {
        self.reset();
        // Would process HTML tokens through forward pass
        // and generate markdown tokens autoregressively
        Vec::with_capacity(max_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = ReaderLmEngine::new();
        assert_eq!(engine.layers.len(), NUM_LAYERS);
        assert_eq!(engine.position, 0);
    }

    #[test]
    fn test_engine_reset() {
        let mut engine = ReaderLmEngine::new();
        engine.position = 42;
        engine.reset();
        assert_eq!(engine.position, 0);
    }

    #[test]
    fn test_gqa_ratio() {
        // 12 query heads, 2 KV heads → 6:1 GQA ratio
        assert_eq!(NUM_HEADS / NUM_KV_HEADS, 6);
        // Each GQA group: 6 Q heads share 1 KV head
    }
}
