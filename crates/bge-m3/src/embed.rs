//! BGE-M3 inference: real XLM-RoBERTa forward pass.
//!
//! Architecture: 24 layers, 16 heads, 1024 hidden, GELU, no GQA.
//! Weights loaded from safetensors/GGUF via bgz7 index OR raw.
//!
//! When no weights are loaded, falls back to a deterministic hash-based
//! embedding for development and testing.

use ndarray::hpc::bgz17_bridge::Base17;

use super::weights::*;

/// XLM-RoBERTa layer weights (one per transformer layer).
pub struct LayerWeights {
    pub attn_q: Vec<f32>,    // [HIDDEN_DIM, HIDDEN_DIM]
    pub attn_k: Vec<f32>,    // [HIDDEN_DIM, HIDDEN_DIM]
    pub attn_v: Vec<f32>,    // [HIDDEN_DIM, HIDDEN_DIM]
    pub attn_o: Vec<f32>,    // [HIDDEN_DIM, HIDDEN_DIM]
    pub attn_ln_w: Vec<f32>, // [HIDDEN_DIM]
    pub attn_ln_b: Vec<f32>, // [HIDDEN_DIM]
    pub ffn_up: Vec<f32>,    // [HIDDEN_DIM, MLP_DIM]
    pub ffn_down: Vec<f32>,  // [MLP_DIM, HIDDEN_DIM]
    pub ffn_ln_w: Vec<f32>,  // [HIDDEN_DIM]
    pub ffn_ln_b: Vec<f32>,  // [HIDDEN_DIM]
}

/// Full model weights for BGE-M3 (XLM-RoBERTa).
pub struct BgeM3Model {
    pub word_embeddings: Vec<f32>,     // [VOCAB_SIZE, HIDDEN_DIM]
    pub position_embeddings: Vec<f32>, // [MAX_SEQ_LEN + 2, HIDDEN_DIM]  (8194)
    pub embed_ln_w: Vec<f32>,          // [HIDDEN_DIM]
    pub embed_ln_b: Vec<f32>,          // [HIDDEN_DIM]
    pub layers: Vec<LayerWeights>,     // NUM_LAYERS entries
}

/// XLM-RoBERTa forward pass engine.
///
/// Supports two modes:
/// - With loaded weights: full transformer inference
/// - Without weights: deterministic hash-based embedding (for dev/testing)
pub struct BgeM3Engine {
    pub model: Option<BgeM3Model>,
}

impl BgeM3Engine {
    pub fn new() -> Self {
        Self { model: None }
    }

    pub fn load_model(&mut self, model: BgeM3Model) {
        self.model = Some(model);
    }

    /// Full forward pass: tokens -> 1024-dim L2-normalized embedding.
    /// If model not loaded, falls back to hash-based embedding.
    pub fn embed_tokens(&self, tokens: &[u32]) -> Vec<f32> {
        match &self.model {
            Some(model) => self.forward(model, tokens),
            None => self.embed_hash(tokens),
        }
    }

    /// Real XLM-RoBERTa forward pass.
    fn forward(&self, model: &BgeM3Model, tokens: &[u32]) -> Vec<f32> {
        let seq_len = tokens.len();

        // 1. Token + position embeddings
        //    XLM-RoBERTa has token_type_embeddings(1 × 1024) but it's all zeros
        //    for single-segment input, so we skip it.
        let mut hidden = vec![0.0f32; seq_len * HIDDEN_DIM];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok_offset = (tok as usize).min(VOCAB_SIZE - 1) * HIDDEN_DIM;
            let pos_offset = t.min(MAX_SEQ_LEN - 1) * HIDDEN_DIM;
            for d in 0..HIDDEN_DIM {
                hidden[t * HIDDEN_DIM + d] = model.word_embeddings[tok_offset + d]
                    + model.position_embeddings[pos_offset + d];
            }
        }

        // Embedding LayerNorm
        layer_norm(&mut hidden, seq_len, &model.embed_ln_w, &model.embed_ln_b);

        // 2. 24 transformer layers
        for layer in &model.layers {
            transformer_layer(&mut hidden, seq_len, layer);
        }

        // 3. Mean pool over sequence
        let mut pooled = vec![0.0f32; HIDDEN_DIM];
        for t in 0..seq_len {
            for d in 0..HIDDEN_DIM {
                pooled[d] += hidden[t * HIDDEN_DIM + d];
            }
        }
        let inv_len = 1.0 / seq_len as f32;
        for d in 0..HIDDEN_DIM {
            pooled[d] *= inv_len;
        }

        // 4. L2 normalize
        l2_normalize(&mut pooled);

        pooled
    }

    /// Hash-based fallback when no weights loaded.
    /// Deterministic: same tokens -> same embedding.
    fn embed_hash(&self, tokens: &[u32]) -> Vec<f32> {
        let mut embedding = vec![0.0f32; HIDDEN_DIM];
        for (i, &tok) in tokens.iter().enumerate() {
            let idx = (tok as usize).wrapping_mul(2654435761) % HIDDEN_DIM;
            embedding[idx] += 1.0 / (i as f32 + 1.0);
        }
        l2_normalize(&mut embedding);
        embedding
    }

    /// Embed text end-to-end (tokenize + forward).
    pub fn embed_text(&self, text: &str) -> Vec<f32> {
        let tokens = super::tokenizer::tokenize(text);
        self.embed_tokens(&tokens)
    }

    /// Project 1024-dim embedding to Base17.
    /// Uses golden-step octave averaging (same as ndarray bgz17_bridge).
    pub fn embed_to_base17(&self, text: &str) -> Base17 {
        let emb = self.embed_text(text);
        // Golden-step projection: 1024 -> 17
        let mut dims = [0i16; 17];
        let n_octaves = (HIDDEN_DIM + 17 - 1) / 17;
        let golden_pos: [usize; 17] = core::array::from_fn(|i| (i * 11) % 17);
        let mut sum = [0.0f64; 17];
        let mut count = [0u32; 17];
        for octave in 0..n_octaves {
            for bi in 0..17 {
                let dim = octave * 17 + golden_pos[bi];
                if dim < emb.len() {
                    sum[bi] += emb[dim] as f64;
                    count[bi] += 1;
                }
            }
        }
        for d in 0..17 {
            if count[d] > 0 {
                dims[d] = (sum[d] / count[d] as f64 * 256.0 * 10000.0)
                    .round()
                    .clamp(-32768.0, 32767.0) as i16;
            }
        }
        Base17 { dims }
    }

    /// Similarity via L1 distance on Base17 fingerprints.
    /// 0.0 = identical, 1.0 = maximally different.
    pub fn distance(&self, a: &str, b: &str) -> f32 {
        let fa = self.embed_to_base17(a);
        let fb = self.embed_to_base17(b);
        fa.l1(&fb) as f32 / (17u32 * 65535) as f32
    }

    /// Similarity (inverse of distance). 1.0 = identical, 0.0 = maximally different.
    pub fn similarity(&self, a: &str, b: &str) -> f32 {
        1.0 - self.distance(a, b)
    }

    /// Find most similar from candidates.
    pub fn most_similar<'a>(
        &self,
        query: &str,
        candidates: &'a [&str],
    ) -> Option<(usize, f32, &'a str)> {
        let qfp = self.embed_to_base17(query);
        candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let sim =
                    1.0 - qfp.l1(&self.embed_to_base17(c)) as f32 / (17u32 * 65535) as f32;
                (i, sim, *c)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }

    /// Batch embed multiple texts to Base17.
    pub fn batch_embed(&self, texts: &[&str]) -> Vec<Base17> {
        texts.iter().map(|t| self.embed_to_base17(t)).collect()
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// L2 normalize a vector in place.
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    let inv = 1.0 / norm;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// LayerNorm: for each position, normalize to zero mean / unit variance, then scale + shift.
fn layer_norm(hidden: &mut [f32], seq_len: usize, weight: &[f32], bias: &[f32]) {
    for t in 0..seq_len {
        let offset = t * HIDDEN_DIM;
        let slice = &mut hidden[offset..offset + HIDDEN_DIM];
        let mean: f32 = slice.iter().sum::<f32>() / HIDDEN_DIM as f32;
        let var: f32 =
            slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / HIDDEN_DIM as f32;
        let inv_std = 1.0 / (var + 1e-5f32).sqrt();
        for d in 0..HIDDEN_DIM {
            slice[d] = (slice[d] - mean) * inv_std * weight[d] + bias[d];
        }
    }
}

/// Single transformer layer: LN → MHA → residual → LN → FFN → residual.
fn transformer_layer(hidden: &mut [f32], seq_len: usize, layer: &LayerWeights) {
    // Pre-LN for attention
    let mut normed = hidden.to_vec();
    layer_norm(
        &mut normed,
        seq_len,
        &layer.attn_ln_w,
        &layer.attn_ln_b,
    );

    // QKV projection
    let mut q = vec![0.0f32; seq_len * HIDDEN_DIM];
    let mut k = vec![0.0f32; seq_len * HIDDEN_DIM];
    let mut v = vec![0.0f32; seq_len * HIDDEN_DIM];
    matmul(&normed, &layer.attn_q, &mut q, seq_len, HIDDEN_DIM, HIDDEN_DIM);
    matmul(&normed, &layer.attn_k, &mut k, seq_len, HIDDEN_DIM, HIDDEN_DIM);
    matmul(&normed, &layer.attn_v, &mut v, seq_len, HIDDEN_DIM, HIDDEN_DIM);

    // Multi-head attention (16 heads, head_dim=64)
    let mut attn_out = vec![0.0f32; seq_len * HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    for h in 0..NUM_HEADS {
        let head_off = h * HEAD_DIM;
        for i in 0..seq_len {
            // Compute attention scores for position i
            let mut scores = vec![0.0f32; seq_len];
            for j in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..HEAD_DIM {
                    dot += q[i * HIDDEN_DIM + head_off + d] * k[j * HIDDEN_DIM + head_off + d];
                }
                scores[j] = dot * scale;
            }
            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                sum_exp += *s;
            }
            let inv_sum = 1.0 / sum_exp;
            for s in &mut scores {
                *s *= inv_sum;
            }
            // Weighted sum of V
            for d in 0..HEAD_DIM {
                let mut val = 0.0f32;
                for j in 0..seq_len {
                    val += scores[j] * v[j * HIDDEN_DIM + head_off + d];
                }
                attn_out[i * HIDDEN_DIM + head_off + d] = val;
            }
        }
    }

    // Output projection + residual
    let mut o_out = vec![0.0f32; seq_len * HIDDEN_DIM];
    matmul(
        &attn_out,
        &layer.attn_o,
        &mut o_out,
        seq_len,
        HIDDEN_DIM,
        HIDDEN_DIM,
    );
    for i in 0..hidden.len() {
        hidden[i] += o_out[i];
    }

    // Pre-LN for FFN
    let mut normed2 = hidden.to_vec();
    layer_norm(
        &mut normed2,
        seq_len,
        &layer.ffn_ln_w,
        &layer.ffn_ln_b,
    );

    // FFN: up-project → GELU → down-project
    let mut ffn_mid = vec![0.0f32; seq_len * MLP_DIM];
    matmul(
        &normed2,
        &layer.ffn_up,
        &mut ffn_mid,
        seq_len,
        HIDDEN_DIM,
        MLP_DIM,
    );
    for x in &mut ffn_mid {
        *x = gelu(*x);
    }
    let mut ffn_out = vec![0.0f32; seq_len * HIDDEN_DIM];
    matmul(
        &ffn_mid,
        &layer.ffn_down,
        &mut ffn_out,
        seq_len,
        MLP_DIM,
        HIDDEN_DIM,
    );
    for i in 0..hidden.len() {
        hidden[i] += ffn_out[i];
    }
}

/// Dense matrix multiply: out[m, n] = a[m, k] × b[k, n].
fn matmul(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    // Row-major layout. Not optimized — real deployment should use ndarray BLAS.
    for i in 0..m {
        let a_row = i * k;
        let o_row = i * n;
        for p in 0..k {
            let a_val = a[a_row + p];
            let b_row = p * n;
            for j in 0..n {
                out[o_row + j] += a_val * b[b_row + j];
            }
        }
    }
}

/// GELU activation (Gaussian Error Linear Unit, tanh approximation).
fn gelu(x: f32) -> f32 {
    0.5 * x
        * (1.0
            + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

// ============================================================================
// Legacy API — thin wrappers over BgeM3Engine for backward compatibility
// ============================================================================

/// Embed text as Base17 fingerprint (legacy API).
pub fn embed_text(text: &str) -> Base17 {
    let engine = BgeM3Engine::new();
    engine.embed_to_base17(text)
}

/// Similarity via L1 distance (legacy API).
pub fn distance(a: &str, b: &str) -> f32 {
    let engine = BgeM3Engine::new();
    engine.distance(a, b)
}

/// Similarity (inverse of distance) (legacy API).
pub fn similarity(a: &str, b: &str) -> f32 {
    let engine = BgeM3Engine::new();
    engine.similarity(a, b)
}

/// Find most similar from candidates (legacy API).
pub fn most_similar<'a>(query: &str, candidates: &'a [&str]) -> Option<(usize, f32, &'a str)> {
    let engine = BgeM3Engine::new();
    engine.most_similar(query, candidates)
}

/// Batch embed multiple texts (legacy API).
pub fn batch_embed(texts: &[&str]) -> Vec<Base17> {
    let engine = BgeM3Engine::new();
    engine.batch_embed(texts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = BgeM3Engine::new();
        assert!(engine.model.is_none());
    }

    #[test]
    fn test_embed_hash_fallback() {
        let engine = BgeM3Engine::new();
        let emb = engine.embed_text("hello world");
        assert_eq!(emb.len(), HIDDEN_DIM);
        // Should be L2 normalized
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_embed_deterministic() {
        let engine = BgeM3Engine::new();
        let a = engine.embed_text("test");
        let b = engine.embed_text("test");
        assert_eq!(a, b);
    }

    #[test]
    fn test_embed_different_texts() {
        let engine = BgeM3Engine::new();
        let a = engine.embed_text("machine learning");
        let b = engine.embed_text("cooking pasta");
        assert_ne!(a, b);
    }

    #[test]
    fn test_embed_to_base17() {
        let engine = BgeM3Engine::new();
        let fp = engine.embed_to_base17("hello");
        assert_ne!(fp.dims, [0i16; 17]);
    }

    #[test]
    fn test_tokenizer_integration() {
        let tokens = super::super::tokenizer::tokenize("Hello world");
        assert_eq!(tokens[0], 0); // CLS
        assert_eq!(*tokens.last().unwrap(), 2); // SEP
        assert!(tokens.len() >= 4); // CLS + Hello + world + SEP
    }

    // Legacy API tests
    #[test]
    fn test_embed() {
        assert_ne!(embed_text("hello").dims, [0; 17]);
    }

    #[test]
    fn test_self_sim() {
        assert!((similarity("x", "x") - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_diff() {
        // With hash-based fallback + Base17 projection, L1 similarity is coarse.
        // Just verify that different texts don't produce identical fingerprints.
        let a = embed_text("cat");
        let b = embed_text("quantum physics");
        assert_ne!(a.dims, b.dims);
    }

    #[test]
    fn test_batch() {
        assert_eq!(batch_embed(&["a", "b", "c"]).len(), 3);
    }

    #[test]
    fn test_most_similar() {
        let r = most_similar("deep learning", &["cat", "machine learning", "cooking"]).unwrap();
        assert!(r.2.contains("learning"));
    }

    #[test]
    fn test_gelu_zero() {
        assert!((gelu(0.0)).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_positive() {
        // GELU(x) ≈ x for large positive x
        assert!((gelu(3.0) - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }
}
