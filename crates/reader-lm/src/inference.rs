//! Qwen2 1.5B forward pass for Reader-LM.
//! Architecture: RoPE + GQA(12:2) + SwiGLU + RMSNorm.

use super::weights::*;

pub struct Qwen2LayerWeights {
    pub attn_q: Vec<f32>,      // [1536, 1536] (12 heads x 128)
    pub attn_k: Vec<f32>,      // [1536, 256]  (2 KV heads x 128)
    pub attn_v: Vec<f32>,      // [1536, 256]
    pub attn_o: Vec<f32>,      // [1536, 1536]
    pub attn_norm: Vec<f32>,   // [1536] RMSNorm
    pub ffn_gate: Vec<f32>,    // [1536, 8960]
    pub ffn_up: Vec<f32>,      // [1536, 8960]
    pub ffn_down: Vec<f32>,    // [8960, 1536]
    pub ffn_norm: Vec<f32>,    // [1536] RMSNorm
}

pub struct Qwen2Model {
    pub wte: Vec<f32>,                // [151936, 1536]
    pub layers: Vec<Qwen2LayerWeights>,
    pub final_norm: Vec<f32>,         // [1536]
    pub lm_head: Vec<f32>,            // [1536, 151936]
}

pub struct ReaderLmEngine {
    pub model: Option<Qwen2Model>,
    pub kv_cache: Vec<KvCache>,
    pub position: usize,
}

pub struct KvCache {
    pub k: Vec<f32>, // [seq, NUM_KV_HEADS * HEAD_DIM]
    pub v: Vec<f32>,
    pub len: usize,
}

impl ReaderLmEngine {
    pub fn new() -> Self {
        Self { model: None, kv_cache: Vec::new(), position: 0 }
    }

    pub fn load_model(&mut self, model: Qwen2Model) {
        self.kv_cache = (0..model.layers.len()).map(|_| KvCache {
            k: Vec::new(), v: Vec::new(), len: 0,
        }).collect();
        self.model = Some(model);
    }

    pub fn reset(&mut self) {
        self.position = 0;
        for cache in &mut self.kv_cache {
            cache.k.clear();
            cache.v.clear();
            cache.len = 0;
        }
    }

    /// Forward pass: one token -> logits.
    pub fn forward(&mut self, token_id: u32) -> Vec<f32> {
        if self.model.is_none() {
            return vec![0.0; VOCAB_SIZE];
        }
        let pos = self.position;
        self.position += 1;

        // 1. Token embedding
        let model = self.model.as_ref().unwrap();
        let tok_offset = (token_id as usize).min(VOCAB_SIZE - 1) * HIDDEN_DIM;
        let mut hidden = vec![0.0f32; HIDDEN_DIM];
        for d in 0..HIDDEN_DIM {
            hidden[d] = model.wte[tok_offset + d];
        }

        // 2. 28 transformer layers
        let num_layers = model.layers.len();
        for l in 0..num_layers {
            Self::qwen2_layer(&mut self.kv_cache, self.model.as_ref().unwrap(), &mut hidden, l, pos);
        }

        // 3. Final RMSNorm
        let model = self.model.as_ref().unwrap();
        rms_norm_inplace(&mut hidden, &model.final_norm);

        // 4. LM head
        let mut logits = vec![0.0f32; VOCAB_SIZE];
        for v in 0..VOCAB_SIZE {
            let mut dot = 0.0f32;
            for d in 0..HIDDEN_DIM {
                dot += hidden[d] * model.lm_head[d * VOCAB_SIZE + v];
            }
            logits[v] = dot;
        }

        logits
    }

    fn qwen2_layer(kv_cache: &mut [KvCache], model: &Qwen2Model, hidden: &mut [f32], layer_idx: usize, pos: usize) {
        let layer = &model.layers[layer_idx];
        let kv_dim = NUM_KV_HEADS * HEAD_DIM; // 2 x 128 = 256

        // Pre-attention RMSNorm
        let mut normed = hidden.to_vec();
        rms_norm_inplace(&mut normed, &layer.attn_norm);

        // Q projection: [1536] -> [1536] (12 heads x 128)
        let mut q = vec![0.0f32; HIDDEN_DIM];
        matmul_vec(&normed, &layer.attn_q, &mut q, HIDDEN_DIM, HIDDEN_DIM);

        // K projection: [1536] -> [256] (2 KV heads x 128)
        let mut k = vec![0.0f32; kv_dim];
        matmul_vec(&normed, &layer.attn_k, &mut k, HIDDEN_DIM, kv_dim);

        // V projection: [1536] -> [256]
        let mut v = vec![0.0f32; kv_dim];
        matmul_vec(&normed, &layer.attn_v, &mut v, HIDDEN_DIM, kv_dim);

        // Apply RoPE to Q and K
        apply_rope_inplace(&mut q, NUM_HEADS, HEAD_DIM, pos);
        apply_rope_inplace(&mut k, NUM_KV_HEADS, HEAD_DIM, pos);

        // Append to KV cache
        let cache = &mut kv_cache[layer_idx];
        cache.k.extend_from_slice(&k);
        cache.v.extend_from_slice(&v);
        cache.len += 1;

        // GQA attention: 12 Q heads, 2 KV heads (6:1 ratio)
        let mut attn_out = vec![0.0f32; HIDDEN_DIM];
        let kv_group_size = NUM_HEADS / NUM_KV_HEADS; // 6

        for h in 0..NUM_HEADS {
            let kv_h = h / kv_group_size; // which KV head
            let q_off = h * HEAD_DIM;

            // Compute scores against all cached K
            let mut scores = vec![0.0f32; cache.len];
            for t in 0..cache.len {
                let k_off = t * kv_dim + kv_h * HEAD_DIM;
                let mut dot = 0.0f32;
                for d in 0..HEAD_DIM {
                    dot += q[q_off + d] * cache.k[k_off + d];
                }
                scores[t] = dot / (HEAD_DIM as f32).sqrt();
            }

            // Causal softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for s in &mut scores { *s = (*s - max_s).exp(); sum_exp += *s; }
            for s in &mut scores { *s /= sum_exp; }

            // Weighted sum of V
            for d in 0..HEAD_DIM {
                let mut val = 0.0f32;
                for t in 0..cache.len {
                    val += scores[t] * cache.v[t * kv_dim + kv_h * HEAD_DIM + d];
                }
                attn_out[q_off + d] = val;
            }
        }

        // O projection + residual
        let mut o_out = vec![0.0f32; HIDDEN_DIM];
        matmul_vec(&attn_out, &layer.attn_o, &mut o_out, HIDDEN_DIM, HIDDEN_DIM);
        for d in 0..HIDDEN_DIM { hidden[d] += o_out[d]; }

        // Pre-FFN RMSNorm
        let mut normed2 = hidden.to_vec();
        rms_norm_inplace(&mut normed2, &layer.ffn_norm);

        // SwiGLU FFN: gate(x) * up(x) -> down
        let mut gate = vec![0.0f32; MLP_DIM];
        let mut up = vec![0.0f32; MLP_DIM];
        matmul_vec(&normed2, &layer.ffn_gate, &mut gate, HIDDEN_DIM, MLP_DIM);
        matmul_vec(&normed2, &layer.ffn_up, &mut up, HIDDEN_DIM, MLP_DIM);
        // SiLU(gate) * up
        for i in 0..MLP_DIM {
            gate[i] = silu(gate[i]) * up[i];
        }
        let mut down_out = vec![0.0f32; HIDDEN_DIM];
        matmul_vec(&gate, &layer.ffn_down, &mut down_out, MLP_DIM, HIDDEN_DIM);
        for d in 0..HIDDEN_DIM { hidden[d] += down_out[d]; }
    }

    /// Generate tokens autoregressively.
    pub fn generate(&mut self, prompt_tokens: &[u32], max_new: usize) -> Vec<u32> {
        self.reset();
        let mut generated = Vec::new();

        // Process prompt
        let mut last_logits = vec![0.0f32; VOCAB_SIZE];
        for &tok in prompt_tokens {
            last_logits = self.forward(tok);
        }

        // Generate
        for _ in 0..max_new {
            let best_id = last_logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            if best_id == 151645 { break; } // EOS
            generated.push(best_id);
            last_logits = self.forward(best_id);
        }
        generated
    }

    /// HTML -> Markdown conversion.
    pub fn html_to_markdown(&mut self, html: &str, max_tokens: usize) -> Vec<u32> {
        let tokens = super::tokenizer::tokenize(html);
        self.generate(&tokens, max_tokens)
    }
}

// Helpers

fn rms_norm_inplace(x: &mut [f32], weight: &[f32]) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (ss + 1e-6).sqrt();
    for i in 0..n { x[i] = x[i] * inv_rms * weight[i]; }
}

fn apply_rope_inplace(x: &mut [f32], n_heads: usize, head_dim: usize, pos: usize) {
    let theta_base: f64 = 2000000.0; // rope_theta from config
    for h in 0..n_heads {
        let off = h * head_dim;
        for i in (0..head_dim).step_by(2) {
            let freq = 1.0 / theta_base.powf(i as f64 / head_dim as f64);
            let angle = pos as f64 * freq;
            let cos_a = angle.cos() as f32;
            let sin_a = angle.sin() as f32;
            let x0 = x[off + i];
            let x1 = x[off + i + 1];
            x[off + i] = x0 * cos_a - x1 * sin_a;
            x[off + i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

fn matmul_vec(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize) {
    for j in 0..n {
        let mut sum = 0.0f32;
        for i in 0..m { sum += a[i] * b[i * n + j]; }
        out[j] = sum;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_no_model() {
        let mut engine = ReaderLmEngine::new();
        let logits = engine.forward(0);
        assert_eq!(logits.len(), VOCAB_SIZE);
    }

    #[test]
    fn test_rms_norm() {
        let weight = vec![1.0f32; 4];
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        rms_norm_inplace(&mut x, &weight);
        // Should be normalized
        let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((ss - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 0.01);
        assert!(silu(5.0) > 4.9); // silu(large) ~ x
        assert!(silu(-5.0).abs() < 0.05); // silu(large neg) ~ 0
    }

    #[test]
    fn test_rope() {
        let mut x = vec![1.0, 0.0, 1.0, 0.0];
        apply_rope_inplace(&mut x, 1, 4, 0);
        // At position 0, cos(0)=1, sin(0)=0, so no change
        assert!((x[0] - 1.0).abs() < 0.01);
        assert!((x[1] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_gqa_ratio() {
        assert_eq!(NUM_HEADS / NUM_KV_HEADS, 6);
    }

    #[test]
    fn test_tokenizer() {
        let tokens = super::super::tokenizer::tokenize("Hello world");
        assert_eq!(tokens[0], 151643); // BOS
        assert!(tokens.len() >= 3);
    }
}
