//! CAM-PQ distance contract.
//!
//! Defines the trait for CAM-PQ operations. ndarray provides the codec;
//! lance-graph provides the UDF + storage; consumers call this trait.

/// CAM fingerprint size in bytes.
pub const CAM_SIZE: usize = 6;

/// Number of subspaces.
pub const NUM_SUBSPACES: usize = 6;

/// Number of centroids per subspace.
pub const NUM_CENTROIDS: usize = 256;

/// Minimum element count for a tensor to be worth encoding via CAM-PQ.
/// Below this, codebook storage overhead dominates.
pub const CAM_PQ_MIN_ELEMENTS: u64 = 4096;

/// Routing decision for a single tensor in a model checkpoint.
///
/// Enforces invariant I1 (two regimes): index-regime tensors (embeddings,
/// lm_head) MUST stay Passthrough to preserve identity lookup; argmax-regime
/// tensors (attention Q/K/V/O, MLP gate/up/down) route through CAM-PQ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecRoute {
    /// Encode via CAM-PQ: 6-byte fingerprint + per-tensor codebook.
    /// Target: attention projections and MLP feed-forward layers.
    CamPq,
    /// Store as f32 (no compression). Required for index-regime tensors:
    /// embedding table, lm_head output projection, any tensor where row
    /// identity must round-trip exactly.
    Passthrough,
    /// Skip codec entirely — leave as f32 alongside other small tensors.
    /// Target: norms, biases, anything too small to benefit from codec.
    Skip,
}

/// Route a single tensor by name + dimensions.
///
/// Matching rules (applied in order; first match wins):
/// 1. `token_embd`, `embed_tokens`, `lm_head`, `wte`, `wpe` → `Passthrough`.
///    Identity lookup must be exact — no codec can survive Invariant I1.
/// 2. `norm`, `ln_`, `layer_norm` → `Skip`. Small; codec overhead wastes space.
/// 3. Attention `q/k/v/o_proj`, `attn_q/k/v/output`, `self_attn` → `CamPq`.
/// 4. MLP `gate_proj`, `up_proj`, `down_proj`, `ffn_gate/up/down`, `fc1/fc2`,
///    `w1/w2/w3`, generic `mlp`/`ffn` → `CamPq`.
/// 5. 4D tensors (Conv2D kernels) → `Skip` — not our target.
/// 6. Small tensors (< [`CAM_PQ_MIN_ELEMENTS`]) → `Skip`.
/// 7. Ambiguous 2D matrix ≥ `CAM_PQ_MIN_ELEMENTS` → `CamPq` (argmax default).
/// 8. Everything else → `Skip`.
///
/// # Example
///
/// ```
/// use lance_graph_contract::cam::{route_tensor, CodecRoute};
///
/// assert_eq!(route_tensor("model.layers.0.self_attn.q_proj.weight", &[4096, 4096]), CodecRoute::CamPq);
/// assert_eq!(route_tensor("model.embed_tokens.weight", &[151936, 1024]), CodecRoute::Passthrough);
/// assert_eq!(route_tensor("lm_head.weight", &[151936, 1024]), CodecRoute::Passthrough);
/// assert_eq!(route_tensor("model.layers.0.post_attention_layernorm.weight", &[4096]), CodecRoute::Skip);
/// ```
pub fn route_tensor(name: &str, dims: &[u64]) -> CodecRoute {
    // Rule 1: index-regime tensors — must be exact. Check before size/shape
    // rules so lm_head (which is 2D and large) isn't misrouted as CamPq.
    // Use `wte.` / `wpe.` as anchors to avoid matching unrelated 3-letter runs.
    let n_lower = name.to_ascii_lowercase();
    let is_wte_wpe = n_lower == "wte"
        || n_lower == "wpe"
        || n_lower.starts_with("wte.")
        || n_lower.starts_with("wpe.")
        || n_lower.ends_with(".wte")
        || n_lower.ends_with(".wpe")
        || n_lower.contains(".wte.")
        || n_lower.contains(".wpe.");
    if n_lower.contains("token_embd")
        || n_lower.contains("embed_tokens")
        || n_lower.contains("lm_head")
        || is_wte_wpe
    {
        return CodecRoute::Passthrough;
    }

    // Rule 2: norms are small and not worth encoding.
    if n_lower.contains("norm") || n_lower.contains("ln_") || n_lower.contains("layer_norm") {
        return CodecRoute::Skip;
    }

    // Rule 5 (applied early): skip conv kernels.
    if dims.len() == 4 {
        return CodecRoute::Skip;
    }

    // Rule 6: skip anything too small to benefit.
    let total: u64 = dims.iter().product();
    if total < CAM_PQ_MIN_ELEMENTS {
        return CodecRoute::Skip;
    }

    // Rule 3: attention projections.
    if n_lower.contains("q_proj")
        || n_lower.contains("k_proj")
        || n_lower.contains("v_proj")
        || n_lower.contains("o_proj")
        || n_lower.contains("attn_q")
        || n_lower.contains("attn_k")
        || n_lower.contains("attn_v")
        || n_lower.contains("attn_output")
        || n_lower.contains("self_attn")
    {
        return CodecRoute::CamPq;
    }

    // Rule 4: MLP / feed-forward.
    if n_lower.contains("gate_proj")
        || n_lower.contains("up_proj")
        || n_lower.contains("down_proj")
        || n_lower.contains("ffn_gate")
        || n_lower.contains("ffn_up")
        || n_lower.contains("ffn_down")
        || n_lower.contains("mlp")
        || n_lower.contains("ffn")
        || n_lower.contains("fc1")
        || n_lower.contains("fc2")
        || n_lower.contains("w1")
        || n_lower.contains("w2")
        || n_lower.contains("w3")
    {
        return CodecRoute::CamPq;
    }

    // Rule 7: ambiguous 2D matrix that's large enough → CamPq by default.
    if dims.len() == 2 && total >= CAM_PQ_MIN_ELEMENTS {
        return CodecRoute::CamPq;
    }

    CodecRoute::Skip
}

/// Named CAM bytes (stroke positions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CamByte {
    Heel   = 0,
    Branch = 1,
    TwigA  = 2,
    TwigB  = 3,
    Leaf   = 4,
    Gamma  = 5,
}

/// CAM-PQ scan strategy (selected by cost model).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CamStrategy {
    /// Full ADC: 6 table lookups per candidate. Good for < 10M.
    FullAdc,
    /// Stroke cascade: HEEL → BRANCH → full. 99% rejection. Good for > 10M.
    Cascade,
    /// IVF + Cascade: coarse partition probe then cascade. Good for > 100M.
    IvfCascade,
}

/// Precomputed distance tables (6 subspaces × 256 centroids = 6KB).
///
/// Fits in L1 cache. Computed once per query, reused for all candidates.
pub trait DistanceTableProvider: Send + Sync {
    /// Precompute distance tables from query vector + codebook.
    fn precompute(&self, query: &[f32], codebook: &[Vec<Vec<f32>>]) -> [[f32; 256]; 6];

    /// ADC distance to one 6-byte CAM fingerprint using precomputed tables.
    fn distance(&self, tables: &[[f32; 256]; 6], cam: &[u8; 6]) -> f32;

    /// Batch distance for N candidates. Returns distances.
    fn distance_batch(&self, tables: &[[f32; 256]; 6], cams: &[[u8; 6]]) -> Vec<f32>;
}

/// CAM-PQ codec contract.
///
/// ndarray implements this with its AVX-512 kernels.
/// lance-graph-planner uses it via the CamPqScanOp physical operator.
pub trait CamCodecContract: Send + Sync {
    /// Encode a batch of vectors to CAM fingerprints.
    fn encode_batch(&self, vectors: &[Vec<f32>], codebook: &[Vec<Vec<f32>>]) -> Vec<[u8; 6]>;

    /// Decode a CAM fingerprint back to approximate vector.
    fn decode(&self, cam: &[u8; 6], codebook: &[Vec<Vec<f32>>]) -> Vec<f32>;

    /// Train codebook from sample vectors (k-means).
    fn train_codebook(&self, vectors: &[Vec<f32>], dim: usize) -> Vec<Vec<Vec<f32>>>;

    /// Select scan strategy based on candidate count.
    fn select_strategy(&self, num_candidates: u64) -> CamStrategy;
}

/// IVF (Inverted File) contract for billion-scale search.
pub trait IvfContract: Send + Sync {
    /// Train coarse centroids from sample vectors.
    fn train(&self, vectors: &[Vec<f32>], num_partitions: usize);

    /// Assign a vector to its nearest partition.
    fn assign(&self, vector: &[f32]) -> u32;

    /// Find top-P closest partitions for a query.
    fn probe(&self, query: &[f32], num_probes: usize) -> Vec<(u32, f32)>;
}

#[cfg(test)]
mod route_tests {
    use super::*;

    #[test]
    fn attention_projections_route_campq() {
        assert_eq!(
            route_tensor("model.layers.0.self_attn.q_proj.weight", &[4096, 4096]),
            CodecRoute::CamPq,
        );
        assert_eq!(
            route_tensor("model.layers.12.self_attn.k_proj.weight", &[4096, 1024]),
            CodecRoute::CamPq,
        );
        assert_eq!(
            route_tensor("model.layers.5.self_attn.v_proj.weight", &[4096, 1024]),
            CodecRoute::CamPq,
        );
        assert_eq!(
            route_tensor("model.layers.0.self_attn.o_proj.weight", &[4096, 4096]),
            CodecRoute::CamPq,
        );
    }

    #[test]
    fn mlp_projections_route_campq() {
        assert_eq!(
            route_tensor("model.layers.0.mlp.gate_proj.weight", &[4096, 11008]),
            CodecRoute::CamPq,
        );
        assert_eq!(
            route_tensor("model.layers.0.mlp.up_proj.weight", &[4096, 11008]),
            CodecRoute::CamPq,
        );
        assert_eq!(
            route_tensor("model.layers.0.mlp.down_proj.weight", &[11008, 4096]),
            CodecRoute::CamPq,
        );
    }

    #[test]
    fn embeddings_stay_passthrough() {
        assert_eq!(
            route_tensor("model.embed_tokens.weight", &[151936, 1024]),
            CodecRoute::Passthrough,
        );
        assert_eq!(
            route_tensor("lm_head.weight", &[151936, 1024]),
            CodecRoute::Passthrough,
        );
        // GGUF naming
        assert_eq!(
            route_tensor("token_embd.weight", &[151936, 1024]),
            CodecRoute::Passthrough,
        );
        // GPT-2 naming
        assert_eq!(
            route_tensor("wte.weight", &[50257, 768]),
            CodecRoute::Passthrough,
        );
    }

    #[test]
    fn norms_skipped() {
        assert_eq!(
            route_tensor("model.layers.0.input_layernorm.weight", &[4096]),
            CodecRoute::Skip,
        );
        assert_eq!(
            route_tensor("model.norm.weight", &[4096]),
            CodecRoute::Skip,
        );
        assert_eq!(
            route_tensor("ln_1.weight", &[768]),
            CodecRoute::Skip,
        );
    }

    #[test]
    fn small_tensors_skipped() {
        // Under 4096 elements — biases, small projections.
        assert_eq!(
            route_tensor("model.layers.0.self_attn.q_proj.bias", &[256]),
            CodecRoute::Skip,
        );
    }

    #[test]
    fn conv2d_skipped() {
        // 4D tensor — conv kernel, not our target.
        assert_eq!(
            route_tensor("vision.patch_embed.proj.weight", &[768, 3, 16, 16]),
            CodecRoute::Skip,
        );
    }

    #[test]
    fn lm_head_not_misrouted_as_campq() {
        // lm_head is 2D, large, would match the ambiguous-2D fallback.
        // Must be caught by rule 1 first.
        let r = route_tensor("lm_head.weight", &[151936, 4096]);
        assert_eq!(r, CodecRoute::Passthrough, "lm_head must NOT route to CamPq");
    }

    #[test]
    fn ambiguous_large_2d_routes_campq() {
        // Generic 2D weight matrix, no clear role name → argmax default.
        assert_eq!(
            route_tensor("encoder.linear_fallback.weight", &[1024, 1024]),
            CodecRoute::CamPq,
        );
    }
}
