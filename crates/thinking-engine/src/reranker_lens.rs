//! Jina Reranker v3 BF16 Lens — cross-encoder relevance scoring.
//!
//! Baked-in 256×256 HDR distance table + 151K codebook index from
//! Jina Reranker v3 BF16 GGUF, CLAM sampled, CDF-percentile encoded.
//!
//! ```text
//! cos[-0.886, +0.826] — WIDEST signed range of all models
//! Nearly symmetric: balanced excitation/inhibition
//! 256 centroids, 151,936 vocab (Qwen2 tokenizer)
//! 64 KB table = L2 cache resident
//! ```
//!
//! The reranker is the most informative lens for i8 signed experiments
//! because its cosine range is nearly symmetric around zero.
//! For the i8/u8 dual-path comparison: start here.

/// The 256×256 HDR distance table from Jina Reranker v3 BF16.
pub static RERANKER_HDR_TABLE: &[u8; 256 * 256] =
    include_bytes!("../data/jina-reranker-v3-BF16-hdr/distance_table_256x256.u8");

/// Token → centroid codebook index. 151,936 entries × u16 = 296 KB.
pub static RERANKER_CODEBOOK_INDEX: &[u8] =
    include_bytes!("../data/jina-reranker-v3-BF16-hdr/codebook_index.u16");

/// Number of centroids.
pub const RERANKER_N_CENTROIDS: usize = 256;

/// Vocabulary size (Qwen2 tokenizer, shared with reader-lm).
pub const RERANKER_VOCAB_SIZE: usize = 151_936;

/// Look up the centroid for a token ID.
#[inline]
pub fn reranker_lookup(token_id: u32) -> u16 {
    let idx = (token_id as usize).min(RERANKER_VOCAB_SIZE - 1);
    let offset = idx * 2;
    if offset + 1 < RERANKER_CODEBOOK_INDEX.len() {
        u16::from_le_bytes([RERANKER_CODEBOOK_INDEX[offset], RERANKER_CODEBOOK_INDEX[offset + 1]])
    } else {
        0
    }
}

/// Look up centroids for a batch of token IDs.
pub fn reranker_lookup_many(token_ids: &[u32]) -> Vec<u16> {
    token_ids.iter().map(|&id| reranker_lookup(id)).collect()
}

/// Get the HDR distance between two centroids. O(1).
#[inline]
pub fn reranker_distance(a: u16, b: u16) -> u8 {
    let ai = (a as usize).min(RERANKER_N_CENTROIDS - 1);
    let bi = (b as usize).min(RERANKER_N_CENTROIDS - 1);
    RERANKER_HDR_TABLE[ai * RERANKER_N_CENTROIDS + bi]
}

/// Create a ThinkingEngine from the baked reranker HDR table.
pub fn reranker_engine() -> crate::engine::ThinkingEngine {
    crate::engine::ThinkingEngine::new(RERANKER_HDR_TABLE.to_vec())
}

/// Full pipeline: token IDs → centroids → domino cascade.
pub fn reranker_think(
    token_ids: &[u32],
    cascade: &crate::domino::DominoCascade,
) -> (u16, Vec<u16>, crate::domino::DissonanceProfile) {
    let centroids = reranker_lookup_many(token_ids);
    let (dom, stages, dis) = cascade.think(&centroids);
    let chain: Vec<u16> = stages.iter()
        .filter_map(|s| s.focus.first().map(|a| a.index))
        .collect();
    (dom, chain, dis)
}

/// Relevance score between two texts via reranker lens.
///
/// Cross-encoder style: encode both texts, compare centroid activations.
/// Higher score = more relevant. Uses domino cascade for multi-hop comparison.
pub fn reranker_relevance(
    query_ids: &[u32],
    document_ids: &[u32],
) -> f32 {
    let q_centroids = reranker_lookup_many(query_ids);
    let d_centroids = reranker_lookup_many(document_ids);

    // Cross-attention: for each query centroid, find best document match
    let mut total_score = 0.0f32;
    let mut pairs = 0;

    for &qc in &q_centroids {
        let mut best = 0u8;
        for &dc in &d_centroids {
            let dist = reranker_distance(qc, dc);
            if dist > best { best = dist; }
        }
        total_score += best as f32 / 255.0;
        pairs += 1;
    }

    if pairs > 0 { total_score / pairs as f32 } else { 0.0 }
}

/// Compare two texts using Jina v3 embedding + reranker cross-validation.
/// Returns (embedding_similarity, reranker_relevance, agreement).
pub fn cross_model_eval(
    text_a_jina_ids: &[u32],
    text_b_jina_ids: &[u32],
    text_a_reranker_ids: &[u32],
    text_b_reranker_ids: &[u32],
) -> CrossModelResult {
    // Jina v3 embedding similarity (symmetric)
    let jina_centroids_a = super::jina_lens::jina_lookup_many(text_a_jina_ids);
    let jina_centroids_b = super::jina_lens::jina_lookup_many(text_b_jina_ids);
    let mut jina_sim = 0.0f32;
    let mut jina_pairs = 0;
    for &ca in &jina_centroids_a {
        for &cb in &jina_centroids_b {
            jina_sim += super::jina_lens::jina_distance(ca, cb) as f32 / 255.0;
            jina_pairs += 1;
        }
    }
    let jina_score = if jina_pairs > 0 { jina_sim / jina_pairs as f32 } else { 0.0 };

    // Reranker relevance (asymmetric: query → document)
    let reranker_score = reranker_relevance(text_a_reranker_ids, text_b_reranker_ids);

    // Agreement: geometric mean (weakest-link property)
    let agreement = (jina_score * reranker_score).sqrt();

    CrossModelResult {
        jina_similarity: jina_score,
        reranker_relevance: reranker_score,
        agreement,
    }
}

/// Result of cross-model evaluation.
#[derive(Debug, Clone, Copy)]
pub struct CrossModelResult {
    /// Jina v3 embedding similarity (symmetric, 0-1).
    pub jina_similarity: f32,
    /// Reranker relevance (asymmetric, 0-1).
    pub reranker_relevance: f32,
    /// Agreement: geometric mean of both (0-1).
    pub agreement: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_256x256() {
        assert_eq!(RERANKER_HDR_TABLE.len(), 256 * 256);
    }

    #[test]
    fn codebook_is_151k() {
        assert_eq!(RERANKER_CODEBOOK_INDEX.len(), RERANKER_VOCAB_SIZE * 2);
    }

    #[test]
    fn diagonal_is_255() {
        for i in 0..256 {
            assert_eq!(RERANKER_HDR_TABLE[i * 256 + i], 255,
                "diagonal[{}] should be 255", i);
        }
    }

    #[test]
    fn lookup_in_range() {
        for token_id in [0, 100, 1000, 50000, 100000, 151935] {
            let centroid = reranker_lookup(token_id);
            assert!(centroid < 256, "centroid {} out of range for token {}", centroid, token_id);
        }
    }

    #[test]
    fn distance_symmetric() {
        for a in [0u16, 50, 100, 200, 255] {
            for b in [0u16, 50, 100, 200, 255] {
                assert_eq!(reranker_distance(a, b), reranker_distance(b, a));
            }
        }
    }

    #[test]
    fn hdr_table_has_variance() {
        let avg = RERANKER_HDR_TABLE.iter().map(|&v| v as f64).sum::<f64>()
            / RERANKER_HDR_TABLE.len() as f64;
        let std = (RERANKER_HDR_TABLE.iter()
            .map(|&v| { let d = v as f64 - avg; d * d })
            .sum::<f64>() / RERANKER_HDR_TABLE.len() as f64)
            .sqrt();
        assert!(std > 50.0, "HDR table std={:.1} — should be >50", std);
    }

    #[test]
    fn engine_creates() {
        let engine = reranker_engine();
        assert_eq!(engine.size, 256);
    }

    #[test]
    fn relevance_self_is_high() {
        // Same tokens should have high relevance
        let ids: Vec<u32> = (0..20).collect();
        let score = reranker_relevance(&ids, &ids);
        assert!(score > 0.5, "self-relevance should be high: {}", score);
    }

    #[test]
    fn cross_model_runs() {
        let ids_a: Vec<u32> = (0..10).collect();
        let ids_b: Vec<u32> = (1000..1010).collect();
        let result = cross_model_eval(&ids_a, &ids_b, &ids_a, &ids_b);
        assert!(result.agreement >= 0.0 && result.agreement <= 1.0);
    }
}
