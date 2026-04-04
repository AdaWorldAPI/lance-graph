//! Jina v3 Semantic Precision Lens
//!
//! Baked-in 256×256 HDR distance table + 250K codebook index from
//! Jina v3 F16 GGUF token embeddings, CLAM sampled, CDF-percentile encoded.
//!
//! ```text
//! Table std=73.6 (vs linear 6.8) — 10.8× more topology
//! 256 centroids, balanced (766-1273 tokens each)
//! cos[-0.067, 0.234] → HDR u8[0,254] via CDF percentile
//! 64 KB table = L2 cache resident
//! ```
//!
//! This is a PRECISION LENS, not truth. The truth is in the domino chain.
//! The lens renders semantic distance for THIS model (Jina v3 XLM-RoBERTa).
//! Different models = different lenses on the same topology.

/// The 256×256 HDR distance table. Baked at build time from Jina v3 F16 GGUF.
/// Each u8 value is the CDF percentile rank of the pairwise cosine between
/// two CLAM-sampled centroid averages of token embeddings.
///
/// 64 KB = fits in L2 cache. No file I/O. No allocation.
pub static JINA_HDR_TABLE: &[u8; 256 * 256] = include_bytes!("../data/jina-v3-hdr/distance_table_256x256.u8");

/// Token → centroid codebook index. 250,002 entries × u16 = 488 KB.
/// Maps each Jina/XLM-RoBERTa BPE token ID to its nearest CLAM centroid.
pub static JINA_CODEBOOK_INDEX: &[u8] = include_bytes!("../data/jina-v3-hdr/codebook_index.u16");

/// Number of centroids in the Jina lens.
pub const JINA_N_CENTROIDS: usize = 256;

/// Vocabulary size (XLM-RoBERTa).
pub const JINA_VOCAB_SIZE: usize = 250_002;

/// Look up the centroid for a token ID.
#[inline]
pub fn jina_lookup(token_id: u32) -> u16 {
    let idx = (token_id as usize).min(JINA_VOCAB_SIZE - 1);
    let offset = idx * 2;
    if offset + 1 < JINA_CODEBOOK_INDEX.len() {
        u16::from_le_bytes([JINA_CODEBOOK_INDEX[offset], JINA_CODEBOOK_INDEX[offset + 1]])
    } else {
        0
    }
}

/// Look up centroids for a batch of token IDs.
pub fn jina_lookup_many(token_ids: &[u32]) -> Vec<u16> {
    token_ids.iter().map(|&id| jina_lookup(id)).collect()
}

/// Get the HDR distance between two centroids. O(1).
#[inline]
pub fn jina_distance(a: u16, b: u16) -> u8 {
    let ai = (a as usize).min(JINA_N_CENTROIDS - 1);
    let bi = (b as usize).min(JINA_N_CENTROIDS - 1);
    JINA_HDR_TABLE[ai * JINA_N_CENTROIDS + bi]
}

/// Create a ThinkingEngine from the baked Jina HDR table.
/// No file I/O. No allocation beyond the engine itself.
pub fn jina_engine() -> crate::engine::ThinkingEngine {
    crate::engine::ThinkingEngine::new(JINA_HDR_TABLE.to_vec())
}

/// Full pipeline: token IDs → centroids → domino cascade.
/// Returns (dominant_atom, chain, dissonance).
pub fn jina_think(
    token_ids: &[u32],
    cascade: &crate::domino::DominoCascade,
) -> (u16, Vec<u16>, crate::domino::DissonanceProfile) {
    let centroids = jina_lookup_many(token_ids);
    let (dom, stages, dis) = cascade.think(&centroids);
    let chain: Vec<u16> = stages.iter()
        .filter_map(|s| s.focus.first().map(|a| a.index))
        .collect();
    (dom, chain, dis)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_256x256() {
        assert_eq!(JINA_HDR_TABLE.len(), 256 * 256);
    }

    #[test]
    fn codebook_is_250k() {
        assert_eq!(JINA_CODEBOOK_INDEX.len(), JINA_VOCAB_SIZE * 2);
    }

    #[test]
    fn diagonal_is_255() {
        for i in 0..256 {
            assert_eq!(JINA_HDR_TABLE[i * 256 + i], 255, "diagonal[{}] should be 255", i);
        }
    }

    #[test]
    fn lookup_in_range() {
        for token_id in [0, 100, 1000, 50000, 200000, 249999] {
            let centroid = jina_lookup(token_id);
            assert!(centroid < 256, "centroid {} out of range for token {}", centroid, token_id);
        }
    }

    #[test]
    fn distance_symmetric() {
        for a in [0u16, 50, 100, 200, 255] {
            for b in [0u16, 50, 100, 200, 255] {
                assert_eq!(jina_distance(a, b), jina_distance(b, a),
                    "distance({},{}) != distance({},{})", a, b, b, a);
            }
        }
    }

    #[test]
    fn hdr_table_has_variance() {
        let avg = JINA_HDR_TABLE.iter().map(|&v| v as f64).sum::<f64>() / JINA_HDR_TABLE.len() as f64;
        let std = (JINA_HDR_TABLE.iter().map(|&v| { let d = v as f64 - avg; d * d })
            .sum::<f64>() / JINA_HDR_TABLE.len() as f64).sqrt();
        assert!(std > 50.0, "HDR table std={:.1} — should be >50 (was 73.6 at build)", std);
    }

    #[test]
    fn engine_creates_from_const() {
        let engine = jina_engine();
        assert_eq!(engine.size, 256);
    }
}
