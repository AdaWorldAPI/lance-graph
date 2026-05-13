//! BGE-M3 BF16 Semantic Precision Lens
//!
//! Same architecture as jina_lens but from BGE-M3 BF16 GGUF.
//! XLM-RoBERTa base, different training focus (multi-lingual retrieval).
//!
//! ```text
//! Table std=73.6 (HDR CDF percentile encoding)
//! 256 centroids, CLAM sampled from 250K token embeddings
//! BF16 source: f32::from_bits((u16 as u32) << 16) — one shift
//! ```
//!
//! Multi-lens voting: if Jina AND BGE-M3 agree on semantic distance,
//! NARS confidence increases. Disagreement = investigate.

/// BGE-M3 BF16 HDR distance table. 256×256 = 64 KB.
pub static BGE_M3_HDR_TABLE: &[u8; 256 * 256] =
    include_bytes!("../data/bge-m3-hdr/distance_table_256x256.u8");

/// BGE-M3 codebook index. 250,002 tokens × u16 = 488 KB.
pub static BGE_M3_CODEBOOK_INDEX: &[u8] =
    include_bytes!("../data/bge-m3-hdr/codebook_index.u16");

pub const BGE_M3_N_CENTROIDS: usize = 256;
pub const BGE_M3_VOCAB_SIZE: usize = 250_002;

#[inline]
pub fn bge_m3_lookup(token_id: u32) -> u16 {
    let idx = (token_id as usize).min(BGE_M3_VOCAB_SIZE - 1);
    let offset = idx * 2;
    if offset + 1 < BGE_M3_CODEBOOK_INDEX.len() {
        u16::from_le_bytes([BGE_M3_CODEBOOK_INDEX[offset], BGE_M3_CODEBOOK_INDEX[offset + 1]])
    } else { 0 }
}

pub fn bge_m3_lookup_many(token_ids: &[u32]) -> Vec<u16> {
    token_ids.iter().map(|&id| bge_m3_lookup(id)).collect()
}

#[inline]
pub fn bge_m3_distance(a: u16, b: u16) -> u8 {
    let ai = (a as usize).min(BGE_M3_N_CENTROIDS - 1);
    let bi = (b as usize).min(BGE_M3_N_CENTROIDS - 1);
    BGE_M3_HDR_TABLE[ai * BGE_M3_N_CENTROIDS + bi]
}

pub fn bge_m3_engine() -> crate::engine::ThinkingEngine {
    crate::engine::ThinkingEngine::new(BGE_M3_HDR_TABLE.to_vec())
}

/// Multi-lens vote: compare Jina and BGE-M3 distances for a token pair.
/// Returns (jina_dist, bge_dist, agreement) where agreement is 0.0-1.0.
pub fn vote_distance(token_a: u32, token_b: u32) -> (u8, u8, f32) {
    let jina_a = crate::jina_lens::jina_lookup(token_a);
    let jina_b = crate::jina_lens::jina_lookup(token_b);
    let jina_d = crate::jina_lens::jina_distance(jina_a, jina_b);

    let bge_a = bge_m3_lookup(token_a);
    let bge_b = bge_m3_lookup(token_b);
    let bge_d = bge_m3_distance(bge_a, bge_b);

    let diff = (jina_d as i16 - bge_d as i16).unsigned_abs() as f32;
    let agreement = 1.0 - (diff / 255.0);

    (jina_d, bge_d, agreement)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_size() { assert_eq!(BGE_M3_HDR_TABLE.len(), 256 * 256); }

    #[test]
    fn codebook_size() { assert_eq!(BGE_M3_CODEBOOK_INDEX.len(), BGE_M3_VOCAB_SIZE * 2); }

    #[test]
    fn diagonal_255() {
        for i in 0..256 {
            assert_eq!(BGE_M3_HDR_TABLE[i * 256 + i], 255);
        }
    }

    #[test]
    fn hdr_variance() {
        let avg = BGE_M3_HDR_TABLE.iter().map(|&v| v as f64).sum::<f64>() / BGE_M3_HDR_TABLE.len() as f64;
        let std = (BGE_M3_HDR_TABLE.iter().map(|&v| { let d = v as f64 - avg; d * d })
            .sum::<f64>() / BGE_M3_HDR_TABLE.len() as f64).sqrt();
        assert!(std > 50.0, "HDR std={:.1} should be >50", std);
    }

    #[test]
    fn vote_agreement() {
        let (j, b, agree) = vote_distance(100, 200);
        assert!(agree >= 0.0 && agree <= 1.0);
        // Both lenses should produce valid distances
        assert!(j <= 255);
        assert!(b <= 255);
    }
}
