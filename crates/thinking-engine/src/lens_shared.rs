//! Shared machinery for baked HDR lens modules (`jina_lens`, `bge_m3_lens`,
//! `reranker_lens`).
//!
//! Part of D-TEH-4 (ENTROPY M8 engine collapse). The three lens modules
//! carried byte-identical `_lookup` / `_lookup_many` / `_distance` bodies,
//! differing only in which baked table/codebook slice and vocab size they
//! closed over. This module holds the one generic implementation; each lens
//! module keeps its own public function names (unchanged call sites, unchanged
//! tests) as thin one-line delegations.
//!
//! What does NOT move here: the `_engine()` constructors (trivial,
//! model-specific `ThinkingEngine::new(TABLE.to_vec())` one-liners not worth
//! indirecting) and `bge_m3_lens::vote_distance` (cross-lens comparison logic,
//! not per-lens machinery).

/// Look up the centroid index for a token ID in a baked `token_id -> u16`
/// codebook (little-endian, 2 bytes/entry, out-of-range bytes read as `0`).
///
/// `vocab_size` bounds the token id (mirrors each lens's own `.min(VOCAB-1)`
/// clamp); `index` is the raw codebook byte slice.
#[inline]
pub fn codebook_lookup(index: &[u8], vocab_size: usize, token_id: u32) -> u16 {
    let idx = (token_id as usize).min(vocab_size.saturating_sub(1));
    let offset = idx * 2;
    if offset + 1 < index.len() {
        u16::from_le_bytes([index[offset], index[offset + 1]])
    } else {
        0
    }
}

/// Batch form of [`codebook_lookup`].
pub fn codebook_lookup_many(index: &[u8], vocab_size: usize, token_ids: &[u32]) -> Vec<u16> {
    token_ids
        .iter()
        .map(|&id| codebook_lookup(index, vocab_size, id))
        .collect()
}

/// Read the HDR distance between two centroids from an `n_centroids ×
/// n_centroids` baked table, clamping out-of-range centroid indices to the
/// last row/column (mirrors each lens's own `.min(N_CENTROIDS-1)` clamp).
#[inline]
pub fn hdr_distance(table: &[u8], n_centroids: usize, a: u16, b: u16) -> u8 {
    let ai = (a as usize).min(n_centroids.saturating_sub(1));
    let bi = (b as usize).min(n_centroids.saturating_sub(1));
    table[ai * n_centroids + bi]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_lookup_reads_le_pairs() {
        // token 0 -> bytes [1,0] = 1u16; token 1 -> bytes [0,1] = 256u16.
        let index = [1u8, 0u8, 0u8, 1u8];
        assert_eq!(codebook_lookup(&index, 2, 0), 1);
        assert_eq!(codebook_lookup(&index, 2, 1), 256);
    }

    #[test]
    fn codebook_lookup_clamps_out_of_range_token_to_last_entry() {
        let index = [1u8, 0u8, 0u8, 1u8];
        // vocab_size=2 -> token 99 clamps to index 1 (bytes [0,1] = 256).
        assert_eq!(codebook_lookup(&index, 2, 99), 256);
    }

    #[test]
    fn codebook_lookup_out_of_bounds_offset_reads_zero() {
        // vocab_size larger than the actual index slice: offset falls off
        // the end and must read 0, not panic.
        let index = [1u8, 0u8];
        assert_eq!(codebook_lookup(&index, 5, 4), 0);
    }

    #[test]
    fn hdr_distance_reads_the_expected_cell_not_a_neighbor() {
        // 3x3 table; distinguish (0,2) from every other cell so a transposed
        // or off-by-one index read would be caught.
        #[rustfmt::skip]
        let table = [
            10, 11, 12,
            13, 14, 15,
            16, 17, 18,
        ];
        assert_eq!(hdr_distance(&table, 3, 0, 2), 12);
        assert_eq!(hdr_distance(&table, 3, 2, 0), 16);
    }

    #[test]
    fn hdr_distance_clamps_out_of_range_centroid() {
        let table = [10u8, 11, 12, 13];
        // n_centroids=2 -> centroid 9 clamps to 1.
        assert_eq!(hdr_distance(&table, 2, 9, 0), hdr_distance(&table, 2, 1, 0));
    }
}
