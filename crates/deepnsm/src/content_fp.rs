//! 10K-dim content fingerprints from vocabulary ranks.
//!
//! Each word in the COCA vocabulary gets a deterministic `Vsa10k`
//! fingerprint: pseudo-random bits spread across all 10,000 dims
//! via SplitMix64 seeded from rank. These are the CONTENT vectors
//! that get bound into role-key slices via `RoleKey::bind`.
//!
//! Gated on `grammar-10k` feature (pulls in `lance-graph-contract`).

use lance_graph_contract::grammar::role_keys::{Vsa10k, VSA_WORDS, VSA_DIMS};

/// Generate a 10K-dim content fingerprint from a vocabulary rank.
///
/// Deterministic: same rank always produces the same vector.
/// ~50% bits set (balanced), good avalanche via SplitMix64.
pub fn content_fp(rank: u16) -> Vsa10k {
    let seed = (rank as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(0xBF58476D1CE4E5B9);
    let mut v = [0u64; VSA_WORDS];
    let mut state = seed;
    for w in 0..VSA_WORDS {
        state = splitmix64(state);
        v[w] = state;
    }
    // Zero slack bits above VSA_DIMS (10000) to prevent noise.
    let last_word = (VSA_DIMS - 1) / 64;
    let last_bit = VSA_DIMS % 64;
    if last_bit > 0 {
        v[last_word] &= (1u64 << last_bit) - 1;
    }
    for w in (last_word + 1)..VSA_WORDS {
        v[w] = 0;
    }
    v
}

/// Batch-generate content fingerprints for ranks `0..n`.
pub fn content_fp_table(n: u16) -> Vec<Vsa10k> {
    (0..n).map(content_fp).collect()
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        assert_eq!(content_fp(42), content_fp(42));
    }

    #[test]
    fn distinct_ranks_distinct_vectors() {
        let a = content_fp(0);
        let b = content_fp(1);
        assert_ne!(a, b);
    }

    #[test]
    fn balanced_popcount() {
        let v = content_fp(42);
        let pop: u32 = v.iter().map(|w| w.count_ones()).sum();
        let expected = VSA_DIMS as u32 / 2;
        let tolerance = (VSA_DIMS as f32).sqrt() as u32 * 4;
        assert!(
            pop.abs_diff(expected) < tolerance,
            "popcount {} should be near {}, tolerance {}",
            pop, expected, tolerance
        );
    }

    #[test]
    fn no_bits_above_vsa_dims() {
        let v = content_fp(999);
        for dim in VSA_DIMS..(VSA_WORDS * 64) {
            let word = dim / 64;
            let bit = dim % 64;
            assert_eq!(
                (v[word] >> bit) & 1, 0,
                "bit set above VSA_DIMS at dim {dim}"
            );
        }
    }

    #[test]
    fn table_has_correct_length() {
        let t = content_fp_table(100);
        assert_eq!(t.len(), 100);
    }
}
