// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bitset population-count primitives — the SIMD seam.
//!
//! The data-confirmation hot loop (counting rows that satisfy a conjunction of
//! items) is the `faiss-homology` "SIMD batch-AND over the SoA facet column"
//! workload. With the row-bitset SoA ([`crate::bitset::RowMasks`]) it reduces
//! to `AND` + popcount over `&[u64]`.
//!
//! Per `.claude/knowledge/ndarray-vertical-simd-alien-magic.md` (MANDATORY for
//! consumer SIMD), this routes through **`ndarray::simd::U64x8`** — zero raw
//! intrinsics, zero `cfg(target_arch)`, zero feature detection in this crate;
//! the polyfill owns dispatch. The scalar path is the default (so the crate
//! stays std-only and independently verifiable); the `ndarray-simd` feature
//! swaps in the vectorised path.

/// Population count of a bitset (`Σ count_ones`).
#[cfg(not(feature = "ndarray-simd"))]
#[must_use]
pub fn popcount(words: &[u64]) -> u32 {
    words.iter().map(|w| w.count_ones()).sum()
}

/// Population count of `a AND b` (the co-occurrence count of two item masks).
#[cfg(not(feature = "ndarray-simd"))]
#[must_use]
pub fn and_popcount(a: &[u64], b: &[u64]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x & y).count_ones()).sum()
}

// ── ndarray::simd vectorised path (feature `ndarray-simd`) ──────────────────
//
// Uses ONLY `ndarray::simd::U64x8` (`from_slice` + `&` (BitAnd) + `popcnt` +
// `to_array`) — the canonical W1b consumer surface, zero raw intrinsics.
// Verified: `cargo test/clippy --features ndarray-simd` → 33/33, clippy clean.
// The real AVX-512 VPOPCNTQ / AMX kernels need `-C target-cpu=native` or
// `x86-64-v4`; otherwise this is ndarray's scalar polyfill (correct, not
// accelerated).

/// Population count of a bitset, via `ndarray::simd::U64x8::popcnt`.
#[cfg(feature = "ndarray-simd")]
#[must_use]
pub fn popcount(words: &[u64]) -> u32 {
    use ndarray::simd::U64x8;
    let chunks = words.len() / 8;
    let mut acc = 0u64;
    for i in 0..chunks {
        acc += U64x8::from_slice(&words[i * 8..i * 8 + 8])
            .popcnt()
            .to_array()
            .iter()
            .sum::<u64>();
    }
    for &w in &words[chunks * 8..] {
        acc += u64::from(w.count_ones());
    }
    acc as u32
}

/// Population count of `a AND b`, via `ndarray::simd::U64x8`.
#[cfg(feature = "ndarray-simd")]
#[must_use]
pub fn and_popcount(a: &[u64], b: &[u64]) -> u32 {
    use ndarray::simd::U64x8;
    let chunks = a.len() / 8;
    let mut acc = 0u64;
    for i in 0..chunks {
        let va = U64x8::from_slice(&a[i * 8..i * 8 + 8]);
        let vb = U64x8::from_slice(&b[i * 8..i * 8 + 8]);
        acc += (va & vb).popcnt().to_array().iter().sum::<u64>();
    }
    for i in (chunks * 8)..a.len() {
        acc += u64::from((a[i] & b[i]).count_ones());
    }
    acc as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn popcount_counts_set_bits() {
        assert_eq!(popcount(&[0u64]), 0);
        assert_eq!(popcount(&[u64::MAX]), 64);
        assert_eq!(popcount(&[0b1011, 0b11]), 5); // 3 + 2 set bits
                                                  // exercise the 8-word vector chunk + tail
        assert_eq!(popcount(&[u64::MAX; 9]), 64 * 9);
    }

    #[test]
    fn and_popcount_is_conjunction_count() {
        assert_eq!(and_popcount(&[0b1100], &[0b1010]), 1); // only bit 3 shared
        assert_eq!(and_popcount(&[u64::MAX; 9], &[u64::MAX; 9]), 64 * 9);
        assert_eq!(and_popcount(&[u64::MAX; 9], &[0u64; 9]), 0);
    }
}
