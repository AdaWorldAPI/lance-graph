// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bridge types for lance-graph <-> ndarray interop.
//!
//! ndarray's `Fingerprint2K` (`Fingerprint<256>`) is the canonical 16384-bit
//! binary vector with SIMD-dispatched hamming/popcount (VPOPCNTDQ → AVX-512BW
//! → AVX2 → scalar). This module re-exports it and provides zero-copy
//! conversion to/from lance-graph's `BitVec`.
//!
//! All SIMD dispatch lives in ndarray — no duplication here.

use super::types::{BitVec, VECTOR_WORDS};

// Re-export ndarray's canonical fingerprint types.
pub use ndarray::hpc::fingerprint::{Fingerprint, Fingerprint1K, Fingerprint2K, Fingerprint64K};

// Re-export raw SIMD-dispatched operations for callers that work on `&[u8]`.
pub use ndarray::hpc::bitwise::{
    hamming_batch_raw as dispatch_hamming_batch, hamming_distance_raw as dispatch_hamming,
    popcount_raw as dispatch_popcount,
};

// ---------------------------------------------------------------------------
// NdarrayFingerprint — thin wrapper keeping the existing API surface
// ---------------------------------------------------------------------------

/// Newtype over ndarray's `Fingerprint2K` that provides the lance-graph API.
///
/// This exists so that downstream code in blasgraph (semirings, HDR cascade)
/// can keep using `NdarrayFingerprint` without a mass rename. All hot-path
/// operations delegate to ndarray's SIMD dispatch.
#[derive(Clone, PartialEq, Eq)]
pub struct NdarrayFingerprint {
    /// Raw storage: 256 x 64-bit words = 16384 bits.
    /// Public to match ndarray's `Fingerprint { pub words }` layout.
    pub words: [u64; VECTOR_WORDS],
}

impl std::fmt::Debug for NdarrayFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NdarrayFingerprint {{ popcount: {} }}", self.popcount())
    }
}

impl Default for NdarrayFingerprint {
    fn default() -> Self {
        Self {
            words: [0u64; VECTOR_WORDS],
        }
    }
}

impl NdarrayFingerprint {
    /// Create a zeroed fingerprint.
    pub fn zero() -> Self {
        Self::default()
    }

    /// Create a fingerprint with all bits set.
    pub fn ones() -> Self {
        Self {
            words: [u64::MAX; VECTOR_WORDS],
        }
    }

    /// Construct from a word array.
    pub fn from_words(words: [u64; VECTOR_WORDS]) -> Self {
        Self { words }
    }

    /// Return a byte-level view of the fingerprint (zero-copy).
    pub fn as_bytes(&self) -> &[u8] {
        self.as_fingerprint2k().as_bytes()
    }

    /// Population count (number of set bits).
    /// Delegates to ndarray's SIMD dispatch.
    #[inline]
    pub fn popcount(&self) -> u64 {
        self.as_fingerprint2k().popcount() as u64
    }

    /// Hamming distance to another fingerprint.
    /// Delegates to ndarray's SIMD dispatch.
    #[inline]
    pub fn hamming_distance(&self, other: &NdarrayFingerprint) -> u64 {
        self.as_fingerprint2k()
            .hamming_distance(other.as_fingerprint2k()) as u64
    }

    /// View as ndarray Fingerprint2K (zero-copy reinterpret).
    fn as_fingerprint2k(&self) -> &Fingerprint2K {
        // SAFETY: NdarrayFingerprint has layout `[u64; 256]` which is identical
        // to Fingerprint2K's `words: [u64; 256]`. Same size, same alignment.
        unsafe { &*(self as *const NdarrayFingerprint as *const Fingerprint2K) }
    }

    /// Convert to an owned Fingerprint2K.
    pub fn to_fingerprint2k(&self) -> Fingerprint2K {
        Fingerprint2K::from_words(self.words)
    }

    /// Consume and return the inner ndarray Fingerprint2K.
    pub fn into_fingerprint2k(self) -> Fingerprint2K {
        Fingerprint2K::from_words(self.words)
    }
}

// ---------------------------------------------------------------------------
// From conversions: BitVec <-> NdarrayFingerprint
// ---------------------------------------------------------------------------

impl From<&BitVec> for NdarrayFingerprint {
    fn from(bv: &BitVec) -> Self {
        NdarrayFingerprint {
            words: *bv.words(),
        }
    }
}

impl From<&NdarrayFingerprint> for BitVec {
    fn from(fp: &NdarrayFingerprint) -> Self {
        BitVec::from_words(&fp.words)
    }
}

// ---------------------------------------------------------------------------
// From conversions: Fingerprint2K <-> NdarrayFingerprint
// ---------------------------------------------------------------------------

impl From<Fingerprint2K> for NdarrayFingerprint {
    fn from(fp: Fingerprint2K) -> Self {
        Self { words: fp.words }
    }
}

impl From<NdarrayFingerprint> for Fingerprint2K {
    fn from(fp: NdarrayFingerprint) -> Self {
        Fingerprint2K::from_words(fp.words)
    }
}

impl From<&Fingerprint2K> for NdarrayFingerprint {
    fn from(fp: &Fingerprint2K) -> Self {
        Self { words: fp.words }
    }
}

// ---------------------------------------------------------------------------
// From conversions: ndarray::Array1<u64> <-> NdarrayFingerprint
// ---------------------------------------------------------------------------

impl From<&ndarray::Array1<u64>> for NdarrayFingerprint {
    fn from(arr: &ndarray::Array1<u64>) -> Self {
        assert_eq!(
            arr.len(),
            VECTOR_WORDS,
            "Array1 must have exactly {} elements",
            VECTOR_WORDS
        );
        let mut words = [0u64; VECTOR_WORDS];
        for (i, &w) in arr.iter().enumerate() {
            words[i] = w;
        }
        Self { words }
    }
}

impl From<&NdarrayFingerprint> for ndarray::Array1<u64> {
    fn from(fp: &NdarrayFingerprint) -> Self {
        ndarray::Array1::from_vec(fp.words.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_default_is_zero() {
        let fp = NdarrayFingerprint::default();
        assert_eq!(fp.popcount(), 0);
    }

    #[test]
    fn test_fingerprint_ones() {
        let fp = NdarrayFingerprint::ones();
        assert_eq!(fp.popcount(), (VECTOR_WORDS * 64) as u64);
    }

    #[test]
    fn test_fingerprint_from_bitvec_roundtrip() {
        let bv = BitVec::random(42);
        let fp = NdarrayFingerprint::from(&bv);
        let bv2 = BitVec::from(&fp);
        assert_eq!(bv, bv2);
    }

    #[test]
    fn test_fingerprint_hamming_self_is_zero() {
        let fp = NdarrayFingerprint::from(&BitVec::random(99));
        assert_eq!(fp.hamming_distance(&fp), 0);
    }

    #[test]
    fn test_fingerprint_hamming_complement() {
        let fp = NdarrayFingerprint::from(&BitVec::random(99));
        let mut inv_words = [0u64; VECTOR_WORDS];
        for i in 0..VECTOR_WORDS {
            inv_words[i] = !fp.words[i];
        }
        let inv = NdarrayFingerprint::from_words(inv_words);
        assert_eq!(fp.hamming_distance(&inv), (VECTOR_WORDS * 64) as u64);
    }

    #[test]
    fn test_dispatch_hamming_matches_bitvec() {
        let a = BitVec::random(1);
        let b = BitVec::random(2);
        let expected = a.hamming_distance(&b) as u64;
        let fa = NdarrayFingerprint::from(&a);
        let fb = NdarrayFingerprint::from(&b);
        assert_eq!(dispatch_hamming(fa.as_bytes(), fb.as_bytes()), expected);
    }

    #[test]
    fn test_dispatch_popcount_matches_bitvec() {
        let bv = BitVec::random(12345);
        let expected = bv.popcount() as u64;
        let fp = NdarrayFingerprint::from(&bv);
        assert_eq!(dispatch_popcount(fp.as_bytes()), expected);
    }

    #[test]
    fn test_fingerprint2k_roundtrip() {
        let bv = BitVec::random(42);
        let fp = NdarrayFingerprint::from(&bv);
        let f2k: Fingerprint2K = fp.clone().into();
        let fp2: NdarrayFingerprint = f2k.into();
        assert_eq!(fp, fp2);
    }

    #[test]
    fn test_ndarray_array1_roundtrip() {
        let bv = BitVec::random(42);
        let fp = NdarrayFingerprint::from(&bv);
        let arr: ndarray::Array1<u64> = ndarray::Array1::from(&fp);
        let fp2 = NdarrayFingerprint::from(&arr);
        assert_eq!(fp, fp2);
    }

    #[test]
    fn test_ndarray_hamming_via_bridge() {
        let a = BitVec::random(1);
        let b = BitVec::random(2);
        let expected = a.hamming_distance(&b) as u64;
        let fa = NdarrayFingerprint::from(&a);
        let fb = NdarrayFingerprint::from(&b);
        let arr_a: ndarray::Array1<u64> = ndarray::Array1::from(&fa);
        let arr_b: ndarray::Array1<u64> = ndarray::Array1::from(&fb);
        let fa2 = NdarrayFingerprint::from(&arr_a);
        let fb2 = NdarrayFingerprint::from(&arr_b);
        assert_eq!(fa2.hamming_distance(&fb2), expected);
    }

    #[test]
    fn test_dispatch_hamming_empty() {
        assert_eq!(dispatch_hamming(&[], &[]), 0);
    }

    #[test]
    fn test_dispatch_popcount_empty() {
        assert_eq!(dispatch_popcount(&[]), 0);
    }

    #[test]
    fn test_fingerprint_words_preserved() {
        let bv = BitVec::random(777);
        let fp = NdarrayFingerprint::from(&bv);
        assert_eq!(&fp.words, bv.words());
    }
}
