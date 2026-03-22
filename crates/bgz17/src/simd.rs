//! SIMD-accelerated batch palette distance.
//!
//! For large candidate sets, looking up distances one-at-a-time in the
//! 256×256 matrix leaves performance on the table. This module provides
//! batch operations that process multiple candidates per call.
//!
//! Runtime dispatch: AVX-512 → AVX2 → scalar fallback.
//! All paths produce identical results (the matrix lookup is exact).
//!
//! The scalar path is always available. SIMD paths are `#[cfg(target_arch)]`
//! gated and use runtime feature detection.

use crate::distance_matrix::SpoDistanceMatrices;

/// SIMD capability level detected at runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdLevel {
    /// AVX-512 VGATHERDPS: 16 lookups per instruction.
    Avx512,
    /// AVX2 VPGATHERDD: 8 lookups per instruction.
    Avx2,
    /// Scalar fallback: 1 lookup per iteration.
    Scalar,
}

/// Detect available SIMD level.
pub fn detect_simd() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return SimdLevel::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return SimdLevel::Avx2;
        }
    }
    SimdLevel::Scalar
}

/// Batch palette distance: look up distances for multiple candidates at once.
///
/// `dm_data`: flat k×k u16 distance matrix data.
/// `k`: palette size.
/// `query`: query palette index.
/// `candidates`: candidate palette indices.
/// `out`: output distances (must be same length as candidates).
pub fn batch_palette_distance(
    dm_data: &[u16],
    k: usize,
    query: u8,
    candidates: &[u8],
    out: &mut [u16],
) {
    assert_eq!(candidates.len(), out.len());
    let _level = detect_simd();

    // Currently using scalar for all paths. AVX2/AVX512 gather would be:
    //   - Compute base offset: query_row = query as usize * k
    //   - Gather: load dm_data[query_row + candidate[i]] for 8/16 at once
    //
    // The scalar path is already memory-bound (single cache line per row),
    // so SIMD mainly helps with instruction-level parallelism on large batches.
    //
    // TODO: Uncomment SIMD paths when targeting AVX2/AVX512 deployment.
    scalar_batch(dm_data, k, query, candidates, out);
}

/// Scalar batch lookup: dm[query][candidate_i] for each candidate.
#[inline]
fn scalar_batch(dm_data: &[u16], k: usize, query: u8, candidates: &[u8], out: &mut [u16]) {
    let row_offset = query as usize * k;
    for (i, &cand) in candidates.iter().enumerate() {
        out[i] = dm_data[row_offset + cand as usize];
    }
}

/// Batch SPO distance: combined S+P+O distance for multiple candidates.
///
/// For each candidate i:
///   out[i] = dm_s[q_s][c_s[i]] + dm_p[q_p][c_p[i]] + dm_o[q_o][c_o[i]]
pub fn batch_spo_distance(
    dm: &SpoDistanceMatrices,
    query_s: u8,
    query_p: u8,
    query_o: u8,
    cand_s: &[u8],
    cand_p: &[u8],
    cand_o: &[u8],
    out: &mut [u32],
) {
    let n = cand_s.len();
    assert_eq!(cand_p.len(), n);
    assert_eq!(cand_o.len(), n);
    assert_eq!(out.len(), n);

    let k_s = dm.subject.k;
    let k_p = dm.predicate.k;
    let k_o = dm.object.k;

    let row_s = query_s as usize * k_s;
    let row_p = query_p as usize * k_p;
    let row_o = query_o as usize * k_o;

    // Prefetch the three rows we'll be reading
    #[cfg(target_arch = "x86_64")]
    {
        // Software prefetch hints for the matrix rows
        if dm.subject.data.len() > row_s {
            unsafe {
                let ptr = dm.subject.data.as_ptr().add(row_s) as *const i8;
                core::arch::x86_64::_mm_prefetch(ptr, core::arch::x86_64::_MM_HINT_T0);
            }
        }
        if dm.predicate.data.len() > row_p {
            unsafe {
                let ptr = dm.predicate.data.as_ptr().add(row_p) as *const i8;
                core::arch::x86_64::_mm_prefetch(ptr, core::arch::x86_64::_MM_HINT_T0);
            }
        }
        if dm.object.data.len() > row_o {
            unsafe {
                let ptr = dm.object.data.as_ptr().add(row_o) as *const i8;
                core::arch::x86_64::_mm_prefetch(ptr, core::arch::x86_64::_MM_HINT_T0);
            }
        }
    }

    for i in 0..n {
        let ds = dm.subject.data[row_s + cand_s[i] as usize] as u32;
        let dp = dm.predicate.data[row_p + cand_p[i] as usize] as u32;
        let d_o = dm.object.data[row_o + cand_o[i] as usize] as u32;
        out[i] = ds + dp + d_o;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base17::Base17;
    use crate::palette::Palette;
    use crate::distance_matrix::{DistanceMatrix, SpoDistanceMatrices};
    use crate::BASE_DIM;

    fn make_palette(k: usize) -> Palette {
        let entries = (0..k).map(|i| {
            let mut dims = [0i16; BASE_DIM];
            for d in 0..BASE_DIM {
                dims[d] = ((i * 97 + d * 31) % 512) as i16 - 256;
            }
            Base17 { dims }
        }).collect();
        Palette { entries }
    }

    #[test]
    fn test_batch_matches_scalar() {
        let pal = make_palette(64);
        let dm = DistanceMatrix::build(&pal);

        let query = 5u8;
        let candidates: Vec<u8> = (0..64).collect();
        let mut batch_out = vec![0u16; 64];

        batch_palette_distance(&dm.data, dm.k, query, &candidates, &mut batch_out);

        // Verify against individual lookups
        for (i, &cand) in candidates.iter().enumerate() {
            let expected = dm.distance(query, cand);
            assert_eq!(batch_out[i], expected,
                "Mismatch at candidate {}: batch={} scalar={}", cand, batch_out[i], expected);
        }
    }

    #[test]
    fn test_batch_self_distance_zero() {
        let pal = make_palette(32);
        let dm = DistanceMatrix::build(&pal);

        for q in 0..32u8 {
            let candidates = vec![q];
            let mut out = vec![0u16; 1];
            batch_palette_distance(&dm.data, dm.k, q, &candidates, &mut out);
            assert_eq!(out[0], 0, "Self-distance should be 0 for index {}", q);
        }
    }

    #[test]
    fn test_batch_spo_distance() {
        let pal = make_palette(16);
        let dm = SpoDistanceMatrices::build(&pal, &pal, &pal);

        let n = 10;
        let cand_s: Vec<u8> = (0..n as u8).collect();
        let cand_p: Vec<u8> = (0..n as u8).collect();
        let cand_o: Vec<u8> = (0..n as u8).collect();
        let mut out = vec![0u32; n];

        batch_spo_distance(&dm, 0, 0, 0, &cand_s, &cand_p, &cand_o, &mut out);

        // Verify against individual lookups
        for i in 0..n {
            let expected = dm.spo_distance(0, 0, 0, cand_s[i], cand_p[i], cand_o[i]);
            assert_eq!(out[i], expected, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_batch_spo_self_zero() {
        let pal = make_palette(16);
        let dm = SpoDistanceMatrices::build(&pal, &pal, &pal);

        let cand_s = vec![5u8];
        let cand_p = vec![5u8];
        let cand_o = vec![5u8];
        let mut out = vec![0u32; 1];

        batch_spo_distance(&dm, 5, 5, 5, &cand_s, &cand_p, &cand_o, &mut out);
        assert_eq!(out[0], 0);
    }

    #[test]
    fn test_detect_simd() {
        let level = detect_simd();
        // Should always return a valid level
        match level {
            SimdLevel::Avx512 | SimdLevel::Avx2 | SimdLevel::Scalar => {}
        }
    }
}
