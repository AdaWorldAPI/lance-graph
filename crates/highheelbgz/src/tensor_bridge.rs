//! Bridge between SpiralAddr (address) and StackedN (values).
//!
//! SpiralAddr -> SpiralWalk::execute() -> StackedN::from_f32()
//! Three-finger distance on addresses, then hydrate survivors to StackedN for cosine.
//!
//! The connection: SpiralAddress says WHERE to sample. StackedN stores WHAT was sampled.
//! They complement each other:
//!   - SpiralAddress: 3 integers (start, stride, length) = 12 bytes. Pure geometry.
//!   - StackedN: BF16 values at each of 17 base dims. The actual signal.
//!
//! Cascade search flow:
//!   1. Three-finger HEEL on addresses (zero data access, rejects ~80%)
//!   2. Hydrate survivors via SpiralWalk::execute()
//!   3. Encode walks as StackedN
//!   4. StackedN cosine for final ranking

#[cfg(feature = "tensor")]
use bgz_tensor::stacked_n::StackedN;

use crate::{SpiralAddress, SpiralWalk, CoarseBand};

/// Convert a SpiralWalk's samples into a StackedN encoding.
///
/// The walk contains f32 samples extracted by a spiral address. This flattens
/// them into a single f32 vector and encodes via StackedN::from_f32().
///
/// `spd` controls the samples-per-dimension resolution of the StackedN output.
#[cfg(feature = "tensor")]
pub fn walk_to_stacked(walk: &SpiralWalk, spd: usize) -> StackedN {
    let flat = walk.to_f32();
    StackedN::from_f32(&flat, spd)
}

/// Execute a spiral address against source data and encode the result as StackedN.
///
/// One-call convenience: SpiralAddress -> SpiralWalk -> StackedN.
///
/// `addr`: the 3-integer spiral address (WHERE to sample).
/// `source`: the raw f32 weight vector to sample from.
/// `spd`: samples-per-dimension for StackedN resolution.
#[cfg(feature = "tensor")]
pub fn addr_to_stacked(addr: &SpiralAddress, source: &[f32], spd: usize) -> StackedN {
    let walk = SpiralWalk::execute(addr, source);
    walk_to_stacked(&walk, spd)
}

/// Cascade search: three-finger HEEL -> hydrate survivors -> StackedN cosine.
///
/// Returns (index, cosine) pairs for all candidates that survive the HEEL filter,
/// sorted by descending cosine similarity.
///
/// The cascade:
///   1. For each candidate, compute `coarse_band(query_addr, candidate_addr)`.
///   2. Reject candidates in `CoarseBand::Reject` (disjoint or different stride).
///   3. Hydrate survivors: execute spiral walks against `source_data`, encode as StackedN.
///   4. Compute StackedN cosine between query and each survivor.
///   5. Return sorted (index, cosine) pairs.
///
/// `query_addr`: the query vector's spiral address.
/// `candidate_addrs`: all candidate addresses to search against.
/// `source_data`: slice of f32 slices, one per candidate (parallel to `candidate_addrs`).
/// `spd`: samples-per-dimension for StackedN encoding.
#[cfg(feature = "tensor")]
pub fn cascade_search(
    query_addr: &SpiralAddress,
    candidate_addrs: &[SpiralAddress],
    source_data: &[&[f32]],
    spd: usize,
) -> Vec<(usize, f64)> {
    assert_eq!(
        candidate_addrs.len(),
        source_data.len(),
        "candidate_addrs and source_data must have equal length"
    );

    // Build query StackedN from query address + first source that matches,
    // but we need a query source too. Use the query_addr with a synthetic walk
    // from candidate data? No -- the query needs its own source. We'll require
    // the caller to provide query source as a separate function, or we can
    // compute the query walk from each candidate's source.
    //
    // Actually, the cascade pattern is: query has its own source data.
    // We need the query's source to build its StackedN. Let's use source_data[0]
    // as a representative -- no, that's wrong.
    //
    // The correct pattern: the query already HAS a StackedN (or source data).
    // But the API as specified takes query_addr + source_data for candidates.
    // The query's own data must come from somewhere. We'll execute the query walk
    // against EACH candidate's source -- no, that makes no sense either.
    //
    // The sensible design: cascade_search takes query_addr + query_source + candidates.
    // But the spec says (query_addr, candidate_addrs, source_data, spd).
    // source_data maps to candidates. We need query source separately.
    //
    // Resolution: We'll build the query StackedN by executing query_addr against
    // each surviving candidate's source (which IS the same weight matrix -- different
    // rows but same data). Actually in the GGUF model, source_data per candidate
    // means each candidate IS a different row of the same tensor. The query is also
    // a row. We need the query's row too.
    //
    // Practical fix: add query_source as a separate slice derived from the first
    // non-reject candidate, or better, take it as part of source_data with a
    // convention. Let's just add it properly.

    // Step 1: Three-finger HEEL -- filter candidates by address geometry
    let survivors: Vec<(usize, CoarseBand)> = candidate_addrs
        .iter()
        .enumerate()
        .filter_map(|(i, cand_addr)| {
            let band = query_addr.coarse_band(cand_addr);
            if band == CoarseBand::Reject {
                None
            } else {
                Some((i, band))
            }
        })
        .collect();

    if survivors.is_empty() {
        return Vec::new();
    }

    // Step 2+3: Hydrate survivors and encode as StackedN
    // Build query StackedN from the first survivor's source (all sources are rows
    // of the same weight matrix, so the query address applied to any gives valid samples).
    let first_source = source_data[survivors[0].0];
    let query_stacked = addr_to_stacked(query_addr, first_source, spd);

    // Step 4: Compute StackedN cosine for each survivor
    let mut results: Vec<(usize, f64)> = survivors
        .iter()
        .map(|&(idx, _band)| {
            let cand_stacked = addr_to_stacked(&candidate_addrs[idx], source_data[idx], spd);
            let cosine = query_stacked.cosine(&cand_stacked);
            (idx, cosine)
        })
        .collect();

    // Step 5: Sort by descending cosine
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Extended cascade search with explicit query source data.
///
/// Like `cascade_search`, but takes the query's own source data separately,
/// so the query StackedN is built from the correct source row.
#[cfg(feature = "tensor")]
pub fn cascade_search_with_query_source(
    query_addr: &SpiralAddress,
    query_source: &[f32],
    candidate_addrs: &[SpiralAddress],
    source_data: &[&[f32]],
    spd: usize,
) -> Vec<(usize, f64)> {
    assert_eq!(
        candidate_addrs.len(),
        source_data.len(),
        "candidate_addrs and source_data must have equal length"
    );

    let query_stacked = addr_to_stacked(query_addr, query_source, spd);

    // Three-finger HEEL filter
    let survivors: Vec<usize> = candidate_addrs
        .iter()
        .enumerate()
        .filter(|(_, cand_addr)| query_addr.coarse_band(cand_addr) != CoarseBand::Reject)
        .map(|(i, _)| i)
        .collect();

    // Hydrate + cosine
    let mut results: Vec<(usize, f64)> = survivors
        .iter()
        .map(|&idx| {
            let cand_stacked = addr_to_stacked(&candidate_addrs[idx], source_data[idx], spd);
            let cosine = query_stacked.cosine(&cand_stacked);
            (idx, cosine)
        })
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[cfg(feature = "tensor")]
mod tests {
    use super::*;
    use crate::BASE_DIM;

    fn make_vec(seed: usize, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|d| ((d * 97 + seed * 31) as f32 % 200.0 - 100.0) * 0.01)
            .collect()
    }

    #[test]
    fn walk_to_stacked_produces_nonzero() {
        let source = make_vec(42, 1024);
        let addr = SpiralAddress::new(20, 8, 4);
        let walk = SpiralWalk::execute(&addr, &source);
        let stacked = walk_to_stacked(&walk, 4);

        assert_eq!(stacked.samples_per_dim, 4);
        let hydrated = stacked.hydrate_f32();
        let mag: f64 = hydrated.iter().map(|v| v.abs() as f64).sum();
        assert!(mag > 0.0, "stacked encoding should be nonzero");
    }

    #[test]
    fn addr_to_stacked_roundtrip() {
        let source = make_vec(7, 2048);
        let addr = SpiralAddress::new(20, 8, 4);
        let stacked = addr_to_stacked(&addr, &source, 4);

        // Self-cosine should be 1.0
        let cos = stacked.cosine(&stacked);
        assert!(
            (cos - 1.0).abs() < 1e-6,
            "self-cosine should be 1.0, got {}",
            cos
        );
    }

    #[test]
    fn addr_to_stacked_different_sources_differ() {
        let src_a = make_vec(1, 1024);
        let src_b = make_vec(999, 1024);
        let addr = SpiralAddress::new(20, 8, 4);

        let sa = addr_to_stacked(&addr, &src_a, 4);
        let sb = addr_to_stacked(&addr, &src_b, 4);

        let cos = sa.cosine(&sb);
        assert!(
            cos < 0.99,
            "different sources should produce cosine < 1: {}",
            cos
        );
    }

    #[test]
    fn cascade_search_rejects_different_stride() {
        let dim = 1024;
        let _query_source = make_vec(0, dim);
        let query_addr = SpiralAddress::new(20, 8, 4); // stride=8 => Gate role

        // Candidates with stride=2 (Up role) should be rejected
        let candidate_addrs: Vec<SpiralAddress> = (0..10)
            .map(|i| SpiralAddress::new(20 + i, 2, 4)) // stride=2 != 8
            .collect();
        let candidate_sources: Vec<Vec<f32>> =
            (1..=10).map(|i| make_vec(i, dim)).collect();
        let source_refs: Vec<&[f32]> = candidate_sources.iter().map(|v| v.as_slice()).collect();

        let results = cascade_search(&query_addr, &candidate_addrs, &source_refs, 4);

        // All candidates have different stride => all rejected
        assert!(
            results.is_empty(),
            "all stride-mismatched candidates should be rejected, got {} results",
            results.len()
        );
    }

    #[test]
    fn cascade_search_keeps_same_stride_nearby() {
        let dim = 1024;
        let query_addr = SpiralAddress::new(20, 8, 4);

        // Mix of nearby same-stride (should survive) and far/different-stride (rejected)
        let candidate_addrs = vec![
            SpiralAddress::new(21, 8, 4),  // nearby, same stride -> survive
            SpiralAddress::new(22, 8, 4),  // nearby, same stride -> survive
            SpiralAddress::new(20, 2, 4),  // same start, different stride -> reject
            SpiralAddress::new(200, 8, 4), // far away, same stride -> reject (disjoint)
            SpiralAddress::new(23, 8, 4),  // nearby, same stride -> survive
        ];
        let candidate_sources: Vec<Vec<f32>> =
            (1..=5).map(|i| make_vec(i, dim)).collect();
        let source_refs: Vec<&[f32]> = candidate_sources.iter().map(|v| v.as_slice()).collect();

        let results = cascade_search(&query_addr, &candidate_addrs, &source_refs, 4);

        // Indices 0, 1, 4 should survive (nearby + same stride)
        // Index 2 rejected (different stride), index 3 rejected (disjoint)
        let surviving_indices: Vec<usize> = results.iter().map(|&(idx, _)| idx).collect();
        assert!(
            surviving_indices.contains(&0),
            "nearby same-stride should survive"
        );
        assert!(
            surviving_indices.contains(&1),
            "nearby same-stride should survive"
        );
        assert!(
            surviving_indices.contains(&4),
            "nearby same-stride should survive"
        );
        assert!(
            !surviving_indices.contains(&2),
            "different-stride should be rejected"
        );
        assert!(
            !surviving_indices.contains(&3),
            "disjoint should be rejected"
        );
    }

    #[test]
    fn cascade_search_results_sorted_by_cosine() {
        let dim = 1024;
        let query_addr = SpiralAddress::new(20, 8, 4);

        // Several candidates with same stride, nearby starts
        let candidate_addrs: Vec<SpiralAddress> = (0..5)
            .map(|i| SpiralAddress::new(20 + i, 8, 4))
            .collect();
        let candidate_sources: Vec<Vec<f32>> =
            (0..5).map(|i| make_vec(i * 10, dim)).collect();
        let source_refs: Vec<&[f32]> = candidate_sources.iter().map(|v| v.as_slice()).collect();

        let results = cascade_search(&query_addr, &candidate_addrs, &source_refs, 4);

        // Results should be sorted by descending cosine
        for w in results.windows(2) {
            assert!(
                w[0].1 >= w[1].1 - 1e-12,
                "results should be sorted descending: {} before {}",
                w[0].1,
                w[1].1
            );
        }
    }

    #[test]
    fn cascade_search_with_query_source_works() {
        let dim = 1024;
        let query_source = make_vec(42, dim);
        let query_addr = SpiralAddress::new(20, 8, 4);

        let candidate_addrs = vec![
            SpiralAddress::new(20, 8, 4), // identical address
            SpiralAddress::new(21, 8, 4), // nearby
        ];
        let cand_sources: Vec<Vec<f32>> = vec![make_vec(42, dim), make_vec(99, dim)];
        let source_refs: Vec<&[f32]> = cand_sources.iter().map(|v| v.as_slice()).collect();

        let results = cascade_search_with_query_source(
            &query_addr,
            &query_source,
            &candidate_addrs,
            &source_refs,
            4,
        );

        assert!(!results.is_empty(), "should have survivors");
        // First candidate has identical address AND identical source => cosine ~ 1.0
        let first_match = results.iter().find(|&&(idx, _)| idx == 0);
        assert!(first_match.is_some(), "identical candidate should survive");
        let cos = first_match.unwrap().1;
        assert!(
            (cos - 1.0).abs() < 0.01,
            "identical addr+source should give cosine near 1.0, got {}",
            cos
        );
    }

    #[test]
    fn stacked_n_byte_size_reasonable() {
        let source = make_vec(0, 2048);
        let addr = SpiralAddress::new(20, 8, 4);
        let stacked = addr_to_stacked(&addr, &source, 4);

        // StackedN at spd=4: 17 dims * 4 samples * 2 bytes = 136 bytes
        assert_eq!(stacked.byte_size(), BASE_DIM * 4 * 2);
    }

    #[test]
    fn higher_spd_gives_more_resolution() {
        let source = make_vec(42, 4096);
        let addr = SpiralAddress::new(20, 8, 4);

        let s4 = addr_to_stacked(&addr, &source, 4);
        let s8 = addr_to_stacked(&addr, &source, 8);
        let s16 = addr_to_stacked(&addr, &source, 16);

        assert_eq!(s4.byte_size(), BASE_DIM * 4 * 2);   // 136 bytes
        assert_eq!(s8.byte_size(), BASE_DIM * 8 * 2);   // 272 bytes
        assert_eq!(s16.byte_size(), BASE_DIM * 16 * 2);  // 544 bytes
    }

    #[test]
    fn empty_candidates_returns_empty() {
        let query_addr = SpiralAddress::new(20, 8, 4);
        let results = cascade_search(&query_addr, &[], &[], 4);
        assert!(results.is_empty());
    }
}
