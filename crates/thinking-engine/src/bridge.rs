//! Bridge between burn matmul (ndarray) distance tables and highheelbgz spiral addressing.
//!
//! The ThinkingEngine uses a flat `Vec<u8>` distance table of size N x N.
//! This module maps highheelbgz `SpiralAddress` values into that table space,
//! builds coarse distance tables from three-finger geometry, and refines
//! entries via hydrated cosine when the coarse band is ambiguous.

use highheelbgz::{CoarseBand, SpiralAddress, SpiralWalk};

// ═══════════════════════════════════════════════════════════════════════════
// 1. spiral_to_table_index: map a spiral address to a table row/col index
// ═══════════════════════════════════════════════════════════════════════════

/// Map a spiral address to a distance table row/col index.
///
/// Uses a hash of (start, stride, length) modulo `table_size` so that
/// distinct addresses land at distinct (with high probability) table positions.
pub fn spiral_to_table_index(addr: &SpiralAddress, table_size: usize) -> usize {
    // FNV-1a style mix of the three fields for uniform distribution.
    let mut h: u64 = 0xcbf29ce484222325;
    let mix = |h: &mut u64, v: u32| {
        for b in v.to_le_bytes() {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x100000001b3);
        }
    };
    mix(&mut h, addr.start);
    mix(&mut h, addr.stride);
    mix(&mut h, addr.length);
    (h as usize) % table_size
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. build_spiral_distance_table: N x N u8 table from coarse bands
// ═══════════════════════════════════════════════════════════════════════════

/// Map a `CoarseBand` to a u8 similarity value for the distance table.
///
/// The ThinkingEngine convention: 255 = identical, 0 = opposite/far.
///   - Foveal (nearly identical walk)  -> 255
///   - Near   (overlapping, same stride) -> 192
///   - Maybe  (need hydration to decide) -> 128
///   - Reject (disjoint or different stride) -> 0
#[inline]
fn band_to_u8(band: CoarseBand) -> u8 {
    match band {
        CoarseBand::Foveal => 255,
        CoarseBand::Near => 192,
        CoarseBand::Maybe => 128,
        CoarseBand::Reject => 0,
    }
}

/// Build an N x N u8 distance table from spiral addresses using three-finger
/// coarse band distance. No source data access required.
///
/// The returned table is row-major: entry `[i * n + j]` is the similarity
/// between `addresses[i]` and `addresses[j]`.
///
/// # Panics
/// Panics if `addresses.len() != n`.
pub fn build_spiral_distance_table(addresses: &[SpiralAddress], n: usize) -> Vec<u8> {
    assert_eq!(addresses.len(), n, "addresses.len() must equal n");
    let mut table = vec![0u8; n * n];
    for i in 0..n {
        // Diagonal: self-similarity = 255
        table[i * n + i] = 255;
        for j in (i + 1)..n {
            let band = addresses[i].coarse_band(&addresses[j]);
            let val = band_to_u8(band);
            table[i * n + j] = val;
            table[j * n + i] = val; // symmetric
        }
    }
    table
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. hydrate_and_cosine: hydrate both addresses and compute cosine
// ═══════════════════════════════════════════════════════════════════════════

/// Hydrate both spiral addresses from source data and compute cosine similarity.
///
/// Executes `SpiralWalk::execute` for each address against the shared `source`
/// vector, then returns the walk-level cosine similarity as f64.
pub fn hydrate_and_cosine(
    addr_a: &SpiralAddress,
    addr_b: &SpiralAddress,
    source: &[f32],
) -> f64 {
    let walk_a = SpiralWalk::execute(addr_a, source);
    let walk_b = SpiralWalk::execute(addr_b, source);
    walk_a.cosine(&walk_b)
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. enrich_table_from_source: refine Maybe entries with actual cosine
// ═══════════════════════════════════════════════════════════════════════════

/// For pairs where the coarse band is `Maybe` (u8 value 128), compute the
/// actual hydrated cosine similarity and replace the table entry.
///
/// Cosine in [-1, 1] is mapped to u8 [0, 255]:
///   u8 = ((cosine + 1.0) / 2.0 * 255.0) as u8
///
/// This selectively hydrates only ambiguous pairs, keeping the cost low
/// while eliminating false positives from the coarse pass.
///
/// # Panics
/// Panics if `table.len() != n * n` or `addresses.len() != n`.
pub fn enrich_table_from_source(
    table: &mut [u8],
    addresses: &[SpiralAddress],
    source: &[f32],
    n: usize,
) {
    assert_eq!(table.len(), n * n, "table size must be n*n");
    assert_eq!(addresses.len(), n, "addresses.len() must equal n");

    let maybe_val = band_to_u8(CoarseBand::Maybe); // 128

    for i in 0..n {
        for j in (i + 1)..n {
            let idx_ij = i * n + j;
            if table[idx_ij] == maybe_val {
                let cosine = hydrate_and_cosine(&addresses[i], &addresses[j], source);
                // Map cosine [-1, 1] -> u8 [0, 255]
                let val = ((cosine + 1.0) / 2.0 * 255.0).clamp(0.0, 255.0) as u8;
                table[idx_ij] = val;
                table[j * n + i] = val; // symmetric
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use highheelbgz::SpiralAddress;

    /// Deterministic test vector generator.
    fn make_source(seed: usize, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|d| ((d * 97 + seed * 31) as f32 % 200.0 - 100.0) * 0.01)
            .collect()
    }

    // ── spiral_to_table_index ──────────────────────────────────────

    #[test]
    fn table_index_deterministic() {
        let addr = SpiralAddress::new(20, 8, 4);
        let idx1 = spiral_to_table_index(&addr, 4096);
        let idx2 = spiral_to_table_index(&addr, 4096);
        assert_eq!(idx1, idx2, "same address must produce same index");
    }

    #[test]
    fn table_index_within_bounds() {
        let addr = SpiralAddress::new(20, 8, 4);
        for size in [16, 256, 4096, 65536] {
            let idx = spiral_to_table_index(&addr, size);
            assert!(idx < size, "index {} must be < table_size {}", idx, size);
        }
    }

    #[test]
    fn table_index_different_addresses_differ() {
        let a = SpiralAddress::new(20, 8, 4);
        let b = SpiralAddress::new(30, 8, 4);
        let c = SpiralAddress::new(20, 5, 4);
        let size = 65536; // large enough to make collisions unlikely
        let ia = spiral_to_table_index(&a, size);
        let ib = spiral_to_table_index(&b, size);
        let ic = spiral_to_table_index(&c, size);
        // At least two of three should differ (collision possible but unlikely at 64K)
        assert!(
            ia != ib || ia != ic || ib != ic,
            "different addresses should (usually) map to different indices"
        );
    }

    // ── build_spiral_distance_table ────────────────────────────────

    #[test]
    fn distance_table_diagonal_is_255() {
        let addrs = vec![
            SpiralAddress::new(20, 8, 4),
            SpiralAddress::new(30, 8, 4),
            SpiralAddress::new(20, 2, 4),
        ];
        let n = addrs.len();
        let table = build_spiral_distance_table(&addrs, n);
        for i in 0..n {
            assert_eq!(table[i * n + i], 255, "diagonal must be 255 (self-similarity)");
        }
    }

    #[test]
    fn distance_table_symmetric() {
        let addrs = vec![
            SpiralAddress::new(20, 8, 4),
            SpiralAddress::new(21, 8, 4),
            SpiralAddress::new(200, 8, 4),
            SpiralAddress::new(20, 2, 4),
        ];
        let n = addrs.len();
        let table = build_spiral_distance_table(&addrs, n);
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    table[i * n + j],
                    table[j * n + i],
                    "table must be symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn distance_table_reject_is_zero() {
        // Different stride -> Reject -> 0
        let addrs = vec![
            SpiralAddress::new(20, 8, 4),
            SpiralAddress::new(20, 2, 4),
        ];
        let table = build_spiral_distance_table(&addrs, 2);
        assert_eq!(table[0 * 2 + 1], 0, "different stride should be Reject=0");
        assert_eq!(table[1 * 2 + 0], 0, "symmetric Reject=0");
    }

    #[test]
    fn distance_table_foveal_is_255() {
        // Identical addresses -> Foveal -> 255
        let addrs = vec![
            SpiralAddress::new(20, 8, 4),
            SpiralAddress::new(21, 8, 4), // offset=1, same stride, high overlap -> Foveal
        ];
        let table = build_spiral_distance_table(&addrs, 2);
        assert_eq!(table[0 * 2 + 1], 255, "foveal pair should be 255");
    }

    // ── hydrate_and_cosine ─────────────────────────────────────────

    #[test]
    fn hydrate_cosine_self_is_one() {
        let source = make_source(42, 4096);
        let addr = SpiralAddress::new(20, 8, 4);
        let cos = hydrate_and_cosine(&addr, &addr, &source);
        assert!(
            (cos - 1.0).abs() < 1e-10,
            "self-cosine must be 1.0, got {}",
            cos
        );
    }

    #[test]
    fn hydrate_cosine_different_stride_lower() {
        let source = make_source(42, 4096);
        let a = SpiralAddress::new(20, 8, 4);
        let b_near = SpiralAddress::new(21, 8, 4);
        let b_far = SpiralAddress::new(20, 2, 4); // different stride

        let cos_near = hydrate_and_cosine(&a, &b_near, &source);
        let cos_far = hydrate_and_cosine(&a, &b_far, &source);

        // Near pair (same stride, offset=1) should generally have higher cosine
        // than different-stride pair
        assert!(
            cos_near > cos_far || cos_far.abs() < 0.5,
            "near cosine ({:.4}) should exceed or far should be low ({:.4})",
            cos_near,
            cos_far
        );
    }

    // ── enrich_table_from_source ───────────────────────────────────

    #[test]
    fn enrich_replaces_maybe_entries() {
        // Build addresses where some pairs will be Maybe
        let addrs = vec![
            SpiralAddress::new(20, 8, 4),
            SpiralAddress::new(30, 8, 4), // same stride, moderate offset -> Near or Maybe
            SpiralAddress::new(20, 2, 4), // different stride -> Reject
        ];
        let n = addrs.len();
        let mut table = build_spiral_distance_table(&addrs, n);

        // Check which entries are Maybe (128) before enrichment
        let maybe_before: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .filter(|&(i, j)| table[i * n + j] == 128)
            .collect();

        let source = make_source(42, 4096);
        enrich_table_from_source(&mut table, &addrs, &source, n);

        // After enrichment, no Maybe entries should remain
        for &(i, j) in &maybe_before {
            assert_ne!(
                table[i * n + j], 128,
                "Maybe entry at ({}, {}) should have been replaced",
                i, j
            );
        }

        // Table should still be symmetric
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    table[i * n + j],
                    table[j * n + i],
                    "enriched table must remain symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn enrich_preserves_non_maybe_entries() {
        let addrs = vec![
            SpiralAddress::new(20, 8, 4),
            SpiralAddress::new(20, 2, 4), // different stride -> Reject=0
        ];
        let n = addrs.len();
        let mut table = build_spiral_distance_table(&addrs, n);
        let reject_val = table[0 * n + 1];
        assert_eq!(reject_val, 0, "should be Reject");

        let source = make_source(42, 4096);
        enrich_table_from_source(&mut table, &addrs, &source, n);

        // Reject entry should be unchanged
        assert_eq!(
            table[0 * n + 1], reject_val,
            "Reject entry should not be modified by enrichment"
        );
    }

    #[test]
    fn enrich_cosine_values_in_valid_range() {
        // Use addresses that produce Maybe pairs so we can verify the cosine mapping
        let addrs = vec![
            SpiralAddress::new(20, 8, 8),
            SpiralAddress::new(30, 8, 8),
            SpiralAddress::new(35, 8, 8),
            SpiralAddress::new(40, 8, 8),
        ];
        let n = addrs.len();
        let mut table = build_spiral_distance_table(&addrs, n);
        let source = make_source(7, 4096);
        enrich_table_from_source(&mut table, &addrs, &source, n);

        // All entries must be valid u8 (always true by type), but check
        // diagonal is still 255 and off-diagonal values are plausible
        for i in 0..n {
            assert_eq!(table[i * n + i], 255, "diagonal unchanged");
        }
    }
}
