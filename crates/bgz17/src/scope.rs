//! Scope storage: the bgz17 layout for neighborhoods.lance.
//!
//! A scope is 10,000 edges. The storage layout is:
//!
//! ```text
//! neighborhoods.lance columns (per scope):
//!   scent:      [u8; 10000]          =  10 KB   Layer 0 (always hot)
//!   palette_s:  [u8; 10000]          =  10 KB   Layer 1 indices
//!   palette_p:  [u8; 10000]          =  10 KB
//!   palette_o:  [u8; 10000]          =  10 KB
//!
//! scopes.lance (per scope, amortized):
//!   codebook_s: [i16[17]; k]         ≤  8.7 KB  Palette codebook
//!   codebook_p: [i16[17]; k]         ≤  8.7 KB
//!   codebook_o: [i16[17]; k]         ≤  8.7 KB
//!   dist_s:     [u16; k×k]           ≤ 128 KB   Distance matrix
//!   dist_p:     [u16; k×k]           ≤ 128 KB
//!   dist_o:     [u16; k×k]           ≤ 128 KB
//!
//! Total: ~40 KB edges + ~410 KB codebooks = ~450 KB per scope
//! vs current: 10 KB scent + 480 MB full planes = 480 MB
//! Compression: ~1,067:1
//! ```

use crate::base17::{Base17, SpoBase17};
use crate::palette::{Palette, PaletteEdge};
use crate::distance_matrix::{DistanceMatrix, SpoDistanceMatrices};
use crate::layered::LayeredScope;

/// Complete bgz17 scope: everything needed for layered search.
pub struct Bgz17Scope {
    pub scope_id: u64,
    /// Per-plane palettes.
    pub palette_s: Palette,
    pub palette_p: Palette,
    pub palette_o: Palette,
    /// Precomputed distance matrices.
    pub matrices: SpoDistanceMatrices,
    /// Per-edge data.
    pub scent: Vec<u8>,
    pub palette_indices: Vec<PaletteEdge>,
    pub base_patterns: Vec<SpoBase17>,
    /// Edge count.
    pub edge_count: usize,
}

impl Bgz17Scope {
    /// Build a bgz17 scope from raw accumulator planes.
    ///
    /// `planes`: Vec of (subject, predicate, object) as i8[16384] each.
    /// `k`: palette size (max 256).
    pub fn build(
        scope_id: u64,
        planes: &[(Vec<i8>, Vec<i8>, Vec<i8>)],
        k: usize,
    ) -> Self {
        let edge_count = planes.len();

        // Step 1: Encode all planes to Base17
        let base_patterns: Vec<SpoBase17> = planes.iter()
            .map(|(s, p, o)| SpoBase17::encode(s, p, o))
            .collect();

        // Step 2: Build palettes (k-means, 10 iterations)
        let (pal_s, pal_p, pal_o) = Palette::build_spo(&base_patterns, k, 10);

        // Step 3: Build distance matrices
        let matrices = SpoDistanceMatrices::build(&pal_s, &pal_p, &pal_o);

        // Step 4: Encode all edges to palette indices
        let palette_indices: Vec<PaletteEdge> = base_patterns.iter()
            .map(|bp| PaletteEdge {
                s_idx: pal_s.nearest(&bp.subject),
                p_idx: pal_p.nearest(&bp.predicate),
                o_idx: pal_o.nearest(&bp.object),
            })
            .collect();

        // Step 5: Compute scent bytes (pairwise with a reference — typically self)
        // For storage, scent is computed per-pair during search, not precomputed.
        // But we store a "self-scent" for each edge (comparison with scope centroid).
        let centroid = if !base_patterns.is_empty() {
            scope_centroid(&base_patterns)
        } else {
            SpoBase17 {
                subject: Base17::zero(),
                predicate: Base17::zero(),
                object: Base17::zero(),
            }
        };

        let scent: Vec<u8> = base_patterns.iter()
            .map(|bp| centroid.scent(bp))
            .collect();

        Bgz17Scope {
            scope_id,
            palette_s: pal_s,
            palette_p: pal_p,
            palette_o: pal_o,
            matrices,
            scent,
            palette_indices,
            base_patterns,
            edge_count,
        }
    }

    /// Convert to a LayeredScope for search.
    pub fn to_layered_scope(&self) -> LayeredScope {
        LayeredScope {
            scent: self.scent.clone(),
            palette_edges: self.palette_indices.clone(),
            distance_matrices: self.matrices.clone(),
            base_patterns: self.base_patterns.clone(),
            edge_count: self.edge_count,
        }
    }

    /// Total storage cost in bytes.
    pub fn total_bytes(&self) -> usize {
        let scent = self.edge_count;
        let palette_idx = self.edge_count * 3;
        let codebooks = self.palette_s.codebook_bytes()
            + self.palette_p.codebook_bytes()
            + self.palette_o.codebook_bytes();
        let matrices = self.matrices.byte_size();
        let bases = self.edge_count * SpoBase17::BYTE_SIZE;
        scent + palette_idx + codebooks + matrices + bases
    }

    /// Storage without Layer 2 (base patterns).
    pub fn compact_bytes(&self) -> usize {
        let scent = self.edge_count;
        let palette_idx = self.edge_count * 3;
        let codebooks = self.palette_s.codebook_bytes()
            + self.palette_p.codebook_bytes()
            + self.palette_o.codebook_bytes();
        let matrices = self.matrices.byte_size();
        scent + palette_idx + codebooks + matrices
    }
}

/// Compute the centroid of a set of SpoBase17 patterns.
fn scope_centroid(patterns: &[SpoBase17]) -> SpoBase17 {
    let n = patterns.len() as i64;
    let mut s_sum = [0i64; 17];
    let mut p_sum = [0i64; 17];
    let mut o_sum = [0i64; 17];

    for bp in patterns {
        for d in 0..17 {
            s_sum[d] += bp.subject.dims[d] as i64;
            p_sum[d] += bp.predicate.dims[d] as i64;
            o_sum[d] += bp.object.dims[d] as i64;
        }
    }

    let to_base = |sum: &[i64; 17]| -> Base17 {
        let mut dims = [0i16; 17];
        for d in 0..17 { dims[d] = (sum[d] / n) as i16; }
        Base17 { dims }
    };

    SpoBase17 {
        subject: to_base(&s_sum),
        predicate: to_base(&p_sum),
        object: to_base(&o_sum),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_plane(seed: u64) -> Vec<i8> {
        let mut v = vec![0i8; 16384];
        let mut s = seed;
        for x in v.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *x = (s >> 33) as i8;
        }
        v
    }

    #[test]
    fn test_build_scope() {
        let planes: Vec<(Vec<i8>, Vec<i8>, Vec<i8>)> = (0..100)
            .map(|i| (random_plane(i * 3), random_plane(i * 3 + 1), random_plane(i * 3 + 2)))
            .collect();

        let scope = Bgz17Scope::build(1, &planes, 32);

        assert_eq!(scope.edge_count, 100);
        assert_eq!(scope.palette_s.len(), 32);
        assert_eq!(scope.palette_indices.len(), 100);
        assert_eq!(scope.scent.len(), 100);

        println!("Scope storage:");
        println!("  Total:   {} bytes ({:.0}:1 vs full planes)",
            scope.total_bytes(),
            (100 * 6144) as f64 / scope.total_bytes() as f64);
        println!("  Compact: {} bytes ({:.0}:1 vs full planes)",
            scope.compact_bytes(),
            (100 * 6144) as f64 / scope.compact_bytes() as f64);
    }

    #[test]
    fn test_layered_search_from_scope() {
        let planes: Vec<(Vec<i8>, Vec<i8>, Vec<i8>)> = (0..50)
            .map(|i| (random_plane(i * 3), random_plane(i * 3 + 1), random_plane(i * 3 + 2)))
            .collect();

        let scope = Bgz17Scope::build(1, &planes, 16);
        let layered = scope.to_layered_scope();

        // Query: find edges similar to edge 0
        let query = &scope.base_patterns[0];
        let query_pe = scope.palette_indices[0];
        let query_scent = scope.scent[0];

        let results = layered.search(query_scent, &query_pe, query, 20, 10, 5);
        assert!(!results.is_empty());
        // Self should be top result
        assert_eq!(results[0].position, 0);
    }
}
