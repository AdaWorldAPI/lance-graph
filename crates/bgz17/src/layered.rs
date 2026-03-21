//! Layered Distance Codec: SID→Opus for knowledge graph search.
//!
//! ```text
//! Layer 0: Scent (1 byte)      — "is this edge even in the neighborhood?"
//! Layer 1: Palette (3 bytes)   — precomputed matrix lookup, O(1)
//! Layer 2: ZeckBF17 (102 bytes)— full base pattern L1
//! Layer 3: Full planes (6 KB)  — exact Hamming, ground truth
//! ```
//!
//! CAKES triangle inequality pruning terminates 95%+ of computations
//! at Layer 0-1. Only decision-boundary cases need Layer 2-3.

use crate::base17::SpoBase17;
use crate::palette::{Palette, PaletteEdge};
use crate::distance_matrix::SpoDistanceMatrices;

/// A search hit with layered distance information.
#[derive(Clone, Debug)]
pub struct LayeredHit {
    /// Position in the scope.
    pub position: usize,
    /// Layer 0: scent byte distance (0-8 bits differing).
    pub scent_distance: Option<u32>,
    /// Layer 1: palette distance (sum of 3 matrix lookups).
    pub palette_distance: Option<u32>,
    /// Layer 2: ZeckBF17 base L1 distance.
    pub base_distance: Option<u32>,
    /// Layer 3: exact Hamming distance (if loaded).
    pub exact_distance: Option<u32>,
    /// The best available distance (lowest layer computed).
    pub best_distance: u32,
    /// Which layer provided the best_distance.
    pub resolved_layer: u8,
}

/// A scope indexed for layered search.
///
/// Contains all four layers of distance information, loaded on demand.
/// Lance column pruning ensures reading Layer 0 never loads Layer 1-3.
pub struct LayeredScope {
    /// Layer 0: scent bytes (always loaded, 10 KB).
    pub scent: Vec<u8>,
    /// Layer 1: palette indices (loaded on demand, 30 KB).
    pub palette_edges: Vec<PaletteEdge>,
    /// Layer 1 support: precomputed distance matrices.
    pub distance_matrices: SpoDistanceMatrices,
    /// Layer 2: full base patterns (loaded on demand, ~1 MB).
    pub base_patterns: Vec<SpoBase17>,
    /// Number of edges in scope.
    pub edge_count: usize,
}

impl LayeredScope {
    /// Layer 0: scent-only search. Returns candidates sorted by scent distance.
    ///
    /// This is the HEEL stage of HHTL. Compares query scent against all
    /// edges' scent bytes. O(N) where N = edge count.
    pub fn search_scent(&self, query_scent: u8, max_candidates: usize) -> Vec<LayeredHit> {
        let mut hits: Vec<LayeredHit> = (0..self.edge_count)
            .map(|pos| {
                let d = (query_scent ^ self.scent[pos]).count_ones();
                LayeredHit {
                    position: pos,
                    scent_distance: Some(d),
                    palette_distance: None,
                    base_distance: None,
                    exact_distance: None,
                    best_distance: d,
                    resolved_layer: 0,
                }
            })
            .collect();

        hits.sort_by_key(|h| h.best_distance);
        hits.truncate(max_candidates);
        hits
    }

    /// Layer 1: refine candidates with palette lookup. O(1) per candidate.
    ///
    /// This is the HIP stage. For each candidate from Layer 0, look up
    /// the S+P+O palette distance in the precomputed matrices.
    /// Replaces ~200 candidates' distances with palette-resolution values.
    pub fn refine_palette(&self, candidates: &mut [LayeredHit], query: &PaletteEdge) {
        for hit in candidates.iter_mut() {
            let pe = &self.palette_edges[hit.position];
            let d = self.distance_matrices.spo_distance(
                query.s_idx, query.p_idx, query.o_idx,
                pe.s_idx, pe.p_idx, pe.o_idx,
            );
            hit.palette_distance = Some(d);
            hit.best_distance = d;
            hit.resolved_layer = 1;
        }
        candidates.sort_by_key(|h| h.best_distance);
    }

    /// Layer 2: refine top-N with full base L1. O(17) per candidate.
    ///
    /// Only needed for decision-boundary candidates where palette
    /// resolution is insufficient to distinguish rank order.
    pub fn refine_base(&self, candidates: &mut [LayeredHit], query: &SpoBase17, top_n: usize) {
        for hit in candidates.iter_mut().take(top_n) {
            let d = query.l1(&self.base_patterns[hit.position]);
            hit.base_distance = Some(d);
            hit.best_distance = d;
            hit.resolved_layer = 2;
        }
        candidates.sort_by_key(|h| h.best_distance);
    }

    /// Full layered search: scent → palette → base → (exact if needed).
    ///
    /// Matches the Opus analogy:
    /// - SILK (palette) handles most of the work
    /// - CELT (base L1) activates for transients/boundaries
    /// - Psychoacoustic masking (scent pruning) eliminates 95%+ of candidates
    pub fn search(
        &self,
        query_scent: u8,
        query_palette: &PaletteEdge,
        query_base: &SpoBase17,
        max_scent_candidates: usize,
        max_palette_candidates: usize,
        max_base_candidates: usize,
    ) -> Vec<LayeredHit> {
        // Layer 0: scent prune
        let mut candidates = self.search_scent(query_scent, max_scent_candidates);

        // Layer 1: palette refine
        let n_palette = candidates.len().min(max_palette_candidates);
        self.refine_palette(&mut candidates[..n_palette], query_palette);
        candidates.sort_by_key(|h| h.best_distance);

        // Layer 2: base refine (top-N only)
        self.refine_base(&mut candidates, query_base, max_base_candidates);

        candidates
    }

    /// Storage cost breakdown.
    pub fn storage_breakdown(&self) -> StorageBreakdown {
        StorageBreakdown {
            scent_bytes: self.edge_count,
            palette_bytes: self.edge_count * 3,
            matrix_bytes: self.distance_matrices.byte_size(),
            base_bytes: self.edge_count * SpoBase17::BYTE_SIZE,
            total_bytes: self.edge_count * (1 + 3 + SpoBase17::BYTE_SIZE)
                + self.distance_matrices.byte_size(),
            edge_count: self.edge_count,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StorageBreakdown {
    pub scent_bytes: usize,
    pub palette_bytes: usize,
    pub matrix_bytes: usize,
    pub base_bytes: usize,
    pub total_bytes: usize,
    pub edge_count: usize,
}

impl std::fmt::Display for StorageBreakdown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let full_plane_bytes = self.edge_count * 6144;
        writeln!(f, "Layered Scope Storage ({} edges)", self.edge_count)?;
        writeln!(f, "  Layer 0 (scent):   {:>8} bytes", self.scent_bytes)?;
        writeln!(f, "  Layer 1 (palette): {:>8} bytes (edges) + {:>8} bytes (matrices)",
            self.palette_bytes, self.matrix_bytes)?;
        writeln!(f, "  Layer 2 (base):    {:>8} bytes", self.base_bytes)?;
        writeln!(f, "  Total:             {:>8} bytes", self.total_bytes)?;
        writeln!(f, "  Full planes:       {:>8} bytes", full_plane_bytes)?;
        writeln!(f, "  Compression:       {:>8.0}:1", full_plane_bytes as f64 / self.total_bytes as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base17::Base17;
    use crate::palette::Palette;

    fn make_test_scope(n_edges: usize) -> (LayeredScope, SpoBase17, PaletteEdge) {
        // Generate synthetic edges
        let edges: Vec<SpoBase17> = (0..n_edges).map(|i| {
            let make_base = |seed: usize| {
                let mut dims = [0i16; 17];
                for d in 0..17 { dims[d] = ((seed * 97 + d * 31) % 512) as i16 - 256; }
                Base17 { dims }
            };
            SpoBase17 {
                subject: make_base(i * 3),
                predicate: make_base(i * 3 + 1),
                object: make_base(i * 3 + 2),
            }
        }).collect();

        // Build palettes
        let (s_pal, p_pal, o_pal) = Palette::build_spo(&edges, 32, 5);
        let matrices = SpoDistanceMatrices::build(&s_pal, &p_pal, &o_pal);

        // Encode all edges
        let palette_edges: Vec<PaletteEdge> = edges.iter()
            .map(|e| PaletteEdge {
                s_idx: s_pal.nearest(&e.subject),
                p_idx: p_pal.nearest(&e.predicate),
                o_idx: o_pal.nearest(&e.object),
            })
            .collect();

        // Compute scent bytes (self-referential for testing)
        let query = &edges[0];
        let scent: Vec<u8> = edges.iter()
            .map(|e| query.scent(e))
            .collect();

        let query_palette = PaletteEdge {
            s_idx: s_pal.nearest(&query.subject),
            p_idx: p_pal.nearest(&query.predicate),
            o_idx: o_pal.nearest(&query.object),
        };

        let scope = LayeredScope {
            scent,
            palette_edges,
            distance_matrices: matrices,
            base_patterns: edges,
            edge_count: n_edges,
        };

        (scope, query.clone(), query_palette)
    }

    #[test]
    fn test_layered_search() {
        let (scope, query_base, query_palette) = make_test_scope(100);
        let query_scent = query_base.scent(&query_base);

        let results = scope.search(
            query_scent,
            &query_palette,
            &query_base,
            50,   // scent candidates
            20,   // palette candidates
            10,   // base candidates
        );

        assert!(!results.is_empty());
        // First result should be the query itself (position 0)
        assert_eq!(results[0].position, 0);
        assert_eq!(results[0].best_distance, 0);
        // Results should be sorted by distance
        for w in results.windows(2) {
            assert!(w[0].best_distance <= w[1].best_distance);
        }
    }

    #[test]
    fn test_storage_breakdown() {
        let (scope, _, _) = make_test_scope(1000);
        let breakdown = scope.storage_breakdown();
        println!("{}", breakdown);
        assert!(breakdown.total_bytes < 1_000_000); // < 1 MB for 1000 edges
    }

    #[test]
    fn test_layer_refinement_improves() {
        let (scope, query_base, query_palette) = make_test_scope(200);
        let query_scent = query_base.scent(&query_base);

        // Layer 0 only
        let scent_results = scope.search_scent(query_scent, 50);

        // Layer 0 + 1
        let mut palette_results = scent_results.clone();
        scope.refine_palette(&mut palette_results, &query_palette);

        // The top-1 should still be position 0 at both layers
        assert_eq!(scent_results[0].position, 0);
        assert_eq!(palette_results[0].position, 0);
    }
}
