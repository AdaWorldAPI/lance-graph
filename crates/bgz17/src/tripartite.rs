//! Tripartite palette graph: cross-plane S×P×O reasoning.
//!
//! Three palettes form a tripartite graph. Edges between archetypes
//! in different planes encode cross-plane correlations. The blasgraph
//! semirings (HammingMin, SimilarityMax) operate on this graph to
//! answer queries like "which Subject archetypes are close to which
//! Predicate archetypes?" — the SP_ interaction at palette resolution.

use crate::base17::Base17;
use crate::palette::Palette;
use crate::distance_matrix::DistanceMatrix;
use crate::scalar_sparse::ScalarCsr;

/// Cross-plane distance between two palette entries from DIFFERENT planes.
/// Uses L1 on the i16[17] base patterns directly.
pub fn cross_plane_distance(a: &Base17, b: &Base17) -> u32 {
    a.l1(b)
}

/// Build a cross-plane adjacency matrix (ScalarCsr) between two palettes.
///
/// Each entry (i, j) = L1 distance between palette_a[i] and palette_b[j].
/// Only stores entries below `threshold`. The resulting sparse matrix can
/// be used with `spmv_min_plus` for shortest-path queries in cross-plane space.
pub fn cross_plane_matrix(pal_a: &Palette, pal_b: &Palette, threshold: u32) -> ScalarCsr {
    let nrows = pal_a.len();
    let ncols = pal_b.len();

    let mut row_ptr = Vec::with_capacity(nrows + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();

    row_ptr.push(0);
    for i in 0..nrows {
        for j in 0..ncols {
            let d = cross_plane_distance(&pal_a.entries[i], &pal_b.entries[j]);
            if d < threshold {
                col_idx.push(j);
                vals.push(d as f32);
            }
        }
        row_ptr.push(col_idx.len());
    }

    ScalarCsr { nrows, ncols, row_ptr, col_idx, vals }
}

/// The SP_, S_O, _PO cross-plane interaction matrices.
///
/// These generalize the scent byte's 3 pairwise bits (SP close? SO close? PO close?)
/// to continuous distances at palette resolution.
#[derive(Clone, Debug)]
pub struct CrossPlaneMatrices {
    /// Subject × Predicate cross-plane distances.
    pub sp: ScalarCsr,
    /// Subject × Object cross-plane distances.
    pub so: ScalarCsr,
    /// Predicate × Object cross-plane distances.
    pub po: ScalarCsr,
}

impl CrossPlaneMatrices {
    /// Build all three cross-plane matrices from palettes.
    pub fn build(
        s_pal: &Palette,
        p_pal: &Palette,
        o_pal: &Palette,
        threshold: u32,
    ) -> Self {
        CrossPlaneMatrices {
            sp: cross_plane_matrix(s_pal, p_pal, threshold),
            so: cross_plane_matrix(s_pal, o_pal, threshold),
            po: cross_plane_matrix(p_pal, o_pal, threshold),
        }
    }

    /// Query: given a Subject archetype index, find the closest Predicate archetypes.
    /// Returns (predicate_index, distance) pairs sorted by distance.
    pub fn nearest_predicates(&self, s_idx: u8, top_k: usize) -> Vec<(u8, f32)> {
        let mut results: Vec<(u8, f32)> = Vec::new();
        let row = s_idx as usize;
        if row >= self.sp.nrows { return results; }

        let start = self.sp.row_ptr[row];
        let end = self.sp.row_ptr[row + 1];
        for idx in start..end {
            results.push((self.sp.col_idx[idx] as u8, self.sp.vals[idx]));
        }
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(top_k);
        results
    }

    /// Query: given S and P archetype indices, find the closest O archetypes.
    /// Combines SO and PO distances via min-plus composition.
    pub fn nearest_objects(&self, s_idx: u8, p_idx: u8, top_k: usize) -> Vec<(u8, f32)> {
        // Get SO distances for this S
        let so_dists = row_distances(&self.so, s_idx as usize);
        // Get PO distances for this P
        let po_dists = row_distances(&self.po, p_idx as usize);

        // Combined distance = SO + PO (both must be reachable)
        let ncols = self.so.ncols.max(self.po.ncols);
        let mut combined: Vec<(u8, f32)> = Vec::new();

        for o in 0..ncols {
            let so_d = so_dists.get(&o).copied().unwrap_or(f32::INFINITY);
            let po_d = po_dists.get(&o).copied().unwrap_or(f32::INFINITY);
            if so_d < f32::INFINITY && po_d < f32::INFINITY {
                combined.push((o as u8, so_d + po_d));
            }
        }

        combined.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        combined.truncate(top_k);
        combined
    }
}

/// Extract row distances as a HashMap for sparse lookup.
fn row_distances(csr: &ScalarCsr, row: usize) -> std::collections::HashMap<usize, f32> {
    let mut map = std::collections::HashMap::new();
    if row >= csr.nrows { return map; }
    let start = csr.row_ptr[row];
    let end = csr.row_ptr[row + 1];
    for idx in start..end {
        map.insert(csr.col_idx[idx], csr.vals[idx]);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_palette(k: usize, seed: usize) -> Palette {
        let entries = (0..k).map(|i| {
            let mut dims = [0i16; 17];
            for d in 0..17 { dims[d] = ((i * 97 + d * 31 + seed * 53) % 512) as i16 - 256; }
            Base17 { dims }
        }).collect();
        Palette { entries }
    }

    #[test]
    fn test_cross_plane_build() {
        let s = make_palette(16, 0);
        let p = make_palette(16, 1);
        let o = make_palette(16, 2);
        let cp = CrossPlaneMatrices::build(&s, &p, &o, u32::MAX);

        assert!(cp.sp.nnz() > 0);
        assert!(cp.so.nnz() > 0);
        assert!(cp.po.nnz() > 0);
    }

    #[test]
    fn test_nearest_predicates() {
        let s = make_palette(16, 0);
        let p = make_palette(16, 1);
        let o = make_palette(16, 2);
        let cp = CrossPlaneMatrices::build(&s, &p, &o, u32::MAX);

        let nearest = cp.nearest_predicates(0, 5);
        assert!(!nearest.is_empty());
        // Should be sorted by distance
        for w in nearest.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }
}
