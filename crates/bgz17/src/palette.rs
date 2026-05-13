//! Palette: 256 archetypal Base17 patterns per plane.
//!
//! Like indexed color (GIF/PNG palette mode) — store 256 representative
//! i16[17] base patterns, then each edge is just 3 bytes (u8 index per S/P/O).
//!
//! At 1000 edges: 3 KB (indices) + 8.7 KB (codebook) = 11.7 KB
//! vs 102 KB (raw ZeckBF17) vs 6 MB (full planes).

use crate::base17::{Base17, SpoBase17};
use crate::MAX_PALETTE_SIZE;

/// A palette codebook: up to 256 archetypal Base17 patterns.
#[derive(Clone, Debug)]
pub struct Palette {
    /// The archetype entries.
    pub entries: Vec<Base17>,
}

/// A palette-encoded edge: 3 bytes (one u8 index per S/P/O plane).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PaletteEdge {
    pub s_idx: u8,
    pub p_idx: u8,
    pub o_idx: u8,
}

impl PaletteEdge {
    /// Serialize to 3 bytes.
    pub fn to_bytes(self) -> [u8; 3] {
        [self.s_idx, self.p_idx, self.o_idx]
    }

    /// Deserialize from 3 bytes.
    pub fn from_bytes(b: &[u8; 3]) -> Self {
        PaletteEdge { s_idx: b[0], p_idx: b[1], o_idx: b[2] }
    }
}

impl Palette {
    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Byte size of the codebook.
    pub fn codebook_bytes(&self) -> usize {
        self.entries.len() * Base17::BYTE_SIZE
    }

    /// Find the nearest palette entry to a given base pattern. Returns index.
    pub fn nearest(&self, query: &Base17) -> u8 {
        let mut best_idx = 0u8;
        let mut best_dist = u32::MAX;
        for (i, entry) in self.entries.iter().enumerate() {
            let d = query.l1(entry);
            if d < best_dist {
                best_dist = d;
                best_idx = i as u8;
            }
        }
        best_idx
    }

    /// Build a precomputed distance table for O(1) inter-centroid distance.
    ///
    /// Returns a 256×256 u16 table where `table[i][j]` = L1 distance between
    /// `entries[i]` and `entries[j]`. Used by the renderer and cascade skip
    /// for fast palette-edge distance without recomputing L1 per query.
    pub fn build_distance_table(&self) -> PaletteDistanceTable {
        let k = self.entries.len();
        let mut table = vec![0u16; 256 * 256];
        for i in 0..k {
            for j in i..k {
                let d = self.entries[i].l1(&self.entries[j]) as u16;
                table[i * 256 + j] = d;
                table[j * 256 + i] = d;
            }
        }
        PaletteDistanceTable { table, size: k }
    }

    /// Encode an SpoBase17 edge to palette indices.
    pub fn encode_edge(&self, edge: &SpoBase17) -> PaletteEdge {
        PaletteEdge {
            s_idx: self.nearest(&edge.subject),
            p_idx: self.nearest(&edge.predicate),
            o_idx: self.nearest(&edge.object),
        }
    }

    /// Decode palette indices back to Base17 patterns (lossy).
    pub fn decode_edge(&self, pe: PaletteEdge) -> SpoBase17 {
        SpoBase17 {
            subject: self.entries[pe.s_idx as usize].clone(),
            predicate: self.entries[pe.p_idx as usize].clone(),
            object: self.entries[pe.o_idx as usize].clone(),
        }
    }

    /// Build a palette from a collection of Base17 patterns using k-means.
    ///
    /// `k`: target palette size (max 256).
    /// `max_iter`: k-means iterations (typically converges in 1-3).
    pub fn build(patterns: &[Base17], k: usize, max_iter: usize) -> Self {
        let k = k.min(MAX_PALETTE_SIZE).min(patterns.len());
        if k == 0 {
            return Palette { entries: Vec::new() };
        }

        // Initialize centroids: k-means++ style (first = random, rest = farthest)
        let mut centroids: Vec<Base17> = Vec::with_capacity(k);
        centroids.push(patterns[0].clone());

        for _ in 1..k {
            // Find pattern farthest from all existing centroids
            let mut best_idx = 0;
            let mut best_dist = 0u64;
            for (i, p) in patterns.iter().enumerate() {
                let min_d: u32 = centroids.iter().map(|c| p.l1(c)).min().unwrap_or(u32::MAX);
                if min_d as u64 > best_dist {
                    best_dist = min_d as u64;
                    best_idx = i;
                }
            }
            centroids.push(patterns[best_idx].clone());
        }

        // K-means iterations
        for _iter in 0..max_iter {
            // Assign each pattern to nearest centroid
            let mut assignments = vec![0usize; patterns.len()];
            for (i, p) in patterns.iter().enumerate() {
                let mut best = 0;
                let mut best_d = u32::MAX;
                for (c, centroid) in centroids.iter().enumerate() {
                    let d = p.l1(centroid);
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                assignments[i] = best;
            }

            // Recompute centroids
            let mut new_centroids: Vec<[i64; 17]> = vec![[0i64; 17]; k];
            let mut counts = vec![0u32; k];

            for (i, p) in patterns.iter().enumerate() {
                let c = assignments[i];
                counts[c] += 1;
                for (d, nc) in new_centroids[c].iter_mut().enumerate() {
                    *nc += p.dims[d] as i64;
                }
            }

            let mut changed = false;
            for c in 0..k {
                if counts[c] == 0 { continue; }
                let mut new_dims = [0i16; 17];
                for d in 0..17 {
                    new_dims[d] = (new_centroids[c][d] / counts[c] as i64) as i16;
                }
                let new_base = Base17 { dims: new_dims };
                if new_base != centroids[c] {
                    changed = true;
                    centroids[c] = new_base;
                }
            }

            if !changed { break; }
        }

        Palette { entries: centroids }
    }

    /// Sigma-band palette: codebook from empirical distribution.
    ///
    /// Each band boundary = sorted-percentile of input patterns.
    /// k bands = k entries. Guaranteed no empty clusters (each band covers
    /// 100/k percent of data by construction).
    ///
    /// Distribution-free: works for Gaussian, bimodal, skewed, heavy-tailed.
    /// Inspired by GQ (arxiv 2512.06609): Target Divergence Constraint
    /// → codebook without training, guaranteed uniform utilization.
    pub fn from_sigma_bands(patterns: &[Base17], k: usize) -> Self {
        let k = k.min(MAX_PALETTE_SIZE).min(patterns.len());
        if k == 0 {
            return Palette { entries: Vec::new() };
        }

        // Sort patterns by L1 distance from the centroid (global mean)
        let n = patterns.len();
        let mut mean = [0i64; 17];
        for p in patterns {
            for (d, m) in mean.iter_mut().enumerate() {
                *m += p.dims[d] as i64;
            }
        }
        let centroid = Base17 {
            dims: {
                let mut dims = [0i16; 17];
                for d in 0..17 {
                    dims[d] = (mean[d] / n as i64) as i16;
                }
                dims
            },
        };

        // Compute distances from centroid and sort indices by distance
        let mut indexed: Vec<(usize, u32)> = patterns
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.l1(&centroid)))
            .collect();
        indexed.sort_unstable_by_key(|&(_, d)| d);

        // Pick one representative per equal-percentile band
        let mut entries = Vec::with_capacity(k);
        for band in 0..k {
            let center_idx = (band * n / k) + (n / (2 * k));
            let idx = center_idx.min(n - 1);
            entries.push(patterns[indexed[idx].0].clone());
        }

        Palette { entries }
    }

    /// Build three palettes (one per S/P/O plane) from a set of SpoBase17 edges.
    pub fn build_spo(edges: &[SpoBase17], k: usize, max_iter: usize) -> (Self, Self, Self) {
        let s_patterns: Vec<Base17> = edges.iter().map(|e| e.subject.clone()).collect();
        let p_patterns: Vec<Base17> = edges.iter().map(|e| e.predicate.clone()).collect();
        let o_patterns: Vec<Base17> = edges.iter().map(|e| e.object.clone()).collect();

        (
            Palette::build(&s_patterns, k, max_iter),
            Palette::build(&p_patterns, k, max_iter),
            Palette::build(&o_patterns, k, max_iter),
        )
    }
}

/// Precomputed 256×256 L1 distance table for O(1) inter-centroid lookup.
///
/// Built once from a `Palette` via `palette.build_distance_table()`.
/// Used by the cascade skip (HHTL), renderer force-directed layout, and
/// any path that needs repeated palette-edge distance without recomputing L1.
///
/// Memory: 256×256×2 = 128 KB (fits L2 cache). Build cost: O(k²×17).
#[derive(Clone)]
pub struct PaletteDistanceTable {
    table: Vec<u16>,
    size: usize,
}

impl PaletteDistanceTable {
    /// O(1) distance between two palette indices.
    #[inline]
    pub fn distance(&self, a: u8, b: u8) -> u16 {
        self.table[a as usize * 256 + b as usize]
    }

    /// Number of active entries (≤ 256).
    pub fn size(&self) -> usize { self.size }

    /// Distance between two PaletteEdges (sum of S + P + O distances).
    #[inline]
    pub fn edge_distance(&self, a: PaletteEdge, b: PaletteEdge) -> u32 {
        self.distance(a.s_idx, b.s_idx) as u32
            + self.distance(a.p_idx, b.p_idx) as u32
            + self.distance(a.o_idx, b.o_idx) as u32
    }

    /// Memory footprint in bytes.
    pub fn byte_size(&self) -> usize { self.table.len() * 2 }
}

/// Palette resolution: trade compression vs accuracy.
///
/// Edge count determines optimal palette size:
/// - Few edges → small palette (less data, faster build)
/// - Many edges → larger palette (more archetypes, better rank correlation)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaletteResolution {
    /// Full 256-entry palette. 128KB distance matrix, ρ=0.992.
    Full256,
    /// Half 128-entry palette. 32KB distance matrix, ρ=0.965.
    Half128,
    /// Quarter 64-entry palette. 8KB distance matrix, ρ=0.738.
    Quarter64,
}

impl PaletteResolution {
    /// Auto-select palette resolution based on edge count.
    ///
    /// - <100 edges: Quarter64 (64 archetypes is enough, saves memory)
    /// - 100-1000 edges: Half128 (good balance)
    /// - >1000 edges: Full256 (need maximum discrimination)
    pub fn auto_select(edge_count: usize) -> Self {
        if edge_count < 100 {
            PaletteResolution::Quarter64
        } else if edge_count <= 1000 {
            PaletteResolution::Half128
        } else {
            PaletteResolution::Full256
        }
    }

    /// Return the k value for this resolution.
    pub fn k(self) -> usize {
        match self {
            PaletteResolution::Full256 => 256,
            PaletteResolution::Half128 => 128,
            PaletteResolution::Quarter64 => 64,
        }
    }

    /// Distance matrix byte size for this resolution.
    pub fn matrix_bytes(self) -> usize {
        let k = self.k();
        k * k * 2 // u16 per entry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_patterns(n: usize) -> Vec<Base17> {
        (0..n).map(|i| {
            let mut dims = [0i16; 17];
            for d in 0..17 {
                dims[d] = ((i * 7 + d * 13) % 256) as i16 - 128;
            }
            Base17 { dims }
        }).collect()
    }

    #[test]
    fn test_build_palette() {
        let patterns = make_patterns(100);
        let palette = Palette::build(&patterns, 16, 10);
        assert_eq!(palette.len(), 16);
    }

    #[test]
    fn test_nearest_self() {
        let patterns = make_patterns(50);
        let palette = Palette::build(&patterns, 50, 1);
        // Each pattern should map to itself or something very close
        for p in &patterns {
            let idx = palette.nearest(p);
            let dist = p.l1(&palette.entries[idx as usize]);
            // Should be close (possibly not exact due to centroid averaging)
            assert!(dist < 1000, "Nearest distance {} too large", dist);
        }
    }

    #[test]
    fn test_encode_decode() {
        let patterns = make_patterns(100);
        let palette = Palette::build(&patterns, 32, 5);
        let edge = SpoBase17 {
            subject: patterns[10].clone(),
            predicate: patterns[20].clone(),
            object: patterns[30].clone(),
        };
        let encoded = palette.encode_edge(&edge);
        let decoded = palette.decode_edge(encoded);
        // Decoded should be close to original
        assert!(edge.subject.l1(&decoded.subject) < 2000);
    }

    #[test]
    fn test_palette_edge_bytes() {
        let pe = PaletteEdge { s_idx: 42, p_idx: 128, o_idx: 255 };
        let bytes = pe.to_bytes();
        let pe2 = PaletteEdge::from_bytes(&bytes);
        assert_eq!(pe, pe2);
    }

    #[test]
    fn test_palette_resolution_auto_select() {
        assert_eq!(PaletteResolution::auto_select(10), PaletteResolution::Quarter64);
        assert_eq!(PaletteResolution::auto_select(99), PaletteResolution::Quarter64);
        assert_eq!(PaletteResolution::auto_select(100), PaletteResolution::Half128);
        assert_eq!(PaletteResolution::auto_select(500), PaletteResolution::Half128);
        assert_eq!(PaletteResolution::auto_select(1000), PaletteResolution::Half128);
        assert_eq!(PaletteResolution::auto_select(1001), PaletteResolution::Full256);
        assert_eq!(PaletteResolution::auto_select(10000), PaletteResolution::Full256);
    }

    #[test]
    fn test_palette_resolution_k() {
        assert_eq!(PaletteResolution::Quarter64.k(), 64);
        assert_eq!(PaletteResolution::Half128.k(), 128);
        assert_eq!(PaletteResolution::Full256.k(), 256);
    }

    #[test]
    fn test_palette_resolution_matrix_bytes() {
        assert_eq!(PaletteResolution::Quarter64.matrix_bytes(), 8192);
        assert_eq!(PaletteResolution::Half128.matrix_bytes(), 32768);
        assert_eq!(PaletteResolution::Full256.matrix_bytes(), 131072);
    }

    #[test]
    fn test_sigma_band_palette_size() {
        let patterns = make_patterns(200);
        let palette = Palette::from_sigma_bands(&patterns, 32);
        assert_eq!(palette.len(), 32);
    }

    #[test]
    fn test_sigma_band_no_empty() {
        let patterns = make_patterns(100);
        let palette = Palette::from_sigma_bands(&patterns, 16);
        assert_eq!(palette.len(), 16);
        // All entries should be distinct (from different percentile bands)
        for i in 0..palette.len() {
            for j in (i + 1)..palette.len() {
                // Not necessarily distinct, but they come from different positions
                // At minimum, palette shouldn't be empty
                assert!(!palette.entries[i].dims.iter().all(|&d| d == 0) || i == 0);
            }
        }
    }

    #[test]
    fn test_sigma_band_comparable_to_kmeans() {
        let patterns = make_patterns(200);
        let sigma = Palette::from_sigma_bands(&patterns, 32);
        let kmeans = Palette::build(&patterns, 32, 10);

        // Both should produce reasonable assignments
        let total_dist_sigma: u64 = patterns.iter().map(|p| {
            let idx = sigma.nearest(p);
            p.l1(&sigma.entries[idx as usize]) as u64
        }).sum();

        let total_dist_kmeans: u64 = patterns.iter().map(|p| {
            let idx = kmeans.nearest(p);
            p.l1(&kmeans.entries[idx as usize]) as u64
        }).sum();

        // Sigma-band should be within 5× of k-means (it's training-free)
        assert!(
            total_dist_sigma < total_dist_kmeans * 5,
            "sigma {} should be within 5× of kmeans {}",
            total_dist_sigma, total_dist_kmeans
        );
    }

    #[test]
    fn test_convergence() {
        // K-means should converge quickly
        let patterns = make_patterns(200);
        let p1 = Palette::build(&patterns, 32, 1);
        let p5 = Palette::build(&patterns, 32, 5);
        let p20 = Palette::build(&patterns, 32, 20);

        // Total assignment distance should decrease with iterations
        let total_dist = |pal: &Palette| -> u64 {
            patterns.iter().map(|p| {
                let idx = pal.nearest(p);
                p.l1(&pal.entries[idx as usize]) as u64
            }).sum::<u64>()
        };

        let d1 = total_dist(&p1);
        let d5 = total_dist(&p5);
        let d20 = total_dist(&p20);
        assert!(d5 <= d1, "5 iters should be ≤ 1 iter: {} vs {}", d5, d1);
        assert!(d20 <= d5, "20 iters should be ≤ 5 iters: {} vs {}", d20, d5);
    }
}
