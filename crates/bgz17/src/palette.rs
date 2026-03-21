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
                for d in 0..17 {
                    new_centroids[c][d] += p.dims[d] as i64;
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
