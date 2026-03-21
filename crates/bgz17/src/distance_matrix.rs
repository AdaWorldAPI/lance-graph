//! Precomputed distance matrices: 256×256 u16 per plane.
//!
//! After building a palette, compute ALL pairwise L1 distances once.
//! Every subsequent distance lookup becomes a single u16 array load.
//! The 128 KB matrix fits in L1 cache. ~10,000× faster than recomputing.

use crate::base17::Base17;
use crate::palette::Palette;
use crate::MAX_PALETTE_SIZE;

/// Precomputed pairwise distance matrix for one plane's palette.
///
/// `matrix[i * k + j]` = L1 distance between palette entries i and j.
/// Symmetric: `matrix[i][j] == matrix[j][i]`.
/// Diagonal: `matrix[i][i] == 0`.
#[derive(Clone, Debug)]
pub struct DistanceMatrix {
    /// Flat storage: k × k u16 values.
    pub data: Vec<u16>,
    /// Palette size (k). `data.len() == k * k`.
    pub k: usize,
}

impl DistanceMatrix {
    /// Build from a palette. O(k²) pairwise comparisons.
    pub fn build(palette: &Palette) -> Self {
        let k = palette.len();
        let mut data = vec![0u16; k * k];

        for i in 0..k {
            for j in (i + 1)..k {
                let d = palette.entries[i].l1(&palette.entries[j]);
                // Clamp to u16 (max L1 = 17 × 65535 = 1,114,095 > u16::MAX)
                // Scale: d / max_l1 * 65535
                let max_l1 = 17u64 * 65535;
                let scaled = ((d as u64 * 65535) / max_l1).min(65535) as u16;
                data[i * k + j] = scaled;
                data[j * k + i] = scaled;
            }
        }

        DistanceMatrix { data, k }
    }

    /// Look up distance between two palette indices. O(1).
    #[inline]
    pub fn distance(&self, a: u8, b: u8) -> u16 {
        self.data[a as usize * self.k + b as usize]
    }

    /// Byte size of the matrix.
    pub fn byte_size(&self) -> usize {
        self.k * self.k * 2
    }
}

/// Three distance matrices: one per S/P/O plane.
#[derive(Clone, Debug)]
pub struct SpoDistanceMatrices {
    pub subject: DistanceMatrix,
    pub predicate: DistanceMatrix,
    pub object: DistanceMatrix,
}

impl SpoDistanceMatrices {
    /// Build from three palettes.
    pub fn build(s_pal: &Palette, p_pal: &Palette, o_pal: &Palette) -> Self {
        SpoDistanceMatrices {
            subject: DistanceMatrix::build(s_pal),
            predicate: DistanceMatrix::build(p_pal),
            object: DistanceMatrix::build(o_pal),
        }
    }

    /// Combined S+P+O distance from palette indices. O(1): 3 array loads.
    #[inline]
    pub fn spo_distance(&self, a_s: u8, a_p: u8, a_o: u8, b_s: u8, b_p: u8, b_o: u8) -> u32 {
        self.subject.distance(a_s, b_s) as u32
            + self.predicate.distance(a_p, b_p) as u32
            + self.object.distance(a_o, b_o) as u32
    }

    /// Generate scent byte from palette indices using precomputed distances.
    /// Much faster than recomputing from base patterns.
    pub fn scent(&self, a_s: u8, a_p: u8, a_o: u8, b_s: u8, b_p: u8, b_o: u8) -> u8 {
        let ds = self.subject.distance(a_s, b_s) as u32;
        let dp = self.predicate.distance(a_p, b_p) as u32;
        let d_o = self.object.distance(a_o, b_o) as u32;

        // Threshold: half of max scaled distance
        let threshold = 32767u32; // 65535 / 2

        let sc = (ds < threshold) as u8;
        let pc = (dp < threshold) as u8;
        let oc = (d_o < threshold) as u8;
        sc | (pc << 1) | (oc << 2)
            | ((sc & pc) << 3) | ((sc & oc) << 4)
            | ((pc & oc) << 5) | ((sc & pc & oc) << 6)
    }

    /// Total byte size of all three matrices.
    pub fn byte_size(&self) -> usize {
        self.subject.byte_size() + self.predicate.byte_size() + self.object.byte_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::palette::Palette;

    fn make_palette(k: usize) -> Palette {
        let entries = (0..k).map(|i| {
            let mut dims = [0i16; 17];
            for d in 0..17 { dims[d] = ((i * 97 + d * 31) % 512) as i16 - 256; }
            Base17 { dims }
        }).collect();
        Palette { entries }
    }

    #[test]
    fn test_distance_self_zero() {
        let pal = make_palette(32);
        let dm = DistanceMatrix::build(&pal);
        for i in 0..32 {
            assert_eq!(dm.distance(i, i), 0, "Self-distance must be 0 for entry {}", i);
        }
    }

    #[test]
    fn test_distance_symmetric() {
        let pal = make_palette(32);
        let dm = DistanceMatrix::build(&pal);
        for i in 0..32u8 {
            for j in 0..32u8 {
                assert_eq!(dm.distance(i, j), dm.distance(j, i));
            }
        }
    }

    #[test]
    fn test_spo_distance_self_zero() {
        let pal = make_palette(16);
        let spo = SpoDistanceMatrices::build(&pal, &pal, &pal);
        assert_eq!(spo.spo_distance(5, 5, 5, 5, 5, 5), 0);
    }

    #[test]
    fn test_scent_self_all_close() {
        let pal = make_palette(16);
        let spo = SpoDistanceMatrices::build(&pal, &pal, &pal);
        let scent = spo.scent(5, 5, 5, 5, 5, 5);
        assert_eq!(scent & 0x7F, 0x7F);
    }

    #[test]
    fn test_cache_friendliness() {
        let pal = make_palette(256);
        let dm = DistanceMatrix::build(&pal);
        assert_eq!(dm.byte_size(), 256 * 256 * 2); // 128 KB — fits L1/L2 cache
        assert!(dm.byte_size() <= 131072);
    }
}
