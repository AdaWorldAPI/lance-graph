//! PaletteSemiring: algebraic structure over palette indices.
//!
//! The 256×256 distance matrix defines a metric space. The compose table
//! defines a multiplication (path composition via XOR bind in Base17 space).
//! Together they form a semiring compatible with Session A's TypedGraph.
//!
//! compose_table[a * k + b] = palette index of `palette[a].xor_bind(palette[b])`.
//! This is the "path through a then b" operation — XOR bind in Base17 is
//! self-inverse and associative (bitwise XOR on i16 dims).

use crate::base17::Base17;
use crate::distance_matrix::DistanceMatrix;
use crate::palette::Palette;

/// Semiring over palette indices: distance (metric) + compose (path algebra).
#[derive(Clone, Debug)]
pub struct PaletteSemiring {
    /// Precomputed pairwise distance matrix.
    pub distance_matrix: DistanceMatrix,
    /// compose_table[a * k + b] = palette index of path(a → b).
    /// Size: k × k bytes.
    pub compose_table: Vec<u8>,
    /// Palette size.
    pub k: usize,
}

impl PaletteSemiring {
    /// Build from a palette: compute distance matrix and compose table.
    pub fn build(palette: &Palette) -> Self {
        let dm = DistanceMatrix::build(palette);
        let compose_table = build_compose(palette);
        PaletteSemiring {
            k: palette.len(),
            distance_matrix: dm,
            compose_table,
        }
    }

    /// Look up composed path: a → b.
    #[inline]
    pub fn compose(&self, a: u8, b: u8) -> u8 {
        self.compose_table[a as usize * self.k + b as usize]
    }

    /// Look up distance between two palette indices.
    #[inline]
    pub fn distance(&self, a: u8, b: u8) -> u16 {
        self.distance_matrix.distance(a, b)
    }

    /// Identity element: the palette entry closest to Base17::zero().
    pub fn identity(&self, palette: &Palette) -> u8 {
        palette.nearest(&Base17::zero())
    }

    /// Total byte size: distance matrix + compose table.
    pub fn byte_size(&self) -> usize {
        self.distance_matrix.byte_size() + self.compose_table.len()
    }
}

/// Build the compose table from a palette's Base17 entries.
///
/// For each pair (a, b), compute `palette[a].xor_bind(palette[b])`,
/// then find the nearest palette entry to the result.
pub fn build_compose(palette: &Palette) -> Vec<u8> {
    let k = palette.len();
    let mut table = vec![0u8; k * k];
    for a in 0..k {
        for b in 0..k {
            let composed = palette.entries[a].xor_bind(&palette.entries[b]);
            table[a * k + b] = palette.nearest(&composed);
        }
    }
    table
}

/// Three PaletteSemirings: one per S/P/O plane.
#[derive(Clone, Debug)]
pub struct SpoPaletteSemiring {
    pub subject: PaletteSemiring,
    pub predicate: PaletteSemiring,
    pub object: PaletteSemiring,
}

impl SpoPaletteSemiring {
    /// Build from three palettes.
    pub fn build(s_pal: &Palette, p_pal: &Palette, o_pal: &Palette) -> Self {
        SpoPaletteSemiring {
            subject: PaletteSemiring::build(s_pal),
            predicate: PaletteSemiring::build(p_pal),
            object: PaletteSemiring::build(o_pal),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base17::Base17;
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
    fn test_compose_identity() {
        // Build a palette that includes an actual zero entry for exact identity
        let mut entries: Vec<Base17> = (0..31).map(|i| {
            let mut dims = [0i16; BASE_DIM];
            for d in 0..BASE_DIM {
                dims[d] = ((i * 97 + d * 31) % 512) as i16 - 256;
            }
            Base17 { dims }
        }).collect();
        entries.push(Base17::zero()); // entry 31 is exact zero
        let pal = Palette { entries };

        let sr = PaletteSemiring::build(&pal);
        let id = sr.identity(&pal);
        assert_eq!(id, 31, "identity should be the zero entry");

        // compose(a, identity) = a when identity is exact zero
        for a in 0..32u8 {
            let composed = sr.compose(a, id);
            assert_eq!(composed, a,
                "compose({}, identity={}) should equal {} but got {}", a, id, a, composed);
        }
    }

    #[test]
    fn test_compose_self_inverse() {
        let pal = make_palette(32);
        let sr = PaletteSemiring::build(&pal);
        // xor_bind is self-inverse, so compose(compose(a, b), b) ≈ a
        for a in 0..8u8 {
            for b in 0..8u8 {
                let ab = sr.compose(a, b);
                let abb = sr.compose(ab, b);
                // Should be close to a (quantization error possible)
                let dist = sr.distance(a, abb);
                // Allow some quantization slack (palette approximation)
                assert!(dist < 10000,
                    "compose(compose({}, {}), {}) = {}, dist to {} = {}",
                    a, b, b, abb, a, dist);
            }
        }
    }

    #[test]
    fn test_compose_table_size() {
        let pal = make_palette(64);
        let sr = PaletteSemiring::build(&pal);
        assert_eq!(sr.compose_table.len(), 64 * 64);
        assert_eq!(sr.k, 64);
    }

    #[test]
    fn test_spo_semiring_build() {
        let s_pal = make_palette(16);
        let p_pal = make_palette(32);
        let o_pal = make_palette(24);
        let spo = SpoPaletteSemiring::build(&s_pal, &p_pal, &o_pal);
        assert_eq!(spo.subject.k, 16);
        assert_eq!(spo.predicate.k, 32);
        assert_eq!(spo.object.k, 24);
    }

    #[test]
    fn test_byte_size() {
        let pal = make_palette(256);
        let sr = PaletteSemiring::build(&pal);
        // distance matrix: 256*256*2 = 128KB
        // compose table: 256*256 = 64KB
        assert_eq!(sr.byte_size(), 256 * 256 * 2 + 256 * 256);
    }
}
