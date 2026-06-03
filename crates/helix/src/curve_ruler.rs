//! Stage 2 — the φ-spiral curve-ruler: stride-4-over-17 arc coupling from the
//! HHTL place offset.
//!
//! **The Curve-Ruler Principle:** every curve is determined by start and end on
//! the curve-ruler. A draughtsman's French curve has a fixed shape; mark two
//! points and the whole curve between them is determined. The φ-low-discrepancy
//! sequence IS the template — once you know "it is the φ-spiral" and "index a to
//! index b", every interior point is determined by `(offset + STRIDE·k) mod
//! MODULUS`. Endpoints are stored; the interior regenerates. This lifts the
//! `bgz17` 680× compression to the *curve* level: do not compress the points —
//! recognise they lie on a template, keep only the endpoints.
//!
//! The PLACE (an HHTL trie address) sets WHERE on the ruler the arc begins; the
//! stride-4-over-17 walk (`gcd(4,17)=1` → full permutation of all 17 residues)
//! is how the RESIDUE rides the same φ skeleton the Base17 palette uses.
use crate::constants::{MODULUS, STRIDE};

/// The φ-spiral curve-ruler, fixed by a single `start_offset ∈ [0, 17)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CurveRuler {
    start_offset: u8,
}

impl CurveRuler {
    /// `M = 17` — the prime modulus (Base17 / zeck17 alignment).
    pub const MODULUS: u8 = MODULUS;
    /// stride = 4 — coprime to 17, so the arc is a full permutation of the 17 residues.
    pub const STRIDE: u8 = STRIDE;

    /// Build a ruler from a raw place offset: `start = place mod 17`.
    pub fn from_place(place: u64) -> Self {
        Self {
            start_offset: (place % MODULUS as u64) as u8,
        }
    }

    /// Build a ruler from an HHTL address `(path, depth)` — the `NiblePath` packed
    /// form — WITHOUT importing the HHTL type (keeps the crate standalone). The
    /// depth is folded in so two addresses with the same path but different depth
    /// begin at different arc positions.
    pub fn from_hhtl(path: u64, depth: u8) -> Self {
        Self::from_place(path.wrapping_add(depth as u64))
    }

    /// The arc start position on the ruler ∈ [0, 17).
    pub fn start_offset(&self) -> u8 {
        self.start_offset
    }

    /// The k-th index along the stride arc: `(start + 4·k) mod 17`. Because
    /// `gcd(4, 17) = 1`, `index(0)..index(17)` is a full permutation of all 17
    /// residues (no residue is missed — unlike the banned Fibonacci-mod-17 walk).
    pub fn index(&self, k: u32) -> u8 {
        let step = (STRIDE as u64 * k as u64) % MODULUS as u64;
        ((self.start_offset as u64 + step) % MODULUS as u64) as u8
    }

    /// One full period of the stride arc (all 17 residues, in stride order).
    pub fn arc(&self) -> [u8; MODULUS as usize] {
        let mut out = [0u8; MODULUS as usize];
        for (k, slot) in out.iter_mut().enumerate() {
            *slot = self.index(k as u32);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_offset_is_place_mod_17() {
        assert_eq!(
            CurveRuler::from_place(0x1234).start_offset(),
            (0x1234u64 % 17) as u8
        );
        assert_eq!(CurveRuler::from_place(0).start_offset(), 0);
        assert_eq!(CurveRuler::from_place(17).start_offset(), 0);
        assert_eq!(CurveRuler::from_place(18).start_offset(), 1);
    }

    #[test]
    fn index_zero_is_start() {
        let r = CurveRuler::from_place(5);
        assert_eq!(r.index(0), 5);
    }

    #[test]
    fn arc_is_full_permutation() {
        for place in [0u64, 1, 7, 16, 0x1234, u64::MAX] {
            let arc = CurveRuler::from_place(place).arc();
            let mut seen = [false; MODULUS as usize];
            for &idx in arc.iter() {
                assert!(idx < MODULUS, "index out of range");
                seen[idx as usize] = true;
            }
            assert!(seen.iter().all(|&s| s), "arc must visit all 17 residues");
        }
    }

    #[test]
    fn index_wraps_with_period_17() {
        let r = CurveRuler::from_place(3);
        for k in 0..50u32 {
            assert_eq!(r.index(k), r.index(k + 17));
        }
    }

    #[test]
    fn hhtl_depth_distinguishes() {
        let a = CurveRuler::from_hhtl(0x10, 0);
        let b = CurveRuler::from_hhtl(0x10, 1);
        assert_ne!(a.start_offset(), b.start_offset());
    }
}
