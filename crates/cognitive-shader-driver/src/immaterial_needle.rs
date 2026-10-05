//! Immaterial needle. Two `(8:8)` pairs, folded in register, never a cross table.
//!
//! `[a, b][c, d]` is two indexes. Each index is the interleave of a palette
//! pair. The fold combines the two reads with a law and drops them.
//!
//! Important distinction: a four-code cross table is refused, but a closed
//! Palette256 law `P × P -> P` is legal. Two relation reads may therefore feed
//! one 256 × 256 perturbation LUT as long as both relation results collapse
//! back to native `u8` ordinals. That is three 64-KiB LUT reads, not a 256^4
//! materialization.
//!
//! Not wired into the dispatch cycle. `CausalEdge64` is not touched.

/// A palette code. One byte of a `(8:8)` pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Palette(pub u8);

/// One needle. The interleave is the slot. The table is not carried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Needle {
    pub slot: u16,
}

impl Needle {
    /// `morton(a, b)` as a `u16`. Bit 0 is `b`'s low bit.
    pub fn pair(a: Palette, b: Palette) -> Self {
        Self {
            slot: morton(a.0, b.0),
        }
    }
}

/// Two needles. The fold input. Not a four-dimensional index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImmaterialNeedle {
    pub left: Needle,
    pub right: Needle,
}

impl ImmaterialNeedle {
    pub fn pairs(a: Palette, b: Palette, c: Palette, d: Palette) -> Self {
        Self {
            left: Needle::pair(a, b),
            right: Needle::pair(c, d),
        }
    }

    /// Combine two already-read bins. The bins are not stored as a pair.
    pub fn fold(self, left_bin: u8, right_bin: u8) -> u8 {
        let _ = self;
        left_bin.min(right_bin)
    }
}

/// A four-code cross table would be `256^4` slots and is the materialization
/// this type refuses. A 256 × 256 binary Palette256 law is *not* refused; see
/// `crate::palette_perturbation`, where each pair collapses back to one byte
/// before the next law is addressed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrossRefused;

impl ImmaterialNeedle {
    pub fn cross_table(self) -> Result<(), CrossRefused> {
        let _ = self;
        Err(CrossRefused)
    }
}

fn morton(a: u8, b: u8) -> u16 {
    let mut out = 0u16;
    for i in 0..8 {
        out |= u16::from((b >> i) & 1) << (2 * i);
        out |= u16::from((a >> i) & 1) << (2 * i + 1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_pairs_are_two_needles() {
        let n = ImmaterialNeedle::pairs(Palette(1), Palette(2), Palette(3), Palette(4));
        assert_eq!(n.left, Needle::pair(Palette(1), Palette(2)));
        assert_eq!(n.right, Needle::pair(Palette(3), Palette(4)));
        assert_ne!(n.left.slot, n.right.slot);
    }

    #[test]
    fn the_fold_drops_the_bins() {
        let n = ImmaterialNeedle::pairs(Palette(0), Palette(1), Palette(2), Palette(3));
        assert_eq!(n.fold(9, 4), 4);
    }

    #[test]
    fn the_cross_table_is_refused() {
        let n = ImmaterialNeedle::pairs(Palette(0), Palette(0), Palette(0), Palette(0));
        assert_eq!(n.cross_table(), Err(CrossRefused));
    }
}
