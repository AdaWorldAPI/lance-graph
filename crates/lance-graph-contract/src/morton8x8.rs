//! `morton8x8` — checked 8:8 Cartesian address arithmetic on the Morton code
//! (`D-MORTON-0`).
//!
//! An `(x, y)` pair of bytes is interleaved into one `u16` Z-order code: `x` on
//! the even bits, `y` on the odd bits. This is the same code as
//! [`FacetTier::morton`](crate::facet::FacetTier::morton) with `x = lo` and
//! `y = hi`; that function stays the reference and the tests check the two
//! agree over all 65,536 pairs.
//!
//! # What the code gives without decoding
//!
//! - **A neighbour.** [`Morton8x8::checked_offset`] moves `x` and `y` by signed
//!   deltas directly on the code (dilated add and subtract), and returns `None`
//!   when a coordinate would leave `0..=255`. No `(x, y)` is decoded on the way.
//! - **The trie ascent.** Each hex nibble of the code holds two `x` bits and two
//!   `y` bits: one 4 x 4 refinement, fan-out 16, the same fan-out as
//!   [`crate::hhtl::FAN_OUT`]. [`Morton8x8::nibble_climb`] is the number of
//!   nibbles, counted from the finest, up to and including the highest one where
//!   two codes differ. It is read from `a XOR b`; nothing is searched.
//!
//! Over a full 256 x 256 tile the eight Moore directions give 521,220 in-tile
//! visits. 344,064 of them climb one nibble, 132,096 climb two, 35,904 three and
//! 9,156 four, so 91.35 % stay within two nibbles (pinned in the tests).
//!
//! # What this module does not do
//!
//! It holds no topology. Which offsets count as neighbours (Moore, von Neumann,
//! anything else) is the caller's choice. There is no neighbour list and no
//! stored edge.

/// The `x` bits of a code: bits 0, 2, …, 14.
const X_BITS: u16 = 0x5555;
/// The `y` bits of a code: bits 1, 3, …, 15.
const Y_BITS: u16 = 0xAAAA;

/// An `(x, y)` byte pair as one Morton (Z-order) code: `x` on even bits, `y` on
/// odd bits. Every `u16` is a valid code.
///
/// # Examples
///
/// ```
/// use lance_graph_contract::morton8x8::Morton8x8;
///
/// let m = Morton8x8::from_xy(5, 3);
/// assert_eq!((m.x(), m.y()), (5, 3));
/// assert_eq!(m.checked_offset(1, -1), Some(Morton8x8::from_xy(6, 2)));
/// assert_eq!(Morton8x8::from_xy(255, 0).checked_offset(1, 0), None);
/// assert_eq!(m.nibble_climb(m), 0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Morton8x8(u16);

impl Morton8x8 {
    /// Interleave `x` (even bits) and `y` (odd bits).
    #[inline]
    #[must_use]
    pub const fn from_xy(x: u8, y: u8) -> Self {
        Self(spread(x) | (spread(y) << 1))
    }

    /// Wrap an existing code. Every `u16` is a valid `(x, y)` pair.
    #[inline]
    #[must_use]
    pub const fn from_code(code: u16) -> Self {
        Self(code)
    }

    /// The raw code.
    #[inline]
    #[must_use]
    pub const fn code(self) -> u16 {
        self.0
    }

    /// The `x` coordinate (the even bits gathered back).
    #[inline]
    #[must_use]
    pub const fn x(self) -> u8 {
        gather(self.0)
    }

    /// The `y` coordinate (the odd bits gathered back).
    #[inline]
    #[must_use]
    pub const fn y(self) -> u8 {
        gather(self.0 >> 1)
    }

    /// The code moved by `dx` along `x` and `dy` along `y`, or `None` if either
    /// coordinate would leave `0..=255`.
    ///
    /// Works on the code: the other axis's bits are filled with ones for an add
    /// (so a carry runs through them) or cleared for a subtract (so a borrow
    /// does), and leaving the range is read from the carry or the borrow.
    #[inline]
    #[must_use]
    pub const fn checked_offset(self, dx: i8, dy: i8) -> Option<Self> {
        let code = match lane_offset(self.0, X_BITS, dx, 0) {
            Some(c) => c,
            None => return None,
        };
        match lane_offset(code, Y_BITS, dy, 1) {
            Some(c) => Some(Self(c)),
            None => None,
        }
    }

    /// How many nibbles, counted from the finest, must be climbed to reach the
    /// smallest 16-ary subtree holding both codes. `0` for equal codes, at most
    /// `4`.
    #[inline]
    #[must_use]
    pub const fn nibble_climb(self, other: Self) -> u8 {
        let diff = self.0 ^ other.0;
        let bits = 16 - diff.leading_zeros();
        bits.div_ceil(4) as u8
    }
}

/// Spread a byte's bits to the even positions of a `u16`.
#[inline]
const fn spread(v: u8) -> u16 {
    let mut x = v as u16;
    x = (x | (x << 4)) & 0x0F0F;
    x = (x | (x << 2)) & 0x3333;
    x = (x | (x << 1)) & 0x5555;
    x
}

/// Gather the even bits of `v` back into a byte (inverse of [`spread`]).
#[inline]
const fn gather(v: u16) -> u8 {
    let mut x = v & 0x5555;
    x = (x | (x >> 1)) & 0x3333;
    x = (x | (x >> 2)) & 0x0F0F;
    x = (x | (x >> 4)) & 0x00FF;
    x as u8
}

/// Move one axis of `code` by `delta`. `lane` is that axis's bit mask and
/// `shift` places a spread delta onto it (0 for `x`, 1 for `y`).
#[inline]
const fn lane_offset(code: u16, lane: u16, delta: i8, shift: u32) -> Option<u16> {
    let holes = !lane;
    let d = spread(delta.unsigned_abs()) << shift;
    let a = code & lane;
    if delta >= 0 {
        // Holes set to one carry through; a carry past bit 15 means the
        // coordinate passed 255.
        let sum = ((a | holes) as u32) + d as u32;
        if sum > 0xFFFF {
            None
        } else {
            Some((sum as u16 & lane) | (code & holes))
        }
    } else if d > a {
        // Spreading keeps order, so the spread compare is the coordinate compare.
        None
    } else {
        // Holes are zero in both operands, so a borrow runs through them.
        Some(((a - d) & lane) | (code & holes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facet::FacetTier;
    use crate::hhtl::NiblePath;

    /// The eight Moore offsets.
    const MOORE: [(i8, i8); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    fn all() -> impl Iterator<Item = (u8, u8)> {
        (0..=255u8).flat_map(|x| (0..=255u8).map(move |y| (x, y)))
    }

    /// Decode, move, bounds-check, encode: the slow geometry the code path
    /// must agree with.
    fn geometry(x: u8, y: u8, dx: i8, dy: i8) -> Option<Morton8x8> {
        let nx = i16::from(x) + i16::from(dx);
        let ny = i16::from(y) + i16::from(dy);
        if (0..=255).contains(&nx) && (0..=255).contains(&ny) {
            Some(Morton8x8::from_xy(nx as u8, ny as u8))
        } else {
            None
        }
    }

    /// FAILS IF: the code differs from `FacetTier::morton` for any pair, or
    /// decoding does not return the pair.
    #[test]
    fn code_matches_facet_tier_morton_and_round_trips() {
        for (x, y) in all() {
            let m = Morton8x8::from_xy(x, y);
            assert_eq!(m.code(), FacetTier { lo: x, hi: y }.morton(), "({x},{y})");
            assert_eq!((m.x(), m.y()), (x, y));
            assert_eq!(Morton8x8::from_code(m.code()), m);
        }
    }

    /// FAILS IF: any Moore step on the code disagrees with decoded geometry,
    /// including the edges and corners where it must return `None`.
    #[test]
    fn moore_steps_match_geometry_over_the_whole_tile() {
        for (x, y) in all() {
            let m = Morton8x8::from_xy(x, y);
            for (dx, dy) in MOORE {
                assert_eq!(
                    m.checked_offset(dx, dy),
                    geometry(x, y, dx, dy),
                    "({x},{y}) + ({dx},{dy})"
                );
            }
        }
    }

    /// FAILS IF: larger offsets, up to the full `i8` range, disagree with
    /// geometry. Sampled: every 17th x and every 13th y.
    #[test]
    fn wide_offsets_match_geometry() {
        let deltas = [-128i8, -127, -64, -17, -2, 0, 2, 17, 64, 127];
        for x in (0..=255u8).step_by(17) {
            for y in (0..=255u8).step_by(13) {
                let m = Morton8x8::from_xy(x, y);
                for dx in deltas {
                    for dy in deltas {
                        assert_eq!(m.checked_offset(dx, dy), geometry(x, y, dx, dy));
                    }
                }
            }
        }
    }

    /// FAILS IF: a zero offset moves the code, or a code climbs from itself.
    #[test]
    fn zero_offset_and_self_climb_stay_silent() {
        for code in [0u16, 1, 0x1234, 0xFFFF] {
            let m = Morton8x8::from_code(code);
            assert_eq!(m.checked_offset(0, 0), Some(m));
            assert_eq!(m.nibble_climb(m), 0);
        }
    }

    /// The nibble path of a code, coarsest nibble first, as an independent
    /// trie oracle.
    fn path(m: Morton8x8) -> NiblePath {
        let c = m.code();
        NiblePath::root((c >> 12) as u8 & 0xF)
            .child((c >> 8) as u8 & 0xF)
            .child((c >> 4) as u8 & 0xF)
            .child(c as u8 & 0xF)
    }

    /// FAILS IF: the XOR climb disagrees with `NiblePath::common_prefix_depth`
    /// for any Moore visit, or the visit count or climb histogram moves.
    #[test]
    fn nibble_climb_matches_the_trie_and_the_counts_are_pinned() {
        let mut visits = 0u32;
        let mut by_climb = [0u32; 5];
        for (x, y) in all() {
            let m = Morton8x8::from_xy(x, y);
            for (dx, dy) in MOORE {
                if let Some(n) = m.checked_offset(dx, dy) {
                    let climb = m.nibble_climb(n);
                    assert_eq!(climb, 4 - path(m).common_prefix_depth(path(n)));
                    visits += 1;
                    by_climb[climb as usize] += 1;
                }
            }
        }
        assert_eq!(visits, 521_220);
        assert_eq!(by_climb, [0, 344_064, 132_096, 35_904, 9_156]);
    }

    /// FAILS IF: on a 4 x 4 grid (2-bit axes) the in-grid test `code < 16`
    /// disagrees with the geometry, or the visit count is not 84.
    #[test]
    fn a_power_of_two_subgrid_is_a_code_prefix() {
        let mut visits = 0;
        for x in 0..4u8 {
            for y in 0..4u8 {
                let m = Morton8x8::from_xy(x, y);
                assert!(m.code() < 16);
                for (dx, dy) in MOORE {
                    let on_grid = (0..4).contains(&(i16::from(x) + i16::from(dx)))
                        && (0..4).contains(&(i16::from(y) + i16::from(dy)));
                    let in_prefix = m.checked_offset(dx, dy).is_some_and(|n| n.code() < 16);
                    assert_eq!(in_prefix, on_grid, "({x},{y}) + ({dx},{dy})");
                    visits += usize::from(in_prefix);
                }
            }
        }
        assert_eq!(visits, 84);
    }
}
