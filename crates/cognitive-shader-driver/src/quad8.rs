//! Four bytes, two zero-copy readings.
//!
//! `Quad8` is the common 32-bit shape underneath two different kernels:
//!
//! ```text
//! structural view                 calibrated view
//! ----------------               ----------------
//! [a][b][c][d]                   [a:b][c:d]
//!  8 × 8 × 8 × 8                 256 × 256 each
//!  implicit Cartesian space       Palette256 pair addresses
//!  deterministic / no LUT         LUT law from palette_perturbation
//! ```
//!
//! The bytes never move and no Cartesian product is materialized.  The
//! structural view enumerates only occupied coordinates and feeds them
//! directly to a consumer/fold.  The calibrated view reuses the exact same
//! four bytes as the two 8:8 addresses introduced by the Palette256 kernel.
//!
//! This module deliberately contains no LUT and no matrix/tensor allocation.

use crate::palette_perturbation::{PairAddress, PaletteState};

/// Four resident bytes with view-selected semantics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Quad8([u8; 4]);

impl Quad8 {
    #[inline(always)]
    pub const fn new(a: u8, b: u8, c: u8, d: u8) -> Self {
        Self([a, b, c, d])
    }

    #[inline(always)]
    pub const fn from_bytes(bytes: [u8; 4]) -> Self {
        Self(bytes)
    }

    #[inline(always)]
    pub const fn bytes(self) -> [u8; 4] {
        self.0
    }

    /// Calibrated reading: the same four bytes are two direct 8:8
    /// Palette256 addresses. No conversion or reshuffle occurs.
    #[inline(always)]
    pub const fn palette_pairs(self) -> (PairAddress, PairAddress) {
        (
            PairAddress::new(PaletteState(self.0[0]), PaletteState(self.0[1])),
            PairAddress::new(PaletteState(self.0[2]), PaletteState(self.0[3])),
        )
    }

    /// Number of occupied coordinates in the implicit 8×8×8×8 Cartesian
    /// product. This is the product of the four byte popcounts, never 4096
    /// unless every bit in every byte is live.
    #[inline]
    pub fn occupied_len(self) -> usize {
        self.0
            .iter()
            .map(|byte| byte.count_ones() as usize)
            .product()
    }

    /// Visit occupied structural coordinates in deterministic lexicographic
    /// order (a, then b, then c, then d), with no intermediate matrix.
    ///
    /// Each set bit contributes its ordinal 0..7. A fully dense Quad8 visits
    /// exactly 4096 addresses; sparse inputs visit only the Cartesian product
    /// of their live bits.
    #[inline]
    pub fn for_each_cartesian(self, mut visit: impl FnMut(CartesianAddress12)) {
        let mut a_mask = self.0[0];
        while a_mask != 0 {
            let a = a_mask.trailing_zeros() as u8;
            a_mask &= a_mask - 1;

            let mut b_mask = self.0[1];
            while b_mask != 0 {
                let b = b_mask.trailing_zeros() as u8;
                b_mask &= b_mask - 1;

                let mut c_mask = self.0[2];
                while c_mask != 0 {
                    let c = c_mask.trailing_zeros() as u8;
                    c_mask &= c_mask - 1;

                    let mut d_mask = self.0[3];
                    while d_mask != 0 {
                        let d = d_mask.trailing_zeros() as u8;
                        d_mask &= d_mask - 1;
                        visit(CartesianAddress12::new(a, b, c, d));
                    }
                }
            }
        }
    }

    /// Fold the implicit matrix/tensor directly into a terminal accumulator.
    ///
    /// This is the matrix kernel without materialization: coordinates are
    /// generated and consumed one at a time.
    #[inline]
    pub fn fold_cartesian<T>(
        self,
        mut acc: T,
        mut fold: impl FnMut(T, CartesianAddress12) -> T,
    ) -> T {
        self.for_each_cartesian(|address| {
            // Move the accumulator through the fold without requiring T: Copy.
            // Option is only a stack control slot; no heap or population exists.
            let current = unsafe { core::ptr::read(&acc) };
            acc = fold(current, address);
        });
        acc
    }
}

/// One occupied coordinate in the implicit 8×8×8×8 space.
///
/// Four 3-bit ordinals fit exactly in 12 bits:
/// `aaaa bbb ccc ddd` conceptually, packed as `a<<9 | b<<6 | c<<3 | d`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct CartesianAddress12(u16);

impl CartesianAddress12 {
    #[inline(always)]
    pub const fn new(a: u8, b: u8, c: u8, d: u8) -> Self {
        debug_assert!(a < 8 && b < 8 && c < 8 && d < 8);
        Self(((a as u16) << 9) | ((b as u16) << 6) | ((c as u16) << 3) | d as u16)
    }

    #[inline(always)]
    pub const fn raw(self) -> u16 {
        self.0
    }

    #[inline(always)]
    pub const fn ordinals(self) -> [u8; 4] {
        [
            ((self.0 >> 9) & 0x7) as u8,
            ((self.0 >> 6) & 0x7) as u8,
            ((self.0 >> 3) & 0x7) as u8,
            (self.0 & 0x7) as u8,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad8_is_exactly_four_bytes() {
        assert_eq!(core::mem::size_of::<Quad8>(), 4);
    }

    #[test]
    fn same_bytes_are_two_palette_pair_addresses() {
        let q = Quad8::new(0x12, 0x34, 0x56, 0x78);
        let (left, right) = q.palette_pairs();
        assert_eq!(left.0, 0x1234);
        assert_eq!(right.0, 0x5678);
        assert_eq!(q.bytes(), [0x12, 0x34, 0x56, 0x78]);
    }

    #[test]
    fn structural_address_is_four_three_bit_ordinals() {
        let address = CartesianAddress12::new(1, 2, 3, 4);
        assert_eq!(address.raw(), 0x29c);
        assert_eq!(address.ordinals(), [1, 2, 3, 4]);
    }

    #[test]
    fn sparse_cartesian_visits_only_live_coordinates_in_order() {
        let q = Quad8::new(
            0b0000_0101, // a = {0,2}
            0b0000_1010, // b = {1,3}
            0b0001_0000, // c = {4}
            0b1000_0001, // d = {0,7}
        );
        assert_eq!(q.occupied_len(), 8);

        let mut seen = Vec::new();
        q.for_each_cartesian(|address| seen.push(address.ordinals()));

        assert_eq!(
            seen,
            vec![
                [0, 1, 4, 0],
                [0, 1, 4, 7],
                [0, 3, 4, 0],
                [0, 3, 4, 7],
                [2, 1, 4, 0],
                [2, 1, 4, 7],
                [2, 3, 4, 0],
                [2, 3, 4, 7],
            ]
        );
    }

    #[test]
    fn dense_cartesian_is_4096_addresses_without_a_4096_cell_object() {
        let q = Quad8::new(0xff, 0xff, 0xff, 0xff);
        assert_eq!(q.occupied_len(), 4096);

        let mut count = 0usize;
        let mut first = None;
        let mut last = None;
        q.for_each_cartesian(|address| {
            first.get_or_insert(address);
            last = Some(address);
            count += 1;
        });

        assert_eq!(count, 4096);
        assert_eq!(first.unwrap().raw(), 0);
        assert_eq!(last.unwrap().raw(), 4095);
    }

    #[test]
    fn fold_consumes_coordinates_directly() {
        let q = Quad8::new(0b0000_0011, 0b0000_0001, 0b0000_0001, 0b0000_0011);
        let sum = q.fold_cartesian(0u32, |acc, address| acc + address.raw() as u32);

        let expected = CartesianAddress12::new(0, 0, 0, 0).raw() as u32
            + CartesianAddress12::new(0, 0, 0, 1).raw() as u32
            + CartesianAddress12::new(1, 0, 0, 0).raw() as u32
            + CartesianAddress12::new(1, 0, 0, 1).raw() as u32;

        assert_eq!(sum, expected);
    }
}
