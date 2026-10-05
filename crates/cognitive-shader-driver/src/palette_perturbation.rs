//! Palette256 binary law and surfel-adjacent perturbation kernel.
//!
//! The hot substrate is already normalized. Runtime code does not reconstruct
//! cosine, Fisher-Z, covariance, or any other continuous representation.
//!
//! One law has the closed shape:
//!
//! ```text
//! Palette256 × Palette256 -> Palette256
//!       u8          u8    ->    u8
//! ```
//!
//! A law is therefore exactly 256 × 256 bytes = 64 KiB.  The pair itself is
//! the 16-bit address; the table returns the next native palette ordinal.
//! Composing laws never widens the live state beyond one byte.
//!
//! The perturbation form used here is:
//!
//! ```text
//! [a,b] -> relation_lut -> r0 : u8
//! [c,d] -> relation_lut -> r1 : u8
//! [r0,r1] -> perturb_lut -> delta : u8
//! ```
//!
//! The two intermediate ordinals are register values.  No 256^4 table and no
//! intermediate population are materialized.

/// Palette256 cardinality.
pub const PALETTE_CARDINALITY: usize = 256;

/// One complete binary Palette256 law: 256 × 256 one-byte results.
pub const PALETTE_LUT_LEN: usize = PALETTE_CARDINALITY * PALETTE_CARDINALITY;

/// Native Palette256 state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct PaletteState(pub u8);

/// A 16-bit address formed directly from two native palette ordinals.
///
/// This is deliberately row-major rather than a geometric Morton address:
/// the pair is an address into a 256 × 256 law table. Geometry belongs to the
/// caller's adjacency topology; the law only consumes the two ordinals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct PairAddress(pub u16);

impl PairAddress {
    #[inline(always)]
    pub const fn new(left: PaletteState, right: PaletteState) -> Self {
        Self(((left.0 as u16) << 8) | right.0 as u16)
    }

    #[inline(always)]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Borrowed 64-KiB Palette256 binary law.
///
/// The table is data, not a hidden runtime metric. Calibration pays the entry
/// tax before the hot path; evaluation is one indexed byte load.
#[derive(Clone, Copy, Debug)]
pub struct PaletteLut<'a> {
    entries: &'a [u8; PALETTE_LUT_LEN],
}

impl<'a> PaletteLut<'a> {
    #[inline]
    pub const fn new(entries: &'a [u8; PALETTE_LUT_LEN]) -> Self {
        Self { entries }
    }

    #[inline(always)]
    pub fn at(self, left: PaletteState, right: PaletteState) -> PaletteState {
        PaletteState(self.entries[PairAddress::new(left, right).index()])
    }

    #[inline(always)]
    pub fn at_address(self, address: PairAddress) -> PaletteState {
        PaletteState(self.entries[address.index()])
    }

    #[inline(always)]
    pub const fn entries(self) -> &'a [u8; PALETTE_LUT_LEN] {
        self.entries
    }
}

/// Two closed Palette256 laws: one relation law and one perturbation law.
///
/// This is the smallest useful "surfel-adjacent perturbation field" shape.
/// Adjacency is supplied by the caller. The kernel owns no geometry and
/// allocates no intermediate field.
#[derive(Clone, Copy, Debug)]
pub struct PalettePerturbation<'a> {
    relation: PaletteLut<'a>,
    perturb: PaletteLut<'a>,
}

impl<'a> PalettePerturbation<'a> {
    #[inline]
    pub const fn new(relation: PaletteLut<'a>, perturb: PaletteLut<'a>) -> Self {
        Self { relation, perturb }
    }

    /// Compose two pair relations into one perturbation ordinal.
    ///
    /// Three LUT reads, two register-resident intermediate bytes, one byte out.
    #[inline(always)]
    pub fn compose4(
        self,
        a: PaletteState,
        b: PaletteState,
        c: PaletteState,
        d: PaletteState,
    ) -> PaletteState {
        let left = self.relation.at(a, b);
        let right = self.relation.at(c, d);
        self.perturb.at(left, right)
    }

    /// Advance one resident perturbation state across one adjacent pair.
    ///
    /// `current` is the field state already resident at the destination. The
    /// caller supplies the two adjacent surfel/palette states. No child plane
    /// or relation population is retained.
    #[inline(always)]
    pub fn hop(
        self,
        current: PaletteState,
        local: PaletteState,
        neighbor: PaletteState,
    ) -> PaletteState {
        let relation = self.relation.at(local, neighbor);
        self.perturb.at(current, relation)
    }

    /// Run a bounded path of adjacent pairs with one live byte of state.
    ///
    /// This is intentionally a slice rather than a geometry type: Moore, hex,
    /// Cartesian, Queen, or another topology decides which pairs are visited.
    /// The kernel only executes the closed Palette256 law.
    #[inline]
    pub fn hops(
        self,
        initial: PaletteState,
        adjacent_pairs: &[(PaletteState, PaletteState)],
    ) -> PaletteState {
        adjacent_pairs
            .iter()
            .fold(initial, |state, &(local, neighbor)| {
                self.hop(state, local, neighbor)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table_from(mut f: impl FnMut(u8, u8) -> u8) -> Box<[u8; PALETTE_LUT_LEN]> {
        let mut v = vec![0u8; PALETTE_LUT_LEN];
        for left in 0u16..=255 {
            for right in 0u16..=255 {
                let addr = PairAddress::new(
                    PaletteState(left as u8),
                    PaletteState(right as u8),
                );
                v[addr.index()] = f(left as u8, right as u8);
            }
        }
        v.into_boxed_slice()
            .try_into()
            .expect("Palette256 law must contain exactly 65,536 bytes")
    }

    #[test]
    fn pair_address_spans_the_full_u16_space() {
        assert_eq!(
            PairAddress::new(PaletteState(0), PaletteState(0)).0,
            0
        );
        assert_eq!(
            PairAddress::new(PaletteState(255), PaletteState(255)).0,
            u16::MAX
        );
        assert_eq!(PALETTE_LUT_LEN, 65_536);
    }

    #[test]
    fn one_lut_read_is_closed_over_palette256() {
        let table = table_from(|a, b| a ^ b);
        let law = PaletteLut::new(&table);
        assert_eq!(
            law.at(PaletteState(0b1010_0001), PaletteState(0b0011_1100)),
            PaletteState(0b1001_1101)
        );
    }

    #[test]
    fn lut_of_lut_is_three_reads_not_a_cross_table() {
        let relation_table = table_from(|a, b| a ^ b);
        let perturb_table = table_from(|a, b| a.wrapping_add(b));
        let kernel = PalettePerturbation::new(
            PaletteLut::new(&relation_table),
            PaletteLut::new(&perturb_table),
        );

        // r0 = 10^3 = 9; r1 = 7^5 = 2; delta = 9+2 = 11.
        assert_eq!(
            kernel.compose4(
                PaletteState(10),
                PaletteState(3),
                PaletteState(7),
                PaletteState(5),
            ),
            PaletteState(11)
        );
    }

    #[test]
    fn twelve_hops_keep_exactly_one_byte_of_live_state() {
        let relation_table = table_from(|a, b| a ^ b);
        let perturb_table = table_from(|state, relation| state.wrapping_add(relation));
        let kernel = PalettePerturbation::new(
            PaletteLut::new(&relation_table),
            PaletteLut::new(&perturb_table),
        );

        let path = [
            (PaletteState(1), PaletteState(2)),
            (PaletteState(2), PaletteState(4)),
            (PaletteState(3), PaletteState(6)),
            (PaletteState(4), PaletteState(8)),
            (PaletteState(5), PaletteState(10)),
            (PaletteState(6), PaletteState(12)),
            (PaletteState(7), PaletteState(14)),
            (PaletteState(8), PaletteState(16)),
            (PaletteState(9), PaletteState(18)),
            (PaletteState(10), PaletteState(20)),
            (PaletteState(11), PaletteState(22)),
            (PaletteState(12), PaletteState(24)),
        ];

        let expected = path.iter().fold(0u8, |state, &(a, b)| {
            state.wrapping_add(a.0 ^ b.0)
        });

        assert_eq!(
            kernel.hops(PaletteState(0), &path),
            PaletteState(expected)
        );
    }
}
