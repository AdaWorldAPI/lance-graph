//! The HHTL / NARS / Moore value tenants: byte access and the MooreNars16 reading.
//!
//! Three append-only value tenants of the 480-byte slab, one lane per
//! [`MooreSlot`] in the canonical order `NW, N, NE, W, E, SW, S, SE`:
//!
//! | tenant | lane | reading |
//! |---|---|---|
//! | [`ValueTenant::Nars16x8`] | LE `u16` | CANDIDATE, not declared here |
//! | [`ValueTenant::MoorePalettePairs`] | `(u8:u8)` | Palette256 pair, first operand at byte `2i` |
//! | [`ValueTenant::MooreNars16`] | LE `u16` | [`MooreNars16`] |
//!
//! Every byte position comes from [`ValueTenant::value_offset`]; this module
//! writes no literal offset. The accessors borrow the slab and copy nothing
//! but the lane they return.
//!
//! **Status.** The physical layout (ordinals, widths, byte order, slot order)
//! is RATIFIED. The MooreNars16 field reading is CANDIDATE: it is the
//! representation PR #1406 measured as observationally equivalent to CE64
//! under the shipped ISA operations, and it carries two refusals that a
//! consumer must honour:
//!
//! - **Direction.** A Moore lane's direction is `(slot, polarity)`. It has no
//!   S/P/O sign triple, so [`DirectionReading::require_sign_triple`] refuses it.
//! - **Witness.** One witness covers all eight lanes. It is held by the owner,
//!   not stored in these bytes, and [`MooreTenantMut::lift_ce64`] refuses
//!   input whose lanes carry different witnesses.
//!
//! This module adds no truth formula and changes no CE64 field.

use crate::canonical_node::{ValueTenant, VALUE_SLAB_LEN};
use crate::epistemic_state5::{Epi5Gen, EpistemicState5};

/// Number of lanes in each Moore tenant.
pub const MOORE_LANES: usize = 8;

/// One of the eight Moore neighbours, in canonical lane order.
///
/// The order is the one the Moore probes use (`moore_plasticity_probe`,
/// `moore_nars16_probe`): row by row from the north-west, skipping the centre.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MooreSlot {
    /// `(-1, -1)`
    Nw = 0,
    /// `(0, -1)`
    N = 1,
    /// `(1, -1)`
    Ne = 2,
    /// `(-1, 0)`
    W = 3,
    /// `(1, 0)`
    E = 4,
    /// `(-1, 1)`
    Sw = 5,
    /// `(0, 1)`
    S = 6,
    /// `(1, 1)`
    Se = 7,
}

impl MooreSlot {
    /// Every slot in lane order. `ALL[i] as usize == i`.
    pub const ALL: [MooreSlot; MOORE_LANES] = [
        MooreSlot::Nw,
        MooreSlot::N,
        MooreSlot::Ne,
        MooreSlot::W,
        MooreSlot::E,
        MooreSlot::Sw,
        MooreSlot::S,
        MooreSlot::Se,
    ];

    /// The slot at lane `index`, or `None` past the eighth lane.
    #[must_use]
    pub const fn from_index(index: usize) -> Option<Self> {
        if index < MOORE_LANES {
            Some(Self::ALL[index])
        } else {
            None
        }
    }

    /// The lane index of this slot.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The neighbour's `(dx, dy)` offset from the centre, `y` growing south.
    #[must_use]
    pub const fn offset(self) -> (i8, i8) {
        match self {
            MooreSlot::Nw => (-1, -1),
            MooreSlot::N => (0, -1),
            MooreSlot::Ne => (1, -1),
            MooreSlot::W => (-1, 0),
            MooreSlot::E => (1, 0),
            MooreSlot::Sw => (-1, 1),
            MooreSlot::S => (0, 1),
            MooreSlot::Se => (1, 1),
        }
    }
}

/// A Moore-local reading of the ISA-visible CE64 state of one lane.
///
/// ```text
/// bits  0..2   Pearl3       (CE64 40..42)
/// bits  3..6   Energy4      (CE64 46..49, the raw signed-mantissa nibble)
/// bits  7..9   Plasticity3  (CE64 50..52)
/// bit   10     Polarity1    (Moore-local; no CE64 home)
/// bits 11..15  Epi5         (CE64 59..63; codes 24..31 reserved)
/// ```
///
/// CE64 bits 43..45 (the S/P/O sign triple) and 53..58 (the witness) are not
/// here: direction is `(slot, polarity)` and the witness is tenant-scoped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct MooreNars16(pub u16);

impl MooreNars16 {
    const PEARL_SHIFT: u32 = 0;
    const ENERGY_SHIFT: u32 = 3;
    const PLASTICITY_SHIFT: u32 = 7;
    const POLARITY_SHIFT: u32 = 10;
    const EPI5_SHIFT: u32 = 11;

    /// Pack the five fields. Each input is masked to its width.
    #[must_use]
    pub const fn new(pearl: u8, energy: u8, plasticity: u8, polarity: bool, epi5: u8) -> Self {
        Self(
            ((pearl as u16 & 0x7) << Self::PEARL_SHIFT)
                | ((energy as u16 & 0xF) << Self::ENERGY_SHIFT)
                | ((plasticity as u16 & 0x7) << Self::PLASTICITY_SHIFT)
                | ((polarity as u16) << Self::POLARITY_SHIFT)
                | ((epi5 as u16 & 0x1F) << Self::EPI5_SHIFT),
        )
    }

    /// Pearl mask, CE64 bits 40..42.
    #[must_use]
    pub const fn pearl(self) -> u8 {
        ((self.0 >> Self::PEARL_SHIFT) & 0x7) as u8
    }

    /// The raw signed-mantissa nibble, CE64 bits 46..49.
    #[must_use]
    pub const fn energy(self) -> u8 {
        ((self.0 >> Self::ENERGY_SHIFT) & 0xF) as u8
    }

    /// Plasticity triad, CE64 bits 50..52.
    #[must_use]
    pub const fn plasticity(self) -> u8 {
        ((self.0 >> Self::PLASTICITY_SHIFT) & 0x7) as u8
    }

    /// `false` = outbound (centre to neighbour), `true` = inbound.
    #[must_use]
    pub const fn polarity(self) -> bool {
        (self.0 >> Self::POLARITY_SHIFT) & 1 != 0
    }

    /// The raw 5-bit epistemic code, CE64 bits 59..63.
    #[must_use]
    pub const fn epi5_raw(self) -> u8 {
        (self.0 >> Self::EPI5_SHIFT) as u8
    }

    /// The epistemic state, refusing the reserved codes 24..31.
    ///
    /// # Errors
    ///
    /// [`MooreRefusal::ReservedEpi5`] for a code outside the canonical codebook.
    pub const fn epistemic(self, slot: MooreSlot) -> Result<EpistemicState5, MooreRefusal> {
        match EpistemicState5::decode(Epi5Gen::V1, self.epi5_raw()) {
            Ok(s) => Ok(s),
            Err(_) => Err(MooreRefusal::ReservedEpi5 {
                slot,
                code: self.epi5_raw(),
            }),
        }
    }

    /// How this lane's direction is read: `(slot, polarity)`.
    #[must_use]
    pub const fn direction(self, slot: MooreSlot) -> DirectionReading {
        DirectionReading::Moore {
            slot,
            polarity: self.polarity(),
        }
    }

    /// Read the lane from a CE64 register. Direction (43..45) and witness
    /// (53..58) are not read; `polarity` supplies the Moore-local bit.
    #[must_use]
    pub const fn from_ce64(edge: u64, polarity: bool) -> Self {
        Self::new(
            ((edge >> 40) & 0x7) as u8,
            ((edge >> 46) & 0xF) as u8,
            ((edge >> 50) & 0x7) as u8,
            polarity,
            ((edge >> 59) & 0x1F) as u8,
        )
    }

    /// CE64 bits 40..63 for this lane under `witness`. The direction bits
    /// (43..45) are zero: the Moore reading carries no sign triple. Bits
    /// 0..39 (S/P/O, F, C) are the caller's.
    #[must_use]
    pub const fn ce64_upper(self, witness: u8) -> u64 {
        ((self.pearl() as u64) << 40)
            | ((self.energy() as u64) << 46)
            | ((self.plasticity() as u64) << 50)
            | (((witness & 0x3F) as u64) << 53)
            | ((self.epi5_raw() as u64) << 59)
    }
}

/// How a CE64 direction is to be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DirectionReading {
    /// Bits 43..45 are the canonical S/P/O sign triple.
    SignTriple(u8),
    /// The direction is the lane's slot and polarity; there is no sign triple.
    Moore {
        /// The lane's neighbour.
        slot: MooreSlot,
        /// `false` = outbound, `true` = inbound.
        polarity: bool,
    },
}

impl DirectionReading {
    /// The S/P/O sign triple, for a consumer that reads one.
    ///
    /// # Errors
    ///
    /// [`DirectionRefusal`] for a Moore-local reading, which has no sign
    /// triple. A sign-triple consumer must not reinterpret it.
    pub const fn require_sign_triple(self) -> Result<u8, DirectionRefusal> {
        match self {
            DirectionReading::SignTriple(bits) => Ok(bits & 0x7),
            DirectionReading::Moore { slot, polarity } => Err(DirectionRefusal { slot, polarity }),
        }
    }
}

/// A sign-triple consumer was handed a Moore-local direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DirectionRefusal {
    /// The lane the refused direction came from.
    pub slot: MooreSlot,
    /// Its polarity.
    pub polarity: bool,
}

/// Why a Moore tenant read or lift was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MooreRefusal {
    /// An Epi5 code outside the canonical codebook (24..31).
    ReservedEpi5 {
        /// The offending lane.
        slot: MooreSlot,
        /// The code it held.
        code: u8,
    },
    /// The lanes of one tenant carry different witnesses.
    MixedWitness {
        /// The first lane that disagrees with lane 0.
        slot: MooreSlot,
        /// Lane 0's witness.
        expected: u8,
        /// That lane's witness.
        found: u8,
    },
}

/// Read access to the three tenants in a value slab.
#[derive(Debug, Clone, Copy)]
pub struct MooreTenantView<'a> {
    value: &'a [u8; VALUE_SLAB_LEN],
}

impl<'a> MooreTenantView<'a> {
    /// Borrow a value slab.
    #[must_use]
    pub const fn new(value: &'a [u8; VALUE_SLAB_LEN]) -> Self {
        Self { value }
    }

    const fn u16_at(&self, tenant: ValueTenant, slot: MooreSlot) -> u16 {
        let at = tenant.value_offset() + 2 * slot.index();
        u16::from_le_bytes([self.value[at], self.value[at + 1]])
    }

    /// The [`ValueTenant::Nars16x8`] lane for `slot`.
    #[must_use]
    pub const fn nars16(&self, slot: MooreSlot) -> u16 {
        self.u16_at(ValueTenant::Nars16x8, slot)
    }

    /// The [`ValueTenant::MoorePalettePairs`] pair for `slot`: `(first, second)`.
    #[must_use]
    pub const fn palette_pair(&self, slot: MooreSlot) -> (u8, u8) {
        let at = ValueTenant::MoorePalettePairs.value_offset() + 2 * slot.index();
        (self.value[at], self.value[at + 1])
    }

    /// The [`ValueTenant::MooreNars16`] lane for `slot`.
    #[must_use]
    pub const fn moore_nars16(&self, slot: MooreSlot) -> MooreNars16 {
        MooreNars16(self.u16_at(ValueTenant::MooreNars16, slot))
    }
}

/// Write access to the three tenants in a value slab. Each setter touches
/// only its own lane's bytes.
#[derive(Debug)]
pub struct MooreTenantMut<'a> {
    value: &'a mut [u8; VALUE_SLAB_LEN],
}

impl<'a> MooreTenantMut<'a> {
    /// Borrow a value slab mutably.
    pub fn new(value: &'a mut [u8; VALUE_SLAB_LEN]) -> Self {
        Self { value }
    }

    /// Read access to the same slab.
    #[must_use]
    pub fn view(&self) -> MooreTenantView<'_> {
        MooreTenantView::new(self.value)
    }

    fn set_u16(&mut self, tenant: ValueTenant, slot: MooreSlot, v: u16) {
        let at = tenant.value_offset() + 2 * slot.index();
        self.value[at..at + 2].copy_from_slice(&v.to_le_bytes());
    }

    /// Write the [`ValueTenant::Nars16x8`] lane for `slot`.
    pub fn set_nars16(&mut self, slot: MooreSlot, v: u16) {
        self.set_u16(ValueTenant::Nars16x8, slot, v);
    }

    /// Write the [`ValueTenant::MoorePalettePairs`] pair for `slot`.
    pub fn set_palette_pair(&mut self, slot: MooreSlot, first: u8, second: u8) {
        let at = ValueTenant::MoorePalettePairs.value_offset() + 2 * slot.index();
        self.value[at] = first;
        self.value[at + 1] = second;
    }

    /// Write the [`ValueTenant::MooreNars16`] lane for `slot`.
    pub fn set_moore_nars16(&mut self, slot: MooreSlot, lane: MooreNars16) {
        self.set_u16(ValueTenant::MooreNars16, slot, lane.0);
    }

    /// Project eight CE64 registers into the [`ValueTenant::MooreNars16`]
    /// lanes and return their common witness.
    ///
    /// Everything is checked before anything is written, so a refusal leaves
    /// the slab unchanged.
    ///
    /// # Errors
    ///
    /// - [`MooreRefusal::MixedWitness`] when the lanes' witnesses differ.
    /// - [`MooreRefusal::ReservedEpi5`] when a lane holds Epi5 code 24..31.
    pub fn lift_ce64(
        &mut self,
        edges: &[u64; MOORE_LANES],
        polarity: [bool; MOORE_LANES],
    ) -> Result<u8, MooreRefusal> {
        let witness = ((edges[0] >> 53) & 0x3F) as u8;
        let mut lanes = [MooreNars16(0); MOORE_LANES];
        for slot in MooreSlot::ALL {
            let edge = edges[slot.index()];
            let w = ((edge >> 53) & 0x3F) as u8;
            if w != witness {
                return Err(MooreRefusal::MixedWitness {
                    slot,
                    expected: witness,
                    found: w,
                });
            }
            let lane = MooreNars16::from_ce64(edge, polarity[slot.index()]);
            lane.epistemic(slot)?;
            lanes[slot.index()] = lane;
        }
        for slot in MooreSlot::ALL {
            self.set_moore_nars16(slot, lanes[slot.index()]);
        }
        Ok(witness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical_node::{ValueSchema, VALUE_TENANTS};
    use crate::soa_envelope::ColumnKind;

    const NEW: [ValueTenant; 3] = [
        ValueTenant::Nars16x8,
        ValueTenant::MoorePalettePairs,
        ValueTenant::MooreNars16,
    ];

    /// A. The three tenants sit at ordinals 18..20, row bytes 284..332, LE
    /// u16 / u8 kinds, and in `Full`.
    #[test]
    fn a_layout() {
        let expect = [
            (ValueTenant::Nars16x8, 18, ColumnKind::U16, 8, 284),
            (ValueTenant::MoorePalettePairs, 19, ColumnKind::U8, 16, 300),
            (ValueTenant::MooreNars16, 20, ColumnKind::U16, 8, 316),
        ];
        for (t, ord, kind, elems, row) in expect {
            assert_eq!(t as usize, ord);
            let c = &VALUE_TENANTS[ord];
            assert_eq!(c.name_id as usize, ord);
            assert_eq!(c.kind, kind);
            assert_eq!(c.elems_per_row, elems);
            assert_eq!(c.row_offset as usize, row);
            assert_eq!(t.byte_len(), 16);
            assert_eq!(t.value_offset(), row - 32);
            assert!(ValueSchema::Full.has(t));
            for p in [
                ValueSchema::Bootstrap,
                ValueSchema::Cognitive,
                ValueSchema::Compressed,
            ] {
                assert!(!p.has(t), "{p:?} must not carry {t:?}");
            }
        }
        assert_eq!(VALUE_TENANTS.len(), 21);
        assert_eq!(ValueSchema::Full.tenant_bytes(), 300);
    }

    /// B. Tenants 0..17 keep their exact descriptors.
    #[test]
    fn b_existing_tenants_preserved() {
        let pinned: [(ColumnKind, u16, u32); 18] = [
            (ColumnKind::U64, 1, 32),
            (ColumnKind::U64, 1, 40),
            (ColumnKind::U64, 4, 48),
            (ColumnKind::U8, 32, 80),
            (ColumnKind::U8, 6, 112),
            (ColumnKind::U8, 16, 118),
            (ColumnKind::F32, 1, 134),
            (ColumnKind::U32, 1, 138),
            (ColumnKind::U16, 1, 142),
            (ColumnKind::U64, 1, 144),
            (ColumnKind::U8, 12, 152),
            (ColumnKind::U8, 12, 164),
            (ColumnKind::U8, 12, 176),
            (ColumnKind::U8, 16, 188),
            (ColumnKind::U8, 16, 204),
            (ColumnKind::U8, 32, 220),
            (ColumnKind::U8, 16, 252),
            (ColumnKind::U8, 16, 268),
        ];
        for (i, (kind, elems, row)) in pinned.into_iter().enumerate() {
            let c = &VALUE_TENANTS[i];
            assert_eq!(c.name_id as usize, i);
            assert_eq!(
                (c.kind, c.elems_per_row, c.row_offset),
                (kind, elems, row),
                "tenant {i}"
            );
        }
    }

    fn patterned() -> [u8; VALUE_SLAB_LEN] {
        let mut v = [0u8; VALUE_SLAB_LEN];
        for (i, b) in v.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37).wrapping_add(11);
        }
        v
    }

    /// C. Writing one lane changes exactly that lane's two bytes.
    #[test]
    fn c_new_tenant_isolation() {
        for t in NEW {
            for slot in MooreSlot::ALL {
                let before = patterned();
                let mut after = before;
                {
                    let mut m = MooreTenantMut::new(&mut after);
                    let pick = before[t.value_offset() + 2 * slot.index()] ^ 0xFF;
                    match t {
                        ValueTenant::Nars16x8 => m.set_nars16(slot, u16::from(pick) * 257),
                        ValueTenant::MoorePalettePairs => m.set_palette_pair(slot, pick, pick),
                        _ => m.set_moore_nars16(slot, MooreNars16(u16::from(pick) * 257)),
                    }
                }
                let lane = t.value_offset() + 2 * slot.index();
                for i in 0..VALUE_SLAB_LEN {
                    if i == lane || i == lane + 1 {
                        assert_ne!(before[i], after[i], "{t:?} {slot:?} byte {i} not written");
                    } else {
                        assert_eq!(before[i], after[i], "{t:?} {slot:?} wrote byte {i}");
                    }
                }
            }
        }
    }

    /// D. Little-endian round trip with the fixed fixtures.
    #[test]
    fn d_le_round_trip() {
        for v in [0x0102u16, 0x1234, 0xABCD, 0xFF00] {
            for slot in MooreSlot::ALL {
                let mut slab = [0u8; VALUE_SLAB_LEN];
                let mut m = MooreTenantMut::new(&mut slab);
                m.set_nars16(slot, v);
                m.set_moore_nars16(slot, MooreNars16(v));
                m.set_palette_pair(slot, (v >> 8) as u8, v as u8);
                let r = m.view();
                assert_eq!(r.nars16(slot), v);
                assert_eq!(r.moore_nars16(slot), MooreNars16(v));
                assert_eq!(r.palette_pair(slot), ((v >> 8) as u8, v as u8));
                let at = |t: ValueTenant| t.value_offset() + 2 * slot.index();
                let le = v.to_le_bytes();
                assert_eq!(slab[at(ValueTenant::Nars16x8)..][..2], le);
                assert_eq!(slab[at(ValueTenant::MooreNars16)..][..2], le);
                // The pair is two bytes in operand order, NOT a LE u16.
                assert_eq!(
                    slab[at(ValueTenant::MoorePalettePairs)..][..2],
                    [(v >> 8) as u8, v as u8]
                );
            }
        }
    }

    /// E. Slot identity: lane i is slot i, the offsets are the eight Moore
    /// neighbours in row order, and lane i lives at byte 2i of each tenant.
    #[test]
    fn e_moore_slot_identity() {
        let order = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ];
        for (i, slot) in MooreSlot::ALL.into_iter().enumerate() {
            assert_eq!(slot.index(), i);
            assert_eq!(MooreSlot::from_index(i), Some(slot));
            assert_eq!(slot.offset(), order[i]);
        }
        assert_eq!(MooreSlot::from_index(8), None);
        for t in NEW {
            for slot in MooreSlot::ALL {
                let mut slab = [0u8; VALUE_SLAB_LEN];
                let at = t.value_offset() + 2 * slot.index();
                slab[at] = 0x5A;
                slab[at + 1] = 0xA5;
                let r = MooreTenantView::new(&slab);
                for other in MooreSlot::ALL {
                    let hit = match t {
                        ValueTenant::Nars16x8 => r.nars16(other) != 0,
                        ValueTenant::MoorePalettePairs => r.palette_pair(other) != (0, 0),
                        _ => r.moore_nars16(other).0 != 0,
                    };
                    assert_eq!(
                        hit,
                        other == slot,
                        "{t:?}: byte 2*{} read by {other:?}",
                        slot.index()
                    );
                }
            }
        }
    }

    /// G. Exhaustive: every u16 decodes to five fields that re-encode to the
    /// same u16, and every field combination round-trips.
    #[test]
    fn g_exhaustive_moore_nars16_fields() {
        for raw in 0..=u16::MAX {
            let l = MooreNars16(raw);
            let back = MooreNars16::new(
                l.pearl(),
                l.energy(),
                l.plasticity(),
                l.polarity(),
                l.epi5_raw(),
            );
            assert_eq!(back, l, "{raw:#06x}");
        }
        let mut seen = 0u32;
        for pearl in 0..8u8 {
            for energy in 0..16u8 {
                for plast in 0..8u8 {
                    for pol in [false, true] {
                        for epi in 0..32u8 {
                            let l = MooreNars16::new(pearl, energy, plast, pol, epi);
                            assert_eq!(
                                (
                                    l.pearl(),
                                    l.energy(),
                                    l.plasticity(),
                                    l.polarity(),
                                    l.epi5_raw()
                                ),
                                (pearl, energy, plast, pol, epi)
                            );
                            seen += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(seen, 1 << 16, "the five fields tile the u16 exactly");
        // Reserved Epi5 codes are refused; declared ones are not.
        for epi in 0..32u8 {
            let r = MooreNars16::new(0, 0, 0, false, epi).epistemic(MooreSlot::E);
            assert_eq!(r.is_err(), epi >= 24, "epi5 {epi}");
        }
    }

    /// G (CE64 side). `from_ce64` reads exactly bits 40..42, 46..52 and
    /// 59..63, and `ce64_upper` writes them back with W and zero direction.
    #[test]
    fn g_ce64_bit_mapping() {
        for bit in 0..64u32 {
            let lane = MooreNars16::from_ce64(1u64 << bit, false);
            let read = matches!(bit, 40..=42 | 46..=52 | 59..=63);
            assert_eq!(lane.0 != 0, read, "CE64 bit {bit}");
        }
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..4096 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let w = ((x >> 53) & 0x3F) as u8;
            let upper = MooreNars16::from_ce64(x, true).ce64_upper(w);
            let keep = !((1u64 << 40) - 1) & !(0b111u64 << 43);
            assert_eq!(upper, x & keep);
        }
    }

    /// H. A Moore direction is refused by a sign-triple consumer; a sign
    /// triple is passed through.
    #[test]
    fn h_direction_refusal() {
        for slot in MooreSlot::ALL {
            for pol in [false, true] {
                let d = MooreNars16::new(0, 0, 0, pol, 0).direction(slot);
                assert_eq!(
                    d,
                    DirectionReading::Moore {
                        slot,
                        polarity: pol
                    }
                );
                assert_eq!(
                    d.require_sign_triple(),
                    Err(DirectionRefusal {
                        slot,
                        polarity: pol
                    })
                );
            }
        }
        for bits in 0..8u8 {
            assert_eq!(
                DirectionReading::SignTriple(bits).require_sign_triple(),
                Ok(bits)
            );
        }
    }

    fn edge(w: u8, epi: u8, salt: u64) -> u64 {
        let base = salt.wrapping_mul(0x9E37_79B9_7F4A_7C15) & !(0x3Fu64 << 53) & !(0x1Fu64 << 59);
        base | (u64::from(w & 0x3F) << 53) | (u64::from(epi & 0x1F) << 59)
    }

    /// I. Homogeneous witness lifts and returns W; any mixed lane or reserved
    /// Epi5 is refused, and a refusal writes nothing.
    #[test]
    fn i_witness_ownership() {
        let pol = [false, true, false, true, true, false, true, false];
        for w in [0u8, 1, 37, 63] {
            let edges: [u64; 8] =
                core::array::from_fn(|k| edge(w, (k as u8 * 3) % 24, k as u64 + 1));
            let mut slab = [0u8; VALUE_SLAB_LEN];
            let got = MooreTenantMut::new(&mut slab).lift_ce64(&edges, pol);
            assert_eq!(got, Ok(w));
            let r = MooreTenantView::new(&slab);
            for slot in MooreSlot::ALL {
                assert_eq!(
                    r.moore_nars16(slot),
                    MooreNars16::from_ce64(edges[slot.index()], pol[slot.index()])
                );
            }
            for k in 1..8 {
                let mut mixed = edges;
                let other = (w + 1) & 0x3F;
                mixed[k] = edge(other, 0, 99);
                let mut slab = patterned();
                let before = slab;
                let got = MooreTenantMut::new(&mut slab).lift_ce64(&mixed, pol);
                assert_eq!(
                    got,
                    Err(MooreRefusal::MixedWitness {
                        slot: MooreSlot::ALL[k],
                        expected: w,
                        found: other
                    })
                );
                assert_eq!(slab, before, "a refused lift must not write");
            }
            let mut reserved = edges;
            reserved[7] = edge(w, 24, 7);
            let mut slab = patterned();
            let before = slab;
            assert_eq!(
                MooreTenantMut::new(&mut slab).lift_ce64(&reserved, pol),
                Err(MooreRefusal::ReservedEpi5 {
                    slot: MooreSlot::Se,
                    code: 24
                })
            );
            assert_eq!(slab, before);
        }
    }

    /// K. An all-zero slab reads as zero lanes, null pairs, and a declared
    /// (not reserved) Epi5 code.
    #[test]
    fn k_zero_fallback() {
        let slab = [0u8; VALUE_SLAB_LEN];
        let r = MooreTenantView::new(&slab);
        for slot in MooreSlot::ALL {
            assert_eq!(r.nars16(slot), 0);
            assert_eq!(r.palette_pair(slot), (0, 0));
            let l = r.moore_nars16(slot);
            assert_eq!(l, MooreNars16::default());
            assert!(l.epistemic(slot).is_ok());
            assert!(!l.polarity());
        }
    }

    /// L. Each codec mutation is caught by the exhaustive round trip. Pins
    /// that every field width and position is load-bearing.
    #[test]
    fn l_codec_mutations_are_caught() {
        type Enc = fn(u8, u8, u8, bool, u8) -> u16;
        let mutants: [(&str, Enc); 5] = [
            ("energy 3 bits", |p, e, pl, po, ep| {
                MooreNars16::new(p, e & 7, pl, po, ep).0
            }),
            ("plasticity 2 bits", |p, e, pl, po, ep| {
                MooreNars16::new(p, e, pl & 3, po, ep).0
            }),
            ("epi5 dropped", |p, e, pl, po, _| {
                MooreNars16::new(p, e, pl, po, 0).0
            }),
            ("polarity over plasticity", |p, e, pl, po, ep| {
                MooreNars16::new(p, e, (pl & !1) | u8::from(po), false, ep).0
            }),
            ("pearl and plasticity swapped", |p, e, pl, po, ep| {
                MooreNars16::new(pl, e, p, po, ep).0
            }),
        ];
        for (name, enc) in mutants {
            let mut caught = false;
            'all: for p in 0..8u8 {
                for e in 0..16u8 {
                    for pl in 0..8u8 {
                        for po in [false, true] {
                            for ep in 0..32u8 {
                                let l = MooreNars16(enc(p, e, pl, po, ep));
                                if (
                                    l.pearl(),
                                    l.energy(),
                                    l.plasticity(),
                                    l.polarity(),
                                    l.epi5_raw(),
                                ) != (p, e, pl, po, ep)
                                {
                                    caught = true;
                                    break 'all;
                                }
                            }
                        }
                    }
                }
            }
            assert!(caught, "mutation {name} survived the round trip");
        }
    }
}
