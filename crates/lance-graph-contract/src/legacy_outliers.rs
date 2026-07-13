//! Legacy outlier carvings — the strongly-discouraged V3 migration waiting room.
//!
//! The V3 facet's 12-byte payload is a **content-blind register** the ClassView
//! projects. The sanctioned readings ([`crate::facet::CascadeShape`] +
//! `le-contract.md §3` L1–L8) are all **byte-axis** projections: every tier is a
//! *byte* (`6×(8:8)` rails / `4×(8:8:8)` triplets / `3×(8:8:8:8)` quads), so the
//! classview projects real rails and `group_of` is a pure shift.
//!
//! This module is the **opposite** — the three **wide contiguous** carvings a
//! class may sit in *temporarily* when it has **not yet** decomposed into
//! byte-axis tiles. They are still exactly 96 bits, still one content-blind
//! register — but they carry **no axis, no rail, no shift-addressable group**,
//! which is precisely why they are outliers and why [`CascadeShape`] refuses to
//! bless them (this module exists *because* they must not become `CascadeShape`
//! variants).
//!
//! [`CascadeShape`]: crate::facet::CascadeShape
//!
//! # STRONGLY DISCOURAGED
//!
//! A carving here is a smell, tagged by the operator ruling (2026-07-13,
//! `le-contract.md §3a`) with its own diagnosis:
//!
//! - **god-object-related** → a wide field means the class carries too many
//!   concerns; you owe a **decomposition** (split the class / focus the lens),
//!   never a wider field.
//! - **lacking proper bucket rollover** → a wide contiguous field with no HHTL
//!   cascade spill has nothing to overflow *into*; it saturates silently. Give
//!   it rollover, or narrow it.
//! - **the exit** → the real destination is the L4 `6×(8:8)` **palette256²**
//!   cosine-replacement (each byte pair indexes the 256×256 palette LUT), the
//!   axis-grouped shape the wide field is standing in for.
//!
//! This is **not** a revival of the V1 *tail* model. There is no path/tail
//! split; this is one content-blind register read coarsely. The V1
//! `family:identity` u24 fragment is simply the degenerate [`WideMixed`] /
//! [`WideTriple`] case, given a legal V3 home during migration instead of a
//! crash. **New classes MUST NOT be born into these carvings.** The waiting room
//! is not a destination.

/// The 12-byte content-blind payload every carving reads (facet bytes `4..16`).
pub const PAYLOAD_LEN: usize = 12;

/// Read a little-endian `u24` from three bytes (as a `u32`, high byte zero).
#[inline]
const fn u24_le(a: u8, b: u8, c: u8) -> u32 {
    (a as u32) | ((b as u32) << 8) | ((c as u32) << 16)
}

/// Write a little-endian `u24` into three bytes. Debug-asserts the 24-bit fit —
/// the same no-silent-truncation guard `NodeGuid::new` uses for its u24 groups.
#[inline]
fn put_u24_le(dst: &mut [u8], v: u32) {
    debug_assert!(v <= 0x00FF_FFFF, "legacy outlier u24 field must fit 24 bits");
    dst[0] = (v & 0xFF) as u8;
    dst[1] = ((v >> 8) & 0xFF) as u8;
    dst[2] = ((v >> 16) & 0xFF) as u8;
}

/// The three sanctioned-but-discouraged wide carvings of the 96-bit payload.
///
/// Selection is the classview's job (slot purity) exactly as for the byte-axis
/// layouts — never by inspecting payload bytes. A carving is chosen here only
/// because the class has not yet been decomposed onto the byte axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyOutlier {
    /// **G1** — `3 × u16 + 2 × u24` (48 + 48 = 96 bit). The coarsest mixed
    /// waiting room; the V1 `family:identity` u24 pair lands in the two u24
    /// fields.
    WideMixed,
    /// **G2** — `4 × u24` (96 bit), contiguous. **NOT** the axis-grouped
    /// `4×(8:8:8)` triplets ([`CascadeShape::G4D3`]) — no per-byte rail.
    ///
    /// [`CascadeShape::G4D3`]: crate::facet::CascadeShape::G4D3
    WideTriple,
    /// **G3** — `3 × u32` (96 bit), contiguous. **NOT** the axis-grouped
    /// `3×(8:8:8:8)` quads ([`CascadeShape::G3D4`]) — no per-byte rail.
    ///
    /// [`CascadeShape::G3D4`]: crate::facet::CascadeShape::G3D4
    WideQuad,
}

impl LegacyOutlier {
    /// Every carving, for exhaustive audits.
    pub const ALL: [LegacyOutlier; 3] = [
        LegacyOutlier::WideMixed,
        LegacyOutlier::WideTriple,
        LegacyOutlier::WideQuad,
    ];

    /// G1 read: `[u16; 3]` (bytes `0..6`) then `[u24 as u32; 2]` (bytes `6..12`).
    #[inline]
    pub const fn read_wide_mixed(p: &[u8; PAYLOAD_LEN]) -> ([u16; 3], [u32; 2]) {
        (
            [
                u16::from_le_bytes([p[0], p[1]]),
                u16::from_le_bytes([p[2], p[3]]),
                u16::from_le_bytes([p[4], p[5]]),
            ],
            [u24_le(p[6], p[7], p[8]), u24_le(p[9], p[10], p[11])],
        )
    }

    /// G2 read: `[u24 as u32; 4]`, little-endian, contiguous.
    #[inline]
    pub const fn read_wide_triple(p: &[u8; PAYLOAD_LEN]) -> [u32; 4] {
        [
            u24_le(p[0], p[1], p[2]),
            u24_le(p[3], p[4], p[5]),
            u24_le(p[6], p[7], p[8]),
            u24_le(p[9], p[10], p[11]),
        ]
    }

    /// G3 read: `[u32; 3]`, little-endian, contiguous.
    #[inline]
    pub const fn read_wide_quad(p: &[u8; PAYLOAD_LEN]) -> [u32; 3] {
        [
            u32::from_le_bytes([p[0], p[1], p[2], p[3]]),
            u32::from_le_bytes([p[4], p[5], p[6], p[7]]),
            u32::from_le_bytes([p[8], p[9], p[10], p[11]]),
        ]
    }

    /// G1 write. Debug-asserts each u24 fits 24 bits.
    #[inline]
    pub fn write_wide_mixed(shorts: [u16; 3], longs: [u32; 2]) -> [u8; PAYLOAD_LEN] {
        let mut p = [0u8; PAYLOAD_LEN];
        p[0..2].copy_from_slice(&shorts[0].to_le_bytes());
        p[2..4].copy_from_slice(&shorts[1].to_le_bytes());
        p[4..6].copy_from_slice(&shorts[2].to_le_bytes());
        put_u24_le(&mut p[6..9], longs[0]);
        put_u24_le(&mut p[9..12], longs[1]);
        p
    }

    /// G2 write. Debug-asserts each u24 fits 24 bits.
    #[inline]
    pub fn write_wide_triple(vals: [u32; 4]) -> [u8; PAYLOAD_LEN] {
        let mut p = [0u8; PAYLOAD_LEN];
        put_u24_le(&mut p[0..3], vals[0]);
        put_u24_le(&mut p[3..6], vals[1]);
        put_u24_le(&mut p[6..9], vals[2]);
        put_u24_le(&mut p[9..12], vals[3]);
        p
    }

    /// G3 write.
    #[inline]
    pub fn write_wide_quad(vals: [u32; 3]) -> [u8; PAYLOAD_LEN] {
        let mut p = [0u8; PAYLOAD_LEN];
        p[0..4].copy_from_slice(&vals[0].to_le_bytes());
        p[4..8].copy_from_slice(&vals[1].to_le_bytes());
        p[8..12].copy_from_slice(&vals[2].to_le_bytes());
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_carving_is_exactly_96_bits() {
        // G1: 3×16 + 2×24 = 48 + 48;  G2: 4×24;  G3: 3×32 — all 96 bit = 12 byte.
        assert_eq!(3 * 16 + 2 * 24, 96);
        assert_eq!(4 * 24, 96);
        assert_eq!(3 * 32, 96);
        assert_eq!(PAYLOAD_LEN * 8, 96);
    }

    #[test]
    fn wide_mixed_round_trips() {
        let p = LegacyOutlier::write_wide_mixed([0x1234, 0x5678, 0x9ABC], [0x11_2233, 0x44_5566]);
        let (shorts, longs) = LegacyOutlier::read_wide_mixed(&p);
        assert_eq!(shorts, [0x1234, 0x5678, 0x9ABC]);
        assert_eq!(longs, [0x11_2233, 0x44_5566]);
    }

    #[test]
    fn wide_triple_round_trips() {
        let p = LegacyOutlier::write_wide_triple([0x00_0001, 0x0A_0B0C, 0xFF_FFFF, 0x12_3456]);
        assert_eq!(
            LegacyOutlier::read_wide_triple(&p),
            [0x00_0001, 0x0A_0B0C, 0xFF_FFFF, 0x12_3456]
        );
    }

    #[test]
    fn wide_quad_round_trips() {
        let p = LegacyOutlier::write_wide_quad([0xDEAD_BEEF, 0x0000_0001, 0xCAFE_BABE]);
        assert_eq!(
            LegacyOutlier::read_wide_quad(&p),
            [0xDEAD_BEEF, 0x0000_0001, 0xCAFE_BABE]
        );
    }

    #[test]
    fn v1_family_identity_fragment_is_the_degenerate_wide_mixed_case() {
        // The retired V1 tail = family(u24) ++ identity(u24) — the two u24
        // fields of G1, the three u16 zero. Proves the "waiting room" claim:
        // an un-migrated V1 unit has a legal home here, not a crash.
        let p = LegacyOutlier::write_wide_mixed([0, 0, 0], [0x00_00AB, 0x00_00CD]);
        let (shorts, longs) = LegacyOutlier::read_wide_mixed(&p);
        assert_eq!(shorts, [0, 0, 0]);
        assert_eq!(longs, [0xAB, 0xCD]);
    }
}
