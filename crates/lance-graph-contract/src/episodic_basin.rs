// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `episodic_basin` — the LE codec for the [`ValueTenant::EpisodicBasin`] rail
//! (`D-ACR-6`).
//!
//! One home for the 32-byte carve, so a writer and a reader cannot disagree
//! about it. (The cost of two homes was measured elsewhere this same day: an
//! artifact whose tail was composed V3 by its index and read back V1 by its
//! consumer returned every identity 256× too large.)
//!
//! # The row is REFERENCES
//!
//! A basin is a subject's outgoing-object neighborhood. The naive promotion
//! inlines that neighborhood — member names, source text, a per-basin bundle —
//! which is the fat-concept failure `§3a` forbids. This row stores the basin's
//! ADDRESS: the members are reached by following `(subject, [from, to))` into
//! the triple stream.
//!
//! Width is deliberately absent: it is recomputable from the members those
//! references reach, so storing it would be a second projection of data the row
//! already addresses. [`self_code`](BasinRow::self_code) IS stored, because it
//! is the basin's own identity at a strictly higher rung than its inputs — not
//! a cached member.
//!
//! # Where the HHTL position is — and is NOT
//!
//! **Not here.** A basin row is a node, and a node's cascade position lives in
//! its own KEY (`HEEL`/`HIP`/`TWIG`, key bytes `4..10`) — the canon's "the key
//! prerenders nodes with zero value decode". Putting a `NiblePath` in the value
//! slab would be a second home for an address the key already carries.
//!
//! What that implies is a **precondition on the promotion, not a field**, and
//! it is asymmetric between the two corpora this rail serves:
//!
//! - **Ontologies:** the `part_of:is_a` rails are already hydrated nodes, so
//!   the anchor a basin attaches to exists before the basin does.
//! - **Books:** the HHTL tree does NOT exist and must be **spawned first** —
//!   the table of contents minted as the full tree skeleton, an SoA node per
//!   entry, *before* any triple-level reasoning runs. A basin promoted against
//!   an unspawned book tree has no ancestor to ascend to, which is exactly the
//!   gap `D-ACR-12`'s ascent primitive would fall into.
//!
//! So: same rail, same bytes, different readiness. A promoter must check that
//! the anchor node exists; for books that check is "has the TOC been spawned".
//!
//! ## What this section does NOT say (operator, 2026-09-01)
//!
//! The ruling above is about the ADDRESS and nothing else. Read as a statement
//! about the NODE it is wrong, and it was in fact misread that way once — a
//! planning survey turned "the position lives in the key" into "the node is
//! key-only", which would forbid giving an HHTL position a value at all.
//!
//! An HHTL node is a node: `key(16) | edges(16) | value(480)`. The plan is for
//! HHTL positions to BE SoA rows whose value slab carries a self-organizing
//! summary of the position's children (upstream/downstream inheritance, basin
//! agreement, disagreement, missing links). Nothing here forbids that; what is
//! forbidden is a second copy of the cascade POSITION, which the key holds.
//!
//! The two halves do not compete, and the reason is mechanical: a summary
//! changes when its children change, and a key is an identity — a mutable
//! summary in the key would re-address the node on every sweep. Address in the
//! key, summary in a value lane, necessarily.
//!
//! ## A third readiness state
//!
//! The two corpora above are not the whole ladder. A position can also be
//! **implicit in the rails but not hydrated**: a `part_of`/`is_a` edge names
//! the position, so it is not absent, yet no node was ever hydrated there, so
//! it is not present either. That is the state a node-level hydrate step would
//! consume, and no such step exists in this tree (`lance-graph-hydrate` is
//! artifact-level — an object store to a local volume — and shares only the
//! vocabulary).

/// Bytes one basin occupies on the rail.
pub const BASIN_ROW_BYTES: usize = 32;

/// Length of the Cam96 centroid self-code, in bytes (`6×(u8:u8)`).
pub const SELF_CODE_BYTES: usize = 12;

/// One promoted basin, as it lives in the [`ValueTenant::EpisodicBasin`] lane.
///
/// Every field is a reference or a count; nothing here is a copy of anything
/// the triple stream holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BasinRow {
    /// Vocabulary reference: the subject that anchors this neighborhood.
    pub subject: u16,
    /// How many members the neighborhood has — a guard, never the members.
    pub member_count: u16,
    /// The basin's own Cam96 centroid — its identity, not a cached member.
    pub self_code: [u8; SELF_CODE_BYTES],
    /// Inclusive lower bound of the version range the members live in.
    pub version_from: u64,
    /// Exclusive upper bound.
    pub version_to: u64,
}

impl BasinRow {
    /// The all-zero row: **no basin promoted**. Zero-fallback, matching the
    /// canon's ladder — an unwritten lane reads as absent, never as a basin
    /// over nothing.
    pub const EMPTY: Self = Self {
        subject: 0,
        member_count: 0,
        self_code: [0; SELF_CODE_BYTES],
        version_from: 0,
        version_to: 0,
    };

    /// Is this lane unwritten?
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == Self::EMPTY
    }

    /// Encode to the lane's little-endian carve.
    #[must_use]
    pub fn to_le_bytes(&self) -> [u8; BASIN_ROW_BYTES] {
        let mut b = [0u8; BASIN_ROW_BYTES];
        b[0..2].copy_from_slice(&self.subject.to_le_bytes());
        b[2..4].copy_from_slice(&self.member_count.to_le_bytes());
        b[4..16].copy_from_slice(&self.self_code);
        b[16..24].copy_from_slice(&self.version_from.to_le_bytes());
        b[24..32].copy_from_slice(&self.version_to.to_le_bytes());
        b
    }

    /// Decode from the lane's little-endian carve — total, no failure mode:
    /// every 32-byte pattern is a representable row (an all-zero one being
    /// [`EMPTY`](Self::EMPTY)).
    #[must_use]
    pub fn from_le_bytes(b: &[u8; BASIN_ROW_BYTES]) -> Self {
        let mut self_code = [0u8; SELF_CODE_BYTES];
        self_code.copy_from_slice(&b[4..16]);
        Self {
            subject: u16::from_le_bytes([b[0], b[1]]),
            member_count: u16::from_le_bytes([b[2], b[3]]),
            self_code,
            version_from: u64::from_le_bytes(b[16..24].try_into().expect("8 bytes")),
            version_to: u64::from_le_bytes(b[24..32].try_into().expect("8 bytes")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical_node::{ValueTenant, VALUE_TENANTS};

    /// **The codec matches the DESCRIPTOR, not a second literal.** If the lane
    /// is ever re-widened, this fails rather than letting the codec write past
    /// its column into whatever lands next.
    #[test]
    fn the_carve_is_the_one_the_tenant_table_declares() {
        let d = VALUE_TENANTS[ValueTenant::EpisodicBasin as usize];
        assert_eq!(
            d.col_bytes_per_row(),
            BASIN_ROW_BYTES,
            "codec width must equal the declared column width"
        );
    }

    /// Round-trip with every field DISTINCT, so a swapped pair cannot pass.
    #[test]
    fn a_row_round_trips_and_no_two_fields_alias() {
        let r = BasinRow {
            subject: 0x1234,
            member_count: 0x5678,
            self_code: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            version_from: 0x0102_0304_0506_0708,
            version_to: 0x1112_1314_1516_1718,
        };
        let back = BasinRow::from_le_bytes(&r.to_le_bytes());
        assert_eq!(back, r);

        // Field isolation: moving ONE field must change only its own bytes.
        let base = r.to_le_bytes();
        for (mutate, span) in [
            (
                Box::new(|x: &mut BasinRow| x.subject ^= 0xFFFF) as Box<dyn Fn(&mut BasinRow)>,
                0..2,
            ),
            (Box::new(|x: &mut BasinRow| x.member_count ^= 0xFFFF), 2..4),
            (Box::new(|x: &mut BasinRow| x.self_code[0] ^= 0xFF), 4..16),
            (
                Box::new(|x: &mut BasinRow| x.version_from ^= u64::MAX),
                16..24,
            ),
            (
                Box::new(|x: &mut BasinRow| x.version_to ^= u64::MAX),
                24..32,
            ),
        ] {
            let mut m = r;
            mutate(&mut m);
            let got = m.to_le_bytes();
            for i in 0..BASIN_ROW_BYTES {
                if span.contains(&i) {
                    continue;
                }
                assert_eq!(got[i], base[i], "byte {i} moved but is outside {span:?}");
            }
            assert_ne!(got[span.start], base[span.start], "the field itself moved");
        }
    }

    /// Zero-fallback: an unwritten lane is ABSENT, not a basin over nothing.
    /// Both halves — the empty row reads empty, and a row with any content does
    /// not.
    #[test]
    fn an_unwritten_lane_reads_as_no_basin() {
        assert!(BasinRow::from_le_bytes(&[0u8; BASIN_ROW_BYTES]).is_empty());
        assert!(BasinRow::EMPTY.is_empty());
        let one = BasinRow {
            subject: 1,
            ..BasinRow::EMPTY
        };
        assert!(!one.is_empty(), "a real subject is not an absent basin");
    }
}
