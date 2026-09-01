// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `basin_lanes` — the **24 × i4** agreement-lane reading of the 12-byte
//! content-blind register (`D-DCR-2b`, the field map's value-side carrier).
//!
//! An HHTL position is a node, and the node's VALUE summarises the position's
//! children so the node speaks for itself without a second read. The carrier
//! is the register the node already owns — 12 bytes — read at NIBBLE
//! granularity: 24 signed i4 lanes in `[−8, 7]`. Every shipped
//! [`CascadeShape`](crate::facet::CascadeShape) carves the same register at
//! BYTE granularity (6×2 / 4×3 / 3×4, all 12 units); this is the fourth,
//! nibble-granular reading, and the only signed one.
//!
//! # The sign IS the semantics
//!
//! One lane, three cells: **positive = agreement, negative = disagreement,
//! zero = silence.** The whale case falls out of the carrier itself — a whale
//! disagreeing with the mammal neighbourhood is a NEGATIVE lane, a value on
//! the node, never a removal from the set. And silence stays distinct from
//! denial (`0` vs any negative), the same distinction
//! [`Supports::NoEvidence`](crate::dismech_evidence::Supports) refuses to
//! collapse on the evidence side.
//!
//! # Mechanical hydration writes NOTHING here (operator-ruled, 2026-09-01)
//!
//! Minting a row at a nameable DN (from a book TOC, a rail's parent/child, an
//! OWL hierarchy) is MECHANICAL — structure only. Its epistemic output is
//! exactly [`SILENT`](BasinLanes::SILENT): all 24 lanes zero, the
//! zero-fallback ladder holding one level up. Original causality predicates —
//! the dismech palette, Tarski-precise assertions, these signed lanes — come
//! only from evidence and propagation. A mint that writes a nonzero lane is
//! structure impersonating knowledge; [`is_silent`](BasinLanes::is_silent) is
//! the checkable half of that rule.
//!
//! The lane-filling ARITHMETIC (how a child's stance becomes a signed value,
//! how siblings merge at the parent) is deliberately NOT here — it is
//! unmeasured, and per the plan it is a probe, not a codec. This module is
//! the carrier only.

use crate::atoms::I4x32;

/// Lanes in one register: 12 bytes × 2 nibbles.
pub const BASIN_LANES: usize = 24;

/// Bytes the lanes occupy — the node's content-blind register width.
pub const BASIN_LANE_BYTES: usize = 12;

/// Packed 24-lane signed-i4 agreement register (12 bytes, two lanes per
/// byte). Same nibble codec as [`I4x32`] / `I4x64` at the node's own width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BasinLanes {
    bytes: [u8; BASIN_LANE_BYTES],
}

impl BasinLanes {
    /// Epistemic silence: every lane `0`. The ONLY value mechanical
    /// hydration may leave behind, and what an unwritten register reads as —
    /// absent, never an assertion.
    pub const SILENT: Self = Self {
        bytes: [0; BASIN_LANE_BYTES],
    };

    /// Pack 24 signed lanes, saturating to `[−8, 7]`. Two's-complement
    /// nibble: lane `2k` → low nibble of byte `k`, lane `2k+1` → high.
    #[must_use]
    pub fn pack(lanes: &[i8; BASIN_LANES]) -> Self {
        let mut bytes = [0u8; BASIN_LANE_BYTES];
        for (k, b) in bytes.iter_mut().enumerate() {
            let lo = lanes[2 * k].clamp(-8, 7) as u8 & 0x0F;
            let hi = lanes[2 * k + 1].clamp(-8, 7) as u8 & 0x0F;
            *b = (hi << 4) | lo;
        }
        Self { bytes }
    }

    /// Unpack to signed lanes (sign-extended i4, `[−8, 7]`).
    #[must_use]
    pub fn unpack(&self) -> [i8; BASIN_LANES] {
        let mut out = [0i8; BASIN_LANES];
        for (k, b) in self.bytes.iter().enumerate() {
            out[2 * k] = I4x32::sext4(b & 0x0F);
            out[2 * k + 1] = I4x32::sext4(b >> 4);
        }
        out
    }

    /// The register's raw little-endian bytes, as they sit in the row.
    #[must_use]
    pub const fn to_le_bytes(&self) -> [u8; BASIN_LANE_BYTES] {
        self.bytes
    }

    /// Read a register from its raw bytes — total, no failure mode: every
    /// 12-byte pattern is 24 representable lanes.
    #[must_use]
    pub const fn from_le_bytes(bytes: [u8; BASIN_LANE_BYTES]) -> Self {
        Self { bytes }
    }

    /// Is every lane zero? `true` = epistemic silence — a mechanically
    /// hydrated (or never-written) register. The mint-side guard: a hydration
    /// step must leave this `true`.
    #[must_use]
    pub fn is_silent(&self) -> bool {
        *self == Self::SILENT
    }

    /// How many lanes carry a NEGATIVE value — recorded disagreement. A
    /// disagreeing child is a value here, never a removal from the set.
    #[must_use]
    pub fn disagreement_count(&self) -> usize {
        self.unpack().iter().filter(|&&v| v < 0).count()
    }

    /// How many lanes carry a POSITIVE value — recorded agreement.
    #[must_use]
    pub fn agreement_count(&self) -> usize {
        self.unpack().iter().filter(|&&v| v > 0).count()
    }

    /// **One-hop accumulation** (operator-ruled, 2026-09-01): a parent's
    /// register expresses its DIRECT children accumulated — agreement AND
    /// disagreement — never the grandchildren. Grandchild information reaches
    /// a grandparent only through the child's own accumulated register, one
    /// hop at a time; the global field map is the composition of one-hop
    /// summaries, never a node reaching past its children (the same locality
    /// that makes `FieldMask::inherit` a parent∪delta and the substrate
    /// Markov, `I-SUBSTRATE-MARKOV`).
    ///
    /// Per-lane merge: **saturating signed sum** in `[−8, 7]` — agreement
    /// stacks, disagreement pulls down, the bound is the carrier's own range
    /// (associative/commutative in expectation, the sanctioned bundle shape).
    ///
    /// **Measured limitation, pinned rather than hidden:** in ONE register a
    /// balanced conflict (`+3` child and `−3` child on the same lane) sums to
    /// `0` — indistinguishable from silence. That is the concrete case for
    /// the ruled escape hatch "the nibbles can be expanded if necessary /
    /// multiple 24×i4 further up": a contested-mass register beside the net
    /// register. NOT built here — the multi-register semantics are the
    /// operator's to shape; the collapse is pinned by
    /// `a_balanced_conflict_collapses_to_silence_in_one_register` so the gap
    /// stays loud.
    ///
    /// `accumulate_children(&[])` is [`SILENT`](Self::SILENT) — a childless
    /// position asserts nothing.
    #[must_use]
    pub fn accumulate_children(children: &[Self]) -> Self {
        let mut acc = [0i8; BASIN_LANES];
        for c in children {
            let lanes = c.unpack();
            for (a, v) in acc.iter_mut().zip(lanes.iter()) {
                *a = a.saturating_add(*v).clamp(-8, 7);
            }
        }
        Self::pack(&acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facet::{CascadeShape, CASCADE_UNITS};

    #[test]
    fn round_trips_every_representable_lane_value() {
        for val in -8i8..=7 {
            let lanes = [val; BASIN_LANES];
            assert_eq!(
                BasinLanes::pack(&lanes).unpack(),
                lanes,
                "lane value {val} must round-trip"
            );
        }
        // A mixed pattern too — a uniform fixture cannot see a lane-order
        // swap (lane 2k vs 2k+1 confusion round-trips on uniform input).
        let mut mixed = [0i8; BASIN_LANES];
        for (i, l) in mixed.iter_mut().enumerate() {
            *l = (i as i8 % 16) - 8;
        }
        assert_eq!(BasinLanes::pack(&mixed).unpack(), mixed);
    }

    #[test]
    fn saturates_outside_the_i4_range_instead_of_wrapping() {
        assert_eq!(
            BasinLanes::pack(&[100; BASIN_LANES]).unpack(),
            [7; BASIN_LANES]
        );
        assert_eq!(
            BasinLanes::pack(&[-100; BASIN_LANES]).unpack(),
            [-8; BASIN_LANES]
        );
        // Just outside each bound: a wrapping codec would flip the SIGN here,
        // which for this carrier turns agreement into disagreement — the
        // worst possible corruption, so the boundary is pinned exactly.
        assert_eq!(
            BasinLanes::pack(&[8; BASIN_LANES]).unpack(),
            [7; BASIN_LANES]
        );
        assert_eq!(
            BasinLanes::pack(&[-9; BASIN_LANES]).unpack(),
            [-8; BASIN_LANES]
        );
    }

    #[test]
    fn silence_is_the_default_and_the_zero_register() {
        // Zero-fallback one level up: an unwritten register IS silence.
        assert_eq!(BasinLanes::default(), BasinLanes::SILENT);
        assert!(BasinLanes::from_le_bytes([0; BASIN_LANE_BYTES]).is_silent());
        assert!(BasinLanes::SILENT.is_silent());
        assert_eq!(BasinLanes::SILENT.disagreement_count(), 0);
        assert_eq!(BasinLanes::SILENT.agreement_count(), 0);
    }

    /// The whale case at the carrier level, two-sided: disagreement is a
    /// VALUE (recorded, countable, round-trips) and is distinct from BOTH
    /// silence and agreement. A carrier that collapsed `−` into `0` (or into
    /// removal) fails all three arms.
    #[test]
    fn a_negative_lane_is_recorded_disagreement_not_silence_and_not_removal() {
        let mut lanes = [1i8; BASIN_LANES]; // a mammal neighbourhood agreeing
        lanes[3] = -5; // the whale
        let reg = BasinLanes::pack(&lanes);
        assert!(!reg.is_silent(), "a disagreeing lane is not silence");
        assert_eq!(reg.disagreement_count(), 1, "the whale is RECORDED");
        assert_eq!(
            reg.agreement_count(),
            BASIN_LANES - 1,
            "the rest of the neighbourhood is untouched — nothing was removed"
        );
        assert_eq!(
            reg.unpack()[3],
            -5,
            "the disagreement value itself survives"
        );
    }

    /// Missing ≠ refuted, at the carrier level: lane 0 (silence) and a
    /// negative lane are different cells. Collapsing them is the same error
    /// class as `NoEvidence` narrowing a set.
    #[test]
    fn silence_and_denial_are_different_cells() {
        let mut lanes = [0i8; BASIN_LANES];
        lanes[7] = -1;
        let denied = BasinLanes::pack(&lanes);
        lanes[7] = 0;
        let silent = BasinLanes::pack(&lanes);
        assert_ne!(denied, silent);
        assert_eq!(denied.disagreement_count(), 1);
        assert_eq!(silent.disagreement_count(), 0);
        assert!(silent.is_silent());
    }

    /// The register width is the SAME 12 units every CascadeShape carves —
    /// this reading adds no bytes to the node, it re-reads what is there.
    #[test]
    fn the_lane_register_is_the_cascade_register_re_read_at_nibble_grain() {
        assert_eq!(BASIN_LANE_BYTES, CASCADE_UNITS);
        assert_eq!(BASIN_LANES, 2 * CASCADE_UNITS);
        for s in CascadeShape::ROTATIONS {
            assert_eq!(
                s.groups() as usize * s.levels() as usize,
                BASIN_LANE_BYTES,
                "every byte-granular shape covers the same register these lanes re-read"
            );
        }
    }
    // ---- one-hop accumulation ----

    #[test]
    fn accumulation_stacks_agreement_and_records_disagreement_as_pull_down() {
        let mut a = [0i8; BASIN_LANES];
        let mut b = [0i8; BASIN_LANES];
        a[0] = 2;
        b[0] = 3; // both agree on lane 0
        a[1] = 4;
        b[1] = -1; // contested lane 1: net stays positive but is PULLED DOWN
        let acc = BasinLanes::accumulate_children(&[BasinLanes::pack(&a), BasinLanes::pack(&b)]);
        let lanes = acc.unpack();
        assert_eq!(lanes[0], 5, "agreement stacks");
        assert_eq!(
            lanes[1], 3,
            "disagreement is IN the accumulation, not dropped"
        );
        assert!(
            lanes[2..].iter().all(|&v| v == 0),
            "untouched lanes stay silent"
        );
    }

    #[test]
    fn accumulation_saturates_at_the_carrier_bound_in_both_directions() {
        let up = BasinLanes::pack(&[5; BASIN_LANES]);
        let down = BasinLanes::pack(&[-5; BASIN_LANES]);
        assert_eq!(
            BasinLanes::accumulate_children(&[up, up, up]).unpack(),
            [7; BASIN_LANES]
        );
        assert_eq!(
            BasinLanes::accumulate_children(&[down, down, down]).unpack(),
            [-8; BASIN_LANES]
        );
    }

    #[test]
    fn a_childless_position_accumulates_to_silence() {
        assert!(BasinLanes::accumulate_children(&[]).is_silent());
    }

    /// Pinned MEASURED LIMITATION, not desired behaviour: in ONE register a
    /// balanced conflict is indistinguishable from silence. This is the
    /// concrete case for the ruled multi-register expansion; when that lands
    /// this pin must fail and force the deliberate re-shape.
    #[test]
    fn a_balanced_conflict_collapses_to_silence_in_one_register() {
        let mut a = [0i8; BASIN_LANES];
        let mut b = [0i8; BASIN_LANES];
        a[5] = 3;
        b[5] = -3;
        let acc = BasinLanes::accumulate_children(&[BasinLanes::pack(&a), BasinLanes::pack(&b)]);
        assert!(
            acc.is_silent(),
            "one net register cannot carry contested-ness; the day it can, re-pin"
        );
    }

    /// One-hop locality made OBSERVABLE across two levels: the grandparent
    /// reads the grandchild only through the child's accumulated register.
    /// Can-fire: a grandchild move that moves the child moves the
    /// grandparent. Silence: a grandchild move ABSORBED by the child's own
    /// saturation leaves the grandparent byte-identical.
    #[test]
    fn a_grandchild_reaches_the_grandparent_only_through_the_child() {
        use crate::hhtl::{direct_children, NiblePath};
        let gp = NiblePath::root(1);
        let child = gp.child(2);
        let gc1 = child.child(0);
        let gc2 = child.child(1);
        let occupied = [gp, child, gc1, gc2];
        assert_eq!(direct_children(gp, &occupied), vec![child]);
        assert_eq!(direct_children(child, &occupied), vec![gc1, gc2]);

        let lanes_of = |v: i8| {
            let mut l = [0i8; BASIN_LANES];
            l[0] = v;
            BasinLanes::pack(&l)
        };
        let two_level = |gc1_v: i8, gc2_v: i8| {
            let child_reg = BasinLanes::accumulate_children(&[lanes_of(gc1_v), lanes_of(gc2_v)]);
            BasinLanes::accumulate_children(&[child_reg])
        };
        // Can-fire: the grandchild flips sign hard -> the child moves -> the
        // grandparent moves.
        assert_ne!(two_level(3, 3), two_level(-6, 3));
        // Silence: both grandchild states saturate the child at +7, so the
        // grandparent cannot tell them apart -- the grandchild's detail is
        // the CHILD's knowledge, one hop only.
        assert_eq!(two_level(7, 5), two_level(6, 7));
    }
}
