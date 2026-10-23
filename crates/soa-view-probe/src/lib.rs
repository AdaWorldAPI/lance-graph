//! D-NFR-6 lab probe: native operations over a [`SoaEnvelope`] borrow.
//!
//! **Mutation safety is the borrow; the mathematics is ndarray.** There is no
//! table type, no second ownership mechanism and no runtime guard here.
//!
//! - **Owner:** whoever holds the `Vec<NodeRow>` (the mailbox). It mutates
//!   through `&mut` and nothing else.
//! - **View:** [`NodeRowPacket`], the contract's existing `&self` borrow of
//!   those rows. As [`SoaEnvelope`]'s own docs put it, a view cannot outlive
//!   or alias the owner, and the owner's `&mut` cannot begin while any view is
//!   live: "Rust's borrow rules ARE the inheritance; nothing is checked at
//!   runtime." The packet is deliberately not `Clone`, so it cannot become a
//!   second holder either. The `compile_fail` examples below pin all three.
//! - **Operations:** [`FacetFold`], one ndarray kernel per method, over the
//!   packet's LE bytes at the 512-byte row stride and the 16-byte facet
//!   stride every other SoA in lance-graph uses. Each method body is that one
//!   call; there is no per-operation check to pay for.
//!
//! What this crate does not do, on purpose: it does not check frame
//! provenance or external authorization. Those are real questions (who sealed
//! these bytes; may this caller read them), but they belong to whoever hands
//! out the packet, once, and they are not a reason to re-check, per
//! operation, something the borrow checker already guarantees. The earlier
//! version of this crate did exactly that (a `Table` beside the SoA plus a
//! frame guard on each access) and is superseded; see the plan.
//!
//! # The borrow does the work
//!
//! The owner cannot write while a view is live (E0502):
//!
//! ```compile_fail,E0502
//! use lance_graph_contract::NodeRowPacket;
//! let mut rows = soa_view_probe::fixture::rows(64);
//! let view = NodeRowPacket::new(&rows, 1);
//! rows[0].value[0] = 7;
//! let _ = view.rows().len();
//! ```
//!
//! A view cannot outlive its owner (E0597):
//!
//! ```compile_fail,E0597
//! use lance_graph_contract::NodeRowPacket;
//! let view;
//! {
//!     let rows = soa_view_probe::fixture::rows(64);
//!     view = NodeRowPacket::new(&rows, 1);
//! }
//! let _ = view.rows().len();
//! ```
//!
//! A view cannot be duplicated into a second holder (E0599: no `clone`):
//!
//! ```compile_fail,E0599
//! use lance_graph_contract::NodeRowPacket;
//! let rows = soa_view_probe::fixture::rows(64);
//! let view = NodeRowPacket::new(&rows, 1);
//! let second = view.clone();
//! ```
//!
//! The twin that must compile: once the view is gone the owner writes, and a
//! new view sees the new bytes.
//!
//! ```
//! use lance_graph_contract::NodeRowPacket;
//! use soa_view_probe::{FacetFold, FacetSlot, fixture};
//! let mut rows = fixture::rows(64);
//! let slot = FacetSlot::value(0).unwrap();
//! let mut sel = [0u64; 1];
//! for class in [fixture::CLASS_A, fixture::CLASS_B] {
//!     rows[1].value[..4].copy_from_slice(&class.to_le_bytes());
//!     let view = NodeRowPacket::new(&rows, 2);
//!     view.classid_mask(slot, class, &mut sel);
//!     assert_eq!(sel[0] >> 1 & 1, 1);
//! }
//! ```

use lance_graph_contract::canonical_node::{
    GUIDS_PER_NODE, NODE_ROW_STRIDE, VALUE_SLAB_ROW_OFFSET,
};
use lance_graph_contract::soa_envelope::SoaEnvelope;
use lance_graph_contract::{NodeRow, NodeRowPacket};

/// Bytes per facet: `classid(4) + 12-byte payload`.
pub const FACET_BYTES: usize = 16;

/// One of the 32 facet slots of a 512-byte row (slot 0 is the key, slot 1 the
/// edge facet, slots 2.. the value slab). Built once; operations then trust
/// it, because a value of this type is in range by construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FacetSlot(u8);

impl FacetSlot {
    /// The key facet.
    pub const KEY: Self = Self(0);

    /// Slot `k` of the row, if it exists.
    pub const fn new(k: usize) -> Option<Self> {
        if k < GUIDS_PER_NODE {
            Some(Self(k as u8))
        } else {
            None
        }
    }

    /// Facet `k` of the value slab (bytes `32 + 16k ..`), if it exists.
    pub const fn value(k: usize) -> Option<Self> {
        Self::new(VALUE_SLAB_ROW_OFFSET / FACET_BYTES + k)
    }

    /// Byte offset of this facet inside a row.
    pub const fn offset(self) -> usize {
        self.0 as usize * FACET_BYTES
    }
}

/// Native operations over a borrowed packet. Every method is one ndarray call
/// on [`SoaEnvelope::as_le_bytes`] at the row stride; masks have one bit per
/// row (`n_rows().div_ceil(64)` words).
pub trait FacetFold {
    /// Rows whose facet `slot` carries `classid` in its first four bytes.
    fn classid_mask(&self, slot: FacetSlot, classid: u32, out: &mut [u64]);

    /// Rows whose 16-byte facet `slot` agrees with `pattern` on every set
    /// bit of `care`.
    fn facet_match_mask(
        &self,
        slot: FacetSlot,
        pattern: &[u8; 16],
        care: &[u8; 16],
        out: &mut [u64],
    );

    /// Sum of the twelve payload bytes (each `u8`, never widened) of facet
    /// `slot` over the rows selected by `mask`.
    fn payload_byte_sum(&self, slot: FacetSlot, mask: &[u64]) -> i64;
}

impl FacetFold for NodeRowPacket<'_> {
    #[inline]
    fn classid_mask(&self, slot: FacetSlot, classid: u32, out: &mut [u64]) {
        ndarray::simd::eq_u32_strided_to_mask(
            self.as_le_bytes(),
            slot.offset(),
            NODE_ROW_STRIDE,
            self.n_rows(),
            classid,
            out,
        );
    }

    #[inline]
    fn facet_match_mask(
        &self,
        slot: FacetSlot,
        pattern: &[u8; 16],
        care: &[u8; 16],
        out: &mut [u64],
    ) {
        ndarray::simd::ternary_match_strided16_to_mask(
            self.as_le_bytes(),
            slot.offset(),
            NODE_ROW_STRIDE,
            self.n_rows(),
            pattern,
            care,
            out,
        );
    }

    #[inline]
    fn payload_byte_sum(&self, slot: FacetSlot, mask: &[u64]) -> i64 {
        // 12 bytes per row, at most 255 each: the i128 accumulator inside the
        // kernel cannot leave i64 below ~3e15 rows, so `None` is unreachable
        // for any slice that fits in memory.
        ndarray::simd::masked_strided_group_sum(
            self.as_le_bytes(),
            slot.offset() + 4,
            NODE_ROW_STRIDE,
            self.n_rows(),
            12,
            1,
            mask,
        )
        .expect("12 u8 per row cannot overflow i64 for an in-memory slice")
    }
}

/// Owner-side write: facet `k` of the value slab of one row. The owner holds
/// `&mut`, so no view of these rows can be live while this runs.
pub fn set_value_facet(row: &mut NodeRow, k: usize, facet: [u8; FACET_BYTES]) {
    let at = k * FACET_BYTES;
    row.value[at..at + FACET_BYTES].copy_from_slice(&facet);
}

/// Deterministic rows for tests and the cost example.
pub mod fixture {
    use super::{FACET_BYTES, set_value_facet};
    use lance_graph_contract::canonical_node::TailVariant;
    use lance_graph_contract::{EdgeBlock, NodeGuid, NodeRow};

    /// Two classids (canon high: concept, low: app) for the value facet.
    pub const CLASS_A: u32 = 0x0B05_0001;
    /// The other classid.
    pub const CLASS_B: u32 = 0x080B_0001;

    /// `n` rows: key minted V3, value facet 0 alternating between the two
    /// classids by a fixed pseudo-random pattern, payload bytes varying.
    pub fn rows(n: usize) -> Vec<NodeRow> {
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        (0..n)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let key = NodeGuid::mint_for(
                    TailVariant::V3,
                    CLASS_A,
                    1,
                    2,
                    3,
                    4,
                    5,
                    (i % 0xFFFF) as u32,
                );
                let mut row = NodeRow {
                    key,
                    edges: EdgeBlock::from_bytes(&[0; 16]),
                    value: [0; 480],
                };
                let class = if s % 3 == 0 { CLASS_B } else { CLASS_A };
                let mut facet = [0u8; FACET_BYTES];
                facet[..4].copy_from_slice(&class.to_le_bytes());
                for (j, b) in facet[4..].iter_mut().enumerate() {
                    *b = (s >> (j * 5)) as u8;
                }
                set_value_facet(&mut row, 0, facet);
                row
            })
            .collect()
    }
}
