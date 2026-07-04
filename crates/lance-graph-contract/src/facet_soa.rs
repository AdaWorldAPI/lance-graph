//! `facet_soa` — the class-facet registry SoA: a stride-16 column of
//! [`FacetCascade`] rows.
//!
//! # What this is (and what it is NOT)
//!
//! When a producer (`ogar-from-ruff`) compiles a class it mints a 16-byte
//! **rail facet** — `facet_classid(4) | 6×(8:8) tiers` — the class's
//! concept-cascade *address*. This module is the **sink**: N compiled classes
//! → N × 16-byte [`FacetCascade`] rows in one aligned columnar backing store,
//! a narrow SoA orthogonal to the 512-byte per-instance node SoA.
//!
//! ```text
//!   ogar-from-ruff: compile → Facet::to_bytes()  (16B per class)
//!                                   │  (byte-identical carrier crossing)
//!   here:           FacetSoa::from_le_bytes  → &[FacetCascade]  (the registry)
//! ```
//!
//! **All substrate here is V3-shaped.** The 16-byte key is
//! `facet_classid(4) | 6 × (8:8) tiles` (`HEEL·HIP·TWIG·LEAF·family·identity`),
//! byte-identical to [`FacetCascade`]. The legacy V1 `family-u24 + identity-u24`
//! node-key tail is **not used** by any new code in this module — new work is
//! tile-shaped only.
//!
//! This registry stays a **separate** stride-16 column for a *population*
//! reason, not a byte-layout one: a compiled **class** rail-address is not a
//! per-**instance** node. A class facet carries no instance `identity` — its
//! tiers are the concept cascade, and the id tier is unused (0) for a pure
//! class address. Rows are read only through [`FacetCascade`]'s own tile
//! accessors.
//!
//! [`facet_cascades_from_le_bytes`] is the exact stride-16 twin of
//! `node_rows_from_le_bytes` (`canonical_node.rs`): a checked zero-copy
//! reinterpret of a 16-aligned LE slab as `&[FacetCascade]`.

use crate::facet::FacetCascade;
use crate::soa_envelope::{ColumnDescriptor, ColumnKind, SoaEnvelope};

/// Stride of one facet row = `size_of::<FacetCascade>()` = 16 bytes.
pub const FACET_ROW_STRIDE: usize = 16;

/// Stable column ordinal for the single facet column (consumer-side `name_id`).
/// The registry is one column; the ordinal is its identity in the descriptor
/// table, not a value-slab tenant id (this SoA is standalone, not a `NodeRow`).
pub const FACET_COLUMN_ID: u16 = 0;

/// Zero-copy reinterpret of a 16-aligned LE byte slab as `&[FacetCascade]` —
/// the stride-16 twin of `node_rows_from_le_bytes`.
///
/// Returns `Some(&[])` for an empty slice, `None` if `bytes.len()` is not a
/// multiple of [`FACET_ROW_STRIDE`] or the pointer is not 16-aligned. Bytes
/// produced by [`FacetSoa::as_le_bytes`] are always 16-aligned (the backing
/// `Vec<FacetCascade>` allocates at `align(16)`), so the round-trip
/// `facet_cascades_from_le_bytes(soa.as_le_bytes())` always succeeds.
#[must_use]
pub fn facet_cascades_from_le_bytes(bytes: &[u8]) -> Option<&[FacetCascade]> {
    if bytes.is_empty() {
        return Some(&[]);
    }
    if !bytes.len().is_multiple_of(FACET_ROW_STRIDE) {
        return None;
    }
    if !(bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<FacetCascade>()) {
        return None;
    }
    let n = bytes.len() / FACET_ROW_STRIDE;
    // SAFETY: FacetCascade is #[repr(C, align(16))], size_of == 16 == FACET_ROW_STRIDE
    // (const-asserted in facet.rs). We checked (1) bytes.len() is an exact multiple of
    // the stride, so n rows span the whole slice with no trailing bytes, and (2) the
    // pointer is aligned to align_of::<FacetCascade>() (16). Every 16-byte pattern is a
    // valid FacetCascade (facet_classid is a u32, tiers are [u8;2] pairs — no niche/enum
    // to invalidate), so the reinterpretation is sound. The returned slice borrows
    // `bytes` for its lifetime (no copy).
    Some(unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast::<FacetCascade>(), n) })
}

/// The class-facet registry: N compiled classes → N × 16-byte [`FacetCascade`]
/// rows in a 16-aligned columnar backing store.
///
/// This is the class **rail-address** registry — distinct from the per-instance
/// [`NodeRow`](crate::canonical_node::NodeRow) store. Each row is self-keying:
/// the facet's own `facet_classid` prefix routes it, so no external key column
/// is needed (`lookup_by_classid` scans; a radix index is a follow-up).
#[derive(Clone, Debug, Default)]
pub struct FacetSoa {
    /// 16-aligned backing store — one [`FacetCascade`] per class.
    rows: Vec<FacetCascade>,
    /// Cycle stamp this snapshot carries (the [`SoaEnvelope`] version stamp).
    cycle: u32,
}

impl FacetSoa {
    /// Build a registry from already-decoded facets.
    #[must_use]
    pub fn from_facets(rows: Vec<FacetCascade>, cycle: u32) -> Self {
        Self { rows, cycle }
    }

    /// Build the registry from concatenated 16-byte facet keys — e.g. the
    /// producer's flattened `Vec<[u8;16]>` from `Facet::to_bytes()`. Each
    /// 16-byte chunk is parsed via [`FacetCascade::from_bytes`] into the
    /// 16-aligned backing store (a copy — this is the base-header parse leg).
    /// Returns `None` if `bytes.len()` is not a multiple of the stride.
    #[must_use]
    pub fn from_le_bytes(bytes: &[u8], cycle: u32) -> Option<Self> {
        if !bytes.len().is_multiple_of(FACET_ROW_STRIDE) {
            return None;
        }
        let rows = bytes
            .chunks_exact(FACET_ROW_STRIDE)
            .map(|c| {
                let mut b = [0u8; FACET_ROW_STRIDE];
                b.copy_from_slice(c);
                FacetCascade::from_bytes(&b)
            })
            .collect();
        Some(Self { rows, cycle })
    }

    /// The registry rows as a borrowed slice.
    #[must_use]
    pub fn facets(&self) -> &[FacetCascade] {
        &self.rows
    }

    /// The `i`-th facet, if in range.
    #[must_use]
    pub fn get(&self, i: usize) -> Option<&FacetCascade> {
        self.rows.get(i)
    }

    /// Number of classes in the registry.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The first row whose `facet_classid` equals `classid` (linear scan). The
    /// facet is prefix-routable by its own classid; a radix index over the
    /// registry is a follow-up. Returns `None` on a miss (e.g. the bootstrap
    /// `0x0000_0000` when it was never sunk), never a wrong neighbor.
    #[must_use]
    pub fn lookup_by_classid(&self, classid: u32) -> Option<&FacetCascade> {
        self.rows.iter().find(|f| f.facet_classid == classid)
    }
}

/// The single-column descriptor: 16 × `u8` at offset 0 — one `FacetCascade`
/// per row. Static so the descriptor table is itself a stable LE artifact.
const FACET_COLUMNS: [ColumnDescriptor; 1] = [ColumnDescriptor {
    name_id: FACET_COLUMN_ID,
    kind: ColumnKind::U8,
    elems_per_row: FACET_ROW_STRIDE as u16,
    row_offset: 0,
}];

impl SoaEnvelope for FacetSoa {
    fn columns(&self) -> &[ColumnDescriptor] {
        &FACET_COLUMNS
    }

    fn row_stride(&self) -> usize {
        FACET_ROW_STRIDE
    }

    fn n_rows(&self) -> usize {
        self.rows.len()
    }

    fn cycle(&self) -> u32 {
        self.cycle
    }

    fn as_le_bytes(&self) -> &[u8] {
        // SAFETY: FacetCascade is #[repr(C, align(16))], size_of == 16 (const-asserted
        // in facet.rs). A &[FacetCascade] of len n reinterprets as &[u8] of len n*16:
        // u8 has alignment 1 (≤ 16, so the pointer is trivially aligned), and every byte
        // of a FacetCascade is a valid u8 (no niche). The borrow ties the &[u8] to &self.
        unsafe {
            core::slice::from_raw_parts(
                self.rows.as_ptr().cast::<u8>(),
                self.rows.len() * FACET_ROW_STRIDE,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soa_envelope::SoaEnvelope;

    fn sample(classid: u32, part_of: [u8; 6], is_a: [u8; 6]) -> FacetCascade {
        // Build via the 16 facet bytes in the locked wire order:
        // classid LE in [0..4); byte[4+2t]=is_a[t] (lo), byte[5+2t]=part_of[t] (hi).
        let mut b = [0u8; 16];
        b[0..4].copy_from_slice(&classid.to_le_bytes());
        for t in 0..6 {
            b[4 + 2 * t] = is_a[t];
            b[5 + 2 * t] = part_of[t];
        }
        FacetCascade::from_bytes(&b)
    }

    #[test]
    fn from_le_bytes_round_trips_every_row_byte_exact() {
        let a = sample(0x0202_0002, [1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]);
        let c = sample(0x0102_0001, [9, 9, 9, 0, 0, 0], [1, 1, 1, 0, 0, 0]);
        let src: Vec<u8> = [a, c].iter().flat_map(|f| f.to_bytes()).collect();

        let soa = FacetSoa::from_le_bytes(&src, 7).expect("multiple of 16");
        assert_eq!(soa.len(), 2);
        assert_eq!(soa.cycle(), 7);
        // Byte-parity on ALL rows (not one hand-picked value).
        assert_eq!(soa.as_le_bytes(), &src[..]);
        assert_eq!(soa.get(0).unwrap().to_bytes(), a.to_bytes());
        assert_eq!(soa.get(1).unwrap().to_bytes(), c.to_bytes());
    }

    #[test]
    fn as_le_bytes_is_16_aligned_and_reinterprets_zero_copy() {
        // The slab-addressability guard truth-architect required: as_le_bytes()
        // is 16-aligned (Vec<FacetCascade> allocates at align 16), so the
        // free-function reinterpret returns Some and points at the same bytes.
        let soa = FacetSoa::from_facets(
            vec![sample(0x0202_0002, [1; 6], [2; 6]), sample(7, [0; 6], [0; 6])],
            0,
        );
        let bytes = soa.as_le_bytes();
        assert_eq!(bytes.len(), 32);
        let view = facet_cascades_from_le_bytes(bytes).expect("aligned slab reinterprets");
        assert_eq!(view.len(), 2);
        assert_eq!(view[0].facet_classid, 0x0202_0002);
        // Same backing bytes — a reinterpret, not a copy.
        assert_eq!(view.as_ptr() as usize, soa.facets().as_ptr() as usize);
    }

    #[test]
    fn lookup_by_classid_indexes_and_misses_cleanly() {
        let soa = FacetSoa::from_facets(
            vec![
                sample(0x0202_0002, [1; 6], [2; 6]),
                sample(0x0102_0001, [3; 6], [4; 6]),
            ],
            0,
        );
        assert_eq!(
            soa.lookup_by_classid(0x0102_0001).unwrap().facet_classid,
            0x0102_0001
        );
        assert!(soa.lookup_by_classid(0x0000_0000).is_none(), "clean miss");
    }

    #[test]
    fn from_le_bytes_rejects_non_stride_length() {
        assert!(FacetSoa::from_le_bytes(&[0u8; 15], 0).is_none());
        assert!(FacetSoa::from_le_bytes(&[0u8; 17], 0).is_none());
        assert!(FacetSoa::from_le_bytes(&[], 0).unwrap().is_empty());
    }

    #[test]
    fn envelope_layout_verifies() {
        let soa = FacetSoa::from_facets(vec![sample(1, [0; 6], [0; 6])], 3);
        // One 16-byte column at offset 0, stride 16 — geometry is self-consistent.
        assert_eq!(soa.row_stride(), 16);
        assert_eq!(soa.columns().len(), 1);
        assert_eq!(soa.columns()[0].col_bytes_per_row(), 16);
        soa.verify_layout().expect("single-column 16B geometry is valid");
    }

    #[test]
    fn free_fn_rejects_unaligned_and_bad_length() {
        // A Vec<u8> is only 1-aligned; the reinterpret must refuse it.
        let unaligned: Vec<u8> = vec![0u8; 32];
        // Force a very likely misalignment by slicing off the front byte.
        let off = &unaligned[1..17];
        assert!(
            facet_cascades_from_le_bytes(off).is_none()
                || (off.as_ptr() as usize).is_multiple_of(16)
        );
        // Bad length.
        assert!(facet_cascades_from_le_bytes(&[0u8; 15]).is_none());
    }
}
