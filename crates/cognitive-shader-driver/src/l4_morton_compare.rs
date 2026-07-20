//! L4 morton-cascade read over a real `MailboxSoA` style lane — the V3
//! thinking-compute wiring (ENTROPY-MILESTONES M4/M14).
//!
//! Operator directive: *"consolidate the P64 cognitive-shader-driver ancestry
//! pipeline into the Morton-cascade … for V3 since it wasn't wired properly"*
//! and *"keep the old and add a `_v3` … and compare its results."* This module
//! is the **keep-old + add-v3 + compare** wedge at the driver level, and it is
//! deliberately the same shape as [`crate::edge_v3_compare`]: **additive,
//! read-only, no hot-loop change, no new stored column.**
//!
//! ## Why the L4 lane already IS a morton tenant
//!
//! `MailboxSoA`'s P4 autopoiesis lanes (`frozen`/`learned`/`explore_style`) are
//! each a **12-byte L4 register** (`[u8; 12]` per row), read via the
//! `MailboxSoaView::style_lane_at` trait accessor. Per the operator-locked
//! content-blind facet (`E-V3-FACET-4-PLUS-12`, le-contract §3), those 12 bytes
//! are *"a dumb byte register the ClassView projects — it holds every sanctioned
//! reading at once."* One sanctioned carving is `6×(u8:u8)` — exactly the
//! [`bgz_tensor::morton_cascade::L4Tenant`] shape (3 SPO pairs + 3
//! AriGraph-basin pairs). So this module does **not** reshape or re-store the
//! lane; it reads the *same bytes* the policy-lane reader reads and scores them
//! through the shipped 256×256 [`FisherZTable`] over the Morton 2bit×2bit
//! inverse-pyramid cascade.
//!
//! ## What it replaces (the "wasn't wired properly" path)
//!
//! The clean V3 compute REPLACES running the cluttered V1 `CausalEdge64`
//! (`3×8` bare palette indices) over the V3 substrate — the poor design the
//! operator named. The legacy V1 bare-index read is kept in parallel as the
//! comparand ([`bgz_tensor::morton_cascade::legacy`]); the divergence when a
//! lane's pair partners (the `.1` of each `(a, b)`) carry signal is the measured
//! cost of the old path. On the `a == b` diagonal the two agree — the legacy
//! read is the diagonal of the V3 read.
//!
//! ## Codebook-agnostic by construction
//!
//! `read`/`compare` take the [`FisherZTable`] as a parameter — this module never
//! fabricates a global codebook. The driver's real call site passes its own
//! trained table (a later, separately-gated step, exactly as `edge_v3_compare`
//! is the wedge *before* any stored `edges_v3` column). The tests here build a
//! small synthetic smooth codebook, the same fixture `morton_cascade`'s own
//! tests use.

use crate::mailbox_soa::MailboxSoA;
use bgz_tensor::morton_cascade::{self, L4Tenant, Reading};
use bgz_tensor::FisherZTable;
use lance_graph_contract::soa_view::{MailboxSoaView, StyleLane};

/// Read a mailbox row's L4 style `lane` as a morton-cascade tenant (the same
/// `[u8; 12]` the policy-lane reader sees, carved `6×(u8:u8)`). Returns `None`
/// for an out-of-range / unpopulated row (the `style_lane_at` logical-row
/// discipline).
#[inline]
pub fn tenant_at<const N: usize>(
    mb: &MailboxSoA<N>,
    row: usize,
    lane: StyleLane,
) -> Option<L4Tenant> {
    mb.style_lane_at(row, lane)
        .map(|b| L4Tenant::from_bytes(&b))
}

/// The clean V3 morton read of a row's lane through `fz`, via the active backend
/// ([`morton_cascade::read`] — LazyLock, default V3). `None` for an
/// out-of-range / unpopulated row.
#[inline]
pub fn read_lane<const N: usize>(
    mb: &MailboxSoA<N>,
    row: usize,
    lane: StyleLane,
    fz: &FisherZTable,
) -> Option<Reading> {
    tenant_at(mb, row, lane).map(|t| morton_cascade::read(&t, fz))
}

/// Compare thinking: run BOTH substrates (V3 full-resolution pairwise vs the V1
/// bare-index legacy path) on the same real mailbox lane. Returns
/// `(v3, legacy)`; `None` for an out-of-range / unpopulated row. When the lane's
/// pair partners carry signal the V1 read cannot see, the two `Reading`s
/// diverge — the measured cost of running V1 over the V3 substrate.
#[inline]
pub fn compare_lane<const N: usize>(
    mb: &MailboxSoA<N>,
    row: usize,
    lane: StyleLane,
    fz: &FisherZTable,
) -> Option<(Reading, Reading)> {
    tenant_at(mb, row, lane).map(|t| morton_cascade::compare(&t, fz))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::collapse_gate::MailboxId;

    /// A smooth 256-centroid codebook so neighbouring indices correlate — the
    /// same fixture shape `morton_cascade`'s own tests use.
    fn table() -> FisherZTable {
        let dim = 32usize;
        let reps: Vec<Vec<f32>> = (0..256)
            .map(|c| {
                (0..dim)
                    .map(|d| ((c as f32) * 0.19 + (d as f32) * 0.37).sin())
                    .collect()
            })
            .collect();
        FisherZTable::build(&reps, 256)
    }

    /// One populated mailbox with a single row, ready for lane writes.
    fn one_row_mailbox() -> MailboxSoA<8> {
        let mut mb = MailboxSoA::<8>::new(MailboxId::default(), 0, 0.5);
        mb.set_populated(1);
        mb
    }

    #[test]
    fn reads_the_same_bytes_the_policy_lane_reader_sees() {
        let mut mb = one_row_mailbox();
        let bytes = [10u8, 200, 11, 205, 12, 195, 13, 15, 14, 11, 15, 12];
        mb.set_style_lane(0, StyleLane::Learned, bytes);
        // the morton tenant is a *view* of the same 12 bytes, carved 6×(u8:u8).
        let t = tenant_at(&mb, 0, StyleLane::Learned).expect("populated row");
        assert_eq!(
            t.to_bytes(),
            bytes,
            "tenant must be the lane bytes verbatim"
        );
        assert_eq!(t.spo[0], (10, 200));
        assert_eq!(t.basins[2], (15, 12));
    }

    #[test]
    fn v3_lane_read_differs_from_legacy_when_partners_carry_signal() {
        let fz = table();
        let mut mb = one_row_mailbox();
        // second byte of each SPO pair (the AriGraph / full-res half) carries
        // signal the V1 bare-index read drops.
        mb.set_style_lane(
            0,
            StyleLane::Learned,
            [10, 200, 11, 205, 12, 195, 13, 15, 14, 11, 15, 12],
        );
        let (v3, legacy) = compare_lane(&mb, 0, StyleLane::Learned, &fz).expect("populated row");
        assert!(
            (v3.fact_coherence - legacy.fact_coherence).abs() > 1e-3,
            "V3 full-res lane read must differ from the V1 bare-index read when the pair partners carry signal"
        );
    }

    #[test]
    fn v3_and_legacy_agree_on_the_degenerate_diagonal_lane() {
        let fz = table();
        let mut mb = one_row_mailbox();
        // a == b in every pair ⇒ the pairwise collapses to the bare index; the
        // legacy read is the diagonal of the V3 read, so the two must AGREE.
        mb.set_style_lane(
            0,
            StyleLane::Frozen,
            [10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15],
        );
        let (v3, legacy) = compare_lane(&mb, 0, StyleLane::Frozen, &fz).expect("populated row");
        assert!(
            (v3.fact_coherence - legacy.fact_coherence).abs() < 1e-6,
            "on the a==b diagonal the V3 pairwise lane read collapses to the V1 bare-index read"
        );
    }

    #[test]
    fn out_of_range_row_is_none() {
        let fz = table();
        let mb = one_row_mailbox(); // populated == 1
        assert!(tenant_at(&mb, 1, StyleLane::Learned).is_none());
        assert!(read_lane(&mb, 7, StyleLane::Learned, &fz).is_none());
        assert!(compare_lane(&mb, 3, StyleLane::Explore, &fz).is_none());
    }

    #[test]
    fn read_lane_matches_the_v3_half_of_compare() {
        let fz = table();
        let mut mb = one_row_mailbox();
        mb.set_style_lane(
            0,
            StyleLane::Explore,
            [3, 40, 5, 60, 7, 80, 9, 20, 11, 30, 13, 44],
        );
        // default backend is V3, so read_lane == the v3 half of compare_lane.
        let v3 = read_lane(&mb, 0, StyleLane::Explore, &fz).expect("populated");
        let (cmp_v3, _legacy) = compare_lane(&mb, 0, StyleLane::Explore, &fz).expect("populated");
        assert_eq!(v3, cmp_v3);
    }
}
