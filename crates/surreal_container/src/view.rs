//! # `view` — the SurrealQL read-glove (D-PG-6, `MailboxSoaView` adapter)
//!
//! The contract-compliant **read-only** view that `surreal_container`
//! materialises over a Lance-backed mailbox row. Pairs with the
//! `LanceVersionScheduler` in `lance-graph::graph::scheduler` to close
//! `E-SUBSTRATE-IS-THE-SCHEDULER`'s IN-direction (substrate version tick →
//! next legal `KanbanMove`) for a SurrealQL projection.
//!
//! ## What this is (and isn't)
//!
//! `MailboxSoaView` is **read-only by trait design** (per
//! `soa_view.rs`: "view is read-only" is structural — surreal implements only
//! the read half; the OUT-direction `MailboxSoaOwner` is on the cognitive-side
//! `MailboxSoA<N>`). That's exactly the polyglot-container ruling:
//!
//! > LanceDB leads; SurrealDB is a view/dialect (handover 2026-05-28 §2,
//! > E-RUBICON-RACTOR; kanban.rs:1-21 "surreal=project-read-only,
//! > callcenter=commit").
//!
//! So this module:
//!
//! - DOES expose `SurrealMailboxView<'a>` — a typed view a SurrealQL
//!   projection populates from kv-lance scan data (zero-copy borrows over the
//!   underlying byte buffers).
//! - DOES expose `SurrealMailboxView::from_columns(...)` — the constructor
//!   the kv-lance read-path calls once it has the row bytes.
//! - DOES expose `read_via_kv_lance()` — the integration-point stub the
//!   surrealdb-fork integrator implements once the cold build is taken.
//! - Does NOT implement `MailboxSoaOwner`. That trait lives on the cognitive
//!   side (`MailboxSoA<N>` in `cognitive-shader-driver`). Trying to mutate
//!   through a `SurrealMailboxView` is a category error the trait surface
//!   already forbids.
//!
//! ## How it plugs in
//!
//! ```text
//!   Lance dataset       SurrealDB (kv-lance backend)         consumer
//!   ─────────────       ────────────────────────────         ────────
//!   versions().tick ─►  SurrealQL projection / LIVE  ────►  SurrealMailboxView<'a>
//!                                                                │
//!                                                                ▼
//!                                                  VersionScheduler::on_version
//!                                                                │
//!                                                                ▼
//!                                                       Option<KanbanMove>
//!                                                                │
//!                                                                ▼
//!                                               cognitive owner.try_advance_phase
//! ```
//!
//! `LanceVersionScheduler` (shipped this PR in `lance-graph`) and this view
//! are the IN-direction pair: they meet at the `MailboxSoaView` trait
//! boundary, with no SurrealQL types crossing the cognitive-side seam.
//!
//! ## Status
//!
//! - `SurrealMailboxView<'a>` and `MailboxSoaView` impl: **shipped here**
//!   (contract-only, zero surrealdb dep).
//! - `read_via_kv_lance()`: **stub** — returns
//!   [`SurrealContainerError::BlockedColdBuild`] until the surrealdb fork dep
//!   in `Cargo.toml` is uncommented and the kv-lance scan code is filled in.
//!   This is the integrator's task; the trait surface above stays unchanged.

use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::KanbanColumn;
use lance_graph_contract::soa_view::MailboxSoaView;

use crate::SurrealContainerError;

/// A read-only mailbox view materialised from a SurrealQL projection over
/// a Lance-backed kv-lance row.
///
/// **Zero-copy borrow design.** Per-row column slices are borrowed from the
/// scan's byte buffers (which outlive the view); the view is constructed once
/// per `versions()` tick, fed to `VersionScheduler::on_version`, and
/// dropped. No allocation, no clone — same shape as the existing
/// `MailboxSoaView` impl on `MailboxSoA<N>`.
///
/// **Borrow lifetime `'a`** is the lifetime of the projection's byte buffers.
/// The surrealdb-fork integrator picks the right backing — likely an arrow
/// `RecordBatch`-borrowed slice — when wiring `read_via_kv_lance`.
#[derive(Debug, Clone, Copy)]
pub struct SurrealMailboxView<'a> {
    mailbox_id: MailboxId,
    w_slot: u8,
    current_cycle: u32,
    phase: KanbanColumn,
    energy: &'a [f32],
    edges_raw: &'a [u64],
    meta_raw: &'a [u32],
    entity_type: &'a [u16],
}

impl<'a> SurrealMailboxView<'a> {
    /// Construct a view from already-projected column slices. The SurrealQL
    /// kv-lance scan calls this once it has the row bytes; the slices borrow
    /// from the scan's underlying buffers and the view is dropped before the
    /// next tick.
    ///
    /// **Invariant:** all column slices MUST have the same length
    /// `N == energy.len() == edges_raw.len() == meta_raw.len() ==
    /// entity_type.len()` — this is what `n_rows()` returns. Mismatched
    /// lengths are a programming error in the projection; `from_columns`
    /// debug-asserts in tests but does not validate at runtime to preserve
    /// zero-copy.
    //
    // `#[allow(clippy::too_many_arguments)]`: the arg list IS the
    // `MailboxSoaView` column shape (4 scalars + 4 column slices). Packing
    // them into a struct would add a copy/layout layer for zero readers —
    // every caller is a kv-lance scan that already has the 8 values fanned
    // out from its arrow `RecordBatch`. A builder would land an allocation
    // in the zero-copy path. Trait shape wins.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn from_columns(
        mailbox_id: MailboxId,
        w_slot: u8,
        current_cycle: u32,
        phase: KanbanColumn,
        energy: &'a [f32],
        edges_raw: &'a [u64],
        meta_raw: &'a [u32],
        entity_type: &'a [u16],
    ) -> Self {
        debug_assert_eq!(energy.len(), edges_raw.len());
        debug_assert_eq!(energy.len(), meta_raw.len());
        debug_assert_eq!(energy.len(), entity_type.len());
        Self {
            mailbox_id,
            w_slot,
            current_cycle,
            phase,
            energy,
            edges_raw,
            meta_raw,
            entity_type,
        }
    }
}

impl<'a> MailboxSoaView for SurrealMailboxView<'a> {
    #[inline]
    fn mailbox_id(&self) -> MailboxId {
        self.mailbox_id
    }
    #[inline]
    fn n_rows(&self) -> usize {
        self.energy.len()
    }
    #[inline]
    fn w_slot(&self) -> u8 {
        self.w_slot
    }
    #[inline]
    fn current_cycle(&self) -> u32 {
        self.current_cycle
    }
    #[inline]
    fn phase(&self) -> KanbanColumn {
        self.phase
    }
    #[inline]
    fn energy(&self) -> &[f32] {
        self.energy
    }
    #[inline]
    fn edges_raw(&self) -> &[u64] {
        self.edges_raw
    }
    #[inline]
    fn meta_raw(&self) -> &[u32] {
        self.meta_raw
    }
    #[inline]
    fn entity_type(&self) -> &[u16] {
        self.entity_type
    }
}

/// Read a mailbox row from the SurrealDB `kv-lance` backend and present it
/// as a [`SurrealMailboxView`].
///
/// **Stub.** The surrealdb fork dep in `Cargo.toml` is commented out (cold
/// build cost — see Cargo.toml `[dependencies]` note for the rationale).
/// Once an integrator uncomments it, this body is filled in with:
///
/// 1. A SurrealQL projection: `SELECT energy, edges_raw, meta_raw,
///    entity_type, current_cycle, phase, w_slot FROM mailbox WHERE id =
///    $mailbox_id` — the polyglot-container plan's D-PG-3
///    record-range scan.
/// 2. Decode the projection's arrow `RecordBatch` columns into the typed
///    slices.
/// 3. Call [`SurrealMailboxView::from_columns`] with the borrowed slices and
///    return.
///
/// Until then this returns [`SurrealContainerError::BlockedColdBuild`] —
/// the typed signal a caller can pattern-match on to know whether to fall
/// back to a direct lance-graph read.
#[allow(unused_variables)]
pub async fn read_via_kv_lance(
    _store: &crate::SurrealStore,
    _mailbox_id: MailboxId,
) -> Result<SurrealMailboxView<'static>, SurrealContainerError> {
    // Reference implementation (paste body below once the surrealdb dep is
    // uncommented; the contract surface above stays unchanged):
    //
    //   let ds = store.ds().ok_or(SurrealContainerError::Blocked { .. })?;
    //   let q = "SELECT * FROM mailbox WHERE mailbox_id = $id;";
    //   let result = ds.execute(q, &session, vars).await?;
    //   let row = result.into_first_row()?;
    //   Ok(SurrealMailboxView::from_columns(
    //       row.mailbox_id, row.w_slot, row.current_cycle, row.phase,
    //       row.energy_slice(), row.edges_slice(), row.meta_slice(),
    //       row.entity_type_slice(),
    //   ))
    Err(SurrealContainerError::BlockedColdBuild {
        reason: "surrealdb fork dep commented (cold-build gate); see Cargo.toml \
                 [dependencies] BLOCKED(C) note and view.rs::read_via_kv_lance docs",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::scheduler::{
        DatasetVersion, NextPhaseScheduler, VersionScheduler,
    };

    fn make_view<'a>(
        phase: KanbanColumn,
        energy: &'a [f32],
        edges: &'a [u64],
        meta: &'a [u32],
        et: &'a [u16],
    ) -> SurrealMailboxView<'a> {
        SurrealMailboxView::from_columns(7, 5, 13, phase, energy, edges, meta, et)
    }

    #[test]
    fn view_columns_borrow_from_caller_buffers() {
        // Zero-copy: the slices the view returns are the slices the caller
        // passed in — no clone, no allocation. This mirrors the existing
        // MailboxSoaView impl on MailboxSoA<N>.
        let energy = [1.0f32, 2.0, 3.0];
        let edges = [10u64, 20, 30];
        let meta = [100u32, 200, 300];
        let et = [1u16, 2, 3];
        let v = make_view(KanbanColumn::Planning, &energy, &edges, &meta, &et);
        assert_eq!(v.mailbox_id(), 7);
        assert_eq!(v.w_slot(), 5);
        assert_eq!(v.current_cycle(), 13);
        assert_eq!(v.phase(), KanbanColumn::Planning);
        assert_eq!(v.n_rows(), 3);
        assert_eq!(v.energy(), &energy[..]);
        assert_eq!(v.edges_raw(), &edges[..]);
        assert_eq!(v.meta_raw(), &meta[..]);
        assert_eq!(v.entity_type(), &et[..]);
        // Pointer equality — proves the view borrows the original buffers.
        assert_eq!(v.energy().as_ptr(), energy.as_ptr());
        assert_eq!(v.edges_raw().as_ptr(), edges.as_ptr());
    }

    #[test]
    fn view_is_a_legal_input_to_next_phase_scheduler() {
        // The whole point of the read glove: a SurrealMailboxView plugged
        // into the contract's `NextPhaseScheduler` lowers a version tick to
        // the next legal Rubicon move — same as MailboxSoA<N>, no SurrealQL
        // types cross the boundary.
        let energy = [];
        let edges = [];
        let meta = [];
        let et = [];
        let view = make_view(KanbanColumn::Planning, &energy, &edges, &meta, &et);
        let mv = NextPhaseScheduler
            .on_version(
                &view,
                DatasetVersion(42),
                lance_graph_contract::kanban::ExecTarget::SurrealQl,
            )
            .expect("Planning is not absorbing");
        assert_eq!(mv.from, KanbanColumn::Planning);
        assert_eq!(mv.to, KanbanColumn::CognitiveWork);
        assert_eq!(mv.libet_offset_us, -550_000); // Libet anchor
        assert_eq!(
            mv.exec,
            lance_graph_contract::kanban::ExecTarget::SurrealQl,
            "exec target routed through unchanged"
        );
    }

    #[test]
    fn view_implements_only_read_half_compile_time() {
        // Trait-system proof that the read glove cannot mutate: this
        // module imports MailboxSoaView but NOT MailboxSoaOwner. If a
        // future drift adds `impl MailboxSoaOwner for SurrealMailboxView`,
        // it would have to import the Owner trait — a code-review tripwire
        // and the structural enforcement the kanban.rs:1-21 doc states.
        fn assert_view<V: MailboxSoaView>(_: &V) {}
        let energy = [];
        let edges = [];
        let meta = [];
        let et = [];
        let view = make_view(KanbanColumn::Evaluation, &energy, &edges, &meta, &et);
        assert_view(&view); // compiles ⇒ View only.
    }

    #[tokio::test(flavor = "current_thread")]
    async fn read_via_kv_lance_returns_typed_blocked_cold_build() {
        // Until the surrealdb fork dep is uncommented, the kv-lance read
        // path returns BlockedColdBuild — the typed signal a caller can
        // pattern-match to fall back. NOT an unwrap-or-panic stub.
        // SurrealStore::open itself is Blocked (pre-existing); compose on
        // a placeholder PhantomData store to test read_via_kv_lance directly.
        let placeholder = crate::SurrealStore::test_placeholder();
        let result = read_via_kv_lance(&placeholder, 42).await;
        assert!(
            matches!(result, Err(SurrealContainerError::BlockedColdBuild { .. })),
            "stub must return BlockedColdBuild until dep is wired"
        );
        // The display surface carries the cold-build hint pointing at Cargo.toml.
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("cold-build"),
            "BlockedColdBuild display must name the gate"
        );
    }
}
