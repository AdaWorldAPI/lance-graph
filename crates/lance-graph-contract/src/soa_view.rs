//! # `soa_view` — the transparent, zero-copy read view over the ONE SoA.
//!
//! **R1 "one SoA never transformed":** the per-mailbox SoA is never serialized or
//! copied; it lives from mailbox spawn to tombstone and is mutated only by
//! cognitive operations. This module is the **zero-dep borrow vocabulary** that
//! lets three holders read the SAME bytes:
//!
//! - `cognitive-shader-driver`'s `MailboxSoA<N>` — the in-RAM hot owner (implements
//!   [`MailboxSoaOwner`]; ractor drives it),
//! - `surreal_container` — the transparent kv-lance-backed VIEW (implements the
//!   read-only [`MailboxSoaView`] over the same Lance columns; no Arrow re-encode),
//! - `lance-graph-planner` — a CONSUMER (plans over the columns directly).
//!
//! The contract owns **no** SoA storage — only this lens. It cannot name
//! `MailboxSoA<N>` (another crate) without a dependency, so the lens is a trait the
//! owner/view implement — the same dependency-inversion pattern as
//! [`crate::plan::PlannerContract`] and [`crate::orchestration::OrchestrationBridge`].

use crate::collapse_gate::MailboxId;
use crate::kanban::{KanbanColumn, KanbanMove};

/// A transparent, read-only view over one mailbox's SoA columns.
///
/// Implementors return **borrows** (`&[T]`) or `Copy` scalars — never clones of the
/// backing store. A `surreal_container` view and the in-RAM `MailboxSoA` are both
/// valid implementors over the *same* bytes; that two-implementor symmetry is what
/// "transparent view" means here (R1).
pub trait MailboxSoaView {
    /// Identity of the mailbox this view reads.
    fn mailbox_id(&self) -> MailboxId;
    /// Number of populated rows in the SoA.
    fn n_rows(&self) -> usize;
    /// 6-bit witness-table slot (0..=63) the mailbox occupies.
    fn w_slot(&self) -> u8;
    /// Monotonic cognitive cycle stamp.
    fn current_cycle(&self) -> u32;
    /// The Rubicon phase the mailbox is currently in (kanban column).
    fn phase(&self) -> KanbanColumn;

    // ── zero-copy column borrows (the SIMD / surreal-projection surface) ──

    /// Per-row spatial-temporal energy accumulator.
    fn energy(&self) -> &[f32];
    /// Per-row packed `CausalEdge64` as raw `u64` (reconstruct via `CausalEdge64(raw)`;
    /// kept raw so the contract stays zero-dep — `causal-edge` is not a contract dep).
    fn edges_raw(&self) -> &[u64];
    /// Per-row packed `MetaWord` as raw `u32`.
    fn meta_raw(&self) -> &[u32];
    /// Per-row entity-type id.
    fn entity_type(&self) -> &[u16];

    /// Per-row **class discriminator** — the Cognitive-RISC `class_id` / `shape_id`
    /// (a.k.a. the OGIT `EntityTypeId`). Aliases
    /// [`entity_type`](MailboxSoaView::entity_type) today: the existing `u16` slot IS
    /// the class hook, so no new column is added (honors R1 "one SoA never
    /// transformed"). Only the `u16` discriminator lives on the SoA; the machinery it
    /// keys — label inheritance, column projection, jinja templates — resolves ONE
    /// LAYER UP via the OGIT ontology cache (`lance-graph-ontology`), never in the SoA
    /// / kv-lance columns. This is the Cognitive-RISC N1 freeze-time hook.
    #[inline]
    fn class_id(&self) -> &[u16] {
        self.entity_type()
    }

    /// The `class_id` of a single row.
    #[inline]
    fn class_id_at(&self, row: usize) -> u16 {
        self.entity_type()[row]
    }

    // NOTE (follow-up): the qualia column (`QualiaI4_16D`) accessor is intentionally omitted —
    // add `fn qualia(&self) -> &[crate::qualia::QualiaI4_16D]` when the first consumer
    // (planner strategy selection) needs it; keep the read surface minimal until then.

    // ── per-row scalar read (mirrors `MailboxSoA::energy_at`) ──

    /// Energy at `row`. Default indexes [`energy`](MailboxSoaView::energy); override
    /// if the implementor can read a single row more cheaply.
    #[inline]
    fn energy_at(&self, row: usize) -> f32 {
        self.energy()[row]
    }
}

/// The mutation airgap for the SoA **owner** only (the ractor-driven hot path).
///
/// A read-only view (e.g. `surreal_container`) deliberately does **not** implement
/// this — that is what makes "the view is read-only" a structural guarantee rather
/// than a convention. Only the in-RAM `MailboxSoA` owner advances phases.
pub trait MailboxSoaOwner: MailboxSoaView {
    /// Drive one Rubicon phase transition to `to`; return the emitted move.
    ///
    /// The only mutation surface at the contract level: cognitive operations advance
    /// the lifecycle column. The SoA columns themselves are mutated by the owner's
    /// own (crate-private) cognitive ops, never serialized through here (R1).
    fn advance_phase(&mut self, to: KanbanColumn) -> KanbanMove;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal in-memory implementor proving the trait is satisfiable and the
    /// `&[T]` borrows compile + read zero-copy — without any consumer crate.
    struct FakeSoa {
        id: MailboxId,
        phase: KanbanColumn,
        energy: Vec<f32>,
        edges: Vec<u64>,
        meta: Vec<u32>,
        etype: Vec<u16>,
        cycle: u32,
    }

    impl MailboxSoaView for FakeSoa {
        fn mailbox_id(&self) -> MailboxId {
            self.id
        }
        fn n_rows(&self) -> usize {
            self.energy.len()
        }
        fn w_slot(&self) -> u8 {
            (self.id & 0x3F) as u8
        }
        fn current_cycle(&self) -> u32 {
            self.cycle
        }
        fn phase(&self) -> KanbanColumn {
            self.phase
        }
        fn energy(&self) -> &[f32] {
            &self.energy
        }
        fn edges_raw(&self) -> &[u64] {
            &self.edges
        }
        fn meta_raw(&self) -> &[u32] {
            &self.meta
        }
        fn entity_type(&self) -> &[u16] {
            &self.etype
        }
    }

    impl MailboxSoaOwner for FakeSoa {
        fn advance_phase(&mut self, to: KanbanColumn) -> KanbanMove {
            let from = self.phase;
            self.phase = to;
            KanbanMove {
                mailbox: self.id,
                from,
                to,
                witness_chain_position: 0,
                libet_offset_us: if to == KanbanColumn::CognitiveWork {
                    -550_000
                } else {
                    0
                },
            }
        }
    }

    fn sample() -> FakeSoa {
        FakeSoa {
            id: 7,
            phase: KanbanColumn::Planning,
            energy: vec![0.1, 0.2, 0.3],
            edges: vec![0, 1, 2],
            meta: vec![10, 11, 12],
            etype: vec![100, 101, 102],
            cycle: 1,
        }
    }

    #[test]
    fn view_reads_columns_zero_copy() {
        let soa = sample();
        // Borrow points INTO the backing store (zero-copy): identical pointer.
        assert_eq!(soa.energy().as_ptr(), soa.energy.as_ptr());
        assert_eq!(soa.n_rows(), 3);
        assert_eq!(soa.edges_raw(), &[0, 1, 2]);
        assert_eq!(soa.meta_raw(), &[10, 11, 12]);
        assert_eq!(soa.entity_type(), &[100, 101, 102]);
        // class_id is the Cognitive-RISC N1 hook aliasing the entity_type slot.
        assert_eq!(soa.class_id(), &[100, 101, 102]);
        assert_eq!(soa.class_id_at(0), 100);
        assert_eq!(soa.energy_at(1), 0.2);
        assert_eq!(soa.phase(), KanbanColumn::Planning);
        assert_eq!(soa.w_slot(), 7);
    }

    #[test]
    fn owner_advances_phase_and_sets_libet_anchor() {
        let mut soa = sample();
        let m = soa.advance_phase(KanbanColumn::CognitiveWork);
        assert_eq!(m.from, KanbanColumn::Planning);
        assert_eq!(m.to, KanbanColumn::CognitiveWork);
        assert_eq!(m.libet_offset_us, -550_000);
        assert_eq!(soa.phase(), KanbanColumn::CognitiveWork);
    }
}
