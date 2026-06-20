//! D2 — the kanban loop, the pure-SoA slice (no Lance, no ractor, no async).
//!
//! Proves the loop SHAPE the operator named — surrealdb (version tick) + ractor
//! (owner/driver) + lance-graph-planner (move policy) = one planner SoA — using
//! ONLY shipped contract types: `KanbanColumn`/`KanbanMove`/`ExecTarget`
//! (`kanban`), `MailboxSoaView`+`MailboxSoaOwner` (`soa_view`),
//! `NextPhaseScheduler`+`VersionScheduler::on_version`+`DatasetVersion`
//! (`scheduler`). This module is the ~glue: a `SymbiontBoard` owner over the
//! existing `Vec<NodeRow>` board-set that impls the two traits, driven by a `u32`
//! version tick standing in for the Lance subscription.
//!
//! IN-direction loop, verbatim from the contract (`scheduler.rs` §IN):
//!   version tick → `NextPhaseScheduler::on_version(view)` → `Option<KanbanMove>`
//!     → `owner.try_advance_phase(move.to)`   [CognitiveWork runs the Domino sweep]
//! Forward arc (`next_phases().first()`): `Planning → CognitiveWork`[sweep]` →
//! Evaluation → Commit` (absorbing → halt). The scheduler PROPOSES (`&view`); the
//! owner DISPOSES (`&mut`) — R1 read/write split, the same as in the contract.
//!
//! OWNERSHIP (operator, 2026-06-20): ractor is the **runtime ownership
//! guarantee**, NOT a message bus — "the mailbox-as-owner ... Rust move/ownership
//! semantics prove no aliasing / no data race / no use-after-free at compile
//! time" (CLAUDE.md E-CE64-MB-4; PR #477: nothing is serialized or transmitted
//! between mailboxes). So there is **no ractor message actor here and no tokio**:
//! `SymbiontBoard`'s single `&mut self` owner IS that guarantee, in plain Rust —
//! ractor would host the exact same ownership in prod (a structural/dummy
//! wrapper, never a message handler). `step()` drives the loop by direct owned
//! mutation, never by sending a message.
//!
//! THE TRIGGER IS SYNCHRONOUS — the writer fires it. `VersionScheduler::on_version(
//! &view, DatasetVersion(u64), exec)` is a **sync pure function** (contract
//! `scheduler.rs`); a batch writer that commits a SoA batch already KNOWS the
//! version it wrote, so it fires the kanban update inline — `on_version` →
//! `try_advance_phase` — with NO async. `surreal_container/tests/scheduler_seam.rs`
//! drives the WHOLE Rubicon arc this way with plain `#[test]`s feeding
//! `DatasetVersion(i)` directly, and `cognitive-shader-driver`'s `MailboxSoA`
//! (test 11) runs the same in-RAM OUT+IN loop ("no surreal / ractor message bus
//! needed", `mailbox_soa.rs:700`). This loop's `u32` `version_tick` IS that
//! pattern — the writer's monotonic version, lowered synchronously.
//!
//! Async enters in only two places, NEITHER of which is the kanban firing: the
//! Lance WRITE I/O itself, and the SUBSCRIPTION variant
//! `lance_graph::graph::scheduler::LanceVersionScheduler::drive_once` (async
//! SOLELY because it READS a version it did NOT write — opening `nodes.lance`;
//! shipped + 5 `#[tokio::test]`s) — for consumers that aren't the writer. The one
//! remaining stub is the SurrealQL re-read (`surreal_container::view::read_via_kv_lance`).

use lance_graph_contract::canonical_node::NodeRow;
use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
use lance_graph_contract::scheduler::{DatasetVersion, NextPhaseScheduler, VersionScheduler};
use lance_graph_contract::soa_view::{MailboxSoaOwner, MailboxSoaView};

use crate::domino;

/// A mailbox-as-owner over symbiont's flat `Vec<NodeRow>` board-set. The SoA
/// columns are kept parallel to the rows so the trait's zero-copy `&[T]` borrows
/// are real slices: `energy` is synced from the boards' `Energy` tenant after a
/// sweep; `edges`/`meta`/`entity` are zeroed for the POC (not read by
/// `NextPhaseScheduler`, whose policy is `phase`/`cycle`/`mailbox_id` only).
pub struct SymbiontBoard {
    rows: Vec<NodeRow>,
    energy: Vec<f32>,
    edges: Vec<u64>,
    meta: Vec<u32>,
    entity: Vec<u16>,
    phase: KanbanColumn,
    cycle: u32,
    mailbox: u32, // MailboxId = u32 (collapse_gate::MailboxId)
}

impl SymbiontBoard {
    /// Spawn a mailbox in `Planning` (the canonical spawn column) over `n_boards`
    /// seeded BF16-tile boards.
    pub fn spawn(n_boards: usize, mailbox: u32) -> Self {
        let rows = domino::seed_boards(n_boards);
        let n = rows.len();
        Self {
            rows,
            energy: vec![0.0; n],
            edges: vec![0; n],
            meta: vec![0; n],
            entity: vec![0; n],
            phase: KanbanColumn::Planning,
            cycle: 0,
            mailbox,
        }
    }

    /// Project each board's `Energy` tenant into the SoA energy column.
    fn sync_energy(&mut self) {
        for (e, row) in self.energy.iter_mut().zip(self.rows.iter()) {
            *e = domino::energy_of(row);
        }
    }

    /// The `u32` version tick — the stand-in for one Lance dataset `versions()`
    /// event (the IN-direction trigger).
    fn version_tick(&mut self) -> DatasetVersion {
        self.cycle += 1;
        DatasetVersion(self.cycle as u64)
    }

    /// The `CognitiveWork` phase: the BF16 Domino sweep over the boards, then the
    /// result projected into the energy column.
    fn cognitive_work(&mut self) {
        domino::domino_sweep(&mut self.rows, 3);
        self.sync_energy();
    }

    /// One IN-direction step: tick → scheduler PROPOSES → owner DISPOSES. Runs the
    /// sweep on the `CognitiveWork` crossing. Returns the applied move, or `None`
    /// once the mailbox has reached an absorbing column.
    pub fn step(&mut self, sched: &NextPhaseScheduler) -> Option<KanbanMove> {
        let at = self.version_tick();
        let proposed = sched.on_version(&*self, at, ExecTarget::Native)?;
        if proposed.to == KanbanColumn::CognitiveWork {
            self.cognitive_work();
        }
        self.try_advance_phase(proposed.to).ok()
    }

    /// Drive the forward arc to an absorbing column, returning the move trail.
    pub fn run_to_absorbing(&mut self, sched: &NextPhaseScheduler) -> Vec<KanbanMove> {
        let mut trail = Vec::new();
        while let Some(mv) = self.step(sched) {
            let absorbing = self.phase.is_absorbing();
            trail.push(mv);
            if absorbing {
                break;
            }
        }
        trail
    }
}

impl MailboxSoaView for SymbiontBoard {
    fn mailbox_id(&self) -> u32 {
        self.mailbox
    }
    fn n_rows(&self) -> usize {
        self.rows.len()
    }
    fn w_slot(&self) -> u8 {
        (self.mailbox & 0x3F) as u8
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
        &self.entity
    }
}

impl MailboxSoaOwner for SymbiontBoard {
    fn advance_phase(&mut self, to: KanbanColumn) -> KanbanMove {
        let from = self.phase;
        self.phase = to;
        let libet_offset_us =
            if from == KanbanColumn::Planning && to == KanbanColumn::CognitiveWork {
                -550_000
            } else {
                0
            };
        KanbanMove {
            mailbox: self.mailbox,
            from,
            to,
            witness_chain_position: self.cycle,
            libet_offset_us,
            exec: ExecTarget::Native,
        }
    }
}

/// The D2 demo: one mailbox drives the Rubicon forward arc; the `CognitiveWork`
/// crossing burns the BF16 Domino sweep through the SoA; the NaN-projection
/// surface keeps it finite; the mailbox halts at the absorbing `Commit`.
pub fn run_demo() {
    let mut board = SymbiontBoard::spawn(64, 7);
    let trail = board.run_to_absorbing(&NextPhaseScheduler);
    let arc: Vec<KanbanColumn> = trail.iter().map(|m| m.to).collect();
    let max_e = board.energy().iter().copied().fold(0.0_f32, f32::max);
    println!(
        "D2 kanban loop: mailbox {} ({} boards) — version-tick → NextPhaseScheduler → \
         try_advance_phase drove {arc:?}; CognitiveWork ran the BF16 Domino sweep; halted \
         absorbing at {:?} in {} cycles; max Energy = {max_e:.4}",
        board.mailbox_id(),
        board.n_rows(),
        board.phase(),
        board.current_cycle(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_drives_forward_arc_to_commit() {
        let mut board = SymbiontBoard::spawn(32, 1);
        assert_eq!(board.phase(), KanbanColumn::Planning);
        let trail = board.run_to_absorbing(&NextPhaseScheduler);
        let arc: Vec<KanbanColumn> = trail.iter().map(|m| m.to).collect();
        assert_eq!(
            arc,
            vec![
                KanbanColumn::CognitiveWork,
                KanbanColumn::Evaluation,
                KanbanColumn::Commit,
            ]
        );
        assert!(board.phase().is_absorbing());
        // the Planning→CognitiveWork crossing carries the Libet anchor; others 0.
        assert_eq!(trail[0].libet_offset_us, -550_000);
        assert_eq!(trail[1].libet_offset_us, 0);
        // monotonic cycle stamps (the SoA cycle-ownership stamp, R4).
        assert_eq!(
            trail.iter().map(|m| m.cycle()).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        // CognitiveWork actually ran the sweep, and it stayed finite (else the NaN
        // projection surface inside the sweep would have caught it).
        assert!(board.energy().iter().all(|e| e.is_finite()));
        assert!(board.energy().iter().any(|&e| e != 0.0));
    }

    #[test]
    fn illegal_skip_is_rejected_no_mutation() {
        let mut board = SymbiontBoard::spawn(16, 2);
        // Planning → Evaluation is not a legal Rubicon edge.
        assert!(board.try_advance_phase(KanbanColumn::Evaluation).is_err());
        assert_eq!(board.phase(), KanbanColumn::Planning);
    }
}
