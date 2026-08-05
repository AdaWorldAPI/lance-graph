//! Lane E — kanban-journaled batches over the direct exclusive owner.
//!
//! Per Addendum-13 lane E (see `README.md` §3), this lane measures the V3
//! kanban **journaling** tax on top of the SAME groupby-aggregate workload
//! lanes A/C/D already measure. The corpus is split into `batches`
//! newline-aligned chunks (`batches >= workers`, `chunk_bounds`), pulled by
//! `workers` puller tasks from a shared lock-free queue (`AtomicUsize`
//! index into the batch list), and EVERY batch is journaled as one kanban
//! card: a fresh [`ProbeBoard`] held `&mut` and driven through the full
//! Rubicon **forward arc** (`Planning -> CognitiveWork -> Evaluation ->
//! Commit`) around the actual per-batch work
//! ([`crate::lane_a_scalar`](super::lane_a_scalar)).
//!
//! ## 2026-08-05 migration — the actor variant is retired with the message path
//!
//! This lane originally spawned a `KanbanActor` per batch and drove it through
//! `KanbanMsg::Tick` RPCs — it was the last library consumer of that surface,
//! and its E−D reading existed to isolate journaling cost from the actor-model
//! tax lane D prices. The actor/tick surface was DELETED
//! (`E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1`: a version tick is knowledge,
//! never permission to advance; `&mut` IS the serialization). What lane E
//! prices now is the journaling itself — `KanbanMove` minting + collection at
//! kanban-card granularity over the direct exclusive owner, zero message
//! overhead. Lane D still prices the actor model on its own; the old E−D
//! "journaling minus actor tax" subtraction is retired with the actors.
//! This still feeds W2d (the 550 ms Libet budget question — how many kanban
//! cards per wall-clock second the substrate can journal).
//!
//! ## Journal invariant
//!
//! Each batch drives exactly 3 [`KanbanMove`]s (`Planning->CognitiveWork`,
//! `CognitiveWork->Evaluation`, `Evaluation->Commit` — the pure forward arc
//! to the absorbing `Commit` column). Every worker collects its own moves
//! into a local `Vec<KanbanMove>`; at the end of [`lane_e_kanban`] the
//! combined journal is asserted to have exactly `3 * batches` moves, and
//! every move is asserted legal via [`KanbanColumn::can_transition_to`] — a
//! violated assert here is a probe bug, not a measurement.

use crate::{chunk_bounds, lane_a_scalar, merge_maps, Stats};
use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
use lance_graph_contract::soa_view::{MailboxSoaOwner, MailboxSoaView};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// The probe's stand-in kanban board — a minimal in-RAM [`MailboxSoaView`] +
/// [`MailboxSoaOwner`] with empty column slices (`n_rows() == 0`, no
/// energy/edges/meta/entity_type data). This lane measures the KANBAN
/// JOURNALING overhead only, not SoA storage — a real SoA board wired to
/// actual rows is lane F's business (Morton-tile cascaded shader, per README
/// §5.1's closing note).
struct ProbeBoard {
    id: MailboxId,
    phase: KanbanColumn,
    cycle: u32,
}

impl ProbeBoard {
    /// A fresh board for kanban card `id`, starting at the spawn state
    /// ([`KanbanColumn::Planning`], the `#[default]` variant).
    fn new(id: MailboxId) -> Self {
        Self {
            id,
            phase: KanbanColumn::default(),
            cycle: 0,
        }
    }

    /// Advance one step along the Rubicon forward arc
    /// (`phase().next_phases().first()`), or `None` at an absorbing column.
    /// A plain `&mut` method — the exclusive borrow is the single-writer
    /// guarantee; no message, no RPC, no scheduler.
    fn forward_tick(&mut self) -> Option<KanbanMove> {
        self.phase
            .next_phases()
            .first()
            .map(|&to| self.advance_phase(to))
    }
}

impl MailboxSoaView for ProbeBoard {
    fn mailbox_id(&self) -> MailboxId {
        self.id
    }
    fn n_rows(&self) -> usize {
        0
    }
    fn w_slot(&self) -> u8 {
        // `id` here is a probe-local kanban-card counter, not a composed
        // classid — this is a plain bit-op over `MailboxId` (a plain `u32`),
        // not classid discrimination.
        (self.id & 0x3F) as u8
    }
    fn current_cycle(&self) -> u32 {
        self.cycle
    }
    fn phase(&self) -> KanbanColumn {
        self.phase
    }
    fn energy(&self) -> &[f32] {
        &[]
    }
    fn edges_raw(&self) -> &[u64] {
        &[]
    }
    fn meta_raw(&self) -> &[u32] {
        &[]
    }
    fn entity_type(&self) -> &[u16] {
        &[]
    }
}

impl MailboxSoaOwner for ProbeBoard {
    fn advance_phase(&mut self, to: KanbanColumn) -> KanbanMove {
        let from = self.phase;
        self.phase = to;
        self.cycle = self.cycle.wrapping_add(1);
        KanbanMove {
            mailbox: self.id,
            from,
            to,
            witness_chain_position: self.cycle,
            exec: ExecTarget::Native,
        }
    }
}

/// Lane E — kanban-journaled batches. See module doc for the design and the
/// 2026-08-05 migration off the actor surface.
///
/// `batches` is clamped to `>= workers.max(1)` — a batch queue thinner than
/// the worker pool would leave pullers idle and defeat the point of the
/// shared-queue design.
pub fn lane_e_kanban(data: &[u8], workers: usize, batches: usize) -> BTreeMap<String, Stats> {
    let workers = workers.max(1);
    let batches = batches.max(workers.max(1));
    let bounds = chunk_bounds(data, batches);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .build()
        .expect("build tokio runtime for lane E");

    runtime.block_on(async move {
        // One-time corpus copy into a shared Arc — the same boundary cost
        // Lane D pays (see `lane_d.rs` module doc).
        let shared = Arc::new(data.to_vec());
        let bounds = Arc::new(bounds);
        // Lock-free shared batch queue: each puller atomically claims the
        // next batch index until the queue is exhausted.
        let next = Arc::new(AtomicUsize::new(0));

        let mut join_handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let shared = Arc::clone(&shared);
            let bounds = Arc::clone(&bounds);
            let next = Arc::clone(&next);
            join_handles.push(tokio::spawn(async move {
                let mut local_map: BTreeMap<String, Stats> = BTreeMap::new();
                let mut journal: Vec<KanbanMove> = Vec::new();

                loop {
                    let idx = next.fetch_add(1, Ordering::Relaxed);
                    if idx >= bounds.len() {
                        break;
                    }
                    let (start, end) = bounds[idx];

                    // One kanban card per batch: a fresh exclusively-owned
                    // board starting at Planning. `&mut` is the single-writer
                    // guarantee — no actor, no message loop.
                    let mut board = ProbeBoard::new(idx as MailboxId);

                    // Step 1: Planning -> CognitiveWork.
                    let mv1 = board
                        .forward_tick()
                        .expect("Planning -> CognitiveWork must advance");
                    journal.push(mv1);

                    // The actual work — same per-record helper every lane
                    // shares (see `lib.rs` module doc "Reference inventory").
                    let batch_map = lane_a_scalar(&shared[start..end]);

                    // Step 2: CognitiveWork -> Evaluation. Merge the batch's
                    // map into the worker-local accumulator here — mirrors
                    // the commutative BUNDLE step `merge_maps` uses, applied
                    // per-batch instead of per-worker (see `Stats::merge`
                    // struct-level doc).
                    let mv2 = board
                        .forward_tick()
                        .expect("CognitiveWork -> Evaluation must advance");
                    journal.push(mv2);
                    for (name, stats) in batch_map {
                        match local_map.get_mut(&name) {
                            Some(existing) => existing.merge(&stats),
                            None => {
                                local_map.insert(name, stats);
                            }
                        }
                    }

                    // Step 3: Evaluation -> Commit (absorbing).
                    let mv3 = board
                        .forward_tick()
                        .expect("Evaluation -> Commit must advance");
                    journal.push(mv3);
                    debug_assert!(
                        board.forward_tick().is_none(),
                        "Commit is absorbing — a fourth forward tick must yield nothing"
                    );
                }

                (local_map, journal)
            }));
        }

        let mut worker_maps = Vec::with_capacity(join_handles.len());
        let mut all_moves: Vec<KanbanMove> = Vec::new();
        for jh in join_handles {
            let (map, journal) = jh.await.expect("lane E worker task join");
            worker_maps.push(map);
            all_moves.extend(journal);
        }

        // Journal invariant (see module doc "Journal invariant"): exactly 3
        // moves per batch, every move a legal Rubicon edge.
        assert_eq!(
            all_moves.len(),
            3 * batches,
            "lane E journal must record exactly 3 kanban moves per batch"
        );
        for mv in &all_moves {
            assert!(
                mv.from.can_transition_to(mv.to),
                "lane E journal move {:?} -> {:?} must be a legal Rubicon edge",
                mv.from,
                mv.to
            );
        }

        merge_maps(worker_maps)
    })
}
