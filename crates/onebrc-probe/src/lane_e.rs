//! Lane E — kanban-scheduled batches.
//!
//! Per Addendum-13 lane E (see `README.md` §3), this lane measures the V3
//! kanban scheduling/journaling tax on top of the SAME groupby-aggregate
//! workload lanes A/C/D already measure. The corpus is split into `batches`
//! newline-aligned chunks (`batches >= workers`, `chunk_bounds`), pulled by
//! `workers` puller tasks from a shared lock-free queue (`AtomicUsize`
//! index into the batch list), and EVERY batch is journaled as one kanban
//! card: a fresh [`KanbanActor`] (from `lance-graph-supervisor`, feature
//! `supervisor`) whose owned [`ProbeBoard`] is driven through the full
//! Rubicon **forward arc** (`Planning -> CognitiveWork -> Evaluation ->
//! Commit`) around the actual per-batch work
//! ([`crate::lane_a_scalar`](super::lane_a_scalar)).
//!
//! Two readings this lane is built to support:
//!
//! - **E at `batches == workers`** vs Lane D: identical `chunk_bounds`
//!   split, identical `Arc<Vec<u8>>` corpus-copy tax (see `lane_d.rs`
//!   module doc "Actor-model boundary cost") — the only variable is
//!   swapping Lane D's stateless `ChunkWorker` ask-pattern actor for a
//!   `KanbanActor<ProbeBoard>` driven through 3 Rubicon ticks per batch.
//!   E-D isolates the **journaling cost** in isolation from the actor-model
//!   tax Lane D already prices.
//! - **E at fine granularity** (`batches >> workers`, e.g.
//!   `batches = workers * 16`): each puller spawns, ticks 3×, and stops
//!   many short-lived actors instead of one long-lived one per worker —
//!   prices the **per-card scheduling overhead** the V3 substrate pays when
//!   work is journaled at kanban-card granularity rather than
//!   worker-chunk granularity. This feeds W2d (the 550 ms Libet budget
//!   question — how many kanban cards per wall-clock second the substrate
//!   can actually journal).
//!
//! ## Journal invariant
//!
//! Each batch drives exactly 3 [`KanbanMove`]s (`Planning->CognitiveWork`,
//! `CognitiveWork->Evaluation`, `Evaluation->Commit` — the pure forward arc
//! to the absorbing `Commit` column, mirroring `kanban_actor.rs`'s
//! `run_to_absorbing` test). Every worker collects its own moves into a
//! local `Vec<KanbanMove>`; at the end of [`lane_e_kanban`] the combined
//! journal is asserted to have exactly `3 * batches` moves, and every move
//! is asserted legal via [`KanbanColumn::can_transition_to`] — a violated
//! assert here is a probe bug, not a measurement.

use crate::{chunk_bounds, lane_a_scalar, merge_maps, Stats};
use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
use lance_graph_contract::scheduler::DatasetVersion;
use lance_graph_contract::soa_view::{MailboxSoaOwner, MailboxSoaView};
use lance_graph_supervisor::{drive_version_tick, KanbanActor};
use ractor::Actor;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// The probe's stand-in kanban-owned board — mirrors the shape of
/// `lance-graph-supervisor`'s own `TestBoard` (`kanban_actor.rs`'s test
/// module): a minimal in-RAM [`MailboxSoaView`] + [`MailboxSoaOwner`] with
/// empty column slices (`n_rows() == 0`, no energy/edges/meta/entity_type
/// data). This lane measures the KANBAN JOURNALING overhead only, not SoA
/// storage — a real SoA board wired to actual rows is lane F's business
/// (Morton-tile cascaded shader, per README §5.1's closing note).
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
        // classid — this is the same bit-op `TestBoard::w_slot` uses over
        // `MailboxId` (a plain `u32`), not classid discrimination.
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
            libet_offset_us: 0,
            exec: ExecTarget::Native,
        }
    }
}

/// Lane E — kanban-scheduled batches. See module doc for the full design
/// and the two readings (E vs D at `batches == workers`; E at fine
/// granularity for per-card scheduling cost).
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
        // One-time corpus copy into a shared Arc — the same actor-model
        // boundary cost Lane D pays (see `lane_d.rs` module doc).
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

                    // One kanban card per batch: a fresh KanbanActor whose
                    // owned board starts at Planning.
                    let (actor, handle) = Actor::spawn(
                        None,
                        KanbanActor::<ProbeBoard>::default(),
                        ProbeBoard::new(idx as MailboxId),
                    )
                    .await
                    .expect("spawn lane E kanban actor");

                    // Tick 1: Planning -> CognitiveWork.
                    let mv1 = drive_version_tick(&actor, DatasetVersion(1))
                        .await
                        .expect("lane E tick 1 rpc")
                        .expect("Planning -> CognitiveWork must advance");
                    journal.push(mv1);

                    // The actual work — same per-record helper every lane
                    // shares (see `lib.rs` module doc "Reference inventory").
                    let batch_map = lane_a_scalar(&shared[start..end]);

                    // Tick 2: CognitiveWork -> Evaluation. Merge the batch's
                    // map into the worker-local accumulator here — mirrors
                    // the commutative BUNDLE step `merge_maps` uses, applied
                    // per-batch instead of per-worker (see `Stats::merge`
                    // struct-level doc).
                    let mv2 = drive_version_tick(&actor, DatasetVersion(2))
                        .await
                        .expect("lane E tick 2 rpc")
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

                    // Tick 3: Evaluation -> Commit (absorbing).
                    let mv3 = drive_version_tick(&actor, DatasetVersion(3))
                        .await
                        .expect("lane E tick 3 rpc")
                        .expect("Evaluation -> Commit must advance");
                    journal.push(mv3);

                    actor.stop(None);
                    handle.await.expect("lane E actor join");
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
