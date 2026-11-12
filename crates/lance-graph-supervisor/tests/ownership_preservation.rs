#![cfg(feature = "cycle-driver")]
//! Ownership falsifier suite (D-SCHEMA-WARDROBE-P0).
//!
//! Real `MailboxSoA<8>` owners, the real `BatchWriter` / `collect_casts` /
//! `seal_cycle` / `apply_sealed_transitions` / `cycle_writer` path, and one
//! in-file `MemWal` sink. No production code is added or changed here.

use std::collections::HashMap;

use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
use lance_graph_contract::scheduler::DatasetVersion;
use lance_graph_contract::soa_view::{MailboxSoaOwner, MailboxSoaView};
use lance_graph_planner::batch_writer::BatchWriter;
use lance_graph_planner::persist_sink::{
    CommitError, CommitOutcome, CycleFrame, CycleId, DetachedCycleBatch, FrameMeta, LandedSlot,
    PersistError, SweepSlot, WalSink, WriteFailed,
};
use lance_graph_supervisor::cycle_driver::{
    apply_sealed_transitions, collect_casts, seal_cycle, SealRecovery,
};
use lance_graph_supervisor::cycle_writer::{cycle_writer, SubmitError};

type Owner = MailboxSoA<8>;

fn owner(id: MailboxId) -> Owner {
    MailboxSoA::new(id, (id & 0x3F) as u8, 1.0)
}

fn run<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(f)
}

fn mv(mailbox: MailboxId, from: KanbanColumn, to: KanbanColumn) -> KanbanMove {
    KanbanMove {
        mailbox,
        from,
        to,
        witness_chain_position: 0,
        exec: ExecTarget::Native,
    }
}

/// A legal Planning -> CognitiveWork move naming `mailbox`.
fn legal(mailbox: MailboxId) -> KanbanMove {
    mv(mailbox, KanbanColumn::Planning, KanbanColumn::CognitiveWork)
}

/// One artifact cast (non-empty payload, no move) for `owner` through the real
/// writer + `collect_casts` path.
fn cycle_slots(cycle: u64, owner: MailboxId, pos_base: u64) -> Vec<SweepSlot> {
    let mut w: BatchWriter<Vec<u8>> = BatchWriter::new();
    w.cast(owner, vec![], vec![0xAB, cycle as u8]);
    collect_casts(&mut w, CycleId(cycle), pos_base, |_| 0).slots
}

/// In-process sink: counts commits, reconciles by (cycle, batch_hash), fences
/// on a stale base version, and can fail the next commit with a retryable error.
struct MemWal {
    commits: u64,
    fail_next: bool,
    durable: Vec<(CycleId, u64)>,
    head: u64,
}

impl MemWal {
    fn new() -> Self {
        Self {
            commits: 0,
            fail_next: false,
            durable: Vec::new(),
            head: 0,
        }
    }
}

impl WalSink for MemWal {
    async fn commit_cycle(
        &mut self,
        batch: DetachedCycleBatch,
    ) -> Result<CommitOutcome, CommitError> {
        if std::mem::take(&mut self.fail_next) {
            return Err(CommitError::Io(WriteFailed("injected".into())));
        }
        let cycle = batch.frame.cycle;
        if let Some(&(_, hash)) = self.durable.iter().find(|(c, _)| *c == cycle) {
            return if hash == batch.batch_hash {
                Ok(CommitOutcome::Reconciled {
                    current_head: DatasetVersion(self.head),
                    cycle,
                    batch_hash: hash,
                })
            } else {
                Err(CommitError::HashConflict {
                    cycle,
                    stored_hash: hash,
                    offered_hash: batch.batch_hash,
                })
            };
        }
        if batch.frame.base_version != DatasetVersion(self.head) {
            return Err(CommitError::Fenced {
                current_head: DatasetVersion(self.head),
            });
        }
        self.commits += 1;
        self.head += 1;
        self.durable.push((cycle, batch.batch_hash));
        Ok(CommitOutcome::Committed {
            version: DatasetVersion(self.head),
            cycle,
            batch_hash: batch.batch_hash,
        })
    }

    async fn scan_sealed(
        &self,
        _after_cycle: Option<CycleId>,
    ) -> Result<Vec<LandedSlot>, WriteFailed> {
        Ok(Vec::new())
    }

    async fn timeline(&self) -> Result<Vec<FrameMeta>, WriteFailed> {
        Ok(Vec::new())
    }
}

/// FAILS IF: the owner's `cast_on_behalf` records any mailbox other than its
/// own id on the cast.
#[test]
fn the_owner_stamps_its_own_cast() {
    let o = owner(7);
    let mut writer: BatchWriter<Vec<u8>> = BatchWriter::new();
    let cast = o.cast_on_behalf(&mut writer, vec![], vec![1]);
    assert_eq!(writer.on_behalf_of(cast), Some(7));
}

/// FAILS IF: the public `BatchWriter::cast` starts demanding owner evidence
/// (it then stops accepting a forged owner id) or records a different owner.
///
/// This pins a GAP, not a guarantee: code holding NO mailbox can cast on behalf
/// of mailbox 7 and it is recorded exactly like the owner's own cast. Re-pin
/// this test when an enforcement seam lands on the public cast.
#[test]
fn finding_public_cast_accepts_any_mailbox_id() {
    let mut writer: BatchWriter<Vec<u8>> = BatchWriter::new();
    let cast = writer.cast(7, vec![], vec![0xAB]);
    assert_eq!(writer.on_behalf_of(cast), Some(7));
    let collected = collect_casts(&mut writer, CycleId(1), 0, |_| 0);
    assert_eq!(collected.slots.len(), 1);
    assert_eq!(collected.slots[0].owner, 7);
}

/// FAILS IF: a cast on behalf of mailbox 7 carrying a move minted for mailbox 9
/// is sealed, or if either real mailbox changes phase because of it; or if the
/// correctly-owned control move no longer advances mailbox 7.
#[test]
fn a_forged_cross_owner_move_is_refused_before_it_changes_a_mailbox() {
    let mut fleet: HashMap<MailboxId, Owner> = HashMap::from([(7, owner(7)), (9, owner(9))]);
    let mut sink = MemWal::new();
    let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();

    let mut w: BatchWriter<Vec<u8>> = BatchWriter::new();
    fleet[&7].cast_on_behalf(&mut w, vec![legal(9)], vec![0xAB]);
    let collected = collect_casts(&mut w, CycleId(1), 0, |_| 0);
    let failure = run(seal_cycle(
        &mut sink,
        CycleFrame::new(CycleId(1), DatasetVersion(0)),
        collected.slots,
    ))
    .expect_err("a cross-owner move must not seal");
    assert_eq!(
        failure.cause,
        PersistError::OwnerMismatch {
            move_owner: 9,
            landing_owner: 7
        }
    );
    assert_eq!(sink.commits, 0, "nothing reached the store");
    assert_eq!(fleet[&7].phase(), KanbanColumn::Planning);
    assert_eq!(fleet[&9].phase(), KanbanColumn::Planning);

    // Positive control: the same flow with a correctly-owned move advances 7 only.
    let mut w: BatchWriter<Vec<u8>> = BatchWriter::new();
    fleet[&7].cast_on_behalf(&mut w, vec![legal(7)], vec![0xAB]);
    let collected = collect_casts(&mut w, CycleId(2), 0, |_| 0);
    let sealed = run(seal_cycle(
        &mut sink,
        CycleFrame::new(CycleId(2), DatasetVersion(0)),
        collected.slots,
    ))
    .expect("a correctly-owned move seals");
    let applied = apply_sealed_transitions(&mut fleet, &sealed, &mut wm).expect("applies");
    assert_eq!(applied.applied.len(), 1);
    assert_eq!(fleet[&7].phase(), KanbanColumn::CognitiveWork);
    assert_eq!(fleet[&9].phase(), KanbanColumn::Planning);
}

/// FAILS IF: casting (staging intent) commits anything to the sink or moves the
/// owner's phase before a seal runs.
#[test]
fn a_cast_is_not_a_commit() {
    let o = owner(7);
    let sink = MemWal::new();
    let mut w: BatchWriter<Vec<u8>> = BatchWriter::new();
    o.cast_on_behalf(&mut w, vec![legal(7)], vec![0xAB]);
    assert_eq!(w.casts().len(), 1, "the cast was really recorded");
    assert_eq!(sink.commits, 0);
    assert_eq!(o.phase(), KanbanColumn::Planning);
}

/// FAILS IF: a failed seal advances an owner, commits, or loses the casts; or
/// if the retry of the returned frame + casts does not commit once and advance
/// the owner exactly one legal step.
#[test]
fn a_failed_seal_advances_nothing_and_returns_the_casts() {
    let mut fleet: HashMap<MailboxId, Owner> = HashMap::from([(7, owner(7))]);
    let mut sink = MemWal::new();
    let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();

    let mut w: BatchWriter<Vec<u8>> = BatchWriter::new();
    fleet[&7].cast_on_behalf(&mut w, vec![legal(7)], vec![0xAB]);
    let collected = collect_casts(&mut w, CycleId(1), 0, |_| 0);
    let submitted = collected.slots.clone();
    assert_eq!(submitted.len(), 1);

    sink.fail_next = true;
    let failure = run(seal_cycle(
        &mut sink,
        CycleFrame::new(CycleId(1), DatasetVersion(0)),
        collected.slots,
    ))
    .expect_err("injected commit failure");
    assert_eq!(failure.casts, submitted, "the casts come back unchanged");
    assert_eq!(failure.recovery(), SealRecovery::Regenerate);
    assert_eq!(fleet[&7].phase(), KanbanColumn::Planning);
    assert_eq!(sink.commits, 0);

    let failure = *failure;
    let sealed = run(seal_cycle(&mut sink, failure.frame, failure.casts)).expect("retry seals");
    assert_eq!(sink.commits, 1);
    let applied = apply_sealed_transitions(&mut fleet, &sealed, &mut wm).expect("applies");
    assert_eq!(applied.applied.len(), 1);
    assert_eq!(fleet[&7].phase(), KanbanColumn::CognitiveWork);
}

/// FAILS IF: a full queue swallows the casts instead of handing them back
/// untouched in `SubmitError::Full`.
#[test]
fn a_full_queue_hands_the_casts_back() {
    // The writer is held but never run, so the queue (depth 1) stays occupied.
    let (_writer, handle) = cycle_writer(MemWal::new(), DatasetVersion(0), 1);
    let first = cycle_slots(1, 7, 0);
    let _ticket = handle
        .try_submit(CycleId(1), first)
        .expect("the first cycle fits");

    let second = cycle_slots(2, 7, 100);
    let expected = second.clone();
    assert!(!expected.is_empty());
    match handle.try_submit(CycleId(2), second) {
        Err(SubmitError::Full { cycle, casts }) => {
            assert_eq!(cycle, CycleId(2));
            assert_eq!(casts, expected);
        }
        other => panic!("a depth-1 queue holding one cycle must be full, got {other:?}"),
    }
}

/// FAILS IF: a second cycle framed on a stale base version commits, or the
/// fenced error is not the cause; or if the same cycle re-framed on the current
/// head no longer commits.
#[test]
fn a_second_writer_on_a_stale_head_is_fenced() {
    let mut sink = MemWal::new();
    let first = run(seal_cycle(
        &mut sink,
        CycleFrame::new(CycleId(1), DatasetVersion(0)),
        cycle_slots(1, 7, 0),
    ))
    .expect("the first writer commits");
    assert!(matches!(first.outcome, CommitOutcome::Committed { .. }));
    assert_eq!(sink.commits, 1);

    let failure = run(seal_cycle(
        &mut sink,
        CycleFrame::new(CycleId(2), DatasetVersion(0)),
        cycle_slots(2, 8, 100),
    ))
    .expect_err("a stale base version must be fenced");
    assert_eq!(
        failure.cause,
        PersistError::Commit(CommitError::Fenced {
            current_head: DatasetVersion(1)
        })
    );
    assert_eq!(sink.commits, 1, "the fenced writer published nothing");

    // Positive control: framed on the current head, the same cycle commits.
    let again = run(seal_cycle(
        &mut sink,
        CycleFrame::new(CycleId(2), DatasetVersion(1)),
        cycle_slots(2, 8, 100),
    ))
    .expect("a fresh base version commits");
    assert!(matches!(again.outcome, CommitOutcome::Committed { .. }));
    assert_eq!(sink.commits, 2);
}

/// FAILS IF: the read view returns a slice that is not the owner's own column
/// (a second population), for any slice accessor the view trait has.
#[test]
fn a_view_is_the_owners_own_columns() {
    let mut o = owner(7);
    o.energy[3] = 1.5;
    let view: &dyn MailboxSoaView = &o;
    assert_eq!(view.energy().as_ptr(), o.energy.as_ptr());
    assert_eq!(
        view.energy()[3],
        1.5,
        "the view reads the owner's own value"
    );
    assert_eq!(view.edges_raw().as_ptr(), o.edges.as_ptr() as *const u64);
    assert_eq!(view.meta_raw().as_ptr(), o.meta.as_ptr() as *const u32);
    assert_eq!(view.entity_type().as_ptr(), o.entity_type.as_ptr());
    assert_eq!(view.class_id().as_ptr(), o.entity_type.as_ptr());
}

/// FAILS IF: advancing one mailbox changes another's phase or slices, or two
/// mailboxes share a backing store.
#[test]
fn two_mailboxes_do_not_share_state() {
    let mut a = owner(7);
    let b = owner(9);
    let b_energy = b.energy().as_ptr();
    let b_edges = b.edges_raw().as_ptr();
    assert_ne!(a.energy().as_ptr(), b_energy);
    assert_ne!(a.edges_raw().as_ptr(), b_edges);

    let step = a
        .try_advance_phase(KanbanColumn::CognitiveWork)
        .expect("Planning -> CognitiveWork is legal");
    assert_eq!(step.mailbox, 7);
    assert_eq!(a.phase(), KanbanColumn::CognitiveWork);
    assert_eq!(b.phase(), KanbanColumn::Planning);
    assert_eq!(b.energy().as_ptr(), b_energy);
    assert_eq!(b.edges_raw().as_ptr(), b_edges);
}
