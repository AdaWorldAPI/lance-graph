//! D-DRF W4–W7: flushed row images handed to the real `CycleWriter`
//! (lance-graph-supervisor) over a reconciliation-first fake WAL.
use drf_focus_probe::handoff::into_slots;
use drf_focus_probe::*;
use lance_graph_contract::scheduler::DatasetVersion;
use lance_graph_planner::persist_sink::{
    CommitError, CommitOutcome, CycleId, DetachedCycleBatch, FrameMeta, LandedSlot, SweepSlot,
    WalSink, WriteFailed,
};
use lance_graph_supervisor::cycle_driver::SealRecovery;
use lance_graph_supervisor::cycle_writer::{SealError, SubmitError, cycle_writer};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;

const NROWS: usize = 1000;
const OFF: usize = 16;
const OFF2: usize = 64;
/// A byte no test stages: changing it changes a row without any write touching it.
const UNSTAGED: usize = 200;
const OWNER: u32 = 7;

fn rows() -> Vec<Row> {
    (0..NROWS)
        .map(|i| {
            let mut r = [0u8; ROW];
            r[OFF..OFF + 4].copy_from_slice(&(i as i32).to_le_bytes());
            r
        })
        .collect()
}
fn sid() -> SourceId {
    SourceId {
        dataset: 1,
        branch: 1,
        version: 1,
        index_seal: 9,
    }
}
fn map_from(id: u64, mut f: impl FnMut(usize) -> u16) -> Arc<SealedMap> {
    let mut o = Box::new([0u16; SLOTS]);
    for (s, v) in o.iter_mut().enumerate() {
        *v = f(s);
    }
    Arc::new(SealedMap::new(MapId(id), 9, o))
}
fn facet(off: usize) -> Facet {
    Facet::new(off).unwrap()
}

/// What landed: one entry per durable commit.
#[derive(Clone)]
struct Landed {
    cycle: CycleId,
    batch_hash: u64,
    image: BTreeMap<u64, Vec<u8>>,
}

/// One-writer fake WAL. Reconciles by `(cycle, batch_hash)` before appending,
/// fences on `base_version`, can hold commits behind a gate, and can lose the
/// acknowledgement of one cycle (commit lands, caller sees `Ambiguous`).
struct FakeWal {
    head: u64,
    landed: Arc<Mutex<Vec<Landed>>>,
    gate: Option<Arc<Semaphore>>,
    entered: Arc<AtomicUsize>,
    lose_ack_of: Option<CycleId>,
}

impl FakeWal {
    fn new() -> Self {
        Self {
            head: 0,
            landed: Arc::default(),
            gate: None,
            entered: Arc::default(),
            lose_ack_of: None,
        }
    }
}

impl WalSink for FakeWal {
    async fn commit_cycle(
        &mut self,
        batch: DetachedCycleBatch,
    ) -> Result<CommitOutcome, CommitError> {
        self.entered.fetch_add(1, Ordering::SeqCst);
        if let Some(gate) = &self.gate {
            gate.acquire().await.expect("gate open").forget();
        }
        let (cycle, hash) = (batch.frame.cycle, batch.batch_hash);
        let existing = self
            .landed
            .lock()
            .unwrap()
            .iter()
            .find(|l| l.cycle == cycle)
            .map(|l| l.batch_hash);
        match existing {
            Some(h) if h == hash => {
                return Ok(CommitOutcome::Reconciled {
                    current_head: DatasetVersion(self.head),
                    cycle,
                    batch_hash: hash,
                });
            }
            Some(h) => {
                return Err(CommitError::HashConflict {
                    cycle,
                    stored_hash: h,
                    offered_hash: hash,
                });
            }
            None => {}
        }
        if batch.frame.base_version.0 != self.head {
            return Err(CommitError::Fenced {
                current_head: DatasetVersion(self.head),
            });
        }
        for (row, p) in &batch.image {
            if p.len() != ROW {
                return Err(CommitError::InvalidArtifact {
                    row: *row,
                    len: p.len(),
                });
            }
        }
        self.head += 1;
        self.landed.lock().unwrap().push(Landed {
            cycle,
            batch_hash: hash,
            image: batch.image,
        });
        if self.lose_ack_of.take() == Some(cycle) {
            return Err(CommitError::Ambiguous {
                cycle,
                batch_hash: hash,
                cause: "ack lost".into(),
            });
        }
        Ok(CommitOutcome::Committed {
            version: DatasetVersion(self.head),
            cycle,
            batch_hash: hash,
        })
    }

    async fn scan_sealed(&self, _after: Option<CycleId>) -> Result<Vec<LandedSlot>, WriteFailed> {
        Ok(Vec::new())
    }

    async fn timeline(&self) -> Result<Vec<FrameMeta>, WriteFailed> {
        Ok(Vec::new())
    }
}

/// Stage `n` writes over 37 distinct rows (slot s -> row (s % 37) * 11) and
/// flush them into casts for `cycle`.
fn dirty_cycle(src: &[Row], cycle: u64, next_pos: u64) -> (Vec<SweepSlot>, u64) {
    let soa = Soa {
        rows: src,
        id: sid(),
    };
    let m = map_from(400 + cycle, |s| ((s % 37) * 11) as u16);
    let mgr = FocusManager::new();
    let f = mgr.load(&soa, &m, Scope(1), facet(OFF)).unwrap();
    let mut wb = f.writeback();
    for s in 0..64 {
        wb.stage(s, facet(OFF), 1000 + ((s % 37) as i32)).unwrap();
    }
    into_slots(wb.flush(&soa).unwrap(), CycleId(cycle), OWNER, next_pos)
}

fn payloads(slots: &[SweepSlot]) -> Vec<(u64, Vec<u8>)> {
    slots.iter().map(|s| (s.row, s.payload.clone())).collect()
}

async fn until(n: &AtomicUsize, v: usize) {
    while n.load(Ordering::SeqCst) < v {
        tokio::task::yield_now().await;
    }
}

#[tokio::test(flavor = "current_thread")]
async fn w4_owned_handoff_outlives_the_source_and_durability_waits_for_the_wal() {
    let mut src = rows();
    let (slots, next) = dirty_cycle(&src, 1, 0);
    assert_eq!(slots.len(), 37, "one cast per distinct row");
    assert_eq!(next, 37);
    let expect: BTreeMap<u64, Vec<u8>> = slots.iter().map(|s| (s.row, s.payload.clone())).collect();
    // The focus and the borrow of `src` ended inside `dirty_cycle`; the thought
    // goes on and changes the source while the writer still holds the casts.
    // The changed byte is one no write staged, so a regeneration now would
    // produce different rows.
    for r in src.iter_mut() {
        r[UNSTAGED] ^= 0xFF;
    }
    let (now, _) = dirty_cycle(&src, 1, 0);
    assert!(now.iter().all(|s| s.payload != expect[&s.row]));

    let mut wal = FakeWal::new();
    let gate = Arc::new(Semaphore::new(0));
    wal.gate = Some(gate.clone());
    let (entered, landed) = (wal.entered.clone(), wal.landed.clone());
    let (writer, mut h) = cycle_writer(wal, DatasetVersion(0), 4);
    let run = tokio::spawn(writer.run());

    let mut ticket = h.try_submit(CycleId(1), slots).expect("queue has room");
    until(&entered, 1).await;
    // The WAL is holding the commit: nothing is durable yet.
    assert!(ticket.try_sealed().is_none());
    assert_eq!(h.durable().through, None);
    assert!(landed.lock().unwrap().is_empty());

    gate.add_permits(1);
    let sealed = ticket.sealed().await.expect("sealed");
    assert!(matches!(sealed.outcome, CommitOutcome::Committed { .. }));
    let mark = h.wait_durable(CycleId(1)).await.expect("durable");
    assert_eq!(mark.through, Some(CycleId(1)));
    assert_eq!(mark.head, DatasetVersion(1));
    // What landed is the flushed snapshot, not the source as it is now.
    assert_eq!(landed.lock().unwrap()[0].image, expect);

    // A second cycle continues the stream positions (recovery watermark).
    let (slots2, next2) = dirty_cycle(&src, 2, next);
    assert_eq!(slots2[0].stream_position, 37);
    assert_eq!(next2, 74);
    gate.add_permits(1);
    h.try_submit(CycleId(2), slots2)
        .unwrap()
        .sealed()
        .await
        .unwrap();
    drop(h);
    let exit = run.await.unwrap();
    assert_eq!(exit.mark.through, Some(CycleId(2)));
    assert_eq!(landed.lock().unwrap().len(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn w2_through_the_writer_two_facets_of_one_row_both_land() {
    let src = rows();
    let soa = Soa {
        rows: &src,
        id: sid(),
    };
    let m = map_from(500, |s| if s < 2 { 500 } else { (s % NROWS) as u16 });
    let mgr = FocusManager::new();
    let f = mgr.load(&soa, &m, Scope(1), facet(OFF)).unwrap();
    let mut wb = f.writeback();
    wb.stage(0, facet(OFF), 7).unwrap();
    wb.stage(1, facet(OFF2), 9).unwrap();
    let (slots, _) = into_slots(wb.flush(&soa).unwrap(), CycleId(1), OWNER, 0);

    // Contrast arm, built by hand: one cast per staged write, each the sealed
    // row with only its own facet. The writer's freeze keeps the later
    // payload whole, so the first write is lost. This is why the handoff takes
    // `flush` output and never raw patches.
    let raw: Vec<SweepSlot> = [(OFF, 7i32), (OFF2, 9)]
        .into_iter()
        .zip(10u64..)
        .map(|((off, v), pos)| {
            let mut p = src[500].to_vec();
            p[off..off + 4].copy_from_slice(&v.to_le_bytes());
            SweepSlot {
                cycle: CycleId(2),
                stream_position: pos,
                owner: OWNER,
                row: 500,
                paired_move: None,
                payload: p,
            }
        })
        .collect();

    let wal = FakeWal::new();
    let landed = wal.landed.clone();
    let (writer, h) = cycle_writer(wal, DatasetVersion(0), 4);
    let run = tokio::spawn(writer.run());
    h.try_submit(CycleId(1), slots)
        .unwrap()
        .sealed()
        .await
        .unwrap();
    h.try_submit(CycleId(2), raw)
        .unwrap()
        .sealed()
        .await
        .unwrap();
    drop(h);
    run.await.unwrap();

    let l = landed.lock().unwrap();
    let get = |c: usize, off: usize| {
        i32::from_le_bytes(l[c].image[&500][off..off + 4].try_into().unwrap())
    };
    assert_eq!(
        (get(0, OFF), get(0, OFF2)),
        (7, 9),
        "coalesced: both facets"
    );
    assert_eq!(get(1, OFF), 500, "raw: the first facet write was replaced");
    assert_eq!(get(1, OFF2), 9);
}

#[tokio::test(flavor = "current_thread")]
async fn w6_lost_ack_resubmits_the_same_frozen_casts_and_resampling_fails_closed() {
    let mut src = rows();
    let (slots, _) = dirty_cycle(&src, 1, 0);
    let original: Vec<(u64, Vec<u8>)> = slots.iter().map(|s| (s.row, s.payload.clone())).collect();

    let mut wal = FakeWal::new();
    wal.lose_ack_of = Some(CycleId(1));
    let landed = wal.landed.clone();
    let (writer, h) = cycle_writer(wal, DatasetVersion(0), 4);
    let run = tokio::spawn(writer.run());
    let err = h
        .try_submit(CycleId(1), slots)
        .unwrap()
        .sealed()
        .await
        .unwrap_err();
    let failure = match err {
        SealError::Failed(f) => f,
        other => panic!("expected Failed, got {other:?}"),
    };
    assert_eq!(failure.recovery(), SealRecovery::ResubmitFrozen);
    // The handed-back casts are the frozen batch, byte for byte.
    let back: Vec<(u64, Vec<u8>)> = failure
        .casts
        .iter()
        .map(|s| (s.row, s.payload.clone()))
        .collect();
    assert_eq!(back, original);
    // The writer is poisoned: the next submission comes back with its casts.
    // The writer closes its queue before it answers the failed ticket.
    assert!(matches!(
        h.try_submit(CycleId(2), Vec::new()),
        Err(SubmitError::Closed { .. })
    ));
    drop(h);
    let exit = run.await.unwrap();
    assert_eq!(landed.lock().unwrap().len(), 1, "it did land once");

    // Meanwhile the source moved on, in a byte no write staged.
    for r in src.iter_mut() {
        r[UNSTAGED] ^= 0x55;
    }
    // Every arm below frames cycle 1 on its own base, 0. The batch hash covers
    // the base, so a different base alone would already conflict; holding it
    // fixed leaves the payload bytes as the only thing that can differ.
    assert_eq!(exit.mark.head, DatasetVersion(0));

    // Right: re-submit the SAME frozen casts on a fresh writer. Reconciled, no
    // second append.
    let (writer, h) = cycle_writer(exit.sink, DatasetVersion(0), 4);
    let run = tokio::spawn(writer.run());
    let ok = h
        .try_submit(CycleId(1), failure.casts.clone())
        .unwrap()
        .sealed()
        .await
        .unwrap();
    assert!(matches!(ok.outcome, CommitOutcome::Reconciled { .. }));
    assert_eq!(landed.lock().unwrap().len(), 1);

    drop(h);
    let exit = run.await.unwrap();

    // Control: regenerating from the source as it WAS gives the same batch,
    // so it reconciles too. Regeneration itself is not what conflicts.
    let (same, _) = dirty_cycle(&rows(), 1, 0);
    assert_eq!(payloads(&same), original);
    let (writer, h) = cycle_writer(exit.sink, DatasetVersion(0), 4);
    let run = tokio::spawn(writer.run());
    let ok = h
        .try_submit(CycleId(1), same)
        .unwrap()
        .sealed()
        .await
        .unwrap();
    assert!(matches!(ok.outcome, CommitOutcome::Reconciled { .. }));
    drop(h);
    let exit = run.await.unwrap();

    // Wrong: regenerate cycle 1 from the live (changed) source. Different
    // bytes, same base, different hash: the WAL fails closed instead of
    // overwriting.
    let (resampled, _) = dirty_cycle(&src, 1, 0);
    assert!(
        payloads(&resampled)
            .iter()
            .zip(&original)
            .all(|(a, b)| a.0 == b.0 && a.1 != b.1)
    );
    let (writer, h) = cycle_writer(exit.sink, DatasetVersion(0), 4);
    let run = tokio::spawn(writer.run());
    let err = h
        .try_submit(CycleId(1), resampled)
        .unwrap()
        .sealed()
        .await
        .unwrap_err();
    match err {
        SealError::Failed(f) => assert_eq!(f.recovery(), SealRecovery::Escalate),
        other => panic!("expected Failed, got {other:?}"),
    }
    assert_eq!(landed.lock().unwrap().len(), 1);
    drop(h);
    run.await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn w6_backpressure_hands_the_casts_back_untouched() {
    let src = rows();
    let mut wal = FakeWal::new();
    let gate = Arc::new(Semaphore::new(0));
    wal.gate = Some(gate.clone());
    let entered = wal.entered.clone();
    let (writer, h) = cycle_writer(wal, DatasetVersion(0), 1);
    let run = tokio::spawn(writer.run());
    let (a, next) = dirty_cycle(&src, 1, 0);
    let t1 = h.try_submit(CycleId(1), a).unwrap();
    until(&entered, 1).await; // cycle 1 is in the WAL, the queue is empty
    let (b, next) = dirty_cycle(&src, 2, next);
    let t2 = h.try_submit(CycleId(2), b).unwrap(); // fills the queue (depth 1)
    let (c, _) = dirty_cycle(&src, 3, next);
    let want: Vec<Vec<u8>> = c.iter().map(|s| s.payload.clone()).collect();
    match h.try_submit(CycleId(3), c) {
        Err(SubmitError::Full { casts, .. }) => {
            assert_eq!(
                casts.iter().map(|s| s.payload.clone()).collect::<Vec<_>>(),
                want
            )
        }
        other => panic!("expected Full, got {:?}", other.map(|t| t.cycle)),
    }
    gate.add_permits(2);
    t1.sealed().await.unwrap();
    t2.sealed().await.unwrap();
    drop(h);
    run.await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn w7_a_cycle_with_no_dirty_row_publishes_nothing() {
    let src = rows();
    let soa = Soa {
        rows: &src,
        id: sid(),
    };
    let m = map_from(700, |s| (s % NROWS) as u16);
    let mgr = FocusManager::new();
    let f = mgr.load(&soa, &m, Scope(1), facet(OFF)).unwrap();
    let (none, next) = into_slots(f.writeback().flush(&soa).unwrap(), CycleId(1), OWNER, 0);
    assert!(none.is_empty());
    assert_eq!(next, 0);

    let wal = FakeWal::new();
    let (entered, landed) = (wal.entered.clone(), wal.landed.clone());
    let (writer, mut h) = cycle_writer(wal, DatasetVersion(0), 4);
    let run = tokio::spawn(writer.run());
    let s = h
        .try_submit(CycleId(1), none)
        .unwrap()
        .sealed()
        .await
        .unwrap();
    assert!(matches!(s.outcome, CommitOutcome::NoChange { .. }));
    assert_eq!(entered.load(Ordering::SeqCst), 0, "the WAL is never called");
    assert_eq!(
        h.wait_durable(CycleId(1)).await.unwrap().head,
        DatasetVersion(0)
    );
    // A dirty cycle after it publishes exactly one version.
    let (slots, _) = dirty_cycle(&src, 2, next);
    h.try_submit(CycleId(2), slots)
        .unwrap()
        .sealed()
        .await
        .unwrap();
    assert_eq!(
        h.wait_durable(CycleId(2)).await.unwrap().head,
        DatasetVersion(1)
    );
    assert_eq!(landed.lock().unwrap().len(), 1);
    drop(h);
    run.await.unwrap();
}
