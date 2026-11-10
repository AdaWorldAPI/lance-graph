//! A DR map table made durable through the fire-and-forget cycle writer.
//!
//! - The owner submits and returns at once; an advance is applied only after
//!   its cycle seals, at the route version claimed when it was built.
//! - A failed seal applies nothing and leaves a gap.
//! - Every landing is on behalf of the owning mailbox.
//! - A restore answers every past version exactly.
//!
//! The interleavings are driven deterministically by a gated in-memory sink
//! (`GateWal`): commits wait on a semaphore the test opens, and a chosen cycle
//! fails. The round trip runs on the real `LanceCycleWriter`.

use std::collections::BTreeMap;
use std::sync::Arc;

use lance_graph::graph::cycle_sink::LanceCycleWriter;
use lance_graph_contract::scheduler::DatasetVersion;
use lance_graph_dt64::dr::{
    FrameRefusal, LocalMap, MapRead, MapTable, RouteVersion, VersionClock, Width,
};
use lance_graph_dt64_lance::{
    restore, AdvanceError, MapLog, ReplayError, RestoreError, SealedImages, Settled, RESERVED_ID,
};
use lance_graph_planner::persist_sink::{
    CommitError, CommitOutcome, CycleId, DetachedCycleBatch, FrameMeta, LandedSlot, WalSink,
    WriteFailed,
};
use lance_graph_supervisor::cycle_writer::{cycle_writer, SealError};
use tokio::sync::Semaphore;

const OWNER: u32 = 42;
const TILES: [u32; 3] = [1, 2, 3];

fn map(w: Width, ids: &[u64]) -> Option<LocalMap> {
    Some(LocalMap::prepare(w, ids).unwrap())
}

/// What a table answers for `tile` at `version`: `(width, globals)`, or the
/// refusal.
fn answer(t: &MapTable, version: u64, tile: u32) -> Result<(u16, Vec<u64>), MapRead> {
    let d = t.diagnose_at(RouteVersion(version), tile)?;
    let width = match d.width() {
        Width::W64 => 64,
        Width::W256 => 256,
    };
    let globals = (0..d.len())
        .map(|l| d.global(u8::try_from(l).unwrap()).unwrap())
        .collect();
    Ok((width, globals))
}

/// Every version up to `upto`, every tile: the two tables answer alike.
fn assert_same(a: &MapTable, b: &MapTable, upto: u64) {
    for v in 0..=upto {
        for tile in TILES {
            assert_eq!(
                answer(a, v, tile),
                answer(b, v, tile),
                "version {v} tile {tile}"
            );
        }
    }
}

/// Six advances covering every kind of history event.
fn history() -> Vec<Vec<(u32, Option<LocalMap>)>> {
    vec![
        vec![
            (1, map(Width::W64, &[10, 11, 12])),
            (2, map(Width::W256, &[7, 3, 9, 1])),
        ],
        vec![(1, map(Width::W64, &[10, 11, 12, 13]))], // extension
        vec![(1, map(Width::W64, &[13, 10, 11, 12]))], // replacement (reorder)
        vec![(2, None)],                               // removal
        vec![],                                        // advance with no change
        vec![(2, map(Width::W64, &[7])), (3, map(Width::W256, &[]))], // re-add, empty map
    ]
}

/// One-writer in-memory sink. Fences on `base_version`, publishes head + 1,
/// keeps each cycle's landings and image, can hold commits behind a gate and
/// can fail a chosen cycle. Like the real writer, scanned landings carry no
/// payload.
#[derive(Default)]
struct GateWal {
    head: u64,
    cycles: Vec<(CycleId, Vec<LandedSlot>, BTreeMap<u64, Vec<u8>>)>,
    gate: Option<Arc<Semaphore>>,
    fail_on: Option<CycleId>,
}

impl WalSink for GateWal {
    async fn commit_cycle(
        &mut self,
        batch: DetachedCycleBatch,
    ) -> Result<CommitOutcome, CommitError> {
        if let Some(gate) = &self.gate {
            gate.acquire().await.expect("gate open").forget();
        }
        if Some(batch.frame.cycle) == self.fail_on {
            return Err(CommitError::Io(WriteFailed("injected".into())));
        }
        if batch.frame.base_version.0 != self.head {
            return Err(CommitError::Fenced {
                current_head: DatasetVersion(self.head),
            });
        }
        self.head += 1;
        let landings = batch
            .landings
            .iter()
            .map(|s| {
                let mut slot = s.clone();
                slot.payload = Vec::new();
                LandedSlot {
                    cycle: batch.frame.cycle,
                    slot,
                }
            })
            .collect();
        self.cycles
            .push((batch.frame.cycle, landings, batch.image.clone()));
        Ok(CommitOutcome::Committed {
            version: DatasetVersion(self.head),
            cycle: batch.frame.cycle,
            batch_hash: batch.batch_hash,
        })
    }

    async fn scan_sealed(&self, after: Option<CycleId>) -> Result<Vec<LandedSlot>, WriteFailed> {
        Ok(self
            .cycles
            .iter()
            .filter(|(c, _, _)| after.is_none_or(|a| *c > a))
            .flat_map(|(_, l, _)| l.iter().cloned())
            .collect())
    }

    async fn timeline(&self) -> Result<Vec<FrameMeta>, WriteFailed> {
        Ok(Vec::new())
    }
}

impl SealedImages for GateWal {
    async fn sealed_image(&self, cycle: CycleId) -> Result<BTreeMap<u64, Vec<u8>>, WriteFailed> {
        Ok(self
            .cycles
            .iter()
            .find(|(c, _, _)| *c == cycle)
            .map(|(_, _, i)| i.clone())
            .unwrap_or_default())
    }
}

fn applied(s: Option<Settled>) -> RouteVersion {
    match s {
        Some(Settled::Applied(v)) => v,
        other => panic!("expected Applied, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_restored_table_answers_every_past_version_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("map.lance");
    let sink = LanceCycleWriter::open(path.to_str().unwrap()).await.unwrap();
    let head = sink.head();
    let (writer, handle) = cycle_writer(sink, head, 8);

    let (exit, live) = tokio::join!(writer.run(), async move {
        let mut log = MapLog::create(handle, OWNER, 1, 1, VersionClock::new()).unwrap();
        for (k, changes) in history().into_iter().enumerate() {
            let v = log.advance(1, 2 + k as u64, changes).unwrap();
            assert_eq!(v, RouteVersion(k as u64 + 1));
        }
        assert!(log.settle_next().await.is_none(), "genesis applies nothing");
        for k in 1..=6 {
            assert_eq!(applied(log.settle_next().await), RouteVersion(k));
        }
        log
    });
    let live = live.table();

    // One Lance version per cycle: genesis plus six advances.
    assert_eq!(exit.sink.head(), DatasetVersion(head.0 + 7));

    let restored = restore(&exit.sink, OWNER, VersionClock::new()).await.unwrap();
    assert_eq!(restored.last, RouteVersion(6));
    assert_same(live, &restored.table, 6);
    for tile in TILES {
        assert_eq!(
            live.generation_count(tile),
            restored.table.generation_count(tile)
        );
        assert_eq!(live.event_count(tile), restored.table.event_count(tile));
    }
}

#[tokio::test]
async fn the_owner_returns_before_the_seal_and_applies_only_after_it() {
    let gate = Arc::new(Semaphore::new(0));
    let sink = GateWal {
        gate: Some(gate.clone()),
        ..GateWal::default()
    };
    let (writer, handle) = cycle_writer(sink, DatasetVersion(0), 8);

    let (_exit, ()) = tokio::join!(writer.run(), async move {
        let mut log = MapLog::create(handle, OWNER, 1, 1, VersionClock::new()).unwrap();
        let v1 = log.advance(1, 1, vec![(1, map(Width::W64, &[5]))]).unwrap();
        let v2 = log.advance(1, 1, vec![(1, map(Width::W64, &[5, 6]))]).unwrap();
        assert_eq!((v1, v2), (RouteVersion(1), RouteVersion(2)));

        // Nothing sealed yet: the table has not moved.
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
        assert!(log.settle().is_empty());
        assert_eq!(log.pending(), 3);
        assert_eq!(
            answer(log.table(), 2, 1),
            Err(MapRead::NoSuchTile(1)),
            "unsealed advances must not be applied"
        );

        gate.add_permits(3);
        assert!(log.settle_next().await.is_none());
        assert_eq!(applied(log.settle_next().await), v1);
        assert_eq!(applied(log.settle_next().await), v2);
        assert_eq!(answer(log.table(), 2, 1), Ok((64, vec![5, 6])));
        assert_eq!(answer(log.table(), 1, 1), Ok((64, vec![5])));
    });
}

#[tokio::test]
async fn a_clock_tick_between_claim_and_seal_lands_at_the_claimed_version() {
    let gate = Arc::new(Semaphore::new(0));
    let sink = GateWal {
        gate: Some(gate.clone()),
        ..GateWal::default()
    };
    let (writer, handle) = cycle_writer(sink, DatasetVersion(0), 8);

    let (exit, ()) = tokio::join!(writer.run(), async move {
        let clock = VersionClock::new();
        let mut log = MapLog::create(handle, OWNER, 1, 1, clock.clone()).unwrap();
        let v = log.advance(1, 1, vec![(1, map(Width::W64, &[9]))]).unwrap();
        // Another holder of the clock (a RouteLog) ticks while the cycle seals.
        clock.tick();
        clock.tick();
        gate.add_permits(2);
        assert!(log.settle_next().await.is_none());
        assert_eq!(applied(log.settle_next().await), v);
        assert_eq!(log.table().current_version(), RouteVersion(v.0 + 2));
        assert_eq!(answer(log.table(), v.0, 1), Ok((64, vec![9])));
        assert_eq!(answer(log.table(), v.0 - 1, 1), Err(MapRead::NoSuchTile(1)));
    });

    let restored = restore(&exit.sink, OWNER, VersionClock::new()).await.unwrap();
    assert_eq!(restored.last, RouteVersion(1));
    assert_eq!(answer(&restored.table, 1, 1), Ok((64, vec![9])));
}

#[tokio::test]
async fn a_failed_seal_applies_nothing_and_poisons_what_follows() {
    let gate = Arc::new(Semaphore::new(0));
    let sink = GateWal {
        gate: Some(gate.clone()),
        fail_on: Some(CycleId(2)),
        ..GateWal::default()
    };
    let (writer, handle) = cycle_writer(sink, DatasetVersion(0), 8);

    let (exit, ()) = tokio::join!(writer.run(), async move {
        let mut log = MapLog::create(handle, OWNER, 1, 1, VersionClock::new()).unwrap();
        let v1 = log.advance(1, 1, vec![(1, map(Width::W64, &[1]))]).unwrap();
        let v2 = log.advance(1, 1, vec![(1, map(Width::W64, &[1, 2]))]).unwrap();
        let v3 = log.advance(1, 1, vec![(2, map(Width::W64, &[3]))]).unwrap();
        gate.add_permits(4);

        assert!(log.settle_next().await.is_none());
        assert_eq!(applied(log.settle_next().await), v1);
        match log.settle_next().await {
            Some(Settled::NotDurable {
                version,
                error: SealError::Failed(f),
            }) => {
                assert_eq!(version, v2);
                assert_eq!(f.casts.len(), 2, "marker + one tile cast come back");
            }
            other => panic!("expected the failed seal, got {other:?}"),
        }
        match log.settle_next().await {
            Some(Settled::NotDurable {
                version,
                error: SealError::Poisoned { casts, .. },
            }) => {
                assert_eq!(version, v3);
                assert_eq!(casts.len(), 2);
            }
            other => panic!("expected the poisoned seal, got {other:?}"),
        }

        // Neither v2 nor v3 was applied.
        assert_eq!(answer(log.table(), v2.0, 1), Ok((64, vec![1])));
        assert_eq!(answer(log.table(), v3.0, 2), Err(MapRead::NoSuchTile(2)));
        // And nothing more is submitted.
        let before = log.table().current_version();
        match log.advance(1, 1, vec![]) {
            Err(AdvanceError::Poisoned { failed }) => assert_eq!(failed, v2),
            other => panic!("expected Poisoned, got {other:?}"),
        }
        assert_eq!(log.table().current_version(), before, "nothing was claimed");
    });

    // The store holds genesis and v1 only, and replays to the same table.
    let restored = restore(&exit.sink, OWNER, VersionClock::new()).await.unwrap();
    assert_eq!(restored.last, RouteVersion(1));
    assert_eq!(answer(&restored.table, 1, 1), Ok((64, vec![1])));
    assert_eq!(answer(&restored.table, 1, 2), Err(MapRead::NoSuchTile(2)));
}

#[tokio::test]
async fn every_landing_is_on_behalf_of_the_owner() {
    let (writer, handle) = cycle_writer(GateWal::default(), DatasetVersion(0), 8);
    let (exit, ()) = tokio::join!(writer.run(), async move {
        let mut log = MapLog::create(handle, OWNER, 1, 1, VersionClock::new()).unwrap();
        log.advance(1, 1, vec![(1, map(Width::W256, &[1, 2]))])
            .unwrap();
        while log.settle_next().await.is_some() {}
    });
    let landings = exit.sink.scan_sealed(None).await.unwrap();
    // genesis marker + advance marker + four W256 parts
    assert_eq!(landings.len(), 6);
    assert!(landings.iter().all(|l| l.slot.owner == OWNER));
    assert!(landings.iter().all(|l| l.slot.paired_move.is_none()));

    match restore(&exit.sink, OWNER + 1, VersionClock::new()).await {
        Err(RestoreError::Replay(ReplayError::ForeignOwner { owner, .. })) => {
            assert_eq!(owner, OWNER)
        }
        Ok(_) => panic!("a restore on behalf of another mailbox must be refused"),
        Err(e) => panic!("expected ForeignOwner, got {e:?}"),
    }
}

#[tokio::test]
async fn a_reserved_id_or_a_repeated_tile_is_refused_before_a_version_is_claimed() {
    let (writer, handle) = cycle_writer(GateWal::default(), DatasetVersion(0), 8);
    let (_exit, ()) = tokio::join!(writer.run(), async move {
        let mut log = MapLog::create(handle, OWNER, 1, 1, VersionClock::new()).unwrap();
        let before = log.table().current_version();
        let reserved = LocalMap::prepare(Width::W64, &[1, RESERVED_ID]).unwrap();
        assert!(matches!(
            log.advance(1, 1, vec![(4, Some(reserved))]),
            Err(AdvanceError::ReservedId(4))
        ));
        assert!(matches!(
            log.advance(1, 1, vec![(4, None), (4, map(Width::W64, &[1]))]),
            Err(AdvanceError::DuplicateTile(4))
        ));
        assert_eq!(log.table().current_version(), before);
        // The silence twin: a valid advance is claimed and applied.
        let v = log.advance(1, 1, vec![(4, map(Width::W64, &[1]))]).unwrap();
        assert_eq!(v.0, before.0 + 1);
        while log.settle_next().await.is_some() {}
        assert_eq!(answer(log.table(), v.0, 4), Ok((64, vec![1])));
    });
}

#[tokio::test]
async fn a_restored_log_continues_and_restores_again() {
    let (writer, handle) = cycle_writer(GateWal::default(), DatasetVersion(0), 8);
    let (exit, ()) = tokio::join!(writer.run(), async move {
        let mut log = MapLog::create(handle, OWNER, 1, 1, VersionClock::new()).unwrap();
        log.advance(1, 1, vec![(1, map(Width::W64, &[1]))]).unwrap();
        while log.settle_next().await.is_some() {}
    });
    let sink = exit.sink;
    let restored = restore(&sink, OWNER, VersionClock::new()).await.unwrap();
    let head = DatasetVersion(sink.head);
    let (writer, handle) = cycle_writer(sink, head, 8);
    let (exit, first) = tokio::join!(writer.run(), async move {
        let mut log = MapLog::resume(restored, handle, OWNER);
        let v = log.advance(1, 1, vec![(1, None)]).unwrap();
        assert_eq!(v, RouteVersion(2));
        assert_eq!(applied(log.settle_next().await), v);
        log
    });
    let again = restore(&exit.sink, OWNER, VersionClock::new()).await.unwrap();
    assert_eq!(again.last, RouteVersion(2));
    assert_same(first.table(), &again.table, 2);
}

#[tokio::test]
async fn a_frame_from_before_a_restore_is_foreign() {
    let (writer, handle) = cycle_writer(GateWal::default(), DatasetVersion(0), 8);
    let (exit, log) = tokio::join!(writer.run(), async move {
        let mut log = MapLog::create(handle, OWNER, 1, 1, VersionClock::new()).unwrap();
        log.advance(1, 1, vec![(1, map(Width::W64, &[1]))]).unwrap();
        while log.settle_next().await.is_some() {}
        log
    });
    let frame = log.table().seal();
    let restored = restore(&exit.sink, OWNER, VersionClock::new()).await.unwrap();
    assert!(matches!(
        restored.table.map(&frame, 1),
        Err(MapRead::Stale(FrameRefusal::ForeignFrame))
    ));
    // The same version still answers through diagnose_at.
    assert_eq!(answer(&restored.table, 1, 1), Ok((64, vec![1])));
}
