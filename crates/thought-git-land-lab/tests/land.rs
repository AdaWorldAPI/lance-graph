//! Land a cycle of thoughts through the writer contract and read it back.

use std::collections::BTreeMap;
use std::future::Future;

use lance_graph_contract::scheduler::DatasetVersion;
use lance_graph_planner::persist_sink::{
    CommitError, CommitOutcome, CycleFrame, CycleId, DetachedCycleBatch, FrameMeta, LandedSlot,
    PersistError, SweepSlot, WalSink, WriteFailed,
};
use thought_git_land_lab::land::{
    LandError, MAX_LANES, ROW_BYTES, ReplayImage, ReplayRefusal, RowError, decode_row, encode_row,
    land_cycle, replay_landed,
};
use thought_git_probe::fixtures::{Cell, Outcome, Thought, cell};
use thought_git_probe::oracle::{Givens, Version};

const ROWS: u32 = 20;
const LANES: u8 = 6;
const OWNER: u32 = 3;

struct Sealed {
    frame_base: DatasetVersion,
    cycle: CycleId,
    batch_hash: u64,
    landings: Vec<SweepSlot>,
    image: BTreeMap<u64, Vec<u8>>,
}

/// In-memory sink with the production commit semantics, plus counters.
struct MemWal {
    head: DatasetVersion,
    sealed: Vec<Sealed>,
    /// Every call to `commit_cycle`.
    commit_calls: u64,
    /// Durable appends only.
    appends: u64,
    /// Reads of a durable image through [`ReplayImage`]: the recovery path.
    /// A live read must leave this at zero.
    image_reads: std::cell::Cell<u64>,
}

impl MemWal {
    fn new(head: u64) -> Self {
        Self {
            head: DatasetVersion(head),
            sealed: Vec::new(),
            commit_calls: 0,
            image_reads: std::cell::Cell::new(0),
            appends: 0,
        }
    }
}

impl WalSink for MemWal {
    async fn commit_cycle(
        &mut self,
        batch: DetachedCycleBatch,
    ) -> Result<CommitOutcome, CommitError> {
        self.commit_calls += 1;
        if let Some(rec) = self.sealed.iter().find(|s| s.cycle == batch.frame.cycle) {
            return if rec.batch_hash == batch.batch_hash {
                Ok(CommitOutcome::Reconciled {
                    current_head: self.head,
                    cycle: batch.frame.cycle,
                    batch_hash: batch.batch_hash,
                })
            } else {
                Err(CommitError::HashConflict {
                    cycle: batch.frame.cycle,
                    stored_hash: rec.batch_hash,
                    offered_hash: batch.batch_hash,
                })
            };
        }
        if batch.frame.base_version != self.head {
            return Err(CommitError::Fenced {
                current_head: self.head,
            });
        }
        self.appends += 1;
        self.head = DatasetVersion(self.head.0 + 1);
        let (cycle, batch_hash) = (batch.frame.cycle, batch.batch_hash);
        self.sealed.push(Sealed {
            frame_base: batch.frame.base_version,
            cycle,
            batch_hash,
            landings: batch.landings,
            image: batch.image,
        });
        Ok(CommitOutcome::Committed {
            version: self.head,
            cycle,
            batch_hash,
        })
    }

    async fn scan_sealed(
        &self,
        after_cycle: Option<CycleId>,
    ) -> Result<Vec<LandedSlot>, WriteFailed> {
        Ok(self
            .sealed
            .iter()
            .filter(|s| after_cycle.is_none_or(|c| s.cycle > c))
            .flat_map(|s| {
                s.landings.iter().map(|slot| LandedSlot {
                    cycle: s.cycle,
                    slot: SweepSlot {
                        payload: Vec::new(),
                        ..slot.clone()
                    },
                })
            })
            .collect())
    }

    async fn timeline(&self) -> Result<Vec<FrameMeta>, WriteFailed> {
        Ok(self
            .sealed
            .iter()
            .map(|s| FrameMeta {
                cycle: s.cycle,
                base_version: s.frame_base,
                batch_hash: s.batch_hash,
            })
            .collect())
    }
}

impl MemWal {
    /// Test inspection of what was written. Not counted: it is the test
    /// looking at the store, not the code under test reading it.
    fn peek(&self, cycle: CycleId) -> Option<&BTreeMap<u64, Vec<u8>>> {
        self.sealed
            .iter()
            .find(|s| s.cycle == cycle)
            .map(|s| &s.image)
    }
}

impl ReplayImage for MemWal {
    fn image(&self, cycle: CycleId) -> Option<&BTreeMap<u64, Vec<u8>>> {
        self.image_reads.set(self.image_reads.get() + 1);
        self.sealed
            .iter()
            .find(|s| s.cycle == cycle)
            .map(|s| &s.image)
    }
}

fn run<F: Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(f)
}

fn givens() -> Givens {
    Givens((0..3).map(|r| (cell(r, 0), 100 + u64::from(r))).collect())
}

fn th(id: u32, writes: Vec<(Cell, u64)>, outcome: Outcome) -> Thought {
    Thought::in_space(id, ROWS, LANES, writes, outcome)
}

fn commit(id: u32, writes: Vec<(Cell, u64)>) -> Thought {
    th(id, writes, Outcome::Commit)
}

/// Land `batch` as `cycle` reading `base`, starting at stream position `pos`.
fn land(
    wal: &mut MemWal,
    cycle: u64,
    base: DatasetVersion,
    pos: u64,
    prior: &Version,
    batch: &[Thought],
) -> Result<(CommitOutcome, Version), LandError> {
    let frame = CycleFrame::new(CycleId(cycle), base);
    run(land_cycle(wal, frame, OWNER, pos, &givens(), prior, batch))
}

/// FAILS IF: a cycle with nothing to land touches the sink, advances the head,
/// or reports anything but `NoChange` at the frame's base version.
#[test]
fn an_empty_cycle_writes_nothing() {
    let cases: Vec<Vec<Thought>> = vec![
        vec![],
        vec![
            th(1, vec![(cell(5, 1), 9)], Outcome::Plan),
            th(2, vec![(cell(6, 1), 9)], Outcome::Prune),
            th(3, vec![(cell(7, 1), 9)], Outcome::Veto),
        ],
        vec![commit(4, vec![(cell(0, 0), 100)])],
    ];
    for (i, batch) in cases.iter().enumerate() {
        let mut wal = MemWal::new(5);
        let prior = Version::default();
        let base = wal.head;
        let (outcome, next) = land(&mut wal, 1, base, 0, &prior, batch).unwrap();
        assert_eq!(
            outcome,
            CommitOutcome::NoChange {
                head: DatasetVersion(5)
            },
            "case {i}"
        );
        assert_eq!(wal.commit_calls, 0, "case {i}");
        assert_eq!(wal.appends, 0, "case {i}");
        assert_eq!(wal.head, DatasetVersion(5), "case {i}");
        assert_eq!(next.cells, prior.cells, "case {i}");
    }
}

/// FAILS IF: two thoughts of one cycle land as more or fewer than one version, a
/// live read touches the store, the replayed image disagrees with the resident
/// version, or a read differs from the resident read anywhere in the space.
#[test]
fn two_thoughts_in_one_cycle_land_as_one_version() {
    let mut wal = MemWal::new(5);
    let a = commit(1, vec![(cell(4, 0), 7), (cell(4, 1), 8), (cell(5, 2), 9)]);
    let b = commit(2, vec![(cell(4, 3), 11), (cell(0, 0), 100)]);
    let prior = Version::default();
    let base = wal.head;
    let (outcome, next) = land(&mut wal, 1, base, 0, &prior, &[a, b]).unwrap();

    assert!(
        matches!(
            outcome,
            CommitOutcome::Committed { version, cycle, .. }
                if version == DatasetVersion(6) && cycle == CycleId(1)
        ),
        "{outcome:?}"
    );
    assert_eq!(wal.appends, 1);
    assert_eq!(wal.commit_calls, 1);
    assert_eq!(wal.head, DatasetVersion(6));
    assert_eq!(wal.peek(CycleId(1)).unwrap().len(), 2, "rows 4 and 5");

    // Live path: the resident version answers; the store is not read.
    let g = givens();
    assert_eq!(next.read(&g, cell(4, 0)), Some(7));
    assert_eq!(next.read(&g, cell(4, 3)), Some(11));
    assert_eq!(next.read(&g, cell(5, 2)), Some(9));
    assert_eq!(next.read(&g, cell(5, 0)), None);
    assert_eq!(wal.image_reads.get(), 0, "a live read touched the store");

    // Recovery: one replay rebuilds exactly the resident cells.
    let landed = replay_landed(&wal, &[CycleId(1)]).unwrap();
    assert_eq!(wal.image_reads.get(), 1);
    assert_eq!(landed.cells, next.cells);

    let (mut from_given, mut from_landed, mut absent) = (0, 0, 0);
    for row in 0..ROWS {
        for lane in 0..LANES {
            let c = cell(row, lane);
            let got = landed.read(&g, c);
            assert_eq!(got, next.read(&g, c), "{c:?}");
            match got {
                Some(_) if g.0.contains_key(&c) => from_given += 1,
                Some(_) => from_landed += 1,
                None => absent += 1,
            }
        }
    }
    assert!(from_given >= 1 && from_landed >= 1 && absent >= 1);
}

/// FAILS IF: re-submitting the same cycle and batch appends a second time or
/// does not reconcile.
#[test]
fn a_resubmitted_cycle_reconciles() {
    let mut wal = MemWal::new(5);
    let batch = vec![commit(1, vec![(cell(4, 0), 7)])];
    let prior = Version::default();
    let base = wal.head;
    let (first, _) = land(&mut wal, 1, base, 0, &prior, &batch).unwrap();
    assert!(matches!(first, CommitOutcome::Committed { .. }));
    let (second, _) = land(&mut wal, 1, base, 0, &prior, &batch).unwrap();
    assert!(
        matches!(second, CommitOutcome::Reconciled { .. }),
        "{second:?}"
    );
    assert_eq!(wal.appends, 1);
    assert_eq!(wal.commit_calls, 2);
}

/// FAILS IF: two different batches for one cycle id are both accepted, or the
/// conflict is not the sink's `HashConflict`.
#[test]
fn one_batch_per_thought_conflicts_in_the_same_cycle() {
    let mut wal = MemWal::new(5);
    let prior = Version::default();
    let base = wal.head;
    let a = vec![commit(1, vec![(cell(4, 0), 7)])];
    let b = vec![commit(2, vec![(cell(6, 1), 8)])];
    land(&mut wal, 7, base, 0, &prior, &a).unwrap();
    let err = land(&mut wal, 7, base, 0, &prior, &b).unwrap_err();
    assert!(
        matches!(
            err,
            LandError::Persist(PersistError::Commit(CommitError::HashConflict { .. }))
        ),
        "{err:?}"
    );
    assert_eq!(wal.appends, 1);
}

/// FAILS IF: an observed zero reads as absent, a never-written lane reads as
/// zero, or a newer row fails to keep the older row's lanes.
#[test]
fn observed_zero_and_absent_survive_a_newer_row() {
    let mut wal = MemWal::new(5);
    let g = givens();
    let base1 = wal.head;
    let (_, v1) = land(
        &mut wal,
        1,
        base1,
        0,
        &Version::default(),
        &[commit(1, vec![(cell(9, 1), 0)])],
    )
    .unwrap();
    let base2 = wal.head;
    let (_, v2) = land(
        &mut wal,
        2,
        base2,
        1,
        &v1,
        &[commit(2, vec![(cell(9, 3), 5)])],
    )
    .unwrap();
    assert_eq!(v2.cells.len(), 2);

    // Live path, from the resident version.
    assert_eq!(v2.read(&g, cell(9, 1)), Some(0));
    assert_eq!(v2.read(&g, cell(9, 2)), None);
    assert_eq!(v2.read(&g, cell(9, 3)), Some(5));
    assert_eq!(wal.image_reads.get(), 0, "a live read touched the store");

    // Recovery: the same three states come back from the durable rows.
    let landed = replay_landed(&wal, &[CycleId(1), CycleId(2)]).unwrap();
    assert_eq!(landed.read(&g, cell(9, 1)), Some(0));
    assert_eq!(landed.read(&g, cell(9, 2)), None);
    assert_eq!(landed.read(&g, cell(9, 3)), Some(5));
    assert_eq!(landed.cells, v2.cells);

    let first = &wal.peek(CycleId(1)).unwrap()[&9];
    let second = &wal.peek(CycleId(2)).unwrap()[&9];
    assert_eq!(first.len(), ROW_BYTES);
    assert_eq!(second.len(), ROW_BYTES);
    assert_ne!(first, second);
}

/// FAILS IF: a cycle whose base is not the sink head is appended, or the
/// refusal is not `Fenced`.
#[test]
fn a_stale_base_is_fenced() {
    let mut wal = MemWal::new(5);
    let batch = vec![commit(1, vec![(cell(4, 0), 7)])];
    let err = land(
        &mut wal,
        1,
        DatasetVersion(3),
        0,
        &Version::default(),
        &batch,
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            LandError::Persist(PersistError::Commit(CommitError::Fenced { current_head }))
                if current_head == DatasetVersion(5)
        ),
        "{err:?}"
    );
    assert_eq!(wal.appends, 0);
}

/// FAILS IF: a cell offered two different values lands anyway, is not recorded
/// as a divergence, or blocks another cell of the same batch.
#[test]
fn a_divergent_cell_does_not_land() {
    let mut wal = MemWal::new(5);
    let g = givens();
    let a = commit(1, vec![(cell(10, 0), 1), (cell(11, 0), 3)]);
    let b = commit(2, vec![(cell(10, 0), 2)]);
    let base = wal.head;
    let (outcome, next) = land(&mut wal, 1, base, 0, &Version::default(), &[a, b]).unwrap();
    assert!(matches!(outcome, CommitOutcome::Committed { .. }));
    assert!(!next.divergences.is_empty());

    assert_eq!(next.read(&g, cell(10, 0)), None);
    assert_eq!(next.read(&g, cell(11, 0)), Some(3));
    assert_eq!(wal.image_reads.get(), 0, "a live read touched the store");

    let landed = replay_landed(&wal, &[CycleId(1)]).unwrap();
    assert_eq!(landed.cells, next.cells);
    assert_eq!(wal.peek(CycleId(1)).unwrap().len(), 1, "only row 11");
}

/// FAILS IF: a malformed row decodes, a refusal names the wrong error, lane 61
/// encodes, or a round trip (zero value included) is not exact.
#[test]
fn row_codec_refuses_malformed_rows() {
    let empty = encode_row(&BTreeMap::new()).unwrap();
    assert_eq!(empty.len(), ROW_BYTES);

    assert_eq!(decode_row(&empty[..511]), Err(RowError::Length(511)));
    let mut long = empty.clone();
    long.push(0);
    assert_eq!(decode_row(&long), Err(RowError::Length(513)));

    let mut stamped = empty.clone();
    stamped[3] = 1;
    assert_eq!(decode_row(&stamped), Err(RowError::StampNotZero));

    let mut high_bit = empty.clone();
    high_bit[16 + 7] |= 1 << (61 - 56);
    assert_eq!(decode_row(&high_bit), Err(RowError::LaneOutOfRange(61)));

    let mut orphan = empty.clone();
    orphan[24 + 8 * 2] = 1;
    assert_eq!(decode_row(&orphan), Err(RowError::ValueWithoutPresence(2)));

    let too_far: BTreeMap<u8, u64> = [(MAX_LANES, 1)].into_iter().collect();
    assert_eq!(encode_row(&too_far), Err(RowError::LaneOutOfRange(61)));

    let lanes: BTreeMap<u8, u64> = [(0, 42), (5, 0), (60, u64::MAX)].into_iter().collect();
    let back = decode_row(&encode_row(&lanes).unwrap()).unwrap();
    assert_eq!(back, lanes);
    assert_eq!(back.get(&5), Some(&0));
    assert_eq!(back.get(&6), None);
}

/// FAILS IF: reading a cycle that was never sealed succeeds or reports
/// anything but `MissingCycle`.
#[test]
fn a_missing_cycle_is_refused_on_read() {
    let wal = MemWal::new(5);
    let got = replay_landed(&wal, &[CycleId(99)]);
    assert!(
        matches!(got, Err(ReplayRefusal::MissingCycle(CycleId(99)))),
        "{got:?}"
    );
}
