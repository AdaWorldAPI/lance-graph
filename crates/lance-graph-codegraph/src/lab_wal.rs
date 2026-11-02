//! Lab only (D-TGL-PUB step 2): an in-memory [`WalSink`] for the G10 × G11
//! publication lab. It honours the production writer contract
//! (`lance-graph-planner/src/persist_sink.rs`) and adds two lab seams: injected
//! faults, and a read of the durable image (`scan_sealed` returns landings
//! without payload, so recovery must read land records from the image).
//!
//! Not a store: it holds what the test wrote, in memory, for one test.

use std::collections::BTreeMap;

use lance_graph_contract::scheduler::DatasetVersion;
use lance_graph_planner::persist_sink::{
    CommitError, CommitOutcome, CycleId, DetachedCycleBatch, FrameMeta, LandedSlot, SweepSlot,
    WalSink, WriteFailed,
};

/// A fault the next `commit_cycle` injects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fault {
    /// I/O failed before anything was published.
    IoBeforePublish,
    /// The batch was published, then the acknowledgement was lost.
    AmbiguousAfterPublish,
}

struct Sealed {
    frame_base: DatasetVersion,
    cycle: CycleId,
    batch_hash: u64,
    landings: Vec<SweepSlot>,
    image: BTreeMap<u64, Vec<u8>>,
}

/// The lab writer. One logical writer (`&mut self` on commit), like the real one.
pub(crate) struct LabWal {
    head: DatasetVersion,
    sealed: Vec<Sealed>,
    fault: Option<Fault>,
    /// Durable appends performed (a reconcile performs none).
    pub(crate) appends: u64,
}

impl LabWal {
    pub(crate) fn new(head: DatasetVersion) -> Self {
        Self {
            head,
            sealed: Vec::new(),
            fault: None,
            appends: 0,
        }
    }

    pub(crate) fn head(&self) -> DatasetVersion {
        self.head
    }

    /// Inject a fault into the next `commit_cycle` only.
    pub(crate) fn inject(&mut self, fault: Fault) {
        self.fault = Some(fault);
    }

    /// Lab seam: reverse the stored (physical) order of the sealed cycles,
    /// so a recovery that trusts physical order replays them wrongly.
    pub(crate) fn reverse_physical_order(&mut self) {
        self.sealed.reverse();
    }

    /// The durable image of `cycle` (row → payload): where recovery reads
    /// land records, because `scan_sealed` carries no payload.
    pub(crate) fn image(&self, cycle: CycleId) -> Option<&BTreeMap<u64, Vec<u8>>> {
        self.sealed
            .iter()
            .find(|s| s.cycle == cycle)
            .map(|s| &s.image)
    }
}

impl WalSink for LabWal {
    async fn commit_cycle(
        &mut self,
        batch: DetachedCycleBatch,
    ) -> Result<CommitOutcome, CommitError> {
        let fault = self.fault.take();
        if fault == Some(Fault::IoBeforePublish) {
            return Err(CommitError::Io(WriteFailed("lab: injected I/O".into())));
        }
        // Reconciliation first: an already durable (cycle, hash) is success, a
        // durable cycle with another hash fails closed.
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
        // The fence: the batch must target the current head.
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
        if fault == Some(Fault::AmbiguousAfterPublish) {
            return Err(CommitError::Ambiguous {
                cycle,
                batch_hash,
                cause: "lab: acknowledgement lost after publish".into(),
            });
        }
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
        // Stored order, payload-free: the production projection, not better.
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

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_planner::persist_sink::CycleFrame;

    fn run<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime")
            .block_on(f)
    }

    fn batch(cycle: u64, base: u64, payload: &[u8]) -> DetachedCycleBatch {
        DetachedCycleBatch::freeze(
            CycleFrame::new(CycleId(cycle), DatasetVersion(base)),
            vec![SweepSlot {
                cycle: CycleId(cycle),
                stream_position: cycle,
                owner: 1,
                row: 7,
                paired_move: None,
                payload: payload.to_vec(),
            }],
        )
    }

    /// Everything the writer exposes, for "nothing was written" checks.
    fn state(w: &LabWal) -> (DatasetVersion, u64, usize, usize) {
        let t = run(w.timeline()).expect("timeline").len();
        let s = run(w.scan_sealed(None)).expect("scan").len();
        (w.head(), w.appends, t, s)
    }

    /// A batch against a stale base is `Fenced` and writes nothing. Red if
    /// the fence is skipped (disable: drop the `base_version != head` check).
    #[test]
    fn a_stale_base_is_fenced_and_writes_nothing() {
        let mut w = LabWal::new(DatasetVersion(5));
        let before = state(&w);
        let r = run(w.commit_cycle(batch(1, 4, b"land")));
        assert_eq!(
            r,
            Err(CommitError::Fenced {
                current_head: DatasetVersion(5)
            })
        );
        assert_eq!(state(&w), before);
    }

    /// A commit against the head publishes exactly one version. `scan_sealed`
    /// carries no payload; the image does. Red if scan leaks the payload
    /// (disable: return the stored slot unchanged), which would let a
    /// recovery read from the wrong place and pass in the lab only.
    #[test]
    fn a_commit_publishes_one_version_and_the_payload_lives_in_the_image() {
        let mut w = LabWal::new(DatasetVersion(5));
        let b = batch(1, 5, b"land");
        let hash = b.batch_hash;
        let r = run(w.commit_cycle(b));
        assert_eq!(
            r,
            Ok(CommitOutcome::Committed {
                version: DatasetVersion(6),
                cycle: CycleId(1),
                batch_hash: hash
            })
        );
        let scanned = run(w.scan_sealed(None)).expect("scan");
        assert_eq!(scanned.len(), 1);
        assert!(
            scanned[0].slot.payload.is_empty(),
            "scan carries no payload"
        );
        assert_eq!(
            w.image(CycleId(1))
                .and_then(|i| i.get(&7))
                .map(Vec::as_slice),
            Some(&b"land"[..])
        );
    }

    /// Re-submitting the same frozen batch reconciles with no second append;
    /// another batch for a durable cycle fails closed. Red if the
    /// reconciliation lookup is skipped (disable): the re-submit is then
    /// `Fenced` (head moved) and a retry after a lost ack cannot confirm.
    #[test]
    fn a_resubmit_reconciles_and_a_different_batch_conflicts() {
        let mut w = LabWal::new(DatasetVersion(5));
        let b = batch(1, 5, b"land");
        run(w.commit_cycle(b.clone())).expect("commit");
        let after = state(&w);
        assert!(matches!(
            run(w.commit_cycle(b)),
            Ok(CommitOutcome::Reconciled { .. })
        ));
        assert_eq!(state(&w), after, "no second append");
        assert!(matches!(
            run(w.commit_cycle(batch(1, 5, b"other"))),
            Err(CommitError::HashConflict { .. })
        ));
        assert_eq!(state(&w), after);
    }

    /// B3: the acknowledgement is lost after publication. The batch IS
    /// durable; re-submitting it reconciles. An injected I/O fault before
    /// publication writes nothing. Twin pair: the two faults must leave
    /// different durable states, or the injection is not doing anything.
    #[test]
    fn faults_before_and_after_publish_leave_different_durable_states() {
        let mut w = LabWal::new(DatasetVersion(5));
        let before = state(&w);
        w.inject(Fault::IoBeforePublish);
        assert!(matches!(
            run(w.commit_cycle(batch(1, 5, b"land"))),
            Err(CommitError::Io(_))
        ));
        assert_eq!(state(&w), before, "I/O before publish writes nothing");

        let b = batch(1, 5, b"land");
        w.inject(Fault::AmbiguousAfterPublish);
        assert!(matches!(
            run(w.commit_cycle(b.clone())),
            Err(CommitError::Ambiguous { .. })
        ));
        assert_ne!(state(&w), before, "published despite the lost ack");
        assert!(matches!(
            run(w.commit_cycle(b)),
            Ok(CommitOutcome::Reconciled { .. })
        ));
        assert_eq!(w.appends, 1);
    }
}
