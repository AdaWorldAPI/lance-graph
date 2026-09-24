//! The background cycle writer — the seal runs off the thought loop.
//!
//! [`seal_cycle`] is the whole durable path for one cycle: order the casts,
//! fold them into the coalesced image (a copy of every payload), hash the
//! canonical content, keep a retry copy of the casts, and commit one Lance
//! version. Measured at 64k casts × 512 B that is ~60–110 ms before the Lance
//! write even starts (board entry
//! `2026-09-23-freeze-at-64k-is-hash-and-copy-not-order.md`), and a loop that
//! calls it inline holds every thought until it returns.
//!
//! This module moves all of it onto ONE background task that owns the sink:
//!
//! ```text
//! thought loop:   collect_casts ──try_submit(cycle, casts)──▶ (returns at once)
//!                                        │ moves the Vec, no copy
//! writer task:                           ▼
//!                 frame = (cycle, head) → seal_cycle(sink, frame, casts)
//!                 head  ← the version this cycle published
//!                 ticket ← SealedCycle      durable mark ← (cycle, head)
//! ```
//!
//! **What stays exactly the same.** The writer calls [`seal_cycle`] itself, so
//! the frozen batch, its hash, the reconciliation-first commit and the
//! [`SealFailure`] taxonomy are unchanged — a cycle sealed here is
//! byte-identical to one sealed inline (pinned by a test). There is still
//! exactly ONE writer: the task owns the sink, and `WalSink::commit_cycle`'s
//! `&mut self` is the same boundary it always was.
//!
//! **The frame's `base_version` is assigned by the writer, not the caller.**
//! Cycle N+1 is submitted before cycle N has landed, so the caller cannot know
//! N's publication version yet. The writer does: it frames each cycle against
//! the head the previous one published (unchanged on `NoChange`, the
//! reconciliation-time head on `Reconciled`). Seals therefore run strictly in
//! submission order, and submissions must carry increasing cycle ids.
//!
//! **When a step may be applied.** A cycle's transitions arrive on its
//! [`SealTicket`]; apply them ([`apply_sealed_transitions`]) only after the
//! ticket resolves `Ok`. That keeps the existing rule — no sealed version, no
//! applied step — while the loop computes the next cycle in the meantime.
//! Whether steps may run ahead of durability is a separate decision this
//! module does not make.
//!
//! **Failure poisons the pipeline, and nothing is lost.** Every queued cycle
//! was going to be framed on top of the one that failed, so after a failed
//! seal the writer seals nothing further: the failing ticket gets the
//! [`SealFailure`] (its [`SealFailure::recovery`] says what to do), every cycle
//! still queued gets [`SealError::Poisoned`] with its casts handed back, new
//! submissions are refused with their casts handed back, and the durable mark
//! stops at the last cycle that landed. Recovery is the existing one: resolve
//! the failure, then start a fresh writer on the durable head. Durable
//! progress is therefore always a gap-free prefix of the submitted cycles.
//!
//! **Driving it.** [`CycleWriter::run`] is a future the caller drives. It is
//! `Send` exactly when the sink's commit future is, so spawn it with
//! `tokio::spawn` for a `Send` sink or on a `LocalSet` otherwise. This is the
//! outbound sink boundary where the supervisor's tokio use belongs (I-2); the
//! handle's `try_submit` is plain sync code and never blocks the loop.
//!
//! [`apply_sealed_transitions`]: crate::cycle_driver::apply_sealed_transitions

use lance_graph_contract::scheduler::DatasetVersion;
use lance_graph_planner::persist_sink::{CommitOutcome, CycleFrame, CycleId, SweepSlot, WalSink};
use tokio::sync::{mpsc, oneshot, watch};

use crate::cycle_driver::{seal_cycle, SealFailure, SealedCycle};

/// How far durable progress has reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DurableMark {
    /// The last cycle whose seal landed (`None` until the first one does).
    /// Every submitted cycle up to it has landed — the prefix has no gaps.
    pub through: Option<CycleId>,
    /// The store head after that cycle — the base the next cycle is framed on.
    pub head: DatasetVersion,
    /// Set once the writer has stopped sealing.
    pub stopped: Option<WriterStop>,
}

/// Why the writer stopped sealing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriterStop {
    /// The seal of `failed` did not land; nothing after it was sealed.
    Poisoned { failed: CycleId },
    /// Every handle was dropped and the queue drained normally.
    Closed,
}

/// Why a submitted cycle did not seal.
#[derive(Debug)]
pub enum SealError {
    /// This cycle's own seal failed at the WAL. [`SealFailure::recovery`]
    /// classifies it; the failure carries the frame and the casts.
    Failed(Box<SealFailure>),
    /// An earlier cycle's seal failed, so this one was never framed or frozen.
    /// Its casts come back untouched.
    Poisoned {
        failed: CycleId,
        cycle: CycleId,
        casts: Vec<SweepSlot>,
    },
    /// The cycle id did not increase past the last sealed cycle. Refused
    /// without sealing; the writer keeps going.
    OutOfOrder {
        last: CycleId,
        cycle: CycleId,
        casts: Vec<SweepSlot>,
    },
    /// The writer was dropped before it answered (its future was never driven
    /// to this job, or the task was aborted).
    WriterGone,
}

/// Why a cycle could not be handed to the writer. The casts always come back.
#[derive(Debug)]
pub enum SubmitError {
    /// The queue is at its depth — backpressure. Retry, or `submit().await`.
    Full {
        cycle: CycleId,
        casts: Vec<SweepSlot>,
    },
    /// The writer has stopped (poisoned or dropped) and accepts nothing more.
    Closed {
        cycle: CycleId,
        casts: Vec<SweepSlot>,
    },
}

struct Job {
    cycle: CycleId,
    casts: Vec<SweepSlot>,
    reply: oneshot::Sender<Result<SealedCycle, SealError>>,
}

/// The pending seal of one submitted cycle.
#[derive(Debug)]
pub struct SealTicket {
    pub cycle: CycleId,
    rx: oneshot::Receiver<Result<SealedCycle, SealError>>,
}

impl SealTicket {
    /// Wait for this cycle's seal.
    pub async fn sealed(self) -> Result<SealedCycle, SealError> {
        self.rx.await.unwrap_or(Err(SealError::WriterGone))
    }

    /// The seal if it has already resolved, without waiting. `None` while the
    /// writer is still on it. After this returns `Some`, the ticket is spent.
    pub fn try_sealed(&mut self) -> Option<Result<SealedCycle, SealError>> {
        match self.rx.try_recv() {
            Ok(r) => Some(r),
            Err(oneshot::error::TryRecvError::Empty) => None,
            Err(oneshot::error::TryRecvError::Closed) => Some(Err(SealError::WriterGone)),
        }
    }
}

/// The thought-loop side: submit cycles, read durable progress.
#[derive(Clone)]
pub struct CycleWriterHandle {
    tx: mpsc::Sender<Job>,
    durable: watch::Receiver<DurableMark>,
}

impl CycleWriterHandle {
    /// Hand a cycle to the writer without blocking. The cast vector is moved,
    /// never copied; every byte of work on it happens on the writer.
    pub fn try_submit(
        &self,
        cycle: CycleId,
        casts: Vec<SweepSlot>,
    ) -> Result<SealTicket, SubmitError> {
        let (reply, rx) = oneshot::channel();
        match self.tx.try_send(Job {
            cycle,
            casts,
            reply,
        }) {
            Ok(()) => Ok(SealTicket { cycle, rx }),
            Err(mpsc::error::TrySendError::Full(job)) => Err(SubmitError::Full {
                cycle,
                casts: job.casts,
            }),
            Err(mpsc::error::TrySendError::Closed(job)) => Err(SubmitError::Closed {
                cycle,
                casts: job.casts,
            }),
        }
    }

    /// Like [`try_submit`](Self::try_submit), but waits for queue space
    /// instead of returning [`SubmitError::Full`].
    pub async fn submit(
        &self,
        cycle: CycleId,
        casts: Vec<SweepSlot>,
    ) -> Result<SealTicket, SubmitError> {
        let (reply, rx) = oneshot::channel();
        match self
            .tx
            .send(Job {
                cycle,
                casts,
                reply,
            })
            .await
        {
            Ok(()) => Ok(SealTicket { cycle, rx }),
            Err(mpsc::error::SendError(job)) => Err(SubmitError::Closed {
                cycle,
                casts: job.casts,
            }),
        }
    }

    /// A snapshot of durable progress. Cheap; never waits.
    #[must_use]
    pub fn durable(&self) -> DurableMark {
        *self.durable.borrow()
    }

    /// The durability barrier: wait until `cycle` and every cycle before it
    /// have landed. `Err` if the writer stops first — `cycle` will not land
    /// through this writer.
    pub async fn wait_durable(&mut self, cycle: CycleId) -> Result<DurableMark, WriterStop> {
        loop {
            let mark = *self.durable.borrow_and_update();
            if mark.through.is_some_and(|t| t >= cycle) {
                return Ok(mark);
            }
            if let Some(stop) = mark.stopped {
                return Err(stop);
            }
            if self.durable.changed().await.is_err() {
                // The writer is gone; report what it last published.
                let last = *self.durable.borrow();
                return if last.through.is_some_and(|t| t >= cycle) {
                    Ok(last)
                } else {
                    Err(last.stopped.unwrap_or(WriterStop::Closed))
                };
            }
        }
    }
}

/// The writer side: owns the sink and seals cycles in submission order.
pub struct CycleWriter<S> {
    sink: S,
    rx: mpsc::Receiver<Job>,
    head: DatasetVersion,
    last: Option<CycleId>,
    durable: watch::Sender<DurableMark>,
}

/// What [`CycleWriter::run`] hands back when it stops: the sink (for recovery
/// or a fresh writer) and the final durable mark.
pub struct WriterExit<S> {
    pub sink: S,
    pub mark: DurableMark,
}

/// Build a writer over `sink`, whose current durable head is `head`, with a
/// queue of `depth` cycles (at least 1). Each queued cycle holds its casts in
/// memory, so `depth` bounds that memory.
pub fn cycle_writer<S: WalSink>(
    sink: S,
    head: DatasetVersion,
    depth: usize,
) -> (CycleWriter<S>, CycleWriterHandle) {
    let (tx, rx) = mpsc::channel(depth.max(1));
    let (durable, durable_rx) = watch::channel(DurableMark {
        through: None,
        head,
        stopped: None,
    });
    (
        CycleWriter {
            sink,
            rx,
            head,
            last: None,
            durable,
        },
        CycleWriterHandle {
            tx,
            durable: durable_rx,
        },
    )
}

impl<S: WalSink> CycleWriter<S> {
    /// Seal queued cycles until every handle is dropped or a seal fails.
    pub async fn run(mut self) -> WriterExit<S> {
        let mut failed: Option<CycleId> = None;
        while let Some(job) = self.rx.recv().await {
            if let Some(failed) = failed {
                // Queued behind a failure: hand the casts back, seal nothing.
                let _ = job.reply.send(Err(SealError::Poisoned {
                    failed,
                    cycle: job.cycle,
                    casts: job.casts,
                }));
                continue;
            }
            if let Some(last) = self.last.filter(|&last| job.cycle <= last) {
                let _ = job.reply.send(Err(SealError::OutOfOrder {
                    last,
                    cycle: job.cycle,
                    casts: job.casts,
                }));
                continue;
            }
            let frame = CycleFrame::new(job.cycle, self.head);
            match seal_cycle(&mut self.sink, frame, job.casts).await {
                Ok(sealed) => {
                    self.head = match sealed.outcome {
                        CommitOutcome::NoChange { .. } => self.head,
                        CommitOutcome::Committed { version, .. } => version,
                        CommitOutcome::Reconciled { current_head, .. } => current_head,
                    };
                    self.last = Some(job.cycle);
                    self.durable.send_replace(DurableMark {
                        through: self.last,
                        head: self.head,
                        stopped: None,
                    });
                    let _ = job.reply.send(Ok(sealed));
                }
                Err(failure) => {
                    failed = Some(job.cycle);
                    // Refuse new submissions; the ones already queued are
                    // still received below and get their casts back.
                    self.rx.close();
                    self.durable.send_modify(|m| {
                        m.stopped = Some(WriterStop::Poisoned { failed: job.cycle });
                    });
                    let _ = job.reply.send(Err(SealError::Failed(failure)));
                }
            }
        }
        if failed.is_none() {
            self.durable
                .send_modify(|m| m.stopped = Some(WriterStop::Closed));
        }
        let mark = *self.durable.borrow();
        WriterExit {
            sink: self.sink,
            mark,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cycle_driver::SealRecovery;
    use lance_graph_planner::persist_sink::{
        CommitError, DetachedCycleBatch, FrameMeta, LandedSlot, WriteFailed,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    /// One-writer fake: fences on `base_version`, publishes head + 1, and can
    /// hold commits behind a gate or fail a chosen cycle.
    struct GateWal {
        head: u64,
        /// `(cycle, base_version, batch_hash)` per landed commit.
        commits: Vec<(CycleId, DatasetVersion, u64)>,
        gate: Option<Arc<Semaphore>>,
        entered: Arc<AtomicUsize>,
        fail_on: Option<CycleId>,
    }

    impl GateWal {
        fn new(head: u64) -> Self {
            Self {
                head,
                commits: Vec::new(),
                gate: None,
                entered: Arc::new(AtomicUsize::new(0)),
                fail_on: None,
            }
        }
    }

    impl WalSink for GateWal {
        async fn commit_cycle(
            &mut self,
            batch: DetachedCycleBatch,
        ) -> Result<CommitOutcome, CommitError> {
            self.entered.fetch_add(1, Ordering::SeqCst);
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
            self.commits.push((
                batch.frame.cycle,
                batch.frame.base_version,
                batch.batch_hash,
            ));
            Ok(CommitOutcome::Committed {
                version: DatasetVersion(self.head),
                cycle: batch.frame.cycle,
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

    /// `n` artifact casts for `cycle`, one owner per row, 512-byte payloads.
    fn casts(cycle: u64, n: u64) -> Vec<SweepSlot> {
        (0..n)
            .map(|i| {
                let mut payload = vec![0u8; 512];
                payload[..8].copy_from_slice(&(cycle * 1_000_000 + i).to_le_bytes());
                SweepSlot {
                    cycle: CycleId(cycle),
                    stream_position: cycle * 1_000_000 + i,
                    owner: i as _,
                    row: i,
                    paired_move: None,
                    payload,
                }
            })
            .collect()
    }

    async fn yield_until(mut done: impl FnMut() -> bool) {
        for _ in 0..10_000 {
            if done() {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("condition never became true");
    }

    #[tokio::test]
    async fn submit_returns_before_the_commit_lands_and_cycles_chain_their_heads() {
        let gate = Arc::new(Semaphore::new(0));
        let mut sink = GateWal::new(7);
        sink.gate = Some(gate.clone());
        let entered = sink.entered.clone();
        let (writer, handle) = cycle_writer(sink, DatasetVersion(7), 4);

        let (exit, ()) = tokio::join!(writer.run(), async move {
            let mut t1 = handle
                .try_submit(CycleId(1), casts(1, 64))
                .expect("submit 1");
            let t2 = handle
                .try_submit(CycleId(2), casts(2, 64))
                .expect("submit 2");
            // The writer is inside cycle 1's commit, held by the gate …
            yield_until(|| entered.load(Ordering::SeqCst) == 1).await;
            // … and both submissions already returned: nothing has landed.
            assert!(t1.try_sealed().is_none(), "cycle 1 cannot have sealed yet");
            assert_eq!(handle.durable().through, None);

            gate.add_permits(2);
            let s1 = t1.sealed().await.expect("cycle 1 seals");
            let s2 = t2.sealed().await.expect("cycle 2 seals");
            assert_eq!(s1.publication_version, Some(DatasetVersion(8)));
            assert_eq!(s2.publication_version, Some(DatasetVersion(9)));
            let mut h = handle.clone();
            let mark = h.wait_durable(CycleId(2)).await.expect("durable");
            assert_eq!(mark.through, Some(CycleId(2)));
            assert_eq!(mark.head, DatasetVersion(9));
            drop(h);
            drop(handle);
        });
        // Cycle 2 was framed on the version cycle 1 published, not on the
        // head the writer started from — otherwise the fake would fence it.
        let bases: Vec<_> = exit.sink.commits.iter().map(|c| c.1).collect();
        assert_eq!(bases, vec![DatasetVersion(7), DatasetVersion(8)]);
        assert_eq!(exit.mark.stopped, Some(WriterStop::Closed));
    }

    #[tokio::test]
    async fn the_writer_seals_the_same_batch_as_an_inline_seal() {
        let mut inline_sink = GateWal::new(3);
        let inline = seal_cycle(
            &mut inline_sink,
            CycleFrame::new(CycleId(5), DatasetVersion(3)),
            casts(5, 200),
        )
        .await
        .expect("inline seal");

        let (writer, handle) = cycle_writer(GateWal::new(3), DatasetVersion(3), 2);
        let (exit, background) = tokio::join!(writer.run(), async move {
            let t = handle
                .try_submit(CycleId(5), casts(5, 200))
                .expect("submit");
            drop(handle);
            t.sealed().await.expect("background seal")
        });
        assert_eq!(background.outcome, inline.outcome, "same batch identity");
        assert_eq!(background.next_position_base, inline.next_position_base);
        assert_eq!(exit.sink.commits, inline_sink.commits);
    }

    #[tokio::test]
    async fn a_failed_seal_poisons_the_queue_and_hands_every_cast_back() {
        let gate = Arc::new(Semaphore::new(0));
        let mut sink = GateWal::new(0);
        sink.gate = Some(gate.clone());
        sink.fail_on = Some(CycleId(2));
        let entered = sink.entered.clone();
        let (writer, handle) = cycle_writer(sink, DatasetVersion(0), 8);

        let (exit, ()) = tokio::join!(writer.run(), async move {
            let t1 = handle.try_submit(CycleId(1), casts(1, 16)).expect("1");
            let t2 = handle.try_submit(CycleId(2), casts(2, 16)).expect("2");
            let t3 = handle.try_submit(CycleId(3), casts(3, 16)).expect("3");
            yield_until(|| entered.load(Ordering::SeqCst) == 1).await;
            gate.add_permits(8);

            assert!(t1.sealed().await.is_ok(), "cycle 1 lands");
            match t2.sealed().await {
                Err(SealError::Failed(f)) => {
                    assert_eq!(f.recovery(), SealRecovery::Regenerate);
                    assert_eq!(f.casts, casts(2, 16), "failed casts come back");
                }
                other => panic!("cycle 2 must fail at the WAL, got {other:?}"),
            }
            match t3.sealed().await {
                Err(SealError::Poisoned {
                    failed,
                    cycle,
                    casts: back,
                }) => {
                    assert_eq!((failed, cycle), (CycleId(2), CycleId(3)));
                    assert_eq!(back, casts(3, 16), "queued casts come back untouched");
                }
                other => panic!("cycle 3 must be poisoned, got {other:?}"),
            }
            match handle.try_submit(CycleId(4), casts(4, 16)) {
                Err(SubmitError::Closed { cycle, casts: back }) => {
                    assert_eq!(cycle, CycleId(4));
                    assert_eq!(back, casts(4, 16));
                }
                other => panic!("a poisoned writer must refuse cycle 4, got {other:?}"),
            }
            let mut h = handle.clone();
            assert_eq!(
                h.wait_durable(CycleId(3)).await,
                Err(WriterStop::Poisoned { failed: CycleId(2) })
            );
            drop(h);
            drop(handle);
        });
        // Durable progress stopped at the gap-free prefix: only cycle 1.
        assert_eq!(exit.mark.through, Some(CycleId(1)));
        assert_eq!(exit.sink.commits.len(), 1);
    }

    #[tokio::test]
    async fn an_out_of_order_cycle_is_refused_without_poisoning() {
        let (writer, handle) = cycle_writer(GateWal::new(0), DatasetVersion(0), 8);
        let (exit, ()) = tokio::join!(writer.run(), async move {
            let t2 = handle.try_submit(CycleId(2), casts(2, 8)).expect("2");
            let t1 = handle.try_submit(CycleId(1), casts(1, 8)).expect("1");
            let t3 = handle.try_submit(CycleId(3), casts(3, 8)).expect("3");
            assert!(t2.sealed().await.is_ok());
            match t1.sealed().await {
                Err(SealError::OutOfOrder {
                    last,
                    cycle,
                    casts: back,
                }) => {
                    assert_eq!((last, cycle), (CycleId(2), CycleId(1)));
                    assert_eq!(back, casts(1, 8));
                }
                other => panic!("cycle 1 after 2 must be refused, got {other:?}"),
            }
            assert!(t3.sealed().await.is_ok(), "the writer kept going");
            drop(handle);
        });
        assert_eq!(exit.mark.through, Some(CycleId(3)));
        assert_eq!(exit.mark.stopped, Some(WriterStop::Closed));
    }

    #[tokio::test]
    async fn a_full_queue_hands_the_casts_back() {
        let gate = Arc::new(Semaphore::new(0));
        let mut sink = GateWal::new(0);
        sink.gate = Some(gate.clone());
        let entered = sink.entered.clone();
        let (writer, handle) = cycle_writer(sink, DatasetVersion(0), 1);
        let (_exit, ()) = tokio::join!(writer.run(), async move {
            let t1 = handle.try_submit(CycleId(1), casts(1, 4)).expect("1");
            // Cycle 1 leaves the queue for the commit; cycle 2 fills it.
            yield_until(|| entered.load(Ordering::SeqCst) == 1).await;
            let t2 = handle.try_submit(CycleId(2), casts(2, 4)).expect("2");
            match handle.try_submit(CycleId(3), casts(3, 4)) {
                Err(SubmitError::Full { cycle, casts: back }) => {
                    assert_eq!(cycle, CycleId(3));
                    assert_eq!(back, casts(3, 4));
                }
                other => panic!("depth 1 with one queued must be full, got {other:?}"),
            }
            gate.add_permits(4);
            assert!(t1.sealed().await.is_ok());
            assert!(t2.sealed().await.is_ok());
            drop(handle);
        });
    }

    /// What the loop pays per cycle at 64k × 512 B: handing the casts to the
    /// writer, against sealing them inline (the sink commits instantly, so the
    /// inline figure is freeze + retry copy only — no Lance I/O).
    ///
    /// `cargo test -p lance-graph-supervisor --features cycle-driver --release
    ///  --lib submit_vs_inline_seal_at_64k -- --ignored --nocapture`
    #[tokio::test]
    #[ignore = "timing probe; run in release"]
    async fn submit_vs_inline_seal_at_64k() {
        use std::time::Instant;
        const N: u64 = 65_536;
        const RUNS: usize = 7;
        let median = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let mut inline_ms = Vec::new();
        for r in 0..RUNS {
            let mut sink = GateWal::new(r as u64);
            let c = casts(1, N);
            let t = Instant::now();
            let sealed = seal_cycle(
                &mut sink,
                CycleFrame::new(CycleId(1), DatasetVersion(r as u64)),
                c,
            )
            .await
            .expect("inline");
            inline_ms.push(t.elapsed().as_secs_f64() * 1e3);
            drop(sealed);
        }
        let (writer, handle) = cycle_writer(GateWal::new(0), DatasetVersion(0), RUNS + 1);
        let (_exit, submit_us) = tokio::join!(writer.run(), async move {
            let mut us = Vec::new();
            let mut tickets = Vec::new();
            for r in 0..RUNS as u64 {
                let c = casts(r + 1, N);
                let t = Instant::now();
                tickets.push(handle.try_submit(CycleId(r + 1), c).expect("submit"));
                us.push(t.elapsed().as_secs_f64() * 1e6);
            }
            for t in tickets {
                t.sealed().await.expect("seal");
            }
            drop(handle);
            us
        });
        eprintln!(
            "64k x 512B per cycle: inline seal {:.2} ms, try_submit {:.2} us (median of {RUNS})",
            median(inline_ms),
            median(submit_us)
        );
    }
}
