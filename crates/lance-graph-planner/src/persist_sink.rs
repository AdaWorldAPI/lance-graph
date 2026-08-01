//! The D-MBX-A6 persistence sink — **two cleanly separated clock domains**.
//!
//! This is the POST-write half of the fire-and-forget flow (pre-write half =
//! [`crate::owner_adapter`]). The thinker already reported and moved on:
//!
//! ```text
//! thinking path:      SoA → BatchWriter.cast(on_behalf, paired_move, descriptor) → keep thinking
//! persistence path:   PersistCast → async durable append → DurableReceipt      (NO owner borrow)
//! owner-local step:   DurableReceipt + paired move → try_advance_phase          (NO await)
//! ```
//!
//! ## Why the two phases must NOT be one async function (operator-ruled)
//!
//! A monolithic `persist_then_step(&mut owner, …, bytes).await` would hold BOTH
//! `&mut owner` and `&[u8]` borrowed from live owner state **across the WAL I/O
//! await** — immobilizing the owner while the object store completes, which
//! defeats the latency masking the architecture exists to obtain. So the durable
//! write ([`persist_cast`], async, **no `O` in its signature**) and the lifecycle
//! step ([`apply_durable_step`], sync, **no await**) are split. The exclusive
//! `&mut owner` is held only in the sync phase, outside the storage-latency window;
//! `MailboxSoaOwner::try_advance_phase` stays the SOLE lifecycle mutator.
//!
//! ## The ordering invariant (what the falsifiers prove)
//!
//! - **No durable write ⇒ no receipt ⇒ no step.** A failed append yields no
//!   [`DurableReceipt`], and [`apply_durable_step`] cannot be reached without one.
//! - **The step is the PAIRED move** (`receipt.paired_move.to`), never a generic
//!   `next_phases().first()`. The move the thought cast rides in the receipt.
//! - A receipt is applied only to **its own** owner ([`PersistError::OwnerMismatch`]).
//!
//! ## The `DurableWrite` seam + the durability type
//!
//! A WAL append produces a durable **coordinate** (shard + writer epoch + WAL
//! entry position), NOT a base `DatasetVersion` — the dataset version arrives
//! later via MemTable flush + manifest commit. So [`DurableWrite::append`] returns
//! [`DurableCoordinate`], never a version. The concrete impl (a lance-having
//! crate) wires lance 7.0.0's OFFICIAL MemWAL — preferring the high-level
//! `ShardWriter::put` (`enable_memtable + durable_write`: insert into the
//! queryable MemTable, release the lock, then await WAL durability off-lock) over
//! the raw `WalAppender::append` primitive — over an Arrow `RecordBatch` whose
//! buffers are independently owned (Arrow shared-buffer ownership), so "zero-copy"
//! never means holding the SoA owner borrowed across I/O.
//!
//! Keeping the seam a trait leaves the ordering core lance-free and `protoc`-free.

use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::{KanbanMove, RubiconTransitionError};
use lance_graph_contract::soa_view::MailboxSoaOwner;

/// A durable write that did not land (the WAL append failed / was fenced).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteFailed(pub String);

/// A backend-neutral **durable coordinate** — where the write landed in the
/// durable log. This is NOT a base Lance `DatasetVersion` (that arrives later via
/// MemTable flush + manifest commit); it is the WAL/LSM coordinate that proves
/// durability now. For lance: `shard` = shard `Uuid` (as `u128`),
/// `writer_epoch` + `wal_entry_position` from the `ShardWriter`/`WalAppendResult`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DurableCoordinate {
    /// Opaque shard identity (lance: the shard `Uuid` as `u128`).
    pub shard: u128,
    /// The writer epoch that fenced this append.
    pub writer_epoch: u64,
    /// The monotonic WAL entry position of this durable append.
    pub wal_entry_position: u64,
}

/// The async durable-append seam.
///
/// `&self` (shared — many casts drain concurrently) and **no owner borrow**: the
/// persistence path must not hold the SoA owner across object-store I/O. `payload`
/// is an independently-owned buffer (never a borrow of live owner state), so the
/// owner stays free while the WAL hums. `async` because lance's WAL is async and
/// the sink runs on the background persistence path (never the hot thinker path);
/// `async_fn_in_trait` is allowed (generic use only, never `dyn`).
#[allow(async_fn_in_trait)]
pub trait DurableWrite {
    /// Durably append `payload` on behalf of `owner`. `Ok(coordinate)` = it landed
    /// (durable now); `Err` = it did not (⇒ no receipt, ⇒ no step).
    async fn append(
        &self,
        owner: MailboxId,
        payload: &[u8],
    ) -> Result<DurableCoordinate, WriteFailed>;
}

/// A **detached** cast envelope — the thinker's report, carrying its OWN payload
/// (independently owned, never a borrow of live owner state) so persistence runs
/// with the owner free. Mirrors what `BatchWriter::cast(on_behalf, moves, payload)`
/// stages; at drain the descriptor is resolved to `payload` bytes.
pub struct PersistCast {
    /// The mailbox this write is on behalf of.
    pub owner: MailboxId,
    /// The lifecycle move the thought cast with this write (its `to` is applied
    /// post-durability). `None` when the cast carries no lifecycle intent.
    pub paired_move: Option<KanbanMove>,
    /// The bytes to persist — an owned/independent buffer (the concrete sink forms
    /// an Arrow `RecordBatch` over it with shared-buffer ownership).
    pub payload: Vec<u8>,
}

/// Proof the write landed (a [`DurableCoordinate`]) plus the paired move to apply
/// and the owner it belongs to. Produced by [`persist_cast`] on success; consumed
/// by [`apply_durable_step`]. Its mere existence is the "the write landed" fact —
/// there is no separate ack/confirmation ledger (`E-ACK-ELIMINATED-1`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableReceipt {
    /// The mailbox the durable write was on behalf of.
    pub owner: MailboxId,
    /// The paired lifecycle move to apply post-durability.
    pub paired_move: Option<KanbanMove>,
    /// Where the write landed in the durable log.
    pub coordinate: DurableCoordinate,
}

/// Why a persist / step operation produced no lifecycle advance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistError {
    /// The durable write did not land — **no receipt exists, so no step can be
    /// applied** (the ordering invariant's negative half).
    Write(WriteFailed),
    /// The paired move is not a legal Rubicon edge from the owner's current phase —
    /// surfaced, never silently applied.
    Illegal(RubiconTransitionError),
    /// A receipt was applied to the wrong owner — refused (a receipt only advances
    /// the mailbox it was minted for; write-on-behalf never crosses owners).
    OwnerMismatch {
        receipt_owner: MailboxId,
        applying_to: MailboxId,
    },
}

/// **Phase 1 — async persistence, NO owner borrow.** Durably append the cast's
/// payload; on success return a [`DurableReceipt`], on failure a
/// [`PersistError::Write`] and **no receipt**. `O` (the owner) does not appear in
/// this signature at all — it is never borrowed across the WAL / object-store
/// await, so the owner stays free while durability runs.
pub async fn persist_cast<W: DurableWrite>(
    sink: &W,
    cast: PersistCast,
) -> Result<DurableReceipt, PersistError> {
    let coordinate = sink
        .append(cast.owner, &cast.payload)
        .await
        .map_err(PersistError::Write)?;
    Ok(DurableReceipt {
        owner: cast.owner,
        paired_move: cast.paired_move,
        coordinate,
    })
}

/// **Phase 2 — synchronous owner-local completion, NO await.** Given a
/// [`DurableReceipt`] (⇒ the write already landed), apply the PAIRED move via the
/// owner's checked `try_advance_phase`. The exclusive `&mut owner` is held only
/// here, outside the storage-latency window. Refuses a receipt minted for a
/// different owner ([`PersistError::OwnerMismatch`]).
///
/// Returns `Ok(Some(step))` (paired move applied), `Ok(None)` (receipt carried no
/// move — a durable no-step), or an error. The transition target is
/// `receipt.paired_move.to` — never a generic successor.
pub fn apply_durable_step<O: MailboxSoaOwner>(
    owner: &mut O,
    receipt: DurableReceipt,
) -> Result<Option<KanbanMove>, PersistError> {
    if receipt.owner != owner.mailbox_id() {
        return Err(PersistError::OwnerMismatch {
            receipt_owner: receipt.owner,
            applying_to: owner.mailbox_id(),
        });
    }
    match receipt.paired_move {
        Some(mv) => owner
            .try_advance_phase(mv.to)
            .map(Some)
            .map_err(PersistError::Illegal),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::collapse_gate::MailboxId as MId;
    use lance_graph_contract::kanban::{ExecTarget, KanbanColumn};
    use lance_graph_contract::soa_view::{MailboxSoaOwner, MailboxSoaView};

    /// Minimal in-RAM owner (mirrors `kanban_actor::tests::TestBoard`).
    struct FakeOwner {
        id: MId,
        phase: KanbanColumn,
        cycle: u32,
    }
    impl MailboxSoaView for FakeOwner {
        fn mailbox_id(&self) -> MId {
            self.id
        }
        fn n_rows(&self) -> usize {
            0
        }
        fn w_slot(&self) -> u8 {
            0
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
    impl MailboxSoaOwner for FakeOwner {
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

    /// A `DurableWrite` whose success/failure and call-count are observable.
    /// `&self` + interior-mutable counter, mirroring the real sink's shared shape.
    struct FakeSink {
        succeed: bool,
        calls: std::cell::Cell<u32>,
    }
    impl FakeSink {
        fn new(succeed: bool) -> Self {
            Self {
                succeed,
                calls: std::cell::Cell::new(0),
            }
        }
    }
    impl DurableWrite for FakeSink {
        async fn append(
            &self,
            _owner: MId,
            _payload: &[u8],
        ) -> Result<DurableCoordinate, WriteFailed> {
            self.calls.set(self.calls.get() + 1);
            if self.succeed {
                Ok(DurableCoordinate {
                    shard: 0xABCD,
                    writer_epoch: 1,
                    wal_entry_position: 7,
                })
            } else {
                Err(WriteFailed("wal fenced".into()))
            }
        }
    }

    fn owner(phase: KanbanColumn) -> FakeOwner {
        FakeOwner {
            id: 42,
            phase,
            cycle: 5,
        }
    }
    fn cast(paired_to: Option<KanbanColumn>) -> PersistCast {
        PersistCast {
            owner: 42,
            paired_move: paired_to.map(|to| KanbanMove {
                mailbox: 42,
                from: KanbanColumn::Planning,
                to,
                witness_chain_position: 0,
                exec: ExecTarget::Elixir,
            }),
            payload: vec![1, 2, 3, 4],
        }
    }
    fn receipt_for(paired_to: Option<KanbanColumn>) -> DurableReceipt {
        DurableReceipt {
            owner: 42,
            paired_move: paired_to.map(|to| KanbanMove {
                mailbox: 42,
                from: KanbanColumn::Planning,
                to,
                witness_chain_position: 0,
                exec: ExecTarget::Elixir,
            }),
            coordinate: DurableCoordinate {
                shard: 0xABCD,
                writer_epoch: 1,
                wal_entry_position: 7,
            },
        }
    }

    // ── Phase 1: async persistence, NO owner borrow ────────────────────────────

    #[tokio::test]
    async fn a_successful_write_yields_a_receipt_carrying_the_paired_move() {
        let sink = FakeSink::new(true);
        let r = persist_cast(&sink, cast(Some(KanbanColumn::CognitiveWork)))
            .await
            .expect("write landed");
        assert_eq!(r.owner, 42);
        assert_eq!(
            r.paired_move.map(|m| m.to),
            Some(KanbanColumn::CognitiveWork)
        );
        assert_eq!(
            r.coordinate.wal_entry_position, 7,
            "the durable coordinate rides"
        );
        assert_eq!(sink.calls.get(), 1, "the write was attempted once");
    }

    #[tokio::test]
    async fn a_failed_write_yields_no_receipt() {
        // The negative half: no durable write ⇒ NO receipt exists. There is
        // therefore nothing to hand to apply_durable_step ⇒ no step can occur.
        let sink = FakeSink::new(false);
        let r = persist_cast(&sink, cast(Some(KanbanColumn::CognitiveWork))).await;
        assert_eq!(
            r,
            Err(PersistError::Write(WriteFailed("wal fenced".into())))
        );
        assert_eq!(sink.calls.get(), 1, "the write was attempted");
    }

    // ── Phase 2: synchronous owner-local completion, NO await ───────────────────

    #[test]
    fn applying_a_receipt_advances_the_owner_by_the_paired_move() {
        let mut o = owner(KanbanColumn::Planning);
        let step = apply_durable_step(&mut o, receipt_for(Some(KanbanColumn::CognitiveWork)))
            .expect("legal")
            .expect("a paired move");
        assert_eq!(step.to, KanbanColumn::CognitiveWork);
        assert_eq!(o.phase(), KanbanColumn::CognitiveWork, "advanced once");
        assert_eq!(o.current_cycle(), 6);
    }

    #[test]
    fn applying_a_receipt_uses_the_paired_move_not_the_generic_successor() {
        // THE key falsifier. From `Evaluation` the generic forward arc is `Commit`;
        // the receipt carries the free-won't veto `Evaluation → Prune`. The step
        // must be the PAIRED move (Prune), never the generic successor (Commit).
        assert_eq!(
            KanbanColumn::Evaluation.next_phases().first(),
            Some(&KanbanColumn::Commit),
            "precondition: generic successor is Commit",
        );
        let mut o = owner(KanbanColumn::Evaluation);
        let step = apply_durable_step(&mut o, receipt_for(Some(KanbanColumn::Prune)))
            .expect("legal")
            .expect("paired veto");
        assert_eq!(step.to, KanbanColumn::Prune, "the paired veto, not Commit");
        assert_eq!(o.phase(), KanbanColumn::Prune);
        assert_ne!(
            o.phase(),
            KanbanColumn::Commit,
            "generic successor NOT taken"
        );
    }

    #[test]
    fn a_receipt_with_no_paired_move_is_a_durable_no_step() {
        let mut o = owner(KanbanColumn::CognitiveWork);
        let r = apply_durable_step(&mut o, receipt_for(None));
        assert_eq!(r, Ok(None), "durable, but no step");
        assert_eq!(o.phase(), KanbanColumn::CognitiveWork, "phase unchanged");
    }

    #[test]
    fn an_illegal_paired_edge_is_surfaced_not_applied() {
        // Planning → Evaluation skips CognitiveWork — illegal. Surfaced; owner
        // untouched (the checked airgap holds in the owner-local phase).
        let mut o = owner(KanbanColumn::Planning);
        let r = apply_durable_step(&mut o, receipt_for(Some(KanbanColumn::Evaluation)));
        assert!(matches!(r, Err(PersistError::Illegal(_))));
        assert_eq!(o.phase(), KanbanColumn::Planning, "owner untouched");
    }

    #[test]
    fn a_receipt_is_refused_for_a_foreign_owner() {
        // A receipt minted for mailbox 42 must never advance mailbox 99.
        let mut other = FakeOwner {
            id: 99,
            phase: KanbanColumn::Planning,
            cycle: 5,
        };
        let r = apply_durable_step(&mut other, receipt_for(Some(KanbanColumn::CognitiveWork)));
        assert!(matches!(
            r,
            Err(PersistError::OwnerMismatch {
                receipt_owner: 42,
                applying_to: 99
            })
        ));
        assert_eq!(
            other.phase(),
            KanbanColumn::Planning,
            "foreign owner untouched"
        );
    }
}
