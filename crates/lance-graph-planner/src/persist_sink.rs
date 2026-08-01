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
//! ## Crash-durability: the paired move is CO-LOCATED, not in-memory-only
//!
//! The paired transition witness is **durably co-located with the SoA state in
//! the same persistence generation** (operator ruling). The naïve split persists
//! only the payload and keeps the paired move alive solely in the in-memory
//! [`DurableReceipt`]; then *WAL append lands → process dies before
//! [`apply_durable_step`] → the receipt evaporates → the paired move is lost →
//! the KanbanStep never fires*, even though the durable SoA state moved. The gap
//! is silent: storage advanced, lifecycle did not.
//!
//! So [`DurableWrite::append`] takes the [`DurableWitness`] (owner, cast id,
//! cycle, paired move) **alongside** the payload and lands BOTH atomically. The
//! [`DurableReceipt`] then merely *references* that durable material (via its
//! [`DurableCoordinate`]) for the fast in-process apply; it is **not the only
//! copy**. On restart, [`recover_and_apply`] reads the witnesses back
//! ([`DurableWrite::scan_witnesses`]), runs `temporal` layer-1 causal
//! deinterlacing to find each owner's pending tail, and re-applies the moves in
//! cast order. This is a read of what durable storage already holds — **not** a
//! separate ack / confirmation ledger (`E-ACK-ELIMINATED-1`).
//!
//! ## The `DurableWrite` seam + the durability type
//!
//! A WAL append produces a durable **coordinate** (shard + writer epoch + WAL
//! entry position), NOT a base `DatasetVersion` — the dataset version arrives
//! later via MemTable flush + manifest commit. So [`DurableWrite::append`] returns
//! [`DurableCoordinate`], never a version. This is exactly why the coordinate,
//! not `LanceVersion`, is the durability proof: the write is queryable and
//! durable *before* any base manifest version attaches (falsifier 5 —
//! WAL-visible-before-manifest — proves the latest local state is identified from
//! the co-located witness's cast order, not from a dataset version that does not
//! exist yet).
//!
//! The concrete impl (a lance-having crate) wires lance 7.0.0's OFFICIAL MemWAL —
//! preferring the high-level `ShardWriter::put` (`enable_memtable + durable_write`:
//! insert into the queryable MemTable, release the lock, then await WAL durability
//! off-lock) over the raw `WalAppender::append` primitive. The witness and the
//! payload are two columns of ONE `RecordBatch` (one generation, atomic), whose
//! buffers are independently owned, so "no owner borrow across I/O" holds without
//! any claim of magic zero-copy.
//!
//! Keeping the seam a trait leaves the ordering core lance-free and `protoc`-free.
//! **This module builds NO concrete `LanceShardSink`** — per operator ruling the
//! durable-witness reshape + `temporal` layer-1 land first; the production sink
//! comes after, gated on the crash-recovery falsifiers here going green.

use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::{KanbanColumn, KanbanMove, RubiconTransitionError};
use lance_graph_contract::soa_view::MailboxSoaOwner;

use crate::batch_writer::CastId;
use crate::temporal::{local_trajectory_of, LocalCausalRow};

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

/// The durable transition witness — CO-LOCATED with the SoA payload in the same
/// persistence generation so it survives a crash (module doc § Crash-durability).
///
/// Everything needed to re-apply a pending KanbanStep after a restart lives here:
/// WHO (`owner`), WHICH cast (`cast_id` — the owner-local replay order), the
/// owner-local `cycle`, and the `paired_move` to apply. On recovery,
/// [`recover_and_apply`] reads these back and replays in `cast_id` order.
///
/// Implements [`LocalCausalRow`] so `temporal` layer-1 can deinterlace a global
/// interleaved witness stream into each owner's local trajectory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableWitness {
    /// The mailbox this write is on behalf of — the deinterlace grouping key.
    pub owner: MailboxId,
    /// The cast this write realises — its `.0` is the owner-local replay order.
    pub cast_id: CastId,
    /// The owner-local cycle at the time of the cast (audit / trajectory).
    pub cycle: u32,
    /// The lifecycle move to apply once this generation is durable. `None` when
    /// the cast carried no lifecycle intent (a durable no-step).
    pub paired_move: Option<KanbanMove>,
}

impl LocalCausalRow for DurableWitness {
    fn owner(&self) -> MailboxId {
        self.owner
    }
    fn cast_seq(&self) -> u64 {
        self.cast_id.0
    }
}

/// The async durable-append + replay seam.
///
/// `&self` (shared — many casts drain concurrently) and **no owner borrow**: the
/// persistence path must not hold the SoA owner across object-store I/O. Both the
/// [`DurableWitness`] and the `payload` are independently owned (never a borrow of
/// live owner state), so the owner stays free while the WAL hums. `async` because
/// lance's WAL is async and the sink runs on the background persistence path
/// (never the hot thinker path); `async_fn_in_trait` is allowed (generic use
/// only, never `dyn`).
#[allow(async_fn_in_trait)]
pub trait DurableWrite {
    /// Durably append the `witness` CO-LOCATED with `payload`, atomically, in ONE
    /// persistence generation. `Ok(coordinate)` = both landed together (durable
    /// now); `Err` = neither did (⇒ no receipt, ⇒ no step). The witness is the
    /// crash-durable copy of the paired move — never in-memory-only.
    async fn append(
        &self,
        witness: &DurableWitness,
        payload: &[u8],
    ) -> Result<DurableCoordinate, WriteFailed>;

    /// Replay seam: read back every durably-landed [`DurableWitness`], in the
    /// durable log's own (globally interleaved) order. Crash recovery scans this,
    /// runs `temporal` layer-1 to split it per owner, and re-applies pending moves
    /// in cast order ([`recover_and_apply`]). Reading the SAME co-located material
    /// the receipts merely referenced — NOT a separate ledger.
    async fn scan_witnesses(&self) -> Result<Vec<DurableWitness>, WriteFailed>;
}

/// A **detached** cast envelope — the thinker's report, carrying its OWN payload
/// (independently owned, never a borrow of live owner state) so persistence runs
/// with the owner free. Mirrors what `BatchWriter::cast(on_behalf, moves, payload)`
/// stages; at drain the descriptor is resolved to `payload` bytes.
pub struct PersistCast {
    /// The mailbox this write is on behalf of.
    pub owner: MailboxId,
    /// The cast this write realises — the owner-local replay order (from
    /// `BatchWriter::cast`). Rides into the [`DurableWitness`] for crash replay.
    pub cast_id: CastId,
    /// The owner-local cycle at cast time — rides into the witness (audit).
    pub cycle: u32,
    /// The lifecycle move the thought cast with this write (its `to` is applied
    /// post-durability). `None` when the cast carries no lifecycle intent.
    pub paired_move: Option<KanbanMove>,
    /// The bytes to persist — an owned/independent buffer (the concrete sink forms
    /// one Arrow `RecordBatch` with the witness as a second column, one generation).
    pub payload: Vec<u8>,
}

/// Proof the write landed (a [`DurableCoordinate`]) plus the paired move to apply
/// and the owner it belongs to. Produced by [`persist_cast`] on success; consumed
/// by [`apply_durable_step`]. Its mere existence is the "the write landed" fact —
/// there is no separate ack/confirmation ledger (`E-ACK-ELIMINATED-1`).
///
/// **This receipt REFERENCES durable material; it is not the only copy of the
/// paired move.** The move is co-located in the durable generation the
/// [`coordinate`](Self::coordinate) points at (via the [`DurableWitness`] that
/// [`persist_cast`] appended). If this in-memory receipt is lost to a crash
/// before [`apply_durable_step`] runs, [`recover_and_apply`] reconstructs the
/// move from that durable generation — the KanbanStep is not lost with the
/// process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableReceipt {
    /// The mailbox the durable write was on behalf of.
    pub owner: MailboxId,
    /// The cast this receipt realises — mirrors the co-located witness's cast id.
    pub cast_id: CastId,
    /// The paired lifecycle move to apply post-durability. A convenience copy of
    /// the co-located witness's move (see the type doc — durable, not only here).
    pub paired_move: Option<KanbanMove>,
    /// Where the write landed in the durable log — the reference INTO the durable
    /// generation that co-locates the witness with the SoA state.
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
    /// The paired move's `from` does not match the owner's current phase — the
    /// move is stale (already applied, or out of cast order). Surfaced on the
    /// synchronous single-receipt path ([`apply_durable_step`]) so a mis-ordered
    /// or double apply is loud; the crash-recovery path ([`recover_and_apply`])
    /// instead SKIPS a stale move (it is already reflected in the durable state).
    StalePhase {
        owner_phase: KanbanColumn,
        move_from: KanbanColumn,
    },
}

/// **Phase 1 — async persistence, NO owner borrow.** Form the crash-durable
/// [`DurableWitness`] and append it CO-LOCATED with the cast's payload in one
/// generation; on success return a [`DurableReceipt`] (which merely references
/// that durable material), on failure a [`PersistError::Write`] and **no
/// receipt**. `O` (the owner) does not appear in this signature at all — it is
/// never borrowed across the WAL / object-store await, so the owner stays free
/// while durability runs.
pub async fn persist_cast<W: DurableWrite>(
    sink: &W,
    cast: PersistCast,
) -> Result<DurableReceipt, PersistError> {
    let witness = DurableWitness {
        owner: cast.owner,
        cast_id: cast.cast_id,
        cycle: cast.cycle,
        paired_move: cast.paired_move,
    };
    let coordinate = sink
        .append(&witness, &cast.payload)
        .await
        .map_err(PersistError::Write)?;
    Ok(DurableReceipt {
        owner: cast.owner,
        cast_id: cast.cast_id,
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
        Some(mv) => {
            // The move's `from` must match the owner's current phase: on the
            // synchronous path a mismatch is a stale / out-of-cast-order apply
            // and is surfaced loudly (a double apply would otherwise silently
            // re-run through `try_advance_phase`). Enforces cast order.
            if mv.from != owner.phase() {
                return Err(PersistError::StalePhase {
                    owner_phase: owner.phase(),
                    move_from: mv.from,
                });
            }
            owner
                .try_advance_phase(mv.to)
                .map(Some)
                .map_err(PersistError::Illegal)
        }
        None => Ok(None),
    }
}

/// **Crash recovery.** Given the durable witnesses read back via
/// [`DurableWrite::scan_witnesses`] (a globally-interleaved stream from every
/// owner) and an `owner` reconstructed at its durable phase, re-apply the owner's
/// PENDING paired moves in cast order and return them.
///
/// The read is `temporal` layer-1: [`local_trajectory_of`] deinterlaces the
/// global stream down to this owner's own chain, in `cast_id` order — the exact
/// replay order. Each move is applied only when its `from` matches the owner's
/// current phase; a move whose `from` no longer matches is **already reflected in
/// the recovered durable SoA state** and is SKIPPED (idempotent — re-running
/// recovery after catching up applies nothing). A move that matches `from` but is
/// not a legal Rubicon edge is a genuine corruption and is surfaced
/// ([`PersistError::Illegal`]).
///
/// This is why the paired move MUST be co-located in the durable generation: the
/// witnesses are the only reason a KanbanStep survives a crash between the WAL
/// append and [`apply_durable_step`]. No witnesses ⇒ nothing to replay ⇒ the step
/// is lost (the gap this reshape closes).
pub fn recover_and_apply<O: MailboxSoaOwner>(
    owner: &mut O,
    witnesses: &[DurableWitness],
) -> Result<Vec<KanbanMove>, PersistError> {
    let chain = local_trajectory_of(witnesses, owner.mailbox_id());
    let mut applied = Vec::new();
    for w in chain {
        let Some(mv) = w.paired_move else { continue };
        if mv.from != owner.phase() {
            // Stale: already reflected in the recovered durable state. Skip
            // (recovery is idempotent), do not surface — unlike the synchronous
            // single-receipt path, a stale move here is expected, not an error.
            continue;
        }
        let step = owner
            .try_advance_phase(mv.to)
            .map_err(PersistError::Illegal)?;
        applied.push(step);
    }
    Ok(applied)
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

    /// A `DurableWrite` whose success/failure and call-count are observable, and
    /// which RECORDS every witness it lands so [`scan_witnesses`] can replay them
    /// — this is what makes the crash-recovery falsifier real: the durable
    /// generation (here, `landed`) outlives the in-memory receipts. `&self` +
    /// interior mutability, mirroring the real sink's shared shape.
    struct FakeSink {
        succeed: bool,
        calls: std::cell::Cell<u32>,
        landed: std::cell::RefCell<Vec<DurableWitness>>,
    }
    impl FakeSink {
        fn new(succeed: bool) -> Self {
            Self {
                succeed,
                calls: std::cell::Cell::new(0),
                landed: std::cell::RefCell::new(Vec::new()),
            }
        }
    }
    impl DurableWrite for FakeSink {
        async fn append(
            &self,
            witness: &DurableWitness,
            _payload: &[u8],
        ) -> Result<DurableCoordinate, WriteFailed> {
            self.calls.set(self.calls.get() + 1);
            if self.succeed {
                // The witness lands CO-LOCATED and DURABLE — it survives even if
                // every in-memory receipt is dropped (the crash the reshape guards).
                self.landed.borrow_mut().push(witness.clone());
                Ok(DurableCoordinate {
                    shard: 0xABCD,
                    writer_epoch: 1,
                    wal_entry_position: self.landed.borrow().len() as u64,
                })
            } else {
                // The negative half: a fenced WAL lands NOTHING — no witness, so
                // nothing to replay, so no move and no step.
                Err(WriteFailed("wal fenced".into()))
            }
        }
        async fn scan_witnesses(&self) -> Result<Vec<DurableWitness>, WriteFailed> {
            Ok(self.landed.borrow().clone())
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
        cast_from(KanbanColumn::Planning, paired_to)
    }
    fn cast_from(from: KanbanColumn, paired_to: Option<KanbanColumn>) -> PersistCast {
        PersistCast {
            owner: 42,
            cast_id: CastId(0),
            cycle: 5,
            paired_move: paired_to.map(|to| KanbanMove {
                mailbox: 42,
                from,
                to,
                witness_chain_position: 0,
                exec: ExecTarget::Elixir,
            }),
            payload: vec![1, 2, 3, 4],
        }
    }
    fn receipt_for(paired_to: Option<KanbanColumn>) -> DurableReceipt {
        receipt_from(KanbanColumn::Planning, paired_to)
    }
    fn receipt_from(from: KanbanColumn, paired_to: Option<KanbanColumn>) -> DurableReceipt {
        DurableReceipt {
            owner: 42,
            cast_id: CastId(0),
            paired_move: paired_to.map(|to| KanbanMove {
                mailbox: 42,
                from,
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
            r.coordinate.wal_entry_position, 1,
            "the durable coordinate rides (first witness landed)"
        );
        assert_eq!(r.cast_id, CastId(0), "the receipt mirrors the cast id");
        assert_eq!(sink.calls.get(), 1, "the write was attempted once");
        // The witness LANDED durably (co-located) — not only in the receipt.
        let scanned = sink.scan_witnesses().await.expect("scan");
        assert_eq!(scanned.len(), 1, "one witness durably co-located");
        assert_eq!(
            scanned[0].paired_move.map(|m| m.to),
            Some(KanbanColumn::CognitiveWork),
            "the paired move is in the DURABLE witness, not only the in-memory receipt",
        );
    }

    #[tokio::test]
    async fn a_fenced_write_lands_no_witness_so_nothing_can_be_replayed() {
        // Negative half, at the durable layer: a failed append records NO witness,
        // so a subsequent crash-recovery scan finds nothing to replay ⇒ no move.
        let sink = FakeSink::new(false);
        let _ = persist_cast(&sink, cast(Some(KanbanColumn::CognitiveWork))).await;
        assert_eq!(
            sink.scan_witnesses().await.expect("scan").len(),
            0,
            "a fenced WAL leaves no durable witness — no move, no step",
        );
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
        let step = apply_durable_step(
            &mut o,
            receipt_from(KanbanColumn::Evaluation, Some(KanbanColumn::Prune)),
        )
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

    // ── Crash recovery: the co-located witness replays after the receipt is lost ─

    /// Build a `PersistCast` for owner `owner` with an explicit cast id and move.
    fn cast_of(owner: MId, cast_id: u64, from: KanbanColumn, to: KanbanColumn) -> PersistCast {
        PersistCast {
            owner,
            cast_id: CastId(cast_id),
            cycle: 0,
            paired_move: Some(KanbanMove {
                mailbox: owner,
                from,
                to,
                witness_chain_position: cast_id as u32,
                exec: ExecTarget::Elixir,
            }),
            payload: vec![cast_id as u8],
        }
    }

    /// THE crash falsifier (operator): WAL append succeeds, then the process
    /// dies BEFORE the sync step — every in-memory [`DurableReceipt`] is dropped.
    /// On restart the owner is reconstructed at its durable phase and
    /// [`recover_and_apply`] replays the PAIRED move from the co-located witness.
    /// Without the reshape the move lived only in the dropped receipt and the
    /// KanbanStep would be lost; here it is reconstructed and applied.
    #[tokio::test]
    async fn a_crash_after_durable_write_replays_the_move_from_the_witness() {
        let sink = FakeSink::new(true);
        // The write lands durably (witness co-located).
        let receipt = persist_cast(
            &sink,
            cast_of(42, 0, KanbanColumn::Planning, KanbanColumn::CognitiveWork),
        )
        .await
        .expect("write landed");
        // ── CRASH ── drop the receipt without ever calling apply_durable_step.
        drop(receipt);

        // Restart: reconstruct the owner at its DURABLE phase (Planning — the
        // step never applied) and recover from what durable storage holds.
        let scanned = sink.scan_witnesses().await.expect("scan");
        let mut o = owner(KanbanColumn::Planning);
        let applied = recover_and_apply(&mut o, &scanned).expect("recovery legal");
        assert_eq!(
            applied.iter().map(|m| m.to).collect::<Vec<_>>(),
            vec![KanbanColumn::CognitiveWork],
            "the pending move was reconstructed from the durable witness and applied",
        );
        assert_eq!(
            o.phase(),
            KanbanColumn::CognitiveWork,
            "the step fired on recovery"
        );

        // Idempotent: recovery re-run after catching up applies nothing (the move
        // is now reflected in the recovered state — stale, skipped, not errored).
        let again = recover_and_apply(&mut o, &scanned).expect("idempotent");
        assert!(again.is_empty(), "second recovery is a no-op");
        assert_eq!(o.phase(), KanbanColumn::CognitiveWork);
    }

    /// FALSIFIER (operator): one owner's durable batch replays its moves in
    /// CAST order, and the globally-interleaved OTHER owner's witnesses are
    /// deinterlaced away (temporal layer-1). The durable log interleaves owner
    /// 42 and owner 99; recovery of 42 applies only 42's chain, in cast order.
    #[tokio::test]
    async fn recovery_replays_one_owners_batch_in_cast_order_ignoring_interleaved_owners() {
        let sink = FakeSink::new(true);
        // Global durable-log order interleaves two owners:
        //   c0: 42 Planning→CognitiveWork
        //   c1: 99 Planning→CognitiveWork   (other owner, interleaved)
        //   c2: 42 CognitiveWork→Evaluation
        for c in [
            cast_of(42, 0, KanbanColumn::Planning, KanbanColumn::CognitiveWork),
            cast_of(99, 1, KanbanColumn::Planning, KanbanColumn::CognitiveWork),
            cast_of(42, 2, KanbanColumn::CognitiveWork, KanbanColumn::Evaluation),
        ] {
            persist_cast(&sink, c).await.expect("landed");
        }
        let scanned = sink.scan_witnesses().await.expect("scan");
        assert_eq!(scanned.len(), 3, "all three witnesses are durable");

        let mut o = owner(KanbanColumn::Planning); // owner 42 at its durable phase
        let applied = recover_and_apply(&mut o, &scanned).expect("legal");
        assert_eq!(
            applied.iter().map(|m| m.to).collect::<Vec<_>>(),
            vec![KanbanColumn::CognitiveWork, KanbanColumn::Evaluation],
            "owner 42's OWN chain, in cast order — 99's interleaved cast deinterlaced away",
        );
        assert_eq!(
            o.phase(),
            KanbanColumn::Evaluation,
            "advanced two local steps"
        );
    }

    /// FALSIFIER 5 (operator): the durable proof is the coordinate, NOT a
    /// `LanceVersion`. A witness is queryable/replayable the instant it lands —
    /// before any base manifest version attaches — so recovery identifies the
    /// latest LOCAL state from the co-located witness cast order, not a dataset
    /// version. Here no dataset version exists at all, yet recovery is exact.
    #[tokio::test]
    async fn recovery_uses_cast_order_not_a_dataset_version() {
        let sink = FakeSink::new(true);
        let receipt = persist_cast(
            &sink,
            cast_of(42, 0, KanbanColumn::Planning, KanbanColumn::CognitiveWork),
        )
        .await
        .expect("landed");
        // The durability proof carries no dataset version — only a WAL/LSM
        // coordinate. (The type has no `DatasetVersion` field to read.)
        assert!(
            receipt.coordinate.wal_entry_position > 0,
            "durable via the WAL coordinate, before any manifest version",
        );
        let scanned = sink.scan_witnesses().await.expect("scan");
        let mut o = owner(KanbanColumn::Planning);
        let applied = recover_and_apply(&mut o, &scanned).expect("legal");
        assert_eq!(
            applied.len(),
            1,
            "latest local state found from cast order alone"
        );
    }

    /// The synchronous single-receipt path surfaces a stale/out-of-order move
    /// LOUDLY ([`PersistError::StalePhase`]) — the counterpart to recovery's
    /// silent stale-skip. A receipt whose move is `Planning→…` applied to an
    /// owner already at `CognitiveWork` is refused, owner untouched.
    #[test]
    fn a_stale_move_on_the_sync_path_is_surfaced_not_silently_reapplied() {
        let mut o = owner(KanbanColumn::CognitiveWork);
        let r = apply_durable_step(&mut o, receipt_for(Some(KanbanColumn::CognitiveWork)));
        assert_eq!(
            r,
            Err(PersistError::StalePhase {
                owner_phase: KanbanColumn::CognitiveWork,
                move_from: KanbanColumn::Planning,
            }),
        );
        assert_eq!(o.phase(), KanbanColumn::CognitiveWork, "owner untouched");
    }
}
