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
//! - A receipt is applied only to **its own** owner ([`PersistError::OwnerMismatch`]),
//!   and only if the move itself was minted for that owner (`mv.mailbox == owner` —
//!   checked at persist time so a cross-owner move never becomes durable).
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
//! ([`DurableWrite::scan_witnesses`]) and re-applies each owner's pending tail.
//! This is a read of what durable storage already holds — **not** a separate
//! ack / confirmation ledger (`E-ACK-ELIMINATED-1`).
//!
//! ## Replay order + idempotence — the DURABLE coordinate, never `CastId`
//!
//! Two distinct keys, both durable, neither the resettable `CastId` counter:
//!
//! - **Replay ORDER = the durable-log position** ([`DurableCoordinate::log_order`],
//!   i.e. `wal_entry_position`). `BatchWriter`'s `CastId` counter **resets to 0 on
//!   every restart**, so two witnesses from different writer lifetimes can share a
//!   `cast_id` — ordering by it is unstable across crashes. The WAL position is
//!   monotonic across the shard's whole life and never resets; `writer_epoch`
//!   fences overlapping writers so positions never collide. One owner writes one
//!   shard ⇒ a total order over that owner's witnesses, valid across crashes.
//!   `cast_id` survives on the witness only as provenance/audit.
//! - **Idempotence = a durable WATERMARK** (`applied_through`), the last durable
//!   coordinate whose move is already reflected in the recovered SoA state.
//!   **Phase equality is NOT a sound idempotence key**: the Rubicon lifecycle is
//!   cyclic (`Planning → CognitiveWork → Evaluation → Plan → Planning`), so after
//!   a completed lap the owner is back at `Planning` and a phase-only check would
//!   replay the whole lap. [`recover_and_apply`] skips every witness at or below
//!   the watermark; above it, the chain must be contiguous (a `from` that does not
//!   match the current phase is a genuine gap/corruption, surfaced as
//!   [`PersistError::StalePhase`], never silently skipped). **The watermark MUST
//!   be persisted with the SoA state** (same generation as the phase it agrees
//!   with) or the cyclic ambiguity returns after a second crash — a single
//!   per-owner scalar high-water mark, not a per-thought ledger.
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
//! the co-located witness's durable position, not from a dataset version that
//! does not exist yet).
//!
//! The concrete impl (a lance-having crate) wires lance 7.0.0's OFFICIAL MemWAL —
//! preferring the high-level `ShardWriter::put` (`enable_memtable + durable_write`:
//! insert into the queryable MemTable, release the lock, then await WAL durability
//! off-lock) over the raw `WalAppender::append` primitive. The witness and the
//! payload are two columns of ONE `RecordBatch` (one generation, atomic), whose
//! buffers are independently owned, so "no owner borrow across I/O" holds without
//! any claim of magic zero-copy. [`DurableWrite::scan_witnesses`] takes a `from`
//! lower bound so recovery reads only the tail after the last applied coordinate,
//! never the whole log.
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

impl std::fmt::Display for WriteFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "durable write did not land: {}", self.0)
    }
}
impl std::error::Error for WriteFailed {}

/// A backend-neutral **durable coordinate** — where the write landed in the
/// durable log. This is NOT a base Lance `DatasetVersion` (that arrives later via
/// MemTable flush + manifest commit); it is the WAL/LSM coordinate that proves
/// durability now. For lance: `shard` = shard `Uuid` (as `u128`),
/// `writer_epoch` + `wal_entry_position` from the `ShardWriter`/`WalAppendResult`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DurableCoordinate {
    /// Opaque shard identity (lance: the shard `Uuid` as `u128`).
    pub shard: u128,
    /// The writer epoch that fenced this append. A new writer lifetime takes a
    /// higher epoch; fencing guarantees two epochs never share a WAL position.
    pub writer_epoch: u64,
    /// The monotonic WAL entry position of this durable append. Monotonic across
    /// the shard's whole life — it does **not** reset on writer restart.
    pub wal_entry_position: u64,
}

impl DurableCoordinate {
    /// The durable-log total-order key for replay + the watermark comparison.
    ///
    /// `wal_entry_position` is monotonic across the shard's whole life and never
    /// resets (unlike `BatchWriter`'s `CastId` counter), and `writer_epoch` fences
    /// overlapping writers so positions never collide. One owner writes one shard,
    /// so this is a total order over that owner's witnesses, valid across crashes.
    #[must_use]
    pub fn log_order(&self) -> u64 {
        self.wal_entry_position
    }
}

/// The durable transition witness — CO-LOCATED with the SoA payload in the same
/// persistence generation so it survives a crash (module doc § Crash-durability).
///
/// Everything needed to re-apply a pending KanbanStep after a restart lives here:
/// WHO (`owner`), the `paired_move` to apply, and the owner-local `cycle`.
/// `cast_id` rides as **provenance/audit only** — it is NOT the replay-order key
/// (it resets across restarts); the durable [`DurableCoordinate`] the sink assigns
/// at append time is, and it is carried by [`LandedWitness`] on the read path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableWitness {
    /// The mailbox this write is on behalf of — the deinterlace grouping key. The
    /// `paired_move` (when present) is minted for this same mailbox — checked at
    /// persist time, so a cross-owner move never becomes durable.
    pub owner: MailboxId,
    /// The cast this write realises — **provenance only** (resets across restarts;
    /// never the replay-order key). The durable coordinate is the order.
    pub cast_id: CastId,
    /// The owner-local cycle at the time of the cast (audit / trajectory).
    pub cycle: u32,
    /// The lifecycle move to apply once this generation is durable. `None` when
    /// the cast carried no lifecycle intent (a durable no-step).
    pub paired_move: Option<KanbanMove>,
}

/// A [`DurableWitness`] read back from the durable log WITH the durable coordinate
/// the sink assigned at append time — the read-path shape [`scan_witnesses`]
/// returns. The coordinate is what orders replay + drives the idempotence
/// watermark (never the resettable `cast_id`), so it implements [`LocalCausalRow`]
/// keyed on [`DurableCoordinate::log_order`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LandedWitness {
    /// Where this witness durably landed — the total-order key across restarts.
    pub coordinate: DurableCoordinate,
    /// The witness itself.
    pub witness: DurableWitness,
}

impl LocalCausalRow for LandedWitness {
    fn owner(&self) -> MailboxId {
        self.witness.owner
    }
    fn cast_seq(&self) -> u64 {
        // The DURABLE log position — monotonic across restarts — NOT `cast_id.0`.
        self.coordinate.log_order()
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

    /// Replay seam: read back durably-landed witnesses (each with its
    /// [`DurableCoordinate`]), in ascending durable-log order, **strictly after**
    /// `from` when given (so recovery reads only the tail past the last applied
    /// coordinate, never the whole log). Crash recovery scans this, splits it per
    /// owner (`temporal` layer-1), and re-applies pending moves in durable order
    /// ([`recover_and_apply`]). Reading the SAME co-located material the receipts
    /// merely referenced — NOT a separate ledger.
    async fn scan_witnesses(
        &self,
        from: Option<DurableCoordinate>,
    ) -> Result<Vec<LandedWitness>, WriteFailed>;
}

/// A **detached** cast envelope — the thinker's report, carrying its OWN payload
/// (independently owned, never a borrow of live owner state) so persistence runs
/// with the owner free. Mirrors what `BatchWriter::cast(on_behalf, moves, payload)`
/// stages; at drain the descriptor is resolved to `payload` bytes.
pub struct PersistCast {
    /// The mailbox this write is on behalf of.
    pub owner: MailboxId,
    /// The cast this write realises (from `BatchWriter::cast`). Rides into the
    /// [`DurableWitness`] as provenance (NOT the replay-order key).
    pub cast_id: CastId,
    /// The owner-local cycle at cast time — rides into the witness (audit).
    pub cycle: u32,
    /// The lifecycle move the thought cast with this write (its `to` is applied
    /// post-durability). `None` when the cast carries no lifecycle intent. When
    /// present, `paired_move.mailbox` MUST equal `owner` (checked in
    /// [`persist_cast`]).
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
    /// A receipt (or the move it carries) was applied to the wrong owner — refused
    /// (a receipt only advances the mailbox it was minted for; write-on-behalf
    /// never crosses owners). Also raised at persist time when
    /// `paired_move.mailbox != owner`, so a cross-owner move never becomes durable.
    OwnerMismatch {
        receipt_owner: MailboxId,
        applying_to: MailboxId,
    },
    /// The paired move's `from` does not match the owner's current phase. On the
    /// synchronous single-receipt path ([`apply_durable_step`]) this is a stale /
    /// out-of-order apply — surfaced, and SAFE to drop because the move is durable
    /// and [`recover_and_apply`] will replay it. In recovery it means an
    /// above-watermark witness does not chain — a genuine gap/corruption.
    StalePhase {
        owner_phase: KanbanColumn,
        move_from: KanbanColumn,
    },
}

impl std::fmt::Display for PersistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Write(e) => write!(f, "{e}"),
            Self::Illegal(e) => {
                write!(f, "illegal Rubicon transition {:?} -> {:?}", e.from, e.to)
            }
            Self::OwnerMismatch {
                receipt_owner,
                applying_to,
            } => write!(
                f,
                "move for mailbox {receipt_owner} applied to mailbox {applying_to}"
            ),
            Self::StalePhase {
                owner_phase,
                move_from,
            } => write!(
                f,
                "stale move: owner at {owner_phase:?}, move.from {move_from:?}"
            ),
        }
    }
}
impl std::error::Error for PersistError {}

/// **Phase 1 — async persistence, NO owner borrow.** Form the crash-durable
/// [`DurableWitness`] and append it CO-LOCATED with the cast's payload in one
/// generation; on success return a [`DurableReceipt`] (which merely references
/// that durable material), on failure a [`PersistError::Write`] and **no
/// receipt**. `O` (the owner) does not appear in this signature at all — it is
/// never borrowed across the WAL / object-store await, so the owner stays free
/// while durability runs.
///
/// Rejects a `paired_move` minted for a different mailbox
/// ([`PersistError::OwnerMismatch`]) BEFORE the append, so a cross-owner move
/// never becomes durable (the write-on-behalf invariant, enforced at the source).
pub async fn persist_cast<W: DurableWrite>(
    sink: &W,
    cast: PersistCast,
) -> Result<DurableReceipt, PersistError> {
    if let Some(mv) = cast.paired_move {
        if mv.mailbox != cast.owner {
            return Err(PersistError::OwnerMismatch {
                receipt_owner: mv.mailbox,
                applying_to: cast.owner,
            });
        }
    }
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
///
/// **Reordered / stale receipts are safe to drop.** `DurableWrite` explicitly
/// supports concurrent drains, so a later append can complete first; applying its
/// receipt while the owner is still at the earlier phase yields
/// [`PersistError::StalePhase`]. Dropping it loses NOTHING — the move is durable,
/// and [`recover_and_apply`] (or a re-drive that re-reads the durable tail)
/// replays it in durable order. The sync path is the fast happy path; the durable
/// witness is the correctness backstop.
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
            // Defense in depth: the move must also be minted for this owner (the
            // envelope check above only compares receipt.owner).
            if mv.mailbox != owner.mailbox_id() {
                return Err(PersistError::OwnerMismatch {
                    receipt_owner: mv.mailbox,
                    applying_to: owner.mailbox_id(),
                });
            }
            // The move's `from` must match the owner's current phase: on the
            // synchronous path a mismatch is a stale / out-of-order apply and is
            // surfaced (safe to drop — the durable witness replays it).
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

/// The result of a [`recover_and_apply`] pass: the moves actually applied, and the
/// new durable **watermark** the caller must persist WITH the recovered SoA state
/// so the next recovery is idempotent (skips everything at or below it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recovered {
    /// The paired moves applied this pass, in durable order.
    pub applied: Vec<KanbanMove>,
    /// The highest durable coordinate now accounted for (the new watermark), or
    /// the input `applied_through` when nothing was applied. Persist it with the
    /// owner's phase (same generation) — see the module doc § Replay order.
    pub watermark: Option<DurableCoordinate>,
}

/// **Crash recovery.** Given landed witnesses read back via
/// [`DurableWrite::scan_witnesses`] (a globally-interleaved stream from every
/// owner), an `owner` reconstructed at its durable phase, and the durable
/// `applied_through` watermark persisted alongside that phase, re-apply the
/// owner's PENDING tail in **durable-log order** and return the applied moves plus
/// the new watermark.
///
/// - `temporal` layer-1 ([`local_trajectory_of`]) deinterlaces the global stream
///   to this owner's own chain, ordered by the durable [`DurableCoordinate`] (NOT
///   the resettable `cast_id`).
/// - Every witness at or below `applied_through` is **already reflected in the
///   recovered SoA state** and is skipped. This watermark — not phase equality —
///   is the idempotence key: the Rubicon lifecycle is cyclic, so after a completed
///   lap the owner is back at `Planning`, and a phase-only check would replay the
///   whole lap (`E-…-NOT-IN-MEMORY-ONLY-1`).
/// - Above the watermark the chain must be contiguous: a move whose `from` does
///   not match the owner's current phase is a genuine gap/corruption and is
///   surfaced ([`PersistError::StalePhase`]); a matching-`from` move that is not a
///   legal Rubicon edge is [`PersistError::Illegal`].
///
/// **On error the owner is left mid-chain** (every earlier move in this pass is
/// already applied). The returned `Recovered` is only produced on full success;
/// re-drive from the persisted watermark after resolving the corruption.
///
/// This is why the paired move MUST be co-located in the durable generation: the
/// witnesses are the only reason a KanbanStep survives a crash between the WAL
/// append and [`apply_durable_step`]. No witnesses ⇒ nothing to replay ⇒ the step
/// is lost (the gap this reshape closes).
pub fn recover_and_apply<O: MailboxSoaOwner>(
    owner: &mut O,
    landed: &[LandedWitness],
    applied_through: Option<DurableCoordinate>,
) -> Result<Recovered, PersistError> {
    let chain = local_trajectory_of(landed, owner.mailbox_id());
    let hw = applied_through.map(|c| c.log_order());
    let mut applied = Vec::new();
    let mut watermark = applied_through;
    for lw in chain {
        // Skip everything at or below the durable watermark — already reflected in
        // the recovered SoA state (the cyclic-safe idempotence key).
        if hw.is_some_and(|hw| lw.coordinate.log_order() <= hw) {
            continue;
        }
        match lw.witness.paired_move {
            None => {
                // A durable no-step still advances the watermark past this
                // generation (it is accounted for; nothing to apply).
                watermark = Some(lw.coordinate);
            }
            Some(mv) => {
                // Above the watermark the chain MUST be contiguous: a non-matching
                // `from` is a gap/corruption, not a benign already-applied move.
                if mv.from != owner.phase() {
                    return Err(PersistError::StalePhase {
                        owner_phase: owner.phase(),
                        move_from: mv.from,
                    });
                }
                let step = owner
                    .try_advance_phase(mv.to)
                    .map_err(PersistError::Illegal)?;
                applied.push(step);
                watermark = Some(lw.coordinate);
            }
        }
    }
    Ok(Recovered { applied, watermark })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::collapse_gate::MailboxId as MId;
    use lance_graph_contract::kanban::{ExecTarget, KanbanColumn};
    use lance_graph_contract::soa_view::{MailboxSoaOwner, MailboxSoaView};
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Mutex;

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
    /// which RECORDS every witness it lands (with the durable coordinate it
    /// assigns) so [`scan_witnesses`] can replay them — this is what makes the
    /// crash-recovery falsifier real: the durable log (`landed`) outlives the
    /// in-memory receipts. `Sync` (Mutex + Atomic) so the documented concurrent
    /// drain can actually be exercised.
    struct FakeSink {
        succeed: bool,
        calls: AtomicU32,
        /// The durable log: (coordinate, witness), assigned a monotonic WAL
        /// position that does NOT reset across simulated writer lifetimes.
        landed: Mutex<Vec<LandedWitness>>,
    }
    impl FakeSink {
        fn new(succeed: bool) -> Self {
            Self {
                succeed,
                calls: AtomicU32::new(0),
                landed: Mutex::new(Vec::new()),
            }
        }
        fn calls(&self) -> u32 {
            self.calls.load(Ordering::SeqCst)
        }
    }
    impl DurableWrite for FakeSink {
        async fn append(
            &self,
            witness: &DurableWitness,
            _payload: &[u8],
        ) -> Result<DurableCoordinate, WriteFailed> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if !self.succeed {
                // The negative half: a fenced WAL lands NOTHING — no witness, so
                // nothing to replay, so no move and no step.
                return Err(WriteFailed("wal fenced".into()));
            }
            let mut log = self.landed.lock().unwrap();
            // WAL position: monotonic over the log's whole life, 1-based.
            let coordinate = DurableCoordinate {
                shard: 0xABCD,
                writer_epoch: 1,
                wal_entry_position: log.len() as u64 + 1,
            };
            log.push(LandedWitness {
                coordinate,
                witness: witness.clone(),
            });
            Ok(coordinate)
        }
        async fn scan_witnesses(
            &self,
            from: Option<DurableCoordinate>,
        ) -> Result<Vec<LandedWitness>, WriteFailed> {
            let lb = from.map(|c| c.log_order());
            Ok(self
                .landed
                .lock()
                .unwrap()
                .iter()
                .filter(|lw| lb.is_none_or(|lb| lw.coordinate.log_order() > lb))
                .cloned()
                .collect())
        }
    }

    fn owner(phase: KanbanColumn) -> FakeOwner {
        owner_id(42, phase)
    }
    fn owner_id(id: MId, phase: KanbanColumn) -> FakeOwner {
        FakeOwner {
            id,
            phase,
            cycle: 5,
        }
    }
    fn mv(owner: MId, from: KanbanColumn, to: KanbanColumn) -> KanbanMove {
        KanbanMove {
            mailbox: owner,
            from,
            to,
            witness_chain_position: 0,
            exec: ExecTarget::Elixir,
        }
    }
    fn cast(paired_to: Option<KanbanColumn>) -> PersistCast {
        cast_of(42, 0, KanbanColumn::Planning, paired_to)
    }
    fn cast_of(
        owner: MId,
        cast_id: u64,
        from: KanbanColumn,
        paired_to: Option<KanbanColumn>,
    ) -> PersistCast {
        PersistCast {
            owner,
            cast_id: CastId(cast_id),
            cycle: 0,
            paired_move: paired_to.map(|to| mv(owner, from, to)),
            payload: vec![cast_id as u8],
        }
    }
    fn receipt_from(from: KanbanColumn, paired_to: Option<KanbanColumn>) -> DurableReceipt {
        DurableReceipt {
            owner: 42,
            cast_id: CastId(0),
            paired_move: paired_to.map(|to| mv(42, from, to)),
            coordinate: DurableCoordinate {
                shard: 0xABCD,
                writer_epoch: 1,
                wal_entry_position: 7,
            },
        }
    }
    fn receipt_for(paired_to: Option<KanbanColumn>) -> DurableReceipt {
        receipt_from(KanbanColumn::Planning, paired_to)
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
        assert_eq!(sink.calls(), 1, "the write was attempted once");
        // The witness LANDED durably (co-located) — not only in the receipt.
        let scanned = sink.scan_witnesses(None).await.expect("scan");
        assert_eq!(scanned.len(), 1, "one witness durably co-located");
        assert_eq!(
            scanned[0].witness.paired_move.map(|m| m.to),
            Some(KanbanColumn::CognitiveWork),
            "the paired move is in the DURABLE witness, not only the in-memory receipt",
        );
    }

    #[tokio::test]
    async fn a_failed_write_yields_no_receipt() {
        let sink = FakeSink::new(false);
        let r = persist_cast(&sink, cast(Some(KanbanColumn::CognitiveWork))).await;
        assert_eq!(
            r,
            Err(PersistError::Write(WriteFailed("wal fenced".into())))
        );
        assert_eq!(sink.calls(), 1, "the write was attempted");
    }

    #[tokio::test]
    async fn a_fenced_write_lands_no_witness_so_nothing_can_be_replayed() {
        // Negative half, at the durable layer: a failed append records NO witness,
        // so a subsequent crash-recovery scan finds nothing to replay ⇒ no move.
        let sink = FakeSink::new(false);
        let _ = persist_cast(&sink, cast(Some(KanbanColumn::CognitiveWork))).await;
        assert_eq!(
            sink.scan_witnesses(None).await.expect("scan").len(),
            0,
            "a fenced WAL leaves no durable witness — no move, no step",
        );
    }

    #[tokio::test]
    async fn a_cross_owner_paired_move_is_rejected_before_it_becomes_durable() {
        // A move minted for mailbox 99 riding in a cast on behalf of 42 must be
        // refused at persist time — a cross-owner move never lands durably.
        let sink = FakeSink::new(true);
        let cast = PersistCast {
            owner: 42,
            cast_id: CastId(0),
            cycle: 0,
            paired_move: Some(mv(99, KanbanColumn::Planning, KanbanColumn::CognitiveWork)),
            payload: vec![],
        };
        assert_eq!(
            persist_cast(&sink, cast).await,
            Err(PersistError::OwnerMismatch {
                receipt_owner: 99,
                applying_to: 42
            }),
        );
        assert_eq!(sink.calls(), 0, "the write was never attempted");
        assert_eq!(
            sink.scan_witnesses(None).await.expect("scan").len(),
            0,
            "nothing durable",
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_drains_both_land_durably() {
        // The DurableWrite doc claims &self is shared because many casts drain
        // concurrently. Exercise it: two persist_cast futures on a shared sink.
        let sink = std::sync::Arc::new(FakeSink::new(true));
        let s1 = sink.clone();
        let s2 = sink.clone();
        let (a, b) = tokio::join!(
            async move {
                persist_cast(
                    &*s1,
                    cast_of(
                        42,
                        0,
                        KanbanColumn::Planning,
                        Some(KanbanColumn::CognitiveWork),
                    ),
                )
                .await
            },
            async move {
                persist_cast(
                    &*s2,
                    cast_of(
                        43,
                        0,
                        KanbanColumn::Planning,
                        Some(KanbanColumn::CognitiveWork),
                    ),
                )
                .await
            },
        );
        assert!(a.is_ok() && b.is_ok(), "both drains landed");
        assert_eq!(sink.calls(), 2);
        assert_eq!(
            sink.scan_witnesses(None).await.expect("scan").len(),
            2,
            "both witnesses durably recorded under concurrent drain",
        );
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
        let mut other = owner_id(99, KanbanColumn::Planning);
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

    #[test]
    fn a_stale_move_on_the_sync_path_is_surfaced_not_silently_reapplied() {
        // The sync path surfaces a stale/out-of-order move LOUDLY (safe to drop —
        // the durable witness replays it). A `Planning→…` move applied to an owner
        // already at `CognitiveWork` is refused, owner untouched.
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

    // ── Crash recovery: the co-located witness replays after the receipt is lost ─

    /// Persist a chain of casts through the sink (they land durably) and return the
    /// scanned landed witnesses — the "durable log" a restart reads back.
    async fn persist_and_scan(sink: &FakeSink, casts: Vec<PersistCast>) -> Vec<LandedWitness> {
        for c in casts {
            persist_cast(sink, c).await.expect("landed");
        }
        sink.scan_witnesses(None).await.expect("scan")
    }

    /// THE crash falsifier (operator): WAL append succeeds, then the process dies
    /// BEFORE the sync step — every in-memory [`DurableReceipt`] is dropped. On
    /// restart the owner is reconstructed at its durable phase (+ its durable
    /// watermark) and [`recover_and_apply`] replays the PAIRED move from the
    /// co-located witness. Idempotent: a second recovery from the returned
    /// watermark applies nothing.
    #[tokio::test]
    async fn a_crash_after_durable_write_replays_the_move_from_the_witness() {
        let sink = FakeSink::new(true);
        let receipt = persist_cast(
            &sink,
            cast_of(
                42,
                0,
                KanbanColumn::Planning,
                Some(KanbanColumn::CognitiveWork),
            ),
        )
        .await
        .expect("write landed");
        drop(receipt); // ── CRASH ── never called apply_durable_step.

        let landed = sink.scan_witnesses(None).await.expect("scan");
        let mut o = owner(KanbanColumn::Planning); // durable phase; no watermark yet
        let rec = recover_and_apply(&mut o, &landed, None).expect("recovery legal");
        assert_eq!(
            rec.applied.iter().map(|m| m.to).collect::<Vec<_>>(),
            vec![KanbanColumn::CognitiveWork],
            "the pending move was reconstructed from the durable witness and applied",
        );
        assert_eq!(
            o.phase(),
            KanbanColumn::CognitiveWork,
            "the step fired on recovery"
        );
        assert!(rec.watermark.is_some(), "a new watermark to persist");

        // Idempotent: recovery re-run FROM THE WATERMARK applies nothing.
        let again = recover_and_apply(&mut o, &landed, rec.watermark).expect("idempotent");
        assert!(again.applied.is_empty(), "second recovery is a no-op");
        assert_eq!(o.phase(), KanbanColumn::CognitiveWork);
    }

    /// FALSIFIER (operator): one owner's durable batch replays in DURABLE order,
    /// and the globally-interleaved OTHER owner's witnesses are deinterlaced away
    /// (temporal layer-1). A durable no-step interleaved into the chain is skipped
    /// while the following move still applies.
    #[tokio::test]
    async fn recovery_replays_one_owners_batch_in_order_skipping_no_steps_and_other_owners() {
        let sink = FakeSink::new(true);
        // Durable-log order interleaves two owners + a no-step for 42:
        //   c0: 42 Planning→CognitiveWork
        //   c1: 99 Planning→CognitiveWork   (other owner, interleaved)
        //   c2: 42 no-step (paired_move None)
        //   c3: 42 CognitiveWork→Evaluation
        let landed = persist_and_scan(
            &sink,
            vec![
                cast_of(
                    42,
                    0,
                    KanbanColumn::Planning,
                    Some(KanbanColumn::CognitiveWork),
                ),
                cast_of(
                    99,
                    0,
                    KanbanColumn::Planning,
                    Some(KanbanColumn::CognitiveWork),
                ),
                cast_of(42, 1, KanbanColumn::Planning, None),
                cast_of(
                    42,
                    2,
                    KanbanColumn::CognitiveWork,
                    Some(KanbanColumn::Evaluation),
                ),
            ],
        )
        .await;
        assert_eq!(landed.len(), 4, "all four witnesses are durable");

        let mut o = owner(KanbanColumn::Planning); // owner 42 at its durable phase
        let rec = recover_and_apply(&mut o, &landed, None).expect("legal");
        assert_eq!(
            rec.applied.iter().map(|m| m.to).collect::<Vec<_>>(),
            vec![KanbanColumn::CognitiveWork, KanbanColumn::Evaluation],
            "owner 42's OWN chain, in durable order — 99's cast + the no-step skipped",
        );
        assert_eq!(
            o.phase(),
            KanbanColumn::Evaluation,
            "advanced two local steps"
        );
    }

    /// THE cyclic idempotence falsifier (Codex/CodeRabbit Critical). A full lap
    /// `Planning → CognitiveWork → Evaluation → Plan → Planning` leaves the owner
    /// back at `Planning`. Phase equality alone would replay the whole lap; the
    /// durable WATERMARK makes the second recovery a no-op. The negative control
    /// (recovering WITHOUT the persisted watermark) proves the watermark is
    /// load-bearing by reproducing the double-lap.
    #[tokio::test]
    async fn cyclic_recovery_is_idempotent_only_with_the_durable_watermark() {
        // Precondition: the lap is a real cycle in the Rubicon graph.
        assert_eq!(KanbanColumn::Plan.next_phases(), &[KanbanColumn::Planning]);
        let sink = FakeSink::new(true);
        let landed = persist_and_scan(
            &sink,
            vec![
                cast_of(
                    42,
                    0,
                    KanbanColumn::Planning,
                    Some(KanbanColumn::CognitiveWork),
                ),
                cast_of(
                    42,
                    1,
                    KanbanColumn::CognitiveWork,
                    Some(KanbanColumn::Evaluation),
                ),
                cast_of(42, 2, KanbanColumn::Evaluation, Some(KanbanColumn::Plan)),
                cast_of(42, 3, KanbanColumn::Plan, Some(KanbanColumn::Planning)),
            ],
        )
        .await;

        let mut o = owner(KanbanColumn::Planning);
        let rec = recover_and_apply(&mut o, &landed, None).expect("legal");
        assert_eq!(rec.applied.len(), 4, "the whole lap applied once");
        assert_eq!(
            o.phase(),
            KanbanColumn::Planning,
            "back at Planning after a lap"
        );

        // WITH the persisted watermark: second recovery skips the entire lap.
        let good = recover_and_apply(&mut o, &landed, rec.watermark).expect("legal");
        assert!(
            good.applied.is_empty(),
            "watermark makes cyclic recovery idempotent",
        );
        assert_eq!(o.phase(), KanbanColumn::Planning);

        // NEGATIVE CONTROL: WITHOUT the watermark (as if it were not persisted),
        // phase equality replays the whole lap — the bug the watermark fixes.
        let bug = recover_and_apply(&mut o, &landed, None).expect("legal");
        assert_eq!(
            bug.applied.len(),
            4,
            "without the durable watermark the cyclic lap is replayed — watermark is load-bearing",
        );
    }

    /// FALSIFIER (Bugbot High): `CastId` resets across writer restarts, so two
    /// lifetimes can share `cast_id` 0. Replay MUST order by the durable WAL
    /// position (which does not reset), not `cast_id`. Two lifetimes each cast
    /// `cast_id 0`, at distinct WAL positions, forming one owner's chain — recovery
    /// applies them in durable order regardless of the colliding cast ids.
    #[tokio::test]
    async fn recovery_orders_by_durable_position_not_the_resettable_cast_id() {
        let sink = FakeSink::new(true);
        // Lifetime 1 (cast_id 0) then lifetime 2 (cast_id 0 again) — the counter
        // reset. Distinct WAL positions (1, 2) come from the durable log.
        let landed = persist_and_scan(
            &sink,
            vec![
                cast_of(
                    42,
                    0,
                    KanbanColumn::Planning,
                    Some(KanbanColumn::CognitiveWork),
                ),
                cast_of(
                    42,
                    0,
                    KanbanColumn::CognitiveWork,
                    Some(KanbanColumn::Evaluation),
                ),
            ],
        )
        .await;
        assert_eq!(
            landed
                .iter()
                .map(|l| l.witness.cast_id.0)
                .collect::<Vec<_>>(),
            vec![0, 0],
            "the cast ids collide (counter reset across lifetimes)",
        );
        assert_eq!(
            landed
                .iter()
                .map(|l| l.coordinate.wal_entry_position)
                .collect::<Vec<_>>(),
            vec![1, 2],
            "but the durable WAL positions are distinct + monotonic",
        );

        let mut o = owner(KanbanColumn::Planning);
        let rec = recover_and_apply(&mut o, &landed, None).expect("legal");
        assert_eq!(
            rec.applied.iter().map(|m| m.to).collect::<Vec<_>>(),
            vec![KanbanColumn::CognitiveWork, KanbanColumn::Evaluation],
            "ordered by durable position despite the colliding cast ids",
        );
        assert_eq!(o.phase(), KanbanColumn::Evaluation);
    }

    /// FALSIFIER 5 (operator): the durable proof is the coordinate, NOT a
    /// `LanceVersion`. A witness is replayable the instant it lands — before any
    /// base manifest version attaches — so recovery identifies the latest LOCAL
    /// state from the co-located witness's durable position, not a dataset version.
    #[tokio::test]
    async fn recovery_uses_durable_position_not_a_dataset_version() {
        let sink = FakeSink::new(true);
        let receipt = persist_cast(
            &sink,
            cast_of(
                42,
                0,
                KanbanColumn::Planning,
                Some(KanbanColumn::CognitiveWork),
            ),
        )
        .await
        .expect("landed");
        // The durability proof carries no dataset version — only a WAL/LSM
        // coordinate. (The type has no `DatasetVersion` field to read.)
        assert!(
            receipt.coordinate.log_order() > 0,
            "durable via the WAL coordinate, before any manifest version",
        );
        let landed = sink.scan_witnesses(None).await.expect("scan");
        let mut o = owner(KanbanColumn::Planning);
        let rec = recover_and_apply(&mut o, &landed, None).expect("legal");
        assert_eq!(
            rec.applied.len(),
            1,
            "latest local state found from the durable position alone"
        );
    }

    /// The bounded scan seam: `scan_witnesses(from)` returns only the tail past a
    /// coordinate — recovery need not read the whole log.
    #[tokio::test]
    async fn scan_from_a_coordinate_returns_only_the_tail() {
        let sink = FakeSink::new(true);
        let _ = persist_and_scan(
            &sink,
            vec![
                cast_of(
                    42,
                    0,
                    KanbanColumn::Planning,
                    Some(KanbanColumn::CognitiveWork),
                ),
                cast_of(
                    42,
                    1,
                    KanbanColumn::CognitiveWork,
                    Some(KanbanColumn::Evaluation),
                ),
                cast_of(42, 2, KanbanColumn::Evaluation, Some(KanbanColumn::Commit)),
            ],
        )
        .await;
        let after_first = DurableCoordinate {
            shard: 0xABCD,
            writer_epoch: 1,
            wal_entry_position: 1,
        };
        let tail = sink.scan_witnesses(Some(after_first)).await.expect("scan");
        assert_eq!(
            tail.iter()
                .map(|l| l.coordinate.wal_entry_position)
                .collect::<Vec<_>>(),
            vec![2, 3],
            "only witnesses strictly after the given coordinate",
        );
    }
}
