//! The D-MBX-A6 persistence sink — **one WAL write per cycle** (the amortization).
//!
//! This is the POST-write half of the fire-and-forget flow (pre-write half =
//! [`crate::owner_adapter`]). Its ONLY concerns are **storage** and that 64k
//! concurrent thoughts hit **no physical or epistemic race condition** (operator
//! ruling). It does not model semantics: semantic causality already lives in the
//! witness substrate (`causal_witness`, AriGraph SPO, NARS inference direction) —
//! this layer never invents a second causality graph.
//!
//! > The sweep globally aligns **durability**; `temporal.rs` locally aligns
//! > **order** before durability. One semantic generation = one globally-aligned
//! > 64k-row **cycle** that seals into exactly one [`DatasetVersion`].
//!
//! ## Three levels — cast ⊂ chunk ⊂ cycle
//!
//! - **cast** — one logical thought staged into the open cycle (no WAL write).
//! - **chunk** — a disjoint / row-indexed slice of the cycle's final image.
//! - **cycle** — the 64k-row frame. Its **seal is ONE WAL write** → one version.
//!
//! Modelling the durable unit as one *cast* (a WAL write per thought) is the
//! anti-pattern this file corrects: it would pay WAL header + durability-wait +
//! version-metadata 64k times. Here the WAL is amortized — many logical thoughts
//! → one durable boundary → one published version.
//!
//! ## Physical race safety — WRITE-SIDE order, not read-time repair
//!
//! 64k thoughts run concurrently and finish in ARBITRARY CPU order. In
//! [`DetachedCycleBatch::freeze`], the casts are stable-ordered by their existing
//! `stream_position` via [`order_cycle_stably`] BEFORE the single WAL append — so
//! the durable image is already canonical. [`WalSink::scan_sealed`] returns rows in
//! that stored order and **never sorts**: completion order is fixed to canonical
//! order at WRITE time, not repaired on every read (which would lose the whole
//! amortization and hand every downstream reader the wrong weave first). The
//! ordering is write-path WAL preparation, so it lives here — NOT in `temporal.rs`,
//! which owns query-time reading.
//!
//! ## Epistemic race safety — the sealed read horizon
//!
//! Every thought in an open cycle reads only the sealed predecessor
//! (`frame.base_version` = `Vn`); the seal publishes `Vn+1` once. An open cycle's
//! output is **never visible as `Vn` input** — [`WalSink::scan_sealed`] excludes
//! any unsealed cycle. So no thought can read a concurrent sibling's in-flight
//! result as though it were sealed history (`read Vn / write Vn+1`).
//!
//! ## Coalescing — fold ordered updates into owned rows, then chunk
//!
//! The final cycle image is the ordered per-row fold of the staged updates (a
//! later stream position overwrites the same row); chunks are disjoint / mergeable
//! slices of that final image, NOT successive whole-image replacements.
//!
//! ## Scope boundary — storage only, no semantic / rung types minted here
//!
//! This layer owns storage + race-freedom, nothing else. It does NOT mint (or
//! carry) semantic-witness, ancestry, awareness, branch, rung, or temporal-policy
//! types — those live upstream (the semantic witness substrate) and downstream
//! (`temporal.rs` owns *query-time* reader-horizon / version reading). The
//! write-side ordering here is generic over a caller-supplied canonical key
//! ([`order_cycle_stably`]); the rung a cycle is aligned at is decided by whoever
//! schedules it, not modelled in [`CycleFrame`].
//!
//! ## What the tests prove — CONTRACT probes, NOT crash/WAL integration (honest)
//!
//! The `#[cfg(test)]` `FakeWalSink` is in-process maps, not a WAL/Lance table. The
//! tests are **storage/race CONTRACT probes**: one WAL write per cycle, write-side
//! order under scrambled completion, per-row coalescing, no unsealed visibility,
//! sealed read horizon, recovery idempotence. They do **NOT** prove real
//! durability (no MemWAL, restart, atomic `RecordBatch`, or manifest).
//! **`compile+test green ≠ storage proven`** (the Ladybug lesson). **This module
//! builds NO concrete Lance sink.**

use lance_graph_contract::kanban::{KanbanColumn, KanbanMove, RubiconTransitionError};
use lance_graph_contract::scheduler::DatasetVersion;
use lance_graph_contract::soa_view::MailboxSoaOwner;

pub use lance_graph_contract::collapse_gate::MailboxId;

/// **Write-side stable ordering** — the loom before the WAL. Stable-order
/// concurrently-produced rows by a caller-supplied canonical key BEFORE the single
/// WAL append, so completion order never becomes storage order (physical race) and
/// no read-time repair is ever needed. Generic over the key until the repository
/// proves a canonical key type; equal keys keep arrival order (stable).
///
/// **Note (the dense-key nuance):** when the key is already a dense slot index into
/// the cycle image, a stable scatter into predetermined positions is cheaper than a
/// general `O(n log n)` sort on the 64k path — that choice belongs beside the cycle
/// batch (here), not in `temporal.rs`. This general sort is the fallback form.
///
/// This lives with the persistence cycle, NOT in `temporal.rs`: that module owns
/// query-time reader-horizon / version reading; this is write-path WAL preparation.
pub fn order_cycle_stably<T, K: Ord>(rows: &mut [T], key: impl FnMut(&T) -> K) {
    rows.sort_by_key(key);
}

/// A cycle — one globally-aligned 64k-row sweep whose commit is ONE WAL append and
/// ONE [`DatasetVersion`]. A boring scheduling identity (monotonic), not a
/// semantic object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CycleId(pub u64);

/// A cycle's storage identity: which cycle, and the sealed predecessor its thoughts
/// read (`Vn` — the epistemic read horizon; the commit publishes its successor).
/// Storage-only: it carries NO rung / projection / branch / semantic tags (those
/// are decided upstream, never minted in this layer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CycleFrame {
    pub cycle: CycleId,
    /// The sealed predecessor every thought in this cycle reads (`Vn`).
    pub base_version: DatasetVersion,
}

impl CycleFrame {
    /// A cycle reading `base_version` (`Vn`) as its sealed predecessor.
    #[must_use]
    pub fn new(cycle: CycleId, base_version: DatasetVersion) -> Self {
        Self {
            cycle,
            base_version,
        }
    }
}

/// A persistence **landing** record — WHERE a staged thought lands in the cycle.
/// Boring by design: it says where, never why. Semantic causality (ancestry,
/// evidence, revision) stays in the witness substrate — this record has no
/// `basis`. `stream_position` is the caller's EXISTING canonical order key, the
/// sole write-side ordering input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SweepSlot {
    pub cycle: CycleId,
    /// The existing canonical (textual/stream) order key — the write-side
    /// deinterlace input. NOT a new coordinate.
    pub stream_position: u64,
    /// The mailbox this landing is on behalf of.
    pub owner: MailboxId,
    /// The SoA row this update writes (coalescing folds same-row updates).
    pub row: u64,
    /// The lifecycle move this thought cast; applied post-SEAL only. `None` = a
    /// no-step landing.
    pub paired_move: Option<KanbanMove>,
    /// The update payload — a descriptor (a `NodeRowPacket` slice in production;
    /// bytes here). Never a semantic witness.
    pub payload: Vec<u8>,
}

/// A landing read back from a SEALED cycle — its slot plus the version its cycle
/// sealed into (shared by every landing of that cycle). Only ever produced for
/// SEALED cycles; returned in the stored (already canonical) order, NEVER re-sorted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LandedSlot {
    /// The version the cycle sealed into (the cheap coarse timeline).
    pub version: DatasetVersion,
    pub slot: SweepSlot,
}

/// A durable write that did not land (the cycle seal failed / was fenced).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteFailed(pub String);

impl std::fmt::Display for WriteFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cycle seal did not land: {}", self.0)
    }
}
impl std::error::Error for WriteFailed {}

/// A **detached, frozen, already-deinterlaced** cycle image — the loom's finished
/// fabric, ready for the single WAL append. Built by [`DetachedCycleBatch::freeze`]
/// (temporal deinterlace + per-row coalesce) BEFORE [`WalSink::commit_cycle`], so
/// the durable operation is a pure atomic append of an already-coherent frame.
/// Detached = a snapshot, never a borrow of live mutable SoA.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetachedCycleBatch {
    pub frame: CycleFrame,
    /// Landings in canonical `stream_position` order (the temporal deinterlace
    /// already ran — the WAL never sees worker-completion order).
    pub landings: Vec<SweepSlot>,
    /// The coalesced final image: `row -> last payload in stream order`.
    pub image: std::collections::BTreeMap<u64, Vec<u8>>,
}

impl DetachedCycleBatch {
    /// Freeze concurrently-produced casts into the ordered, coalesced cycle image:
    /// (1) [`order_cycle_stably`] stable-orders by `stream_position` —
    /// completion order never becomes storage order (physical race); (2) fold
    /// same-row updates into owned rows (later stream position wins). The result is
    /// detached from any live SoA and ready for exactly one WAL append.
    #[must_use]
    pub fn freeze(frame: CycleFrame, mut casts: Vec<SweepSlot>) -> Self {
        order_cycle_stably(&mut casts, |s| s.stream_position);
        let mut image = std::collections::BTreeMap::new();
        for s in &casts {
            image.insert(s.row, s.payload.clone());
        }
        Self {
            frame,
            landings: casts,
            image,
        }
    }
}

/// Why a persist / seal / step operation produced no lifecycle advance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistError {
    /// The cycle seal did not land — nothing published, so no step.
    Write(WriteFailed),
    /// The paired move is not a legal Rubicon edge from the owner's current phase.
    Illegal(RubiconTransitionError),
    /// A move minted for a different mailbox than the landing's owner — refused so
    /// a cross-owner move never becomes durable (write-on-behalf never crosses
    /// owners), and again as a recovery corruption guard.
    OwnerMismatch {
        move_owner: MailboxId,
        landing_owner: MailboxId,
    },
    /// An above-watermark landing's move `from` does not match the owner's current
    /// phase — a gap/corruption in the sealed replay chain (phase equality is a
    /// corruption guard, NOT the idempotence key — the watermark is).
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
                move_owner,
                landing_owner,
            } => write!(
                f,
                "move for mailbox {move_owner} on a landing owned by {landing_owner}"
            ),
            Self::StalePhase {
                owner_phase,
                move_from,
            } => write!(
                f,
                "stale/corrupt replay: owner at {owner_phase:?}, move.from {move_from:?}"
            ),
        }
    }
}
impl std::error::Error for PersistError {}

/// The WAL seam — **one durable append per cycle**. [`commit_cycle`](WalSink::commit_cycle)
/// takes an ALREADY-deinterlaced, ALREADY-frozen [`DetachedCycleBatch`] (the loom
/// ran before the WAL) and performs the single atomic append → one
/// [`DatasetVersion`], visible all-or-nothing. Reads return committed landings
/// only, in the STORED canonical order — this seam NEVER sorts (order is a
/// write-side property, fixed before the append).
///
/// `&self` (shared) and **no owner borrow**: the persistence path never holds the
/// SoA owner across I/O.
#[allow(async_fn_in_trait)]
pub trait WalSink {
    /// The SINGLE amortized WAL append for a whole cycle: commit `batch` (already
    /// ordered + coalesced) against the sealed predecessor `base`, publishing
    /// exactly one new [`DatasetVersion`] atomically. This is the ONLY durable op —
    /// 64k thoughts → one append, not 64k appends. `async` because the concrete WAL
    /// append is async.
    async fn commit_cycle(
        &self,
        base: DatasetVersion,
        batch: DetachedCycleBatch,
    ) -> Result<DatasetVersion, WriteFailed>;

    /// Read back COMMITTED landings (never an uncommitted cycle), in the STORED
    /// canonical order — this seam does NOT sort. Optionally only cycles after
    /// `from_version`.
    async fn scan_sealed(
        &self,
        from_version: Option<DatasetVersion>,
    ) -> Result<Vec<LandedSlot>, WriteFailed>;

    /// The CHEAP coarse timeline: `(CycleId, DatasetVersion)` per committed cycle,
    /// WITHOUT replaying any landing. The read a downstream time-series consumer
    /// (e.g. `stockfish-rs`, another session) does — a lookup of the version table
    /// over already-coherent frames.
    async fn versions(&self) -> Result<Vec<(CycleId, DatasetVersion)>, WriteFailed>;
}

/// Deinterlace + freeze a whole cycle's casts and commit it in ONE WAL append.
/// The loom ([`order_cycle_stably`]) and the freeze run in
/// [`DetachedCycleBatch::freeze`] BEFORE the single [`WalSink::commit_cycle`], so
/// completion order never reaches storage. Rejects a cross-owner move (and a
/// cycle-id mismatch) before the batch is built. Returns the one published version.
pub async fn persist_cycle<S: WalSink>(
    sink: &S,
    frame: CycleFrame,
    casts: Vec<SweepSlot>,
) -> Result<DatasetVersion, PersistError> {
    for c in &casts {
        if c.cycle != frame.cycle {
            return Err(PersistError::Write(WriteFailed(format!(
                "cast cycle {:?} != frame cycle {:?}",
                c.cycle, frame.cycle
            ))));
        }
        if let Some(mv) = c.paired_move {
            if mv.mailbox != c.owner {
                return Err(PersistError::OwnerMismatch {
                    move_owner: mv.mailbox,
                    landing_owner: c.owner,
                });
            }
        }
    }
    // Loom-before-WAL: order + coalesce + detach, THEN one atomic append.
    let batch = DetachedCycleBatch::freeze(frame, casts);
    sink.commit_cycle(frame.base_version, batch)
        .await
        .map_err(PersistError::Write)
}

/// The result of a [`recover_and_apply`] pass: the moves applied this pass, and the
/// new stream-position **watermark** the caller must persist WITH the owner's
/// sealed cycle state so the next recovery is idempotent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recovered {
    pub applied: Vec<KanbanMove>,
    /// The highest `stream_position` now accounted for (per owner), or the input
    /// watermark when nothing was applied.
    pub watermark: Option<u64>,
}

/// **Crash recovery — post-SEAL only.** Given SEALED landings read back via
/// [`WalSink::scan_sealed`] (an unsealed cycle is never here — steps are eligible
/// only after the seal), an `owner` reconstructed at its durable phase, and the
/// durable `applied_through` watermark (`stream_position`) persisted alongside that
/// phase, re-apply the owner's PENDING tail in canonical `stream_position` order.
///
/// The landings arrive already ordered (write-side deinterlace at seal), so no
/// sort is done here. The watermark — not phase equality — is the idempotence key
/// (the Rubicon lifecycle is cyclic). Above the watermark the chain must be
/// contiguous: a non-matching `from` is a gap/corruption ([`PersistError::StalePhase`]).
/// On error the owner is left mid-chain; re-drive from the persisted watermark.
pub fn recover_and_apply<O: MailboxSoaOwner>(
    owner: &mut O,
    sealed: &[LandedSlot],
    applied_through: Option<u64>,
) -> Result<Recovered, PersistError> {
    let mut applied = Vec::new();
    let mut watermark = applied_through;
    let me = owner.mailbox_id();
    // Already in canonical stream order from the write side; filter to this owner.
    for ls in sealed.iter().filter(|ls| ls.slot.owner == me) {
        if applied_through.is_some_and(|hw| ls.slot.stream_position <= hw) {
            continue;
        }
        match ls.slot.paired_move {
            None => watermark = Some(ls.slot.stream_position),
            Some(mv) => {
                if mv.mailbox != owner.mailbox_id() {
                    return Err(PersistError::OwnerMismatch {
                        move_owner: mv.mailbox,
                        landing_owner: owner.mailbox_id(),
                    });
                }
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
                watermark = Some(ls.slot.stream_position);
            }
        }
    }
    Ok(Recovered { applied, watermark })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::kanban::{ExecTarget, KanbanColumn};
    use lance_graph_contract::soa_view::{MailboxSoaOwner, MailboxSoaView};
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    // ── Minimal in-RAM owner ────────────────────────────────────────────────────
    struct FakeOwner {
        id: MailboxId,
        phase: KanbanColumn,
        cycle: u32,
    }
    impl MailboxSoaView for FakeOwner {
        fn mailbox_id(&self) -> MailboxId {
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
    fn owner(phase: KanbanColumn) -> FakeOwner {
        FakeOwner {
            id: 42,
            phase,
            cycle: 5,
        }
    }

    // ── The WAL sink fake ────────────────────────────────────────────────────────
    struct SealedCycle {
        frame: CycleFrame,
        version: DatasetVersion,
        /// The landings AS COMMITTED (already write-side ordered before the append).
        landings: Vec<SweepSlot>,
        /// The coalesced final image: row -> last value in stream order.
        image: BTreeMap<u64, Vec<u8>>,
    }
    struct FakeWalSink {
        succeed: bool,
        sealed: Mutex<Vec<SealedCycle>>,
        next_version: AtomicU64,
        /// The number of physical WAL appends — must be ONE per committed cycle.
        wal_writes: AtomicU64,
    }
    impl FakeWalSink {
        fn new() -> Self {
            Self {
                succeed: true,
                sealed: Mutex::new(Vec::new()),
                next_version: AtomicU64::new(1),
                wal_writes: AtomicU64::new(0),
            }
        }
        fn failing() -> Self {
            Self {
                succeed: false,
                ..Self::new()
            }
        }
        fn wal_writes(&self) -> u64 {
            self.wal_writes.load(Ordering::SeqCst)
        }
        fn version_count(&self) -> usize {
            self.sealed.lock().unwrap().len()
        }
        fn image_of(&self, cycle: CycleId) -> Option<BTreeMap<u64, Vec<u8>>> {
            self.sealed
                .lock()
                .unwrap()
                .iter()
                .find(|s| s.frame.cycle == cycle)
                .map(|s| s.image.clone())
        }
        /// Test-only: inject a committed cycle whose landings are DELIBERATELY out
        /// of order, to prove `scan_sealed` does not re-sort (order is a write-side
        /// property, fixed before the append — never a read-time repair).
        fn inject_unordered_committed(&self, frame: CycleFrame, landings: Vec<SweepSlot>) {
            let v = DatasetVersion(self.next_version.fetch_add(1, Ordering::SeqCst));
            self.sealed.lock().unwrap().push(SealedCycle {
                frame,
                version: v,
                landings,
                image: BTreeMap::new(),
            });
        }
    }
    impl WalSink for FakeWalSink {
        async fn commit_cycle(
            &self,
            _base: DatasetVersion,
            batch: DetachedCycleBatch,
        ) -> Result<DatasetVersion, WriteFailed> {
            if !self.succeed {
                return Err(WriteFailed("cycle fenced".into()));
            }
            // The batch arrives ALREADY deinterlaced + coalesced (loom before WAL).
            // THE single amortized WAL append for the whole cycle.
            self.wal_writes.fetch_add(1, Ordering::SeqCst);
            let version = DatasetVersion(self.next_version.fetch_add(1, Ordering::SeqCst));
            self.sealed.lock().unwrap().push(SealedCycle {
                frame: batch.frame,
                version,
                landings: batch.landings,
                image: batch.image,
            });
            Ok(version)
        }
        async fn scan_sealed(
            &self,
            from_version: Option<DatasetVersion>,
        ) -> Result<Vec<LandedSlot>, WriteFailed> {
            // Returned in STORED order — no sort. (The order was fixed at seal.)
            Ok(self
                .sealed
                .lock()
                .unwrap()
                .iter()
                .filter(|s| from_version.is_none_or(|f| s.version > f))
                .flat_map(|s| {
                    s.landings.iter().map(|slot| LandedSlot {
                        version: s.version,
                        slot: slot.clone(),
                    })
                })
                .collect())
        }
        async fn versions(&self) -> Result<Vec<(CycleId, DatasetVersion)>, WriteFailed> {
            Ok(self
                .sealed
                .lock()
                .unwrap()
                .iter()
                .map(|s| (s.frame.cycle, s.version))
                .collect())
        }
    }

    fn mv(owner: MailboxId, from: KanbanColumn, to: KanbanColumn) -> KanbanMove {
        KanbanMove {
            mailbox: owner,
            from,
            to,
            witness_chain_position: 0,
            exec: ExecTarget::Elixir,
        }
    }
    /// A landing for `owner` at `stream_position`, moving `from→to`, writing `row`.
    fn slot(
        owner: MailboxId,
        cycle: u64,
        stream_position: u64,
        row: u64,
        m: Option<(KanbanColumn, KanbanColumn)>,
    ) -> SweepSlot {
        SweepSlot {
            cycle: CycleId(cycle),
            stream_position,
            owner,
            row,
            paired_move: m.map(|(f, t)| mv(owner, f, t)),
            payload: vec![stream_position as u8],
        }
    }

    // ── FALSIFIER (headline): one WAL write per cycle — the amortization ─────────
    #[tokio::test]
    async fn a_whole_cycle_of_casts_is_one_wal_write_one_version() {
        let sink = FakeWalSink::new();
        let frame = CycleFrame::new(CycleId(1), DatasetVersion(0));
        // 100 concurrent thoughts stage into an owned cast vector — building it
        // touches NO WAL (staging is the caller's, not the sink's).
        let casts: Vec<SweepSlot> = (0..100u64).map(|i| slot(42, 1, i, i, None)).collect();
        assert_eq!(
            sink.wal_writes(),
            0,
            "staging writes no WAL — pure amortization"
        );
        assert_eq!(sink.version_count(), 0, "no version before the seal");
        // persist_cycle is the SINGLE amortized WAL write for the whole cycle.
        let version = persist_cycle(&sink, frame, casts).await.unwrap();
        assert_eq!(sink.wal_writes(), 1, "100 casts → exactly ONE WAL write");
        assert_eq!(version, DatasetVersion(1), "→ exactly one version");
        assert_eq!(sink.version_count(), 1);
    }

    // ── FALSIFIER: write-side order under scrambled completion (physical race) ───
    #[tokio::test]
    async fn scrambled_completion_lands_in_canonical_stream_order_at_write_time() {
        let sink = FakeWalSink::new();
        // Thoughts "finish" (stage) in scrambled CPU order: stream 2, 0, 3, 1.
        let sealed = persist_cycle(
            &sink,
            CycleFrame::new(CycleId(5), DatasetVersion(0)),
            vec![
                slot(42, 5, 2, 20, None),
                slot(42, 5, 0, 10, None),
                slot(42, 5, 3, 30, None),
                slot(42, 5, 1, 11, None),
            ],
        )
        .await
        .unwrap();
        // scan_sealed does NOT sort; the order came from the WRITE side (seal).
        let order: Vec<u64> = sink
            .scan_sealed(None)
            .await
            .unwrap()
            .iter()
            .filter(|l| l.version == sealed)
            .map(|l| l.slot.stream_position)
            .collect();
        assert_eq!(
            order,
            vec![0, 1, 2, 3],
            "canonical stream order was fixed at WRITE time, not repaired on read",
        );
    }

    /// The complement: `scan_sealed` itself performs NO sort — order is purely a
    /// write-side property. Inject a deliberately out-of-order sealed cycle and
    /// prove scan returns it AS-STORED (a read-time sort would "fix" it).
    #[tokio::test]
    async fn scan_sealed_does_not_repair_order_on_read() {
        let sink = FakeWalSink::new();
        sink.inject_unordered_committed(
            CycleFrame::new(CycleId(9), DatasetVersion(0)),
            vec![
                slot(42, 9, 3, 30, None),
                slot(42, 9, 1, 10, None),
                slot(42, 9, 2, 20, None),
            ],
        );
        let order: Vec<u64> = sink
            .scan_sealed(None)
            .await
            .unwrap()
            .iter()
            .map(|l| l.slot.stream_position)
            .collect();
        assert_eq!(
            order,
            vec![3, 1, 2],
            "scan_sealed returns stored order verbatim — it never sorts (order is write-side)",
        );
    }

    // ── FALSIFIER: per-row coalescing (not last-chunk-wins) ──────────────────────
    #[tokio::test]
    async fn same_row_updates_coalesce_distinct_rows_survive() {
        let sink = FakeWalSink::new();
        persist_cycle(
            &sink,
            CycleFrame::new(CycleId(3), DatasetVersion(0)),
            vec![
                // Two updates to ROW 7 (stream 0 then 2) — coalesce to the later.
                SweepSlot {
                    payload: vec![0xA0],
                    ..slot(42, 3, 0, 7, None)
                },
                // A different ROW 8 — survives independently.
                SweepSlot {
                    payload: vec![0xB0],
                    ..slot(42, 3, 1, 8, None)
                },
                SweepSlot {
                    payload: vec![0xA1],
                    ..slot(42, 3, 2, 7, None)
                },
            ],
        )
        .await
        .unwrap();
        let image = sink.image_of(CycleId(3)).unwrap();
        assert_eq!(image.len(), 2, "two distinct rows in the final image");
        assert_eq!(
            image.get(&7),
            Some(&vec![0xA1]),
            "row 7 coalesced to the later update"
        );
        assert_eq!(image.get(&8), Some(&vec![0xB0]), "row 8 survived, disjoint");
    }

    // ── FALSIFIER: no partial visibility — an unsealed cycle is invisible ────────
    #[tokio::test]
    async fn an_unsealed_cycle_is_invisible_epistemic_horizon() {
        let sink = FakeWalSink::new();
        // A cycle's casts are staged in an owned vector held by the caller — NOT in
        // the sink. Until the single atomic commit_cycle runs, nothing is visible:
        // an open cycle's output is NOT readable as Vn input (read Vn / write Vn+1).
        let casts = vec![slot(
            42,
            9,
            0,
            0,
            Some((KanbanColumn::Planning, KanbanColumn::CognitiveWork)),
        )];
        assert_eq!(
            sink.scan_sealed(None).await.unwrap().len(),
            0,
            "before the seal, no landing is visible",
        );
        assert!(
            sink.versions().await.unwrap().is_empty(),
            "no version before seal"
        );
        // The seal is all-or-nothing: the whole cycle appears at once, never partially.
        persist_cycle(&sink, CycleFrame::new(CycleId(9), DatasetVersion(0)), casts)
            .await
            .unwrap();
        assert_eq!(
            sink.scan_sealed(None).await.unwrap().len(),
            1,
            "visible only after seal"
        );
    }

    // ── FALSIFIER: sealed read horizon — a cycle reads the sealed predecessor ────
    #[tokio::test]
    async fn a_cycle_reads_the_sealed_predecessor_not_an_in_flight_sibling() {
        let sink = FakeWalSink::new();
        // Cycle 1 seals → V1.
        let s1 = persist_cycle(
            &sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![slot(42, 1, 0, 0, None)],
        )
        .await
        .unwrap();
        // Cycle 2's base_version is the SEALED V1 — never an in-flight value.
        let f2 = CycleFrame::new(CycleId(2), s1);
        assert_eq!(
            f2.base_version, s1,
            "the next cycle reads the SEALED predecessor version (Vn), not in-flight state",
        );
        let s2 = persist_cycle(&sink, f2, vec![slot(99, 2, 0, 0, None)])
            .await
            .unwrap();
        assert_eq!(s2, DatasetVersion(2), "and publishes exactly its own Vn+1");
    }

    // ── FALSIFIER: cheap time-series — version table only, no landing replay ─────
    #[tokio::test]
    async fn downstream_time_series_reads_the_version_table_not_the_landings() {
        let sink = FakeWalSink::new();
        for c in 1..=3u64 {
            persist_cycle(
                &sink,
                CycleFrame::new(CycleId(c), DatasetVersion(c - 1)),
                vec![slot(42, c, 0, 0, None)],
            )
            .await
            .unwrap();
        }
        assert_eq!(
            sink.versions().await.unwrap(),
            vec![
                (CycleId(1), DatasetVersion(1)),
                (CycleId(2), DatasetVersion(2)),
                (CycleId(3), DatasetVersion(3)),
            ],
            "history scales with CYCLES; a time-series consumer just looks up the version table",
        );
    }

    // ── Recovery over sealed landings (post-seal only) ───────────────────────────
    #[tokio::test]
    async fn recovery_replays_an_owners_chain_in_stream_order_skipping_others() {
        let sink = FakeWalSink::new();
        persist_cycle(
            &sink,
            CycleFrame::new(CycleId(4), DatasetVersion(0)),
            vec![
                slot(
                    42,
                    4,
                    0,
                    0,
                    Some((KanbanColumn::Planning, KanbanColumn::CognitiveWork)),
                ),
                slot(
                    99,
                    4,
                    1,
                    1,
                    Some((KanbanColumn::Planning, KanbanColumn::CognitiveWork)),
                ),
                slot(
                    42,
                    4,
                    2,
                    2,
                    Some((KanbanColumn::CognitiveWork, KanbanColumn::Evaluation)),
                ),
            ],
        )
        .await
        .unwrap();
        let sealed = sink.scan_sealed(None).await.unwrap();
        let mut o = owner(KanbanColumn::Planning);
        let rec = recover_and_apply(&mut o, &sealed, None).unwrap();
        assert_eq!(
            rec.applied.iter().map(|m| m.to).collect::<Vec<_>>(),
            vec![KanbanColumn::CognitiveWork, KanbanColumn::Evaluation],
            "owner 42's chain in stream order — owner 99's interleaved landing skipped",
        );
        assert_eq!(o.phase(), KanbanColumn::Evaluation);
    }

    #[tokio::test]
    async fn cyclic_recovery_is_idempotent_only_with_the_watermark() {
        assert_eq!(KanbanColumn::Plan.next_phases(), &[KanbanColumn::Planning]);
        let sink = FakeWalSink::new();
        persist_cycle(
            &sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![
                slot(
                    42,
                    1,
                    0,
                    0,
                    Some((KanbanColumn::Planning, KanbanColumn::CognitiveWork)),
                ),
                slot(
                    42,
                    1,
                    1,
                    1,
                    Some((KanbanColumn::CognitiveWork, KanbanColumn::Evaluation)),
                ),
                slot(
                    42,
                    1,
                    2,
                    2,
                    Some((KanbanColumn::Evaluation, KanbanColumn::Plan)),
                ),
                slot(
                    42,
                    1,
                    3,
                    3,
                    Some((KanbanColumn::Plan, KanbanColumn::Planning)),
                ),
            ],
        )
        .await
        .unwrap();
        let sealed = sink.scan_sealed(None).await.unwrap();
        let mut o = owner(KanbanColumn::Planning);

        let rec = recover_and_apply(&mut o, &sealed, None).unwrap();
        assert_eq!(rec.applied.len(), 4, "the whole lap applied once");
        assert_eq!(
            o.phase(),
            KanbanColumn::Planning,
            "back at Planning after a lap"
        );

        let good = recover_and_apply(&mut o, &sealed, rec.watermark).unwrap();
        assert!(
            good.applied.is_empty(),
            "watermark makes cyclic recovery idempotent"
        );

        let bug = recover_and_apply(&mut o, &sealed, None).unwrap();
        assert_eq!(
            bug.applied.len(),
            4,
            "without the watermark the cyclic lap replays"
        );
    }

    #[tokio::test]
    async fn a_cross_owner_move_is_rejected_before_the_cycle_persists() {
        let bad = SweepSlot {
            paired_move: Some(mv(99, KanbanColumn::Planning, KanbanColumn::CognitiveWork)),
            ..slot(42, 1, 0, 0, None)
        };
        let sink = FakeWalSink::new();
        assert!(matches!(
            persist_cycle(
                &sink,
                CycleFrame::new(CycleId(1), DatasetVersion(0)),
                vec![bad]
            )
            .await,
            Err(PersistError::OwnerMismatch {
                move_owner: 99,
                landing_owner: 42
            }),
        ));
        assert_eq!(sink.wal_writes(), 0, "nothing written on a bad cast");
        assert_eq!(sink.version_count(), 0);
    }

    #[tokio::test]
    async fn a_fenced_cycle_writes_nothing() {
        let sink = FakeWalSink::failing();
        let r = persist_cycle(
            &sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![slot(
                42,
                1,
                0,
                0,
                Some((KanbanColumn::Planning, KanbanColumn::CognitiveWork)),
            )],
        )
        .await;
        assert!(matches!(r, Err(PersistError::Write(_))));
        assert_eq!(sink.wal_writes(), 0, "no WAL write, no version, no step");
        assert_eq!(sink.scan_sealed(None).await.unwrap().len(), 0);
    }

    #[test]
    fn cycle_frame_is_storage_only() {
        // The frame carries ONLY storage identity — no rung / projection / branch
        // / semantic tags are minted in this layer (they live upstream).
        let f = CycleFrame::new(CycleId(1), DatasetVersion(7));
        assert_eq!(f.cycle, CycleId(1));
        assert_eq!(f.base_version, DatasetVersion(7));
    }

    #[test]
    fn order_cycle_stably_is_stable_and_generic() {
        // Scrambled → ordered by the supplied key; equal keys keep arrival order.
        let mut rows = vec![(3u64, "d"), (1u64, "b1"), (2u64, "c"), (1u64, "b2")];
        order_cycle_stably(&mut rows, |r| r.0);
        assert_eq!(rows, vec![(1, "b1"), (1, "b2"), (2, "c"), (3, "d")]);
    }
}
