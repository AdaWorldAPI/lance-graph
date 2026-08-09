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
//! tests are **storage/race CONTRACT probes**: one durable commit per cycle,
//! write-side order under scrambled completion, per-row coalescing, no unsealed
//! visibility, sealed read horizon, recovery idempotence. They do **NOT** prove
//! real durability. **`compile+test green ≠ storage proven`** (the Ladybug
//! lesson). The concrete sink is `lance_graph::graph::cycle_sink::LanceCycleWriter`.
//!
//! ## The governing storage rule (operator-ruled 2026-08-09 — supersedes the
//! ## earlier "one version per cycle, empty cycles included" contract)
//!
//! **No artifact-backed semantic change → no write → no new [`DatasetVersion`].**
//!
//! - A cast with a NON-EMPTY payload is an **artifact cast** — a real semantic
//!   delta (grounding result, reusable walk product, adjudication, terminal
//!   conclusion). Only these persist.
//! - A cast with an EMPTY payload is an **intent-only cast** (a held-intent
//!   re-stage, a pure kanban step). It is EPHEMERAL: its move still applies to
//!   the in-memory fleet, but it never reaches the store, never trips a
//!   payload gate, and after a crash is cheaply regenerated from the pinned
//!   sealed inputs. Kanban never decides when to persist; the anyway-happening
//!   artifact write is what carries kanban progress into durability.
//! - A cycle with ZERO artifact casts is [`CommitOutcome::NoChange`]: zero
//!   store calls, zero rows, unchanged head. (The predecessor contract's
//!   deliberate empty-cycle versioning is REMOVED, not repaired.)
//!
//! **Honest limit of the Phase-A gate (review-flagged):** the gate tests
//! payload PRESENCE, not semantic CHANGE — an identical artifact re-emitted in
//! a later cycle still commits (a new cycle genuinely concluded, but with
//! unchanged content). Change-detection needs conclusion IDENTITY (the
//! `(episode, plan, revision)` model + producer-side digest dedup) and is the
//! Phase-D conclusion-boundary refinement; a typed
//! `IntentOnly | ArtifactChanged` cast kind belongs there too, replacing the
//! empty-payload CONVENTION with a type. Until then this doc — not the type
//! system — is what says an empty payload means intent.
//!
//! ## One logical writer — commit is `&mut self`, no rollback, no delete
//!
//! There is exactly ONE logical application writer per cycle store; the 64k
//! thoughts are parallel PRODUCERS, not writers. The trait therefore commits
//! through `&mut self` (two application commits cannot interleave through the
//! type boundary), and a published manifest is HISTORY — there is no
//! compensating delete and no rollback. An unexpected head is a
//! fence/reconciliation condition ([`CommitError::Fenced`] /
//! [`CommitOutcome::Reconciled`]), never normal competition. Idempotency is
//! durable: every committed batch carries its `(cycle, batch_hash)` in the
//! same commit, so a lost acknowledgement reconciles by re-submitting the SAME
//! frozen batch — [`WalSink::commit_cycle`] finds it and returns
//! [`CommitOutcome::Reconciled`] instead of appending twice.

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
    ///
    /// **Contract: monotonically increasing per owner ACROSS cycles, not
    /// per cycle.** [`recover_and_apply`] uses it as the durable watermark over a
    /// multi-cycle [`WalSink::scan_sealed`] stream (`stream_position <= watermark`
    /// ⇒ already applied), so a per-cycle restart to 0 would make recovery skip
    /// every later-cycle landing at or below an earlier cycle's watermark. The
    /// caller owns this monotonicity (it is the witness-fabric order key, already
    /// monotonic); this layer does not re-key by `(CycleId, stream_position)`.
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

/// A landing read back from a SEALED cycle — its slot plus the cycle it sealed
/// with. Only ever produced for SEALED cycles; returned in the stored (already
/// canonical) order, NEVER re-sorted.
///
/// Keyed by CYCLE, not by dataset version: the physical [`DatasetVersion`] is
/// the publication position of the sole writer's commit (returned in
/// [`CommitOutcome::Committed`]) and is deliberately NOT re-derivable per row —
/// semantic identity lives in `(cycle, batch_hash)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LandedSlot {
    /// The cycle this landing sealed with (the recovery bound / grouping key).
    pub cycle: CycleId,
    pub slot: SweepSlot,
}

/// One committed cycle's frame metadata — the coarse timeline row. Compact:
/// never carries payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameMeta {
    pub cycle: CycleId,
    /// The sealed read horizon this cycle's thoughts read (`Vn`).
    pub base_version: DatasetVersion,
    /// The batch's deterministic content hash (the durable idempotency key).
    pub batch_hash: u64,
}

/// The honest result states of a commit attempt. There is NO state meaning
/// "nothing landed" that can be returned after publication may have occurred —
/// that case either reconciles to [`Reconciled`](CommitOutcome::Reconciled) or
/// surfaces as [`CommitError::Ambiguous`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitOutcome {
    /// Zero artifact casts: zero store calls, zero rows, unchanged head.
    NoChange { head: DatasetVersion },
    /// The batch is durable at the ACTUAL returned physical PUBLICATION
    /// version — the audit-grade reference (unlike
    /// [`Reconciled`](CommitOutcome::Reconciled)'s `current_head`).
    Committed {
        version: DatasetVersion,
        cycle: CycleId,
        batch_hash: u64,
    },
    /// The batch was ALREADY durable (a lost acknowledgement / retry): found by
    /// its `(cycle, batch_hash)` identity; no second append happened.
    ///
    /// **`current_head` is deliberately NOT named `version`**: it is the store
    /// head AT RECONCILIATION TIME, not the version this cycle originally
    /// published at (retrying cycle 1 after cycle 5 reconciles at head V5).
    /// An audit trail must reference the durable identity
    /// `(cycle, batch_hash)` — the exact publication position is recoverable
    /// from the version history on the audit path, never inferred from this
    /// field. Only [`Committed`](CommitOutcome::Committed)'s `version` is a
    /// publication version.
    Reconciled {
        current_head: DatasetVersion,
        cycle: CycleId,
        batch_hash: u64,
    },
}

/// Why a commit attempt did NOT yield a durable outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommitError {
    /// The store head is not the frame's sealed predecessor and this batch is
    /// not already durable. NOTHING was written: regenerate against the
    /// current head. Under the one-writer topology this means a lost-response
    /// restart, a stale cached head, an unauthorized writer, or corruption —
    /// never normal competition.
    Fenced { current_head: DatasetVersion },
    /// This cycle is durable with a DIFFERENT batch hash — fail closed
    /// (corruption / identity conflict). Never promoted, never overwritten.
    HashConflict {
        cycle: CycleId,
        stored_hash: u64,
        offered_hash: u64,
    },
    /// I/O failed with provably NOTHING published — safe to regenerate from
    /// the unchanged horizon.
    Io(WriteFailed),
    /// The append's outcome could not be determined AND reconciliation itself
    /// failed. Re-submit the SAME frozen batch: `commit_cycle` reconciles
    /// first, so the retry cannot double-append.
    Ambiguous {
        cycle: CycleId,
        batch_hash: u64,
        cause: String,
    },
}

impl std::fmt::Display for CommitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fenced { current_head } => {
                write!(f, "fenced: store head is {current_head:?}, nothing written")
            }
            Self::HashConflict {
                cycle,
                stored_hash,
                offered_hash,
            } => write!(
                f,
                "cycle {cycle:?} durable with hash {stored_hash:#018x}, offered {offered_hash:#018x} — fail closed"
            ),
            Self::Io(e) => write!(f, "commit I/O (nothing published): {e}"),
            Self::Ambiguous {
                cycle,
                batch_hash,
                cause,
            } => write!(
                f,
                "AMBIGUOUS: cycle {cycle:?} (hash {batch_hash:#018x}) may or may not be durable: {cause}"
            ),
        }
    }
}
impl std::error::Error for CommitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
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
    /// ARTIFACT landings (non-empty payloads) in canonical `stream_position`
    /// order (the temporal deinterlace already ran — the store never sees
    /// worker-completion order). Intent-only casts never enter a batch.
    pub landings: Vec<SweepSlot>,
    /// The coalesced final image: `row -> last payload in stream order`.
    pub image: std::collections::BTreeMap<u64, Vec<u8>>,
    /// Deterministic content hash over the CANONICAL landings — the durable
    /// idempotency key. Identical completed sets yield identical hashes
    /// regardless of worker completion order (the freeze canonicalizes first).
    pub batch_hash: u64,
}

impl DetachedCycleBatch {
    /// Freeze concurrently-produced ARTIFACT casts into the ordered, coalesced
    /// cycle image: (1) [`order_cycle_stably`] stable-orders by
    /// `stream_position` — completion order never becomes storage order
    /// (physical race); (2) fold same-row updates into owned rows (later
    /// stream position wins); (3) hash the canonical content. The result is
    /// detached from any live SoA and ready for exactly one durable commit.
    ///
    /// Callers pass artifact casts only ([`persist_cycle`] partitions);
    /// passing intent-only casts here would persist control flow, which the
    /// governing storage rule forbids.
    #[must_use]
    pub fn freeze(frame: CycleFrame, mut casts: Vec<SweepSlot>) -> Self {
        order_cycle_stably(&mut casts, |s| s.stream_position);
        let mut image = std::collections::BTreeMap::new();
        for s in &casts {
            image.insert(s.row, s.payload.clone());
        }
        let batch_hash = Self::content_hash(frame, &casts);
        Self {
            frame,
            landings: casts,
            image,
            batch_hash,
        }
    }

    /// FNV-1a 64 over the frame identity + canonical landing content.
    fn content_hash(frame: CycleFrame, canonical: &[SweepSlot]) -> u64 {
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = OFFSET;
        let mut eat = |bytes: &[u8]| {
            for b in bytes {
                h ^= u64::from(*b);
                h = h.wrapping_mul(PRIME);
            }
        };
        eat(&frame.cycle.0.to_le_bytes());
        eat(&frame.base_version.0.to_le_bytes());
        for s in canonical {
            eat(&s.stream_position.to_le_bytes());
            eat(&s.owner.to_le_bytes());
            eat(&s.row.to_le_bytes());
            match &s.paired_move {
                Some(m) => eat(&[1, m.from as u8, m.to as u8, m.exec as u8]),
                None => eat(&[0]),
            }
            if let Some(m) = &s.paired_move {
                eat(&m.mailbox.to_le_bytes());
                eat(&m.witness_chain_position.to_le_bytes());
            }
            eat(&(s.payload.len() as u64).to_le_bytes());
            eat(&s.payload);
        }
        h
    }
}

/// Why a persist / seal / step operation produced no lifecycle advance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistError {
    /// The cycle seal did not land — nothing published, so no step. This is the
    /// RETRYABLE class (a fenced / failed append may succeed on retry).
    Write(WriteFailed),
    /// The durable commit did not yield an outcome — see [`CommitError`] for
    /// the honest sub-states ([`Fenced`](CommitError::Fenced) = regenerate
    /// against the new head; [`Io`](CommitError::Io) = nothing landed, safe
    /// regenerate; [`Ambiguous`](CommitError::Ambiguous) = re-submit the SAME
    /// frozen batch, reconciliation decides; [`HashConflict`](CommitError::HashConflict)
    /// = fail closed).
    Commit(CommitError),
    /// A cast was staged against a different cycle than the frame — a caller
    /// programming error, PERMANENT and never retryable (distinct from [`Write`]
    /// so retry logic never loops on it). [`Write`]: PersistError::Write
    CycleMismatch {
        cast_cycle: CycleId,
        frame_cycle: CycleId,
    },
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
            Self::Commit(e) => write!(f, "{e}"),
            Self::CycleMismatch {
                cast_cycle,
                frame_cycle,
            } => write!(
                f,
                "cast cycle {cast_cycle:?} != frame cycle {frame_cycle:?}"
            ),
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
impl std::error::Error for PersistError {
    /// Expose the wrapped cause so callers can walk the error chain — the two
    /// wrapping variants ([`Write`](PersistError::Write) /
    /// [`Illegal`](PersistError::Illegal)) forward to their inner error; the
    /// self-describing variants have no deeper cause.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Write(e) => Some(e),
            Self::Commit(e) => Some(e),
            Self::Illegal(e) => Some(e),
            Self::CycleMismatch { .. } | Self::OwnerMismatch { .. } | Self::StalePhase { .. } => {
                None
            }
        }
    }
}

/// The WAL seam — **one durable append per cycle**. [`commit_cycle`](WalSink::commit_cycle)
/// takes an ALREADY-deinterlaced, ALREADY-frozen [`DetachedCycleBatch`] (the loom
/// ran before the WAL) and performs the single atomic append → one
/// [`DatasetVersion`], visible all-or-nothing. Reads return committed landings
/// only, in the STORED canonical order — this seam NEVER sorts (order is a
/// write-side property, fixed before the append).
///
/// `&self` (shared) and **no owner borrow**: the persistence path never holds the
/// SoA owner across I/O.
///
/// **Futures are not `Send`** (native `async fn` in trait, no `Send` bound): callers
/// drive these on the crate's single-task persistence path and must NOT
/// `tokio::spawn` them onto a multi-threaded runtime. If a concrete sink ever needs
/// to be spawned, give it explicit `impl Future<Output = …> + Send` methods there.
#[allow(async_fn_in_trait)]
pub trait WalSink {
    /// The SINGLE amortized durable commit for a whole cycle: commit `batch`
    /// (already ordered + coalesced + hashed, artifact casts only) against the
    /// sealed predecessor `batch.frame.base_version`, publishing exactly one
    /// new [`DatasetVersion`] atomically — 64k thoughts → one commit.
    ///
    /// **`&mut self` — the one-logical-writer boundary.** Two application
    /// commits cannot interleave through this type boundary (a concrete writer
    /// is additionally non-`Clone`). Producers stay fire-and-forget on their
    /// own `&self`/staging surfaces; only the sole writer drives this method
    /// and it FULLY honors the result.
    ///
    /// Reconciliation-first: an implementation looks the batch's
    /// `(cycle, batch_hash)` up BEFORE appending, so re-submitting the same
    /// frozen batch after a lost acknowledgement returns
    /// [`CommitOutcome::Reconciled`] instead of double-appending. There is no
    /// rollback and no compensating delete: a published manifest is history.
    async fn commit_cycle(
        &mut self,
        batch: DetachedCycleBatch,
    ) -> Result<CommitOutcome, CommitError>;

    /// Read back COMMITTED landings (never an uncommitted cycle), in the STORED
    /// canonical order — this seam does NOT sort. `after_cycle` bounds the read
    /// to cycles strictly after it (the recovery tail bound); implementations
    /// push the bound into the storage scan, never full-scan-and-filter.
    async fn scan_sealed(
        &self,
        after_cycle: Option<CycleId>,
    ) -> Result<Vec<LandedSlot>, WriteFailed>;

    /// The CHEAP coarse timeline: one [`FrameMeta`] per committed cycle,
    /// WITHOUT touching any payload. NOT a normal-path read — after a
    /// successful commit the caller already holds the outcome; this is for
    /// recovery, audit, and downstream consumers.
    async fn timeline(&self) -> Result<Vec<FrameMeta>, WriteFailed>;
}

/// Partition, deinterlace + freeze a cycle's ARTIFACT casts and commit them in
/// ONE durable operation — or perform NONE when there is nothing artifact-backed.
///
/// The governing storage rule, applied here:
/// - Intent-only casts (EMPTY payload — held-intent re-stages, pure kanban
///   steps) are partitioned OUT before the freeze. They stay ephemeral: their
///   moves still reach the fleet through the caller's transition set, but no
///   payload gate sees them and no store row carries them.
/// - Zero artifact casts ⇒ [`CommitOutcome::NoChange`], with the sink NEVER
///   called: zero store operations, zero rows, unchanged head.
/// - Otherwise the loom ([`order_cycle_stably`]) and freeze run in
///   [`DetachedCycleBatch::freeze`] BEFORE the single
///   [`WalSink::commit_cycle`], so completion order never reaches storage.
///
/// Rejects a cross-owner move (and a cycle-id mismatch) across ALL casts —
/// ephemeral ones included — before anything is built.
pub async fn persist_cycle<S: WalSink>(
    sink: &mut S,
    frame: CycleFrame,
    casts: Vec<SweepSlot>,
) -> Result<CommitOutcome, PersistError> {
    for c in &casts {
        if c.cycle != frame.cycle {
            // Permanent caller error — NOT a retryable class (retry would loop).
            return Err(PersistError::CycleMismatch {
                cast_cycle: c.cycle,
                frame_cycle: frame.cycle,
            });
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
    // The artifact gate: only non-empty payloads are semantic deltas.
    let artifacts: Vec<SweepSlot> = casts
        .into_iter()
        .filter(|c| !c.payload.is_empty())
        .collect();
    if artifacts.is_empty() {
        return Ok(CommitOutcome::NoChange {
            head: frame.base_version,
        });
    }
    // Loom-before-commit: order + coalesce + hash + detach, THEN one commit.
    let batch = DetachedCycleBatch::freeze(frame, artifacts);
    sink.commit_cycle(batch).await.map_err(PersistError::Commit)
}

/// The result of a [`recover_and_apply`] pass: the moves applied this pass, and the
/// new stream-position **watermark** the caller must persist WITH the owner's
/// sealed cycle state so the next recovery is idempotent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recovered {
    pub applied: Vec<KanbanMove>,
    /// The highest `stream_position` now accounted for (per owner), or the input
    /// watermark when nothing was applied. Same cross-cycle scope as
    /// [`SweepSlot::stream_position`] — monotonic per owner across sealed cycles,
    /// so it is a durable watermark over the whole multi-cycle `scan_sealed` stream,
    /// not a per-cycle counter.
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
///
/// **On error the accumulated [`Recovered`] is returned WITH the error**
/// (`Err((partial, cause))`). The function mutates `owner` as it walks the chain,
/// so a mid-chain failure has already advanced the owner's phase for the applied
/// prefix; returning the partial lets the caller persist the watermark it earned.
/// Discarding it would leave the watermark below the applied prefix, replaying those
/// moves against an already-advanced owner (a permanent `StalePhase` stall in the
/// acyclic case, a double-apply in a cyclic lap).
pub fn recover_and_apply<O: MailboxSoaOwner>(
    owner: &mut O,
    sealed: &[LandedSlot],
    applied_through: Option<u64>,
) -> Result<Recovered, (Recovered, PersistError)> {
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
                if mv.mailbox != me {
                    return Err((
                        Recovered { applied, watermark },
                        PersistError::OwnerMismatch {
                            move_owner: mv.mailbox,
                            landing_owner: me,
                        },
                    ));
                }
                if mv.from != owner.phase() {
                    return Err((
                        Recovered { applied, watermark },
                        PersistError::StalePhase {
                            owner_phase: owner.phase(),
                            move_from: mv.from,
                        },
                    ));
                }
                match owner.try_advance_phase(mv.to) {
                    Ok(step) => {
                        applied.push(step);
                        watermark = Some(ls.slot.stream_position);
                    }
                    Err(e) => {
                        return Err((Recovered { applied, watermark }, PersistError::Illegal(e)))
                    }
                }
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

    // ── The store fake (contract probe; concrete = LanceCycleWriter) ────────────
    struct SealedRec {
        frame: CycleFrame,
        batch_hash: u64,
        /// The landings AS COMMITTED (already write-side ordered before the commit).
        landings: Vec<SweepSlot>,
        /// The coalesced final image: row -> last value in stream order.
        image: BTreeMap<u64, Vec<u8>>,
    }
    struct FakeWalSink {
        succeed: bool,
        sealed: Mutex<Vec<SealedRec>>,
        /// The store's physical head version (0 = empty store).
        head: AtomicU64,
        /// The number of physical durable commits — must be ONE per committed cycle.
        wal_writes: AtomicU64,
    }
    impl FakeWalSink {
        fn new() -> Self {
            Self {
                succeed: true,
                sealed: Mutex::new(Vec::new()),
                head: AtomicU64::new(0),
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
        fn head(&self) -> DatasetVersion {
            DatasetVersion(self.head.load(Ordering::SeqCst))
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
        /// property, fixed before the commit — never a read-time repair).
        fn inject_unordered_committed(&self, frame: CycleFrame, landings: Vec<SweepSlot>) {
            self.head.fetch_add(1, Ordering::SeqCst);
            self.sealed.lock().unwrap().push(SealedRec {
                frame,
                batch_hash: 0,
                landings,
                image: BTreeMap::new(),
            });
        }
    }
    impl WalSink for FakeWalSink {
        async fn commit_cycle(
            &mut self,
            batch: DetachedCycleBatch,
        ) -> Result<CommitOutcome, CommitError> {
            if !self.succeed {
                return Err(CommitError::Io(WriteFailed("cycle store I/O".into())));
            }
            let mut sealed = self.sealed.lock().unwrap();
            // Reconciliation-first: an already-durable (cycle, hash) is success,
            // a matching cycle with a different hash fails closed.
            if let Some(rec) = sealed.iter().find(|s| s.frame.cycle == batch.frame.cycle) {
                return if rec.batch_hash == batch.batch_hash {
                    Ok(CommitOutcome::Reconciled {
                        current_head: DatasetVersion(self.head.load(Ordering::SeqCst)),
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
            // The fence: a commit MUST target the current head (`Vn`). Under the
            // one-writer topology a mismatch is a stale horizon, never competition.
            let head = DatasetVersion(self.head.load(Ordering::SeqCst));
            if batch.frame.base_version != head {
                return Err(CommitError::Fenced { current_head: head });
            }
            // THE single amortized durable commit for the whole cycle.
            self.wal_writes.fetch_add(1, Ordering::SeqCst);
            let version = DatasetVersion(self.head.fetch_add(1, Ordering::SeqCst) + 1);
            let (cycle, batch_hash) = (batch.frame.cycle, batch.batch_hash);
            sealed.push(SealedRec {
                frame: batch.frame,
                batch_hash,
                landings: batch.landings,
                image: batch.image,
            });
            Ok(CommitOutcome::Committed {
                version,
                cycle,
                batch_hash,
            })
        }
        async fn scan_sealed(
            &self,
            after_cycle: Option<CycleId>,
        ) -> Result<Vec<LandedSlot>, WriteFailed> {
            // Returned in STORED order — no sort. (The order was fixed at seal.)
            Ok(self
                .sealed
                .lock()
                .unwrap()
                .iter()
                .filter(|s| after_cycle.is_none_or(|c| s.frame.cycle > c))
                .flat_map(|s| {
                    s.landings.iter().map(|slot| LandedSlot {
                        cycle: s.frame.cycle,
                        slot: slot.clone(),
                    })
                })
                .collect())
        }
        async fn timeline(&self) -> Result<Vec<FrameMeta>, WriteFailed> {
            Ok(self
                .sealed
                .lock()
                .unwrap()
                .iter()
                .map(|s| FrameMeta {
                    cycle: s.frame.cycle,
                    base_version: s.frame.base_version,
                    batch_hash: s.batch_hash,
                })
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
        let mut sink = FakeWalSink::new();
        let frame = CycleFrame::new(CycleId(1), DatasetVersion(0));
        // 100 concurrent thoughts stage into an owned cast vector — building it
        // touches NO store (staging is the caller's, not the sink's).
        let casts: Vec<SweepSlot> = (0..100u64).map(|i| slot(42, 1, i, i, None)).collect();
        assert_eq!(
            sink.wal_writes(),
            0,
            "staging writes nothing — pure amortization"
        );
        assert_eq!(sink.version_count(), 0, "no version before the seal");
        // persist_cycle is the SINGLE amortized durable commit for the cycle.
        let out = persist_cycle(&mut sink, frame, casts).await.unwrap();
        assert_eq!(sink.wal_writes(), 1, "100 casts → exactly ONE commit");
        assert!(
            matches!(
                out,
                CommitOutcome::Committed {
                    version: DatasetVersion(1),
                    cycle: CycleId(1),
                    ..
                }
            ),
            "→ exactly one version, honestly reported: {out:?}"
        );
        assert_eq!(sink.version_count(), 1);
    }

    // ── FALSIFIER: write-side order under scrambled completion (physical race) ───
    #[tokio::test]
    async fn scrambled_completion_lands_in_canonical_stream_order_at_write_time() {
        let mut sink = FakeWalSink::new();
        // Thoughts "finish" (stage) in scrambled CPU order: stream 2, 0, 3, 1.
        persist_cycle(
            &mut sink,
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
            .filter(|l| l.cycle == CycleId(5))
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
        let mut sink = FakeWalSink::new();
        persist_cycle(
            &mut sink,
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
        let mut sink = FakeWalSink::new();
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
            sink.timeline().await.unwrap().is_empty(),
            "no frame before seal"
        );
        // The seal is all-or-nothing: the whole cycle appears at once, never partially.
        persist_cycle(
            &mut sink,
            CycleFrame::new(CycleId(9), DatasetVersion(0)),
            casts,
        )
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
        let mut sink = FakeWalSink::new();
        // Cycle 1 seals → V1 (the sealed head advances to V1).
        let s1 = persist_cycle(
            &mut sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![slot(42, 1, 0, 0, None)],
        )
        .await
        .unwrap();
        let CommitOutcome::Committed { version: v1, .. } = s1 else {
            panic!("expected Committed, got {s1:?}");
        };
        assert_eq!(v1, DatasetVersion(1));
        // A sibling that read the STALE predecessor V0 (an in-flight base, not the
        // sealed head V1) is FENCED — the sink refuses the commit with the
        // current head, writing NOTHING (never normal competition under the
        // one-writer topology; a fence/reconciliation condition).
        let stale = persist_cycle(
            &mut sink,
            CycleFrame::new(CycleId(2), DatasetVersion(0)),
            vec![slot(99, 2, 0, 0, None)],
        )
        .await;
        assert!(
            matches!(
                stale,
                Err(PersistError::Commit(CommitError::Fenced {
                    current_head: DatasetVersion(1)
                }))
            ),
            "committing against a stale base is Fenced with the current head: {stale:?}",
        );
        assert_eq!(
            sink.version_count(),
            1,
            "the fenced commit published nothing"
        );
        // Committing against the SEALED head V1 succeeds and publishes exactly V2.
        let s2 = persist_cycle(
            &mut sink,
            CycleFrame::new(CycleId(2), v1),
            vec![slot(99, 2, 0, 0, None)],
        )
        .await
        .unwrap();
        assert!(
            matches!(
                s2,
                CommitOutcome::Committed {
                    version: DatasetVersion(2),
                    ..
                }
            ),
            "and publishes exactly its own Vn+1: {s2:?}"
        );
    }

    // ── FALSIFIER: cheap time-series — version table only, no landing replay ─────
    #[tokio::test]
    async fn downstream_time_series_reads_the_version_table_not_the_landings() {
        let mut sink = FakeWalSink::new();
        for c in 1..=3u64 {
            persist_cycle(
                &mut sink,
                CycleFrame::new(CycleId(c), DatasetVersion(c - 1)),
                vec![slot(42, c, 0, 0, None)],
            )
            .await
            .unwrap();
        }
        let frames = sink.timeline().await.unwrap();
        assert_eq!(
            frames
                .iter()
                .map(|f| (f.cycle, f.base_version))
                .collect::<Vec<_>>(),
            vec![
                (CycleId(1), DatasetVersion(0)),
                (CycleId(2), DatasetVersion(1)),
                (CycleId(3), DatasetVersion(2)),
            ],
            "history scales with CYCLES; the timeline is frame metadata, no payloads",
        );
        assert!(
            frames.iter().all(|f| f.batch_hash != 0),
            "every frame carries its durable idempotency hash"
        );
    }

    // ── Recovery over sealed landings (post-seal only) ───────────────────────────
    #[tokio::test]
    async fn recovery_replays_an_owners_chain_in_stream_order_skipping_others() {
        let mut sink = FakeWalSink::new();
        persist_cycle(
            &mut sink,
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
        let mut sink = FakeWalSink::new();
        persist_cycle(
            &mut sink,
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

    // ── FALSIFIER: a no-step landing advances the watermark past itself ──────────
    #[tokio::test]
    async fn a_no_step_landing_advances_the_watermark_and_is_not_re_scanned() {
        // A mixed chain: step@0, a NO-STEP landing@1, step@2. If the no-step
        // landing did not advance the watermark, it would be re-scanned forever and
        // block the watermark behind every following step.
        let mut sink = FakeWalSink::new();
        persist_cycle(
            &mut sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![
                slot(
                    42,
                    1,
                    0,
                    0,
                    Some((KanbanColumn::Planning, KanbanColumn::CognitiveWork)),
                ),
                slot(42, 1, 1, 1, None), // no-step (artifact) landing at position 1
                slot(
                    42,
                    1,
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
        assert_eq!(rec.applied.len(), 2, "the two real steps applied");
        assert_eq!(
            rec.watermark,
            Some(2),
            "the watermark reaches the last landing, past the no-step at 1",
        );
        // A second pass with that watermark applies nothing (no re-scan of the no-step).
        let again = recover_and_apply(&mut o, &sealed, rec.watermark).unwrap();
        assert!(again.applied.is_empty(), "idempotent — nothing re-applied");
    }

    // ── FALSIFIER: a mid-chain failure returns the watermark it already earned ───
    #[tokio::test]
    async fn a_mid_chain_failure_returns_the_applied_prefix_watermark() {
        // step@0 is valid and advances the owner; step@1 has a `from` that no longer
        // matches (corruption) → StalePhase. The Err must carry the partial Recovered
        // so the caller can persist the watermark for the prefix that DID apply.
        let mut sink = FakeWalSink::new();
        persist_cycle(
            &mut sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![
                slot(
                    42,
                    1,
                    0,
                    0,
                    Some((KanbanColumn::Planning, KanbanColumn::CognitiveWork)),
                ),
                // Above-watermark landing whose `from` (Planning) mismatches the
                // owner's now-current phase (CognitiveWork) → StalePhase.
                slot(
                    42,
                    1,
                    1,
                    1,
                    Some((KanbanColumn::Planning, KanbanColumn::CognitiveWork)),
                ),
            ],
        )
        .await
        .unwrap();
        let sealed = sink.scan_sealed(None).await.unwrap();
        let mut o = owner(KanbanColumn::Planning);
        let (partial, err) =
            recover_and_apply(&mut o, &sealed, None).expect_err("the mid-chain mismatch must fail");
        assert!(matches!(err, PersistError::StalePhase { .. }));
        assert_eq!(
            partial.applied.len(),
            1,
            "step 0 applied before the failure"
        );
        assert_eq!(
            partial.watermark,
            Some(0),
            "the returned watermark covers exactly the applied prefix (position 0)",
        );
        assert_eq!(
            o.phase(),
            KanbanColumn::CognitiveWork,
            "the owner's phase advanced for the applied prefix — the reason the \
             partial watermark must be persistable",
        );
    }

    // ── FALSIFIER: the watermark spans cycles (stream_position is cross-cycle) ───
    #[tokio::test]
    async fn recovery_watermark_spans_multiple_sealed_cycles() {
        // stream_position is monotonic per owner ACROSS cycles: cycle 1 carries
        // position 0, cycle 2 carries position 1. Recovery over the multi-cycle
        // scan_sealed stream must treat the watermark as cross-cycle.
        let mut sink = FakeWalSink::new();
        persist_cycle(
            &mut sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![slot(
                42,
                1,
                0,
                0,
                Some((KanbanColumn::Planning, KanbanColumn::CognitiveWork)),
            )],
        )
        .await
        .unwrap();
        persist_cycle(
            &mut sink,
            CycleFrame::new(CycleId(2), DatasetVersion(1)),
            vec![slot(
                42,
                2,
                1,
                1,
                Some((KanbanColumn::CognitiveWork, KanbanColumn::Evaluation)),
            )],
        )
        .await
        .unwrap();
        let sealed = sink.scan_sealed(None).await.unwrap();
        assert_eq!(sealed.len(), 2, "landings from BOTH sealed cycles");

        // A fresh recovery applies the whole cross-cycle chain; watermark = 1.
        let mut o = owner(KanbanColumn::Planning);
        let rec = recover_and_apply(&mut o, &sealed, None).unwrap();
        assert_eq!(
            rec.applied.iter().map(|m| m.to).collect::<Vec<_>>(),
            vec![KanbanColumn::CognitiveWork, KanbanColumn::Evaluation],
            "both cycles' steps applied in cross-cycle stream order",
        );
        assert_eq!(
            rec.watermark,
            Some(1),
            "watermark reaches the cycle-2 landing"
        );

        // A watermark from cycle 1 (position 0) skips ONLY cycle 1, still applies
        // cycle 2 — proving the watermark is not per-cycle.
        let mut o2 = owner(KanbanColumn::CognitiveWork);
        let rec2 = recover_and_apply(&mut o2, &sealed, Some(0)).unwrap();
        assert_eq!(
            rec2.applied.iter().map(|m| m.to).collect::<Vec<_>>(),
            vec![KanbanColumn::Evaluation],
            "cycle-1 landing skipped by watermark 0; cycle-2 landing still applied",
        );
    }

    // ── FALSIFIER: a cast for the wrong cycle is a PERMANENT error, not Write ────
    #[tokio::test]
    async fn a_cast_for_the_wrong_cycle_is_a_permanent_cycle_mismatch_not_write() {
        // A cast whose cycle != frame.cycle is a caller programming error — it must
        // NOT surface as the RETRYABLE Write class (retry would loop forever).
        let mut sink = FakeWalSink::new();
        let r = persist_cycle(
            &mut sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![slot(42, 2, 0, 0, None)], // cast cycle 2 ≠ frame cycle 1
        )
        .await;
        assert!(
            matches!(
                r,
                Err(PersistError::CycleMismatch {
                    cast_cycle: CycleId(2),
                    frame_cycle: CycleId(1),
                }),
            ),
            "wrong-cycle cast is CycleMismatch (permanent), never Write (retryable)",
        );
        assert_eq!(
            sink.wal_writes(),
            0,
            "nothing written on a wrong-cycle cast"
        );
    }

    #[tokio::test]
    async fn a_cross_owner_move_is_rejected_before_the_cycle_persists() {
        let bad = SweepSlot {
            paired_move: Some(mv(99, KanbanColumn::Planning, KanbanColumn::CognitiveWork)),
            ..slot(42, 1, 0, 0, None)
        };
        let mut sink = FakeWalSink::new();
        assert!(matches!(
            persist_cycle(
                &mut sink,
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
        let mut sink = FakeWalSink::failing();
        let r = persist_cycle(
            &mut sink,
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
        assert!(matches!(r, Err(PersistError::Commit(CommitError::Io(_)))));
        assert_eq!(sink.wal_writes(), 0, "no commit, no version, no step");
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

    // ── FALSIFIER (Phase A): no artifact-backed delta → zero store operations ────
    #[tokio::test]
    async fn intent_only_cycle_is_nochange_zero_store_calls() {
        let mut sink = FakeWalSink::new();
        // A cycle of PURE kanban movement: every cast is intent-only (empty
        // payload — the restage_held shape). Kanban never decides to persist.
        let intents: Vec<SweepSlot> = (0..5u64)
            .map(|i| SweepSlot {
                payload: Vec::new(),
                ..slot(
                    42,
                    1,
                    i,
                    i,
                    Some((KanbanColumn::Planning, KanbanColumn::CognitiveWork)),
                )
            })
            .collect();
        let out = persist_cycle(
            &mut sink,
            CycleFrame::new(CycleId(1), DatasetVersion(7)),
            intents,
        )
        .await
        .unwrap();
        assert_eq!(
            out,
            CommitOutcome::NoChange {
                head: DatasetVersion(7)
            },
            "pure movement is NoChange with the unchanged head"
        );
        assert_eq!(sink.wal_writes(), 0, "zero store operations");
        assert_eq!(sink.version_count(), 0, "zero rows, zero frames");
        // And a MIXED cycle persists ONLY its artifact casts.
        let mixed = vec![
            SweepSlot {
                payload: Vec::new(),
                ..slot(
                    42,
                    2,
                    10,
                    1,
                    Some((KanbanColumn::Planning, KanbanColumn::CognitiveWork)),
                )
            },
            slot(42, 2, 11, 2, None), // artifact (non-empty payload)
        ];
        let out = persist_cycle(
            &mut sink,
            CycleFrame::new(CycleId(2), DatasetVersion(0)),
            mixed,
        )
        .await
        .unwrap();
        assert!(matches!(out, CommitOutcome::Committed { .. }));
        let sealed = sink.scan_sealed(None).await.unwrap();
        assert_eq!(sealed.len(), 1, "only the artifact cast persisted");
        assert_eq!(sealed[0].slot.stream_position, 11);
    }

    // ── FALSIFIER (Phase A): retrying the same frozen batch reconciles ───────────
    #[tokio::test]
    async fn retrying_the_same_batch_reconciles_never_duplicates() {
        let mut sink = FakeWalSink::new();
        let casts = || vec![slot(42, 1, 0, 0, None), slot(42, 1, 1, 1, None)];
        let first = persist_cycle(
            &mut sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            casts(),
        )
        .await
        .unwrap();
        assert!(matches!(first, CommitOutcome::Committed { .. }));
        // The acknowledgement was "lost"; the caller re-submits the SAME batch.
        let retry = persist_cycle(
            &mut sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            casts(),
        )
        .await
        .unwrap();
        assert!(
            matches!(
                retry,
                CommitOutcome::Reconciled {
                    cycle: CycleId(1),
                    ..
                }
            ),
            "the retry reconciles instead of appending twice: {retry:?}"
        );
        assert_eq!(sink.wal_writes(), 1, "exactly one physical commit");
        assert_eq!(
            sink.scan_sealed(None).await.unwrap().len(),
            2,
            "no duplicate landings"
        );
    }

    // ── FALSIFIER (Phase A): a conflicting batch for a committed cycle fails closed
    #[tokio::test]
    async fn a_different_batch_for_a_committed_cycle_fails_closed() {
        let mut sink = FakeWalSink::new();
        persist_cycle(
            &mut sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![slot(42, 1, 0, 0, None)],
        )
        .await
        .unwrap();
        // Same cycle identity, DIFFERENT content — corruption/conflict, never promoted.
        let conflict = persist_cycle(
            &mut sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            vec![slot(42, 1, 5, 5, None)],
        )
        .await;
        assert!(
            matches!(
                conflict,
                Err(PersistError::Commit(CommitError::HashConflict {
                    cycle: CycleId(1),
                    ..
                }))
            ),
            "{conflict:?}"
        );
        assert_eq!(sink.wal_writes(), 1, "the conflicting batch wrote nothing");
    }

    // ── FALSIFIER (Phase A): completion order never reaches the batch identity ───
    #[test]
    fn randomized_completion_order_yields_the_same_batch_hash() {
        let frame = CycleFrame::new(CycleId(3), DatasetVersion(2));
        let ordered = vec![
            slot(1, 3, 0, 0, None),
            slot(2, 3, 1, 1, None),
            slot(3, 3, 2, 2, None),
        ];
        let scrambled = vec![
            slot(3, 3, 2, 2, None),
            slot(1, 3, 0, 0, None),
            slot(2, 3, 1, 1, None),
        ];
        let a = DetachedCycleBatch::freeze(frame, ordered);
        let b = DetachedCycleBatch::freeze(frame, scrambled);
        assert_eq!(
            a.batch_hash, b.batch_hash,
            "identity comes from canonical content"
        );
        assert_eq!(a.landings, b.landings, "identical canonical landings");
        // And DIFFERENT content yields a different hash (the hash discriminates).
        let c = DetachedCycleBatch::freeze(frame, vec![slot(1, 3, 0, 9, None)]);
        assert_ne!(a.batch_hash, c.batch_hash);
    }

    // ── FALSIFIER (Phase A): the recovery tail bound excludes earlier cycles ─────
    #[tokio::test]
    async fn after_cycle_bound_limits_the_sealed_scan() {
        let mut sink = FakeWalSink::new();
        for c in 1..=3u64 {
            persist_cycle(
                &mut sink,
                CycleFrame::new(CycleId(c), DatasetVersion(c - 1)),
                vec![slot(42, c, c, c, None)],
            )
            .await
            .unwrap();
        }
        let tail = sink.scan_sealed(Some(CycleId(1))).await.unwrap();
        assert_eq!(
            tail.iter().map(|l| l.cycle).collect::<Vec<_>>(),
            vec![CycleId(2), CycleId(3)],
            "strictly after the bound — bounded recovery, not full history"
        );
    }
}
