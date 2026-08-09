//! P4 — the cycle loop-closure driver (supervisor-side).
//!
//! This is the seam that makes the merged `persist_sink` cycle/WAL bootstrap
//! **load-bearing**. Before this, `persist_sink::persist_cycle` had zero
//! production callers. The driver composes the already-shipped organs — it mints
//! **no** new semantic / temporal / rung / witness type:
//!
//! - **P4a** ([`collect_casts`] + [`seal_cycle`]): drain the fleet's staged
//!   planner casts into one `Vec<SweepSlot>`, `persist_cycle` them → **exactly
//!   one WAL write → one `DatasetVersion`**, and read out the **sparse** sealed
//!   paired-transition set (only the owners that actually cast a move).
//! - **P4b** ([`apply_sealed_transitions`]): iterate **only** the sealed sparse
//!   transitions, resolve each owner in the fleet, apply **one** legal
//!   `try_advance_phase` AND advance the owner's durable **recovery watermark**
//!   in the same pass. **Every unrepresented owner is left byte-identical.**
//!
//! ## The load-bearing rule (E-D-MBX-SPINE-IS-STRAIGHT-TRACK-VERSION-IS-NOT-A-FLEET-STEP-SIGNAL-1)
//!
//! A `DatasetVersion` is **global knowledge**, NOT permission to advance every
//! mailbox. Kanban mutation is **sparse and owner-specific**: an owner advances
//! only because it produced a sealed `paired_move`. The version tick never fans a
//! step across the fleet. Applying only the sealed sparse set is what keeps a
//! globally-complete cycle physically sparse
//! (`E-COMPLETE-CYCLE-IS-PHYSICALLY-SPARSE-NOT-A-FULL-REWRITE-1`).
//!
//! ## Failure semantics (authoritative, operator-ruled) — TWO separate mechanisms
//!
//! **Pre-commit failure (no `Vn+1` exists): deterministic regeneration is the
//! correctness mechanism.** The authoritative rule:
//!
//! ```text
//! sealed Vn + unchanged Kanban task + deterministic computation
//!     = the same provisional intent on the next sweep
//! ```
//!
//! So when a commit fails before `Vn+1` exists: publish nothing · mutate no
//! owner · advance no watermark · **discard** provisional slots, held moves and
//! planning results · rerun the unchanged Kanban task from `Vn` — the same
//! semantic sparse cycle regenerates. Correctness NEVER requires retaining the
//! first heap batch: [`SealFailure::casts`] is an **optional retry cache** (an
//! implementation convenience that skips the re-stage pass), never the source
//! of truth and never a provisional-planning ledger. Discarding it is always
//! sound; the regeneration falsifier pins this.
//!
//! **Post-commit interruption (`Vn+1` EXISTS): committed-history recovery.**
//! [`recover_fleet`] replays already-SEALED landings whose application (or a
//! restart) was interrupted, idempotent over per-owner watermarks. It is for
//! committed history ONLY — never a substitute for, or conceptually grouped
//! with, ordinary pre-commit write failure. The two mechanisms share no state.
//!
//! ## The ≤1-transition-per-owner rule is enforced BEFORE sealing
//!
//! [`collect_casts`] partitions: the **first** intent move per owner (in cast
//! order) seals as that owner's `paired_move`; **every further move** — a second
//! move in the same cast, or a later same-owner cast — is returned as a
//! [`HeldIntent`] for the caller to re-stage into a FUTURE cycle
//! ([`restage_held`]). Nothing is sealed and then discarded: the sealed
//! transition set and the normally-applied transition set are the **same set**,
//! so crash recovery and normal operation agree. (`AppliedCycle::deferred`
//! survives only as a defence-in-depth counter for sealed inputs produced
//! elsewhere; cycles sealed through [`collect_casts`] keep it at 0.)
//!
//! ## Ownership (D-MBX spine) — scope honesty
//!
//! The supervisor crate is the **runtime-owner side** of the spine; the planner
//! **decides** (produces the casts + the persistence contract). This driver
//! depends **one-way** on the planner (planner never deps supervisor — no
//! cycle). Per the ratified sparse-cycle ruling it applies **writer-fires-inline**
//! through `MailboxSoaOwner::try_advance_phase` — no ractor message bus, no
//! dataset re-read, NOT a 64k async `drive_once` fan.
//!
//! **What [`MailboxFleet`] is:** the driver's owner-resolution seam, with its
//! `HashMap` impl as the keyed fleet the driver and its tests use. This
//! sealed-cycle path (#879) is the complete and independent production
//! phase-progression path — there is no actor bridge waiting to be completed.
//! (The `kanban_actor` module's actor/ack/tick surface was DELETED 2026-08-05,
//! `E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1`; what remains there is the
//! message-free visibility census + pure helpers this driver composes.)
//!
//! ## Honesty ledger (what is proven vs not)
//!
//! - **Control-loop contract: proven** (this module's falsifiers, over fakes).
//! - **cognitive-shader-driver / MailboxSoA thought: NOT proven** — the P4c
//!   thought body here is the real **MUL gate** ([`shade_owner`] =
//!   `gate_decision_i4` + `advance_on_gate`), but its qualia/mantissa inputs
//!   come from a caller-supplied extractor, NOT from a live `MailboxSoA` /
//!   shader-driver dispatch. A MedCare thought is real only when that path runs.
//! - **Durability: fake** — the WAL leg is the contract-probe `WalSink` fake
//!   until the concrete `LanceShardSink` lands (`compile+test green ≠ storage
//!   proven`, the Ladybug rule).

use std::collections::{HashMap, HashSet};

use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
use lance_graph_contract::mul::i4_eval::gate_decision_i4;
use lance_graph_contract::scheduler::DatasetVersion;
use lance_graph_contract::soa_view::{MailboxSoaOwner, MailboxSoaView};
use lance_graph_contract::QualiaI4_16D;

use lance_graph_planner::batch_writer::BatchWriter;
use lance_graph_planner::owner_adapter::emit_bootstrap_intent;
use lance_graph_planner::persist_sink::{
    persist_cycle, recover_and_apply, CommitOutcome, CycleFrame, CycleId, LandedSlot, PersistError,
    SweepSlot, WalSink,
};
use lance_graph_planner::traits::StrategyOutcome;

/// One sealed paired transition — a member of the **sparse** set a sealed cycle
/// publishes. Carries `stream_position` so P4b applies transitions in canonical
/// order and advances the recovery watermark. Boring by design: it says *which
/// owner advances where*, nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SealedTransition {
    /// The canonical durable order key (see [`collect_casts`]'s
    /// `position_base` contract — restart-stable, NOT a raw `CastId`).
    pub stream_position: u64,
    /// The owner that produced this transition (== `mv.mailbox`).
    pub owner: MailboxId,
    /// The lifecycle move the owner cast.
    pub mv: KanbanMove,
}

/// P4a output — one sealed cycle: the single published `DatasetVersion` and the
/// **sparse** sealed paired-transition set. `transitions` is empty for a no-op
/// cycle (no owner cast a move); its length is ≤ the number of participants and
/// is typically ≪ the fleet size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SealedCycle {
    /// The raw commit result — the honest source of truth. Under the
    /// governing storage rule a cycle with zero artifact casts is
    /// [`CommitOutcome::NoChange`] and publishes nothing.
    pub outcome: CommitOutcome,
    /// The version this cycle sealed into, or `None` for
    /// [`CommitOutcome::NoChange`] (nothing published, head unchanged).
    /// Derived from [`Self::outcome`] at construction — kept as a plain field
    /// (not a method) so existing `sealed.version` call sites stay readable;
    /// `Some` for [`CommitOutcome::Committed`] / [`CommitOutcome::Reconciled`].
    pub version: Option<DatasetVersion>,
    /// Only the owners that cast a `paired_move` — the sparse subset.
    /// With the pre-seal ≤1-per-owner partition, at most one per owner.
    /// **Computed over ALL collected casts, not just artifact ones** — an
    /// intent-only (empty-payload) cast's move still seals here even when the
    /// cycle as a whole is [`CommitOutcome::NoChange`]: kanban movement is
    /// ephemeral in the STORE, never in the in-memory fleet (see
    /// [`restage_held`] and the governing storage rule in `persist_sink`).
    pub transitions: Vec<SealedTransition>,
    /// The first unused stream position AFTER this cycle, computed over **all**
    /// sealed slots (including no-move landings). The caller's next
    /// `position_base` is `max(previous_base, next_position_base)` — carrying
    /// the previous base forward covers the empty-cycle case (where this is 0).
    pub next_position_base: u64,
}

/// A seal that failed at the WAL. **No owner was mutated, no version published,
/// no watermark advanced** (apply never ran).
///
/// **Classification (module docs § Failure semantics): the retained `casts` are
/// an OPTIONAL retry cache, NOT the correctness mechanism.** The authoritative
/// pre-commit rule is deterministic regeneration: discard everything and rerun
/// the unchanged Kanban task from the sealed `Vn` — the same semantic sparse
/// cycle regenerates (sealed `Vn` + unchanged task + deterministic computation
/// = the same provisional intent). Callers that keep a hot loop MAY resubmit
/// `casts` via [`seal_cycle`] to skip the re-stage pass; callers that simply
/// drop this value are equally correct. This must never grow into a
/// provisional-planning ledger.
///
/// **`cause`'s honest sub-states** (routed from `persist_sink::CommitError`
/// via `PersistError::Commit`): [`CommitError::Fenced`](lance_graph_planner::persist_sink::CommitError::Fenced)
/// and [`CommitError::Io`](lance_graph_planner::persist_sink::CommitError::Io) both mean nothing was published —
/// regenerating from the unchanged `Vn` is always safe.
/// [`CommitError::Ambiguous`](lance_graph_planner::persist_sink::CommitError::Ambiguous) means the outcome is UNKNOWN — do NOT
/// regenerate a fresh cycle; re-submit the SAME frozen `casts` (via
/// [`seal_cycle`]) and let `commit_cycle`'s reconciliation-first lookup
/// decide. [`CommitError::HashConflict`](lance_graph_planner::persist_sink::CommitError::HashConflict) is a fail-closed corruption/identity
/// conflict — never promoted, never overwritten; this driver mints no new
/// recovery machinery for it, it simply surfaces the error.
#[derive(Debug)]
pub struct SealFailure {
    /// The frame the failed seal was submitted under (regenerate with the same
    /// one, or a refreshed `base_version` after a fence conflict).
    pub frame: CycleFrame,
    /// Optional retry cache (see type docs) — safe to discard; regeneration
    /// from `Vn` is the prescribed path.
    pub casts: Vec<SweepSlot>,
    /// Why the WAL write failed.
    pub cause: PersistError,
}

/// P4b output — the effect of applying a sealed cycle's sparse transition set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedCycle {
    /// The version whose sealed transitions were applied — mirrors
    /// [`SealedCycle::version`]: `None` for a [`CommitOutcome::NoChange`]
    /// cycle (an all-intent-only cycle can still apply sparse transitions to
    /// the in-memory fleet even though nothing was published to the store).
    pub version: Option<DatasetVersion>,
    /// One move per **advanced** owner (distinct owners; ≤1 per cycle).
    pub applied: Vec<KanbanMove>,
    /// Defence-in-depth counter: same-owner extras in a sealed input NOT
    /// produced by [`collect_casts`] (which enforces ≤1/owner pre-seal, so
    /// cycles sealed through it keep this at 0). A non-zero value means the
    /// sealed set and the applied set diverge — investigate the producer.
    pub deferred: usize,
    /// Sealed transitions whose owner is not registered in the fleet (a
    /// registration gap — surfaced as a count, not a crash; the durable move
    /// stays in the log and is replayed by [`recover_fleet`] once the owner
    /// registers).
    pub missing: usize,
}

/// An intent move held back by the pre-seal ≤1-per-owner partition — NOT sealed
/// this cycle; re-stage it into a future cycle via [`restage_held`].
///
/// **Scheduling convenience within a SUCCESSFUL cycle, not a durable ledger:**
/// on a failed seal, held intents are discarded together with everything else
/// (the pre-commit rule is deterministic regeneration from `Vn`, module docs
/// § Failure semantics — [`run_cycle`]'s seal-error path drops them). Even on
/// success, a dropped held intent is regenerated by the owner's next thought
/// pass; retaining it merely saves that pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeldIntent {
    /// The owner whose extra move is held.
    pub owner: MailboxId,
    /// The move to re-stage next cycle.
    pub mv: KanbanMove,
}

/// P4a (drain) output: the slots to seal this cycle + the intents held for a
/// future cycle by the ≤1-per-owner partition.
#[derive(Debug)]
pub struct CollectedCasts {
    /// One landing per drained cast (payloads always land; at most the first
    /// per-owner move is paired).
    pub slots: Vec<SweepSlot>,
    /// Every intent move beyond the first per owner — nothing is silently
    /// truncated; re-stage via [`restage_held`].
    pub held: Vec<HeldIntent>,
}

/// A resolvable collection of mailbox owners — the driver's owner-resolution
/// seam. P4b resolves each sealed transition's owner through this;
/// **unrepresented owners are never resolved and never touched** (that is what
/// keeps them byte-identical).
///
/// The `HashMap` impl below is the driver's keyed fleet (order-free access;
/// cross-mailbox ordering is the read side's job, per the temporal contract).
pub trait MailboxFleet {
    /// The concrete owner type this fleet holds.
    type Owner: MailboxSoaOwner;
    /// Resolve a mailbox to its owner (read-only) — used by P4c's thought body.
    fn owner(&self, id: MailboxId) -> Option<&Self::Owner>;
    /// Resolve a mailbox to its owner for mutation, or `None` if not registered.
    fn owner_mut(&mut self, id: MailboxId) -> Option<&mut Self::Owner>;
}

/// A `HashMap` keyed by `MailboxId` — the driver's keyed fleet (order-free
/// access).
impl<O: MailboxSoaOwner> MailboxFleet for HashMap<MailboxId, O> {
    type Owner = O;
    fn owner(&self, id: MailboxId) -> Option<&O> {
        self.get(&id)
    }
    fn owner_mut(&mut self, id: MailboxId) -> Option<&mut O> {
        self.get_mut(&id)
    }
}

/// **P4a (drain).** Map the fleet's staged `BatchWriter` casts into `SweepSlot`s
/// for one cycle, enforcing the **≤1-move-per-owner** rule BEFORE sealing.
///
/// - `stream_position = position_base + CastId` — **`position_base` is the
///   caller's durable cursor**, NOT 0 after a restart. `CastId` alone is a
///   resettable in-memory counter (a reconstructed writer restarts it at 0),
///   while `stream_position` is the cross-cycle recovery watermark: a position
///   that ever repeats below an owner's watermark is silently skipped by
///   `recover_and_apply`. Take `position_base` from durable state — the
///   previous [`SealedCycle::next_position_base`] (max'd with the prior base)
///   or a scan of the sealed log. The restart falsifier pins this.
/// - The first intent move per owner (cast order) becomes that owner's
///   `paired_move`. **Every further move** — same cast or a later same-owner
///   cast — is returned in [`CollectedCasts::held`], never silently dropped
///   and never sealed-then-ignored.
/// - Casts with no move are no-step landings (`paired_move = None`) — they
///   coalesce into the cycle image but contribute no transition.
///
/// Draining clears the writer's payload staging.
#[must_use]
pub fn collect_casts(
    writer: &mut BatchWriter<Vec<u8>>,
    cycle: CycleId,
    position_base: u64,
    mut row_of: impl FnMut(MailboxId) -> u64,
) -> CollectedCasts {
    // Drain first (ends the &mut borrow), then read the intent moves immutably.
    let drained: Vec<_> = writer.drain_pending_payloads().collect();
    let mut paired_owners: HashSet<MailboxId> = HashSet::new();
    let mut slots = Vec::with_capacity(drained.len());
    let mut held = Vec::new();
    for (cast, payload) in drained {
        let Some(owner) = writer.on_behalf_of(cast) else {
            continue;
        };
        let mut paired_move = None;
        for &mv in writer.intent_moves(cast).unwrap_or(&[]) {
            if paired_move.is_none() && !paired_owners.contains(&owner) {
                paired_move = Some(mv);
                paired_owners.insert(owner);
            } else {
                // ≤1/owner/cycle, enforced pre-seal: the extra move is HELD for
                // a future cycle — not sealed, not discarded, not truncated.
                held.push(HeldIntent { owner, mv });
            }
        }
        slots.push(SweepSlot {
            cycle,
            stream_position: position_base + cast.0,
            owner,
            row: row_of(owner),
            paired_move,
            payload,
        });
    }
    CollectedCasts { slots, held }
}

/// Re-stage held intents ([`CollectedCasts::held`]) into the writer for the
/// NEXT cycle. The re-cast is intent-only (empty payload — the original cast's
/// payload already sealed with its cycle). Returns the number re-staged.
///
/// **This empty payload is precisely what makes the re-staged intent
/// EPHEMERAL under the governing storage rule** (`persist_sink`'s artifact
/// gate): an intent-only cast never trips the payload gate and never reaches
/// the store, so no held-intent re-stage can ever accidentally become a
/// persisted artifact.
pub fn restage_held(writer: &mut BatchWriter<Vec<u8>>, held: Vec<HeldIntent>) -> usize {
    let n = held.len();
    for h in held {
        writer.cast(h.owner, vec![h.mv], Vec::new());
    }
    n
}

/// **P4a (seal).** Freeze the collected casts into one cycle: read out the
/// **sparse** sealed paired-transition set (the slots that carry a `paired_move`,
/// ordered by `stream_position`), then `persist_cycle` → **exactly one WAL write,
/// one `DatasetVersion`**. `persist_cycle`'s own guards still run
/// (cross-owner-move reject, cast/frame cycle-match). The version is published
/// before any owner advances (P4b).
///
/// **On WAL failure:** nothing is published, no owner is mutated, no watermark
/// advances. The prescribed recovery is **deterministic regeneration from the
/// unchanged `Vn`** (module docs § Failure semantics); the [`SealFailure`]
/// carries the frozen casts only as an optional retry cache (dropping it and
/// re-staging is equally correct — the regeneration falsifier proves the same
/// semantic cycle re-derives). The one clone held until commit success is the
/// price of offering that cache.
///
/// **`sink: &mut S`** — `persist_cycle`'s `WalSink::commit_cycle` is the
/// one-logical-writer boundary (`&mut self`); this function's own
/// `PersistError::Commit` sub-states stay the honest routing they always
/// were: `CommitError::Fenced` / `CommitError::Io` mean nothing published
/// (regenerate from `Vn`); `CommitError::Ambiguous` means UNKNOWN (re-submit
/// the SAME frozen `casts` — reconciliation decides); `CommitError::HashConflict`
/// is fail closed (never promoted, never overwritten).
pub async fn seal_cycle<S: WalSink>(
    sink: &mut S,
    frame: CycleFrame,
    casts: Vec<SweepSlot>,
) -> Result<SealedCycle, Box<SealFailure>> {
    // Read the transitions + next base out BEFORE `persist_cycle` takes ownership.
    // Computed over ALL collected casts (artifact AND intent-only) — the
    // in-memory fleet step is not gated by the store's artifact filter.
    let mut transitions: Vec<SealedTransition> = casts
        .iter()
        .filter_map(|s| {
            s.paired_move.map(|mv| SealedTransition {
                stream_position: s.stream_position,
                owner: s.owner,
                mv,
            })
        })
        .collect();
    transitions.sort_by_key(|t| t.stream_position);
    let next_position_base = casts
        .iter()
        .map(|s| s.stream_position + 1)
        .max()
        .unwrap_or(0);
    // Held until commit success — the price of a byte-identical retry.
    let frozen = casts.clone();
    match persist_cycle(sink, frame, casts).await {
        Ok(outcome) => {
            let version = match outcome {
                CommitOutcome::NoChange { .. } => None,
                CommitOutcome::Committed { version, .. } => Some(version),
                // `current_head` is the store head AT RECONCILIATION TIME, not
                // the publication version this cycle originally committed at.
                CommitOutcome::Reconciled { current_head, .. } => Some(current_head),
            };
            Ok(SealedCycle {
                outcome,
                version,
                transitions,
                next_position_base,
            })
        }
        Err(cause) => Err(Box::new(SealFailure {
            frame,
            casts: frozen,
            cause,
        })),
    }
}

/// **P4b.** Apply **only** the sealed sparse transition set to the fleet, and
/// advance each applied owner's durable **recovery watermark** in the same
/// pass — the phase transition and the applied-through watermark move together,
/// so a crash after normal apply does NOT cause [`recover_fleet`] to replay it.
///
/// Iterate the transitions in canonical `stream_position` order; resolve each
/// owner and apply **one** legal `try_advance_phase`. **Every unrepresented
/// owner is left byte-identical** — never resolved, never touched.
///
/// A sealed move whose `from` no longer matches the owner's phase is corruption
/// ([`PersistError::StalePhase`]); a cross-owner move is
/// [`PersistError::OwnerMismatch`] (defence in depth — `persist_cycle` already
/// rejected it). On such an error the **applied prefix is returned**
/// (`Err((partial, cause))`, mirroring `recover_and_apply`'s contract) with its
/// watermarks already advanced — the caller persists the prefix's watermarks and
/// re-drives the tail from durable state.
///
/// Reads **no** dataset: the version was sealed by [`seal_cycle`]; this is a pure
/// in-memory fan over the sparse set (no `scan_sealed`, no `versions`, no
/// `drive_once`).
pub fn apply_sealed_transitions<F: MailboxFleet>(
    fleet: &mut F,
    sealed: &SealedCycle,
    watermarks: &mut HashMap<MailboxId, Option<u64>>,
) -> Result<AppliedCycle, (AppliedCycle, PersistError)> {
    let mut ordered: Vec<&SealedTransition> = sealed.transitions.iter().collect();
    ordered.sort_by_key(|t| t.stream_position);

    let mut applied = Vec::new();
    let mut advanced: HashSet<MailboxId> = HashSet::new();
    let mut deferred = 0usize;
    let mut missing = 0usize;

    let partial = |applied: Vec<KanbanMove>, deferred, missing| AppliedCycle {
        version: sealed.version,
        applied,
        deferred,
        missing,
    };

    for t in ordered {
        // Defence-in-depth: collect_casts enforces ≤1/owner pre-seal, so this
        // only fires for sealed inputs produced elsewhere.
        if advanced.contains(&t.owner) {
            deferred += 1;
            continue;
        }
        let Some(owner) = fleet.owner_mut(t.owner) else {
            missing += 1;
            continue;
        };
        // Defence in depth (persist_cycle already validated at seal time).
        if t.mv.mailbox != t.owner {
            return Err((
                partial(applied, deferred, missing),
                PersistError::OwnerMismatch {
                    move_owner: t.mv.mailbox,
                    landing_owner: t.owner,
                },
            ));
        }
        // Corruption guard: the sealed move's `from` must match the owner's phase.
        if t.mv.from != owner.phase() {
            return Err((
                partial(applied, deferred, missing),
                PersistError::StalePhase {
                    owner_phase: owner.phase(),
                    move_from: t.mv.from,
                },
            ));
        }
        match owner.try_advance_phase(t.mv.to) {
            Ok(step) => {
                applied.push(step);
                advanced.insert(t.owner);
                // Phase transition + watermark advance TOGETHER (one owner state).
                let wm = watermarks.entry(t.owner).or_insert(None);
                if wm.is_none_or(|w| w < t.stream_position) {
                    *wm = Some(t.stream_position);
                }
            }
            Err(e) => {
                return Err((
                    partial(applied, deferred, missing),
                    PersistError::Illegal(e),
                ));
            }
        }
    }

    Ok(partial(applied, deferred, missing))
}

/// The full effect of one driven cycle ([`run_cycle`]).
#[derive(Debug)]
pub struct CycleOutcome {
    /// The sealed cycle (version + sparse transitions + next position base).
    pub sealed: SealedCycle,
    /// The applied effect (advanced owners + counters), watermarks advanced.
    pub applied: AppliedCycle,
    /// Intents held by the ≤1/owner partition — re-stage via [`restage_held`].
    pub held: Vec<HeldIntent>,
}

/// A [`run_cycle`] failure.
#[derive(Debug)]
pub enum CycleError {
    /// The WAL commit failed — **nothing published, no owner mutated, no
    /// watermark advanced**. Held intents from this pass were discarded.
    /// Prescribed recovery: rerun the unchanged Kanban task from `Vn`
    /// (deterministic regeneration, module docs § Failure semantics); the
    /// boxed [`SealFailure`] is an optional retry cache only.
    Seal(Box<SealFailure>),
    /// A guard tripped mid-apply — the applied prefix (with its watermarks
    /// already advanced) is preserved; re-drive the tail via [`recover_fleet`].
    Apply {
        /// The applied prefix before the error.
        partial: AppliedCycle,
        /// The guard that tripped.
        cause: PersistError,
    },
}

/// Convenience: run one full cycle — **P4a** (drain the writer + seal) then
/// **P4b** (apply the sparse set + advance watermarks) — as ONE sequential
/// call.
///
/// **Borrow honesty (operator-flagged):** this signature holds the exclusive
/// `&mut F` fleet borrow for the ENTIRE future — including across the seal's
/// storage `.await` — because a future captures its parameters for its whole
/// lifetime regardless of when they are dereferenced. A caller awaiting
/// `run_cycle` therefore cannot touch the fleet concurrently with storage
/// I/O. This makes `run_cycle` the SEQUENTIAL convenience (tests, probes,
/// single-task loops). The PRODUCTION detached path is the split the pieces
/// already expose: `collect_casts` (writer only) → drop the fleet borrow →
/// `seal_cycle(&mut sink, …).await` (sink only, NO fleet parameter) →
/// re-acquire the fleet → `apply_sealed_transitions` (fleet only, no sink).
/// Unrelated thoughts keep computing during the await because nothing they
/// need is borrowed.
///
/// `position_base` is the durable stream cursor (see [`collect_casts`]);
/// `watermarks` is the fleet's per-owner recovery watermark map, advanced in
/// place alongside the phases. On [`CycleError::Seal`] the frozen cycle is
/// retryable; on [`CycleError::Apply`] the applied prefix is preserved.
///
/// **Borrow note (operator-ruled): `fleet: &mut F` is not touched across the
/// seal's `.await`.** The parameter's lifetime spans the whole function, but
/// the body only reads/writes through it in [`apply_sealed_transitions`],
/// AFTER `seal_cycle`'s I/O has already completed — the exclusive fleet
/// borrow is effectively taken post-I/O, not held live across the WAL commit.
pub async fn run_cycle<S, F>(
    sink: &mut S,
    fleet: &mut F,
    writer: &mut BatchWriter<Vec<u8>>,
    frame: CycleFrame,
    position_base: u64,
    watermarks: &mut HashMap<MailboxId, Option<u64>>,
    row_of: impl FnMut(MailboxId) -> u64,
) -> Result<CycleOutcome, CycleError>
where
    S: WalSink,
    F: MailboxFleet,
{
    let collected = collect_casts(writer, frame.cycle, position_base, row_of);
    // `fleet` is untouched up to and including this await — the seal's I/O
    // runs before the fleet is ever resolved (see the borrow note above).
    let sealed = seal_cycle(sink, frame, collected.slots)
        .await
        .map_err(CycleError::Seal)?;
    match apply_sealed_transitions(fleet, &sealed, watermarks) {
        Ok(applied) => Ok(CycleOutcome {
            sealed,
            applied,
            held: collected.held,
        }),
        Err((partial, cause)) => Err(CycleError::Apply { partial, cause }),
    }
}

/// P4c output — what a cognitive pass did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CognitiveWorkOutcome {
    /// Next-cycle intents cast (one per owner whose thought produced one).
    pub cast: usize,
    /// Owners evaluated this pass that produced NO cast (a gate `Hold`, a
    /// declined/unfinished thought, or a non-sentinel outcome). **A Hold is a
    /// reschedule, not a strand:** feed these back into
    /// [`run_cognitive_work_over`] / [`run_cognitive_work_gated_over`] on a
    /// later cycle so a Held owner is evaluated again.
    pub held_owners: Vec<MailboxId>,
}

/// Shared body: evaluate the thought seam over an explicit owner set. Only
/// owners currently in `CognitiveWork` are evaluated; each either casts a
/// next-cycle intent (via `owner_adapter::emit_bootstrap_intent`) or lands in
/// `held_owners` for a later pass.
fn cognitive_pass<F>(
    fleet: &F,
    owners: impl IntoIterator<Item = MailboxId>,
    writer: &mut BatchWriter<Vec<u8>>,
    mut think: impl FnMut(&F::Owner) -> Option<(StrategyOutcome, Vec<u8>)>,
) -> CognitiveWorkOutcome
where
    F: MailboxFleet,
{
    let mut cast = 0usize;
    let mut held_owners = Vec::new();
    for id in owners {
        let Some(owner) = fleet.owner(id) else {
            continue;
        };
        if owner.phase() != KanbanColumn::CognitiveWork {
            continue;
        }
        let mut did_cast = false;
        if let Some((outcome, payload)) = think(owner) {
            // owner_adapter rebinds the bootstrap sentinel to this owner + casts
            // it write-on-behalf; a non-sentinel / no-intent outcome stages nothing.
            if emit_bootstrap_intent(
                &outcome,
                owner.mailbox_id(),
                owner.current_cycle(),
                writer,
                payload,
            )
            .is_some()
            {
                did_cast = true;
            }
        }
        if did_cast {
            cast += 1;
        } else {
            held_owners.push(id);
        }
    }
    CognitiveWorkOutcome { cast, held_owners }
}

/// **P4c.** For each owner that ENTERED `CognitiveWork` this cycle (an applied
/// move whose `to` is [`KanbanColumn::CognitiveWork`]), run the pluggable
/// thought body and route its Outcome into the NEXT cycle's casts via
/// `owner_adapter::emit_bootstrap_intent`.
///
/// The thought body is a **seam**: `think` is
/// `FnMut(&Owner) -> Option<(StrategyOutcome, payload)>`. `None` = no next
/// intent this pass — the owner is returned in
/// [`CognitiveWorkOutcome::held_owners`] and MUST be re-evaluated on a later
/// cycle via [`run_cognitive_work_over`] (a rest is a reschedule, never a
/// strand). The Outcome's `intended_move` must be a **bootstrap sentinel**
/// (`mailbox 0`) — `owner_adapter` rebinds it to the live owner (no-theft) and
/// casts it **write-on-behalf**. This never mutates a mailbox (the step is P4b,
/// post-seal); it only stages the next intent.
///
/// `run_cognitive_work` is a sequential contract-probe adapter used to prove
/// the seal→apply→intent roundtrip. It does not define the production
/// execution model.
///
/// Production cognition may run independently and concurrently over the sealed
/// `Vn`. Completed immutable outcomes converge only at the deterministic
/// ordering/coalescing/seal boundary.
pub fn run_cognitive_work<F>(
    fleet: &F,
    applied: &AppliedCycle,
    writer: &mut BatchWriter<Vec<u8>>,
    think: impl FnMut(&F::Owner) -> Option<(StrategyOutcome, Vec<u8>)>,
) -> CognitiveWorkOutcome
where
    F: MailboxFleet,
{
    let entered = applied
        .applied
        .iter()
        .filter(|mv| mv.to == KanbanColumn::CognitiveWork)
        .map(|mv| mv.mailbox)
        .collect::<Vec<_>>();
    cognitive_pass(fleet, entered, writer, think)
}

/// **P4c re-poll.** Evaluate the thought seam over an EXPLICIT owner set — the
/// re-scheduling lane for owners a previous pass returned in
/// [`CognitiveWorkOutcome::held_owners`]. Owners no longer in `CognitiveWork`
/// are skipped.
pub fn run_cognitive_work_over<F>(
    fleet: &F,
    owners: &[MailboxId],
    writer: &mut BatchWriter<Vec<u8>>,
    think: impl FnMut(&F::Owner) -> Option<(StrategyOutcome, Vec<u8>)>,
) -> CognitiveWorkOutcome
where
    F: MailboxFleet,
{
    cognitive_pass(fleet, owners.iter().copied(), writer, think)
}

/// **The MUL-gate plug (P4c gate).** Run the MUL cognitive gate over an owner in
/// `CognitiveWork` and lower the decision to the owner's next intended move.
/// This composes `kanban_actor::mul_target` for the driver — it mints **no**
/// new decision logic, reusing the two shipped contract primitives:
///
/// 1. read the owner's *current* phase,
/// 2. run [`gate_decision_i4`]`(qualia, mantissa)` — the i4 TrustTexture × FlowState
///    gate (`Flow` / `Hold` / `Block`),
/// 3. lower the `GateDecision` to a DAG-legal next phase via
///    [`KanbanColumn::advance_on_gate`] (`Flow` → forward, `Block` →
///    Prune-where-legal, `Hold` → rest).
///
/// **Scope honesty:** this is the real MUL *gate*, NOT the
/// `cognitive-shader-driver` / `MailboxSoA` dispatch path — no SoA columns are
/// read here, and a MedCare thought is proven only when that path runs. The
/// result is a **bootstrap sentinel** [`StrategyOutcome`] (`mailbox 0`,
/// `witness_chain_position 0`) for `owner_adapter` to rebind; `None` = the gate
/// **Holds** (or no legal successor) — the owner rests this pass and is
/// re-scheduled via [`CognitiveWorkOutcome::held_owners`].
///
/// `qualia` + `mantissa` are supplied by the caller because `MailboxSoaView`
/// does not yet expose `qualia()` (deferred — `soa_view.rs` "add `fn qualia`
/// when the first consumer arrives"). P4c is that first consumer; until the
/// trait method lands, a fleet-specific extractor bridges the seam. This keeps
/// the MailboxSoa contract UNCHANGED — no trait redesign.
#[must_use]
pub fn shade_owner<O: MailboxSoaOwner>(
    owner: &O,
    qualia: &QualiaI4_16D,
    mantissa: i8,
    reliability: f32,
) -> Option<StrategyOutcome> {
    let phase = owner.phase();
    let gate = gate_decision_i4(qualia, mantissa);
    let to = phase.advance_on_gate(&gate)?;
    Some(StrategyOutcome {
        reliability,
        intended_move: Some(KanbanMove {
            // bootstrap sentinel — owner_adapter rebinds mailbox 0 to the live owner.
            mailbox: 0,
            from: phase,
            to,
            witness_chain_position: 0,
            exec: ExecTarget::Native,
        }),
    })
}

/// **P4c with the MUL gate wired in.** Like [`run_cognitive_work`], but the
/// thought body IS the MUL cognitive gate ([`shade_owner`]). For each owner that
/// just entered `CognitiveWork`, `read_gate` extracts that owner's `(qualia,
/// signed_mantissa, reliability, payload)` — the qualia seam the deferred
/// `MailboxSoaView::qualia()` will eventually close — and the gate decides. A
/// **Hold** (or a declined `read_gate`) lands the owner in
/// [`CognitiveWorkOutcome::held_owners`] for re-scheduling.
pub fn run_cognitive_work_gated<F>(
    fleet: &F,
    applied: &AppliedCycle,
    writer: &mut BatchWriter<Vec<u8>>,
    mut read_gate: impl FnMut(&F::Owner) -> Option<(QualiaI4_16D, i8, f32, Vec<u8>)>,
) -> CognitiveWorkOutcome
where
    F: MailboxFleet,
{
    run_cognitive_work(fleet, applied, writer, |owner| {
        let (qualia, mantissa, reliability, payload) = read_gate(owner)?;
        let outcome = shade_owner(owner, &qualia, mantissa, reliability)?;
        Some((outcome, payload))
    })
}

/// **P4c gated re-poll.** [`run_cognitive_work_gated`] over an explicit owner
/// set — how a Held owner gets its gate re-evaluated on a later cycle.
pub fn run_cognitive_work_gated_over<F>(
    fleet: &F,
    owners: &[MailboxId],
    writer: &mut BatchWriter<Vec<u8>>,
    mut read_gate: impl FnMut(&F::Owner) -> Option<(QualiaI4_16D, i8, f32, Vec<u8>)>,
) -> CognitiveWorkOutcome
where
    F: MailboxFleet,
{
    run_cognitive_work_over(fleet, owners, writer, |owner| {
        let (qualia, mantissa, reliability, payload) = read_gate(owner)?;
        let outcome = shade_owner(owner, &qualia, mantissa, reliability)?;
        Some((outcome, payload))
    })
}

/// The effect of a [`recover_fleet`] pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FleetRecovery {
    /// Total moves re-applied across all recovered owners this pass.
    pub total_applied: usize,
    /// Owners that had a pending tail replayed (non-empty applied set).
    pub owners_recovered: usize,
    /// Landings observed in the scanned tail for owners NOT in this pass's
    /// `fleet_ids` (or absent from the fleet) — latecomers whose recovery is
    /// still owed. `0` means the scanned tail belonged entirely to this pass.
    pub foreign_landings: usize,
    /// The SMALLEST cycle carrying such a foreign landing. **The latecomer
    /// fence:** the caller must NEVER raise its durable `after_cycle` bound to
    /// or past this cycle until those owners have recovered — advancing the
    /// global bound over an unrecovered latecomer's tail silences it
    /// permanently. `None` = no foreign landings in the scanned tail.
    pub foreign_min_cycle: Option<CycleId>,
}

/// **P4e — COMMITTED-HISTORY recovery ONLY.** Valid solely when a commit
/// SUCCEEDED (`Vn+1` exists as a sealed landing) and its application — or a
/// restart — was interrupted. It is NEVER the path for an ordinary pre-commit
/// write failure: that path is deterministic regeneration from the unchanged
/// `Vn` (module docs § Failure semantics), and the two mechanisms share no
/// state.
///
/// Scan the sealed landings ONCE, partition
/// them per owner (one pass over history, not O(fleet × history)), and replay
/// each owner's PENDING tail via `persist_sink::recover_and_apply`, idempotent
/// with a **per-owner watermark**. Only unreplayed moves (above the owner's
/// watermark) are applied; already-applied moves — including those applied by
/// the NORMAL path, whose watermarks [`apply_sealed_transitions`] advanced —
/// are skipped; unrepresented owners are untouched. `watermarks` is updated in
/// place with the new per-owner watermark to persist alongside the SoA phase.
///
/// **Bounded recovery is now the contract.** `after_cycle` is threaded
/// straight into [`WalSink::scan_sealed`]'s own bound: the scan reads only
/// cycles strictly after it, a BOUNDED tail — never the unbounded full sealed
/// history. Callers durably track the last cycle they fully recovered through
/// (e.g. the highest [`SealedCycle`] cycle whose transitions were applied) and
/// pass it back in on the next recovery pass; pass `None` only for a
/// from-scratch recovery over the whole sealed history.
///
/// On a mid-owner failure the partial progress is kept: the failing owner's
/// watermark is still advanced for its applied prefix (per
/// `recover_and_apply`'s `Err((partial, cause))` contract) before the error is
/// returned, so a re-drive does not replay the applied prefix.
pub async fn recover_fleet<S, F>(
    sink: &S,
    fleet: &mut F,
    fleet_ids: &[MailboxId],
    watermarks: &mut HashMap<MailboxId, Option<u64>>,
    after_cycle: Option<CycleId>,
) -> Result<FleetRecovery, PersistError>
where
    S: WalSink,
    F: MailboxFleet,
{
    let sealed: Vec<LandedSlot> = sink
        .scan_sealed(after_cycle)
        .await
        .map_err(PersistError::Write)?;
    // Partition once: per-owner tails in stored order (O(history), then each
    // owner replays only its own tail).
    let mut by_owner: HashMap<MailboxId, Vec<LandedSlot>> = HashMap::new();
    for ls in sealed {
        by_owner.entry(ls.slot.owner).or_default().push(ls);
    }
    let mut total_applied = 0usize;
    let mut owners_recovered = 0usize;
    // Latecomer accounting: landings owned by mailboxes this pass does NOT
    // recover (not listed, or listed but absent from the fleet). Their
    // smallest cycle is the floor below which the caller's global
    // `after_cycle` bound must stay until they recover.
    let ids: std::collections::HashSet<MailboxId> = fleet_ids.iter().copied().collect();
    let mut foreign_landings = 0usize;
    let mut foreign_min_cycle: Option<CycleId> = None;
    for (owner_id, slots) in &by_owner {
        if !ids.contains(owner_id) || fleet.owner(*owner_id).is_none() {
            foreign_landings += slots.len();
            if let Some(mc) = slots.iter().map(|l| l.cycle).min() {
                foreign_min_cycle = Some(foreign_min_cycle.map_or(mc, |cur: CycleId| cur.min(mc)));
            }
        }
    }
    for &id in fleet_ids {
        let Some(owner) = fleet.owner_mut(id) else {
            continue;
        };
        let tail: &[LandedSlot] = by_owner.get(&id).map_or(&[], Vec::as_slice);
        let wm = watermarks.get(&id).copied().flatten();
        match recover_and_apply(owner, tail, wm) {
            Ok(rec) => {
                if !rec.applied.is_empty() {
                    owners_recovered += 1;
                }
                total_applied += rec.applied.len();
                watermarks.insert(id, rec.watermark);
            }
            Err((partial, cause)) => {
                // Keep the earned watermark for the applied prefix, then surface the
                // error (the partial `applied` count is discarded — the caller
                // re-drives from the persisted watermark).
                watermarks.insert(id, partial.watermark);
                return Err(cause);
            }
        }
    }
    Ok(FleetRecovery {
        total_applied,
        owners_recovered,
        foreign_landings,
        foreign_min_cycle,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::kanban::{ExecTarget, KanbanColumn};
    use lance_graph_contract::soa_view::MailboxSoaView;
    use lance_graph_planner::persist_sink::{
        CommitError, CommitOutcome, DetachedCycleBatch, FrameMeta, LandedSlot, WriteFailed,
    };
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Mutex;

    // ── Minimal in-RAM owner (mirrors the persist_sink FakeOwner) ───────────────
    #[derive(Clone, PartialEq, Eq, Debug)]
    struct FakeOwner {
        id: MailboxId,
        phase: KanbanColumn,
        cycle: u32,
    }
    impl FakeOwner {
        fn at(id: MailboxId, phase: KanbanColumn) -> Self {
            Self {
                id,
                phase,
                cycle: 0,
            }
        }
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

    // ── The WAL sink fake — counts writes AND reads (base-fenced, failable) ─────
    struct SealedRec {
        frame: CycleFrame,
        version: DatasetVersion,
        /// The batch's deterministic content hash — the reconciliation-first
        /// idempotency key `commit_cycle` looks up BEFORE appending.
        batch_hash: u64,
        landings: Vec<SweepSlot>,
    }
    struct FakeWalSink {
        sealed: Mutex<Vec<SealedRec>>,
        next_version: AtomicU64,
        wal_writes: AtomicU64,
        reads: AtomicU64,      // scan_sealed + timeline — MUST stay 0 across P4a+P4b
        fail_next: AtomicBool, // injects ONE retryable (Io) WAL failure
    }
    impl FakeWalSink {
        fn new() -> Self {
            Self {
                sealed: Mutex::new(Vec::new()),
                next_version: AtomicU64::new(1),
                wal_writes: AtomicU64::new(0),
                reads: AtomicU64::new(0),
                fail_next: AtomicBool::new(false),
            }
        }
        fn wal_writes(&self) -> u64 {
            self.wal_writes.load(Ordering::SeqCst)
        }
        fn reads(&self) -> u64 {
            self.reads.load(Ordering::SeqCst)
        }
        fn fail_next_commit(&self) {
            self.fail_next.store(true, Ordering::SeqCst);
        }
    }
    impl WalSink for FakeWalSink {
        async fn commit_cycle(
            &mut self,
            batch: DetachedCycleBatch,
        ) -> Result<CommitOutcome, CommitError> {
            if self.fail_next.swap(false, Ordering::SeqCst) {
                return Err(CommitError::Io(WriteFailed(
                    "injected retryable WAL failure".into(),
                )));
            }
            let mut sealed = self.sealed.lock().unwrap();
            // Reconciliation-first: an already-durable (cycle, hash) is success,
            // a matching cycle with a different hash fails closed.
            if let Some(rec) = sealed.iter().find(|s| s.frame.cycle == batch.frame.cycle) {
                return if rec.batch_hash == batch.batch_hash {
                    Ok(CommitOutcome::Reconciled {
                        current_head: rec.version,
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
            let head = sealed.last().map_or(DatasetVersion(0), |s| s.version);
            if batch.frame.base_version != head {
                return Err(CommitError::Fenced { current_head: head });
            }
            self.wal_writes.fetch_add(1, Ordering::SeqCst);
            let version = DatasetVersion(self.next_version.fetch_add(1, Ordering::SeqCst));
            let (cycle, batch_hash) = (batch.frame.cycle, batch.batch_hash);
            sealed.push(SealedRec {
                frame: batch.frame,
                version,
                batch_hash,
                landings: batch.landings,
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
            self.reads.fetch_add(1, Ordering::SeqCst);
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
            self.reads.fetch_add(1, Ordering::SeqCst);
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
            exec: ExecTarget::Native,
        }
    }

    /// Stage one Planning→CognitiveWork cast for each of `owners`.
    fn writer_with_moves(owners: &[MailboxId]) -> BatchWriter<Vec<u8>> {
        let mut w: BatchWriter<Vec<u8>> = BatchWriter::new();
        for &o in owners {
            w.cast(
                o,
                vec![mv(o, KanbanColumn::Planning, KanbanColumn::CognitiveWork)],
                vec![0xAB],
            );
        }
        w
    }

    // ── P4a FALSIFIER: drain N casts → exactly one WAL write, one version ───────
    #[tokio::test]
    async fn p4a_drains_casts_and_seals_one_wal_write_one_version() {
        let mut sink = FakeWalSink::new();
        let owners: Vec<MailboxId> = (0..100).collect();
        let mut w = writer_with_moves(&owners);
        let collected = collect_casts(&mut w, CycleId(1), 0, u64::from);
        assert_eq!(collected.slots.len(), 100, "one slot per staged cast");
        assert!(collected.held.is_empty(), "one move each → nothing held");
        assert_eq!(sink.wal_writes(), 0, "collecting writes no WAL");

        let sealed = seal_cycle(
            &mut sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            collected.slots,
        )
        .await
        .unwrap();
        assert_eq!(sink.wal_writes(), 1, "100 casts → exactly ONE WAL write");
        assert_eq!(
            sealed.version,
            Some(DatasetVersion(1)),
            "→ exactly one version"
        );
        assert_eq!(
            sealed.transitions.len(),
            100,
            "all 100 casts carried a move → sparse set = 100 here"
        );
        assert_eq!(
            sealed.next_position_base, 100,
            "next base = max position + 1"
        );
        // A second drain of the same writer is empty (payload staging cleared).
        let again = collect_casts(&mut w, CycleId(1), 0, u64::from);
        assert!(again.slots.is_empty(), "drain cleared the writer's staging");
    }

    // ── AUTHORITATIVE PRE-COMMIT FALSIFIER: deterministic regeneration from Vn ──
    // The operator-ruled contract: sealed Vn + unchanged Kanban task +
    // deterministic computation = the same provisional intent on the next sweep.
    // Correctness NEVER requires retaining the first heap batch — this test
    // DROPS the SealFailure (the optional cache) and proves the same semantic
    // sparse cycle re-derives from the unchanged Vn. Object identity of the
    // first batch is deliberately NOT asserted.
    #[tokio::test]
    async fn pre_commit_failure_discards_everything_and_regenerates_from_vn() {
        let mut sink = FakeWalSink::new();
        let mut fleet: HashMap<MailboxId, FakeOwner> = HashMap::from([
            (3, FakeOwner::at(3, KanbanColumn::Planning)),
            (8, FakeOwner::at(8, KanbanColumn::Planning)),
        ]);
        let before = fleet.clone();
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();

        // The deterministic Kanban task: a pure function of the (unchanged)
        // fleet state — every owner still in Planning stages Planning→CognitiveWork.
        let stage = |fleet: &HashMap<MailboxId, FakeOwner>| -> BatchWriter<Vec<u8>> {
            let mut w: BatchWriter<Vec<u8>> = BatchWriter::new();
            let mut ids: Vec<MailboxId> = fleet
                .values()
                .filter(|o| o.phase() == KanbanColumn::Planning)
                .map(|o| o.mailbox_id())
                .collect();
            ids.sort_unstable();
            for id in ids {
                w.cast(
                    id,
                    vec![mv(id, KanbanColumn::Planning, KanbanColumn::CognitiveWork)],
                    vec![id as u8],
                );
            }
            w
        };

        // Derive cycle C deterministically from Vn; inject a commit failure.
        let mut w1 = stage(&fleet);
        sink.fail_next_commit();
        let err = run_cycle(
            &mut sink,
            &mut fleet,
            &mut w1,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            0,
            &mut wm,
            u64::from,
        )
        .await
        .expect_err("injected commit failure");
        let CycleError::Seal(failure) = err else {
            panic!("expected a Seal failure");
        };
        // Snapshot the failed cycle's SEMANTIC content, then DISCARD the batch —
        // the optional cache is deliberately NOT used for recovery.
        let semantic_c1: Vec<(MailboxId, KanbanColumn, KanbanColumn)> = failure
            .casts
            .iter()
            .filter_map(|s| s.paired_move.map(|m| (s.owner, m.from, m.to)))
            .collect();
        drop(failure);

        // Publish nothing · mutate no owner · advance no watermark.
        assert_eq!(sink.wal_writes(), 0, "no Vn+1 was published");
        assert_eq!(fleet, before, "no owner mutated");
        assert!(wm.is_empty(), "no watermark advanced");

        // Rerun the UNCHANGED task from the unchanged Vn (fresh writer, fresh
        // staging — nothing retained from the failed attempt).
        let mut w2 = stage(&fleet);
        let out = run_cycle(
            &mut sink,
            &mut fleet,
            &mut w2,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            0,
            &mut wm,
            u64::from,
        )
        .await
        .expect("regenerated cycle commits");

        // The SAME semantic sparse cycle regenerated (owner set + from→to).
        let semantic_c2: Vec<(MailboxId, KanbanColumn, KanbanColumn)> = out
            .sealed
            .transitions
            .iter()
            .map(|t| (t.owner, t.mv.from, t.mv.to))
            .collect();
        assert_eq!(
            semantic_c2, semantic_c1,
            "deterministic regeneration: the same semantic sparse cycle"
        );

        // Exactly one Vn+1; exactly the represented owners advance once.
        assert_eq!(sink.wal_writes(), 1, "exactly one successful WAL write");
        assert_eq!(out.sealed.version, Some(DatasetVersion(1)));
        assert_eq!(out.applied.applied.len(), 2);
        assert_eq!(fleet[&3].phase(), KanbanColumn::CognitiveWork);
        assert_eq!(fleet[&8].phase(), KanbanColumn::CognitiveWork);
    }

    // ── Optional-cache probe: the SealFailure retry cache also works ────────────
    // Secondary to the regeneration falsifier above: a caller that DOES use the
    // cache gets a byte-identical resubmit. Convenience path, not the contract.
    #[tokio::test]
    async fn failed_seal_preserves_the_frozen_cycle_for_byte_identical_retry() {
        let mut sink = FakeWalSink::new();
        let mut fleet: HashMap<MailboxId, FakeOwner> =
            HashMap::from([(9, FakeOwner::at(9, KanbanColumn::Planning))]);
        let before = fleet.clone();
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();
        let mut w = writer_with_moves(&[9]);

        sink.fail_next_commit();
        let err = run_cycle(
            &mut sink,
            &mut fleet,
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            0,
            &mut wm,
            u64::from,
        )
        .await
        .expect_err("injected WAL failure");

        // Failed seal → zero owner mutation, and the writer stayed drained.
        assert_eq!(fleet, before, "no owner advanced on a failed seal");
        let CycleError::Seal(failure) = err else {
            panic!("expected a Seal failure");
        };
        assert_eq!(failure.casts.len(), 1, "the frozen cycle survived");
        let frozen_copy = failure.casts.clone();

        // Retry submits the SAME frozen cycle → exactly one version lands.
        let sealed = seal_cycle(&mut sink, failure.frame, failure.casts)
            .await
            .expect("retry succeeds");
        assert_eq!(sealed.version, Some(DatasetVersion(1)));
        assert_eq!(sink.wal_writes(), 1, "one successful WAL write total");
        // Byte-identical: what landed is exactly the frozen set.
        let landed = sink.scan_sealed(None).await.unwrap();
        assert_eq!(landed.len(), 1);
        assert_eq!(landed[0].slot, frozen_copy[0], "no cast lost or mutated");

        let applied = apply_sealed_transitions(&mut fleet, &sealed, &mut wm).unwrap();
        assert_eq!(applied.applied.len(), 1, "owner advances exactly once");
        assert_eq!(fleet[&9].phase(), KanbanColumn::CognitiveWork);
    }

    // ── RESTART FALSIFIER: stream positions stay monotonic across writer rebuilds ─
    #[tokio::test]
    async fn restart_stable_stream_positions_survive_writer_reconstruction() {
        let mut sink = FakeWalSink::new();
        let mut fleet: HashMap<MailboxId, FakeOwner> =
            HashMap::from([(5, FakeOwner::at(5, KanbanColumn::Planning))]);
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();

        // Cycle 1 (base 0): owner 5 Planning→CognitiveWork at position 0.
        let mut w1 = writer_with_moves(&[5]);
        let c1 = collect_casts(&mut w1, CycleId(1), 0, u64::from);
        let s1 = seal_cycle(
            &mut sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            c1.slots,
        )
        .await
        .unwrap();
        apply_sealed_transitions(&mut fleet, &s1, &mut wm).unwrap();
        assert_eq!(wm[&5], Some(0), "cycle-1 watermark at position 0");

        // RESTART: the writer is RECONSTRUCTED (CastId restarts at 0). The
        // durable cursor carries forward: base = s1.next_position_base.
        let mut w2: BatchWriter<Vec<u8>> = BatchWriter::new();
        w2.cast(
            5,
            vec![mv(5, KanbanColumn::CognitiveWork, KanbanColumn::Evaluation)],
            vec![0xCD],
        );
        let base2 = s1.next_position_base;
        assert_eq!(base2, 1);
        let c2 = collect_casts(&mut w2, CycleId(2), base2, u64::from);
        assert_eq!(
            c2.slots[0].stream_position, 1,
            "restart-stable: NOT the raw CastId 0"
        );
        seal_cycle(
            &mut sink,
            CycleFrame::new(CycleId(2), DatasetVersion(1)),
            c2.slots,
        )
        .await
        .unwrap();

        // Crash AFTER cycle 2 sealed but BEFORE it applied: recovery from the
        // cycle-1 watermark must NOT skip the cycle-2 landing. (With the raw
        // CastId scheme its position would be 0 ≤ watermark 0 → silently lost.)
        let rec = recover_fleet(&mut sink, &mut fleet, &[5], &mut wm, None)
            .await
            .unwrap();
        assert_eq!(rec.total_applied, 1, "the later landing was replayed");
        assert_eq!(fleet[&5].phase(), KanbanColumn::Evaluation);
        assert_eq!(wm[&5], Some(1));
    }

    // ── WATERMARK-COUPLING FALSIFIER: normal apply advances recovery watermarks ─
    #[tokio::test]
    async fn normal_apply_advances_the_recovery_watermark_no_replay_after_crash() {
        let mut sink = FakeWalSink::new();
        let mut fleet: HashMap<MailboxId, FakeOwner> =
            HashMap::from([(5, FakeOwner::at(5, KanbanColumn::Planning))]);
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();
        let mut w = writer_with_moves(&[5]);

        // Normal path: seal + apply. Watermark advances WITH the phase.
        run_cycle(
            &mut sink,
            &mut fleet,
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            0,
            &mut wm,
            u64::from,
        )
        .await
        .unwrap();
        assert_eq!(fleet[&5].phase(), KanbanColumn::CognitiveWork);
        assert_eq!(wm[&5], Some(0), "normal apply advanced the watermark");

        // Crash + recovery with the SAME persisted watermarks: nothing replays,
        // no StalePhase — the normal path and recovery share one rule.
        let rec = recover_fleet(&mut sink, &mut fleet, &[5], &mut wm, None)
            .await
            .expect("recovery after a normal apply must not StalePhase-stall");
        assert_eq!(rec.total_applied, 0, "already-applied move NOT replayed");
        assert_eq!(fleet[&5].phase(), KanbanColumn::CognitiveWork);
    }

    // ── P4b HEADLINE FALSIFIER: 64k mailboxes / 17 sealed → exactly 17 advance ──
    #[tokio::test]
    async fn p4b_applies_only_the_sealed_sparse_set_64k_of_17_advance_rest_byte_identical() {
        const FLEET: u32 = 65_536;
        // 17 owners produce a material update; the other 65_519 do not.
        let represented: Vec<MailboxId> = (0..17).map(|i| i * 3_855).collect();
        let represented_set: HashSet<MailboxId> = represented.iter().copied().collect();

        let mut fleet: HashMap<MailboxId, FakeOwner> = (0..FLEET)
            .map(|id| (id, FakeOwner::at(id, KanbanColumn::Planning)))
            .collect();
        let before = fleet.clone();
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();

        let mut sink = FakeWalSink::new();
        let mut w = writer_with_moves(&represented);
        let collected = collect_casts(&mut w, CycleId(1), 0, u64::from);
        let sealed = seal_cycle(
            &mut sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            collected.slots,
        )
        .await
        .unwrap();
        assert_eq!(
            sealed.transitions.len(),
            17,
            "sparse sealed set = 17, not 64k"
        );

        let applied = apply_sealed_transitions(&mut fleet, &sealed, &mut wm).unwrap();

        // Exactly the 17 represented owners advanced.
        assert_eq!(applied.applied.len(), 17, "exactly 17 owners advanced");
        assert_eq!(applied.deferred, 0);
        assert_eq!(applied.missing, 0);
        assert_eq!(applied.version, Some(DatasetVersion(1)));
        assert_eq!(wm.len(), 17, "exactly 17 watermarks advanced");

        // Every represented owner is now at CognitiveWork (cycle bumped); every
        // OTHER owner is byte-identical to its pre-cycle state.
        let mut advanced_count = 0usize;
        let mut untouched_count = 0usize;
        for (id, owner) in &fleet {
            if represented_set.contains(id) {
                assert_eq!(owner.phase(), KanbanColumn::CognitiveWork);
                assert_eq!(owner.current_cycle(), 1, "advanced owner bumped its cycle");
                advanced_count += 1;
            } else {
                assert_eq!(
                    owner, &before[id],
                    "unrepresented owner {id} must be BYTE-IDENTICAL"
                );
                untouched_count += 1;
            }
        }
        assert_eq!(advanced_count, 17);
        assert_eq!(
            untouched_count,
            (FLEET as usize) - 17,
            "anti-vacuity: the untouched set is 64k−17, not merely 'some'"
        );

        // Storage: exactly one WAL write, one version, and P4b read NO dataset.
        assert_eq!(sink.wal_writes(), 1, "one WAL write for the whole sweep");
        assert_eq!(
            sink.reads(),
            0,
            "P4b reads no dataset (no second dataset read)"
        );
    }

    // ── P4b: a DatasetVersion is NOT a fleet-step signal (empty sparse set) ──────
    #[tokio::test]
    async fn p4b_a_version_with_no_sealed_transitions_advances_nobody() {
        // Every owner "participated" (a no-move cast) but NONE produced a move.
        let mut fleet: HashMap<MailboxId, FakeOwner> = (0..1_000)
            .map(|id| (id, FakeOwner::at(id, KanbanColumn::Planning)))
            .collect();
        let before = fleet.clone();
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();

        let mut sink = FakeWalSink::new();
        let mut w: BatchWriter<Vec<u8>> = BatchWriter::new();
        for id in 0..1_000u32 {
            w.cast(id, vec![], vec![0x00]); // no move → no transition
        }
        let collected = collect_casts(&mut w, CycleId(1), 0, u64::from);
        let sealed = seal_cycle(
            &mut sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            collected.slots,
        )
        .await
        .unwrap();
        assert!(sealed.transitions.is_empty(), "no moves → empty sparse set");

        let applied = apply_sealed_transitions(&mut fleet, &sealed, &mut wm).unwrap();
        assert!(
            applied.applied.is_empty(),
            "a version alone advances nobody"
        );
        assert_eq!(fleet, before, "the whole fleet is byte-identical");
        assert_eq!(sink.wal_writes(), 1, "the cycle still sealed one version");
    }

    // ── ≤1/owner is enforced BEFORE sealing: extras are HELD, never discarded ───
    #[tokio::test]
    async fn second_same_owner_move_is_held_pre_seal_and_lands_next_cycle() {
        let mut fleet: HashMap<MailboxId, FakeOwner> =
            HashMap::from([(42, FakeOwner::at(42, KanbanColumn::Planning))]);
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();
        let mut sink = FakeWalSink::new();
        // Two casts for owner 42 staged in one cycle.
        let mut w: BatchWriter<Vec<u8>> = BatchWriter::new();
        w.cast(
            42,
            vec![mv(42, KanbanColumn::Planning, KanbanColumn::CognitiveWork)],
            vec![1],
        );
        w.cast(
            42,
            vec![mv(
                42,
                KanbanColumn::CognitiveWork,
                KanbanColumn::Evaluation,
            )],
            vec![2],
        );
        let collected = collect_casts(&mut w, CycleId(1), 0, u64::from);
        // The partition happened PRE-SEAL: one paired transition, one held.
        assert_eq!(collected.held.len(), 1, "the second move is HELD");
        assert_eq!(collected.held[0].mv.to, KanbanColumn::Evaluation);
        assert_eq!(
            collected.slots[1].paired_move, None,
            "the extra cast still lands its payload, move-free"
        );

        let sealed = seal_cycle(
            &mut sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            collected.slots,
        )
        .await
        .unwrap();
        assert_eq!(
            sealed.transitions.len(),
            1,
            "sealed set == applied set (recovery and normal op agree)"
        );
        let applied = apply_sealed_transitions(&mut fleet, &sealed, &mut wm).unwrap();
        assert_eq!(applied.applied.len(), 1);
        assert_eq!(applied.deferred, 0, "nothing sealed was skipped");
        assert_eq!(fleet[&42].phase(), KanbanColumn::CognitiveWork);

        // Recovery from scratch applies EXACTLY the same set as normal op did.
        let mut fresh = HashMap::from([(42, FakeOwner::at(42, KanbanColumn::Planning))]);
        let mut wm2: HashMap<MailboxId, Option<u64>> = HashMap::new();
        let rec = recover_fleet(&mut sink, &mut fresh, &[42], &mut wm2, None)
            .await
            .unwrap();
        assert_eq!(rec.total_applied, 1, "recovery applies the same ONE move");
        assert_eq!(fresh[&42].phase(), KanbanColumn::CognitiveWork);

        // The held intent re-stages and lands in the NEXT cycle.
        assert_eq!(restage_held(&mut w, collected.held), 1);
        let c2 = collect_casts(&mut w, CycleId(2), sealed.next_position_base, u64::from);
        let s2 = seal_cycle(
            &mut sink,
            CycleFrame::new(CycleId(2), DatasetVersion(1)),
            c2.slots,
        )
        .await
        .unwrap();
        let a2 = apply_sealed_transitions(&mut fleet, &s2, &mut wm).unwrap();
        assert_eq!(a2.applied.len(), 1, "the held move applied next cycle");
        assert_eq!(fleet[&42].phase(), KanbanColumn::Evaluation);
    }

    // ── No silent truncation: a multi-move cast holds its extras too ────────────
    #[tokio::test]
    async fn multi_move_single_cast_holds_extras_never_truncates() {
        let mut w: BatchWriter<Vec<u8>> = BatchWriter::new();
        w.cast(
            7,
            vec![
                mv(7, KanbanColumn::Planning, KanbanColumn::CognitiveWork),
                mv(7, KanbanColumn::CognitiveWork, KanbanColumn::Evaluation),
                mv(7, KanbanColumn::Evaluation, KanbanColumn::Commit),
            ],
            vec![0xEE],
        );
        let collected = collect_casts(&mut w, CycleId(1), 0, u64::from);
        assert_eq!(collected.slots.len(), 1);
        assert_eq!(
            collected.slots[0].paired_move.unwrap().to,
            KanbanColumn::CognitiveWork,
            "first move seals"
        );
        assert_eq!(
            collected.held.len(),
            2,
            "the other two moves are HELD, not silently dropped"
        );
    }

    // ── P4b: a stale sealed move (from ≠ phase) is a corruption error ───────────
    #[tokio::test]
    async fn p4b_stale_phase_is_a_corruption_error() {
        // Owner is at CognitiveWork; the sealed move claims from=Planning.
        let mut fleet: HashMap<MailboxId, FakeOwner> =
            HashMap::from([(7, FakeOwner::at(7, KanbanColumn::CognitiveWork))]);
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();
        let sealed = SealedCycle {
            outcome: CommitOutcome::Committed {
                version: DatasetVersion(1),
                cycle: CycleId(1),
                batch_hash: 0,
            },
            version: Some(DatasetVersion(1)),
            transitions: vec![SealedTransition {
                stream_position: 0,
                owner: 7,
                mv: mv(7, KanbanColumn::Planning, KanbanColumn::CognitiveWork),
            }],
            next_position_base: 1,
        };
        assert!(matches!(
            apply_sealed_transitions(&mut fleet, &sealed, &mut wm),
            Err((_, PersistError::StalePhase { .. }))
        ));
    }

    // ── P4b: a mid-apply error preserves the applied prefix + its watermarks ────
    #[tokio::test]
    async fn p4b_mid_apply_error_returns_the_applied_prefix_with_watermarks() {
        // Owner 1 has a good move; owner 2's sealed move is stale (corruption).
        let mut fleet: HashMap<MailboxId, FakeOwner> = HashMap::from([
            (1, FakeOwner::at(1, KanbanColumn::Planning)),
            (2, FakeOwner::at(2, KanbanColumn::CognitiveWork)),
        ]);
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();
        let sealed = SealedCycle {
            outcome: CommitOutcome::Committed {
                version: DatasetVersion(1),
                cycle: CycleId(1),
                batch_hash: 0,
            },
            version: Some(DatasetVersion(1)),
            transitions: vec![
                SealedTransition {
                    stream_position: 0,
                    owner: 1,
                    mv: mv(1, KanbanColumn::Planning, KanbanColumn::CognitiveWork),
                },
                SealedTransition {
                    stream_position: 1,
                    owner: 2,
                    mv: mv(2, KanbanColumn::Planning, KanbanColumn::CognitiveWork),
                },
            ],
            next_position_base: 2,
        };
        let (partial, cause) =
            apply_sealed_transitions(&mut fleet, &sealed, &mut wm).expect_err("owner 2 is stale");
        assert!(matches!(cause, PersistError::StalePhase { .. }));
        assert_eq!(partial.applied.len(), 1, "the applied prefix is preserved");
        assert_eq!(partial.applied[0].mailbox, 1);
        assert_eq!(
            wm.get(&1).copied().flatten(),
            Some(0),
            "the prefix's watermark was advanced before the error"
        );
        assert_eq!(fleet[&1].phase(), KanbanColumn::CognitiveWork);
    }

    // ── P4b: a sealed move for an unregistered owner is counted, not a crash ────
    #[tokio::test]
    async fn p4b_missing_owner_is_counted_not_applied() {
        let mut fleet: HashMap<MailboxId, FakeOwner> = HashMap::new(); // empty fleet
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();
        let sealed = SealedCycle {
            outcome: CommitOutcome::Committed {
                version: DatasetVersion(1),
                cycle: CycleId(1),
                batch_hash: 0,
            },
            version: Some(DatasetVersion(1)),
            transitions: vec![SealedTransition {
                stream_position: 0,
                owner: 99,
                mv: mv(99, KanbanColumn::Planning, KanbanColumn::CognitiveWork),
            }],
            next_position_base: 1,
        };
        let applied = apply_sealed_transitions(&mut fleet, &sealed, &mut wm).unwrap();
        assert_eq!(applied.missing, 1);
        assert!(applied.applied.is_empty());
        // The durable landing stays in the log: once the owner registers,
        // recover_fleet replays it (the normal-path retry mechanism).
    }

    // ── End-to-end: run_cycle closes P4a→P4b in one call ────────────────────────
    #[tokio::test]
    async fn run_cycle_seals_then_applies_the_sparse_set() {
        let mut fleet: HashMap<MailboxId, FakeOwner> = (0..10)
            .map(|id| (id, FakeOwner::at(id, KanbanColumn::Planning)))
            .collect();
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();
        let mut sink = FakeWalSink::new();
        let mut w = writer_with_moves(&[3, 7]); // only owners 3 and 7 produce a move

        let out = run_cycle(
            &mut sink,
            &mut fleet,
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            0,
            &mut wm,
            u64::from,
        )
        .await
        .unwrap();

        assert_eq!(out.sealed.version, Some(DatasetVersion(1)));
        assert_eq!(
            out.applied.applied.len(),
            2,
            "exactly owners 3 and 7 advanced"
        );
        assert!(out.held.is_empty());
        assert_eq!(fleet[&3].phase(), KanbanColumn::CognitiveWork);
        assert_eq!(fleet[&7].phase(), KanbanColumn::CognitiveWork);
        assert_eq!(
            fleet[&0].phase(),
            KanbanColumn::Planning,
            "owner 0 untouched"
        );
        assert_eq!(sink.wal_writes(), 1);
        assert_eq!(sink.reads(), 0);
    }

    /// A bootstrap-sentinel move (owner 0, cycle 0) that `owner_adapter` rebinds.
    fn sentinel(from: KanbanColumn, to: KanbanColumn) -> KanbanMove {
        KanbanMove {
            mailbox: 0,
            from,
            to,
            witness_chain_position: 0,
            exec: ExecTarget::Native,
        }
    }

    // ── P4c FALSIFIER: CognitiveWork thought → next-cycle cast → round-trip ─────
    #[tokio::test]
    async fn p4c_cognitive_work_casts_the_next_intent_and_round_trips() {
        let mut sink = FakeWalSink::new();
        let mut fleet: HashMap<MailboxId, FakeOwner> =
            HashMap::from([(5, FakeOwner::at(5, KanbanColumn::Planning))]);
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();
        let mut w = writer_with_moves(&[5]);

        // Cycle 1: owner 5 casts Planning→CognitiveWork; the driver applies it.
        let out1 = run_cycle(
            &mut sink,
            &mut fleet,
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            0,
            &mut wm,
            u64::from,
        )
        .await
        .unwrap();
        assert_eq!(fleet[&5].phase(), KanbanColumn::CognitiveWork);

        // P4c: owner 5 (now in CognitiveWork) thinks → intends CognitiveWork→Evaluation,
        // cast as a bootstrap sentinel into the writer for cycle 2.
        let cw = run_cognitive_work(&fleet, &out1.applied, &mut w, |owner| {
            assert_eq!(owner.phase(), KanbanColumn::CognitiveWork);
            let outcome = StrategyOutcome {
                reliability: 0.9,
                intended_move: Some(sentinel(
                    KanbanColumn::CognitiveWork,
                    KanbanColumn::Evaluation,
                )),
            };
            Some((outcome, vec![0xCC]))
        });
        assert_eq!(cw.cast, 1, "one next-cycle intent cast");
        assert!(cw.held_owners.is_empty());

        // Cycle 2: the driver drains that cast → seals V2 → applies → owner 5 → Evaluation.
        let out2 = run_cycle(
            &mut sink,
            &mut fleet,
            &mut w,
            CycleFrame::new(CycleId(2), DatasetVersion(1)),
            out1.sealed.next_position_base,
            &mut wm,
            u64::from,
        )
        .await
        .unwrap();
        assert_eq!(out2.sealed.version, Some(DatasetVersion(2)));
        assert_eq!(
            out2.applied.applied.len(),
            1,
            "the round-tripped intent advanced owner 5 one further step"
        );
        assert_eq!(fleet[&5].phase(), KanbanColumn::Evaluation);
    }

    // ── WAIT-FREE (strengthened): an unfinished thought never blocks a cast ─────
    #[tokio::test]
    async fn p4d_an_unfinished_owner_never_blocks_a_completed_owners_cast() {
        // BOTH owners are represented and BOTH enter CognitiveWork; A's thought
        // stays unfinished (declines), B completes — B's cast lands regardless.
        let mut sink = FakeWalSink::new();
        let mut fleet: HashMap<MailboxId, FakeOwner> = HashMap::from([
            (1, FakeOwner::at(1, KanbanColumn::Planning)),
            (2, FakeOwner::at(2, KanbanColumn::Planning)),
        ]);
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();
        let mut w = writer_with_moves(&[1, 2]);

        let out = run_cycle(
            &mut sink,
            &mut fleet,
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            0,
            &mut wm,
            u64::from,
        )
        .await
        .unwrap();
        assert_eq!(out.applied.applied.len(), 2, "both entered CognitiveWork");

        let cw = run_cognitive_work(&fleet, &out.applied, &mut w, |owner| {
            if owner.mailbox_id() == 1 {
                None // A: unfinished / declines this pass
            } else {
                Some((
                    StrategyOutcome {
                        reliability: 0.8,
                        intended_move: Some(sentinel(
                            KanbanColumn::CognitiveWork,
                            KanbanColumn::Evaluation,
                        )),
                    },
                    vec![0xB2],
                ))
            }
        });
        assert_eq!(cw.cast, 1, "B cast without waiting for A");
        assert_eq!(cw.held_owners, vec![1], "A is rescheduled, not lost");

        // Cycle 2: B advances; A stays in CognitiveWork (unblocked, unfinished).
        let out2 = run_cycle(
            &mut sink,
            &mut fleet,
            &mut w,
            CycleFrame::new(CycleId(2), DatasetVersion(1)),
            out.sealed.next_position_base,
            &mut wm,
            u64::from,
        )
        .await
        .unwrap();
        assert_eq!(out2.applied.applied.len(), 1);
        assert_eq!(fleet[&2].phase(), KanbanColumn::Evaluation, "B advanced");
        assert_eq!(
            fleet[&1].phase(),
            KanbanColumn::CognitiveWork,
            "A unfinished, never a barrier for B"
        );
    }

    // ── P4e FALSIFIER: the `after_cycle` bound BITES (inertness both ways) ───────
    /// A knob that changes nothing is decoration. This drives the SAME durable
    /// history twice — once unbounded, once bounded past the landing's cycle —
    /// and asserts the bound both EXCLUDES (nothing replays, the owner stays
    /// put) and, at a lower bound, ADMITS (the move replays). Without both
    /// halves, `after_cycle` could be ignored inside `recover_fleet` and every
    /// existing test would still pass, since they all pass `None`.
    #[tokio::test]
    async fn recover_fleet_after_cycle_bound_excludes_and_admits() {
        // Seal owner 5's Planning→CognitiveWork move in CYCLE 2.
        let mut sink = FakeWalSink::new();
        let mut w = writer_with_moves(&[5]);
        let collected = collect_casts(&mut w, CycleId(2), 0, u64::from);
        seal_cycle(
            &mut sink,
            CycleFrame::new(CycleId(2), DatasetVersion(0)),
            collected.slots,
        )
        .await
        .unwrap();

        // Bounded PAST the landing's cycle → its tail is outside the read.
        let mut fleet: HashMap<MailboxId, FakeOwner> =
            HashMap::from([(5, FakeOwner::at(5, KanbanColumn::Planning))]);
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();
        let excluded = recover_fleet(&mut sink, &mut fleet, &[5], &mut wm, Some(CycleId(2)))
            .await
            .unwrap();
        assert_eq!(
            excluded.total_applied, 0,
            "a bound past the landing's cycle excludes it — the bound is not inert"
        );
        assert_eq!(
            fleet[&5].phase(),
            KanbanColumn::Planning,
            "and the owner is untouched"
        );

        // Bounded BELOW it (strictly-after semantics) → the same tail replays.
        let admitted = recover_fleet(&mut sink, &mut fleet, &[5], &mut wm, Some(CycleId(1)))
            .await
            .unwrap();
        assert_eq!(
            admitted.total_applied, 1,
            "a lower bound admits the same landing — the bound discriminates"
        );
        assert_eq!(fleet[&5].phase(), KanbanColumn::CognitiveWork);
    }

    // ── P4e FALSIFIER: recovery replays the pending tail, idempotent w/ watermark ─
    #[tokio::test]
    async fn p4e_recover_fleet_replays_pending_tail_idempotent_with_watermark() {
        // Seal a cycle with owner 5's Planning→CognitiveWork move (a durable landing).
        let mut sink = FakeWalSink::new();
        let mut w = writer_with_moves(&[5]);
        let collected = collect_casts(&mut w, CycleId(1), 0, u64::from);
        seal_cycle(
            &mut sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            collected.slots,
        )
        .await
        .unwrap();

        // Restart: a FRESH owner 5 at its pre-move phase, empty watermarks.
        let mut fleet: HashMap<MailboxId, FakeOwner> =
            HashMap::from([(5, FakeOwner::at(5, KanbanColumn::Planning))]);
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();

        let rec = recover_fleet(&mut sink, &mut fleet, &[5], &mut wm, None)
            .await
            .unwrap();
        assert_eq!(rec.total_applied, 1, "the pending move was replayed");
        assert_eq!(rec.owners_recovered, 1);
        assert_eq!(fleet[&5].phase(), KanbanColumn::CognitiveWork);

        // Re-drive with the returned watermark → idempotent (nothing re-applied).
        let again = recover_fleet(&mut sink, &mut fleet, &[5], &mut wm, None)
            .await
            .unwrap();
        assert_eq!(
            again.total_applied, 0,
            "watermark makes recovery idempotent"
        );

        // Negative control: watermark LOST → re-driving the already-advanced owner
        // stalls (from=Planning ≠ phase=CognitiveWork) → the watermark is load-bearing.
        let mut wm_lost: HashMap<MailboxId, Option<u64>> = HashMap::new();
        let stalled = recover_fleet(&mut sink, &mut fleet, &[5], &mut wm_lost, None).await;
        assert!(
            matches!(stalled, Err(PersistError::StalePhase { .. })),
            "without the watermark an acyclic re-drive stalls — watermark is load-bearing"
        );
    }

    // A fleet that COUNTS owner resolutions — proves apply cost is O(sparse set).
    struct CountingFleet {
        inner: HashMap<MailboxId, FakeOwner>,
        resolves: usize,
    }
    impl MailboxFleet for CountingFleet {
        type Owner = FakeOwner;
        fn owner(&self, id: MailboxId) -> Option<&FakeOwner> {
            self.inner.get(&id)
        }
        fn owner_mut(&mut self, id: MailboxId) -> Option<&mut FakeOwner> {
            self.resolves += 1;
            self.inner.get_mut(&id)
        }
    }

    // ── P4f (SCALE): apply cost scales with the SPARSE set, not the fleet ───────
    #[tokio::test]
    async fn p4f_apply_cost_scales_with_the_sparse_set_not_the_fleet() {
        const FLEET: u32 = 65_536;
        const DIRTY: u32 = 640; // ~1% sparse dirty fraction
        let inner: HashMap<MailboxId, FakeOwner> = (0..FLEET)
            .map(|id| (id, FakeOwner::at(id, KanbanColumn::Planning)))
            .collect();
        let mut fleet = CountingFleet { inner, resolves: 0 };
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();

        let mut sink = FakeWalSink::new();
        let represented: Vec<MailboxId> = (0..DIRTY).map(|i| i * 100).collect();
        let mut w = writer_with_moves(&represented);
        let collected = collect_casts(&mut w, CycleId(1), 0, u64::from);
        let sealed = seal_cycle(
            &mut sink,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            collected.slots,
        )
        .await
        .unwrap();
        assert_eq!(sealed.transitions.len(), DIRTY as usize);

        let applied = apply_sealed_transitions(&mut fleet, &sealed, &mut wm).unwrap();
        assert_eq!(
            applied.applied.len(),
            DIRTY as usize,
            "exactly the dirty set advanced"
        );
        // THE MEASUREMENT: owner resolution touched the sparse set ONLY — 640, not 64k.
        assert_eq!(
            fleet.resolves, DIRTY as usize,
            "apply resolves exactly the sparse set (O(dirty)), never the {FLEET}-owner fleet"
        );
        eprintln!(
            "perf.p4f fleet={FLEET} dirty={DIRTY} resolves={} (sparse routing: apply is O(dirty), not O(fleet))",
            fleet.resolves
        );
    }

    // ── The MUL-gate plug (P4c gate) ────────────────────────────────────────────

    /// Flow qualia (warmth=4, groundedness=3, coherence=4, valence=2) — the same
    /// construction `kanban_actor::tests::mul_target_flow_advances_and_hold_holds`
    /// uses: `flow_proxy = 4+3−0 = 7 ≥ 4` + mantissa>0 → FlowState::Flow;
    /// coherence≥4 + valence≥2 + tension≤1 → TrustTexture::Calibrated ⇒ gate `Flow`.
    fn flow_qualia() -> QualiaI4_16D {
        QualiaI4_16D(0).with(3, 4).with(14, 3).with(9, 4).with(1, 2)
    }

    /// Uncertain qualia (coherence=−3, tension=3) ⇒ TrustTexture::Uncertain ⇒
    /// gate `Block`.
    fn block_qualia() -> QualiaI4_16D {
        QualiaI4_16D(0).with(9, -3).with(2, 3)
    }

    // shade_owner is the REAL gate: Flow→forward, Block→Prune, Hold→None. Three
    // distinct outputs for three distinct inputs — the gate discriminates (it is
    // not a constant that always fires the same way).
    #[test]
    fn shade_owner_flow_advances_forward_block_prunes_hold_rests() {
        // Flow at CognitiveWork → forward to Evaluation (the only non-Prune next).
        let cw = FakeOwner::at(1, KanbanColumn::CognitiveWork);
        let out = shade_owner(&cw, &flow_qualia(), 4, 0.9).expect("Flow yields a move");
        let m = out.intended_move.expect("Flow carries an intended move");
        assert_eq!(m.from, KanbanColumn::CognitiveWork);
        assert_eq!(m.to, KanbanColumn::Evaluation, "Flow → forward");
        assert_eq!(
            m.mailbox, 0,
            "bootstrap sentinel — owner_adapter rebinds it"
        );
        assert_eq!(m.witness_chain_position, 0, "sentinel witness position");
        assert!((out.reliability - 0.9).abs() < f32::EPSILON);

        // Block at Planning → Prune (the Prune-where-legal branch).
        let plan = FakeOwner::at(2, KanbanColumn::Planning);
        let out = shade_owner(&plan, &block_qualia(), -4, 0.5).expect("Block yields a Prune");
        assert_eq!(
            out.intended_move.unwrap().to,
            KanbanColumn::Prune,
            "Block → Prune-where-legal"
        );

        // Hold (neutral qualia, mantissa 0) → None: the owner rests, casts nothing.
        assert!(
            shade_owner(&cw, &QualiaI4_16D(0), 0, 0.5).is_none(),
            "Hold must not produce a move"
        );
    }

    // shade_owner respects the DAG: Block at an absorbing column (Commit) has no
    // legal successor (`next_phases` empty) → None, even though the gate said Block.
    #[test]
    fn shade_owner_at_absorbing_column_yields_nothing() {
        let done = FakeOwner::at(3, KanbanColumn::Commit);
        assert!(
            shade_owner(&done, &flow_qualia(), 4, 1.0).is_none(),
            "Flow at Commit has no forward successor"
        );
        assert!(
            shade_owner(&done, &block_qualia(), -4, 1.0).is_none(),
            "Block at Commit has no Prune successor"
        );
    }

    // ── P4c GATED FALSIFIER: Flow casts + round-trips; Hold is RESCHEDULED ──────
    #[tokio::test]
    async fn gated_flow_casts_hold_is_rescheduled_and_wakes_on_a_later_cycle() {
        let mut sink = FakeWalSink::new();
        // Owner 5 will FLOW (advances); owner 6 will HOLD (rests, re-polled later).
        let mut fleet: HashMap<MailboxId, FakeOwner> = HashMap::from([
            (5, FakeOwner::at(5, KanbanColumn::Planning)),
            (6, FakeOwner::at(6, KanbanColumn::Planning)),
        ]);
        let mut wm: HashMap<MailboxId, Option<u64>> = HashMap::new();
        let mut w = writer_with_moves(&[5, 6]);

        // Cycle 1: both cast Planning→CognitiveWork; the driver applies both.
        let out1 = run_cycle(
            &mut sink,
            &mut fleet,
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            0,
            &mut wm,
            u64::from,
        )
        .await
        .unwrap();
        assert_eq!(out1.applied.applied.len(), 2);

        // P4c: the REAL gate runs. Owner 5 Flows → casts CognitiveWork→Evaluation.
        // Owner 6 gets neutral qualia → Hold → RESCHEDULED (held_owners), not lost.
        let cw = run_cognitive_work_gated(&fleet, &out1.applied, &mut w, |owner| {
            let payload = vec![owner.mailbox_id() as u8];
            if owner.mailbox_id() == 5 {
                Some((flow_qualia(), 4, 0.9, payload)) // FLOW
            } else {
                Some((QualiaI4_16D(0), 0, 0.5, payload)) // HOLD
            }
        });
        assert_eq!(cw.cast, 1, "only the Flow owner cast a next intent");
        assert_eq!(cw.held_owners, vec![6], "the Hold owner is RESCHEDULED");

        // Cycle 2: owner 5 advances to Evaluation; owner 6 rested this round.
        let out2 = run_cycle(
            &mut sink,
            &mut fleet,
            &mut w,
            CycleFrame::new(CycleId(2), DatasetVersion(1)),
            out1.sealed.next_position_base,
            &mut wm,
            u64::from,
        )
        .await
        .unwrap();
        assert_eq!(fleet[&5].phase(), KanbanColumn::Evaluation);
        assert_eq!(fleet[&6].phase(), KanbanColumn::CognitiveWork);

        // The Held owner WAKES: re-poll the held set with (now) Flow qualia —
        // it casts, and the NEXT cycle advances it. No permanent strand.
        let woken = run_cognitive_work_gated_over(&fleet, &cw.held_owners, &mut w, |owner| {
            Some((flow_qualia(), 4, 0.7, vec![owner.mailbox_id() as u8]))
        });
        assert_eq!(woken.cast, 1, "the Held owner cast on re-poll");
        assert!(woken.held_owners.is_empty());

        let out3 = run_cycle(
            &mut sink,
            &mut fleet,
            &mut w,
            CycleFrame::new(CycleId(3), DatasetVersion(2)),
            out2.sealed
                .next_position_base
                .max(out1.sealed.next_position_base),
            &mut wm,
            u64::from,
        )
        .await
        .unwrap();
        assert_eq!(out3.applied.applied.len(), 1);
        assert_eq!(
            fleet[&6].phase(),
            KanbanColumn::Evaluation,
            "a Hold is a reschedule, never a permanent strand"
        );
    }
}
