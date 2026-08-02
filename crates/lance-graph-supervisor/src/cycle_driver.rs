//! P4a / P4b — the cycle loop-closure driver (supervisor-side).
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
//!   `try_advance_phase`. **Every unrepresented owner is left byte-identical.**
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
//! **Interim conservative rule:** at most **one** durable phase transition per
//! owner per sealed cycle. A second sealed transition for an already-advanced
//! owner is **deferred** (held for the next sealed horizon), not applied.
//!
//! ## Ownership (D-MBX spine)
//!
//! The supervisor is the **exclusive runtime owner** of the fleet; the planner
//! **decides** (produces the casts + the persistence contract). This driver lives
//! in the supervisor and depends **one-way** on the planner (planner never deps
//! supervisor — no cycle). It applies through `MailboxSoaOwner::try_advance_phase`
//! directly — no ractor message bus needed for P4a/P4b.
//!
//! ## Honesty
//!
//! The control loop closes here; the **durability leg stays the contract-probe
//! fake** until the concrete `LanceShardSink` lands (`compile+test green ≠
//! storage proven`, the Ladybug rule). `apply_sealed_transitions` reads **no**
//! dataset — the version was already sealed by `seal_cycle`.

use std::collections::{HashMap, HashSet};

use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::KanbanMove;
use lance_graph_contract::scheduler::DatasetVersion;
use lance_graph_contract::soa_view::{MailboxSoaOwner, MailboxSoaView};

use lance_graph_planner::batch_writer::BatchWriter;
use lance_graph_planner::persist_sink::{
    persist_cycle, CycleFrame, CycleId, PersistError, SweepSlot, WalSink,
};

/// One sealed paired transition — a member of the **sparse** set a sealed cycle
/// publishes. Carries `stream_position` so P4b applies transitions in canonical
/// order (later-position wins the interim one-per-owner slot). Boring by design:
/// it says *which owner advances where*, nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SealedTransition {
    /// The canonical order key inherited from the cast's `SweepSlot`.
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
    /// The one version this cycle sealed into.
    pub version: DatasetVersion,
    /// Only the owners that cast a `paired_move` — the sparse subset.
    pub transitions: Vec<SealedTransition>,
}

/// P4b output — the effect of applying a sealed cycle's sparse transition set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedCycle {
    /// The version whose sealed transitions were applied.
    pub version: DatasetVersion,
    /// One move per **advanced** owner (distinct owners; interim ≤1 per cycle).
    pub applied: Vec<KanbanMove>,
    /// Extra same-owner sealed transitions held for the next horizon (interim
    /// one-transition-per-owner rule).
    pub deferred: usize,
    /// Sealed transitions whose owner is not registered in the fleet (a
    /// registration gap — surfaced as a count, not a crash; the durable move
    /// stays in the log for a later recovery pass).
    pub missing: usize,
}

/// A resolvable collection of mailbox owners — the supervisor's production fleet.
/// P4b resolves each sealed transition's owner through this; **unrepresented
/// owners are never resolved and never touched** (that is what keeps them
/// byte-identical).
pub trait MailboxFleet {
    /// The concrete owner type this fleet holds.
    type Owner: MailboxSoaOwner;
    /// Resolve a mailbox to its owner, or `None` if it is not registered.
    fn owner_mut(&mut self, id: MailboxId) -> Option<&mut Self::Owner>;
}

/// A `HashMap` keyed by `MailboxId` is the simplest concrete fleet — used by the
/// production supervisor's owner registry and by the tests.
impl<O: MailboxSoaOwner> MailboxFleet for HashMap<MailboxId, O> {
    type Owner = O;
    fn owner_mut(&mut self, id: MailboxId) -> Option<&mut O> {
        self.get_mut(&id)
    }
}

/// **P4a (drain).** Map the fleet's staged `BatchWriter` casts into `SweepSlot`s
/// for one cycle. One slot per cast: `stream_position` = the monotonic `CastId`
/// (cast order), `paired_move` = the cast's **first** intended move (interim: one
/// intended move per cast), `row` resolved by the caller's `row_of`, `payload` =
/// the drained descriptor bytes. Draining clears the writer's payload staging.
///
/// Casts with no recorded move become no-step landings (`paired_move = None`) —
/// they still coalesce into the cycle image but contribute no transition.
#[must_use]
pub fn collect_casts(
    writer: &mut BatchWriter<Vec<u8>>,
    cycle: CycleId,
    mut row_of: impl FnMut(MailboxId) -> u64,
) -> Vec<SweepSlot> {
    // Drain first (ends the &mut borrow), then read the intent moves immutably.
    let drained: Vec<_> = writer.drain_pending_payloads().collect();
    drained
        .into_iter()
        .filter_map(|(cast, payload)| {
            let owner = writer.on_behalf_of(cast)?;
            let paired_move = writer
                .intent_moves(cast)
                .and_then(|moves| moves.first().copied());
            Some(SweepSlot {
                cycle,
                stream_position: cast.0,
                owner,
                row: row_of(owner),
                paired_move,
                payload,
            })
        })
        .collect()
}

/// **P4a (seal).** Freeze the collected casts into one cycle: read out the
/// **sparse** sealed paired-transition set (the slots that carry a `paired_move`,
/// ordered by `stream_position`), then `persist_cycle` → **exactly one WAL write,
/// one `DatasetVersion`**. `persist_cycle`'s own guards still run
/// (cross-owner-move reject, cast/frame cycle-match). The version is published
/// before any owner advances (P4b).
pub async fn seal_cycle<S: WalSink>(
    sink: &S,
    frame: CycleFrame,
    casts: Vec<SweepSlot>,
) -> Result<SealedCycle, PersistError> {
    // Read the transitions out BEFORE `persist_cycle` takes ownership of `casts`.
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
    let version = persist_cycle(sink, frame, casts).await?;
    Ok(SealedCycle {
        version,
        transitions,
    })
}

/// **P4b.** Apply **only** the sealed sparse transition set to the fleet. Iterate
/// the transitions in canonical `stream_position` order; resolve each owner and
/// apply **one** legal `try_advance_phase`. **Every unrepresented owner is left
/// byte-identical** — it is never resolved, never touched.
///
/// Interim rule: at most one durable transition per owner per cycle — a second
/// sealed transition for an already-advanced owner is **deferred**. A sealed
/// move whose `from` no longer matches the owner's phase is corruption
/// ([`PersistError::StalePhase`]); a cross-owner move is
/// [`PersistError::OwnerMismatch`] (defence in depth — `persist_cycle` already
/// rejected it). On such an error the fleet is left mid-apply; re-drive from the
/// persisted watermark via `persist_sink::recover_and_apply`.
///
/// Reads **no** dataset: the version was sealed by [`seal_cycle`]; this is a pure
/// in-memory fan over the sparse set (no `scan_sealed`, no `versions`, no
/// `drive_once`).
pub fn apply_sealed_transitions<F: MailboxFleet>(
    fleet: &mut F,
    sealed: &SealedCycle,
) -> Result<AppliedCycle, PersistError> {
    let mut ordered: Vec<&SealedTransition> = sealed.transitions.iter().collect();
    ordered.sort_by_key(|t| t.stream_position);

    let mut applied = Vec::new();
    let mut advanced: HashSet<MailboxId> = HashSet::new();
    let mut deferred = 0usize;
    let mut missing = 0usize;

    for t in ordered {
        // Interim ≤1-per-owner: a later sealed move for an already-advanced owner
        // waits for the next sealed horizon.
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
            return Err(PersistError::OwnerMismatch {
                move_owner: t.mv.mailbox,
                landing_owner: t.owner,
            });
        }
        // Corruption guard: the sealed move's `from` must match the owner's phase.
        if t.mv.from != owner.phase() {
            return Err(PersistError::StalePhase {
                owner_phase: owner.phase(),
                move_from: t.mv.from,
            });
        }
        let step = owner
            .try_advance_phase(t.mv.to)
            .map_err(PersistError::Illegal)?;
        applied.push(step);
        advanced.insert(t.owner);
    }

    Ok(AppliedCycle {
        version: sealed.version,
        applied,
        deferred,
        missing,
    })
}

/// Convenience: run one full cycle — **P4a** (drain the writer + seal) then
/// **P4b** (apply the sparse set). The one seam a running loop calls per cycle.
/// Returns both the sealed cycle and the applied effect.
pub async fn run_cycle<S, F>(
    sink: &S,
    fleet: &mut F,
    writer: &mut BatchWriter<Vec<u8>>,
    frame: CycleFrame,
    row_of: impl FnMut(MailboxId) -> u64,
) -> Result<(SealedCycle, AppliedCycle), PersistError>
where
    S: WalSink,
    F: MailboxFleet,
{
    let casts = collect_casts(writer, frame.cycle, row_of);
    let sealed = seal_cycle(sink, frame, casts).await?;
    let applied = apply_sealed_transitions(fleet, &sealed)?;
    Ok((sealed, applied))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::kanban::{ExecTarget, KanbanColumn};
    use lance_graph_contract::soa_view::MailboxSoaView;
    use lance_graph_planner::persist_sink::{DetachedCycleBatch, LandedSlot, WriteFailed};
    use std::sync::atomic::{AtomicU64, Ordering};
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

    // ── The WAL sink fake — counts writes AND reads (base-fenced) ────────────────
    struct SealedRec {
        frame: CycleFrame,
        version: DatasetVersion,
    }
    struct FakeWalSink {
        sealed: Mutex<Vec<SealedRec>>,
        next_version: AtomicU64,
        wal_writes: AtomicU64,
        reads: AtomicU64, // scan_sealed + versions — MUST stay 0 across P4a+P4b
    }
    impl FakeWalSink {
        fn new() -> Self {
            Self {
                sealed: Mutex::new(Vec::new()),
                next_version: AtomicU64::new(1),
                wal_writes: AtomicU64::new(0),
                reads: AtomicU64::new(0),
            }
        }
        fn wal_writes(&self) -> u64 {
            self.wal_writes.load(Ordering::SeqCst)
        }
        fn reads(&self) -> u64 {
            self.reads.load(Ordering::SeqCst)
        }
    }
    impl WalSink for FakeWalSink {
        async fn commit_cycle(
            &self,
            base: DatasetVersion,
            batch: DetachedCycleBatch,
        ) -> Result<DatasetVersion, WriteFailed> {
            let mut sealed = self.sealed.lock().unwrap();
            let head = sealed.last().map_or(DatasetVersion(0), |s| s.version);
            if base != head {
                return Err(WriteFailed(format!("stale base {base:?}, head {head:?}")));
            }
            self.wal_writes.fetch_add(1, Ordering::SeqCst);
            let version = DatasetVersion(self.next_version.fetch_add(1, Ordering::SeqCst));
            sealed.push(SealedRec {
                frame: batch.frame,
                version,
            });
            Ok(version)
        }
        async fn scan_sealed(
            &self,
            _from: Option<DatasetVersion>,
        ) -> Result<Vec<LandedSlot>, WriteFailed> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            Ok(Vec::new())
        }
        async fn versions(&self) -> Result<Vec<(CycleId, DatasetVersion)>, WriteFailed> {
            self.reads.fetch_add(1, Ordering::SeqCst);
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
        let sink = FakeWalSink::new();
        let owners: Vec<MailboxId> = (0..100).collect();
        let mut w = writer_with_moves(&owners);
        let casts = collect_casts(&mut w, CycleId(1), |id| u64::from(id));
        assert_eq!(casts.len(), 100, "one slot per staged cast");
        assert_eq!(sink.wal_writes(), 0, "collecting writes no WAL");

        let sealed = seal_cycle(&sink, CycleFrame::new(CycleId(1), DatasetVersion(0)), casts)
            .await
            .unwrap();
        assert_eq!(sink.wal_writes(), 1, "100 casts → exactly ONE WAL write");
        assert_eq!(sealed.version, DatasetVersion(1), "→ exactly one version");
        assert_eq!(
            sealed.transitions.len(),
            100,
            "all 100 casts carried a move → sparse set = 100 here"
        );
        // A second drain of the same writer is empty (payload staging cleared).
        let again = collect_casts(&mut w, CycleId(1), |id| u64::from(id));
        assert!(again.is_empty(), "drain cleared the writer's staging");
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

        let sink = FakeWalSink::new();
        let mut w = writer_with_moves(&represented);
        let casts = collect_casts(&mut w, CycleId(1), |id| u64::from(id));
        let sealed = seal_cycle(&sink, CycleFrame::new(CycleId(1), DatasetVersion(0)), casts)
            .await
            .unwrap();
        assert_eq!(
            sealed.transitions.len(),
            17,
            "sparse sealed set = 17, not 64k"
        );

        let applied = apply_sealed_transitions(&mut fleet, &sealed).unwrap();

        // Exactly the 17 represented owners advanced.
        assert_eq!(applied.applied.len(), 17, "exactly 17 owners advanced");
        assert_eq!(applied.deferred, 0);
        assert_eq!(applied.missing, 0);
        assert_eq!(applied.version, DatasetVersion(1));

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

        let sink = FakeWalSink::new();
        let mut w: BatchWriter<Vec<u8>> = BatchWriter::new();
        for id in 0..1_000u32 {
            w.cast(id, vec![], vec![0x00]); // no move → no transition
        }
        let casts = collect_casts(&mut w, CycleId(1), |id| u64::from(id));
        let sealed = seal_cycle(&sink, CycleFrame::new(CycleId(1), DatasetVersion(0)), casts)
            .await
            .unwrap();
        assert!(sealed.transitions.is_empty(), "no moves → empty sparse set");

        let applied = apply_sealed_transitions(&mut fleet, &sealed).unwrap();
        assert!(
            applied.applied.is_empty(),
            "a version alone advances nobody"
        );
        assert_eq!(fleet, before, "the whole fleet is byte-identical");
        assert_eq!(sink.wal_writes(), 1, "the cycle still sealed one version");
    }

    // ── P4b: interim one-transition-per-owner defers the rest ────────────────────
    #[tokio::test]
    async fn p4b_defers_a_second_transition_for_the_same_owner() {
        let mut fleet: HashMap<MailboxId, FakeOwner> =
            HashMap::from([(42, FakeOwner::at(42, KanbanColumn::Planning))]);
        let sink = FakeWalSink::new();
        // Two casts for owner 42 in one cycle.
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
        let casts = collect_casts(&mut w, CycleId(1), |id| u64::from(id));
        let sealed = seal_cycle(&sink, CycleFrame::new(CycleId(1), DatasetVersion(0)), casts)
            .await
            .unwrap();
        assert_eq!(sealed.transitions.len(), 2, "both cast a move");

        let applied = apply_sealed_transitions(&mut fleet, &sealed).unwrap();
        assert_eq!(
            applied.applied.len(),
            1,
            "≤1 durable transition per owner per cycle"
        );
        assert_eq!(applied.deferred, 1, "the second transition is deferred");
        assert_eq!(
            fleet[&42].phase(),
            KanbanColumn::CognitiveWork,
            "owner advanced exactly one step (the first in stream order)"
        );
    }

    // ── P4b: a stale sealed move (from ≠ phase) is a corruption error ───────────
    #[tokio::test]
    async fn p4b_stale_phase_is_a_corruption_error() {
        // Owner is at CognitiveWork; the sealed move claims from=Planning.
        let mut fleet: HashMap<MailboxId, FakeOwner> =
            HashMap::from([(7, FakeOwner::at(7, KanbanColumn::CognitiveWork))]);
        let sealed = SealedCycle {
            version: DatasetVersion(1),
            transitions: vec![SealedTransition {
                stream_position: 0,
                owner: 7,
                mv: mv(7, KanbanColumn::Planning, KanbanColumn::CognitiveWork),
            }],
        };
        assert!(matches!(
            apply_sealed_transitions(&mut fleet, &sealed),
            Err(PersistError::StalePhase { .. })
        ));
    }

    // ── P4b: a sealed move for an unregistered owner is counted, not a crash ────
    #[tokio::test]
    async fn p4b_missing_owner_is_counted_not_applied() {
        let mut fleet: HashMap<MailboxId, FakeOwner> = HashMap::new(); // empty fleet
        let sealed = SealedCycle {
            version: DatasetVersion(1),
            transitions: vec![SealedTransition {
                stream_position: 0,
                owner: 99,
                mv: mv(99, KanbanColumn::Planning, KanbanColumn::CognitiveWork),
            }],
        };
        let applied = apply_sealed_transitions(&mut fleet, &sealed).unwrap();
        assert_eq!(applied.missing, 1);
        assert!(applied.applied.is_empty());
    }

    // ── End-to-end: run_cycle closes P4a→P4b in one call ────────────────────────
    #[tokio::test]
    async fn run_cycle_seals_then_applies_the_sparse_set() {
        let mut fleet: HashMap<MailboxId, FakeOwner> = (0..10)
            .map(|id| (id, FakeOwner::at(id, KanbanColumn::Planning)))
            .collect();
        let sink = FakeWalSink::new();
        let mut w = writer_with_moves(&[3, 7]); // only owners 3 and 7 produce a move

        let (sealed, applied) = run_cycle(
            &sink,
            &mut fleet,
            &mut w,
            CycleFrame::new(CycleId(1), DatasetVersion(0)),
            |id| u64::from(id),
        )
        .await
        .unwrap();

        assert_eq!(sealed.version, DatasetVersion(1));
        assert_eq!(applied.applied.len(), 2, "exactly owners 3 and 7 advanced");
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
}
