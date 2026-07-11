//! W1b ahead-firing batch writer — the kanban board IS the write-ahead log (M24).
//!
//! `cast()` records intent moves AHEAD of any storage ack; `ack()` confirms
//! at the `LanceVersion` the sink assigned (the CastId↔LanceVersion join
//! wiring the WAL into the temporal classifier; see `crate::temporal`);
//! `unacked()` is the crash-replay surface. Payload-generic: the writer never
//! inspects `P` (DTO purity — ownership rides the cast pairing, never the DTO).
//!
//! **Zero-copy sink (operator ruling, plan Addendum-6):** `P` is a DESCRIPTOR
//! — (mailbox, dirty row-range, cycle) — never owned delta bytes. Deltas stay
//! in the SoA backing store; the sink reads them through
//! `NodeRowPacket::as_le_bytes` at flush time. The sink drains EAGERLY
//! (ASAP on cast, background), and the write masks the thinking and vice
//! versa: the thinker reports (casts) and moves on — "melden macht frei" —
//! it is NEVER refused because earlier casts are unacked. Stacked casts on
//! the same row are stacked WAL entries; the sink reads the LIVE store at
//! flush, so one physical flush coalesces all earlier intents for a row
//! (last-state-wins; the move log keeps the full ordered history).
//!
//! Uses the REAL shipped kanban contract types
//! ([`lance_graph_contract::kanban::KanbanMove`],
//! [`lance_graph_contract::kanban::KanbanColumn`],
//! [`lance_graph_contract::collapse_gate::MailboxId`]) — this module does not
//! mint a parallel `KanbanMove`; see the D-MBX-A6 Outcome adapter context in
//! `crate::strategy::style_strategy`.

use std::collections::{BTreeMap, HashMap};

use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::{ExecTarget, KanbanMove};
use lance_graph_contract::scheduler::{DatasetVersion, VersionScheduler};
use lance_graph_contract::soa_view::MailboxSoaView;

use crate::temporal::LanceVersion;

/// Identity of one `cast()` — a write-ahead intent record on the kanban board.
///
/// `next_id` is monotonically increasing, so `CastId`'s derived `Ord` gives
/// insertion order — this is what lets `board`/`acked` use `BTreeMap` and
/// have `unacked()` iterate in cast order for free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CastId(pub u64);

/// Ahead-firing batch writer: intent (`cast`) is visible on the board before
/// any ack; `unacked()` is the M24 crash-replay surface; `resolve_owner` is
/// the W1c delegation cache (resolve-once, cache-hit thereafter).
pub struct BatchWriter<P> {
    /// Monotonic id generator for the next cast.
    next_id: u64,
    /// Board: intent moves recorded per cast, keyed by `CastId`, alongside the
    /// mailbox the cast was recorded on behalf of. Visible between `cast()`
    /// and `ack()` (and beyond, for crash-replay via `unacked()`).
    /// `BTreeMap` (not `HashMap`) so iteration is in ascending `CastId` order
    /// — i.e. cast (insertion) order, matching `unacked()`'s ordering contract.
    board: BTreeMap<CastId, (MailboxId, Vec<KanbanMove>)>,
    /// Casts that have been confirmed (acked), mapped to the `LanceVersion`
    /// the sink assigned when it flushed — the CastId↔LanceVersion join
    /// wiring the WAL into the temporal classifier (see `crate::temporal`).
    /// A cast present in `board` but absent from `acked` is the crash-replay
    /// surface (`unacked()`).
    acked: BTreeMap<CastId, LanceVersion>,
    /// W1c delegation cache: `on_behalf` mailbox -> resolved owner mailbox.
    delegation_cache: HashMap<MailboxId, MailboxId>,
    /// Payloads recorded per cast (payload-generic; the writer never inspects `P`).
    pending_payloads: Vec<(CastId, P)>,
}

impl<P> Default for BatchWriter<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> BatchWriter<P> {
    /// Construct an empty batch writer.
    pub fn new() -> Self {
        Self {
            next_id: 0,
            board: BTreeMap::new(),
            acked: BTreeMap::new(),
            delegation_cache: HashMap::new(),
            pending_payloads: Vec::new(),
        }
    }

    /// AHEAD: records the intent (moves visible on the board) BEFORE any ack.
    /// Returns the cast id.
    ///
    /// "Melden macht frei" (plan Addendum-7): casting is REPORTING, never
    /// refused because earlier casts on the same mailbox are still unacked —
    /// stacked casts are stacked WAL entries, each with its own `CastId`.
    pub fn cast(&mut self, on_behalf: MailboxId, moves: Vec<KanbanMove>, payload: P) -> CastId {
        let cast = CastId(self.next_id);
        self.next_id += 1;
        self.board.insert(cast, (on_behalf, moves));
        self.pending_payloads.push((cast, payload));
        cast
    }

    /// Confirmation — marks the cast acked at the Lance version the sink
    /// assigned (the CastId↔LanceVersion join wiring the WAL into the
    /// temporal classifier; see planner `temporal.rs`).
    pub fn ack(&mut self, cast: CastId, version: LanceVersion) {
        self.acked.insert(cast, version);
    }

    /// **The SLA gate (operator rulings, 2026-07-10/11):
    /// an ack-gated advance for SLA/audit-bearing work.** Records the ack,
    /// then lowers the assigned `LanceVersion` to the next legal
    /// [`KanbanMove`] proposal via the provided [`VersionScheduler`] over
    /// `view`.
    ///
    /// **Tier note (E-ACK-HARD-GATE-VS-KANBANSTEP-STREAM-1; name refined to
    /// "SLA gate" 2026-07-11):** this is the TICKET tier — work items that
    /// want an explicit SLA and an auditable goalstate advance only on
    /// confirmed durability, and the WAL board + `acked` map is their audit
    /// trail. The STREAM/reasoning tier (the kanbanstep) never routes through
    /// here: a writer that already holds the version it committed fires
    /// `on_version → try_advance_phase(&mut)` inline — thinking never waits on
    /// an ack. This SLA-gate mechanism is preserved (not retired) and is
    /// repurposed on the OGAR consumer side as the **actionhandler queue** —
    /// where action handlers (tickets, e-mails, …) wait for their durability
    /// ack before advancing.
    ///
    /// The loop this closes — the StreamDto "can't stop thinking" lineage:
    ///
    /// ```text
    /// think → cast (fire-and-forget: the thinker is freed at once)
    ///       → sink drains async → Lance ack
    ///       → THIS method: ack ⇒ next-move proposal (version tick)
    ///       → owner disposes it → the board reprioritizes the live set
    ///       → thinking, which never stopped, sees the new priorities
    /// ```
    ///
    /// The cognitive-shader-driver NEVER waits on scheduling or writing:
    /// `cast` returns immediately ("melden macht frei"), the write is async,
    /// and the board update is a *consequence of the write completing*, not
    /// something the thinker schedules. Orchestration is self-updating —
    /// every completed write releases the next kanban update through the
    /// gate.
    ///
    /// Propose-don't-dispose is preserved: the writer only PROPOSES; the
    /// owner (`MailboxSoaOwner` / the KanbanActor) applies the move — the
    /// sole-mutator proof is untouched. `None` = the view is absorbing or
    /// policy-filtered (the loop rests; a no-op tick is suppressed, never an
    /// error). A `Planning → CognitiveWork` proposal carries the Libet
    /// anchor, so `elevation::cycle::CycleBudget::from_move` opens the next
    /// cycle's window from this same proposal (M12).
    ///
    /// **At-least-once delivery (codex P2 on #674):** only the FIRST
    /// unacked→acked transition releases a proposal. A duplicate ack (async
    /// sink retry, version-watcher replaying the same `CastId`) returns
    /// `None` WITHOUT overwriting the first recorded version — first ack
    /// wins, keeping the CastId↔LanceVersion temporal join stable. A stray
    /// `CastId` (never cast on this board) also returns `None` and records
    /// nothing. The low-level [`ack`](Self::ack) stays the unconditional
    /// recording primitive; dedup lives at the gate, because the gate is
    /// what would otherwise multiply lifecycle moves.
    pub fn ack_and_propose<S, V>(
        &mut self,
        cast: CastId,
        version: LanceVersion,
        scheduler: &S,
        view: &V,
        exec: ExecTarget,
    ) -> Option<KanbanMove>
    where
        S: VersionScheduler,
        V: MailboxSoaView,
    {
        if !self.board.contains_key(&cast) || self.acked.contains_key(&cast) {
            return None;
        }
        self.ack(cast, version);
        scheduler.on_version(view, DatasetVersion(version), exec)
    }

    /// The `LanceVersion` a cast was acked at, if it has been acked.
    #[must_use]
    pub fn acked_version(&self, cast: CastId) -> Option<LanceVersion> {
        self.acked.get(&cast).copied()
    }

    /// Crash-replay surface (M24): casts recorded but not yet acked, in
    /// ascending `CastId` (cast) order.
    #[must_use]
    pub fn unacked(&self) -> Vec<CastId> {
        self.board
            .keys()
            .filter(|cast| !self.acked.contains_key(cast))
            .copied()
            .collect()
    }

    /// Board read: intent moves recorded for a cast (visible between cast and ack).
    #[must_use]
    pub fn intent_moves(&self, cast: CastId) -> Option<&[KanbanMove]> {
        self.board.get(&cast).map(|(_, moves)| moves.as_slice())
    }

    /// Board read: the mailbox a cast was recorded ON BEHALF OF — the
    /// delegation-audit surface (W4a cast pairing: consumers pair the owner
    /// at cast; wardens verify the pairing here, never on the DTO).
    #[must_use]
    pub fn on_behalf_of(&self, cast: CastId) -> Option<MailboxId> {
        self.board.get(&cast).map(|(on_behalf, _)| *on_behalf)
    }

    /// W1c delegation cache: resolve an owner once, cache; returns `(owner, was_cache_hit)`.
    pub fn resolve_owner(
        &mut self,
        on_behalf: MailboxId,
        resolver: impl FnOnce(MailboxId) -> MailboxId,
    ) -> (MailboxId, bool) {
        if let Some(&owner) = self.delegation_cache.get(&on_behalf) {
            return (owner, true);
        }
        let owner = resolver(on_behalf);
        self.delegation_cache.insert(on_behalf, owner);
        (owner, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::kanban::KanbanColumn;
    use lance_graph_contract::scheduler::NextPhaseScheduler;

    /// Minimal view at a given phase (mirrors elevation::cycle's PhaseView).
    struct PhaseView(KanbanColumn);
    impl MailboxSoaView for PhaseView {
        fn mailbox_id(&self) -> MailboxId {
            7
        }
        fn n_rows(&self) -> usize {
            0
        }
        fn w_slot(&self) -> u8 {
            7
        }
        fn current_cycle(&self) -> u32 {
            0
        }
        fn phase(&self) -> KanbanColumn {
            self.0
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

    #[test]
    fn ack_is_the_next_kanban_trigger_and_the_thinker_never_waited() {
        let mut w: BatchWriter<u8> = BatchWriter::new();

        // Fire-and-forget: the thinker casts and is freed at once — no ack
        // exists yet, nothing to wait on, and further casts stack freely.
        let c1 = w.cast(7, vec![], 0);
        let c2 = w.cast(7, vec![], 1);
        assert_eq!(w.unacked(), vec![c1, c2], "thinking never blocked on I/O");

        // The async sink completes c1: the ack ITSELF proposes the next
        // kanban update (Planning → CognitiveWork), carrying the Libet
        // anchor — orchestration self-updates off write completions.
        let mv = w
            .ack_and_propose(
                c1,
                41,
                &NextPhaseScheduler,
                &PhaseView(KanbanColumn::Planning),
                ExecTarget::Native,
            )
            .expect("write completion triggers the next board update");
        assert_eq!(mv.from, KanbanColumn::Planning);
        assert_eq!(mv.to, KanbanColumn::CognitiveWork);
        assert!(mv.libet_offset_us < 0, "Σ-commit opens the next window");
        assert_eq!(w.acked_version(c1), Some(41), "the ack was recorded");
        assert_eq!(w.unacked(), vec![c2], "c2 still in flight, still no wait");
    }

    #[test]
    fn ack_gated_chain_walks_the_arc_off_write_completions_alone() {
        // Three async write completions, each triggering the next update:
        // the board walks Planning → CognitiveWork → Evaluation → Commit
        // with NO scheduler call from the thinker's side.
        let mut w: BatchWriter<u8> = BatchWriter::new();
        let mut phase = KanbanColumn::Planning;
        for (i, expected) in [
            KanbanColumn::CognitiveWork,
            KanbanColumn::Evaluation,
            KanbanColumn::Commit,
        ]
        .into_iter()
        .enumerate()
        {
            let cast = w.cast(7, vec![], i as u8);
            let mv = w
                .ack_and_propose(
                    cast,
                    i as u64 + 1,
                    &NextPhaseScheduler,
                    &PhaseView(phase),
                    ExecTarget::Native,
                )
                .expect("non-absorbing view advances");
            assert_eq!(mv.to, expected);
            phase = mv.to; // the owner disposed the proposal; next view reflects it
        }

        // At the absorbing column the gate rests: ack recorded, no proposal,
        // no error — the no-op tick is suppressed, never a deadlock.
        let last = w.cast(7, vec![], 9);
        let rest = w.ack_and_propose(
            last,
            99,
            &NextPhaseScheduler,
            &PhaseView(KanbanColumn::Commit),
            ExecTarget::Native,
        );
        assert!(rest.is_none(), "absorbing view: the loop rests");
        assert_eq!(w.acked_version(last), Some(99));
    }

    /// codex P2 on #674: at-least-once sink delivery must not multiply
    /// lifecycle moves — only the FIRST unacked→acked transition proposes;
    /// duplicate acks keep the first version; stray casts record nothing.
    #[test]
    fn duplicate_and_stray_acks_never_release_twice() {
        let mut w: BatchWriter<u8> = BatchWriter::new();
        let c1 = w.cast(7, vec![], 0);

        // First transition: proposes.
        let first = w.ack_and_propose(
            c1,
            41,
            &NextPhaseScheduler,
            &PhaseView(KanbanColumn::Planning),
            ExecTarget::Native,
        );
        assert!(first.is_some(), "first unacked->acked transition releases");

        // Sink retry / watcher replay of the same CastId: NO second
        // proposal, and the first recorded version wins (the temporal
        // join stays stable even if the retry carries a later version).
        let dup = w.ack_and_propose(
            c1,
            42,
            &NextPhaseScheduler,
            &PhaseView(KanbanColumn::Planning),
            ExecTarget::Native,
        );
        assert!(dup.is_none(), "duplicate ack must not release a second move");
        assert_eq!(w.acked_version(c1), Some(41), "first ack wins");

        // Stray CastId (never cast on this board): no proposal, no record.
        let stray = CastId(9999);
        let ghost = w.ack_and_propose(
            stray,
            77,
            &NextPhaseScheduler,
            &PhaseView(KanbanColumn::Planning),
            ExecTarget::Native,
        );
        assert!(ghost.is_none(), "stray cast id is ignored");
        assert_eq!(w.acked_version(stray), None, "nothing recorded for it");
    }
}
