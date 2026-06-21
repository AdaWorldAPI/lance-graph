//! S4 — the kanban-advance ractor actor (the smallest *true* OUT-leg wire).
//!
//! Per the operator ownership model ("every SoA is owned by its ractor actor —
//! mailbox-as-owner"), the Rubicon phase of a per-mailbox SoA is advanced by the
//! actor that OWNS it, in reaction to a message. There is no owner-registry held
//! by a bridge and no "absent owner" case: the actor's `State` IS the owner
//! ([`MailboxSoaOwner`]).
//!
//! ## Why this is the safe substrate
//!
//! A ractor actor processes ONE message at a time, so the owner sees a strict
//! single-writer: `&mut state` during `handle` cannot alias another writer. That
//! is the compile-time "no aliasing / no data race / no use-after-free" guarantee
//! the canon attributes to mailbox-as-owner (E-CE64-MB-4) — realized here by
//! Rust's `&mut` + ractor's serialized mailbox, not by a lock.
//!
//! ## What it does NOT do (kept honest)
//!
//! This is the OWNER-advance mechanism only. It does NOT resolve a `kanban.*`
//! `UnifiedStep` to a mailbox (that delivery edge is `step_type` → mailbox id →
//! `ractor::registry::where_is` → `cast`, a separate seam) and it does NOT drive
//! the advance from a MUL gate (S2) or a Lance version tick (S3) — those compose
//! ON TOP by sending [`KanbanMsg::Advance`]. The owner advances **itself** via
//! the contract's checked [`MailboxSoaOwner::try_advance_phase`]; an illegal
//! Rubicon edge is a typed [`RubiconTransitionError`], never silent corruption.

use lance_graph_contract::kanban::{KanbanColumn, KanbanMove, RubiconTransitionError};
use lance_graph_contract::mul::i4_eval::gate_decision_i4;
use lance_graph_contract::scheduler::{DatasetVersion, VersionScheduler};
use lance_graph_contract::soa_view::{MailboxSoaOwner, MailboxSoaView};
use lance_graph_contract::QualiaI4_16D;
use ractor::{Actor, ActorProcessingErr, ActorRef, RpcReplyPort};

/// Messages the kanban actor accepts.
pub enum KanbanMsg {
    /// Advance the owned mailbox's Rubicon phase to `to` (checked against the
    /// lifecycle DAG). Replies with the emitted [`KanbanMove`] on a legal edge,
    /// or a [`RubiconTransitionError`] on an illegal one (no mutation occurs).
    Advance {
        to: KanbanColumn,
        reply: RpcReplyPort<Result<KanbanMove, RubiconTransitionError>>,
    },
    /// Read the owned mailbox's current Rubicon phase (no mutation).
    Phase { reply: RpcReplyPort<KanbanColumn> },
    /// **Atomic** S2 step: run the MUL gate (`gate_decision_i4` over `qualia` +
    /// `mantissa`) against the owner's CURRENT phase and advance in ONE message.
    /// Replies `Ok(Some(move))` on advance, `Ok(None)` on Hold, or the typed
    /// error on an illegal edge. Gate-read and transition are serialized with the
    /// owner state (one mailbox message), so a concurrent sender cannot make the
    /// phase read stale between decision and mutation (codex #578).
    MulAdvance {
        qualia: QualiaI4_16D,
        mantissa: i8,
        reply: RpcReplyPort<Result<Option<KanbanMove>, RubiconTransitionError>>,
    },
    /// **Atomic** S3 IN-leg step: a substrate version tick (`at`) advances the
    /// owner along the Rubicon **forward arc** — `phase().next_phases().first()` —
    /// in ONE message, reading the owner's phase at the instant of mutation. This
    /// is the in-actor realization of [`scheduler::NextPhaseScheduler`]'s policy
    /// (`E-SUBSTRATE-IS-THE-SCHEDULER`): a Lance `versions()` event lowers to the
    /// next legal move and the owner applies it. Replies `Some(move)` on advance,
    /// or `None` when the owner is in an absorbing column (`Commit`/`Prune`) — a
    /// **no-op tick is suppressed**, not an error. No error variant: the forward
    /// arc is legal by construction.
    ///
    /// [`scheduler::NextPhaseScheduler`]: lance_graph_contract::scheduler::NextPhaseScheduler
    Tick {
        at: DatasetVersion,
        reply: RpcReplyPort<Option<KanbanMove>>,
    },
}

/// A ractor actor whose `State` IS a [`MailboxSoaOwner`] — the SoA mailbox and
/// its owning actor are the same thing (mailbox-as-owner). On
/// [`KanbanMsg::Advance`] the owner advances its own phase via
/// [`MailboxSoaOwner::try_advance_phase`].
pub struct KanbanActor<O: MailboxSoaOwner> {
    _marker: core::marker::PhantomData<O>,
}

impl<O: MailboxSoaOwner> Default for KanbanActor<O> {
    fn default() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }
}

impl<O> Actor for KanbanActor<O>
where
    O: MailboxSoaOwner + Send + Sync + 'static,
{
    type Msg = KanbanMsg;
    type State = O;
    type Arguments = O;

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        owner: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        // The actor takes ownership of the SoA mailbox at spawn. From here on the
        // ONLY mutator of this owner is this actor's serialized message loop.
        Ok(owner)
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        msg: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match msg {
            KanbanMsg::Advance { to, reply } => {
                // Single-writer by construction: one message at a time. The owner
                // advances ITSELF; nothing else holds it.
                let result = state.try_advance_phase(to);
                let _ = reply.send(result);
            }
            KanbanMsg::Phase { reply } => {
                let _ = reply.send(state.phase());
            }
            KanbanMsg::MulAdvance {
                qualia,
                mantissa,
                reply,
            } => {
                // Gate-decision + transition in ONE serialized message: the gate
                // reads `state.phase()` at the instant of mutation, so a
                // concurrent sender can't make it stale (mailbox-as-owner
                // atomicity — codex #578).
                let result = match mul_target(state.phase(), &qualia, mantissa) {
                    None => Ok(None),                                  // Hold
                    Some(to) => state.try_advance_phase(to).map(Some), // advance
                };
                let _ = reply.send(result);
            }
            KanbanMsg::Tick { at: _, reply } => {
                // Forward-arc advance, atomic against the owner's live phase. The
                // first legal successor is empty exactly for absorbing columns
                // (`Commit`/`Prune`) → `None` suppresses the no-op tick. The arc
                // is legal by construction, so the infallible `advance_phase` is
                // correct here (no `try_`/error path).
                let from = state.phase();
                let moved = from
                    .next_phases()
                    .first()
                    .map(|&to| state.advance_phase(to));
                let _ = reply.send(moved);
            }
        }
        Ok(())
    }
}

// ─── S4 delivery edge: `kanban.<mailbox>.<phase>` → where_is → cast(Advance) ───

/// Error from delivering a `kanban.*` step to its owning actor.
#[derive(Debug, thiserror::Error)]
pub enum KanbanRouteError {
    /// `step_type` was not a well-formed `kanban.<mailbox>.<phase>`.
    #[error("malformed kanban step_type: {0}")]
    BadStepType(String),
    /// No live actor is registered under `<mailbox>`. A routing MISS, distinct
    /// from the (impossible) "no owner" case: a live mailbox is always owned by
    /// its actor — this means the *named* mailbox isn't registered/live.
    #[error("no live mailbox registered as `{0}`")]
    NoMailbox(String),
    /// The owner rejected the transition (illegal Rubicon edge; no mutation).
    #[error("illegal transition {from:?} -> {to:?}")]
    Illegal {
        from: KanbanColumn,
        to: KanbanColumn,
    },
    /// The actor RPC failed (mailbox closed, timeout, …).
    #[error("kanban rpc failed: {0}")]
    Rpc(String),
}

/// Parse a `kanban.<mailbox>.<phase>` step type into `(mailbox, target_phase)`,
/// where `<phase>` is the snake_case [`KanbanColumn`] name (e.g. `cognitive_work`).
/// Returns `None` for anything that isn't a well-formed kanban step.
pub fn parse_kanban_step(step_type: &str) -> Option<(&str, KanbanColumn)> {
    let mut it = step_type.splitn(3, '.');
    match (it.next(), it.next(), it.next()) {
        (Some("kanban"), Some(mailbox), Some(phase)) if !mailbox.is_empty() => {
            phase_from_name(phase).map(|p| (mailbox, p))
        }
        _ => None,
    }
}

/// Snake_case phase name → [`KanbanColumn`] (the `kanban.*` routing vocabulary;
/// inverse of the canonical column names).
fn phase_from_name(name: &str) -> Option<KanbanColumn> {
    Some(match name {
        "planning" => KanbanColumn::Planning,
        "cognitive_work" => KanbanColumn::CognitiveWork,
        "evaluation" => KanbanColumn::Evaluation,
        "commit" => KanbanColumn::Commit,
        "plan" => KanbanColumn::Plan,
        "prune" => KanbanColumn::Prune,
        _ => return None,
    })
}

/// The S4 **delivery edge**: resolve a `kanban.<mailbox>.<phase>` step to its
/// owning ractor actor via the actor system's OWN name registry
/// ([`ractor::registry::where_is`]) and RPC it [`KanbanMsg::Advance`]. The owner
/// advances ITSELF; this only delivers. No bridge-held owner, no `UnifiedStep`
/// field — the target is recovered from the step string + the registry
/// (mailbox-as-owner addressing). Multi-mailbox resolves because `where_is`
/// looks up any registered mailbox by name.
pub async fn deliver_kanban_step(step_type: &str) -> Result<KanbanMove, KanbanRouteError> {
    let (mailbox, to) = parse_kanban_step(step_type)
        .ok_or_else(|| KanbanRouteError::BadStepType(step_type.to_string()))?;
    let cell = ractor::registry::where_is(mailbox)
        .ok_or_else(|| KanbanRouteError::NoMailbox(mailbox.to_string()))?;
    let actor: ActorRef<KanbanMsg> = cell.into();
    let inner = ractor::call!(actor, |reply| KanbanMsg::Advance { to, reply })
        .map_err(|e| KanbanRouteError::Rpc(e.to_string()))?;
    inner.map_err(|e| KanbanRouteError::Illegal {
        from: e.from,
        to: e.to,
    })
}

// ─── S2 driver: MUL gate (`gate_decision_i4`) → owner advance ─────────────────

/// The MUL-gated target phase for `phase` given a node's `qualia` + inference
/// `mantissa`: run the i4 gate ([`gate_decision_i4`]) and lower it to the
/// DAG-legal next phase via [`KanbanColumn::advance_on_gate`] (Flow → forward,
/// Block → Prune-where-legal, Hold → `None`). Pure + integer-only (no f64/NaN).
pub fn mul_target(
    phase: KanbanColumn,
    qualia: &QualiaI4_16D,
    mantissa: i8,
) -> Option<KanbanColumn> {
    let gate = gate_decision_i4(qualia, mantissa);
    phase.advance_on_gate(&gate)
}

/// S2 driver: the MUL gate decides, the owner advances ITSELF — in ONE atomic
/// actor message ([`KanbanMsg::MulAdvance`]). Returns the emitted [`KanbanMove`]
/// on advance, `None` on Hold, or [`KanbanRouteError::Illegal`] on an illegal
/// edge.
///
/// **Atomicity (codex #578):** the gate-read and the transition run inside the
/// SAME serialized mailbox message, so the gate sees the owner's phase at the
/// instant of mutation — two concurrent drivers can't both read a stale
/// `Planning` and collide. (The earlier two-RPC `Phase`-then-`Advance` shape had
/// that race.) `advance_on_gate` only yields a DAG-legal successor, so `Illegal`
/// here would signal a gate/DAG drift bug — surfaced, not panicked.
pub async fn drive_mul_advance(
    actor: &ActorRef<KanbanMsg>,
    qualia: QualiaI4_16D,
    mantissa: i8,
) -> Result<Option<KanbanMove>, KanbanRouteError> {
    let inner = ractor::call!(actor, |reply| KanbanMsg::MulAdvance {
        qualia,
        mantissa,
        reply
    })
    .map_err(|e| KanbanRouteError::Rpc(e.to_string()))?;
    inner.map_err(|e| KanbanRouteError::Illegal {
        from: e.from,
        to: e.to,
    })
}

// ─── S3 IN-leg: substrate version tick → owner forward-arc advance ─────────────

/// S3 driver: a substrate version tick advances the owner along the Rubicon
/// forward arc, in ONE atomic actor message ([`KanbanMsg::Tick`]). Returns the
/// emitted [`KanbanMove`] on advance, or `None` when the owner is absorbing
/// (`Commit`/`Prune`) — the **no-op tick is suppressed** (D-MBX-9-IN,
/// `E-SUBSTRATE-IS-THE-SCHEDULER`).
///
/// **Atomicity:** like [`drive_mul_advance`], the next-phase decision and the
/// transition run inside the SAME serialized mailbox message, so concurrent ticks
/// cannot read a stale phase and collide — they chain along the arc instead
/// (codex #578 lesson, applied to the IN-leg). This is the actor-side realization
/// of the contract's [`NextPhaseScheduler`] policy; use [`drive_scheduled_tick`]
/// when a custom [`VersionScheduler`] policy (version-delta gating, `Plan`/`Prune`
/// over the forward arc, batching) reads a richer view.
///
/// [`NextPhaseScheduler`]: lance_graph_contract::scheduler::NextPhaseScheduler
pub async fn drive_version_tick(
    actor: &ActorRef<KanbanMsg>,
    at: DatasetVersion,
) -> Result<Option<KanbanMove>, KanbanRouteError> {
    ractor::call!(actor, |reply| KanbanMsg::Tick { at, reply })
        .map_err(|e| KanbanRouteError::Rpc(e.to_string()))
}

/// S3 driver (custom policy): drive an arbitrary [`VersionScheduler`] for one
/// version tick. The scheduler **proposes** the next move from `view`; if it
/// yields `Some`, the owner **disposes** it via [`KanbanMsg::Advance`]; `None`
/// **suppresses the no-op tick** ("propose, don't dispose" — the scheduler reads,
/// the owner is the sole mutator).
///
/// Unlike [`drive_version_tick`], the proposal is computed OUTSIDE the owner's
/// message (from the supplied `view`), so it is **advisory**: if the owner's phase
/// changes between the proposal and the `Advance`, the edge may be rejected
/// ([`KanbanRouteError::Illegal`]) rather than silently corrupting — surfaced, not
/// panicked. The returned move is the owner's (authoritative phase transition,
/// witness position, and libet anchor from the REAL mutation) with the
/// **scheduler's `exec` overlaid** — the backend routing tag is the policy's
/// decision, which the owner (defaulting to `Native`) can't make. For the pure
/// forward-arc policy prefer the atomic [`drive_version_tick`]; reach for this
/// only when the policy needs a richer view than the owner computes internally.
pub async fn drive_scheduled_tick<S, V>(
    scheduler: &S,
    view: &V,
    at: DatasetVersion,
    exec: lance_graph_contract::kanban::ExecTarget,
    actor: &ActorRef<KanbanMsg>,
) -> Result<Option<KanbanMove>, KanbanRouteError>
where
    S: VersionScheduler,
    V: MailboxSoaView,
{
    // Propose: lower the version event to the next legal move (or `None`).
    let Some(proposed) = scheduler.on_version(view, at, exec) else {
        return Ok(None); // absorbing / policy-filtered → suppress the no-op tick
    };
    // Dispose: the owner applies it (checked); relay an illegal edge as typed.
    let inner = ractor::call!(actor, |reply| KanbanMsg::Advance {
        to: proposed.to,
        reply
    })
    .map_err(|e| KanbanRouteError::Rpc(e.to_string()))?;
    match inner {
        // The owner's emitted move is authoritative for the phase transition,
        // witness position, and libet anchor (from the REAL mutation), but it
        // defaults to `ExecTarget::Native` and can't know which backend the policy
        // chose — overlay the scheduler's selection so a `Jit`/`SurrealQl`/`Elixir`
        // target is not silently reported/routed as Native (codex #579 P2).
        Ok(mut emitted) => {
            emitted.exec = proposed.exec;
            Ok(Some(emitted))
        }
        Err(e) => Err(KanbanRouteError::Illegal {
            from: e.from,
            to: e.to,
        }),
    }
}

// ─── Capstone run-to-absorbing: drive a mailbox to its terminal column ─────────

/// Drive a mailbox to its **absorbing column** by repeatedly ticking
/// ([`drive_version_tick`]) until the owner reports no further move
/// (`Commit`/`Prune`). Returns the full forward-arc [`KanbanMove`] trace.
///
/// This is the actor-side, lance-free analog of the cognitive loop's
/// run-to-absorbing: it proves the OUT/IN-leg substrate carries a mailbox through
/// a complete Rubicon cycle to a terminal state with no panic and no spurious
/// rejection (the integer-only phase/i4 path cannot produce NaN). The live S3
/// source feeds real `versions()` ticks through the same `drive_version_tick`;
/// here the loop counter stands in for the version stream.
///
/// `max_ticks` bounds the loop defensively. The pure forward arc always reaches
/// `Commit` (`Planning → CognitiveWork → Evaluation → Commit`), so the bound is a
/// guard against a future non-terminating policy, not a normal exit: exceeding it
/// returns [`KanbanRouteError::Rpc`] with a non-termination note rather than
/// looping forever.
pub async fn run_to_absorbing(
    actor: &ActorRef<KanbanMsg>,
    max_ticks: usize,
) -> Result<Vec<KanbanMove>, KanbanRouteError> {
    let mut trace = Vec::new();
    for tick in 0..max_ticks {
        match drive_version_tick(actor, DatasetVersion(tick as u64 + 1)).await? {
            Some(mv) => trace.push(mv),
            None => return Ok(trace), // absorbing column reached — the cycle ended
        }
    }
    Err(KanbanRouteError::Rpc(format!(
        "run_to_absorbing did not reach an absorbing column within {max_ticks} ticks"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::collapse_gate::MailboxId;
    use lance_graph_contract::kanban::ExecTarget;
    use lance_graph_contract::soa_view::MailboxSoaView;

    /// Minimal in-RAM owner (mirrors the contract's `FakeSoa`) — proves the actor
    /// owns and advances a real `MailboxSoaOwner` without any heavy SoA backing.
    struct TestBoard {
        id: MailboxId,
        phase: KanbanColumn,
        cycle: u32,
    }

    impl MailboxSoaView for TestBoard {
        fn mailbox_id(&self) -> MailboxId {
            self.id
        }
        fn n_rows(&self) -> usize {
            0
        }
        fn w_slot(&self) -> u8 {
            (self.id & 0x3F) as u8
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

    impl MailboxSoaOwner for TestBoard {
        fn advance_phase(&mut self, to: KanbanColumn) -> KanbanMove {
            let from = self.phase;
            self.phase = to;
            self.cycle = self.cycle.wrapping_add(1);
            KanbanMove {
                mailbox: self.id,
                from,
                to,
                witness_chain_position: self.cycle,
                libet_offset_us: 0,
                exec: ExecTarget::Native,
            }
        }
    }

    fn board(phase: KanbanColumn) -> TestBoard {
        TestBoard {
            id: 42,
            phase,
            cycle: 0,
        }
    }

    #[tokio::test]
    async fn actor_advances_its_own_phase_on_message() {
        let (actor, handle) = Actor::spawn(
            None,
            KanbanActor::<TestBoard>::default(),
            board(KanbanColumn::Planning),
        )
        .await
        .expect("spawn kanban actor");

        // Legal forward arc Planning -> CognitiveWork: the owner advances itself.
        let mv = ractor::call!(actor, |reply| KanbanMsg::Advance {
            to: KanbanColumn::CognitiveWork,
            reply
        })
        .expect("rpc")
        .expect("legal Rubicon edge");
        assert_eq!(mv.from, KanbanColumn::Planning);
        assert_eq!(mv.to, KanbanColumn::CognitiveWork);

        // The advance persisted in the owned SoA.
        let phase = ractor::call!(actor, |reply| KanbanMsg::Phase { reply }).expect("rpc");
        assert_eq!(phase, KanbanColumn::CognitiveWork);

        actor.stop(None);
        handle.await.expect("actor join");
    }

    #[tokio::test]
    async fn illegal_edge_is_a_typed_error_no_mutation() {
        let (actor, handle) = Actor::spawn(
            None,
            KanbanActor::<TestBoard>::default(),
            board(KanbanColumn::Planning),
        )
        .await
        .expect("spawn kanban actor");

        // Planning -> Commit is NOT a legal Rubicon edge: typed error, no mutation.
        let err = ractor::call!(actor, |reply| KanbanMsg::Advance {
            to: KanbanColumn::Commit,
            reply
        })
        .expect("rpc")
        .expect_err("illegal edge must be rejected");
        assert_eq!(err.from, KanbanColumn::Planning);
        assert_eq!(err.to, KanbanColumn::Commit);

        // Phase unchanged — the owner did not mutate on the rejected edge.
        let phase = ractor::call!(actor, |reply| KanbanMsg::Phase { reply }).expect("rpc");
        assert_eq!(phase, KanbanColumn::Planning);

        actor.stop(None);
        handle.await.expect("actor join");
    }

    #[test]
    fn parse_kanban_step_shapes() {
        assert_eq!(
            parse_kanban_step("kanban.mb42.cognitive_work"),
            Some(("mb42", KanbanColumn::CognitiveWork))
        );
        assert_eq!(parse_kanban_step("lg.foo"), None); // wrong domain
        assert_eq!(parse_kanban_step("kanban.mb42"), None); // no phase
        assert_eq!(parse_kanban_step("kanban..commit"), None); // empty mailbox
        assert_eq!(parse_kanban_step("kanban.mb42.bogus"), None); // unknown phase
    }

    #[tokio::test]
    async fn delivery_edge_resolves_via_registry_then_advances() {
        // Register the owning actor under a name — `where_is` is the actor
        // system's own registry, the S4 addressing source (no bespoke registry).
        let name = "mb-kanban-route-test";
        let (actor, handle) = Actor::spawn(
            Some(name.to_string()),
            KanbanActor::<TestBoard>::default(),
            board(KanbanColumn::Planning),
        )
        .await
        .expect("spawn named");

        // Legal: kanban.<mailbox>.cognitive_work → resolves → owner advances.
        let mv = deliver_kanban_step(&format!("kanban.{name}.cognitive_work"))
            .await
            .expect("delivered + advanced");
        assert_eq!(mv.to, KanbanColumn::CognitiveWork);

        // Unknown mailbox → graceful routing miss (NOT a panic, NOT a no-owner).
        assert!(matches!(
            deliver_kanban_step("kanban.no-such-mailbox.cognitive_work").await,
            Err(KanbanRouteError::NoMailbox(_))
        ));

        // Illegal Rubicon edge → typed Illegal, relayed from the owner.
        assert!(matches!(
            deliver_kanban_step(&format!("kanban.{name}.commit")).await,
            Err(KanbanRouteError::Illegal { .. })
        ));

        // Malformed step type → BadStepType.
        assert!(matches!(
            deliver_kanban_step("lg.noop").await,
            Err(KanbanRouteError::BadStepType(_))
        ));

        actor.stop(None);
        handle.await.expect("actor join");
    }

    #[tokio::test]
    async fn s2_driver_gate_advances_then_holds() {
        let (actor, handle) = Actor::spawn(
            None,
            KanbanActor::<TestBoard>::default(),
            board(KanbanColumn::Planning),
        )
        .await
        .expect("spawn");

        // Flow qualia (warmth/groundedness high, low tension, calibrated) +
        // mantissa>0 → gate Flow → forward advance Planning → CognitiveWork.
        let flow_q = QualiaI4_16D(0).with(3, 4).with(14, 3).with(9, 4).with(1, 2);
        let mv = drive_mul_advance(&actor, flow_q, 4)
            .await
            .expect("driver ok")
            .expect("advanced on Flow");
        assert_eq!(mv.from, KanbanColumn::Planning);
        assert_eq!(mv.to, KanbanColumn::CognitiveWork);

        // Neutral qualia + mantissa 0 → gate Hold → None (owner stays put).
        let held = drive_mul_advance(&actor, QualiaI4_16D(0), 0)
            .await
            .expect("driver ok");
        assert!(held.is_none(), "Hold must not advance");
        let phase = ractor::call!(actor, |reply| KanbanMsg::Phase { reply }).expect("rpc");
        assert_eq!(phase, KanbanColumn::CognitiveWork);

        actor.stop(None);
        handle.await.expect("actor join");
    }

    #[tokio::test]
    async fn concurrent_mul_drivers_serialize_no_spurious_rejection() {
        // codex #578: two concurrent Flow drivers must NOT both read a stale
        // `Planning` and collide. The atomic `MulAdvance` serializes gate+advance
        // in the owner's mailbox, so they chain Planning→CognitiveWork→Evaluation
        // — both succeed, neither is a spurious `Illegal`.
        let (actor, handle) = Actor::spawn(
            None,
            KanbanActor::<TestBoard>::default(),
            board(KanbanColumn::Planning),
        )
        .await
        .expect("spawn");

        let flow = || QualiaI4_16D(0).with(3, 4).with(14, 3).with(9, 4).with(1, 2);
        let a1 = actor.clone();
        let a2 = actor.clone();
        let (r1, r2) = tokio::join!(
            drive_mul_advance(&a1, flow(), 4),
            drive_mul_advance(&a2, flow(), 4),
        );

        // Neither call is a spurious rejection; both advanced along the arc.
        assert!(r1.expect("driver1 ok").is_some(), "first advanced");
        assert!(r2.expect("driver2 ok").is_some(), "second advanced");

        // Serialized chain: Planning → CognitiveWork → Evaluation.
        let phase = ractor::call!(actor, |reply| KanbanMsg::Phase { reply }).expect("rpc");
        assert_eq!(phase, KanbanColumn::Evaluation);

        actor.stop(None);
        handle.await.expect("actor join");
    }

    #[tokio::test]
    async fn version_tick_advances_forward_arc_then_suppresses_at_absorbing() {
        // S3 IN-leg: a version tick advances along the forward arc; once the owner
        // reaches an absorbing column the tick is a suppressed no-op (`None`).
        let (actor, handle) = Actor::spawn(
            None,
            KanbanActor::<TestBoard>::default(),
            board(KanbanColumn::Planning),
        )
        .await
        .expect("spawn");

        // Planning → CognitiveWork → Evaluation → Commit, one tick per version.
        let expected = [
            KanbanColumn::CognitiveWork,
            KanbanColumn::Evaluation,
            KanbanColumn::Commit,
        ];
        for (i, want) in expected.iter().enumerate() {
            let mv = drive_version_tick(&actor, DatasetVersion(i as u64 + 1))
                .await
                .expect("tick ok")
                .expect("non-absorbing advances");
            assert_eq!(mv.to, *want);
        }

        // Commit is absorbing: the next tick advances nothing (no-op suppressed).
        let noop = drive_version_tick(&actor, DatasetVersion(99))
            .await
            .expect("tick ok");
        assert!(noop.is_none(), "absorbing column must suppress the tick");
        let phase = ractor::call!(actor, |reply| KanbanMsg::Phase { reply }).expect("rpc");
        assert_eq!(phase, KanbanColumn::Commit);

        actor.stop(None);
        handle.await.expect("actor join");
    }

    #[tokio::test]
    async fn concurrent_version_ticks_serialize_along_the_arc() {
        // Two concurrent ticks must NOT both read a stale `Planning`; the atomic
        // `Tick` serializes decision+advance in the owner's mailbox, so they chain
        // Planning → CognitiveWork → Evaluation (both advance, neither is lost).
        let (actor, handle) = Actor::spawn(
            None,
            KanbanActor::<TestBoard>::default(),
            board(KanbanColumn::Planning),
        )
        .await
        .expect("spawn");

        let a1 = actor.clone();
        let a2 = actor.clone();
        let (r1, r2) = tokio::join!(
            drive_version_tick(&a1, DatasetVersion(1)),
            drive_version_tick(&a2, DatasetVersion(2)),
        );
        assert!(r1.expect("tick1 ok").is_some(), "first advanced");
        assert!(r2.expect("tick2 ok").is_some(), "second advanced");

        let phase = ractor::call!(actor, |reply| KanbanMsg::Phase { reply }).expect("rpc");
        assert_eq!(phase, KanbanColumn::Evaluation);

        actor.stop(None);
        handle.await.expect("actor join");
    }

    #[tokio::test]
    async fn custom_scheduler_proposes_and_owner_disposes() {
        use lance_graph_contract::scheduler::NextPhaseScheduler;

        // The generic consumer drives the EXISTING `VersionScheduler` trait: the
        // reference `NextPhaseScheduler` proposes from a view, the owner disposes.
        let (actor, handle) = Actor::spawn(
            None,
            KanbanActor::<TestBoard>::default(),
            board(KanbanColumn::Planning),
        )
        .await
        .expect("spawn");

        // View mirrors the owner's current phase; scheduler proposes CognitiveWork.
        let view = board(KanbanColumn::Planning);
        let mv = drive_scheduled_tick(
            &NextPhaseScheduler,
            &view,
            DatasetVersion(1),
            ExecTarget::Native,
            &actor,
        )
        .await
        .expect("scheduled ok")
        .expect("forward arc proposed + disposed");
        assert_eq!(mv.from, KanbanColumn::Planning);
        assert_eq!(mv.to, KanbanColumn::CognitiveWork);

        // An absorbing view → scheduler yields `None` → suppressed, no RPC needed.
        let absorbing_view = board(KanbanColumn::Commit);
        let noop = drive_scheduled_tick(
            &NextPhaseScheduler,
            &absorbing_view,
            DatasetVersion(2),
            ExecTarget::Native,
            &actor,
        )
        .await
        .expect("scheduled ok");
        assert!(noop.is_none(), "absorbing proposal is suppressed");

        actor.stop(None);
        handle.await.expect("actor join");
    }

    #[tokio::test]
    async fn scheduled_tick_preserves_non_native_exec_target() {
        use lance_graph_contract::scheduler::NextPhaseScheduler;

        // codex #579 P2: the scheduler selects the backend; the owner defaults to
        // `Native`. The returned move must carry the scheduler's exec, NOT be
        // flattened to the owner's Native default.
        for exec in [ExecTarget::Jit, ExecTarget::SurrealQl, ExecTarget::Elixir] {
            // Fresh owner per exec so the phase starts at Planning each iteration.
            let (actor, handle) = Actor::spawn(
                None,
                KanbanActor::<TestBoard>::default(),
                board(KanbanColumn::Planning),
            )
            .await
            .expect("spawn");

            let view = board(KanbanColumn::Planning);
            let mv =
                drive_scheduled_tick(&NextPhaseScheduler, &view, DatasetVersion(1), exec, &actor)
                    .await
                    .expect("scheduled ok")
                    .expect("forward arc proposed + disposed");
            assert_eq!(mv.to, KanbanColumn::CognitiveWork);
            assert_eq!(
                mv.exec, exec,
                "scheduler's backend must survive, not be overwritten with Native"
            );

            actor.stop(None);
            handle.await.expect("actor join");
        }
    }

    #[tokio::test]
    async fn run_to_absorbing_drives_a_full_rubicon_cycle_no_nan_no_panic() {
        // Capstone run-NaN (actor-side, lance-free): a mailbox driven from
        // Planning runs to the absorbing Commit column through the REAL actor
        // messages — it terminates, never panics, never emits a spurious Illegal,
        // and the trace is the deterministic forward arc. The integer phase/i4
        // path cannot produce NaN, so a green run here IS the actor-side half of
        // the loop's run-NaN answer.
        let (actor, handle) = Actor::spawn(
            None,
            KanbanActor::<TestBoard>::default(),
            board(KanbanColumn::Planning),
        )
        .await
        .expect("spawn");

        let trace = run_to_absorbing(&actor, 16)
            .await
            .expect("reaches an absorbing column within the bound");

        // Forward arc: Planning → CognitiveWork → Evaluation → Commit (3 moves).
        let arc: Vec<_> = trace.iter().map(|m| m.to).collect();
        assert_eq!(
            arc,
            vec![
                KanbanColumn::CognitiveWork,
                KanbanColumn::Evaluation,
                KanbanColumn::Commit,
            ]
        );
        // Every move en route is a legal Rubicon edge (no corruption).
        for m in &trace {
            assert!(
                m.from.can_transition_to(m.to),
                "{:?} -> {:?} must be legal",
                m.from,
                m.to
            );
        }

        // The owner rests in the absorbing column: a further run is empty, and
        // the phase is unchanged (idempotent at rest — no spurious advance/error).
        let again = run_to_absorbing(&actor, 4)
            .await
            .expect("idempotent at the absorbing column");
        assert!(again.is_empty(), "absorbing column yields no further moves");
        let phase = ractor::call!(actor, |reply| KanbanMsg::Phase { reply }).expect("rpc");
        assert_eq!(phase, KanbanColumn::Commit);

        actor.stop(None);
        handle.await.expect("actor join");
    }
}
