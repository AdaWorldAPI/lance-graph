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
use lance_graph_contract::soa_view::MailboxSoaOwner;
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
}
