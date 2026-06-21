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
}
