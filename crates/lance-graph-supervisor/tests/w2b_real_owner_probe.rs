//! D-V3-W2b integration probe: KanbanActor spawned over the REAL production
//! `MailboxSoaOwner` (`cognitive_shader_driver::mailbox_soa::MailboxSoA`), not
//! the in-file `TestBoard` fake that `kanban_actor.rs`'s own unit tests use.
//!
//! Closes the gap named in D-V3-W2b: until this probe, `KanbanActor<O>` was
//! only ever exercised against `kanban_actor::tests::TestBoard` — a minimal
//! in-RAM stand-in with no SoA columns. This probe proves the SAME actor
//! message surface (`KanbanMsg::Advance` / `KanbanMsg::Phase`) drives the
//! REAL owner's `try_advance_phase` (via the contract's `MailboxSoaOwner`
//! trait), that illegal transitions are rejected with no mutation on the
//! real SoA, and that the actor is the ONLY path this probe ever uses to
//! mutate the row (no direct `advance_phase`/`try_advance_phase` call from
//! the probe itself — only through `KanbanMsg`).
//!
//! Spec: `.claude/board/*` D-V3-W2b (KanbanActor never spawned over real
//! MailboxSoA — this file closes that gap).

#[cfg(feature = "supervisor")]
mod w2b_real_owner_probe {
    use cognitive_shader_driver::mailbox_soa::MailboxSoA;
    use lance_graph_contract::kanban::KanbanColumn;
    use lance_graph_contract::soa_view::MailboxSoaView;
    use lance_graph_supervisor::kanban_actor::{KanbanActor, KanbanMsg};
    use ractor::Actor;

    /// Small capacity — the probe only needs the owner's phase column, not a
    /// realistic row count. Mirrors `mailbox_soa.rs`'s own unit tests
    /// (`MailboxSoA<8>` in `test_mailbox_soa_new_zero`).
    type ProbeMailbox = MailboxSoA<8>;

    /// Construct a real `MailboxSoA` the same way `mailbox_soa.rs`'s own
    /// tests do: `MailboxSoA::<N>::new(mailbox_id, w_slot, threshold)`
    /// followed by `set_populated` (W1c discipline). This mirrors the
    /// crate's own construction idiom, not an invented shape.
    fn real_mailbox() -> ProbeMailbox {
        let mut mb = MailboxSoA::new(/* mailbox_id */ 77, /* w_slot */ 3, /* threshold */ 1.0);
        // Declare 1 populated row so MailboxSoaView::n_rows() is non-zero,
        // matching how a real spawn would declare its logical size
        // (`MailboxSoA::set_populated` docs: "mirrors fixing BindSpace::len
        // at construction"). `phase()` itself is a mailbox-level field, not
        // per-row, so this is not required for the phase assertions below —
        // it is here so the probe's owner is representative of a real spawn
        // rather than a zero-row empty shell.
        mb.set_populated(1);
        mb
    }

    #[tokio::test]
    async fn w2b_real_owner_two_legal_advances_persist_on_the_real_soa() {
        let mb = real_mailbox();
        assert_eq!(
            mb.phase(),
            KanbanColumn::Planning,
            "MailboxSoA::new starts in Planning (mirrors TestBoard's board(Planning) helper \
             in kanban_actor.rs's own unit tests)"
        );

        let (actor, handle) = Actor::spawn(None, KanbanActor::<ProbeMailbox>::default(), mb)
            .await
            .expect("spawn kanban actor over the REAL MailboxSoA");

        // Legal edge #1: Planning -> CognitiveWork, driven ONLY through the actor.
        let mv1 = ractor::call!(actor, |reply| KanbanMsg::Advance {
            to: KanbanColumn::CognitiveWork,
            reply
        })
        .expect("rpc")
        .expect("Planning -> CognitiveWork is a legal Rubicon edge");
        assert_eq!(mv1.from, KanbanColumn::Planning);
        assert_eq!(mv1.to, KanbanColumn::CognitiveWork);

        // Read back through MailboxSoaView::phase() (via KanbanMsg::Phase) —
        // the real SoA row reflects the advance.
        let phase1 = ractor::call!(actor, |reply| KanbanMsg::Phase { reply }).expect("rpc");
        assert_eq!(phase1, KanbanColumn::CognitiveWork);

        // Legal edge #2: CognitiveWork -> Evaluation.
        let mv2 = ractor::call!(actor, |reply| KanbanMsg::Advance {
            to: KanbanColumn::Evaluation,
            reply
        })
        .expect("rpc")
        .expect("CognitiveWork -> Evaluation is a legal Rubicon edge");
        assert_eq!(mv2.from, KanbanColumn::CognitiveWork);
        assert_eq!(mv2.to, KanbanColumn::Evaluation);

        let phase2 = ractor::call!(actor, |reply| KanbanMsg::Phase { reply }).expect("rpc");
        assert_eq!(
            phase2,
            KanbanColumn::Evaluation,
            "the real MailboxSoA row reflects both advances, read back via MailboxSoaView"
        );

        actor.stop(None);
        handle.await.expect("actor join");
    }

    #[tokio::test]
    async fn w2b_real_owner_illegal_edge_rejected_no_mutation_on_the_real_soa() {
        let mb = real_mailbox();
        let (actor, handle) = Actor::spawn(None, KanbanActor::<ProbeMailbox>::default(), mb)
            .await
            .expect("spawn kanban actor over the REAL MailboxSoA");

        // Planning -> Commit is NOT a legal Rubicon edge (same DAG the
        // in-file TestBoard tests exercise) — must surface the typed
        // RubiconTransitionError from MailboxSoaOwner::try_advance_phase,
        // relayed through the actor's Advance message, with NO mutation on
        // the real row.
        let err = ractor::call!(actor, |reply| KanbanMsg::Advance {
            to: KanbanColumn::Commit,
            reply
        })
        .expect("rpc")
        .expect_err("Planning -> Commit must be rejected by the real owner's lifecycle DAG");
        assert_eq!(err.from, KanbanColumn::Planning);
        assert_eq!(err.to, KanbanColumn::Commit);

        // The real SoA's phase column is UNCHANGED after the rejected edge.
        let phase = ractor::call!(actor, |reply| KanbanMsg::Phase { reply }).expect("rpc");
        assert_eq!(
            phase,
            KanbanColumn::Planning,
            "rejected transition must not mutate the real MailboxSoA row"
        );

        actor.stop(None);
        handle.await.expect("actor join");
    }

    #[tokio::test]
    async fn w2b_real_owner_actor_is_the_sole_mutator_structural_check() {
        // Structural proof (mailbox-as-owner, E-CE64-MB-4): the probe never
        // calls `MailboxSoaOwner::advance_phase` / `try_advance_phase`
        // directly on a `MailboxSoA` value it holds after spawn — the real
        // `MailboxSoA` is MOVED into `Actor::spawn` (ownership transfer),
        // and the only handle this test touches from that point on is the
        // `ActorRef<KanbanMsg>`. Any mutation not routed through
        // `KanbanMsg::Advance` would be a compile error here (`mb` is no
        // longer in scope), not a runtime bug this test could silently miss.
        let mb = real_mailbox();
        let (actor, handle) = Actor::spawn(None, KanbanActor::<ProbeMailbox>::default(), mb)
            .await
            .expect("spawn kanban actor over the REAL MailboxSoA");
        // `mb` was moved into `Actor::spawn` above and is not usable here —
        // the only remaining handle to the owner is `actor`.

        let mv = ractor::call!(actor, |reply| KanbanMsg::Advance {
            to: KanbanColumn::CognitiveWork,
            reply
        })
        .expect("rpc")
        .expect("legal edge applied via the actor, the only mutation surface reachable here");
        assert_eq!(mv.to, KanbanColumn::CognitiveWork);

        actor.stop(None);
        handle.await.expect("actor join");
    }
}
