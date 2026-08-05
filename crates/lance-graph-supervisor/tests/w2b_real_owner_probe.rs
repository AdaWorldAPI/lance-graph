//! D-V3-W2b integration probe, direct-owner form: the REAL production
//! `MailboxSoaOwner` (`cognitive_shader_driver::mailbox_soa::MailboxSoA`)
//! implements the Rubicon lifecycle DAG through the contract trait — legal
//! advances persist on the real SoA, illegal edges are rejected with no
//! mutation.
//!
//! ## 2026-08-05 migration — the actor surface this probe drove is deleted
//!
//! The original W2b closed the gap "`KanbanActor` never spawned over a real
//! `MailboxSoA`" by driving `KanbanMsg::{Advance, Phase}` RPCs. That actor
//! surface is DELETED (`E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1`); what
//! remains worth pinning is the half that was never about messages: the real
//! owner's lifecycle DAG behind `try_advance_phase`, exercised through plain
//! `&mut` — which IS the single-writer guarantee (a second mutator is a
//! compile error, not a runtime property this test could miss). The probe
//! also exercises the replacement visibility surface ([`PhaseCensus`]) over
//! the real SoA.

#[cfg(feature = "supervisor")]
mod w2b_real_owner_probe {
    use cognitive_shader_driver::mailbox_soa::MailboxSoA;
    use lance_graph_contract::kanban::KanbanColumn;
    use lance_graph_contract::soa_view::{MailboxSoaOwner, MailboxSoaView};
    use lance_graph_supervisor::PhaseCensus;

    /// Small capacity — the probe only needs the owner's phase column, not a
    /// realistic row count. Mirrors `mailbox_soa.rs`'s own unit tests
    /// (`MailboxSoA<8>` in `test_mailbox_soa_new_zero`).
    type ProbeMailbox = MailboxSoA<8>;

    /// Construct a real `MailboxSoA` the same way `mailbox_soa.rs`'s own
    /// tests do: `MailboxSoA::<N>::new(mailbox_id, w_slot, threshold)`
    /// followed by `set_populated` (W1c discipline). This mirrors the
    /// crate's own construction idiom, not an invented shape.
    fn real_mailbox() -> ProbeMailbox {
        let mut mb = MailboxSoA::new(
            /* mailbox_id */ 77, /* w_slot */ 3, /* threshold */ 1.0,
        );
        // Declare 1 populated row so MailboxSoaView::n_rows() is non-zero,
        // matching how a real spawn would declare its logical size
        // (`MailboxSoA::set_populated` docs: "mirrors fixing BindSpace::len
        // at construction"). `phase()` itself is a mailbox-level field, not
        // per-row — this keeps the probe's owner representative of a real
        // spawn rather than a zero-row empty shell.
        mb.set_populated(1);
        mb
    }

    #[test]
    fn w2b_real_owner_two_legal_advances_persist_on_the_real_soa() {
        let mut mb = real_mailbox();
        assert_eq!(
            mb.phase(),
            KanbanColumn::Planning,
            "MailboxSoA::new starts in Planning"
        );

        // Legal edge #1: Planning -> CognitiveWork, through the contract's
        // owner trait on the exclusive borrow.
        let mv1 = mb
            .try_advance_phase(KanbanColumn::CognitiveWork)
            .expect("Planning -> CognitiveWork is a legal Rubicon edge");
        assert_eq!(mv1.from, KanbanColumn::Planning);
        assert_eq!(mv1.to, KanbanColumn::CognitiveWork);
        assert_eq!(mb.phase(), KanbanColumn::CognitiveWork);

        // Legal edge #2: CognitiveWork -> Evaluation.
        let mv2 = mb
            .try_advance_phase(KanbanColumn::Evaluation)
            .expect("CognitiveWork -> Evaluation is a legal Rubicon edge");
        assert_eq!(mv2.from, KanbanColumn::CognitiveWork);
        assert_eq!(mv2.to, KanbanColumn::Evaluation);
        assert_eq!(
            mb.phase(),
            KanbanColumn::Evaluation,
            "the real MailboxSoA row reflects both advances, read back via MailboxSoaView"
        );
    }

    #[test]
    fn w2b_real_owner_illegal_edge_rejected_no_mutation_on_the_real_soa() {
        let mut mb = real_mailbox();

        // Planning -> Commit is NOT a legal Rubicon edge — the typed
        // RubiconTransitionError surfaces from the real owner's lifecycle
        // DAG, with NO mutation on the real row.
        let err = mb
            .try_advance_phase(KanbanColumn::Commit)
            .expect_err("Planning -> Commit must be rejected by the real owner's lifecycle DAG");
        assert_eq!(err.from, KanbanColumn::Planning);
        assert_eq!(err.to, KanbanColumn::Commit);
        assert_eq!(
            mb.phase(),
            KanbanColumn::Planning,
            "rejected transition must not mutate the real MailboxSoA row"
        );
    }

    #[test]
    fn w2b_phase_census_observes_the_real_soa_without_mutating_it() {
        // The replacement visibility surface over REAL owners: a mixed pair
        // of mailboxes is counted correctly (can-fire), and driving both to
        // absorbing columns flips at_rest (can-stay-silent) — all through
        // `&self` reads, no message, no RPC.
        let mut a = real_mailbox();
        let mut b = real_mailbox();
        a.try_advance_phase(KanbanColumn::CognitiveWork)
            .expect("legal edge");

        let mid = PhaseCensus::observe([&a, &b]);
        assert_eq!(mid.total(), 2);
        assert_eq!(mid.count(KanbanColumn::CognitiveWork), 1);
        assert_eq!(mid.count(KanbanColumn::Planning), 1);
        assert!(!mid.at_rest(), "a mid-arc fleet is not at rest");

        // Drive both along legal arcs into absorbing columns.
        a.try_advance_phase(KanbanColumn::Evaluation)
            .expect("legal edge");
        a.try_advance_phase(KanbanColumn::Commit)
            .expect("legal edge");
        b.try_advance_phase(KanbanColumn::Prune)
            .expect("legal edge");

        let done = PhaseCensus::observe([&a, &b]);
        assert_eq!(done.absorbing(), 2);
        assert!(done.at_rest(), "a fully absorbed fleet reads at rest");
        // Observation mutated nothing: phases are exactly where the owner
        // left them.
        assert_eq!(a.phase(), KanbanColumn::Commit);
        assert_eq!(b.phase(), KanbanColumn::Prune);
    }
}
