//! Seam falsifier: `SurrealMailboxView` → `VersionScheduler::on_version` → `KanbanMove`.
//!
//! ADDITIVE. Asserts the IN-direction wiring the module doc-comments *claim*
//! ("a version tick, fed to `VersionScheduler::on_version`, lowers to the next
//! legal Rubicon move over a `MailboxSoaView`") actually holds end-to-end —
//! driven by the REAL `SurrealMailboxView` read-glove, not a hand-rolled fake,
//! across the FULL Rubicon lifecycle (the in-crate unit test covers one tick).
//!
//! Each test states its kill-condition: the observation that would prove the
//! seam WRONG. None of these pin "current behaviour" — they pin the contract
//! the doc-comments assert.

use lance_graph_contract::kanban::{ExecTarget, KanbanColumn};
use lance_graph_contract::scheduler::{DatasetVersion, NextPhaseScheduler, VersionScheduler};
use lance_graph_contract::soa_view::MailboxSoaView;
use surreal_container::view::SurrealMailboxView;

/// One empty-column view at `phase` (the scheduler reads only `phase()` +
/// `mailbox_id()` + `current_cycle()`; row columns are irrelevant to lowering).
fn view_at(phase: KanbanColumn) -> SurrealMailboxView<'static> {
    SurrealMailboxView::from_columns(7, 5, 13, phase, &[], &[], &[], &[])
}

/// KILL-CONDITION: if any forward tick lowers to a column the Rubicon DAG does
/// not sanction (`can_transition_to` == false), the scheduler is mis-wired.
/// Walks the whole canonical arc through the real view.
#[test]
fn full_rubicon_arc_lowers_to_legal_successors() {
    let scheduler = NextPhaseScheduler;
    // The canonical forward arc the doc-comment names.
    let arc = [
        (KanbanColumn::Planning, KanbanColumn::CognitiveWork),
        (KanbanColumn::CognitiveWork, KanbanColumn::Evaluation),
        (KanbanColumn::Evaluation, KanbanColumn::Commit),
        (KanbanColumn::Plan, KanbanColumn::Planning), // re-deliberate
    ];
    for (i, (from, want_to)) in arc.iter().enumerate() {
        let mv = scheduler
            .on_version(&view_at(*from), DatasetVersion(i as u64 + 1), ExecTarget::Native)
            .unwrap_or_else(|| panic!("{from:?} must schedule a forward move"));
        assert_eq!(mv.from, *from, "move.from must echo the observed phase");
        assert_eq!(mv.to, *want_to, "{from:?} must lower to {want_to:?}");
        // The DAG itself must sanction the emitted edge — the falsifier.
        assert!(
            from.can_transition_to(mv.to),
            "scheduler emitted an illegal Rubicon edge {from:?}->{:?}",
            mv.to
        );
    }
}

/// KILL-CONDITION: an absorbing column (`Commit`/`Prune`) is cycle-end — it must
/// schedule NOTHING. If `on_version` returns `Some` here, the lifecycle would
/// never terminate (a mailbox past Commit would keep being advanced).
#[test]
fn absorbing_columns_schedule_no_move() {
    let scheduler = NextPhaseScheduler;
    for phase in [KanbanColumn::Commit, KanbanColumn::Prune] {
        assert!(phase.is_absorbing(), "{phase:?} must be absorbing (precondition)");
        assert!(
            scheduler
                .on_version(&view_at(phase), DatasetVersion(99), ExecTarget::Native)
                .is_none(),
            "{phase:?} is absorbing — it must schedule no advance"
        );
    }
}

/// KILL-CONDITION: the Libet `-550_000 µs` readiness-potential anchor is stamped
/// ONLY on the Planning→CognitiveWork Σ-commit crossing. If any other tick
/// carries a non-zero offset (or that crossing carries zero), the Libet anchor
/// is mis-placed.
#[test]
fn libet_anchor_only_on_sigma_commit_crossing() {
    let scheduler = NextPhaseScheduler;

    let crossing = scheduler
        .on_version(&view_at(KanbanColumn::Planning), DatasetVersion(1), ExecTarget::Native)
        .expect("Planning advances");
    assert_eq!(crossing.to, KanbanColumn::CognitiveWork);
    assert_eq!(
        crossing.libet_offset_us, -550_000,
        "the Σ-commit crossing must carry the -550ms Libet anchor"
    );

    for from in [KanbanColumn::CognitiveWork, KanbanColumn::Evaluation, KanbanColumn::Plan] {
        let mv = scheduler
            .on_version(&view_at(from), DatasetVersion(2), ExecTarget::Native)
            .expect("non-absorbing column advances");
        assert_eq!(
            mv.libet_offset_us, 0,
            "{from:?} is not the Σ-commit crossing — Libet offset must be 0"
        );
    }
}

/// KILL-CONDITION: the scheduler is a pure function of (view, version, exec).
/// Same inputs MUST yield the same move — the determinism the whole
/// version-tick → lifecycle mechanism stands on. A difference proves hidden
/// state leaked in.
#[test]
fn lowering_is_deterministic() {
    let scheduler = NextPhaseScheduler;
    let a = scheduler
        .on_version(&view_at(KanbanColumn::CognitiveWork), DatasetVersion(7), ExecTarget::Jit)
        .expect("advances");
    let b = scheduler
        .on_version(&view_at(KanbanColumn::CognitiveWork), DatasetVersion(7), ExecTarget::Jit)
        .expect("advances");
    assert_eq!(a, b, "same (view, version, exec) must lower to the same move");
}

/// KILL-CONDITION: the `exec` backend selector must ride through the lowering
/// onto the emitted move unchanged (it names where the precipitated plan runs).
/// If it is dropped or rewritten, the planner's execution-target choice is lost.
#[test]
fn exec_target_rides_onto_the_move() {
    let scheduler = NextPhaseScheduler;
    for exec in [ExecTarget::Native, ExecTarget::Jit, ExecTarget::SurrealQl, ExecTarget::Elixir] {
        let mv = scheduler
            .on_version(&view_at(KanbanColumn::Planning), DatasetVersion(3), exec)
            .expect("advances");
        assert_eq!(mv.exec, exec, "exec target must survive the lowering");
    }
}
