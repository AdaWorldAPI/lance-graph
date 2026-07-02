//! D-V3-W1e probe-first: three failing probes for the W1b ahead-firing batch
//! writer + W1c delegation cache, pinned against
//! `lance_graph_planner::batch_writer::BatchWriter`.
//!
//! All three are `#[ignore]`d — the writer's methods are `todo!()` stubs.
//! Un-ignore in the W1b implementation commit once the bodies are filled in.
//!
//! Uses the REAL shipped kanban contract types
//! (`lance_graph_contract::kanban::{KanbanColumn, KanbanMove, ExecTarget}`,
//! `lance_graph_contract::collapse_gate::MailboxId`) — no hand-rolled
//! composites (F12).

use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
use lance_graph_planner::batch_writer::BatchWriter;

/// Build a `KanbanMove` from a `mailbox`/`from`/`to` triple using only public
/// constructors/fields on the shipped contract type — no hand-rolled bit math.
fn make_move(mailbox: u32, from: KanbanColumn, to: KanbanColumn, witness: u32) -> KanbanMove {
    KanbanMove {
        mailbox,
        from,
        to,
        witness_chain_position: witness,
        libet_offset_us: 0,
        exec: ExecTarget::Native,
    }
}

/// Probe 1 (W1b): cast() makes intent moves visible on the board AHEAD of any
/// ack; ack() then removes the cast from unacked().
#[test]
#[ignore = "probe-first: W1b mechanism pending — un-ignore in the W1b implementation commit"]
fn probe_ahead_update_ordering() {
    let mut writer: BatchWriter<()> = BatchWriter::new();

    let moves = vec![
        make_move(7, KanbanColumn::Planning, KanbanColumn::CognitiveWork, 0),
        make_move(7, KanbanColumn::CognitiveWork, KanbanColumn::Evaluation, 1),
        make_move(7, KanbanColumn::Evaluation, KanbanColumn::Commit, 2),
    ];

    let cast = writer.cast(7, moves.clone(), ());

    // AHEAD: intent is visible on the board BEFORE any ack.
    assert_eq!(writer.intent_moves(cast), Some(moves.as_slice()));

    writer.ack(cast);

    // After ack, the cast is no longer in the unacked (crash-replay) surface.
    assert!(!writer.unacked().contains(&cast));
}

/// Probe 2 (M24): a cast that is never acked stays on the crash-replay
/// surface (`unacked()`), and its intent moves remain replayable.
#[test]
#[ignore = "probe-first: W1b mechanism pending — un-ignore in the W1b implementation commit"]
fn probe_kill_after_cast_replay() {
    let mut writer: BatchWriter<()> = BatchWriter::new();

    let moves = vec![make_move(
        11,
        KanbanColumn::Planning,
        KanbanColumn::CognitiveWork,
        0,
    )];

    let cast = writer.cast(11, moves.clone(), ());
    // Deliberately no ack() — simulates a crash between cast and ack.

    let unacked = writer.unacked();
    assert_eq!(unacked, vec![cast]);

    let replayed = writer
        .intent_moves(cast)
        .expect("unacked cast must still have replayable intent moves");
    assert!(!replayed.is_empty());
    assert_eq!(replayed, moves.as_slice());
}

/// Probe 3 (W1c): resolve_owner() calls the resolver on the first lookup for
/// a mailbox (cache miss) and skips it on the second lookup for the same
/// mailbox (cache hit), returning the same owner both times.
#[test]
#[ignore = "probe-first: W1b mechanism pending — un-ignore in the W1b implementation commit"]
fn probe_delegation_miss_then_hit() {
    let mut writer: BatchWriter<()> = BatchWriter::new();

    let on_behalf: u32 = 3;
    let mut resolver_calls = 0u32;

    let (owner_first, was_hit_first) = writer.resolve_owner(on_behalf, |mailbox| {
        resolver_calls += 1;
        mailbox + 1000 // arbitrary deterministic "resolved owner" transform
    });
    assert!(!was_hit_first, "first resolve for a mailbox must be a cache miss");
    assert_eq!(resolver_calls, 1);

    let (owner_second, was_hit_second) = writer.resolve_owner(on_behalf, |mailbox| {
        resolver_calls += 1;
        mailbox + 1000
    });
    assert!(was_hit_second, "second resolve for the same mailbox must be a cache hit");
    assert_eq!(resolver_calls, 1, "resolver must not be called again on cache hit");
    assert_eq!(owner_first, owner_second);
}
