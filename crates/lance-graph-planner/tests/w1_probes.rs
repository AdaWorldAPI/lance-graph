//! D-V3-W1e probe-first: four probes for the W1b ahead-firing batch writer +
//! W1c delegation cache, pinned against
//! `lance_graph_planner::batch_writer::BatchWriter`.
//!
//! Live (W1b implementation landed) — all four run and pass.
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

    writer.ack(cast, 1);

    // After ack, the cast is no longer in the unacked (crash-replay) surface.
    assert!(!writer.unacked().contains(&cast));

    // The CastId<->LanceVersion join: the ack recorded the assigned version.
    assert_eq!(writer.acked_version(cast), Some(1));
}

/// Probe 2 (M24): a cast that is never acked stays on the crash-replay
/// surface (`unacked()`), and its intent moves remain replayable.
#[test]
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
fn probe_delegation_miss_then_hit() {
    let mut writer: BatchWriter<()> = BatchWriter::new();

    let on_behalf: u32 = 3;
    let mut resolver_calls = 0u32;

    let (owner_first, was_hit_first) = writer.resolve_owner(on_behalf, |mailbox| {
        resolver_calls += 1;
        mailbox + 1000 // arbitrary deterministic "resolved owner" transform
    });
    assert!(
        !was_hit_first,
        "first resolve for a mailbox must be a cache miss"
    );
    assert_eq!(resolver_calls, 1);

    let (owner_second, was_hit_second) = writer.resolve_owner(on_behalf, |mailbox| {
        resolver_calls += 1;
        mailbox + 1000
    });
    assert!(
        was_hit_second,
        "second resolve for the same mailbox must be a cache hit"
    );
    assert_eq!(
        resolver_calls, 1,
        "resolver must not be called again on cache hit"
    );
    assert_eq!(owner_first, owner_second);
}

/// Probe 4 (M24 / operator ruling "melden macht frei", plan Addendum-7):
/// casting is REPORTING, and reporting frees the thinker — the writer NEVER
/// refuses a cast because earlier casts on the same mailbox are still
/// unacked. Three stacked casts are three WAL entries: distinct ids, full
/// ordered history retained, acks retire independently. (Physical sink
/// coalescing — one flush of the live store satisfying all earlier intents
/// for a row — is sink-side behavior, exercised in the W1b implementation
/// tests, not at this API surface.)
#[test]
fn probe_stacked_casts_never_refused() {
    let mut writer: BatchWriter<()> = BatchWriter::new();

    let mv = |w| make_move(7, KanbanColumn::Planning, KanbanColumn::CognitiveWork, w);

    // Three stacked writes on the SAME mailbox, zero acks in between.
    let c1 = writer.cast(7, vec![mv(0)], ());
    let c2 = writer.cast(7, vec![mv(1)], ());
    let c3 = writer.cast(7, vec![mv(2)], ());

    // No refusal: three distinct WAL entries, cast order preserved.
    assert_ne!(c1, c2);
    assert_ne!(c2, c3);
    assert_eq!(writer.unacked(), vec![c1, c2, c3]);

    // Every stacked intent stays independently replayable.
    assert!(writer.intent_moves(c1).is_some());
    assert!(writer.intent_moves(c2).is_some());
    assert!(writer.intent_moves(c3).is_some());

    // Acks retire independently and in any order.
    writer.ack(c2, 1);
    assert_eq!(writer.unacked(), vec![c1, c3]);
    writer.ack(c1, 2);
    writer.ack(c3, 3);
    assert!(writer.unacked().is_empty());
}
