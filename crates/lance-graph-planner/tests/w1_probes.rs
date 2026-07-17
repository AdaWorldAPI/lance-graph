//! D-V3-W1e probe-first: probes for the W1b ahead-firing batch writer +
//! W1c delegation cache, pinned against
//! `lance_graph_planner::batch_writer::BatchWriter`.
//!
//! There is no confirmation bookkeeping in the writer (operator ruling
//! 2026-07-17, E-ACK-ELIMINATED-1): durability evidence is the written
//! row's own `LanceVersion` in Lance, read through `temporal.rs`; replay
//! is a temporal READ comparing recorded intents against what Lance holds.
//! These probes therefore pin exactly two things: intent recording and
//! delegation caching.
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

/// Probe 1 (W1b): cast() makes intent moves visible immediately — AHEAD of
/// any storage write completing.
#[test]
fn probe_ahead_update_ordering() {
    let mut writer: BatchWriter<()> = BatchWriter::new();

    let moves = vec![
        make_move(7, KanbanColumn::Planning, KanbanColumn::CognitiveWork, 0),
        make_move(7, KanbanColumn::CognitiveWork, KanbanColumn::Evaluation, 1),
        make_move(7, KanbanColumn::Evaluation, KanbanColumn::Commit, 2),
    ];

    let cast = writer.cast(7, moves.clone(), ());

    // AHEAD: intent is visible immediately.
    assert_eq!(writer.intent_moves(cast), Some(moves.as_slice()));
    assert_eq!(writer.casts(), vec![cast]);
}

/// Probe 2 (M24): recorded intents stay readable after a simulated crash —
/// replay compares them against what Lance holds (a temporal READ, done
/// elsewhere); the writer's only obligation is that the intent record
/// survives and stays readable.
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
    // Simulates a crash right after cast: nothing else happens.

    assert_eq!(writer.casts(), vec![cast]);
    let replayed = writer
        .intent_moves(cast)
        .expect("cast must still have replayable intent moves");
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

/// Probe 4 (operator ruling "melden macht frei", plan Addendum-7):
/// casting is REPORTING, and reporting frees the thinker — the writer NEVER
/// refuses a cast. Three stacked casts on the same mailbox are three intent
/// records: distinct ids, full ordered history retained. (Physical sink
/// coalescing — one flush of the live store satisfying all earlier intents
/// for a row — is sink-side behavior, not at this API surface.)
#[test]
fn probe_stacked_casts_never_refused() {
    let mut writer: BatchWriter<()> = BatchWriter::new();

    let mv = |w| make_move(7, KanbanColumn::Planning, KanbanColumn::CognitiveWork, w);

    // Three stacked writes on the SAME mailbox, nothing in between.
    let c1 = writer.cast(7, vec![mv(0)], ());
    let c2 = writer.cast(7, vec![mv(1)], ());
    let c3 = writer.cast(7, vec![mv(2)], ());

    // No refusal: three distinct intent records, cast order preserved.
    assert_ne!(c1, c2);
    assert_ne!(c2, c3);
    assert_eq!(writer.casts(), vec![c1, c2, c3]);

    // Every stacked intent stays independently replayable.
    assert!(writer.intent_moves(c1).is_some());
    assert!(writer.intent_moves(c2).is_some());
    assert!(writer.intent_moves(c3).is_some());
}
