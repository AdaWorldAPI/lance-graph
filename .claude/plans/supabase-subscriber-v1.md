# Supabase-shape Subscriber Flow Wire-up — v1

> **Status:** In progress (2026-04-24)
> **Owner:** @callcenter-specialist, @bus-compiler
> **Scope:** `lance-graph-callcenter` crate only (plus `external_membrane.rs` if trait surface bends)
> **Depends on:** none (substrate-independent)

## Goal

Flip `LanceMembrane::subscribe()` from GHOST to PARTIAL. Ship DM-4 `LanceVersionWatcher` + DM-6 `DrainTask`. Close the Outside-BBB loop so Lance dataset version bumps fire notifications to subscribers with filtered `CognitiveEventRow` payloads.

## Deliverables

- **DM-4a** — Swap `Subscription` associated type from `mpsc::Receiver<u64>` to `tokio::sync::watch::Receiver<CognitiveEventRow>` in `lance_membrane.rs`.
- **DM-4b** — Create `crates/lance-graph-callcenter/src/version_watcher.rs`. Holds the `watch::Sender<CognitiveEventRow>`; `bump(row)` on each `project()` commit.
- **DM-4c** — Uncomment `pub mod version_watcher` in `lib.rs:71-72`; export `LanceVersionWatcher`.
- **DM-5a** — `subscribe(filter)` returns the `watch::Receiver` wrapped with a `CommitFilter` predicate combinator.
- **DM-6a** — Create `crates/lance-graph-callcenter/src/drain.rs`. Scaffold only — `DrainTask` struct + `drain()` method that currently returns `Poll::Pending`. Wiring to `OrchestrationBridge::route()` is the follow-up PR.
- **DM-6b** — Uncomment `pub mod drain` in `lib.rs:78-79`; export `DrainTask`.
- **DM-7** — Flip test `subscribe_returns_disconnected_receiver` to `subscribe_receives_on_project` — assert `rx.borrow().version > 0` after a `project()` call.

## Non-goals (explicit)

- `dialect` Phase-B source wiring — separate TECH_DEBT row (@callcenter-specialist).
- `scent` Phase-C CAM-PQ cascade — blocked on substrate migration (PR B, pending).
- `PhoenixServer` DM-5 — Queued separately.
- `DrainTask` runtime drain of `steering_intent` — this PR ships only the scaffold.

## Acceptance criteria

- `cargo test -p lance-graph-callcenter --lib` — 11 existing tests pass + new `subscribe_receives_on_project` test passes. Zero regressions.
- `bbb_scalar_only_compile_check` still compiles.
- `cargo check --workspace` compiles.
- Verdict flip in `.claude/plans/unified-integration-v1.md §6`: Supabase row `GHOST` → `PARTIAL` (one-line Edit on the table row).
- `.claude/board/INTEGRATION_PLANS.md` — prepend entry pointing to this plan file.
- `.claude/board/STATUS_BOARD.md` — DM-4 / DM-6 rows status updated.
- `.claude/board/EPIPHANIES.md` — prepend short FINDING entry noting subscribe wire-up.

## Architecture notes

Per CLAUDE.md BBB invariant, `Subscription` must carry Arrow-scalar content only — `CognitiveEventRow` is the canonical outbound DTO and is already scalar-only (compile-time enforced by `bbb_scalar_only_compile_check`). `tokio::sync::watch` is the right primitive for the supabase-realtime-shaped fan-out: single-producer (the membrane), many-consumer (subscribers), always-latest semantics (skip stale revisions).

Implementation sketch:

```rust
// version_watcher.rs
pub struct LanceVersionWatcher {
    tx: tokio::sync::watch::Sender<CognitiveEventRow>,
}
impl LanceVersionWatcher {
    pub fn new(initial: CognitiveEventRow) -> Self { ... }
    pub fn bump(&self, row: CognitiveEventRow) { let _ = self.tx.send(row); }
    pub fn subscribe(&self) -> tokio::sync::watch::Receiver<CognitiveEventRow> { self.tx.subscribe() }
}
```

The `CommitFilter` wrapper is NOT a new trait — it's a method `LanceMembrane::subscribe_filtered(filter: CommitFilter) -> impl Stream<Item = CognitiveEventRow>` that calls `self.watcher.subscribe()` and uses `tokio_stream::wrappers::WatchStream::new(rx).filter(move |row| filter.matches(row))`.

## File-level edits (full list)

1. `crates/lance-graph-callcenter/Cargo.toml` — add `tokio = { workspace = true, features = ["sync"] }` if not present; add `tokio-stream = { workspace = true }` (minimal features).
2. `crates/lance-graph-callcenter/src/version_watcher.rs` — NEW.
3. `crates/lance-graph-callcenter/src/drain.rs` — NEW (scaffold).
4. `crates/lance-graph-callcenter/src/lib.rs:71-72` — uncomment `pub mod version_watcher`.
5. `crates/lance-graph-callcenter/src/lib.rs:78-79` — uncomment `pub mod drain`.
6. `crates/lance-graph-callcenter/src/lance_membrane.rs:56` — field `watcher: LanceVersionWatcher`.
7. `crates/lance-graph-callcenter/src/lance_membrane.rs:117` — `type Subscription = watch::Receiver<CognitiveEventRow>`.
8. `crates/lance-graph-callcenter/src/lance_membrane.rs:186-189` — `subscribe()` body = `self.watcher.subscribe()`.
9. `crates/lance-graph-callcenter/src/lance_membrane.rs` — `project()` path (wherever it completes a commit) — call `self.watcher.bump(row)` after the Lance write.
10. `crates/lance-graph-callcenter/src/lance_membrane.rs` tests — flip `subscribe_returns_disconnected_receiver` to `subscribe_receives_on_project`.
11. `crates/lance-graph-contract/src/external_membrane.rs` — IF the `Subscription` associated type is declared here with bounds, widen to `Clone + Send` as needed. Do NOT add Vsa/RoleKey/NarsTruth — BBB deny-list stays intact.
12. Board files (INTEGRATION_PLANS, STATUS_BOARD, EPIPHANIES, unified-integration-v1 §6) — per Acceptance.

## Test

- New test `subscribe_receives_on_project` — construct `LanceMembrane`, call `subscribe()` → `rx`, call `project(some_event)` → assert `rx.borrow().<relevant field>` matches `some_event`.
