# AGENT_LOG

Append-only log of agent sessions. Prepend new entries at the top.

---

## 2026-04-24T16:30 — Supabase subscriber v2 (sonnet, claude/supabase-subscriber-wire-up)

**D-ids:** DM-4a/b/c, DM-6a/b
**Commit:** `ec3b5c7`
**Tests:** 17 pass with realtime feature (13 without); 5 new tests total (4 in version_watcher.rs, 1 subscribe_receives_on_project in lance_membrane.rs)
**Outcome:** Wired LanceMembrane::subscribe() from Phase-A disconnected mpsc stub to live tokio::sync::watch::Receiver<CognitiveEventRow> under [realtime] feature. project() now calls watcher.bump(row.clone()) on every projected cycle. DrainTask scaffold (Poll::Pending) ships unconditionally. Tokio was already a dep — no Cargo.toml changes needed. PR 255: https://github.com/AdaWorldAPI/lance-graph/pull/255

## 2026-04-24T16:30 — Archetype scaffold v2 (sonnet, claude/archetype-crate-scaffold)

**D-ids:** DU-2.1..2.6
**Commit:** `816a7c0`
**Tests:** 12 pass
**Outcome:** Shipped `lance-graph-archetype` crate scaffold: Component + Processor traits (Arrow-backed), World meta-state with tick/fork/at_tick stubs, CommandBroker FIFO queue, ArchetypeError (thiserror). Added to root workspace members. No compile errors; 12 unit tests green.
