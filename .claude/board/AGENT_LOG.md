# Agent Log — Append-Only Session Record

> **APPEND-ONLY.** Each agent session appends one entry with timestamp,
> D-ids, commit hash, test count, and outcome. Never edit past entries.

---

## 2026-04-24T16:30 — Supabase subscriber v2 (sonnet, claude/supabase-subscriber-wire-up)

**D-ids:** DM-4a/b/c, DM-6a/b
**Commit:** `ec3b5c7`
**Tests:** 17 pass with realtime feature (13 without); 5 new tests total (4 in version_watcher.rs, 1 subscribe_receives_on_project in lance_membrane.rs)
**Outcome:** Wired LanceMembrane::subscribe() from Phase-A disconnected mpsc stub to live tokio::sync::watch::Receiver<CognitiveEventRow> under [realtime] feature. project() now calls watcher.bump(row.clone()) on every projected cycle. DrainTask scaffold (Poll::Pending) ships unconditionally. Tokio was already a dep - no Cargo.toml changes needed. PR 255: https://github.com/AdaWorldAPI/lance-graph/pull/255
