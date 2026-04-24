# Cross-Session Broadcast — Committed, Curated Append-Only

> **This IS committed.** Unlike AGENT_LOG.md (gitignored, ephemeral),
> every entry here travels with the repo. Use ONLY for messages another
> session MUST see before starting work — architectural decisions,
> urgent corrections, findings that can't wait for the next PR merge.
>
> Most coordination belongs in Layer A (teleport role switch) or Layer B
> (local AGENT_LOG.md). See `.claude/AGENT_COORDINATION.md` §Layer C for
> when to use this channel instead.
>
> **Append via `cat >>` heredoc** — no Read, no overwrite, pre-allowed.

---

## Entries (reverse chronological)

## 2026-04-24 — AGENT_LOG.md gitignored; architecture moved to .claude/AGENT_COORDINATION.md

After 3 merge conflicts in one session from parallel agents appending
to a committed AGENT_LOG.md, the split landed: architectural docs
(three coordination layers, canonical append pattern) moved to
`.claude/AGENT_COORDINATION.md` (committed). Per-session log is now
gitignored. Durable findings continue in EPIPHANIES.md. See
`.claude/AGENT_COORDINATION.md` for the new governance.
