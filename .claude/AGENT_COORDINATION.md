# Agent Coordination — Architectural Governance

> **Committed governance doc.** Read-through, not append-to. Describes HOW
> agents coordinate in this workspace. The per-session activity log that
> was briefly in `.claude/board/AGENT_LOG.md` is now **gitignored** (local
> blackboard, ephemeral by design) because agent-by-agent appends kept
> producing merge conflicts between parallel branches.
>
> The durable record of what agents did lives in:
> - **Git log** — commit messages on each agent's branch
> - **PR descriptions** — what the agent shipped, tests, verdict
> - **EPIPHANIES.md** — findings that outlive a single PR
> - **`.claude/board/CROSS_SESSION_BROADCAST.md`** — intentional cross-session messages (committed, rare)

---

## Three Coordination Layers

All three use the **same structured-entry format**. Only the transport differs.

### Layer A — Teleportation (in-context role switch)

**Transport:** None (same context window).
**Latency:** Instant. **Context loss:** Zero.

The model loads an agent card (`.claude/agents/*.md`), adopts its role
and knowledge, does the work, and switches back. No process boundary,
no serialization. The 19 specialist + 5 meta-agent cards in this
workspace are **teleportation roles**, not delegation targets. The
agent IS the main thread wearing a different hat.

```
[main thread] → load @family-codec-smith card → do codec work
             → load @truth-architect card → review with full context
             → back to main thread (nothing lost)
```

### Layer B — Per-session file blackboard (in-session, between agents)

**Transport:** Local filesystem (`.claude/board/AGENT_LOG.md`, gitignored).
**Latency:** Instant. **Context loss:** File-level.

Agents spawned via `Agent()` in the SAME session share a working tree.
They can read each other's AGENT_LOG.md entries to see what's been done,
same as Layer-1 experts reading prior `BlackboardEntry` rounds.

**Critical:** AGENT_LOG.md is **gitignored**. It does NOT travel between
sessions or branches. It's pure local blackboard state. Each new session
starts with an empty log.

Why gitignored: every branch that ended up with appended entries from
parallel agents generated merge conflicts when its PR rebased against
other merged PRs. The durable record was already preserved in git log
+ PR description; the log was pure redundant metadata causing friction.

### Layer C — Cross-session broadcast (between sessions)

**Transport:** `.claude/board/CROSS_SESSION_BROADCAST.md` (committed,
append-only but curated) + `git push` + `subscribe_pr_activity` webhook.

**Latency:** Minutes. **Context loss:** Entry-level.

When two sessions genuinely need to talk across the boundary, they:

1. Open a coordination PR (or use an existing long-running one).
2. Subscribe via `mcp__github__subscribe_pr_activity`.
3. Append to `CROSS_SESSION_BROADCAST.md` (curated — small, intentional).
4. Push; the webhook notifies subscribers.

**Do NOT append here carelessly.** Unlike the gitignored AGENT_LOG.md,
every entry here is a durable commit that travels with the repo. Think
of it as a system-wide announcement channel. Most coordination should
stay in Layer A (teleport) or Layer B (local log). Use Layer C only for:

- Findings another session MUST see before starting work
- Architectural decisions that change what future sessions can assume
- Urgent corrections that can't wait for the next PR to merge

Everything else is either Layer B (don't leave the session) or belongs
in `EPIPHANIES.md` as a findings-of-record.

### Summary

| Layer | Scope | Transport | Committed? | Conflict risk |
|---|---|---|---|---|
| **A: Teleport** | In-context | None | N/A | None |
| **B: File** | In-session | `AGENT_LOG.md` | **No (gitignored)** | None |
| **C: Broadcast** | Cross-session | `CROSS_SESSION_BROADCAST.md` | Yes | Low (curated) |

All three share one invariant: **append-only, structured entries,
newest-first.** A `BlackboardEntry` by any other transport.

---

## Canonical Append Pattern

For AGENT_LOG.md (local, gitignored) and CROSS_SESSION_BROADCAST.md
(committed), use the `cat >>` heredoc — no Read required, no overwrite
risk, pre-allowed in `.claude/settings.json`:

```bash
cat >> .claude/board/AGENT_LOG.md <<'EOF'

## YYYY-MM-DDTHH:MM — description (model, branch)

**D-ids:** ...
**Commit:** `abc1234`
**Tests:** N pass (M new)
**Outcome:** One-line summary.
EOF
```

This is the ONLY sanctioned write pattern for both logs. Do not use
`Edit` or `Write` tools on them — they risk overwriting prior entries.
`cat >>` is append-only by construction.

---

## Durable Record (not ephemeral blackboard)

For findings, architectural changes, and PR outcomes that need to survive
beyond the session:

| What | Where | Format |
|---|---|---|
| Multi-source insight / "aha" / correction | `.claude/board/EPIPHANIES.md` | PREPEND dated entry |
| Tech-debt observation | `.claude/board/TECH_DEBT.md` | PREPEND |
| Unresolved blocker | `.claude/board/ISSUES.md` | PREPEND |
| Contract type added | `.claude/board/LATEST_STATE.md` | Edit inventory |
| Merged PR | PR description + git log | Automatic |
| New D-id / deliverable | `.claude/board/STATUS_BOARD.md` | Row entry |
| Plan version | `.claude/plans/<name>-v<N>.md` + `INTEGRATION_PLANS.md` PREPEND | New file |

Do NOT put findings in AGENT_LOG.md — it's local-only and will be lost
when the session ends. Findings go in EPIPHANIES.md.

---

## The Rule of Thumb

> "Would a future session need to read this?"
>
> **Yes** → EPIPHANIES.md / LATEST_STATE.md / CROSS_SESSION_BROADCAST.md (committed)
>
> **No, just this session** → AGENT_LOG.md (gitignored)
>
> **Automatic from git** → nothing to write — commit messages and PR
> descriptions already capture it

This is the split that was forced by three merge conflicts in one
session (2026-04-24): the architectural governance survives, the
ephemeral per-run log doesn't.
