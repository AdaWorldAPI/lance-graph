---
name: cca2a
description: >
  Explains the Claude Code Agent-to-Agent (CCA2A) pattern used in
  this workspace: BOOT.md session entry, agent ensemble with
  meta-agents, append-only knowledge state (LATEST_STATE +
  PR_ARC_INVENTORY), handover protocol, governance permissions.
  Read this once to understand the pattern; then stop re-explaining
  it across sessions. Canonical files live at their normal
  workspace paths, not in this skill.
---

# CCA2A — Claude Code Agent-to-Agent

Explanation-only skill. Documents the pattern so a new session can
grok it once instead of the user re-deriving it for 30 turns.

**Canonical files** (read these, not template copies):

**Entry points:**
- `.claude/BOOT.md` — session entry point (one page).
- `CLAUDE.md` — workspace spec, links to all of the below.

**Eight bookkeeping files** (rows are history; specific fields are
state; never delete a row; method hierarchy: APPEND preferred →
Edit-field after prior Read → Write prompts for confirmation):

_History / dashboards:_
- `.claude/knowledge/LATEST_STATE.md` — current contract inventory.
- `.claude/knowledge/PR_ARC_INVENTORY.md` — APPEND-ONLY per-PR arc.
- `.claude/knowledge/STATUS_BOARD.md` — deliverable-level dashboard.
- `.claude/knowledge/INTEGRATION_PLANS.md` — versioned plan index.

_Kanban / audit (priority + scope tags on every entry):_
- `.claude/knowledge/EPIPHANIES.md` — date-prefixed insight log.
  Status: FINDING / CONJECTURE / SUPERSEDED.
- `.claude/knowledge/ISSUES.md` — Open + Resolved bugs / regressions.
  Double-entry: issue captured on discovery; Resolution appends on
  close. Priority P0-P3, Scope @agent D<N> domain:<tag>.
- `.claude/knowledge/IDEAS.md` — Open + Implemented + Rejected
  speculation. Triple-entry: idea captured, shipped appended,
  plan-update logged. Priority + Scope.
- `.claude/knowledge/TECH_DEBT.md` — Open + Paid knowingly-deferred
  work. Priority + Scope + Introduced-by-PR + Payoff-estimate.

**Agent ensemble + orchestration:**
- `.claude/agents/BOOT.md` — orchestration spec + Knowledge
  Activation + Handover Protocol.
- `.claude/agents/README.md` — function inventory of all agents.

**Governance + hooks:**
- `.claude/settings.json` — team-shared governance (ask-on-Edit for
  the two strictest bookkeeping files, deny on destructive ops,
  SessionStart + PostCompact hooks).
- `.claude/hooks/session-start.sh` / `post-compact.sh` — inject
  bootload context at turn 0 / after compaction.

**Active plans:**
- `.claude/plans/<name>-v<N>.md` — indexed via
  `INTEGRATION_PLANS.md`. Old versions stay with Status annotation;
  new versions prepend.

## What to read when

- **New session on this workspace:** `.claude/BOOT.md` first, then
  the three mandatory reads it lists.
- **Understanding the A2A two-layer model:**
  [concepts.md](concepts.md).
- **Comparing to official Claude Code conventions:**
  [divergence.md](divergence.md).

## The idea in one paragraph

Every new session pays a 30-turn rediscovery tax rebuilding context
about what types exist, which conventions are locked, what work is
queued. CCA2A is the scaffold that cuts that to 3-5 turns: two
mandatory knowledge files that state current inventory and decision
history, one agent ensemble file that routes subagents, one hook
that injects critical state after compaction. Append-only
governance on the history files so drift can't erode the arc.
