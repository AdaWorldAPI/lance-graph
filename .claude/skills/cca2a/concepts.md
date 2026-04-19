# CCA2A — Concepts

## The Problem

A new Claude Code session on a non-trivial workspace burns 20-30
turns rediscovering:
- What types / modules already exist (re-proposes duplicates).
- What conventions have been locked (violates them, gets reverted).
- What work is already queued (proposes things in the plan).
- How to coordinate subagents (re-invents the wheel).

At ~$15-$75 per 1M tokens of Opus, that's **$20-$40 of waste per
cold start.** CCA2A is the scaffolding that reduces this to 3-5
turns.

## Two A2A Layers

### Layer 1 — Runtime A2A (code-level inside the system)

If the workspace is building a cognitive system, its runtime
orchestration needs a blackboard where experts write entries and
later rounds read them. Typical structure:

```rust
struct Blackboard {
    entries: Vec<BlackboardEntry>,
    round: u32,
}

struct BlackboardEntry {
    expert_id: ExpertId,
    capability: ExpertCapability,
    result: u16,
    confidence: f32,
    support: [u16; 4],
    dissonance: f32,
    cost_us: u32,
}
```

This is the A2A bus for multi-expert inference inside the system.
Name it consistently; reference from BOOT.md.

### Layer 2 — Session A2A (Claude Code subagent coordination)

For subagent coordination *during a Claude session*:
- The two mandatory knowledge files (LATEST_STATE +
  PR_ARC_INVENTORY) are the shared blackboard.
- Domain-specific knowledge docs are extended blackboard entries
  (load-on-trigger via Knowledge Activation table).
- Handover files (`.claude/handovers/*.md`) carry per-chain state
  transfer.
- Parallel subagent spawns in one main-thread turn is the cheapest
  coordination pattern.

Layers 1 and 2 don't conflict — they operate at different time
scales. Don't entangle them.

## Governance Rules

### Append-only history (eight bookkeeping files, unified rule)

The workspace carries eight bookkeeping files. Rows / entries
inside them are immutable historical record; specific fields per
file are mutable state. Never delete a row. Supersedure is a new
row that cites the old; the old row's Status updates to
"Superseded by <new>".

| File | Role | Immutable (rows) | Mutable (state fields) |
|---|---|---|---|
| `PR_ARC_INVENTORY.md` | Per-PR decision arc | PR rows (Added / Locked / Deferred / Docs) | Confidence + Corrections APPEND |
| `LATEST_STATE.md` | Current-state snapshot | Recently-shipped PR table | Snapshot sections updated by replacement |
| `STATUS_BOARD.md` | Deliverable-level dashboard | Rows (D-id / title / plan / scope) | Status, PR/Evidence per row |
| `INTEGRATION_PLANS.md` | Versioned plan index | Entries (scope / path / deliverables) | Status + Confidence per entry |
| `EPIPHANIES.md` | Dated insight log | Entry bodies | Status line (FINDING/CONJECTURE/SUPERSEDED) |
| `ISSUES.md` | Open + Resolved bugs | Entry bodies | Status + Resolution line (append on close) |
| `IDEAS.md` | Open + Implemented + Rejected speculation | Entry bodies | Status + Rationale line (append) |
| `TECH_DEBT.md` | Open + Paid debt | Entry bodies | Status + Payoff line (append) |

**Method hierarchy (preferred → discouraged):**

1. **APPEND** (the method of choice) — add a NEW dated row. Either
   Edit-to-prepend inside a bounded section, or `Bash cat >>
   file << EOF`. No prompt. Old rows stay untouched. This is the
   double-bookkeeping pattern: new state = new row, not mutation.
2. **Edit field with prior Read** — flip a mutable field
   (Status / Confidence / Resolution / Payoff) on an existing row
   after reading the file. No prompt. Use for clarifications and
   status transitions on already-captured entries.
3. **Write (full overwrite)** — prompts via
   `.claude/settings.json::permissions.ask`. Discouraged; only for
   wholesale replacement after explicit review.

**Rule:** when in doubt, APPEND. Double-bookkeeping keeps the
arc intact and makes the audit trail legible. Edit-a-field is the
lighter touch for clear status transitions. Write is the escape
hatch that costs a confirmation because it can destroy history.

**Kanban discipline** — three files track work items that must
not get buried: `ISSUES.md`, `IDEAS.md`, `TECH_DEBT.md`. Every
entry carries:

- **Priority** — P0 blocker / P1 high / P2 medium / P3 low.
- **Scope** — `@<agent-name>`, `D<N>` (plan D-id),
  `domain:<grammar|codec|arigraph|infra|...>`.

Agents filter by their own `@`-mention or their domain. Status
moves through Open → In Progress → Resolved / Implemented / Paid.
Nothing falls through the cracks because every ticket has an owner
named by `@`.

Core invariant: **the arc is the record; rewriting it destroys the
"why was this decided that way" context that prevents future
rediscovery.**

### Model policy

- Main thread: Opus, deep thinking, full `effortLevel: high`.
- Subagent grindwork (single-source mechanical): Sonnet.
- Subagent accumulation (multi-source synthesis): Opus.
- NEVER Haiku — quality floor for this pattern is Sonnet.

Concrete test before spawning a subagent:
> "Does this agent have to read N sources and produce something that
> only makes sense when those sources are held in mind together?"
>
> Yes → Opus. No → Sonnet.

### GitHub access

- Reads (3+ files from same repo): zipball to `/tmp/sources/` +
  local grep. Each `mcp__github__get_file_contents` call drops the
  full file into session context and re-bills on every subsequent
  turn. Zipball-then-grep is ~20x cheaper.
- Reads (single targeted file, known path): MCP is fine.
- Writes (PR creation, comments, reviews, review threads): MCP.

### Destructive ops denied globally

`.claude/settings.json::permissions.deny` blocks: `git push --force`,
`git push -f`, `git branch -D`, `git reset --hard`, `rm -rf`,
MCP `merge_pull_request`, `delete_file`, `fork_repository`,
`create_repository`, `enable_pr_auto_merge`.

## Why This Is Different From Official Claude Code Docs

See [divergence.md](divergence.md) for what's official, what's
invented, and what's recommended for adoption.
