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

### Append-only history (four bookkeeping files, unified rule)

The workspace carries four bookkeeping files. Rows / entries inside
them are immutable historical record; specific fields per file are
mutable state. Never delete a row. Supersedure is a new row that
cites the old; the old row's Status updates to "Superseded by
<new>".

| File | Immutable (rows) | Mutable (state fields) |
|---|---|---|
| `PR_ARC_INVENTORY.md` | PR rows (Added / Locked / Deferred / Docs) | Confidence line per entry; Corrections APPEND as dated lines |
| `LATEST_STATE.md` | Recently-shipped PR table | Current Inventory / Active Branches / Queued / Deferred snapshots (updated by replacement) |
| `STATUS_BOARD.md` | Deliverable rows (D-id / title / plan-version / scope) | Status column + PR / Evidence column per row |
| `INTEGRATION_PLANS.md` | Plan entries (scope / path / deliverables) | Status + Confidence lines per entry |

Governance enforcement: `.claude/settings.json::permissions.ask` on
Edit of `PR_ARC_INVENTORY.md` and `LATEST_STATE.md` surfaces any
edit as an approval prompt. Write for appends stays unprompted.
`STATUS_BOARD.md` and `INTEGRATION_PLANS.md` are less strict (Edit
is allowed without prompt since their Status fields move often) but
the same immutable-rows discipline applies by convention.

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
