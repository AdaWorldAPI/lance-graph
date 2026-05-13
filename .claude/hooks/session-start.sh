#!/usr/bin/env bash
# SessionStart hook — inject workspace bootload context at turn 0.
# Emits JSON on stdout; Claude consumes hookSpecificOutput.additionalContext
# as a system reminder in the first model turn.
#
# Wired from: .claude/settings.json
# Documented by: .claude/skills/cca2a/SKILL.md
set -euo pipefail

cat <<'JSON'
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "WORKSPACE BOOTLOAD (lance-graph, via cca2a pattern):\n\n## Mandatory reads this session (in order)\n\n1. .claude/board/LATEST_STATE.md    — current contract inventory, recently shipped PRs, queued work, explicit deferrals. What exists.\n2. .claude/board/PR_ARC_INVENTORY.md — APPEND-ONLY decision history per PR (Added / Locked / Deferred / Docs + mutable Confidence line). Why it exists.\n3. .claude/agents/BOOT.md                — 19 specialist + 5 meta-agent ensemble, Knowledge Activation trigger table, Handover Protocol. How to coordinate.\n\nDo NOT propose any new type, module, or convention without grepping LATEST_STATE first.\n\n## Governance (never violate)\n\n- APPEND-ONLY: LATEST_STATE + PR_ARC_INVENTORY Edit prompts for approval; Write to append stays unprompted. Only the Confidence line per PR entry is updatable; corrections append as dated lines; reversals get their own PR entry.\n- Model policy: main thread Opus + deep thinking; subagent grindwork (single-source mechanical) → Sonnet; accumulation (multi-source synthesis) → Opus; NEVER Haiku.\n- GitHub reads: zipball to /tmp/sources/ + local grep for 3+ reads per external repo. MCP only for writes (PR, comments, reviews) and single-path reads.\n- Contract zero-dep invariant: lance-graph-contract has no external crate deps.\n- Read before Write: always Read a file before overwriting.\n- No JSON serialization in runtime types (serde is debug-only).\n\n## A2A two layers\n\n- Layer 1 (runtime): contract::a2a_blackboard. Cognitive-cycle bus.\n- Layer 2 (session): knowledge docs + .claude/handovers/*.md. Parallel subagent spawns in one main-thread turn is cheapest.\n\n## Pattern explained\n\n.claude/skills/cca2a/SKILL.md — read once to grok the pattern, skip re-deriving.\n\n## Full spec\n\nCLAUDE.md at workspace root."
  }
}
JSON
