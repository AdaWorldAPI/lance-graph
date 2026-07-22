#!/usr/bin/env bash
# SessionStart hook — inject workspace bootload context + preflight drift signal.
# Emits JSON on stdout; Claude consumes hookSpecificOutput.additionalContext
# as a system reminder in the first model turn.
#
# Wired from: .claude/settings.json
# Documented by: .claude/skills/cca2a/SKILL.md
# Drift tool: .claude/tools/preflight_drift.rs (compile once with rustc to enable)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIFT_TOOL="$REPO_ROOT/.claude/tools/preflight_drift"
DRIFT_SRC="$REPO_ROOT/.claude/tools/preflight_drift.rs"

# Optional drift signal. Compiled binary at $DRIFT_TOOL → run it. Missing binary
# but source present → emit a build hint. Neither → silent. Never blocks.
DRIFT_BLOCK=""
if [[ -x "$DRIFT_TOOL" ]]; then
  if ! DRIFT_OUT="$(cd "$REPO_ROOT" && "$DRIFT_TOOL" 2>&1)"; then
    DRIFT_BLOCK=$'\n\n## ⚠ Board drift detected (preflight_drift exit 1)\n\n```\n'"$DRIFT_OUT"$'\n```\n\nThe board claims do not match Cargo.toml workspace state. Before any new sprint spawn, update `CLAUDE.md` / `.claude/board/LATEST_STATE.md` to match real counts, OR document why the drift is intentional. The Mandatory Board-Hygiene Rule in CLAUDE.md applies.'
  fi
elif [[ -f "$DRIFT_SRC" ]]; then
  DRIFT_BLOCK=$'\n\n## ℹ preflight_drift not compiled\n\nBuild once to enable board-vs-cargo drift signal at session start:\n\n```\nrustc '"$DRIFT_SRC"$' -o '"$DRIFT_TOOL"$'\n```'
fi

BASE_CONTEXT='WORKSPACE BOOTLOAD (lance-graph, via cca2a pattern):

## Mandatory reads this session (in order)

1. .claude/board/LATEST_STATE.md    — current contract inventory, recently shipped PRs, queued work, explicit deferrals. What exists.
2. .claude/board/PR_ARC_INVENTORY.md — APPEND-ONLY decision history per PR (Added / Locked / Deferred / Docs + mutable Confidence line). Why it exists.
3. .claude/agents/BOOT.md                — 19 specialist + 5 meta-agent ensemble, Knowledge Activation trigger table, Handover Protocol. How to coordinate.

Do NOT propose any new type, module, or convention without grepping LATEST_STATE first.

## Governance (never violate)

- APPEND-ONLY: LATEST_STATE + PR_ARC_INVENTORY Edit prompts for approval; Write to append stays unprompted. Only the Confidence line per PR entry is updatable; corrections append as dated lines; reversals get their own PR entry.
- Model policy: main thread Opus + deep thinking; subagent grindwork (single-source mechanical) → Sonnet; accumulation (multi-source synthesis) → Opus; NEVER Haiku EXCEPT the one contract-gated guarded-executor role (.claude/knowledge/tiered-agent-execution-protocol.md).
- GitHub reads: zipball to /tmp/sources/ + local grep for 3+ reads per external repo. MCP only for writes (PR, comments, reviews) and single-path reads.
- Contract zero-dep invariant: lance-graph-contract has no external crate deps.
- Read before Write: always Read a file before overwriting.
- No JSON serialization in runtime types (serde is debug-only).

## A2A two layers

- Layer 1 (runtime): contract::a2a_blackboard. Cognitive-cycle bus.
- Layer 2 (session): knowledge docs + .claude/handovers/*.md. Parallel subagent spawns in one main-thread turn is cheapest.

## Pattern explained

.claude/skills/cca2a/SKILL.md — read once to grok the pattern, skip re-deriving.

## Engineering spec (when implementing or auditing the agent kit)

.claude/ATT/ — Attractor-style NLSpecs (DoD checklists + Cross-Language/Provider parity matrices):
  - autoattended-orchestrator-spec.md  (wave loop, 4-savant slots PP-13/14/15/16, worker iron rules, validation WAVE-001..WAVE-017)
  - anti-skim-agent-spec.md            (Reading-Depth Ladder, Lie-Detector LD-1..5, stuck-protocol blockers, AP1..AP9 anti-patterns)
  - agent-coordination-mcp-spec.md     (Layer-0/1/2 coordination, handover schema, append-only governance §7.2)
.claude/ATT/ACTIVATION.md — receipt: which DoD items this repo currently satisfies + which are explicit gaps.

## Full spec

CLAUDE.md at workspace root.'

# Compose final additionalContext (base + optional drift block) and JSON-encode
# via python3 (avoids hand-rolled escaping). python3 is a documented prereq for
# this workspace's harvest tooling (cf. WoA/.claude/v0.2/tools/*.py).
FULL_CONTEXT="${BASE_CONTEXT}${DRIFT_BLOCK}"

python3 - <<PY
import json
print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "SessionStart",
        "additionalContext": $(python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' <<< "$FULL_CONTEXT")
    }
}))
PY
