#!/usr/bin/env bash
# PostCompact hook — re-inject critical state after context compaction
# drops the workspace bootload. Without this, the post-compact model
# turn rediscovers what existed before compaction at full cost.
#
# Wired from: .claude/settings.json
# Documented by: .claude/skills/cca2a/SKILL.md
set -euo pipefail

cat <<'JSON'
{
  "hookSpecificOutput": {
    "hookEventName": "PostCompact",
    "additionalContext": "POST-COMPACT RE-INJECT (lance-graph, via cca2a pattern):\n\nCompaction just dropped prior context. Re-read mandatory state BEFORE answering the next turn:\n\n1. .claude/board/LATEST_STATE.md    — current contract inventory.\n2. .claude/board/PR_ARC_INVENTORY.md — top 3 PR entries (reverse chronological).\n3. .claude/agents/BOOT.md                — if this session was doing agent orchestration.\n4. .claude/handovers/                    — if a chain was in progress, read the most recent handover file.\n\nDo NOT propose new types without re-grepping LATEST_STATE — compaction may have dropped your awareness that the type already exists.\n\nActive branch and current work-in-progress state should be discoverable via git status + the handover (if any). If no handover exists but work was clearly in progress, ask the user to clarify before resuming."
  }
}
JSON
