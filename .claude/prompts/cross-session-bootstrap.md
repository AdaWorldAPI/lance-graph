# Cross-Session Bootstrap — Paste into the OTHER Session

> Copy this entire file into the new Claude Code session that will
> coordinate with an existing one. The receiving session subscribes
> to the shared MCP-emulation PR, learns the kanban-ack protocol,
> and starts claiming work without duplicating what the other
> session is already doing.

---

## What this is

Two (or more) Claude Code sessions are working on the same repo at
the same time. Native A2A doesn't exist yet, so we use the **Branch
Pub/Sub** pattern from `.claude/knowledge/A2Aworkarounds.md`
Workaround #2: a dedicated draft PR carries the bus file
(`.claude/board/CROSS_SESSION_BROADCAST.md`); every push fires a
GitHub webhook delivered to all subscribed sessions as
`<github-webhook-activity>`.

## Bootstrap steps (run these first, in order)

### 1. Identify your session_id

Your session ID is in the URL of every Claude Code commit you make:
`https://claude.ai/code/session_<HASH>`. Memorize it (or jot it on a
sticky); you'll tag every cross-session entry with it.

If you don't know it yet: open your most recent commit's footer, OR
`git log -1 --pretty=%B | grep -oE 'session_[A-Za-z0-9]+'`, OR ask
the user to paste their current session URL.

### 2. Fetch and read the bus

```bash
cd <your-repo>
git fetch origin claude/blackboard
git show origin/claude/blackboard:.claude/board/CROSS_SESSION_BROADCAST.md | tail -200
```

Read what the other session(s) shipped, claimed, and asked you to
NOT redo. Open coordination questions are usually at the bottom of
the most recent entry.

### 3. Subscribe to the bus PR

```
mcp__github__subscribe_pr_activity(
  owner="AdaWorldAPI", repo="lance-graph", pullNumber=261
)
```

PR #261 is the dedicated coordination bus. **Do not merge it.** It
stays draft forever; merging would close the channel.

From this point on, every push from any subscribed session arrives
in your conversation as `<github-webhook-activity>`.

### 4. Switch to your work branch (NOT the blackboard branch)

The blackboard branch is the bus, not your workspace. Do your code
on a feature branch:

```bash
git checkout -b claude/<your-topic>-<short-id>
```

(Or whatever branch the user has assigned you.)

---

## The Kanban-Ack Protocol

We coordinate via **3-state kanban ack** posted to
`.claude/board/CROSS_SESSION_BROADCAST.md` on the `claude/blackboard`
branch. Three transitions per work item:

| State | When you post | What it tells the other session |
|---|---|---|
| **CLAIM** | Before starting work on an item | "I own this — don't take it" |
| **WIP** *(optional)* | Periodic checkpoints on long work | "Still alive, here's progress" |
| **DONE** | Immediately after committing the work | "Free for the next caller; here's the commit" |

### Posting an entry

Always append (never edit). Always tag with your session_id and a
work-item ID. Always commit + push immediately so the webhook fires.

```bash
git checkout claude/blackboard
git pull origin claude/blackboard

cat >> .claude/board/CROSS_SESSION_BROADCAST.md <<'EOF'

## YYYY-MM-DDTHH:MM — CLAIM <ITEM-ID> — session_<HASH>

**Item:** <ITEM-ID> (e.g., LF-3, TD-INT-5)
**Owner:** session_<HASH> (yours)
**Branch:** claude/<your-feature-branch>
**Estimate:** <small / medium / large>
**Note:** <one-line context>

EOF

git add .claude/board/CROSS_SESSION_BROADCAST.md
git commit -m "kanban: CLAIM <ITEM-ID> on session_<HASH>"
git push origin claude/blackboard
git checkout -  # back to your work branch
```

DONE entry mirrors CLAIM but adds the commit hash:

```
## YYYY-MM-DDTHH:MM — DONE <ITEM-ID> — session_<HASH>

**Item:** <ITEM-ID>
**Owner:** session_<HASH>
**Branch:** claude/<your-feature-branch>
**Commit:** `<commit-hash>`
**Tests:** N pass (M new)
**Outcome:** <one-line>
```

### Conflict avoidance

- **Pull before posting** — `git pull origin claude/blackboard`
  always, just before the cat >>. Concurrent appends merge cleanly
  only if both sides are based on the latest tip.
- **Commit immediately** — don't queue multiple appends; push each
  CLAIM/DONE on its own commit so the webhook fires the other
  session before you take the next action.
- **Read CLAIMs before claiming** — if the item already has an
  open CLAIM (no matching DONE), don't claim it. Pick another.

### What if you both CLAIM the same item simultaneously?

Latest push wins. The other session sees its own CLAIM accepted
(it merged first) and the loser session sees the webhook for the
winner's claim arrive AFTER its own push, with their own claim
either rebased away or visible-but-late. The losing session
reverts its CLAIM with a DONE-CANCEL entry:

```
## YYYY-MM-DDTHH:MM — CANCEL-CLAIM <ITEM-ID> — session_<HASH>

**Reason:** session_<OTHER-HASH> claimed first at <timestamp>.
Reverting; picking next item.
```

Then push that and pick a different item.

---

## Session-id in agent cards (work scoping)

When you spawn subagents during your session, include your
session_id in every spawn prompt:

```
You are working on behalf of session_<HASH> on branch
claude/<your-branch>. Before starting, read
.claude/board/CROSS_SESSION_BROADCAST.md (latest entries) to see
what other sessions claimed. Your task is <X>; tag your final
report with session_<HASH>/<sub-task-id>.
```

This way every commit, every broadcast entry, every blackboard
ack carries provenance back to the spawning session. Multi-agent
work isn't anonymous; it's session-scoped.

If you maintain `.claude/agents/<role>.md` cards for specialist
agents, the cards themselves stay session-agnostic (they describe
roles, not owners). The session_id flows through the **prompt**
the parent passes when spawning, not through the card's static
content.

---

## Open work for whoever takes it next

(See the latest CROSS_SESSION_BROADCAST.md entry on
`claude/blackboard` for the live list. The list below is a
snapshot from 2026-04-25.)

### Inside BBB (8 of 14 dormant intelligence features remain)

- **TD-INT-5** — RoleKey bind/unbind in content cascade (replace
  Hamming with role-indexed VSA cosine). P1.
- **TD-INT-6** — ContextChain disambiguation in route handler.
  P2 (activates when real Cypher parser lands).
- **TD-INT-7** — Pearl 2³ causal mask query path (causal_type
  WHERE filter on graph queries). P1.
- **TD-INT-8** — Schema validation on SPO commit (run
  `Schema::validate` before AriGraph commit; emit FailureTicket
  on missing-required). P1.
- **TD-INT-9** — RBAC `Policy::evaluate` at `LanceMembrane::project`
  emit time. P1.
- **TD-INT-11** — Populate `neural-debug` runtime registry from
  live dispatch (currently `WireHealth.neural_debug = None`). P2.
- **TD-INT-12** — `DrainTask::poll()` actually drains
  `steering_intent` rows into `OrchestrationBridge::route`. P2.
- **TD-INT-13** — `CommitFilter` applied server-side in
  `LanceMembrane::project` (currently subscriber-side only). P2.

### Outside BBB (2 of 8 SMB foundry-parity items remain)

- **LF-2** — RoleKey slice band for SMB roles. Range [9910..10000)
  is fully allocated by tense + NARS keys; needs Vsa10k → Vsa16k
  upgrade decision before allocation can land.
- **LF-3** — Uncomment callcenter `[auth]` DM-7 (RLS rewriter).
  Blocked on UNKNOWN-3 (pgwire choice) and UNKNOWN-4 (actor_id
  type).

### Foundry-parity Tier 2 (28 chunks across 9 stages — pick by stage)

Stage 1 Data Integration: LF-10..14
Stage 2 Ontology: LF-20..23
Stage 3 Storage v2: LF-30..33
Stage 4 Search: LF-40..42
Stage 5 Models: LF-50..53
Stage 6 Decisions: LF-60..62
Stage 7 Scenarios: LF-70..72
Stage 8 Marketplace: LF-80..81
Cross-cutting: LF-90..92

See `docs/foundry-parity-checklist.md` (commit `164a476`) for the
full per-item descriptions.

---

## When you're done

- **Always** post DONE entries before declaring work complete.
- **Don't merge PR #261.** It's the bus, not a deliverable.
- **Do open feature PRs** for your actual code work — those go
  through normal review.
- **Stay subscribed.** Even after your work, leave the subscription
  active so future sessions can ping you with questions. Unsubscribe
  only when leaving the project entirely.

---

## Cross-ref

- `.claude/knowledge/A2Aworkarounds.md` — full pattern catalog
  (file blackboard / branch pub-sub / role teleportation / handovers)
- `.claude/board/CROSS_SESSION_BROADCAST.md` — the bus file (this is
  what you append to)
- PR #261 — the dedicated MCP-emulation channel
  (https://github.com/AdaWorldAPI/lance-graph/pull/261)
- `CLAUDE.md` § Agent-to-Agent (A2A) Orchestration — Two Layers
  — the runtime + session distinction
