# BOOT — Session Entry Point for lance-graph

> **If you are a new session on this workspace, read this file first.**
> It is the one page that tells you everything you need to bootstrap
> before proposing any work. Target cold-start cost: 3–5 turns, not 30.

---

## Read Order (MANDATORY)

Read these in order before proposing anything:

1. **`.claude/knowledge/LATEST_STATE.md`** — current contract
   inventory, recently shipped PRs, active branches, queued work,
   explicit deferrals. Tells you **what exists**.
2. **`.claude/knowledge/PR_ARC_INVENTORY.md`** — per-PR Added /
   Locked / Deferred / Docs / Confidence, reverse-chronological.
   APPEND-ONLY (only Confidence is mutable; corrections append as
   new dated lines; reversals get their own PR entry). Tells you
   **why it exists**.
3. **`.claude/agents/BOOT.md`** — the 19-specialist + 5-meta-agent
   ensemble, the Knowledge Activation trigger table (domain →
   agent → required knowledge docs), the Handover Protocol spec.
   Tells you **how to coordinate**.

These three files give you ~90 % of the context you need to avoid
re-proposing what's shipped or violating a locked convention.

---

## The Governance Rules (never violate)

1. **Append-only architectural history.** `PR_ARC_INVENTORY.md` and
   `LATEST_STATE.md` are governed by the append-only rule — Edit
   prompts for explicit approval, Write for appends is allowed.
   Old PR entries are immutable historical record; only the
   Confidence line per entry is updatable.
2. **Model policy.** Main thread on Opus with deep thinking.
   Subagents: Sonnet for grindwork (single-source mechanical),
   Opus for accumulation (multi-source synthesis). **Never Haiku**,
   regardless of task. Full spec in CLAUDE.md §Model Policy.
3. **GitHub access.** Zipball to `/tmp/sources/` + local grep for
   3+ reads per external repo. MCP github only for writes (PR,
   comments, reviews) and single-path reads. Full spec in
   CLAUDE.md §GitHub Access Policy.
4. **Contract zero-dep invariant.** `lance-graph-contract` has no
   external crate deps. Do not add any.
5. **Read before Write.** Always Read a file before overwriting.
   Write-over-self without Read is the documented failure mode;
   see CLAUDE.md §In-Session Orchestration Discipline.
6. **No JSON serialization in types.** Serde stays debug-only; wire
   formats are explicit.
7. **Pumpkin framing** for externally-visible examples (clinical /
   game-AI disguise for the AGI primitives).

---

## A2A Orchestration — Two Layers

Orchestration on this workspace runs at two distinct layers:

- **Layer 1 — Runtime A2A:** `lance_graph_contract::a2a_blackboard`
  `Blackboard` + `BlackboardEntry` (experts write, later rounds
  read), `OrchestrationBridge` + `StepDomain` + `UnifiedStep`,
  `orchestration_mode` for composition modes. For
  cognitive-cycle composition inside the running system.
- **Layer 2 — Session A2A:** The mandatory-read files above are
  the shared blackboard across subagents. `.claude/knowledge/*.md`
  are extended blackboard entries. `.claude/handovers/*.md`
  (created per agent chain) carry per-handover state. Parallel
  `Agent` spawns in one main-thread turn are the cheapest
  coordination pattern.

Full description: CLAUDE.md §Agent-to-Agent (A2A) Orchestration.
Detailed agent ensemble + trigger table + handover protocol:
`.claude/agents/BOOT.md`.

---

## The 30-Turn Rediscovery Tax (why this file exists)

Without this BOOT.md, a new session would typically:

1. Not know the current contract — propose duplicate types.
2. Not know recently locked conventions — violate them, get
   reverted, propose again.
3. Not know queued work — suggest things already in the plan.
4. Not know the agent ensemble — re-invent subagent coordination.
5. Re-read the same large knowledge docs multiple times because
   nothing tells it which docs matter for which work.

Cumulative cost: 20–30 turns of main-thread context before the
session is productive. That's ~$20–40 in Opus charges per fresh
session, most of which is wasted rediscovery.

Reading the three mandatory files above reduces this to ~3–5 turns.
The entire BOOT spec (this file) is ~2 KB. The three mandatory
reads total ~15 KB. Compared to 20–30 wasted turns, the bootload
is a 5–10× cost reduction on every session start.

---

## Existing content — don't duplicate, link

This workspace already has substantial curated content. New work
should reference these, not recreate them:

- **`.claude/prompts/`** — 41 scoped prompts (sessions,
  certifications, probes, handovers, research surfaces). Each is a
  self-contained task brief. See `.claude/prompts/SCOPED_PROMPTS.md`
  as the natural index.
- **`.claude/plans/`** — versioned integration plans. Index at
  `.claude/knowledge/INTEGRATION_PLANS.md` (APPEND-ONLY — new
  versions prepend; prior plans stay with Status annotation).
  Active: `.claude/plans/elegant-herding-rocket-v1.md`.
- **`.claude/*.md`** (top-level, 61 docs) — calibration reports,
  handover logs, epiphanies compressed, integration-plan snapshots,
  cross-repo audits, invariant matrices. Browse before writing new
  reference docs. Examples: `EPIPHANIES_COMPRESSED.md` in prompts/,
  `SESSION_CAPSTONE.md`, `INTEGRATIONSPLAN_2026_04_01.md`,
  `INTEGRATION_SESSIONS.md`, `INVENTORY_MAP.md`.
- **`.claude/knowledge/*.md`** (newer, structured) — the knowledge
  base proper with `READ BY:` headers + Knowledge Activation
  triggers. See `.claude/agents/BOOT.md` § Knowledge Activation for
  the trigger table.
- **`.claude/agents/*.md`** (19 specialists + 5 meta-agents) —
  ensemble cards. See `.claude/agents/README.md` for the function
  inventory or `BOOT.md` (sibling) for the orchestration spec.
- **`.claude/hooks/*.sh`** — SessionStart and PostCompact hooks
  wired via `.claude/settings.json`.
- **`.claude/skills/cca2a/`** — the A2A pattern explanation skill.

Before creating a new `.claude/*.md` file, grep the existing 61
docs and 41 prompts for the topic. Most architectural concerns have
prior art.

## Fallback — CLAUDE.md is the source of truth for everything else

If this file doesn't answer your question, CLAUDE.md does. Tables
of contents:

- CLAUDE.md §Session Start — mandatory reads (same three as above)
- CLAUDE.md §A2A Orchestration — both layers in detail
- CLAUDE.md §Model Policy — grindwork vs accumulation, never Haiku
- CLAUDE.md §GitHub Access Policy — zipball for reads
- CLAUDE.md §Workspace Structure — 11-crate layout
- CLAUDE.md §Knowledge Base — all `.claude/knowledge/` files
- CLAUDE.md §Knowledge Activation (MANDATORY) — agent protocol
- CLAUDE.md §In-Session Orchestration Discipline — Read before Write

---

## One sentence

**Read `LATEST_STATE.md`, `PR_ARC_INVENTORY.md`, and
`.claude/agents/BOOT.md` before proposing anything, then the
trigger-table domain doc, then start work.**
