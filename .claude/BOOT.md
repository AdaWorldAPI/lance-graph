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

Two companion dashboards (consult when deliverable status or plan
version matters — typically mid-session, not at cold start):

- **`.claude/knowledge/STATUS_BOARD.md`** — deliverable-level
  dashboard. All D-ids across every active plan with Status
  (Shipped / In PR / In progress / Queued / Backlog / Deferred /
  Abandoned). Plus infrastructure status, research threads, and
  the 102-file prior-art audit.
- **`.claude/knowledge/INTEGRATION_PLANS.md`** — versioned plan
  index, APPEND-ONLY. New plan versions prepend; prior versions
  stay with Status annotation. Active plan lives at
  `.claude/plans/<name>-v<N>.md`.

---

## The Governance Rules (never violate)

1. **Append-only on bookkeeping files.** Eight files carry the
   workspace's historical record; rows / sections inside them are
   immutable, with a short list of mutable fields per file.

   **Method hierarchy (preferred → discouraged):**

   1. **APPEND** (the method of choice) — add a NEW dated row to
      an existing section. Either via Edit (prepend inside a `---`
      bounded section) or Bash `cat >> file << EOF`. No prompt.
      The old row stays untouched. This is the double-bookkeeping
      pattern — new state enters as a new entry, not by mutating
      the old.
   2. **Edit field with prior Read** — flip a mutable field
      (Status / Confidence / Resolution / Payoff) on an existing
      row after reading the file. No prompt. Use for clarifications,
      status transitions, and minor updates. Prior Read is the
      workspace discipline (already in `CLAUDE.md`).
   3. **Write (full overwrite)** — prompts for approval on every
      bookkeeping file via `.claude/settings.json::permissions.ask`.
      Discouraged; only for wholesale replacement of a file that's
      been through explicit review. Never the default answer.

   Rule: **when in doubt, append**. The double-bookkeeping habit
   (new row rather than edited old row) preserves the full arc and
   keeps the audit trail legible. Edit-a-field is acceptable when
   the change is clearly a status transition on an existing entry;
   Write is the escape hatch that costs a confirmation.

   | File | Immutable | Mutable fields |
   |---|---|---|
   | `PR_ARC_INVENTORY.md` | PR rows (Added / Locked / Deferred / Docs) | Confidence line per entry. Corrections APPEND as dated lines. |
   | `LATEST_STATE.md` | Recently-shipped PR table rows | Snapshot sections (Current Inventory / Active Branches / Queued / Deferred) are updated by replacement, not history |
   | `STATUS_BOARD.md` | Deliverable rows (D-id / title / plan-version / scope) | Status column, PR / Evidence column per row |
   | `INTEGRATION_PLANS.md` | Plan entries (scope / path / deliverables) | Status and Confidence lines per entry |
   | `EPIPHANIES.md` | Dated entry bodies | Status line (FINDING / CONJECTURE / SUPERSEDED) |
   | `ISSUES.md` | Dated entry bodies | Status line (Open / Resolved / Wontfix / Superseded) + Resolution line (append on close) |
   | `IDEAS.md` | Dated entry bodies | Status line (Open / Implemented / Rejected / Deferred / Reactivated) + Rationale (append) |
   | `TECH_DEBT.md` | Dated entry bodies | Status line (Open / Paid / Moot) + Payoff line (append) + Priority / Scope (filled at creation, stable) |

   Core invariant: **rows are history; specific fields are state;
   never delete a row.** Supersedure is a new row that cites the
   old; the old row's Status updates to "Superseded by <new>".

   **Kanban discipline** — every entry in `ISSUES.md`, `IDEAS.md`,
   `TECH_DEBT.md` carries `**Priority:**` (P0-P3) and
   `**Scope:**` (@agent D<N> domain:<tag>) fields. Agents filter
   the board by their own `@`-mention or domain. Status moves
   through Open → In Progress → Resolved/Implemented/Paid. Nothing
   falls through the cracks because every ticket has an owner
   mentioned by `@`.
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
