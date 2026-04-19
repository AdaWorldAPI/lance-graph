# Status Board — Cross-Deliverable View

> Deliverable-level status across all active integration plans.
> **Status** and **PR / Evidence** columns are the only mutable
> fields — title, plan-version, and scope are immutable.
>
> For plan-level status see `INTEGRATION_PLANS.md`.
> For per-PR decision history see `PR_ARC_INVENTORY.md`.
> For current contract inventory see `LATEST_STATE.md`.

---

## Status Legend

| Status | Meaning |
|---|---|
| **Shipped** | Merged to main. PR column cites the merge commit. |
| **In PR** | PR open, under review. Not yet merged. |
| **In progress** | Active branch, code in flight, not yet PR. |
| **Queued** | Next up; spec is clear; work not started. |
| **Backlog** | Future; still in scope but not yet queued for a phase. |
| **Deferred** | Explicitly parked. Rationale recorded. Will be revisited. |
| **Abandoned** | Removed from scope. Rationale recorded. Will not be revisited. |

Rules:
- New rows APPEND (at the bottom of the relevant section).
- Status field is the ONLY field that gets edited in place.
- When a deliverable ships, record the PR number — never delete the
  row.
- When a deliverable is superseded by a different design, keep the
  row with Status = Abandoned and cite the replacement.

---

## elegant-herding-rocket-v1 — Phase-structured

Active integration plan, 12 deliverables D0 + D2–D11 (D1 dropped
early — CausalityFlow extension deferred). Plan path:
`.claude/plans/elegant-herding-rocket-v1.md`.

### Phase 1 — Shipped (PR #210, merged 2026-04-19)

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D0  | grammar-landscape.md + linguistic-epiphanies + fractal-codec knowledge docs | **Shipped** | #210 — 3 docs, 1151 LOC |
| D4  | ContextChain reasoning ops (coherence / replay / disambiguate / WeightingKernel) | **Shipped** | #210 — 396 LOC, 8 tests |
| D6  | Role-key catalogue with contiguous `[start:stop]` slice addressing | **Shipped** | #210 — 404 LOC, 7 tests |

### Phase 2 — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D2  | DeepNSM emits `FailureTicket` on low coverage | **Queued** | — |
| D3  | Grammar Triangle wired into DeepNSM via `triangle_bridge.rs` | **Queued** | — |
| D5  | Markov ±5 SPO+TEKAMOLO bundler with role-indexed VSA | **Queued** | — |
| D7  | NARS-tested grammar thinking styles (meta-inference policies) | **Queued** | — |

### Phase 3 — Queued

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D8  | Story-context bridge (AriGraph episodic + triplet-graph + orthogonal global-context) | **Queued** | — |
| D10 | Forward-validation harness (Animal Farm benchmark) | **Queued** | — |

### Phase 4 — Backlog

| D-id | Title | Status | PR / Evidence |
|---|---|---|---|
| D9  | ONNX story-arc export + ArcPressure / ArcDerivative awareness hook | **Backlog** | — |
| D11 | Bundle-perturb emergence interface (transformer-free generative stack) | **Backlog** | — |

### Dropped / Deferred from the plan itself

| D-id | Title | Status | Notes |
|---|---|---|---|
| D1  | CausalityFlow 3→9 slot extension (modal/local/instrument + beneficiary/goal/source) | **Deferred** | User decision; follow-up PR after Phase 2 |

---

## Infrastructure / governance (not in elegant-herding-rocket)

Workspace-level bootstrap work. Tracked here rather than PR_ARC
because it's process, not architecture.

| Item | Status | PR / Evidence |
|---|---|---|
| CLAUDE.md §Session Start — three mandatory reads | **Shipped** | #211 |
| CLAUDE.md §A2A Orchestration — two layers (runtime + session) | **Shipped** | #211 |
| CLAUDE.md §Model Policy — grindwork vs accumulation + never Haiku | **Shipped** | #211 |
| CLAUDE.md §GitHub Access Policy — zipball-for-reads | **Shipped** | #211 |
| `.claude/BOOT.md` session entry + prior-art links | **Shipped** | #211 |
| `.claude/agents/BOOT.md` orchestration spec (renamed from README) | **Shipped** | #211 |
| `.claude/agents/README.md` function inventory | **Shipped** | #211 |
| `.claude/knowledge/LATEST_STATE.md` current-state snapshot | **Shipped** | #211 |
| `.claude/knowledge/PR_ARC_INVENTORY.md` append-only decision arc | **Shipped** | #211 |
| `.claude/knowledge/INTEGRATION_PLANS.md` versioned plan index | **Shipped** | #211 |
| `.claude/knowledge/STATUS_BOARD.md` this file | **Shipped** | #211 |
| `.claude/settings.json` team-shared governance (ask/deny + hooks) | **Shipped** | #211 |
| `.claude/hooks/session-start.sh` + `post-compact.sh` | **Shipped** | #211 |
| `.claude/skills/cca2a/` pattern-explanation skill | **Shipped** | #211 |
| `.claude/plans/elegant-herding-rocket-v1.md` plan in workspace | **Shipped** | #211 |

## Infrastructure — queued

| Item | Status | Notes |
|---|---|---|
| `.claude/rules/` with `paths:` frontmatter | **Backlog** | Audit rec 2; replace / complement `READ BY:` headers with path-scoped loading |
| Skill `context: fork` + `agent:` field | **Backlog** | Audit rec 4; read-only isolation for search-only skill variants |
| Auto memory (`~/.claude/projects/<proj>/memory/`) | **Backlog** | Audit rec; unstructured addition to curated LATEST_STATE |

---

## Cross-cutting research threads (orthogonal to grammar work)

Separate research thread — not entangled with grammar/crystal/A2A.
Tracked here so it doesn't get lost.

| Item | Status | Notes |
|---|---|---|
| Named-Entity pre-pass (NER) — biggest OSINT blocker | **Deferred** | Dedicated PR after Phase 2 |
| FP_WORDS = 160 migration (currently 157) | **Deferred** | Needs coordinated ndarray change |
| Crystal4K 41:1 persistence compression | **Deferred** | ladybug-rs owns it; would port later |
| 200–500 YAML TEKAMOLO templates per language | **Deferred** | Training pipeline; future |
| Cross-linguistic active parsers (EN+FI+RU+TR) | **Deferred** | Role keys exist; parsers later |
| Fractal-descriptor leaf codec (MFDFA on Hadamard) | **Research** | `.claude/knowledge/fractal-codec-argmax-regime.md`. 30-min probe first. |
| UK Biobank cardiac MRI benchmark | **Research** | Downstream of fractal-codec probe |
| Chess vertical (ruci + lichess-bot integration) | **Deferred** | Capstone Tier 0, parallel stream |
| Wikidata ingest (1.2 B triples → 14.4 GB) | **Deferred** | `.claude/knowledge/wikidata-spo-nars-at-scale.md` |
| OSINT pipeline (spider + reader-lm + DeepNSM) | **Deferred** | `.claude/knowledge/osint-pipeline-openclaw.md` |
| Python/TypeScript grammar-stack convergence | **Deferred** | `.claude/knowledge/grammar-landscape.md` §7 |

---

## Prior-art audit (61 + 41 = 102 existing docs)

Before this session, the workspace accumulated 61 `.claude/*.md`
top-level docs + 41 `.claude/prompts/*.md` files across prior
sessions. They are indexed in `.claude/BOOT.md §Existing content`
and `CLAUDE.md §Prior art`, but their individual **status** (still
active / superseded / archival) has not been audited.

Status rows per bucket, not per file (102 rows would drown the
board — use filesystem + INTEGRATION_PLANS + PR_ARC for per-file
history):

| Bucket | Count | Status | Notes |
|---|---|---|---|
| `.claude/*.md` top-level calibration reports / handovers / audits / snapshots | 61 | **Audited** | PR #211 baseline; per-file vicinity classification in `ARCHIVE_INDEX.md`. |
| `.claude/prompts/*.md` scoped session / probe / handover prompts (incl. 12 in `fisher-z-wiring/`) | 41 (52 w/ subdir) | **Audited** | PR #211 baseline; per-file vicinity classification in `ARCHIVE_INDEX.md`. |
| `.claude/knowledge/*.md` structured knowledge | 12 | **Active** | Current; each has `READ BY:` header; used by Knowledge Activation triggers. |
| `.claude/agents/*.md` specialist + meta-agent cards | 24 | **Active** | Current; used by spawning + Knowledge Activation. |
| `.claude/hooks/*.sh` | 2 | **Active** | Wired via settings.json. |
| `.claude/skills/cca2a/*.md` | 3 | **Active** | Current. |
| `.claude/plans/*.md` integration plans | 1 (v1) | **Active** | Elegant herding rocket v1, Phase 1 shipped. |

**Backlog item — prior-art audit.** One-pass sweep across the
61+41 files. Per file: label as active / superseded / archival
with a one-line note. Deliverable = an `ARCHIVE_INDEX.md` that
splits the 102 into current vs historical, plus rename/move of
superseded files into an `archive/` subdirectory. Estimate ~200
LOC of meta work, ~2 hours of reading. **Not urgent**; useful
before the next major planning session.

## Update protocol

When a deliverable ships:
1. Edit this file's Status column in place for the row → **Shipped**.
2. Fill in PR / Evidence column with the merge commit or PR #.
3. Append a new section to `PR_ARC_INVENTORY.md` (Added / Locked /
   Deferred / Docs / Confidence).
4. Update `LATEST_STATE.md` (Recently Shipped PRs + Current Inventory
   if types change).

When a deliverable moves phase (e.g. Queued → In progress → In PR):
1. Edit Status column in place. Don't reorder rows.
2. If the move reflects scope correction, also update
   `INTEGRATION_PLANS.md` Status line for the parent plan.

When a new deliverable is added to a plan:
1. Append a new row at the bottom of the plan's section.
2. D-id is sequential in the plan (D12, D13, etc.).
3. Original scope becomes immutable once committed.

When a deliverable is abandoned:
1. Edit Status → **Abandoned**. Don't remove the row.
2. Cite the replacement in Notes.
