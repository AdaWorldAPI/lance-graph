# `.claude/ATT/` — Activation Receipt for `lance-graph`

> **Activated:** 2026-05-21 on branch `claude/activate-lance-graph-att-k2pHI`
> **Scope:** Single-repo activation (lance-graph only). Other repos in
> the 26-repo rollout activate via their own branches.

## What "activated" means in this repo

The three NLSpecs at `.claude/ATT/*.md` (`autoattended-orchestrator-spec.md`,
`anti-skim-agent-spec.md`, `agent-coordination-mcp-spec.md`) were
imported as the engineering specification governing this repo's
`.claude/agents/` ensemble. Activation wires the spec into four
workspace plumbing surfaces:

| Surface | Change | Purpose |
|---|---|---|
| `.claude/settings.json` | Added Edit/Write/MultiEdit deny rules for the 8 append-only bookkeeping files (`PR_ARC_INVENTORY.md`, `LATEST_STATE.md`, `STATUS_BOARD.md`, `INTEGRATION_PLANS.md`, `EPIPHANIES.md`, `ISSUES.md`, `IDEAS.md`, `TECH_DEBT.md`). | Closes [coordination-mcp-spec §7.2 / §9 DoD](./agent-coordination-mcp-spec.md#72-enforcement) — append-only governance now enforced at the settings layer, matching what `.claude/BOOT.md` already claimed was enforced. |
| `.claude/agents/BOOT.md` | New "ATT NLSpec Coupling" section near the top maps the four quality-lifecycle agents (`convergence-architect`, `preflight-drift-auditor`, `baton-handoff-auditor`, `brutally-honest-tester`) to ATT slots PP-14, PP-16, PP-15, PP-13 with non-overlapping verdict vocabularies. | Documents the operator ↔ engineering mapping. The agents already existed; this records the slot they occupy in the ATT taxonomy. |
| `.claude/BOOT.md` | "Existing content" section now points at both `.claude/EN/` (operator) and `.claude/ATT/` (engineering), naming the three NLSpecs. | New sessions learn the spec exists during the mandatory bootload. |
| `.claude/hooks/session-start.sh` | Added an "Engineering spec" block to the `additionalContext` injected at SessionStart. | New sessions see the ATT pointer as a system reminder on turn 1. |

No code under `crates/` was touched. Activation is documentation /
plumbing only.

## Definition-of-Done audit (per spec)

The three ATT specs each ship a DoD checklist. The status below is
honest as of the activation date — claims marked SATISFIED have a
named home in this repo; PARTIAL items have some coverage with a
known gap; GAP items have no current implementation and are filed
for follow-up work.

### `autoattended-orchestrator-spec.md` §10

| DoD item | Status | Lives at |
|---|---|---|
| Sprint plans are DOT graphs or §6.4-YAML mirrors with required attrs | PARTIAL | `.claude/board/sprint-log-{10..13}/` carry YAML-ish plans; not all required node/edge attrs are present. Gap for new sprints. |
| Validation rules §7 WAVE-001..WAVE-015 run via `preflight-drift-auditor` and block on ERROR | PARTIAL | `preflight-drift-auditor` exists (`.claude/agents/preflight-drift-auditor.md`) and `.claude/tools/preflight_drift.rs` exists; the WAVE-NNN rule IDs are not yet a one-to-one match. |
| Every worker spawns with `isolation: "worktree"` | SATISFIED | Agent tool default for spawned workers in current sessions. |
| Every worker writes a `status.json` matching §9.1 | GAP | Workers currently emit free-form completion text. `status.json` schema not enforced. |
| Missing `status.json` = FAIL (`auto_status=false`) | GAP | No status file is required today; this is the headline gap. |
| Four savants present with non-overlapping verdict vocabularies + non-use route-tables | SATISFIED | Mapping documented in `.claude/agents/BOOT.md § ATT NLSpec Coupling` (this activation). |
| PP-13 owns Rust Tier-1 toolchain | SATISFIED | `.claude/agents/brutally-honest-tester.md` is wired to `cargo clippy / cargo fmt / cargo test / cargo audit / cargo deny`. |
| PP-15 owns BAP1..BAP10 + 8 boundary classes | PARTIAL | `.claude/agents/baton-handoff-auditor.md` exists; explicit BAP1..BAP10 enumeration not in the card. |
| PP-16 owns PD1..PD10 + 6 axes + §7 validation | PARTIAL | `.claude/agents/preflight-drift-auditor.md` + `.claude/tools/preflight_drift.rs`; PD-IDs not yet enumerated. |
| Worker briefs declare `proof_of_read: true` + unique `sentinel_token` | GAP | No worker template enforces sentinel tokens today. |
| Meta-agent drains REQUESTS-FROM-AGENTS.md ≥ 2×/day | GAP | No `META/REQUESTS-FROM-AGENTS.md` file; `.claude/board/AGENT_LOG.md` is the closest analogue. |
| PR review classifies findings ONLY as P0 or P1 | SATISFIED | Existing convention in `.claude/board/PR_ARC_INVENTORY.md`. |
| `INVARIANTS.md` ≤ 500 lines | N/A | No `INVARIANTS.md` in this repo (iron rules live in `CLAUDE.md`). |
| One consolidation commit per wave at shared registry; zero N-mini-commit anti-pattern | PARTIAL | Convention followed by recent sprints; not yet a validation rule. |
| Files > 150 lines written via `tee -a` (chunking discipline) | PARTIAL | BOOT.md mandates `tee -a` for the 8 bookkeeping files; not yet enforced for all > 150-line writes. |
| Context Fidelity ladder §11A with precedence edge > node > graph > default | GAP | Context fidelity is not currently configured. |
| `fidelity=truncate` does NOT exempt the §3.3 reading-depth ladder | N/A | Depends on the fidelity ladder being live; see above. |

### `anti-skim-agent-spec.md` §10

| DoD item | Status | Lives at |
|---|---|---|
| Every worker brief contains a unique `sentinel_token` | GAP | Not enforced. |
| Worker first-reply begins with sentinel verbatim | GAP | Not enforced. |
| Workers emit `status.json` with proof-of-read entries per file | GAP | See orchestrator §9.1 gap above. |
| Proof-of-read entries declare `sha256` + `lines` + depth | GAP | No structured proof-of-read today. |
| Workers run §6.1 tool-call loop detector after every tool call (N=10) | GAP | Tool-call loop detection is not implemented; AP9 is described in the spec but not detected automatically. |
| Stuck workers use one of five §5.1 blocker types | GAP | No structured blocker schema today. |
| Meta-agent spot-checks one of LD-1..5 per savant per wave | GAP | No rotating spot-check protocol today. |
| Drift signals §4.3 scanned before goal-gate verdict | PARTIAL | `preflight-drift-auditor` runs board-vs-Cargo drift; LD-style drift signals not a separate scan. |
| PP-13 runs Tier-1 toolchain; Tier-1 failure → FAIL | SATISFIED | See orchestrator audit row above. |
| PP-13 anti-pattern scan covers AP1..AP9 | PARTIAL | `brutally-honest-tester.md` covers codex-class bugs; not all 9 APs are enumerated. |
| Tool-output truncation surfaces in proof-of-read as `truncated:head-N` / `truncated:tail-N` | GAP | Proof-of-read not yet structured. |
| `auto_status=false` mandatory; missing `status.json` = FAIL | GAP | See orchestrator §9.1 gap above. |

### `agent-coordination-mcp-spec.md` §9

| DoD item | Status | Lives at |
|---|---|---|
| Three layers (Teleport / Blackboard / Branch Pub/Sub) implementable with existing primitives | SATISFIED | `.claude/agents/*.md` (Layer-0), `.claude/board/AGENT_LOG.md` (Layer-1), `mcp__github__subscribe_pr_activity` (Layer-2). |
| Native MCP endpoints §5.1–§5.4 OR documented workaround | SATISFIED | Workaround mode (file blackboard + GitHub MCP) is the documented fallback. |
| `BlackboardEntry`, `ProofOfRead`, `Handover`, `RequestEntry`, `AnswerEntry` schemas implemented + validated | PARTIAL | `Handover` schema present (`.claude/agents/BOOT.md § Handover Protocol`); other four are described in the ATT spec but not implemented as validators in this repo. |
| Append-only governance §7 enforced at `.claude/settings.json` for the 8 bookkeeping files | **SATISFIED** | **Closed by this activation** — see `.claude/settings.json` deny rules added in this commit. |
| Single-mutable-file invariant: `Stand.md` / `STATUS_BOARD.md` is the ONLY file workers may overwrite | PARTIAL | `STATUS_BOARD.md` is in the deny list now (immutable rows; only Status column is mutable per BOOT.md table). The "only file workers may overwrite" invariant is not yet enforced as a positive allowlist. |
| Decision matrix §4 followed: workers use right transport for right message | SATISFIED | Documented in `.claude/agents/BOOT.md § A2A Orchestration`. |
| Coordination PRs are draft, named `claude/<topic>`, marked `Do not merge` | SATISFIED | Branch convention followed; "Do not merge" body marker is per-PR. |
| Handover files use §6.3 schema with required sections | SATISFIED | `.claude/agents/BOOT.md § Handover Protocol` matches §6.3 schema. |

## Headline gaps for follow-up

The activation closes one DoD item outright (append-only governance
at settings layer) and documents the operator ↔ engineering mapping
for the four quality-lifecycle savants. The structural gaps that
remain — and are filed for separate PRs, not this activation:

1. **`status.json` + proof-of-read schema** (orchestrator §9.1,
   anti-skim §7) — workers do not yet emit structured status with
   per-file `sha256` + `lines` + depth. This is the largest single
   gap and the headline conformance item.
2. **Sentinel-token / Lie-Detector LD-1..5 enforcement** (anti-skim
   §4) — no spot-checks today.
3. **Tool-call loop detection N=10** (anti-skim §6) — not
   implemented.
4. **Typed blocker schema for stuck workers** (anti-skim §5) — no
   schema today.
5. **DOT-graph sprint plans with WAVE-001..WAVE-015 validation**
   (orchestrator §6–§7) — current YAML-ish plans don't carry all
   required node/edge attrs.

Each gap should be addressed via a dedicated PR rather than bundled
into this activation, so the "spec is wired, code-level gaps are
visible" property of this receipt remains intact.

## Provenance

- Source spec: `.claude/ATT/README.md` (this directory)
- EN sibling: `.claude/EN/README.md`
- Format inspiration: [strongdm/attractor](https://github.com/strongdm/attractor)
- Activation branch: `claude/activate-lance-graph-att-k2pHI`
