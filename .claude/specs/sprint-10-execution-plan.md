# Sprint-10 Execution Plan — CCA2A Orchestration + Sprint-11 Governance

> **Status:** Draft, 2026-05-14
> **Author:** W12 (Sonnet), sprint-log-10 board-hygiene-execution worker
> **Output target:** `.claude/specs/sprint-10-execution-plan.md`
> **Scope:** Orchestration meta-document for sprint-11 implementation fleet.
>   Defines: worker prompt template, CCA2A scratchpad protocol, board hygiene
>   per-PR, post-merge governance hand-off, STATUS_BOARD / LATEST_STATE /
>   PR_ARC_INVENTORY row discipline.
> **Parent plan:** `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §14 + §15
> **Format precedent:** `.claude/plans/sprint-5-through-9-roadmap-v1.md`
> **Delta vs. prior:** W12's unique deliverables = sprint-11 worker prompt template
>   (§3), board hygiene per-PR enforcement table (§5), OQ resolution tracking (§8),
>   cross-session coordination protocol (§9). Everything else is composition of
>   named-and-reviewed pieces.

---

## §1 Statement of Scope

This document is the **orchestration spec** for sprint-11 implementation. It does **not** re-derive architectural decisions made in `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` (the parent plan) or any of the 12 per-PR spec files produced by sprint-10 workers W1-W11. Those specs are the authoritative implementation contracts.

This document defines:

1. **Sprint-11 worker fleet** — 12 impl workers + roles (§2)
2. **Worker prompt template** — verbatim spawn text for each impl worker (§3)
3. **CCA2A scratchpad protocol** — per-worker append-only log discipline (§4)
4. **Board hygiene per-PR** — which board files update when, in the same commit (§5)
5. **Post-merge governance** — how the main thread closes out each merged PR (§6)
6. **Sprint-10 → sprint-11 hand-off** — gating criteria + sequencing (§7)
7. **OQ resolution tracking** — 8 open questions, per-worker owners, ratification path (§8)
8. **Cross-session coordination** — Branch Pub/Sub + File Blackboard protocols (§9)
9. **Meta-reviewer responsibility** — M agent scope + grading criteria (§10)
10. **Sprint-11 completion criteria** — what "done" means before sprint-12 spawns (§11)
11. **Risk matrix** — HIGH/MED/LOW risks + mitigations (§12)

---

## §2 Sprint-11 Worker Fleet

12 implementation workers, each owns exactly one branch and one PR. Parallel where the dep-graph (`.claude/specs/sprint-10-pr-dep-graph.md`) allows.

| # | Worker ID | Deliverable | Spec file | Branch pattern |
|---|---|---|---|---|
| W1-impl | par-tile-crate | `Mailbox<T>` trait + InMemoryMailbox + TokioMailbox + SupabaseSubMailbox backings + workspace `Cargo.toml` entry | `.claude/specs/pr-ce64-mb-1-par-tile-crate.md` | `claude/sprint-11-W1-par-tile` |
| W2-impl | causaledge64-v2 | CausalEdge64 v2 layout (G:5+W:6+truth:2 in-place reclaim of reserved 13 bits) + accessor methods + `#[cfg(feature = "causal-edge-v2")]` feature flag | `.claude/specs/pr-ce64-mb-2-causaledge64-v2.md` | `claude/sprint-11-W2-ce64-v2` |
| W3-impl | pal8-nars-regression | PAL8 round-trip tests + NarsTables LUT regression (bit-layout invariants must not shift post-v2) | `.claude/specs/pr-ce64-mb-2-pal8-nars-regression.md` | `claude/sprint-11-W3-pal8-nars` |
| W4-impl | bindspace-efgh | BindSpace Columns E (`OntologyDelta`) + F (`AwarenessColumn`) + G (`ModelBindingColumn`) + H (`TypeColumn EntityTypeId u16`) + `CollapseGate::Superposition` mode | `.claude/specs/pr-ce64-mb-3-bindspace-efgh.md` | `claude/sprint-11-W4-bindspace-efgh` |
| W5-impl | arigraph-spo-g | SPO-G quad upgrade in AriGraph (G = OGIT domain u32 slot) + ghost-edge persistence (Pearl rung 3/7) + `SpoWitnessChain<N>` Cow-shaped witness chain | `.claude/specs/pr-ce64-mb-4-arigraph-spo-g.md` | `claude/sprint-11-W5-arigraph-spog` |
| W6-impl | mailbox-soa-attentionmask | `MailboxSoA<N>` compartment SoA + `AttentionMask` rename SoA (G/W/style/truth slot tables) + `AttentionMaskActor` lifecycle (LRU eviction) | `.claude/specs/pr-ce64-mb-5-mailbox-soa-attentionmask.md` | `claude/sprint-11-W6-mailbox-soa` |
| W7-impl | sigma-tier-router | `SigmaTierRouter` (Σ1-Σ10 mailbox backing dispatch) + `InMemoryMailbox` cycle-speed backing + plasticity bit-counter + pruning triggers + attention budget | `.claude/specs/pr-ce64-mb-6-sigma-tier-router.md` | `claude/sprint-11-W7-sigma-router` |
| W8-impl | ndarray-miri-complete | u-word method gaps (`simd_eq/ne/ge/gt/le/lt/clamp/select/zero` on `U16x32`/`U32x16`/`U64x8` + I-word symmetric) + `cfg(miri)` dispatch reroute in `ndarray/src/simd.rs` | `.claude/specs/pr-ndarray-miri-complete.md` | `claude/sprint-11-W8-ndarray-miri` |
| W9-impl | bevy-cull-plugin | `NdarrayCullPlugin` proof plugin (consumes `MailboxSoA` for frustum cull via `intersects_sphere_x16`) + bevy integration smoke test | `.claude/specs/pr-ce64-mb-7-bevy-cull-plugin.md` | `claude/sprint-11-W9-bevy-cull` |
| W10-impl | pr-sequencing-coord | Observer role: opens PRs in sequence per dep graph, tracks per-PR CI status, resolves merge conflicts, escalates OQ-blocked PRs to main thread | `.claude/specs/sprint-10-pr-dep-graph.md` | (no own branch; works on existing impl branches) |
| W11-impl | test-plan-executor | Observer role: runs cross-PR integration tests + Miri extended sweep + perf benchmarks (bevy cull 2-10x target); reports per-PR test matrix results | `.claude/specs/sprint-10-test-plan.md` | (no own branch; runs against impl branches) |
| W12-impl | post-merge-hygiene | Post-merge board hygiene per CLAUDE.md Mandatory Board-Hygiene Rule: LATEST_STATE.md + PR_ARC_INVENTORY.md + STATUS_BOARD.md updates per merged PR; this spec is the continuation of W12's deliverable | (this document) | (no own branch; hygiene commits go in impl PRs) |

**Observation role vs. implementation role:** W10-impl and W11-impl are observer/coordinator roles — they do not own code branches. Their job is cross-PR sequencing and test execution. The meta-reviewer M is Opus and runs after all 12 impl PRs are open (not before).

---

## §3 Worker Prompt Template (Sprint-11)

Copy verbatim for each impl worker. Substitute `{N}`, `{topic}`, `{spec-file}`.

```
You are sprint-11 worker W{N}-impl, Sonnet, CWD /home/user/lance-graph. CCA2A pattern.

WRONG-REPO GUARDRAIL — run this before any write:
  import os; from github import Github, Auth
  tok = os.environ["GITHUB_TOKEN"].strip().strip('"').strip("'")
  g = Github(auth=Auth.Token(tok))
  repo = g.get_repo("AdaWorldAPI/lance-graph")
  assert repo.full_name == "AdaWorldAPI/lance-graph", f"WRONG REPO: {repo.full_name}"

Your spec: .claude/specs/{spec-file}.md — READ IN FULL FIRST
Your scratchpad: .claude/board/sprint-log-11/agents/agent-W{N}-impl.md (tee -a, append-only)
Your branch: claude/sprint-11-W{N}-{topic} (create new from origin/main; one branch per worker)

Mandatory read-order BEFORE coding:
1. .claude/board/sprint-log-11/MANIFEST.md
2. .claude/specs/{spec-file}.md  <- your spec — implement exactly this
3. .claude/specs/sprint-10-pr-dep-graph.md  <- where your PR lands in wave sequence
4. .claude/specs/sprint-10-test-plan.md  <- tests your PR must pass
5. .claude/board/AGENT_LOG.md  <- what other sprint-11 workers already shipped
6. .claude/board/LATEST_STATE.md  <- Tier-0 mandatory (what exists, do NOT re-propose)
7. .claude/plans/causaledge64-mailbox-rename-soa-v1.md  <- parent plan sections 0-3 + 10

Implementation discipline:
- Implement EXACTLY what the spec says.
- If you need to deviate, append OQ entry to AGENT_LOG and STOP — escalate to main thread.
- Cargo clippy -D warnings is the pre-merge gate (per CLAUDE.md Clippy-first rule).
- Write the unit tests named in your spec. Aim for 100% spec-driven test coverage.
- Commit incrementally (at logical boundaries, not per-file).
- Board hygiene is part of YOUR commit — do NOT open a PR without it.

Board hygiene (per CLAUDE.md Mandatory Board-Hygiene Rule — SAME COMMIT as code):
- STATUS_BOARD.md: update your D-CE64-MB-N row from Queued to In progress to In PR
- When PR opens: confirm STATUS_BOARD row is "In PR" and AGENT_LOG entry exists
- DO NOT open PR without these board updates in the same branch commit

Scratchpad protocol:
- After each commit, run:
    DATE=$(date -u +%Y-%m-%dT%H:%M)
    tee -a .claude/board/sprint-log-11/agents/agent-W{N}-impl.md > /dev/null <<EOF
    ## $DATE — <action> (sonnet)
    **Commit:** $(git rev-parse --short HEAD)
    **LOC delta:** +N / -M across X files
    **Tests:** N passed (M new); clippy clean: yes/no
    **Open questions:** <bullet list or "none">
    EOF
- After PR opens, also append one-liner to .claude/board/AGENT_LOG.md

Reporting (< 200 words to main thread when done):
- PR URL
- LOC delta (total)
- Test count (new + total passing)
- Clippy status (clean / warnings count)
- Board hygiene status (STATUS_BOARD row, AGENT_LOG entry)
- Top 3 open questions (if any)
```

---

## §4 CCA2A Scratchpad Protocol

Every sprint-11 implementation worker MUST follow this protocol. No exceptions.

### Scratchpad file

`.claude/board/sprint-log-11/agents/agent-W{N}-impl.md`

Created at worker start via:
```bash
mkdir -p .claude/board/sprint-log-11/agents/
DATE=$(date -u +%Y-%m-%dT%H:%M)
tee -a .claude/board/sprint-log-11/agents/agent-W{N}-impl.md > /dev/null <<EOF
## $DATE — W{N}-impl started (sonnet)
**Spec:** .claude/specs/{spec-file}.md
**Branch:** claude/sprint-11-W{N}-{topic}
**Plans cited:** causaledge64-mailbox-rename-soa-v1.md + sprint-10-pr-dep-graph.md + sprint-10-test-plan.md
EOF
```

### Entry format (per commit)

```
## YYYY-MM-DDTHH:MM — <action> (sonnet)

**Commit:** `<short-hash>`
**LOC delta:** +N / -M across X files
**Tests:** N passed (M new); clippy clean: yes/no
**Open questions:** <bullet list or "none">
```

### Rules

1. **Append-only via `tee -a`** — never overwrite. Overwriting voids the append-only guarantee.
2. **One entry per commit** — not one per file, not one per session. Commits = entries.
3. **Timestamped** — `date -u +%Y-%m-%dT%H:%M` at time of write.
4. **Clippy status mandatory** — clippy clean yes/no is load-bearing (CI gate).
5. **AGENT_LOG.md also gets one-liner** — per CLAUDE.md Layer-2 pattern. The AGENT_LOG entry is the cross-worker blackboard; the per-agent scratchpad is the worker's private log.

### Layer-2 blackboard one-liner format (AGENT_LOG.md)

```
## YYYY-MM-DDTHH:MM — W{N}-impl <action> (sonnet, sprint-11)
**D-ids:** D-CE64-MB-N
**Commit:** `<short-hash>`
**Tests:** N pass (M new)
**Outcome:** <one-line summary>
```

---

## §5 Board Hygiene Per-PR

Per CLAUDE.md Mandatory Board-Hygiene Rule. **Board hygiene updates are part of the implementation PR — NOT a separate cleanup commit.** The retroactive-hygiene anti-pattern (merge then clean up) is explicitly forbidden.

### Trigger table

| Lifecycle moment | Required board file updates (same commit) |
|---|---|
| PR branch created | `STATUS_BOARD.md` row updated: Status = "In progress" |
| PR opened for review | `STATUS_BOARD.md` row updated: Status = "In PR"; `AGENT_LOG.md` one-liner appended |
| PR merged | `LATEST_STATE.md` "Recently Shipped PRs" table PREPEND (new top row); `LATEST_STATE.md` "Current Contract Inventory" APPEND (if new types ship); `PR_ARC_INVENTORY.md` PREPEND entry (Added/Locked/Deferred/Docs/Confidence fields); `STATUS_BOARD.md` row updated: Status = "Shipped" |
| Finding/correction surfaces during impl | `EPIPHANIES.md` PREPEND dated entry |
| Tech debt discovered | `TECH_DEBT.md` row append |
| New unresolved issue / blocker | `ISSUES.md` row append |

### Who commits the post-merge hygiene

**The implementation worker** commits the `LATEST_STATE.md` + `PR_ARC_INVENTORY.md` + `STATUS_BOARD.md` changes in the SAME commit that introduces the feature code. The main thread reviews and confirms before merge. This is enforced by the worker prompt in §3: "DO NOT open a PR without board hygiene updates in the same branch commit."

The W12-impl post-merge hygiene role (§2) is a **review and verification** role, not a catch-up role. W12-impl reads merged PRs, checks that all board fields are populated correctly, and appends correction entries if any fields are missing — but missing fields should be rare if the worker prompt template is followed correctly.

### PR_ARC_INVENTORY.md entry format (per merged PR)

```markdown
## PR #N — <title>

**Added:** <comma-separated list of new types/crates/files>
**Locked:** <architectural decisions made irreversible by this PR>
**Deferred:** <items explicitly deferred to later PR>
**Docs:** <spec file(s) that governed this PR>
**Confidence (post-merge):** High / Med / Low — <one-line rationale>
```

### STATUS_BOARD.md row format

```
| D-CE64-MB-N | <title> | Queued / In progress / In PR / Shipped | PR #N | `<commit-hash>` |
```

---

## §6 Post-Merge Governance Per-PR

The post-merge governance process runs after each sprint-11 PR merges. **Main thread (not the implementation worker)** performs final verification.

### Per-PR governance checklist (main thread)

After PR #N merges to main:

1. **Verify board hygiene completeness** — read `LATEST_STATE.md`, `PR_ARC_INVENTORY.md`, `STATUS_BOARD.md`. Confirm the merged PR's entries are present and correctly populated.
2. **If any field is missing** — W12-impl opens a one-commit correction PR directly on main that appends the missing entry. Cite the missed PR. Flag as "W12-impl post-merge correction" in the PR title.
3. **Update `INTEGRATION_PLANS.md`** — if the merged PR resolves an OQ listed in §11 of the parent plan, update the plan's **Status** field (the only mutable field in INTEGRATION_PLANS.md entries).
4. **Cross-worker unblocking** — check `sprint-10-pr-dep-graph.md` wave table; if this merge unblocks a downstream worker branch, notify W10-impl (or append an AGENT_LOG entry that W10-impl will read).
5. **Epiphany check** — if the impl surfaced a new architectural finding, prepend to `EPIPHANIES.md`.

### Anti-patterns (explicitly forbidden)

| Anti-pattern | Why forbidden | Correct alternative |
|---|---|---|
| Retroactive cleanup commit (merge then notice board stale then separate PR) | Violates CLAUDE.md "same commit" rule | Board hygiene goes IN the implementation PR |
| LATEST_STATE.md edits in a standalone "governance" PR after the code PR | Same as above | LATEST_STATE row prepend goes in the code PR |
| Bulk board hygiene commit that covers multiple PRs | Loses per-PR traceability | One board-hygiene block per PR, in that PR |
| W12-impl rewrites prior board entries | APPEND-ONLY rule | Corrections append as new dated entries |

---

## §7 Sprint-10 to Sprint-11 Hand-Off

### Sprint-10 deliverables (this sprint)

Sprint-10 produces:
- 12 per-PR spec files (W1-W11 outputs) at `.claude/specs/`
- 1 execution plan (this document, W12 output)
- 1 meta-review at `.claude/board/sprint-log-10/meta-review.md` (M agent, Opus)
- All aggregated into one commit on `claude/causaledge64-mailbox-rename-soa-v1` then PR #371

### Gating criteria before sprint-11 spawns

The following must be true before the main thread spawns the sprint-11 fleet:

| Gate | Owner | Status trigger |
|---|---|---|
| User ratifies OQ-1 (Sigma-tier banding policy) | User + main thread | Explicit "go" on OQ-1 in session |
| User ratifies OQ-3 (compartment plasticity update granularity) | User + main thread | Explicit "go" on OQ-3 |
| User ratifies OQ-5 (rayon vendor decision — std::thread::scope first) | User + main thread | Explicit "go" on OQ-5 |
| PR #371 merged (sprint-10 specs on main) | main thread | GitHub merge |
| Meta-review grades >= B for all 12 workers (or re-spawn corrections complete) | M agent | meta-review.md verdict |
| `sprint-log-11/MANIFEST.md` scaffold created | main thread | File exists on main |

If meta-review surfaces a worker that needs re-spawn (grade < B), that worker re-runs with the meta-review's "super-helpful concrete next-step" as the corrective prompt before sprint-11 spawns. Sprint-11 does NOT spawn against incomplete specs.

### Sprint-11 wave sequencing

Follows `.claude/specs/sprint-10-pr-dep-graph.md`. Expected waves:

| Wave | Workers | Blocked by |
|---|---|---|
| Wave 0 | W8-impl (ndarray prerequisites) | nothing (independent repo) |
| Wave 1 | W1-impl (par-tile crate) | Wave 0 merged |
| Wave 2 | W2-impl + W3-impl (CausalEdge64 v2 + regression) | Wave 1 merged |
| Wave 3 | W4-impl (BindSpace E/F/G/H) + W5-impl (AriGraph SPO-G) | Wave 2 merged |
| Wave 4 | W6-impl (MailboxSoA) | Wave 3 merged |
| Wave 5 | W7-impl (SigmaTierRouter) | Wave 4 merged |
| Wave 6 | W9-impl (bevy cull plugin) | Wave 5 merged |

W10-impl and W11-impl run as observers across all waves. W12-impl runs post each merge.

**Sprint-11 expected duration:** 3-6 weeks (per parent plan §7 LOC estimates aggregated).

---

## §8 OQ Resolution Tracking

Parent plan §11 identifies 8 open questions. This section tracks their resolution owners and ratification path.

| OQ | Summary | Tentative resolution | Worker spec resolving | Ratification path |
|---|---|---|---|---|
| **OQ-1** | Sigma-tier banding (Sigma1-5 Tokio / Sigma6-8 InMemory / Sigma9-10 escalate; or should Sigma4-5 be cycle-speed?) | Keep Sigma1-5 Tokio (Zone-2 reflexes); Sigma6-8 InMemoryMailbox (Zone-1 cycle-speed) | W7-impl spec | **User must ratify before sprint-11 Wave 5** |
| **OQ-2** | Ghost edge decay in AriGraph (NARS confidence drift vs. fixed Pearl rung 3) | NARS truth-revise on ghosts at AriGraph-commit boundaries (low-frequency, batched) | W5-impl spec | Meta-review acceptance of W5 spec |
| **OQ-3** | Compartment plasticity update granularity (bit-counter per emission + NARS at AriGraph commit) | bit-counter per emission (high-freq, AttentionMask-side) + NARS truth-refine at AriGraph commit (low-freq, batched) | W6-impl + W7-impl specs | **User must ratify before sprint-11 Wave 4** |
| **OQ-4** | INT4-32D cold-start wiring (SigmaTierRouter spawn path K-NN fallback) | yes — compartment spawn path uses INT4-32D K-NN fallback; wiring deferred until PR-CE64-MB-6 | W7-impl spec | Meta-review acceptance of W7 spec |
| **OQ-5** | Rayon vendor decision (vendored rayon-shape vs std::thread::scope) | Start with `std::thread::scope`; promote to vendored rayon if profiling shows throughput cliff | W1-impl spec | **User must ratify before sprint-11 Wave 1** |
| **OQ-6** | Vsa16kF32 residence (single-cycle Markov bundle, dropped at cycle end) | Stays in `crystal/fingerprint.rs` for within-cycle Markov bundle; dropped at cycle end; NO cumulative state | E-CE64-MB-2 epiphany (locked) | Already resolved — no ratification needed |
| **OQ-7** | AwarenessColumn sizing (256 B/row per bindspace-columns-v1.md §3) | 256 B/row — stays full BindSpace width; compartments write via gated delta | W4-impl spec | Meta-review acceptance of W4 spec |
| **OQ-8** | Witness shape (SpoWitness64 packed vs. SpoWitnessChain<N>) | Both supported — `SpoWitness64` (Copy, 8B, peer mailbox edges) + `SpoWitnessChain<N>` (Cow-shaped, supervisor + AriGraph-commit edges) | W5-impl spec | Meta-review acceptance of W5 spec |

### OQ-to-PR gating

| Unratified OQ | Blocks |
|---|---|
| OQ-1 unratified | Wave 5 (PR-CE64-MB-6 / W7-impl) |
| OQ-3 unratified | Wave 4 (PR-CE64-MB-5 / W6-impl) |
| OQ-5 unratified | Wave 1 (PR-CE64-MB-1 / W1-impl) |
| OQ-2, OQ-4, OQ-7, OQ-8 unratified | Non-blocking — tentative resolution in spec is sufficient; meta-review validates |

When a worker's PR opens on a spec with an unratified OQ affecting that PR, W10-impl must **hold** the PR from merging until the user ratifies. W10-impl appends a hold notice to AGENT_LOG.md.

---

## §9 Cross-Session Coordination

Sprint-10 and sprint-11 may overlap across multiple Claude Code sessions. Two coordination primitives (per `.claude/knowledge/A2Aworkarounds.md`):

### Primitive 1: File Blackboard (Workaround 1)

`.claude/board/AGENT_LOG.md` is the Layer-2 blackboard. Every agent run (sprint-10 or sprint-11) appends one entry before exiting.

- All sprint-11 workers MUST read `AGENT_LOG.md` before starting work.
- All sprint-11 workers MUST append a one-liner after opening their PR.
- The log is append-only; main thread may reorder at board-hygiene time.

### Primitive 2: Branch Pub/Sub (Workaround 2)

Sprint-10 and sprint-11 coordination branch is PR #371 (`claude/causaledge64-mailbox-rename-soa-v1`). Sessions that need real-time push notifications subscribe via:

```python
mcp__github__subscribe_pr_activity(owner="AdaWorldAPI", repo="lance-graph", pullNumber=371)
```

When a sprint-11 worker pushes a new commit, subscribed sessions receive a `<github-webhook-activity>` event, re-pull `AGENT_LOG.md`, see the new entry, and build on it without a direct inter-agent message.

### Session start protocol (every new sprint-11 session)

A session resuming sprint-11 work MUST, in this order:

1. Read `.claude/board/LATEST_STATE.md` (what is merged)
2. Read `.claude/board/AGENT_LOG.md` (what is in flight)
3. Read `.claude/board/sprint-log-11/agents/agent-W{N}-impl.md` (own prior scratchpad)
4. Check `STATUS_BOARD.md` for D-id row status
5. Then resume where left off

**DO NOT** start fresh each session. The scratchpad is the resumption point.

---

## §10 Meta-Reviewer Responsibility

The M agent in sprint-10 is Opus (per CLAUDE.md Model Policy — accumulation requires Opus).

### Reading list (M agent mandatory reads)

1. All 12 worker scratchpads at `.claude/board/sprint-log-10/agents/agent-W{N}.md`
2. All 12 spec output files (W1-W12)
3. `.claude/board/AGENT_LOG.md` (sprint-10 entries)
4. Parent plan §11 (OQs) and §15 (readiness checklist)
5. `.claude/plans/sprint-5-through-9-roadmap-v1.md` (format precedent for meta-reviews)

### meta-review.md required sections

Output at `.claude/board/sprint-log-10/meta-review.md`:

```markdown
# Sprint-10 Meta-Review (Opus)

## Overall sprint grade: <letter> — <one-line rationale>

## Per-worker grades

| Worker | Grade | Rationale | Super-helpful concrete next-step |
|---|---|---|---|
| W1 | <A/B/C/D/F> | ... | ... |
...

## Cross-spec inconsistencies

For each pair (Wi, Wj) where Wi's spec contradicts Wj's: cite both specs, the conflict,
and recommended resolution.

## Cross-cutting epiphanies

Patterns that only emerge when all 12 specs are held in mind together.

## Recommended PR-merge sequencing adjustments

Any wave-order changes recommended based on spec contents. W10 spec is authoritative;
M can recommend adjustments with rationale.

## Sprint-11 spawn decision

- Spawn immediately: YES / NO
- If NO: which workers need re-spawn, with corrected prompt?
```

### Grading rubric

| Grade | Criteria |
|---|---|
| A | Spec is a proper DELTA against parent plan. Cites the section being extended. Includes files-to-touch table, test plan with named tests, risk matrix, OQ callouts. LOC estimates credible. No re-derivation of already-decided architecture. |
| B | One of the above missing or shallow. Otherwise solid. |
| C | Two or more missing. Spec is too sparse or re-derives already-decided architecture. Usable but needs revision before sprint-11. |
| D | Fundamental misunderstanding of scope. Spec would cause worker to implement the wrong thing. Must be re-spawned. |
| F | Empty or wrong repo. Re-spawn required. |

---

## §11 Sprint-11 Completion Criteria

Sprint-11 is **done** when all of the following are true:

| Criterion | Owner | Verification |
|---|---|---|
| All 8 PRs merged (PRs CE64-MB-1 through CE64-MB-7 + ndarray-miri-complete) | W10-impl | GitHub PR list |
| `cargo clippy --tests --no-deps -D warnings` exits 0 across workspace | W11-impl | CI pass |
| Miri extended sweep green (~1500 Miri-clean tests per W11 spec target) | W11-impl | `scripts/miri-tests.sh` output |
| Bevy cull plugin perf benchmark validates 2-10x speedup vs. stock cull | W11-impl | W9-impl bench output |
| Board hygiene complete per §5: all 8 PRs have LATEST_STATE + PR_ARC_INVENTORY + STATUS_BOARD rows | W12-impl | Board file audit |
| OQ-1 + OQ-3 + OQ-5 formally ratified (user explicit "go") | User + main thread | Session record |
| All other OQs resolved via meta-review acceptance of corresponding spec | M agent verdict | meta-review.md |
| `STATUS_BOARD.md` D-CE64-MB-1..9 rows all show "Shipped" | W12-impl | Board audit |
| No open blockers in `ISSUES.md` related to this plan | main thread | Issues audit |

When all criteria are met, main thread spawns sprint-12 to consume the now-live substrate (additional bevy plugins, INT4-32D cold-start wiring, vendored rayon-shape, etc.).

---

## §12 Risk Matrix

| Risk | Severity | Probability | Mitigation |
|---|---|---|---|
| Worker drift from spec ("worker improvises" anti-pattern) | HIGH | Med | Sprint-11 prompt template mandates "implement EXACTLY what the spec says; if deviation needed, escalate via OQ." Meta-review (M agent) catches drift in sprint-10 specs before sprint-11 spawn. |
| Parallel-merge conflicts on shared crate files | MED | High | Wave sequencing in W10-impl spec serializes merges in dep order. Workers on the same wave touch disjoint crates. W10-impl holds merge until conflicts resolved. |
| OQ ratification delay blocks gating PRs | MED | Med | OQ-1/OQ-3/OQ-5 are flagged as user-ratification-required in §8. W10-impl explicitly holds affected PRs until ratified. Main thread escalates OQs to user in session. |
| Board hygiene drift (worker forgets LATEST_STATE / PR_ARC) | LOW | Med | Worker prompt template includes explicit board-hygiene checklist. W12-impl does post-merge verification and can open correction PRs. |
| Meta-review surfaces multiple grade-D specs | MED | Low | M agent (Opus) catches this in sprint-10; re-spawn specific workers with corrective prompts before sprint-11 spawns. Sprint-11 does NOT spawn against grade-D specs. |
| PAL8 / NarsTables bit-layout regression from CausalEdge64 v2 | HIGH | Low | W3-impl spec is dedicated to PAL8 + NarsTables regression tests. W3-impl PR lands in Wave 2 alongside W2-impl and must pass before Wave 3 unblocks. |
| ndarray-side prerequisites not merged in time (PR-NDARRAY-MIRI-COMPLETE) | MED | Low | W8-impl is Wave 0 (independent; no lance-graph deps). Should land before Wave 1 starts. W10-impl monitors ndarray PR status. |
| Bevy cull plugin depends on MailboxSoA interface (Wave 6 is last) | LOW | Low | W9-impl is in Wave 6; W6+W7 must merge first. If bevy interface differs from W6 spec, W9-impl escalates. Risk is interface mismatch, not correctness — easy to fix. |

---

## §13 Cross-References

**Plans this composes + depends on (DELTA discipline — no re-derivation):**

- `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §14 (board entries post-spec-ratify) + §15 (final readiness checklist) — this document implements the checklist as runnable governance
- `.claude/plans/sprint-5-through-9-roadmap-v1.md` — canonical sprint roadmap format; worker prompt template in §3 above extends the format from that doc's "Worker-prompt template" section
- `.claude/knowledge/A2Aworkarounds.md` Workaround 1 (File Blackboard) + Workaround 2 (Branch Pub/Sub) — §9 of this doc applies these directly
- `.claude/knowledge/cca2a-sprint-prompt-template.md` — wrong-repo guardrail snippet embedded verbatim in §3 worker prompt template
- CLAUDE.md Mandatory Board-Hygiene Rule — §5 trigger table above is a direct implementation of that rule's per-type trigger table

**This document does NOT:**
- Re-derive the CausalEdge64 bit layout (§3 of parent plan owns this)
- Re-derive the MailboxSoA or AttentionMask structure (W6-impl spec owns this)
- Propose new architectural patterns (all patterns named in E-CE64-MB-1..10 epiphanies)
- Define per-PR test cases (W11-impl spec owns this at `.claude/specs/sprint-10-test-plan.md`)

**Board files this spec triggers (per CLAUDE.md Mandatory Board-Hygiene Rule):**

When this spec lands with the sprint-10 commit batch:
- `.claude/board/INTEGRATION_PLANS.md` — PREPEND new entry for `causaledge64-mailbox-rename-soa-v1` plan
- `.claude/board/STATUS_BOARD.md` — append D-CE64-MB-1 through D-CE64-MB-9 rows (Status = Queued)
- `.claude/board/EPIPHANIES.md` — PREPEND E-CE64-MB-1..10 entries (already authored in §9 of parent plan)
- `.claude/board/PR_ARC_INVENTORY.md` — one entry per merged PR in sprint-11 series (added post-merge by impl workers per §5)
- `.claude/board/LATEST_STATE.md` — one row per merged PR (added post-merge by impl workers per §5)

---

*End of sprint-10-execution-plan.md — W12 deliverable, sprint-log-10 board-hygiene-execution worker.*
