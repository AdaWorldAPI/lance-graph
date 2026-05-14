# Sprint-log-10 — CausalEdge64 mailbox + sparse-rename SoA composition spec fleet

**Branch:** `claude/causaledge64-mailbox-rename-soa-v1` (PR #371)
**Goal:** Produce 12 per-PR/per-deliverable specs that realize `.claude/plans/causaledge64-mailbox-rename-soa-v1.md`. The parent plan has 9 D-ids across 7 PRs + 1 ndarray-side prerequisite. This sprint produces the meticulously detailed specs each PR will follow during sprint-11+ implementation.

**Parent plan:** `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` (15 §, ~600 lines, ratified by all active sessions)
**Pattern:** CCA2A 12-worker + 1-meta (Sonnet workers, Opus meta), per `.claude/knowledge/A2Aworkarounds.md` Workaround 1 (File Blackboard)
**Permission:** `tee -a .claude/board/sprint-log-10/**` pre-allowed in `.claude/settings.json`; workers DO NOT commit (main thread aggregates).

## Fleet (12 Sonnet workers + 1 Opus meta)

| # | Worker | Output spec | Composes (plan §) | Output target |
|---|---|---|---|---|
| W1 | par-tile-crate | par-tile crate apex (Mailbox<T> trait + 3 backings + workspace deps) | §6 par-tile NEW · §7 PR-CE64-MB-1 | `.claude/specs/pr-ce64-mb-1-par-tile-crate.md` |
| W2 | causaledge64-v2 | CausalEdge64 v2 layout (G:5 + W:6 + truth:2 reclaim) + accessors + feature flag | §3 layout extension · §7 PR-CE64-MB-2 | `.claude/specs/pr-ce64-mb-2-causaledge64-v2.md` |
| W3 | pal8-nars-regression | PAL8 round-trip + NarsTables LUT regression tests (D-CE64-MB-2 + D-CE64-MB-3) | §3 compat invariants · §7 PR-CE64-MB-2 test plan | `.claude/specs/pr-ce64-mb-2-pal8-nars-regression.md` |
| W4 | bindspace-efgh | BindSpace Columns E (OntologyDelta) + F (AwarenessColumn) + G (ModelBindingColumn) + H (TypeColumn EntityTypeId u16) | §6 cognitive-shader-driver · §7 PR-CE64-MB-3 · `bindspace-columns-v1.md` Phase 2 | `.claude/specs/pr-ce64-mb-3-bindspace-efgh.md` |
| W5 | arigraph-spo-g | SPO-G quad upgrade in AriGraph + ghost-edge persistence + SpoWitnessChain<N> | §6 lance-graph/arigraph · §7 PR-CE64-MB-4 · `ogit-g-context-bundle-v1.md` D-OGIT-G-1 · `oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` §1-§8 | `.claude/specs/pr-ce64-mb-4-arigraph-spo-g.md` |
| W6 | mailbox-soa-attentionmask | MailboxSoA<N> + AttentionMask SoA + AttentionMaskActor + lifecycle | §4 + §5 + §7 PR-CE64-MB-5 | `.claude/specs/pr-ce64-mb-5-mailbox-soa-attentionmask.md` |
| W7 | sigma-tier-router | SigmaTierRouter (Σ1-Σ10 → mailbox backing) + InMemoryMailbox cycle-speed backing + plasticity + pruning triggers + budget | §7 PR-CE64-MB-6 · Σ10 Rubicon from `linguistic-epiphanies-2026-04-19.md` E21 · `THINKING_ORCHESTRATION_WIRING.md` Gap 4 closure | `.claude/specs/pr-ce64-mb-6-sigma-tier-router.md` |
| W8 | ndarray-miri-complete | u-word method gaps (simd_eq/ne/ge/gt/le/lt/clamp/select/zero on U16x32/U32x16/U64x8 + I-word symmetric) + cfg(miri) dispatch reroute in `ndarray/src/simd.rs` | §8 ndarray-side prerequisites · `PR-NDARRAY-MIRI-COMPLETE` | `.claude/specs/pr-ndarray-miri-complete.md` |
| W9 | bevy-cull-plugin | NdarrayCullPlugin proof plugin (consumes MailboxSoA for frustum cull) | §7 PR-CE64-MB-7 · bevy session round-2 agent #7 `intersects_sphere_x16` sketch | `.claude/specs/pr-ce64-mb-7-bevy-cull-plugin.md` |
| W10 | pr-dep-graph | Cross-PR dependency graph + sequencing + parallel-landability + gating | §7 sequencing table · §10 blast radius · §11 OQs that gate each PR | `.claude/specs/sprint-10-pr-dep-graph.md` |
| W11 | test-plan-unification | Regression test plan unified across all 7 PRs + Miri coverage extension + clippy gate | §12 iron rules · §3 compat invariants · `scripts/miri-tests.sh` extension | `.claude/specs/sprint-10-test-plan.md` |
| W12 | board-hygiene-execution | Sprint-10 execution plan + post-merge governance (LATEST_STATE + STATUS_BOARD + PR_ARC_INVENTORY hygiene per CLAUDE.md Mandatory Board-Hygiene Rule) | §14 board entries to add post-spec-ratify | `.claude/specs/sprint-10-execution-plan.md` |
| **M** | **meta-reviewer (Opus)** | Brutally honest cross-spec review + super-helpful per-worker solutions + cross-spec inconsistency flagging | reads all W1-W12 outputs + scratchpads | `.claude/board/sprint-log-10/meta-review.md` |

## CCA2A coordination protocol (per `.claude/knowledge/A2Aworkarounds.md`)

**Each worker MUST**:
1. Read `.claude/board/AGENT_LOG.md` BEFORE drafting (see what other workers already shipped this sprint).
2. Read mandatory plan(s) listed in worker's row.
3. Write ONE spec file to the path in their row (~10 KB target, but expand as needed for detail).
4. APPEND to `.claude/board/sprint-log-10/agents/agent-W{N}.md` via `tee -a` with their report.
5. APPEND to `.claude/board/AGENT_LOG.md` via `tee -a` with a one-line entry.
6. DO NOT git commit/push — main thread aggregates.
7. Report <120 words to main thread: spec byte size + plans cited + key delta-vs-prior + open questions surfaced.

**Meta-agent (Opus) MUST**:
1. Read all 12 worker scratchpads `.claude/board/sprint-log-10/agents/agent-W{N}.md`.
2. Read all 12 spec output files.
3. Read `.claude/board/AGENT_LOG.md` for context.
4. Produce **brutally honest** cross-spec review at `.claude/board/sprint-log-10/meta-review.md`:
   - Per-worker grade (A/B/C/D/F) with rationale
   - Cross-spec inconsistencies (where W2's spec conflicts with W4's, etc.)
   - Per-worker **super-helpful solutions** (concrete next-step that worker should take to address gaps)
   - Cross-cutting epiphanies (patterns that emerge only when all 12 specs are held in mind together)
   - Recommended PR-merge sequencing adjustments based on what specs surfaced
5. Report <300 words to main thread with overall sprint grade + critical-path next steps.

## Spec scope norms

- Each spec ~5-20 KB markdown (detail depth proportional to PR risk).
- **DELTA against parent plan**, not fresh draft. Cite §X.Y of parent plan being extended.
- Include: rationale, code/struct sketches, files-to-touch table, test plan, risk matrix, open questions.
- Forbidden: re-deriving architectural decisions already made in parent plan or composed plans.

## Output collation

Main thread after fleet completes:
1. Aggregate all spec files + scratchpads + meta-review into one commit on `claude/causaledge64-mailbox-rename-soa-v1`.
2. Update `STATUS_BOARD.md` with D-CE64-MB-1..9 rows (Status = Queued, PR/Evidence = `sprint-log-10/agent-W{N}.md`).
3. Update `INTEGRATION_PLANS.md` plan entry's **Status** if any OQ resolved.
4. Push to update PR #371.
