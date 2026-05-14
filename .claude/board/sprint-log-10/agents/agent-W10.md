# agent-W10 scratchpad — sprint-log-10 CCA2A

## Identity
- Worker: W10 (pr-dep-graph)
- Role: Cross-PR dependency graph + sequencing + parallel-landability + OQ gating
- Output: `.claude/specs/sprint-10-pr-dep-graph.md`
- Date: 2026-05-14

## Mandatory reads completed
1. `.claude/board/sprint-log-10/MANIFEST.md` — confirmed W10 scope
2. `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §7 (PR sequencing) + §10 (blast radius) + §11 (OQs) + §15 (readiness checklist) — primary source
3. `.claude/board/LATEST_STATE.md` — checked current contract inventory, active branches
4. `.claude/board/AGENT_LOG.md` — file does not exist yet (first W10 agent in sprint-10)
5. Worker spec files W1-W9 — not yet authored (sprint-10 workers running in parallel); this graph is based on parent plan §7 + crate-dep inference

## Spec file
- Path: `/home/user/lance-graph/.claude/specs/sprint-10-pr-dep-graph.md`
- Size: 25183 bytes (~24.6 KB)
- Sections: 11 (§1 scope, §2 PR inventory, §3 dep graph ASCII, §4 parallel-landability 6 waves, §5 OQ-to-PR gating table, §6 cross-spec consistency 6 checks, §7 risk matrix, §8 handover, §9 user gating recommendations, §10 files-to-touch, §11 cross-ref index)

## Key findings

### Dependency graph structure
- 8 PRs total (7 from parent plan §7 + 1 ndarray prerequisite from W8)
- 6 waves: Wave 0 (ndarray independent), Waves 1+2 (parallel: par-tile + causal-edge), Wave 3 (parallel pair: BindSpace + AriGraph), Wave 4 (convergence: MailboxSoA), Wave 5 (SigmaTierRouter), Wave 6 (bevy proof)
- Maximum parallelism: 3 workers simultaneous (W1 + W2 + W3 in Waves 1+2)

### Key delta vs parent plan §7
- Parent plan shows linear sequence PR-CE64-MB-1 through MB-7. This spec surfaces the ACTUAL parallel structure: Waves 1+2 are parallel; Wave 3 is a parallel pair; Wave 6 (bevy) can start after Wave 4 without waiting for Wave 5.
- Bevy plugin (PR-CE64-MB-7) does NOT hard-depend on SigmaTierRouter (PR-CE64-MB-6). Primary plugin path uses MailboxSoA directly. This is a meaningful unblocking: Wave 6 can proceed in parallel with Wave 5.

### OQ gating table (primary contribution)
- OQ-5 -> gates W1 (par-tile apex = Wave 1 foundation): must ratify before sprint-11 spawns
- OQ-1 -> gates W7 (SigmaTierRouter = Wave 5, highest-risk PR): must ratify before MB-6 merges
- OQ-3 -> gates W6 (MailboxSoA = Wave 4 convergence): must ratify before MB-5 merges
- OQ-2, OQ-7, OQ-8 -> ratify at meta-review or during sprint-11 W4/W5 prep
- OQ-4, OQ-6 -> documentation-only, defer

### Top 3 open questions for meta-review
1. **C-1 (accessor naming)**: W2's spec will define the actual accessor names for CausalEdge64 G/W/truth fields. Meta-reviewer MUST verify W6 and W7 specs use identical names. If they diverge, integration fails silently. This is the single highest-probability cross-spec integration bug.
2. **C-5 (SigmaTier enum residence)**: Recommend moving SigmaTier from par-tile to lance-graph-contract to avoid import cycle risk. Meta-reviewer should check if W6 and W7 independently defined SigmaTier and force reconciliation.
3. **C-2 (BindSpaceView lifetime)**: Parent plan §5 shows `BindSpaceView<'static>` in MailboxSoA rows, which requires Arc<BindSpace> interior. If W1 (par-tile) uses raw refs instead, W6 (MailboxSoA) cannot compile without unsound lifetime extension. This should be caught at meta-review before sprint-11 spawns.

## Coordination
- No other W1-W9 agents found to have shipped yet when W10 ran (sprint-log-10/agents/ was empty)
- This graph is based on parent plan + crate-dep inference, not on reading other worker specs
- Meta-reviewer should update this graph after reading all W1-W9 specs if any structural deps changed
