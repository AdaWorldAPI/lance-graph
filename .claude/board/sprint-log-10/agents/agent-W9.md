# agent-W9 scratchpad — sprint-log-10 CCA2A

## Identity
- Worker: W9 (bevy-cull-plugin)
- Role: NdarrayCullPlugin proof plugin spec (consumes MailboxSoA for frustum cull)
- Output: `.claude/specs/pr-ce64-mb-7-bevy-cull-plugin.md`
- Date: 2026-05-14
- Authored from main thread (per CCA2A pattern — main thread can stand in for any W{N} if the worker was not spawned)

## Mandatory reads completed
1. `.claude/board/sprint-log-10/MANIFEST.md` — confirmed W9 scope + output target + bevy session round-2 agent #7 reference
2. `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §7 PR-CE64-MB-7 (primary scope) + §9 epiphanies + §13 fresh-eyes recursion 3rd pair (bevy session — diamond dep graph + Slice↔Plane bridge + NdarrayCullPlugin proof-first)
3. `.claude/specs/pr-ce64-mb-1-par-tile-crate.md` — diamond dep graph §572-593 confirms bevy as one of three par-tile consumers (ndarray / lance-graph / bevy)
4. `.claude/specs/pr-ce64-mb-5-mailbox-soa-attentionmask.md` (W6) §4.1-§4.2 — MailboxSoA<N> layout + push_row/drop_row signatures consumed in §6 of W9 spec
5. `.claude/specs/sprint-10-test-plan.md` §3 PR-CE64-MB-7 row + §7.6 bevy-cull-plugin × par-tile + §8.6 perf benches + §9.1 CI job
6. `.claude/specs/sprint-10-pr-dep-graph.md` — Wave 6 placement confirmed; bevy plugin lands last
7. `.claude/specs/sprint-10-execution-plan.md` (W12) §2 sprint-11 worker fleet row W9-impl + §7 wave sequencing
8. `.claude/board/AGENT_LOG.md` — existing entries reviewed; no prior W9 entry (this is first)

## Spec file
- Path: `/home/user/lance-graph/.claude/specs/pr-ce64-mb-7-bevy-cull-plugin.md`
- Size: ~14 KB (~500 LOC plugin estimate per parent plan §7; spec scaled to lowest-risk PR in the sprint)
- Sections: 13 (§1 scope, §2 crate layout, §3 Plugin impl, §4 cull_system, §5 FrustumSoA, §6 spawn system, §7 test plan 5+4+1+2, §8 benches, §9 CI, §10 files-to-touch, §11 risk matrix, §12 OQs, §13 cross-references)

## Key spec decisions

### Decision 1 — `ambiguous_with` not full replacement
Bevy stock `check_visibility` stays in schedule alongside `cull_system` with `ambiguous_with` marker. Conservative-write rationale: plugin can only HIDE, not REVEAL — never produces false-positive visibility. Defers the "kill stock cull" decision to sprint-12+. Lowers risk for the proof PR.

### Decision 2 — `BindSpaceView::empty_static()` helper in par-tile, not bevy-cull-plugin
Cross-spec touchpoint flagged as W9-OQ-1. The helper is generic (any consumer crate needing a placeholder view benefits), so it lives in W1's crate. Meta-review must verify W1 spec includes this; if absent, W1-impl adds before sprint-11 Wave 1.

### Decision 3 — Mailbox capacity default = 512 (per W6 OQ-N bevy bracket)
Matches W6 spec §4.1 comment: "Consumers pick: N=512 for bevy, N=4096 for Sigma8-branching." This spec defaults to 512 and exposes `NdarrayCullPlugin::mailbox_capacity` as runtime-configurable (saturating to 512 minimum).

### Decision 4 — Compartment-per-visible-entity (frame-ephemeral)
The plugin **produces** compartments, not just consumes them. Each frame drains previous frame's MailboxSoA rows then spawns one row per visible entity. Lifetime = 1 frame. Proves the producer side of the substrate as well as the consumer side. SigmaTier = TwigBranching (Σ7) — appropriate for per-frame ephemeral work.

### Decision 5 — Single-camera scope (sprint-11) → multi-camera deferred
`cameras.get_single()` narrows the proof. Multi-camera fan-out is a 1-line change (`.iter()`), deferred to sprint-12+. Flagged as W9-OQ-4 (soft limit).

### Decision 6 — Test count = 10 (5 correctness + 4 integration + 1 Miri-compat) + 2 schedule sanity
Slightly above sprint-10-test-plan.md §3.2 row's ~8 count because schedule-sanity tests (cull runs in PostUpdate, spawn runs after cull) are cheap to write and catch a class of bevy `Plugin::build` bugs that pure correctness tests miss.

## Key delta vs parent plan §7
Parent plan §7 names PR-CE64-MB-7 in a single one-line row ("Bevy plugin `NdarrayCullPlugin` consuming MailboxSoA for frustum cull"). This spec resolves it into:
- Concrete crate layout (`crates/bevy-cull-plugin/` with 11 source/test/bench files)
- `Plugin` impl with schedule-set placement (`VisibilitySystems::CheckVisibility` + `ambiguous_with` conservative-write)
- `intersects_sphere_x16` x16-lane SIMD consumption pattern (scalar tail for non-multiple-of-16 batches)
- Compartment-per-visible-entity producer-side proof (closes the loop: plugin consumes AND produces)
- Cross-spec touchpoint with W1 (`BindSpaceView::empty_static()`) flagged for meta-review verification
- Risk matrix (9 risks) + OQs (5) + files-to-touch (16 files including 1 par-tile cross-touch + 1 CI workflow edit)

## Open questions surfaced (for meta-review)
1. **W9-OQ-1 (cross-spec touchpoint):** `BindSpaceView::empty_static()` belongs in W1 or W9? Spec recommends W1. Meta-review verifies W1 spec; if absent, W1-impl adds before sprint-11 Wave 1. (HIGH — blocks W9-impl compile if unresolved.)
2. **W9-OQ-2 (bevy version pin):** Pin to bevy 0.14 stable for proof PR. Bump in sprint-12+. Verify workspace `Cargo.toml` has no conflicting bevy entry today.
3. **W9-OQ-3 (`ambiguous_with` schedule):** Bevy 0.14 supports `ambiguous_with` for Component-write conflicts in dev experience; if drift surfaces on first compile, fall back to feature-gating stock cull out entirely.
4. **W9-OQ-4 (multi-camera):** Soft scope limit — single-camera proof for sprint-11; multi-camera fan-out deferred to sprint-12+.
5. **W9-OQ-5 (CI feature matrix):** Run `cull-ndarray` ON in dedicated bevy-cull-plugin-tests job, OFF in workspace sweep. Documented in §9 of spec.

## Coordination notes
- This is the LAST spec in sprint-log-10 (Wave 6, depends on Waves 1-5 in the dep graph).
- Spec is the smallest of the W1-W9 series by design: low-risk proof PR with ~500 LOC envelope per parent plan §7.
- Meta-reviewer should verify the cross-spec touchpoint with W1 (BindSpaceView::empty_static) — single most important integration point.
- W7 (sigma-tier-router) spec is **also missing** from sprint-log-10/agents/ alongside this W9 — main thread should spawn W7 next OR write it from main thread before meta-review fires.

## Status: COMPLETE — spec at `.claude/specs/pr-ce64-mb-7-bevy-cull-plugin.md` (~14 KB)
