# Sprint-10 PR Dependency Graph — Cross-PR Sequencing, Parallel-Landability, and OQ Gating

> **Worker:** W10 (pr-dep-graph) — sprint-log-10 CCA2A fleet
> **Role:** Meta-spec. Sequences code-authoring PRs; does not author new architecture.
> **Parent plan:** `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §7 (PR sequencing table), §10 (blast radius), §11 (OQs), §15 (readiness checklist)
> **Scope:** Cross-PR dependency graph + sequencing + parallel-landability + OQ gating for sprint-11+ implementation
> **Date drafted:** 2026-05-14
> **Confidence:** High — graph is a composition of named-and-reviewed pieces per parent plan §1.

---

## §1 Statement of Scope

This spec is the **canonical PR-merge order** for sprint-11+ implementation of `.claude/plans/causaledge64-mailbox-rename-soa-v1.md`. It identifies:

1. Which PRs can land **in parallel** (different crates, no shared types being introduced in the same window)
2. Which PRs must **serialize** (downstream depends on upstream type definitions or binary-layout stability)
3. Which **OQs from §11 of the parent plan** must be ratified before a given PR can merge (not just before it can be drafted)
4. The **cross-spec consistency invariants** that workers W1-W9 must hold simultaneously

This spec is a meta-spec: it reads the 9 worker specs (W1-W9) and produces the ordering graph that the sprint-11 implementation wave follows. Where W1-W9 specs are still being authored (sprint-10), this graph is based on the parent plan's §7 table plus what can be inferred from crate dependencies. The meta-reviewer (Opus, sprint-10 close) should reconcile any delta between this graph and the actual W1-W9 spec outputs.

**What this spec does NOT do:**
- Author new architecture (the parent plan is the authority)
- Re-derive the CausalEdge64 bit layout, MailboxSoA structure, or AttentionMask semantics
- Reproduce code sketches (those live in the per-PR worker specs W1-W9)

---

## §2 PR Inventory (8 PRs total)

7 PRs from parent plan §7 + 1 ndarray-side prerequisite from W8 spec.

| PR# | Source | Owner spec | Crate(s) | Reverse deps |
|---|---|---|---|---|
| **PR-NDARRAY-MIRI-COMPLETE** | parent §8 | W8 spec | `ndarray` (sibling repo) | W9 bevy-cull-plugin: Miri-checkable cull paths; W1 par-tile: Miri-clean polyfill paths |
| **PR-CE64-MB-1** | parent §7 | W1 spec | `crates/par-tile/` (new crate) | W6 MailboxSoA, W7 SigmaTierRouter, W9 bevy-cull-plugin |
| **PR-CE64-MB-2** | parent §7 | W2 spec | `crates/causal-edge/` | W6 MailboxSoA, W7 SigmaTierRouter, W5 AriGraph |
| **PR-CE64-MB-2-test** | parent §7 (test gate) | W3 spec | `crates/causal-edge/` (test-only) | Merge gate for PR-CE64-MB-2 |
| **PR-CE64-MB-3** | parent §7 | W4 spec | `crates/cognitive-shader-driver/` | W6 MailboxSoA, W5 AriGraph, W9 bevy-cull-plugin |
| **PR-CE64-MB-4** | parent §7 | W5 spec | `crates/lance-graph/src/graph/arigraph/` | W7 SigmaTierRouter, W6 MailboxSoA |
| **PR-CE64-MB-5** | parent §7 | W6 spec | `crates/par-tile/` + `crates/lance-graph-supervisor/` | W7 SigmaTierRouter |
| **PR-CE64-MB-6** | parent §7 | W7 spec | `crates/lance-graph-supervisor/` + `crates/par-tile/` | W9 bevy-cull-plugin (soft dep — plugin primary path uses MailboxSoA directly without SigmaTierRouter) |
| **PR-CE64-MB-7** | parent §7 | W9 spec | `bevy-cull-plugin` (new crate, bevy fork side) | None — leaf node, proof step |

**Note on PR-CE64-MB-7 vs SigmaTierRouter dependency:** The bevy plugin (W9) consumes `MailboxSoA` directly for its core frustum-cull loop. It does NOT require `SigmaTierRouter` for its primary proof path. SigmaTierRouter integration is additive (advanced routing for Sigma9-Sigma10 escalation). PR-CE64-MB-7 hard-depends on PR-CE64-MB-5 (MailboxSoA) but soft-depends on PR-CE64-MB-6 (SigmaTierRouter). Plugin can proceed in parallel with Wave 5 once Wave 4 merges.

---

## §3 Dependency Graph (ASCII)

```
    PR-NDARRAY-MIRI-COMPLETE  (W8, ndarray sibling repo — independent)
           |
           |  unblocks Miri coverage for par-tile + bevy-cull-plugin
           v
    +------------------------------------------------------+
    |  WAVE 1+2 — can land in parallel                     |
    |                                                      |
    |  PR-CE64-MB-1          PR-CE64-MB-2 -----> PR-CE64-MB-2-test
    |  (W1, par-tile new)    (W2, causal-edge v2) (W3, test gate)
    |  no consumers yet      in-place extension   merge gate for MB-2
    |                                                      |
    +------------------------------------+-----------------+
                                         |  MB-2 merges after MB-2-test passes
                    +--------------------+------------------+
                    |  WAVE 3 — parallel pair               |
                    |                                       |
                    |  PR-CE64-MB-3          PR-CE64-MB-4  |
                    |  (W4, BindSpace E/F/G/H) (W5, AriGraph SPO-G)
                    |  cognitive-shader-driver  arigraph/   |
                    |  closes PR355#6+FIX-5    closes D-OGIT-G-1
                    |                                       |
                    +-----------------+---------------------+
                                      |
                                      v
                    PR-CE64-MB-5  (W6, MailboxSoA + AttentionMaskActor)
                    par-tile + lance-graph-supervisor
                    depends: MB-1 + MB-2 + MB-3 + MB-4 all merged
                                      |
                                      v
                    PR-CE64-MB-6  (W7, SigmaTierRouter + InMemoryMailbox)
                    lance-graph-supervisor + par-tile
                    depends: MB-5 merged
                    gates on: OQ-1 ratification (Sigma-tier banding policy)
                                      |
                                      | soft dep (MailboxSoA sufficient for
                                      | plugin primary path without this)
                                      v
                    PR-CE64-MB-7  (W9, NdarrayCullPlugin bevy proof)
                    bevy-cull-plugin (new crate, bevy fork)
                    hard depends: MB-5 merged
                    soft depends: MB-6 (advanced routing)
                    also depends: PR-NDARRAY-MIRI-COMPLETE (Miri CI gate)
                    LEAF NODE — no reverse deps
```

**Crate boundary summary:**

| Crate | Changed by | Depends on (this sprint) |
|---|---|---|
| `ndarray` (sibling) | W8 | — |
| `crates/par-tile/` (new) | W1, W6 | ndarray (Miri coverage) |
| `crates/causal-edge/` | W2, W3 | — (in-place extension) |
| `crates/cognitive-shader-driver/` | W4 | causal-edge v2 |
| `crates/lance-graph/arigraph/` | W5 | causal-edge v2 |
| `crates/lance-graph-supervisor/` | W6, W7 | par-tile, arigraph |
| `bevy-cull-plugin` (new) | W9 | par-tile, ndarray |

---

## §4 Parallel-Landability Analysis

### Wave 0 (prerequisite, any time — independent of lance-graph)

**PR-NDARRAY-MIRI-COMPLETE (W8)** lives in the ndarray sibling repo. Independent of all lance-graph PRs. Should land on ndarray's own branch/PR cycle. Effect on this sprint: unblocks Miri CI gate for par-tile dispatch paths and bevy-cull-plugin cull paths. **Par-tile CAN develop and ship without Miri completeness; Miri completeness is needed for the Miri CI gate in PR-CE64-MB-1 and PR-CE64-MB-7, not for compilation.**

Recommendation: W8 drafts ndarray spec; main thread lands ndarray PR in parallel with sprint-11 waves 1-3, not as a strict gate.

### Wave 1 (sprint-11 kickoff — par-tile apex)

**PR-CE64-MB-1 (W1, par-tile crate)** — the diamond apex. Pure new crate, no consumers yet. Can be authored and merged on its own branch before any other MB PR. Once merged, par-tile's `Mailbox<T>` trait + `InMemoryMailbox` + `AttentionMask` struct are stable.

**Must precede:** PR-CE64-MB-5, PR-CE64-MB-6, PR-CE64-MB-7.

**Gates on OQ-5** (rayon vendor decision) — must be ratified before PR-CE64-MB-1 merges.

### Wave 2 (parallel with Wave 1 — causal-edge extension)

**PR-CE64-MB-2 + PR-CE64-MB-2-test (W2 + W3)** — in-place extension to `crates/causal-edge/`. Different crate from par-tile. PR-CE64-MB-2-test is a required CI gate for PR-CE64-MB-2: W3 test spec defines the test plan, W2 implementation must pass W3's tests before PR-CE64-MB-2 merges.

**Can land in parallel with Wave 1.** No shared types introduced in the same window. `causal-edge` does not depend on `par-tile`; Wave 1 par-tile does not yet depend on causal-edge v2 accessors (those come in Wave 4 via MailboxSoA).

**Must precede:** PR-CE64-MB-3, PR-CE64-MB-4, PR-CE64-MB-5.

### Wave 3 (parallel pair — after Wave 2 merges)

**PR-CE64-MB-3 (W4)** and **PR-CE64-MB-4 (W5)** can land **in parallel with each other.** Neither depends on the other at the type level:

- PR-CE64-MB-3 adds Columns E/F/G/H + `BindSpaceView` + `CollapseGate MergeMode::Superposition` to `cognitive-shader-driver`. Does not touch arigraph.
- PR-CE64-MB-4 adds SPO-G quad mode + ghost-edge persistence + `SpoWitnessChain<N>` to `lance-graph/arigraph`. Does not touch cognitive-shader-driver.

Both depend on causal-edge v2 (Wave 2) because both use `CausalEdge64::g_slot()` (PR-CE64-MB-3 in BindSpaceView column-mask filtering; PR-CE64-MB-4 for SPO-G quad G-slot storage).

**Merge conflict risk:** Low — different crates, different modules.

**Must precede:** PR-CE64-MB-5.

### Wave 4 (sequential convergence point)

**PR-CE64-MB-5 (W6, MailboxSoA + AttentionMaskActor)** — highest-fan-in PR. Converges: par-tile crate (Wave 1), causal-edge v2 (Wave 2), BindSpaceView type (Wave 3 MB-3), AriGraph commit target (Wave 3 MB-4). Also edits `crates/lance-graph-supervisor/`. **Merge conflict risk MEDIUM** with PR-CE64-MB-6 (W7 also edits lance-graph-supervisor/src/lib.rs). Mitigation: W6 lands PR-CE64-MB-5 first; W7 rebases PR-CE64-MB-6 on top.

**Gates on OQ-3** (plasticity granularity) — must be ratified before PR-CE64-MB-5 merges.

### Wave 5 (sequential — after Wave 4)

**PR-CE64-MB-6 (W7, SigmaTierRouter)** — extends lance-graph-supervisor with Sigma-tier dispatcher. Calls `MailboxSoA::push_row`, `dispatch_cycle`, `drop_row`.

**Gates on OQ-1** (Sigma-tier banding policy) — must be ratified before PR-CE64-MB-6 merges.

### Wave 6 (proof step — leaf, can start after Wave 4)

**PR-CE64-MB-7 (W9, NdarrayCullPlugin)** — proof leaf. Hard dep on PR-CE64-MB-5 (MailboxSoA stable). Soft dep on PR-CE64-MB-6 (SigmaTierRouter optional for advanced routing). **Can proceed in parallel with Wave 5** once Wave 4 merges. If OQ-1 delays PR-CE64-MB-6, W9 is not blocked.

---

## §5 OQ-to-PR Gating Table

Derived from parent plan §11 (OQ-1 through OQ-8).

| OQ | Title | Gates PR(s) | Timing | Resolution path |
|---|---|---|---|---|
| **OQ-1** | Sigma-tier banding policy (should Sigma4-Sigma5 also use cycle-speed InMemoryMailbox?) | PR-CE64-MB-6 (W7) | Must ratify before MB-6 merges | W7 spec §2 banding policy recommendation; user ratifies before W7 spec accepted. Core question: are Sigma4-Sigma5 "fast reflexes" better served by cycle-speed InMemoryMailbox or Tokio-backed mailbox? |
| **OQ-2** | Ghost-edge decay policy (NARS confidence drift vs fixed Pearl rung 3?) | PR-CE64-MB-4 (W5) | Must ratify before MB-4 merges | W5 spec recommends NARS truth-revise on ghosts at AriGraph-commit boundaries (low-frequency, batched). Ratify at meta-review or pre-sprint-11-W5. |
| **OQ-3** | Compartment plasticity update granularity | PR-CE64-MB-5 (W6) + PR-CE64-MB-6 (W7) | Must ratify before MB-5 merges | Recommended: bit-counter per emission (high-freq, AttentionMask-side) + NARS truth-refine at AriGraph commit (low-freq, batched). W6 + W7 must agree on interface boundary. |
| **OQ-4** | INT4-32D cold-start wiring in SigmaTierRouter spawn path | PR-CE64-MB-6 (W7) | Defer post-merge — document fallback only | W7 spec documents K-NN cold-start path but defers wiring to sprint-11+. No PR blocked on this OQ. |
| **OQ-5** | Rayon vendor decision (vendored rayon-shape vs std::thread::scope in par-tile) | PR-CE64-MB-1 (W1) | Must ratify before MB-1 merges | W1 spec recommends: start with std::thread::scope + crossbeam (~500 LOC, Miri-friendly); defer rayon-vendor (2 KLOC) until profiling shows throughput cliff. User ratifies this deferral. |
| **OQ-6** | Vsa16kF32 final residence (stays in `crystal/fingerprint.rs`?) | None (documentation-only) | Confirm at meta-review | Confirmed in parent plan §9 E-CE64-MB-2: Vsa16kF32 stays in crystal/fingerprint.rs for within-cycle Markov bundle only; dropped at cycle end. No PR action needed. |
| **OQ-7** | AwarenessColumn (Column F) sizing (256 B/row?) | PR-CE64-MB-3 (W4) | Must ratify before MB-3 merges | Parent plan §11 confirms 256 B/row is the full BindSpace width. W4 spec confirms. Ratify at meta-review or pre-sprint-11-W4. |
| **OQ-8** | SpoWitness shape: SpoWitness64 packed vs SpoWitnessChain<N> | PR-CE64-MB-4 (W5) + PR-CE64-MB-6 (W7 consumes) | Must ratify before MB-4 merges | Parent plan §11: both supported; sender picks by destination (SpoWitness64 for peer mailbox edges, SpoWitnessChain<N> for parent-supervisor + AriGraph-commit edges). W5 + W7 must agree. |

**User ratification priority order (before sprint-11 spawns):**

1. **OQ-5 first** — gates W1 (par-tile = Wave 1 = everything's foundation)
2. **OQ-1 second** — gates W7 (SigmaTierRouter = highest-risk PR, Wave 5)
3. **OQ-3 third** — gates W6 (MailboxSoA = Wave 4 convergence point)
4. OQ-2, OQ-7, OQ-8 — ratify at meta-review or during sprint-11 W5/W4 prep
5. OQ-4, OQ-6 — documentation-only, defer indefinitely

---

## §6 Cross-Spec Consistency Checks

The following invariants MUST hold across all worker specs simultaneously. The meta-reviewer (Opus) must verify each at sprint-10 close before sprint-11 spawns.

### C-1: CausalEdge64 accessor naming (W2 defines, W6+W7 consume)

| Accessor | Defined by | Used by | Required name |
|---|---|---|---|
| OGIT domain slot reader | W2 | W6, W7 | `.g_slot() -> u8` |
| Witness palette slot reader | W2 | W6, W7 | `.w_slot() -> u8` |
| Truth band reader | W2 | W6, W7 | `.truth() -> TrustTexture` |
| Setters (builder pattern) | W2 | W6 (constructs outbound CE64 in dispatch_cycle) | `.with_g_slot(u8)`, `.with_w_slot(u8)`, `.with_truth(TrustTexture)` |

**Failure mode:** If W2 names `.g()` and W6 calls `.g_slot()`, integration fails at compile time with method-not-found. Must be caught in meta-review before sprint-11.

### C-2: BindSpaceView lifetime strategy (W1 defines, W4+W6+W9 consume)

Parent plan §5 shows `pub bindspace_views: [BindSpaceView<'static>; N]` in `MailboxSoA`. This requires the BindSpaceView to hold an `Arc<BindSpace>` internally (not a raw reference) so MailboxSoA rows can own their views without lifetime entanglement. All four workers must use the same strategy:

- W1 defines `BindSpaceView` with `Arc<BindSpace>` interior
- W4 adds Columns E/F/G/H to the `Arc<BindSpace>` interior
- W6 stores `BindSpaceView` in `MailboxSoA<N>` rows
- W9 borrows a `BindSpaceView` per cull frame

**Failure mode:** If W1 uses raw refs (`&'a BindSpace`) and W6 uses `'static`, the SoA cannot be constructed without unsound lifetime extension. Catch at meta-review.

### C-3: CollapseGate MergeMode::Superposition semantics (W4 defines, W6+W7 consume)

`MergeMode::Superposition` = preserve both delta buffers when two CausalEdge64 emissions are XOR-equal (neither cancel as in XOR mode nor merge as in Bundle mode). W4 must include a docstring stating this exactly. W6 and W7 must consume it with the same semantics. If semantics disagree, EPIPHANY escalation silently misroutes.

### C-4: SpoWitness shape variants (W5 defines, W7 consumes)

- `SpoWitness64`: `Copy, u64`-packed — for peer mailbox edges (high-frequency, hot path)
- `SpoWitnessChain<N>`: `Cow`-shaped — for parent-supervisor + AriGraph-commit edges (low-frequency)

W7 SigmaTierRouter must use `SpoWitness64` for intra-tier routing and `SpoWitnessChain<N>` for AriGraph commits. If W7 emits the wrong variant, AriGraph sees malformed quads.

### C-5: SigmaTier enum canonical residence

`SigmaTier` (Sigma1-Sigma10 tier enum) must live in ONE canonical location. Parent plan §5 places it in `crates/par-tile/src/mailbox_soa.rs`. W7 SigmaTierRouter must import from par-tile, not re-define.

**Recommended alternative:** Add `SigmaTier` to `lance-graph-contract` (zero-dep crate) following the established pattern for `TrustTexture`, `ThinkingStyle`, etc. Both W6 and W7 then import from contract. Meta-reviewer should flag any worker that re-defines `SigmaTier` locally.

### C-6: AriGraph SPO-G quad commit API (W5 defines, W6+W7 consume)

W5 must expose a stable `arigraph.commit_spog(s, p, o, g, witness)` surface. W6's `MailboxSoA::dispatch_cycle` calls it when `intents[i].is_some() || sigma_tier >= Sigma7`. W7's SigmaTierRouter calls it on Sigma9-Sigma10 EPIPHANY escalation. Both must call the same API.

Ghost-edge emission API `arigraph.emit_ghost(rung, s, p, o, g)` must also be stable before W7 can call it from the pruning path.

---

## §7 Risk Matrix (Cross-PR)

| Risk | Severity | Prob | Affected PRs | Mitigation |
|---|---|---|---|---|
| Type-naming drift (C-1) — W2 accessor names differ from what W6/W7 call | HIGH | Medium | MB-2, MB-5, MB-6 | Explicit accessor name table in §6 C-1 is the contract. Meta-review audit before sprint-11. |
| BindSpaceView lifetime confusion (C-2) — raw ref vs Arc approach varies | HIGH | Medium | MB-1, MB-3, MB-5, MB-7 | W1 spec MUST define Arc-backed view as the canonical approach. Meta-reviewer flags any deviation. |
| Merge conflict on lance-graph-supervisor/lib.rs (W6 + W7 both edit) | MED | High (guaranteed overlap) | MB-5, MB-6 | Serial merge: MB-5 first, W7 rebases MB-6 on top. W6 + W7 coordinate via scratchpad. |
| OQ-1 ratification delay blocks W7 | MED | Low-Medium | MB-6 | Escalate OQ-1 to user early. W7 can draft spec with OQ-1 flagged but not merge. |
| SpoWitness shape disagreement (C-4) — W5 and W7 pick wrong variants for EPIPHANY path | MED | Medium | MB-4, MB-6 | OQ-8 ratification + C-4 check at meta-review. |
| SigmaTier enum duplication (C-5) — W6 defines in par-tile, W7 re-defines in supervisor | MED | Low (if C-5 recommendation followed) | MB-5, MB-6 | Pre-agree before sprint-11 W6/W7 spawn: SigmaTier goes to lance-graph-contract. |
| AriGraph commit API mismatch (C-6) | MED | Low-Medium | MB-4, MB-5, MB-6 | W5 exposes stable commit API in spec; meta-reviewer verifies signature consistency across W5/W6/W7. |
| ndarray Miri delay (W8) blocks W9 Miri CI gate | LOW | Low | NDARRAY-MIRI, MB-7 | Plugin ships without Miri CI gate first; Miri gate added as follow-up once ndarray PR lands. |
| Bevy plugin (W9) accidentally coupling to SigmaTierRouter primary path | LOW | Low | MB-7, MB-6 | W9 spec must explicitly note: "primary path: MailboxSoA only; SigmaTierRouter: optional advanced routing." |

---

## §8 Sprint-10 to Sprint-11+ Handover

### Sprint-10 deliverable

At sprint-10 end (all 12 specs land via PR #371 merge):
- 12 ratified specs + 1 Opus meta-review + this PR-dep-graph spec
- OQ-5 and OQ-1 queued for user ratification (can happen before sprint-11 spawns)
- Cross-spec consistency checks C-1 through C-6 audited by meta-reviewer

### Sprint-11 implementation wave

| Wave | PRs | Parallelism | Gate condition |
|---|---|---|---|
| Wave 0 | PR-NDARRAY-MIRI-COMPLETE | Independent (ndarray repo) | None |
| Wave 1 | PR-CE64-MB-1 | Yes (no consumers yet) | OQ-5 ratified |
| Wave 2 | PR-CE64-MB-2 + MB-2-test | Yes, parallel with Wave 1 | None (in-place extension) |
| Wave 3 | PR-CE64-MB-3 + MB-4 | Yes, parallel with each other; after Wave 2 merges | OQ-2, OQ-7, OQ-8 ratified |
| Wave 4 | PR-CE64-MB-5 | No (convergence) | Waves 1-3 all merged; OQ-3 ratified |
| Wave 5 | PR-CE64-MB-6 | No (depends Wave 4) | Wave 4 merged + OQ-1 ratified |
| Wave 6 | PR-CE64-MB-7 | Can start after Wave 4 (parallel with Wave 5) | Wave 4 merged |

**Sprint-11 parallelism budget:** W1 + W2 + W3 spawn simultaneously (Waves 1+2). W4 + W5 spawn in parallel once Wave 2 merges. W6 then W7 (serial). W9 can start alongside W7 once W6 (MB-5) merges.

### Sprint-12+ continuation

- Remaining bevy plugins (NdarraySplat / NdarrayCognitive / NdarrayAudio, bevy session Phase 4) — depend on MB-1 through MB-7 substrate stable
- Deferred OQs: rayon-vendor (OQ-5 follow-on if profiling shows cliff), INT4-32D cold-start path (OQ-4), Sigma4-Sigma5 banding follow-on (OQ-1 residual)
- lance-graph-callcenter Zone-3 surface (postgrest / drain / grpc / supabase-realtime) remains **completely unchanged** per parent plan §10 blast radius

---

## §9 Gating Recommendations for the User

### Before sprint-11 spawns (action required)

1. **Ratify OQ-5** (rayon vendor): confirm "start with std::thread::scope; defer rayon-vendor." Unblocks W1 (par-tile = Wave 1).

2. **Ratify OQ-1** (Sigma-tier banding): confirm or amend:
   - Sigma1-Sigma5 STATIC repair -> Tokio-backed mailbox
   - Sigma6 EMERGENT + Sigma7-Sigma8 TWIG -> InMemoryMailbox cycle-speed
   - Sigma9-Sigma10 EPIPHANY -> escalate to L4 lance-graph-planner
   
   Key question: should Sigma4-Sigma5 also use cycle-speed InMemoryMailbox?

3. **Ratify OQ-3** (plasticity granularity): confirm "bit-counter per emission + NARS truth-refine at AriGraph commit boundaries." Unblocks W6 spec finalization.

### During sprint-11 (per-wave gates)

- Each wave's merge gates on previous wave all merged + per-PR test plan in worker spec
- C-1 through C-6 audit findings from sprint-10 meta-review must be addressed before affected PR merges

### Post sprint-11 (validation)

- Bevy plugin (PR-CE64-MB-7) is the cross-cutting proof: if it ships clean, the planar-carrier abstraction is validated end-to-end
- Plugin shipping clean = MailboxSoA-backed frustum cull compiles, tests pass, Miri-clean with PR-NDARRAY-MIRI-COMPLETE merged

---

## §10 Files-to-Touch Summary (sprint-11 conflict detection)

| File path | Workers | Notes |
|---|---|---|
| `crates/par-tile/` (entire new crate) | W1, W6 | W1 lays foundation; W6 adds MailboxSoA + AttentionMaskActor. Serial: W1 first. |
| `crates/causal-edge/src/edge.rs` | W2 | Bit layout constants + new accessor methods. In-place. |
| `crates/causal-edge/tests/` | W3 | New test files only; no edits to W2's source. |
| `crates/cognitive-shader-driver/src/bindspace.rs` | W4 | Columns E/F/G/H added. Existing Columns A-D untouched. |
| `crates/cognitive-shader-driver/src/collapse_gate.rs` | W4 | New MergeMode::Superposition variant. |
| `crates/lance-graph/src/graph/arigraph/triplet_graph.rs` | W5 | SPO-G quad mode. SPO triple API preserved. |
| `crates/lance-graph/src/graph/arigraph/episodic.rs` | W5 | Ghost-edge persistence at Pearl rung 3/7. |
| `crates/lance-graph-supervisor/src/lib.rs` | W6, W7 | **HIGH CONFLICT RISK.** W6 first, W7 rebases. |
| `crates/lance-graph-supervisor/src/sigma_tier_router.rs` | W7 | New file. Verify W6 spec scope does NOT create this file. |
| `bevy-cull-plugin/` (entire new crate, bevy fork) | W9 | New crate. No conflict. |

**Files NOT touched (blast radius boundary per parent plan §10):**
- `crates/lance-graph-callcenter/src/drain.rs`
- `crates/lance-graph-callcenter/src/postgrest.rs`
- `crates/lance-graph-callcenter/src/version_watcher.rs`
- `crates/lance-graph-callcenter/src/auth.rs` / `policy.rs` / `rbac.rs`
- `crates/cognitive-shader-driver/src/grpc.rs`
- `crates/lance-graph-contract/` (no new types needed; SigmaTier recommended for addition here per C-5)

---

## §11 Cross-Reference Index

| Reference | Section used |
|---|---|
| `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §7 | PR inventory + sequencing gates |
| `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §10 | Blast radius + files NOT touched |
| `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §11 | OQ-1 through OQ-8 definitions |
| `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §15 | Readiness checklist => §9 gating recommendations |
| `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §3 | CausalEdge64 v2 layout (C-1 accessor names) |
| `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §4 | AttentionMask struct (C-2 lifetime context) |
| `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §5 | MailboxSoA struct (C-2 BindSpaceView field) |
| `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §6 | Per-crate change inventory (§2 PR table) |
| `.claude/plans/bindspace-columns-v1.md` | W4 scope — Columns E/F/G/H Phase 2 |
| `.claude/plans/ogit-g-context-bundle-v1.md` | W5 scope — D-OGIT-G-1 SPO-G u32 slot |
| `.claude/plans/oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` | W5 scope — SPOW §1-§8 |
| `.claude/knowledge/linguistic-epiphanies-2026-04-19.md` E21 | Sigma10 Rubicon doctrine (W7 SigmaTierRouter grounding) |
| `.claude/board/sprint-log-10/MANIFEST.md` | Fleet overview, per-worker scope assignment |

---

*End of spec. This is a meta-spec — it sequences code-authoring PRs; all architecture authority remains in the parent plan and the individual worker specs W1-W9.*
