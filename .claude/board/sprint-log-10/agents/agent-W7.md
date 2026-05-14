# agent-W7 scratchpad — sprint-log-10 CCA2A

## Identity
- Worker: W7 (sigma-tier-router)
- Role: SigmaTierRouter + banding policy + plasticity feedback + 3-trigger pruning + JIT KernelHandle pipeline + Σ9-Σ10 escalation
- Output: `.claude/specs/pr-ce64-mb-6-sigma-tier-router.md` (~48 KB, 15 sections)
- Date: 2026-05-14
- Authored from main thread (per CCA2A pattern — W7 was not spawned in original fan-out; W9 written first then W7 to complete sprint-log-10 fleet)

## Mandatory reads completed
1. `.claude/board/sprint-log-10/MANIFEST.md` — W7 row + scope + composes
2. `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §6 (mechanism) + §7 PR-CE64-MB-6 (LOC + scope) + §10 (Zone-2 topology) + §11 OQ-1/OQ-3/OQ-4 (gating ratifications) + E-CE64-MB-8/9/10
3. `.claude/specs/pr-ce64-mb-1-par-tile-crate.md` (W1) — Mailbox<T> trait + 3 backings (TokioMailbox/InMemoryMailbox/SupabaseSubMailbox) + AttentionMaskActor surface
4. `.claude/specs/pr-ce64-mb-5-mailbox-soa-attentionmask.md` (W6) — MailboxSoA<N>::push_row + dispatch_cycle + drop_row signatures + CompartmentReport return type (CRITICAL cross-spec touchpoint: g_slot_at_drop missing)
5. `.claude/specs/pr-ce64-mb-2-causaledge64-v2.md` (W2) — CausalEdge64 v2 G/W/truth accessors (read via grep due to context budget)
6. `.claude/specs/pr-ce64-mb-4-arigraph-spo-g.md` (W5) — ghost reactivation event (Σ9 EPIPHANY → AriGraph commit path; via grep)
7. `.claude/specs/sprint-10-pr-dep-graph.md` (W10) — Wave 5 placement; identified OQ-4 gap (INT4-32D not listed as hard dep) — flagged W7-OQ-4
8. `.claude/specs/sprint-10-test-plan.md` (W11) — §3 PR-CE64-MB-6 row + §8 bench targets
9. `.claude/board/AGENT_LOG.md` — existing entries reviewed (W5, W6, W8, W10, W12 + W9 prior); no prior W7 entry

## Spec decisions

### Decision 1 — Policy/mechanism separation (router does not own MailboxSoA)
The router queues prune drops; the dispatch-cycle owner (W6 MailboxSoA) executes them. Branch-light dispatch hot path. Aligns with parent plan §6 "single point of policy" framing.

### Decision 2 — 10-tier numeric Σ-band table + 4-variant coarse enum (W6)
W6 ships 4-variant SigmaTier; W7 needs 10-tier resolution for banding (Σ4 vs Σ5 differ in OQ-1). Solution: `SpawnArgs.sigma_fine: u8 (1..=10)` carries fine grain; `Banding::validate_compat` enforces coarse/fine consistency. **No re-derivation of W6 enum.**

### Decision 3 — Default banding = conservative (all Σ4-Σ5 → Tokio)
Pre-OQ-1-ratification default. Promoting Σ4-Σ5 to cycle-speed via `Banding::alternative_fast_reflex()` is the runtime override path triggered by `SetBanding` msg. Default is safe-to-ship; ratification can only PROMOTE, never demote — so OQ-1 unratified does not block sprint-11 Wave 5 (only OQ-2 cross-spec touchpoint does).

### Decision 4 — KernelHandleCache lives in router, not planner
Closes JIT pipeline Gap 3. Cache is router-state — populated on first-spawn-for-style. Lookup → JitCompiler trait via trait object (avoids hard supervisor→planner build dep cycle). Hit rate target: > 95% after 1K spawns over 10 distinct styles.

### Decision 5 — Σ9 vs Σ10 split via Pearl rung
Pearl rung < 7 → `EpiphanyWitness` (commit to AriGraph SPO-G); Pearl rung ≥ 7 → `RubiconWitness` (AriGraph + Wire DTO egress). Both flow through CallcenterSupervisor parent actor (Zone-2). Adds 2 msg variants + handlers to existing PR #366 supervisor (~15 LOC patch).

### Decision 6 — Plasticity Hebbian rollup at drop_row, NOT mid-cycle
Per E-CE64-MB-10: bit-counter increments in dispatch_cycle (W6), router absorbs via CompartmentReport on drop. `SpawnPriorBias::refresh_from(aggregator)` recomputes top-K bias every refresh epoch. Hand-tuned threshold (1.5) flagged W7-OQ-3 (Jirak-derivation deferred to sprint-12+).

### Decision 7 — Escalation backpressure via 1024-entry bounded queue
Σ9-Σ10 flood load-shedding: oldest dropped + TECH_DEBT entry appended. Better than locking the cycle-speed path.

### Decision 8 — Cross-spec touchpoint with W6: `CompartmentReport.g_slot_at_drop`
W6 spec §4.2 returns CompartmentReport WITHOUT g_slot_at_drop field. W7 plasticity aggregator NEEDS this field to key (role, G) pairs correctly. **W7-OQ-2 escalates this to meta-review.** ~3 LOC patch on W6's CompartmentReport struct definition. Without it, plasticity aggregator can only key on role, losing G-slot granularity.

## Test count = 30
- §10.1 Banding: 10
- §10.2 Cold-start K-NN: 5
- §10.3 Pruning: 4
- §10.4 Escalation: 3
- §10.5 Plasticity: 5
- §10.6 KernelHandle: 3
Matches sprint-10-test-plan.md §3 PR-CE64-MB-6 row target.

## Key delta vs parent plan §7
Parent plan §7 names PR-CE64-MB-6 in a single one-line row with LOC + composes. This spec resolves:
- SigmaTierRouter struct + 6 message variants + per-tier state
- 10-tier numeric banding (resolves W6's 4-variant SigmaTier ambiguity for OQ-1)
- Cold-start INT4-32D K-NN fallback (RESOLVES parent plan OQ-4)
- 3-trigger pruning policy with queue-then-drain pattern (branch-light hot path)
- Plasticity Hebbian rollup → spawn-prior bias (CLOSES E-CE64-MB-10)
- KernelHandleCache (CLOSES THINKING_ORCHESTRATION_WIRING.md Gap 3)
- Σ9-Σ10 escalation via supervisor msg + 1024-entry backpressure queue
- 30 tests, 4 criterion benches, 1 CI job
- Risk matrix (9 risks) + 6 OQs + files-to-touch (15 files)

## Open questions surfaced (for meta-review)
1. **W7-OQ-1 (= parent plan OQ-1):** Σ4-Σ5 banding ratification — HIGH/BLOCKS Wave 5; conservative default ships safe
2. **W7-OQ-2 (cross-spec critical):** W6 CompartmentReport missing g_slot_at_drop field — HIGH/BLOCKS plasticity correctness; meta-review must enforce ~3 LOC W6 patch
3. **W7-OQ-3:** Hand-tuned 1.5 spawn-prior bias threshold not Jirak-derived — non-blocking TECH_DEBT
4. **W7-OQ-4 (W10 gap):** INT4-32D codebook dep not in sprint-10-pr-dep-graph.md as hard prerequisite — HIGH/BLOCKS Wave 5; meta-review patches W10 graph
5. **W7-OQ-5:** supervisor→planner build dep cycle risk for JitCompiler — non-blocking (trait object fallback)
6. **W7-OQ-6 (= parent plan OQ-3):** plasticity granularity bit-counter + AriGraph NARS revise — coupled to OQ-3 ratification (W6, not W7 direct)

## Status: COMPLETE — spec at `.claude/specs/pr-ce64-mb-6-sigma-tier-router.md` (~48 KB)

## Sprint-log-10 agents/ folder NOW complete: W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11, W12 (12/12)
