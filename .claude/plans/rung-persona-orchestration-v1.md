# rung-persona-orchestration-v1 — time-bound persona orchestration: boring checklist → meta-recipe → hot/cold/feedback anneal

**Status:** PROPOSAL (sibling to `rung-mul-grounding-v1`; the time-bound + composition layer over it)
**Date:** 2026-05-26
**Confidence:** HIGH on structure (the hot/cold/feedback loop is the original ladybug architecture + the OpenAI macro-eval pipeline + ADK Memory Bank, all converging); MED on ractor adoption + macro-eval harness scope (net-new builds).
**Predecessors:** `rung-mul-grounding-v1` (b4efb55), `rung-ladder-grounding-v1` (b0ef6fa), `cognitive-substrate-convergence-v1` (hot/cold split, CausalEdge64 baton).
**Design refs (read-only, general-web — ladybug-rs is outside GitHub-MCP scope):** ladybug-rs `INTEGRATION_PLAN.md` @177a321 — §"BF16 Superposition Architecture (Hot/Cold/Feedback)" (lines 542+), the 4-phase `[DONE]/[TODO]` gate checklist, the 3 composition modes, BindSpace-as-blackboard hub; ladybug-rs `src/spectroscopy/detector.rs` @177a321 (RungLevel/StyleProfile classify, `fanout = base·(1+bridgeness·0.5)`, `noise_tolerance = base·(1+(1−confidence)·0.5)`). External: Claude chief-of-staff (Task delegation), OpenAI macro-evals (offline distillation + suspect-bridge), Google ADK (Memory Bank Preload/Load).

---

## 0. Grounding stance — restore-on-SoA, not port (per E-AGICHAT-DIMENSION-CONTRACT)

ladybug's INTEGRATION_PLAN is the **original**. We ground its hot/cold/feedback loop, phase-gate checklist, and blackboard composition onto **our** contract types + SoA floor — not by copying its crewai/n8n/BindSpace stack. Where ladybug diverges from what we need (no dead-end/null handling; raw lived-history vs distilled wisdom), we say so explicitly (§8).

## 1. Two orthogonal orderings, arbitrated by the time budget

- **Axis A — epistemic** (the `rung-mul` experience curve / DK): where am I in *evidence*? → which work/sparse-route move fires.
- **Axis B — social** (persona etiquette arc): where am I in the *conversation*? → greeting…empathy…curiosity…reflection…Coda.
- **`budget.latency_budget` = the arbiter** (`elevation/mod.rs:131`, `trigger.elapsed > budget.latency_budget`): schedules Axis B; within each phase runs Axis A's work *anytime*, priority-first over the 2³×heads×styles space, truncating at the deadline.

The menu is a **2D grid** — phase × DK-position → strategy. Etiquette = soft transition prior (not a rigid FSM); free-energy/evidence can preempt it. Etiquette **is** the anytime graceful-degradation contract: out of time mid-curiosity → still close politely (Coda).

## 2. The boring checklist (verify — temp≈0)

Flat, deterministic, each item green/red. Grounds ladybug's per-phase `[DONE]/[TODO]` gate. Split into **hard** (must be green to boot) vs **soft** (degrade gracefully if red — anytime):

```
HARD (boot gate):
[ ] contract types load + Pod sizes exact   (RungState=16B, SpoHead, MulAssessment)
[ ] SoA floor up                            (SoaColumns, i4-32 unpack)
[ ] operational store reachable             (Lance / SQLite — NOT surreal; §6)
[ ] NARS tables loaded                      (NarsTables lookup hot)
[ ] thresholds loaded                       (MUL profile, SD_FLOW/BLOCK, rung thresholds)
[ ] free-energy path wired                  (FreeEnergy::compose available)

SOFT (degrade if red):
[ ] each capability registered              (ExpertCapability / actor / MCP — route around if down)
[ ] wisdom-marker store hydratable          (cold start → foot of curve, NOT Mount Stupid)
[ ] macro-eval harness present              (run without offline updates if absent)
```

**Continuous, not one-time.** A green item going red at runtime (actor crash, capability degrade, evidence→NaN) is a **let-it-crash** event → supervisor restart / escalation = our `rung-shift` on sustained-block + NaN→cautious-exploration→Lab. The checklist items **are** the supervision health-checks.

## 3. The meta-recipe (compose — cold)

A declarative child-spec **manifest** (data, not code — AGI-as-SoA: recipe = column config), consumed by the supervisor:

```
recipe = with green(store), green(contracts), green(caps):
            supervise [ store, capability_actors…, orchestrator, eval ]
              strategy = one_for_one / rest_for_one
              → compose phase-ordered UnifiedSteps → run
         else: degrade / escalate
```

**Composition = blackboard, not direct calls** — ladybug composes via BindSpace (services read/write addressable memory, coherence via CollapseGate). We keep that on **`a2a_blackboard` / SoA columns**; `ractor` is the supervised outer-swarm runtime carrying **Batons** between specialist actors (§6). Recipe-as-data ⇒ the macro-eval pipeline can score *which recipes* produced *which outcomes* (recipe = a `behavior_pattern` unit).

## 4. Hot / cold / feedback (grounding ladybug §542)

| Loop | ladybug original | our grounding |
|---|---|---|
| **Hot** (µs, online) | `superposition_decompose()` → 4-state {Crystallized, Tensioned, Uncertain, Noise} | the cognitive cycle, temperature-annealed; per-projection NARS state + gate (D-RUNG-MUL-1) |
| **Cold** (offline, Lance) | `CrystalCodebook` — *"a lived history, not a trained model"*; 125-cell learned centroids | the **macro-eval pipeline = the wisdom-marker factory**; clustered DK-patterns; *"3 dims that agree ARE the address"* ≈ the screening-off-identified causal triple |
| **Feedback** (lazy) | codebook biases which dims weighted next | **hydrate-before-the-fact** (ADK `PreloadMemoryTool`) = wisdom-marker as the KL prior |

The cold loop is where ladybug's "lived history" becomes our **calibrated** wisdom marker — the distillation (§8) is the difference.

## 5. Temperature anneal (the squeeze)

Explore **hot** (high-temp fanout, Staunen-driven, NaN→cautious-exploration), exploit **cold** (low-temp calibrated commit). **Evidence-gated, not time-gated** — cool too early = premature convergence = Mount Stupid. `temperature ~ 1/calibrated_confidence`; **cool only as fast as the Boole-bound lets confidence rise** (`conf(SPO) ≤ weakest link`). Free-energy descent IS temperature-annealed variational inference.

Grounded in `detector.rs`: `noise_tolerance = base·(1 + (1−confidence)·0.5)` (low confidence = hotter); `fanout = base·(1 + bridgeness·0.5)`, clamp [1,30] (**bridgeness drives fanout = the macro-eval suspect-bridge centrality = our work-metric** — triple convergence); rung-shift on `emergence>0.5 ∧ coherence<0.4` (+1) / `coherence>0.8 ∧ emergence<0.1` (−1).

Cold scaffold (§2+§3) runs at temp≈0; cognition runs hot on top; the experience curve anneals between them.

## 6. Runtime substrate decision

- **ractor — YES, scoped to the outer swarm.** Supervised specialist actors under `OrchestrationBridge`; ractor messages carry Batons; async boundary at the swarm layer only; the SoA Click stays inner + sync. Don't double-mailbox with the existing mailbox-as-owner (E-BATON-1).
- **surrealdb — NO for the cognitive store** (redundant with lance-graph/AriGraph + Lance; introduces a second graph + second truth; not actually "boring stable"). **Open for the operational trace/session store only** — but prefer **SQLite/Lance** there too. **AriGraph stays the one graph.**
- **Composition = blackboard** (`a2a_blackboard`/SoA), per ladybug's BindSpace choice; ractor supervises, blackboard composes.

## 7. Deliverables

| D-id | title | crate | ~LOC | risk |
|---|---|---|---|---|
| D-PERSONA-1 | hard/soft checklist verifier (contract types + readiness probes; continuous health) | contract + planner | 180 | LOW |
| D-PERSONA-2 | meta-recipe manifest (declarative child-spec, recipe-as-data, macro-evaluable) | contract | 150 | MED |
| D-PERSONA-3 | hot/cold/feedback wiring — anneal + `CrystalCodebook`→wisdom-marker cold path + Preload hydrate | planner + Lance | 240 | MED |
| D-PERSONA-4 | macro-eval harness (scenario→trace→discover→diagnose; suspect-bridge = blasgraph betweenness; 5 rubrics from D-RUNG-MUL) | planner + Lance | 280 | HIGH |
| D-PERSONA-5 | ractor outer-swarm runtime under `OrchestrationBridge` (batons as messages, async only at boundary) | planner | 200 | MED |

## 8. Honest gaps vs the original

- **ladybug `detector.rs` has no null/dead-end/escalation** — *"all classifications produce valid output; no dead-end states."* Our **NaN→cautious-exploration→Lab + dead-end-as-work is net-new**, not a restore.
- **`CrystalCodebook` = ladybug's cold path, but it dumps "lived history."** We reframe it as **macro-eval-distilled** wisdom markers (a calibrated prior + the Boole-bound, ≤32 identities per I-VSA-IDENTITIES) — *not* a raw history accumulation. The distillation is the contribution (same gap as ADK's whole-transcript Memory Bank).
- **ractor + the persona etiquette arc are not in the original** (ladybug uses crewai/n8n + BindSpace; etiquette/anytime is new).

## 9. Invariants honored

restore-on-SoA not port · hard-gate vs soft graceful-degradation (anytime/etiquette) · recipe-as-data (AGI-as-SoA, macro-evaluable) · evidence-gated anneal — Boole-bound caps cooling rate (no premature Mount-Stupid) · blackboard composition (`a2a_blackboard`/SoA, not direct calls) · ractor async only at the swarm boundary, SoA inner sync · no second graph (AriGraph is the one graph) · I-VSA-IDENTITIES (wisdom markers ≤32 identities, content in store) · `latency_budget` is the time arbiter (no wall-clock in the hot Pod).

## 10. Cross-refs

`rung-mul-grounding-v1` (the experience curve this schedules), `rung-ladder-grounding-v1` (the integer rung the supervisor restarts on); `elevation/{mod,homeostasis,learning}.rs` (latency_budget, MUL-L6); `a2a_blackboard` + `OrchestrationBridge` (Layer-1 composition); `FreeEnergy::compose`; design refs ladybug-rs `INTEGRATION_PLAN.md` §542 + `detector.rs` @177a321, Claude chief-of-staff, OpenAI macro-evals, ADK Memory Bank. Iron rules: E-BATON-1 (mailbox-as-owner), E-AGICHAT-DIMENSION-CONTRACT (restore-on-SoA), I-VSA-IDENTITIES, data-flow (no `&mut` during compute).
