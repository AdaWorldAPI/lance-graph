# Supersession index — GENERATED, do not edit

> Regenerate: `python3 .claude/tools/supersession_index.py > .claude/board/SUPERSESSION-INDEX.md`
>
> This table is derived from `COMPONENT-MAP.md` + `crates/` + `plans/` on every run,
> so it cannot disagree with the repo. A hand-kept supersession table goes stale at
> the next rename, and a stale one is worse than none: it authorises work against a
> symbol that is already retired.
>
> The commentary below is generated too: every measurement in it is interpolated from
> the same values as the tables, so prose and table cannot disagree. Methods are
> literal; measurements never are.

## What this table says

**`BindSpace` is the shape of the problem.** Marked RETIRE, and simultaneously the
most-referenced symbol here: **68 crate files, 47 plans, 42 of them blind.**
That is a programme, not a cleanup.

**`GateState` is the sharpest case: 1 plan names it and *every one* is blind.**
Its COMPONENT-MAP note reads: intra-cascade SD gate — fine IF intra-mailbox; warden sign-off queued, not assumed

**`ResonanceDto` → `PerturbationDto` gives the rule that needs no map at all:**
2 crate files against 12 plans. The code moved; the plans did not.
**Plan-mentions exceeding crate-mentions is a staleness signal on its own.**

### The limit of the mechanical route

`RESCOPE` fires on "names a RETIRE symbol" alone. That is a *reason to look*, not a
verdict: a plan mentioning the symbol once in background and a plan built around
retiring it land in the same bucket. Separating them needs a read; this table's job
is to make that read finite and ordered, not to replace it.

### What cannot be found this way

A rename that left no trace. `Blumenstrauss -> cognitive-shader-driver SoA` has zero
hits in `.claude/`, `crates/`, plans, or board, so nothing connects prior reasoning to
the current name and no script recovers what was never written down. An alias enters
this table only if someone records it in COMPONENT-MAP -- the method's single
dependency and single failure mode.

### Deliberately not a signal

`git log` dates on plans. 2026-07-24 is a bulk import -- one merge, 2,718 files -- so
git dates the import, not the work. Routing uses self-declared status and board
coverage instead.

## Table 1 — ruled symbols: verdict, successor, and where each side actually lives

| symbol | verdict | successor | live in crates | named in plans | blind plans |
|---|---|---|---|---|---|
| `A2AMessage` | BLOCKED | — | 2 | 1 | 0 |
| `StepMask` | BLOCKED | — | 3 | 9 | 4 |
| `commit_to_l4` | BLOCKED | — | 2 | 2 | 0 |
| `dispatch_busdto` | BLOCKED | — | 3 | 7 | 5 |
| `persist_cycle` | BLOCKED | — | 10 | 7 | 5 |
| `CognitiveMarkers` | REPURPOSE | `Commit` | 1 | 0 | 0 |
| `DominoCascade` | REPURPOSE | `Commit` | 7 | 1 | 0 |
| `GateDecision` | REPURPOSE | — | 25 | 27 | 24 |
| `GateState` | REPURPOSE | — | 14 | 1 | 1 |
| `MergeMode` | REPURPOSE | — | 8 | 13 | 12 |
| `ResonanceDto` | REPURPOSE | `PerturbationDto` | 2 | 12 | 7 |
| `BindSpace` | RETIRE | — | 68 | 47 | 42 |
| `CollapseGateEmission` | RETIRE | — | 5 | 13 | 12 |
| `ThinkingStyle` | RETIRE-toward-contract | — | 51 | 28 | 24 |

## Table 2 — plans naming a ruled symbol without citing the ruling (74)

Route is **mechanical triage, not a verdict**: `ARCHIVE?` = the plan's own status says
it shipped; `RESCOPE` = it targets a symbol marked RETIRE; `READ` = neither signal fires
and a human read decides. Board coverage counts this plan's D-ids cited on the board.

`ARCHIVE?` reads the status's LEADING token only. The first batch it produced was
**3/3 false positives** -- an unanchored match on a word that did not predicate the
plan (`ACTIVE ... Phase A/B COMPLETE`, `PROPOSAL. v1 SHIPPED in PR #420`, and a
`Status legend:` defining the tick mark). All three were live, and archiving them
would have retired work in flight. A route here is a prompt to read the plan, never
a licence to act on it.

| route | plan | ruled symbols named | self-declared status | board coverage |
|---|---|---|---|---|
| **READ** | `alpha-reason-witness-cognitive-fabric-v1` | `ResonanceDto` | PROPOSED / PLAN ONLY. No production implemen | 2/9 |
| **READ** | `archetype-scaffold-v1` | `GateDecision` | In progress (2026-04-24) | 0/0 |
| **READ** | `capstone-cognitive-loop-wiring-nan-census-v1` | `GateDecision` | PROPOSED (2026-06-20). The measurement compa | 0/0 |
| **READ** | `deepnsm-v3-convergence-v1` | `StepMask` | PROPOSED (doc-only). Extends `v3-convergence | 5/5 |
| **READ** | `epistemic-quadrant-materialization-v1` | `MergeMode` | PROPOSED.** Operator direction 2026-07-29: * | 4/4 |
| **READ** | `graphrag-doc-retrieval-soa-integration-v1` | `GateDecision` | DESIGN + FIRST CODE. **v1.2 (2026-07-17):**  | 7/10 |
| **READ** | `integration-actionhandler-rbac-orchestration-v1` | `GateDecision` | HARDENING (5+3 in progress). | 0/0 |
| **READ** | `mask-algebra-revision-read-v1` | `StepMask` | DRAFT, awaiting operator ruling on §5 | 2/3 |
| **READ** | `mul-calibration-not-verdict-v1` | `GateDecision` | PROPOSAL (unbuilt) — 2026-08-26. PLAN/BOARD  | 6/12 |
| **READ** | `mul-consumer-build-gate-v1` | `GateDecision` | GATE RUN — 2026-08-27. Discharges D-MCAL-6 a | 3/7 |
| **READ** | `mul-consumer-census-v1` | `GateDecision` | MEASUREMENT COMPLETE — 2026-08-27. Measureme | 1/2 |
| **READ** | `mul-ewa-trust-propagation-v1` | `GateDecision` | PROPOSED — PLAN/BOARD ONLY. Measure-before-c | 1/3 |
| **READ** | `persistence-artifact-backed-commit-v1` | `persist_cycle` | RATIFIED (operator ruling 2026-08-09). Phase | 0/0 |
| **READ** | `post-teardown-buildup-survey-v1` | `StepMask` | SURVEY, read-only, plan-only (no code, no te | 5/5 |
| **READ** | `r2il-bpe-typed-genetic-recombination-v1` | `GateDecision` | PROPOSAL, §7's three falsifiers now RUN (see | 1/1 |
| **READ** | `scientific-kg-substrate-v1` | `GateDecision` | PROPOSED — **scoping doc**, no code. Records | 8/9 |
| **READ** | `self-reasoning-substrate-v1` | `GateDecision` | PROPOSED — doc-only. No code, no contract ch | 5/5 |
| **READ** | `v3-convergence-wiring-v1` | `GateDecision` | ACTIVE (2026-07-01). Operator: "I'm all in f | 0/0 |
| **RESCOPE** | `soa-migration-diff-resolution-2026-06-13` | `BindSpace`, `CollapseGateEmission`, `GateDecision`, `MergeMode` … | — | 3/5 |
| **RESCOPE** | `cognitive-substrate-convergence-v1` | `BindSpace`, `CollapseGateEmission`, `GateDecision`, `MergeMode` … | PROPOSAL (sprint-10 architectural decisions  | 7/13 |
| **RESCOPE** | `cognitive-substrate-convergence-v2` | `BindSpace`, `CollapseGateEmission`, `GateDecision`, `MergeMode` … | ACTIVE — sprint-11 Phase A/B COMPLETE (pendi | 8/15 |
| **RESCOPE** | `bindspace-singleton-to-mailbox-soa-v1` | `BindSpace`, `CollapseGateEmission`, `ResonanceDto`, `ThinkingStyle` | CONJECTURE / design (migration spec). NOT ye | 18/19 |
| **RESCOPE** | `callcenter-membrane-v1` | `BindSpace`, `GateDecision`, `MergeMode`, `ThinkingStyle` | Active | 0/0 |
| **RESCOPE** | `causaledge64-mailbox-rename-soa-v1` | `BindSpace`, `GateDecision`, `MergeMode`, `ThinkingStyle` | Active (draft, 2026-05-14) | 1/10 |
| **RESCOPE** | `integrated-cognitive-planner-v1` | `BindSpace`, `GateDecision`, `ResonanceDto`, `dispatch_busdto` | — | 2/3 |
| **RESCOPE** | `palantir-parity-cascade-v2` | `BindSpace`, `MergeMode`, `ResonanceDto`, `ThinkingStyle` | plan, not implementation. | 1/17 |
| **RESCOPE** | `temporal-markov-and-style-classes-v1` | `BindSpace`, `MergeMode`, `StepMask`, `ThinkingStyle` | ACTIVE (operator-ratified 2026-07-10: "other | 16/19 |
| **RESCOPE** | `unified-soa-convergence-v1` | `BindSpace`, `CollapseGateEmission`, `ResonanceDto`, `ThinkingStyle` | PROPOSAL / integration plan. Design-spec onl | 22/25 |
| **RESCOPE** | `alpha-reason-witness-shader-field-archaeology-pass-1` | `BindSpace`, `MergeMode`, `ResonanceDto` | SOURCE AUDIT / PLAN ONLY. No production wiri | 1/1 |
| **RESCOPE** | `bindspace-mailbox-soa-dependency-map-v1` | `BindSpace`, `dispatch_busdto`, `persist_cycle` | MAP / preflight. No source wired yet. Read-b | 2/2 |
| **RESCOPE** | `bindspace-mailbox-soa-w3-w4a-impl-v1` | `BindSpace`, `dispatch_busdto`, `persist_cycle` | v2 — 5-consolidation + 3-brutal-critic pass  | 1/1 |
| **RESCOPE** | `cognitive-substrate-convergence-v3` | `BindSpace`, `CollapseGateEmission`, `MergeMode` | ACTIVE — sprint-12 Wave F + Wave G complete  | 8/16 |
| **RESCOPE** | `mailbox-cycle-aware-write-contract-v1` | `BindSpace`, `dispatch_busdto`, `persist_cycle` | CONJECTURE / design. 5+3-gated before code. | 1/1 |
| **RESCOPE** | `unified-integration-v1` | `BindSpace`, `MergeMode`, `ThinkingStyle` | Active — brainstorm phase complete; delivera | 0/0 |
| **RESCOPE** | `2026-05-06-splat-osint-ingestion-v1` | `BindSpace`, `MergeMode` | Active — PR 1+2 of 6 in flight on `claude/sp | 1/7 |
| **RESCOPE** | `Palette256-3DSB-PhiSpiral-attention-integration-plan` | `BindSpace`, `CollapseGateEmission` | — | 0/0 |
| **RESCOPE** | `alpha-interventional-faithfulness-v1` | `GateDecision`, `ThinkingStyle` | PROPOSAL (measured targets, unbuilt) — 2026- | 1/7 |
| **RESCOPE** | `anatomy-realtime-v1` | `BindSpace`, `ThinkingStyle` | — | 0/1 |
| **RESCOPE** | `elegant-herding-rocket-v1` | `BindSpace`, `ThinkingStyle` | — | 0/0 |
| **RESCOPE** | `grounding-descent-cognitive-maslow-v1` | `GateDecision`, `ThinkingStyle` | PROPOSED (unbuilt; every mechanism cited exi | 1/7 |
| **RESCOPE** | `kognitionswirtschaft-v1` | `GateDecision`, `ThinkingStyle` | PROPOSED (unbuilt, unprobed). Operator-initi | 1/7 |
| **RESCOPE** | `lance-graph-ontology-v5` | `BindSpace`, `GateDecision` | Drafted (2026-05-07). Picks up where v4 (`cl | 3/16 |
| **RESCOPE** | `north-star-integration-v1` | `CollapseGateEmission`, `ThinkingStyle` | RATIFIED (council resolved + gates ratified  | 1/1 |
| **RESCOPE** | `octopus-causal-cot-audit-v1` | `GateDecision`, `ThinkingStyle` | MEASUREMENT REPORT. **No code. No new type.  | 2/9 |
| **RESCOPE** | `odoo-savant-reasoners-v2` | `BindSpace`, `CollapseGateEmission` | PROPOSAL. v1 SHIPPED in PR #420 (`D-ODOO-SAV | 2/3 |
| **RESCOPE** | `ogar-ar-shape-endgame-v1` | `GateDecision`, `ThinkingStyle` | when filed:** PLAN (pre-council). Becomes PL | 0/0 |
| **RESCOPE** | `rung-ladder-grounding-v1` | `CollapseGateEmission`, `GateState` | PROPOSAL (the most-obvious first grounding p | 0/4 |
| **RESCOPE** | `rung-persona-orchestration-v1` | `BindSpace`, `ThinkingStyle` | PROPOSAL (sibling to `rung-mul-grounding-v1` | 4/9 |
| **RESCOPE** | `streaming-arm-nars-discovery-v1` | `BindSpace`, `CollapseGateEmission` | PROPOSAL / integration plan. Spec only; **no | 11/20 |
| **RESCOPE** | `3DGS-4x4-cognitive-shader-integration-plan` | `BindSpace` | — | 0/0 |
| **RESCOPE** | `3DGS-neuronal-network-4x4-plan` | `BindSpace` | — | 0/0 |
| **RESCOPE** | `a3-carrier-v1` | `ThinkingStyle` | — | 1/1 |
| **RESCOPE** | `atom-mailbox-substrate-v1` | `ThinkingStyle` | PROPOSAL (implements `EPIPHANIES.md` E-LADDE | 7/8 |
| **RESCOPE** | `bindspace-columns-v1` | `BindSpace` | Active | 0/0 |
| **RESCOPE** | `codec-sweep-via-lab-infra-v1` | `BindSpace` | — | 0/0 |
| **RESCOPE** | `cycle-coherent-soa-snapshot-v1` | `CollapseGateEmission` | Queued | 0/6 |
| **RESCOPE** | `dacr7-band-reading-contract-v1` | `ThinkingStyle` | — | 4/5 |
| **RESCOPE** | `entropy-closure-causal-ground-v1` | `ThinkingStyle` | PROPOSAL (measured, unbuilt) — 2026-08-26. P | 3/8 |
| **RESCOPE** | `foundry-consumer-parity-v1` | `BindSpace` | Active | 0/0 |
| **RESCOPE** | `foundry-roadmap-unified-smb-medcare-v1` | `BindSpace` | Active | 0/0 |
| **RESCOPE** | `lf-integration-mapping-v1` | `BindSpace` | Active (2026-04-25) | 0/0 |
| **RESCOPE** | `lite-unified-surrealql-lance-v1` | `BindSpace` | CONJECTURE / design. **Test via feature gate | 0/0 |
| **RESCOPE** | `ogit-cascade-supabase-callcenter-v1` | `BindSpace` | plan, not implementation. | 0/16 |
| **RESCOPE** | `open-ideas-fetch-v1` | `BindSpace` | MEASURED / ready-to-execute — PLANNING ONLY  | 2/10 |
| **RESCOPE** | `q2-foundry-integration-v1` | `BindSpace` | Proposed (2026-04-24) | 0/0 |
| **RESCOPE** | `reliability-checklist-arc-v1` | `ThinkingStyle` | PROPOSAL / possibility menu (2026-05-30). NO | 3/3 |
| **RESCOPE** | `singleton-to-snapshot-nudge-v1` | `BindSpace` | PROPOSAL | 1/12 |
| **RESCOPE** | `soa-value-tenant-migration-v1` | `BindSpace` | BRIEF (2026-06-24). This is NOT the migratio | 0/0 |
| **RESCOPE** | `splat-native-ultrasound-v1` | `BindSpace` | PROPOSAL / integration plan. Design-spec onl | 3/17 |
| **RESCOPE** | `sql-spo-ontology-bridge-v1` | `BindSpace` | Active | 0/0 |
| **RESCOPE** | `super-domain-rbac-tenancy-v1` | `BindSpace` | Active | 23/41 |
| **RESCOPE** | `thought-cycle-soa-awareness-integration-v1` | `BindSpace` | integration plan. No implementation claimed  | 0/0 |
| **RESCOPE** | `unified-ogit-architecture-v1` | `ThinkingStyle` | — | 0/0 |
| **RESCOPE** | `unified-soa-rubikon-integration-v1` | `BindSpace` | — | 8/8 |

- **ARCHIVE?**: 0
- **RESCOPE**: 56
- **READ**: 18
- ruled symbols tracked: 14
