# V3 Substrate Integration Plan — waves W0–W6

> **Status:** ACTIVE (W0 ships with this PR; W1–W6 sequenced below)
> **Date:** 2026-07-02 · **Authority:** operator rulings of 2026-07-02
> (board entries E-MAILBOX-KANBAN-NO-COLLAPSEGATE, E-COMPILED-THINKING-TEMPLATES,
> E-DTO-LADDER-OWNERSHIP-SPLIT, E-TWO-RESONANCES-SPLIT, E-V3-MARKER-IS-A-MONITOR,
> E-CLASSID-* arc) under standing full authorization.
> **Index entry:** `.claude/board/INTEGRATION_PLANS.md` (prepended same commit).
> **Deliverable dashboard:** STATUS_BOARD rows `D-V3-*` + the existing
> D-MBX-A6 / D-PERT-1 / D-CC-* / D-VCW-* / D-CCF-4 rows this plan adopts.

The organizing principle is unchanged from v3-convergence-wiring: **wire,
don't invent** — every wave below wires machinery that already exists
(envelope, kanban types, DSL triple, graph-flow, classid helpers) into the
ruled model. Anything that looks like a new layer is a defect in this plan.

---

## Wave map (dependency order)

```
W0 ratify/document ──► W1 envelope+ownership ──► W2 kanban executors ──► W5 consumers
                              │                        │
                              └──► W4 DTO ladder       └──► W3 templates ──► (catalogue ⤳ W6/P4)
                                                                 W6 monitor & retirement (P4 = operator checkpoint)
```

Rule of thumb: **W1 is the keystone** — the cast pairing
(`cast(on_behalf = envelope.mailbox_owner(), payload = BusDto)`) is what
every later wave consumes. W4 can run any time (rename + doc). W6's scanner
can be built early; its retirements only fire on proof.

---

## W0 — Ratify & document (THIS PR)

| D-id | Deliverable | State |
|---|---|---|
| D-V3-W0a | `.claude/v3/` tree (README, this plan, COMPONENT-MAP, ENTROPY-MILESTONES, MODULE-TABLE, soa_layout/*) | this commit |
| D-V3-W0b | V3 awareness layer: knowledge docs, 4 agent cards (`v3-*`), `/v3` skill, `/v3-audit` command, CLAUDE.md + BOOT.md entrypoints | this commit |

Gate: board hygiene same-commit (INTEGRATION_PLANS prepend, STATUS_BOARD
rows, AGENT_LOG, handover pointer).

## W1 — Envelope & ownership (the keystone)

| D-id | Deliverable | Notes |
|---|---|---|
| D-V3-W1a | `SoaEnvelope::mailbox_owner()` stamp | **SHIPPED** (this branch) — default 0 = bootstrap, zero-fallback ladder |
| D-V3-W1b | **Batch writer**: `cast(on_behalf, payload)` pairing + AHEAD kanban update fired at cast (never at write-ack) | new module, planner-adjacent; the ONLY sanctioned online write shape |
| D-V3-W1c | **Delegation cache**: cast id vs envelope stamp; hit = proceed, miss = resolve once + cache | lives in the batch writer; consumers never re-implement |
| D-V3-W1d | MailboxId minting path (non-zero owners; `debug_assert` uniqueness in default basin) | per CANON zero-fallback ladder |
| D-V3-W1e | Probe: ahead-update ordering test (kanban board reflects intent BEFORE Lance ack) + delegation-miss test | probe-first; W2 consumes |

Gates: `v3-envelope-auditor` LAYOUT-CLEAN; `v3-mailbox-warden` OWNED on the
writer's own paths; no serialization on the hot path (zero-copy invariant).

## W2 — Kanban executors (two arms + structural owner)

| D-id | Deliverable | Notes |
|---|---|---|
| D-MBX-A6 | `Outcome → KanbanMove` adapter emit in `lance-graph-planner strategy/style_strategy.rs` (arm #1) | adopts existing STATUS_BOARD row; converged/cycle_count from BusDto feed it |
| D-V3-W2a | **Per-mailbox kanban board as TENANT** (type + lane; sibling of per-row `KanbanTenant`) | envelope-auditor gate: field-isolation matrix |
| D-V3-W2b | Supervisor wiring: `kanban_actor.rs` applies moves via `MailboxSoaOwner::advance_phase` (sole mutator); ractor stays spawn-only | structural-owner proof unchanged |
| D-V3-W2c | symbiont arm (SurrealDB-on-kv-lance): kanban updates as KV transactions | **BLOCKED(C)**: AdaWorldAPI surrealdb fork `kv-lance` coordinates; POC = `symbiont/kanban_loop.rs` |
| D-V3-W2d | 550 ms budget: load-balancing hooks into planner `elevation/` (extend, don't shadow) + budget instrumentation | 64k–256k SoA prioritization |

Gate: W1b/W1c shipped (moves fire off casts). Standing rule: updates
reprioritize, never gate — a missing update must not deadlock a cycle.

## W3 — Compiled templates

| D-id | Deliverable | Notes |
|---|---|---|
| D-V3-W3a | `StepMask` in contract (sibling of FieldMask; selection, not control flow) | zero-dep |
| D-V3-W3b | ElixirTemplate → graph-flow `GraphBuilder` adapter; Session threads the MailboxId (ownership inheritance) | rs-graph-llm seam; Iron-Rule-5-style contract pull |
| D-V3-W3c | Rig oracle node: FailureTicket → oracle run → `template-equivalence` gate → `cognitive-compiler` compile-down into catalogue | D-VCW-7 lineage; deterministic-first |
| D-V3-W3d | Catalogue keyed INTERNALLY (template-id); classid custom-half keying deferred to W6/P4 | styles-as-lenses F2 lands here post-P4 |

Gate: replay equivalence green on every template change (template-smith rule).

## W4 — DTO ladder (parallel, any time)

| D-id | Deliverable | Notes |
|---|---|---|
| D-PERT-1 | Rename `dto.rs::ResonanceDto` → `PerturbationDto` (~9 files thinking-engine + cognitive-shader-driver engine_bridge), deprecated alias | adopts existing row; "cascade" key-tier vocabulary NOT renamed (canon) |
| D-V3-W4a | BusDto/cast pairing call sites in cognitive-shader-driver (consumes W1b) | BusDto never grows ownership fields — warden check 2 |
| D-V3-W4b | L4 learning loop doc-to-code check: converged residue → owner-stamped tenant lane → template reads row next cycle | probe over an end-to-end cycle |

## W5 — Consumer adoption (write-on-behalf fleet-wide)

| D-id | Deliverable | Notes |
|---|---|---|
| D-V3-W5a | q2: CI re-bakes (`osint-bake`, `fma`) + `body.soa` re-release; drop `FMA_V3_CLASSID_LEGACY` from BodyV3.tsx | handover continuation §1 |
| D-V3-W5b | q2 cpic contract pull with mereology (kinds → cascade positions under `0x0E01_1000`) — dissolves interim `0x0E01_000N` + `ISS-Q2-CPIC-MIRROR` | handover F3 |
| D-V3-W5c | Bake pipelines annotated bootstrap-owner; NEW online consumer writes route through the batch writer | per write-on-behalf.md interim rules |
| D-V3-W5d | Probes D-VCW-3 (P7 render, bitmask→askama) + D-VCW-5 (cascade3 nibble falsifier) — q2 gate already WAIVED | validate V3 keys end-to-end |
| D-V3-W5e | Stragglers: ladybug-rs (pre-V3/rustynum-era) + smb-office-rs contract pulls only | never bridges |

## W6 — Monitor & retirement (proof-gated)

| D-id | Deliverable | Notes |
|---|---|---|
| D-V3-W6a | **Adoption/corpus scanner**: ONE two-metric range-count tool (canon-high adoption % + old-form row count) over Lance datasets | canon-high = clustered index → both metrics are key-range counts |
| D-CCF-4 | `0x1000` marker retirement (P4) — trigger DEFINED: adoption reads 100% | **operator checkpoint** |
| D-V3-W6b | Legacy alias retirement (`CLASSID_*_LEGACY`, compat reader narrowing) | corpus proof = zero old-form rows; never before |
| D-V3-W6c | Custom half opens: 64k ClassView render catalogue + template catalogue dispatch (completes W3d/F2) | post-P4 only |

---

## Standing gates (every wave)

1. **Probe-first** — mechanism lands after its failing probe (workspace rule).
2. **Warden + auditor green** on any write-path / layout diff (`/v3-audit` is the cheap pre-check).
3. **Board hygiene same-commit** (STATUS_BOARD row transitions, EPIPHANIES for findings).
4. **Model economy** — Sonnet 5 grindwork, Fable/Opus decisions (operator ruling 2026-07-02). Every Sonnet brief carries the §1 preamble of `knowledge/sonnet-worker-guardrails.md` verbatim; §5 escalation triggers are STOP+report, never worker-resolved.
5. **Wire, don't invent** — a new struct/trait/layer proposal must first fail the "existing machinery" search (COMPONENT-MAP is that search, precomputed).

## What this plan supersedes / adopts

- Adopts (does not duplicate): D-MBX-A6, D-PERT-1, D-CC-RUNTIME/EQUIV/COMPILER rows, D-VCW-3/5/7, D-CCF-4.
- Extends: `v3-convergence-wiring-v1` (its seam list is W1–W3's ancestry) and `soa-value-tenant-migration-v2` (Phase-2 tenant shaping proceeds under W2a's tenant discipline).
- Supersedes in prose only: any remaining CollapseGate-as-singleton framing in older docs (primer §6 table governs).
