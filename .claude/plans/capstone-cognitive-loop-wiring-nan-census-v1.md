# Capstone Validation Plan — Phase-Aligned Cognitive Loop: Wiring & NaN-Census (v1)

> **Status:** PROPOSED (2026-06-20). The measurement companion to the kanban×Rubicon
> tenant arc. **Not a build plan — a measurement plan.** Per the workspace
> measurement-before-synthesis iron rule (truth-architect) + probe-first protocol:
> every seam below is a **CONJECTURE until its probe runs green**. The deliverable
> per seam is the probe + the number, never more prose.
> **Operator framing (2026-06-20):** "99% is there unused, 28% wiring gaps
> resulting in 72% NaN" — treated here as THREE orthogonal measured quantities,
> not estimates to assert.

---

## 0 — Why this exists (the anti-synthesis-spiral guard)

Two Opus mapping passes (AGENT_LOG 2026-06-20) established the qualitative truth:
the substrate has **all the pieces** of a phase-aligned, per-SoA-owned, version-arc-
scheduled cognitive loop (`kanban` Rubicon phases, `MailboxSoaOwner`/`View` split,
`VersionScheduler`/`NextPhaseScheduler`, `CycleAccumulator`, the `Meta`/`Qualia`/
`Plasticity` tenants, `mul.rs` DK/Trust/Flow/Gate, `OrchestrationBridge`/`BridgeSlot`,
surrealdb `timeline.rs`), **but the runtime loop is not closed** — `UnifiedStep`
carries no SoA pointer, the batch writer pushes nothing, the timeline is unwired
(`#[allow(dead_code)]`), kanban is not yet a value tenant, and no pipelining exists.

The risk this plan removes: shipping the kanban tenant + calling the loop "done"
by assertion. Instead we **measure** the three quantities below and drive run-NaN
toward 0 seam by seam.

## 1 — The three measured quantities (the dashboard)

| Metric | Definition | How measured | Baseline (Wave 0) |
|---|---|---|---|
| **Piece-presence %** | of the N named loop components, how many TYPES exist | static: grep the contract + surrealdb for each named type | ~99% (claim — Wave-0 confirms) |
| **Seam-wiring %** | of the M seams (S1..S7), how many are CONNECTED (caller→callee actually invoked in a non-test path) | runtime-trace: instrument one cycle, count seams that fire | ~72% wired ⇒ ~28% gap (claim) |
| **Run-NaN %** | of the K observable outputs in one end-to-end cycle, fraction that are NaN / `None` / unhandled-`BridgeSlot` / default-fallback | run one cycle on shipped code, count valid vs NaN | ~72% NaN (HYPOTHESIS) |

**These are distinct.** A piece can be present (99%) yet its seam unwired (28% gap)
yet its output NaN at runtime (72%). The plan's success metric is **run-NaN → <5%**
with **seam-wiring → 100%**, each step backed by a probe number.

## 2 — The loop under test (one cycle)

```text
SoA node (Kanban tenant: phase)                         ── S1
  → MUL reads Qualia(flow/trust/DK)+Meta(NARS/FE)+Plasticity ── S2
  → GateDecision (flow vs mismatch)                     ── S2
  → owner advances Kanban phase (gated write)           ── S1/S2
  → VersionScheduler lowers version-arc → KanbanMove    ── S3
  → UnifiedStep routed via BridgeSlot (lance/surreal/lancedb) ── S4
  → batch writer commits → Lance version (push?)        ── S5
  → timeline view renders kanbanview (zero-copy?)       ── S6
  → SoA self-NaN-census written to Meta (meta-awareness)── S7
  → (next cycle, pipelined against this one's disk-push)── S5/G2
```

## 3 — The seams (each a probe with PASS / KILL)

| Seam | Claim | Probe (pass/fail) | KILL condition | Status |
|---|---|---|---|---|
| **S1** kanban tenant | `ValueTenant::Kanban` at `row_offset 144` (`[112,120)` in the value slab) carries phase+cycle+exec, layout-preserving | field-isolation matrix: write each tenant, assert all others unchanged; `NodeRowPacket` round-trips kanban byte-exact | stride ≠ 512, or any other tenant perturbed | ✅ **FINDING (green 2026-06-20)** — shipped `ValueTenant::Kanban`/`KanbanTenant`/`NodeRow::{kanban,set_kanban}`; field-isolation + schema-membership tests pass; Full 112→120 ≤ 480, stride 512. + `tenant_counter` instrument. |
| **S2** MUL→phase | `MUL::GateDecision(Qualia,Meta,Plasticity)` mismatch ⇒ owner advances phase | feed a known flow-vs-mismatch qualia vector; assert the gate returns the expected `KanbanMove` (or HOLD) | gate ignores qualia (constant output), or reads uninitialized → NaN | ✅ **FINDING (green 2026-06-21)** — `NodeRow::{qualia,mul_phase_step}` + `KanbanColumn::advance_on_gate` wire `gate_decision_i4(node.qualia(), mantissa)` → DAG-legal phase advance (Flow→forward, Block→Prune-where-legal, Hold→None). Probe `s2_mul_phase_step_*` asserts all three on known qualia vectors. Pure read (owner applies via `set_kanban`). Uses the i4 integer gate — no f64/NaN path. |
| **S3** version→move | `VersionScheduler::on_version` lowers a real Lance version to the next legal `KanbanMove` | drive a 2-version dataset; assert the forward-arc move emitted | no move on a legal transition, or illegal edge emitted | ✅ **FINDING (lowering green — already shipped)** — `scheduler.rs::NextPhaseScheduler::on_version` + 6 tests (`scheduled_move_is_a_legal_rubicon_edge`, forward-arc Planning→CW→Eval→Commit, `absorbing_columns_schedule_nothing`, Libet anchor, exec-thread). Grade corrected from CONJECTURE: the *version→move lowering* is proven. **Remaining gap = the live subscription** (a `LanceVersionScheduler` subscribing to `Dataset::versions()` via the callcenter `LanceVersionWatcher` — downstream crate, not the contract). |
| **S4** envelope route | a kanban `UnifiedStep` reaches the present `BridgeSlot` (surreal plan engaged by Cargo presence) | register a surreal slot; route a `step_type:"kanban.*"`; assert it lands; with slot absent assert graceful unhandled (not panic) | routes to wrong domain, or panics when slot absent | CONJECTURE (UnifiedStep has no SoA pointer — G1) |
| **S5** batch push | commit PUSHES a kanban update to the planner (not pull-only) | commit a row; assert a push/notify carrying the `KanbanMove` is observed | commit only notifies the optimizer (current state) → gap stays | GAP (measured: pull-only) |
| **S6** timeline view | `TimelineView` renders the kanbanview **zero-copy** (`FixedSizeBinary(512)` → `&[NodeRow]`) | store a node as `FixedSizeBinary(512)`; `node_rows_from_le_bytes` over the column buffer; assert ptr-identity (no copy) | `val` is variable `Binary` (current) → cannot zero-copy → NaN/copy | GAP (val is opaque Binary) |
| **S7** meta-awareness | the SoA carries its OWN wiring-completeness census in the `Meta` tenant | compute the run-NaN% of one cycle; write it as a free-energy/awareness field; assert it reads back and reflects the real gap count | census not written, or hard-coded (doesn't track real gaps) | CONJECTURE (the capstone itself) |

## 4 — Waves (probe-ordered; no brick lands before its probe is green)

- **Wave 0 — measure the baseline on SHIPPED code (no new code).** Instrument one
  end-to-end cycle against `main` as it stands; produce the first real
  `(piece-presence%, seam-wiring%, run-NaN%)` triple. **This is the honest
  number behind "99/28/72."** Output: a `nan_census` example/bench + a recorded
  baseline. Until this runs, "72% NaN" stays a HYPOTHESIS in this doc.
- **Wave 1 — S1 + S2** (the kanban tenant keystone + the MUL trigger). Build the
  8-byte tenant (gated on operator go), field-isolation test (S1 green), wire the
  MUL→phase function (S2 green). Re-measure the triple; expect run-NaN to drop.
- **Wave 2 — S4 + S5 + S6** (envelope routing, batch push, `FixedSizeBinary(512)`).
  The heavier surrealdb-side + contract envelope work. Each gated on its probe.
- **Wave 3 — S7** the meta-awareness self-census: the SoA writes its own run-NaN%
  into `Meta`. The capstone — the system measuring its own wiring.

## 5 — The honest AGI-adjacency framing (overclaim guard)

What the green end-state proves: **the substrate can close an active-inference
loop over the SoA, per-SoA-owned, NaN-free, with the system carrying a census of
its own remaining gaps.** That is *adjacency* — the structure of the thing — not a
claim of AGI. The load-bearing word is **IF**: IF all seams go green AND run-NaN→0
AND S7 self-census tracks real gaps, THEN we are at aspiration-adjacency. The plan
converts the IF into a measured fact; it never asserts the THEN. (truth-architect /
overclaim-auditor: AGI-adjacency is the hypothesis the census tests, not a headline.)

The genuinely novel measurable: **S7 — a system aware of what it cannot yet do**
(the φ-1 permanent-humility ceiling, made a number in the `Meta` tenant). That
self-NaN-census is the "Orchestration meta-awareness plan" the operator named.

## 6 — Cross-refs

`canonical_node.rs` (`ValueTenant`/`VALUE_TENANTS` carve, free `[144,512)`),
`mul.rs` (Dk/Trust/Flow/Gate), `soa_view.rs` (`MailboxSoaOwner`/`View`),
`scheduler.rs` (`VersionScheduler`/`NextPhaseScheduler`), `cycle_accumulator.rs`,
`orchestration.rs` (`UnifiedStep`/`BridgeSlot`), surrealdb `core/src/kvs/lance/
timeline.rs`; AGENT_LOG 2026-06-20 (cont.¹⁴ the two mapping passes); EPIPHANIES
`E-SURREALDB-SECOND-BRAIN-IS-ZERO-COPY-IFF-FIXEDSIZEBINARY`; the AGI-as-glove
doctrine (four columns ARE the surface); I-VSA-IDENTITIES (register laziness —
why thinking-style is ClassView + Meta, not a new 128-bit tenant).
