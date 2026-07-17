# FUTURE-DESIGN — the V3 meta board / landing zone for post-V3 design rulings

> **APPEND-ONLY** (prepend new entries; only Status lines mutate). This is the
> landing zone the operator asked for (2026-07-10): future substrate design
> lands HERE first, referencing the V3 inventory (`COMPONENT-MAP.md`,
> `MODULE-TABLE.md`, `soa_layout/*`) so design converges on what exists.
> **The WHY lives in `VISION.md`** (same folder — the graded canon synthesis).
> Canonical ruling text lives on the board (`.claude/board/EPIPHANIES.md`);
> this file is the design-side index + the wiring queue. Read AFTER the V3
> README; cite EPIPHANIES entries in PRs, not this mirror.

---

## 2026-07-17 (sixth wave) — ACK ELIMINATED everywhere (operator directive)

| Ruling | One line | Wiring queue |
|---|---|---|
| E-ACK-ELIMINATED-1 | "remove the ack from everywhere, completely eliminate it" — the concept is expelled from code, doctrine, and vocabulary (it twice produced tier-crossing designs and became a lens distorting every session). SHIPPED: `batch_writer.rs` = pure intent recorder (`ack`/`ack_and_propose`/`acked`/`unacked`/`acked_version` deleted; zero production callers existed); probes + driver tests rewritten; durability evidence = the written row's own LanceVersion via temporal.rs; crash-replay = a temporal READ of intents vs Lance. D-AHG-1 is DEAD; the SLA-gate / actionhandler-queue concepts are retired UNBUILT; the membrane admits/serializes/splits tasks with the transcode's own native semantics (no confirmation abstraction of ours). /v3-audit check 7 = the elimination scan; guardrails trigger 7 bans rebuilding the mechanism under any name | none — the elimination is the wiring. Rows below referencing ack/SLA-gate/actionhandler are HISTORY, read through this row |

## 2026-07-17 (seventh wave) — triangle tenants × one-register-two-readings × surface separation × chess quarantine

| Ruling / design | One line | Wiring queue |
|---|---|---|
| triangle-tenants-gestalt-separation v1 (operator-directed design; plan at `.claude/plans/triangle-tenants-gestalt-separation-v1.md`) | the autopoiesis triangle = THREE value tenants (Frozen/Learned/Explore Style: 12 slots × palette256 atom; slot = StyleFamily ordinal OR template step — **one content-blind register, two ClassView readings**, so a ≤12-step compiled elixir-like template IS one facet with StepMask selecting active steps); four surfaces kept apart (triangle=policy / episodic witness+basins=memory / gestalt awareness=ResonanceDto+palette resonance incl. the opponent mirror-SoA meta / chess=quarantined corpus); **nail→hammer dispatch direction** (object resonance fetches the tool — the inverted read path structurally doesn't exist); 12-slot redundancy landed old+new and MEASURED with the jc battery before any collapse; the 226-atom cognitive palette256 (144∥34∥36∥12, 30 reserved append-only) is the value codebook | D-TRI-1..4 (STATUS_BOARD); ALL layout mint-gated — triangle tenants + chess classids + Tasks-SoA task classid ride the SAME batched mint as W2a BoardAggregates; wires the thinking-engine→P64→shader ancestry (explore←perturbation gems, learned←l4_bridge W4b with the orphan-write resolved, frozen←dispatch) |

## 2026-07-17 (fifth wave) — ack-violation regrade + the stream tier's economics named

| Ruling | One line | Wiring queue |
|---|---|---|
| E-ACK-VIOLATION-REGRADE-1 | the ack-gated advance regraded from "correct code, wrong tier" to a hard architecture violation (duplicate of batch-writer→kanbanstep, operator hard-break + internal reversal); quantified: 40 ns-class stream ops vs 10–400 µs acks (250×–10,000× stall), ractor messaging = Tokio = 10⁴–10⁷× — ractor stays compile-time-ownership-only; NEW mechanism banked: SPO 2³ rung ladder = one SPO cache load + ≤8 L1-resident cycles → CE64 NARS candidates; kanbanstep = auto-progression on the Rubicon aktionale phase (550→200 ms window); the ONLY ack home = the external ontology membrane (callcenter Supabase-realtime tickets/SLA + OGAR actionhandler queues) | doc-only here; chess/expert-iteration Phase-1 wording corrected before start (episode casts fire-and-forget at the storage boundary; teacher/NARS loop paces on nothing); drift signal unchanged |
| E-SOA-OWN-BOARD-NO-SIDECAR-1 | the SECOND ack-model violation: BatchWriter's `board`/`acked` BTreeMaps = a sidecar kanban state OUTSIDE the SoA — M24's own gate ("grep writer-internal intent queues = zero") already forbids the shape; the ruled model is the self-controlling actor-SoA (board INSIDE the SoA: ownership proof + board + budget in the owned rows); `ack_and_propose` = zero production callers, kanbanstep primitive still wired inside the ack path (b49ccf3f) | **Stage A (unblocked):** excise ack_and_propose from batch_writer.rs (pure WAL restored) → SLA gate re-homes at the callcenter membrane; /v3-audit check-7 exemption narrows. **Stage B (gated on the W2a board-tenant mint, T1–T6):** sidecar maps migrate into board-tenant rows (ack = the row's LanceVersion; unacked = a temporal.rs Strict projection); M24 goes green. **W2a priority RAISED — it now gates closing both violations** |
| E-TASKS-SOA-AT-THE-MEMBRANE-1 (⊘ partially retracted — see next row) | ack's POSITIVE role: the outer serialization boundary of INCOMING tasks, which split into SoAs; multi-SoA reasoning first-class (chess opponent "mirror neuron" SoA; Odoo multi-loop business logic never evicted from its reasoning loops); the membrane state = a **Tasks SoA** at lance-graph-callcenter (task = V3 row, 32-bit classid + 96-bit payload; SLA audit = Lance versions + temporal.rs — no bespoke acked map); P64's 64×64-tenant era superseded by the 4+12 facet cast (gridlake stays a measured batch topology, not the tenant atom) | REVISES Stage A's membrane home: Tasks SoA design (envelope-auditor + batched-mint gated — task-row classid rides the next mint), not a module with relocated maps; chess arc gains the opponent-model sibling SoA as a named Phase-1/3 element; P64-doc read-through sweep queued (low priority) |
| E-KANBAN-STAYS-INTERIOR-MEMBRANE-IS-ACK-ONLY-1 (CORRECTION) | kanban does NOT move to the membrane — it is the Rubicon reasoning advance, **reasoning-interior ONLY**. RETRACTS the previous row's "kanban board on the task row" clause: the membrane does ACK-serialization only; the Tasks SoA holds admission + SLA/ack ledger + fan-out edges (no sidecar) — the SLA-gate/actionhandler-queue tier, categorically NOT the Rubicon kanban. Cross-routing either way is the prime violation (thought→ack = E-ACK-VIOLATION-REGRADE-1; task→kanban = this) | Tasks SoA spec carries task lifecycle as an ack-gated SLA ledger, NEVER a Rubicon KanbanColumn; the reasoning kanban stays inside the child SoAs the task splits into |

## 2026-07-10 (fourth wave) — the chess arc: E-CHESS gets its measured instantiation (sibling repo)

| Seed | One line | Wiring queue |
|---|---|---|
| operator suggestion → `stockfish-rs` bootstrap (`b987c4b`) | shakmaty (rules, consumed never re-implemented) × ruff_cpp_spo Stockfish harvest (oracle-only) × the 64×64=4096 cascade — from×to move matrix / ButterflyHistory / NNUE HalfKAv2_hm are natively the gridlake shape; NNUE's incremental accumulator = **E-CHESS (#539) made literal and measurable**; int8 GEMM reuses the Tesseract-proven `ndarray::simd` primitives | Plan: `stockfish-rs/.claude/plans/stockfish-harvest-64x64-v1.md` (P0 oracle → L1 net loader → L2 index parity → L3 refresh → L4 incremental keystone → L5 evaluate() parity → L6 search on the 64×64 butterfly SoA lane, measured-not-parity; later UCI + Lichess-API arm). L4's incremental-vs-refresh speedup = the compute_dag receipt E-CHESS has been waiting for |

## 2026-07-10 (third wave) — the two-tier advance split: hard gate (OGAR tickets) vs stream (kanbanstep)

| Ruling | One line | Wiring queue |
|---|---|---|
| E-ACK-HARD-GATE-VS-KANBANSTEP-STREAM-1 | `ack_and_propose` = the low-level orchestration **HARD GATE** → belongs to OGAR ticket orchestration (explicit SLA + auditable goalstate; WAL board + `acked` map = the audit trail; waiting is a FEATURE for tickets); **kanbanstep** = the stream reasoning (in-stream synchronous `on_version → try_advance_phase(&mut)`; nobody waits, 550 ms budget) | **D-AHG-1**: move/mirror propose-on-ack into OGAR's ticket-orchestration surface — rides the batched-mint discipline, NEVER a solo OGAR PR; planner `BatchWriter` keeps the WAL/ack durability bookkeeping. Tier test for reviews: ticket (SLA/audit) → hard gate; thought (cycle budget) → stream; never cross-route |
| E-KANBANSTEP-IS-THE-TRIGGER-1 | genealogy corrected: kanbanstep (contract `scheduler.rs` §IN + symbiont `kanban_loop.rs` + supervisor `deliver_kanban_step` step shape) predates and outranks the ack-gated advance; it was obscured in the SurrealQL-AST carve-out | doc-only (leave-as-is ruling stands); drift signal = new code pacing a cycle on an ack |

## 2026-07-10 (second wave) — families vs runbooks; thinking tenants → V3; the smaller-CE64 × comma awareness probe

| Ruling | One line | Wiring queue |
|---|---|---|
| E-STYLE-FAMILY-VS-RUNBOOK-1 | 12 = abstract orchestration FAMILIES (`StyleFamily`); 36 = literal NARS RUNBOOKS → seed the rung ladder → replayable unit for rs-graph-llm chaining → feed the elixir-like notation (M10) | D-TSC-1 absorbs naming/semantics (spec `dtsc1-thinkingstyle-dedup-spec-v1.md`); runbook content attaches in D-TSC-3 |
| E-THINKING-TENANTS-V3-1 | thinking tenants → V3 substrate; old CausalEdge64 KEPT as perturbation baseline | D-TTV-1 (engineering, envelope-auditor gated) |
| (probe) | smaller CausalEdge64 × comma vs old-CE64 baseline: find the awareness knee (bits as a projection axis — the storage twin of comma-replay's scale axis) | **D-MTS-6 MEASURED GREEN 2026-07-10**: k\*=1 — one stored truth bit per comma level (2 bits/edge vs baseline 16) matches all three awareness proxies (|ΔE| 0.0084 / surprise agree 0.9688 / descent ρ 0.9792); aligned needs k\*=4; the comma lattice = low-discrepancy dither buying ≈3.4 effective bits ≈ log₂(12). D-MTS-6b (driver-integrated fixture) gates any real CE64 shrink |

Process note: this wave ran under the new **5+3 council** harness
(`.claude/agents/5plus3-council.md`, `/5plus3` skill) — first live run =
D-TSC-1; the 5-savant fan-out surfaced the three-divergent-tables finding.

## 2026-07-10 — the ratified cognition arc (four rulings + one measured probe)

> **⊘ READ-THROUGH (2026-07-17):** the first row below
> (E-ACK-IS-THE-KANBAN-TRIGGER-1, "the Lance ack pumps the next
> KanbanMove") is read THROUGH E-KANBANSTEP-IS-THE-TRIGGER-1 and
> E-ACK-VIOLATION-REGRADE-1 (fifth wave, top of this file): the
> ack-trigger framing was a hard architecture violation, reversed —
> kanbanstep is the only reasoning advance; acks are durability
> bookkeeping (and since 2026-07-17, E-ACK-ELIMINATED-1: no such
> bookkeeping exists at all); SLA-gated waiting lives only at the callcenter/OGAR
> membrane. The row's fire-and-forget clause stands; its trigger
> ranking does not. Append-only: the row is not edited.

**Rulings** (canonical: EPIPHANIES — measured results in
`E-COMMA-QUORUM-MEASURED-1`; plan: `temporal-markov-and-style-classes-v1`):

| Ruling | One line | Wiring queue |
|---|---|---|
| E-ACK-IS-THE-KANBAN-TRIGGER-1 | write fire-and-forget; the Lance ack pumps the next KanbanMove; the driver never waits | `BatchWriter::ack_and_propose` SHIPPED; sink wiring = W2 residue |
| E-ORCHESTRATION-ORGANS-1 | rs-graph-llm = consumer shell + slow path; rig = oracle-frequency; surrealdb = storage/read-glove/lowering only | W2c re-scoped; consumer adoption = W5 |
| E-THINKING-STYLES-ARE-CLASSES-1 | style = class: StepMask × WideFieldMask + rung set + KausalSpec, classid-resolved; classes project, never copy | D-TSC-1 (M9 dedup) BLOCKS; D-TSC-4 W6c re-ruling ESCALATED |
| E-MARKOV-TEMPORAL-STREAM-1 | Markov = temporal.rs sorted stream; VSA demoted to its four-test niche; L4 palette² Morton cascade | probes D-MTS-1..3 gate migration |
| E-COMMA-QUORUM-1 / E-COMMA-REPLAY-1 | comma vertical quorum + scale-as-projection-axis (replayable upper bounds) | **D-MTS-5 MEASURED GREEN 2026-07-10**: comma N_eff 11.00/12 vs strict 1.00 / unit 2.49 / rational 3.92; replay bit-identical any order; fresh level +0.83 witnesses at max\|ρ\|=0.156; 82 KB touched vs ~69 GB dense. **Boundary condition (measured):** N_eff(comma) = min(L, spectral participation of the detail) — the quorum cannot exceed the detail's information content (participation curve: 2.55 → 9.79 → 11.00). Latent granularity holds for BROADBAND residues; smooth content saturates early, measurably. |

## The migration arc (operator context, 2026-07-10) — where the gems are

```
ladybug-rs ─(migration)→ thinking-engine ─→ p64 (the DTO ladder:
  StreamDto Φ / PerturbationDto Ψ [renamed from dto.rs::ResonanceDto, D-PERT-1
  SHIPPED #630; awareness_dto.rs::ResonanceDto KEEPS its name — perspectival]
  / BusDto B / ThoughtStruct Γ) ─→ cognitive-shader-driver ─→ SoA (MailboxSoA,
  value tenants — MID-MIGRATION: 10 lanes shipped, BoardAggregates 11th gated)
```

**thinking-engine carries unwired gems** — built during the ladybug arc,
not yet on the hot path. The wiring pass consults `MODULE-TABLE.md` (per-file
census: consumes / emits / LE / debt / duplication / wave) + `COMPONENT-MAP.md`
(reuse / repurpose / retire verdicts) BEFORE proposing anything new. Known
candidates from the crate inventory (verify against MODULE-TABLE rows before
wiring; some are calibration-only by design):

- `layered.rs::CascadeChannels8` — NOW NAMED as the per-level cognitive-
  mantissa carrier of the Morton cascade (E-COMMA-QUORUM-1); collapses into
  `causal_edge::CausalEdge64`'s signed mantissa slot. First wiring target.
- `spiral_segment.rs` / `prime_fingerprint.rs` — VSA bundle perturbation
  (re-scope under the VSA-niche ruling before wiring).
- `cronbach.rs` / `ground_truth.rs` / `reencode_safety.rs` — the calibration
  battery (feeds D-MTS-2's certification gate).
- `ghosts.rs` / `persona.rs` / `qualia.rs` / `world_model.rs` — cognition
  organs; persona/archetype are Layer-2 role catalogues → style-class
  candidates (Track B, after M9).
- `domino.rs` / `composite_engine.rs` / `dual_engine.rs` — M8's collapse
  surface (4 near-duplicate engines → one dispatched engine).
- `l4_bridge.rs` / `tensor_bridge.rs` — the L4 learning-loop seam (W4b).
- **D-EPIPHANY-SIG-1 (queued, [H]/CONJECTURE)** — Hambly–Lyons epiphany-vs-rumination detector: the Lévy area of the `Think` belief-trajectory signature distinguishes real belief-update (non-tree, area swept) from circular rumination (tree-like, collapses). In-tree parts: `sigker` + `jc/hambly_lyons` + `Think`/NARS + `temporal.rs`. Probe-gated; full text EPIPHANIES `E-CHESS-ARC-TO-V3-TRANSFER-1`.

**Standing constraint:** the value-tenant migration is MID-FLIGHT (10 lanes,
`ENVELOPE_LAYOUT_VERSION=2`, BoardAggregates gated on the batched mint +
T1-T6) — any gem wired to the hot path lands as a ClassView READING of
existing lanes or through the envelope-auditor gate, never as an ad-hoc lane.
