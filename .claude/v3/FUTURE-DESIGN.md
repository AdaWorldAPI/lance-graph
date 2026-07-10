# FUTURE-DESIGN — the V3 meta board / landing zone for post-V3 design rulings

> **APPEND-ONLY** (prepend new entries; only Status lines mutate). This is the
> landing zone the operator asked for (2026-07-10): future substrate design
> lands HERE first, referencing the V3 inventory (`COMPONENT-MAP.md`,
> `MODULE-TABLE.md`, `soa_layout/*`) so design converges on what exists.
> Canonical ruling text lives on the board (`.claude/board/EPIPHANIES.md`);
> this file is the design-side index + the wiring queue. Read AFTER the V3
> README; cite EPIPHANIES entries in PRs, not this mirror.

---

## 2026-07-10 — the ratified cognition arc (four rulings + one measured probe)

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

**Standing constraint:** the value-tenant migration is MID-FLIGHT (10 lanes,
`ENVELOPE_LAYOUT_VERSION=2`, BoardAggregates gated on the batched mint +
T1-T6) — any gem wired to the hot path lands as a ClassView READING of
existing lanes or through the envelope-auditor gate, never as an ad-hoc lane.
