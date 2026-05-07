# SoA DTO Dependency Ledger — entropy table

> **READ BY:** integration-lead, truth-architect, bus-compiler, host-glove-designer, palette-engineer, cert-officer, family-codec-smith, callcenter-specialist.
> **APPEND-ONLY.** Newest entries prepend. Status (CONJECTURE / FINDING / OPEN) per row IS updatable; everything else is immutable history.
> **First authored:** 2026-05-07 (immediately after PR #352 merge, paired with `palantir-parity-cascade-v2.md`).
> **Maintenance contract:** every PR that adds a `*Dto` / `*Row` / `*Intent` / `*Event` / `*Filter` / `*Step` / `*Slot` type prepends a row here. CI gate `tools/dto-class-check` (planned D-PARITY-V2-10) enforces.

## Why this ledger exists

The 2026-05-07 audit found 22+ DTO types across the workspace that admit three orthogonal classifications:

1. **Tier** (where in the data ladder): sensor → engine → contract → callcenter (0 → 4).
2. **Classification** (structural shape): bare-metal vs SoA-glue vs bridge-projection.
3. **Zone** (BBB membrane position): Zone 1 (BindSpace SoA, no Serialize) / Zone 2 (Arrow scalar membrane) / Zone 3 (outbound serialization).

These three axes determine **whether a type may carry `serde::Serialize`**, **which crate it belongs in**, and **whether it composes through the codec cascade or routes around it**. Classifying every DTO once prevents the recurring "should this be a struct, an enum, or a column projection" rediscovery tax.

## Definitions

- **Bare-metal DTO** — single-row scalar payload, no batch column structure. May serialize. Examples: `CognitiveEventRow`, `ExternalIntent`, `BusDto`.
- **SoA-glue DTO** — projects column slices over many rows; used for batch operations, compose chains, semiring composition. MUST NOT serialize. Examples: `UnifiedStep`, `ResonanceDto`, `TekamoloSlots`.
- **Bridge-projection DTO** — `LazyLock<&Registry>` or trait-bounded scoped view over a canonical store. Owns no data; only references. Examples: `MedcareBridge`, `OntologyDto`.
- **Tier 0** = sensor / lens (raw input, codebook indices).
- **Tier 1** = engine compute (ripple field, energy distribution).
- **Tier 2** = engine commit (single resolved thought).
- **Tier 3** = lance-graph-contract (canonical zero-dep DTOs consumed by all crates).
- **Tier 4** = lance-graph-callcenter (membrane DTOs and Zone 3 transcode).
- **Zone 1** = inside BBB; only `cognitive-shader-driver` BindSpace + `thinking-engine` ResonanceDto.
- **Zone 2** = `lance-graph-callcenter::lance_membrane`, BBB-scalar-only Arrow projection.
- **Zone 3** = `lance-graph-callcenter::transcode/`, `phoenix`, `postgrest`, `drain`, `supabase` — Serialize allowed.

## The ladder (canonical flow)

```
Tier 0  StreamDto       (codebook_indices: Vec<u16>, timestamp: u64)
        thinking-engine::dto.rs:40                                  bare-metal · Zone 1
                  │
                  ▼ engine.compute(stream)
Tier 1  ResonanceDto    (energy: Vec<f32; 4096>, top_k: [(u16, f32); 8])
        thinking-engine::dto.rs:59                                  SoA-glue · Zone 1
                  │  (this IS the SoA — not a glue layer; the substrate)
                  ▼ engine.commit(resonance)
Tier 2  BusDto          (codebook_index: u16, energy: f32, top_k: [(u16, f32); 8])
        thinking-engine::dto.rs:115                                 bare-metal · Zone 1
                  │
                  ▼ ShaderDispatch.encode(BusDto) — see engine_bridge.rs:6,37,80,121
Tier 3  ShaderDispatch / ShaderResonance / ShaderBus / ShaderCrystal
        lance-graph-contract::cognitive_shader                      bare-metal · Zone 1
        UnifiedStep         (depends_on: Vec<StepId>, status, …)    SoA-glue · Zone 1
        TekamoloSlots       (4 role-slice pairs)                    SoA-glue · Zone 1
        SlotPrior           (5×f32 axes)                             SoA-glue · Zone 1
        WorldModelDto       (qualia + axes + proprioception + cycle) SoA-glue · Zone 1
        ShaderEvent         (cycle, free_energy, resolution, style)  bare-metal · Zone 1
        StepResult          (style, mode, reason, free_will, step)   bare-metal · Zone 1
        MetaFilter          (thinking_mask, awareness_min, 3×u8)     bare-metal · Zone 1
        CommitFilter        (actor_id, max_free_energy, …)           bare-metal · Zone 2
        WorldMapDto         (state_vector[11], anchor, distance)    bridge-projection · Zone 1
                  │
                  ▼ LanceMembrane.project()
Tier 4  CognitiveEventRow   (15 scalar fields, repr(C)-style)        bare-metal · Zone 2
                            ← BBB invariant: scalar-only enforced
        ExternalIntent       (role, dn, body, routing, kind)         bare-metal · Zone 2
        OntologyDto          (entity_types + link_types + action_types)
                                                                     bridge-projection · Zone 3
        EntityTypeDto / PropertyDto / LinkTypeDto / ActionTypeDto    bridge-projection · Zone 3
        Filter (postgrest)   (5+ predicate terms)                    SoA-glue · Zone 3
        FilterTerm (spo_filter)                                       SoA-glue · Zone 3
        DriftEvent           (parallelbetrieb event)                  bridge-projection · Zone 3
        RowEncryptionPolicy / RowEncryptionRegistry                   bare-metal · Zone 3
                  │
                  ▼ Zone 3 transcode (phoenix WS / postgrest / supabase REST)
External payload  JSON-LD / Arrow RecordBatch / Phoenix channel msg
```

## Per-DTO entropy table

> Schema: `Tier · Crate · File · Line · Type · Classification · Zone · Has Serialize? · Has repr(C)? · Status`. Status is the only mutable column.

| Tier | Crate | File | Line | Type | Class | Zone | Serialize | repr(C) | Status (2026-05-07) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | thinking-engine | dto.rs | 40 | `StreamDto` | bare-metal | 1 | No | No | FINDING |
| 1 | thinking-engine | dto.rs | 59 | `ResonanceDto` | SoA-glue | 1 | No | No | FINDING (this IS the SoA) |
| 2 | thinking-engine | dto.rs | 115 | `BusDto` | bare-metal | 1 | No | No | FINDING |
| 3 | lance-graph-contract | cognitive_shader.rs | * | `ShaderDispatch` | bare-metal | 1 | No | No | FINDING |
| 3 | lance-graph-contract | cognitive_shader.rs | * | `ShaderResonance` | bare-metal | 1 | No | No | FINDING |
| 3 | lance-graph-contract | cognitive_shader.rs | * | `ShaderBus` | bare-metal | 1 | No | No | FINDING |
| 3 | lance-graph-contract | cognitive_shader.rs | * | `ShaderCrystal` | bare-metal | 1 | No | No | FINDING |
| 3 | lance-graph-contract | cognitive_shader.rs | 81 | `MetaFilter` | bare-metal | 1 | No | No | FINDING |
| 3 | lance-graph-contract | orchestration.rs | 333 | `UnifiedStep` | SoA-glue | 1 | No | No | FINDING |
| 3 | lance-graph-contract | sensorium.rs | 145 | `StepResult` | bare-metal | 1 | No | No | FINDING |
| 3 | lance-graph-contract | world_model.rs | 165 | `WorldModelDto` | SoA-glue | 1 | No | No | FINDING |
| 3 | lance-graph-contract | world_map.rs | 43 | `WorldMapDto` | bridge-projection | 1 | No | No | OPEN — classification needs explicit confirmation in registry projection layer |
| 3 | lance-graph-contract | graph_render.rs | 248 | `ShaderEvent` | bare-metal | 1 | No | No | FINDING |
| 3 | lance-graph-contract | grammar/tekamolo.rs | 18 | `TekamoloSlots` | SoA-glue | 1 | No | No | FINDING |
| 3 | lance-graph-contract | grammar/verb_table.rs | 83 | `SlotPrior` | SoA-glue | 1 | No | No | FINDING |
| 3 | lance-graph-contract | grammar/verb_table.rs | 123 | `SlotPriorDelta` | SoA-glue | 1 | No | No | FINDING |
| 3 | lance-graph-contract | external_membrane.rs | 75 | `CommitFilter` | bare-metal | 2 | No | No | FINDING — Zone 2 boundary type |
| 4 | lance-graph-callcenter | external_intent.rs | 33 | `ExternalIntent` | bare-metal | 2 | No | No | FINDING |
| 4 | lance-graph-callcenter | external_intent.rs | 111 | `CognitiveEventRow` | bare-metal | 2 | No | No | FINDING — `bbb_scalar_only_compile_check` enforces |
| 4 | lance-graph-callcenter | ontology_dto.rs | 19 | `OntologyDto` | bridge-projection | 3 | No | No | OPEN — Pillar 3 of v1 cascade collapses this to a 2-line projection over `OntologyRegistry::enumerate(ns)` |
| 4 | lance-graph-callcenter | ontology_dto.rs | 29 | `EntityTypeDto` | bridge-projection | 3 | No | No | FINDING |
| 4 | lance-graph-callcenter | ontology_dto.rs | 38 | `PropertyDto` | bridge-projection | 3 | No | No | FINDING |
| 4 | lance-graph-callcenter | ontology_dto.rs | 46 | `LinkTypeDto` | bridge-projection | 3 | No | No | FINDING |
| 4 | lance-graph-callcenter | ontology_dto.rs | 54 | `ActionTypeDto` | bridge-projection | 3 | No | No | FINDING |
| 4 | lance-graph-callcenter | postgrest.rs | 138 | `Filter` | SoA-glue | 3 | No | No | FINDING |
| 4 | lance-graph-callcenter | transcode/spo_filter.rs | 34 | `FilterTerm` | SoA-glue | 3 | No | No | FINDING |
| 4 | lance-graph-callcenter | transcode/parallelbetrieb.rs | 109 | `DriftEvent` | bridge-projection | 3 | No | No | OPEN — borderline; could be re-classified bare-metal |
| 4 | lance-graph-callcenter | policy.rs | 353 | `RowEncryptionPolicy` | bare-metal | 3 | No | No | FINDING |
| 4 | lance-graph-callcenter | policy.rs | 365 | `RowEncryptionRegistry` | bare-metal | 3 | No | No | FINDING |

**Counts** (FINDING-grade as of 2026-05-07): 22 classified, 3 OPEN (`WorldMapDto` projection-layer confirmation, `OntologyDto` collapse target, `DriftEvent` ambiguity), 0 CONJECTURE.

## Internal vs external O(1) mapping (per the user's framing)

### Internal O(1) — cognitive-shader-driver, no Serialize, in-memory

```
ThinkingEngine.commit() returns BusDto  (bare-metal, Tier 2, Zone 1)
        │
        │ ShaderDispatch.encode(BusDto):
        │   ├── BindSpace.write_cycle_fingerprint([u64; 256])
        │   │     converts to Vsa16kF32 internally
        │   ├── CollapseGate.merge(MergeMode::Bundle | Xor)
        │   ├── CAM-PQ encode → cam_pq_code: [u8; 6]   (column read O(1))
        │   ├── bgz17 encode → base17: [u8; 34]         (column read O(1))
        │   ├── PaletteSemiring → palette_key: u32      (column read O(1))
        │   └── Scent compute → scent: u8               (column read O(1))
        ▼
TripletGraph (AriGraph) write
        │ SpoBridge::promote_to_spo (v5 D-2, queued)
        ▼
CausalEdge64 column update (Pearl 2³ mask, per-row)
```

**Every step is a method on the carrier.** Zero DTO indirection. The codec cascade columns live in the SAME `BindSpace` SoA address space as Columns A..H (per `EPIPHANIES.md` 2026-04-26). A SIMD sweep can read all of them without indirection, which is why the Pillar 0 click works.

### External O(1) — lance-graph-callcenter, Zone 3, Serialize allowed

```
LanceVersionWatcher.bump(CognitiveEventRow)  (bare-metal, Zone 2)
        │
        ▼ LanceMembrane.subscribe(CommitFilter) → watch::Receiver<CognitiveEventRow>
        │
        ▼ Zone 2 → Zone 3 fan-out
                │
                ├── transcode/phoenix.rs       → JSON-LD over WebSocket
                ├── transcode/postgrest.rs     → Arrow RecordBatch over HTTP
                └── transcode/supabase.rs      → Cypher → SPARQL CONSTRUCT → JSON-LD over Realtime
                        (D-CASCADE-V1-10)
```

**The path is one-way; an inbound message reverses it**: Supabase → SemanticQuad → SPO → CollapseGate → BindSpace (D-CASCADE-V1-9 wires this).

## Encoding cascade column status (FINDING from 2026-05-07 audit)

`OntologyRegistry` (`registry.rs:33-86`, `proposal.rs:81-96 MappingRow`) carries:

| Column | Status |
|---|---|
| `bridge_id`, `public_name`, `ogit_uri` | FINDING — used as primary key tuple |
| `namespace_id: NamespaceId` | FINDING |
| `schema_ptr: SchemaPtr` | FINDING |
| `kind: SchemaKind` | FINDING |
| `semantic_type: SemanticType` | FINDING |
| `marking: Marking` | FINDING |
| `confidence: f32` | FINDING |
| `created_at_us: i64` + `created_by: String` | FINDING |
| `source_uri: String` | FINDING (carries v5 D-1 dcterms:source target) |
| `active: bool` | FINDING |
| `checksum: String` | FINDING |
| `identity_fp: Vsa16kF32` | OPEN — D-CASCADE-V1-7 wires |
| `cam_pq_code: [u8; 6]` | OPEN — D-CASCADE-V1-7 wires |
| `base17: [u8; 34]` | OPEN — D-CASCADE-V1-7 wires |
| `palette_key: u32` | OPEN — D-CASCADE-V1-7 wires |
| `scent: u8` | OPEN — D-CASCADE-V1-7 wires |
| `qualia: [f32; 18]` | OPEN — D-CASCADE-V1-7 wires |
| `meta: MetaWord` | OPEN — D-CASCADE-V1-7 wires |
| `edge: CausalEdge64` | OPEN — D-CASCADE-V1-7 wires |
| `thinking_style: Option<ThinkingStyle>` | OPEN — D-PARITY-V2-12 wires |
| `ontology_context_id: u32` | OPEN — D-CASCADE-V1-2 wires |

**Today the registry uses `(bridge_id, public_name)` tuples + `ogit_uri` hashing** for indexing (per agent audit, no codec cascade columns). The codec cascade is target state, NOT current state. This ledger row is the canonical place to track that gap.

## Bare-metal vs SoA-glue decision matrix

When designing a new DTO, ask in order:

1. **Does it project column slices over many rows?** → SoA-glue. MUST NOT carry `serde::Serialize`. Examples: `UnifiedStep.depends_on: Vec<StepId>`, `ResonanceDto.energy: Vec<f32; 4096>`, `TekamoloSlots` (4 role-slice pairs).
2. **Is it a single-row scalar payload that may exit the BBB?** → bare-metal. May carry `serde::Serialize` IF and only IF it lives in Zone 3. Examples: `CognitiveEventRow` (Zone 2 — no Serialize), `OntologyDto`'s leaf types (Zone 3 — Serialize allowed when wire emit is added).
3. **Is it a `LazyLock<&Registry>` reference + a name→row map?** → bridge-projection. MUST own no data. Examples: `MedcareBridge`, `WoaBridge`, `OntologyDto` (after Pillar 3 collapse).
4. **None of the above?** → AMBIGUOUS — flag for explicit classification in a new ledger row before merging.

## Probe queue (CONJECTURE-grade until probed)

| Probe | What to measure | Pass criterion | Status |
|---|---|---|---|
| O(1) lookup latency | `name → cam_pq_code` p99 vs raw oxigraph SPARQL p99 | ≥ 100× speedup | OPEN — D-CASCADE-V1-11 |
| `Serialize` static check | `cert-officer` parses every Zone 1 / Zone 2 type and asserts no `#[derive(Serialize)]` | zero Zone 1 / Zone 2 types with Serialize | OPEN — D-CASCADE-V1-1 |
| DTO classification CI | `tools/dto-class-check` parses every `*Dto` / `*Row` / `*Filter` and asserts a `// classification:` doc comment | 100% coverage | OPEN — D-PARITY-V2-10 |
| BusDto round-trip | `BusDto → ShaderDispatch.encode → BindSpace → ShaderResonance.unbind → BusDto` is identity | bit-exact identity | OPEN — D-PARITY-V2-3 |
| Codec cascade column SoA layout | `cargo asm` confirms a SIMD sweep over `OntologyRegistry` rows reads `cam_pq_code` + `base17` + `palette_key` from contiguous memory | one cache line per row group | OPEN — D-CASCADE-V1-7 |

## Open questions

1. **`WorldMapDto` re-classification**: currently bridge-projection because its state vector is computed from a classifier-backed lookup. Should it be SoA-glue (state vector IS a column) or bare-metal (single-row classifier output)? **Probe**: read its construction site (`world_map.rs:43-` + classifier wire-up).
2. **`DriftEvent` re-classification**: parallelbetrieb event, structurally a single-row payload but emitted by a bridge. Probe its consumers.
3. **`OntologyDto` collapse target**: after Pillar 3 of v1 cascade lands, `OntologyDto` becomes a 2-line projection. The leaf types (`EntityTypeDto`, `PropertyDto`, `LinkTypeDto`, `ActionTypeDto`) stay as bridge-projections. Is that the right split, or should the leaves disappear too? **Recommend keep leaves** for type safety at the wire boundary.
4. **Should `MetaFilter` and `CommitFilter` merge?** Both are predicate gates; one is Zone 1 (cognitive_shader.rs:81), the other is Zone 2 (external_membrane.rs:75). The Zone boundary justifies the split today; revisit if a Zone 2 type ever needs the cognitive filter shape.
5. **How do `CrystalFingerprint` variants (`Binary16K`, `Vsa16kF32`, `Vsa16kBF16`, `Vsa16kF16`, `Vsa16kI8`) relate to this ledger?** Strictly speaking they're carriers, not DTOs — but they appear in `CognitiveEventRow` as `cycle_fp_hi` / `cycle_fp_lo` `u64` fields. **Recommend: carriers do NOT appear in the ledger; only types ending in Dto/Row/Intent/Event/Filter/Step/Slot do**.

## Cross-references

- `.claude/plans/palantir-parity-cascade-v2.md` — the v2 capstone plan; this ledger ships with it.
- `.claude/plans/ogit-cascade-supabase-callcenter-v1.md` — Pillar 0 SoA-as-canon; codec cascade columns (D-CASCADE-V1-7).
- `.claude/plans/lance-graph-ontology-v5.md` — D-9 thresholds, D-2 SpoBridge.
- `.claude/plans/callcenter-membrane-v1.md` § 10.9 — Membrane Role Place Translation iron rule (parent doctrine).
- `crates/thinking-engine/src/dto.rs` — `StreamDto`, `ResonanceDto`, `BusDto` definitions.
- `crates/cognitive-shader-driver/src/engine_bridge.rs` — `BusDto` consumption point (Tier 2 → Tier 3).
- `crates/lance-graph-contract/src/orchestration.rs` — `UnifiedStep` (Tier 3 SoA-glue).
- `crates/lance-graph-callcenter/src/external_intent.rs` — `CognitiveEventRow` (Tier 4 BBB-scalar).
- `crates/lance-graph-callcenter/src/ontology_dto.rs` — `OntologyDto` family (Tier 4 bridge-projection).
- `crates/lance-graph-ontology/src/registry.rs` — `OntologyRegistry` SoA target (Pillar 0 anchor).
- `.claude/board/EPIPHANIES.md` 2026-04-26 — BindSpace 8-column extension for Foundry parity.
- `.claude/board/SINGLE_BINARY_TOPOLOGY.md` — Layer 2 ecosystem ontology.

## Maintenance protocol

1. **PR adds a new `*Dto`/`*Row`/`*Intent`/`*Event`/`*Filter`/`*Step`/`*Slot` type** → prepend a row to "Per-DTO entropy table" with the exact file:line + classification + zone + Status (CONJECTURE if unverified, FINDING if probed).
2. **PR re-classifies an existing type** → append a `**Reclassified (YYYY-MM-DD from PR #N):**` line under the existing row; do not edit the original Status.
3. **PR closes an OPEN status** → flip the Status column to FINDING and append a one-line note with the closing PR.
4. **PR introduces a new probe** → add a row to "Probe queue" with pass criterion; do not delete probe rows (keep the queue auditable).
5. **CI gate `tools/dto-class-check`** (D-PARITY-V2-10) parses every classified type and confirms its `// classification:` doc comment matches the ledger row.

This ledger is the single source of truth for DTO classification across the workspace. Disagreements between code, plan, and ledger are resolved by **updating the ledger AND the code in the same PR**.
