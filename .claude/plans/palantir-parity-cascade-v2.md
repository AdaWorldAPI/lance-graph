# Palantir Parity · Cascade · DTO Ladder — v2

> **Status:** plan, not implementation.
> **Authored:** 2026-05-07 (immediately after PR #352 merge).
> **Owner crates:** `lance-graph`, `lance-graph-contract`, `lance-graph-callcenter`, `lance-graph-ontology`, `cognitive-shader-driver`, `thinking-engine`, `q2`, `crewai-rust`.
> **Depends on:** `ogit-cascade-supabase-callcenter-v1` (Pillar 0 SoA-as-canon), `lance-graph-ontology-v5` (D-9 thresholds, D-2 SpoBridge), `q2-foundry-integration-v1`, `lf-integration-mapping-v1`, `foundry-consumer-parity-v1`, `medcare-foundry-vision.md`.
> **Carry-over:** prior 4 Foundry parity docs are NOT superseded. v2 is the **integration capstone** that ties them together with v1 cascade's SoA pillar and the newly-located DTO ladder (`StreamDto`, `ResonanceDto`, `BusDto` — all in `thinking-engine::dto.rs`, upstream of contract).

## Why v2 exists

The Foundry/Gotham parity surface is already extensively mapped — 5+ documents, 41 LF + 4 W chunks, PR #272 already shipped Column H (`EntityTypeId = u16` per-row Foundry Object Type bridge). What's missing is **a single capstone** that:

1. Reconciles the prior 4 parity docs (Q2 cockpit + LF integration mapping + foundry consumer parity + MedCare Foundry vision) with v1 cascade's SoA-as-canon click.
2. Maps the **DTO ladder** — `StreamDto` → `ResonanceDto` → `BusDto` → contract DTOs → callcenter DTOs — onto Foundry primitives (Object, Pipeline, Function, ObjectView, Notification).
3. Surfaces the **Business Logic ↔ Thinking-style ↔ OGIT** triangle the user named: each Foundry "Function" maps to a `ThinkingStyle` dispatch + an OGIT verb.
4. Lists the deliverables that close gaps the prior docs left open (LF-12 Pipeline DAG, LF-20 FunctionSpec, LF-22/23 ObjectView, LF-50 ModelRegistry).

## Pillar 0 carry-forward — Foundry parity IS SoA-as-canon parity

The v1 cascade's Pillar 0 (`OntologyRegistry` IS the SoA; schema IS the DTO + index) already provides every Foundry primitive the parity docs targeted, **without a separate object/pipeline/function table per primitive**:

| Foundry primitive | Our equivalent | Where it lives |
|---|---|---|
| **Object** (per-row entity with type) | SoA row + Column H `EntityTypeId: u16` | `cognitive-shader-driver::BindSpace.entity_type` (PR #272 SHIPPED) |
| **Object Type** (schema for a class) | OGIT TTL entity declaration | `OGIT/NTO/<Namespace>/entities/*.ttl` (v4 + v5 work) |
| **Pipeline Builder** (DAG of transforms) | `UnifiedStep.depends_on: Vec<StepId>` | `lance-graph-contract::orchestration::UnifiedStep` (LF-12 QUEUED) |
| **Function** (named transform) | `ThinkingStyle` dispatch + `FunctionSpec` row | `lance-graph-contract::thinking` (36 styles) + LF-20 (QUEUED) |
| **Object Explorer** (UI property scrolling) | Q2 Object Explorer with PrefetchDepth L0→L3 | `q2-foundry-integration-v1.md` Q2-1.2 (QUEUED) |
| **Cypher / Workshop console** | Q2 Cypher Console (polyglot) | Q2-2.x (QUEUED) |
| **ObjectView** (card / detail / summary) | `Schema::ObjectView` + `NotificationSpec` | LF-22/23 (QUEUED) |
| **Model Registry** (ML deployment surface) | `lance-graph-models` crate (planned) | LF-50/52 (QUEUED) |
| **Helix** (time-series histogram) | NARS `(frequency, confidence)` per triple + `CausalEdge64` Pearl 2³ mask | NOT replicated — architecturally novel; Vertex has zero causal typing |
| **Code Workbook** (notebook surface) | `lance-graph-python` + Jupyter (existing) | `crates/lance-graph-python/` |
| **Action** (state-mutating call) | `ExternalIntent` → `UnifiedStep` → BindSpace write | `lance-graph-callcenter::external_intent` |
| **Foundry Function (Compute Module)** | `crewai-rust` agent + `n8n-rs` workflow | sibling repos via `lance-graph-contract` |

**Click**: there is no separate "Foundry Object table" — there is a `BindSpace` SoA row whose Column H gives it a Foundry-equivalent type tag. The whole architectural distance between Foundry and our stack is *which primitive owns the row*. Foundry says "an Object row in the Ontology table". We say "a row in the canonical SoA, with its EntityTypeId column = ontology link". Same outcome, our structure is one-fewer-layer-deep.

## The 8 column map (extended with codec cascade)

Per `EPIPHANIES.md` 2026-04-26 (Foundry parity discussion) the BindSpace already carries 8 columns A–H. v1 cascade's codec cascade extends **inside** Column H's per-row state, not as new columns:

| Col | Name | Type | Foundry primitive | Codec cascade row state |
|---|---|---|---|---|
| A | Entity | row id | Object | (this row's address) |
| B | Content | `Box<[u64; 256]>` | Property values | shadow of inner content |
| C | Cycle | `Box<[f32]>` (Vsa16kF32, 64 KB) | — | identity fingerprint |
| D | Topic | `Box<[u64]>` | Object discussion topic | bundled context |
| E | Angle | `Box<[u64]>` | Object viewing-angle / persona | dispatched perspective |
| F | OntologyDelta | per-row CRDT | Branch / scenario diff | Foundry Scenario equivalent |
| G | Awareness | per-row pressure | Object access metadata | DK + flow + compass |
| **H** | **EntityType** | `EntityTypeId: u16` | **Foundry Object Type** | **opaque link to OGIT row** |

**v2 extension** — the codec cascade per H-row (one per OntologyRegistry entry):

| Sub-column | Type | Bytes | Foundry analog |
|---|---|---|---|
| `identity_fp` | `Vsa16kF32` | 64 KB | Object embedding |
| `cam_pq_code` | `[u8; 6]` | 6 | k-NN search index |
| `base17` | `[u8; 34]` | 34 | bgz17 archetype |
| `palette_key` | `u32` | 4 | PaletteSemiring composition handle |
| `scent` | `u8` | 1 | final scent tier |
| `qualia` | `[f32; 18]` | 72 | NARS truth + DK + flow + compass |
| `meta` | `MetaWord` | 8 | dispatch bits + dcterms:source pointer |
| `edge` | `CausalEdge64` | 8 | Pearl 2³ causal mask (Foundry has none) |

**Foundry parity differentiator**: every row's Pearl 2³ mask is a **causal type tag** Foundry/Vertex don't have. Our `Helix`-equivalent isn't a histogram of *time*; it's a histogram of *causal masks across time* — strictly more expressive.

## DTO ladder — the bare-metal vs SoA-glue split

The DTO inventory (per the 2026-05-07 audit) found `StreamDto` / `ResonanceDto` / `BusDto` in `thinking-engine::dto.rs` (NOT in lance-graph-contract). The ladder runs across four tiers:

```
Tier 0  sensor / lens               StreamDto     (codebook indices, bare-metal)
                  │
                  ▼
Tier 1  thinking-engine compute     ResonanceDto  (4096 ripple energies, SoA-glue)
                  │                                ← this is where the SoA actually is
                  ▼
Tier 2  thinking-engine commit      BusDto        (single committed thought, bare-metal)
                  │
                  ▼
Tier 3  lance-graph-contract        UnifiedStep   (DAG vector, SoA-glue)
                                    ShaderEvent   (single render output, bare-metal)
                                    StepResult    (single decision, bare-metal)
                                    TekamoloSlots (4 role-slice pairs, SoA-glue)
                  │
                  ▼
Tier 4  lance-graph-callcenter      OntologyDto      (entity + link + action vectors, bridge-projection)
                                    CognitiveEventRow (15 scalar fields, bare-metal — BBB-scalar-only)
                                    ExternalIntent   (gate-crossing, bare-metal)
                                    CommitFilter     (fanout predicate, bare-metal)
```

**Bare-metal DTO** = single-row scalar payload, no batch column structure, eagerly serialized at Zone 3 (or never). 9 instances across crates.

**SoA-glue DTO** = projects column slices over many rows, used for batch operations or compose chains. 7 instances. **`ResonanceDto.energy: Vec<f32>` over 4096 codebook entries IS the SoA — not a glue layer; it's the substrate.**

**Bridge-projection DTO** = `LazyLock<&Registry>` or trait-bounded view over a canonical store. 6 instances (3 clear, 3 ambiguous — flagged in the ledger).

**The classification matters because**:
- Bare-metal DTOs may carry `serde::Serialize` if (and only if) they live in Zone 3.
- SoA-glue DTOs MUST NOT carry `serde::Serialize` — they project columns; serializing the projection breaks the SIMD sweep.
- Bridge-projection DTOs MUST NOT own data — only `LazyLock<&Registry>` references.

## Internal vs external O(1) mappings

Per the user's framing — two distinct surfaces:

### Internal (cognitive-shader-driver, in-memory, no Serialize)

```
ThinkingEngine.commit() → BusDto
        │
        ▼
ShaderDispatch.encode(BusDto)
        │
        ├── BindSpace SoA write (Column A, B, C, D, E, F, G, H)
        ├── CollapseGate.merge (MergeMode::Bundle | MergeMode::Xor)
        ├── CAM-PQ encode (DistanceTableProvider)
        ├── bgz17 encode (Base17)
        ├── palette key compute (PaletteSemiring)
        └── Scent compute (final cascade tier)
        │
        ▼
TripletGraph (AriGraph) write → SpoBridge::promote_to_spo (v5 D-2)
        │
        ▼
CausalEdge64 column update (per-row Pearl 2³ mask)
```

**No DTO crosses this path eagerly.** Every step is a method on the carrier. `BindSpace.write_cycle_fingerprint(&[u64; 256])` is the canonical entry point; it converts internally to `Vsa16kF32`. The codec cascade columns live in the SAME address space as `Column A..H`, so a SIMD sweep can read all of them without indirection.

### External (lance-graph-callcenter, Zone 3, Serialize allowed)

```
LanceVersionWatcher.bump(CognitiveEventRow)  ← Zone 2 fans out
        │
        ▼
LanceMembrane.subscribe(CommitFilter) → watch::Receiver<CognitiveEventRow>
        │
        ▼
Zone 3 transcode/{phoenix, postgrest, supabase}.rs
        ├── CognitiveEventRow → JSON-LD (Phoenix WS payload)
        ├── CognitiveEventRow → Arrow RecordBatch (PostgREST)
        └── CognitiveEventRow → Supabase realtime payload (Cypher → SPARQL CONSTRUCT → JSON-LD)
        │
        ▼
External consumer (browser, agent, n8n flow, crewai task)
```

**Every external DTO crosses Zone 2's BBB membrane as Arrow scalar columns first** (`bbb_scalar_only_compile_check`), then becomes a Serialize-derived shape ONLY in Zone 3. The path is one-way; an inbound message reverses it (Supabase → SemanticQuad → SPO → CollapseGate → BindSpace).

## Business Logic ↔ Thinking-style ↔ OGIT (the third triangle)

The user's ask: *"the Business Logic needs some Thinking style OGIT mapping later"*. The shape is a triangle, not a chain. Each business operation has three faces:

```
                      Thinking style
                      (lance-graph-contract::thinking, 36 styles)
                            ▲
                            │
                            │  dispatches
                            │
       Business operation ──┼── OGIT verb
       (industry case)      │  (OGIT/NTO/<Namespace>/verbs/*.ttl)
                            │
                            │  describes
                            │
                            ▼
                      OGIT entity
                      (OGIT/NTO/<Namespace>/entities/*.ttl)
```

Example: `WorkOrder::Issue` is an operation. It dispatches `ThinkingStyle::PracticalCommit` (the cluster that commits transactional facts). It maps to the OGIT verb `Issued` (Customer→Order). Its arguments are OGIT entities (`Customer`, `Order`, `Position`).

v2 introduces this triangle as an **append-only knowledge artifact**, not as a new schema column. Each business operation gets one row in `.claude/knowledge/business-thinking-ogit-triangle.md` (NEW, deliverable D-PARITY-V2-9 below) carrying `(operation_name, thinking_style, ogit_verb, ogit_entities[])`.

The crewai-rust + n8n-rs consumers read this knowledge file as a routing table; the lance-graph-planner consults it when picking a strategy.

## Deliverables (15 total, ranked by leverage / cost)

| Rank | D-id | Scope | LOC | Owner crate / file |
|---|---|---|---|---|
| 1 | **D-PARITY-V2-1** | Knowledge doc: `soa-dto-dependency-ledger.md` (the entropy ledger; this v2 plan ships it) | ~250 | `.claude/knowledge/` |
| 2 | **D-PARITY-V2-2** | Knowledge doc: `business-thinking-ogit-triangle.md` (operation → style → verb → entities routing) | ~150 | `.claude/knowledge/` |
| 3 | **D-PARITY-V2-3** | Wire `cognitive-shader-driver::engine_bridge` to consume `BusDto` directly (close the StreamDto → ResonanceDto → BusDto → ShaderDispatch path) | ~200 | `cognitive-shader-driver::engine_bridge` |
| 4 | **D-PARITY-V2-4** | LF-22/23: `Schema::ObjectView` + `NotificationSpec` in `lance-graph-contract` | ~180 | `lance-graph-contract::ontology` |
| 5 | **D-PARITY-V2-5** | LF-20: `FunctionSpec { name, signature, body, thinking_style: ThinkingStyle }` in `lance-graph-contract` | ~150 | `lance-graph-contract::function` (new module) |
| 6 | **D-PARITY-V2-6** | LF-12: wire `UnifiedStep.depends_on` resolution into `OrchestrationBridge::route()` (Pipeline DAG hot path) | ~250 | `lance-graph-contract::orchestration` |
| 7 | **D-PARITY-V2-7** | Q2-1.2: Object Explorer property scrolling at PrefetchDepth L0→L3, consuming `OntologyDto` projection | ~400 | `q2/src/ui/explorer.rs` (cross-repo) |
| 8 | **D-PARITY-V2-8** | LF-50/52: scaffold `lance-graph-models` crate (ModelRegistry, LlmProvider trait, deployment manifest) | ~300 | `crates/lance-graph-models/` (new) |
| 9 | **D-PARITY-V2-9** | Populate first 12 rows of `business-thinking-ogit-triangle.md` (one per WorkOrder operation in v4 OGIT TTL emit) | ~12 entries × ~5 lines | `.claude/knowledge/business-thinking-ogit-triangle.md` |
| 10 | **D-PARITY-V2-10** | DTO classification CI check: `cargo metadata` + `syn` parse asserts every `*Dto`/`*Row`/`*Filter` type carries a `// classification: bare-metal | soa-glue | bridge-projection` doc comment | ~250 | `tools/dto-class-check/` (new bin) |
| 11 | **D-PARITY-V2-11** | Foundry parity test: assert that for every `Schema::ObjectView` we can render a Q2 cockpit panel without writing a single new endpoint (Wire DTOs are lab quarantine per PR #223) | ~300 | `q2/tests/parity.rs` |
| 12 | **D-PARITY-V2-12** | `OntologyRegistry::SchemaPtr` carries `thinking_style: Option<ThinkingStyle>` so per-entity dispatch is a column read | ~80 | `lance-graph-ontology::registry` (extends v5 D-9 surface) |
| 13 | **D-PARITY-V2-13** | bgz-tensor `AttentionSemiring` integration with codec cascade columns (composes over `palette_key`) | ~200 | `bgz-tensor::hhtl_cache` |
| 14 | **D-PARITY-V2-14** | Helix-equivalent: `CausalEdge64` time-series histogram operator in `lance-graph-planner` (Foundry-superseding causal histogram) | ~250 | `lance-graph-planner::physical` |
| 15 | **D-PARITY-V2-15** | End-to-end parity test: open a Q2 Object Explorer panel, click an entity, hit Function (`ThinkingStyle` dispatch), observe the Pipeline DAG (`UnifiedStep.depends_on`), see the result render through `Schema::ObjectView` — all without leaving the SoA | ~500 | `q2/tests/end_to_end_foundry_parity.rs` |

## Dependencies (path graph)

```
        v5 D-9 (MulThresholdProfile)
               │
               ▼
     v1 D-CASCADE-V1-2 (SchemaPtr.context_id)
               │
               ▼
     v1 D-CASCADE-V1-7 (codec cascade columns)
               │
               ├──► v2 D-PARITY-V2-12 (SchemaPtr.thinking_style)
               │
               ├──► v2 D-PARITY-V2-4 (Schema::ObjectView)
               │           │
               │           ▼
               │   v2 D-PARITY-V2-7 (Q2 Object Explorer)
               │           │
               │           ▼
               │   v2 D-PARITY-V2-11 (parity test)
               │
               ├──► v2 D-PARITY-V2-5 (FunctionSpec)
               │           │
               │           ▼
               │   v2 D-PARITY-V2-6 (Pipeline DAG resolver)
               │           │
               │           ▼
               │   v2 D-PARITY-V2-15 (end-to-end test)
               │
               └──► v2 D-PARITY-V2-13 (bgz-tensor cascade composition)
                           │
                           ▼
                   v2 D-PARITY-V2-14 (causal histogram operator)

     v2 D-PARITY-V2-1 (DTO ledger)        ← independent, ships with this plan
     v2 D-PARITY-V2-2 (triangle ledger)   ← independent, ships with this plan
     v2 D-PARITY-V2-3 (BusDto bridge)     ← independent, only needs thinking-engine
     v2 D-PARITY-V2-9 (12 triangle rows)  ← needs V2-2
     v2 D-PARITY-V2-10 (CI class check)   ← needs V2-1
     v2 D-PARITY-V2-8 (lance-graph-models) ← independent, scaffold only
```

## Acceptance criteria

- [ ] `cargo test -p lance-graph-callcenter --features full` passes; `bbb_scalar_only_compile_check` still compiles.
- [ ] `tools/dto-class-check` (D-PARITY-V2-10) fails CI if any `*Dto`/`*Row`/`*Filter` type lacks a classification doc comment.
- [ ] Q2 parity panel (D-PARITY-V2-11) renders an `OntologyDto` projection without using any Wire DTO defined under `cognitive-shader-driver/src/wire.rs`.
- [ ] D-PARITY-V2-15 end-to-end test passes, asserting (via `cert-officer`) that no Zone 1 / Zone 2 type was reachable through `Serialize` during the round-trip.
- [ ] DTO ledger (D-PARITY-V2-1) classifies all 22+ DTOs found in the 2026-05-07 audit; new DTOs added by deliverables 4–8 above append rows.
- [ ] Triangle ledger (D-PARITY-V2-9) carries one row per WorkOrder operation in OGIT/NTO/WorkOrder/verbs/.
- [ ] No upstream PRs to `almatoai/OGIT` (per v5 ratification).

## Out of v2 scope

- **CRDT scenario branching** (Foundry's `Branch` and `MasterBranch`) — Column F already provides per-row OntologyDelta; surfacing it as a UI affordance is a v3 concern.
- **Foundry Marketplace / Compass** (data discovery surfaces) — adjacent UX; our equivalent is `OntologyRegistry::enumerate(namespace)` exposed via Q2.
- **Foundry Code Repositories** (managed git surface) — orthogonal; we use external git.
- **Vertex / Workshop** specific UI affordances — covered by the existing q2-foundry-integration-v1 plan; v2 does not extend Q2's UX scope.
- **Ontology import from Foundry export format** — speculative; the OGIT TTL surface is the canonical input.

## Open questions

1. **DTO ledger maintenance pattern**: append-only knowledge doc (current proposal) vs cargo-doc-generated (auto-extracted from `// classification:` doc comments)? **Recommend manual append + auto-validate via D-PARITY-V2-10**, so the ledger is human-curated but machine-verified.
2. **Triangle ledger scope**: just WorkOrder for v2 (12 rows), or also Healthcare and SMB (~50 rows)? **Recommend WorkOrder-only for v2**; expand in v3 once the pattern is proven.
3. **Q2 dependency direction**: should `q2` depend on `lance-graph-callcenter`, or only on `lance-graph-contract`? **Recommend contract-only** to preserve the lab-quarantine boundary; `q2` consumes `OntologyDto` via the contract surface.
4. **`thinking_style: Option<ThinkingStyle>` on `SchemaPtr`**: column on every row (D-PARITY-V2-12) vs lookup table keyed by `(namespace, public_name)`? **Recommend column** — makes the SoA sweep coherent; lookup table reintroduces indirection.
5. **bgz-tensor causal histogram** (D-PARITY-V2-14) — is this `lance-graph-planner::physical` territory or a new `lance-graph-helix` crate? **Recommend planner-physical** unless it grows past 800 LOC.

## Self-bootstrapping prompt for next session

```
Read .claude/plans/palantir-parity-cascade-v2.md cover-to-cover. The Pillar 0
carry-forward (Foundry parity IS SoA-as-canon parity) is the architectural
anchor. If you find yourself proposing a new ObjectTable / FunctionTable /
PipelineTable, stop — those are the SoA columns A-H, and v2's job is to make
the SoA carry the Foundry-equivalent shape, not duplicate the table set.

Top-3 deliverables to start: D-PARITY-V2-1 (the DTO ledger; ships with this
plan as `.claude/knowledge/soa-dto-dependency-ledger.md`), D-PARITY-V2-2 (the
triangle ledger; ships as `.claude/knowledge/business-thinking-ogit-triangle.md`),
D-PARITY-V2-3 (BusDto bridge into engine_bridge.rs).

Cross-plan deps: v1 D-CASCADE-V1-7 (codec cascade columns) MUST land before
V2-12 (SchemaPtr.thinking_style); v1 D-CASCADE-V1-2 (SchemaPtr.context_id) MUST
land before V2-4 (Schema::ObjectView). Confirm those before starting V2-12 / V2-4.

Branch: claude/create-graph-ontology-crate-gkuJG (per workspace policy).
PR target: AdaWorldAPI/lance-graph base=main.
```

## Cross-references

- `.claude/plans/lance-graph-ontology-v5.md` — D-9 thresholds, D-2 SpoBridge.
- `.claude/plans/ogit-cascade-supabase-callcenter-v1.md` — Pillar 0 SoA-as-canon, codec cascade columns.
- `.claude/plans/q2-foundry-integration-v1.md` — Q2 = Gotham UI equivalent (4 phases).
- `.claude/plans/lf-integration-mapping-v1.md` — 41 LF + 4 W chunks across stages 1-8.
- `.claude/plans/foundry-consumer-parity-v1.md` — SMB + MedCare shared surface.
- `.claude/medcare-foundry-vision.md` — F1-F5 phases with cost/sovereignty positioning.
- `.claude/board/SINGLE_BINARY_TOPOLOGY.md` — Layer 2 ecosystem ontology.
- `.claude/board/EPIPHANIES.md` 2026-04-26 — BindSpace 8-column extension for Foundry parity.
- `.claude/knowledge/soa-dto-fma-map.md:215-217` — Column H Foundry Object Type bridge (PR #272).
- `.claude/knowledge/soa-dto-dependency-ledger.md` (NEW, ships with this plan) — entropy ledger.
- `.claude/knowledge/business-thinking-ogit-triangle.md` (NEW, ships with this plan) — operation routing.

## Confidence (2026-05-07)

Pre-execution. Pillar 0 carry-forward is the only architectural commitment that admits no rollback — and it is right per the Foundry parity prior art (Column H is already the bridge; the rest is column population, not new tables). The 15 deliverables are bounded; D-PARITY-V2-1, V2-2, V2-3 land first because they have no upstream blockers and ship with this plan.
