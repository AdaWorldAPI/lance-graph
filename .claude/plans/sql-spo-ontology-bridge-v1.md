# SQL ↔ SPO Ontology Bridge — v1 Implementation Plan

> **Status:** Active
> **Author:** Plan agent (Opus), session 2026-04-30
> **Cross-ref:** `foundry-consumer-parity-v1.md`, `foundry-soa-row-projection-v1.md` (queued)

---

## Overview

Production-grade SQL↔SPO ontology bridge enabling MedCare-rs and smb-office-rs
consumers to query AriGraph SPO triples via DataFusion SQL, reconcile against
MySQL ground truth, and receive NARS truth values as auditable SQL columns.

Architecture follows the existing callcenter membrane pattern: contract types
in `lance-graph-contract` (zero-dep), SPO machinery in `lance-graph`, and the
DataFusion/external surface in `lance-graph-callcenter` (the timing adapter
between internal 20-200ns and external 2-200ms).

---

## Phase 1: Schema → SPO Expansion (the bridge contract)

### Problem

An `Ontology` (Schema + LinkSpec) is a typed object model. The SPO triple store
operates on `Fingerprint`-addressed triples with `TruthValue`. There's no
automated expansion. Today, triples are inserted manually via
`SpoBuilder::build_edge`. The bridge must automate: given an entity row
conforming to a Schema, expand its properties and links into SPO triples
with appropriate truth values.

### Design

**New trait `SchemaExpander` in `lance-graph-contract::ontology`:**

```rust
trait SchemaExpander {
    fn expand_entity(&self, entity_type: &str, entity_id: u64,
                     properties: &[(&str, &[u8])]) -> Vec<ExpandedTriple>;
    fn expand_link(&self, link: &LinkSpec, subject_id: u64, object_id: u64) -> ExpandedTriple;
}
```

**New struct `ExpandedTriple` (zero-dep DTO):**

```rust
struct ExpandedTriple {
    subject_label: String,
    predicate: &'static str,
    object_label: String,
    truth: (f32, f32),                  // frequency, confidence
    property_kind: PropertyKind,
    marking: Marking,
    semantic_type: SemanticType,
}
```

Each `PropertySpec.predicate` becomes `(entity:{type}:{id}, {predicate}, value:{hash})`.
Each `LinkSpec` becomes `(entity:{subject_type}:{subject_id}, {link.predicate},
entity:{object_type}:{object_id})`. Truth values from `PropertySpec.nars_floor`
(Required start at floor; Optional/Free start at `TruthValue::unknown()`).

### Crate locations
- `crates/lance-graph-contract/src/ontology.rs` — extend with `ExpandedTriple` + `SchemaExpander` (~70 LOC)
- `crates/lance-graph/src/graph/spo/ontology_bridge.rs` — new file (~200 LOC)
- Re-export in `crates/lance-graph/src/graph/spo/mod.rs`

### LOC: ~270 | Dependencies: none

### Tests
- Unit: expand `Schema::builder("Customer").required("name").required("tax_id").build()` → 2 triples with correct labels
- Unit: expand `LinkSpec::one_to_many("Customer", "issued", "Invoice")` → 1 edge triple
- Integration: round-trip through `SpoStore`
- Integration: expand `smb_ontology()` and `medcare_ontology()`, verify counts

---

## Phase 2: DataFusion SQL Surface over SPO

### Problem

Consumers want `SELECT * FROM customer_triples WHERE predicate = 'tax_id' AND nars_frequency > 0.7`.
Today SPO is an in-memory `HashMap<u64, SpoRecord>` with Hamming queries. No `TableProvider`.

### Design

**New struct `SpoTableProvider` in `lance-graph-callcenter`** implementing
`datafusion::datasource::TableProvider` over `Arc<SpoStore>` (or production
Lance dataset).

**Arrow schema:**

| Column | Arrow Type |
|---|---|
| `entity_type` | Utf8 |
| `entity_id` | UInt64 |
| `subject` | FixedSizeBinary(64) |
| `predicate` | Utf8 (denormalized) |
| `object` | FixedSizeBinary(64) |
| `object_label` | Utf8 |
| `nars_frequency` | Float32 |
| `nars_confidence` | Float32 |
| `nars_expectation` | Float32 |
| `property_kind` | Utf8 |
| `marking` | Utf8 |
| `tenant_id` | Utf8 |

**Predicate pushdown:** `predicate = 'tax_id'` → fingerprint match on `label_fp("tax_id")`.
`nars_frequency > 0.7` → `TruthGate` threshold. `entity_type = 'Customer'` → `EntityTypeId` filter.

**RLS integration:** `SpoTableProvider` registered as table per entity type
(`customer_triples`, `invoice_triples`). Existing `RlsRewriter` (PR-F1, shipped)
injects `tenant_id` predicates unchanged.

### Crate location
- `crates/lance-graph-callcenter/src/spo_table.rs` (new, ~350 LOC)
- Gated `query-lite`

### LOC: ~350 | Dependencies: Phase 1

### Tests
- Unit: register `SpoTableProvider`, run `SELECT predicate, nars_frequency FROM customer_triples WHERE entity_id = 42`
- Unit: predicate pushdown reduces scan
- Integration: SQL JOIN across entity types via `LinkSpec`
- Integration: `RlsRewriter` injects tenant predicate

---

## Phase 3: MySQL Ground-Truth Reconciler

### Problem

Both consumers run MySQL as parity oracle. Reconciler compares DataFusion (Lance-backed SPO)
against MySQL, produces audit entries for drift.

### Design

**New trait `Reconciler`:**

```rust
trait Reconciler {
    type Error;
    async fn reconcile(&self, entity_type: &str,
                       lance_batch: &RecordBatch,
                       mysql_batch: &RecordBatch) -> Result<ReconcileReport, Self::Error>;
}
```

**`ReconcileReport`:**

```rust
struct ReconcileReport {
    entity_type: &'static str,
    total_rows: usize,
    matched: usize,
    drifted: Vec<DriftEntry>,
    lance_only: usize,
    mysql_only: usize,
    timestamp_ms: u64,
}

struct DriftEntry {
    entity_id: u64,
    column: String,
    lance_value: String,
    mysql_value: String,
    drift_kind: DriftKind,
}

enum DriftKind {
    ValueMismatch,
    TypeMismatch,
    NullMismatch,
    TruthBelowFloor { frequency: f32, confidence: f32 },
}
```

**Zero-copy comparison:** Walk both batches column-by-column via Arrow accessors
(`StringArray::value(i)`, `Float32Array::value(i)`). No row materialization.

**Drift → Audit:** Each `DriftEntry` → `AuditEntry { action_kind: AuditAction::Import }`
→ `LanceAuditSink` (PR-F3, shipped).

### Crate locations
- `ReconcileReport`, `DriftEntry`, `DriftKind`, `Reconciler`: contract (~100 LOC)
- `crates/lance-graph-callcenter/src/reconciler.rs` (new, ~300 LOC)
- Feature: `mysql-reconciler = ["dep:sqlx"]` or `reconciler` (no MySQL dep variant)

### LOC: ~400 | Dependencies: Phase 2

### Tests
- Unit: identical batches → `matched == total`
- Unit: one column differs → `DriftKind::ValueMismatch`
- Unit: NULL vs value → `DriftKind::NullMismatch`
- Unit: truth below `nars_floor` → `DriftKind::TruthBelowFloor`
- Integration: `smb_ontology()` populated, mutate one row, verify exactly one drift
- Integration: drifts flow through `AuditSink::append()`

---

## Phase 4: NARS Cold-Path Sink

### Problem

NARS revisions (from cognitive shaders or business logic) must sink to SQL as
auditable columns. Unidirectional NARS→SQL. Cold path: 2-200ms, batched.

### Design

**New struct `NarsColdSink` in callcenter:**

1. Receive `NarsRevision` events
2. Batch in bounded `VecDeque` (like `InMemoryAuditSink`)
3. On flush (timer 100ms or 64 entries): convert to Arrow `RecordBatch`, update SPO truth, append audit, optionally write Lance dataset

**`NarsRevision`:**

```rust
struct NarsRevision {
    entity_type: &'static str,
    entity_id: u64,
    predicate: &'static str,
    old_truth: (f32, f32),
    new_truth: (f32, f32),
    source: RevisionSource,
    timestamp_ms: u64,
}

enum RevisionSource {
    CognitiveShader,
    BusinessRule,
    UserCorrection,
    ModelPrediction,
    Reconciler,
}
```

### Crate locations
- `NarsRevision`, `RevisionSource`: contract (~50 LOC)
- `crates/lance-graph-callcenter/src/nars_cold_sink.rs` (new, ~250 LOC)
- Feature: `audit-log` (reuses existing)

### LOC: ~290 | Dependencies: Phase 1, Phase 2

### Tests
- Unit: 10 revisions → flush → 10-row RecordBatch
- Unit: `UserCorrection` → `AuditEntry { action_kind: AuditAction::Update }`
- Unit: 64 entries at 1ms → single flush
- Integration: revision drops below `nars_floor` → reconciler detects on next pass

---

## Phase 5: BindSpaceRowDto (zero-copy row projection)

### Problem

Internal BindSpace SoA → external DTO at row level for PostgREST/Phoenix consumers.
Zero-copy (Arrow views, not owned). Internal types (Fingerprint, MetaWord,
CausalEdge64) never leak; decoded scalars cross BBB.

### Design

**`BindSpaceRowDto<'a>`** — zero-copy view backed by Arrow arrays:

```rust
struct BindSpaceRowDto<'a> {
    // Decoded from MetaWord (u32 → 5 scalars)
    thinking_style: u8,
    awareness: u8,
    nars_frequency: u8,
    nars_confidence: u8,
    free_energy: u8,

    // QualiaColumn (18×f32 slice view)
    qualia: &'a [f32],

    // EdgeColumn (u64 → decoded)
    edge: CausalEdgeDto,

    // entity_type column (u16 → name lookup)
    entity_type_name: &'a str,
    entity_type_id: u16,

    temporal: u64,
    expert: u16,

    // Fingerprints stay BEHIND BBB
    content_hash: u64,  // FNV of content fp (correlation only)
}
```

**`CausalEdgeDto`:** decodes packed u64 into readable fields.

**`project_bindspace_batch(bs, window, ontology) -> RecordBatch`:** columnar
projection. Each column built from SoA buffers. Compile-time BBB guarantee:
`RecordBatch` cannot contain `Fingerprint` or `MetaWord` (not Arrow types).

### Crate locations
- `crates/lance-graph-callcenter/src/bindspace_dto.rs` (new, ~300 LOC)
- Feature: `persist`

### LOC: ~300 | Dependencies: Phase 1 (EntityTypeId mapping)

### Tests
- Unit: 4-row BindSpace → projected batch → 4 rows
- Unit: `MetaWord::new(5,3,200,150,30)` → decoded scalars match
- Unit: `entity_type[0]=1` + Ontology with "Customer" at index 0 → name = "Customer"
- Unit: BBB test — fingerprint columns absent from RecordBatch schema
- Integration: SQL query over projected batch

---

## Sequencing

```
Phase 1 (Schema→SPO)
    │
    ├──→ Phase 2 (DataFusion TableProvider)
    │        │
    │        ├──→ Phase 3 (MySQL Reconciler)
    │        └──→ Phase 4 (NARS Cold Sink)
    │
    └──→ Phase 5 (BindSpaceRowDto)
```

**Parallelization:** Phases 2 and 5 run in parallel after Phase 1.
Phases 3 and 4 run in parallel after Phase 2.

## Summary

| Phase | Title | Crate | LOC | New file | Feature |
|---|---|---|---|---|---|
| 1 | Schema→SPO | contract + lance-graph | ~270 | `ontology_bridge.rs` | none |
| 2 | DataFusion SQL | callcenter | ~350 | `spo_table.rs` | `query-lite` |
| 3 | MySQL reconciler | contract + callcenter | ~400 | `reconciler.rs` | `reconciler` |
| 4 | NARS cold sink | contract + callcenter | ~290 | `nars_cold_sink.rs` | `audit-log` |
| 5 | BindSpaceRowDto | callcenter | ~300 | `bindspace_dto.rs` | `persist` |
| **Total** | | | **~1,610** | **5 new files** | |

---

## Critical Files (read before implementation)

- `crates/lance-graph-contract/src/ontology.rs` — extend with new types
- `crates/lance-graph-contract/src/property.rs` — Schema, PropertySpec, LinkSpec, EntityStore
- `crates/lance-graph/src/graph/spo/builder.rs` — SpoBuilder, label_fp
- `crates/lance-graph/src/graph/spo/nsm_bridge.rs` — pattern: Arrow export with NARS truth
- `crates/lance-graph-callcenter/src/audit.rs` — LanceAuditSink, AuditSink trait
- `crates/lance-graph-callcenter/src/rls.rs` — RlsRewriter (DataFusion OptimizerRule)
- `crates/lance-graph-callcenter/src/ontology_dto.rs` — schema-level external surface (PR #308)
- `crates/cognitive-shader-driver/src/bindspace.rs` — BindSpace SoA columns

## Out of scope (separate plans)

- Cognitive shader integration (deferred — needs OntologyDelta column on BindSpace)
- bgz-tensor full integration (deferred — see `ndarray-feature-exports-v1.md`)
- WebSocket / Phoenix realtime push (existing in `version_watcher.rs`)
