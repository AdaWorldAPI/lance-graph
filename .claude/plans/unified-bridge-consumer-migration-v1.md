# Unified-Bridge Consumer Migration — v1

> **Author:** main thread (Opus 4.7), session 2026-05-21 (branch `claude/activate-lance-graph-att-k2pHI`)
> **Status:** Draft
> **Scope:** Migrate the per-consumer bridges (`woa-bridge` — not yet started; `smb-bridge` — partial; `medcare-bridge` — in flight) onto a single shared constructor pattern that yields `lance_graph_callcenter::UnifiedBridge<NamespaceBridge>` (already shipped). The unified bridge is the DTO mapper that pulls CAM-codebook + schema + label + verbs from OGIT (Native) and OWL/DOLCE (cross-walks) and presents them as O(1) per-`OwlIdentity` lookups, with the dictionary materialised once and persisted append-only in a LanceDB column under `lance-graph-ontology`'s `lance-cache` feature.
> **Path:** `.claude/plans/unified-bridge-consumer-migration-v1.md`
> **Confidence:** Working (architecture: super-domain plan §3.9 locked the shape and the smb-bridge wiring file is the working reference); Partial (per-consumer `<repo>_unified_bridge()` constructors are absent in woa-rs + medcare-rs, smb-bridge ships its constructor against `OgitBridge` not the future `SmbBridge`).

---

## 1 — Why this exists

`lance_graph_callcenter::UnifiedBridge` is the canonical 4-stage authorize surface (chinese-wall → super-domain → role group → slot redaction) sitting on top of `lance_graph_callcenter::policy::PolicyRewriter` (`RowFilter` / `ColumnMask` / `RowEncryption` / `DifferentialPrivacy` / `Audit`). The shape is fixed; the deliverable is **per-consumer wiring** that says "given a hydrated registry + an actor role + a tenant id, hand back a `UnifiedBridge<NamespaceBridge>` locked to my OGIT namespace." Today this lives only in `smb-office-rs/crates/smb-bridge/src/unified_bridge_wiring.rs::smb_unified_bridge`. The plan extends the same shape to `woa-rs` (greenfield) and `MedCare-rs` (mid-migration), and promotes each consumer from the placeholder `OgitBridge::for_namespace(...)` parameterisation to its own typed namespace bridge once the bridge ships in `lance-graph-ontology::bridges`.

The "DTO mapper" framing in the user-facing prompt maps to two existing artefacts:

1. **Inline per-family codebook** (`super-domain-rbac-tenancy-v1.md` §3.3 `OgitFamilyTable` + `FamilyEntry`) — one 256-slot dense array per OGIT family, slot index IS `OwlIdentity.slot()`. Each slot carries `label_uri` + `kind` + `owl_characteristics` + `dolce_marker` + `axiom_blob` + `provenance` + `verbs` INLINE. ~50–200 KB per family, ~75 families ≈ 5–15 MB resident.
2. **Lance-cache persistence** (`lance-graph-ontology::lance_cache`, `lance-cache` feature) — append-only `MappingRow` log on a Lance dataset that survives process restarts. The in-memory dictionary is rebuilt from the Lance scan in `O(rows)` once at boot; the steady state is O(1) array index.

The migration plan unifies the three consumer wirings around this substrate without introducing a parallel enforcement path. `UnifiedBridge::authorize()` composes onto `PolicyRewriter` per `super-domain-rbac-tenancy-v1.md` §13.1; this plan only ships the **constructor** layer that each consumer's HTTP / FFI / route handler imports.

## 2 — Current state per consumer (2026-05-21)

| Consumer | `<repo>-bridge` crate | `<repo>-ontology` crate | `<repo>_unified_bridge()` constructor | Namespace lock | Lance-cache wired |
|---|---|---|---|---|---|
| `woa-rs` | — (greenfield) | — (greenfield) | absent | `WorkOrder` (`lance_graph_ontology::bridges::WoaBridge::NAMESPACE`) | no |
| `smb-office-rs` | `crates/smb-bridge/` (Mongo + Lance EntityStore impls; `unified_bridge_wiring.rs`) | `crates/smb-ontology/` (`customer.rs`, `mahnung.rs`, `markings.rs`) | **present** (`smb_unified_bridge`, parameterised over `OgitBridge`) | `Network` / `WorkOrder` / future `SMB` | yes (Lance feature on smb-bridge) |
| `MedCare-rs` | `crates/medcare-bridge/` (`registry.rs::MedcareRegistry`) | (none — schema declarations live in `medcare-analytics::soa_mapping`) | absent | `Healthcare` (`lance_graph_ontology::bridges::MedcareBridge::NAMESPACE`) | yes (gated `--features ontology`) |


### 2.1 What "in flight" means for medcare-bridge

Per `MedCare-rs/crates/medcare-bridge/src/lib.rs` "## What is NOT wired yet" block, the medcare-side gaps are:

1. `medcare-analytics::ontology_dto::MedcareOntology::default()` calls a now-broken `upstream_medcare_ontology()` no-arg form → **lance-phase2 build is currently broken at `ontology_dto.rs:85`**. This is the headline blocker; nothing else in the medcare plan moves until it's fixed.
2. `medcare-analytics::soa_mapping::ALL_SCHEMAS` iteration in `medcare-server::state::rls_registry` is NOT yet replaced by `OntologyRegistry::enumerate("Healthcare")`. Without the replacement, the 3 NEW entities surfaced by OGIT (Treatment, Visit, VitalSign — beyond the legacy 4 Patient/Diagnosis/LabResult/Prescription) have no RLS policy → fail-OPEN bypass risk. Safety-critical.
3. `MulThresholdProfile::MEDICAL` consumption at the gate site (per `D-ONTO-V5-9` in lance-graph#355).
4. `ontology_context_id`-keyed RLS extension (third axis after `(table, praxis_id)`) per §73 SGB V Überweisung shape.

These four items are the medcare-rs side of this plan's Tier C.

## 3 — Target shape: one constructor per consumer

Every consumer ends up exposing **one** public function in its `<repo>-bridge` crate:

```rust
// woa-rs/crates/woa-bridge/src/unified_bridge_wiring.rs (NEW)
pub fn woa_unified_bridge(
    registry: Arc<OntologyRegistry>,
    actor_role: &'static str,                              // mandant / sachbearbeiter / kasse / buchhaltung / admin
    tenant: TenantId,                                       // mapped from Mandant.id
) -> Result<UnifiedBridge<WoaBridge>, lance_graph_ontology::Error>;

// smb-office-rs/crates/smb-bridge/src/unified_bridge_wiring.rs (EXISTS)
pub fn smb_unified_bridge(
    registry: Arc<OntologyRegistry>,
    namespace: &str,                                        // "Network" | "WorkOrder" | future "SMB"
    actor_role: &'static str,                              // accountant / auditor / admin (D-SDR-2: + tax_clerk / partner / client_user / audit_observer)
    tenant: TenantId,                                       // mapped from auth::TenantId.praxis_id or kdnr
) -> Result<UnifiedBridge<OgitBridge>, lance_graph_ontology::Error>;

// MedCare-rs/crates/medcare-bridge/src/unified_bridge_wiring.rs (NEW)
pub fn medcare_unified_bridge(
    registry: Arc<OntologyRegistry>,
    actor_role: &'static str,                              // physician / nurse / cashier / researcher / hipaa_audit / admin
    tenant: TenantId,                                       // mapped from Praxis.id
) -> Result<UnifiedBridge<MedcareBridge>, lance_graph_ontology::Error>;
```

The signature shape is uniform; the parameterisation differs by which concrete `NamespaceBridge` impl each consumer uses. smb-office-rs is the **outlier**: it ships against `OgitBridge::for_namespace(...)` because `SmbBridge` does not yet exist in `lance-graph-ontology::bridges` (see `unified_bridge_wiring.rs` lines 9-14). woa-rs + MedCare-rs ship against their dedicated `WoaBridge` / `MedcareBridge` directly.

### 3.1 Why uniform signature matters

Each consumer's HTTP / FFI / route-handler layer imports the constructor and stops there. Call sites stay unchanged when the bridge type parameter switches (e.g., when smb-office-rs promotes to a dedicated `SmbBridge` locked to `OGIT/NTO/SMB/`). The 4-stage authorize flow (`authorize(owl, row_tenant, op) → RowAccess`) lives on the `UnifiedBridge<B>` regardless of which `B` is plugged in. The DataFusion predicate lowering (`super-domain-rbac-tenancy-v1.md` §3.10) is independent of `B`.


## 4 — DTO mapper layer (CAM codebook + schema + label, O(1) per OwlIdentity)

The "unified as DTO mapper" framing in the user-facing prompt materialises as the **per-family codebook tables** described in `super-domain-rbac-tenancy-v1.md` §3.3. Concretely:

```
Hydration phase (boot or first registry.resolve()):
    AdaWorldAPI/OGIT/NTO/<Namespace>/entities/*.ttl
        + AdaWorldAPI/OGIT/NTO/<Namespace>/verbs/*.ttl
        + AdaWorldAPI/OGIT/NTO/<Namespace>/attributes/*.ttl
        + OWL/DOLCE cross-walks from MetaAnchors
    │
    ▼
    parse_ttl_directory_with_provenance       ←  in lance-graph-ontology::ttl_parse
    │
    ▼
    MappingProposal stream                     ←  in lance-graph-ontology::proposal
    │
    ▼
    OntologyRegistry.append_proposals          ←  in lance-graph-ontology::registry
        ├─ in-memory dict keyed by (bridge_id, public_name)
        └─ Lance dataset (under `lance-cache` feature) — append-only
    │
    ▼
    Per-family bake: OgitFamilyTable[OgitFamily] { entries: [Option<FamilyEntry>; 256], codebook: PerFamilyCodebook }
                                                                      ↑                              ↑
                                                            slot index IS OwlIdentity.slot()    5-8 bit centroids
                                                                                                (per-family local)

Hot path (every Cypher/SPARQL/Gremlin query that touches a row):
    LanceDB row → OwlIdentity (u16) → owl.family() → OgitFamilyTable lookup → FamilyEntry { label_uri, kind, axiom_blob, verbs, codebook entry }
                                          ↑                                       ↑
                                  high-byte index into                  O(1) array index into the 256-slot
                                  static [OgitFamilyTable; 256]            dense array (no map lookup)
```

**Per-row LanceDB overhead is 6 bytes** (tenant_id u32 + owl_id u16). The codebook + label + schema + verbs do NOT live on each row — they live ONCE in the static OgitFamilyTable, addressed by the 2-byte owl_id. This is the "O(1) lookup cached in lancedb column" property: the table is materialised at boot, persisted by the `lance-cache` feature so re-hydration is O(rows) once not O(rows × ttl-parse-cost), and consulted by the same masked u16 compare DataFusion lowers Cypher MATCH to (§3.10).

### 4.1 OWL/DOLCE cross-walk surface

The `MetaAnchors` field (super-domain-rbac §3.5) on each `SuperDomainEntry` carries pointers to upper ontologies:

| Cross-walk | Field | Consulted when |
|---|---|---|
| Foundry ObjectType | `foundry_object_type: Option<&'static str>` | Customer requests Foundry-shape export |
| OWL upper class | `owl_upper_class: Option<&'static str>` | OWL reasoner runs over the same registry |
| DOLCE marker | `dolce_marker: DolceMarker` (Endurant / Perdurant / Quality / Abstract) | Per-row philosophical-category tagging |
| Wikidata QID | `wikidata_qid: Option<u64>` | Wikidata sync / sameAs link generation |

These are **interop crutches**, not the hot path. The 4-stage `authorize()` never consults them; the per-family codebook is the runtime fast lane. OWL/DOLCE values ride in `FamilyEntry.axiom_blob` for slots that carry semantic obligations (functional / transitive / inverseFunctional / etc., per `GLUE_LAYER_OGIT_TO_OWL_SPEC.md` 1-byte bitfield).

### 4.2 Lance-cache persistence (the "cached in lancedb column" property)

`lance-graph-ontology::lance_cache` (gated on `lance-cache` feature) writes a Lance dataset under `<state-dir>/.ontology-cache.lance/` with the schema:

| Column | Type | Purpose |
|---|---|---|
| `bridge_id` | utf8 | Which tenant bridge (`woa` / `medcare` / `smb` / `ogit` / `spear` / `sharepoint`) |
| `namespace` | utf8 | OGIT namespace (`WorkOrder` / `Healthcare` / `Network` / ...) |
| `public_name` | utf8 | Public-facing name (`Customer`, `WorkOrder`, `Position`, ...) |
| `ogit_uri` | utf8 | Canonical OGIT URI (`ogit.WorkOrder:Customer`) |
| `kind` | u8 | `SchemaKind` (Entity / Edge / Attribute) |
| `dolce_marker` | u8 | `DolceMarker` |
| `owl_characteristics` | u8 | bitfield from `GLUE_LAYER_OGIT_TO_OWL_SPEC` |
| `provenance` | utf8 | `dcterms:source` lineage (carries off-label OSLC fit etc.) |
| `axiom_blob` | binary | OWL subClassOf / equivalentClass axioms |
| `verb_slots` | binary | outgoing verb slots within this family |
| `centroid_blob` | binary | 5-8 bit per-family codebook centroid (CAM-PQ shape) |
| `proposal_sha256` | binary[32] | idempotent dedup key — same proposal landed twice yields one row not two |
| `appended_at` | timestamp[ms] | append timestamp for audit chain |

Reads are append-only; writes go through `MappingProposal::sha256()`-keyed dedup. Boot path: `OntologyRegistry::hydrate_once_sync(ttl_root, &[namespace])` walks TTL once and builds the in-memory registry; `lance_cache::append_proposals(...)` mirrors it to the Lance dataset; subsequent boots can skip the TTL walk by reading the Lance scan first and only re-TTL-walking if the on-disk root checksum mismatches.

**This is what makes the codebook "O(1) cached in lancedb column":** the column store IS the cache; reads are scan-then-build-in-memory once at boot; lookups are array-index on `OwlIdentity.slot()` after.


## 5 — Deliverables

### Tier A — Shared substrate (lance-graph workspace)

Most of Tier A is already shipped or scoped under `super-domain-rbac-tenancy-v1.md` Tier A (D-SDR-1..5). This plan adds three follow-ons:

- **D-UB-1** — Stable public constructor pattern doc at `lance-graph/.claude/knowledge/unified-bridge-consumer-pattern.md`. Specifies the signature shape (§3 above), the migration path from `OgitBridge`-parameterised → dedicated bridge, the deprecation strategy when a consumer switches its type parameter. ~250 lines markdown. READ BY: consumer-crate authors before adding a new bridge consumer.
- **D-UB-2** — `lance-graph-ontology::bridges::SmbBridge` skeleton locked to `Network` (placeholder until `OGIT/NTO/SMB/` namespace exists). Mirrors `WoaBridge` (`bridges/woa_bridge.rs:1-50`) — `bridge_id="smb"`, `g_lock = Network` for now, `BridgeFromRegistry` impl. ~50 LOC + 2 tests. Unblocks smb-bridge's promotion from `OgitBridge` parameterisation to `SmbBridge` per `unified_bridge_wiring.rs` lines 9-14.
- **D-UB-3** — `lance-graph-ontology::lance_cache::ontology_cache_schema()` returning a stable Arrow schema (the §4.2 table) + a `LanceCacheBootStrategy` (TTL-first / Lance-first / TTL-with-Lance-mirror) selector. ~150 LOC + 6 tests. Needed because each consumer makes a different choice (woa-rs Lance-first for cold-start latency, MedCare-rs TTL-first for HIPAA audit traceability).

### Tier B — Per-consumer constructors

- **D-UB-4** — `woa-rs/crates/woa-bridge/src/unified_bridge_wiring.rs::woa_unified_bridge`. ~50 LOC mirroring `smb_unified_bridge` but parameterised over `WoaBridge` (not `OgitBridge`). Depends on Tier A of `lance-graph-in-woa-rs-v1.md` shipping the `woa-bridge` + `woa-ontology` crate scaffolding first. + 2 integration tests (constructor errors on unhydrated registry; round-trip resolution of `Customer` → `ogit.WorkOrder:Customer`).
- **D-UB-5** — `smb-office-rs/crates/smb-bridge/src/unified_bridge_wiring.rs::smb_unified_bridge` already exists; this deliverable is the **type-parameter swap** from `UnifiedBridge<OgitBridge>` → `UnifiedBridge<SmbBridge>` after D-UB-2 lands. ~15 LOC change + 1 test asserting the previous OgitBridge surface still resolves correctly through the new SmbBridge type parameter (no regression at call sites).
- **D-UB-6** — `MedCare-rs/crates/medcare-bridge/src/unified_bridge_wiring.rs::medcare_unified_bridge`. NEW file. ~50 LOC parameterised over `MedcareBridge`. Depends on D-UB-7 (the `ontology_dto.rs:85` build-fix) landing first. + 2 integration tests.
- **D-UB-7** — Fix `medcare-analytics::ontology_dto::MedcareOntology::default()` to call a registry-driven constructor instead of the broken `upstream_medcare_ontology()` no-arg form. This is the lance-phase2 build-fix that unblocks all subsequent medcare work. ~30 LOC fix + 2 tests covering the constructor's failure mode (unhydrated registry) and the success path. Highest priority across this entire plan; everything medcare-side blocks on it.

### Tier C — RLS coverage closure (medcare-rs only — safety-critical)

- **D-UB-8** — Replace `medcare-analytics::soa_mapping::ALL_SCHEMAS` iteration in `medcare-server::state::rls_registry` with `OntologyRegistry::enumerate("Healthcare")`. Adds RLS coverage for the 3 NEW Healthcare entities (Treatment, Visit, VitalSign) that OGIT surfaces beyond the legacy 4 (Patient, Diagnosis, LabResult, Prescription). ~80 LOC + 4 tests covering full coverage (`Test C — rls_coverage_parity_with_all_schemas` in `medcare-bridge/Cargo.toml [dev-dependencies]` already references this contract). Blocking on D-UB-7. **Fail-OPEN bypass risk if shipped without this** — the unmapped entities have no row-level policy.
- **D-UB-9** — `MulThresholdProfile::MEDICAL` consumption at the gate site (per `D-ONTO-V5-9` in lance-graph#355). ~60 LOC + 2 tests. Cross-references `super-domain-rbac-tenancy-v1.md` §13.5 (researcher role: anonymized projection only).
- **D-UB-10** — `ontology_context_id`-keyed RLS extension. Adds a third axis to the `(table, praxis_id)` tuple per §73 SGB V Überweisung shape. ~100 LOC + 4 tests covering the cross-tenant referral case where a Patient row's `ontology_context_id` differs from the requesting tenant's id.

### Tier D — Cross-consumer parity tests

- **D-UB-11** — `lance-graph-ontology/tests/integration_unified_bridge_parity.rs`. Spawns three consumers in-process (in-memory `OntologyRegistry` hydrated from the same TTL root) and asserts that `<woa|smb|medcare>_unified_bridge(...)` each return a `UnifiedBridge` that resolves the same shared concepts (e.g., `ogit.Auth:User`) to the same `OwlIdentity` across all three. ~120 LOC + 4 tests. Prevents per-consumer drift from sneaking into the dictionary layer.


## 6 — Sequencing

```
        ┌─────────────────────────────────────────────────┐
        │ D-UB-7  fix ontology_dto.rs:85 (lance-phase2)   │  ← unblocks medcare-side
        │   30 LOC + 2 tests · MedCare-rs                 │
        └────────────────────────┬────────────────────────┘
                                 │
   ┌────────────────────────┬───-┴────────────────────────┐
   │                        │                             │
┌──▼───────────────┐ ┌──────▼──────────┐    ┌─────────────▼──────────────────┐
│ D-UB-1  doc      │ │ D-UB-2  SmbBridge│   │ D-UB-3  lance_cache schema/strat│
│ ~250 lines       │ │ skeleton ~50 LOC │   │ ~150 LOC + 6 tests             │
│ lance-graph      │ │ lance-graph      │   │ lance-graph                     │
└──────────────────┘ └───────┬──────────┘    └────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
       ┌──────▼──────┐ ┌─────▼────────┐ ┌──▼────────────────────────┐
       │ D-UB-4 woa  │ │ D-UB-5 smb   │ │ D-UB-6 medcare            │
       │ unified-    │ │ swap to      │ │ unified_bridge_wiring     │
       │ bridge      │ │ SmbBridge    │ │ NEW file (~50 LOC + 2 t)  │
       │ NEW (~50 +2)│ │ (~15 LOC +1) │ │ MedCare-rs                │
       └─────────────┘ └──────────────┘ └────────────┬──────────────┘
                                                     │
                                  ┌──────────────────┼──────────────────┐
                                  │                  │                  │
                              ┌───▼──────┐ ┌─────────▼─────┐ ┌──────────▼──────┐
                              │ D-UB-8   │ │ D-UB-9        │ │ D-UB-10         │
                              │ RLS      │ │ MUL MEDICAL   │ │ ontology_context│
                              │ coverage │ │ ~60 LOC +2 t  │ │ _id RLS axis    │
                              │ 80 LOC+4 │ │ MedCare-rs    │ │ 100 LOC+4 tests │
                              └──────────┘ └───────────────┘ └─────────────────┘

                                  └────────────────┬───────────────┘
                                                   │
                                       ┌───────────▼────────────┐
                                       │ D-UB-11 cross-consumer │
                                       │ parity test  120 LOC+4 │
                                       │ lance-graph            │
                                       └────────────────────────┘
```

The critical path is D-UB-7 → D-UB-6 → D-UB-8. Everything else can fan out in parallel once Tier A lands.

## 7 — Cross-references

- `super-domain-rbac-tenancy-v1.md` §3.9 — `UnifiedBridge::authorize` (the trait this plan wires consumers onto)
- `super-domain-rbac-tenancy-v1.md` §13.1 — `lance_graph_callcenter::policy::PolicyRewriter` (the enforcement chain UnifiedBridge composes onto, not parallels)
- `super-domain-rbac-tenancy-v1.md` §3.3 — `OgitFamilyTable` + `FamilyEntry` (the DTO mapper's storage shape)
- `super-domain-rbac-tenancy-v1.md` §3.5 — `MetaAnchors` (OWL/DOLCE/Foundry/Wikidata cross-walks)
- `super-domain-rbac-tenancy-v1.md` §14 — Bridge harvest from medcare/sharepoint as canonical pattern source
- `super-domain-rbac-tenancy-v1.md` §18.7 — `D-SDR-35..39` (medcare-rs parity ingest endpoints; orthogonal to this plan but adjacent)
- `lance-graph-in-woa-rs-v1.md` — depends on Tier A of that plan for crate scaffolding
- `lance-graph-in-smb-office-rs-v1.md` — D-UB-5 (type-parameter swap) is the smb-office-rs sequel to that plan's Tier B
- `lance-graph-in-medcare-rs-v1.md` — D-UB-7..10 are the medcare-rs deliverables for unblocking lance-phase2 and closing the RLS coverage gap

## 8 — Open questions

1. **SmbBridge namespace name** — The unified_bridge_wiring comment names a future `OGIT/NTO/SMB/`. Is `SMB` the locked name or do we want `Steuerberater` / `Buchhaltung` / something more specific? Affects D-UB-2.
2. **Lance-cache vs TTL-first per consumer** — woa-rs is greenfield so Lance-first is cheap; MedCare-rs needs HIPAA audit traceability so TTL-first with explicit dataset-as-mirror is safer. D-UB-3 names the selector; the per-consumer plans confirm which strategy each ships with.
3. **D-UB-8 fail-OPEN window** — Is there a way to gate `medcare-server` boot on a "registry enumeration matches expected entity count" check so the 3 NEW Healthcare entities (Treatment / Visit / VitalSign) cannot land without RLS policy? Probably yes via `HydrationReport` already returned by `medcare_bridge::registry::MedcareRegistry::hydrate_with_report`.
4. **Cross-consumer parity test scope** — D-UB-11 asserts `ogit.Auth:User` resolves identically across three consumers, but what about cross-namespace concepts (a Customer in WorkOrder vs a Patient in Healthcare — both have addresses)? Out of scope for v1; the same parity test would catch it later if needed.

## 9 — Status

- **Architecture:** Working — UnifiedBridge ships, smb_unified_bridge is the reference wiring, the three target signatures are uniform.
- **Tier A:** Not started. D-UB-1..3 author this session or next.
- **Tier B:** D-UB-4 + D-UB-6 not started. D-UB-5 is a 15-LOC swap once D-UB-2 lands.
- **Tier C:** D-UB-7 is the blocker; D-UB-8 is the safety-critical follow-on. D-UB-9..10 land after.
- **Tier D:** D-UB-11 ships last as the regression gate.

**Confidence:** Working. The substrate is locked in `super-domain-rbac-tenancy-v1.md`; this plan only formalises the per-consumer wiring + closes the 4 medcare-rs gaps named in `medcare-bridge/src/lib.rs`.

## 10 — One-line summary

> Three consumers, three `<repo>_unified_bridge()` constructors, one `lance_graph_callcenter::UnifiedBridge<NamespaceBridge>` shape, one per-family codebook materialised once at boot and cached as a LanceDB append-only column. The fail-OPEN risk on the 3 newly-OGIT-surfaced Healthcare entities is the safety-critical headline; D-UB-7 → D-UB-6 → D-UB-8 is the critical path.

