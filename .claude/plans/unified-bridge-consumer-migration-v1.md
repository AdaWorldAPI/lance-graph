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

### 4.1 OWL/DOLCE cross-walk — the source material for the OGIT mapping

**The cross-walk table below is not an interop crutch — it is the source material from which `lance-graph-ontology` *constructs* the OGIT/OWL/DOLCE mapping.** External standards (DOLCE+DUL, OWL-Time, PROV-O, QUDT, schema.org, FIBO, SKOS, …) ride in as TTL/OWL artifacts; the hydrators ingest them; the OGIT canonical classification (per-family codebook of §3.3, edge whitelists, inherits-from chain at `OGIT::*_V1` slots) is the *synthesis* that emerges. lance-graph-ontology is the construction site; the hydrators are the construction tool; the cross-walk standards are the bricks.

Direction of build:

```
External standards          →   Hydrator (consumes TTL/OWL/XSD/SKR)  →   OGIT canonical surface
─────────────────────────       ──────────────────────────────────       ──────────────────────────────
DOLCE+DUL.owl             ──►  hydrate_dolce                       ──►  OGIT::DOLCE_V1 bundle
  + DUL extensions                                                       (root, inherits_from: None)
                                                                         + 17-IRI edge whitelist

OWL-Time.ttl              ──►  hydrate_owltime                     ──►  OGIT::OWLTIME_V1 bundle
PROV-O.ttl                ──►  hydrate_provo                       ──►  OGIT::PROVO_V1 bundle
QUDT 2.1                  ──►  hydrate_qudt                        ──►  OGIT::QUDT_V1 bundle
schema.org.ttl            ──►  hydrate_schemaorg                   ──►  OGIT::SCHEMAORG_V1 bundle
                                  (all four inherits_from: Some(OGIT::DOLCE_V1.0))

SKOS.ttl                  ──►  hydrate_skos                        ──►  OGIT::SKOS_V1 bundle
FIBO-FND                  ──►  hydrate_fibo_fnd                    ──►  OGIT::FIBO_FND_V1 bundle
FIBO-BE                   ──►  hydrate_fibo_be                     ──►  OGIT::FIBO_BE_V1 bundle
                                                                         (inherits FND, not DOLCE direct)

ISO Schematron rules      ──►  SchematronHydrator                  ──►  rule-pattern bundle entries
XSD type defs             ──►  XsdHydrator                         ──►  type-def bundle entries
ZUGFeRD/Factur-X (EN16931)──►  hydrate_zugferd + hydrate_zugferd_rules
                               (composes XsdHydrator + Schematron)  ──►  ZUGFeRD-specific bundle
DATEV SKR 03/04/03-Bau    ──►  SkrHydrator + hydrate_skr*          ──►  account-hierarchy bundles
                                                                         (SKR03_IRI_PREFIX etc.)

                          The OGIT mapping IS the union of:
                            • OGIT::*_V1 G-slots populated by the above
                            • the inherits-from DAG that chains them
                            • the per-family codebook (§3.3) with
                              FamilyEntry { label_uri, kind, dolce_marker,
                                             owl_characteristics, axiom_blob,
                                             provenance, verbs }
                              at every OwlIdentity slot
```

The `MetaAnchors` struct on `SuperDomainEntry` (super-domain-rbac §3.5) is the **runtime read surface** over this constructed mapping — the field shapes are:

| MetaAnchors field | What it points at | Built from which hydrator's output | Read at runtime by |
|---|---|---|---|
| `foundry_object_type: Option<&'static str>` | Foundry ObjectType string like `"PhysicalSystem"` | Cross-walk authored from `hydrate_schemaorg` + `hydrate_fibo_be` outputs (the upper-class anchors Foundry maps to). NO auto-population — per `super-domain-rbac-tenancy-v1.md` §10 OQ-1 ("Foundry ObjectType cross-walk targets … need product-side input"). | Foundry-shape export path; not by `authorize()`. |
| `owl_upper_class: Option<&'static str>` | OWL upper-class IRI like `"BiomedicalOntology"` | `hydrate_dolce`'s `OGIT::DOLCE_V1` bundle (the 17-IRI cascade preserves `rdfs:subClassOf` / `owl:equivalentClass` chains). | OWL reasoner running over the registry. |
| `dolce_marker: DolceMarker` (Endurant / Perdurant / Quality / Abstract) | DOLCE primary category at the per-row philosophical-category tag | `hydrate_dolce` populates DOLCE_V1; downstream `hydrate_owltime` / `provo` / `qudt` / `schemaorg` align via `inherits_from: Some(OGIT::DOLCE_V1.0)` + `dul:isClassifiedBy` / `rdfs:subClassOf dul:Event\|Object\|Quality\|Abstract`. **DolceMarker enum naming open question:** canonical DOLCE+DUL renamed `Endurant → Object` and `Perdurant → Event` in the DUL header — the enum variants either stay on the legacy `Endurant/Perdurant` (with downstream consumers aware of the rename) or migrate to `Object/Event`. Decide before D-UB-3 + reflect in the §4.3 hydrator inventory. | Per-row tagging in `FamilyEntry.dolce_marker`. |
| `wikidata_qid: Option<u64>` | Wikidata QID like `Q11190` | NOT YET BUILT — no `hydrate_wikidata` hydrator exists. Adding one is ~50 LOC of `OwlHydrator { g: OGIT::WIKIDATA_V1.0, inherits_from: Some(OGIT::DOLCE_V1.0), … }` glue once a tenant requests Wikidata sync. | Wikidata sync / `sameAs` link generation. |

The `MetaAnchors` fields are **the read API over what the hydrators built**. The 4-stage `authorize()` doesn't consult them directly — it operates on `OwlIdentity` against the per-family codebook. The cross-walks are consulted when interop with an external system (Foundry export, OWL reasoner, Wikidata sync) is requested.

OWL property characteristics (functional / transitive / inverseFunctional / etc., per `GLUE_LAYER_OGIT_TO_OWL_SPEC.md` 1-byte bitfield in `FamilyEntry.owl_characteristics`) flow the same way — `hydrate_dolce` + the L2/L3 hydrators preserve the OWL axioms in `FamilyEntry.axiom_blob`, and the bitfield is the runtime cache over them.

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

### 4.3 Hydrator inventory (post-PR #407) — the construction tool

The hydrators that build the OGIT/OWL/DOLCE mapping (per §4.1) are now concrete in `lance_graph_ontology::hydrators::*` (PR #407 + the ~11 preceding feature commits; PR #408 wires the corresponding `NamespaceRegistry::seed_defaults` entries so the OGIT G-slots register at boot). Every hydrator follows the same shape — a free function `hydrate_<name>(registry: &OntologyRegistry) -> Result<u32, HydrateErr>` that constructs an `OwlHydrator { g: OGIT::<SLOT>.0, version: OGIT::<SLOT>.1, domain_name: "<name>".to_string(), inherits_from: Option<u32>, starting_entity_id: 100 }`, calls `.hydrate(ttl_path, registry)` (or `.hydrate_many(&paths, registry)` for multi-file bundles like DOLCE+DUL+extensions), then registers an edge-IRI whitelist via `registry.register_edge_types(g, &EDGE_WHITELIST)`. The return value is the OGIT `G` slot u32 the bundle landed in — i.e., the address of the synthesised mapping in OGIT space.

**Generic substrate (the bO-* scaffold every hydrator instantiates):**

| Type | Path | Role |
|---|---|---|
| `OwlHydrator` | `lance_graph_ontology::hydrators::owl::OwlHydrator` | The generic struct with `g`, `version`, `domain_name`, `inherits_from`, `starting_entity_id`. Reads OWL/Turtle via the existing TTL parser, emits `MappingProposal`s into the registry. |
| `MetaStructureHydrator` | `lance_graph_ontology::hydrators::owl::MetaStructureHydrator` (trait) | Trait the per-ontology glue implements; `OwlHydrator` is the default impl. Pattern-D meta-structure hydration. |
| `ContextBundle` | `lance_graph_ontology::hydrators::owl::ContextBundle` | Typed bundle keyed by an OGIT `G` slot. One bundle per hydrator output. |
| `EntityId`, `OntologySlot`, `HydrateErr` | `lance_graph_ontology::hydrators::owl::*` | Per-entity slot index inside a bundle; the error enum. |

**Layered ontologies (L1 root → L4 sector):**

| Layer | Hydrator function | OGIT slot | `inherits_from` | TTL artifact | Edge whitelist size |
|---|---|---|---|---|---|
| **L1 upper** | `hydrate_dolce` (+ `hydrate_dolce_from`, `hydrate_dolce_from_many` for tests / multi-file) | `OGIT::DOLCE_V1.0` | `None` (root) | `data/ontologies/dul.ttl` + `dul-extensions/{conceptualization.owl, lmm-l2.owl}` | 17 IRIs (rdfs:subClassOf + owl:equivalentClass/disjointWith + DnS classify/role-binding + dul:hasPart/isPartOf + dul:hasTimeInterval/isObservableAt) |
| **L2 universal** | `hydrate_owltime` (+ `_from`) | `OGIT::OWLTIME_V1.0` | `Some(OGIT::DOLCE_V1.0)` | `data/ontologies/owltime.ttl` | — |
| **L2 universal** | `hydrate_provo` (+ `_from`) | `OGIT::PROVO_V1.0` | `Some(OGIT::DOLCE_V1.0)` | `data/ontologies/provo.ttl` | 22 IRIs (subClassOf/subPropertyOf + 8 load-bearing PROV relations: wasGeneratedBy/used/wasDerivedFrom/wasAttributedTo/wasAssociatedWith/wasInformedBy/actedOnBehalfOf/wasInvalidatedBy + qualified* + activity-lifecycle + delegation) |
| **L2 universal** | `hydrate_qudt` (+ `_from`) | `OGIT::QUDT_V1.0` | `Some(OGIT::DOLCE_V1.0)` | `data/ontologies/qudt/*` (+ quantitykinds catalogue, bO-4) | — |
| **L3 commercial-web** | `hydrate_schemaorg` (+ `_from`) | `OGIT::SCHEMAORG_V1.0` | `Some(OGIT::DOLCE_V1.0)` | `data/ontologies/schemaorg.ttl` | — |
| **Sector** | `hydrate_skos` (+ `_from`) | `OGIT::SKOS_V1.0` | `Some(OGIT::DOLCE_V1.0)` | `data/ontologies/skos.ttl` (+ DUL extension modules per bO-5) | — |
| **Sector — finance** | `hydrate_fibo_fnd` (+ `_from`) | `OGIT::FIBO_FND_V1.0` | `Some(OGIT::DOLCE_V1.0)` | FIBO Foundations | — |
| **Sector — finance** | `hydrate_fibo_be` (+ `_from`) | `OGIT::FIBO_BE_V1.0` | `Some(OGIT::FIBO_FND_V1.0)` | FIBO Business Entities | — |

**Dedicated (non-OWL) hydrators:**

| Type | Path | Domain | Role |
|---|---|---|---|
| `SchematronHydrator` | `lance_graph_ontology::hydrators::schematron::SchematronHydrator` | ISO Schematron rule sets | Hydrates Schematron rule patterns (asserts + reports) as ContextBundle entries; used by `hydrate_zugferd_rules` + `hydrate_zugferd_rules_from` for the EN16931 compliance rules ZUGFeRD/Factur-X invoices must satisfy. |
| `XsdHydrator` | `lance_graph_ontology::hydrators::xsd::XsdHydrator` (+ `collect_xsd_files()` helper) | XML Schema | Hydrates XSD type definitions as ContextBundle entries; used by `hydrate_zugferd` + `hydrate_zugferd_from` for the ZUGFeRD/Factur-X EN16931 invoice schema (PR-bO-16). |
| `SkrHydrator` | `lance_graph_ontology::hydrators::skr::SkrHydrator` | DATEV SKR charts of accounts | Hydrates the SKR account hierarchy from the CSV data files (`data/skr/SKR03.csv`, `SKR04.csv`, `SKR03_bau.csv`). Consumed by `hydrate_skr03` / `hydrate_skr04` / `hydrate_skr03_bau` (and their `_from` test variants). Three IRI-prefix constants exposed: `SKR03_IRI_PREFIX`, `SKR04_IRI_PREFIX`, `SKR03_BAU_IRI_PREFIX`. |

**Re-export surface (consumers import from the crate root, not the submodule):**

```rust
// Single import for the whole hydrator surface:
use lance_graph_ontology::{
    // generic
    OwlHydrator, MetaStructureHydrator, ContextBundle, EntityId, OntologySlot, HydrateErr,
    // layered
    hydrate_dolce, hydrate_dolce_from, hydrate_dolce_from_many,
    hydrate_owltime, hydrate_owltime_from,
    hydrate_provo, hydrate_provo_from,
    hydrate_qudt, hydrate_qudt_from,
    hydrate_schemaorg, hydrate_schemaorg_from,
    hydrate_skos, hydrate_skos_from,
    hydrate_fibo_fnd, hydrate_fibo_fnd_from,
    hydrate_fibo_be, hydrate_fibo_be_from,
    // sector + dedicated
    SchematronHydrator,
    XsdHydrator, collect_xsd_files,
    SkrHydrator,
    hydrate_skr03, hydrate_skr03_from,
    hydrate_skr04, hydrate_skr04_from,
    hydrate_skr03_bau, hydrate_skr03_bau_from,
    SKR03_IRI_PREFIX, SKR04_IRI_PREFIX, SKR03_BAU_IRI_PREFIX,
    hydrate_zugferd, hydrate_zugferd_from,
    hydrate_zugferd_rules, hydrate_zugferd_rules_from,
};
```

**How this plan's deliverables use the hydrator surface:**

- **D-UB-1** (consumer-pattern doc) names the `hydrate_<name>(registry)` shape as the producer side of `OntologyRegistry`. Consumers do not call hydrators directly — the bridge constructor (`<repo>_unified_bridge`) takes an already-hydrated `Arc<OntologyRegistry>` and the deployment code chooses which hydrators to run at boot.
- **D-UB-2** (`SmbBridge` skeleton) declares no hydrator dependency; SMB locks to a future `OGIT::SMB_V1` slot. Once `OGIT/NTO/SMB/` ships, an `hydrate_smb_namespace()` follows the same `OwlHydrator { inherits_from: Some(OGIT::DOLCE_V1.0), ... }` pattern.
- **D-UB-3** (`lance_cache::ontology_cache_schema()` + `LanceCacheBootStrategy`) persists the per-hydrator `ContextBundle` output as the Lance dataset rows. Each row carries `bridge_id`, `namespace`, `public_name`, `ogit_uri`, `kind`, `dolce_marker`, `owl_characteristics`, `provenance`, `axiom_blob`, `verb_slots`, `centroid_blob` (§4.2 table) — populated by the hydrators that ran at boot.
- **D-UB-4..6** (per-consumer constructors) all take `Arc<OntologyRegistry>` already hydrated by `hydrate_dolce` + the appropriate L2/L3/sector hydrators for that consumer's namespace. Concretely:
  - **woa-rs** boots: `hydrate_dolce` + `hydrate_provo` (audit trails, GoBD) + `hydrate_qudt` (Mengen, Stundenzahl) + `hydrate_schemaorg` (Customer / Invoice cross-walks) + `hydrate_fibo_fnd` (financial primitives) + `hydrate_skr03`/`hydrate_skr04` (chart of accounts) + future `OGIT/NTO/WorkOrder` namespace hydration.
  - **smb-office-rs** boots: same as woa-rs minus the WorkOrder namespace, plus `hydrate_skr03_bau` (Baugewerbe) for construction tenants and `hydrate_zugferd` + `hydrate_zugferd_rules` for X-Rechnung output validation.
  - **MedCare-rs** boots: `hydrate_dolce` + `hydrate_owltime` (Visit / Treatment chronology) + `hydrate_provo` (HIPAA audit trail) + `hydrate_qudt` (LabValue units, VitalSign measurements) + `hydrate_skos` (clinical-code thesauri — ICD-10 mappings) + future `OGIT/NTO/Healthcare` hydration.

The bridge constructor never sees `OwlHydrator` directly — it sees a hydrated `OntologyRegistry`. The deployment / `main.rs` of each consumer chooses the hydration menu.

### 4.4 The hydrators are the spine that makes inheritance O(1) from family buckets

The shape that earns the cost-model gains: **`inherits_from` + per-family codebook + family-bucket dense array together make schema, label, and codebook inheritance O(1) at lookup time.** Spelled out:

| Inheritance flavour | Substrate | Lookup cost | Pre-bake step |
|---|---|---|---|
| **Schema inheritance** (a `Patient` in Healthcare resolves to `dul:Object → dul:Agent → dul:PhysicalAgent` per the DOLCE chain) | Each hydrator's `inherits_from: Option<u32>` chains its OGIT G-slot to its parent. `OGIT::DOLCE_V1` is the root (None). At hydration time, `OntologyRegistry` walks the `rdfs:subClassOf` / `owl:equivalentClass` chain ONCE and **flattens** it into `FamilyEntry.axiom_blob`. | O(1) — one masked u16 compare into `OgitFamilyTable[OgitFamily.0 as usize]`, then `.entries[OwlIdentity.slot() as usize]`. **Zero chain-walks at query time.** The flattened chain rides in the static blob. | At hydration: walk the OWL/DOLCE/PROV-O `subClassOf` graph from leaf to root, materialise as `axiom_blob`. Per-family lock-in by `inherits_from`. |
| **Label inheritance** (`rdfs:label@de` / `rdfs:label@en` propagated from parent when child lacks an override) | `hydrate_<name>` reads `rdfs:label` per-locale at TTL parse time. When a slot has no own label, the inherits-from walker copies the parent's label into `FamilyEntry.label_uri` (or `.label_de` / `.label_en` once those columns ride the FamilyEntry — see `lance-cache` schema §4.2). | O(1) — same `OgitFamilyTable` array index → `FamilyEntry.label_*` field read. **Zero parent lookup at query time.** | At hydration: per-locale label collapse during the subClassOf walk. |
| **Codebook inheritance** (per-family centroid reuses when a slot's content distribution overlaps the parent family's) | `PerFamilyCodebook` (§3.3) is sized 5-8 bit per family; when a child family's slot range overlaps the parent's centroid space, the child's `centroid_blob` REFERENCES the parent's centroid index directly (no per-child re-quantisation). The reference is a u8 offset into the parent codebook. | O(1) — `FamilyEntry.centroid_blob` → `parent_family.codebook[centroid_idx]`. **One indirection max.** | At hydration: when a child family's content distribution is statistically close to the parent's (Jirak-aware Berry-Esseen bound per `I-NOISE-FLOOR-JIRAK`), emit a centroid-reference instead of re-fitting. |

**Why family buckets, not a flat dict:** A flat OGIT URI → entry HashMap costs ~50-100ns per lookup at scale (hash + collision walk + cache miss). The 256-slot dense `[Option<FamilyEntry>; 256]` per family + the high-byte family index makes it **two array indices, both predictable, both L1-cache-resident**: ~5ns. The 20× cost difference is the headline gain from the layered substrate.

**What this means for downstream consumers:**

- `woa-bridge` doesn't pay the OWL-reasoning cost at every `wo_detail` query. Whatever schema / label / codebook inheritance applies to `ogit.WorkOrder:Customer` is pre-baked into the `FamilyEntry` at boot; the route handler reads one entry in one array index.
- `medcare-bridge`'s clinical-code lookups (ICD-10 / SNOMED via `hydrate_skos`) inherit their hierarchy chain at hydration; runtime cohort queries walk only the per-row identity, never the standards' subClassOf chain.
- `smb-bridge`'s X-Rechnung validation against ZUGFeRD Schematron rules doesn't re-parse the rule set at every invoice — the rules are baked once at boot via `hydrate_zugferd_rules` into the same family-bucket structure, and the per-invoice check is one masked predicate per rule.

**The construction tool (hydrators) and the runtime substrate (family buckets) are co-designed:** the hydrators only earn the O(1) property because they flatten the inheritance chains at construction time; the family buckets only earn the cost-model gain because the hydrators populate them with per-slot dense entries. Neither half works alone.

---

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

