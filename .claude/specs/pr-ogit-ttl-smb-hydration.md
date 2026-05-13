# PR-OGIT-SMB — OGIT/NTO/SMB TTL Authoring + Lance-Graph Hydration

> **Sprint:** sprint-log-5-6, W13
> **Worker:** W13 (claude-sonnet-4-6), 2026-05-13
> **Source inventory:** `AdaWorldAPI/smb-office-rs:main:.claude/board/OGIT_TTL_INVENTORY.md`
> **Status:** SPEC READY — OGIT-side authoring is a deliverable handoff; lance-graph-side is
>              ready-to-consume once OGIT delivers (0 LOC blocker on this side)
> **Prior plan extended:** `lance-graph-ontology-v5.md` D-ONTO-V5-4 (smb-ontology export-only)
> **Siblings:** `pr-d4-family-hydration.md` (W3), `pr-e2-smb-retrofit.md` (W7)

---

## 0 — Scope Statement

**PR-OGIT-SMB is a two-surface PR.** The OGIT repository (`AdaWorldAPI/OGIT`) is outside
the lance-graph workspace MCP scope and cannot be committed to from this session. The OGIT-side
TTL authoring is therefore described as a **deliverable handoff**: a concrete spec the OGIT-side
author can execute to produce the `OGIT/NTO/SMB/` directory. The lance-graph-side hydration is
**concrete and 0 LOC**: the existing `parse_ttl_directory_with_provenance` function in
`crates/lance-graph-ontology/src/ttl_parse.rs` already handles the core hydration; the only
new entry point is the `parse_family_registry()` option-(c) stub that W3 OQ-1 deferred.

The downstream beneficiary is `crates/smb-realtime/src/ontology.rs` in `smb-office-rs`. Once
`OGIT/NTO/SMB/` lands and the registry is hydrated, the consumer-side hand-rolled
`build_smb_ontology()` stopgap (commit `c204819` in smb-office-rs) can be retired. That
retirement is a follow-on PR scoped to `smb-realtime` only and is estimated separately in §8.

The `smb_ontology()` factory in `crates/lance-graph-callcenter/src/ontology_dto.rs` already
calls `OntologyDto::project(registry, "SMB", ...)` with the correct signature. The test
`smb_projects_three_entities` confirms the 3-entity Foundry shape. No changes to
`ontology_dto.rs` are required by this PR.

---

## 1 — Two-Surface Ontology Summary

The SMB ontology has two distinct representations that must coexist in the same OGIT TTL
directory.

**Foundry shape (membrane layer, B.1):** 3 entities in English, used by `OntologyDto::project`
and consumed by PostgREST / Phoenix downstream. These map to the 3 entities confirmed by the
`smb_registry()` test in `ontology_dto.rs`:

- `ogit.SMB:Customer`
- `ogit.SMB:Invoice`
- `ogit.SMB:TaxDeclaration`

**BSON shape (storage layer, B.2):** 14 entities in German wire names with `smb.<entity>`
prefix, matching the 13 entries in W7's `smb_owl_id_for()` table plus one reconciliation entity.
The 13 confirmed entities from W7 §5.2 are: `customer/kunde`, `rechnung`, `mahnung`, `dokument`,
`bank`, `fibu`, `steuer`, `lieferant`, `mitarbeiter`, `auftrag`, `angebot`, `zahlung`,
`schuldner`. The 14th entity reconciles to `smb.kanzlei` (practice entity) which anchors
multi-tenant ownership; its absence from W7's mapping is noted in §5.2 below.

The two shapes are held in the same `OGIT/NTO/SMB/` directory but in separate entity files.
The namespace `"SMB"` in `OntologyRegistry` covers the Foundry surface; `"SMB.bson"` covers
the BSON surface (see §3 namespace decision). `OntologyDto::project` already routes on namespace
string, so the Foundry projection passes `"SMB"` and a BSON projection would pass `"SMB.bson"`.

---

## 2 — Foundry-Shape TTL Skeleton (B.1)

The following is a complete, copy-pasteable TTL template for one Foundry entity — `Customer`.
The OGIT-side author copies this pattern for `Invoice` and `TaxDeclaration`, adjusting the
entity name, label strings, and property predicates.

```turtle
@prefix rdf:       <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:      <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:       <http://www.w3.org/2002/07/owl#> .
@prefix xsd:       <http://www.w3.org/2001/XMLSchema#> .
@prefix dcterms:   <http://purl.org/dc/terms/> .
@prefix ogit:      <http://www.purl.org/ogit/> .
@prefix ogit.SMB:  <http://www.purl.org/ogit/SMB/> .
@prefix ogit.meta: <http://www.purl.org/ogit/meta/> .

# ── Entity declaration ────────────────────────────────────────────────────────

ogit.SMB:Customer
    a                       ogit:Entity , owl:Class ;
    rdfs:label              "Customer"@en , "Kunde"@de ;
    rdfs:comment            "A client of the tax advisory practice."@en ;
    ogit:kind               ogit:Entity ;
    ogit:marking            ogit.meta:Internal ;
    ogit:surface            ogit.meta:FoundryShape ;
    dcterms:source          <https://github.com/AdaWorldAPI/smb-office-rs/blob/main/crates/smb-ontology/> ;

# ── Properties ────────────────────────────────────────────────────────────────

    ogit.meta:hasAttribute  ogit.SMB:Customer.name ,
                            ogit.SMB:Customer.email ,
                            ogit.SMB:Customer.phone ,
                            ogit.SMB:Customer.taxId ,
                            ogit.SMB:Customer.iban ,
                            ogit.SMB:Customer.customerId ,
                            ogit.SMB:Customer.address .

# ── Per-property attribute declarations ──────────────────────────────────────

ogit.SMB:Customer.name
    a                       ogit:Attribute ;
    rdfs:label              "Name"@de , "Name"@en ;
    ogit:kind               ogit:Attribute ;
    ogit:marking            ogit.meta:Internal ;
    ogit.meta:semanticType  ogit.meta:PlainText ;
    ogit.meta:propertyKind  ogit.meta:Required ;
    ogit.meta:predicateIri  "name" .

ogit.SMB:Customer.email
    a                       ogit:Attribute ;
    rdfs:label              "E-Mail"@de , "Email"@en ;
    ogit:kind               ogit:Attribute ;
    ogit:marking            ogit.meta:Pii ;
    ogit.meta:semanticType  ogit.meta:Email ;
    ogit.meta:propertyKind  ogit.meta:Optional ;
    ogit.meta:predicateIri  "email" .

ogit.SMB:Customer.phone
    a                       ogit:Attribute ;
    rdfs:label              "Telefon"@de , "Phone"@en ;
    ogit:kind               ogit:Attribute ;
    ogit:marking            ogit.meta:Pii ;
    ogit.meta:semanticType  ogit.meta:Phone ;
    ogit.meta:propertyKind  ogit.meta:Optional ;
    ogit.meta:predicateIri  "telefon" .

ogit.SMB:Customer.taxId
    a                       ogit:Attribute ;
    rdfs:label              "Steuernummer"@de , "Tax ID"@en ;
    ogit:kind               ogit:Attribute ;
    ogit:marking            ogit.meta:Financial ;
    ogit.meta:semanticType  ogit.meta:TaxId ;
    ogit.meta:propertyKind  ogit.meta:Required ;
    ogit.meta:predicateIri  "steuernummer" .

ogit.SMB:Customer.iban
    a                       ogit:Attribute ;
    rdfs:label              "IBAN"@de , "IBAN"@en ;
    ogit:kind               ogit:Attribute ;
    ogit:marking            ogit.meta:Financial ;
    ogit.meta:semanticType  ogit.meta:Iban ;
    ogit.meta:propertyKind  ogit.meta:Optional ;
    ogit.meta:predicateIri  "iban" .

ogit.SMB:Customer.customerId
    a                       ogit:Attribute ;
    rdfs:label              "Kundennummer"@de , "Customer ID"@en ;
    ogit:kind               ogit:Attribute ;
    ogit:marking            ogit.meta:Internal ;
    ogit.meta:semanticType  ogit.meta:CustomerId ;
    ogit.meta:propertyKind  ogit.meta:Required ;
    ogit.meta:predicateIri  "kdnr" .

ogit.SMB:Customer.address
    a                       ogit:Attribute ;
    rdfs:label              "Adresse"@de , "Address"@en ;
    ogit:kind               ogit:Attribute ;
    ogit:marking            ogit.meta:Pii ;
    ogit.meta:semanticType  ogit.meta:Address ;
    ogit.meta:propertyKind  ogit.meta:Optional ;
    ogit.meta:predicateIri  "adresse" .
```

**Template for `Invoice` (`ogit.SMB:Invoice`):** Mirror the pattern above. Key properties:
`rechnungsnummer` (InvoiceNumber, Required), `datum` (Date, Required), `betrag` (PlainText,
Required), `mwst` (PlainText, Optional), `bezahlt` (Date, Optional), `kundenRef` linking to
`ogit.SMB:Customer`.

**Template for `TaxDeclaration` (`ogit.SMB:TaxDeclaration`):** Key properties:
`steuerart` (PlainText, Required), `zeitraum` (PlainText, Required), `eingereicht_am`
(Date, Optional — note this is a B.2 gap in `smb.steuer`, not a B.1 gap),
`steuernummer` (TaxId, Required), `kundenRef` linking to `ogit.SMB:Customer`.

---

## 3 — BSON-Shape TTL Skeleton (B.2)

### 3.1 Namespace Decision: `ogit.SMB.bson:customer` (recommended)

**Decision (§E.1 resolution):** Use the **sub-namespace form** `ogit.SMB.bson:customer`
rather than the single-namespace form `ogit.SMB:smb.customer`.

**Justification:** The OGIT `OgitUri::parse` implementation in
`crates/lance-graph-ontology/src/namespace/mod.rs` splits on `:` to separate namespace from
entity name. The single-namespace form `ogit.SMB:smb.customer` would force the entity name to
contain a `.` which the parser tolerates as `name()` returning `"smb.customer"`, but it
creates a collision risk: the `OntologyRegistry` indexes by `(namespace, public_name)` pair,
so `"smb.customer"` and `"customer"` in the same `"SMB"` namespace would be distinct rows
but would be confusing for downstream consumers enumerating the namespace. The sub-namespace
form `ogit.SMB.bson` gives a clean separation: `registry.enumerate("SMB")` returns only
Foundry-shape entities (3); `registry.enumerate("SMB.bson")` returns only BSON-shape entities
(14). The `OntologyDto::project` factory already routes on namespace string so no code change
is needed — the Foundry projection passes `"SMB"`, and a future BSON projection would pass
`"SMB.bson"`.

```turtle
@prefix rdf:           <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:          <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:           <http://www.w3.org/2002/07/owl#> .
@prefix xsd:           <http://www.w3.org/2001/XMLSchema#> .
@prefix dcterms:       <http://purl.org/dc/terms/> .
@prefix ogit:          <http://www.purl.org/ogit/> .
@prefix ogit.SMB.bson: <http://www.purl.org/ogit/SMB/bson/> .
@prefix ogit.meta:     <http://www.purl.org/ogit/meta/> .

# ── BSON entity: smb.customer (wire name: kunde) ─────────────────────────────

ogit.SMB.bson:customer
    a                       ogit:Entity , owl:Class ;
    rdfs:label              "Kunde"@de , "Customer (BSON)"@en ;
    rdfs:comment            "Storage-layer BSON representation of a client."@en ;
    ogit:kind               ogit:Entity ;
    ogit:marking            ogit.meta:Internal ;
    ogit:surface            ogit.meta:BsonShape ;
    ogit.meta:wirePrefix    "smb.customer" ;
    ogit.meta:foundryRef    <http://www.purl.org/ogit/SMB/Customer> ;
    dcterms:source          <https://github.com/AdaWorldAPI/smb-office-rs/blob/main/crates/smb-ontology/> ;

    ogit.meta:hasAttribute  ogit.SMB.bson:customer.kdnr ,
                            ogit.SMB.bson:customer.firma ,
                            ogit.SMB.bson:customer.vorname ,
                            ogit.SMB.bson:customer.nachname ,
                            ogit.SMB.bson:customer.email ,
                            ogit.SMB.bson:customer.telefon ,
                            ogit.SMB.bson:customer.iban ,
                            ogit.SMB.bson:customer.steuernummer ,
                            ogit.SMB.bson:customer.adresse .

ogit.SMB.bson:customer.kdnr
    a                       ogit:Attribute ;
    rdfs:label              "Kundennummer"@de ;
    ogit.meta:semanticType  ogit.meta:CustomerId ;
    ogit.meta:propertyKind  ogit.meta:Required ;
    ogit.meta:predicateIri  "kdnr" .

ogit.SMB.bson:customer.firma
    a                       ogit:Attribute ;
    rdfs:label              "Firma"@de ;
    ogit.meta:semanticType  ogit.meta:PlainText ;
    ogit.meta:propertyKind  ogit.meta:Optional ;
    ogit.meta:predicateIri  "firma" .

ogit.SMB.bson:customer.vorname
    a                       ogit:Attribute ;
    rdfs:label              "Vorname"@de ;
    ogit.meta:semanticType  ogit.meta:PlainText ;
    ogit.meta:propertyKind  ogit.meta:Optional ;
    ogit.meta:predicateIri  "vorname" .

ogit.SMB.bson:customer.nachname
    a                       ogit:Attribute ;
    rdfs:label              "Nachname"@de ;
    ogit.meta:semanticType  ogit.meta:PlainText ;
    ogit.meta:propertyKind  ogit.meta:Required ;
    ogit.meta:predicateIri  "nachname" .

ogit.SMB.bson:customer.email
    a                       ogit:Attribute ;
    rdfs:label              "E-Mail"@de ;
    ogit.meta:semanticType  ogit.meta:Email ;
    ogit.meta:propertyKind  ogit.meta:Optional ;
    ogit.meta:predicateIri  "email" .

ogit.SMB.bson:customer.telefon
    a                       ogit:Attribute ;
    rdfs:label              "Telefon"@de ;
    ogit.meta:semanticType  ogit.meta:Phone ;
    ogit.meta:propertyKind  ogit.meta:Optional ;
    ogit.meta:predicateIri  "telefon" .

ogit.SMB.bson:customer.iban
    a                       ogit:Attribute ;
    rdfs:label              "IBAN"@de ;
    ogit.meta:semanticType  ogit.meta:Iban ;
    ogit.meta:propertyKind  ogit.meta:Optional ;
    ogit.meta:predicateIri  "iban" .

ogit.SMB.bson:customer.steuernummer
    a                       ogit:Attribute ;
    rdfs:label              "Steuernummer"@de ;
    ogit.meta:semanticType  ogit.meta:TaxId ;
    ogit.meta:propertyKind  ogit.meta:Optional ;
    ogit.meta:predicateIri  "steuernummer" .

ogit.SMB.bson:customer.adresse
    a                       ogit:Attribute ;
    rdfs:label              "Adresse"@de ;
    ogit.meta:semanticType  ogit.meta:Address ;
    ogit.meta:propertyKind  ogit.meta:Optional ;
    ogit.meta:predicateIri  "adresse" .
```

The `ogit.meta:foundryRef` predicate links each BSON entity to its Foundry counterpart —
`customer` to `Customer`, `rechnung` to `Invoice`, `steuer` to `TaxDeclaration`. The 11 BSON
entities without a Foundry counterpart (`mahnung`, `dokument`, `bank`, `fibu`, `lieferant`,
`mitarbeiter`, `auftrag`, `angebot`, `zahlung`, `schuldner`, `kanzlei`) carry no `foundryRef`
predicate.

---

## 4 — Lance-Graph Hydrator Changes (if any)

**Net lance-graph-side LOC delta: approximately 0 to 60 LOC.**

The core hydration path for the Foundry-shape entities (B.1) requires **zero code changes**.
`parse_ttl_directory_with_provenance` in `crates/lance-graph-ontology/src/ttl_parse.rs`
already walks a directory of `.ttl` files and emits `MappingProposal` rows keyed by namespace.
Pointing it at `OGIT/NTO/SMB/` will hydrate `ogit.SMB:Customer`, `ogit.SMB:Invoice`, and
`ogit.SMB:TaxDeclaration` into the registry via the existing `"SMB"` namespace bucket. The
`smb_ontology()` factory in `ontology_dto.rs` already calls `registry.enumerate("SMB")` and
the test `smb_projects_three_entities` already asserts the 3-entity output.

The BSON-shape entities (B.2) under `ogit.SMB.bson:*` will likewise hydrate automatically via
the same parser, arriving in the `"SMB.bson"` namespace bucket. No consumer currently reads
this bucket, so it is a no-op until `smb-realtime` is retrofitted (§8 follow-on).

The only new lance-graph code in this PR is the **`parse_family_registry()` stub** from W3
OQ-1 option-(c). W3's spec deferred the decision about how `hydration::load_overlay` extracts
the two custom predicates (`ogit.meta:superDomain` and `ogit.meta:familyId`). Option-(c) —
a thin separate entry point in `lance-graph-ontology` that only looks for those two predicates
— is the recommended approach (cleanest separation, no impact on the existing proposal path).

This PR should include the implementation of that stub:

```rust
// crates/lance-graph-ontology/src/ttl_parse.rs (new public fn)

/// Extracts only `ogit.meta:superDomain` + `ogit.meta:familyId` triples
/// from a TTL byte slice. Used by `lance-graph-callcenter::hydration`
/// to populate FAMILY_TABLE without going through the full MappingProposal path.
///
/// Returns Vec<(family_id: u8, super_domain_name: String)>.
pub fn parse_family_registry(ttl_bytes: &[u8]) -> Result<Vec<(u8, String)>, TtlParseError> {
    // ~40-60 LOC: oxttl MemoryStore load, iterate triples,
    // match on ogit.meta:superDomain + ogit.meta:familyId predicates,
    // return paired (u8, String) entries.
    todo!()
}
```

**Contrast with W3 `parse_family_registry()` scope:** W3 uses this function to hydrate
`FAMILY_TABLE` from the inline `data/family_registry.ttl` seed. PR-OGIT-SMB does NOT call
this function — OGIT-side TTL for SMB contains entity declarations, not family registry
triples. The `parse_family_registry()` entry point is co-developed here because this PR is
the natural home for OGIT-adjacent `ttl_parse.rs` changes.

**W3's `parse_family_registry()` OQ-1 closure:** By implementing option-(c), this PR resolves
W3 OQ-1 as a side effect. The meta-review (§1 W3) recommended (c); this PR delivers it.
W3 can then call `parse_family_registry(SEED_TTL)` directly without touching the
`MappingProposal` path.

**Contrast with W3 `parse_family_registry()` vs this PR:** W3 is about hydrating
`FAMILY_TABLE` (super-domain registry); PR-OGIT-SMB is about hydrating entity schema. The
function is the same; the TTL files passed to it differ.

---

## 5 — Cross-Spec Alignment

### 5.1 Alignment with W7 (`pr-e2-smb-retrofit.md`) — entity count

W7 §5.2 defines `smb_owl_id_for()` mapping 13 entities to `OwlIdentity` slots 1-13:
`customer`, `rechnung`, `mahnung`, `dokument`, `bank`, `fibu`, `steuer`, `lieferant`,
`mitarbeiter`, `auftrag`, `angebot`, `zahlung`, `schuldner`.

The inventory source (OGIT_TTL_INVENTORY.md) specifies 14 BSON entities. The 14th entity
is `kanzlei` (the practice itself — multi-tenant anchor). W7 omits it because the
orchestrator's `ACCEPTED_ENTITIES` does not expose `kanzlei` as a routable entity;
`kanzlei` is only referenced as a parent/ownership anchor in BSON documents, never as a
first-class action target.

**Resolution:** PR-OGIT-SMB includes all 14 BSON entities in the TTL. W7's `smb_owl_id_for()`
should be extended in a follow-up to add slot 14 for `kanzlei`, even if no orchestrator action
currently references it. Slot 14 reserves the OwlIdentity slot so future BSON-layer operations
can reference it. Alignment: W7's 13-entity mapping covers operational entities; PR-OGIT-SMB
TTL covers all 14 BSON + 3 Foundry entities = **17 total declared in OGIT**.

The `smb_owl_id_for()` function in W7 is shape-agnostic (maps entity wire names, not namespace
URIs), so it aligns equally well with `ogit.SMB.bson:customer` as with any other BSON
representation. No changes to `smb_owl_id_for()` are required by this PR's namespace decision.

### 5.2 Alignment with W3 (`pr-d4-family-hydration.md`) — no dependency

This PR does NOT depend on W3 landing first. The OGIT-side TTL authoring and the
`parse_family_registry()` stub addition to `ttl_parse.rs` are independent of W3's
`UnifiedBridge::new_hydrated()` construction. The family-hydration TTL files (W3's
`data/family_registry.ttl`) and the SMB entity TTL files (this PR's `OGIT/NTO/SMB/`) are
separate files parsed by separate functions. W3 can land before or after PR-OGIT-SMB with
no conflict.

The only ordering constraint is that `parse_family_registry()` must be merged into
`lance-graph-ontology` before W3 implements `hydration::load_overlay`. If PR-OGIT-SMB
ships first, W3 picks up the already-existing entry point. If W3 ships first using a
temporary inline implementation, PR-OGIT-SMB replaces it with the canonical entry point.
No hard ordering required; soft recommendation: PR-OGIT-SMB `parse_family_registry()` stub
lands with or before W3 Batch implementation.

---

## 6 — Section D BSON Gaps Reconciliation

The two gaps in `smb.steuer` identified in the inventory source:

1. **`kunde_kdnr` missing from `smb.steuer`** — the `TaxDeclaration` BSON entity lacks a
   foreign key back to the owning `customer` via `kdnr`.
2. **`eingereicht_am` missing from `smb.steuer`** — the filing date is not present as a BSON
   column.

These are **smb-ontology-side fixes** — they require changes to the Rust schema definitions in
`smb-office-rs/crates/smb-ontology/`, not to the OGIT TTL files. The OGIT TTL for
`ogit.SMB.bson:steuer` should declare these as `ogit.meta:propertyKind ogit.meta:Required`
(for `kunde_kdnr`) and `ogit.meta:Optional` (for `eingereicht_am`) so the TTL expresses the
intended schema; the BSON storage layer then becomes the site of the deficit rather than the
TTL being the deficit.

**Recommendation: option (b) — note as a prerequisite/blocker for the `smb-realtime` consumer
cleanup (§8), NOT for this PR.**

Rationale: PR-OGIT-SMB's deliverable is the TTL authoring. The TTL can declare the desired
schema including `kunde_kdnr` and `eingereicht_am` on `ogit.SMB.bson:steuer`. The deficit is
that the current BSON documents in MongoDB do not carry these fields — that is a data migration
concern in `smb-realtime`, not an OGIT-TTL authoring concern. The registry hydration succeeds
regardless; the schema mismatch surfaces when `smb-realtime` tries to query those properties
and finds no data.

Mark the two gaps in a `# BSON-gap` comment within the TTL file:

```turtle
ogit.SMB.bson:steuer.kunde_kdnr
    a                       ogit:Attribute ;
    rdfs:label              "Kundennummer (FK)"@de ;
    ogit.meta:semanticType  ogit.meta:CustomerId ;
    ogit.meta:propertyKind  ogit.meta:Required ;
    ogit.meta:predicateIri  "kunde_kdnr" ;
    rdfs:comment            "BSON-gap: field absent from current MongoDB documents. Requires smb-ontology BSON schema update."@en .

ogit.SMB.bson:steuer.eingereicht_am
    a                       ogit:Attribute ;
    rdfs:label              "Eingereicht am"@de ;
    ogit.meta:semanticType  ogit.meta:Date ;
    ogit.meta:propertyKind  ogit.meta:Optional ;
    ogit.meta:predicateIri  "eingereicht_am" ;
    rdfs:comment            "BSON-gap: field absent from current MongoDB documents. Requires smb-ontology BSON schema update."@en .
```

---

## 7 — Three §E Open Questions: Recommended Answers

### E.1 — BSON namespace shape: single (`ogit.SMB:smb.customer`) vs sub-namespace (`ogit.SMB.bson:customer`)

**Recommendation: `ogit.SMB.bson:customer` (separate sub-namespace).**

The `OntologyRegistry` indexes by `(namespace, public_name)`. Under the single-namespace form,
`registry.enumerate("SMB")` would return all 17 entities (3 Foundry + 14 BSON) mixed together.
`OntologyDto::project` for the Foundry surface — the already-shipping `smb_ontology()` factory
— would then need a filter to exclude BSON entities, and the test `smb_projects_three_entities`
would break. The sub-namespace form gives clean namespace separation at zero cost: `"SMB"` stays
exactly the 3 Foundry entities; `"SMB.bson"` contains the 14 BSON entities. Future consumers
of the BSON surface enumerate `"SMB.bson"` explicitly, with no risk of bleeding into the
Foundry projection. The OGIT URI prefix `ogit.SMB.bson:` is consistent with the `@prefix`
convention used for healthcare sub-namespaces in `OGIT/NTO/Healthcare/`. The `OgitUri::parse`
implementation in `lance-graph-ontology` handles dotted namespace prefixes correctly (WorkOrder
namespace already uses `ogit.WorkOrder:`).

### E.2 — Per-property marking carriage: RDF annotations on each attribute vs entity-level only

**Recommendation: per-property RDF annotations (as shown in §2 and §3 above).**

The `MappingRow` in `crates/lance-graph-ontology/src/registry.rs` carries a single
`marking: Marking` field at the entity level today (per D-CASCADE-V1-7 deferral note in
`ontology_dto.rs`). However, the TTL should be authored with per-property `ogit:marking`
predicates now, not after the registry evolves. Reasons: (1) TTL is an investment that
outlasts the current registry shape; (2) the per-property marking divergence is already visible
in §2 (`Customer.name` = Internal vs `Customer.email` = Pii vs `Customer.taxId` = Financial);
(3) when D-CASCADE-V1-7 extends `MappingRow` to carry per-attribute provenance pairs (per the
`attribute_sources` field already on `MappingRow`), the parser will be able to populate
per-property markings from TTL without a second TTL edit pass. Entity-level-only marking in
TTL would bake in the current approximation, making a future upgrade more expensive. The
OGIT-side author should annotate every `ogit:Attribute` with `ogit:marking` and
`ogit.meta:propertyKind`.

### E.3 — Custom semantic types (`tax_id`, `iban`, `date_de`): add to `semantic_types.toml` first or use existing types?

**Recommendation: use the existing `TaxId`, `Iban`, and `Date` variants — do NOT add custom
`tax_id_de`, `iban_de`, `date_de` types.**

The current `semantic_types.toml` already carries the variants needed:
- `ogit.Compliance:Person.taxId = "TaxId"` — `TaxId` is a first-class variant.
- `ogit.SalesDistribution:Customer.iban = "Iban"` — `Iban` is a first-class variant.
- Date fields use `"Date"` (not `"DateDe"` or `"date_de"`).

Adding locale-specific variants (`tax_id_de`, `iban_de`) would fragment the `SemanticType`
enum and violate the zero-dep invariant by introducing locale-specific codec routing at the
contract layer. IBAN and German tax ID formats are already well-specified by their respective
standards; the `Iban` variant can carry locale context via the `Currency(code)` precedent if
truly needed, but for display and validation purposes `TaxId` + `Iban` are sufficient. The
`ogit.meta:semanticType` predicates in the TTL use `ogit.meta:TaxId` and `ogit.meta:Iban`
(matching the variant names already registered in `semantic_types.toml`). No changes to
`semantic_types.toml` are required for this PR.

If the OGIT-side author wants locale-specific display formatting (e.g., German IBAN grouping),
that is a UI-layer concern handled in the consumer app, not a `SemanticType` variant.
`SemanticType` governs codec routing and PII classification, not display format.

---

## 8 — LOC Estimate

### OGIT-side TTL authoring (deliverable handoff — not lance-graph code)

| File | Action | Estimated lines |
|---|---|---|
| `OGIT/NTO/SMB/entities/Customer.ttl` | New | ~70 |
| `OGIT/NTO/SMB/entities/Invoice.ttl` | New | ~65 |
| `OGIT/NTO/SMB/entities/TaxDeclaration.ttl` | New | ~60 |
| `OGIT/NTO/SMB/bson/customer.ttl` | New | ~90 |
| `OGIT/NTO/SMB/bson/rechnung.ttl` | New | ~75 |
| `OGIT/NTO/SMB/bson/mahnung.ttl` | New | ~50 |
| `OGIT/NTO/SMB/bson/dokument.ttl` | New | ~50 |
| `OGIT/NTO/SMB/bson/bank.ttl` | New | ~55 |
| `OGIT/NTO/SMB/bson/fibu.ttl` | New | ~55 |
| `OGIT/NTO/SMB/bson/steuer.ttl` | New | ~80 (includes gap annotations) |
| `OGIT/NTO/SMB/bson/lieferant.ttl` | New | ~60 |
| `OGIT/NTO/SMB/bson/mitarbeiter.ttl` | New | ~65 |
| `OGIT/NTO/SMB/bson/auftrag.ttl` | New | ~60 |
| `OGIT/NTO/SMB/bson/angebot.ttl` | New | ~55 |
| `OGIT/NTO/SMB/bson/zahlung.ttl` | New | ~55 |
| `OGIT/NTO/SMB/bson/schuldner.ttl` | New | ~50 |
| `OGIT/NTO/SMB/bson/kanzlei.ttl` | New | ~45 |
| `OGIT/NTO/SMB/SMB.ttl` (namespace declaration) | New | ~25 |
| `OGIT/NTO/SMB/bson/namespace.ttl` | New | ~20 |

**OGIT-side total: ~19 files, ~1,085 lines of Turtle.** The OGIT fork PR is one commit
against `AdaWorldAPI/OGIT` master. No pyoxigraph validation failures expected given structural
consistency with `OGIT/NTO/WorkOrder/` (already merged as OGIT#1).

### Lance-graph-side (this codebase)

| File | Action | LOC |
|---|---|---|
| `crates/lance-graph-ontology/src/ttl_parse.rs` | Add `parse_family_registry()` | ~55 |
| `crates/lance-graph-ontology/tests/smb_ttl_round_trip.rs` | New integration test | ~45 |

**Lance-graph-side total: ~100 LOC** (strictly additive, no existing logic touched).

The test verifies: (a) the 3 Foundry entity URIs parse cleanly from the SMB Foundry TTL files,
(b) the 14 BSON entity URIs parse from the SMB BSON TTL files into the `"SMB.bson"` namespace
bucket, (c) `smb_projects_three_entities` (existing) still passes, (d)
`parse_family_registry()` on a minimal test TTL returns the expected `(u8, String)` pairs.

### smb-realtime-side (downstream cleanup — follow-on PR, separate estimate)

The consumer-side stopgap to retire post-hydration is `build_smb_ontology()` in
`crates/smb-realtime/src/ontology.rs` (commit `c204819` in `smb-office-rs`). This function
currently hand-constructs an ontology schema without reading from the OGIT TTL registry.

**Follow-on PR estimate:** ~180 LOC (retire `build_smb_ontology()`, wire `smb_ontology()`
from `ontology_dto.rs` via `OntologyRegistry::hydrate_once_sync`, add one integration test
asserting the 3 Foundry entity names survive the full OGIT to registry to DTO path). Depends
on OGIT/NTO/SMB/ landing on OGIT master AND `lance-graph-ontology` carrying the SMB TTL in
its integration test fixtures.

---

## 9 — Sequencing

```
PR-OGIT-SMB sits AFTER W7 (PR-E2 smb-office retrofit) and PARALLEL TO / BEFORE
consumer-side cleanup.

Dependency graph:

  W3 (pr-d4-family-hydration) ──────────────────────────────────┐
                                                                  │
  W7 (pr-e2-smb-retrofit, Batches A+B)                          │
    │ [provides smb_owl_id_for() 13-entity baseline]             │
    │                                                            │
    ├── PR-OGIT-SMB (this spec)                                  │
    │     ├─ OGIT-side: author OGIT/NTO/SMB/ TTLs               │
    │     │   [blocked on OGIT-side author, outside MCP scope]  │
    │     └─ lance-graph-side: parse_family_registry() stub      │
    │           [~100 LOC, no blockers, shippable now]           │
    │                                                            │
    └── W7 Batch C (depends on W3 landing)                      │
                                                                  │
  [OGIT/NTO/SMB/ merged to OGIT master]                          │
    │                                                            │
    ├── smb-realtime follow-on PR (~180 LOC)                     │
    │   [retires build_smb_ontology() stopgap]                   │
    │                                                            │
    └── smb-realtime integration test                            │
          [full OGIT to registry to DTO to smb-realtime path]   ◄┘
```

Key sequencing notes:

1. The **lance-graph-side `parse_family_registry()` stub** (~55 LOC) has no blockers and can
   merge in the same sprint as W3. It resolves W3 OQ-1 as a side effect.

2. The **OGIT-side TTL authoring** is blocked on the OGIT-side author having access to
   `AdaWorldAPI/OGIT`. This is outside the harness MCP scope per the source inventory doc.
   The lance-graph-side spec is ready-to-consume the moment the OGIT TTL lands.

3. PR-OGIT-SMB is **not a hard prerequisite for W7 Batches A/B**. W7 Batches A and B wire
   authorization gates using the placeholder `SMB_FAMILY = 0` and the existing 13-entity
   mapping. PR-OGIT-SMB (once OGIT-side lands) provides the real OGIT URIs that allow
   `OntologyRegistry::resolve()` to succeed; Batch C of W7 is the natural consumer of that
   resolution path.

4. PR-OGIT-SMB is **not blocked on sprint-5 D-series** (D3A, D3B, D4). Those PRs ship audit
   substrate and family table hydration machinery; PR-OGIT-SMB ships the TTL content that
   those machineries will eventually read.

---

## 10 — DELTA Section

### What this spec concretizes vs OGIT_TTL_INVENTORY.md

| Section | Status in this spec |
|---|---|
| **§A — Healthcare TTL pattern reference** | Applied: TTL skeleton in §2/§3 mirrors the Healthcare entity pattern (same prefix declarations, same `ogit:kind`, `ogit:marking`, `ogit.meta:hasAttribute` structure). |
| **§B.1 — 3 Foundry entities** | Concretized: full TTL template for Customer in §2; Invoice + TaxDeclaration template instructions given. |
| **§B.2 — 14 BSON entities** | Concretized: full TTL template for `smb.customer` in §3; file-by-file breakdown in §8 LOC table. |
| **§C — smb-realtime consumer stopgap** | Noted as follow-on PR in §8; estimate provided (~180 LOC). |
| **§D — property gap matrix** | Reconciled: both gaps (`kunde_kdnr`, `eingereicht_am`) appear in TTL with `rdfs:comment` gap annotation; fix deferred to smb-ontology-side. |
| **§E.1 — BSON namespace shape** | Decided: `ogit.SMB.bson:customer` (sub-namespace) — full justification in §7 E.1. |
| **§E.2 — per-property marking carriage** | Decided: per-property annotations recommended — full justification in §7 E.2. |
| **§E.3 — custom semantic types** | Decided: use existing `TaxId`, `Iban`, `Date` — no new variants needed — full justification in §7 E.3. |
| **Status note: OGIT repo outside MCP scope** | Honored: lance-graph-side spec ready-to-consume; OGIT TTL authoring is handoff; no git commits to OGIT from this session. |

### What remains open after this spec

- **`kanzlei` slot 14 in `smb_owl_id_for()`**: W7 should extend to slot 14 in a follow-up.
- **smb-realtime integration test**: Full path test requires OGIT/NTO/SMB/ on disk during CI.
  Either check the OGIT fork path into `smb-office-rs` test fixtures or use a minimal
  in-memory TTL snapshot in the test.
- **`ogit.meta:foundryRef` predicate spec**: This spec introduces `foundryRef` linking BSON
  entities to their Foundry counterparts. OGIT-side author should confirm `ogit.meta:foundryRef`
  does not collide with an existing predicate.
- **`ogit:surface` predicate**: `ogit.meta:FoundryShape` and `ogit.meta:BsonShape` are new
  OGIT meta concepts. OGIT-side author should declare them in a meta namespace TTL.

### Comparison with sibling Healthcare TTL pattern (§A reference)

The Healthcare pattern in `OGIT/NTO/Healthcare/` (bootstrapped per PR #353: 7 entities + 7
enums, 846 lines) uses the same structural shape as §2 above: same `@prefix ogit.Healthcare:`
declaration, same `rdfs:label "@de" + "@en"` bilingual labels, same `ogit:kind ogit:Entity`
declaration, same `ogit.meta:hasAttribute` predicate list.

The SMB Foundry shape (§2) follows this template precisely. The SMB BSON shape (§3) extends
it with three SMB-specific predicates: `ogit:surface ogit.meta:BsonShape`,
`ogit.meta:wirePrefix`, and `ogit.meta:foundryRef`. These three predicates are
SMB-domain-specific and do not need to be backported to Healthcare unless Healthcare later
grows a BSON storage layer.

---

*End of spec. Estimated implementation: ~1,085 lines TTL (OGIT-side handoff) + ~100 LOC Rust
(lance-graph-side, one PR). Follow-on: ~180 LOC Rust (smb-realtime consumer cleanup, second PR).
OGIT-side authoring is the critical-path blocker; lance-graph-side ships independently.*
