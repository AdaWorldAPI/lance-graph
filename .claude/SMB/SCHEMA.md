# SMB tenant — schema reference for lance-graph sessions

> **Why this folder exists:** lance-graph is the spine; sessions
> working here usually don't have SMB context loaded. When changes to
> `lance-graph-ontology` / `lance-graph-callcenter` / `lance-graph-rbac`
> affect tenant consumers, this folder shows what the SMB tenant
> (`AdaWorldAPI/smb-office-rs`) will hydrate into the registry — so
> upstream changes can be evaluated against a real consumer shape
> without cross-repo grepping.
>
> **Authority:** the source of truth is smb-office-rs:
> - BSON-shape: `crates/smb-ontology/src/{customer,rechnung,mahnung,
>   schuldner,woa_artikel,remaining}.rs`
> - Foundry-shape stopgap: `crates/smb-realtime/src/ontology.rs::
>   build_smb_ontology()`
> - Originating inventory: `smb-office-rs/.claude/board/OGIT_TTL_INVENTORY.md`
>
> **Formal hydration spec:** `lance-graph/.claude/specs/pr-ogit-ttl-smb-hydration.md`
> (W13, sprint-5-6) — 35 KB executable spec, derived from the
> originating inventory above. Read this when the task is OGIT-side
> TTL authoring or registry hydration wiring.
>
> When the two diverge, smb-office-rs wins; update this doc.

---

## 1. Two ontology surfaces (by design)

| Surface | Naming | Entity count | Purpose | Where it lives |
|---|---|---|---|---|
| **BSON-shape** | German wire names, `smb.<entity>` prefix | 14 | Storage layer; matches C# `db_*.cs` verbatim per smb-office-rs CLAUDE.md iron rule 1 | `smb-ontology` crate |
| **Foundry-shape** | English names, no prefix | 3 | API/membrane layer; what RBAC + RLS + Foundry Object Explorer surface externally | `smb-realtime::ontology` (consumer-side stopgap until OGIT TTL hydrates) |

Foundry shape is a **projection** of BSON shape. One Foundry `Customer`
projects from one `smb.customer` row. Translation lives in
`smb_bridge::orchestration` (entity_type → table_name).

---

## 2. Foundry-shape — what `OntologyDto::project(&registry, "SMB", …)` must return

Mirrors the body that lance-graph PR #355 deleted from
`crates/lance-graph-callcenter/src/ontology_dto.rs::smb_ontology()`.
The consumer-side stopgap at `smb-realtime/src/ontology.rs::
build_smb_ontology()` reproduces it verbatim.

### Entities (3)

| Public name | OGIT URI | Required | Optional (Passthrough) | Searchable (CamPq) | Free |
|---|---|---|---|---|---|
| `Customer` | `ogit.SMB:Customer` | `customer_name`, `tax_id` | `address`, `iban` | `industry` | `note` |
| `Invoice` | `ogit.SMB:Invoice` | `invoice_number`, `date`, `total_amount`, `currency`, `customer_ref` | `due_date` | — | `note` |
| `TaxDeclaration` | `ogit.SMB:TaxDeclaration` | `declaration_id`, `tax_year`, `customer_ref`, `declaration_type` | `filing_date`, `status` | — | — |

### Links (`one_to_many`, 2)

- `Customer` —`issued`→ `Invoice`
- `Customer` —`filed`→ `TaxDeclaration`

### Actions (3)

- `approve` (manual) on `Invoice` → predicate `status`
- `classify` (auto) on `Customer` → predicate `industry`
- `submit` (manual) on `TaxDeclaration` → predicate `status`

### Label (bilingual)

- `de` → `Steuerberatungskanzlei`
- `en` → `Tax Practice`

---

## 3. BSON-shape — 14 entities under `smb.*`

German wire names match C# `db_*.cs` verbatim. Per-property markings
carry GDPR / financial classification.

| Schema name | Required predicates | Marking dominance | Source |
|---|---|---|---|
| `smb.customer` | `kdnr`, `firma` | Pii on personal, Financial on bank | `customer.rs` |
| `smb.schuldner` | `schuldner_id`, `debtor_name`, `saldo` | Internal/Pii/Financial mix | `schuldner.rs` |
| `smb.rechnung` | `rechnungsnr`, `kunde_kdnr`, `rechnungsdatum`, `faellig_am`, `betrag_netto`, `betrag_brutto`, `ust_betrag`, `ust_satz`, `zahlungsstatus` | Financial on amounts | `rechnung.rs` |
| `smb.mahnung` | `mahnung_id`, `schuldner_id`, `rechnungsnr`, `stufe`, `ausgestellt_am`, `faellig_am`, `gesamtsumme`, `status` | Internal | `mahnung.rs` |
| `smb.dokument` | `dokument_id`, `parent_entity`, `parent_id`, `dateiname`, `mime_type`, `hochgeladen_am` | Internal | `remaining.rs` |
| `smb.bank` | `buchungssatz_id`, `kontonummer`, `buchungsdatum`, `valutadatum`, `betrag`, `waehrung` | Financial | `remaining.rs` |
| `smb.fibu` | `buchung_id`, `buchungsdatum`, `soll_konto`, `haben_konto`, `betrag` | Financial | `remaining.rs` |
| `smb.steuer` | `steuer_id`, `steuerart`, `zeitraum_von`, `zeitraum_bis`, `bemessungsgrundlage`, `steuerbetrag`, `status` | Financial | `remaining.rs` |
| `smb.lieferant` | `lieferant_id`, `firma` | Internal | `remaining.rs` |
| `smb.mitarbeiter` | `mitarbeiter_id`, `vorname`, `nachname` | Pii on personal, Financial on iban | `remaining.rs` |
| `smb.auftrag` | `auftrag_id`, `kunde_kdnr`, `eingegangen_am`, `auftragssumme`, `status` | Internal | `remaining.rs` |
| `smb.angebot` | `angebot_id`, `kunde_kdnr`, `erstellt_am`, `angebotssumme`, `status` | Internal | `remaining.rs` |
| `smb.zahlung` | (see `remaining.rs:503+`) | Financial | `remaining.rs` |
| `smb.artikel` | `beschreibung` | Internal | `woa_artikel.rs` (WoA tenant) |

All entities also carry `fingerprint` (`CodecRoute::CamPq`) and audit
fields (`ad_create`, `ad_modify`, `ad_delete`).

---

## 4. Projection rules — Foundry shape ← BSON shape

| Foundry property | BSON source |
|---|---|
| `Customer.customer_name` | `smb.customer.firma` |
| `Customer.tax_id` | `smb.customer.umsatzsteuer_id` (preferred) or `steuernummer` |
| `Customer.address` | `smb.customer.anschrift` (+ `plz` + `ort` + `land` joined) |
| `Customer.iban` | `smb.customer.iban` |
| `Customer.industry` | `smb.customer.branche` |
| `Customer.note` | `smb.customer.notiz` |
| `Invoice.invoice_number` | `smb.rechnung.rechnungsnr` |
| `Invoice.date` | `smb.rechnung.rechnungsdatum` |
| `Invoice.total_amount` | `smb.rechnung.betrag_brutto` |
| `Invoice.currency` | `smb.rechnung.waehrung` |
| `Invoice.customer_ref` | `smb.rechnung.kunde_kdnr` |
| `Invoice.due_date` | `smb.rechnung.faellig_am` |
| `Invoice.note` | `smb.rechnung.notiz` |
| `TaxDeclaration.declaration_id` | `smb.steuer.steuer_id` |
| `TaxDeclaration.tax_year` | derived from `smb.steuer.zeitraum_von` (year part) |
| `TaxDeclaration.customer_ref` | **gap** — `smb.steuer` lacks a customer column; planned: add `kunde_kdnr` |
| `TaxDeclaration.declaration_type` | `smb.steuer.steuerart` |
| `TaxDeclaration.filing_date` | **gap** — `smb.steuer` lacks a filing date column; planned: add `eingereicht_am` |
| `TaxDeclaration.status` | `smb.steuer.status` |

Both gaps are smb-ontology-side fixes, independent of OGIT TTL.

---

## 5. What lance-graph changes affect SMB

| Lance-graph change | Affects SMB how |
|---|---|
| `OntologyDto::project(&registry, namespace, …)` signature | `smb-realtime::dto::smb_dto(locale)` calls it for `namespace = "SMB"`. Changes to the projection contract break the consumer. |
| `MappingRow` columns (e.g. PR #355 `IdentityCodec`, `QualiaMeta`) | Computed at hydration; SMB doesn't author them but consumes the projected `OntologyDto` fields. |
| `SchemaExpander` trait | smb-realtime's SPO bridge (`expand_smb_entity`) uses `Ontology::expand_entity`. If SchemaExpander moves to `MappingRow`, smb-realtime adapts. |
| `CachedOntology::new(Ontology)` legacy path | Currently retained for smb-realtime's `smb_cached_ontology()`. If deprecated, smb-realtime needs `CachedOntology::from_registry(&OntologyRegistry, namespace)`. |
| `Policy::evaluate(role, entity, Operation)` | smb-realtime's `SmbMembraneGate` (PR #29) wraps this. Entity keys passed are the Foundry-shape names (`Customer`, `Invoice`, `TaxDeclaration`). |
| `UnifiedBridge<B>` + `TenantId` (lance-graph PR #364 D-SDR-1..5) | smb-bridge wired `smb_unified_bridge(registry, namespace, role, tenant) -> UnifiedBridge<OgitBridge>` per smb-office-rs PR #31. Currently locked to `OgitBridge`; when a dedicated `SmbBridge` (`OGIT/NTO/SMB/` namespace) lands in `lance-graph-ontology::bridges`, the constructor type-param swaps — call sites unchanged. The new `authorize_read/write/act` path is complementary to the legacy `SmbMembraneGate`/`CachedOntology` surface; both coexist until SMB's TTL hydrates the registry. |
| `Locale` enum additions | smb-realtime's DTO cache projects every `Locale` eagerly; growing the enum forces cache init updates. |
| TTL hydrator (`hydrate_once_sync`) input format | Decides what OGIT TTL must carry for SMB (see Section 2 for the shape). |

---

## 6. Stale on the consumer side (will retire when registry hydration lands)

When `OntologyRegistry::hydrate_once_sync(ttl_root)` populates the
`"SMB"` namespace from real OGIT/NTO/SMB TTL:

- `smb-realtime/src/ontology.rs::build_smb_ontology()` — body deletes,
  becomes a thin `OntologyDto::project(&registry, "SMB", …)` facade.
- `smb-realtime/src/dto.rs::cached_smb_ontology() -> &'static Ontology`
  — retired when `SchemaExpander` learns to project from `MappingRow`.
- `smb-realtime/src/transcode.rs::smb_cached_ontology()` — retired when
  `CachedOntology::from_registry` ships.
- Dual cache path in `smb-realtime/src/spo.rs` (`#[cfg(feature = "postgrest")]`)
  — collapses to single path when `dto` becomes unconditional.

---

**Last updated:** 2026-05-13, pairs with smb-office-rs main at `074ce9b`
(PRs #29 + #30 + #31 merged: `SmbMembraneGate` + `OGIT_TTL_INVENTORY` +
`UnifiedBridge<OgitBridge>` wiring) and lance-graph main at `da156eb`
(PR #364 D-SDR-1..5 super-domain + UnifiedBridge surface). Update when
SMB schema changes in smb-ontology or when lance-graph's registry
projection contract shifts.
