# Agent W13 scratchpad — PR-OGIT-SMB TTL hydration spec

**Session:** 2026-05-13, sprint-log-5-6
**Task:** Author `.claude/specs/pr-ogit-ttl-smb-hydration.md`

## Reads completed

1. LATEST_STATE.md — PR #364 shipped OgitFamilyTable (HashMap<u16, FamilyEntry>),
   OwlIdentity 3-byte canonical; D-SDR-3/4/5 complete. smb-office-rs#31 wired
   UnifiedBridge<OgitBridge>.
2. PR_ARC_INVENTORY.md — #352 locked smb-ontology export-only; #364 locked OgitFamilyTable
   sparse HashMap; cross-repo landing pattern documented.
3. pr-d4-family-hydration.md (W3) — TTL hydration parser; OQ-1 proposes parse_family_registry()
   as option (c) for the custom-predicate extraction.
4. pr-e2-smb-retrofit.md (W7) — smb_owl_id_for() maps 13 BSON entities; §8.2 notes
   smb.ttl OWL file not yet authored — this is exactly what PR-OGIT-SMB delivers.
5. super-domain-rbac-tenancy-v1.md §3 — OgitFamilyTable codebook, namespace bytes.
6. lance-graph-ontology-v5.md — D-ONTO-V5-4 says smb-ontology stays Rust export-only;
   OGIT/NTO/SMB/ TTL authoring is adjacent.
7. ontology_dto.rs — smb_ontology() uses "SMB" namespace, 3 Foundry entities (Customer,
   Invoice, TaxDeclaration) confirmed via smb_projects_three_entities test.
8. semantic_types.toml — existing types: PlainText, Iban, Email, Phone, Address, Url,
   TaxId, CustomerId, InvoiceNumber, Image, Date, DateMonth, DateYear, DateTime,
   GeoLatLon, GeoWgs84, GeoPlusCode. No tax_id_de, iban_de, date_de custom types.
9. meta-review.md — W3 OQ-1 recommendation: option (c) parse_family_registry(). W7
   noted §8.2 smb.ttl unblocked by this sprint's work.

## Key findings

- Foundry shape: 3 entities — Customer/Invoice/TaxDeclaration, namespace "SMB",
  URI scheme ogit.SMB:Customer
- BSON shape: 14 entities from W7 §5.2 mapping (13 + reconciliation = 14 per inventory)
  W7 lists 13: customer/kunde, rechnung, mahnung, dokument, bank, fibu, steuer, lieferant,
  mitarbeiter, auftrag, angebot, zahlung, schuldner
- smb.steuer BSON gaps: kunde_kdnr + eingereicht_am missing
- semantic_types.toml has no tax_id_de, iban_de, date_de — these must be added first
- parse_ttl_directory_with_provenance already exists; no new lance-graph parser needed
  for the core hydration path; only the custom-predicate extraction per W3 OQ-1(c) is new
- OGIT repo is outside MCP scope (no direct PR possible from this session)

## Decision: BSON namespace

Recommend ogit.SMB.bson:customer (separate sub-namespace). Justification in spec §3.
