# Foundry Consumer Parity — Shared Ontology + Contracts for SMB + MedCare

> **Status:** Active
> **Author:** main thread (Opus 4.7 1M), session 2026-04-26
> **Scope:** Map the Foundry parity surface shared between smb-office-rs
> and medcare-rs; resolve callcenter UNKNOWNs; document the DataFusion/SQL
> groundtruth pattern both consumers need.
> **Cross-ref:** `smb-office-rs/docs/foundry-parity-checklist.md` (45 LF chunks);
> `medcare-rs` session architecture review (callcenter-as-owner proposal);
> `lance-graph/plans/q2-foundry-integration-v1.md`; `lance-graph/plans/lf-integration-mapping-v1.md`

---

## §1 Callcenter UNKNOWN Resolutions

Resolved by medcare-rs architecture review (2026-04-26). Both SMB and
medcare-rs confirm the same answers — these are consumer-validated.

| UNKNOWN | Question | Resolution | Consumer evidence |
|---|---|---|---|
| **UNKNOWN-2** | Phoenix wire-protocol vs direct Rust API? | **Both.** Phoenix WS for realtime push (SMB: Wartezimmer; medcare: waiting-room/lab-result push). Direct Rust API for bespoke routes (auth, DMS upload, views). | Both consumers |
| **UNKNOWN-3** | Does any consumer need pgwire? | **No.** All clients are HTTP/WS. Neither SMB nor medcare-rs runs a PostgreSQL wire client. PostgREST shape is the target, not pgwire. | Both consumers |
| **UNKNOWN-4** | Right `actor_id` type? | **String** (JWT `sub` claim). Custom claims via `ActorContext::custom()` for domain-specific fields: SMB needs `praxis_id`, medcare needs `praxis_id` + `role`. | Both consumers |
| **UNKNOWN-5** | Lance dataset path convention? | **Single root URI** env var (`LANCE_URI=file:///var/lib/{app}/lance/`), subpaths per dataset. SMB: `MEDCARE__LANCE_PATH`-compatible; medcare: same pattern. | Both consumers |
| **§8 PostgREST** | Confirm PostgREST compat needed? | **Yes.** SMB clinic UI uses REST URL shape. medcare ~87 of 115 routes collapse to PostgREST auto-routes. This is the highest-leverage callcenter deliverable. | Both consumers |

### Action: DM-8 (PostgRestHandler) is UNBLOCKED

The `§8 stop-point` in `callcenter-membrane-v1.md` asked: "confirm
PostgREST compat is needed before building." Two consumers confirm.
DM-8 can proceed.

---

## §2 Shared Foundry Surface — What Both Consumers Need

Both SMB and medcare-rs need the SAME Foundry-equivalent contract surface.
Neither should build its own; both consume from `lance-graph-contract`.

### Already shipped (Tier 0 — consume directly)

| Contract module | SMB use | MedCare use |
|---|---|---|
| `ontology` (Ontology, Schema, LinkSpec, ActionSpec) | Customer/Invoice/TaxDecl ontology | Patient/Diagnosis/Lab/Vital ontology |
| `property` (PropertyKind, PropertySpec, Marking) | GDPR marking per field | GDPR + clinical marking per field |
| `repository` (EntityStore, EntityWriter) | MongoConnector + LanceConnector | MySQL-as-oracle + Lance-as-SoR |
| `external_membrane` (ExternalRole, ExternalEventKind) | SMB roles (operator, debtor) | MedCare roles (doctor, nurse, admin) |
| `auth` (ActorContext, JwtMiddleware) | praxis_id + role claims | praxis_id + role + ward claims |
| `rls` (RlsRewriter) | tenant_id per Steuerberater | tenant_id per Praxis |
| `cognitive_shader` (BindSpace SoA) | extends with SMB entity columns | extends with clinical entity columns |
| `a2a_blackboard` (multi-expert composition) | dunning/tax/doc-lookup experts | triage/prescription/lab-trend experts |
| `distance` (Distance trait + FisherZ) | similarity for customer dedup | similarity for diagnosis suggestion |
| `graph_render` (visual render traits) | q2 Gotham cockpit | q2 clinical dashboard |

### Needed but not shipped (Tier 1+2 — both consumers request)

| LF-id | What | SMB need | MedCare need | Priority |
|---|---|---|---|---|
| **DM-8** | PostgRestHandler | 50+ routes collapse | 87+ routes collapse | **P-0** (highest leverage) |
| **LF-12** | Pipeline DAG | ingestion orchestration | MySQL→Lance sync orchestration | **P-0** |
| **LF-31** | scan_as_of (time travel) | audit trail queries | clinical history queries | **P-1** |
| **LF-20** | FunctionSpec | German tax-rule transforms | clinical rule transforms | P-1 |
| **LF-23** | NotificationSpec | dunning event triggers | lab-result alert triggers | P-1 |
| **LF-40** | Full-text search | customer/document search | patient/diagnosis search | P-1 |
| **LF-41** | Faceted aggregation | filter+facet on invoices | filter+facet on lab values | P-1 |
| **LF-50** | ModelRegistry | tax classifier model | triage classifier model | P-2 |
| **LF-52** | LlmProvider | xAI/OpenAI for reasoning | xAI/OpenAI for diagnosis support | P-2 |
| **LF-60** | Approval workflow | Mahnung approval gates | prescription approval gates | P-2 |
| **LF-61** | NARS decision capture | user corrections on tax class | doctor corrections on diagnosis | P-2 |
| **LF-90** | Audit log | SOC2/GDPR audit trail | clinical audit trail | **P-0** |
| **LF-91** | SLA/health tracking | pipeline freshness | data quality monitoring | P-2 |
| **LF-92** | Multi-tenant isolation | per-Steuerberater isolation | per-Praxis isolation | P-1 |

---

## §3 The DataFusion/SQL Groundtruth Pattern

Both consumers use the SAME architecture:

```
                 lance-graph-callcenter (PostgREST + Phoenix + RLS)
                         │
                         │  ExternalMembrane trait
                         │
              ┌──────────┴──────────┐
              │                     │
         SMB membrane          MedCare membrane
         (schema decls +       (schema decls +
          RLS rules +           RLS rules +
          bespoke routes)       bespoke routes)
              │                     │
              ▼                     ▼
         Lance datasets        Lance datasets
         (system of record)    (system of record)
              │                     │
              │  reconciler         │  reconciler
              ▼                     ▼
         MySQL oracle          MySQL oracle
         (groundtruth,         (groundtruth,
          never retired)        never retired)
```

**DataFusion IS the query surface for both sides:**
- Incoming REST queries → PostgREST translates to DataFusion SQL
- DataFusion plans execute against Lance datasets (via lance-datafusion)
- RLS rewriter injects tenant/actor predicates into every plan
- VSA UDFs (vsa_distance, vsa_similarity) available in WHERE/ORDER BY
- MySQL serves as the parity oracle — reconciler compares SQL results
  from both systems to verify Lance produces identical answers

**lance-datafusion (from lancedb 0.27.2) is required:**
- Registers Lance datasets as DataFusion `TableProvider`
- SQL queries over patient/customer records, joins across entity types
- Predicate pushdown into Lance's native filter engine
- Without it, Lance is just a vector store, not a relational store

---

## §4 Ontology Unification — One Contract, Two Domains

The contract defines the SHAPE; each consumer fills in domain-specific content:

```rust
// lance-graph-contract::ontology — shared shape (zero-dep)
pub struct Ontology {
    pub schemas: Vec<Schema>,     // SMB: Customer, Invoice, ...
                                  // MedCare: Patient, Diagnosis, ...
    pub links: Vec<LinkSpec>,     // SMB: Customer→Invoice
                                  // MedCare: Patient→Diagnosis
    pub actions: Vec<ActionSpec>, // SMB: SendMahnung, ClassifyTax
                                  // MedCare: OrderLab, Prescribe
}

// The AGI surface is IDENTICAL for both:
// - FreeEnergy::compose → Resolution per cycle
// - CausalEdge64 emitted per strong hit
// - OntologyDelta (Column E) per novel pattern
// - AwarenessColumn (Column F) per word
// - EntityTypeId (Column H) per row
// - ModelRef (Column G) per ONNX binding
//
// The ONLY difference is which Schema instances are registered
// and which RLS rules are configured. The cognitive stack doesn't
// care whether it's thinking about a Mahnung or a Prescription —
// it processes fingerprints, emits edges, learns ontology.
```

**"Foundry outside, AGI inside" means:**
- Outside: Ontology + Schema + LinkSpec + ActionSpec = Foundry's typed object model
- Inside: BindSpace SoA + FreeEnergy + CausalEdge64 + NARS = AGI's cognitive substrate
- The callcenter membrane translates between them
- Both consumers see the SAME Foundry surface; neither sees the AGI internals
- The AGI learns from both domains simultaneously (if running in the same BindSpace)

---

## §5 Build Priority (Shared Leverage)

Ordered by "serves both consumers":

1. **DM-8 PostgRestHandler** — 87+ medcare routes + 50+ SMB routes collapse.
   Depends on: UNKNOWN-3 resolved (no pgwire), UNKNOWN-5 resolved (LANCE_URI).
   The single highest-leverage callcenter deliverable.

2. **LF-90 AuditLog** — both consumers need SOC2/GDPR/clinical audit.
   Already drafted (Tier 0 has `AuditEntry` shape). Wire it.

3. **LF-12 Pipeline DAG** — both consumers need orchestrated ingestion.
   SMB: MongoDB→Lance sync. MedCare: MySQL→Lance sync. Same `Pipeline` trait.

4. **LF-31 scan_as_of** — both consumers need time-travel queries.
   lancedb 0.27.2 `Table::checkout(version)` maps directly.

5. **LF-92 Multi-tenant** — both consumers isolate by `praxis_id`.
   The RLS rewriter (already shipped) does this per-query; LF-92 adds
   project-level structural isolation.

---

## §6 What This Plan Does NOT Cover

- **Q2 UI chunks** — tracked in `q2-foundry-integration-v1.md`
- **SMB-specific domain logic** — tracked in `smb-office-rs/docs/foundry-parity-checklist.md`
- **MedCare-specific domain logic** — tracked in `medcare-rs` session
- **ndarray SIMD work** — separate; consumed as-is
- **Per-consumer bespoke routes** — auth, DMS, migration, views stay hand-rolled
