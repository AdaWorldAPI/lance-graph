# lance-graph in smb-office-rs — v1

> **Author:** main thread (Opus 4.7), session 2026-05-21 (branch `claude/activate-lance-graph-att-k2pHI`)
> **Status:** Draft
> **Scope:** Complete the lance-graph integration in smb-office-rs from the current partial state (`smb-bridge` + `smb-ontology` crates exist, `smb_unified_bridge` constructor exists parameterised over `OgitBridge`; smb-bridge implements `EntityStore` + `EntityWriter` for Mongo + Lance; auth + RLS features wired against `lance-graph-callcenter`/`lance-graph-ontology`/`lance-graph-rbac`). End-state: dedicated `SmbBridge` in `lance-graph-ontology::bridges` locked to a future `OGIT/NTO/SMB/` namespace; SMB-shaped role groups (`tax_clerk` / `partner` / `client_user` / `audit_observer`) per D-SDR-2 in the upstream plan; Cypher / SPARQL surface over the Lance projection of MongoDB collections.
> **Path:** `.claude/plans/lance-graph-in-smb-office-rs-v1.md`
> **Confidence:** Working — most substrate already in tree. Partial — the dedicated `SmbBridge` does not yet exist upstream; the rich `auth::TenantId` vs `lance_graph_callcenter::TenantId` consolidation is deferred per `smb-bridge/src/unified_bridge_wiring.rs` lines 16-25.

---

## 1 — Why this exists

smb-office-rs is the **most-progressed consumer** of the lance-graph spine outside MedCare-rs. Per `smb-office-rs/CLAUDE.md`:

- F1–F4 phases ship smb-core → smb-bridge → smb-cache → smb-ontology against the lance-graph workspace already vendored as a sibling clone.
- `smb-bridge` implements `lance-graph-contract::repository::{EntityStore, EntityWriter}` for MongoDB (German BSON wire format with `kdnr`, `firma`, `vorname` field names) AND Lance (Foundry-shape projection).
- `smb-bridge` ships an `auth` feature wiring JWT → `ActorContext` via the `lance-graph-callcenter::auth-jwt` feature (PR #273 lance 2→4 + datafusion 51→52 + deltalake 0.30→0.31 unblocked).
- `smb-bridge` ships an `rls` feature against the `auth-rls` feature in lance-graph-callcenter.
- `smb-bridge/src/unified_bridge_wiring.rs::smb_unified_bridge` is **the working reference** for the cross-consumer unified-bridge migration plan.

What's missing:

1. **Dedicated `SmbBridge` in `lance-graph-ontology::bridges`.** Today `smb_unified_bridge` is parameterised over `OgitBridge::for_namespace(...)` (a pass-through). The `unified_bridge_wiring.rs` doc-comment lines 9-14 names the future swap explicitly.
2. **`OGIT/NTO/SMB/` namespace TTL.** Doesn't exist yet upstream. SMB callers currently lock onto `Network` or `WorkOrder` namespaces shared with other consumers.
3. **SMB-shaped role groups.** Per `unified_bridge_wiring.rs` lines 53-57: "D-SDR-2 will extend this with the SMB-specific role groups (e.g. `tax_clerk`, `partner`, `client_user`, `audit_observer`) once the `RoleGroup` + `FieldRedactionMask` types ship."
4. **Cypher / SPARQL surface.** Today's Lance projection in `smb-bridge::lance` is reachable via DataFusion SQL through lance-graph-catalog but no first-class Cypher endpoint is wired.
5. **`auth::TenantId` rich-tenant ↔ `callcenter::TenantId(u32)` consolidation.** Per `unified_bridge_wiring.rs` lines 16-25: "Future consolidation: when the SMB ontology ships its own SmbBridge, fold the rich tenant context into a super_domain cross-walk so callers see one type."

This plan owns those five items.


## 2 — Phasing

### Phase A — Upstream `SmbBridge` skeleton (3 days, lance-graph workspace)

- **D-LGSMB-1** — Add `lance-graph-ontology/src/bridges/smb_bridge.rs` mirroring `medcare_bridge.rs` (44 lines) verbatim with `NAMESPACE = "SMB"` and `bridge_id() = "smb"`. Add `pub mod smb_bridge;` to `bridges/mod.rs` and the `pub use smb_bridge::SmbBridge;` re-export. ~50 LOC + 2 tests (BridgeFromRegistry round-trip; cross-namespace-leak detection). This is also `D-UB-2` from `unified-bridge-consumer-migration-v1.md`.
- **D-LGSMB-2** — TTL placeholder at `AdaWorldAPI/OGIT/NTO/SMB/entities/` — at least one entity (e.g., `ogit.SMB:TaxClient` extending `ogit.Network:Person`) so the namespace registers at hydration. Real TTL authoring is Phase B. Lands as an OGIT-fork PR.

### Phase B — `SmbBridge` parameterisation swap + role groups (4 days, smb-office-rs side)

- **D-LGSMB-3** — Swap `smb-office-rs/crates/smb-bridge/src/unified_bridge_wiring.rs::smb_unified_bridge` from `UnifiedBridge<OgitBridge>` → `UnifiedBridge<SmbBridge>`. Per the doc-comment promise, call sites stay unchanged — only the type parameter shifts. ~15 LOC change + 1 regression test asserting the previous OgitBridge surface still resolves correctly (no behaviour change at consumers). This is `D-UB-5` from `unified-bridge-consumer-migration-v1.md`.
- **D-LGSMB-4** — Author `OGIT/NTO/SMB/entities/*.ttl` for the SMB business model: `TaxClient` (the Mandant-like concept), `BookingEntry` (FiBu Buchung), `BankStatement`, `TaxReturn`, `WorkSession`, `Document` (DMS), `Reminder` (Mahnung). Includes SemanticType + ObjectView + Marking annotations per the smb-ontology pattern. ~7 TTL files + an OGIT-fork PR + corresponding entries in `smb-office-rs/crates/smb-ontology/src/`.
- **D-LGSMB-5** — SMB-shaped role groups in `lance-graph-rbac::policy`. Promote `smb_policy()` from the starter (`accountant` / `auditor` / `admin`) to the full set per D-SDR-2: `tax_clerk` / `partner` / `client_user` / `audit_observer` plus retain the starter three. Each role gets a `FieldRedactionMask` keyed against the new SMB-namespace slots (booking-entry amounts visible to `accountant` + `partner` but redacted for `client_user`; tax-return drafts writable only by `tax_clerk` and `partner`; everything readable+audit-logged for `audit_observer`). ~120 LOC + 8 tests covering each role's read/write/redact matrix.

### Phase C — Tenant-type consolidation (2 days)

- **D-LGSMB-6** — Cross-walk between `smb_bridge::auth::TenantId` (rich payload: tenant + scope + per-request context) and `lance_graph_callcenter::TenantId(u32)` (transparent). Today the conversion is implicit at every `smb_unified_bridge` call site (`praxis_id` / `kdnr` numeric handles). Centralise into a `From<&auth::TenantId> for callcenter::TenantId` impl on the smb-bridge side plus a `super_domain_context` cross-walk that lets the rich-tenant context ride through `UnifiedBridge::authorize`'s audit emission. ~80 LOC + 4 tests covering the four flow shapes (auth-only / authz-only / dual / mismatch).

### Phase D — Cypher / SPARQL surface (5 days, opt-in)

- **D-LGSMB-7** — Wire `lance-graph-planner` as smb-bridge workspace dep behind `[features] planner`. Add `POST /v1/graph/query` endpoint that accepts Cypher / SPARQL / GQL / Gremlin (polyglot detection per planner Strategy #18) and returns Arrow batches. ~150 LOC + 4 tests against the SMB Lance projection.
- **D-LGSMB-8** — Cypher rewrites of two flagship cross-collection queries:
  - "all open Mahnungen for TaxClients whose latest BankStatement has unresolved entries" — currently hand-rolled across `db_mahnung` + `db_kunde` + `db_kontoauszug` mongo reads, expressible as a single Cypher MATCH against the Lance projection.
  - "BookingEntries by SKR04 account category with audit-log entries from the last quarter" — currently two-step (db_buchungen scan + db_audit join), one Cypher MATCH after the planner is wired.
  ~100 LOC of new handlers + 4 parity tests against the existing MongoDB reference.

### Phase E — CAM-PQ similarity surface (3 days, opt-in)

- **D-LGSMB-9** — `EntityStore::similar_to(entity_id, limit)` over the Lance projection. SMB use case: "TaxClients with similar booking patterns" for sales / collections segmentation. Backed by per-entity Vsa16kF32 over `BookingEntry` history. ~200 LOC + 4 tests. Feature-gated behind `smb-bridge/cam-pq`.

## 3 — Cross-references

- `super-domain-rbac-tenancy-v1.md` §3.9 (`UnifiedBridge::authorize`) — what `smb_unified_bridge` returns after D-LGSMB-3
- `super-domain-rbac-tenancy-v1.md` §3.6 (`RoleGroup` + `FieldRedactionMask`) — D-LGSMB-5 implementation surface
- `super-domain-rbac-tenancy-v1.md` §13.4 — `WorkOrderBilling ↔ OSINT` hard-lock partner row (SMB inherits this)
- `unified-bridge-consumer-migration-v1.md` D-UB-2, D-UB-5 — sister deliverables
- `lance-graph-in-medcare-rs-v1.md` — parallel migration in the Healthcare super domain
- `smb-office-rs/CLAUDE.md` Iron Rule 3 — "`lance-graph` is additive-only"; this plan only adds, never edits
- `smb-office-rs/crates/smb-bridge/src/unified_bridge_wiring.rs` — the working reference (lines 9-14 promise D-LGSMB-3; lines 16-25 promise D-LGSMB-6; lines 53-57 promise D-LGSMB-5)

## 4 — Open questions

1. **OGIT/NTO/SMB/ namespace structure.** Is `SMB` the locked name or do we prefer the German `Steuerberater` (closer to the actual business domain)? Affects D-LGSMB-1 + D-LGSMB-2.
2. **`tax_clerk` permission floor.** D-LGSMB-5 maps `tax_clerk` to `READ` + `WRITE` on TaxReturn drafts. Does the SMB practice require `EXPORT` too (e.g., for DATEV submission)? Affects the Phase B `PermissionSet` bits.
3. **`client_user` scope.** The `client_user` role represents the end-customer of an SMB tax firm (someone whose Buchhaltung the firm handles). Is `client_user` allowed to see their own BookingEntries cross-period or scoped to the current fiscal year only? Affects RLS predicate complexity.
4. **Mongo retire timeline.** Per `smb-office-rs/CLAUDE.md` stage H — "the in-tree twin at `SMB-Office/smb-core-rs` is best deleted at stage H". This plan operates pre-stage-H assuming Mongo stays the writer-oracle through Phase D; Phase E onwards may shift assumptions.

## 5 — Status

- **Phase A:** Not started. Mechanical (3 days). Cross-listed as D-UB-2 in the sister plan.
- **Phase B:** Not started. Blocked on Phase A. ~4 days; the TTL authoring is the labour-intensive part.
- **Phase C:** Not started. Independent of Phases A/B (~2 days).
- **Phase D:** Not started. Phase B-dependent. Opt-in (~5 days).
- **Phase E:** Phase D-dependent. Opt-in (~3 days).

**Confidence:** Working — the smb-bridge + smb-ontology + auth + rls features are already wired. This plan is mostly finishing what `unified_bridge_wiring.rs`'s doc-comments already promise.

## 6 — One-line summary

> Promote `smb_unified_bridge` from `UnifiedBridge<OgitBridge>` to `UnifiedBridge<SmbBridge>` (the doc-comments promise this swap), ship SMB-shaped role groups (`tax_clerk` / `partner` / `client_user` / `audit_observer`) per D-SDR-2, author `OGIT/NTO/SMB/` TTL upstream, and optionally light up the Cypher + CAM-PQ surfaces over the existing Lance projection. Smallest delta of the three consumer plans; most of the substrate already ships.

