# lance-graph in woa-rs — v1

> **Author:** main thread (Opus 4.7), session 2026-05-21 (branch `claude/activate-lance-graph-att-k2pHI`)
> **Status:** Draft
> **Scope:** Full integration of lance-graph (ontology + planner + runtime + CAM-PQ) into `woa-rs` from the current zero-baseline (today: OGIT TTL vendored under `vendor/ogit/v02-harvest/`, sea-orm + MySQL writer-parity established per RFC v02-006, no lance-graph dep, no `woa-bridge`/`woa-ontology` crates). End-state: woa-rs route handlers consume `UnifiedBridge<WoaBridge>` for ontology resolution + RBAC + tenant isolation, with the WorkOrder namespace registry persisted via the `lance-cache` LanceDB column, and a long-arc path to Cypher/SPARQL query over the same LanceDB store (orthogonal to MySQL writer-parity per RFC v02-001 §writer-parity-pivot Goldstaub 2026-05-15).
> **Path:** `.claude/plans/lance-graph-in-woa-rs-v1.md`
> **Confidence:** Working (architecture — `lance_graph_ontology::bridges::WoaBridge` already exists in lance-graph, locked to NAMESPACE="WorkOrder"); Partial (consumer crate scaffolding absent in woa-rs; OGIT NTO/WorkOrder TTL exists per §1.6 of `super-domain-rbac-tenancy-v1.md`); Conjecture (planner integration depth — woa-rs may stop at the ontology+RBAC tier and never wire Cypher).

---

## 1 — Why this exists

woa-rs is the rust transcode of WoA (Python/Flask/SQLAlchemy). Today the workspace ships:

- Root `woa-rs` binary + library + 6 `crates/*` members (`decimal_money`, `skr_data`, `buchungs_validator`, `woa_pdf`, `datev_encoder` planned).
- Sea-orm + MySQL writer-parity (per the 2026-05-15 DualSink-Pivot in `.claude/board/Goldstaub.md` — Python and Rust write to the SAME MySQL).
- OGIT TTL vendored at `vendor/ogit/v02-harvest/entities/*.ttl` + `vendor/ogit/v02-harvest/POLICY.md`.
- RFC v02-006 (route codegen + ontology unification) — DRAFT, classifier landed, skeleton `crates/codegen/` not yet built.
- A single `tests/ontology_cypher_round_trip.rs` integration test — the only existing lance-graph touch-point.

What's missing: **no Cargo dep on lance-graph-contract or lance-graph-ontology**, no `crates/woa-bridge`, no `crates/woa-ontology`. Per `super-domain-rbac-tenancy-v1.md` §4.2 table row "`woa-rs` | WorkOrderBilling | WorkOrder | SOX | Bridge shipped" — that "Bridge shipped" refers to `lance_graph_ontology::bridges::WoaBridge` (the upstream side), not a consumer crate inside woa-rs. The consumer side is the work this plan owns.

The end-state goal is **the same shape MedCare-rs is mid-migration toward** (per `MedCare-rs/CLAUDE.md` Architectural commitment #4 "Thinking lives only in lance-graph. No <repo>-thinking duplicate crate"): woa-rs supplies the route shapes + SoA columns; lance-graph supplies the matrix; sea-orm/MySQL stays the writer-parity oracle.


## 2 — Target workspace layout

```
woa-rs/
├── Cargo.toml                       (workspace manifest — adds vendor/lance-graph exclude block per MedCare-rs §workspace-exclude pattern)
├── vendor/
│   ├── ogit/v02-harvest/            (existing — TTL source of truth)
│   └── lance-graph/                 (NEW — softlink to AdaWorldAPI/lance-graph fork, declares own [workspace] so excluded)
├── crates/
│   ├── decimal_money/               (existing)
│   ├── skr_data/                    (existing)
│   ├── buchungs_validator/          (existing)
│   ├── woa_pdf/                     (existing)
│   ├── codegen/                     (RFC v02-006 — not yet built)
│   ├── woa-ontology/                (NEW Tier B — declarations: SemanticType + ObjectView + per-property Marking per the SMB pattern at smb-office-rs/crates/smb-ontology)
│   └── woa-bridge/                  (NEW Tier B — implements lance-graph-contract::{EntityStore, EntityWriter} for sea-orm/MySQL + Lance projection; ships unified_bridge_wiring per unified-bridge-consumer-migration-v1.md D-UB-4)
└── src/                             (existing root binary + lib + routes/*)
```

The `vendor/lance-graph` softlink mirrors `MedCare-rs/vendor/lance-graph` and is the lockfile-package-collision-avoidance pattern that `MedCare-rs/Cargo.toml` `exclude` documents. Same softlink approach for `vendor/ndarray` if ndarray ever becomes a direct dep (today only transitive via lance-graph).

## 3 — Phasing (six phases, additive)

### Phase 0 — Workspace vendor + exclude (1 day)

- **D-WLG-1** — Add `vendor/lance-graph` softlink (or git-submodule) to `AdaWorldAPI/lance-graph@main`. Append `exclude = ["vendor/lance-graph", "vendor/ndarray"]` to root `Cargo.toml [workspace]`. Mirror `MedCare-rs/Cargo.toml` exclude-block comment verbatim — same rationale (multiple-workspace-roots avoidance + lockfile-package-collision). ~5 LOC Cargo.toml + 0 tests (exclude-only change validated by `cargo metadata` passing).
- **D-WLG-2** — Workspace dep declarations: `lance-graph-contract = { path = "vendor/lance-graph/crates/lance-graph-contract" }`, `lance-graph-ontology = { path = "vendor/lance-graph/crates/lance-graph-ontology" }`, with `optional = true` per the medcare-bridge pattern (`MedCare-rs/crates/medcare-bridge/Cargo.toml` `[features] ontology = [...]`). Lets the lean fallback build skip the transitive lance-graph dep graph until features are flipped on. ~10 LOC Cargo.toml + 1 test (`cargo check --features=ontology` succeeds; `cargo check` without features still succeeds and the lance-graph deps are unreachable in dep tree).

### Phase 1 — woa-bridge skeleton + ontology declarations (3 days)

- **D-WLG-3** — `crates/woa-bridge/` new crate, mirroring `smb-office-rs/crates/smb-bridge` shape:
  - `src/lib.rs` — re-exports `lance_graph_ontology::{bridges::WoaBridge, NamespaceBridge, OntologyRegistry}` per the medcare-bridge `#[cfg(feature = "ontology")] pub use ...` pattern.
  - `src/registry.rs` — `WoaRegistry { registry: Arc<OntologyRegistry>, bridge: WoaBridge }` + `hydrate(ttl_root)` + `hydrate_with_report(ttl_root)`. Mirrors `MedCare-rs/crates/medcare-bridge/src/registry.rs::MedcareRegistry` 1:1 with `NAMESPACE = "WorkOrder"` swapped in.
  - `src/unified_bridge_wiring.rs` — `woa_unified_bridge(registry, actor_role, tenant) -> Result<UnifiedBridge<WoaBridge>>`. Mirrors `smb-office-rs/crates/smb-bridge/src/unified_bridge_wiring.rs::smb_unified_bridge` but parameterised over `WoaBridge` (not `OgitBridge`).
  - `Cargo.toml` — gates everything behind `[features] default = []` + `ontology = ["dep:lance-graph-ontology", "dep:lance-graph-contract", "dep:lance-graph-rbac", "dep:lance-graph-callcenter"]`.
  - ~200 LOC + 6 tests (hydration; bridge round-trip; unified-bridge constructor; ontology-feature-off lean fallback compiles; constructor errors on unhydrated registry; one HIPAA-equivalent SOX-redaction-mask round-trip).
- **D-WLG-4** — `crates/woa-ontology/` new crate, mirroring `smb-office-rs/crates/smb-ontology` shape. Builds a `lance-graph-contract::ontology::Ontology` for the WorkOrder namespace, attaching `SemanticType` + `ObjectView` + per-property `Marking` annotations. Concrete entities: Customer, WorkOrder, Position, HistoryEntry, Setting (Tenant), User, Document. Each entity gets a `Marking` per the GDPR / SOX shape (no PHI flags). ~250 LOC + 4 tests.

### Phase 2 — Route-handler integration (4 days)

- **D-WLG-5** — Wire `woa_unified_bridge` into route handlers as Tower middleware. Add `pub struct OntologyState { pub bridge: UnifiedBridge<WoaBridge> }` to `src/state.rs` (or wherever woa-rs holds Axum state). Plumb through `Extension<OntologyState>` in handlers that need ontology resolution (today: zero handlers; the Phase-3 route-codegen output per RFC v02-006 routes.yaml will be the natural consumers). ~80 LOC + 2 tests against a fake hydrated registry.
- **D-WLG-6** — Mandant ↔ TenantId mapping. WoA's existing `Mandant.id` (numeric MySQL primary key) maps to `lance_graph_callcenter::TenantId(u32)`. Add a `From<Mandant> for TenantId` impl + the inverse for `Mandant::find_by_id(TenantId)`. ~40 LOC + 2 tests. Cross-ref RFC-003 (tenant type-state) — the `Customer<Tenanted>` invariant rides on this mapping.
- **D-WLG-7** — Permission ↔ actor_role mapping. WoA's permissions (`buchhaltung`, `mandant_admin`, `kasse`, etc.) map to the `actor_role: &'static str` strings the unified bridge takes. Add `pub fn actor_role_from_user(u: &CurrentUser) -> &'static str` covering each WoA permission. ~50 LOC + 4 tests covering each role transition.


### Phase 3 — Lance-cache + writer-parity Lance-side projection (5 days)

- **D-WLG-8** — Enable `lance-graph-ontology[lance-cache]` feature in `woa-bridge`. Lance dataset at `<state-dir>/.ontology-cache.lance/` (per `unified-bridge-consumer-migration-v1.md` §4.2 schema). Mirrors the medcare path. Boot strategy: TTL-first with Lance-as-mirror (woa-rs is greenfield so cold-start latency is acceptable; HIPAA-style audit traceability is not required but the same dataset makes a useful SOX evidence trail). ~100 LOC + 4 tests (round-trip via the dataset; idempotent re-hydration; checksum mismatch invalidation; Lance-fallback when TTL root missing).
- **D-WLG-9** — Lance-side projection of WoA's MySQL tables. Implement `lance-graph-contract::repository::EntityWriter` for the Customer, WorkOrder, Position, Tenant, User entities — projecting sea-orm `ActiveModel` writes into a parallel Lance dataset. This is the **woa-rs side of writer-parity** (per the 2026-05-15 DualSink-Pivot: Python and Rust both write MySQL; this adds Lance as a third witness without becoming authoritative). Mirrors `smb-office-rs/crates/smb-bridge/src/lance.rs` shape. ~400 LOC + 8 parity tests (per-entity insert; per-entity update; per-entity soft-delete; row-count parity against MySQL). Feature-gated behind `woa-bridge/lance` (off by default; on only in environments wanting the third witness).
- **D-WLG-10** — RLS policy build via `OntologyRegistry::enumerate("WorkOrder")`. Mirrors the medcare-rs critical gap (D-UB-8 in `unified-bridge-consumer-migration-v1.md`) but woa-rs greenfield so no fail-OPEN window — RLS lands fresh against the enumerated entity set. ~80 LOC + 4 tests. Per-entity `(table, tenant_id)` predicates plus optional `(table, tenant_id, ontology_context_id)` third axis for cross-tenant Mahnwesen / Logbook visibility.

### Phase 4 — Cypher / SPARQL surface (10 days, opt-in)

- **D-WLG-11** — Wire `lance-graph-planner` as workspace dep behind `[features] planner = ["dep:lance-graph-planner"]`. Planner's 16 strategies cover Cypher / GQL / Gremlin / SPARQL parsing → DataFusion. Initial surface: a single `POST /api/__graph` endpoint that accepts a Cypher string and returns Arrow batches. ~150 LOC + 4 tests (Cypher MATCH against a fake hydrated Customer set; SPARQL via the same endpoint with content-negotiation; error path for invalid syntax; tenant-isolation predicate auto-injected per the `UnifiedBridge::authorize` 4-stage flow lowering to DataFusion §3.10).
- **D-WLG-12** — Cross-table queries: WoA's "find all Mahnungen for Customers whose Vorgang.status = 'rechnung_offen' across all tenants this user has access to" — currently this is a hand-rolled sea-orm join. Re-express as Cypher; verify the planner returns identical results to the sea-orm reference. ~80 LOC test + 2 production handlers as examples (Mahnwesen listing + Stundenzettel aggregation). **This is the "lance-graph is the obligatory spine" lift** — the moment Cypher returns to a route handler, the spine is real for WoA.

### Phase 5 — CAM-PQ similarity surface (5 days, opt-in)

- **D-WLG-13** — `EntityStore::similar_to(entity_id, limit)` over Lance + CAM-PQ. WoA use case: "customers similar to Customer X" for sales-pipeline recommendations. Backed by per-entity Vsa16kF32 fingerprints (the Customer fingerprint summarises Customer.address + Customer.industry + recent-Vorgang-history). ~250 LOC + 4 tests. Feature-gated behind `woa-bridge/cam-pq`.
- **D-WLG-14** — Cohort-style similarity for Logbook entries: "customers with similar incident histories." Same shape as D-WLG-13 but over Logbook. ~150 LOC + 2 tests.

## 4 — Build invariants (per super-domain-rbac-tenancy-v1.md §19)

| Layer | Pin |
|---|---|
| `rust-version` | `1.94.1` (lance-graph workspace MSRV; portable_simd patterns) |
| `lance` | `=4.0.0` (exact pin) |
| `lancedb` | `0.27.2` (caret, PR #275) |
| `ndarray` | path = `vendor/ndarray` (softlink, if used; otherwise transitive via lance-graph) |
| SIMD layer | `ndarray::simd::*` (canonical; never raw `std::simd` or hand-rolled intrinsics) |

The woa-rs `Cargo.toml` currently sets `rust-version = "1.95"`. Phase 0 either bumps the lance-graph workspace to 1.95 (preferred — woa-rs's pin is the tighter constraint) or backs woa-rs to 1.94.1 (acceptable as a transient until both align).

## 5 — Cross-references

- `super-domain-rbac-tenancy-v1.md` §3.9 (`UnifiedBridge::authorize`) — what `woa_unified_bridge` returns
- `super-domain-rbac-tenancy-v1.md` §13.1 (`PolicyRewriter` composition) — what the authorize chain composes onto
- `super-domain-rbac-tenancy-v1.md` §19 (pinned versions + ndarray::simd) — Phase 0 build constraints
- `unified-bridge-consumer-migration-v1.md` D-UB-4 — `woa_unified_bridge` constructor (this plan's Phase 1)
- `lance-graph/crates/lance-graph-ontology/src/bridges/woa_bridge.rs` — the `WoaBridge` type this plan's bridge wraps
- `woa-rs/rfcs/v02-006-route-codegen-and-ontology-unification.md` — orthogonal RFC; route codegen consumes the same OGIT TTL surface but doesn't itself require lance-graph runtime. Phase 2 of this plan is the natural counterpart to Phase 1 of v02-006.
- `woa-rs/.claude/board/Goldstaub.md` 2026-05-15 ("DualSink-Pivot") — writer-parity invariant this plan respects (MySQL stays authoritative; Lance is a third witness, not a replacement)
- `MedCare-rs/Cargo.toml` workspace-exclude block — verbatim pattern this plan's Phase 0 mirrors
- `smb-office-rs/crates/smb-bridge/src/unified_bridge_wiring.rs` — the working reference for the Phase-1 constructor

## 6 — Open questions

1. **Phase 4 ceiling.** Is Cypher/SPARQL via `lance-graph-planner` actually wanted in woa-rs production, or is the goal stopping at Phase 3 (ontology + RBAC + Lance-as-third-writer) for the foreseeable future? Affects ~10 days of Phase 4 effort.
2. **Mandant ↔ TenantId surjection.** WoA's `Mandant.id` is `i32` not `u32`. Negative values are absent in production data but Rust's type system can't see that. D-WLG-6 needs either a `TryFrom` (with `MandantIdNegative` error variant) or a workspace invariant that asserts non-negative at the sea-orm boundary.
3. **Permission table evolution.** WoA's permissions are stored in `User.permissions: JSON` per the Python source. The `actor_role_from_user` mapping (D-WLG-7) assumes a fixed enumeration; if WoA adds a new permission tomorrow, where does the new `actor_role` slot live? Likely in a `RoleGroup` registry that hydrates from a small TTL/YAML at boot, mirroring how MedCare-rs handles `physician` / `nurse` / etc.
4. **rust-version alignment.** woa-rs is on 1.95; lance-graph workspace on 1.94.1. Bumping lance-graph to 1.95 is the cleanest path but needs a workspace-wide PR. Pinning woa-rs to 1.94.1 is the temporary alternative.

## 7 — Status

- **Phase 0:** Not started. Mechanical (~1 day).
- **Phase 1:** Not started. Blocked on Phase 0. Mostly mirroring known patterns from MedCare-rs and smb-office-rs (~3 days, low risk).
- **Phase 2:** Not started. Blocked on Phase 1. Per-handler integration is the labour-intensive part (~4 days).
- **Phase 3:** Not started. Blocked on Phase 1. Lance-side projection benefits significantly from the smb-office-rs reference (~5 days).
- **Phase 4:** Open question whether to ship at all (~10 days if yes).
- **Phase 5:** Phase 4-dependent (~5 days if Phase 4 ships).

**Confidence:** Working — every step has a 1:1 mirror in MedCare-rs or smb-office-rs. The architecture is locked upstream; woa-rs is the consumer.

## 8 — One-line summary

> Five additive phases lift woa-rs from "OGIT TTL vendored, sea-orm + MySQL writer-parity" to "ontology + RBAC + Lance-third-writer with an optional Cypher/CAM-PQ surface", each step mirroring an existing pattern from smb-office-rs or MedCare-rs. Phase 0 is mechanical; Phase 1 lands the `woa-bridge` + `woa-ontology` crates; Phase 4+ is opt-in.


---

## 9 — Hot path / cold path refinement (2026-05-21, same session)

User clarification: **woa-rs already has sea-orm as the DTO layer + the MySQL bridge.** The plan is to use `lance-graph-ontology` as the **hot path** and MySQL / sea-orm as the **cold path**. This refines (does not replace) §1–§8.

### 9.1 What "hot path" means here

The hot path is **every per-request operation that needs to identify, authorize, label, or compare an entity at O(1) or O(log n)**. Specifically:

| Hot-path operation | Carrier | Why hot |
|---|---|---|
| "What kind of entity is this row?" — Customer vs WorkOrder vs Position vs Tenant vs User | `OgitFamilyTable.lookup(owl_id) → FamilyEntry.kind` | one masked u16 compare → one array index. Sub-microsecond. |
| "What's the canonical OGIT URI for this row?" | `FamilyEntry.label_uri` | inline in the same `FamilyEntry`; no second fetch. |
| "What's the rdfs:label / German UI string for this row?" | `FamilyEntry.label_uri` resolved via the OWL/DOLCE cross-walk in `MetaAnchors` | static lookup, cached at boot. |
| "Can this actor read this row?" | `UnifiedBridge::authorize(owl, row_tenant, op)` — 4-stage masked-predicate combine | one DataFusion predicate vector. Sub-microsecond per row. |
| "Which rows are similar to this one?" | per-entity Vsa16kF32 + CAM-PQ scan over the Lance projection | O(log n) via codebook compression; no MySQL touch. |
| "Cross-tenant referral visibility under §73 SGB V" | `ontology_context_id` predicate (third RLS axis) | one extra masked compare in the same predicate vector. |
| "Which permissions does this actor have on this slot?" | `RoleGroup.redaction_mask.{readable,writable,redacted}_slots[slot]` | one bit-test in a 256-bit set. |
| "Cypher / SPARQL / Gremlin query" | `lance-graph-planner` → DataFusion plan over the Lance projection | planner's 16 strategies; CAM-PQ-aware. |

**None of these hit MySQL.** They all resolve in-process from the registry (`OntologyRegistry` for the ontology surface; `Policy` for RBAC; Lance datasets under the `lance-cache` feature for persistence of the registry itself plus the per-entity projection).

### 9.2 What "cold path" means here

The cold path is **every operation that mutates business state OR reads a row's authoritative byte-exact value**. Specifically:

| Cold-path operation | Carrier | Why cold |
|---|---|---|
| Insert / update / delete a Customer / WorkOrder / Position / Mahnung row | `sea-orm Entity::insert(.).exec(&db)` → MySQL | DualSink-Pivot 2026-05-15: writer-parity with the Python source is the spec. MySQL is the system of record. |
| Read a row's exact field values (`Customer.firma`, `WorkOrder.betreff`, `Position.netto_summe`) | sea-orm `find_by_id` → MySQL | the byte-exact value is what the Python source produces; parity tests in `tests/parity/` compare row-by-row. |
| DATEV export / X-Rechnung generation / PDF rendering | sea-orm reads → woa_pdf / datev_encoder | output must be byte-identical to Python; MySQL is the only source where this is true. |
| Schema migration | sea-orm migrations | the migrations are the contract between MySQL versions. |
| Bulk historical scans (audit / GoBD retention / financial year close) | sea-orm or raw SQL | reading the audit-trail authoritative state, not the hot-path projection. |

**The cold path is authoritative.** When the hot path and the cold path disagree, the cold path wins. Drift detection rides on this asymmetry.

### 9.3 The pipe between hot and cold

Three concrete bridges:

1. **Write-through projection** (D-WLG-9). Every sea-orm `Entity::insert/update/delete` on a hot-path-projected entity (Customer, WorkOrder, Position, Tenant, User) also dual-writes a Lance row via `lance-graph-contract::repository::EntityWriter`. The write is **synchronous within the same transaction boundary** for the Customer/WorkOrder/Position/Tenant/User set (the entities whose row identity needs to be visible to the next read on the same request); other entities can dual-write async.
2. **Boot-time projection** for entities not yet write-through-projected: a one-shot scan at startup reads every row from MySQL and lands it in the Lance projection. After startup, the write-through path keeps the two in sync.
3. **Drift reconciler** (cron job, opt-in): periodically scans both stores, compares MerkleRoots per-row (computed over the canonical fingerprint of each row's authoritative field set), emits a drift event for any mismatch. Mirrors the `ParityWitness` shape from `MedCareV2/MedCare_2.0/LanceProbe/` per `super-domain-rbac-tenancy-v1.md` §18.2. **Cold path wins on reconciliation:** the Lance projection is rebuilt from MySQL on conflict, never the other way around.

### 9.4 Phase remapping (replaces §3 phase-level framing)

The six phases stay; their internal framing tightens:

| Phase | Hot-path delivery | Cold-path delivery |
|---|---|---|
| 0 (vendor + exclude) | none (mechanical) | none |
| 1 (woa-bridge + woa-ontology) | the `OntologyRegistry` + `WoaBridge` + `UnifiedBridge<WoaBridge>` constructor (every subsequent hot-path operation rides on this) | none |
| 2 (route-handler integration) | wire `UnifiedBridge` into route handlers as Tower middleware (`OntologyState` extension) | unchanged — sea-orm reads stay the cold-path read for byte-exact values |
| 3 (lance-cache + Lance projection) | `lance-cache` feature persists the registry as a LanceDB column (cold-start latency drops); D-WLG-9 stands up the Lance projection of MySQL tables | sea-orm + MySQL stays authoritative; **the projection is a hot-path read replica, NOT a write replacement** |
| 4 (Cypher / SPARQL) | `POST /api/__graph` endpoint over the planner → DataFusion → Lance projection | cold path unchanged |
| 5 (CAM-PQ similarity) | `EntityStore::similar_to` over Lance + CAM-PQ | cold path unchanged |

The **write-through synchronisation** for the Customer/WorkOrder/Position/Tenant/User set lands in Phase 3 (D-WLG-9) as part of the Lance projection. The drift reconciler is a Phase 3 follow-on or Phase 4 opt-in.

### 9.5 What this means for the existing sea-orm code

**Nothing changes about the existing sea-orm code path.** Every route handler that today does `Customer::find_by_id(db, id).await?` keeps doing that. The hot-path integration is **additive**: a route handler that needs ontology resolution or RBAC or similarity adds an `OntologyState` extension parameter and calls `state.bridge.authorize(...)` / `state.registry.resolve(...)`; a route handler that just reads a row's fields stays on sea-orm.

The win is at **routes that today hand-roll cross-entity joins, similarity heuristics, or permission matrices**. Those routes — Mahnwesen aggregation, Stundenzettel cross-customer rollup, "find similar customers" pipelines — become Cypher queries (Phase 4) or CAM-PQ calls (Phase 5) without the hand-rolled join code. The cold-path sea-orm code stays for the byte-exact-row reads those hot-path queries reduce TO.

### 9.6 Consequence for D-WLG-9 scope

D-WLG-9 in §3 was framed as "Lance-side projection of WoA's MySQL tables ... woa-rs side of writer-parity." The hot/cold split tightens this:

- **Lance projection is the hot-path READ replica**, not a writer-parity peer.
- The 2026-05-15 DualSink-Pivot's "Python + Rust both write MySQL" stays the writer-parity contract; **Lance is NOT a third writer-parity peer**.
- D-WLG-9's parity tests compare sea-orm read → MySQL → row vs. EntityStore read → Lance → row for the Customer/WorkOrder/Position/Tenant/User entity set, asserting they agree under the write-through invariant.
- If Lance and MySQL disagree, **MySQL wins** and Lance is rebuilt from MySQL. The Lance projection is replicate, not source.

### 9.7 Open questions (refines §6)

1. **Sync vs async dual-write boundary.** Phase 3 needs to commit which entities sync-dual-write (request-scope-visible) vs async-dual-write (eventually-consistent). My initial pick: Customer / WorkOrder / Position / Tenant / User sync; Mahnung / Stundenzettel-Eintrag / Logbook-Eintrag / Dokument / Setting async. Needs validation against actual route-handler read-after-write patterns.
2. **Hot-path-only entities** (entities that live only in Lance, no MySQL row). E.g., per-entity Vsa16kF32 fingerprints, CAM-PQ codebooks, drift-event log. Phase 5 onwards. These do NOT have a cold-path MySQL counterpart by design.
3. **Cypher rewrite of mid-complexity routes.** The §6 question "is Cypher actually wanted in production" sharpens: yes for cross-entity queries that today hand-roll sea-orm joins; not yet for trivial single-entity find-by-id (those stay on sea-orm). The Phase 4 ceiling becomes "rewrite the 6-8 cross-entity queries that produce the most join code, leave the rest." That's ~1 week not ~2.
4. **Drift reconciler cadence.** Hourly vs nightly vs continuous. Probably nightly for v1 — Phase 4 / 5 features that depend on the projection being recent (similarity search) tolerate one-day-lagged drift if the underlying business velocity is sub-daily.

### 9.8 Cross-references (additive)

- `woa-rs/.claude/board/Goldstaub.md` 2026-05-15 ("DualSink-Pivot") — explicitly preserved; this section refines its read-side framing, not its writer-parity contract.
- `MedCare-rs/CLAUDE.md` § Architectural commitments #1 ("MySQL is the permanent oracle / parity witness") — the same role MySQL plays in MedCare-rs is what it plays here on the woa-rs cold path.
- `super-domain-rbac-tenancy-v1.md` §17.7 ("Net architecture summary") — the per-row LanceDB layout (tenant_id u32 + owl_id u16 + ciphertext + merkle_root cleartext) is the hot-path projection shape this plan inherits.
- `super-domain-rbac-tenancy-v1.md` §16.5 ("MerkleRoot-cleartext-beside-ciphertext") — the drift reconciler in §9.3 of this plan compares MerkleRoots between MySQL and Lance per-row.

### 9.9 One-line summary update

> lance-graph-ontology is the hot path (resolution + RBAC + label + similarity + Cypher in O(1) or O(log n) over the codebook + Lance projection); sea-orm + MySQL is the cold path (writer-parity authoritative; byte-exact row values; the projection rebuilds from MySQL on drift). The 2026-05-15 DualSink-Pivot writer-parity contract is preserved; Lance is a READ replica, not a third writer.


---

## 10 — Ontology-virgin at the hot path, but OGIT already in sea-orm at the cold path (2026-05-21, same session)

User correction: **woa-rs is ontology-virgin at the lance-graph-ontology hot path, BUT it already has the OGIT ontology baked into sea-orm entities at the cold path.** This collapses the work-remaining estimate significantly compared to a true greenfield port.

### 10.1 What's already in tree at the cold-path DTO layer

`vendor/ogit/v02-harvest/entities/*.ttl` is not just vendored TTL sitting unused. It's the **source the existing sea-orm entities were generated/hand-mirrored from**. Per `woa-rs/Cargo.toml` workspace structure + `RFC v02-006` (route codegen + ontology unification, DRAFT):

| Layer | woa-rs status today |
|---|---|
| **Source-of-truth TTL** | `vendor/ogit/v02-harvest/entities/*.ttl` + `vendor/ogit/v02-harvest/POLICY.md` (the divergence ledger) |
| **Sea-orm entities** | Customer / WorkOrder / Position / Tenant / User / Mahnung / Logbook / Dokument / Einsatz / Stundenzettel-Eintrag etc. — all derived from the TTL, hand-edited where Python source diverged. RFC v02-006 §"Ontology unification" §"Layer table" makes the mapping explicit. |
| **MySQL schema** | sea-orm migrations carry the column shapes. DualSink-Pivot 2026-05-15 locks Python+Rust both writing the same MySQL. |
| **Wire DTOs** | `src/dto/*.rs` — hand-written today; future codegen from TTL per RFC v02-006 Phase 5. |
| **lance-graph-ontology hot-path** | NOT WIRED — no `woa-bridge` crate, no `OntologyRegistry` consumption, no `UnifiedBridge<WoaBridge>` constructor. |

**The "ontology virgin" framing is correct at the hot path only.** At the cold path (sea-orm + MySQL writer-parity), the OGIT ontology IS already structurally present — each sea-orm entity name + its property set + its German/English wire labels (per `ogit:label` / `rdfs:label`) maps to the TTL's `ogit.WorkOrder:*` URIs. The mapping is a hand-mirroring today (per RFC v02-006 layer-by-layer status), not a codegen output yet, but the shape is established.

### 10.2 The MedCare-rs reconciler pattern transfers directly

`MedcareMysqlReconciler` (`MedCare-rs/crates/medcare-analytics/src/mysql_reconciler.rs`, 461 LOC, Round-1 = Patient) uses pluggable `PatientFetcher` closures so the production wiring is one config-site away:

```rust
// medcare-rs production pattern (what's in tree):
pub trait PatientFetcher {
    fn fetch_mysql(&self, id: u64) -> Option<CanonicalPatientRow>;
    fn fetch_lance(&self, id: u64) -> Option<CanonicalPatientRow>;
}
// Production impl wraps medcare_db::queries::patient::get_patient(...)
```

**woa-rs gets the same shape for free** because sea-orm is ergonomic and self-explanatory. The Rust port is:

```rust
// woa-rs production pattern (to write):
pub trait CustomerFetcher {
    fn fetch_mysql(&self, kdnr: i32) -> Option<CanonicalCustomerRow>;
    fn fetch_lance(&self, kdnr: i32) -> Option<CanonicalCustomerRow>;
}

// Production impl wraps sea-orm:
impl CustomerFetcher for &DbConnection {
    fn fetch_mysql(&self, kdnr: i32) -> Option<CanonicalCustomerRow> {
        // sea-orm find_by_id is ~3 lines; tokio block_on at the trait boundary
        // or change the trait to async fn (preferred)
        let row = customer::Entity::find_by_id(kdnr).one(self).await.ok().flatten()?;
        Some(CanonicalCustomerRow {
            kdnr: row.kdnr,
            firma: row.firma.unwrap_or_default(),
            // ... 7 more fields the OGIT ontology declares for Customer
        })
    }
    // fetch_lance wraps the WoaConnector::find_by_id over the Lance projection
}
```

The reconciler shell (the `parse_<route>_route` + `diff_<entity>_rows` + `build_event` machinery) is verbatim from MedCare-rs. Only the `Canonical<Entity>Row` types and the `<Entity>Fetcher` traits are per-consumer.

### 10.3 Phase 1 revision — woa-bridge is not greenfield, it's a MedCare-rs port

§3 Phase 1 (`D-WLG-3` woa-bridge skeleton, `D-WLG-4` woa-ontology, ~200 + 250 LOC) was framed as mirroring MedCare-rs and smb-office-rs. The §10 reframe sharpens this: **the mirror is MedCare-rs specifically**, not smb-office-rs. Reasoning:

| Aspect | MedCare-rs (mirror source for woa-rs) | smb-office-rs |
|---|---|---|
| Cold-path store | MySQL via sea-orm | MongoDB via BSON wire shape |
| Cold-path fetcher | `medcare_db::queries::*` (sea-orm-shaped) | `MongoConnector::scan(...)` (BSON document iter) |
| Reconciler | `MedcareMysqlReconciler` (MySQL ↔ Lance) | `SmbMongoReconciler` (Mongo ↔ Lance) |
| Bridge constructor | `medcare_unified_bridge` (TODO, D-LGMC-4) | `smb_unified_bridge` (shipped, parameterised over OgitBridge) |
| Lance-cache wiring | `MedcareRegistry::hydrate_with_report(ttl_root)` returns registry + bridge + HydrationReport | `smb-bridge[lance]` feature gates the LanceConnector |

woa-rs's sea-orm + MySQL shape is **homologous to MedCare-rs's medcare_db + MySQL shape.** The reconciler + constructor + registry-helper all mirror MedCare's structure with sea-orm queries substituted for `medcare_db::queries::*`. Smb-office-rs is the template SOURCE for the bridge-wiring pattern (per `lance-graph-in-smb-office-rs-v1.md` §7), but for the storage-tier MIRROR, MedCare-rs is the right reference.

### 10.4 Revised effort estimate

| Phase | Original §3 estimate | §10 revised |
|---|---|---|
| Phase 0 (vendor + exclude) | 1 day mechanical | **Unchanged** — 1 day. |
| Phase 1 (woa-bridge + woa-ontology) | 3 days, "mirroring MedCare-rs and smb-office-rs references" | **2 days revised.** The sea-orm entity authoring is already done; Phase 1 is `WoaRegistry::hydrate()` helper (~50 LOC mirror of `MedcareRegistry`) + `woa_unified_bridge` constructor (~50 LOC mirror of `smb_unified_bridge` with `WoaBridge` type param) + `woa-ontology` declarations crate (~100 LOC; smaller than MedCare's because the entity shapes already exist in sea-orm, so the ontology crate is mostly the SemanticType + Marking + ObjectView annotations, not the entity declarations themselves). Total ~200 LOC + 4 tests. |
| Phase 2 (route-handler integration) | 4 days, "labour-intensive" | **3 days revised.** `OntologyState` extension + Mandant↔TenantId mapping + permission↔actor_role mapping (~170 LOC + 8 tests). Fewer handlers than original framing because RFC v02-006 codegen (when it lands) will absorb most route-shape work; the manual integration is just for the routes touching the unified bridge surface (~6-8 handlers initially). |
| Phase 3 (lance-cache + Lance projection) | 5 days | **3-4 days revised.** D-WLG-8 (`lance-cache` feature, ~100 LOC + 4 tests) unchanged. D-WLG-9 (Lance-side projection of MySQL tables) IS the WoaMysqlReconciler — NEW deliverable below, see §10.5. D-WLG-10 (RLS via `OntologyRegistry::enumerate("WorkOrder")`) unchanged. |
| Phase 4 + 5 (Cypher / SPARQL, CAM-PQ) | 10 + 5 days opt-in | **Unchanged** — opt-in adopts the planner + CAM-PQ surfaces. |

**Phase 0-3 closure revised from ~13 days to ~9-10 days** with the sea-orm OGIT shortcut + MedCare-rs reconciler pattern transfer. Phases 4-5 stay opt-in (~15 additional days if both ship).

### 10.5 New deliverable — WoaMysqlReconciler

- **D-WLG-15 (NEW)** — `woa-rs/crates/woa-bridge/src/mysql_reconciler.rs::WoaMysqlReconciler<F: CustomerFetcher>` mirroring `medcare-analytics::mysql_reconciler::MedcareMysqlReconciler` 1:1 with `CanonicalCustomerRow` (8-10 fields per `vendor/ogit/v02-harvest/entities/Customer.ttl`) substituted for `CanonicalPatientRow`. Same `Reconciler` trait impl, same `DriftKind` (Match / ValueDrift / ShapeDrift / MissingMysql / MissingLance), same pluggable-fetcher pattern for unit testing. Round-1 scope: `/api/customers/{kdnr}` (the WoA Kunden detail route). ~80 LOC + 11 tests (mirroring the medcare-rs reconciler test suite verbatim). **Round-2 expansion (WorkOrder / Position / Mahnung / Tenant) lands as separate ~80 LOC + 5 test PRs on the same shell.**
- **D-WLG-16 (NEW)** — Production query-handle wiring for `WoaMysqlReconciler`'s `CustomerFetcher`: wraps `customer::Entity::find_by_id(kdnr).one(&db).await` for the MySQL side + the corresponding Lance read via `WoaConnector::find_by_id(kdnr)` for the SPO/Lance side. ~50 LOC + 2 integration tests against a real MySQL fixture + Lance dataset (gated behind `--features mysql-integration lance-phase2`).
- **D-WLG-17 (NEW)** — `POST /api/__parity/csharp` (or equivalent — there is no C# parity tool for woa-rs yet) — actually NOT needed; woa-rs has no diverse-redundancy client like MedCareV2 LanceProbe. The reconciler's drift events sink directly to the persistent `LanceAuditSink` (or in-process ring buffer until D-LGMC-21 lands LanceAuditSink upstream). ~40 LOC + 2 tests for the drift-event dashboard endpoint `GET /api/__parity`.

### 10.6 Implication for the cross-consumer harvest order (per smb-office-rs §7.4)

Updated harvest diagram:

```
              ┌─────────────────────────────────────────────────┐
              │ smb-office-rs                                   │
              │  ships UnifiedBridge wiring template            │
              │  (smb_unified_bridge as the reference)          │
              └────────────────────┬────────────────────────────┘
                                   │ pattern harvest
                ┌──────────────────┴──────────────────┐
                │                                     │
                ▼                                     ▼
   ┌───────────────────────────────┐    ┌───────────────────────────────┐
   │ MedCare-rs                    │    │ woa-rs                        │
   │  has: parallelbetrieb shipped │    │  has: OGIT in sea-orm; cold   │
   │       + MysqlReconciler shell │    │       path entities established│
   │  needs: medcare_unified_bridge│    │  needs: woa-bridge crate +    │
   │         constructor + Phase 5b│    │         WoaMysqlReconciler    │
   │         Round-2 expansion      │    │         (mirror MedCare      │
   │                                │    │          MysqlReconciler 1:1)│
   └────────────────┬───────────────┘    └───────────────────────────────┘
                    │                                     ▲
                    │ reconciler-pattern harvest          │
                    └─────────────────────────────────────┘
```

**Two-axis harvest:** smb-office-rs → bridge-wiring template (`<repo>_unified_bridge`); MedCare-rs → reconciler-pattern (`<repo>MysqlReconciler` + cross-source diff). woa-rs absorbs both axes; the sea-orm cold-path it already has makes the MedCare-rs axis particularly clean to copy.

### 10.7 One-line summary

> woa-rs is ontology-virgin at the lance-graph-ontology hot path BUT already has OGIT baked into sea-orm entities at the cold path; the MedCare-rs MysqlReconciler pattern transfers 1:1 with sea-orm fetchers (sea-orm ergonomics + self-explaining API make this cheap). Phase 0-3 closure revised from ~13 to ~9-10 days; new D-WLG-15..17 add WoaMysqlReconciler + production wiring + drift dashboard mirroring the MedCare-rs shape.

---

## 11 — The rewarding path: woa-rs as integration target, harvesting XRechnung + parallelbetrieb (2026-05-21, same session)

User strategic reframe: **MedCare-rs and smb-office-rs are transcodes that never worked as Rust** — they're pre-prod Rust ports of C# WinForms desktop apps whose UIs can't be moved to web cheaply. They compile, they pass tests, but they don't "look like anything" until the WinForms layer is replaced (a separate large effort outside this plan's scope). **woa-rs is different — it's a web application transcode** (Python/Flask → Rust/axum), which means the same kind of work that produces binaries-only feedback in MedCare/SMB produces VISIBLE WEB-UI feedback in woa. That asymmetry is what makes woa-rs the rewarding integration target.

§9 (hot/cold split) + §10 (OGIT-in-sea-orm) already revised the technical effort estimate downward. §11 reframes the STRATEGIC sequencing — woa-rs is where the integration story should land first, because every PR shows up in a browser, not just in a test report.

### 11.1 Three harvests, one rewarding target

```
                               woa-rs
                  ┌──────────  (web-app target)  ──────────┐
                  │                                         │
                  │   already in tree:                      │
                  │   • OGIT TTL at vendor/ogit/v02-harvest │
                  │   • sea-orm entities mirrored from OGIT │
                  │   • MySQL via DualSink-Pivot writer-    │
                  │     parity                              │
                  │   • axum + askama + tower-sessions      │
                  │     web stack                           │
                  │                                         │
                  └────────────────┬────────────────────────┘
                                   │
                ┌──────────────────┼────────────────────────┐
                ▼                  ▼                        ▼
   ┌─────────────────────┐ ┌──────────────────┐ ┌─────────────────────┐
   │ HARVEST 1 (SMB):    │ │ HARVEST 2 (Med): │ │ HARVEST 3 (in-tree):│
   │ XRechnung pattern   │ │ parallelbetrieb  │ │ OGIT wiring         │
   │                     │ │ reconciler       │ │                     │
   │ source files:       │ │ source files:    │ │ source files:       │
   │ • hydrate_zugferd   │ │ • MedcareMysql-  │ │ • vendor/ogit       │
   │ • hydrate_zugferd_  │ │   Reconciler     │ │ • sea-orm entities  │
   │   rules             │ │   (mysql_recon-  │ │ • lance-graph-      │
   │ • SchematronHydrator│ │   ciler.rs, 461  │ │   ontology hydrators│
   │ • XsdHydrator       │ │   LOC, 11 tests) │ │   (hydrate_dolce +  │
   │ • collect_xsd_files │ │ • parallelbe-    │ │   provo + qudt +    │
   │ • EN16931 rule set  │ │   trieb trait    │ │   schemaorg +       │
   │                     │ │ • C# parity      │ │   fibo_fnd + skr03/ │
   │ delivers to woa-rs: │ │   probe pattern  │ │   skr04)            │
   │ X-Rechnung output   │ │                  │ │                     │
   │ for Stefan's        │ │ delivers:        │ │ delivers:           │
   │ invoices (visible   │ │ MySQL ↔ Lance    │ │ /api/__graph route  │
   │ in browser as       │ │ drift dashboard  │ │ exposing Cypher     │
   │ downloadable XML +  │ │ at /api/__parity │ │ over the woa        │
   │ rendered PDF)       │ │ (visible admin   │ │ ontology (visible   │
   │                     │ │ panel)           │ │ in browser)         │
   └─────────────────────┘ └──────────────────┘ └─────────────────────┘
```

### 11.2 Why this is "quick and rewarding"

The reward function is **time-to-visible-feedback per PR**. For each plan:

| Consumer | Time-to-visible per PR | Why |
|---|---|---|
| **woa-rs** | **Minutes.** Every PR adds a route handler, an askama template, a database column, or a config that surfaces in the browser. PR-1 lands → smoke test against `localhost:8080/vorgaenge/123` shows the change. | It's a web app. Every change has a URL. |
| **smb-office-rs** | Days-to-weeks. PRs land in `smb-bridge`, `smb-ontology`, `smb-mongo`, but the C# WinForms UI on the customer's desktop is what makes the change visible — and that UI isn't ours to touch. FFI smoke tests confirm linkage, not user-visible behavior. | The desktop UI is outside the port. |
| **MedCare-rs** | Days-to-weeks. Same shape — `medcare-server` is technically axum, but the customer-facing experience is the C# WinForms MedCare app via the parity tool. Drift dashboards are admin-only and surface only when the C# probe runs. | Customer-facing surface is C#. |

**This explains why §10's "MedCare is the quickest target for end-to-end proof" is true for ENGINEERING but woa-rs is the right target for DEMONSTRATION.** Engineering completeness ≠ user-visible reward. Stefan (the WoA end user, per `woa-rs/CLAUDE.md` glossary) sees the wins in his browser the day a PR lands.

### 11.3 The harvest sequence — concrete PR ladder

Suggested PR ladder for the "quick and rewarding" path. Each PR is one ascending integration milestone visible in the browser:

| # | Source | Lands in woa-rs as | Visible in browser as |
|---|---|---|---|
| 1 | (greenfield) | Phase 0 vendor symlinks + workspace exclude | (build green; no UI change yet) |
| 2 | smb-office-rs `smb-bridge` shape + MedCare-rs `MedcareRegistry` | `crates/woa-bridge/` + `crates/woa-ontology/` skeleton crates + `WoaRegistry::hydrate(...)` helper (D-WLG-3/D-WLG-4 per §3) | (build green; cargo test passes; no UI yet) |
| 3 | smb-office-rs `unified_bridge_wiring.rs::smb_unified_bridge` | `woa_unified_bridge(registry, actor_role, tenant)` constructor (D-UB-4 per `unified-bridge-consumer-migration-v1.md`) | (`/api/v1/health` returns "ontology hydrated: WorkOrder" — first visible signal) |
| 4 | in-tree OGIT + lance-graph-ontology hydrators | Boot-time hydration menu: `hydrate_dolce + hydrate_provo + hydrate_qudt + hydrate_schemaorg + hydrate_fibo_fnd + hydrate_skr03 + hydrate_skr04` (D-WLG-8 lance-cache feature) | (`/api/__ontology` admin route lists hydrated G-slots + entity counts per family) |
| 5 | **HARVEST 1 (SMB):** `hydrate_zugferd` + `hydrate_zugferd_rules` + `SchematronHydrator` + `XsdHydrator` | woa-rs ZUGFeRD/Factur-X invoice generator: `POST /vorgaenge/<wid>/rechnung/xrechnung` → returns conformant XML + downloadable Factur-X PDF | **Stefan clicks "X-Rechnung erstellen" on a workorder, downloads a valid EN16931-conformant invoice. Visible reward.** |
| 6 | **HARVEST 2 (MedCare):** `MedcareMysqlReconciler` shell + `parallelbetrieb::{Reconciler, DriftEvent, DriftField, DriftKind}` trait | `WoaMysqlReconciler<CustomerFetcher>` with sea-orm fetchers (D-WLG-15/16) + admin route `GET /api/__parity` (D-WLG-17 mirrors medcare-server's parity.rs ring buffer) | **Admin opens `/admin/parity` and sees green/red drift status per Customer row across MySQL ↔ Lance. First-class reconciler dashboard.** |
| 7 | smb-office-rs + MedCare-rs combined: RLS via `OntologyRegistry::enumerate("WorkOrder")` (D-WLG-10) | Per-tenant row filter on every route handler; `tenant_get_or_404` becomes the unified-bridge `authorize(owl, row_tenant, Read)` call | **Cross-tenant URL-guessing returns 404 (not 403); admin sees the per-tenant scope active in the parity panel.** |
| 8 | (opt-in) Phase 4 — lance-graph-planner Cypher endpoint | `POST /api/__graph` accepts Cypher / SPARQL / GQL | **`/admin/graph` becomes a query playground: "MATCH (c:Customer)-[:HAS_WORKORDER]->(wo:WorkOrder) WHERE wo.status = 'offen' RETURN c.firma, wo.betreff" returns results from the Lance projection.** |
| 9 | (opt-in) Phase 5 — CAM-PQ similarity | `EntityStore::similar_to` over Lance | **`/kunden/<kdnr>/similar` returns the 10 most-similar customers by address + industry + recent-Vorgang-history. Sales-pipeline feature.** |

Each rung produces a screenshot. Compare: an equivalent migration in MedCare-rs produces an audit-log entry in JSON, visible only to ops.

### 11.4 What this changes about the harvest order across the three plans

Previous framings:
- §10.6 (this plan) — "two-axis harvest from smb (bridge-wiring) + MedCare (reconciler pattern)"
- `lance-graph-in-smb-office-rs-v1.md` §7.4 — "smb ships UnifiedBridge template; MedCare + woa absorb it"
- `lance-graph-in-medcare-rs-v1.md` §9.4 — "MedCare is the right target for end-to-end proof"

§11 refines: **all three are correct, but they answer different questions.**

| Question | Right target |
|---|---|
| "Where does the canonical `<repo>_unified_bridge` constructor pattern live?" | smb-office-rs (`smb_unified_bridge` in `unified_bridge_wiring.rs`) |
| "Where does the canonical `<repo>MysqlReconciler` shell live?" | MedCare-rs (`MedcareMysqlReconciler` in `medcare-analytics/src/mysql_reconciler.rs`) |
| "Where does the canonical XRechnung / ZUGFeRD invoice flow live?" | smb-office-rs (consumes `hydrate_zugferd` + `hydrate_zugferd_rules` upstream) |
| "Where does an engineer see the unified-bridge stack working end-to-end with HIPAA + diverse-redundancy?" | MedCare-rs (highest substrate maturity; LanceProbe is the cross-language witness) |
| "Where does the customer / end user see the unified-bridge stack at all?" | **woa-rs** (the only web app; the only consumer where every PR has a URL) |

The cross-consumer harvest is **not** linear (smb → MedCare → woa). It's **fan-in to woa**: SMB ships the bridge template + the XRechnung flow; MedCare ships the reconciler shell + the parity dashboard pattern; woa-rs is the customer-facing integration target that absorbs both + adds the visible web-UI surface.

### 11.5 What stays unchanged

- The §9 hot/cold split framing (lance-graph-ontology hot, sea-orm + MySQL cold) stays exactly as written. The harvest path INSIDE this framing.
- The §10 effort estimate revision (Phase 0-3 closure ~9-10 days with the OGIT-in-sea-orm shortcut) is the foundation §11's PR ladder runs on. §11 doesn't change the per-PR cost — it sequences the PRs by visible-reward.
- D-WLG-1..17 deliverable IDs unchanged. §11 just specifies the recommended SHIPPING ORDER.

### 11.6 One-line summary

> woa-rs is the rewarding integration target because it's a web app — every PR shows up in a browser, not just in a test report. The harvest path is fan-in: SMB ships the `<repo>_unified_bridge` template + the XRechnung/ZUGFeRD invoice flow; MedCare ships the parallelbetrieb reconciler shell + the parity-dashboard pattern; woa-rs consumes both + adds the visible web-UI surface that Stefan can click through. Recommended PR sequence: scaffolding (1-2) → ontology hydration (3-4) → **XRechnung visible reward (5)** → **parity dashboard visible reward (6)** → tenant RLS (7) → Cypher playground (8) → similarity (9).

---

## 12 — Three-way reference asymmetry: C# scrape vs Python+SoC+DTO codegen (2026-05-21, same session)

User architecture clarification: **smb-office-rs and MedCare-rs both scraped their business logic from already-working C# WinForms apps.** They have working C# references to compare against (parity-witness pattern fits naturally — LanceProbe in MedCareV2, the future SMB equivalent). **woa-rs is structurally different: it transcoded from Python (Flask/SQLAlchemy), uses reusable Separation-of-Concerns + DTO mapping, and has 660 routes already harvested from Python source via codegen into Jinja → askama templates** (per `woa-rs/rfcs/v02-006-route-codegen-and-ontology-unification.md`).

This changes how the unified-bridge integration should land in woa-rs.

### 12.1 The three sources, restated

| Consumer | Source | Working reference today | Transcoding approach | Scale at the boundary |
|---|---|---|---|---|
| **smb-office-rs** | C# WinForms ERP at `AdaWorldAPI/SMB-Office/1x1-prg` | Yes — the C# app runs in customer deployments | Hand-mirror of `db_*.cs` BSON schemas + per-route logic; `mongo-schema-warden` + `transcode-auditor` agents gate parity | ~13 MongoDB collections, ~30-entity ACCEPTED list per `foundry-roadmap-unified-smb-medcare-v1.md` |
| **MedCare-rs** | C# WinForms clinic-mgmt at `AdaWorldAPI/MedCare` | Yes — the C# app runs in praxis deployments + `MedCareV2/LanceProbe` is the cross-language parity tool | Hand-port of `pf_*`/`combo_*`/`praxis_*`/`glob_*` MySQL tables (104 total per the `MedCare-rs/.MYSQL/Struktur.sql` reality check); auth path's broken 3DES legacy carried forward via `legacy-tripledes-fallback` feature flag | 7 Healthcare OGIT entities (Patient/Diagnosis/LabValue/Medication/Treatment/Visit/VitalSign); ~30-50 codebook slots |
| **woa-rs** | **Python Flask/SQLAlchemy** at `AdaWorldAPI/WoA` | Yes — Stefan's deployment at `stefan280879/WoA` (Python WoA on MySQL) | **Codegen + SoC + DTO architecture: 660 routes classified into 13 buckets (per RFC v02-006); per-family manifest.yaml + routes.yaml + askama templates harvested from Python source via the route-codegen pipeline** | ~660 routes across ~20 functional families (vorgaenge, kunden, einstellungen, mahnwesen, stundenzettel, einsatz, logbook, …) |

The bucket scale + codegen + SoC pivot is the load-bearing structural difference. SMB has ~13 BSON collections to migrate; MedCare has ~7 Healthcare entities; **woa has ~660 routes already classified, manifest-driven, and codegen-emitting**.

### 12.2 What this means for the unified-bridge integration

The §11 PR ladder framed each integration step as a "handler-level" change ("Stefan clicks X-Rechnung erstellen on a workorder"). That's right at the user-visible layer; it's incomplete at the implementation layer. **The actual change for most of the 660 routes is per-BUCKET, via codegen, not per-handler.**

Concretely, the RFC v02-006 bucket taxonomy:

| Bucket | Routes | % | Unified-bridge integration shape |
|---|---|---:|---|
| `csrf_form_post_engine_call` | 194 | 29.7% | Codegen emits `state.unified_bridge.authorize(owl_id, tenant, op)?` as the first line of every generated handler |
| `ajax_json` | 105 | 16.1% | Same — single codegen edit propagates to all 105 handlers |
| `list_for_tenant` | 80 | 12.3% | Codegen emits the per-tenant predicate via the unified bridge's `g_lock` |
| `form_get_post` | 55 | 8.4% | Same shape |
| `detail_for_tenant` | 43 | 6.6% | Same shape |
| `soft_delete` | 41 | 6.3% | Codegen emits a `Write` op authorize check |
| `sa_admin_view` | 34 | 5.2% | Codegen emits the cross-tenant audit hook via the bridge's `AuditSink` |
| `download_blob` | 31 | 4.7% | Codegen emits the `Read` op authorize check |
| `pdf_render` | 22 | 3.4% | Same shape |
| `template_get` | 22 | 3.4% | No tenant scope; codegen skips authorize when route has no entity scope |
| `signed_link_action` | 15 | 2.3% | Codegen emits timing-safe token compare + audit hook; bridge integration on the action handler |
| `get_redirect_shortcut` | 11 | 1.7% | Same as `template_get` |
| **`other`** | **0** | **0.0%** | 100% bucket coverage; no manual fallback needed |

**A single per-bucket codegen edit propagates the unified-bridge integration to every route in that bucket.** 194 routes get authorize-checked from one codegen template change. Compare manual per-handler integration: 194 PR commits, 194 review surfaces, 194 places to drift. The codegen is the force multiplier.

### 12.3 SoC + DTO means the bridge plugs in at the right seam

Separation of Concerns in woa-rs (per the existing `crates/*` layout: `decimal_money`, `skr_data`, `buchungs_validator`, `woa_pdf`, future `crates/codegen`, future `crates/woa-bridge`) means each crate has a single declared responsibility. The unified bridge plugs in at exactly one seam:

```
Route handler (codegen-emitted, per-bucket)
    │
    ▼
state.unified_bridge.authorize(owl_id, tenant, op)?   ←── ONE seam, integrated via codegen template change
    │
    ▼
Sea-orm Entity::find_by_id(...).await   ←── cold-path read (DTO-mapped to render context)
    │
    ▼
Askama template render   ←── visible to Stefan
```

The bridge is invisible inside the bucket-generic handler. The DTO mapping from sea-orm row → render context is unchanged. The Jinja-harvested askama template is unchanged. The codegen emits the bridge call as scaffolding the bucket template requires; per-handler customization stays in the per-handler override layer (per RFC v02-006 §"Architecture" `overrides/<family>/<endpoint>.rs.tmpl`).

### 12.4 PR ladder refinement — codegen-bucket pivots vs per-handler edits

§11's 9-rung PR ladder is mostly right but lumps codegen-bucket changes into "per-handler" framing. Refined:

| § | §11 framing | §12 refinement |
|---|---|---|
| PR-3 | `woa_unified_bridge(...)` constructor + `/api/v1/health` smoke | Unchanged. Constructor lives in `crates/woa-bridge`, not bucket-level. |
| PR-4 | Boot-time hydration menu + `/api/__ontology` admin route | Unchanged. Boot-level wiring; not per-bucket. |
| PR-5 | **HARVEST 1 (SMB): XRechnung** | Refined: lands as ONE handler in the `pdf_render` bucket (per RFC v02-006 — `wo_to_invoice` shape). The codegen-bucket integration of `pdf_render` adds `state.unified_bridge` to the handler context; the XRechnung-specific logic is the per-handler override. Visible reward unchanged (Stefan downloads invoice). |
| PR-6 | **HARVEST 2 (MedCare): parity dashboard** | Refined: ADMIN-only route in the `sa_admin_view` bucket (cross-tenant; bridge `audit_required` is true). Codegen emits the audit hook + cross-tenant scope; per-handler override implements the WoaMysqlReconciler aggregation read. |
| PR-7 | Tenant RLS unification | **THIS IS THE BIG ONE** — refined: ONE codegen-template change to the `list_for_tenant` (80 routes) + `detail_for_tenant` (43 routes) + `csrf_form_post_engine_call` (194 routes; tenant-scoped) buckets replaces ~317 hand-written `tenant_get_or_404(...)` call sites with `state.unified_bridge.authorize(owl_id, tenant, op)?`. One PR, ~317 routes upgraded. Cross-tenant URL-guessing returns 404 across the whole app. |
| PR-8 | Cypher playground | Unchanged (new admin route, not a bucket integration) |
| PR-9 | Similarity | Unchanged (new admin/sales route, not a bucket integration) |

PR-7 is the moment the unified-bridge integration looks **clean** — 317 routes upgraded by a single codegen-template change, all going through the same authorize call, all auditable from one log stream, all per-tenant scoped via the bridge's `g_lock`.

### 12.5 The DTO/SoC layer is the right MIRROR target for the reconciler

§10's framing — "MedCare-rs MysqlReconciler pattern transfers 1:1 with sea-orm fetchers because sea-orm is ergonomic" — is correct but understates the gain. The actual mirror target is **per-bucket DTO mapping**, not per-handler:

- The `CanonicalCustomerRow` shape (§10.5 D-WLG-15) is the **DTO** for the Customer entity. The reconciler diffs DTOs, not Entity rows.
- Sea-orm `Entity::find_by_id` produces an `ActiveModel` that the bucket-generic handler then maps to a `CustomerDto` (per RFC v02-006 Layer table); the `CanonicalCustomerRow` is the deterministic projection of that DTO.
- A `WoaMysqlReconciler` Round-2 expansion (D-WLG-15 follow-ons for WorkOrder / Position / Mahnung / Tenant / User) is **one DTO definition + one fetcher impl per entity** — and the existing DTO definitions per RFC v02-006 layer table are mostly already authored. The reconciler doesn't author entity shapes; it consumes them.

### 12.6 What this changes in §10.4 effort estimates

§10.4 revised Phase 0-3 closure from ~13 days to ~9-10 days based on the OGIT-in-sea-orm shortcut alone. §12 narrows further:

| Phase | §10 revised | §12 refined |
|---|---|---|
| Phase 0 | 1 day | 1 day (unchanged) |
| Phase 1 | 2 days | 2 days (unchanged) |
| Phase 2 (route-handler integration) | 3 days, "~6-8 handlers initially" | **1-2 days** — the route-handler integration is largely a codegen-template-edit, not per-handler. Cost is reading + writing one bucket-template per bucket touched. ~3 buckets × 2 hours each = ~6 hours of template work + per-bucket smoke test. |
| Phase 3 (lance-cache + Lance projection) | 3-4 days | **3 days** — D-WLG-9 (Lance projection) IS the WoaMysqlReconciler at the DTO layer; reuses the DTO shapes that already exist. |

**Revised Phase 0-3 closure: ~7-8 days** (down from ~13 in original §3, then ~9-10 in §10).

### 12.7 One-line summary

> woa-rs's integration cost is bounded by codegen-template edits, not per-handler edits. The 660 routes across 13 buckets means a single per-bucket codegen change propagates the unified-bridge authorize call to all routes in that bucket (e.g., one edit upgrades 317 tenant-scoped routes in one PR — the §11 PR-7 RLS-unification step). The DTO/SoC layer means the reconciler mirror from MedCare-rs lands at the DTO seam, not the entity seam, reusing the existing per-bucket DTO definitions. Net Phase 0-3 closure: ~7-8 days.
