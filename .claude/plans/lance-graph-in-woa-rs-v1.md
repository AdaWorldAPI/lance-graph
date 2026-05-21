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
