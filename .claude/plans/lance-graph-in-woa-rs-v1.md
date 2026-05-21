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

