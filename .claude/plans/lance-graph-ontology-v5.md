# Plan: lance-graph-ontology v5 — post-merge follow-ons

> **Status:** Drafted (2026-05-07). Picks up where v4 (`claude/create-graph-ontology-crate-gkuJG`)
> left off after `AdaWorldAPI/OGIT#1` merged. Doctrine: brutally honest review +
> super helpful solutions. Per CLAUDE.md "Documentation prose, not lists" —
> body sections are prose, lists are reserved for genuine enumerations.
>
> **Author:** integration-lead (Opus 4.7 1M).
> **Branches in flight:** `lance-graph: claude/create-graph-ontology-crate-gkuJG`
> (last commit `34939e8`); `woa-rs: claude/create-graph-ontology-crate-gkuJG`
> (last commit `c881b1c`); `OGIT: master` (PR #1 merged).
> **Reads:** `.claude/board/AGENT_LOG.md`, `.claude/RECON_ONTOLOGY_CRATE.md`,
> `.claude/DECISION_SPO_ARIGRAPH.md`, `.claude/knowledge/ontology-registry.md`,
> `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md`, `.claude/board/TECH_DEBT.md`
> (TTL-PROBE-5), `.claude/board/INTEGRATION_PLANS.md` (top format),
> `.claude/plans/sql-spo-ontology-bridge-v1.md` (partially superseded),
> `.claude/plans/foundry-roadmap-unified-smb-medcare-v1.md`,
> `crates/lance-graph-ontology/src/lib.rs`, `WoA/RUST_TRANSCODE_PLAN.md`.

---

## §1 Brutally honest stocktake

### What v4 actually shipped

The v4 session shipped a working OGIT-canonical ontology spine and the first end-to-end proof that a TTL fork could hydrate into a tenant binary's runtime. Concretely, the new crate at `/home/user/lance-graph/crates/lance-graph-ontology/` is 12 source files plus 3 integration tests. Of those, `registry.rs` (18 KB) and `ttl_parse.rs` (21.6 KB) are the load-bearing pair: the TTL parser builds `MappingProposal` DTOs via `oxttl`, and the registry indexes them by `(bridge_id, public_name)` and by raw OGIT URI. The `lance_cache.rs` module (15 KB, feature-gated) provides the append-only Lance dictionary persistence path. Three default bridges (`woa_bridge.rs`, `medcare_bridge.rs`, `ogit_bridge.rs`) demonstrate the ~45-LOC scoped-view pattern, each with a scope-lock test in `tests/bridge_scope_lock.rs`. The `cognitive-shader-driver` consumer wiring is a single `Option<Arc<OntologyRegistry>>` field on `BindSpace` plus a setter and getter at `bindspace.rs:198/239/244` — not a usage, just an attachment point with a doc-comment flagging future MUL-trust integration. **FINDING (verified end-to-end).**

The cross-repo deliverables are: `AdaWorldAPI/OGIT#1` (commit `3871d37` on the OGIT fork; 27 TTL files at `NTO/WorkOrder/` covering 15 entities and 12 verbs) which **merged to master** and is now baseline for any future OGIT clone; and the woa-rs binary scaffolding at `/home/user/woa-rs/` which compiles against the registry but cannot run end-to-end locally because of a missing `protoc` build dependency for the lance-encoding transitive build script. The `cargo test -p lance-graph-ontology` suite reports 36 passing tests across 4 files (18 lib + 6 bridge_scope_lock + 2 hydrate_real_ogit + 9 round_trip_ttl + 1 from the workorder hydrate added during integration-lead review) and the OGIT fork is verifiably parsed by `oxttl 0.5.8` end-to-end. **FINDING.**

The board-hygiene was honored: `LATEST_STATE.md`, `PR_ARC_INVENTORY.md`, `EPIPHANIES.md`, `INTEGRATION_PLANS.md`, and `TECH_DEBT.md` all received append-only updates totaling 74 insertions / 0 deletions. `AGENT_LOG.md` was created from scratch and now carries the 12-agent + 3-corrections trace. The Phase-7 invariant (cognitive-shader-driver MUL gate at `driver.rs:271-320` untouched) holds — the v4 session deliberately did not touch the trust-threshold machinery and confined itself to the registry attachment point. **FINDING.**

### What v4 punted on

The `smb-ontology` crate (2,079 LOC of declarative Rust at `smb-office-rs/crates/smb-ontology/`, 13 Steuerberater entities) was kept as the OGIT-skeptical-customer fallback rather than being TTL-migrated. Reason: foot-in-the-door deployments per `WoA/RUST_TRANSCODE_PLAN.md` need a self-contained native ontology that survives without an OGIT clone on disk, and the migration would have ballooned the v4 scope past the merge window. **CONJECTURE / status: deliberate deferral, see D-ONTO-V5-4 for the brutal-honest review of whether to actually convert.**

The `callcenter-bridge` was deferred because callcenter has separate auth (JWT middleware, RLS rewriter per actor context — see `foundry-roadmap-unified-smb-medcare-v1.md` §2 LF-3/DM-7) and per-customer scoping concerns. Adding a fourth default bridge that also has to coordinate with `Subject` (currently SUBJECT-DTO-1 in the entropy ledger; Stage 0 / Aspirational) would have coupled two architectural questions; v4 picked the right axis to defer. **FINDING.**

The MySQL and MSSQL `SchemaSource` impls were deferred. The trait shape exists at `src/schema_source.rs` but the only producer today is the OGIT TTL directory walker. Reason: getting one end-to-end TTL path working before a second producer was correct sequencing. **FINDING.**

The customer admin form was deferred because it is React/HTML, not Rust, and the Rust boundary (a `MappingProposal` emitter) is the entire ontology-side concern; the UX layer is woa-rs's territory. **FINDING.**

The ontology-aware MUL trust thresholds (Compliance → Plateau-only; Healthcare → stricter calibration) were deferred to keep Phase-7 to a read-only attachment. The `BindSpace.ontology` slot is reachable but no MUL gate logic reads from it yet. **FINDING.** Brier-history, damage-budget, and sandbox-availability MUL publishers stayed at `SituationInput::default()` for the same reason — those are publisher-side concerns owned by the cognitive-shader-driver crate, not the ontology crate. **FINDING.**

`TTL-PROBE-5` (the dcterms:source provenance drop at parse) was logged as a regression test instead of fixed because the fix path (extend `parse_into_proposals` to look for `<http://purl.org/dc/terms/source>` triples) is well-scoped, the test locks current behavior so future fixes are detectable, and shipping the fix would have required rerunning the OGIT fork validation. **FINDING / proper deferral.**

`SPO-1` (the `arigraph::SpoBridge::promote_to_spo` writer bridge) is unblocked but unowned. The v4 session chose Option B (federated, two-layer cache) per `DECISION_SPO_ARIGRAPH.md` which is the correct architectural reading; it does not close the entropy-ledger row. **FINDING.**

`PARSER-1` (the planner cypher_parse stub) was correctly framed by the main-thread correction at `AGENT_LOG.md` line 187+: cold-path vs hot-path is by design, "consolidation later, not now," not "the planner has a bug to fix." woa-rs uses the cold path (`lance_graph::parser::parse_cypher_query`) for diagnostic clarity. **FINDING.**

## §2 Open ledger rows that v4 did NOT close

`SPO-1` (Stage 3, ×2 distinct purposes / not duplicates by design) is the biggest row v4 had explicit influence over. The L1/L2 cache pair framing (warm string-keyed `triplet_graph` + cold fingerprint-keyed `spo::store`) is now binding doctrine for the ontology crate per `DECISION_SPO_ARIGRAPH.md`. The smallest-cost path to actually closing the row is to ship the one-way writer: `arigraph::SpoBridge::promote_to_spo(&TripletGraph, gate, &mut SpoStore)` as a ~150 LOC addition to `crates/lance-graph/src/graph/arigraph/`, gated by a `gate: PromoteGate` parameter (probably `truth_floor: TruthValue` + `min_episodes: u32`). The ontology crate does not need to ship this — it is `lance-graph` core work — but the v5 plan can name it as D-ONTO-V5-2 since the scope-lock + bridge-trait pattern in `lance-graph-ontology` is the natural template.

`PARSER-1` (Stage 3 lance-graph::parser real + Stage 1 planner stubs) is correctly NOT touched by ontology work. The planner stubs are intentionally lighter than the cold-path parser; consolidation is a separate consolidation track. The ontology crate has zero leverage here and v5 should not pretend otherwise. **No D-id; reference for future readers only.**

`TTL-PROBE-5` (TECH_DEBT row at line 1487, dcterms:source dropped at parse) is the one row the ontology crate fully owns. The fix is local: `crates/lance-graph-ontology/src/ttl_parse.rs` `parse_into_proposals`, plus a flip of the regression test in `tests/round_trip_ttl.rs` from "asserts dropped" to "asserts preserved." This is D-ONTO-V5-1.

`MUL-ASSESS-1` (Stage 2 ×4) is partially leverageable from v5. The ontology registry is the natural lookup surface for namespace-keyed trust thresholds (Compliance → Plateau-only; Healthcare → stricter calibration). The fix is to grow `OntologyRegistry` with `mul_threshold(namespace: NamespaceId) -> MulThreshold` and have `cognitive-shader-driver` read it before constructing `SituationInput`. This is D-ONTO-V5-9. **The row is not closed by D-ONTO-V5-9** (the row is about consolidating the four `MulAssessment` copies into one canonical), but the registry does become a natural canonical source for the per-namespace threshold table.

`TRUST-1` (Stage 2 ×3 incompatible variant sets) and `FLOW-1` (Stage 2 ×3) and `COMPASS-1` (Stage 2 ×3 incompatible) are all consolidation rows in the Thinking cluster. The ontology crate has **zero leverage** on the consolidation step (collapsing the duplicate enums); the registry might at most carry a per-namespace thinking-style preference table, but that is not what the rows ask for. v5 should explicitly NOT pretend to address these.

`CONTRACT-INV-1` (Stage n/a, board hygiene) is partially closed by ontology work — `OntologyRegistry`, `WoaBridge`, `MedcareBridge`, `OgitBridge`, `HydrationReport`, `OgitUri`, `NamespaceBridge` have been added to `LATEST_STATE.md` Contract Inventory per the v4 governance pass. Continued vigilance during v5 deliveries (each new contract type must update the inventory in the same commit). **No D-id; ongoing discipline.**

## §3 Deliverables — D-IDs (rank-ordered by leverage / cost)

D-ids strictly ordered by the leverage-over-cost ratio. The first three are next-3 (ship within 1-2 sessions); D4-6 are 6-months-out; D7-15 are out-past-the-quarter.

### Next-3 (ship now, leverage > cost by ≥ 3×)

**D-ONTO-V5-1 — dcterms:source provenance.** Closes TTL-PROBE-5 properly. Scope: extend `parse_into_proposals` in `crates/lance-graph-ontology/src/ttl_parse.rs` to look for `<http://purl.org/dc/terms/source>` triples per subject and prefer that IRI over the local file path when present. Files touched: `src/ttl_parse.rs` (~80 LOC), `tests/round_trip_ttl.rs` (flip the `dcterms_source_is_currently_dropped` probe assertion + rename), `src/proposal.rs` (`MappingProposal::source_uri` already exists — no API change). Exit criteria: the renamed probe asserts that the dcterms IRI from a TTL like `WoA/models.py:Customer` is preserved verbatim through to `MappingRow::source_uri` for all 27 WorkOrder TTLs in the merged OGIT fork; closes TTL-PROBE-5 in `TECH_DEBT.md` with a "resolved" annotation. Dependencies: none. Risk: low (local, well-tested).

**D-ONTO-V5-9 — Ontology-aware MUL trust thresholds.** The deferred Phase-7 work; the highest-leverage cognitive integration v5 can ship without touching MUL math. Scope: grow `OntologyRegistry` with a small `mul_threshold(namespace: NamespaceId) -> Option<MulThresholdProfile>` returning a per-namespace override (e.g., `Healthcare → MulThresholdProfile::stricter()`, `Compliance → MulThresholdProfile::plateau_only()`); have `cognitive-shader-driver::driver.rs:271-320` consult `bindspace.ontology()` and apply the namespace-keyed override before computing `SituationInput`. Files touched: `crates/lance-graph-ontology/src/registry.rs` (~60 LOC for the lookup table + builder), `crates/cognitive-shader-driver/src/driver.rs:271-320` (~40 LOC to wire the override path; **no MUL math change**, just SituationInput field tightening), one new integration test at `crates/cognitive-shader-driver/tests/ontology_aware_mul_threshold.rs` (~50 LOC). Exit criteria: a Healthcare-namespace cycle goes to HOLD on a SituationInput that a Compliance-namespace cycle would CONTINUE on, with all other state held constant. Dependencies: D-ONTO-V5-1 not strict (parallel-shippable). Risk: medium — the `MulThresholdProfile` shape is new to `lance-graph-contract` and needs a Marking-style enum, not just struct fields, to preserve canonical-source semantics.

**D-ONTO-V5-2 — SPO bridge fn (`promote_to_spo`).** Closes SPO-1. Scope: implement the one-way writer bridge `arigraph::SpoBridge::promote_to_spo(&TripletGraph, gate: PromoteGate, &mut SpoStore)` that promotes warm string-keyed entries into cold fingerprint-keyed storage. Architectural call: the bridge **lives in `lance-graph::graph::arigraph::spo_bridge.rs`, not in a new crate.** Brutal honest take — a separate `crates/spo-bridge` would re-create the inter-store hop (SPO-1's whole point is the L1/L2 cache pair lives in `lance-graph::graph`); pulling it out of that crate breaks the encapsulation that `DECISION_SPO_ARIGRAPH.md` ratifies. The bridge is internal lance-graph plumbing, not a public consumer surface. Files touched: `crates/lance-graph/src/graph/arigraph/spo_bridge.rs` (NEW, ~150 LOC), `crates/lance-graph/src/graph/arigraph/mod.rs` (add `pub mod spo_bridge`), `crates/lance-graph/tests/arigraph_promote_to_spo.rs` (NEW, ~80 LOC). Exit criteria: 50 string-keyed `TripletGraph` entries with `confidence > gate.truth_floor` and `episode_count > gate.min_episodes` round-trip to fingerprint-keyed SPO rows; the entropy-ledger row 70+245 closes with a dated entry. Dependencies: none. Risk: medium — `gate: PromoteGate` shape is new and needs to align with `contract::collapse_gate::GateDecision` semantics without colliding with `mul::GateDecision` (GATE-1 in the ledger).

### Six-months-out (D4-D6, leverage > cost by ≥ 1.5×)

**D-ONTO-V5-3 — Healthcare namespace transcode.** Mirrors the WoA pattern with the same 6-agent ensemble (container-architect / family-codec-smith / bus-compiler for entities; ripple-architect for verbs; truth-architect for hydration probes; certification-officer for the OGIT fork PR). Scope: identify Medcare's domain entities from `MedCare-rs/crates/medcare-core/` (or upstream Python source if available — the workspace currently has only the Rust scaffolding, no `models.py` equivalent), transcode to OGIT-shaped TTL under `OGIT/NTO/Healthcare/`, open the OGIT fork PR. Estimate: 12-15 entities + 8-10 verbs (smaller than WoA because medcare's clinical taxonomy is narrower); ~20 TTL files; ~3 sessions of agent work. Files touched: `OGIT/NTO/Healthcare/{entities,verbs}/*.ttl` (NEW), `crates/lance-graph-ontology/src/bridges/medcare_bridge.rs` (already exists — verify scope-lock against the new namespace), `crates/lance-graph-ontology/tests/hydrate_real_ogit.rs` (extend with `hydrate_healthcare_namespace_from_real_ogit`). Exit criteria: 12-15 Healthcare entity TTLs parse via `pyoxigraph` validation; `MedcareBridge` resolves `Patient`, `Diagnose`, `Laborwert`, `Medikament` URIs; OGIT-fork PR opens. Dependencies: D-ONTO-V5-1 (cleaner provenance for Healthcare TTLs). Risk: low (mechanical mirror of the WoA pattern).

**D-ONTO-V5-6 — SchemaSource for MySQL.** Concrete impl of the trait shape. Scope: pick one tenant DB schema as the proving ground (the WoA `models.py` MySQL schema is the natural choice — same shape as the TTLs, cross-validates the hydration). Implement `MySqlSchemaSource` via `sqlx::MySql` against an information_schema query, map `tables / columns / FK constraints` to `MappingProposal { Schemas, links, rows }`. Files touched: `crates/lance-graph-ontology/src/schema_sources/mysql.rs` (NEW, ~250 LOC), `crates/lance-graph-ontology/Cargo.toml` (add `sqlx` behind feature `mysql-source`), `crates/lance-graph-ontology/tests/mysql_to_proposals.rs` (NEW, ~80 LOC, tempdb-fed). Exit criteria: the MySQL-derived proposals overlay the TTL-derived proposals for the same WorkOrder schema with zero collisions; `HydrationReport` warns on any drift. Dependencies: D-ONTO-V5-1. Risk: medium — `sqlx` is a new transitive dep with build-time gravity (similar to lance-encoding's protoc issue that bit woa-rs).

**D-ONTO-V5-13 — Hydration parallelism.** Scope: profile the TTL hydrate path against the full OGIT fork (66 namespaces, ~3000+ TTL files including upstream + AdaWorldAPI extensions). If wallclock > 5s on a cold cache, parallelize via `rayon::par_iter` over namespace directories with a final merge-into-registry step. Files touched: `crates/lance-graph-ontology/src/registry.rs` `hydrate_once_sync` (~60 LOC delta), `crates/lance-graph-ontology/Cargo.toml` (add `rayon` behind feature `parallel-hydrate`), `crates/lance-graph-ontology/benches/hydrate_full_fork.rs` (NEW, ~40 LOC). Exit criteria: hydrate of full fork wallclock measured + reported; if > 5s, parallel impl ships and the bench passes < 2s; if ≤ 5s, the deliverable closes as "no work needed" with the bench shipping for future regression detection. Dependencies: none. Risk: low (additive feature flag).

### Out-past-the-quarter (D7-D15, defer or partial leverage)

**D-ONTO-V5-4 — smb-ontology TTL migration. Brutal honest take: do NOT convert.** The 2,079-LOC declarative Rust at `smb-office-rs/crates/smb-ontology/` is the foot-in-the-door deployment per `WoA/RUST_TRANSCODE_PLAN.md`'s "backend = local" mode. Customers running smb on a single machine without an OGIT clone need a self-contained native ontology. Converting it to TTL forces an OGIT-fork dependency on every smb deployment, which inverts the foot-in-the-door promise and adds a 600-MB clone to the install footprint. The right move is to **keep smb-ontology as native Rust** AND to add an OGIT-shaped *export* path so an smb deployment can publish its ontology to a fork on demand. Files touched: `smb-office-rs/crates/smb-ontology/src/export_ogit.rs` (NEW, ~120 LOC translating `Schema`/`LinkSpec` into TTL strings using the same shape as `OGIT/NTO/Network/entities/IPAddress.ttl`). Exit criteria: `smb_ontology::export_ogit_ttl()` produces 13 TTL files that parse via `oxttl` and that the `OntologyRegistry` hydrates into a working `SmbBridge`. Dependencies: D-ONTO-V5-1 (consistent provenance). Risk: low. **The "convert smb-ontology to TTL" framing in the v4 deferral list is wrong; do not adopt it.**

**D-ONTO-V5-5 — q2 namespace transcode.** Mirrors WoA + Healthcare. Scope: identify q2's foundry-shape entities (Quarto / Neo4j / Gotham equivalents — see `q2-foundry-integration-v1.md` Q2-1.1..Q2-1.7 for the shape) and transcode to TTL under `OGIT/NTO/Q2/`. Estimate: ~10-12 entities (q2 is more user-interface than data-plane). Files touched: `OGIT/NTO/Q2/{entities,verbs}/*.ttl` (NEW), `crates/lance-graph-ontology/src/bridges/q2_bridge.rs` (NEW, ~45 LOC mirroring `medcare_bridge.rs`), `tests/bridge_scope_lock.rs` (extend with `q2_bridge_scope_lock`). Exit criteria: q2 binary holds an `Arc<OntologyRegistry>` and resolves `Workshop`, `Vertex`, `Doctemplate` via the `Q2Bridge`. Dependencies: D-ONTO-V5-3 (use the Healthcare transcode as a fresher template than WoA). Risk: low (mechanical).

**D-ONTO-V5-7 — SchemaSource for MSSQL.** LOC delta vs MySQL impl: ~30 LOC (different driver string, slightly different information_schema column names; tibero / MSSQL share enough shape with MySQL that the impl is ~80% identical). Files touched: `crates/lance-graph-ontology/src/schema_sources/mssql.rs` (NEW), `Cargo.toml` add `tiberius` behind feature `mssql-source`. Exit criteria: same as D-ONTO-V5-6 for an MSSQL test schema. Dependencies: D-ONTO-V5-6 (do MySQL first; MSSQL is the second tenant). Risk: medium (tiberius is heavier than sqlx-mysql).

**D-ONTO-V5-8 — Customer admin form.** This is React/HTML, not Rust. The Rust boundary is a `MappingProposal` POST endpoint exposed on woa-rs's existing axum server. Brutal honest take: **woa-rs is the first tenant binary, not the platform — the form belongs in woa-rs's surface, not in `lance-graph-ontology`.** Files touched: `woa-rs/src/admin/ontology_form.rs` (NEW, ~150 LOC axum handler accepting a JSON `MappingProposal`), `woa-rs/templates/admin/ontology.askama` (NEW, the actual form), `crates/lance-graph-ontology/src/proposal.rs` (add `serde::{Serialize, Deserialize}` derive behind feature `serde-proposals`, ~10 LOC). Exit criteria: a customer admin can paint a one-entity ontology extension via the form and see it appear in `OntologyRegistry` after a registry rebuild. Dependencies: D-ONTO-V5-1, D-ONTO-V5-9 (admin should be aware of namespace-keyed thresholds). Risk: medium (UX work that doesn't fit the agent ensemble cleanly).

**D-ONTO-V5-10 — callcenter-bridge.** Architectural question: callcenter has separate auth (JWT middleware + RLS rewriter per `foundry-roadmap-unified-smb-medcare-v1.md` §2 LF-3/DM-7) and per-customer scoping. The RBAC × ontology cross is the open architectural question — does a `CallcenterBridge` carry a `Subject` parameter (from SUBJECT-DTO-1; currently Aspirational) on every resolution, or does the registry itself grow a subject-aware lookup? Cite POLICY-1 entropy-ledger row (entropy 4): the RBAC↔BBB bridge is missing — `impl MembraneGate for Arc<rbac::Policy>`. Brutal honest take: **defer until SUBJECT-DTO-1 lands.** The bridge can be drafted but cannot ship until the auth surface is stable. Files touched: `crates/lance-graph-ontology/src/bridges/callcenter_bridge.rs` (DRAFT only, no commit until SUBJECT-DTO-1). Risk: high (depends on entropy-ledger row that is itself Aspirational).

**D-ONTO-V5-11 — woa-rs binary minimum.** The 80/20 cut from `WoA/RUST_TRANSCODE_PLAN.md` (58 KB plan, ~215 dev-hours total scope). Brutal honest take: a useful Rust HTTP service this calendar quarter is ~50 hours of work; pick the chunks that exercise the ontology spine + at least one CRUD path, ship the rest as Python. Concretely: WT-2X chunks (entity round-trip via `EntityStore`) + WT-3X chunk for Customer + WT-4X chunk for the gRPC pair scaffolding (h2c only, defer the 3-mode TLS to next quarter). Files touched: `woa-rs/src/{routes,handlers,models}/customer.rs` (NEW, ~400 LOC), `woa-rs/src/grpc/` (NEW, ~250 LOC). Exit criteria: a `customer-woa-bin` Rust binary serves Customer CRUD over gRPC + axum and round-trips through `OntologyRegistry`. Dependencies: D-ONTO-V5-3 (parallel — Healthcare gives the second tenant template). Risk: high (cross-repo coordination tax per `RUST_TRANSCODE_PLAN.md`).

**D-ONTO-V5-12 — cognitive-shader-driver MUL publishers.** The `SituationInput::default()` fields that v4 left as defaults: `calibration_accuracy`, `allostatic_load`, `max_acceptable_damage`, `sandbox_available`, plus a Brier-history publisher and a damage-budget publisher. These are publisher-side concerns owned by the cognitive-shader-driver crate, NOT the ontology crate. Per-publisher tickets: D-ONTO-V5-12a (Brier history), D-ONTO-V5-12b (damage budget), D-ONTO-V5-12c (sandbox availability). Each is ~80 LOC + 1 integration test in `crates/cognitive-shader-driver/`. Files touched: `crates/cognitive-shader-driver/src/publishers/{brier_history,damage_budget,sandbox_availability}.rs` (NEW). Exit criteria: each publisher emits a `SituationInput` field that varies measurably across cycles; gate decisions reflect the variance. Dependencies: D-ONTO-V5-9 (the namespace-keyed override needs the publishers' outputs to vary). Risk: low individually, sequential composition.

**D-ONTO-V5-14 — Lance dictionary persistence under load.** The append-only contract is correct but unmeasured. Scope: 100K-row hydrate + resolve probe via the Lance-cache feature, measure wallclock + memory + dataset growth across 10 hydrate-resolve-restart cycles. Files touched: `crates/lance-graph-ontology/benches/lance_cache_load.rs` (NEW, ~120 LOC). Exit criteria: 100K hydrates complete < 10s; resolve latency < 100us p50; dataset growth linear in row count. Dependencies: D-ONTO-V5-13 (parallel hydrate makes the benchmark relevant at scale). Risk: low (additive bench).

**D-ONTO-V5-15 — In-memory → Lance-backed registry cutover.** Currently `OntologyRegistry::new_in_memory()` is used everywhere; the Lance variant exists at `src/lance_cache.rs` but has no consumer. Scope: identify the cutover ticket — what's the surface between "in-memory only" and "Lance-backed with in-memory hot cache"? Probably a new constructor `OntologyRegistry::with_lance_cache(path: &Path) -> Result<Self>` that loads existing rows from Lance into the in-memory dictionary at startup, then writes new proposals to both. Files touched: `crates/lance-graph-ontology/src/registry.rs` (~80 LOC to wire the dual-write), `crates/lance-graph-ontology/tests/lance_backed_round_trip.rs` (NEW, ~60 LOC). Exit criteria: a registry constructed with Lance cache survives process restart and serves queries against the warm cache. Dependencies: D-ONTO-V5-14 (load probe must pass before this becomes default). Risk: medium (concurrency story for parallel writers — out of scope for v5 single-writer pattern).

## §4 Test plan

Each deliverable lands with concrete `cargo test` invocations that gate the merge.

D-ONTO-V5-1: `cargo test -p lance-graph-ontology --no-default-features --test round_trip_ttl dcterms_source_is_preserved` (renamed from `_is_currently_dropped`); plus `OGIT_FORK_PATH=/home/user/OGIT cargo test -p lance-graph-ontology --no-default-features --test hydrate_real_ogit` to verify all 27 WorkOrder TTLs.

D-ONTO-V5-9: `cargo test -p cognitive-shader-driver --test ontology_aware_mul_threshold healthcare_holds_compliance_continues`.

D-ONTO-V5-2: `cargo test -p lance-graph --test arigraph_promote_to_spo round_trip_50_warm_to_cold`.

D-ONTO-V5-3: `OGIT_FORK_PATH=/home/user/OGIT cargo test -p lance-graph-ontology --no-default-features --test hydrate_real_ogit hydrate_healthcare_namespace_from_real_ogit`.

D-ONTO-V5-4: `cargo test -p smb-ontology --test export_ogit ttl_roundtrips_through_oxttl`.

D-ONTO-V5-6 / D-ONTO-V5-7: `cargo test -p lance-graph-ontology --features mysql-source --test mysql_to_proposals workorder_schema_round_trip` (and analogous for MSSQL).

D-ONTO-V5-13: `cargo bench -p lance-graph-ontology --bench hydrate_full_fork --features parallel-hydrate` plus a wallclock assertion < 2s in CI.

D-ONTO-V5-14 / D-ONTO-V5-15: `cargo bench -p lance-graph-ontology --bench lance_cache_load --features lance-cache` plus a `cargo test --test lance_backed_round_trip restart_survives`.

The full v5 regression suite is `cargo test -p lance-graph-ontology --all-features` plus the cross-crate integration tests above. Every D milestone adds at least one test; no deliverable merges with a green-on-skip pattern.

## §5 Risk + rollback

The Lance schema migration risk is concentrated in D-ONTO-V5-14 and D-ONTO-V5-15. The current `ontology_dictionary` schema is append-only by design, which makes additive column changes safe (Lance's evolution rules tolerate adding nullable columns). But a non-additive change — say, switching `source_uri: Utf8` to a struct with `{source_uri: Utf8, dcterms_source: Utf8}` — would require a migration script and a backwards-compat read path. The mitigation is: D-ONTO-V5-1 (the dcterms:source fix) ships **into the existing `source_uri` column** as the dcterms IRI when present, falling back to file path when absent. No schema change, no migration. If a future deliverable needs to distinguish provenance from file path, it gets its own additive column. If a non-additive change is ever genuinely needed, the rollback is to read the old dataset, write a new dataset with the new schema, swap pointers — Lance's commit-log makes this cheap, but it is a deliberate operation not an accidental one.

The tenant binary risk is concentrated in D-ONTO-V5-3, D-ONTO-V5-5, and D-ONTO-V5-11. Each tenant binary depends on a registry handle that **can fail to hydrate** — the OGIT fork might not be checked out, the TTL files might be malformed (HydrationFailure), the Lance dataset might be corrupt. The current failure mode is hard panic: `OntologyRegistry::hydrate_once_sync` returns `Err`, the binary's startup fails, the operator gets a stack trace. The rollback path to native ontology is: each tenant binary should accept a `--fallback-to-native` flag that, on hydration failure, constructs an in-memory registry seeded by a hand-rolled `Schema` set (mirroring the smb-ontology pattern). For woa-rs this means committing a `woa-rs/src/native_fallback.rs` that constructs a 15-entity in-memory registry from `models.py`-equivalent Rust data; for medcare-rs and q2 the equivalent. The fallback is intentionally incomplete (no MUL trust thresholds, no admin form, no admin-extended entities) — it is a degraded-mode kept so that a single broken TTL doesn't take down a tenant's entire HTTP surface. Risk class: medium for woa-rs (manual mirror work), low for the others (smaller entity counts).

## §6 Branch + PR strategy

This is multi-repo work spanning `lance-graph`, `woa-rs`, `MedCare-rs` (post-D-ONTO-V5-3), `q2` (post-D-ONTO-V5-5), `OGIT` (every TTL transcode), and `smb-office-rs` (D-ONTO-V5-4). The default branch naming pattern is per-D `claude/onto-v5-<D-id>` (e.g., `claude/onto-v5-1-dcterms-source`), with the exception that closely-coupled deliverables in the same repo can ride one branch (e.g., D-ONTO-V5-1 + D-ONTO-V5-2 are both `lance-graph` repo work but they touch different crates and should land as two separate PRs even on a shared branch).

The repo split is: `lance-graph` gets a PR per Rust deliverable (so D-ONTO-V5-1, D-ONTO-V5-2, D-ONTO-V5-9, D-ONTO-V5-13, D-ONTO-V5-14, D-ONTO-V5-15 are six separate PRs). `OGIT` (the AdaWorldAPI fork) gets a PR per namespace transcode (D-ONTO-V5-3 = one PR for `NTO/Healthcare/`, D-ONTO-V5-5 = one PR for `NTO/Q2/`). `woa-rs`, `MedCare-rs`, `q2` stay on branches until the corresponding ontology deliverable is in master, then each opens its own PR consuming the shipped registry. `smb-office-rs` gets one PR for D-ONTO-V5-4. The OGIT-fork PR cadence is: open the PR when the TTL files validate via `pyoxigraph`, merge after one round of integration-lead review, and the upstream `almatoai/OGIT` repo is intentionally never PR'd — AdaWorldAPI runs an extension fork by design (per `RECON_ONTOLOGY_CRATE.md` §1.9: 66 upstream namespaces unchanged, AdaWorldAPI extensions live under additive directories like `WorkOrder/`, `Healthcare/`, `Q2/`).

Each PR commit message ends with `https://claude.ai/code/<session-id>` per the workspace's git policy. No force-pushes to main/master. No `--no-verify`. Branch lifetime: a per-D branch lives until the PR merges, then is deleted from origin. The session-class agent's parallel branch coordination pattern (one main thread + N specialists each writing to disjoint files + a final integration-lead review) is the recommended ensemble shape for D-ONTO-V5-3 and D-ONTO-V5-5; D-ONTO-V5-1 / D-ONTO-V5-2 / D-ONTO-V5-9 are single-agent grindwork.

## §7 Concrete next-session prompt

> You are a session-class agent on Opus 4.7 (1M). The lance-graph-ontology v5 plan ships
> three deliverables in priority order: D-ONTO-V5-1 (dcterms:source provenance fix,
> closes TTL-PROBE-5; ~80 LOC + 1 test in `crates/lance-graph-ontology/src/ttl_parse.rs`
> + `tests/round_trip_ttl.rs`), D-ONTO-V5-9 (ontology-aware MUL trust thresholds; grow
> `OntologyRegistry` with `mul_threshold(NamespaceId) -> Option<MulThresholdProfile>`,
> wire `cognitive-shader-driver/src/driver.rs:271-320` to consult the namespace-keyed
> override; ~150 LOC + integration test), and D-ONTO-V5-2 (the `arigraph::SpoBridge::
> promote_to_spo` writer, closes SPO-1; ~150 LOC at `crates/lance-graph/src/graph/
> arigraph/spo_bridge.rs`). Mandatory reads: `.claude/plans/lance-graph-ontology-v5.md`
> + `.claude/board/AGENT_LOG.md` + `.claude/DECISION_SPO_ARIGRAPH.md`. Branch:
> `claude/onto-v5-1-dcterms-source` (or per-D as work splits). Do NOT touch the MUL
> gate math at `cognitive-shader-driver/driver.rs:271-320` — only add the override
> path. Do NOT modify the OGIT fork (TTL transcodes are D-ONTO-V5-3 / -5, separate
> sessions). Append AGENT_LOG.md after each deliverable per Layer-2 A2A discipline.

## §8 Append-only commit — INTEGRATION_PLANS.md index entry

Return as text for main-thread to apply per the v4 governance pattern (do not edit `INTEGRATION_PLANS.md` from this session). Prepend the following block at the top of the file, between the file's preamble and the `splat-osint-ingestion-v1` entry:

```markdown
## lance-graph-ontology-v5 — post-merge follow-ons (authored 2026-05-07)

- **Plan:** `.claude/plans/lance-graph-ontology-v5.md`
- **Author + date:** integration-lead (Opus 4.7 1M), 2026-05-07
- **Status:** Active
- **Scope:** Picks up where v4 (`claude/create-graph-ontology-crate-gkuJG`, OGIT#1 merged) left off. 15 deliverables ranked by leverage / cost: D-ONTO-V5-1 (dcterms:source provenance, closes TTL-PROBE-5), D-ONTO-V5-2 (`arigraph::SpoBridge::promote_to_spo`, closes SPO-1), D-ONTO-V5-3 (Healthcare TTL transcode), D-ONTO-V5-4 (smb-ontology export-only, NOT migration — brutal-honest reversal), D-ONTO-V5-5 (q2 TTL transcode), D-ONTO-V5-6/7 (MySQL/MSSQL `SchemaSource` impls), D-ONTO-V5-8 (customer admin form, owned by woa-rs surface), D-ONTO-V5-9 (ontology-aware MUL trust thresholds — registry as namespace-keyed lookup), D-ONTO-V5-10 (callcenter-bridge, deferred until SUBJECT-DTO-1 lands), D-ONTO-V5-11 (woa-rs 80/20 binary cut), D-ONTO-V5-12 (MUL publishers — Brier/damage/sandbox), D-ONTO-V5-13 (hydration parallelism), D-ONTO-V5-14 (Lance dictionary load probe), D-ONTO-V5-15 (in-memory → Lance-backed cutover).
- **Originating context:** v4 OGIT#1 merge (15 entities + 12 verbs in `NTO/WorkOrder/`, master); 36 ontology tests pass; cognitive-shader-driver wired (read-only registry attachment).
- **Resolves ledger rows:** TTL-PROBE-5 (D-ONTO-V5-1), SPO-1 (D-ONTO-V5-2 70+245). Partial leverage on MUL-ASSESS-1 (registry as namespace-keyed threshold table). No leverage on TRUST-1 / FLOW-1 / COMPASS-1 / PARSER-1 (out of scope; the ontology crate has no influence on enum consolidation or the cypher cold/hot split).
- **Branch:** `claude/onto-v5-<D-id>` per deliverable; OGIT-fork PRs per namespace transcode.
- **Confidence (2026-05-07):** Pre-execution. Plan reviews v4's outputs as FINDING-grade and v5's deferrals as honestly-deferred (not punted). Next-3 ranked: D-ONTO-V5-1, D-ONTO-V5-9, D-ONTO-V5-2.
- **Cross-ref:** `.claude/RECON_ONTOLOGY_CRATE.md`, `.claude/DECISION_SPO_ARIGRAPH.md`, `.claude/knowledge/ontology-registry.md`, `sql-spo-ontology-bridge-v1.md` (partially superseded), `foundry-roadmap-unified-smb-medcare-v1.md` (adjacent).
```

---

**End of plan.**
