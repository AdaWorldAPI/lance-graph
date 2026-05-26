# lance-graph in MedCare-rs — v1

> **Author:** main thread (Opus 4.7), session 2026-05-21 (branch `claude/activate-lance-graph-att-k2pHI`)
> **Status:** Draft
> **Scope:** Close the four open items in `medcare-bridge/src/lib.rs` ("What is NOT wired yet") and complete the lance-phase2 wiring. Current state: `medcare-bridge` crate exists with `MedcareRegistry::hydrate()` + `MedcareBridge` re-exports under `--features ontology`; the canonical-branch (`main`) build is broken at `medcare-analytics::ontology_dto::MedcareOntology::default()`; the lean-fallback branch (`claude/scaffold-medcare-rs-rZD5A`) compiles without lance-graph deps. The LanceProbe C# parity tool (`MedCareV2/MedCare_2.0/LanceProbe/`) is scaffolded (M1 done, M2-M6 pending five medcare-rs endpoints).
> **Path:** `.claude/plans/lance-graph-in-medcare-rs-v1.md`
> **Confidence:** Working (architecture — most upstream substrate ships; medcare-bridge crate exists). Partial (build broken at one call site; RLS coverage incomplete for 3 newly-OGIT-surfaced entities → fail-OPEN risk). Conjecture (3DES column inventory beyond `u_pwd` — see §18.5 of super-domain plan).

---

## 1 — Why this exists

MedCare-rs is **mid-migration toward the unified-bridge end-state** — further along than woa-rs but with a known broken build blocking forward progress. Per `MedCare-rs/crates/medcare-bridge/src/lib.rs` "## What is NOT wired yet":

1. **Build broken at `medcare-analytics::ontology_dto.rs:85`.** `MedcareOntology::default()` calls the now-broken no-arg `upstream_medcare_ontology()` form. The lance-phase2 build does not compile. Headline blocker.
2. **`ALL_SCHEMAS` ↔ `OntologyRegistry::enumerate("Healthcare")` migration incomplete.** `medcare-server::state::rls_registry` still iterates the hardcoded 4-entity list (Patient / Diagnosis / LabResult / Prescription). OGIT surfaces 3 NEW Healthcare entities (Treatment / Visit / VitalSign) which therefore have no RLS policy → **fail-OPEN bypass risk**. Safety-critical.
3. **`MulThresholdProfile::MEDICAL` not consumed at the gate site** (per D-ONTO-V5-9 in lance-graph#355).
4. **`ontology_context_id`-keyed RLS extension missing** (third axis after `(table, praxis_id)`) per §73 SGB V Überweisung shape.

Plus the LanceProbe-side gaps from `super-domain-rbac-tenancy-v1.md` §18.7:

5. **`POST /api/__parity/csharp` ingest endpoint absent** (blocks LanceProbe M5).
6. **`GET /api/__parity` canonical dashboard absent** (depends on #5).
7. **`_dto_contracts.md` not published** (blocks LanceProbe M2).
8. **`legacy-tripledes-fallback` feature flag DRAFT only** (blocks LanceProbe M5a).
9. **`/api/__parity/telemetry` endpoint absent** (blocks LanceProbe M6).

This plan owns items 1–4 directly and cross-references items 5–9 to `super-domain-rbac-tenancy-v1.md` Tier H (D-SDR-35..39) where they already have spec coverage.


## 2 — Phasing

### Phase 1 — Unblock the lance-phase2 build (1 day) — CRITICAL PATH

- **D-LGMC-1** — Fix `medcare-analytics::ontology_dto.rs:85`. The broken call is `MedcareOntology::default() → upstream_medcare_ontology()` (no-arg form, removed upstream). Replace with a registry-driven constructor: `MedcareOntology::from_registry(&medcare_bridge::MedcareRegistry)`. Default impl now requires explicit registry construction — `MedcareOntology` becomes `Option<MedcareOntology>` at the State level, populated only when the ontology feature is active. ~30 LOC fix in `medcare-analytics` + 2 tests + 1 propagation change in `medcare-server::state` constructor. This is `D-UB-7` from `unified-bridge-consumer-migration-v1.md`. **No other deliverable in this plan compiles until this lands.**

### Phase 2 — Close the RLS fail-OPEN window (2 days) — SAFETY-CRITICAL

- **D-LGMC-2** — Replace `medcare-analytics::soa_mapping::ALL_SCHEMAS` iteration in `medcare-server::state::rls_registry` with `OntologyRegistry::enumerate("Healthcare")`. The dynamic projection covers all 7 Healthcare entities (Patient, Diagnosis, LabValue, Medication, Treatment, Visit, VitalSign) — adding the 3 newly-OGIT-surfaced (Treatment / Visit / VitalSign) that today bypass RLS. ~80 LOC + 4 tests including the `rls_coverage_parity_with_all_schemas` parity test already wired in `medcare-bridge/Cargo.toml [dev-dependencies] medcare-analytics`. This is `D-UB-8`. **Safety-critical: HIPAA non-conformance until this lands.**
- **D-LGMC-3** — Boot-time enumeration assertion. Gate `medcare-server` startup on `HydrationReport.namespace_entity_count("Healthcare") >= 7` (the OGIT-canonical entity count). Mirrors the F2-A RLS-registry boot pattern already documented in `medcare-bridge::registry::MedcareRegistry::hydrate_with_report`. If the count drifts (TTL adds an entity, RLS coverage falls behind), boot fails closed. ~40 LOC + 2 tests (success path; failure path).

### Phase 3 — Complete the unified-bridge constructor (1 day)

- **D-LGMC-4** — Add `medcare-bridge/src/unified_bridge_wiring.rs::medcare_unified_bridge(registry, actor_role, tenant) -> Result<UnifiedBridge<MedcareBridge>, _>`. Mirrors `smb_unified_bridge` but parameterised over `MedcareBridge` (not `OgitBridge`). actor_role enumerates per `super-domain-rbac-tenancy-v1.md` §4.3: `physician` / `nurse` / `cashier` / `researcher` / `hipaa_audit` / `admin`. ~60 LOC + 4 integration tests covering each role's redaction-mask behaviour. This is `D-UB-6`. Depends on D-LGMC-1.

### Phase 4 — MUL MEDICAL + ontology_context_id (3 days)

- **D-LGMC-5** — `MulThresholdProfile::MEDICAL` consumption at the gate site. Per D-ONTO-V5-9 in lance-graph#355: the medical profile sets the Dunning-Kruger / TrustTexture / FlowState thresholds appropriate for clinical-decision contexts (more conservative than the default `MulThresholdProfile::BALANCED`). Plumb into the `UnifiedBridge::authorize` audit emission so the audit log records which profile was active. ~60 LOC + 2 tests. This is `D-UB-9`.
- **D-LGMC-6** — `ontology_context_id`-keyed RLS extension. Third axis after `(table, praxis_id)` per §73 SGB V Überweisung shape. The use case: a Patient's Diagnosis row created under Praxis A is visible to Praxis B if and only if Praxis A issued a §73 SGB V referral (the `ontology_context_id` carries the referral identity). Adds a new column to the Healthcare Lance dataset + a third clause to the row-filter predicate. ~100 LOC + 4 tests (intra-praxis; cross-praxis with referral; cross-praxis without referral; expired referral). This is `D-UB-10`.

### Phase 5 — LanceProbe-side endpoints (4 days) — UNBLOCKS C# PARITY TOOL

These are the medcare-rs deliverables already named in `super-domain-rbac-tenancy-v1.md` Tier H (D-SDR-35..39). Reproduced here as the smaller "what does this plan ship in MedCare-rs" view:

- **D-LGMC-7** — `POST /api/__parity/csharp` ingest endpoint. Receives `DriftEvent` JSON from LanceProbe's `DriftSink` batch flushes; persists to a Lance table for cross-session aggregation. ~150 LOC + 4 tests. **Blocks LanceProbe M5.** Cross-ref: `D-SDR-35`.
- **D-LGMC-8** — `GET /api/__parity` canonical dashboard endpoint. Aggregates drift events across sessions (group-by route + time bucket); JSON output for the admin-only ParityPanel. ~120 LOC + 3 tests. **Blocks LanceProbe M5.** Cross-ref: `D-SDR-36`.
- **D-LGMC-9** — `_dto_contracts.md` document. Stable JSON DTO contracts for the 5 pilot endpoints (Patient / Lab / Vital / Diagnosis / Wartezimmer) + the planned 40+ additional routes per `MedCare-rs/docs/FUTURE_STACK_ADMIN.md` §4. Doc only, ~300 lines markdown. **Blocks LanceProbe M2.** Cross-ref: `D-SDR-37`.
- **D-LGMC-10** — `legacy-tripledes-fallback` feature flag (currently DRAFT in `AUTH_LEGACY_TRIPLEDES_MIGRATION.md`). When enabled, accepts both Argon2 and legacy-TripleDES password verification; backfills Argon2 on successful legacy auth. Per `super-domain-rbac-tenancy-v1.md` §18.5: the legacy "3DES" is actually broken-3DES-equivalent-to-single-DES with 56-bit effective key strength, so the migration is **upgrade-on-login** (not bulk rewrap-to-AES-GCM). ~180 LOC + 6 tests covering the upgrade-on-login flow + the legacy ciphertext carry-forward. **Blocks LanceProbe M5a.** Cross-ref: `D-SDR-38`. Needs human security review before merge.
- **D-LGMC-11** — `/api/__parity/telemetry` endpoint exposing flat TelemetrySnapshot equivalent for programmatic monitoring (LanceProbe's flat struct). ~80 LOC + 2 tests. **Blocks LanceProbe M6.** Cross-ref: `D-SDR-39`.

### Phase 6 — Cypher / SPARQL surface (5 days, opt-in)

- **D-LGMC-12** — Wire `lance-graph-planner` as `medcare-bridge` workspace dep behind `[features] planner`. Add `POST /api/graph/query` endpoint that accepts Cypher / SPARQL over the Healthcare Lance projection. Tenant-isolation predicate auto-injected per `UnifiedBridge::authorize` lowering to DataFusion §3.10. ~150 LOC + 4 tests.
- **D-LGMC-13** — Cohort similarity rewrites of three flagship clinical queries:
  - "patients with similar comorbidity profiles" (Patient × Diagnosis joins)
  - "lab-value trends across a Treatment cohort" (LabValue + Treatment × Visit joins)
  - "medication-interaction warnings across a patient timeline" (Medication × Patient.history)

  Each is currently expressed as hand-rolled sea-orm / DataFusion. Re-expression as Cypher demonstrates the planner equivalence + uses the per-family codebook for O(1) entity-type lookup. ~120 LOC of new handlers + 6 parity tests against the existing reference path.


### Phase 7 — CAM-PQ similarity surface (3 days, opt-in)

- **D-LGMC-14** — `EntityStore::similar_to(entity_id, limit)` over the Healthcare Lance projection. Uses CAM-PQ with the per-Patient Vsa16kF32 fingerprint summarising demographics + comorbidities + medication history. Clinical use case: "patients similar to this one for cohort suggestion." ~200 LOC + 4 tests. Feature-gated behind `medcare-bridge/cam-pq`. **Researcher role only** per `super-domain-rbac-tenancy-v1.md` §13.5 — direct identifiers always hashed, k-anonymity ≥ 10 (Healthcare default), DP noise on aggregates.

## 3 — Branch strategy (two-target reality)

Per `MedCare-rs/CLAUDE.md` "Branch + PR conventions":

| Deliverable scope | Branch | Notes |
|---|---|---|
| Fix `ontology_dto.rs:85` build (D-LGMC-1) | `main` ONLY | Pulls the full lance-phase2 stack; scaffold (lean) branch cannot host this |
| RLS coverage close (D-LGMC-2, D-LGMC-3) | `main` (cherry-pick to scaffold only if scaffold ever consumes the registry) | Safety-critical; defer scaffold cherry-pick until scaffold needs it |
| `medcare_unified_bridge` constructor (D-LGMC-4) | `main` ONLY | Behind `--features ontology` |
| MUL MEDICAL + ontology_context_id (D-LGMC-5, D-LGMC-6) | `main` ONLY | Pulls upstream dependencies |
| LanceProbe-side endpoints (D-LGMC-7..11) | Start on scaffold (lean / safer) for the endpoint shells; cherry-pick to `main` for the full integration | Per CLAUDE.md "first PR to scaffold, then cherry-pick to main" |
| Cypher / SPARQL surface (D-LGMC-12, D-LGMC-13) | `main` ONLY | Planner dep |
| CAM-PQ similarity (D-LGMC-14) | `main` ONLY | CAM-PQ dep |

Two-branch reality means every `main`-only deliverable is also a "does NOT land on scaffold" decision — the scaffold stays the production-grounded subset. D-LGMC-1's fix must explicitly preserve the scaffold's compile path (the `--no-default-features` build).

## 4 — Cross-references

- `super-domain-rbac-tenancy-v1.md` §3.9 (`UnifiedBridge::authorize`) — what `medcare_unified_bridge` returns
- `super-domain-rbac-tenancy-v1.md` §4.3 — Healthcare role matrix (`physician` / `nurse` / `cashier` / `researcher` / `hipaa_audit` / `admin`)
- `super-domain-rbac-tenancy-v1.md` §13.4 — `Healthcare ↔ OSINT` hard-lock partner row (3 layers of defense)
- `super-domain-rbac-tenancy-v1.md` §13.5 — Researcher role: anonymized projection + k-anonymity floor + DP noise
- `super-domain-rbac-tenancy-v1.md` §16 — Two-track migration (John Doe billing/tickets vs `u_pwd` only)
- `super-domain-rbac-tenancy-v1.md` §18 — Empirical MedCare + MedCareV2 inspection: LanceProbe shape, the 6 concrete canonicalization rules, D-SDR-35..39 medcare-rs deliverables
- `unified-bridge-consumer-migration-v1.md` D-UB-6..10 — sister deliverables
- `MedCare-rs/CLAUDE.md` §Architectural commitments — Iron Rule 4 ("Thinking lives only in lance-graph"); Iron Rule 1 ("MySQL is the permanent oracle / parity witness"); Iron Rule 5 ("Approval gate for upstream lance-graph changes")
- `MedCare-rs/docs/POST_PR355_FINDINGS.md` — full triage of the post-#355 state, including the four "not yet wired" items this plan closes
- `MedCare-rs/docs/CSHARP_HANDOFF_PROMPT.md` — the LanceProbe coordination spec (M1..M6 milestones)
- `MedCare-rs/docs/AUTH_LEGACY_TRIPLEDES_MIGRATION.md` — DRAFT plan that D-LGMC-10 promotes to Active

## 5 — Open questions

1. **`MedcareOntology` `Option<>` vs builder.** D-LGMC-1 makes `MedcareOntology` populated only when `--features ontology` is active. Does `medcare-server` boot need an explicit "ontology-required mode" gate that fails closed if no registry is provided, or is the silent skip acceptable for the lean fallback?
2. **`ontology_context_id` storage shape.** D-LGMC-6 adds a third axis. Is `ontology_context_id` a TenantId-style transparent u32 or a richer struct (referral_id + issued_by + valid_until + scope)? Closer to richer struct given the §73 SGB V semantics, but adds complexity.
3. **`u_pwd` is the only encrypted column.** Per §18.5 of the super-domain plan, confidence is "likely very few or none" beyond `u_pwd`. Needs a focused grep of `MySQL_Connect.cs::EncryptMessage()` call sites in MedCareV2 (~721 KB file). If other columns surface, D-LGMC-10's scope expands.
4. **LanceProbe DTO contract bus.** D-LGMC-9 publishes stable JSON DTO contracts for 5 pilot endpoints. The 40+ additional routes per `FUTURE_STACK_ADMIN.md §4` need enumeration — likely a Sonnet grindwork pass.
5. **MedCareV2 cannot be reshaped.** Per `super-domain-rbac-tenancy-v1.md` §18.1 correction: MedCareV2 is MedCare verbatim + LanceProbe overlay, with explicit "do NOT refactor" constraint. Implications for D-LGMC-7 / D-LGMC-8 / D-LGMC-11: the ingest + dashboard + telemetry endpoint shapes must match what LanceProbe already targets (HTTP+JSON over JWT — Flight SQL is Phase 5+).

## 6 — Status

- **Phase 1:** Not started. **Critical path; blocks everything else.** ~1 day.
- **Phase 2:** Not started. Safety-critical (HIPAA non-conformance window open until D-LGMC-2 lands). ~2 days.
- **Phase 3:** Not started. Blocked on Phase 1. ~1 day.
- **Phase 4:** Not started. Blocked on Phase 1. ~3 days.
- **Phase 5:** Some scaffolding in place (`AUTH_LEGACY_TRIPLEDES_MIGRATION.md` DRAFT). Unblocks LanceProbe M2..M6. ~4 days total.
- **Phase 6:** Phase 4-dependent. Opt-in (~5 days).
- **Phase 7:** Phase 6-dependent. Opt-in (~3 days).

**Confidence:** Working — substrate ships upstream and the four "what is NOT wired" items in `medcare-bridge/src/lib.rs` are concrete and bounded. D-LGMC-1's fix is mechanical (replace one Default impl); D-LGMC-2's fix is the safety-critical one. Once Phase 1+2 land, the rest fans out in parallel.

## 7 — One-line summary

> Critical path is D-LGMC-1 (unblock the lance-phase2 build by fixing `ontology_dto.rs:85`) → D-LGMC-2 (close the RLS fail-OPEN window for Treatment / Visit / VitalSign) → D-LGMC-4 (ship the `medcare_unified_bridge` constructor); the LanceProbe-side endpoints (D-LGMC-7..11) are the second-track work that unblocks the C# parity tool's M2..M6 milestones; Cypher + CAM-PQ surfaces are opt-in after the spine is stable.


---

## 8 — Parallelbetrieb already shipped + MongoDB as alternative cold path (2026-05-21, same session)

User clarifications, two parts:

1. **The sink-with-diff-monitoring is already shipped.** What `lance-graph-in-medcare-rs-v1.md` Phase 5 framed as "D-LGMC-7..11 to design from scratch" is mostly already in the tree. The refinement below documents the existing state and tightens the deliverable list.
2. **MongoDB has been added as an alternative cold path** (parallel to MySQL, not replacing it). MySQL stays the permanent parity oracle per `MedCare-rs/CLAUDE.md` Iron Rule 1; MongoDB is the second cold-path option, propagated from smb-office-rs's three-layer mongo shape. The hot/cold framing for medcare-rs becomes: **hot = lance-graph-ontology + Lance projection; cold (A) = sea-orm + MySQL (legacy parity oracle); cold (B) = MongoDB (alternative).**

### 8.1 What's already in the tree (parallelbetrieb infrastructure)

| Layer | File | LOC | Status |
|---|---|---:|---|
| Upstream trait + DTOs | `lance_graph_callcenter::transcode::parallelbetrieb` | 376 | **Shipped.** `DriftKind` (Match / ValueDrift / ShapeDrift / MissingMysql / MissingLance), `DriftField`, `DriftEvent`, `Reconciler` trait, `validate_route()`. Self-framed as "the one deliberate transition bandaid." |
| Rust-side MedCare reconciler (C1) | `medcare-analytics::mysql_reconciler::MedcareMysqlReconciler` | 461 | **Shipped, Round-1 = Patient only.** Route `/api/patient/{id}`. 11 unit tests cover Match / ValueDrift (single & multi-field) / MissingMysql / MissingLance / ShapeDrift / unsupported / malformed / extra-segments / C#-schema-parity / parser-tests. Pluggable `PatientFetcher` for testability. |
| Sister SMB reconciler (C2) | `smb_bridge::mongo_reconciler::SmbMongoReconciler` | 395 | **Shipped, Round-1 = Customer only.** Route `/api/customer/{kdnr}`. Same `Reconciler` trait, MongoDB instead of MySQL — the cross-source pattern this plan inherits. |
| C# parity probe | `MedCareV2/MedCare_2.0/LanceProbe/` (ParityClient + ParityWitness + DriftSink) | — | **Shipped (M1).** Same `DriftEvent.ToJson()` schema across both languages. |
| medcare-rs ingest + dashboard | `medcare-server::routes::parity` — `POST /api/__parity/csharp` + `GET /api/__parity` | 94 | **Shipped.** 1024-event ring buffer in `OnceLock<Mutex<VecDeque>>`. Auth: any authenticated user POSTs; `GET` is admin-only. |
| 5-phase migration narrative | `docs/medcare-umstellung.md` | 102 | **Shipped.** F1 dual-write → F2 reconciler-live (consumers still MySQL) → F3 consumers-switch → F4 features-live → F5 RBAC. MySQL is permanent witness per Iron Rule 1. |
| CLAUDE.md Iron Rule 1 | `MedCare-rs/CLAUDE.md` § Architectural commitments | — | **Locked.** "MySQL is the permanent oracle / parity witness. It is never retired, even after F4. The reconciler witnesses every promotion gate forever." |

### 8.2 What's still deferred (per source comments)

- **Round-2 routes** on the existing reconciler shell: Lab / Vital / Diagnosis / Prescription. Per `mysql_reconciler.rs:7-11`: "Other routes return a `ShapeDrift` event with reason 'route not in Round-1 scope'. Lab / Vital / Diagnosis / Prescription land in follow-up PRs that all share the same `MedcareMysqlReconciler` shell — only the per-route dispatch table grows."
- **Production query handles**: Closures inside `MedcareMysqlReconciler` are unit-tested with canned rows; production needs them to wrap `medcare_db::queries::patient::get_patient(...)` for the MySQL side + the corresponding Lance read for the SPO side.
- **Canonicalizer table centralization**: 6 field rules (date-only, "F4" doubles, soft-delete coercion, second-truncated timestamps, German locale handling, ciphertext byte-compare for `u_pwd`) need to land in **one place** that both the Rust reconciler and the C# ParityWitness reference. Per `parallelbetrieb.rs:51`: "Land both rules in one place when the Rust query path is wired (Phase 3 of `.claude/plans/sql-spo-ontology-bridge-v1.md`)."
- **Persistent sink**: Today's ring buffer is in-process only (`OnceLock<Mutex<VecDeque>>`, capped 1024). Per parallelbetrieb.rs doc: "The Rust side will publish to the same persistent ring buffer (`crate::audit::LanceAuditSink`) once the wiring lands."
- **Rust-side reconciler runner**: The trait + impl exist; the **loop that periodically issues queries against both sides and feeds `Reconciler::reconcile()`** is not yet wired (deferred per `parallelbetrieb.rs:44-54`).


### 8.3 Phase 5 deliverable corrections (D-LGMC-7..11 status update)

| D-id | §5 original | §8 corrected |
|---|---|---|
| D-LGMC-7 | `POST /api/__parity/csharp` ingest absent — 150 LOC + 4 tests | **SHIPPED** in `medcare-server::routes::parity::ingest_csharp`. Drop from open deliverable list. Optional follow-up: verify 1024-event ring cap is right for production load + add a `purge_after: Duration` config. |
| D-LGMC-8 | `GET /api/__parity` dashboard absent — 120 LOC + 3 tests | **SHIPPED** (same module, admin-only read of the ring buffer). Drop from open deliverable list. Optional follow-up: per-route + per-time-bucket aggregation per the original spec; today's endpoint returns the raw newest-first event stream. |
| D-LGMC-9 | `_dto_contracts.md` for 5 pilot endpoints | **STILL OPEN.** `docs/CSHARP_HANDOFF_PROMPT.md` references the contracts but doesn't enumerate the 5 pilot endpoints' JSON shapes verbatim. Per the C# handoff this blocks LanceProbe M2. |
| D-LGMC-10 | `legacy-tripledes-fallback` feature flag — DRAFT | **STILL DRAFT** per `docs/AUTH_LEGACY_TRIPLEDES_MIGRATION.md`. Blocks LanceProbe M5a. |
| D-LGMC-11 | `/api/__parity/telemetry` endpoint absent | **STILL OPEN.** No telemetry endpoint today. Blocks LanceProbe M6. |

### 8.4 New Phase 5b deliverables (Round-2 routes on the existing reconciler shell)

Extend `MedcareMysqlReconciler` from Round-1 (Patient) to Round-2 (Lab / Vital / Diagnosis / Prescription) by growing the per-route dispatch table. Each adds ~80 LOC of canonical row type + diff impl + tests; the reconciler shell stays untouched.

- **D-LGMC-15** — `CanonicalLabRow` + `LabFetcher` + `parse_lab_route("/api/lab/{id}")` + diff impl. ~80 LOC + 5 tests (match / value-drift / missing-mysql / missing-lance / shape-drift).
- **D-LGMC-16** — `CanonicalVitalRow` + `VitalFetcher` + `parse_vital_route("/api/vital/{id}")` + diff impl. ~80 LOC + 5 tests.
- **D-LGMC-17** — `CanonicalDiagnosisRow` + `DiagnosisFetcher` + `parse_diagnosis_route("/api/diagnosis/{id}")` + diff impl. ~80 LOC + 5 tests.
- **D-LGMC-18** — `CanonicalPrescriptionRow` + `PrescriptionFetcher` + `parse_prescription_route("/api/prescription/{id}")` + diff impl. ~80 LOC + 5 tests.
- **D-LGMC-19** — Production query-handle wiring: replace the closure fetchers with `medcare_db::queries::{patient,lab,vital,diagnosis,prescription}::get_*(...)` for the MySQL side + the corresponding Lance reads for the SPO side. ~150 LOC + 5 integration tests against a real MySQL fixture + Lance dataset (gated behind `--features mysql-integration lance-phase2`).
- **D-LGMC-20** — Canonicalizer-table centralization: hoist the 6 field rules (date-only, "F4" doubles, soft-delete coercion, second-truncated timestamps, German locale, `u_pwd` ciphertext byte-compare) into a shared `lance_graph_callcenter::transcode::canonicalize` module so both the Rust reconciler and the C# ParityWitness reference the same source. ~120 LOC + 6 tests. **Upstream PR required** (Iron Rule 5).
- **D-LGMC-21** — Persistent sink: replace the in-process ring buffer with `lance_graph_callcenter::audit::LanceAuditSink` so drift events survive restart and feed the cross-session dashboard. ~80 LOC + 3 tests covering buffer-overflow eviction + restart-recovery + concurrent-writer ordering.

### 8.5 New Phase 9 — MongoDB cold path (propagated from smb-office-rs)

Add MongoDB as an **alternative** cold path. MySQL stays the permanent oracle per Iron Rule 1; MongoDB is a second cold-path option for tenants / deployments where MongoDB is the system of record (or for new clinical entities authored cleanly against the OGIT TTL shape without a MySQL ancestor). Both feed the same hot path (lance-graph-ontology + Lance projection) and both are witnessed by the same parallelbetrieb reconciler.

Mirror the smb-office-rs three-layer shape:

| smb-office-rs (template) | medcare-rs (mirror, NEW) | Notes |
|---|---|---|
| `crates/smb-mongo/` (connector + `MongoImporter` + `MigrationStats`, 205 LOC, depends on workspace `mongodb` + `bson`) | `crates/medcare-mongo/` (NEW) | Same workspace deps. Importer reads clinical collections (per OGIT/NTO/Healthcare entities) into in-process cache keyed by `<entity_prefix>:<hex _id>`. |
| `crates/smb-bridge/src/mongo.rs` (`MongoConnector` impl of `EntityStore + EntityWriter`, 313 LOC, gated `[features] mongo`) | `crates/medcare-bridge/src/mongo.rs` (NEW) | Same trait surface (`lance-graph-contract::repository::{EntityStore, EntityWriter}`); BSON wire shape carries the Healthcare-namespace entity properties from `medcare-ontology` (when that lands per D-WLG-4-equivalent for medcare). Gated `[features] mongo`. |
| `crates/smb-bridge/src/mongo_reconciler.rs` (`SmbMongoReconciler`, 395 LOC, `Reconciler` impl) | `crates/medcare-analytics/src/mongo_reconciler.rs` (NEW) | Mirrors `mysql_reconciler.rs` 1:1: pluggable `<Entity>Fetcher` trait with `fetch_mongo(...)` + `fetch_lance(...)` methods; per-route dispatch table covering Round-1 Patient + Round-2 Lab/Vital/Diagnosis/Prescription; same `DriftEvent` JSON shape. |

Concrete deliverables:

- **D-LGMC-22** — `crates/medcare-mongo/` new crate. `MongoImporter::new(uri, db)` + `import_all() -> MigrationStats` covering Patient / Diagnosis / LabValue / Medication / Treatment / Visit / VitalSign collections. Mirrors `smb-mongo::MongoImporter` 1:1 with healthcare-namespace entity names. ~200 LOC + 4 tests (per-collection import sanity; full-import stats; error-bag captures per-collection failures without aborting; round-trip BSON roundtrip).
- **D-LGMC-23** — `crates/medcare-bridge/src/mongo.rs` new module. `MongoConnector` impl of `EntityStore + EntityWriter` for the same 7 Healthcare entities. Gated `[features] mongo = ["dep:medcare-mongo", "dep:mongodb", "dep:bson"]`. ~250 LOC + 6 tests (per-entity insert / update / soft-delete / fetch-by-id / list-by-tenant / cross-collection join via Lance projection).
- **D-LGMC-24** — `crates/medcare-analytics/src/mongo_reconciler.rs` new module. `MedcareMongoReconciler<F: PatientFetcher>` (and equivalents per Round-2 entity), `Reconciler` impl over the same trait. Same JSON wire shape as `MedcareMysqlReconciler`. ~400 LOC + 11 tests (mirrors the existing `mysql_reconciler` test suite verbatim, MongoDB substituted for MySQL). The reconciler shell is shared; only `fetch_mongo()` differs from `fetch_mysql()`.
- **D-LGMC-25** — Cold-path selection: `medcare-server` config layer adds `cold_path: ColdPath::{MySql, Mongo}` so a deployment picks one (or both, with double-reconciler for diverse-redundancy). ~50 LOC + 2 tests (config parsing + boot-time selection).
- **D-LGMC-26** — Dual-reconciler mode: when both cold paths are active, the runner issues queries against MySQL **and** MongoDB **and** Lance, emitting a `DriftEvent` per pair. Triple-redundancy diverges into N(N-1)/2 = 3 pairwise comparisons per query (MySQL↔Mongo, MySQL↔Lance, Mongo↔Lance). ~100 LOC + 4 tests.

### 8.6 Hot/cold framing update

Reframes §2 of the woa-rs sister plan (`lance-graph-in-woa-rs-v1.md` §9) for medcare-rs:

| Path | Carrier | Role | Authority |
|---|---|---|---|
| Hot | `OgitFamilyTable` lookup + `UnifiedBridge<MedcareBridge>` 4-stage authorize + Lance projection scans | O(1) resolution / RBAC / Cypher / CAM-PQ similarity | Read-only; rebuilds from cold path on drift |
| Cold A (legacy oracle) | sea-orm + MySQL via `medcare-db` | System of record; byte-exact row reads; DATEV-equivalent regulatory exports | **Authoritative on drift.** Permanent per Iron Rule 1. |
| Cold B (alternative, NEW) | `medcare-bridge::mongo::MongoConnector` over MongoDB | Alternative system of record for deployments / entities authored against OGIT TTL without a MySQL ancestor | Authoritative for its own entities; reconciler witnesses MySQL↔Mongo agreement when both active |

The parallelbetrieb reconciler now has three roles to triangulate:

```
                    ┌────────────────────────────┐
                    │   Hot path                 │
                    │   (Lance projection,       │
                    │    via lance-graph-        │
                    │    ontology)               │
                    └────────────┬───────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │  Reconciler                 │
                  │  emits DriftEvent           │
                  │  (Match/Value/Shape/        │
                  │   Missing*)                 │
                  └──┬────────────────────────┬─┘
                     │                        │
        ┌────────────┴────────┐    ┌──────────┴───────────┐
        │  Cold A: MySQL      │    │  Cold B: MongoDB     │
        │  (legacy oracle,    │    │  (alternative,       │
        │   sea-orm DTO)      │    │   medcare-bridge::    │
        │  Iron Rule 1:       │    │   mongo)              │
        │  permanent witness  │    │                       │
        └─────────────────────┘    └───────────────────────┘
```

Pairwise drift comparisons in dual-reconciler mode (D-LGMC-26):

1. **MySQL ↔ Lance** — already shipped via `MedcareMysqlReconciler` Round-1.
2. **MongoDB ↔ Lance** — new via D-LGMC-24 (`MedcareMongoReconciler` Round-1 + Round-2).
3. **MySQL ↔ MongoDB** — new via D-LGMC-26 dual-mode (compares the two cold paths directly without going through Lance).

### 8.7 Sequencing impact

Phases 1-7 of §2 stay; Phase 5 is reduced (D-LGMC-7+8 are shipped, drop them); Phase 5b is added (D-LGMC-15..21 — Round-2 + production wiring + persistent sink); Phase 9 is new (D-LGMC-22..26 — MongoDB cold-path propagation from smb-office-rs).

```
         ┌──────────────────────────────────────────────────────────┐
         │ Phase 1 (D-LGMC-1)  fix ontology_dto.rs:85 build         │
         │ critical path; nothing else compiles until this lands    │
         └────────────────────────┬─────────────────────────────────┘
                                  │
              ┌───────────────────┼──────────────────────────────┐
              │                   │                              │
         ┌────▼──────┐   ┌────────▼────────┐         ┌───────────▼──────────┐
         │ Phase 2   │   │ Phase 3 (LGMC-4)│         │ Phase 5 (open items) │
         │ (LGMC-2,3)│   │ unified-bridge  │         │ (LGMC-9,10,11)       │
         │ RLS close │   │ constructor     │         │ DTO docs + 3DES + tel│
         └────┬──────┘   └────────┬────────┘         └──────────────────────┘
              │                   │
              │       ┌───────────┴────────────────┐
              │       │                            │
         ┌────▼───────▼───┐         ┌──────────────▼──────────────┐
         │ Phase 4         │         │ Phase 5b (NEW)              │
         │ (LGMC-5,6)      │         │ Round-2 reconciler + prod   │
         │ MUL MEDICAL     │         │ wiring + persistent sink    │
         │ + context_id    │         │ LGMC-15..21                 │
         └─────────────────┘         └──────────────┬──────────────┘
                                                    │
                                          ┌─────────▼─────────────┐
                                          │ Phase 9 (NEW)         │
                                          │ MongoDB cold path     │
                                          │ propagation from SMB  │
                                          │ LGMC-22..26           │
                                          └───────────────────────┘
```

### 8.8 Open questions

1. **Why MongoDB for medcare specifically?** smb-office-rs has it because the inherited C# WinForms ERP persisted to MongoDB with German BSON field names — the cold-path mongo is the legacy data. For medcare-rs the legacy is C# + MySQL + 3DES; what's the equivalent driver for adding MongoDB? Possibilities:
   - **New clinical entities** authored cleanly against OGIT TTL with no MySQL ancestor (Treatment / Visit / VitalSign — the 3 NEW entities D-LGMC-2 closes RLS for) — ship them on MongoDB to avoid MySQL schema migration.
   - **Tenant choice**: some deployments prefer MongoDB ops; per-tenant config.
   - **Foundry-shape ingest path**: external systems (FHIR / HL7 / OpenEHR feeds) land BSON natively; MongoDB is the staging area before promotion to Lance.
   - **Other** (please clarify).
2. **Dual-write or dual-cold-path?** If both MySQL and MongoDB are active, do writes go to both (dual-write, with the reconciler witnessing agreement) or does each entity have a single declared cold path? Affects D-LGMC-25's config shape.
3. **Per-entity cold-path routing.** If different entities have different cold paths (Patient → MySQL, Treatment → MongoDB), the bridge's `EntityWriter::write_<Entity>(...)` needs to dispatch on entity type. Closer to a Router pattern than a single config switch.
4. **Reconciler triple-redundancy cost.** D-LGMC-26's three pairwise comparisons triple the reconciler load per query. Acceptable for sample-gated drift detection (default 1% per `CSHARP_HANDOFF_PROMPT.md`); needs sampling-rate tuning if dual-mode is the default.

### 8.9 Cross-references (additive)

- `MedCare-rs/crates/medcare-analytics/src/mysql_reconciler.rs` — the working reference for D-LGMC-15..18 (Round-2 expansion) and D-LGMC-24 (mongo sister)
- `MedCare-rs/crates/medcare-server/src/routes/parity.rs` — the shipped ingest/dashboard endpoints (drops D-LGMC-7 and D-LGMC-8)
- `MedCare-rs/docs/medcare-umstellung.md` — the 5-phase F1-F5 narrative this plan extends
- `MedCare-rs/docs/foundry-roadmap-unified-smb-medcare-v1.md` §4 — the smb→medcare mirror table that maps `MongoConnector` to `MySqlConnector` (D-LGMC-22..24 expand this to MongoConnector + MySqlConnector BOTH)
- `lance-graph/crates/lance-graph-callcenter/src/transcode/parallelbetrieb.rs` — the upstream trait (D-LGMC-20 lands the canonicalize sub-module here)
- `lance-graph/crates/lance-graph-callcenter/src/audit/` (target for D-LGMC-21 `LanceAuditSink`) — needs confirmation it ships; if absent, D-LGMC-21 first adds it upstream
- `smb-office-rs/crates/smb-mongo/` — the L1 template D-LGMC-22 mirrors
- `smb-office-rs/crates/smb-bridge/src/mongo.rs` — the L2 template D-LGMC-23 mirrors
- `smb-office-rs/crates/smb-bridge/src/mongo_reconciler.rs` — the L3 template D-LGMC-24 mirrors

### 8.10 Status

- **Existing parallelbetrieb infrastructure:** working — Round-1 Patient reconciler + ingest/dashboard endpoints + C# probe + 5-phase narrative all shipped.
- **Round-2 routes (D-LGMC-15..18):** not started. Mechanical extension of the existing shell (~80 LOC each).
- **Production wiring (D-LGMC-19):** not started. Blocked on D-LGMC-1 (build fix).
- **Canonicalizer centralization (D-LGMC-20):** not started. Upstream PR required.
- **Persistent sink (D-LGMC-21):** not started. Depends on `LanceAuditSink` existing upstream.
- **MongoDB cold path (D-LGMC-22..26):** not started. Mirrors three-layer smb-office-rs shape; ~1000 LOC + ~25 tests total.

**Confidence:** Working. The reconciler architecture is locked + partially shipped; the MongoDB propagation has a verbatim template in smb-office-rs; the open questions in §8.8 are about scope / configuration, not architecture.

### 8.11 One-line summary

> Parallelbetrieb infrastructure is already shipped Round-1 (Patient on MySQL ↔ Lance, with the C# ParityWitness, ingest/dashboard endpoints, and 5-phase F1-F5 narrative). This refinement (a) acknowledges what ships, (b) reframes Phase 5 around Round-2 expansion + production wiring + persistent sink, (c) adds Phase 9 to propagate the smb-office-rs three-layer MongoDB shape into medcare-rs as an alternative cold path (alongside, not replacing, MySQL — Iron Rule 1 preserved).


---

## 9 — Position vs sister consumer plans: behind on UnifiedBridge, otherwise the quickest (2026-05-21, same session)

User observation: **MedCare-rs is behind on the UnifiedBridge migration and probably not yet wired to it — but other than that, it's the quickest of the three consumer plans to land.** Spelling that out:

### 9.1 Work-remaining ranking across the three consumers

| Consumer | Substrate maturity | UnifiedBridge constructor | Net work remaining |
|---|---|---|---|
| **MedCare-rs** | **Most mature.** Parallelbetrieb reconciler shipped Round-1 (Patient); `/api/__parity/csharp` ingest + `GET /api/__parity` dashboard endpoints both live; `MedcareRegistry::hydrate(...)` helper shipped; lance-phase2 build wiring shipped (except the one broken call site at `ontology_dto.rs:85`); F1-F5 migration narrative documented; LanceProbe C# parity tool M1 complete; reconciler shell tested with 11 unit tests; MySQL parity-oracle Iron Rule 1 locked. | **Absent.** `medcare_unified_bridge(registry, actor_role, tenant) -> UnifiedBridge<MedcareBridge>` does not exist (D-UB-6 = D-LGMC-4). The consumer is still on the older direct-`OntologyRegistry` + `MedcareBridge::new(...)` shape. | **Smallest.** Once D-LGMC-1 (build fix) + D-LGMC-4 (constructor) land, the remaining work is mostly Round-2 reconciler expansion (D-LGMC-15..18 ~80 LOC each) + LanceProbe-side TODOs (D-LGMC-9..11) + optional MongoDB cold path (D-LGMC-22..26). The headline gap (UnifiedBridge) is ~60 LOC + 4 tests; total Phase-1..4 closure is ~5-7 days. |
| **smb-office-rs** | **Partial.** `smb-bridge` + `smb-ontology` crates shipped (Mongo + Lance EntityStore/EntityWriter impls; auth + RLS features wired against lance-graph-callcenter/-ontology/-rbac). `smb_unified_bridge` constructor EXISTS — but parameterised over `OgitBridge::for_namespace(...)` because `SmbBridge` doesn't exist upstream yet. | **Exists, parameterised over the wrong bridge type.** Type-parameter swap from `UnifiedBridge<OgitBridge>` to `UnifiedBridge<SmbBridge>` is a 15-LOC change (D-UB-5 / D-LGSMB-3) once D-UB-2 (SmbBridge skeleton in lance-graph-ontology) ships. | **Medium.** SmbBridge skeleton (3 days upstream) + parameter swap (15 LOC) + TTL authoring for `OGIT/NTO/SMB/` (4 days) + role groups D-SDR-2 expansion (Phase B) + Phase C tenant-type consolidation (2 days). Phase D-E opt-in. Total Phase-A..C: ~9 days. |
| **woa-rs** | **Greenfield.** OGIT TTL vendored at `vendor/ogit/v02-harvest/`; sea-orm + MySQL writer-parity established per DualSink-Pivot 2026-05-15; RFC v02-006 (route codegen) DRAFT; NO `woa-bridge` crate; NO `woa-ontology` crate; NO `lance-graph-*` Cargo dep declared. Only existing lance-graph touch-point is `tests/ontology_cypher_round_trip.rs`. | **Absent.** `woa_unified_bridge(registry, actor_role, tenant)` would land at D-UB-4, but requires Phase 1 of `lance-graph-in-woa-rs-v1.md` (Phase 0 vendor symlink + exclude block + Phase 1 woa-bridge + woa-ontology crates) to complete first. | **Largest.** Phase 0 (1 day mechanical) + Phase 1 (3 days, crate scaffolding mirroring medcare/smb references) + Phase 2 (4 days, route-handler integration) + Phase 3 (5 days, lance-cache + Lance projection) = ~13 days for the equivalent of medcare's already-shipped baseline. Phases 4-5 opt-in. |

### 9.2 Why MedCare-rs is "behind on UnifiedBridge"

Concretely, the consumer pattern medcare-rs ships today (per `MedCare-rs/crates/medcare-bridge/src/registry.rs`):

```rust
// Today's shape — direct registry + direct bridge:
pub struct MedcareRegistry {
    pub registry: Arc<OntologyRegistry>,
    pub bridge: MedcareBridge,
}

impl MedcareRegistry {
    pub fn hydrate(ttl_root: impl AsRef<Path>) -> Result<Self, Error> { ... }
}
```

Consumers (medcare-server, medcare-analytics) take `&MedcareBridge` for ontology resolution and call `registry.resolve(...)` for raw URI lookups. RBAC and tenant isolation are NOT yet composed via `UnifiedBridge::authorize` — they're handled separately by the existing `medcare-rbac` crate's row-level-security registry. There is no single 4-stage authorize call.

The migration target (per D-UB-6 / D-LGMC-4) is:

```rust
// Target shape — UnifiedBridge composes registry + RBAC + tenant:
pub fn medcare_unified_bridge(
    registry: Arc<OntologyRegistry>,
    actor_role: &'static str,  // physician / nurse / cashier / researcher / hipaa_audit / admin
    tenant: TenantId,           // mapped from Praxis.id
) -> Result<UnifiedBridge<MedcareBridge>, lance_graph_ontology::Error>;
```

Route handlers in medcare-server stop calling `medcare-rbac::rls::check(...)` + `medcare-bridge::registry.resolve(...)` separately and start calling `state.unified_bridge.authorize(owl, row_tenant, op)` (one masked-predicate combine per row, lowered to DataFusion §3.10). The 4-stage flow (chinese-wall → super-domain → role group → slot redaction) becomes one bridge call.

**What "behind" means** (relative to the sister consumers): smb-office-rs has the `smb_unified_bridge` constructor in tree today (parameterised over OgitBridge as a placeholder); medcare-rs's `medcare_unified_bridge` constructor doesn't exist yet. So MedCare-rs is one constructor-and-call-site-migration behind SMB on the headline migration path.

### 9.3 Why MedCare-rs is "the quickest" despite being behind

The substrate gap that makes woa-rs Phase 0 + 1 expensive (vendor symlink + workspace exclude + new crate scaffolding for both `woa-bridge` and `woa-ontology`) is **already done** in MedCare-rs:

- `crates/medcare-bridge/` exists ✓
- `vendor/lance-graph/` softlink + `vendor/ndarray/` softlink + Cargo.toml exclude block exists ✓
- `MedcareRegistry::hydrate(...)` helper exists ✓
- `MedcareBridge` re-export from lance-graph-ontology exists ✓
- `lance-phase2` feature flag exists ✓
- `medcare-rbac` (sister crate, the RLS surface) exists ✓
- LanceProbe parity tool (the C# diverse-redundancy witness) exists ✓
- `POST /api/__parity/csharp` ingest endpoint exists ✓
- 5-phase F1-F5 narrative documented ✓

The remaining gap is **wire-up, not scaffolding**. The headline list:

1. Fix `ontology_dto.rs:85` (~30 LOC).
2. Add `medcare_unified_bridge` constructor (~60 LOC).
3. Close RLS fail-OPEN window for Treatment/Visit/VitalSign (~80 LOC).
4. Round-2 reconciler expansion to Lab/Vital/Diagnosis/Prescription (~320 LOC across D-LGMC-15..18, 80 LOC each).
5. Persistent sink swap from in-process ring → `LanceAuditSink` (~80 LOC).

Total: ~570 LOC + ~25 tests across 5 commits. **Plausibly a single 1-2 day session once D-LGMC-1 unblocks the build.** Compare:

- smb-office-rs Phase A-C closure: ~9 days (TTL authoring is the labour-intensive part).
- woa-rs Phase 0-3 closure: ~13 days (greenfield crate work + route-handler integration + Lance projection).

### 9.4 Sequencing implication

If a session needs to prove the unified-bridge story end-to-end on a single consumer, **MedCare-rs is the right target**. It demonstrates:

- The full reconciler loop (MySQL ↔ Lance + drift events ↔ C# parity probe).
- The 4-stage `UnifiedBridge::authorize` flow on a real consumer.
- The full Lance-cache persistence + boot rehydration cycle.
- The full handover protocol against MedCareV2 LanceProbe (the cross-language diverse-redundancy witness).
- HIPAA-style super-domain compliance regime (the most regulated of the three, per `super-domain-rbac §3.7 ComplianceRegime::HIPAA`).

Recommended micro-sprint shape for the "prove it" session: **D-LGMC-1 → D-LGMC-4 → D-LGMC-2 + D-LGMC-3 → smoke test against the LanceProbe ingest endpoint.** Four PRs, ~200 LOC, one drift-clean window confirms the parallelbetrieb end-to-end.

### 9.5 Status update on §6 (revised ranking)

| Phase | Original §6 status | §9 revised |
|---|---|---|
| Phase 1 (D-LGMC-1 — build fix) | Critical path; ~1 day | Unchanged — still the gate |
| Phase 2 (D-LGMC-2/3 — RLS close) | Safety-critical; ~2 days | Unchanged |
| Phase 3 (D-LGMC-4 — UnifiedBridge constructor) | ~1 day | **HEADLINE GAP** — this is the "MedCare is behind on UnifiedBridge" item the user surfaced. Sequenced after D-LGMC-1, blocking. |
| Phase 4 (D-LGMC-5/6 — MUL MEDICAL + context_id) | ~3 days | Unchanged |
| Phase 5 (D-LGMC-9/10/11 — DTO docs + TripleDES + telemetry) | ~4 days, some scaffolding partial | Mostly unchanged; D-LGMC-7/8 SHIPPED per §8.3 |
| Phase 5b (D-LGMC-15..21 — Round-2 + production wiring + persistent sink) | ~5 days | Unchanged |
| Phase 6 (D-LGMC-12/13 — Cypher/SPARQL) | ~5 days opt-in | Unchanged |
| Phase 7 (D-LGMC-14 — CAM-PQ similarity) | ~3 days opt-in | Unchanged |
| Phase 9 (D-LGMC-22..26 — MongoDB alt cold path) | ~6-8 days | Unchanged |

### 9.6 One-line summary

> MedCare-rs is behind on the UnifiedBridge migration itself (no `medcare_unified_bridge` constructor) but ahead on substrate (parallelbetrieb shipped + ingest + dashboard + MedcareRegistry + F1-F5 narrative + LanceProbe parity + lance-phase2 wiring all live). Net: smallest total remaining work of the three consumers, biggest single point-blocker (D-LGMC-1 build fix → D-LGMC-4 constructor). Best target for the "prove the unified-bridge story end-to-end" micro-sprint.
