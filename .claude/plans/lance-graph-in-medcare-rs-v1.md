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

