# PR-E1 — MedCare Super-Domain Integration: Finalisation Spec

> **Author:** W6 (sprint-log-5-6), 2026-05-13
> **Branch:** claude/lance-datafusion-integration-gv0BF (MedCare-rs)
> **Substrate context:** MedCare-rs#112 (PR-B) merged 2026-05-13. Wired `UnifiedBridge<MedcareBridge>` + medcare-rbac + medcare-realtime (+2963 LOC, 17 files, §73 SGB V + BMV-Ä §57 + BtM regulatory tests).
> **This spec:** PR-E1 = the finalisation — what is still missing from MedCare's super-domain integration after PR-B's initial wiring.
> **Parent plan:** `.claude/plans/super-domain-rbac-tenancy-v1.md`
> **W3 dependency:** `.claude/specs/pr-d4-family-hydration.md` (sprint-5 W9 — TTL hydration that seeds `FAMILY_TO_SUPER_DOMAIN`, unblocks every Healthcare-partition lookup in this spec).

---

## 1 — Gap Analysis: §14 Expected MedCare Surface vs PR-B Substrate

Super-domain-rbac-tenancy-v1.md §14 defines the canonical meta-bridge pattern extracted from `medcare_bridge.rs` as harvest source. The table below maps what §14 requires against what PR-B shipped.

### 1.1 What §14 prescribes (MedCare-specific surface)

| §14 Item | Requirement | Source |
|---|---|---|
| **D-SDR-2** | `SuperDomain::Healthcare` role groups: `physician`, `nurse`, `cashier`, `researcher`, `hipaa_audit`, `admin` as `RoleGroup` entries in `lance-graph-contract::rbac` | §3.4, §4.3, §13.5 |
| **D-SDR-3** | `OgitFamilyTable` seeded with Healthcare-partition basins (FMA, SNOMED, ICD10, RxNorm, LOINC, MONDO, HPO, DRON, CHEBI, RadLex) | §3.3 |
| **D-SDR-5** | `UnifiedBridge::authorize()` wired with real medcare `Policy` (not SMB placeholder); clinical roles resolve correctly against `SuperDomain::Healthcare` | §3.9 |
| **D-SDR-13** | `merkle_salt` on `SuperDomainEntry` + HKDF per-super-domain key derivation | §13.3, §13.4 |
| **D-SDR-14** | `AuditEntry` with `MerkleRoot + ClamPath + super_domain_salt` emitted per clinical access; HIPAA §164.312(b) log | §13.3 |
| **D-SDR-15** | `PolicyKind::DifferentialPrivacy` active for `researcher` role: k-anonymity floor + ε-noise on aggregates | §13.5 |
| **D-SDR-17** | Hard-lock partner matrix: Healthcare ↔ OSINT predicate-time enforcement | §13.4 |
| **D-SDR-18** | Archaeology pass: fix-commits from `medcare_bridge.rs` extracted as named tests in `meta_bridge::tests` | §14.1 |
| **D-SDR-19** | `MetaBridge` trait + `BridgeFromRegistry` extension | §14.2 |
| **D-SDR-21** | `MedCare-rs` retrofit to `MetaBridge` (zero behavior change) | §14.5 |
| **LF-3 / DM-7** | `JwtMiddleware + ActorContext + RlsRewriter::rewrite(LogicalPlan, &ActorContext)` — row-level Ueberweisung filter wired into `medcare-server` | foundry-roadmap §2 |

### 1.2 What PR-B (MedCare-rs#112) actually shipped

| Crate | File | What it provides | Gap vs §14 |
|---|---|---|---|
| `medcare-rbac` | `role.rs` | 4 roles: `doctor`, `auditor`, `receptionist`, `admin` with `PermissionSpec` per entity type (§73 SGB V shaped) | Missing: `physician`, `nurse`, `researcher`, `hipaa_audit`, `cashier` as per super-domain §4.3 naming; existing roles are medcare-server–scoped, not `RoleGroup` DTOs in `lance-graph-contract::rbac` |
| `medcare-rbac` | `permission.rs` | `PermissionSpec` with readable/writable predicates + actions + `PrefetchDepth` | Missing: `FieldRedactionMask` (`BitSet256` slot-level bitmask); `ClearanceLevel`; `audit_required` flag |
| `medcare-rbac` | `policy.rs` | `Policy::evaluate()` + `Operation` enum + `medcare_policy()` factory | `UnifiedBridge` wiring still uses `smb_policy()` placeholder (explicit TODO in `unified_bridge_wiring.rs`) |
| `medcare-rbac` | `access.rs` | `AccessDecision` (Allow/Deny/Escalate) + BtM escalate test | BtM row-context not modelled at gate; dual-control carry-forward |
| `medcare-realtime` | `stack.rs` | `MedCareStack` + `domain_profile()` (delegates to `StepDomain::Medcare.profile()`) | Empty struct: no RLS registry, no `medcare_ontology()` factory, no gate composition; blocked on DM-7/DM-8 |
| `medcare-realtime` | `gate.rs` | `MedCareMembraneGate` + `AllowAllGate`; `from_medcare_policy()` constructor | No `AuditEntry` emission; no `super_domain` lookup (would return `Unknown`); no hard-lock check |
| `medcare-realtime` | `tests/regulatory.rs` | §73 SGB V + BMV-Ä §57 + BtM escalate path tests (gate-layer only) | Row-level Ueberweisung explicitly documented as carry-forward; BtM dual-control documented as carry-forward |
| `medcare-analytics` | `unified_bridge_wiring.rs` | `medcare_unified_bridge()` constructor + `medcare_policy_placeholder()` | Placeholder explicitly wires `smb_policy()`; stringly-typed `authorize_read()` (drops after D-SDR-2/3 land OwlIdentity) |
| `medcare-analytics` | `rls_policies.rs` | `medcare_rls_registry()` with `praxis_id` tenant discriminant; sealed-mode default | Not wired into `MedCareStack`; `medcare-server` wiring pending (F2-B) |
| `medcare-analytics` | `ontology.rs` | Phase-1 stubs: `Node`/`Edge`/`NodeKind`/`EdgeKind` | No OGIT basin cross-walk, no ICD/SNOMED resolution, no vector similarity |
| `medcare-analytics` | `soa_mapping.rs`, `column_mask_bridge.rs` | SoA projection + `SensitivityReason → RedactionMode` mapping | `ColumnMaskRewriter` wired for plan-rewrite layer only; no full OwlIdentity slot-level bitmask |
| `lance-graph` #364 | `super_domain.rs:315` | `FAMILY_TO_SUPER_DOMAIN: [SuperDomain::Unknown; 256]` static | All entries remain `Unknown` — Healthcare family IDs not mapped. Regression test at line 667 proves this. See `td-sdr-family-hydration.md`. |

---

## 2 — Finalisation Items (Concrete Deliverables)

Ordered by unblock dependency. Items 1-3 are independently parallelisable after W3 (pr-d4-family-hydration.md) lands.

### E1-1 — Wire real medcare `Policy` into `UnifiedBridge`

**Repo:** MedCare-rs | **File:** `crates/medcare-analytics/src/unified_bridge_wiring.rs` + new `crates/medcare-rbac/src/super_domain_roles.rs`

**What:** Replace `medcare_policy_placeholder()` (which returns `smb_policy()`) with a proper `medcare_healthcare_policy()` factory that instantiates the 6 clinical `RoleGroup` entries from `super-domain-rbac-tenancy-v1.md §4.3`: `physician`, `nurse`, `cashier`, `researcher`, `hipaa_audit`, `admin`. Each role needs `PermissionSet` bits + `FieldRedactionMask` (3 × `BitSet256`) + `ClearanceLevel` + `audit_required` flag.

**Wiring change in `unified_bridge_wiring.rs`:**
- Drop `use lance_graph_rbac::policy::smb_policy;`
- Import `medcare_healthcare_policy()` from the new module
- `medcare_policy_placeholder()` replaced; stringly-typed `actor_role: &str` retained until D-SDR-2 lands `OwlIdentity` (per inline comment)

**Dependencies:** D-SDR-2 (RoleGroup DTOs in lance-graph-contract — already shipped per #364 D-SDR-1/2). No upstream blocker.

**Estimated LOC:** ~180 LOC in MedCare-rs (`super_domain_roles.rs` ~140 + wiring delta ~40) + 10 unit tests covering each role × entity permission path.

**DELTA vs §14:** Extends D-SDR-2 (shipped in lance-graph #364) to the medcare consumer. §14 listed this as D-SDR-21 retrofit; concretised here with explicit implementation path.

### E1-2 — Seed `OgitFamilyTable` with Healthcare-partition basins

**Repo:** lance-graph | **File:** `crates/lance-graph-ontology/src/namespace_registry.rs` (extend `seed_defaults()`) + `crates/lance-graph-ontology/src/super_domain.rs` (extend `FAMILY_TO_SUPER_DOMAIN` bake)

**What:** The 10 OGIT basins that constitute `SuperDomain::Healthcare` need `OgitFamilyTable` entries seeded. Each basin maps one `OgitFamily` byte to a block of `FamilyEntry` slots.

**Basin → OgitFamily allocation (proposed):**

| Basin | `OgitFamily` byte | OGIT namespace | Example slots |
|---|---|---|---|
| FMA | 0x10 | `ogit.FMA:*` | AnatomicalStructure, Region, Organ (IDs 0x10XX) |
| SNOMED | 0x11 | `ogit.SNOMED:*` | ClinicalFinding, Procedure, Substance |
| ICD10 | 0x12 | `ogit.ICD10:*` | DiagnosticCategory, Block, Code |
| RxNorm | 0x13 | `ogit.RxNorm:*` | ClinicalDrug, Ingredient, DoseForm |
| LOINC | 0x14 | `ogit.LOINC:*` | LabObservation, VitalSign, ClinicalDocument |
| MONDO | 0x15 | `ogit.MONDO:*` | Disease, SynonymousDisease |
| HPO | 0x16 | `ogit.HPO:*` | PhenotypicAbnormality, ClinicalModifier |
| DRON | 0x17 | `ogit.DRON:*` | Drug, DrugProduct, ActiveIngredient |
| CHEBI | 0x18 | `ogit.CHEBI:*` | ChemicalEntity, Molecular function |
| RadLex | 0x19 | `ogit.RadLex:*` | AnatomicLocation, ImagingFinding |

**`seed_defaults()` extension:** each basin adds one entry to `FAMILY_TO_SUPER_DOMAIN[family_byte] = SuperDomain::Healthcare` and registers the `OgitFamilyTable` with starter `FamilyEntry` stubs (label_uri + kind; axiom_blob deferred).

**Dependencies:** BLOCKED on `pr-d4-family-hydration.md` (W3) — the TOML seed file is the data source for these basin → super_domain mappings. Cannot ship without the hydration pipeline from that spec.

**Estimated LOC:** ~220 LOC in lance-graph-ontology (10 tables × ~18 LOC each for starter entries + `seed_defaults()` extension ~40 LOC) + 6 integration tests verifying `FAMILY_TO_SUPER_DOMAIN[0x10..=0x19] == SuperDomain::Healthcare`.

**DELTA vs §14:** D-SDR-3 is marked "shipped" in #364 for the table shape; this item fills the Healthcare-specific data that #364 left as stubs. New sub-item (not in original D-SDR-3 scope which was schema-only).

### E1-3 — `MedCareStack` composition: wire RLS registry + gate + ontology factory

**Repo:** MedCare-rs | **File:** `crates/medcare-realtime/src/stack.rs`

**What:** `MedCareStack` is currently an empty struct with `new()` + `domain_profile()`. Compose it with:

1. `Arc<RlsPolicyRegistry>` — from `medcare-analytics::rls_policies::medcare_rls_registry()`. Wire `stack.rls_registry()` accessor.
2. `Arc<medcare_rbac::Policy>` — from `medcare_healthcare_policy()` (E1-1). Wire `stack.policy()` accessor.
3. `Arc<MedCareMembraneGate>` — built from (2). Wire `stack.gate()` accessor.
4. `medcare_ontology_factory()` stub — returns `OntologyRegistry` seeded with Healthcare basins (E1-2). Wire `stack.ontology_registry()` accessor (lazy `OnceLock`).

**Blockers:** E1-1 (clinical policy) + E1-2 (seeded ontology) must land first. DM-7 (upstream `RlsRewriter` wiring in `medcare-server`) is a separate downstream blocker — this item composes the stack, not the server-level wiring.

**Estimated LOC:** ~130 LOC (`MedCareStack` fields + constructors + 4 accessors) + 4 unit tests verifying each accessor returns non-None.

**DELTA vs §14:** New item — §14 did not specify `MedCareStack` composition explicitly. Extension of D-SDR-21 retrofit scope.

### E1-4 — Audit chain integration: `AuditEntry` emission from `MedCareMembraneGate`

**Repo:** MedCare-rs + lance-graph | **Files:** `crates/medcare-realtime/src/gate.rs` + `crates/lance-graph-callcenter/src/audit.rs`

**What:** Every `authorize()` call through the gate must emit an `AuditEntry` per D-SDR-14. The updated shape (§13.3) adds `MerkleRoot + ClamPath + super_domain_salt` fields. The `super_domain` field on the entry must resolve to `Healthcare` (not `Unknown`) — which requires E1-2 (FAMILY_TO_SUPER_DOMAIN seeded) to be in place.

**Gate change:** `MedCareMembraneGate::evaluate()` (or `should_emit()`) must accept an `AuditSink` reference and emit after each Allow/Deny decision. HIPAA §164.312(b) requires logging all access attempts, not just successful ones.

**Emit shape:**
```rust
AuditEntry {
    tenant:            TenantId(praxis_id),
    super_domain:      SuperDomain::Healthcare,  // from FAMILY_TO_SUPER_DOMAIN[owl.family()]
    actor_role:        "physician",               // resolved role name
    owl:               OwlIdentity::new(OgitFamily(0x11), slot),  // SNOMED example
    op:                PermissionSet::READ,
    merkle_root:       MerkleRoot::from_fingerprint(&row_fp),
    clam_path:         ClamPath { path: "healthcare/patient/...".into(), depth: 3 },
    timestamp:         unix_ms(),
    super_domain_salt: SUPER_DOMAINS[1].merkle_salt,
}
```

**Dependencies:** E1-2 (FAMILY_TO_SUPER_DOMAIN seeded); D-SDR-14 (AuditEntry schema — shipped as part of D-SDR-4 in #364).

**Estimated LOC:** ~160 LOC (`gate.rs` audit emission ~80 + `AuditEntry::healthcare_entry()` constructor ~40 + `MerkleRoot::from_fingerprint()` wire ~40) + 8 integration tests including HIPAA tamper-detection replay.

**DELTA vs §14:** D-SDR-14 is listed as a new item in §13.8; this concretises the MedCare-side wiring (where and how the gate emits). Extending — not replacing — the existing `AuditChain.super_domain()` call (Codex P2 fix in #364).

### E1-5 — Hard-lock enforcement: Healthcare ↔ OSINT predicate-time barrier

**Repo:** lance-graph | **File:** `crates/lance-graph-callcenter/src/unified_bridge.rs` (or `super_domain.rs`)

**What:** D-SDR-17 from §13.8: a static `HARD_LOCK_MATRIX: [(SuperDomain, SuperDomain); 4]` table + check in `authorize()` at stage 2 (super-domain resolution). If the caller's `super_domain` is in the target's `hard_lock_partners`, return `RbacError::HardLockViolation` before any other check.

**Hard-lock pairs (initial matrix):**
```rust
const HARD_LOCK_MATRIX: &[(SuperDomain, SuperDomain)] = &[
    (SuperDomain::Healthcare, SuperDomain::OSINT),
    (SuperDomain::OSINT, SuperDomain::Healthcare),
    (SuperDomain::WorkOrderBilling, SuperDomain::OSINT),
    (SuperDomain::OSINT, SuperDomain::WorkOrderBilling),
];
```

**Dependencies:** D-SDR-17 design is self-contained; no upstream blockers beyond the existing `SuperDomain` enum (shipped in #364).

**Estimated LOC:** ~60 LOC (static matrix + 4-line check in authorize() + `RbacError::HardLockViolation` variant) + 4 tests covering each documented pair.

**DELTA vs §14:** D-SDR-17 is a §13.8 addition (new, not extending a prior D-SDR). The MedCare-specific requirement is that `Healthcare ↔ OSINT` is the primary enforced pair. The OSINT `↔ WorkOrderBilling` pair is a secondary financial-confidentiality requirement.

### E1-6 — Row-level `domain_profile()` completeness: `MedCareStack` DM-7 wiring stub

**Repo:** MedCare-rs | **File:** `crates/medcare-server/src/state.rs` (or new `crates/medcare-server/src/rls_middleware.rs`)

**What:** `MedCareStack::domain_profile()` is functional but the downstream consumers (medcare-server request handlers) do not yet receive an `ActorContext` from JWT middleware or wire the `RlsRewriter` into their DataFusion `SessionContext`. This item creates the server-side stub so DM-7 (upstream lance-graph) has a landing target:

1. `MedCareActorContext` newtype wrapping `praxis_id: u32` + `role: &str` + `tenant_id: TenantId` — extracted from JWT claims.
2. `medcare_rls_middleware()` axum layer that extracts context + calls `stack.rls_registry().build_context(actor)` → `RlsContext` stored in request extensions.
3. Integration test (no real DataFusion yet): verify middleware correctly rejects requests with missing `praxis_id` in JWT.

**Dependencies:** BLOCKED on DM-7 upstream in lance-graph. This item authors the stub/skeleton that DM-7 will fill in.

**Estimated LOC:** ~150 LOC (middleware + `MedCareActorContext` + JWT extraction + 3 integration tests).

**DELTA vs §14:** Extending the foundry-roadmap `F5 (RBAC: doctor/nurse/admin/patient)` stage. Not listed in §14 directly; required by the foundry-roadmap §6 F5 stage dependency on LF-3.

---

## 3 — Estimated LOC per Deliverable

| Item | ID | Files changed | Estimated LOC | Tests |
|---|---|---|---|---|
| Wire real medcare Policy into UnifiedBridge | E1-1 | `unified_bridge_wiring.rs` + `super_domain_roles.rs` (new) | ~180 | 10 |
| Seed OgitFamilyTable Healthcare basins | E1-2 | `namespace_registry.rs` + `super_domain.rs` | ~220 | 6 |
| MedCareStack composition | E1-3 | `stack.rs` | ~130 | 4 |
| Audit chain integration | E1-4 | `gate.rs` + `audit.rs` | ~160 | 8 |
| Hard-lock enforcement | E1-5 | `unified_bridge.rs` or `super_domain.rs` | ~60 | 4 |
| DM-7 wiring stub | E1-6 | `rls_middleware.rs` (new) + `state.rs` | ~150 | 3 |
| **Total** | | **7 files** | **~900 LOC** | **35 tests** |

For comparison: PR-B shipped +2963 LOC across 17 files. PR-E1 closes the finalisation gap at roughly 30% of PR-B's size — a finalisation PR, not a greenfield sprint.

---

## 4 — Dependencies on Sprint-5 W3: Family-Hydration Spec

**Canonical spec file:** `.claude/specs/pr-d4-family-hydration.md`
**Tech-debt precursor:** `.claude/specs/td-sdr-family-hydration.md` (documents the bug: `FAMILY_TO_SUPER_DOMAIN` is all-Unknown; regression test at `super_domain.rs:667` proves it).

The family-hydration spec (sprint-5 W9, mapped to row 3 in SPRINT_LOG.md) authors the TOML seed file + `UnifiedBridge::new_hydrated(config: BridgeConfig)` constructor + Lance overlay + HTTP overlay layers. Every item in this PR-E1 spec that touches `FAMILY_TO_SUPER_DOMAIN` or `OgitFamilyTable` lookup correctness **requires the hydration pipeline from that spec to already be merged**.

**Blocking dependency map:**

| PR-E1 item | Blocked on pr-d4-family-hydration.md | Reason |
|---|---|---|
| E1-1 (wire clinical policy) | No | Role group wiring is independent of super-domain lookup correctness |
| E1-2 (seed Healthcare basins) | **YES — hard block** | Basin → super-domain entries are the TOML seed rows; no pipeline = no seeding |
| E1-3 (MedCareStack composition) | Yes (via E1-2) | `medcare_ontology_factory()` needs hydrated basins to be meaningful |
| E1-4 (audit chain) | **YES — hard block** | `super_domain` in `AuditEntry` resolves via `FAMILY_TO_SUPER_DOMAIN`; all-Unknown = wrong HIPAA log |
| E1-5 (hard-lock) | No | Hard-lock matrix is a static constant; no runtime lookup required |
| E1-6 (DM-7 stub) | No | JWT middleware extraction is upstream-agnostic |

Therefore the recommended merge order is: `pr-d4-family-hydration.md` PR → E1-2 → E1-3 → E1-4 (parallel after hydration); E1-1 + E1-5 + E1-6 can merge independently at any time.

---

## 5 — Open Questions

### OQ-1 — TTL namespace shape for MedCare Healthcare partition

The 10 Healthcare basins listed in E1-2 are inferred from the super-domain spec (§3.4 enumerates them by name) and standard biomedical ontology coverage. However, the exact OGIT TTL namespace URIs for each basin are not yet authored (D-SDR-6 scope in the original plan was Hiro + HubSpot, not Healthcare).

**Concrete question:** Which OGIT fork TTL files provide the Healthcare namespace declarations? The `medcare_bridge.rs` harvest revealed `UnknownNamespace("Healthcare")` as a live error — meaning no `healthcare.ttl` or equivalent exists in the OGIT fork NTO tree yet.

**Blocker status:** OQ-1 blocks E1-2 at the TTL level (the `seed_defaults()` extension needs the OGIT namespace URIs to generate correct `NamespaceId` assignments). Short-term workaround: register Healthcare basins by programmatic `OgitFamily` byte allocation (bypassing TTL — acceptable for v1 seeding, deferred TTL authoring tracks as tech debt).

**Resolution path:** One OGIT-fork PR to add `OGIT/NTO/Healthcare/{FMA,SNOMED,ICD10,RxNorm,LOINC,MONDO,HPO,DRON,CHEBI,RadLex}.ttl` stub files — ~10 stub TTL files × 30-50 lines each. Analogous to the planned D-SDR-6/7 (Hiro/HubSpot) OGIT-fork PRs.

### OQ-2 — SGB V / BMV-Ä mapping into `SuperDomain::Healthcare` partition

§73 SGB V governs cross-doctor patient visibility (Ueberweisung-an-Facharzt referral gating). BMV-Ä §57 governs retention (10-year medical records). Both are German-law rules that must map into the super-domain enforcement surface.

**Open questions:**
- Does the `SuperDomainEntry::compliance` field need a `BMV_Ae_SGB_V` variant alongside `HIPAA`? Or does HIPAA serve as the universal floor with German-law overrides managed at the runtime membrane level (as the `MedCareStack::domain_profile()` doc comment suggests)?
- The `audit_retention_days` field is currently hardcoded as 2190 (HIPAA floor). BMV-Ä §57 requires 3650 (10 years). Should the `SuperDomainEntry` for Healthcare carry both, with the membrane picking the stricter floor for German praxis tenants?
- The Ueberweisung-an-Facharzt referral gating is a **per-row** rule (documented as carry-forward in `regulatory.rs`). Does it belong at the super-domain boundary (as a `FederationPolicy`-style gate), or is it purely an RLS filter pushed down from the gate?

**Current position (from `regulatory.rs` doc comment):** Gate-layer grants Doctor full Patient read; row-level Ueberweisung filter enforced by RLS rewriter (DM-7). This is correct architectural separation — but the spec needs a concrete answer on the retention-days question before E1-3 hardens `MedCareStack`.

### OQ-3 — `researcher` role k-anonymity floor and DP epsilon for Healthcare

§13.5 specifies k=5 as the default k-anonymity floor with a per-super-domain override. For Healthcare, k=10 is mentioned as typical for rare-condition research. The `dp_epsilon` (differential-privacy noise) for `SuperDomain::Healthcare` needs a statistician-level review.

**Open question:** What are the concrete `dp_epsilon` and `k_floor` defaults to hardcode in the Healthcare `SuperDomainEntry`? Healthcare typically uses ε ∈ [0.5, 2.0]; too small = excessive utility loss, too large = re-identification risk. This is a regulatory/clinical decision, not a software decision, but it blocks D-SDR-15 (researcher DP) from compiling correct defaults.

### OQ-4 — OwlIdentity u16 namespace for medcare-rbac roles vs lance-graph-contract::rbac RoleGroup

PR-B shipped `medcare-rbac` with its own `Role` / `Policy` types (mirroring lance-graph-rbac shape). The super-domain spec targets `lance-graph-contract::rbac::RoleGroup` as the canonical type (D-SDR-2). There is now a dual-type situation:

- `medcare_rbac::role::Role` (ships in PR-B, medcare-rs scoped)
- `lance_graph_contract::rbac::RoleGroup` (ships in #364, workspace-canonical)

**Open question:** Is E1-1 a migration (rename medcare-rbac `Role` → `RoleGroup`, adopt the contract type) or a bridge (keep both, with `Role` being a medcare-specific layer above `RoleGroup`)? The `unified_bridge_wiring.rs` inline comment suggests the stringly-typed interface will eventually drop; adopting `RoleGroup` directly removes one adapter layer.

---

## 6 — DELTA vs `super-domain-rbac-tenancy-v1.md` §14

This section classifies each PR-E1 item as either **extending** an existing §14 sub-item or **new** (not present in §14).

| PR-E1 Item | §14 Classification | Cite |
|---|---|---|
| **E1-1** — wire clinical `Policy` into `UnifiedBridge` | **Extending** D-SDR-21 (retrofit MedCare-rs to MetaBridge). §14 specified the retrofit as zero-behavior-change; E1-1 is the prerequisite step that replaces the SMB placeholder with real clinical roles — a behavior change. | §14.5, D-SDR-21 |
| **E1-2** — seed `OgitFamilyTable` Healthcare basins | **New** — D-SDR-3 in §8 covered the table schema; no Healthcare-specific seeding was in scope. Basin byte allocations (0x10–0x19) are new. | §8 Tier A D-SDR-3 |
| **E1-3** — `MedCareStack` composition | **New** — §14 did not address `MedCareStack` internal composition. It follows from D-SDR-21 (retrofit) but is a concrete wiring step not enumerated in §14. | §14.5 |
| **E1-4** — audit chain integration | **Extending** D-SDR-14 (§13.8). D-SDR-14 specifies the `AuditEntry` schema; E1-4 is the MedCare-rs–side wiring (gate emission). Extends — the schema delivery is upstream (lance-graph), the consumer wiring is here. | §13.3, §13.8 D-SDR-14 |
| **E1-5** — hard-lock Healthcare ↔ OSINT | **Extending** D-SDR-17 (§13.8). D-SDR-17 specifies the predicate-time enforcement in `authorize()`; E1-5 is the concrete implementation (static matrix + check). Direct extension. | §13.4, §13.8 D-SDR-17 |
| **E1-6** — DM-7 wiring stub | **New** — §14 does not address server-side JWT middleware. Derived from foundry-roadmap §2 (LF-3 / DM-7 dependency) and §6 (F5 stage). | foundry-roadmap §2, §6 F5 |

**Summary:** 2 items extend §14 sub-items (E1-4, E1-5). 2 items extend in a widened scope (E1-1, E1-3). 2 items are new (E1-2, E1-6). None of the items duplicate or contradict §14 — they fill gaps that §14 left as "MedCare-rs retrofit" without specifying the implementation steps.

---

## 7 — Carry-Forward (Explicitly Out of Scope for PR-E1)

These items are documented as NOT in PR-E1 to prevent scope creep. Each has its own D-SDR or carry-forward reference.

| Item | Deferred to | Reference |
|---|---|---|
| BtM (Betäubungsmittel) dual-control — Doctor.Prescription.issue with `btm_flag=true` → Escalate | Requires `MedCareMembraneGate` row-context extension (gate v2) | `regulatory.rs` doc comment |
| GDPR Art.17 anonymize/merge Escalate | Same row-context gate extension | `regulatory.rs` doc comment |
| `researcher` role DP noise injection (D-SDR-15) | After OQ-3 resolved (ε defaults) | §13.5, §13.8 D-SDR-15 |
| `EncryptedViewAggregate` federation path (D-SDR-16) | Phase 2-3 | §13.2, §13.8 D-SDR-16 |
| `medcare-analytics::ontology` vector similarity (Phase 2) | Separate ontology-similarity PR | `ontology.rs` Phase 2 note |
| MedCareV2 C# alignment via Arrow Flight SQL (D-SDR-23) | Phase 4 cutover | §14.5, §17 |
| OGIT fork TTL authoring for Healthcare basins | Separate OGIT-fork PR (per OQ-1) | D-SDR-6 scope extension |

---

## 8 — Summary

PR-B (MedCare-rs#112) shipped the substrate wiring: `UnifiedBridge<MedcareBridge>` connects, `MedCareMembraneGate` implements the orphan-rule bridge, `medcare-rbac` provides a §73 SGB V–shaped role set, and `medcare-realtime` gives the gate + stack skeleton. What remains is:

1. **Clinical role groups wired** (E1-1) — replacing the SMB placeholder with real physician/nurse/researcher roles.
2. **Healthcare family table seeded** (E1-2, hard-blocked on W3 `pr-d4-family-hydration.md`) — 10 basins baked into `FAMILY_TO_SUPER_DOMAIN`.
3. **Stack composed** (E1-3) — `MedCareStack` grows from empty marker to composed facade.
4. **Audit chain live** (E1-4) — every clinical access emits a HIPAA-complete `AuditEntry` with merkle fingerprint.
5. **Hard-lock enforced** (E1-5) — Healthcare ↔ OSINT predicate-time barrier active.
6. **DM-7 stub ready** (E1-6) — server-side context extraction stub for when the upstream lance-graph RLS rewriter lands.

Total: ~900 LOC, 35 tests, 2 repos (lance-graph + MedCare-rs). Merge order: hydration PR first, then E1-1/E1-5/E1-6 in parallel, then E1-2/E1-4, then E1-3.
