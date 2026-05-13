# PR-E3 — woa-rs Extract: Work-Order-Agent as Super-Domain Subcrate

> **Author:** W8 (claude-sonnet-4-6), sprint-log-5-6, 2026-05-13
> **Sprint slot:** S6-W4
> **Repo target:** AdaWorldAPI/woa-rs (new standalone crate; analogous to MedCare-rs#112)
> **Size estimate:** ~820 LOC net (Phases A+B+C, excluding ~90 LOC tests)
> **Prior plan extended:** `super-domain-rbac-tenancy-v1.md` §14 (D-SDR-18, D-SDR-19, harvest/retrofit for woa_bridge.rs)
> **Follows patterns from:** MedCare-rs#112 (PR-B merged 2026-05-13) + smb-office-rs#31 (PR-C merged 2026-05-13)
> **Status:** SPEC READY — awaiting `pr-d4-family-hydration.md` (W3 sprint-5) for Phase C basin wiring

---

## 0 — What Already Exists (the baseline)

### 0.1 In lance-graph-ontology (shipped as of PR #364)

`crates/lance-graph-ontology/src/bridges/woa_bridge.rs` (~50 LOC) provides:

```rust
pub const NAMESPACE: &str = "WorkOrder";
pub struct WoaBridge { registry: Arc<OntologyRegistry>, g_lock: NamespaceId }
impl NamespaceBridge for WoaBridge { ... }
impl BridgeFromRegistry for WoaBridge { ... }
```

This is the lance-graph-ontology-side bridge. It locks to the `WorkOrder`
namespace via `NamespaceRegistry` and resolves entities like `WorkOrder`,
`Position`, `Customer` to `ogit.WorkOrder:*` URIs. The bridge is listed in
`mod.rs` alongside `MedcareBridge`, `OgitBridge`, `SharePointBridge`,
`SpearBridge`.

### 0.2 In lance-graph-callcenter (shipped as of PR #364)

`crates/lance-graph-callcenter/src/super_domain.rs`:
- `SuperDomain::WorkOrderBilling = 6` (SOX compliance, basins: `&[]` -- not yet seeded)
- `FAMILY_TO_SUPER_DOMAIN: [SuperDomain::Unknown; 256]` -- all-Unknown (same gap as Healthcare, documented in `td-sdr-family-hydration.md`)
- `UnifiedBridge::authorize_read/write` tested against `"WorkOrder"` namespace in `unified_bridge.rs` tests (lines 675, 697, 734, 765)
- `UnifiedAuditEvent` with `super_domain = SuperDomain::WorkOrderBilling` in test assertions

### 0.3 In smb-office-rs (PR-C merged 2026-05-13)

`smb-office-rs#31` wired `UnifiedBridge<OgitBridge>` with +111 LOC (one file:
`unified_bridge_wiring.rs`). The `customer-woa-bin` binary skeleton in
smb-office-rs also references WoA patterns, using `"WorkOrder"` namespace.

### 0.4 What woa-rs does NOT yet have

Based on the MedCare-rs#112 pattern (the canonical model), woa-rs is missing:
- A dedicated `woa-rbac` crate with role groups (`field_tech`, `dispatcher`,
  `accountant`, `sox_audit`, `admin`) and `PermissionSpec` per entity type
- A dedicated `woa-realtime` crate with `WoaStack` + `WoaMembraneGate`
- A dedicated `woa-analytics` crate with `UnifiedBridge<WoaBridge>` wiring,
  RLS policies, SOX compliance stubs, and SOA mapping
- SOX regulatory tests (analogous to §73 SGB V + BMV-A §57 + BtM tests in MedCare-rs)
- A `WoaBridge`-specific `OgitFamily` basin byte allocation in
  `FAMILY_TO_SUPER_DOMAIN`

PR-E3 builds this woa-rs super-domain subcrate following the exact
subcrate cascade pattern from MedCare-rs#112 (3 subcrates) and the bridge
retrofit pattern from smb-office-rs#31.

---

## 1 — Inventory: Types to Extract / Create from q2/geo Context

The `woa-rs` Work-Order-Agent domain covers IT field-service operations,
work orders, asset management, and MRO/MRP billing. These map to
`SuperDomain::WorkOrderBilling` (discriminant 6, SOX compliance).

### 1.1 Roles (WorkOrderBilling super-domain)

Analogous to Healthcare's `physician/nurse/cashier/researcher/hipaa_audit/admin`
(super-domain spec §4.3), WorkOrderBilling requires:

| Role | Permissions | Readable entity slots | Writable entity slots | Audit |
|---|---|---|---|---|
| `field_tech` | READ + WRITE (ops scope) | WorkOrder, Asset, Position, ServiceInterval | WorkOrder.status, WorkOrder.resolution_notes | yes |
| `dispatcher` | READ + WRITE (dispatch) | WorkOrder, Customer, Position, Schedule | WorkOrder.assigned_tech, WorkOrder.priority | yes |
| `accountant` | READ (billing scope) | WorkOrder.billing, Invoice, Customer | none | yes (SOX financial trail) |
| `sox_audit` | READ + EXPORT + AUDIT_BYPASS | all WoA slots | none | yes (every access) |
| `admin` | SCHEMA_VIEW | slot 0xFF (schema-only reserved) | none | no |

These 5 roles become `RoleGroup` entries in `woa-rbac::roles`.

### 1.2 Entity Permissions per Role

Each entity maps to an `OwlIdentity` slot within `OgitFamily::WorkOrder`
(family byte TBD -- see §8.1 of this spec). Initial slot allocation:

| Entity | slot (u16) | Description |
|---|---|---|
| `WorkOrder` | 0x0001 | Primary work-order ticket (status, priority, assigned_tech) |
| `Customer` | 0x0002 | Client/end-user of the work order |
| `Position` | 0x0003 | Job position / geographic location |
| `Asset` | 0x0004 | Physical asset under service (CMDB entry) |
| `Invoice` | 0x0005 | Billing invoice linked to completed work order |
| `Schedule` | 0x0006 | Dispatcher schedule entry |
| `ServiceInterval` | 0x0007 | Planned maintenance interval (MRO trigger) |
| `TechProfile` | 0x0008 | Field technician profile + certifications |

These mirror `smb_owl_ids.rs` in PR-E2 but target `ogit.WorkOrder:*` URIs
(already partially defined in OGIT/NTO/WorkOrder/ TTL, per `woa-rs#2` landing
note in PR_ARC_INVENTORY.md #354).

### 1.3 Policy Operations

`woa-rbac::policy` defines `WoaOperation`:

```rust
pub enum WoaOperation {
    ReadWorkOrder,
    UpdateWorkOrderStatus,
    AssignTechnician,
    CloseWorkOrder,
    ViewBilling,
    ExportAuditLog,
    ViewAsset,
    ScheduleIntervention,
}
```

`Policy::evaluate(actor_role, entity_type, op)` returns `AccessDecision`
(Allow / Deny / Escalate). SOX §404 requires that `ExportAuditLog` escalates
for non-`sox_audit` roles.

### 1.4 Gate Semantics

`WoaMembraneGate` (in `woa-realtime`) follows `MedCareMembraneGate` shape:

- `evaluate(role, entity_owl_id, op)` -> `AccessDecision`
- On `Allow`: emit `UnifiedAuditEvent` with `super_domain = SuperDomain::WorkOrderBilling`
- On `Deny`: emit `UnifiedAuditEvent` with `deny` flag
- `AllowAllGate` variant for tests
- SOX carry-forward: dual-approval for `CloseWorkOrder` when `billing_amount > SOX_THRESHOLD` -- deferred to gate v2

### 1.5 OgitFamily Basin Byte

`OgitFamily::WorkOrder` byte is not yet assigned in `super_domain.rs`.
The `SUPER_DOMAINS[6].basins` array is empty `&[]`. This is the primary
BLOCKER for Phase C (analogous to §8.1 in PR-E2 spec, and E1-2 in PR-E1
spec). Proposed allocation: `OgitFamily(0x05)` (decimal 5). Confirm against
`pr-d4-family-hydration.md` TOML seed before baking.

---

## 2 — Cargo Crate Layout

Decision: mirror medcare-rbac / medcare-realtime / medcare-analytics (3-subcrate
pattern). Justification:

- MedCare-rs#112 is the most recent merged example of this pattern (+2963 LOC, 17 files)
- smb-office-rs#31 shipped only `unified_bridge_wiring.rs` (one file) -- too thin to be
  the model for a full domain subcrate
- woa-rs is a distinct repo (`AdaWorldAPI/woa-rs`) already referenced in PR_ARC_INVENTORY.md
  entries for #354 (`woa-rs#2` cross-repo landing) -- it needs the same 3-crate depth

### 2.1 Target directory structure

```
AdaWorldAPI/woa-rs/
+-- Cargo.toml                          # workspace
+-- crates/
    +-- woa-rbac/                       # RBAC + roles + permissions + policy
    |   +-- Cargo.toml
    |   +-- src/
    |       +-- lib.rs
    |       +-- role.rs                 # WoaRole enum + PermissionSpec
    |       +-- permission.rs           # PermissionSet + FieldRedactionMask
    |       +-- policy.rs               # Policy::evaluate() + WoaOperation + woa_policy()
    |       +-- access.rs               # AccessDecision + SOX escalate path
    +-- woa-realtime/                   # WoaStack + WoaMembraneGate
    |   +-- Cargo.toml
    |   +-- src/
    |   |   +-- lib.rs
    |   |   +-- stack.rs               # WoaStack (fields: rls_registry, policy, gate, ontology)
    |   |   +-- gate.rs                # WoaMembraneGate + from_woa_policy()
    |   +-- tests/
    |       +-- regulatory.rs          # SOX §404 tests
    +-- woa-analytics/                  # UnifiedBridge wiring + RLS + SoA mapping
        +-- Cargo.toml
        +-- src/
            +-- lib.rs
            +-- unified_bridge_wiring.rs # woa_unified_bridge() constructor
            +-- rls_policies.rs          # woa_rls_registry() + tenant discriminant
            +-- soa_mapping.rs           # SoA projection + WoaOwlIds
            +-- ontology.rs             # Phase-1 stubs: WoaNode/Edge/NodeKind/EdgeKind
```

Note on smb-bridge vs woa-rs: smb-office-rs's `customer-woa-bin` binary uses
`"WorkOrder"` namespace today with `UnifiedBridge<OgitBridge>` (a placeholder
bridge per PR-E2 §6.4). After PR-E3 lands, `customer-woa-bin` MUST be updated
to use `UnifiedBridge<WoaBridge>` from the `woa-analytics` crate. That swap is
a single type-parameter change -- the calling interface is identical.

---

## 3 — UnifiedBridge<WoaBridge> Wiring

Follows the exact shape of `MedcareBridge` / `OgitBridge` consumers.

### 3.1 Constructor in woa-analytics

```rust
// crates/woa-analytics/src/unified_bridge_wiring.rs

use lance_graph_callcenter::unified_bridge::UnifiedBridge;
use lance_graph_ontology::bridges::WoaBridge;
use lance_graph_callcenter::super_domain::SuperDomain;

pub fn woa_unified_bridge(
    registry:   Arc<OntologyRegistry>,
    actor_role: WoaRole,       // typed enum; not &str (avoids smb-bridge design gap)
    tenant:     TenantId,
) -> Result<UnifiedBridge<WoaBridge>, BridgeError> {
    let bridge = WoaBridge::from_registry(registry)?;
    let policy = Arc::new(woa_policy(actor_role));
    UnifiedBridge::new(bridge, policy, tenant)
        .with_audit_chain(SuperDomain::WorkOrderBilling, tenant.raw(), /*sink*/ ...)
}
```

Key differences from smb's current `unified_bridge_wiring.rs`:

1. `WoaRole` typed enum from birth -- no `&'static str` actor_role to retroactively
   tighten (the lesson from PR-E2 §3.3)
2. `WoaBridge` not `OgitBridge` -- dedicated bridge, not the pass-through
3. `with_audit_chain(SuperDomain::WorkOrderBilling, ...)` wired from day 1

### 3.2 WoaStack composition (analogous to E1-3 in PR-E1)

```rust
// crates/woa-realtime/src/stack.rs

pub struct WoaStack {
    rls_registry: Arc<RlsPolicyRegistry>,
    policy:       Arc<woa_rbac::Policy>,
    gate:         Arc<WoaMembraneGate>,
    ontology:     OnceLock<Arc<OntologyRegistry>>,
}

impl WoaStack {
    pub fn new() -> Self { ... }
    pub fn rls_registry(&self) -> &Arc<RlsPolicyRegistry> { &self.rls_registry }
    pub fn policy(&self) -> &Arc<woa_rbac::Policy> { &self.policy }
    pub fn gate(&self) -> &Arc<WoaMembraneGate> { &self.gate }
    pub fn ontology_registry(&self) -> &Arc<OntologyRegistry> { /* OnceLock lazy init */ }
    pub fn domain_profile(&self) -> DomainProfile { DomainProfile::for_work_order() }
}
```

`DomainProfile::for_work_order()` delegates to `StepDomain::WorkOrder.profile()`
(analogous to `StepDomain::Medcare.profile()` in MedCare-rs).

### 3.3 Gate audit emission

`WoaMembraneGate::evaluate()` emits `UnifiedAuditEvent` after every Allow/Deny:

```rust
UnifiedAuditEvent {
    tenant:       tenant_id,
    super_domain: SuperDomain::WorkOrderBilling,
    actor_role:   role_str,
    owl:          OwlIdentity::new(OgitFamily(WOA_FAMILY_BYTE), slot),
    op:           PermissionSet::READ | PermissionSet::WRITE,
    timestamp:    unix_ms(),
}
```

The `AuditChain.super_domain()` call (Codex P2 fix from #364) correctly
resolves `WorkOrderBilling` once `FAMILY_TO_SUPER_DOMAIN[WOA_FAMILY_BYTE]`
is seeded (Phase C dependency).

---

## 4 — Super-Domain Assignment

`SuperDomain::WorkOrderBilling` -- already present at discriminant 6 in
`lance-graph-callcenter::super_domain.rs`. No new variant needed.

The compliance regime is `ComplianceRegime::Sox` (SOX §404 internal controls).
WorkOrderBilling covers: IT work orders, field service ops, MRO/MRP billing,
asset management. Cross-cuts with `Networking` for WoA route-handler paths
(per PR-E2 §2.1) -- those paths stamp `SuperDomain::Networking`, which is
not yet in the enum (open blocker, same as PR-E2 §8.1). Networking variant
addition is a separate 2-line PR and does not block PR-E3 Phases A+B.

### 4.1 Basin assignment (OgitFamily)

| Basin name | Proposed `OgitFamily` byte | OGIT namespace prefix | Example entities |
|---|---|---|---|
| WorkOrder | `0x05` | `ogit.WorkOrder:*` | Order, Customer, Position, Asset, Invoice |

A single basin covers the WoA domain. Unlike Healthcare (10 basins), WorkOrder
is a single coherent vocabulary. Expansion to sub-basins (e.g., `OgitFamily::MroPlanning`)
can happen in a later PR if MRO/MRP grows distinct enough.

`SUPER_DOMAINS[6].basins` expands to `&[OgitFamily(0x05)]` once Phase C lands.

---

## 5 — Migration: How Existing q2/geo Consumers Change Post-Extraction

### 5.1 smb-office-rs customer-woa-bin

The binary today uses `UnifiedBridge<OgitBridge>` (pass-through). After PR-E3:

```rust
// Before (smb-office-rs/crates/customer-woa-bin/src/main.rs, Batch C):
let bridge = smb_unified_bridge(registry, "WorkOrder", SmbRole::Accountant, tenant)?;

// After PR-E3 (same file, single type-param swap):
use woa_analytics::woa_unified_bridge;
let bridge = woa_unified_bridge(registry, WoaRole::Accountant, tenant)?;
```

One `use` statement change + one call site update. Zero behavior change at
runtime. `UnifiedBridge<WoaBridge>` and `UnifiedBridge<OgitBridge>` expose
the same `authorize_read/write` surface -- the type parameter is encapsulated.

### 5.2 lance-graph-ontology woa_bridge.rs

`crates/lance-graph-ontology/src/bridges/woa_bridge.rs` stays UNCHANGED.
PR-E3 adds a dependency from `woa-analytics` onto `lance-graph-ontology`
(for `WoaBridge`) -- the bridge itself does not move. The compile-only
`_compile_check` function in `woa_bridge.rs` continues to compile from the
lance-graph side.

### 5.3 lance-graph-callcenter unified_bridge.rs tests

Tests in `unified_bridge.rs` use `"WorkOrder"` namespace with a test
`OntologyRegistry`. No changes required -- these tests exercise the
`UnifiedBridge` substrate, not the woa-specific layer.

### 5.4 OGIT NTO/WorkOrder TTL

Per PR_ARC_INVENTORY.md entry #354: `woa-rs#2` landing added 24 predicate
fills to `NTO/WorkOrder/{Order,Customer,Article}.ttl`. These TTL files are
the authoritative source for the `WoaBridge` namespace resolution. PR-E3
does not change them; it consumes them via `OntologyRegistry::hydrate_from_ttl`.

### 5.5 Future hiro-rs / hubspot-rs consumers

super-domain-rbac-tenancy-v1.md §14 includes `hubspot-rs` (NEW) as cross-cutting
`TicketTool + WorkOrderBilling`. After PR-E3 ships `woa_unified_bridge()` as the
canonical WorkOrderBilling constructor, hubspot-rs inherits this shape for its
billing-side wiring.

---

## 6 — Gap Analysis: §14 Expected woa-rs Surface vs Current Substrate

### 6.1 What §14 prescribes (woa-rs specific surface)

| §14 Item | Requirement | Source |
|---|---|---|
| **D-SDR-18** | Archaeology pass: `git log -p` `woa_bridge.rs`, extract fix-commits as named tests in `meta_bridge::tests` | §14.1, §14.4 |
| **D-SDR-19** | `MetaBridge` trait + `BridgeFromRegistry` extension | §14.2 |
| **§14.2 template** | `woa_bridge.rs` retrofit to meta-bridge surface (~45 LOC after MetaBridge extraction) | §14.2, §14.4 |
| **§4.2 consumer** | `woa-rs`: WorkOrderBilling, WorkOrder basin, SOX compliance | §4.2 table |
| **implicit** | `UnifiedBridge<WoaBridge>` with typed `WoaRole` roles | analogy to MedCare §14.3 |

### 6.2 What currently exists vs what is missing

| Artifact | Exists | Gap |
|---|---|---|
| `WoaBridge` in lance-graph-ontology | Yes (`woa_bridge.rs`, ~50 LOC) | Missing: role groups, typed `WoaRole` catalog |
| `SuperDomain::WorkOrderBilling` | Yes (discriminant 6) | Missing: basin seeding (`basins: &[]`) |
| `FAMILY_TO_SUPER_DOMAIN[0x05]` | No (all-Unknown) | Blocked on `pr-d4-family-hydration.md` |
| `woa-rbac` crate | No | PR-E3 Phase A creates it |
| `woa-realtime` crate | No | PR-E3 Phase B creates it |
| `woa-analytics` crate | No | PR-E3 Phase C creates it |
| SOX regulatory tests | No | PR-E3 Phase B creates them |
| `UnifiedBridge<WoaBridge>` wiring | Partial (test-only in unified_bridge.rs) | Phase C creates production wiring |
| `WoaStack` composition | No | Phase B creates it |
| `OgitFamily(0x05)` allocation | No | Blocked on `pr-d4-family-hydration.md` |

---

## 7 — Phase-by-Phase Deliverables

### Phase A -- woa-rbac (independently mergeable)

No upstream blockers.

| File | Purpose | Est. LOC |
|---|---|---|
| `crates/woa-rbac/src/role.rs` | `WoaRole` enum (5 variants) + `PermissionSpec` per entity | ~120 |
| `crates/woa-rbac/src/permission.rs` | `PermissionSet` + `FieldRedactionMask` (3xBitSet256) + `ClearanceLevel` | ~80 |
| `crates/woa-rbac/src/policy.rs` | `Policy::evaluate()` + `WoaOperation` (8 variants) + `woa_policy()` factory | ~140 |
| `crates/woa-rbac/src/access.rs` | `AccessDecision` + SOX escalate path (CloseWorkOrder gate) | ~60 |
| `crates/woa-rbac/src/lib.rs` | Module wiring | ~10 |
| Unit tests | Role x entity permission paths (5 roles x 8 entities x read/write) | ~80 |
| **Phase A total** | | **~490 LOC** |

### Phase B -- woa-realtime (depends on Phase A)

No upstream blockers beyond Phase A.

| File | Purpose | Est. LOC |
|---|---|---|
| `crates/woa-realtime/src/stack.rs` | `WoaStack` struct + 4 accessors + `domain_profile()` | ~100 |
| `crates/woa-realtime/src/gate.rs` | `WoaMembraneGate` + `from_woa_policy()` + `AllowAllGate` | ~90 |
| `crates/woa-realtime/src/lib.rs` | Module wiring | ~10 |
| `tests/regulatory.rs` | SOX §404: sox_audit bypass; CloseWorkOrder escalate; accountant billing-only; dual-approval carry-forward doc | ~80 |
| **Phase B total** | | **~280 LOC** |

### Phase C -- woa-analytics (depends on Phase B + pr-d4-family-hydration)

Blocked on: Phase B + `pr-d4-family-hydration.md` (W3 spec, family byte assignment).

| File | Purpose | Est. LOC |
|---|---|---|
| `crates/woa-analytics/src/unified_bridge_wiring.rs` | `woa_unified_bridge()` + typed `WoaRole` constructor | ~100 |
| `crates/woa-analytics/src/rls_policies.rs` | `woa_rls_registry()` + `work_order_id` tenant discriminant | ~80 |
| `crates/woa-analytics/src/soa_mapping.rs` | SoA projection + `woa_owl_id_for()` (8 entities) | ~90 |
| `crates/woa-analytics/src/ontology.rs` | Phase-1 stubs: `WoaNode`/`WoaEdge`/`NodeKind`/`EdgeKind` | ~60 |
| `crates/woa-analytics/src/lib.rs` | Module wiring | ~10 |
| Integration tests | `woa_unified_bridge` happy path; deny/allow; `UnifiedAuditEvent` round-trip | ~60 |
| **Phase C total** | | **~400 LOC** |

---

## 8 — LOC Estimate Summary

| Phase | Files | Net LOC | Tests |
|---|---|---|---|
| A -- woa-rbac | 5 | ~410 | ~80 (unit) |
| B -- woa-realtime | 4 | ~200 | ~80 (regulatory) |
| C -- woa-analytics | 5 | ~340 | ~60 (integration) |
| **Total** | **14** | **~950 LOC** | **~220 LOC tests** |

For comparison: MedCare-rs#112 shipped +2963 LOC across 17 files (greenfield
3-crate plus regulatory test suite). PR-E3 at ~950 LOC is roughly 32% of
MedCare's size, appropriate for a WoA domain that has simpler compliance
requirements (SOX vs HIPAA+SGB V+BMV-A+BtM).

---

## 9 — DELTA vs super-domain-rbac-tenancy-v1.md §14 + q2-foundry-integration-v1.md

### 9.1 vs super-domain-rbac-tenancy-v1.md §14

| §14 item | PR-E3 role | Classification |
|---|---|---|
| **D-SDR-18** (archaeology pass on `woa_bridge.rs`) | Phase A implicitly satisfies by naming existing ~50 LOC `woa_bridge.rs` patterns as harvest source. No `git log -p` needed -- bridge is short and already reviewed. `meta_bridge::tests` entry noting its shape suffices. | **Extending** (lightweight satisfy) |
| **D-SDR-19** (`MetaBridge` trait + `BridgeFromRegistry`) | `BridgeFromRegistry` already exists in `woa_bridge.rs` (impl is shipped). `MetaBridge` extraction deferred to follow-on PR (same classification as E1-1 in PR-E1 §6). | **Extending** (partial; MetaBridge extraction deferred) |
| **§14.2 template woa_bridge.rs retrofit (~45 LOC)** | `woa_bridge.rs` remains unchanged in lance-graph-ontology (see §5.2). The "retrofit" = addition of woa-rbac+woa-realtime+woa-analytics atop the existing bridge, not a rewrite. | **Extending** (interpretation: retrofit = surrounding crate build) |
| **§4.2 consumer row** (woa-rs, WorkOrderBilling, SOX, "Bridge shipped") | "Bridge shipped" refers to `WoaBridge` in lance-graph-ontology. PR-E3 ships the consumer-side crates. Upgrades status from "Bridge shipped" to "Consumer subcrates shipped". | **Fulfilling** |

### 9.2 vs q2-foundry-integration-v1.md

The q2-foundry plan treats WoA as the first tenant of the Q2 stack:

| Q2 feature | PR-E3 provides | Gap (post-PR-E3) |
|---|---|---|
| RBAC gate on endpoints (Q2-1.6) | `WoaMembraneGate::evaluate()` + `woa_policy()` | Gate exists; Q2 server Axum wiring is post-PR-E3 |
| Action Panel trigger (Q2-2.1) | `WoaOperation` enum covers `AssignTechnician`, `CloseWorkOrder` etc. | `ActionSpec` integration is Q2-Phase-2 |
| FreeEnergy-gated auto-commit (Q2-3.3) | Gate emits `AccessDecision::Allow` composable with FreeEnergy | FreeEnergy wiring is Q2-Phase-3 |
| SMB as testbed (Q2-1.5) | `WoaRole::Accountant` exercises WoA billing scope | Full Q2 testbed integration post-PR-E3 |

q2-foundry-integration-v1.md §RBAC (Q2-1.6): `Policy.evaluate()` middleware on Axum
routes. PR-E3's `WoaMembraneGate` is the Policy.evaluate() provider for WoA. Axum
wiring lands in a separate Q2 server PR.

---

## 10 — Dependencies and Blockers

### 10.1 Dependency map

```
pr-d4-family-hydration.md (S5-W3)
    |
    +-- Phase A -- woa-rbac      <- NO BLOCKER (can ship immediately)
    +-- Phase B -- woa-realtime  <- Phase A only
    |
    +-- Phase C -- woa-analytics <- Phase B + pr-d4-family-hydration.md
                                    (needs OgitFamily(0x05) assigned)
```

### 10.2 Blockers detail

| Blocker | Affects | Resolution |
|---|---|---|
| `OgitFamily(0x05)` not assigned in `super_domain.rs` | Phase C `woa_owl_id_for()` uses placeholder byte 0 | lance-graph-callcenter owner assigns byte in 2-line PR; Phase C picks up constant |
| `FAMILY_TO_SUPER_DOMAIN[0x05]` all-Unknown | Phase C audit chain stamps `Unknown` not `WorkOrderBilling` | Resolved by `pr-d4-family-hydration.md` (TOML seed) |
| `SuperDomain::Networking` not in enum | WoA route-handler paths (customer-woa-bin) cannot stamp `Networking` | Separate 2-line lance-graph PR; does not block Phases A+B+C (WoA CRUD uses `WorkOrderBilling`) |
| smb-office-rs `customer-woa-bin` bridge swap | After Phase C, binary should use `UnifiedBridge<WoaBridge>` | 1-line swap in `main.rs`; separate follow-on PR from smb-office-rs side |

---

## 11 — SOX Regulatory Tests (Phase B)

Analogous to MedCare-rs#112's §73 SGB V + BMV-A §57 + BtM tests:

| Test | What it verifies | SOX §404 control |
|---|---|---|
| `sox_audit_reads_all_slots` | `sox_audit` role grants `AccessDecision::Allow` for every entity slot | Audit completeness |
| `accountant_billing_only` | `accountant` role denied on `WorkOrder.resolution_notes` (ops field) | Segregation of duties |
| `field_tech_no_billing_access` | `field_tech` denied on `Invoice` entity | Role-scope enforcement |
| `close_workorder_escalate_over_threshold` | `CloseWorkOrder` op with billing_amount > SOX_THRESHOLD -> `Escalate` | SOX §404 dual-approval (skeleton; carry-forward for gate v2) |
| `deny_emits_audit_event` | Deny path emits `UnifiedAuditEvent` with `deny` flag | SOX §404 access logging |
| `allow_emits_workorderbilling_super_domain` | Allow path stamps `SuperDomain::WorkOrderBilling`, never `Unknown` | Domain classification correctness |

---

## 12 — Open Questions

### OQ-1 -- OgitFamily byte assignment for WorkOrder basin

Proposed `OgitFamily(0x05)`. Needs confirmation from `pr-d4-family-hydration.md`
TOML seed file before Phase C hardens the constant. If 0x05 conflicts with an
existing assignment the byte can be adjusted -- impact is one constant change.

### OQ-2 -- SOX threshold for dual-approval escalation

`close_workorder_escalate_over_threshold` test needs a concrete `SOX_THRESHOLD`
value (e.g., $10,000 USD -- typical SOX materiality floor for SMB contexts).
Placeholder: `const SOX_THRESHOLD: u64 = 10_000_00;` (in cents). Documented as
carry-forward pending compliance team input.

### OQ-3 -- woa-rs binary entry point vs customer-woa-bin in smb-office-rs

`smb-office-rs/crates/customer-woa-bin` is the current WoA binary skeleton.
After PR-E3, should the binary live:
- (A) in smb-office-rs as today, swapping to `UnifiedBridge<WoaBridge>` -- easiest path
- (B) moved to `woa-rs` itself as a `woa-server-bin` -- cleaner domain separation

Recommendation: Option A for PR-E3 (minimize scope); Option B as a follow-on
when woa-rs grows a full server stack.

### OQ-4 -- smb-bridge `smb_owl_id_for()` vs woa-analytics `woa_owl_id_for()`

PR-E2 §5.2 defines `smb_owl_id_for()` in `smb-bridge` with `SMB_FAMILY = 0`
placeholder. After PR-E3 ships `woa_owl_id_for()` in `woa-analytics` with the
real `WOA_FAMILY_BYTE`, the smb-bridge version should be removed to avoid
divergence. This is a follow-on cleanup; the two functions use different family
bytes and are in different repos, so they do not conflict in the interim.

---

## 13 — Carry-Forward (Explicitly Out of Scope for PR-E3)

| Item | Deferred to | Reference |
|---|---|---|
| SOX dual-approval real implementation (gate v2) | Requires `WoaMembraneGate` row-context extension | `close_workorder_escalate_over_threshold` test (skeleton only) |
| MRO/MRP sub-basin (`OgitFamily::MroPlanning`) | After WoA domain grows distinct enough; separate OGIT-fork TTL PR | §4.1 note |
| `SuperDomain::Networking` variant | Separate 2-line lance-graph-callcenter PR | PR-E2 §8.1; affects WoA route-handler paths only |
| woa-server binary (Axum + REST endpoints) | Separate PR; Q2-1.6 wiring | q2-foundry-integration-v1.md Phase 1 |
| Arrow Flight SQL client (MedCareV2-style drift, if WoA gets a legacy system) | Phase 4 equivalent | super-domain-rbac-tenancy-v1.md §17 |
| `woa-analytics::ontology` vector similarity (Phase 2) | Separate ontology-similarity PR | `ontology.rs` Phase-1 stubs |
| `ops_analyst` role (de-identified data, WoA analytics) | If WoA analytics use case emerges | Deferred pending customer demand |

---

## 14 — Acceptance Criteria

- [ ] `cargo check -p woa-rbac` green (zero deps beyond `lance-graph-contract`)
- [ ] `cargo check -p woa-realtime` green (depends on woa-rbac)
- [ ] `cargo check -p woa-analytics` green (depends on woa-realtime + lance-graph-ontology + lance-graph-callcenter)
- [ ] SOX regulatory tests pass: `sox_audit_reads_all_slots`, `accountant_billing_only`, `field_tech_no_billing_access`, `deny_emits_audit_event`, `allow_emits_workorderbilling_super_domain`
- [ ] `woa_unified_bridge()` accepts `WoaRole`, not `&str`
- [ ] `UnifiedAuditEvent.super_domain == SuperDomain::WorkOrderBilling` in audit emission tests
- [ ] `woa_owl_id_for()` returns `Some` for all 8 WoA entities; `None` for unknown
- [ ] Existing `woa_bridge.rs` in lance-graph-ontology unchanged (verified by `git diff`)
- [ ] `customer-woa-bin` build still passes with `UnifiedBridge<OgitBridge>` placeholder (bridge swap is post-PR-E3 follow-on)

---

## 15 — PR Sequencing

```
pr-d4-family-hydration.md (S5-W3)
          |
          +-- PR-E3 Phase A  (woa-rbac crate)              <- no blocker
          +-- PR-E3 Phase B  (woa-realtime + SOX tests)    <- Phase A only
          |
 [OgitFamily(0x05) assigned in lance-graph-callcenter super_domain.rs]
          |
          v
 PR-E3 Phase C  (woa-analytics + UnifiedBridge<WoaBridge>)
          |
          v
 smb-office-rs follow-on: customer-woa-bin bridge swap (OgitBridge -> WoaBridge)
```

---

*End of spec. Estimated ~950 LOC total (~820 net + ~130 infrastructure/test LOC).*
*Assign Phase A immediately (no blockers); Phase B after Phase A; Phase C after pr-d4-family-hydration.md lands.*
