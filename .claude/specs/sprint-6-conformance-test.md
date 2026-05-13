# Sprint-6 Cross-Crate Registry Conformance Test Spec

> **Spec-ID:** S6-W10
> **Author:** W12 (claude-sonnet-4-6), sprint-log-5-6, 2026-05-13
> **Deliverable type:** Spec-only (no code committed; engineer executes)
> **Status:** Draft — ready for engineer pickup
> **Cross-refs:**
> - `.claude/plans/super-domain-rbac-tenancy-v1.md` §14 (Harvest + MetaBridge contract)
> - `.claude/specs/td-super-domain-subcrates.md` (UnifiedBridgeImpl trait + CI gate shape from sprint-5 W4)
> - `.claude/plans/foundry-consumer-parity-v1.md` (parity matrix — §2 shared Foundry surface)
> - `.claude/board/LATEST_STATE.md` — D-SDR-5 shipped (PR #364): UnifiedBridge<B> with AuditChain, authorize_read/write/act, super_domain stamping
> - `crates/lance-graph-callcenter/src/unified_bridge.rs` (the trait + AuditChain surface)
> - `crates/lance-graph-contract/src/orchestration.rs` (OrchestrationBridge + DomainProfile)

---

## 1 — Purpose

Sprint-6 ships three super-domain consumer crates in the E-series batch:

| E-id | Crate | Bridge | SuperDomain |
|---|---|---|---|
| E1 | `medcare-rs` finalisation | `MedcareBridge` | `Healthcare = 1` |
| E2 | `smb-office-rs` retrofit | `OgitBridge` (via `callcenter::UnifiedBridge<OgitBridge>`) | `WorkOrderBilling = 6` |
| E3 | `woa-rs` extraction | `WoaBridge` | `WorkOrderBilling = 6` |
| E4 | `hiro-rs` scaffold | `HiroBridge` (stub) | TBD (reserve 9) |
| E5 | `hubspot-rs` scaffold | `HubspotBridge` (stub) | TBD (reserve 8) |

Only E1/E2/E3 are in the sprint-6 parallel batch and must have PASSING conformance tests. E4/E5 are scaffold-only with `#[ignore]` tests.

This spec defines a **cross-crate registry conformance test** that runs against every super-domain consumer and verifies they all implement the `UnifiedBridge` contract correctly — including audit emission shape, super-domain stamping, error path behaviour, and family table coverage.

The conformance test is the CI gate that prevents a consumer crate from shipping a `NamespaceBridge` impl that compiles but violates the contract semantics.

---

## 2 — Contract Assertions (what every consumer must satisfy)

Each consumer bridge `B: NamespaceBridge` wired into `UnifiedBridge<B>` must satisfy the following numbered assertions. The test harness verifies all of them.

### A1 — Audit emission shape (26-byte canonical event)

`UnifiedAuditEvent::canonical_bytes()` must return exactly 26 bytes for every event the bridge emits. The layout is:

```
[0..8]   ts_unix_ms          u64 LE
[8..12]  tenant              u32 LE  (TenantId)
[12]     super_domain        u8
[13..16] owl_identity        3 bytes [family, slot_lo, slot_hi]
[16]     op                  u8  (AuthOp: 0=Read, 1=Write, 2=Act)
[17]     decision            u8  (AuthDecision: 0=Allow, 1=Deny, 2=Escalate)
[18..26] actor_role_hash     u64 LE
```

**Test:** construct a `RecordingSink`, call `authorize_read` on a known entity, assert `events[0].canonical_bytes().len() == 26` and all byte offsets decode correctly.

### A2 — Super-domain stamping (correct field on emitted event)

Every emitted `UnifiedAuditEvent` must carry the `super_domain` that was wired into the `AuditChain` via `UnifiedBridge::with_audit_chain(super_domain, salt, sink)`. The field must NOT be `SuperDomain::Unknown` for any active consumer (E1/E2/E3).

**Test:** wire `with_audit_chain(SuperDomain::Healthcare, test_salt, recording_sink)` for `MedcareBridge`; assert `events[0].super_domain == SuperDomain::Healthcare`. Repeat for each consumer's canonical super-domain.

### A3 — Merkle chain advances across calls (tamper-detection precondition)

The `merkle_root` field on successive events must be strictly different: `events[N].merkle_root != events[N-1].merkle_root` for any sequence of two `authorize_*` calls. The genesis root must not appear as a non-first event's root.

**Test:** make three sequential `authorize_read` calls; assert all three `merkle_root` values are distinct and none equal `AuditMerkleRoot::GENESIS`.

### A4 — Bridge error short-circuits before audit

A `BridgeError` (unknown public name, out-of-scope namespace) must NOT emit an audit event. The audit chain is only advanced when the policy evaluation step is reached.

**Test:** call `authorize_read("__nonexistent_entity__", ...)` on every consumer bridge; assert the recording sink has zero events.

### A5 — Policy evaluates against canonical OGIT name, not bridge alias

When the bridge resolves a consumer-facing alias (e.g. `"WorkOrder"` to `ogit.WorkOrder:Order`), the policy must be evaluated against the canonical OGIT local name (`"Order"`), not the alias (`"WorkOrder"`). A policy keyed on the canonical name grants; a policy keyed only on the alias denies.

**Test (consumer-specific fixtures required):** per-bridge alias table (see Section 5), construct two `Policy` instances — one keyed on the canonical name, one on the alias. Assert the canonical-keyed policy grants and the alias-keyed policy denies via `authorize_read`.

### A6 — SuperDomain is not Unknown for active consumers

The `SuperDomain` wired into an active consumer's `AuditChain` must not be `SuperDomain::Unknown` (discriminant = 0). Scaffold consumers (E4/E5) are exempt.

**Test:** assert `consumer.with_audit_chain(known_super_domain, ...)` produces events where `event.super_domain != SuperDomain::Unknown`.

### A7 — Family table coverage (at least one OWL identity mapping)

Each bridge's backing `OntologyRegistry` must contain at least one `MappingRow` for the bridge's locked namespace. An empty registry means the bridge cannot resolve any entity type, making all `authorize_*` calls produce `BridgeError` — useless in production.

**Test:** seed the registry with one `MappingProposal` per consumer (see Section 5 fixtures); assert `registry.namespace_id(bridge.g_lock_namespace())` succeeds and `bridge.row(seeded_public_name)` returns `Ok(...)`.

### A8 — TenantId isolation (cross-tenant events carry distinct tenant field)

Two `UnifiedBridge` instances with different `TenantId` values must emit events with distinct `tenant` fields. No event from `TenantId(1)` should carry `tenant == TenantId(2)`.

**Test:** create two bridge instances with `TenantId(1)` and `TenantId(42)`; call `authorize_read` on each; assert emitted events carry the respective tenant ids.

### A9 — Actor role hash stability

The `actor_role_hash` field on emitted events must equal `fnv1a_str(actor_role)` computed independently. This validates the audit record is not truncated or zeroed under mutex poison recovery.

**Test:** assert `events[0].actor_role_hash == lance_graph_contract::hash::fnv1a_str("test_role")` for a bridge constructed with `actor_role = "test_role"`.

### A10 — NamespaceBridge::g_lock returns non-zero namespace id

Every active consumer bridge must lock to a non-zero `NamespaceId`. Zero is the "not found" sentinel; a bridge that returns `NamespaceId(0)` has not been initialised against a real registry.

**Test:** assert `bridge.g_lock().raw() != 0` for all E1/E2/E3 bridges after seeding the registry with their fixture data.

---

## 3 — Test Harness Shape

### 3.1 Generic conformance function

The harness is a single generic function called once per consumer. It takes a constructed, seeded `UnifiedBridge<B>` instance plus a per-consumer `ConformanceFixture`:

```rust
// crates/lance-graph-consumer-conformance/src/harness.rs

use std::sync::Arc;
use lance_graph_callcenter::{
    super_domain::SuperDomain,
    unified_audit::{AuditChain, AuditMerkleRoot, AuthDecision, AuthOp, UnifiedAuditEvent, UnifiedAuditSink},
    unified_bridge::{AuthError, TenantId, UnifiedBridge},
};
use lance_graph_ontology::bridge::NamespaceBridge;

/// Per-consumer fixture: one seeded entity name plus its expected canonical OGIT name.
pub struct ConformanceFixture {
    /// Public name the consumer bridge accepts (may differ from canonical).
    pub public_name: &'static str,
    /// Expected canonical OGIT local name (what Policy must key on).
    pub canonical_name: &'static str,
    /// SuperDomain the bridge declares (must not be Unknown for active consumers).
    pub super_domain: SuperDomain,
    /// A policy role name that has read access to `canonical_name`.
    pub role_that_can_read: &'static str,
}

/// Recording sink — captures every emitted event for assertion.
#[derive(Default)]
pub struct RecordingSink {
    pub events: std::sync::Mutex<Vec<UnifiedAuditEvent>>,
}
impl RecordingSink {
    pub fn snapshot(&self) -> Vec<UnifiedAuditEvent> {
        self.events.lock().unwrap().clone()
    }
}
impl UnifiedAuditSink for RecordingSink {
    fn emit(&self, event: &UnifiedAuditEvent) {
        self.events.lock().unwrap().push(*event);
    }
}

/// Assert all contract obligations for a consumer bridge and fixture.
/// Call this from every per-consumer #[test] function.
pub fn assert_consumer_conformance<B: NamespaceBridge>(
    bridge_allow: UnifiedBridge<B>,  // policy: role_that_can_read has access
    bridge_deny:  UnifiedBridge<B>,  // policy: role_that_can_read keyed on alias (should deny)
    bridge_blank: UnifiedBridge<B>,  // empty registry for A4 + A7 bridge-error tests
    fixture: &ConformanceFixture,
    sink_allow: Arc<RecordingSink>,
    sink_blank: Arc<RecordingSink>,
) {
    use lance_graph_contract::property::PrefetchDepth;

    // A1 + A2 + A3 + A5 + A6 + A8 + A9 — allow path
    let _ = bridge_allow.authorize_read(fixture.public_name, PrefetchDepth::Identity).expect("allow");
    let _ = bridge_allow.authorize_read(fixture.public_name, PrefetchDepth::Identity).expect("allow 2");
    let _ = bridge_allow.authorize_read(fixture.public_name, PrefetchDepth::Identity).expect("allow 3");
    let events = sink_allow.snapshot();
    assert_eq!(events.len(), 3, "A1: expect 3 audit events for 3 allow calls");

    // A1: canonical_bytes length
    for ev in &events {
        assert_eq!(ev.canonical_bytes().len(), 26, "A1: canonical_bytes must be 26 bytes");
    }

    // A2: super_domain stamping
    for ev in &events {
        assert_eq!(ev.super_domain, fixture.super_domain, "A2: super_domain mismatch");
        if fixture.super_domain != SuperDomain::Unknown {
            assert_ne!(ev.super_domain, SuperDomain::Unknown,
                "A6: active consumer must not emit Unknown super_domain");
        }
    }

    // A3: merkle chain advances
    assert_ne!(events[0].merkle_root, events[1].merkle_root,
        "A3: chain must advance between calls");
    assert_ne!(events[1].merkle_root, events[2].merkle_root,
        "A3: chain must advance between calls");
    assert_ne!(events[0].merkle_root, AuditMerkleRoot::GENESIS,
        "A3: non-genesis after first emit");

    // A8: tenant field
    for ev in &events {
        assert_eq!(ev.tenant, TenantId(1), "A8: tenant field must match bridge construction");
    }

    // A9: actor_role_hash
    let expected_hash = lance_graph_contract::hash::fnv1a_str(fixture.role_that_can_read);
    for ev in &events {
        assert_eq!(ev.actor_role_hash, expected_hash, "A9: actor_role_hash must match fnv1a(role)");
    }

    // A4: bridge error short-circuits — no audit on unknown entity
    let _ = bridge_blank.authorize_read("__nonexistent__", PrefetchDepth::Identity)
        .expect_err("expect bridge error");
    let blank_events = sink_blank.snapshot();
    assert!(blank_events.is_empty(), "A4: bridge error must not emit audit event");
}
```

### 3.2 Per-consumer test functions

Each consumer gets one `#[test]` function that builds the bridge, seeds the registry, wires the sinks, and calls `assert_consumer_conformance`. The tests are NOT in the same crate as the bridges — they live in the new `crates/lance-graph-consumer-conformance/` crate and import the bridge types as dev-dependencies.

```rust
// crates/lance-graph-consumer-conformance/src/lib.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn medcare_bridge_conforms() { ... }    // E1 — active, must pass

    #[test]
    fn smb_ogit_bridge_conforms() { ... }   // E2 — active, must pass

    #[test]
    fn woa_bridge_conforms() { ... }        // E3 — active, must pass

    #[test]
    #[ignore = "hiro-rs: stub bridge, OWL file not yet seeded (E4 scaffold)"]
    fn hiro_bridge_conforms() { ... }

    #[test]
    #[ignore = "hubspot-rs: stub bridge, OWL file not yet seeded (E5 scaffold)"]
    fn hubspot_bridge_conforms() { ... }
}
```

---

## 4 — Test Crate Location

The conformance test lives in its own dedicated crate: `crates/lance-graph-consumer-conformance/`.

### Rationale for a separate crate (not folded into an existing test crate)

| Option | Verdict | Reason |
|---|---|---|
| Fold into `lance-graph-callcenter` tests | REJECTED | Callcenter tests must not have dev-deps on consumer crates (would create a circular dep: callcenter <- consumer <- callcenter via conformance). |
| Fold into `lance-graph-ontology` tests | REJECTED | Ontology crate does not depend on callcenter (no `UnifiedBridge`, no `AuditChain`); adding that dep would violate the zero-dep discipline. |
| Fold into one consumer crate's tests | REJECTED | Conformance must run across ALL consumers; a per-crate test can only test itself. |
| **New crate `lance-graph-consumer-conformance`** | **ACCEPTED** | Clean dep graph: conformance -> callcenter + ontology + each consumer bridge. No circular deps. One place to add new consumers. CI runs it as a standalone `cargo test -p lance-graph-consumer-conformance`. |

### Cargo.toml shape

```toml
[package]
name = "lance-graph-consumer-conformance"
version = "0.1.0"
edition = "2021"
publish = false   # internal CI gate — not published to crates.io

[dev-dependencies]
lance-graph-callcenter = { path = "../lance-graph-callcenter" }
lance-graph-ontology   = { path = "../lance-graph-ontology" }
lance-graph-contract   = { path = "../lance-graph-contract" }
# E1 — medcare bridge (from lance-graph-ontology::bridges for now; extracted to medcare-rs in Phase 1)
# E2 — OgitBridge (in lance-graph-ontology::bridges::ogit_bridge)
# E3 — WoaBridge  (in lance-graph-ontology::bridges::woa_bridge)
# E4/E5 — hiro-rs / hubspot-rs (path or git deps, added when scaffolded)

[features]
consumer-conformance = []   # feature gate for CI matrix inclusion
```

Note: until medcare-rs, smb-office-rs, and woa-rs are fully extracted to their own repos (D-SDR-21/22/23), the conformance tests import the bridge types from `lance-graph-ontology::bridges::{MedcareBridge, OgitBridge, WoaBridge}`. The import path changes to the external crate in Phase 1-3 of the migration; the test logic does not change.

---

## 5 — Per-Consumer Test Fixtures

### 5.1 MedcareBridge (E1 — Healthcare)

```rust
// Fixture: one MappingProposal registering a canonical Healthcare entity
fn seed_medcare_registry() -> Arc<OntologyRegistry> {
    use lance_graph_contract::property::{Marking, Schema};
    use lance_graph_ontology::namespace::OgitUri;
    use lance_graph_ontology::proposal::{MappingProposal, MappingProposalKind};

    let registry = Arc::new(OntologyRegistry::new_in_memory());
    let uri = OgitUri::parse("ogit.Healthcare:Patient").unwrap();
    registry.append_mapping(MappingProposal {
        public_name: "Patient".to_string(),
        bridge_id: "medcare".to_string(),
        ogit_uri: uri,
        namespace: "Healthcare".to_string(),
        kind: MappingProposalKind::Entity {
            schema: Schema::builder("Patient").required("patient_id").build(),
        },
        marking: Marking::Confidential,   // HIPAA-aware marking
        confidence: 1.0,
        source_uri: "test://medcare-fixture".to_string(),
        checksum: "checksum-medcare-patient".to_string(),
        created_by: "conformance-test".to_string(),
    }).unwrap();
    registry
}

// Expected: public_name == canonical_name for Healthcare (no alias gap)
static MEDCARE_FIXTURE: ConformanceFixture = ConformanceFixture {
    public_name: "Patient",
    canonical_name: "Patient",    // ogit.Healthcare:Patient -> local = "Patient"
    super_domain: SuperDomain::Healthcare,
    role_that_can_read: "doctor",
};
```

**Policy for allow path:** `Role::new("doctor").with_permission(PermissionSpec::read_at("Patient", PrefetchDepth::Identity))`

**Policy for deny path (A5):** `Role::new("doctor").with_permission(PermissionSpec::read_at("WRONG_ALIAS", PrefetchDepth::Identity))` — tests that the policy evaluates on the canonical OGIT name, not a mis-keyed alias.

### 5.2 OgitBridge for smb-office-rs (E2 — WorkOrderBilling)

The `OgitBridge` is a pass-through bridge for callers that already speak OGIT URIs. Its `public_name` IS the canonical name (no alias translation). The conformance test verifies the audit path fires correctly with `SuperDomain::WorkOrderBilling`.

```rust
fn seed_ogit_registry_smb() -> Arc<OntologyRegistry> {
    let registry = Arc::new(OntologyRegistry::new_in_memory());
    let uri = OgitUri::parse("ogit.SMB:Invoice").unwrap();
    registry.append_mapping(MappingProposal {
        public_name: "Invoice".to_string(),
        bridge_id: "ogit".to_string(),
        ogit_uri: uri,
        namespace: "SMB".to_string(),
        kind: MappingProposalKind::Entity {
            schema: Schema::builder("Invoice").required("invoice_id").build(),
        },
        marking: Marking::Internal,
        confidence: 1.0,
        source_uri: "test://smb-fixture".to_string(),
        checksum: "checksum-smb-invoice".to_string(),
        created_by: "conformance-test".to_string(),
    }).unwrap();
    registry
}

static SMB_FIXTURE: ConformanceFixture = ConformanceFixture {
    public_name: "Invoice",
    canonical_name: "Invoice",
    super_domain: SuperDomain::WorkOrderBilling,
    role_that_can_read: "accountant",
};
```

### 5.3 WoaBridge (E3 — WorkOrderBilling)

`WoaBridge` locks to the `WorkOrder` namespace and translates consumer-side aliases like `"WorkOrder"` to `ogit.WorkOrder:Order`. This is the canonical alias-vs-canonical test case (A5):

```rust
fn seed_woa_registry() -> Arc<OntologyRegistry> {
    let registry = Arc::new(OntologyRegistry::new_in_memory());
    let uri = OgitUri::parse("ogit.WorkOrder:Order").unwrap();
    registry.append_mapping(MappingProposal {
        public_name: "WorkOrder".to_string(),   // bridge alias
        bridge_id: "woa".to_string(),
        ogit_uri: uri,                          // canonical = "Order"
        namespace: "WorkOrder".to_string(),
        kind: MappingProposalKind::Entity {
            schema: Schema::builder("Order").required("order_id").build(),
        },
        marking: Marking::Internal,
        confidence: 1.0,
        source_uri: "test://woa-fixture".to_string(),
        checksum: "checksum-woa-order".to_string(),
        created_by: "conformance-test".to_string(),
    }).unwrap();
    registry
}

static WOA_FIXTURE: ConformanceFixture = ConformanceFixture {
    public_name: "WorkOrder",       // bridge alias
    canonical_name: "Order",        // OGIT canonical local name — policy must key on this
    super_domain: SuperDomain::WorkOrderBilling,
    role_that_can_read: "dispatcher",
};
```

**A5 extra assertion for WoaBridge:** policy keyed on `"Order"` grants; policy keyed on `"WorkOrder"` denies. This is the alias-canonical decoupling that PR #364 (Codex P2 fix) hardened in `unified_bridge.rs::canonical_entity_type()`.

### 5.4 E4/E5 scaffold fixtures (stub, #[ignore])

```rust
// hiro-rs: stub bridge with empty registry — all tests #[ignore]
static HIRO_FIXTURE: ConformanceFixture = ConformanceFixture {
    public_name: "Ticket",
    canonical_name: "Ticket",
    super_domain: SuperDomain::TicketTool,  // discriminant = 5
    role_that_can_read: "agent",
};

// hubspot-rs: stub bridge with empty registry — all tests #[ignore]
static HUBSPOT_FIXTURE: ConformanceFixture = ConformanceFixture {
    public_name: "Contact",
    canonical_name: "Contact",
    super_domain: SuperDomain::Unknown,   // TBD discriminant — assigned before un-ignore
    role_that_can_read: "sales_rep",
};
```

---

## 6 — CI Integration

### 6.1 Where this fits in the sprint-5 W4 CI matrix

Sprint-5 W4's CI matrix spec (`.claude/specs/sprint-5-ci-matrix.md`) defines `rust-test.yml` job ordering. The conformance test crate slots in as follows:

```yaml
# .github/workflows/rust-test.yml  (existing file, extend this job)
- name: consumer conformance
  run: cargo test -p lance-graph-consumer-conformance --features consumer-conformance
  # Must run AFTER:
  #   - lance-graph-callcenter tests (UnifiedBridge + AuditChain)
  #   - lance-graph-ontology tests  (bridge impls)
  # May run IN PARALLEL with:
  #   - lance-graph-planner tests
  #   - lance-graph-benches
```

### 6.2 Gating policy

| Test function | CI gating | Unblock condition |
|---|---|---|
| `medcare_bridge_conforms` | Blocking — fails PR | E1 complete (medcare-rs finalised) |
| `smb_ogit_bridge_conforms` | Blocking — fails PR | E2 complete (smb-office-rs retrofit) |
| `woa_bridge_conforms` | Blocking — fails PR | E3 complete (woa-rs extraction) |
| `hiro_bridge_conforms` | Non-blocking `#[ignore]` | E4 OWL file committed to hiro-rs |
| `hubspot_bridge_conforms` | Non-blocking `#[ignore]` | E5 OWL file committed to hubspot-rs |

### 6.3 Dependency order

```
UnifiedBridgeImpl trait defined (callcenter/src/unified_bridge_impl.rs)
    |
    +-- E1: medcare-rs finalisation (D-SDR-21)
    +-- E2: smb-office-rs retrofit  (D-SDR-22)
    +-- E3: woa-rs extraction       (D-SDR-23)
           |
           +---> lance-graph-consumer-conformance CI gate
                    |
                    +---> sprint-6 merge gate (all three must be green)
```

---

## 7 — Conformance Matrix

Full matrix of consumers x assertions:

| Assertion | MedcareBridge (E1) | OgitBridge/smb (E2) | WoaBridge (E3) | HiroBridge (E4) | HubspotBridge (E5) |
|---|---|---|---|---|---|
| A1 Audit bytes = 26 | REQUIRED | REQUIRED | REQUIRED | ignore | ignore |
| A2 super_domain stamped | Healthcare=1 | WorkOrderBilling=6 | WorkOrderBilling=6 | ignore | ignore |
| A3 Merkle chain advances | REQUIRED | REQUIRED | REQUIRED | ignore | ignore |
| A4 BridgeError no audit | REQUIRED | REQUIRED | REQUIRED | ignore | ignore |
| A5 Policy on canonical, not alias | Patient=Patient (trivial) | Invoice=Invoice (trivial) | WorkOrder->Order (alias!) | ignore | ignore |
| A6 SuperDomain != Unknown | Healthcare | WorkOrderBilling | WorkOrderBilling | ignore | ignore |
| A7 Family table non-empty | Patient seeded | Invoice seeded | WorkOrder seeded | ignore (empty) | ignore (empty) |
| A8 TenantId isolation | TenantId(1) | TenantId(1) | TenantId(1) | ignore | ignore |
| A9 actor_role_hash stable | "doctor" | "accountant" | "dispatcher" | ignore | ignore |
| A10 g_lock != 0 | Healthcare ns | SMB ns | WorkOrder ns | ignore | ignore |

---

## 8 — DELTA vs foundry-consumer-parity-v1.md

`foundry-consumer-parity-v1.md` covers the **callcenter Foundry-surface contract** (its Section 2 shared Foundry surface) — which contract modules (`ontology`, `property`, `repository`, `rls`, `auth`, `a2a_blackboard`, etc.) each consumer needs, and which `LF-id` deliverables unblock them (DM-8, LF-12, LF-31, LF-90, LF-92). Its scope is "does the consumer expose the right Foundry-equivalent API surface."

This conformance spec covers the **UnifiedBridge execution contract** — does the bridge correctly implement the D-SDR-series obligations (audit emission, super-domain stamping, merkle chaining, policy-on-canonical-name, error path, tenant isolation). These are orthogonal:

| Dimension | foundry-consumer-parity-v1.md | This spec (sprint-6-conformance-test.md) |
|---|---|---|
| Level | Callcenter Tier-0/1 API surface | UnifiedBridge execution semantics |
| Verified by | Integration + route tests | `#[test]` generic conformance harness |
| Deliverables covered | DM-8, LF-12, LF-31, LF-90, LF-92 | D-SDR-3, D-SDR-4, D-SDR-5, D-SDR-18/19 |
| Key question | Does the consumer expose RLS/PostgREST/audit? | Does authorize_read emit a correct 26-byte audit event? |
| Overlap | None — different assertion targets | None |

One concrete delta: `foundry-consumer-parity-v1.md` Section 5 lists **LF-90 AuditLog** as a P-0 shared deliverable for SOC2/GDPR audit. This spec's A1/A2/A3/A9 assertions ARE the runtime enforcement gate for LF-90 — they verify the merkle-chained `UnifiedAuditEvent` is correctly formed before it reaches any Lance/JSONL sink. The parity plan says "wire it"; this spec defines "verify it was wired correctly."

Second delta: `foundry-consumer-parity-v1.md` Section 2 lists `rls` (RlsRewriter) and `auth` (ActorContext, JwtMiddleware) as Tier-0 already-shipped modules. This spec adds a compile-time enforcement angle: A5 (policy-on-canonical-name) and A10 (g_lock non-zero) catch bugs where the RLS/auth wiring exists but is semantically broken — the bridge compiles, the policy evaluates, but it evaluates on the wrong name. The parity plan cannot detect this; the conformance harness can.

---

## 9 — File Map

```
New files:
  crates/lance-graph-consumer-conformance/Cargo.toml
  crates/lance-graph-consumer-conformance/src/harness.rs   <- generic assert_consumer_conformance<B>
  crates/lance-graph-consumer-conformance/src/lib.rs       <- per-consumer #[test] functions

Modified files (engineer action required):
  crates/lance-graph-callcenter/src/unified_bridge_impl.rs  <- UnifiedBridgeImpl trait (from td-super-domain-subcrates.md Section 6)
  Cargo.toml (workspace root)                               <- add lance-graph-consumer-conformance to members
  .github/workflows/rust-test.yml                           <- add conformance step
```

---

## 10 — Acceptance Criteria

An engineer can mark S6-W10 resolved when ALL of the following are true:

- [ ] `crates/lance-graph-consumer-conformance/` exists as a workspace member.
- [ ] `medcare_bridge_conforms`, `smb_ogit_bridge_conforms`, `woa_bridge_conforms` all pass without `#[ignore]`.
- [ ] `hiro_bridge_conforms` and `hubspot_bridge_conforms` exist with `#[ignore]` and compile.
- [ ] All 10 assertions (A1-A10) are verified by the harness for E1/E2/E3.
- [ ] The `consumer-conformance` CI step runs in `rust-test.yml` after callcenter + ontology tests.
- [ ] No new circular deps introduced (conformance crate depends on consumers, not vice versa).
- [ ] WoaBridge A5 alias-vs-canonical test passes (WorkOrder alias -> Order canonical policy evaluation).
- [ ] `cargo test -p lance-graph-consumer-conformance` exits 0 in CI with no `#[ignore]` suppressions for E1/E2/E3.
