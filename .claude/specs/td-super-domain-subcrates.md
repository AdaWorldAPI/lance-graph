# TD-SUPER-DOMAIN-SUBCRATES-1 — Super-Domain Specialised Subcrates

> **TD-ID:** TD-SUPER-DOMAIN-SUBCRATES-1
> **Priority:** P1
> **Sprint:** Sprint-4 (2026-05-13)
> **Author:** W4-retry (claude-sonnet-4-6)
> **Status:** Spec ready — awaiting engineer pickup
> **Cross-refs:** W3 (column_mask_bridge deprecation), W6 (thinking-engine CognitiveStack wire-up), W8 (audit sink), W9 (family hydration)

---

## 1. Problem Statement

Consumer crates — `medcare-analytics`, `medcare-bridge`, `smb-bridge`, and future `hubspot-rs`, `hiro-rs`, `woa-rs` — are not yet **super-domain-specialised subcrates**. They do not implement the `UnifiedBridgeImpl` trait, they do not declare a `SuperDomain` constant, they have no `OwlHydrator`, no `PolicySeed`, no `FamilyTable`, and no typed `AuditSink`. As a result:

- Each consumer crate reimplements ad-hoc wiring that duplicates logic already captured in `lance-graph-callcenter::UnifiedBridge<B>`.
- The `SuperDomain` enum is unused downstream — routing always falls through to `SuperDomain::Unknown`.
- No CI gate validates that a consumer crate actually satisfies the bridge contract.
- The thinking-engine `CognitiveStack` (W6) cannot be embedded per-subcrate because there is no standardised mount point.
- The audit chain (W8) cannot be wired per-subcrate for the same reason.

---

## 2. Topology

```
+---------------------------------------------------------------------------+
|  lance-graph-contract   (canonical zero-dep types)                        |
|  -----------------------------------------------------------------------  |
|  SuperDomain(u8)   OWLIdentity   FamilyId   PolicySeed   AuditEvent       |
+----------------------------------+-----------------------------------------+
                                   |  trait bounds only
                                   v
+---------------------------------------------------------------------------+
|  lance-graph-callcenter   (orchestration & routing)                       |
|  -----------------------------------------------------------------------  |
|  UnifiedBridge<B: NamespaceBridge>                                        |
|    .with_audit_chain(super_domain, salt, sink)  <- builder                |
|    .authorize_read(identity, ctx)               <- cognitive_stack W6     |
|    .authorize_write(identity, ctx)              <- cognitive_stack W6     |
|  SuperDomain enum { Healthcare=1, ..., WorkOrderBilling=6, TicketTool=5 }|
|  UnifiedBridgeImpl trait  (the specialisation contract)                   |
+----------------------------------+-----------------------------------------+
                                   |  impl UnifiedBridgeImpl
                                   v
+---------------------------------------------------------------------------+
|  lance-graph-ontology :: bridges/   (adapter layer -- to be deprecated)  |
|  -----------------------------------------------------------------------  |
|  MedcareBridge      -> impl NamespaceBridge  (Healthcare)                 |
|  WoaBridge          -> impl NamespaceBridge  (WorkOrderBilling)           |
|  OgitBridge         -> impl NamespaceBridge  (TicketTool)                 |
|  column_mask_bridge -> DEPRECATED (see W3 -> unified_bridge_wiring.rs)   |
+--------+----------------+-------------------+----------------+------------+
         |                |                   |                |
         v                v                   v                v
+-------------+  +-------------+  +---------------+  +-------------+
| medcare-rs  |  |smb-office-rs|  |    woa-rs     |  |  hiro-rs    |
| (in-flight) |  | (in-flight) |  | (extraction)  |  | (scaffold)  |
|             |  |             |  |               |  |             |
| SuperDomain |  | SuperDomain |  | SuperDomain   |  | SuperDomain |
| =Healthcare |  | =SmallBiz   |  | =WorkOrder    |  | =Hiro(TBD)  |
+-------------+  +-------------+  +---------------+  +-------------+
         |
         v
+-------------+
| hubspot-rs  |
| (scaffold)  |
| SuperDomain |
| =Hubspot    |
+-------------+
```

Each super-domain subcrate is a **standalone crate** (potentially in its own repo) that:
1. Depends on `lance-graph-contract` for canonical types.
2. Depends on `lance-graph-callcenter` for `UnifiedBridgeImpl` + `SuperDomain`.
3. Does **not** depend on `lance-graph-ontology` directly after extraction.
4. Implements exactly one specialisation of `UnifiedBridgeImpl`.

---

## 3. Specialisation Pattern — medcare-rs Worked Example

The following ~35 LOC sketch is the normative template. Every subcrate MUST follow this shape.

```rust
// medcare-rs/src/lib.rs

use lance_graph_callcenter::{
    SuperDomain, UnifiedBridgeImpl,
    owl::OwlHydrator,
    policy::PolicySeed,
    family::FamilyTable,
    audit::AuditSink,
};
use lance_graph_callcenter::cognitive::CognitiveStack; // cross-ref W6
use lance_graph_ontology::bridges::MedcareBridge;       // until fully extracted

/// Top-level specialisation struct -- one per super-domain subcrate.
pub struct MedcareSuperDomain;

impl UnifiedBridgeImpl for MedcareSuperDomain {
    /// The bridge adapter this subcrate drives.
    type Bridge = MedcareBridge;

    /// Compile-time binding to the SuperDomain discriminant.
    const SUPER_DOMAIN: SuperDomain = SuperDomain::Healthcare;

    /// Returns the static OWL hydrator for this domain's ontology slice.
    /// The hydrator resolves FMA/SNOMED/LOINC families into FamilyId sets.
    fn owl_hydrator() -> &'static OwlHydrator {
        static HYDRATOR: std::sync::OnceLock<OwlHydrator> = std::sync::OnceLock::new();
        HYDRATOR.get_or_init(|| {
            OwlHydrator::from_ttl(include_bytes!("../ontology/medcare.ttl"))
                .expect("medcare OWL hydrator failed to init")
        })
    }

    /// Returns the static policy seed used to derive column masks and
    /// row-level authorisation decisions. Salt is injected per-request
    /// by UnifiedBridge::with_audit_chain().
    fn policy_seed() -> &'static PolicySeed {
        static SEED: std::sync::OnceLock<PolicySeed> = std::sync::OnceLock::new();
        SEED.get_or_init(|| PolicySeed::from_env("MEDCARE_POLICY_SALT"))
    }

    /// Returns the static FamilyTable mapping OWL identities to FamilyId.
    /// Must contain at least one entry (CI gate: consumer-crates-conformance).
    fn family_table() -> &'static FamilyTable {
        static TABLE: std::sync::OnceLock<FamilyTable> = std::sync::OnceLock::new();
        TABLE.get_or_init(|| {
            FamilyTable::build(Self::owl_hydrator(), Self::SUPER_DOMAIN)
        })
    }

    /// Provides the CognitiveStack for this domain's authorize_* hooks.
    /// cross-ref W6: each subcrate gets its own cognitive_stack instance.
    fn cognitive_stack() -> CognitiveStack {
        CognitiveStack::for_domain(Self::SUPER_DOMAIN)
    }

    /// Provides the audit sink. cross-ref W8: sink is Lance/JSONL.
    fn audit_sink() -> Box<dyn AuditSink> {
        lance_graph_callcenter::audit::default_lance_sink(Self::SUPER_DOMAIN)
    }
}
```

This pattern produces a **zero-argument** entry point: callers construct a
`UnifiedBridge<MedcareSuperDomain>` and get the full specialised stack.

---

## 4. Template Surface — Every Subcrate MUST Expose

| Surface | Type | Description |
|---|---|---|
| `BridgeImpl` | `impl UnifiedBridgeImpl` | Top-level specialisation struct |
| `OwlHydrator` | `&'static OwlHydrator` | Parses domain TTL/OWL into identity maps |
| `PolicySeed` | `&'static PolicySeed` | Derives column masks and row auth policy |
| `FamilyTable` | `&'static FamilyTable` | Maps OWL leaf identities to `FamilyId` (cross-ref W9) |
| `audit_sink` | `Box<dyn AuditSink>` | Lance/JSONL audit chain (cross-ref W8) |
| `cognitive_stack` | `CognitiveStack` | Per-subcrate thinking-engine instance (cross-ref W6) |

A subcrate that omits any of the six surfaces fails the CI gate (Section 7).

---

## 5. Migration Order

Migration proceeds in four phases. Each phase is independently mergeable.

### Phase 1 -- Finalise medcare-rs extraction (highest priority, in-flight)

**Source:** `lance-graph-ontology/src/bridges/medcare.rs`
**Target:** external `medcare-rs` crate (own repo or workspace member TBD -- see Open Questions)

Steps:
1. Copy `MedcareBridge` impl out of `lance-graph-ontology` into `medcare-rs/src/bridge.rs`.
2. Add `MedcareSuperDomain` struct (template above) in `medcare-rs/src/lib.rs`.
3. Add `medcare-rs/ontology/medcare.ttl` (bundled via `include_bytes!`).
4. Wire `FamilyTable::build()` against the bundled OWL.
5. Add `CognitiveStack::for_domain(SuperDomain::Healthcare)` call (W6 dependency).
6. Add `audit_sink()` returning `lance_graph_callcenter::audit::default_lance_sink()` (W8 dependency).
7. Delete re-export from `lance-graph-ontology`; add deprecation shim pointing to `medcare-rs`.
8. Run `consumer-crates-conformance` CI gate (Section 7).

**Blocker:** medcare-rs + smb-office-rs local commits not yet pushed (TD-SDR-CONSUMER-PUSH-1, see W7 spec). Extraction cannot be reviewed until those commits are published.

### Phase 2 -- smb-bridge retrofit

**Source:** `crates/smb-bridge` (current `smb-office-rs` local commits)
**Target:** `smb-office-rs` as standalone subcrate

Steps:
1. Identify current `SmallBiz*` bridge code in local unpushed commits.
2. Apply `UnifiedBridgeImpl` template with `SUPER_DOMAIN = SuperDomain::SmallBusiness` (verify discriminant against `super_domain.rs`).
3. Bundle `smb.ttl` OWL slice.
4. Wire `FamilyTable`, `PolicySeed`, `CognitiveStack`, `AuditSink`.
5. Delete `medcare-analytics`/`medcare-bridge` forwarding layer once medcare-rs ships.

**Note:** `SuperDomain::SmallBusiness` discriminant must be assigned in `super_domain.rs` if not already present (current enum has 8 starters -- verify numbering).

### Phase 3 -- woa-rs extraction

**Source:** `lance-graph-ontology/src/bridges/` -- `WoaBridge` exists, no external subcrate yet.
**Target:** `woa-rs` standalone subcrate

Steps:
1. Create `woa-rs/src/lib.rs` with `WoaSuperDomain` implementing `UnifiedBridgeImpl`.
2. `SUPER_DOMAIN = SuperDomain::WorkOrderBilling` (discriminant = 6 per recon).
3. Extract `WoaBridge` from ontology crate; bundle WOA OWL slice.
4. Wire all six surfaces from template.
5. Deprecate `WoaBridge` re-export in `lance-graph-ontology`.

### Phase 4 -- Scaffold hiro-rs + hubspot-rs from template

These two subcrates have no existing bridge code. Scaffold from the template:

1. Copy template; assign provisional `SuperDomain` discriminants (reserve block 8..15 for future domains: 8=Hubspot, 9=Hiro).
2. Wire stub `OwlHydrator` (empty TTL, returns `FamilyTable::empty()`).
3. Add `#[cfg(test)] #[test] fn family_table_non_empty()` that is `#[ignore]` until OWL file is provided.
4. Open issues in respective repos: "Provide domain OWL file to pass consumer-crates-conformance."

---

## 6. UnifiedBridgeImpl Trait -- Normative Definition

The trait lives in `lance-graph-callcenter`. The following is the normative signature engineers must implement:

```rust
// crates/lance-graph-callcenter/src/unified_bridge_impl.rs  (new file)

use crate::{SuperDomain, NamespaceBridge};
use crate::owl::OwlHydrator;
use crate::policy::PolicySeed;
use crate::family::FamilyTable;
use crate::audit::AuditSink;
use crate::cognitive::CognitiveStack;

/// Specialisation contract every super-domain subcrate implements.
/// Provides all static context needed to construct a UnifiedBridge<Self>.
pub trait UnifiedBridgeImpl: Sized + Send + Sync + 'static {
    /// The NamespaceBridge adapter this specialisation drives.
    type Bridge: NamespaceBridge;

    /// Compile-time super-domain discriminant.
    const SUPER_DOMAIN: SuperDomain;

    /// Static OWL hydrator for this domain's ontology slice.
    fn owl_hydrator() -> &'static OwlHydrator;

    /// Static policy seed for column-mask and row-auth derivation.
    fn policy_seed() -> &'static PolicySeed;

    /// Static FamilyTable mapping OWL identities to FamilyId.
    /// MUST be non-empty in production (CI gate enforces).
    fn family_table() -> &'static FamilyTable;

    /// Per-subcrate CognitiveStack for authorize_* hooks. (cross-ref W6)
    fn cognitive_stack() -> CognitiveStack;

    /// Boxed audit sink -- Lance columnar or JSONL. (cross-ref W8)
    fn audit_sink() -> Box<dyn AuditSink>;
}
```

**Key invariants:**
- `SUPER_DOMAIN` is a compile-time constant -- no runtime dispatch on the discriminant itself.
- `owl_hydrator()` + `policy_seed()` + `family_table()` return `&'static` -- initialised once, zero allocation per request.
- `cognitive_stack()` + `audit_sink()` may allocate per-bridge instance (called once in `UnifiedBridge::new()`).

---

## 7. CI Gate -- consumer-crates-conformance

A new integration test crate `crates/consumer-crates-conformance/` validates all known subcrate registrations at CI time.

```rust
// crates/consumer-crates-conformance/src/lib.rs

#[cfg(test)]
mod conformance {
    use lance_graph_callcenter::UnifiedBridgeImpl;

    /// Asserts a type fully satisfies UnifiedBridgeImpl with non-empty FamilyTable
    /// and at least one OWL identity mapping.
    fn assert_conformance<T: UnifiedBridgeImpl>() {
        // 1. FamilyTable must be non-empty
        let table = T::family_table();
        assert!(
            !table.is_empty(),
            "{} FamilyTable is empty -- add OWL identity mappings",
            std::any::type_name::<T>()
        );

        // 2. At least one OWL identity mapping must resolve to a FamilyId
        let hydrator = T::owl_hydrator();
        let count = hydrator.identity_count();
        assert!(
            count >= 1,
            "{} OwlHydrator has zero identity mappings",
            std::any::type_name::<T>()
        );

        // 3. SuperDomain must not be Unknown
        assert_ne!(
            T::SUPER_DOMAIN,
            lance_graph_callcenter::SuperDomain::Unknown,
            "{} declares SuperDomain::Unknown -- assign a real discriminant",
            std::any::type_name::<T>()
        );
    }

    #[test]
    fn medcare_conforms() {
        assert_conformance::<medcare_rs::MedcareSuperDomain>();
    }

    #[test]
    fn smb_office_conforms() {
        assert_conformance::<smb_office_rs::SmallBizSuperDomain>();
    }

    #[test]
    fn woa_conforms() {
        assert_conformance::<woa_rs::WoaSuperDomain>();
    }

    // hiro-rs + hubspot-rs are #[ignore] until OWL files are provided
    #[test]
    #[ignore = "hiro-rs not yet seeded with OWL -- scaffold only"]
    fn hiro_conforms() {
        assert_conformance::<hiro_rs::HiroSuperDomain>();
    }

    #[test]
    #[ignore = "hubspot-rs not yet seeded with OWL -- scaffold only"]
    fn hubspot_conforms() {
        assert_conformance::<hubspot_rs::HubspotSuperDomain>();
    }
}
```

**CI policy:** The three active subcrates (`medcare_rs`, `smb_office_rs`, `woa_rs`) run as non-ignored tests in every PR touching those crates or `lance-graph-callcenter`. The `#[ignore]` scaffold tests are un-ignored only when the OWL file is committed to the respective subcrate repo.

---

## 8. Cross-References

| Flag | Spec | Interaction |
|---|---|---|
| W3 | `td-api-drift-deprecation.md` | `column_mask_bridge.rs` is deprecated in favour of `unified_bridge_wiring.rs`. Each subcrate's `PolicySeed` replaces the column-mask ad-hoc layer. Subcrate extraction MUST occur AFTER W3 deprecation shims land. |
| W6 | `td-thinking-engine-wire.md` | Each subcrate calls `CognitiveStack::for_domain(Self::SUPER_DOMAIN)` in `cognitive_stack()`. W6 must define `CognitiveStack::for_domain(SuperDomain)` before Phase 1 of this spec can compile. |
| W8 | `td-sdr-audit-persist.md` | Each subcrate calls `default_lance_sink(Self::SUPER_DOMAIN)` in `audit_sink()`. W8 must ship the Lance/JSONL sink before Phase 1 can emit audit events to durable storage. |
| W9 | `td-sdr-family-hydration.md` | Each subcrate's `FamilyTable` provides the per-domain slice that W9's TTL hydration populates. The `FAMILY_TO_SUPER_DOMAIN` reverse lookup (currently all-Unknown) becomes correct once each subcrate's `FamilyTable` is registered at boot. |

**Dependency order for a clean Phase 1 build:**
```
W3 (deprecation shims) --> W6 (CognitiveStack::for_domain) --> W8 (Lance sink) --> W4 Phase 1
                                                                                    ^
W9 (family hydration TTL) runs concurrently with W4 Phase 1; they share FamilyTable
```

---

## 9. Subcrate Inventory

| Crate | SuperDomain | Discriminant | Status | OWL source |
|---|---|---|---|---|
| `medcare-rs` | `Healthcare` | 1 | In-flight, unpushed | FMA / SNOMED / LOINC |
| `smb-office-rs` | `SmallBusiness` (TBD) | TBD | In-flight, unpushed | SMB ontology (TBD) |
| `woa-rs` | `WorkOrderBilling` | 6 | Bridge exists, needs extraction | WOA OWL slice |
| `hiro-rs` | Hiro (TBD) | TBD | Scaffold only | None yet |
| `hubspot-rs` | Hubspot (TBD) | TBD | Scaffold only | None yet |

---

## 10. Open Questions

**(a) Repository topology** -- Where do `hubspot-rs`, `hiro-rs`, and `woa-rs` live?
- Options: external repos under AdaWorldAPI, workspace members in lance-graph, or separate monorepo.
- Recommendation: external repos; add as path-or-git deps gated behind Cargo features in the conformance crate.

**(b) Version cadence** -- Does each subcrate maintain its own SemVer cadence independent of `lance-graph-callcenter`?
- Recommendation: yes, with `lance-graph-callcenter` as a `^X.Y` dep range.

**(c) Workspace vs external monorepo policy** -- If a downstream team wants all five subcrates in one monorepo, the template is compatible: each crate is still a workspace member but publishes independently to crates.io.

**(d) Inter-subcrate dependency policy** -- Subcrates MUST NOT depend on each other. Cross-domain queries are composed at the `UnifiedBridge<B>` layer in `lance-graph-callcenter`, not by importing `medcare-rs` from `woa-rs`. This prevents circular deps and keeps each subcrate independently publishable.

**(e) Shared test fixtures location** -- OWL fixtures used across multiple subcrates (e.g. DOLCE upper ontology, shared OGIT vocabulary) should live in `lance-graph-ontology/fixtures/` and be consumed via `dev-dependencies`. Each subcrate's own domain OWL is private to that crate's `ontology/` directory.

---

## 11. File Map

```
New files after full migration:
  crates/lance-graph-callcenter/src/unified_bridge_impl.rs  <- trait definition (Section 6)
  crates/consumer-crates-conformance/src/lib.rs              <- CI gate (Section 7)
  medcare-rs/src/lib.rs                                      <- Phase 1
  medcare-rs/src/bridge.rs                                   <- extracted MedcareBridge
  medcare-rs/ontology/medcare.ttl                            <- bundled OWL
  smb-office-rs/src/lib.rs                                   <- Phase 2
  smb-office-rs/ontology/smb.ttl
  woa-rs/src/lib.rs                                          <- Phase 3
  woa-rs/src/bridge.rs                                       <- extracted WoaBridge
  woa-rs/ontology/woa.ttl
  hiro-rs/src/lib.rs                                         <- Phase 4 scaffold
  hubspot-rs/src/lib.rs                                      <- Phase 4 scaffold

Modified files:
  crates/lance-graph-ontology/src/bridges/medcare.rs         <- deprecation shim -> medcare-rs
  crates/lance-graph-ontology/src/bridges/woa.rs             <- deprecation shim -> woa-rs
  crates/lance-graph-callcenter/src/unified_bridge.rs        <- add UnifiedBridgeImpl bound
  crates/lance-graph-callcenter/src/super_domain.rs          <- add Hubspot + Hiro discriminants
```

---

## 12. Acceptance Criteria

An engineer can mark TD-SUPER-DOMAIN-SUBCRATES-1 resolved when ALL of the following are true:

- [ ] `UnifiedBridgeImpl` trait defined in `lance-graph-callcenter` with all six surfaces (Section 6).
- [ ] `medcare_rs::MedcareSuperDomain` implements `UnifiedBridgeImpl`; `medcare_conforms` CI test passes.
- [ ] `smb_office_rs::SmallBizSuperDomain` implements `UnifiedBridgeImpl`; `smb_office_conforms` passes.
- [ ] `woa_rs::WoaSuperDomain` implements `UnifiedBridgeImpl`; `woa_conforms` passes.
- [ ] `hiro_rs` and `hubspot_rs` scaffold crates exist with `#[ignore]` conformance tests committed.
- [ ] `column_mask_bridge.rs` is marked `#[deprecated]` (W3 gate -- must land first).
- [ ] Each active subcrate's `FamilyTable` returns at least one entry (CI gate enforces).
- [ ] `CognitiveStack::for_domain(SuperDomain)` is defined and wired (W6 gate -- must land first).
- [ ] `default_lance_sink(SuperDomain)` is defined and wired (W8 gate -- must land first).
- [ ] `FAMILY_TO_SUPER_DOMAIN` reverse lookup returns correct `SuperDomain` for each subcrate's families (W9 validates).
