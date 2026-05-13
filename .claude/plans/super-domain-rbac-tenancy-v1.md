# Super-Domain RBAC + Multi-Tenancy — v1

> **Author:** main thread (Opus 4.7 1M), session 2026-05-13 (branch `claude/lance-datafusion-integration-gv0BF`)
> **Status:** Active
> **Scope:** Lock the 4-level addressing hierarchy (meta-anchors → super domain → OGIT basin → within-basin slot), ship explicit DTOs with byte-sized contracts, and wire RBAC + multi-tenant Chinese walls onto the super-domain boundary. Foundry-parity selling point at the enforcement surface.
> **Path:** `.claude/plans/super-domain-rbac-tenancy-v1.md`
> **Confidence:** Working (architecture); Partial (DOLCE/OWL slot semantics still need probe before per-slot redaction policies harden)

---

## 1 — Why this exists

The lance-graph workspace already ships substantial OGIT substrate (`lance-graph-ontology::namespace`, `bridges::OgitBridge`, `parse_ttl_directory_with_provenance`, `holograph::dntree`, `bgz-tensor::HhtlDEntry`, `highheelbgz::SpiralAddress`, `lance-graph-callcenter::DnPath`). What's missing is an **explicit, byte-sized contract** that says:

- **OGIT is hardware-level addressing** — sub-microsecond bitmask predicates, no Neo4j-style label resolution at query time, no Cypher permission round-trips.
- **Labels and metadata live inline in the per-family codebook** — not in a sidecar. One fetch resolves both for the hot path.
- **Super domains are first-class** — the activation routing layer above OGIT basins, doubling as the RBAC + compliance gate.
- **Tenants are cryptographically isolated** — Chinese walls at the row level, with per-tenant DEKs as crypto backstop to the predicate filter.

This is the spec the Foundry-parity sales motion hangs on. Palantir charges $1M+/yr for ObjectType-level enforcement; we ship it as a single masked predicate at the super-domain boundary.

**This spec corrects an earlier sketch** that proposed a label-vs-metadata sidecar table joined at query time. That was Neo4j-shaped — wrong. The right architecture is **inline per-family codebook**, addressed by the same 16-bit `OwlIdentity`. No join. No sidecar.

---

## 2 — The 4-level hierarchy

```
Level 0 — META-ANCHORS                                  (interop, ~1 KB total)
    Foundry ObjectType / OWL upper class / DOLCE marker / Wikidata QID
    Cross-walks to external standards. Consulted only when interop is requested.

Level 1 — SUPER DOMAIN                                  (governance, ~7-10 today, 256 max)
    Healthcare, Science, Genetics, QuantumPhysics,
    TicketTool, WorkOrderBilling, OSINT
    1 byte. Activation routing + RBAC trust boundary + compliance regime.

Level 2 — OGIT BASIN                                    (the heel, ~75 today, 256 max)
    Hiro, HubSpot, FMA, SNOMED, ICD10, RxNorm, WorkOrder, ...
    1 byte (high byte of OwlIdentity). Per-codebook; new schema = new basin.

Level 3 — WITHIN-BASIN SLOT                             (the row identity, 256 slots/family)
    1 byte (low byte of OwlIdentity). Indexes the per-family codebook table
    that holds label + schema + verbs INLINE (not sidecar).
```

**Per-row LanceDB overhead: 6 bytes total**
- `tenant_id: u32` — 4 bytes (Chinese wall predicate + DEK selector)
- `owl_id:    u16` — 2 bytes (super domain via family lookup + basin + slot)

DataFusion combines all 4 RBAC stages into one predicate vector at plan time.

---

## 3 — Core DTOs

### 3.1 OGIT family pointer (the heel)

```rust
/// 1 byte. Identifies which OGIT family (basin) a row belongs to.
/// 256 families max; ~75 used today (per `RECON_ONTOLOGY_CRATE.md` §1.9).
/// Pure address. No reasoning, no string lookup.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct OgitFamily(pub u8);

impl OgitFamily {
    pub const UNKNOWN:   Self = Self(0);
    pub const NETWORK:   Self = Self(0x01);
    pub const WORKORDER: Self = Self(0x05);
    // ... seeded by NamespaceRegistry::seed_defaults at hydration

    pub const fn raw(self) -> u8 { self.0 }
    pub const fn is_known(self) -> bool { self.0 != 0 }

    /// DataFusion predicate: `(owl_id >> 8) == self.0`.
    /// Single masked compare. No registry deref.
    #[inline] pub const fn matches(self, owl: OwlIdentity) -> bool {
        owl.family().0 == self.0
    }
}
```

### 3.2 OWL identity (per-row, 16-bit)

```rust
/// 2 bytes. BF16-shaped container (interpreted as named bit-fields,
/// not literal floating-point semantics).
/// High 8 bits = OGIT family (the precise heel pointer / "mantissa").
/// Low  8 bits = within-family slot (the OWL/consumer's own identity).
/// This is what rides on every LanceDB row.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct OwlIdentity(pub u16);

impl OwlIdentity {
    pub const UNKNOWN: Self = Self(0);

    #[inline] pub const fn new(family: OgitFamily, slot: u8) -> Self {
        Self(((family.0 as u16) << 8) | slot as u16)
    }
    #[inline] pub const fn family(self) -> OgitFamily { OgitFamily((self.0 >> 8) as u8) }
    #[inline] pub const fn slot(self)   -> u8         { (self.0 & 0xFF) as u8 }
    #[inline] pub const fn raw(self)    -> u16        { self.0 }

    /// Bitmask predicates Cypher MATCH lowers to. No string lookup.
    #[inline] pub const fn is_family(self, f: OgitFamily) -> bool { self.family().0 == f.0 }
    #[inline] pub const fn is_slot(self, s: u8) -> bool { self.slot() == s }
}
```

### 3.3 Per-family codebook table (label + schema + verbs INLINE)

```rust
/// One table per OGIT family. Lives in static memory after hydration.
/// 256-slot dense array — slot index IS OwlIdentity.slot().
/// Each slot carries label + schema + verbs INLINE (not a sidecar).
/// Size per family: ~50-200 KB. ~75 tables total ≈ 5-15 MB resident.
pub struct OgitFamilyTable {
    pub family:   OgitFamily,
    pub entries:  [Option<FamilyEntry>; 256],
    pub codebook: PerFamilyCodebook,   // family-local 5-8 bit centroids
}

/// 1 slot. Resolves a 16-bit OwlIdentity to content.
/// Variable size depending on axiom blob; typical ~80-200 bytes.
pub struct FamilyEntry {
    pub label_uri:           &'static str,        // "ogit.Network:IPAddress"
    pub kind:                SchemaKind,          // Entity / Edge / Attribute (1 byte)
    pub owl_characteristics: OwlCharacteristics,  // 1 byte bitfield
    pub dolce_marker:        DolceMarker,         // 1 byte
    pub axiom_blob:          &'static [u8],       // OWL subClassOf, equivalentClass, etc.
    pub provenance:          &'static str,        // dcterms:source — carries off-label lineage
    pub verbs:               &'static [u8],       // outgoing verb slots within this family
}

impl OgitFamilyTable {
    /// Hot path: O(1) array index. Sub-microsecond.
    #[inline] pub fn lookup(&self, owl: OwlIdentity) -> Option<&FamilyEntry> {
        debug_assert_eq!(owl.family().0, self.family.0);
        self.entries[owl.slot() as usize].as_ref()
    }
    #[inline] pub fn label(&self, owl: OwlIdentity) -> Option<&str> {
        self.lookup(owl).map(|e| e.label_uri)
    }
    #[inline] pub fn kind(&self, owl: OwlIdentity) -> Option<SchemaKind> {
        self.lookup(owl).map(|e| e.kind)
    }
    #[inline] pub fn is_functional(&self, owl: OwlIdentity) -> bool {
        self.lookup(owl).map_or(false, |e| e.owl_characteristics.is_functional())
    }
    #[inline] pub fn is_transitive(&self, owl: OwlIdentity) -> bool {
        self.lookup(owl).map_or(false, |e| e.owl_characteristics.is_transitive())
    }
}
```

### 3.4 Super domain (activation root + RBAC trust boundary)

```rust
/// 1 byte. Activation root + RBAC + compliance regime.
/// 8 starter values; up to 256.
/// Promotes the existing `holograph::dntree::WellKnown` constants
/// to first-class business-named activation roots with formal cross-walks.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SuperDomain {
    Unknown          = 0,
    Healthcare       = 1,    // FMA, SNOMED, ICD10, RxNorm, LOINC, RadLex, MONDO, HPO, DRON, CHEBI
    Science          = 2,    // physics, chemistry, math, materials
    Genetics         = 3,    // genes, sequences, expression, GO
    QuantumPhysics   = 4,    // quantum-specific (specialized lift from Science)
    TicketTool       = 5,    // Hiro, HubSpot, ServiceNow, Jira, Zendesk
    WorkOrderBilling = 6,    // WorkOrder, Billing, Tax, SMB, MRO, MRP, Accounting
    OSINT            = 7,    // Maltego, intel sources, social graph
}

/// One entry per SuperDomain. ~8 entries, ~30 bytes each ≈ 240 bytes total.
pub struct SuperDomainEntry {
    pub super_domain: SuperDomain,
    pub basins:       &'static [OgitFamily],   // 10-30 basins this domain spans
    pub meta:         MetaAnchors,
    pub label:        &'static str,            // "Healthcare"
    pub role_groups:  &'static [RoleGroup],    // nested permission groups (§3.6)
    pub compliance:   ComplianceRegime,        // HIPAA / SOX / PCI-DSS / GDPR / OSINT / ITAR
}

impl SuperDomainEntry {
    /// Activation drill-down: super domain → constituent basins.
    /// Used by spreading-activation queries ("anything in Healthcare").
    #[inline] pub fn activate(&self) -> &[OgitFamily] { self.basins }

    /// Cross-walk: this super domain's anchor in an external upper ontology.
    #[inline] pub fn cross_walk(&self) -> &MetaAnchors { &self.meta }
}

/// Reverse lookup: OgitFamily → SuperDomain it belongs to.
/// 256-entry static array; 1 byte each = 256 bytes total.
/// Single-member by default; multi-member escape hatch via BitSet256
/// reserved for the 2-3 known cross-cutting cases (HPO/MONDO straddle
/// Healthcare ↔ Genetics).
pub static FAMILY_TO_SUPER_DOMAIN: [SuperDomain; 256] = { /* baked at hydration */ };
```

### 3.5 Meta-anchors (cross-walk to formal upper ontologies)

```rust
/// 4 cross-walk pointers — one per upper standard. Each optional.
/// Consulted only when interop with an external system is requested.
pub struct MetaAnchors {
    pub foundry_object_type: Option<&'static str>,  // "PhysicalSystem"
    pub owl_upper_class:     Option<&'static str>,  // "BiomedicalOntology"
    pub dolce_marker:        DolceMarker,           // Endurant / Perdurant / Quality / Abstract
    pub wikidata_qid:        Option<u64>,           // Q11190 (medicine)
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DolceMarker {
    Unknown   = 0,
    Endurant  = 1,
    Perdurant = 2,
    Quality   = 3,
    Abstract  = 4,
}
```

### 3.6 Role groups (nested RBAC within super domain)

```rust
/// One nested role group within a super domain.
/// Healthcare has: physician, nurse, cashier, researcher, hipaa_audit, admin
/// WorkOrderBilling has: cashier, controller, sox_audit
pub struct RoleGroup {
    pub role_name:       &'static str,       // "clinic_personnel", "cashier", "hipaa_audit"
    pub permissions:     PermissionSet,       // 1 byte bitfield
    pub clearance_floor: ClearanceLevel,
    pub audit_required:  bool,
    pub redaction_mask:  FieldRedactionMask,  // 96 bytes: 3 × BitSet256
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PermissionSet(pub u8);

impl PermissionSet {
    pub const READ:         u8 = 1 << 0;
    pub const WRITE:        u8 = 1 << 1;
    pub const DELETE:       u8 = 1 << 2;
    pub const EXPORT:       u8 = 1 << 3;
    pub const ESCALATE:     u8 = 1 << 4;   // request elevated access
    pub const AUDIT_BYPASS: u8 = 1 << 5;   // emergency, must justify
    pub const SCHEMA_VIEW:  u8 = 1 << 6;   // see metadata only, no instances
    pub const REDACT_LIFT:  u8 = 1 << 7;   // see un-redacted values

    #[inline] pub const fn allows(self, op: u8) -> bool { self.0 & op != 0 }
}

/// 96 bytes per role group. Slot-level visibility within a super domain's basins.
pub struct FieldRedactionMask {
    pub readable_slots: BitSet256,   // 32 bytes — slots this role can read
    pub writable_slots: BitSet256,   // 32 bytes — slots this role can mutate
    pub redacted_slots: BitSet256,   // 32 bytes — slots returned hashed/nulled/starred
}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct BitSet256(pub [u64; 4]);

impl BitSet256 {
    #[inline] pub const fn contains(&self, bit: u8) -> bool {
        self.0[(bit >> 6) as usize] & (1u64 << (bit & 0x3F)) != 0
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ClearanceLevel(pub u8);  // 0=public, 1=restricted, 2=confidential, 3=secret
```

### 3.7 Compliance regime

```rust
/// 1 byte. Tagged on each SuperDomainEntry. Drives certification surface.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ComplianceRegime {
    None             = 0,
    HIPAA            = 1,   // Healthcare
    SOX              = 2,   // WorkOrderBilling (financial reporting)
    PCI_DSS          = 3,   // WorkOrderBilling (payment cards)
    GDPR             = 4,   // cross-cutting (any super domain with PII)
    OSINT_CLEARANCE  = 5,   // OSINT (tiered by classification)
    ITAR_EAR         = 6,   // Science / QuantumPhysics (dual-use research)
}
```

### 3.8 Tenant context (multi-tenant Chinese wall)

```rust
/// 4-byte newtype. ~4 billion tenants. Carried on every row.
/// Crypto backstop: per-tenant DEK wraps row payload, so a misconfigured
/// query that bypasses the predicate filter still can't decrypt cross-tenant rows.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TenantId(pub u32);

/// Multi-tenant context the bridge holds.
pub struct TenantContext {
    pub tenant_id:      TenantId,
    pub display_name:   &'static str,                          // "Massachusetts General Hospital"
    pub role_bindings:  &'static [(SuperDomain, &'static str)],
                        // e.g., [(Healthcare, "physician"), (WorkOrderBilling, "cashier")]
    pub encryption_key: KeyHandle,                              // per-tenant DEK
    pub federation:     FederationPolicy,                       // pure wall (default) | k-anonymity escape
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FederationPolicy {
    PureWall            = 0,   // default; no cross-tenant queries
    KAnonymityAggregate = 1,   // aggregate-only with k ≥ 5
    HomomorphicAggregate = 2,  // 2027+ R&D; aggregate without decryption
}
```

### 3.9 Unified bridge (4-stage authorize)

```rust
pub struct UnifiedBridge {
    registry:   Arc<OntologyRegistry>,
    tenant:     TenantContext,
    audit_sink: Arc<dyn AuditSink>,
}

impl UnifiedBridge {
    /// Hot-path authorization. All 4 checks lower to bitmask predicates
    /// in DataFusion — combined into one predicate vector at plan time.
    #[inline] pub fn authorize(
        &self,
        owl:        OwlIdentity,
        row_tenant: TenantId,
        op:         u8,             // PermissionSet bit
    ) -> Result<RowAccess> {
        // 1. CHINESE WALL: tenant isolation (single u32 compare)
        if row_tenant != self.tenant.tenant_id {
            return Err(RbacError::CrossTenantViolation);
        }

        // 2. SUPER DOMAIN: which super domain this basin belongs to
        let sd = FAMILY_TO_SUPER_DOMAIN[owl.family().0 as usize];
        let role_name = self.tenant.role_bindings.iter()
            .find_map(|(s, r)| (*s == sd).then_some(*r))
            .ok_or(RbacError::NoSuperDomainAccess)?;

        // 3. ROLE GROUP: lookup role within super domain
        let role = SUPER_DOMAINS[sd as usize].role_groups.iter()
            .find(|r| r.role_name == role_name)
            .ok_or(RbacError::UnknownRole)?;
        if !role.permissions.allows(op) {
            return Err(RbacError::OperationNotPermitted);
        }

        // 4. SLOT REDACTION: per-slot mask within role
        let slot = owl.slot();
        let access = if op == PermissionSet::READ {
            if role.redaction_mask.readable_slots.contains(slot) { RowAccess::Full }
            else if role.redaction_mask.redacted_slots.contains(slot) { RowAccess::Redacted }
            else { RowAccess::Hidden }
        } else if op == PermissionSet::WRITE {
            if role.redaction_mask.writable_slots.contains(slot) { RowAccess::Full }
            else { return Err(RbacError::SlotNotWritable); }
        } else {
            RowAccess::Full
        };

        if role.audit_required {
            self.audit_sink.emit(AuditEntry::new(self.tenant.tenant_id, role_name, owl, op));
        }
        Ok(access)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RowAccess { Full, Redacted, Hidden }
```

### 3.10 DataFusion lowering (the hot path)

The 4 stages combine into one predicate vector at plan time:

```sql
WHERE tenant_id = $T
  AND ((owl_id >> 8) & 0xFF) IN $authorized_basins
  AND ((owl_id & 0xFF) & $slot_visibility_mask) != 0
```

One vectorized scan, no per-row branching, no per-row policy evaluation. Sub-microsecond stays intact.

---

## 4 — Concrete consumer-to-basin mapping

### 4.1 The 2 ticket consumers (per user constraint)

| Consumer crate | Super domain | OGIT basin | Bridge | Notes |
|---|---|---|---|---|
| **`hiro-rs`** (NEW) | TicketTool | `OgitFamily::Hiro` (NEW) | `HiroBridge::from_registry()` | Absorbs all 4 OSLC-* namespaces with off-label fit acceptable. Per-OSLC-spec lineage rides in `FamilyEntry.provenance`. Bardioc stack (JanusGraph/TinkerPop/Cassandra/BEAM) doesn't care about OSLC-spec-purity at runtime. |
| **`hubspot-rs`** (NEW) | TicketTool, WorkOrderBilling | `OgitFamily::HubSpot` (NEW) | `HubspotBridge::from_registry()` | Fresh basin. CRM codebook (Deal, Contact, Ticket, Pipeline, Property, Engagement) has zero overlap with Hiro's IT-ops vocabulary. Cross-cuts WorkOrderBilling for deals/billing. |

### 4.2 Existing + planned consumers

| Consumer | Super domain(s) | Basin(s) | Compliance | Status |
|---|---|---|---|---|
| `medcare-rs` | Healthcare | Healthcare-* | HIPAA | Bridge shipped |
| `woa-rs` | WorkOrderBilling | WorkOrder | SOX | Bridge shipped |
| `q2-rs` | TBD | Q2 | None | Bridge planned (D-ONTO-V5-5) |
| `smb-office-rs` | WorkOrderBilling | SMB | None | Bridge shipped |
| `osint-rs` (FUTURE) | OSINT | Maltego, intel-* | OSINT_CLEARANCE | Not yet scoped |
| `science-rs` (FUTURE) | Science, QuantumPhysics, Genetics | physics-*, chem-*, GO | ITAR_EAR for dual-use | Not yet scoped |

### 4.3 Healthcare super domain — full role matrix (illustrative)

| Role | Permissions | Readable slots | Redacted slots | Writable slots | Audit |
|---|---|---|---|---|---|
| `physician` | READ + WRITE | clinical (Diagnosis/Vital/Med) + demographics | SSN, full DOB | clinical notes | yes |
| `nurse` | READ + WRITE (limited) | clinical, vitals | SSN, full DOB | vitals only | yes |
| `cashier` | READ | billing slots only | clinical (hashed for matching only) | none | yes (financial trail) |
| `researcher` | READ | de-identified slots only | name, SSN, address all hashed | none | yes (k-anonymity log) |
| `hipaa_audit` | READ + EXPORT + AUDIT_BYPASS | all slots | none | none | yes (every access logged) |
| `admin` | SCHEMA_VIEW | none (slot 0xFF reserved as schema-only) | n/a | none | no |

Sales narrative:

> "Same binary, same database, multiple hospitals — zero leakage. Within Mass General, the physician sees clinical detail, the nurse sees vitals, the cashier sees billing, the auditor sees everything with the audit trail. The researcher sees de-identified data only. The same setup runs for Hopkins next door, with cryptographic isolation between them."

---

## 5 — OSLC absorption decision

**Call: collapse 4 OSLC-* namespaces into `OgitFamily::Hiro`.** Off-label fit is acceptable.

The off-label-ness rides in `FamilyEntry.provenance` — already in the DTO, no new bytes:

```rust
FamilyEntry {
    label_uri:           "ogit.Hiro:PerformanceMetric",
    kind:                SchemaKind::Entity,
    owl_characteristics: OwlCharacteristics(FUNCTIONAL),
    dolce_marker:        DolceMarker::Perdurant,
    axiom_blob:          &[],
    provenance:          "OSLC-perfmon v3.0 § 4.2 (off-label fit: closest match to Hiro's metric model)",
    verbs:               &[/* slots */],
}
```

Why collapse:
- 4-basin alternative preserves OSLC-spec purity but **forks the codebook 4 ways for one consumer** — directly against the "boring agnostic" + "no consultant hours" principle.
- Off-label mapping into Hiro means consumers query `family = Hiro` and get coherent results, with lineage available if anyone wants it.
- The bardioc stack doesn't care about OSLC-spec-of-origin at runtime — it just routes tickets. The basin matches the runtime granularity, not the standards-body granularity.

---

## 6 — Cross-tenant federation policy

**Default: pure Chinese wall (Option A).** Cross-tenant queries always denied. Forces consumers to do anonymization upstream (export to a separate tenant scoped for benchmarking).

**Phase 2 escape hatch (Option B): k-anonymity aggregate.** A special role like `federated_researcher` can run aggregate-only queries across tenants if results pass k-anonymity threshold (k ≥ 5 typical). Uses HyperLogLog-style cardinality estimation; raw row access still impossible.

**2027+ R&D track (Option C): homomorphic-encryption aggregate.** Aggregate without decryption. Most defensible for regulated data, slowest, complex.

A is the conservative HIPAA-defensible default. B is the practical compromise for healthcare analytics. C is research, not deliverable.

---

## 7 — Substrate this builds on (citations)

| Existing artifact | Role in this spec |
|---|---|
| `lance-graph-ontology::namespace::SchemaPtr` | Source for the bit-packing convention. The new `OwlIdentity` is a 16-bit refinement of the same idea (drops the `kind` byte to a slot field, keeps the `(ns, type_id)` core). |
| `lance-graph-ontology::bridges::OgitBridge` (`src/bridges/ogit_bridge.rs`) | Bridge pattern + `BridgeFromRegistry` trait. Hiro/HubSpot bridges follow the same shape (~45 LOC each). |
| `lance-graph-ontology::ttl_parse` + `foundry_map` | TTL hydrator that emits `(ns_id, type_id, kind)` rows. Will emit `OgitFamilyTable` entries instead. |
| `lance-graph-ontology::namespace_registry::NamespaceRegistry` | Codebook seeding. `seed_defaults()` extends to seed `SuperDomain` ↔ `OgitFamily` mapping. |
| `holograph::dntree::WellKnown` (`src/dntree.rs:213-260`) | Existing un-named super-domain ordinals. Promoted to first-class business-named `SuperDomain` enum with cross-walk anchors. |
| `holograph::dntree::CogVerb` (144 verbs in 6 categories) | Cognitive verb taxonomy (separate from OGIT domain verbs). Stays as the spreading-activation substrate; not part of this spec. |
| `lance-graph-callcenter::dn_path::DnPath` | The `tree/{ns}/heel/h/hip/h/branch/b/twig/t/leaf/l` hierarchy. The 16-bit `OwlIdentity` is the runtime address that DnPath compresses to via ZeckBF17→Base17→CAM-PQ→scent (1B, ρ=0.937) — the user's planned compression chain. |
| `bgz-tensor::HhtlDEntry` (`BGZ_HHTL_D.md`) | 4-byte `Slot D + Slot V` layout that proved the bit-packed-hierarchy approach for transformer weights. The 16-bit `OwlIdentity` adapts the *idea* (high bits = family, low bits = slot) to ontology rows. **Different register from Slot D's `Ba[15:14]`** (which is for QK/V/Gate/FFN model-weight roles). |
| `highheelbgz::SpiralAddress` | 12-byte spiral address — unrelated to OWL identity; kept as the model-weight encoding. |
| `lance-graph-contract::cam` (`CamCodecContract`, `DistanceTableProvider`, `IvfContract`) | CAM-PQ substrate for compressed-distance lookup. Not changed by this spec; the per-family codebook compression in `OgitFamilyTable.codebook` plugs into this. |
| `GLUE_LAYER_OGIT_TO_OWL_SPEC.md` (in `.grok/`) | Source for the 1-byte OWL property characteristics bitfield (Functional / InverseFunctional / Transitive / Symmetric / Asymmetric / Reflexive / Irreflexive / Reserved). |
| `palantir-parity-cascade-v2.md` | Foundry parity narrative. This spec adds the **enforcement surface** (RBAC + compliance + tenant isolation) that the parity story needs. |
| `lance-graph-ontology-v5.md` | The ontology registry plan. This spec adds the RBAC + tenancy layer **above** v5; v5 stays as-is. |

---

## 8 — Deliverables

### Tier A — DTOs + bridge surface (lance-graph workspace)

- **D-SDR-1** — `OgitFamily`, `OwlIdentity`, `BitSet256`, `ClearanceLevel`, `PermissionSet`, `DolceMarker`, `SchemaKind`, `OwlCharacteristics` newtypes/enums in `lance-graph-contract`. ~80 LOC + 12 unit tests.
- **D-SDR-2** — `SuperDomain`, `MetaAnchors`, `RoleGroup`, `FieldRedactionMask`, `ComplianceRegime`, `TenantId`, `TenantContext`, `FederationPolicy`, `RbacError`, `RowAccess` in `lance-graph-contract::rbac`. ~150 LOC + 8 unit tests.
- **D-SDR-3** — `OgitFamilyTable`, `FamilyEntry`, `PerFamilyCodebook` in `lance-graph-ontology`. ~120 LOC + 6 unit tests.
- **D-SDR-4** — `SuperDomainEntry` table + `SUPER_DOMAINS` static + `FAMILY_TO_SUPER_DOMAIN` reverse lookup in `lance-graph-ontology::super_domain`. ~200 LOC for the 8 starter super domains + ~75 family mappings; 4 integration tests.
- **D-SDR-5** — `UnifiedBridge::authorize()` 4-stage flow + `AuditSink` trait + DataFusion predicate-lowering in `lance-graph-ontology::bridges::unified`. ~200 LOC + 6 integration tests covering each error path.

### Tier B — TTL namespaces (AdaWorldAPI/OGIT fork PRs)

- **D-SDR-6** — `OGIT/NTO/Hiro/{entities,verbs}/*.ttl` (15-30 entities + 10-20 verbs, absorbs OSLC-* with provenance lineage). One PR on the OGIT fork.
- **D-SDR-7** — `OGIT/NTO/HubSpot/{entities,verbs}/*.ttl` (15-25 entities + 8-15 verbs, fresh CRM vocabulary). One PR on the OGIT fork.

### Tier C — Consumer crate scaffolding

- **D-SDR-8** — `/home/user/hiro-rs` new crate. `HiroBridge::from_registry()` + 1 round-trip integration test (`Ticket` URI → `OwlIdentity`). ~150 LOC.
- **D-SDR-9** — `/home/user/hubspot-rs` new crate. `HubspotBridge::from_registry()` + 1 round-trip test (`Deal` URI → `OwlIdentity`). ~150 LOC.

### Tier D — Compliance + audit surface

- **D-SDR-10** — `AuditEntry` JSON schema + `JsonLinesAuditSink` impl + retention policy doc. ~80 LOC + 1 integration test that verifies every authorized read emits exactly one audit line.
- **D-SDR-11** — Compliance regime certification stub: per-`ComplianceRegime` checklist (HIPAA → §164.312 controls; SOX → §404 internal controls; PCI-DSS → Reqs 3+7+10) cross-referencing which DTOs/methods enforce each control. Doc only, ~200 lines markdown.

### Tier E — Cross-tenant federation gate (Phase 2)

- **D-SDR-12** — `FederationPolicy::KAnonymityAggregate` impl: aggregate-query gate with k-threshold check via HLL cardinality. ~150 LOC + 4 integration tests. **Deferred until a customer demands it.**

---

## 9 — Tradeoffs flagged explicitly

### 9.1 Single-member vs multi-member super-domain assignment
- **Default:** `FAMILY_TO_SUPER_DOMAIN: [SuperDomain; 256]` — one super domain per basin.
- **Escape hatch:** `[BitSet256; 256]` for the 2-3 cross-cutting basins (HPO/MONDO straddle Healthcare ↔ Genetics). Same byte cost (1 byte per basin if we use 8-bit bitset for 8 super domains).
- **Decision:** ship single-member; promote to bitset only when a real cross-cutting case fails the assignment.

### 9.2 Coarse super-domain RBAC vs fine per-slot RBAC
- **Coarse (super-domain only):** single masked predicate, sub-microsecond, default policy. Insufficient for cases where one slot within a basin needs elevated protection (e.g., SSN field within Healthcare needs `Confidential` while other PHI is `Restricted`).
- **Fine (per-slot via redaction mask):** the `FieldRedactionMask` (3 × BitSet256 = 96 bytes per role) handles this — already in the DTO.
- **Decision:** ship both granularities. Coarse covers the 80% case, fine covers PHI-style sub-slot redaction.

### 9.3 SGO meta exclusion from runtime
- SGO has 64 entities + 179 verbs + 265 attributes = **508 items**, overflowing the 256-slot per-family ceiling.
- **Decision:** SGO is **authoring-time scaffolding only**. NTO domain entities reference SGO meta-types at hydration; runtime `OwlIdentity` only addresses NTO. SGO meta is resolved once during codebook bake, then discarded.
- Matches BGZ-HHTL-D's "palette baked offline, runtime reads N bytes per row" pattern.

### 9.4 OSLC collapse vs preserve
- **Decision:** collapse to 1 basin (Hiro) with provenance lineage. See §5.

### 9.5 Cross-tenant federation
- **Decision:** PureWall by default. Phase 2 escape hatch as Option B. See §6.

---

## 10 — Open questions for the next session

1. **Foundry ObjectType cross-walk targets** — which Foundry ObjectType strings anchor each super domain? Need product-side input for the `MetaAnchors.foundry_object_type` field.
2. **Wikidata QID mappings** — concrete QIDs for Healthcare (Q11190?), Science, Genetics, etc. Mechanical lookup but needs a one-shot pass.
3. **Audit format choice** — JSON Lines (default) vs CloudEvents vs OpenTelemetry trace. Which integrates with the customer's SIEM stack?
4. **Key-rotation cadence** for per-tenant DEKs — quarterly default or per-policy?
5. **Escalation protocol UX** — when a role hits `RbacError::OperationNotPermitted`, what's the customer-facing escalation flow? (Out of scope for this spec, but needed for the sales pitch.)
6. **HPO/MONDO multi-member assignment** — confirm these are the only cross-cutting basins (Healthcare ↔ Genetics) before locking single-member as default.
7. **Slot 0xFF reserved as schema-only** — convention or enforced? Lean toward enforced, with a `SchemaOnlySlot` const.

---

## 11 — Status

- **Architecture:** locked (4-level hierarchy, DTOs sized, authorize flow defined).
- **DTOs:** drafted in this spec; ready for `cargo check` once Tier A lands.
- **TTL namespaces:** not authored yet (Tier B = 2 OGIT-fork PRs).
- **Consumer crates:** not scaffolded yet (Tier C = 2 new crates in `/home/user/`).
- **Compliance certification:** stubbed (Tier D doc); legal review required before HIPAA-compliant claim.
- **Federation:** deferred (Tier E, Phase 2).

**Confidence:** Working — the design is internally consistent and grounded in shipped substrate. Per-slot redaction policies (Tier A's `FieldRedactionMask` semantics) need a probe with real OWL/DOLCE slot semantics before they can be hardened against actual PHI.

---

## 12 — One-line summary

> 4-level hierarchy (meta-anchor → super domain → OGIT basin → slot), 6 bytes per row (4-byte tenant + 2-byte OWL identity), inline per-family codebook with label+schema+verbs, single masked predicate enforces tenant + super-domain + role + slot in one DataFusion vector pass. Foundry parity at the enforcement surface, sub-microsecond hot path.

---

## 13 — Refinements (2026-05-13, same session)

Post-draft user feedback surfaced four substrate facts and two requirement upgrades. All folded in as additive corrections — the §3 DTOs, §8 deliverables, and §12 summary remain valid; this section makes the underlying compositor explicit and tightens the federation + hard-lock policy.

### 13.1 The compositor is already shipped: `lance-graph-callcenter::policy`

The 4-stage `UnifiedBridge::authorize()` (§3.9) is **not** a new enforcement layer — it composes against `lance-graph-callcenter/src/policy.rs`'s shipped `PolicyRewriter` trait + `PolicyKind` taxonomy:

```rust
pub enum PolicyKind {
    RowFilter,            // tenant + super-domain + basin bitmask predicates
    ColumnMask,           // per-slot RedactionMask (Null/Constant/Hash/Truncate)
    RowEncryption,        // per-tenant DEK at LanceDB column level
    DifferentialPrivacy,  // k-anonymity aggregate noise (federation Option B)
    Audit,                // side-channel emission per access
}
```

The 4 stages map 1:1 to `PolicyKind` variants:

| `authorize()` stage | `PolicyKind` |
|---|---|
| 1. Chinese wall (tenant) | `RowFilter` (existing `RlsRewriter` as ancestor) |
| 2. Super-domain | `RowFilter` (additional predicate, same rewriter chain) |
| 3. Role group | `ColumnMask` (drives slot-level visibility per `RedactionMode`) |
| 4. Slot redaction | `ColumnMask` + `RowEncryption` (when `clearance_floor ≥ Confidential`) |
| Audit emission | `Audit` (composed last) |

**Consequence:** Tier A deliverables (D-SDR-1..5) **wire onto the existing `PolicyRewriter` chain** rather than introducing a parallel enforcement path. ~30% LOC reduction on Tier A. The DataFusion `OptimizerRule` machinery already handles the predicate-vector composition described in §3.10.

### 13.2 LanceDB transparent encryption upgrades Option C from R&D to viable

Earlier framing of cross-tenant federation (§6) classified Option C (homomorphic-encryption aggregate) as a 2027+ R&D track. **Correction:** LanceDB ships **transparent encrypted views** at the column level — the engine scans/filters/aggregates over encrypted columns without decrypting full rows, with key access gated by tenant DEK. This is the substrate Option C needs without bespoke FHE primitives.

**Updated federation policy:**

```rust
#[repr(u8)]
pub enum FederationPolicy {
    PureWall              = 0,   // default — no cross-tenant queries
    KAnonymityAggregate   = 1,   // Phase 2 — k ≥ 5 via PolicyKind::DifferentialPrivacy
    EncryptedViewAggregate = 2,  // Phase 2-3 — LanceDB transparent encrypted view + per-tenant DEK
                                  //   (was Option C, now viable; not slow)
}
```

**Tier E (D-SDR-12) scope expands:** ship A+B together as Phase 2; add Phase 3 EncryptedViewAggregate path that lifts the k-anonymity threshold for tenants whose data column is encrypted at rest with their own DEK (the engine aggregates over ciphertext when the operation is sum/count/avg with bounded sensitivity).

### 13.3 Merkle + ClamPath integration: audit chain + hard-lock attestation

`crates/lance-graph/src/graph/spo/merkle.rs` ships:

```rust
pub struct MerkleRoot(pub u64);              // XOR-fold hash of fingerprint content
impl MerkleRoot {
    pub fn from_fingerprint(fp: &Fingerprint) -> Self { /* ... */ }
}

pub struct ClamPath {                         // hierarchical DN address
    pub path: String,                         // "agent:test:node"
    pub depth: u32,
}
```

**The merkle/DN-path mixing the user remembered is here** — `MerkleRoot` stamps content, `ClamPath` carries the hierarchical address (the same shape as `DnPath` from `lance-graph-callcenter::dn_path`).

**Wire into the spec:**

- **Audit chain integrity:** every `AuditEntry` (Tier D, D-SDR-10) carries the `MerkleRoot` of the row at access time. A second access produces a new merkle root; the audit log records both, so HIPAA reviewers can detect post-hoc tampering (the merkle would not validate against the recorded fingerprint).
- **Hard-lock attestation (§13.4):** the cryptographic separation between Healthcare and OSINT super domains is attested by **distinct merkle root salts per super domain**. A row whose merkle root validates against the OSINT salt cannot validate against the Healthcare salt, so a leaked row is provably mis-routed at integrity-check time even if the predicate filter is misconfigured.

**Updated `AuditEntry` shape** (Tier D refinement):

```rust
pub struct AuditEntry {
    pub tenant:           TenantId,
    pub super_domain:     SuperDomain,
    pub actor_role:       &'static str,
    pub owl:              OwlIdentity,
    pub op:               u8,                 // PermissionSet bit
    pub merkle_root:      MerkleRoot,         // NEW: fingerprint at access time
    pub clam_path:        ClamPath,           // NEW: hierarchical DN address
    pub timestamp:        u64,
    pub super_domain_salt: u64,                // NEW: per-super-domain merkle salt
}
```

### 13.4 Hard-lock requirement: Healthcare ↔ OSINT crypto barrier

**HIPAA compliance and clinical staff trust require a guarantee stronger than predicate filtering between patient history and OSINT.** The user's framing: "doctors will want to know that patient history and OSINT are hard lock."

**Updated DTO:**

```rust
pub struct SuperDomainEntry {
    // ... fields as in §3.4 ...
    pub merkle_salt:        u64,                          // NEW: per-super-domain integrity salt
    pub hard_lock_partners: &'static [SuperDomain],       // NEW: explicit cryptographic separation
}
```

**Per-super-domain hard-lock matrix (initial):**

| Super domain | `hard_lock_partners` (cannot share rows or be queried jointly under any role) |
|---|---|
| `Healthcare` | `[OSINT]` |
| `OSINT` | `[Healthcare]` |
| `WorkOrderBilling` | `[OSINT]` (financial confidentiality) |
| `Science` (when ITAR-tagged) | `[OSINT]` (export control vs intel) |

**Enforcement mechanism (3 layers of defense):**

1. **Predicate-time:** `authorize()` rejects any query whose `super_domain_target` is in the actor's `hard_lock_partners` list, even if the actor has the source super domain authorized.
2. **Integrity-time:** different `merkle_salt` per super domain. A misconfigured query that bypasses (1) cannot validate cross-domain merkle roots.
3. **Encryption-time:** rows in a hard-locked super domain are encrypted with super-domain-scoped key derivation (per-tenant DEK × per-super-domain HKDF info string). A leaked row decrypts only with both the tenant DEK *and* the super-domain context — neither alone suffices.

**Sales narrative refresh:**

> "Patient history and OSINT are hard-locked. Three layers of defense — predicate, merkle salt, key derivation. A clinician's bridge cannot construct a query that joins patient records with intel; the optimizer rejects it, the merkle would not validate, and the encryption keys won't combine. HIPAA reviewers see a cryptographically attested separation, not a policy promise."

### 13.5 Research role: anonymized projection only

**The `researcher` role from §4.3 is upgraded to a hard requirement, not a configuration knob.** Per user: "research using anonymized."

**Updated `researcher` role definition:**

```rust
RoleGroup {
    role_name:       "researcher",
    permissions:     PermissionSet(PermissionSet::READ),  // no WRITE, no EXPORT, no REDACT_LIFT
    clearance_floor: ClearanceLevel(1),                   // Restricted; never elevated
    audit_required:  true,                                 // every access logged with k-anonymity check
    redaction_mask:  FieldRedactionMask {
        readable_slots: BIT_SET_DEIDENTIFIED_ONLY,   // only de-identified slots visible
        writable_slots: BitSet256([0; 4]),            // empty — researchers never write
        redacted_slots: BIT_SET_DIRECT_IDENTIFIERS,   // name, SSN, DOB, MRN, address — all hashed
    },
},
```

**Composes with `PolicyKind::DifferentialPrivacy`:** when the researcher role queries an aggregate, the optimizer chain auto-injects DP noise per the differential-privacy parameter `ε` configured at the super-domain level (`SuperDomainEntry.dp_epsilon: f32`, NEW field).

**Three additive constraints for the researcher role:**

1. **Field-level:** direct identifiers always hashed (k-anonymity-style pseudonymization).
2. **Row-level:** queries over <k=5 rows error out with `RbacError::KAnonymityViolation` rather than returning thin slices.
3. **Aggregate-level:** when the federated-aggregation gate (§13.2) is enabled, cross-tenant aggregates pass through the encrypted-view path — researcher never sees any tenant's raw values, even pseudonymized.

### 13.6 Net architecture diff vs §1-§12

| Aspect | §1-§12 baseline | §13 refinement |
|---|---|---|
| Enforcement mechanism | New 4-stage `authorize()` | Composes onto shipped `PolicyRewriter` chain in `lance-graph-callcenter::policy` |
| Federation | A+B accepted, C deferred to 2027+ | A+B+C all accepted; C uses LanceDB transparent encrypted view |
| Audit format | TBD (open question) | `AuditEntry` carries `MerkleRoot + ClamPath + super_domain_salt`; tamper-detection built in |
| Cross-domain leakage | Predicate filter + per-tenant DEK | + per-super-domain merkle salt + super-domain-scoped key derivation = 3 layers |
| Researcher role | Optional configuration | Hard requirement: anonymized projection + k-anonymity floor + DP noise on aggregates |
| Hard-lock pairs | Not specified | Healthcare ↔ OSINT, WorkOrderBilling ↔ OSINT, Science(ITAR) ↔ OSINT |

### 13.7 Updated open questions (§10 carry-over + new)

- ~~**Audit format choice**~~ — RESOLVED in §13.3: `AuditEntry` shape with merkle + ClamPath + salt. JSON Lines for serialization, OTel optional bridge.
- ~~**Cross-tenant federation**~~ — RESOLVED in §13.2: A+B+C all accepted.
- **Hard-lock partner matrix** — confirm the initial 4 pairs in §13.4 are correct; any additional pairs (e.g., Genetics ↔ OSINT?) before locking.
- **Per-super-domain DP epsilon defaults** — `dp_epsilon` per super domain (Healthcare = 1.0? OSINT = 0.1?) needs statistician-level review.
- **Merkle salt rotation** — quarterly per-super-domain salt rotation for audit-chain unforgeability; aligns with DEK rotation cadence.
- **K-anonymity floor for `researcher`** — k=5 default; per-super-domain override needed (Healthcare typically k=10 for rare-condition research).

### 13.8 Tier additions

- **D-SDR-13** — `merkle_salt` field on `SuperDomainEntry` + per-super-domain HKDF context derivation in `TenantContext::encryption_key`. ~80 LOC + 4 integration tests covering hard-lock crypto barrier.
- **D-SDR-14** — `AuditEntry` updated schema (merkle + ClamPath + salt) + `JsonLinesAuditSink` impl that includes integrity verification on replay. ~120 LOC + 6 tests including post-hoc tamper detection.
- **D-SDR-15** — `PolicyKind::DifferentialPrivacy` activation for `researcher` role: aggregate-only enforcement + ε-bounded noise injection + k-anonymity floor check. ~150 LOC + 5 tests.
- **D-SDR-16** — `EncryptedViewAggregate` federation policy: LanceDB transparent encrypted view bridge for cross-tenant aggregate. ~200 LOC + 4 integration tests against an actual encrypted column.
- **D-SDR-17** — Hard-lock partner matrix as static table + predicate-time enforcement in `authorize()`. ~60 LOC + 4 tests covering each documented pair.

**Status:** Refinements are additive to the §1-§12 architecture. No prior DTO removed; all existing fields stay. Merkle/audit/hard-lock weave through the existing 4-stage flow as policy-rewriter composition rather than parallel paths.

---

## 14 — Harvest + Templates + Cross-Language Migration (2026-05-13, same session)

The meta-bridge is **extracted from shipped code, not designed clean-room.** Three-step rhythm:

### 14.1 Harvest

`crates/lance-graph-ontology/src/bridges/medcare_bridge.rs` (Healthcare, HIPAA-aware) and `sharepoint_bridge.rs` / `smb_bridge.rs` (WorkOrderBilling, SMB-shaped) are the **canonical pattern source**. Both have absorbed real consumer constraints — HIPAA boundary handling, SharePoint auth quirks — that a clean-room rewrite would lose.

### 14.2 Templates (NEW + retrofit bridges)

| Bridge | Status | Pattern source | Notes |
|---|---|---|---|
| `woa_bridge.rs` | EXISTS — retrofit | mirrors medcare structure | smallest delta — lift to meta-bridge surface |
| `hubspot_bridge.rs` | NEW | medcare (HIPAA-style boundary) | touches PCI-DSS billing slots, mirrors PHI discipline |
| `hiro_bridge.rs` | NEW | sharepoint (off-label tolerance) | absorbs OSLC-* with provenance lineage |

Each ~45 LOC after the meta-pattern is extracted.

### 14.3 Migration of existing consumers

- `MedCare-rs` → retrofit to meta-bridge (lowest risk — it's the harvest source)
- `smb-office-rs` → retrofit (same)
- `MedCareV2 C#` → reshape freely (per §16/§17, MedCareV2 is a partial rewrite, not a retrofit constraint)

### 14.4 Tier F deliverables

- **D-SDR-18** — Archaeology pass: `git log -p` `medcare_bridge.rs` + `sharepoint_bridge.rs` + `woa_bridge.rs`, extract fix-commits as named tests in `meta_bridge::tests`. Captures latent fixes that aren't documented as patterns. ~1 day.
- **D-SDR-19** — `MetaBridge` trait + `BridgeFromRegistry` extension absorbing the harvested patterns. ~150 LOC.
- **D-SDR-20** — **SUPERSEDED by §17.2 — see Arrow Flight SQL endpoint instead of custom Protobuf IDL.**

### 14.5 Tier G migrations

- **D-SDR-21** — `MedCare-rs` retrofit to `MetaBridge` (zero behavior change).
- **D-SDR-22** — `smb-office-rs` retrofit (zero behavior change).
- **D-SDR-23** — `MedCareV2 C#` aligned to `MetaBridge` via Arrow Flight SQL client (per §17).

---

## 15 — Multi-Implementation Drift Detection (2026-05-13, same session)

> **Note:** This section's framing was substantially refined by §16 (Zone 3 boundary placement) and §17 (LanceDB convergence). Read §16+§17 as the controlling architecture; §15 is preserved here as the design arc that surfaced the byte-determinism concerns.

### 15.1 Three implementations under one contract

```rust
pub struct MetaBridgeVersion {
    pub major: u8,    // wire-incompatible
    pub minor: u8,    // wire-compatible additions
    pub patch: u8,    // bug-fix (allowed to differ between drift partners)
}

#[repr(u8)]
pub enum BridgeImpl {
    RustNative   = 0,    // MedCare-rs (authoritative)
    CSharpNative = 1,    // MedCareV2 (parallelbetrieb during migration window)
    MySQLAdapter = 2,    // legacy adapter — DROPPED in §17 (one-shot import only)
}

pub struct DriftableOutput {
    pub version:        MetaBridgeVersion,
    pub source:         BridgeImpl,
    pub owl_identities: Vec<OwlIdentity>,   // sorted ascending by raw u16
    pub fingerprints:   Vec<MerkleRoot>,    // parallel to owl_identities
    pub canonical_hash: u64,                // FNV-1a over interleaved (id, merkle) pairs
}

pub trait DriftDetectionBridge {
    fn compare(&self, query: &Query) -> DriftReport;
}

pub struct DriftReport {
    pub equivalent:        bool,
    pub canonical_hashes:  Vec<(BridgeImpl, u64)>,
    pub divergent_rows:    Vec<DivergentRow>,
    pub timestamp:         u64,
}

pub struct DivergentRow {
    pub owl:               OwlIdentity,
    pub per_impl_merkle:   Vec<(BridgeImpl, MerkleRoot)>,
    pub canonical_winner:  BridgeImpl,         // RustNative wins ties
}
```

### 15.2 Cross-language byte-determinism rules

| Source | Rust default | C# default | Determinism rule |
|---|---|---|---|
| Hash map iteration | randomized | insertion order | **Always sort** before producing `DriftableOutput` |
| f32/f64 summation | left-fold | left-fold | **Use Kahan summation** for any cross-impl FP aggregate |
| Timestamp granularity | `Instant` (nanoseconds) | `DateTime` (100 ns ticks) | **Quantize to milliseconds** before MerkleRoot |
| String hashing | `DefaultHasher` (SipHash) | `string.GetHashCode()` (random per process) | **Always FNV-1a** for cross-boundary hashes |
| Integer overflow | wrapping in release | checked in debug | **`wrapping_*` everywhere** in `DriftableOutput` derivation |
| Unicode normalization | NFC default | NFC default | **Assert NFC at boundary** |
| Decimal arithmetic | `f64` | `decimal` (128-bit) | **String-encoded decimals** for monetary slots |

**Reduction in §16:** Zone 3 placement (Arrow IPC + Supabase RPC) handles 4 of these 7 rules at the wire-format level. Remaining surface = decimal arithmetic + timestamps + side-channel aggregation.

### 15.3 Tier deliverables (§15 originals; revised by §16/§17)

- **D-SDR-24** — `MySQLAdapterBridge` impl. **DROPPED in §16/§17** — replaced by D-SDR-27 one-shot import; the MySQL adapter doesn't live alongside the new system long-term.
- **D-SDR-25** — `DriftDetectionBridge` impl + `JsonLinesDriftSink` for divergence events + dashboards. **Demoted in §17** to Phase-2 dual-write window only; retires after Phase 4 cutover.
- **D-SDR-26** — Determinism rule test suite. **Reduced in §16** from 12 rules to ~3 rules (decimal + timestamp + FP aggregate).

### 15.4 Brutal-honest tradeoff (preserved as design arc)

Original framing: byte-determinism is hard to retrofit; if existing `medcare_bridge.rs` produces non-deterministic output anywhere (HashMap iteration, FP order in vector ops), the harvest pass surfaces this as the highest-priority bug to fix before extracting the meta-bridge. Otherwise the drift bridge produces false-positive divergence events on identical inputs, destroying operational trust in the drift signal.

**Updated by §17:** Because the Phase 4 end-state is "single LanceDB store accessed via Flight SQL", the drift bridge is bounded to the Phase 2-3 migration window. Determinism failures become migration-window incidents (recoverable) rather than perpetual production drift (unrecoverable). 2-3 days of determinism archaeology stays as a budget item but the consequence severity drops.

---

## 16 — Zone 3 Drift Boundary + Two-Track Migration (2026-05-13, same session)

### 16.1 Zone 3 placement collapses determinism rules

MedCare-rs ingests via Zone 3 (Supabase RPC / REST / transcode per the workspace's zone discipline) — the drift comparison happens on serialized wire payloads (Arrow IPC + Supabase RPC), not on internal `BindSpace` state. Most §15.2 rules collapse:

| §15.2 rule | Status at Zone 3 |
|---|---|
| Hash map iteration | Resolved — Arrow column order is part of IPC framing |
| String hashing | Resolved — wire-level string identity is byte-equal |
| Unicode normalization | Already NFC-asserted at Zone 3 boundary |
| f32/f64 summation | Still applies if pre-Zone-3 aggregation occurs |
| Decimal arithmetic | Still applies for monetary slots |
| Timestamp granularity | Still applies |

Determinism test surface (D-SDR-26) shrinks from ~12 rules to ~3 rules.

### 16.2 Pre-prod posture (corrected per user clarification)

**Nothing is in production yet.** No live parallelbetrieb, no production drift to manage:

- Legacy `MedCare` (the original C# code, MySQL+3DES) — pre-prod
- `MedCareV2` (partial C# rewrite, reshapeable) — pre-prod
- `MedCare-rs` (Rust) — pre-prod
- Goal: single one-shot import of MySQL+3DES into the new stores; drift bridge demoted to import sanity check + CI gate

### 16.3 Two-track migration model

| Track | Payload | Volume | Migration shape |
|---|---|---|---|
| **John Doe** | billing + tickets (live in separate WoA / Hiro databases, NOT in MedCare MySQL) | ~95%+ of customer rows | Standard ETL through Zone 3. Maps to `hubspot_bridge` + `hiro_bridge`. No 3DES decrypt step. |
| **3DES PHI** | Healthcare clinical data (MedCare MySQL) | small high-stakes subset (specific PHI columns: name, DOB, contact, free-text diagnosis) | Decrypt 3DES → AES-256-GCM with per-tenant DEK × per-super-domain HKDF context. One-shot import tool. |

### 16.4 3DES rewrap pipeline (single cipher, well-known algorithm)

```rust
/// One-shot import of 3DES-encrypted MySQL rows into the new stores.
/// Runs once during pre-prod cutover; throwaway tool after.
pub fn import_3des_row(
    mysql_row:           &MySQLRow,
    legacy_3des_key:     &TripleDesKey,    // legacy key, destroyed after import
    target_tenant_dek:   &KeyHandle,
    target_super_domain: SuperDomain,
) -> Result<MigratedRow> {
    // 1. Decrypt 3DES (well-known algorithm, no surprises)
    let plaintext = three_des_decrypt(&mysql_row.ciphertext, legacy_3des_key)?;

    // 2. Compute MerkleRoot for drift-bridge sanity check
    let merkle = MerkleRoot::from_fingerprint(&Fingerprint::from(&plaintext));

    // 3. Re-encrypt with per-tenant × per-super-domain AES-256-GCM
    let derived = hkdf_derive(target_tenant_dek, target_super_domain.as_hkdf_context());
    let new_ciphertext = aes_256_gcm_encrypt(&plaintext, &derived, random_nonce())?;

    Ok(MigratedRow {
        owl:          classify_to_owl_identity(&plaintext),
        merkle_root:  merkle,                 // cleartext beside ciphertext
        ciphertext:   new_ciphertext,
        super_domain: target_super_domain,
    })
}
```

### 16.5 MerkleRoot-cleartext-beside-ciphertext (key insight)

Each row stores `(ciphertext, merkle_root_cleartext)` side-by-side. Consequences:

- **Drift bridge compares MerkleRoots without ever decrypting** in steady-state production. No privileged plaintext access role needed for drift detection itself.
- Encryption uses **random nonces** (modern best practice) — no need for deterministic AEAD like AES-GCM-SIV.
- MerkleRoot is computed against fingerprint content (the `Fingerprint` type from `crates/lance-graph/src/graph/fingerprint.rs`), so it's a one-way hash — doesn't reveal plaintext, only enables equality comparison.
- **Trust boundary:** anyone who can write a row needs key access (to encrypt). Anyone who can run drift detection only needs MerkleRoot read access. Read-only drift role is unprivileged from PHI perspective.

### 16.6 Tier F additions (replaces §15's drift-as-production-infra deliverables)

- **D-SDR-27** — 3DES → AES-256-GCM rewrap one-shot tool. ~120 LOC + 2 integration tests against 3DES test vectors. **Throwaway after import succeeds + sanity-check window.**
- **D-SDR-28** — MerkleRoot-cleartext-beside-ciphertext storage layout in MedCare-rs + MedCareV2 stores. ~80 LOC × 2 + 2 tests verifying MerkleRoot consistency across the boundary.
- **D-SDR-29** — Two-track import runner: PHI rows (3DES) → D-SDR-27, John Doe rows → standard transcode. ~100 LOC + 2 tests.
- **D-SDR-30** — Legacy 3DES key handling: hold under audit during import; **destroy on successful import + sanity-check verification**. ~30 LOC + 1-page governance doc.

### 16.7 MedCare MySQL Struktur reality check

The actual schema (`MedCare-rs/.MYSQL/Struktur.sql`, 90 KB, 104 tables) confirms the architecture:

| Prefix | Count | Role |
|---|---|---|
| `pf_*` | 38 | Patient file (diagnosis, allergy, vaccination, vital, lab, operation, therapy) |
| `combo_*` | 29 | Lookup dimensions (anrede=salutation, rasse=race, spez=specialty, drugs, morbidity) |
| `praxis_*` | 12 | Clinic config (patient, mitarbeiter=staff, license, waitingroom) |
| `glob_*` | 9 | Global (formular, mail, update, user_right) |
| `pat_*`, `file_*`, `pdf_*` | 4+4+2 | Patient combinations, attachments, exports |
| Other | 6 | tpk, customerdb, f_2_3, etc. |

**Schema observations:**
- All columns are `VARCHAR(250)` / `TEXT` / `DATETIME` — no `BLOB` or `VARBINARY`. **3DES encryption is application-layer, not at-rest in column types.** Encrypted columns hold opaque base64-encoded ciphertext as strings.
- MySQL TDE is OFF (`DEFAULT ENCRYPTION='N'`).
- Schema is purely clinical — billing/tickets are NOT in this database; they live in WoA / Hiro / HubSpot schemas separately.
- The 38 `pf_*` tables are SUBSTRUCTURE of one Patient entity (allergies, vitals, diagnoses), not 38 distinct OGIT entities.
- Healthcare basin slot count estimate: **~5-8 core clinical entities + ~10-15 verbs + ~29 OWL enumerations rolled into slots = ~30-50 slots used**, well within the 256-slot per-basin ceiling.

**D-SDR-27 implementation gap:** the one-shot import tool needs MedCare's C# app source for the column→3DES key mapping (which specific VARCHAR columns are 3DES-encrypted with which derived key). Pending MCP scope expansion to `AdaWorldAPI/MedCare` + `AdaWorldAPI/MedCareV2`.

---

## 17 — DataFusion SQL inside LanceDB as unified persistence + access (2026-05-13, same session)

### 17.1 Convergence architecture

```
              ┌────────────────────────────────┐
              │  LanceDB (single source)       │
              │  + DataFusion logical plans    │
              └──────────────┬─────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
       ┌────────▼────────┐       ┌────────▼────────┐
       │  MedCare-rs     │       │  MedCareV2 C#   │
       │  (in-process    │       │  (Arrow Flight  │
       │   DataFusion)   │       │   SQL gRPC)     │
       └─────────────────┘       └─────────────────┘
```

### 17.2 Phase sequencing

| Phase | State | Drift bridge role |
|---|---|---|
| 0 (now) | MySQL+3DES is source. MedCareV2 + MedCare-rs both pre-prod. | n/a — no live drift |
| 1 | D-SDR-27..30 import: MySQL+3DES → LanceDB (single-shot, throwaway). | n/a — one-time import sanity check |
| 2 | MedCare-rs reads/writes LanceDB natively. MedCareV2 C# dual-writes via Flight SQL. | Active: compare Arrow batches across paths |
| 3 | Drift-clean window confirms parity. | Active: gate cutover decision |
| 4 | MedCareV2's prior store retires. LanceDB = single source of truth. | Retires — drift bridge becomes CI gate only |

### 17.3 Arrow Flight SQL replaces custom Protobuf IDL (D-SDR-20 resolution)

Earlier draft proposed a custom Protobuf IDL with Rust prost + C# Grpc.Tools. **Redundant** — Arrow Flight SQL already has a defined gRPC schema (`Flight.proto` + Substrait for the logical plan layer). The DTOs from §3 (`OgitFamily`, `OwlIdentity`, `RoleGroup`, etc.) become **Arrow column metadata + Substrait extension types**, not bespoke message definitions.

### 17.4 Drift detection at Flight SQL boundary

```rust
/// Drift comparison at the Flight SQL boundary.
/// Both clients fetch the same query result; we compare MerkleRoots
/// of the Arrow batches they each received and processed.
pub struct ArrowBatchDriftSignal {
    pub query_hash:        u64,                            // FNV-1a of canonical query string
    pub per_client_merkle: Vec<(BridgeImpl, MerkleRoot)>,  // one per client implementation
    pub batch_row_count:   u64,
    pub timestamp:         u64,
}
```

Drift can only come from:
- **Client serialization** — Arrow IPC is byte-deterministic; rare
- **Post-deserialization processing** — MedCareV2 C# applying a transform MedCare-rs doesn't; this IS the drift surface
- **Logical-plan divergence** — only if MedCareV2 builds plans on its own; if it just submits raw SQL/Cypher, this evaporates

### 17.5 Tier F replacements (supersedes §15+§16 drift infrastructure)

- **D-SDR-31** — Arrow Flight SQL endpoint server in `lance-graph`. Wraps the existing DataFusion catalog. ~150 LOC + 4 integration tests. Replaces D-SDR-20 custom-IDL plan.
- **D-SDR-32** — C# `Apache.Arrow.Flight.Sql` client wrapper for MedCareV2 with auth header carrying `TenantContext.encryption_key` handle. ~80 LOC + 2 tests against the D-SDR-31 server. Lands inside MedCareV2's source tree.
- **D-SDR-33** — Substrait extension types for `OwlIdentity` + `MerkleRoot` + `SuperDomain` so plans containing these types serialize cleanly across the Flight SQL wire. ~120 LOC + 3 tests.
- **D-SDR-34** — Phase-2 dual-write coordination: `DriftDetectionBridge` operates against Flight SQL `GetFlightInfo` + `DoGet` to compare per-client MerkleRoots over identical queries. ~150 LOC + 4 integration tests covering the Phase 2 dual-write window.

### 17.6 Dropped from earlier scope

- ~~Custom Protobuf IDL (D-SDR-20 original)~~ — Arrow Flight SQL handles the wire layer
- ~~Multi-trustee key escrow (D-SDR-30 expanded)~~ — no live system to coordinate; one-shot 3DES import retires keys cleanly
- ~~Persistent production drift infrastructure~~ — drift bridge bounded to Phase 2-3 cutover window
- ~~`MySQLAdapterBridge` (D-SDR-24)~~ — replaced by D-SDR-27 one-shot import
- ~~C-ABI FFI option for hot-path C# access~~ — Flight SQL is the single bridge
- ~~"3 unknown ciphers" question~~ — single 3DES, single decryption step

### 17.7 Net architecture summary (after §13 + §14 + §15 + §16 + §17)

```
Per-row data:
    LanceDB row carries: tenant_id u32 + owl_id u16 + ciphertext + merkle_root (cleartext)
    Total per-row identity: 6 bytes; merkle ~8 bytes; payload variable.

Access layer:
    Cypher / SPARQL / Gremlin / SQL → DataFusion logical plan → LanceDB scan
    Both clients (Rust direct, C# via Flight SQL) hit the same plan layer.

RBAC enforcement:
    PolicyRewriter chain in lance-graph-callcenter::policy
    (RowFilter + ColumnMask + RowEncryption + DifferentialPrivacy + Audit)
    composed by UnifiedBridge::authorize() 4-stage flow.

Drift detection:
    Phase 2-3 only. Compares ArrowBatchDriftSignal across clients.
    Retires after Phase 4 cutover; demoted to CI gate.

Migration:
    One-shot D-SDR-27..30 (MySQL+3DES → LanceDB).
    Throwaway tool. 3DES keys destroyed on completion.

Cross-language surface:
    Arrow Flight SQL gRPC (no custom IDL).
    Substrait extension types for OwlIdentity + MerkleRoot + SuperDomain.

Bridge harvest:
    medcare_bridge.rs + sharepoint_bridge.rs as canonical pattern source.
    woa_bridge.rs retrofit; hiro_bridge.rs + hubspot_bridge.rs new templates.
    ~45 LOC each after MetaBridge extraction.
```

### 17.8 Open questions deferred to §18 (pending MCP scope expansion)

- 3DES column inventory for `MedCare` (which VARCHAR columns are encrypted with which derived key)
- Transcoded shape that already accounts for SQL import + app-layer decryption per user note
- Final scope of D-SDR-27 implementation (depends on column inventory)

These unblock once `AdaWorldAPI/MedCare` + `AdaWorldAPI/MedCareV2` are in scope. §18 will fold the findings in as a follow-up commit.
