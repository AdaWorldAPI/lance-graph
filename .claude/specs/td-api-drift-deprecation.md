# TD-API-DRIFT-MIDFLIGHT-1 — D-SDR-1..5 API Drift: Deprecation & Consumer Migration Playbook

**Status:** Spec (not yet implemented)
**Priority:** P0 — blocks medcare-rs and smb-office-rs from completing their UnifiedBridge migrations
**TD-ID:** TD-API-DRIFT-MIDFLIGHT-1
**Author:** W3 (sprint-log-4, 2026-05-13)
**Cross-ref W7:** td-sdr-pr-release.md (owns the PR sequencing + release window timing)

---

## 1. Problem Statement

D-SDR-1 through D-SDR-5 (PRs #355–#363) shipped as a batch of consecutive commits on `main`. Both medcare-rs (`MedcareBridge`) and smb-office-rs (`SmbMembraneGate`) had in-flight local wiring commits that were authored against the pre-D-SDR surfaces. As of 2026-05-13 those consumer commits are locally present but NOT pushed (`TD-SDR-CONSUMER-PUSH-1`). The combination means:

1. Consumer local commits reference symbols that either no longer exist (renamed), have changed signatures, or have widened/narrowed types.
2. The consumers cannot simply rebase — the merge-conflict surface is spread across 5 PRs of structural changes (policy evaluation path, identity types, codebook shape, audit chain builder, authorize wiring).
3. Without a compat shim layer, every consumer must do a full manual migration in a single atomic commit, which is error-prone and blocks the FMA smoke-test integration.

**Solution:** Ship a one-release-cycle deprecation bridge (`lance_graph_contract::compat_v0_4`) that re-exports old-shaped aliases so consumers can migrate incrementally rather than all-at-once.

---

## 2. Drift Catalogue — The 5 Breaking API Surfaces

### 2.1 Policy::evaluate / PolicyRewriter::canonicalize (D-SDR-1)

**Pre-D-SDR shape (consumers authored against):**

```rust
// In lance-graph-rbac (pre-D-SDR-1): policy evaluates against the
// bridge-side public_name directly — no canonical resolution step.
policy.evaluate(actor_role, public_name, operation)
// → AccessDecision
```

**Post-D-SDR-5 shape (what main now ships):**

```rust
// evaluate() still has the same Rust signature, but the SEMANTIC CONTRACT changed:
// the entity_type argument must now be the CANONICAL OGIT URI name part
// (e.g. "Order" for ogit.WorkOrder:Order), NOT the bridge-side alias.
// UnifiedBridge::authorize_*() performs the canonical_entity_type() resolution
// before calling evaluate(). Consumer code calling policy.evaluate() directly
// with a bridge alias now silently fails authorization.
policy.evaluate(actor_role, canonical_entity_type, operation)
// → AccessDecision
```

**Drift type:** Silent behavioral change — compiles but authorizes incorrectly if the consumer passes `public_name` instead of the canonical OGIT name. Consumers that called `policy.evaluate()` directly (bypassing `UnifiedBridge`) are broken.

**Shim path:** `compat_v0_4::policy_evaluate_by_alias(policy, role, alias, op)` — wraps the registry lookup and calls the canonical form. Deprecated at compile time.

---

### 2.2 BridgeError variants (D-SDR-2)

**Pre-D-SDR-2 shape:**

```rust
// BridgeError had only two variants in early code:
pub enum BridgeError {
    NotFound { bridge_id: &'static str, name: String },
    ScopeLeak { bridge_id: &'static str },
}
```

**Post-D-SDR-2 shape (current in `lance-graph-ontology/src/bridge.rs`):**

```rust
pub enum BridgeError {
    NamespaceMissing { bridge_id: &'static str, namespace: &'static str },
    NotInScope { bridge_id: &'static str, public_name: String },
    CrossNamespaceLeak { bridge_id: &'static str, resolved_id: NamespaceId, locked_id: NamespaceId },
}
```

**Drift type:** Hard compile error — `NotFound` and `ScopeLeak` match arms in consumer `match` blocks don't exist. `CrossNamespaceLeak` now carries two `NamespaceId` fields instead of one `bridge_id`.

**Shim path:** A `from_legacy_match!(err)` macro + type aliases in `compat_v0_4` that let consumer match blocks compile while emitting deprecation warnings:

```rust
// In compat_v0_4:
#[deprecated(since = "0.4.0", note = "use BridgeError::NotInScope; removed in 0.5")]
pub const fn bridge_error_not_found(bridge_id: &'static str, name: String) -> BridgeError {
    BridgeError::NotInScope { bridge_id, public_name: name }
}
#[deprecated(since = "0.4.0", note = "use BridgeError::CrossNamespaceLeak; removed in 0.5")]
pub fn bridge_error_scope_leak(bridge_id: &'static str) -> BridgeError {
    // Can't recover NamespaceIds from old shape — use sentinel values
    BridgeError::CrossNamespaceLeak {
        bridge_id,
        resolved_id: NamespaceId(u16::MAX),
        locked_id: NamespaceId(0),
    }
}
```

---

### 2.3 AuditChainBuilder methods (D-SDR-4)

**Pre-D-SDR-4 shape (what consumers expected):**

Consumers expected an `AuditChainBuilder` struct with a fluent builder pattern:
```rust
AuditChainBuilder::new()
    .super_domain(SuperDomain::Healthcare)
    .salt(HIPAA_SALT)
    .sink(JsonLinesAuditSink::new(path))
    .build()
```

**Post-D-SDR-4 shape (what shipped):**

No `AuditChainBuilder` struct exists. Instead, `UnifiedBridge::with_audit_chain()` is a builder method on `UnifiedBridge` itself:
```rust
UnifiedBridge::new(bridge, policy, actor_role, tenant_id)
    .with_audit_chain(super_domain, salt, sink)
// OR for resume:
    .with_audit_chain_resume(super_domain, salt, last_root, sink)
```

The `AuditChain` struct is internal (`unified_audit.rs`) and not part of the public consumer surface.

**Drift type:** Symbol-not-found compile error — `AuditChainBuilder` doesn't exist. The builder pattern moved into `UnifiedBridge`.

**Shim path:**
```rust
// In compat_v0_4:
#[deprecated(since = "0.4.0", note = "use UnifiedBridge::with_audit_chain(); removed in 0.5")]
pub struct AuditChainBuilder {
    super_domain: SuperDomain,
    salt: u64,
    sink: Option<Arc<dyn UnifiedAuditSink>>,
}
impl AuditChainBuilder {
    pub fn new() -> Self { Self { super_domain: SuperDomain::Unknown, salt: 0, sink: None } }
    pub fn super_domain(mut self, sd: SuperDomain) -> Self { self.super_domain = sd; self }
    pub fn salt(mut self, s: u64) -> Self { self.salt = s; self }
    pub fn sink(mut self, sink: Arc<dyn UnifiedAuditSink>) -> Self { self.sink = Some(sink); self }
    pub fn build(self) -> AuditChainConfig {
        AuditChainConfig { super_domain: self.super_domain, salt: self.salt, sink: self.sink }
    }
}
pub struct AuditChainConfig {
    pub super_domain: SuperDomain,
    pub salt: u64,
    pub sink: Option<Arc<dyn UnifiedAuditSink>>,
}
pub fn apply_audit_chain_config<B: NamespaceBridge>(
    bridge: UnifiedBridge<B>,
    cfg: AuditChainConfig,
) -> UnifiedBridge<B> {
    let sink = cfg.sink.unwrap_or_else(|| Arc::new(NoopUnifiedAuditSink));
    bridge.with_audit_chain(cfg.super_domain, cfg.salt, sink)
}
```

---

### 2.4 OwlIdentity slot width (D-SDR-2: 8-bit → 16-bit addressable type)

**Pre-D-SDR-2 shape:**

`OwlIdentity` was a single `u8` — the entire identity fit in one byte:
```rust
pub struct OwlIdentity(pub u8); // 8-bit, no family field
```

**Post-D-SDR-2 shape (current in `unified_bridge.rs`):**

```rust
pub struct OwlIdentity(pub u16); // 16-bit: high 8 = OgitFamily, low 8 = slot
impl OwlIdentity {
    pub const fn new(family: OgitFamily, slot: u8) -> Self { ... }
    pub const fn family(self) -> OgitFamily { ... }
    pub const fn slot(self) -> u8 { ... }
}
```

**Known truncation debt:** `owl_from_schema_ptr()` currently truncates the 16-bit `entity_type_id` from `SchemaPtr` to 8 bits for the slot field. This is documented as `TD-SDR-SLOT-TRUNC-1` and tracked by W10.

**Drift type:** Hard compile error — `OwlIdentity(raw_u8)` constructor no longer compiles; `OwlIdentity::new(family, slot)` is the replacement.

**Shim path:**
```rust
// In compat_v0_4:
#[deprecated(since = "0.4.0", note = "use OwlIdentity::new(OgitFamily(N), slot); removed in 0.5")]
pub fn owl_identity_from_u8(slot: u8) -> OwlIdentity {
    // family=UNKNOWN(0); consumer must update to real domain family constant
    OwlIdentity::new(OgitFamily(0), slot)
}
```

---

### 2.5 FamilyEntry shape (D-SDR-3: label-only → full inline codebook)

**Pre-D-SDR-3 shape (consumers expected):**

```rust
pub struct FamilyEntry {
    pub label: String,       // owned String, heap-allocated
    pub schema_kind: u8,     // raw discriminant, not SchemaKind enum
}
```

**Post-D-SDR-3 shape (current in `family_table.rs`):**

```rust
pub struct FamilyEntry {
    pub label_uri: &'static str,          // interned — no heap alloc
    pub kind: SchemaKind,                 // typed enum (Entity/Edge/Attribute)
    pub owl_characteristics: OwlCharacteristics,  // 1-byte bitfield (NEW)
    pub dolce_marker: DolceMarker,        // upper-ontology marker (NEW)
    pub axiom_blob: &'static [u8],        // OWL axiom bytes (NEW)
    pub provenance: &'static str,         // dcterms:source (NEW)
    pub verbs: &'static [u8],             // outgoing verb slot list (NEW)
}
```

**Drift type:** Hard compile error — field names changed (`label` → `label_uri`), type changed (`String` → `&'static str`), `schema_kind: u8` replaced by `kind: SchemaKind`. 5 new fields have no legacy counterpart.

**Shim path:**
```rust
// In compat_v0_4:
#[deprecated(since = "0.4.0", note = "use FamilyEntry directly; removed in 0.5")]
pub struct LegacyFamilyEntry {
    pub label: String,
    pub schema_kind: u8,
}
impl From<&FamilyEntry> for LegacyFamilyEntry {
    fn from(e: &FamilyEntry) -> Self {
        Self { label: e.label_uri.to_string(), schema_kind: e.kind as u8 }
    }
}
```

---

## 3. Deprecation Policy

All shims follow a single uniform policy:

```rust
#[deprecated(
    since = "0.4.0",
    note = "use <new_symbol>; this shim will be removed in 0.5.0 (30-day window from 2026-05-13)"
)]
```

**Rules:**
1. **One-release-cycle window.** Shims ship in 0.4.0 (`D-SDR-PR-FOLLOWUP-1`). Removal is in 0.5.0, no sooner than 30 calendar days after 0.4.0 release. The 0.5.0 release date is gated on W7's release plan.
2. **Shims are re-exports or thin wrappers only.** No logic beyond the minimal type conversion needed to bridge old shape → new shape. Shims must not introduce new allocation if the canonical form avoids it.
3. **`#[allow(deprecated)]` is NOT permitted in consumer crates after the 30-day window.** CI lint enforces this via `#![deny(deprecated)]` added as a required step at migration end (see Section 6).
4. **Shims are never back-ported to `main` patches.** They exist only in the 0.4.x minor cycle.
5. **Every shim entry has a calendar-dated auto-deletion comment:** `// AUTO-DELETE: 2026-06-12 (30 days after 0.4.0)`.

---

## 4. Compat Shim Layer

**Module:** `lance_graph_contract::compat_v0_4`

Primary location: `crates/lance-graph-contract/src/compat_v0_4.rs` (zero-dep crate).

Re-export locations for convenience:
- `crates/lance-graph-callcenter/src/compat.rs` — callcenter-specific shims
- `crates/lance-graph-ontology/src/compat.rs` — ontology-specific shims

**Feature flag:**
```toml
# lance-graph-contract/Cargo.toml
[features]
default = ["compat_v0_4"]
compat_v0_4 = []  # enables compat_v0_4 module; remove from default in 0.5.0
```

**CI lint for auto-deletion:**
```yaml
# .github/workflows/compat-shim-expiry.yml
name: compat-shim-expiry
on: [push, pull_request]
jobs:
  check-expiry:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Fail if shim module exists past expiry date
        run: |
          EXPIRY="2026-06-12"
          TODAY=$(date +%Y-%m-%d)
          if [ -f "crates/lance-graph-contract/src/compat_v0_4.rs" ] && [[ "$TODAY" > "$EXPIRY" ]]; then
            echo "ERROR: compat_v0_4 past expiry $EXPIRY. Delete compat modules and release 0.5.0."
            exit 1
          fi
```

---

## 5. Per-Consumer Migration Table

> Note: medcare-rs and smb-office-rs sources are not in this repo. File paths and line estimates are based on known naming conventions and the consumer wiring pattern from sprint context.

### 5.1 medcare-rs (MedcareBridge)

| Old Symbol | New Symbol | Likely File | Est. Touch-LOC |
|---|---|---|---|
| `BridgeError::NotFound { name }` | `BridgeError::NotInScope { bridge_id, public_name }` | `medcare-bridge/src/lib.rs` | ~8 |
| `BridgeError::ScopeLeak { .. }` | `BridgeError::CrossNamespaceLeak { bridge_id, resolved_id, locked_id }` | `medcare-bridge/src/lib.rs` | ~6 |
| `OwlIdentity(raw_u8)` | `OwlIdentity::new(OgitFamily(HEALTHCARE_FAMILY), slot)` | `medcare-bridge/src/lib.rs`, `*/src/policy.rs` | ~12 |
| `policy.evaluate(role, alias, op)` | `unified.authorize_read(alias, depth)` (canonical path) | `medcare-bridge/src/lib.rs`, `*/src/policy.rs` | ~20 |
| `AuditChainBuilder::new()...build()` | `UnifiedBridge::with_audit_chain(sd, salt, sink)` | `medcare-bridge/src/lib.rs` | ~15 |
| `FamilyEntry { label: String, schema_kind: u8 }` | `FamilyEntry::plain_entity(label_uri)` or full struct | `*/src/family.rs` (if exists) | ~10 |

**Total estimated:** ~71 LOC source + ~30 LOC test updates.

**Key files:** `medcare-bridge/src/lib.rs`, `medcare-bridge/src/policy.rs`, `*/src/membrane.rs`

### 5.2 smb-office-rs (SmbMembraneGate)

| Old Symbol | New Symbol | Likely File | Est. Touch-LOC |
|---|---|---|---|
| `BridgeError::NotFound { name }` | `BridgeError::NotInScope { bridge_id, public_name }` | `smb-office/src/membrane.rs` | ~8 |
| `BridgeError::ScopeLeak { .. }` | `BridgeError::CrossNamespaceLeak { .. }` | `smb-office/src/membrane.rs` | ~6 |
| `policy.evaluate(role, alias, op)` | `unified.authorize_*()` (canonical path) | `smb-office/src/membrane.rs`, `*/src/policy.rs` | ~25 |
| `OwlIdentity(raw_u8)` | `OwlIdentity::new(OgitFamily(SMB_FAMILY), slot)` | `smb-office/src/membrane.rs` | ~10 |
| `AuditChainBuilder::new()...build()` | `UnifiedBridge::with_audit_chain(...)` | `smb-office/src/membrane.rs` | ~15 |
| `FamilyEntry { label, schema_kind }` | `FamilyEntry::plain_entity()` | `smb-office/src/family.rs` (if exists) | ~8 |

**Total estimated:** ~72 LOC source + ~25 LOC test updates.

**Key files:** `smb-office/src/membrane.rs`, `smb-office/src/policy.rs`, `smb-office/src/lib.rs`

---

## 6. Cutover Sequence

### Step A — Ship D-SDR-PR-FOLLOWUP-1 with shim layer (W7 coordinates)

PR `D-SDR-PR-FOLLOWUP-1` adds:
1. `crates/lance-graph-contract/src/compat_v0_4.rs` — full shim module with all 5 shim groups
2. `crates/lance-graph-callcenter/src/compat.rs` — callcenter shim re-exports
3. `crates/lance-graph-ontology/src/compat.rs` — ontology shim re-exports
4. `.github/workflows/compat-shim-expiry.yml` — CI auto-deletion lint
5. `compat_v0_4` feature flag in `Cargo.toml` (default = on)

**Version bump:** `lance-graph-contract` → 0.4.0, `lance-graph-callcenter` → 0.4.0, `lance-graph-ontology` → 0.4.0.

### Step B — Consumer crates bump dep + flip imports to shim

**medcare-rs:**
```toml
# Cargo.toml
lance-graph-contract = { version = "0.4", features = ["compat_v0_4"] }
```
```rust
// medcare-bridge/src/lib.rs — add temporarily:
#![allow(deprecated)] // TEMPORARY: remove before 2026-06-12
use lance_graph_contract::compat_v0_4::{LegacyFamilyEntry, owl_identity_from_u8, AuditChainBuilder};
```

**smb-office-rs:** identical pattern.

**Estimated time per consumer:** 1 engineer-day.

### Step C — Consumer crates drop shims, use canonical symbols

Within the 30-day window (before 2026-06-12):

1. Remove `#![allow(deprecated)]` from crate root.
2. Replace `LegacyFamilyEntry` with `FamilyEntry::plain_entity()` or full struct literal.
3. Replace `owl_identity_from_u8(slot)` with `OwlIdentity::new(OgitFamily(DOMAIN_FAMILY), slot)`.
4. Replace direct `policy.evaluate()` calls with `unified.authorize_*()`.
5. Replace `AuditChainBuilder` with `UnifiedBridge::with_audit_chain()`.
6. Update `BridgeError` match arms to new variant shapes.
7. Drop `features = ["compat_v0_4"]` from `Cargo.toml`.
8. Run `cargo test` + integration tests.

**Step C PR shape per consumer:** ~100 LOC net change + ~55 LOC test updates.

### Step D — lance-graph removes shim module after 30 days (≥ 2026-06-12)

1. Delete `crates/lance-graph-contract/src/compat_v0_4.rs`.
2. Delete `crates/lance-graph-callcenter/src/compat.rs`.
3. Delete `crates/lance-graph-ontology/src/compat.rs`.
4. Remove `compat_v0_4` feature from all `Cargo.toml` files.
5. Bump `lance-graph-contract` → 0.5.0.
6. CI lint workflow auto-fails if shim file still exists (belt-and-suspenders).

---

## 7. Lint Rule — `#![deny(deprecated)]` After Migration

Once each consumer completes Step C (canonical migration complete, shim imports removed), add to CI:

```yaml
# In consumer CI workflow:
- name: Build (deny deprecated symbols)
  run: RUSTFLAGS="-D deprecated" cargo build --release
```

Or in consumer `.cargo/config.toml`:
```toml
[build]
rustflags = ["-D", "deprecated"]
```

**Timeline:** Add `#![deny(deprecated)]` CI step no later than end of Step C. If it slips past 2026-06-12, Step D's shim deletion will force the issue — any missed deprecated import causes an immediate build failure.

---

## 8. Cross-flag with W7 (PR Sequencing)

W7 owns `td-sdr-pr-release.md` — the release window and PR ordering. Concrete cross-flag items:

- [ ] **Confirm 0.4.0 release date** so the `AUTO-DELETE` comment in `compat_v0_4.rs` has the accurate calendar date (currently placeholder: 2026-06-12 = 30 days from today).
- [ ] **Confirm D-SDR-PR-FOLLOWUP-1 is first in follow-up wave** — compat layer must exist before consumers can bump dep to 0.4.0.
- [ ] **Confirm 0.5.0 is NOT released before 2026-06-12** — premature 0.5.0 would delete the compat window while consumers are still mid-migration.
- [ ] **Co-ship CI lint with shim** — `.github/workflows/compat-shim-expiry.yml` in the same PR as `compat_v0_4.rs`, not a separate follow-up.
- [ ] **Coordinate future consumers** (hubspot-rs, hiro-rs, woa-rs) — if any have pre-D-SDR wiring, they need to be on the same 30-day window, not granted extensions that delay Step D.

If W7's plan compresses the timeline below 30 days, the deprecation policy must be re-negotiated before Step A ships: the 30-day window is the contractual minimum for consumer teams to complete canonical migration safely.

---

## 9. Open Questions

1. **Additional consumers?** hubspot-rs, hiro-rs, and woa-rs are listed as future consumers. If any have pre-D-SDR experimental wiring commits, they need to be identified before Step D. The shim module must not be deleted while an unknown consumer depends on it.

2. **FamilyEntry `&'static str` for runtime codebooks.** Consumers constructing `FamilyEntry` at runtime (not from static baked TTL) cannot use `&'static str` without leaking memory. The shim's `LegacyFamilyEntry` works around this, but the canonical path may need a `FamilyEntry::runtime(label_uri: String)` constructor or an `OwnedFamilyEntry` companion type for non-static use cases.

3. **OgitFamily constants per consumer domain.** The shim `owl_identity_from_u8` falls back to `OgitFamily(0)` (UNKNOWN), which produces incorrect `OwlIdentity` values at runtime. Before consumers reach Step C, `HEALTHCARE_FAMILY: OgitFamily` and `SMB_FAMILY: OgitFamily` constants must be declared in `lance-graph-callcenter::super_domain` (or the per-domain sub-crate) so Step C migration can use the real family value.

4. **BridgeError persistence format.** If consumer crates persist `BridgeError` variant names to logs or event streams, the variant rename (`NotFound` → `NotInScope`, `ScopeLeak` → `CrossNamespaceLeak`) breaks deserialization of existing records. Is there a serde `#[serde(rename)]` shim needed, or is this out of scope for TD-API-DRIFT-MIDFLIGHT-1?

5. **TD-SDR-BRIDGE-ERR-AUDIT-1 interaction.** `BridgeError` currently short-circuits before audit emission by D-SDR-5 design. `AuthDecision::BridgeError` exists in the enum but is never emitted. If W10 wires `BridgeError` into audit emission in a follow-up, the shim's `bridge_error_not_found()` / `bridge_error_scope_leak()` helpers must also trigger the audit path — otherwise probe-detection signals are missing for legacy callers during the shim window. Coordinate with W10 before Step A ships.

---

## Implementation Checklist

- [ ] Create `crates/lance-graph-contract/src/compat_v0_4.rs` with all 5 shim groups
- [ ] Create `crates/lance-graph-callcenter/src/compat.rs` (callcenter re-exports + AuditChainBuilder shim)
- [ ] Create `crates/lance-graph-ontology/src/compat.rs` (ontology BridgeError shims)
- [ ] Add `compat_v0_4` feature flag to `lance-graph-contract/Cargo.toml` (default = on)
- [ ] Add CI lint `.github/workflows/compat-shim-expiry.yml`
- [ ] Confirm 0.4.0 release date with W7; update AUTO-DELETE comment
- [ ] Resolve Open Question #3 (OgitFamily constants) before consumer Step C
- [ ] medcare-rs Step B: bump dep + add `#![allow(deprecated)]` (~1 engineer-day)
- [ ] medcare-rs Step C: canonical migration (~100 LOC + tests, before 2026-06-12)
- [ ] smb-office-rs Step B: bump dep + add `#![allow(deprecated)]` (~1 engineer-day)
- [ ] smb-office-rs Step C: canonical migration (~100 LOC + tests, before 2026-06-12)
- [ ] Add `RUSTFLAGS="-D deprecated"` CI step to both consumer crates after Step C
- [ ] lance-graph Step D (≥ 2026-06-12): delete compat modules, bump to 0.5.0
