# Tech Debt Spec: TD-SDR-SLOT-TRUNC-1 + TD-SDR-BRIDGE-ERR-AUDIT-1

**Sprint:** sprint-log-4  
**Author:** W10-retry  
**Date:** 2026-05-13  
**Branch:** `claude/lance-datafusion-integration-gv0BF`  
**Status:** DRAFT — awaiting review by W12 (governance) and W8 (audit sink)

---

## Overview

This spec addresses two related technical debts in the `lance-graph-callcenter` crate:

1. **TD-SDR-SLOT-TRUNC-1** (P1): Silent truncation of `entity_type_id` from u16 to u8 in the unified bridge, capping OwlIdentity slots to 256 distinct values and silently colliding any id > 255.
2. **TD-SDR-BRIDGE-ERR-AUDIT-1** (P2): `BridgeError` short-circuits authorization paths via `?` before audit emission, making bridge failures invisible to probe-detection dashboards. The `AuthDecision::BridgeError` variant already exists in `unified_audit.rs` but is dead code.

Both items share a single deployment unit and can be landed in one PR.

---

## 1. Slot Truncation Fix (TD-SDR-SLOT-TRUNC-1)

### Problem

`crates/lance-graph-callcenter/src/unified_bridge.rs`, line 449:

```rust
// BEFORE (buggy)
let slot = (ptr.entity_type_id() & 0xFF) as u8;
OwlIdentity::new(family, slot)
```

`ptr.entity_type_id()` returns a `u16`. The bitmask `& 0xFF` combined with `as u8` silently discards the high byte. Any schema pointer with `entity_type_id >= 256` aliases onto one of the first 256 slots. For example, ids 256 and 0 both map to slot 0 — a security-relevant collision that bypasses row-level isolation.

`OwlIdentity::new(family, slot: u8)` enforces the 8-bit cap at the type level, meaning the compiler cannot catch the silent truncation at the call site.

### Root Cause

The original `OwlIdentity` type was designed when the schema had fewer than 256 entity types. The slot field was typed `u8` and the call site was never updated as `entity_type_id` was widened to `u16` in the schema layer.

### Fix

#### 1a. Widen `OwlIdentity::new()` to accept `slot: u16`

```rust
// crates/lance-graph-callcenter/src/owl_identity.rs

pub struct OwlIdentity {
    pub family: OgitFamily,
    pub slot: u16,   // widened from u8
}

impl OwlIdentity {
    pub fn new(family: OgitFamily, slot: u16) -> Self {
        Self { family, slot }
    }

    /// Deprecated shim for callers that have not yet migrated.
    /// Panics in debug builds if slot > 255 to surface hidden truncation.
    #[deprecated(since = "0.4.0", note = "Use new() with u16 slot directly")]
    pub fn slot_u8(&self) -> u8 {
        debug_assert!(
            self.slot <= 0xFF,
            "slot_u8() called on OwlIdentity with slot={} > 255; \
             this would silently truncate", self.slot
        );
        self.slot as u8
    }
}
```

The `#[deprecated]` shim allows a gradual migration for any callers that read back `slot` as `u8`. Callers that only write through `new()` are unaffected.

#### 1b. Fix the single call site in `owl_from_schema_ptr`

```rust
// crates/lance-graph-callcenter/src/unified_bridge.rs, around line 449

// BEFORE
let slot = (ptr.entity_type_id() & 0xFF) as u8;
OwlIdentity::new(family, slot)

// AFTER
OwlIdentity::new(family, ptr.entity_type_id())
```

No mask, no cast. The full u16 passes through.

#### 1c. Serialization / on-wire representation

`OwlIdentity` is serialized in several places (audit log, cache keys, wire format). All serialization paths that currently write `slot` as a 1-byte field must be widened to 2 bytes. This is a **breaking change** for any persisted data. Recommended migration path:

- Add a version byte to the `OwlIdentity` wire format if not already present.
- V1 = 1-byte slot (legacy); V2 = 2-byte slot (new).
- Read path: detect version byte and upcast V1 fields on load.

Exact serialization call sites are out of scope for this spec but must be inventoried before the PR is merged.

### Round-Trip Test — 65 535 Distinct IDs

The following test must be added to `unified_bridge.rs` (or its test module):

```rust
#[test]
fn round_trip_all_u16_entity_type_ids() {
    // Verifies that no two distinct entity_type_ids collapse to the same
    // OwlIdentity slot after the u8 truncation bug is fixed.
    let family = OgitFamily::test_default();
    let mut seen = std::collections::HashSet::new();

    for id in 0u16..=u16::MAX {
        let identity = OwlIdentity::new(family.clone(), id);
        let key = (identity.family.namespace_id(), identity.slot);
        assert!(
            seen.insert(key),
            "Collision: entity_type_id={id} maps to already-seen key {key:?}"
        );
    }
    // All 65 535 ids must be distinct.
    assert_eq!(seen.len(), 65_536);
}
```

This test would have caught the original bug: before the fix, ids 0 and 256 both produce slot=0, causing the `HashSet::insert` to return `false` on the second insertion.

---

## 2. Bridge-Error Audit Fix (TD-SDR-BRIDGE-ERR-AUDIT-1)

### Problem

In `authorize_read`, `authorize_write`, and `authorize_act` (all in `unified_bridge.rs`), the current pattern is:

```rust
// BEFORE (buggy)
let row = bridge.row()?;          // BridgeError short-circuits here
emit_audit(decision, &row, sink); // never reached on bridge failure
```

When `bridge.row()` returns `Err(BridgeError)`, the `?` operator propagates the error immediately, bypassing `emit_audit`. The result:

- Authorization failures caused by bridge faults are **invisible** to the audit trail.
- Attackers who trigger bridge errors (e.g., via malformed schema pointers or resource exhaustion) can probe the authorization boundary without leaving any audit signal.
- `AuthDecision::BridgeError = 3` exists in `unified_audit.rs:80` as dead code — the variant was anticipated but never wired up.

The existing test at `unified_bridge.rs:~690` **validates this broken behavior**:

```rust
// BEFORE (wrong assertion — must be inverted)
assert!(sink.snapshot().is_empty(), "no audit on bridge error");
```

### Fix

Replace the early-exit `?` with a `match` that emits an audit event before returning the error.

#### 2a. Add `emit_bridge_error_audit` helper

```rust
// crates/lance-graph-callcenter/src/unified_bridge.rs

fn emit_bridge_error_audit(
    err: &BridgeError,
    sink: &dyn AuditSink,
    context: &RequestContext,
) {
    let event = AuditEvent {
        decision: AuthDecision::BridgeError,
        subject_id: context.subject_id,
        resource_hint: context.resource_hint.clone(),
        merkle: context.merkle,   // u64 per W8 correction
        timestamp: context.timestamp,
        detail: format!("bridge_err={err:?}"),
    };
    // Best-effort: ignore sink write errors here to avoid double-fault.
    let _ = sink.emit(event);
}
```

#### 2b. Refactor `authorize_read` (and equivalent for write/act)

```rust
// AFTER — approximately 20 LOC change per authorize_* function

pub fn authorize_read(
    &self,
    ptr: SchemaPtr,
    context: &RequestContext,
    sink: &dyn AuditSink,
) -> Result<AuthDecision, BridgeError> {
    let row = match self.bridge.row(ptr) {
        Ok(r) => r,
        Err(e) => {
            emit_bridge_error_audit(&e, sink, context);
            return Err(e);
        }
    };

    let decision = self.policy.evaluate(&row, context);
    emit_audit(decision, &row, sink);
    Ok(decision)
}
```

The same match pattern applies identically to `authorize_write` and `authorize_act`. Total change: ~20 LOC across three functions plus the helper (~10 LOC).

#### 2c. Invert the existing test

```rust
// unified_bridge.rs, around line 690

// BEFORE (wrong — validates missing audit on error)
assert!(sink.snapshot().is_empty(), "no audit on bridge error");

// AFTER (correct — probe-detection signal must be emitted)
let events = sink.snapshot();
assert_eq!(events.len(), 1, "bridge error must emit exactly one audit event");
assert_eq!(
    events[0].decision,
    AuthDecision::BridgeError,
    "emitted event must carry BridgeError decision"
);
assert_eq!(
    events[0].subject_id,
    context.subject_id,
    "subject_id must be preserved even on bridge error"
);
```

#### 2d. Dashboard Probe-Detection Query

Once `BridgeError` events flow into the audit sink, the following query surfaces probe-detection signals in the ops dashboard (assumes an OLAP-compatible audit table — adapt to actual sink schema):

```sql
-- Probe detection: bridge errors by subject over a rolling window
SELECT
    subject_id,
    COUNT(*)                          AS bridge_error_count,
    MIN(timestamp)                    AS first_seen,
    MAX(timestamp)                    AS last_seen,
    ARRAY_AGG(DISTINCT resource_hint) AS resources_probed
FROM audit_events
WHERE
    decision = 3  -- AuthDecision::BridgeError
    AND timestamp >= NOW() - INTERVAL '15 minutes'
GROUP BY subject_id
HAVING COUNT(*) >= 5   -- threshold: 5 bridge errors in 15 min = alert
ORDER BY bridge_error_count DESC;
```

Threshold values (5 errors / 15 min) are illustrative; they should be tuned against baseline noise in staging before alerting is enabled in production.

The `merkle` column (u64) should be included in raw event exports for forensics but excluded from this aggregation query (too high cardinality for grouping).

---

## 3. Open Question: `OgitFamily` Keying

### Background

`OgitFamily` is currently keyed by `namespace_id`. The slot widening in section 1 touches `entity_type_id`, not `namespace_id`. The two fields are orthogonal in the current schema design:

- `namespace_id` — identifies the owning namespace / super-domain (feeds `super_domain_for_family` routing).
- `entity_type_id` — identifies the entity type within that namespace; currently u16.

### Analysis

`super_domain_for_family` takes an `OgitFamily` and routes to a per-super-domain sink (see W4 cross-flag below). This routing is keyed on `namespace_id`, not `entity_type_id`. Widening `slot` (which maps to `entity_type_id`) does **not** affect routing.

**Likely conclusion:** `namespace_id` remains `u8` (capped at 256 namespaces) while `slot` (entity_type_id) widens to `u16`. These are independent dimensions. A namespace can have up to 65 536 entity types without any routing changes.

**Risk:** If any call site computes a combined key as `(namespace_id << 8) | entity_type_id` treating the combined value as a u16, the widening will overflow. Such sites must be audited. A grep for `namespace_id << 8` and `entity_type_id` in combined expressions is recommended before merging.

---

## 4. Cross-Flags

### W8 — Audit Sink Must Accept `AuthDecision::Denied` Events (and `BridgeError`)

W8's audit sink correction established that the `merkle` field is `u64` (not `u32` as in an earlier draft). This spec adopts `u64` throughout (see `emit_bridge_error_audit` above).

W8 confirmed that the audit sink interface accepts `Denied` events. The `BridgeError` fix in section 2 adds a second non-`Allowed` decision variant that the sink must handle. If `AuditSink::emit` has any runtime filtering on `decision` values, it must be verified to pass `AuthDecision::BridgeError = 3` through without dropping or error. Coordinate with W8 before merging the bridge-error fix.

### W4 — Per-Super-Domain Sink Instance

W4 established that each super-domain gets its own `AuditSink` instance, routed through `super_domain_for_family`. The `emit_bridge_error_audit` helper in section 2 calls `sink` directly using the sink that was passed into `authorize_*` — this is the already-routed per-super-domain sink. No additional routing logic is needed in the helper, provided callers continue to pass the correct routed sink.

However: when `bridge.row()` fails, the `OgitFamily` (and therefore the correct routed sink) may not be determinable if the error occurred before family resolution. In that case, the caller may need to pass a fallback "global" sink for bridge-error events. This edge case should be confirmed during implementation.

---

## 5. Open Questions

### OQ-1: Serialization Version Compatibility

How many existing persisted `OwlIdentity` records exist with 1-byte slot fields? Is there a migration tool for on-disk data, or will existing records be abandoned? The answer determines whether V1/V2 version detection is required in the read path or whether a clean cut-over is acceptable.

**Owner to resolve:** Data platform team + W12 (governance, schema migration sign-off).

### OQ-2: Bridge-Error Sink Routing When Family Is Unknown

As noted in the W4 cross-flag: if `bridge.row()` fails before `OgitFamily` resolution, `super_domain_for_family` cannot be called and the routed per-super-domain sink is unavailable. Should `emit_bridge_error_audit` write to a global fallback sink in this case, or should the event be dropped? Dropping is the current (broken) behavior; a global fallback sink is the recommended fix but introduces a new dependency on a global sink reference in `UnifiedBridge`.

**Owner to resolve:** W4 (per-super-domain sink design) + W10 implementer.

### OQ-3: `slot_u8()` Shim Removal Timeline

The `#[deprecated]` shim `slot_u8()` on `OwlIdentity` is a crutch for callers that have not migrated to `u16`. What is the deprecation window before the shim is removed? Are there external crates (outside `lance-graph-callcenter`) that depend on `slot_u8()` through a published crate boundary?

**Owner to resolve:** Crate maintainer + release manager.

---

## Implementation Checklist

- [ ] Widen `OwlIdentity::new()` parameter to `u16`; add `slot_u8()` deprecated shim
- [ ] Remove truncation mask at `unified_bridge.rs:449`; pass `ptr.entity_type_id()` directly
- [ ] Audit all serialization call sites for 1-byte slot assumptions; add V1/V2 version handling if needed
- [ ] Add `round_trip_all_u16_entity_type_ids` test
- [ ] Add `emit_bridge_error_audit` helper
- [ ] Refactor `authorize_read`, `authorize_write`, `authorize_act` to match on `bridge.row()` result
- [ ] Invert existing test at `unified_bridge.rs:~690`
- [ ] Verify `AuditSink::emit` passes `AuthDecision::BridgeError` without filtering
- [ ] Grep for `namespace_id << 8` combined-key patterns
- [ ] Coordinate with W8 (u64 merkle, sink accepts BridgeError) and W4 (fallback sink for unknown family)
- [ ] Add dashboard probe-detection query to ops runbook
- [ ] Resolve OQ-1, OQ-2, OQ-3 before merge

---

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| Serialization breakage for existing OwlIdentity records | High | Medium | V1/V2 version detection; staged rollout |
| `slot_u8()` callers in external crates silently broken by deprecation | Medium | Low | `#[deprecated]` compiler warning; search crate reverse-deps |
| `emit_bridge_error_audit` adds latency on error paths | Low | Low | Best-effort emit; sink write errors ignored |
| Dashboard alert threshold miscalibrated; alert storm at rollout | Medium | Medium | Tune against staging baseline before prod |
| `super_domain_for_family` routing breaks on widened slot | Low | Low | Routing keyed on namespace_id (orthogonal); verify with grep |
