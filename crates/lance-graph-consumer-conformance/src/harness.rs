//! Generic `assert_consumer_conformance` harness.
//!
//! Verifies all 10 contract assertions (A1–A10) for a correctly-implemented
//! `NamespaceBridge`. Call once per consumer from each per-consumer `#[test]`
//! function.
//!
//! # Assertion summary
//!
//! | ID | Name                              | Contract                                         |
//! |----|-----------------------------------|--------------------------------------------------|
//! | A1 | Audit emission shape              | `canonical_bytes().len() == 26`                  |
//! | A2 | Super-domain stamped              | `event.super_domain == fixture.super_domain`     |
//! | A3 | Merkle chain advances             | Consecutive roots are distinct                   |
//! | A4 | BridgeError short-circuits        | No audit event on unknown entity                 |
//! | A5 | Policy on canonical OGIT name     | Canonical-keyed policy grants; alias-keyed denies |
//! | A6 | SuperDomain != Unknown (active)   | Active consumers must not emit Unknown           |
//! | A7 | Family table non-empty            | `bridge.row(public_name)` succeeds post-seed     |
//! | A8 | TenantId isolation                | `event.tenant` matches construction arg          |
//! | A9 | Actor role hash stable            | `event.actor_role_hash == fnv1a_str(role)`       |
//! | A10| g_lock non-zero                   | `bridge.g_lock().raw() != 0`                     |

use std::sync::Arc;

use lance_graph_callcenter::super_domain::SuperDomain;
use lance_graph_callcenter::audit_sink::{AuditError, AuditSink, MerkleRoot};
use lance_graph_callcenter::unified_audit::{AuditMerkleRoot, UnifiedAuditEvent};
use lance_graph_callcenter::unified_bridge::{TenantId, UnifiedBridge};
use lance_graph_contract::hash::fnv1a_str;
use lance_graph_contract::property::PrefetchDepth;
use lance_graph_ontology::bridge::NamespaceBridge;

// ═══════════════════════════════════════════════════════════════════════════
// RecordingSink — captures every emitted event for assertion
// ═══════════════════════════════════════════════════════════════════════════

/// Captures every `UnifiedAuditEvent` emitted by a `UnifiedBridge` under test.
/// Thread-safe (Mutex-guarded Vec) so concurrent `authorize_*` callers can
/// race without corrupting the capture buffer.
#[derive(Default)]
pub struct RecordingSink {
    pub events: std::sync::Mutex<Vec<UnifiedAuditEvent>>,
}

impl RecordingSink {
    /// Returns a snapshot of all events captured so far.
    pub fn snapshot(&self) -> Vec<UnifiedAuditEvent> {
        self.events.lock().unwrap().clone()
    }

    /// Returns the number of events captured so far.
    pub fn len(&self) -> usize {
        self.events.lock().unwrap().len()
    }

    /// Returns true if no events have been captured.
    pub fn is_empty(&self) -> bool {
        self.events.lock().unwrap().is_empty()
    }
}

impl AuditSink for RecordingSink {
    fn emit(&self, event: UnifiedAuditEvent) -> Result<(), AuditError> {
        self.events.lock().unwrap().push(event);
        Ok(())
    }
    fn flush(&self) -> Result<MerkleRoot, AuditError> {
        Ok(0)
    }
    fn checkpoint(&self) -> Result<(), AuditError> {
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ConformanceFixture — per-consumer test constants
// ═══════════════════════════════════════════════════════════════════════════

/// Per-consumer fixture: the entity name the consumer bridge accepts, its
/// expected canonical OGIT local name, the expected `SuperDomain`, and a role
/// that has read access to the canonical name.
///
/// MedcareBridge: `public_name == canonical_name` (no alias gap).
/// WoaBridge: `public_name = "WorkOrder"`, `canonical_name = "Order"` (alias
/// resolved to canonical OGIT local name — the A5 alias/canonical test case).
pub struct ConformanceFixture {
    /// Public name the consumer bridge accepts (may differ from canonical).
    pub public_name: &'static str,
    /// Expected canonical OGIT local name (what `Policy` must key on).
    pub canonical_name: &'static str,
    /// `SuperDomain` the bridge declares (must not be `Unknown` for active
    /// consumers per A6; scaffold E4/E5 may be `Unknown`).
    pub super_domain: SuperDomain,
    /// A policy role name that has read access to `canonical_name`.
    pub role_that_can_read: &'static str,
    /// Whether this is an active consumer (E1/E2/E3) that must pass A6.
    /// Set to `false` for scaffold consumers (E4/E5) where `#[ignore]` is used.
    pub is_active: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// assert_consumer_conformance — the generic harness
// ═══════════════════════════════════════════════════════════════════════════

/// Assert all 10 contract obligations (A1-A10) for a consumer bridge.
///
/// Call this from each per-consumer `#[test]` function. Three pre-built bridge
/// instances are required:
///
/// - `bridge_allow`: policy grants `fixture.role_that_can_read` on `fixture.canonical_name`.
///   Used for A1/A2/A3/A6/A8/A9 (the allow path).
/// - `bridge_deny`: policy grants `fixture.role_that_can_read` ONLY on
///   `fixture.public_name` (the alias), NOT the canonical name. Used for A5
///   (the alias-keyed policy denies).
/// - `bridge_blank`: the same bridge type over an EMPTY registry — `row()` returns
///   `BridgeError` for any name. Used for A4 (bridge error short-circuits audit).
///
/// `sink_allow` and `sink_blank` are the `Arc<RecordingSink>` instances that
/// were wired into `bridge_allow` and `bridge_blank` via `with_audit_chain`.
pub fn assert_consumer_conformance<B: NamespaceBridge>(
    bridge_allow: &UnifiedBridge<B>,
    bridge_deny: Option<&UnifiedBridge<B>>,
    bridge_blank: &UnifiedBridge<B>,
    fixture: &ConformanceFixture,
    sink_allow: &Arc<RecordingSink>,
    sink_blank: &Arc<RecordingSink>,
) {
    // ── Allow path: A1, A2, A3, A6, A8, A9 ──────────────────────────────────

    // Three sequential authorize_read calls on the allow bridge.
    let r1 = bridge_allow.authorize_read(fixture.public_name, PrefetchDepth::Identity);
    assert!(
        r1.is_ok(),
        "A1/A2/A3: first authorize_read should succeed with canonical-keyed policy; got {r1:?}"
    );
    let r2 = bridge_allow.authorize_read(fixture.public_name, PrefetchDepth::Identity);
    assert!(r2.is_ok(), "A3: second authorize_read should succeed");
    let r3 = bridge_allow.authorize_read(fixture.public_name, PrefetchDepth::Identity);
    assert!(r3.is_ok(), "A3: third authorize_read should succeed");

    let events = sink_allow.snapshot();
    assert_eq!(
        events.len(),
        3,
        "A1: expected exactly 3 audit events for 3 allow calls; got {}",
        events.len()
    );

    // A1: canonical_bytes length == 26
    for (i, ev) in events.iter().enumerate() {
        let bytes = ev.canonical_bytes();
        assert_eq!(
            bytes.len(),
            26,
            "A1: canonical_bytes must be 26 bytes; event[{i}] returned {} bytes",
            bytes.len()
        );
        // A1: verify byte-offset layout
        assert_eq!(
            &bytes[0..8],
            &ev.ts_unix_ms.to_le_bytes(),
            "A1: bytes[0..8] must be ts_unix_ms LE"
        );
        assert_eq!(
            &bytes[8..12],
            &ev.tenant.raw().to_le_bytes(),
            "A1: bytes[8..12] must be tenant LE"
        );
        assert_eq!(bytes[12], ev.super_domain.raw(), "A1: bytes[12] must be super_domain");
        assert_eq!(
            &bytes[13..16],
            &ev.owl.to_canonical_bytes(),
            "A1: bytes[13..16] must be owl_identity canonical bytes"
        );
        assert_eq!(bytes[16], ev.op.as_u8(), "A1: bytes[16] must be op");
        assert_eq!(bytes[17], ev.decision.as_u8(), "A1: bytes[17] must be decision");
        assert_eq!(
            &bytes[18..26],
            &ev.actor_role_hash.to_le_bytes(),
            "A1: bytes[18..26] must be actor_role_hash LE"
        );
    }

    // A2: super_domain stamped correctly on every event
    for (i, ev) in events.iter().enumerate() {
        assert_eq!(
            ev.super_domain, fixture.super_domain,
            "A2: event[{i}].super_domain must equal fixture.super_domain ({:?})",
            fixture.super_domain
        );
    }

    // A3: merkle chain strictly advances
    assert_ne!(
        events[0].merkle_root,
        events[1].merkle_root,
        "A3: merkle root must advance between calls (events[0] == events[1])"
    );
    assert_ne!(
        events[1].merkle_root,
        events[2].merkle_root,
        "A3: merkle root must advance between calls (events[1] == events[2])"
    );
    // All three must differ from GENESIS (A3: genesis root must not reappear
    // as a non-first event's root; we check all three to be thorough).
    for (i, ev) in events.iter().enumerate() {
        assert_ne!(
            ev.merkle_root,
            AuditMerkleRoot::GENESIS,
            "A3: event[{i}].merkle_root must not equal GENESIS after chain advance"
        );
    }

    // A6: active consumers must not emit Unknown super_domain
    if fixture.is_active {
        for (i, ev) in events.iter().enumerate() {
            // CC-3 (meta-review): SuperDomain::System is exempt from the
            // hard-lock requirement (infrastructure auditing is not tenant
            // data). All other active-consumer super_domains must not be
            // Unknown.
            let is_system = ev.super_domain == SuperDomain::System;
            assert!(
                is_system || ev.super_domain != SuperDomain::Unknown,
                "A6: active consumer event[{i}].super_domain must not be Unknown \
                 (SystemDomain is exempt per CC-3)"
            );
        }
    }

    // A8: tenant field matches construction arg TenantId(1)
    for (i, ev) in events.iter().enumerate() {
        assert_eq!(
            ev.tenant,
            TenantId(1),
            "A8: event[{i}].tenant must be TenantId(1) (construction arg)"
        );
    }

    // A9: actor_role_hash == fnv1a_str(role_that_can_read)
    let expected_hash = fnv1a_str(fixture.role_that_can_read);
    for (i, ev) in events.iter().enumerate() {
        assert_eq!(
            ev.actor_role_hash, expected_hash,
            "A9: event[{i}].actor_role_hash must equal fnv1a_str({:?}); \
             got {:016x}, expected {:016x}",
            fixture.role_that_can_read, ev.actor_role_hash, expected_hash
        );
    }

    // A10: g_lock is non-zero after seeding
    let g_lock_raw = bridge_allow.bridge().g_lock().raw();
    assert_ne!(
        g_lock_raw, 0,
        "A10: bridge.g_lock().raw() must be non-zero; got 0 (bridge not initialised against a \
         real registry)"
    );

    // ── A5: policy evaluates on canonical OGIT name, not bridge alias ─────────
    //
    // bridge_deny is wired with a policy that keys on the PUBLIC alias name
    // (fixture.public_name) rather than the canonical name (fixture.canonical_name).
    // For consumers where public_name == canonical_name (e.g. MedcareBridge,
    // OgitBridge) this test is trivially satisfied because the alias IS the
    // canonical — we skip the asymmetric deny assertion in that case.
    // For WoaBridge ("WorkOrder" -> "Order") the alias differs; the deny bridge
    // MUST deny.
    if let Some(bd) = bridge_deny {
        if fixture.public_name != fixture.canonical_name {
            // Alias and canonical differ: policy keyed on the alias must deny.
            let deny_result = bd.authorize_read(fixture.public_name, PrefetchDepth::Identity);
            assert!(
                deny_result.is_err(),
                "A5: policy keyed on alias '{}' must deny when canonical is '{}'; got Ok",
                fixture.public_name, fixture.canonical_name
            );
        }
        // In both cases, bridge_allow (policy keyed on canonical name) already
        // succeeded above — that is the affirmative half of A5.
    }

    // ── A4: BridgeError short-circuits before audit ───────────────────────────
    //
    // bridge_blank is the same bridge type with an empty registry so
    // bridge_blank.row("__nonexistent__") returns BridgeError::NotInScope,
    // which must NOT advance the audit chain.
    let blank_before = sink_blank.len();
    let blank_result = bridge_blank.authorize_read("__nonexistent__", PrefetchDepth::Identity);
    assert!(
        blank_result.is_err(),
        "A4: authorize_read on unknown entity must return Err; got Ok"
    );
    let blank_after = sink_blank.len();
    assert_eq!(
        blank_after, blank_before,
        "A4: BridgeError must not emit any audit event (sink had {blank_before} events before, \
         {blank_after} after)"
    );

    // ── A7: family table non-empty ────────────────────────────────────────────
    //
    // The allow bridge was seeded with at least one MappingProposal (the
    // fixture entity). Verify that `bridge.row(public_name)` succeeds — if
    // it did for the three authorize_read calls above it must succeed here too.
    // We make this explicit to document the intent.
    let row_result = bridge_allow.bridge().row(fixture.public_name);
    assert!(
        row_result.is_ok(),
        "A7: bridge.row('{}') must succeed after seeding the registry; got {:?}",
        fixture.public_name, row_result
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Self-test helpers (used by mock-bridge tests in lib.rs)
// ═══════════════════════════════════════════════════════════════════════════

/// Verify that `authorize_read` on a bridge with a deny-keyed policy
/// returns an `AuthError` (Denied or Escalation), not Ok. This is the
/// negative-path complement to the main harness A5 check.
pub fn assert_deny_on_alias_keyed_policy<B: NamespaceBridge>(
    bridge: &UnifiedBridge<B>,
    public_name: &str,
) {
    let result = bridge.authorize_read(public_name, PrefetchDepth::Identity);
    assert!(
        result.is_err(),
        "assert_deny_on_alias_keyed_policy: expected Err for alias-keyed policy on '{}'; got Ok",
        public_name
    );
}
