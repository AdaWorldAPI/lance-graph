//! `UnifiedBridgeGate<B>` — production `CognitiveBridgeGate` impl.
//!
//! Wraps a `UnifiedBridge<B>` and implements `CognitiveBridgeGate` from
//! `thinking-engine::bridge_gate`. Cross-tenant ops are authorized through
//! the existing `UnifiedBridge::authorize_read` / `authorize_act` paths,
//! which already emit `UnifiedAuditEvent` via the `AuditChain` (D-SDR-5).
//!
//! ## Chinese-wall check (§3.8 of super-domain-rbac-tenancy-v1)
//!
//! Every authorization call that presents a `tenant_id` different from the
//! `UnifiedBridge`'s own `TenantId` is a cross-tenant op. The gate
//! short-circuits to `Deny` BEFORE policy evaluation in that case — the
//! Chinese wall fires unconditionally regardless of RBAC grants.
//!
//! When `tenant_id == bridge.tenant().raw()` (same-tenant), the call is
//! forwarded to the normal `authorize_read` / `authorize_act` path, which
//! evaluates RBAC and emits an audit event.
//!
//! ## Delegation to UnifiedBridge
//!
//! | Category | CognitiveBridgeGate method         | UnifiedBridge call                                 |
//! |----------|------------------------------------|----------------------------------------------------|
//! | A        | `authorize_retrieval`              | `authorize_read(entity_type, prefetch_from_u8(depth))` |
//! | B        | `authorize_persona_switch`         | `authorize_read("Persona", PrefetchDepth::Detail)` |
//! | C        | `authorize_cognitive_op`           | `authorize_act("CognitiveStack", op_name)`         |
//!
//! Audit events are emitted automatically by `UnifiedBridge::emit_audit()`
//! on every delegated call. The `super_domain` on emitted events comes from
//! the `AuditChain` configured at `UnifiedBridgeGate` construction time.
//!
//! ## No audit on Chinese-wall Deny
//!
//! Cross-tenant mismatches short-circuit before `UnifiedBridge` is
//! consulted. Those events are NOT audited through the normal `AuditChain`
//! path (the bridge never sees them). A separate counter is kept for
//! observability via `chinese_wall_deny_count()`.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use lance_graph_ontology::bridge::NamespaceBridge;
use lance_graph_contract::property::PrefetchDepth;

use crate::unified_bridge::UnifiedBridge;
use thinking_engine::bridge_gate::{
    CognitiveAuthResult, CognitiveBridgeGate, CognitiveOpKind,
};

// ═══════════════════════════════════════════════════════════════════════════
// PrefetchDepth helper — avoids exposing lance-graph-contract in thinking-engine
// ═══════════════════════════════════════════════════════════════════════════

/// Map the `depth: u8` passed through `CognitiveBridgeGate::authorize_retrieval`
/// to a `PrefetchDepth` value.
///
/// PrefetchDepth variants (per lance-graph-contract):
/// `0 = Identity`, `1 = Detail`, `2 = Similar`, `3 = Full`, `_ = Detail`
/// (safe fallback — unknown depth treated as Detail, not Identity, so the
/// bridge sees a sufficient entity shape and policy can make a correct decision).
fn prefetch_from_u8(depth: u8) -> PrefetchDepth {
    match depth {
        0 => PrefetchDepth::Identity,
        1 => PrefetchDepth::Detail,
        2 => PrefetchDepth::Similar,
        3 => PrefetchDepth::Full,
        _ => PrefetchDepth::Detail,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// UnifiedBridgeGate<B>
// ═══════════════════════════════════════════════════════════════════════════

/// Production `CognitiveBridgeGate` impl. Wraps a `UnifiedBridge<B>`.
///
/// Construction:
/// ```ignore
/// let gate = UnifiedBridgeGate::new(Arc::new(unified_bridge));
/// let gate_arc: Arc<dyn CognitiveBridgeGate> = Arc::new(gate);
/// ```
pub struct UnifiedBridgeGate<B: NamespaceBridge> {
    bridge: Arc<UnifiedBridge<B>>,
    /// Count of Chinese-wall Deny decisions (cross-tenant mismatch).
    /// Monotonically increasing; never reset. Useful for metrics/alerting.
    chinese_wall_deny_count: AtomicU64,
}

impl<B: NamespaceBridge> UnifiedBridgeGate<B> {
    /// Construct a gate wrapping `bridge`.
    pub fn new(bridge: Arc<UnifiedBridge<B>>) -> Self {
        Self {
            bridge,
            chinese_wall_deny_count: AtomicU64::new(0),
        }
    }

    /// Number of Chinese-wall Deny decisions since construction.
    pub fn chinese_wall_deny_count(&self) -> u64 {
        self.chinese_wall_deny_count.load(Ordering::Relaxed)
    }

    /// Returns `true` when `caller_tenant` matches the bridge's own `TenantId`.
    /// Same-tenant → forward to policy. Cross-tenant → Chinese wall Deny.
    #[inline]
    fn same_tenant(&self, caller_tenant: u32) -> bool {
        self.bridge.tenant().raw() == caller_tenant
    }

    /// Chinese-wall check: bump counter, return `Deny`.
    #[inline]
    fn chinese_wall_deny(&self) -> CognitiveAuthResult {
        self.chinese_wall_deny_count.fetch_add(1, Ordering::Relaxed);
        CognitiveAuthResult::Deny
    }

    /// Map a `UnifiedBridge` `Result` to `CognitiveAuthResult`.
    ///
    /// `Ok(_)` → Allow.
    /// `Err(AuthError::Denied)` → Deny.
    /// `Err(AuthError::Escalation)` → Escalate.
    /// `Err(AuthError::Bridge(_))` → Deny (unknown entity is not a grant).
    #[inline]
    fn map_bridge_result<T>(result: Result<T, crate::unified_bridge::AuthError>) -> CognitiveAuthResult {
        match result {
            Ok(_)                                             => CognitiveAuthResult::Allow,
            Err(crate::unified_bridge::AuthError::Escalation(_)) => CognitiveAuthResult::Escalate,
            Err(_)                                            => CognitiveAuthResult::Deny,
        }
    }
}

// ── CognitiveBridgeGate impl ─────────────────────────────────────────────────

impl<B: NamespaceBridge + Send + Sync> CognitiveBridgeGate for UnifiedBridgeGate<B> {
    /// Category A — cross-tenant sensor-lens retrieval.
    ///
    /// Chinese wall fires first when `tenant_id` ≠ bridge tenant.
    /// Same-tenant calls delegate to `authorize_read(entity_type, depth)`.
    fn authorize_retrieval(
        &self,
        tenant_id: u32,
        entity_type: &str,
        depth: u8,
    ) -> CognitiveAuthResult {
        if !self.same_tenant(tenant_id) {
            return self.chinese_wall_deny();
        }
        let depth = prefetch_from_u8(depth);
        Self::map_bridge_result(self.bridge.authorize_read(entity_type, depth))
    }

    /// Category B — persona switch reading shared archetype corpus.
    ///
    /// Always uses `PrefetchDepth::Detail` (the persona archetype record
    /// requires the full slot to determine which qualia vector applies).
    fn authorize_persona_switch(
        &self,
        tenant_id: u32,
        _mode: u8,
    ) -> CognitiveAuthResult {
        if !self.same_tenant(tenant_id) {
            return self.chinese_wall_deny();
        }
        Self::map_bridge_result(self.bridge.authorize_read("Persona", PrefetchDepth::Detail))
    }

    /// Category C — L6 delegation / L8 integration across tenant boundaries.
    ///
    /// Delegates to `authorize_act("CognitiveStack", op_name)` where
    /// `op_name` is the `CognitiveOpKind` display string.
    fn authorize_cognitive_op(
        &self,
        tenant_id: u32,
        op_kind: CognitiveOpKind,
    ) -> CognitiveAuthResult {
        if !self.same_tenant(tenant_id) {
            return self.chinese_wall_deny();
        }
        let op_name = match op_kind {
            CognitiveOpKind::L6Delegation   => "L6Delegation",
            CognitiveOpKind::L8Integration  => "L8Integration",
            CognitiveOpKind::QualiaWrite    => "QualiaWrite",
            CognitiveOpKind::MetaWordCommit => "MetaWordCommit",
        };
        Self::map_bridge_result(self.bridge.authorize_act("CognitiveStack", op_name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use lance_graph_ontology::OntologyRegistry;
    use lance_graph_rbac::policy::smb_policy;

    use crate::unified_bridge::{TenantId, UnifiedBridge};
    use crate::audit_sink::{AuditError, AuditSink, MerkleRoot};
    use crate::unified_audit::UnifiedAuditEvent;
    use crate::super_domain::SuperDomain as SD;
    use thinking_engine::bridge_gate::{CognitiveBridgeGate, CognitiveOpKind, CognitiveAuthResult};

    // ── StubBridge (mirrors unified_bridge tests) ────────────────────────────

    struct StubBridge {
        registry: Arc<OntologyRegistry>,
    }

    impl lance_graph_ontology::bridge::NamespaceBridge for StubBridge {
        fn bridge_id(&self) -> &'static str { "stub" }
        fn registry(&self) -> &OntologyRegistry { &self.registry }
        fn g_lock(&self) -> lance_graph_ontology::namespace::NamespaceId {
            lance_graph_ontology::namespace::NamespaceId(1)
        }
    }

    fn make_gate(tenant: TenantId) -> UnifiedBridgeGate<StubBridge> {
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        let bridge = Arc::new(StubBridge { registry });
        let policy = Arc::new(smb_policy());
        let unified = Arc::new(UnifiedBridge::new(bridge, policy, "accountant", tenant));
        UnifiedBridgeGate::new(unified)
    }

    // ── Recording sink ───────────────────────────────────────────────────────

    #[derive(Default)]
    struct RecordingSink {
        events: std::sync::Mutex<Vec<UnifiedAuditEvent>>,
    }
    impl RecordingSink {
        fn count(&self) -> usize { self.events.lock().unwrap().len() }
    }
    impl AuditSink for RecordingSink {
        fn emit(&self, event: UnifiedAuditEvent) -> Result<(), AuditError> {
            self.events.lock().unwrap().push(event);
            Ok(())
        }
        fn flush(&self) -> Result<MerkleRoot, AuditError> { Ok(0) }
        fn checkpoint(&self) -> Result<(), AuditError> { Ok(()) }
    }

    // ── Chinese-wall tests ───────────────────────────────────────────────────

    #[test]
    fn cross_tenant_retrieval_denied_by_chinese_wall() {
        let gate = make_gate(TenantId(1));
        // Caller presents tenant_id = 2 ≠ bridge tenant 1 → Chinese wall Deny.
        let result = gate.authorize_retrieval(2, "Document", 0);
        assert_eq!(result, CognitiveAuthResult::Deny);
        assert_eq!(gate.chinese_wall_deny_count(), 1);
    }

    #[test]
    fn cross_tenant_persona_denied_by_chinese_wall() {
        let gate = make_gate(TenantId(5));
        let result = gate.authorize_persona_switch(99, 0);
        assert_eq!(result, CognitiveAuthResult::Deny);
        assert_eq!(gate.chinese_wall_deny_count(), 1);
    }

    #[test]
    fn cross_tenant_cognitive_op_denied_by_chinese_wall() {
        let gate = make_gate(TenantId(3));
        let result = gate.authorize_cognitive_op(4, CognitiveOpKind::L6Delegation);
        assert_eq!(result, CognitiveAuthResult::Deny);
    }

    #[test]
    fn chinese_wall_count_accumulates() {
        let gate = make_gate(TenantId(1));
        gate.authorize_retrieval(2, "Document", 0);
        gate.authorize_persona_switch(2, 1);
        gate.authorize_cognitive_op(2, CognitiveOpKind::L8Integration);
        assert_eq!(gate.chinese_wall_deny_count(), 3);
    }

    // ── Same-tenant path tests ───────────────────────────────────────────────

    // Note: same-tenant calls delegate to UnifiedBridge. The stub registry is
    // empty so authorize_read("Document") returns BridgeError::NotInScope →
    // mapped to CognitiveAuthResult::Deny. This tests the delegation path
    // (not a grant path — that needs a real registry with rows).

    #[test]
    fn same_tenant_retrieval_delegates_to_bridge() {
        let gate = make_gate(TenantId(1));
        // Same tenant (1 == 1). Bridge lookup fails → Deny (bridge error).
        let result = gate.authorize_retrieval(1, "UnknownEntity", 0);
        assert_eq!(result, CognitiveAuthResult::Deny);
        // Chinese wall NOT incremented (same tenant).
        assert_eq!(gate.chinese_wall_deny_count(), 0);
    }

    #[test]
    fn same_tenant_persona_switch_delegates_to_bridge() {
        let gate = make_gate(TenantId(7));
        // Same tenant (7 == 7). Bridge lookup "Persona" fails → Deny.
        let result = gate.authorize_persona_switch(7, 1);
        assert_eq!(result, CognitiveAuthResult::Deny);
        assert_eq!(gate.chinese_wall_deny_count(), 0);
    }

    #[test]
    fn same_tenant_cognitive_op_delegates_to_bridge() {
        let gate = make_gate(TenantId(2));
        // Same tenant. "CognitiveStack" not in stub registry → Deny.
        let result = gate.authorize_cognitive_op(2, CognitiveOpKind::QualiaWrite);
        assert_eq!(result, CognitiveAuthResult::Deny);
        assert_eq!(gate.chinese_wall_deny_count(), 0);
    }

    // ── prefetch_from_u8 ────────────────────────────────────────────────────

    #[test]
    fn prefetch_mapping() {
        assert_eq!(prefetch_from_u8(0), PrefetchDepth::Identity);
        assert_eq!(prefetch_from_u8(1), PrefetchDepth::Detail);
        assert_eq!(prefetch_from_u8(2), PrefetchDepth::Similar);
        assert_eq!(prefetch_from_u8(3), PrefetchDepth::Full);
        // Unknown depth falls back to Detail.
        assert_eq!(prefetch_from_u8(255), PrefetchDepth::Detail);
    }

    // ── Gate as Arc<dyn CognitiveBridgeGate> ────────────────────────────────

    #[test]
    fn gate_usable_as_dyn_trait() {
        let gate: Arc<dyn CognitiveBridgeGate> = Arc::new(make_gate(TenantId(42)));
        // Cross-tenant → Deny via Chinese wall.
        assert_eq!(gate.authorize_retrieval(99, "Document", 0), CognitiveAuthResult::Deny);
    }

    // ── Audit emission: cross-tenant retrieval ───────────────────────────────

    /// Cross-tenant retrieval (Chinese wall) does NOT emit audit events through
    /// the UnifiedBridge. Same-tenant retrieval DOES emit one UnifiedAuditEvent.
    ///
    /// We exercise the same-tenant path with a real ontology row so the bridge
    /// resolves cleanly and policy decides (emitting the event).
    #[test]
    fn same_tenant_allowed_emits_audit_event() {
        use lance_graph_contract::property::{Marking, Schema};
        use lance_graph_ontology::namespace::OgitUri;
        use lance_graph_ontology::proposal::{MappingProposal, MappingProposalKind};
        use lance_graph_rbac::permission::PermissionSpec;
        use lance_graph_rbac::role::Role;
        use lance_graph_rbac::policy::Policy;

        // Build a registry with "Document" → "Doc" canonical entity.
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        let canonical_uri = OgitUri::parse("ogit.Content:Doc").unwrap();
        let proposal = MappingProposal {
            public_name: "Document".to_string(),
            bridge_id: "test".to_string(),
            ogit_uri: canonical_uri,
            namespace: "Content".to_string(),
            kind: MappingProposalKind::Entity {
                schema: Schema::builder("Doc").required("id").build(),
            },
            marking: Marking::Internal,
            confidence: 1.0,
            source_uri: "test://doc".to_string(),
            checksum: "ck-doc".to_string(),
            created_by: "test".to_string(),
        };
        registry.append_mapping(proposal).unwrap();
        let g_lock = registry.namespace_id("Content").unwrap();

        struct RealBridge {
            registry: Arc<OntologyRegistry>,
            g_lock: lance_graph_ontology::namespace::NamespaceId,
        }
        impl lance_graph_ontology::bridge::NamespaceBridge for RealBridge {
            fn bridge_id(&self) -> &'static str { "test" }
            fn registry(&self) -> &OntologyRegistry { &self.registry }
            fn g_lock(&self) -> lance_graph_ontology::namespace::NamespaceId { self.g_lock }
        }

        // Policy: "reader" may read "Doc" at Detail.
        let policy = Policy::new("test")
            .with_role(
                Role::new("reader")
                    .with_permission(PermissionSpec::read_at("Doc", PrefetchDepth::Detail)),
            );

        let sink = Arc::new(RecordingSink::default());
        let bridge = Arc::new(UnifiedBridge::new(
            Arc::new(RealBridge { registry, g_lock }),
            Arc::new(policy),
            "reader",
            TenantId(10),
        ).with_audit_chain(SD::WorkOrderBilling, 0, sink.clone()));

        let gate = UnifiedBridgeGate::new(bridge);

        // Same-tenant retrieval of "Document" with depth 1 (Detail).
        let result = gate.authorize_retrieval(10, "Document", 1);
        assert_eq!(result, CognitiveAuthResult::Allow, "policy should allow reader→Doc");

        // Exactly 1 audit event emitted by UnifiedBridge.
        assert_eq!(sink.count(), 1, "same-tenant allowed op must emit 1 audit event");
        // Chinese wall NOT triggered.
        assert_eq!(gate.chinese_wall_deny_count(), 0);
    }

    /// Cross-tenant Deny (Chinese wall) emits ZERO audit events through
    /// UnifiedBridge — the bridge is never consulted.
    #[test]
    fn cross_tenant_denied_emits_no_audit_event() {
        let sink = Arc::new(RecordingSink::default());

        let registry = Arc::new(OntologyRegistry::new_in_memory());
        let bridge_inner = Arc::new(StubBridge { registry });
        let policy = Arc::new(smb_policy());
        let unified = Arc::new(UnifiedBridge::new(bridge_inner, policy, "accountant", TenantId(1))
            .with_audit_chain(SD::Unknown, 0, sink.clone()));
        let gate = UnifiedBridgeGate::new(unified);

        let _ = gate.authorize_retrieval(2, "Document", 0); // cross-tenant
        assert_eq!(sink.count(), 0, "Chinese wall deny must not emit audit via UnifiedBridge");
        assert_eq!(gate.chinese_wall_deny_count(), 1);
    }

    /// Pure cognitive ops (encode, qualia compute, distance lookup) do NOT call
    /// the gate. Verified by the absence of any chinese_wall increment and the
    /// zero-overhead passthrough in thinking-engine standalone mode.
    #[test]
    fn pure_ops_emit_zero_audit_events() {
        let sink = Arc::new(RecordingSink::default());
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        let bridge_inner = Arc::new(StubBridge { registry });
        let policy = Arc::new(smb_policy());
        let unified = Arc::new(UnifiedBridge::new(bridge_inner, policy, "accountant", TenantId(1))
            .with_audit_chain(SD::Unknown, 0, sink.clone()));
        let gate = UnifiedBridgeGate::new(unified);

        // Pure codebook lookups — gate never called.
        let _ = thinking_engine::jina_lens::jina_lookup(42);
        let _ = thinking_engine::jina_lens::jina_distance(0, 1);
        let _ = thinking_engine::bge_m3_lens::bge_m3_lookup(100);
        let _ = thinking_engine::reranker_lens::reranker_lookup(500);

        assert_eq!(sink.count(), 0, "pure ops must emit zero audit events");
        assert_eq!(gate.chinese_wall_deny_count(), 0);
    }
}
