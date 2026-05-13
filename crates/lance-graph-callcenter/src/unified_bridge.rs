//! `UnifiedBridge` — composes a per-namespace `NamespaceBridge` with an RBAC
//! `Policy` so consumers get one entry point that does scope-locking,
//! ontology resolution, and access-decision evaluation in one call.
//!
//! Per `.claude/plans/super-domain-rbac-tenancy-v1.md` §3.9, this is the Tier
//! A starter for the unified bridge surface. The minimal viable shape wires
//! the two existing crates (`lance-graph-ontology` + `lance-graph-rbac`) and
//! adds a `TenantId` field for the multi-tenant Chinese wall (§3.8).
//!
//! Richer surface — super-domain routing (§3.4), nested role groups with
//! `FieldRedactionMask` (§3.6), merkle audit chain (§13.3), 4-stage
//! `authorize()` against `PolicyRewriter` (§3.9 + §13.1) — lands in
//! follow-up commits as Tier A deliverables D-SDR-2..5.
//!
//! ## Wiring at the call site
//!
//! ```ignore
//! use std::sync::Arc;
//! use lance_graph_ontology::{NamespaceBridge, OntologyRegistry, bridges::MedcareBridge};
//! use lance_graph_ontology::bridge::BridgeFromRegistry;
//! use lance_graph_rbac::policy::smb_policy;
//! use lance_graph_callcenter::unified_bridge::{TenantId, UnifiedBridge};
//! use lance_graph_contract::property::PrefetchDepth;
//!
//! let registry = Arc::new(OntologyRegistry::new_in_memory());
//! // ... hydrate registry from TTL ...
//! let bridge = MedcareBridge::from_registry(registry).unwrap();
//! let policy = Arc::new(smb_policy());
//! let unified = UnifiedBridge::new(Arc::new(bridge), policy, "accountant", TenantId(1));
//!
//! match unified.authorize_read("Customer", PrefetchDepth::Detail) {
//!     Ok(entity) => { /* schema_ptr.entity_type_id() = dense index */ }
//!     Err(e)     => { /* denied / escalation / bridge error */ }
//! }
//! ```

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use lance_graph_contract::hash::fnv1a_str;
use lance_graph_contract::property::PrefetchDepth;
use lance_graph_ontology::bridge::{BridgeError, EntityRef, NamespaceBridge};
use lance_graph_ontology::namespace::SchemaPtr;
use lance_graph_ontology::proposal::MappingRow;
use lance_graph_rbac::access::AccessDecision;
use lance_graph_rbac::policy::{Operation, Policy};

use crate::super_domain::SuperDomain;
use crate::unified_audit::{
    AuditChain, AuditMerkleRoot, AuthDecision, AuthOp, NoopUnifiedAuditSink, UnifiedAuditEvent,
    UnifiedAuditSink,
};

/// Extract the canonical ontology entity type name from a resolved
/// [`MappingRow`], for use as the [`Policy::evaluate`] key.
///
/// `row.ogit_uri` carries the canonical OGIT URI (e.g.
/// `ogit.WorkOrder:Order`); `OgitUri::name()` is the local part
/// (`Order`). Falls back to `public_name` if the URI somehow lacks a
/// name part — a malformed URI shouldn't silently bypass the alias
/// resolution, but it shouldn't break authorization either.
fn canonical_entity_type<'a>(row: &'a MappingRow, public_name: &'a str) -> &'a str
{
    row.ogit_uri.name().unwrap_or(public_name)
}

// ═══════════════════════════════════════════════════════════════════════════
// OgitFamily — Level-2 basin pointer (§3.1 of super-domain-rbac-tenancy-v1)
// ═══════════════════════════════════════════════════════════════════════════

/// 1 byte. Identifies which OGIT family (basin) a row belongs to.
/// 256 families max; ~75 used today (per `RECON_ONTOLOGY_CRATE.md` §1.9).
/// Pure address. No reasoning, no string lookup.
///
/// High byte of `OwlIdentity`. Bitmask compare against this is the
/// sub-microsecond hot-path predicate Cypher MATCH lowers to.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OgitFamily(pub u8);

impl OgitFamily
{
    pub const UNKNOWN: Self = Self(0);

    #[inline]
    pub const fn raw(self) -> u8
    {
        self.0
    }

    #[inline]
    pub const fn is_known(self) -> bool
    {
        self.0 != 0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// OwlIdentity — Level-3 per-row identity (§3.2)
// ═══════════════════════════════════════════════════════════════════════════

/// 3 bytes when serialized: 1-byte `OgitFamily` (the precise heel
/// pointer / "mantissa") + 2-byte within-family slot (the OWL /
/// consumer's own identity). Widened from the original 2-byte
/// (family u8 + slot u8) layout after PR #364 review surfaced that
/// `RegistryState::append` allocates entity-type IDs globally as `u16`
/// so any registry with ≥256 entries would alias slot collisions
/// across distinct authorized entities.
///
/// In-memory representation is two named fields (4 bytes with
/// alignment). On-wire layout in `UnifiedAuditEvent::canonical_bytes`
/// is the deterministic 3-byte sequence `[family, slot_le_lo,
/// slot_le_hi]`.
///
/// This is what rides on every LanceDB row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OwlIdentity {
    family: OgitFamily,
    slot: u16,
}

impl OwlIdentity
{
    pub const UNKNOWN: Self = Self {
        family: OgitFamily(0),
        slot: 0,
    };

    #[inline]
    pub const fn new(family: OgitFamily, slot: u16) -> Self
    {
        Self { family, slot }
    }

    #[inline]
    pub const fn family(self) -> OgitFamily
    {
        self.family
    }

    #[inline]
    pub const fn slot(self) -> u16
    {
        self.slot
    }

    /// Deterministic on-wire form: `[family u8, slot_lo u8, slot_hi u8]`.
    /// Used by `UnifiedAuditEvent::canonical_bytes` so the merkle chain
    /// hashes a byte-stable representation across Rust / C# emitters.
    #[inline]
    pub const fn to_canonical_bytes(self) -> [u8; 3]
    {
        let slot = self.slot.to_le_bytes();
        [self.family.0, slot[0], slot[1]]
    }

    /// Bitmask predicate Cypher MATCH lowers to. No string lookup.
    #[inline]
    pub const fn is_family(self, f: OgitFamily) -> bool
    {
        self.family.0 == f.0
    }

    /// Within-family slot predicate.
    #[inline]
    pub const fn is_slot(self, s: u16) -> bool
    {
        self.slot == s
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TenantId — multi-tenant Chinese wall (§3.8 of super-domain-rbac-tenancy-v1)
// ═══════════════════════════════════════════════════════════════════════════

/// 4-byte newtype carried alongside every authorization request.
/// Used by the cross-tenant Chinese-wall predicate at the storage layer.
/// In follow-up commits this couples to a per-tenant DEK for crypto backstop
/// (§13.4 hard-lock); for now it's a pure tag the caller asserts on every
/// `authorize_*` invocation.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TenantId(pub u32);

impl TenantId {
    pub const UNKNOWN: TenantId = TenantId(0);

    pub const fn raw(self) -> u32 {
        self.0
    }

    pub const fn is_known(self) -> bool {
        self.0 != 0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AuthError — unified error across bridge resolution + RBAC evaluation
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a `UnifiedBridge::authorize_*` call.
///
/// Three failure modes:
/// - `Bridge` — the ontology lookup failed (unknown public name, cross-namespace leak).
/// - `Denied` — the bridge resolved cleanly but the RBAC `Policy` denied access.
/// - `Escalation` — the policy returned `AccessDecision::Escalate`; the caller
///   must drive the human-approval / MFA step before retrying.
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("ontology bridge error: {0}")]
    Bridge(#[from] BridgeError),
    #[error("denied: {0}")]
    Denied(&'static str),
    #[error("escalation required: {0}")]
    Escalation(&'static str),
}

impl AuthError {
    pub const fn is_denied(&self) -> bool {
        matches!(self, Self::Denied(_))
    }

    pub const fn is_escalation(&self) -> bool {
        matches!(self, Self::Escalation(_))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// UnifiedBridge — the composition
// ═══════════════════════════════════════════════════════════════════════════

/// Composes a per-namespace `NamespaceBridge` (g-locked ontology lookup) with
/// an RBAC `Policy` (role-based access decisions) and a `TenantId` (multi-
/// tenant Chinese wall tag). One entry point for consumers that hides the
/// two-step resolve-then-evaluate flow.
///
/// Generic over the bridge type so each consumer crate (medcare-rs,
/// smb-office-rs, future hiro-rs / hubspot-rs / woa-rs) constructs its own
/// `UnifiedBridge<MedcareBridge>` / `UnifiedBridge<WoaBridge>` etc. without
/// dynamic dispatch.
pub struct UnifiedBridge<B: NamespaceBridge> {
    bridge: Arc<B>,
    policy: Arc<Policy>,
    actor_role: &'static str,
    /// FNV-1a digest of `actor_role`. Cached at construction so each
    /// `authorize_*` call stamps audit events with a fixed-size identifier
    /// (the `&'static str` doesn't fit into a `Copy` audit record).
    actor_role_hash: u64,
    tenant: TenantId,
    /// Audit sink — every `authorize_*` call that reaches the policy
    /// evaluation step emits one `UnifiedAuditEvent`. Default is
    /// `NoopUnifiedAuditSink` (zero overhead, no persistence). Swap via
    /// [`Self::with_audit_chain`].
    audit_sink: Arc<dyn UnifiedAuditSink>,
    /// Merkle-chained audit advancer. Holds the prior event's root +
    /// per-super-domain salt so each new event chains off it. Mutex
    /// guards the `last_root` advance under concurrent `authorize_*`
    /// callers; contention is bounded (each event mutates once).
    audit_chain: Mutex<AuditChain>,
}

impl<B: NamespaceBridge> UnifiedBridge<B> {
    /// Construct a new unified bridge.
    ///
    /// Defaults audit to `NoopUnifiedAuditSink` + a chain anchored at
    /// `SuperDomain::Unknown` with salt 0. Call
    /// [`Self::with_audit_chain`] to swap in a real sink + the
    /// super-domain-specific salt before authorization traffic starts.
    pub fn new(
        bridge: Arc<B>,
        policy: Arc<Policy>,
        actor_role: &'static str,
        tenant: TenantId,
    ) -> Self {
        Self {
            bridge,
            policy,
            actor_role,
            actor_role_hash: fnv1a_str(actor_role),
            tenant,
            audit_sink: Arc::new(NoopUnifiedAuditSink),
            audit_chain: Mutex::new(AuditChain::new(SuperDomain::Unknown, 0)),
        }
    }

    /// Builder: swap in a real `UnifiedAuditSink` + the super-domain's
    /// `merkle_salt` (§13.4 — cross-domain audit logs unlinkable). Resets
    /// the chain to GENESIS; pass a resume root via
    /// [`Self::with_audit_chain_resume`] if continuing a persisted chain.
    pub fn with_audit_chain(
        mut self,
        super_domain: SuperDomain,
        salt: u64,
        sink: Arc<dyn UnifiedAuditSink>,
    ) -> Self {
        self.audit_sink = sink;
        self.audit_chain = Mutex::new(AuditChain::new(super_domain, salt));
        self
    }

    /// Builder: like [`Self::with_audit_chain`] but resumes from a known
    /// prior root (e.g. on process restart after reading the last
    /// persisted event's root).
    pub fn with_audit_chain_resume(
        mut self,
        super_domain: SuperDomain,
        salt: u64,
        last_root: AuditMerkleRoot,
        sink: Arc<dyn UnifiedAuditSink>,
    ) -> Self {
        self.audit_sink = sink;
        self.audit_chain = Mutex::new(AuditChain::resume(super_domain, salt, last_root));
        self
    }

    /// Returns the underlying namespace bridge.
    pub fn bridge(&self) -> &B {
        &self.bridge
    }

    /// Returns the active actor role name.
    pub fn actor_role(&self) -> &'static str {
        self.actor_role
    }

    /// Returns the tenant id.
    pub fn tenant(&self) -> TenantId {
        self.tenant
    }

    /// Returns the current merkle root of the audit chain (the root that
    /// the next emitted event will chain off of). Useful for persisting
    /// chain state at shutdown so [`Self::with_audit_chain_resume`] can
    /// pick up where we left off.
    pub fn audit_root(&self) -> AuditMerkleRoot {
        self.audit_chain.lock().unwrap().last_root
    }

    // ── Authorization ─────────────────────────────────────────────────────────

    /// Resolve `public_name` through the bridge then evaluate read access at
    /// `depth`. Returns the resolved `EntityRef` on `Allow`, `AuthError` on
    /// `Deny` / `Escalate` / bridge resolution failure.
    ///
    /// **Policy evaluation keys on the canonical ontology entity type
    /// (e.g. `Order` for `ogit.WorkOrder:Order`), not the bridge-side
    /// `public_name` alias.** This means a `Policy` authored against
    /// canonical OGIT names is honored regardless of which bridge / public
    /// alias the caller used, and consumer-facing aliases stay decoupled
    /// from policy authorship.
    ///
    /// On policy evaluation reaching, one `UnifiedAuditEvent` is emitted
    /// through the configured `UnifiedAuditSink` carrying tenant +
    /// super-domain + owl + decision. **`BridgeError` short-circuits
    /// before audit** — bad input names aren't auth decisions, they're
    /// invalid requests (D-SDR-5 minimum; revisit if probing detection
    /// becomes a need).
    pub fn authorize_read(
        &self,
        public_name: &str,
        depth: PrefetchDepth,
    ) -> Result<EntityRef, AuthError> {
        let row = self.bridge.row(public_name)?;
        let canonical = canonical_entity_type(&row, public_name);
        let decision = self.policy.evaluate(self.actor_role, canonical, Operation::Read { depth });
        self.emit_audit(&row.schema_ptr, AuthOp::Read, decision_of(&decision));
        map_decision(decision, row.schema_ptr)
    }

    /// Resolve `public_name` then evaluate write access on `predicate`.
    /// Policy keys on the canonical ontology entity type — see
    /// [`Self::authorize_read`] for the canonical-name + audit contract.
    pub fn authorize_write(
        &self,
        public_name: &str,
        predicate: &str,
    ) -> Result<EntityRef, AuthError> {
        let row = self.bridge.row(public_name)?;
        let canonical = canonical_entity_type(&row, public_name);
        let decision = self.policy.evaluate(self.actor_role, canonical, Operation::Write { predicate });
        self.emit_audit(&row.schema_ptr, AuthOp::Write, decision_of(&decision));
        map_decision(decision, row.schema_ptr)
    }

    /// Resolve `public_name` then evaluate action access on `action`.
    /// Policy keys on the canonical ontology entity type — see
    /// [`Self::authorize_read`] for the canonical-name + audit contract.
    pub fn authorize_act(
        &self,
        public_name: &str,
        action: &str,
    ) -> Result<EntityRef, AuthError> {
        let row = self.bridge.row(public_name)?;
        let canonical = canonical_entity_type(&row, public_name);
        let decision = self.policy.evaluate(self.actor_role, canonical, Operation::Act { action });
        self.emit_audit(&row.schema_ptr, AuthOp::Act, decision_of(&decision));
        map_decision(decision, row.schema_ptr)
    }

    /// Stamp an audit event for the resolved row + op + decision through
    /// the merkle chain and emit to the configured sink.
    fn emit_audit(&self, schema_ptr: &SchemaPtr, op: AuthOp, decision: AuthDecision) {
        let owl = owl_from_schema_ptr(schema_ptr);
        // Hold the chain lock across stamping so the event's super_domain
        // and the chain it commits into are guaranteed to agree.
        // FAMILY_TO_SUPER_DOMAIN is an all-Unknown static today (the family
        // → super-domain hydration table lands in sprint 5); until then,
        // trust the super_domain the caller wired into the chain via
        // with_audit_chain(...) as the single source of truth.
        let mut chain = match self.audit_chain.lock() {
            Ok(g) => g,
            // Mutex poisoned by a panicking holder — keep audit emission
            // best-effort by advancing through the poisoned guard.
            Err(poisoned) => poisoned.into_inner(),
        };
        let event = UnifiedAuditEvent {
            ts_unix_ms: now_unix_ms(),
            tenant: self.tenant,
            super_domain: chain.super_domain,
            owl,
            op,
            decision,
            actor_role_hash: self.actor_role_hash,
            // overwritten by AuditChain::advance
            merkle_root: AuditMerkleRoot::GENESIS,
        };
        let stamped = chain.advance(event);
        drop(chain);
        self.audit_sink.emit(&stamped);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// D-SDR-5 plumbing helpers (free functions, no UnifiedBridge::evaluate)
// ═══════════════════════════════════════════════════════════════════════════

/// Map an `AccessDecision` to the lifetime-free `AuthDecision` tag stored
/// in audit records.
#[inline]
fn decision_of(d: &AccessDecision) -> AuthDecision {
    match d {
        AccessDecision::Allow => AuthDecision::Allow,
        AccessDecision::Deny { .. } => AuthDecision::Deny,
        AccessDecision::Escalate { .. } => AuthDecision::Escalate,
    }
}

/// Map an `AccessDecision` (consumed) to the public `Result<EntityRef,
/// AuthError>` surface. `Allow` produces an `EntityRef` from the
/// already-resolved `SchemaPtr`; `Deny`/`Escalate` propagate the reason
/// through `AuthError`.
#[inline]
fn map_decision(
    d: AccessDecision,
    schema_ptr: SchemaPtr,
) -> Result<EntityRef, AuthError> {
    match d {
        AccessDecision::Allow => Ok(EntityRef { schema_ptr }),
        AccessDecision::Deny { reason } => Err(AuthError::Denied(reason)),
        AccessDecision::Escalate { reason } => Err(AuthError::Escalation(reason)),
    }
}

/// Project a `SchemaPtr` (canonical OGIT pointer with 8-bit namespace +
/// 16-bit entity-type) onto the 3-byte `OwlIdentity` carried in audit
/// events. `entity_type_id` flows through full-width to `slot` — no
/// truncation, no aliasing across the 256-entity boundary.
#[inline]
fn owl_from_schema_ptr(ptr: &SchemaPtr) -> OwlIdentity {
    let family = OgitFamily(ptr.namespace_id().raw());
    let slot = ptr.entity_type_id();
    OwlIdentity::new(family, slot)
}

/// Current wall-clock time in milliseconds since UNIX epoch — used as the
/// `ts_unix_ms` field on audit events. Saturates to `u64::MAX` if the
/// system clock is set absurdly far in the future (~292M years).
fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis().min(u64::MAX as u128) as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_ontology::OntologyRegistry;
    use lance_graph_rbac::policy::smb_policy;

    /// Stub bridge for unit tests — bypasses TTL hydration. Locks to namespace
    /// id 1 and resolves any public_name to a synthetic SchemaPtr.
    struct StubBridge {
        registry: Arc<OntologyRegistry>,
    }

    impl NamespaceBridge for StubBridge {
        fn bridge_id(&self) -> &'static str {
            "stub"
        }
        fn registry(&self) -> &OntologyRegistry {
            &self.registry
        }
        fn g_lock(&self) -> lance_graph_ontology::namespace::NamespaceId {
            lance_graph_ontology::namespace::NamespaceId(1)
        }
    }

    fn make_unified() -> UnifiedBridge<StubBridge> {
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        let bridge = Arc::new(StubBridge { registry });
        let policy = Arc::new(smb_policy());
        UnifiedBridge::new(bridge, policy, "accountant", TenantId(1))
    }

    #[test]
    fn tenant_id_is_known_when_nonzero() {
        assert!(!TenantId::UNKNOWN.is_known());
        assert!(TenantId(1).is_known());
        assert_eq!(TenantId(42).raw(), 42);
    }

    #[test]
    fn auth_error_predicates() {
        let denied = AuthError::Denied("nope");
        assert!(denied.is_denied());
        assert!(!denied.is_escalation());

        let esc = AuthError::Escalation("needs MFA");
        assert!(!esc.is_denied());
        assert!(esc.is_escalation());
    }

    #[test]
    fn unified_bridge_carries_actor_and_tenant() {
        let unified = make_unified();
        assert_eq!(unified.actor_role(), "accountant");
        assert_eq!(unified.tenant(), TenantId(1));
        assert_eq!(unified.bridge().bridge_id(), "stub");
    }

    #[test]
    fn unified_bridge_unknown_public_name_returns_bridge_error() {
        // The stub registry has no rows, so `entity("Customer")` returns
        // `BridgeError::NotInScope` before the policy is even consulted.
        let unified = make_unified();
        let err = unified
            .authorize_read("Customer", PrefetchDepth::Identity)
            .unwrap_err();
        assert!(matches!(err, AuthError::Bridge(_)));
    }

    // ── Alias vs canonical-name resolution (Codex P2 review fix) ──────────────

    /// Bridge that locks to a real namespace + uses a real registry, so
    /// `bridge.row()` resolves and returns a `MappingRow` carrying the
    /// canonical OGIT URI. Used by the alias-resolution tests below.
    struct WoaLikeBridge {
        registry: Arc<OntologyRegistry>,
        g_lock: lance_graph_ontology::namespace::NamespaceId,
    }

    impl NamespaceBridge for WoaLikeBridge {
        fn bridge_id(&self) -> &'static str {
            "woa"
        }
        fn registry(&self) -> &OntologyRegistry {
            &self.registry
        }
        fn g_lock(&self) -> lance_graph_ontology::namespace::NamespaceId {
            self.g_lock
        }
    }

    /// Build a registry with a single mapping where `public_name` ("WorkOrder",
    /// the bridge-side alias) differs from the canonical OGIT URI's name part
    /// ("Order"). Returns the bridge that resolves it.
    fn alias_test_bridge() -> Arc<WoaLikeBridge> {
        use lance_graph_contract::property::{Marking, Schema};
        use lance_graph_ontology::namespace::OgitUri;
        use lance_graph_ontology::proposal::{MappingProposal, MappingProposalKind};

        let registry = Arc::new(OntologyRegistry::new_in_memory());
        let canonical_uri = OgitUri::parse("ogit.WorkOrder:Order").unwrap();
        let proposal = MappingProposal {
            public_name: "WorkOrder".to_string(),
            bridge_id: "woa".to_string(),
            ogit_uri: canonical_uri,
            namespace: "WorkOrder".to_string(),
            kind: MappingProposalKind::Entity {
                schema: Schema::builder("Order").required("id").build(),
            },
            marking: Marking::Internal,
            confidence: 1.0,
            source_uri: "test://woa-alias".to_string(),
            checksum: "checksum-woa-alias".to_string(),
            created_by: "test".to_string(),
        };
        registry.append_mapping(proposal).unwrap();
        let g_lock = registry.namespace_id("WorkOrder").unwrap();
        Arc::new(WoaLikeBridge { registry, g_lock })
    }

    fn policy_with_role(role_name: &'static str, entity_type: &'static str) -> Policy {
        use lance_graph_rbac::permission::PermissionSpec;
        use lance_graph_rbac::role::Role;
        Policy::new("alias-test")
            .with_role(
                Role::new(role_name)
                    .with_permission(PermissionSpec::read_at(entity_type, PrefetchDepth::Detail)),
            )
    }

    #[test]
    fn unified_bridge_evaluates_policy_against_canonical_entity_type() {
        // Caller invokes `authorize_read("WorkOrder", ...)` — the bridge-side
        // alias. Policy is authored against the canonical OGIT entity type
        // "Order" (e.g. shared across multiple bridges that all resolve to
        // the same canonical type). The fix: policy must see "Order", not
        // "WorkOrder".
        let bridge = alias_test_bridge();
        let policy = Arc::new(policy_with_role("clerk", "Order"));
        let unified = UnifiedBridge::new(bridge, policy, "clerk", TenantId(1));

        let entity = unified
            .authorize_read("WorkOrder", PrefetchDepth::Detail)
            .expect("policy keyed on canonical 'Order' should grant alias 'WorkOrder' access");
        assert_eq!(entity.schema_ptr.namespace_id().raw(), 1, "g-lock honored");
    }

    #[test]
    fn unified_bridge_does_not_honor_alias_keyed_policy() {
        // Inverse case: a policy keyed on the bridge-side alias "WorkOrder"
        // does NOT grant access through the canonical-name evaluation path.
        // This is the deliberate decoupling — consumer-facing aliases stay
        // separate from policy authorship; policy authors write canonical
        // OGIT names once and any bridge that resolves to them honors the
        // grant.
        let bridge = alias_test_bridge();
        let policy = Arc::new(policy_with_role("clerk", "WorkOrder"));
        let unified = UnifiedBridge::new(bridge, policy, "clerk", TenantId(1));

        let err = unified
            .authorize_read("WorkOrder", PrefetchDepth::Detail)
            .expect_err("policy keyed on alias 'WorkOrder' should NOT grant access; canonical is 'Order'");
        assert!(matches!(err, AuthError::Denied(_)), "got: {err:?}");
    }

    // ── D-SDR-5: audit emission tests ─────────────────────────────────────────

    /// Recording sink — captures every emitted event for assertions.
    #[derive(Default)]
    struct RecordingSink {
        events: std::sync::Mutex<Vec<UnifiedAuditEvent>>,
    }

    impl RecordingSink {
        fn snapshot(&self) -> Vec<UnifiedAuditEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    impl UnifiedAuditSink for RecordingSink {
        fn emit(&self, event: &UnifiedAuditEvent) {
            self.events.lock().unwrap().push(*event);
        }
    }

    #[test]
    fn authorize_read_emits_allow_audit_event() {
        let bridge = alias_test_bridge();
        let policy = Arc::new(policy_with_role("clerk", "Order"));
        let sink: Arc<RecordingSink> = Arc::new(RecordingSink::default());
        let unified = UnifiedBridge::new(bridge, policy, "clerk", TenantId(7))
            .with_audit_chain(SuperDomain::WorkOrderBilling, 0xDEAD_BEEF, sink.clone());

        let _ = unified
            .authorize_read("WorkOrder", PrefetchDepth::Detail)
            .expect("allow");

        let events = sink.snapshot();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].decision, AuthDecision::Allow);
        assert_eq!(events[0].op, AuthOp::Read);
        assert_eq!(events[0].tenant, TenantId(7));
        // Event super_domain is stamped from the configured AuditChain so
        // compliance partitioning works before FAMILY_TO_SUPER_DOMAIN is
        // hydrated (D-SDR-3b, sprint 5).
        assert_eq!(events[0].super_domain, SuperDomain::WorkOrderBilling);
        assert_ne!(events[0].merkle_root, AuditMerkleRoot::GENESIS);
    }

    #[test]
    fn authorize_read_emits_deny_audit_event() {
        // Policy keyed on the alias → canonical lookup denies; audit still fires.
        let bridge = alias_test_bridge();
        let policy = Arc::new(policy_with_role("clerk", "WorkOrder"));
        let sink: Arc<RecordingSink> = Arc::new(RecordingSink::default());
        let unified = UnifiedBridge::new(bridge, policy, "clerk", TenantId(7))
            .with_audit_chain(SuperDomain::WorkOrderBilling, 0, sink.clone());

        let _ = unified
            .authorize_read("WorkOrder", PrefetchDepth::Detail)
            .expect_err("deny");

        let events = sink.snapshot();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].decision, AuthDecision::Deny);
    }

    #[test]
    fn bridge_error_short_circuits_before_audit() {
        // Stub registry empty → bridge.row returns NotInScope → no audit.
        let sink: Arc<RecordingSink> = Arc::new(RecordingSink::default());
        let registry = Arc::new(OntologyRegistry::new_in_memory());
        let bridge = Arc::new(StubBridge { registry });
        let policy = Arc::new(smb_policy());
        let unified = UnifiedBridge::new(bridge, policy, "accountant", TenantId(1))
            .with_audit_chain(SuperDomain::Unknown, 0, sink.clone());

        let _ = unified
            .authorize_read("Customer", PrefetchDepth::Identity)
            .expect_err("bridge error");

        assert!(sink.snapshot().is_empty(), "no audit on bridge error");
    }

    #[test]
    fn audit_chain_advances_across_calls() {
        let bridge = alias_test_bridge();
        let policy = Arc::new(policy_with_role("clerk", "Order"));
        let sink: Arc<RecordingSink> = Arc::new(RecordingSink::default());
        let unified = UnifiedBridge::new(bridge, policy, "clerk", TenantId(1))
            .with_audit_chain(SuperDomain::WorkOrderBilling, 42, sink.clone());

        let r0 = unified.audit_root();
        let _ = unified.authorize_read("WorkOrder", PrefetchDepth::Detail);
        let r1 = unified.audit_root();
        let _ = unified.authorize_read("WorkOrder", PrefetchDepth::Detail);
        let r2 = unified.audit_root();

        assert_ne!(r0, r1);
        assert_ne!(r1, r2);
        let events = sink.snapshot();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].merkle_root, r1);
        assert_eq!(events[1].merkle_root, r2);
    }

    #[test]
    fn with_audit_chain_resume_picks_up_from_prior_root() {
        let bridge = alias_test_bridge();
        let policy = Arc::new(policy_with_role("clerk", "Order"));
        let sink: Arc<RecordingSink> = Arc::new(RecordingSink::default());
        let prior = AuditMerkleRoot(0xCAFE_F00D);
        let unified = UnifiedBridge::new(bridge, policy, "clerk", TenantId(1))
            .with_audit_chain_resume(
                SuperDomain::WorkOrderBilling,
                7,
                prior,
                sink.clone(),
            );

        assert_eq!(unified.audit_root(), prior);
        let _ = unified.authorize_read("WorkOrder", PrefetchDepth::Detail);
        assert_ne!(unified.audit_root(), prior);
    }
}
