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

use std::sync::Arc;

use lance_graph_contract::property::PrefetchDepth;
use lance_graph_ontology::bridge::{BridgeError, EntityRef, NamespaceBridge};
use lance_graph_rbac::access::AccessDecision;
use lance_graph_rbac::policy::{Operation, Policy};

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

/// 2 bytes. BF16-shaped container (interpreted as named bit-fields, not
/// literal floating-point semantics).
/// High 8 bits = `OgitFamily` (the precise heel pointer / "mantissa").
/// Low  8 bits = within-family slot (the OWL/consumer's own identity).
/// This is what rides on every LanceDB row.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OwlIdentity(pub u16);

impl OwlIdentity
{
    pub const UNKNOWN: Self = Self(0);

    #[inline]
    pub const fn new(family: OgitFamily, slot: u8) -> Self
    {
        Self(((family.0 as u16) << 8) | slot as u16)
    }

    #[inline]
    pub const fn family(self) -> OgitFamily
    {
        OgitFamily((self.0 >> 8) as u8)
    }

    #[inline]
    pub const fn slot(self) -> u8
    {
        (self.0 & 0xFF) as u8
    }

    #[inline]
    pub const fn raw(self) -> u16
    {
        self.0
    }

    /// Bitmask predicate Cypher MATCH lowers to. No string lookup.
    #[inline]
    pub const fn is_family(self, f: OgitFamily) -> bool
    {
        self.family().0 == f.0
    }

    /// Within-family slot predicate.
    #[inline]
    pub const fn is_slot(self, s: u8) -> bool
    {
        self.slot() == s
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
    tenant: TenantId,
}

impl<B: NamespaceBridge> UnifiedBridge<B> {
    /// Construct a new unified bridge.
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
            tenant,
        }
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

    // ── Authorization ─────────────────────────────────────────────────────────

    /// Resolve `public_name` through the bridge then evaluate read access at
    /// `depth`. Returns the resolved `EntityRef` on `Allow`, `AuthError` on
    /// `Deny` / `Escalate` / bridge resolution failure.
    pub fn authorize_read(
        &self,
        public_name: &str,
        depth: PrefetchDepth,
    ) -> Result<EntityRef, AuthError> {
        let entity = self.bridge.entity(public_name)?;
        self.evaluate(public_name, Operation::Read { depth })?;
        Ok(entity)
    }

    /// Resolve `public_name` then evaluate write access on `predicate`.
    pub fn authorize_write(
        &self,
        public_name: &str,
        predicate: &str,
    ) -> Result<EntityRef, AuthError> {
        let entity = self.bridge.entity(public_name)?;
        self.evaluate(public_name, Operation::Write { predicate })?;
        Ok(entity)
    }

    /// Resolve `public_name` then evaluate action access on `action`.
    pub fn authorize_act(
        &self,
        public_name: &str,
        action: &str,
    ) -> Result<EntityRef, AuthError> {
        let entity = self.bridge.entity(public_name)?;
        self.evaluate(public_name, Operation::Act { action })?;
        Ok(entity)
    }

    fn evaluate(&self, entity_type: &str, op: Operation<'_>) -> Result<(), AuthError> {
        match self.policy.evaluate(self.actor_role, entity_type, op) {
            AccessDecision::Allow => Ok(()),
            AccessDecision::Deny { reason } => Err(AuthError::Denied(reason)),
            AccessDecision::Escalate { reason } => Err(AuthError::Escalation(reason)),
        }
    }
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
}
