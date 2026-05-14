//! `CognitiveBridgeGate` — injection point for cross-tenant authorization
//! in the cognitive pipeline.
//!
//! ## Design rules
//!
//! - **No `lance-graph-callcenter` dependency.** This trait lives here so
//!   `thinking-engine` remains the lower-level crate. The production impl
//!   `UnifiedBridgeGate<B>` lives in `lance-graph-callcenter` and depends on
//!   this crate (not the reverse).
//! - **All methods synchronous.** The cognitive pipeline is not async.
//! - **Zero-cost default.** `PassthroughGate` (the default) unconditionally
//!   allows every op with a single branch. No allocations, no locks.
//!
//! ## Gated op categories (per spec §1.2)
//!
//! - **Category A** — Cross-tenant sensor-lens retrieval (jina/bge-m3/reranker).
//! - **Category B** — Persona switch reading shared archetype corpus.
//! - **Category C** — L6 delegation / L8 integration (multi-tenant).
//!
//! Pure ops (encode, qualia compute, l4 learn, spiral geometry, calibration)
//! bypass the gate entirely.

// ═══════════════════════════════════════════════════════════════════════════
// CognitiveOpKind — taxonomy for Category C ops
// ═══════════════════════════════════════════════════════════════════════════

/// Which cognitive operation is requesting authorization (Category C).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CognitiveOpKind {
    /// L6 fan-out: delegating to multiple lenses scoped to different tenants.
    L6Delegation = 1,
    /// L8 integration: aggregating cross-tenant evidence.
    L8Integration = 2,
    /// Direct qualia-vector write crossing a tenant boundary.
    QualiaWrite = 3,
    /// MetaWord style-bits commit crossing a tenant boundary.
    MetaWordCommit = 4,
}

impl std::fmt::Display for CognitiveOpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::L6Delegation => write!(f, "L6Delegation"),
            Self::L8Integration => write!(f, "L8Integration"),
            Self::QualiaWrite => write!(f, "QualiaWrite"),
            Self::MetaWordCommit => write!(f, "MetaWordCommit"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CognitiveAuthResult
// ═══════════════════════════════════════════════════════════════════════════

/// Authorization decision returned by every `CognitiveBridgeGate` method.
///
/// - `Allow` — proceed with the cross-tenant op.
/// - `Deny` — abort; caller must surface `CognitiveBridgeError::Denied`.
/// - `Escalate` — a human-approval / MFA step is required before retrying.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CognitiveAuthResult {
    Allow,
    Deny,
    Escalate,
}

impl CognitiveAuthResult {
    /// `true` when the caller may proceed.
    #[inline]
    pub const fn is_allowed(self) -> bool {
        matches!(self, Self::Allow)
    }

    /// `true` when the caller must not proceed.
    #[inline]
    pub const fn is_denied(self) -> bool {
        !self.is_allowed()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CognitiveBridgeError — surface to callers
// ═══════════════════════════════════════════════════════════════════════════

/// Error returned when a `CognitiveBridgeGate` denies or escalates an op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CognitiveBridgeError {
    /// Gate returned `Deny`. The cross-tenant op was not executed.
    Denied,
    /// Gate returned `Escalate`. Human approval / MFA is required first.
    Escalation,
}

impl std::fmt::Display for CognitiveBridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Denied => write!(f, "cognitive bridge: access denied"),
            Self::Escalation => write!(f, "cognitive bridge: escalation required"),
        }
    }
}

impl std::error::Error for CognitiveBridgeError {}

/// Convert a `CognitiveAuthResult` to `Result<(), CognitiveBridgeError>`.
///
/// Convenience for call sites that want to propagate an error early.
#[inline]
pub fn auth_to_result(result: CognitiveAuthResult) -> Result<(), CognitiveBridgeError> {
    match result {
        CognitiveAuthResult::Allow => Ok(()),
        CognitiveAuthResult::Deny => Err(CognitiveBridgeError::Denied),
        CognitiveAuthResult::Escalate => Err(CognitiveBridgeError::Escalation),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CognitiveBridgeGate — the trait
// ═══════════════════════════════════════════════════════════════════════════

/// Injection point for cross-tenant authorization in the cognitive pipeline.
///
/// **Production impl:** `UnifiedBridgeGate<B>` in `lance-graph-callcenter`.
/// **Default impl:** [`PassthroughGate`] — unconditionally allows everything.
///
/// All methods are synchronous. Implementations must be `Send + Sync` so
/// they can be shared via `Arc<dyn CognitiveBridgeGate>` across threads.
pub trait CognitiveBridgeGate: Send + Sync {
    /// Category A — cross-tenant sensor-lens retrieval.
    ///
    /// Called before any ANN / codebook lookup that queries a shared embedding
    /// index (jina / bge-m3 / reranker). `entity_type` is a human-readable
    /// label ("Document", "Persona", etc.); `depth` maps to `PrefetchDepth`
    /// on the callcenter side.
    fn authorize_retrieval(
        &self,
        tenant_id: u32,
        entity_type: &str,
        depth: u8,
    ) -> CognitiveAuthResult;

    /// Category B — persona switch touching a shared archetype corpus.
    ///
    /// `mode` is the `PersonaMode` ordinal (avoids coupling to the enum):
    /// `0 = Work`, `1 = Personal`, `2 = Hybrid`. Called before the mode is
    /// committed; returning `Deny` / `Escalate` leaves the persona unchanged.
    fn authorize_persona_switch(&self, tenant_id: u32, mode: u8) -> CognitiveAuthResult;

    /// Category C — L6 fan-out or L8 integration across tenant boundaries.
    fn authorize_cognitive_op(
        &self,
        tenant_id: u32,
        op_kind: CognitiveOpKind,
    ) -> CognitiveAuthResult;
}

// ═══════════════════════════════════════════════════════════════════════════
// PassthroughGate — default / standalone impl (NoopGate alias in spec)
// ═══════════════════════════════════════════════════════════════════════════

/// Default gate — unconditionally allows every op. Zero overhead.
///
/// Used when `thinking-engine` runs standalone without a callcenter.
/// The `Arc<dyn CognitiveBridgeGate>` fields in sensors default to
/// `Arc::new(PassthroughGate)`.
pub struct PassthroughGate;

impl CognitiveBridgeGate for PassthroughGate {
    #[inline]
    fn authorize_retrieval(
        &self,
        _tenant_id: u32,
        _entity_type: &str,
        _depth: u8,
    ) -> CognitiveAuthResult {
        CognitiveAuthResult::Allow
    }

    #[inline]
    fn authorize_persona_switch(&self, _tenant_id: u32, _mode: u8) -> CognitiveAuthResult {
        CognitiveAuthResult::Allow
    }

    #[inline]
    fn authorize_cognitive_op(
        &self,
        _tenant_id: u32,
        _op_kind: CognitiveOpKind,
    ) -> CognitiveAuthResult {
        CognitiveAuthResult::Allow
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DenyAllGate — test helper; denies every op
// ═══════════════════════════════════════════════════════════════════════════

/// Gate that denies every op. Useful in tests to verify that gated paths
/// are unreachable when the gate is strict.
pub struct DenyAllGate;

impl CognitiveBridgeGate for DenyAllGate {
    #[inline]
    fn authorize_retrieval(
        &self,
        _tenant_id: u32,
        _entity_type: &str,
        _depth: u8,
    ) -> CognitiveAuthResult {
        CognitiveAuthResult::Deny
    }

    #[inline]
    fn authorize_persona_switch(&self, _tenant_id: u32, _mode: u8) -> CognitiveAuthResult {
        CognitiveAuthResult::Deny
    }

    #[inline]
    fn authorize_cognitive_op(
        &self,
        _tenant_id: u32,
        _op_kind: CognitiveOpKind,
    ) -> CognitiveAuthResult {
        CognitiveAuthResult::Deny
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn passthrough_allows_all() {
        let gate = PassthroughGate;
        assert_eq!(
            gate.authorize_retrieval(1, "Document", 0),
            CognitiveAuthResult::Allow
        );
        assert_eq!(
            gate.authorize_persona_switch(1, 0),
            CognitiveAuthResult::Allow
        );
        assert_eq!(
            gate.authorize_cognitive_op(1, CognitiveOpKind::L6Delegation),
            CognitiveAuthResult::Allow
        );
        assert_eq!(
            gate.authorize_cognitive_op(1, CognitiveOpKind::L8Integration),
            CognitiveAuthResult::Allow
        );
        assert_eq!(
            gate.authorize_cognitive_op(1, CognitiveOpKind::QualiaWrite),
            CognitiveAuthResult::Allow
        );
        assert_eq!(
            gate.authorize_cognitive_op(1, CognitiveOpKind::MetaWordCommit),
            CognitiveAuthResult::Allow
        );
    }

    #[test]
    fn deny_all_denies_all() {
        let gate = DenyAllGate;
        assert_eq!(
            gate.authorize_retrieval(1, "Document", 0),
            CognitiveAuthResult::Deny
        );
        assert_eq!(
            gate.authorize_persona_switch(1, 2),
            CognitiveAuthResult::Deny
        );
        assert_eq!(
            gate.authorize_cognitive_op(1, CognitiveOpKind::L8Integration),
            CognitiveAuthResult::Deny
        );
    }

    #[test]
    fn auth_to_result_allow() {
        assert!(auth_to_result(CognitiveAuthResult::Allow).is_ok());
    }

    #[test]
    fn auth_to_result_deny() {
        assert_eq!(
            auth_to_result(CognitiveAuthResult::Deny),
            Err(CognitiveBridgeError::Denied)
        );
    }

    #[test]
    fn auth_to_result_escalate() {
        assert_eq!(
            auth_to_result(CognitiveAuthResult::Escalate),
            Err(CognitiveBridgeError::Escalation)
        );
    }

    #[test]
    fn gate_as_arc_dyn() {
        let gate: Arc<dyn CognitiveBridgeGate> = Arc::new(PassthroughGate);
        assert!(gate.authorize_retrieval(42, "Persona", 1).is_allowed());
    }

    #[test]
    fn cognitive_op_kind_display() {
        assert_eq!(CognitiveOpKind::L6Delegation.to_string(), "L6Delegation");
        assert_eq!(CognitiveOpKind::L8Integration.to_string(), "L8Integration");
    }

    #[test]
    fn cognitive_auth_result_predicates() {
        assert!(CognitiveAuthResult::Allow.is_allowed());
        assert!(!CognitiveAuthResult::Allow.is_denied());
        assert!(CognitiveAuthResult::Deny.is_denied());
        assert!(CognitiveAuthResult::Escalate.is_denied());
    }

    // ── NoopGate integration: thinking-engine works standalone ───────────────

    /// Simulate the full gated lookup path with PassthroughGate. Confirms that
    /// the default gate never blocks the codebook lookup.
    #[test]
    fn passthrough_gate_noop_integration() {
        let gate: Arc<dyn CognitiveBridgeGate> = Arc::new(PassthroughGate);
        let tenant_id = 7u32;

        // Category A: retrieval
        let result = auth_to_result(gate.authorize_retrieval(tenant_id, "Document", 0));
        assert!(result.is_ok(), "PassthroughGate must allow retrieval");

        // Category B: persona switch
        let result = auth_to_result(gate.authorize_persona_switch(tenant_id, 1));
        assert!(result.is_ok(), "PassthroughGate must allow persona switch");

        // Category C: cognitive op
        let result =
            auth_to_result(gate.authorize_cognitive_op(tenant_id, CognitiveOpKind::L6Delegation));
        assert!(result.is_ok(), "PassthroughGate must allow L6 delegation");
    }

    /// Pure-op test: confirm that pure ops (encode, distance lookup) are
    /// NOT routed through the gate. This is a documentation/design test —
    /// there is no gate call in those paths, so they succeed unconditionally.
    #[test]
    fn pure_ops_dont_touch_gate() {
        // Pure codebook lookup — gate is not called.
        let centroid = crate::jina_lens::jina_lookup(42);
        assert!(centroid < 256);

        let dist = crate::jina_lens::jina_distance(0, 1);
        let _ = dist; // pure math, no gate

        // BGE-M3 pure lookup
        let centroid2 = crate::bge_m3_lens::bge_m3_lookup(100);
        assert!(centroid2 < 256);

        // Reranker pure lookup
        let centroid3 = crate::reranker_lens::reranker_lookup(500);
        assert!(centroid3 < 256);
    }
}
