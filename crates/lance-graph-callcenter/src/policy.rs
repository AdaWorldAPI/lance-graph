//! Policy-layer rewriting framework.
//!
//! Generalization of `crate::rls::RlsRewriter`. The same DataFusion
//! `OptimizerRule` machinery used for tenant-predicate injection serves
//! column masking, row-level encryption, and differential-privacy noise
//! injection. Each policy type is a `PolicyRewriter` impl; they compose
//! via the optimizer rule chain.
//!
//! See PR #278 outlook epiphany E1 for the full motivation.
//!
//! META-AGENT: add `pub mod policy;` to lib.rs gated by `feature = "policy"`.
//! Default to including `policy` feature in `auth-rls-lite` (it's purely
//! additive). Suggested Cargo.toml entry:
//!
//! ```toml
//! policy = ["auth-rls-lite"]
//! ```

use std::sync::Arc;

#[cfg(feature = "auth-rls-lite")]
use datafusion::common::tree_node::Transformed;
#[cfg(feature = "auth-rls-lite")]
use datafusion::common::Result as DFResult;
#[cfg(feature = "auth-rls-lite")]
use datafusion::logical_expr::LogicalPlan;
#[cfg(feature = "auth-rls-lite")]
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};

// ── Policy taxonomy ──────────────────────────────────────────────────────────

/// Policy classification — what kind of transform a rewriter implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PolicyKind {
    /// Inject row-filter predicates (e.g. `tenant_id = 't1'`).
    RowFilter,
    /// Mask / redact / hash columns based on actor role.
    ColumnMask,
    /// Encrypt selected columns at rest using a key handle.
    RowEncryption,
    /// Inject differential-privacy noise into aggregate outputs.
    DifferentialPrivacy,
    /// Emit audit events (read-only side channel).
    Audit,
}

/// Generalized policy rewriter. Implementors transform a LogicalPlan and
/// declare their kind for ordering / introspection.
#[cfg(feature = "auth-rls-lite")]
pub trait PolicyRewriter: Send + Sync + std::fmt::Debug {
    fn kind(&self) -> PolicyKind;
    /// Stable name (e.g. "rls_rewriter", "column_mask"). Used by audit log.
    fn name(&self) -> &'static str;
    /// Rewrite predicate. Default = identity (subclasses override what they need).
    fn rewrite_plan(&self, plan: LogicalPlan) -> DFResult<Transformed<LogicalPlan>> {
        Ok(Transformed::no(plan))
    }
}

// ── Column masking policy ────────────────────────────────────────────────────

/// Per-column redaction mode. Drives how `ColumnMaskRewriter` rewrites
/// referenced expressions in `Projection` nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedactionMode {
    /// Replace with NULL.
    Null,
    /// Replace with a constant ("[REDACTED]").
    Constant,
    /// Hash via FNV-64 (stable across builds).
    Hash,
    /// First-N-chars only (e.g. credit card last-4).
    Truncate(usize),
}

/// Column masking policy: redact / hash / mask values from selected columns
/// based on actor role. Stub UDF reference; concrete UDFs land in a follow-up.
#[derive(Debug, Clone)]
pub struct ColumnMaskPolicy {
    /// Table whose columns this policy applies to.
    pub table_name: String,
    /// Per-column redaction mode. Missing column = unmasked.
    pub columns: std::collections::HashMap<String, RedactionMode>,
}

#[derive(Debug, Default, Clone)]
pub struct ColumnMaskRegistry {
    policies: std::collections::HashMap<String, ColumnMaskPolicy>,
}

impl ColumnMaskRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn register(&mut self, policy: ColumnMaskPolicy) {
        self.policies.insert(policy.table_name.clone(), policy);
    }
    pub fn lookup(&self, table_name: &str) -> Option<&ColumnMaskPolicy> {
        self.policies.get(table_name)
    }
}

#[cfg(feature = "auth-rls-lite")]
#[derive(Debug)]
pub struct ColumnMaskRewriter {
    pub registry: Arc<ColumnMaskRegistry>,
    pub actor_role: String,
}

#[cfg(feature = "auth-rls-lite")]
impl PolicyRewriter for ColumnMaskRewriter {
    fn kind(&self) -> PolicyKind {
        PolicyKind::ColumnMask
    }
    fn name(&self) -> &'static str {
        "column_mask"
    }
    fn rewrite_plan(&self, plan: LogicalPlan) -> DFResult<Transformed<LogicalPlan>> {
        // Walk plan; on Projection, rewrite expressions for redacted columns.
        // For this PR ship the structural skeleton; the actual UDF wrap lands
        // in a follow-up once redaction UDFs are registered.
        // TODO: wrap Expr::Column(c) in mask_udf(...) for c in policy.columns
        Ok(Transformed::no(plan))
    }
}

#[cfg(feature = "auth-rls-lite")]
impl OptimizerRule for ColumnMaskRewriter {
    fn name(&self) -> &str {
        "column_mask_rewriter"
    }
    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }
    fn supports_rewrite(&self) -> bool {
        true
    }
    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> DFResult<Transformed<LogicalPlan>> {
        self.rewrite_plan(plan)
    }
}

// ── Row encryption policy (stub, no executor yet) ────────────────────────────

/// Row encryption policy: encrypt selected columns at rest using a key
/// handle. The actual cipher binding lands in a follow-up; this carries
/// the per-column key-handle association so the registry surface is stable.
#[derive(Debug, Clone)]
pub struct RowEncryptionPolicy {
    /// Table whose columns this policy applies to.
    pub table_name: String,
    /// Per-column key handle. Missing column = unencrypted.
    pub columns: std::collections::HashMap<String, KeyHandle>,
}

/// Opaque key handle. Resolves through a downstream KMS / keyring service.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeyHandle(pub String);

#[derive(Debug, Default, Clone)]
pub struct RowEncryptionRegistry {
    policies: std::collections::HashMap<String, RowEncryptionPolicy>,
}

impl RowEncryptionRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn register(&mut self, policy: RowEncryptionPolicy) {
        self.policies.insert(policy.table_name.clone(), policy);
    }
    pub fn lookup(&self, table_name: &str) -> Option<&RowEncryptionPolicy> {
        self.policies.get(table_name)
    }
}

// ── Differential-privacy policy (stub, no executor yet) ──────────────────────

/// DP noise mechanism. Drives the noise distribution used when the
/// rewriter wraps aggregate outputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DpMechanism {
    /// Laplace mechanism — calibrated to ε.
    Laplace,
    /// Gaussian mechanism — calibrated to (ε, δ).
    Gaussian,
}

/// Differential-privacy policy. Carries the privacy budget and noise
/// mechanism; the rewriter wraps SUM/COUNT/AVG aggregates with calibrated
/// noise injection (follow-up PR).
#[derive(Debug, Clone)]
pub struct DifferentialPrivacyPolicy {
    /// Table this policy applies to.
    pub table_name: String,
    /// Privacy budget. Smaller ε = more noise = stronger privacy.
    pub epsilon: f64,
    /// Noise mechanism.
    pub mechanism: DpMechanism,
}

#[derive(Debug, Default, Clone)]
pub struct DifferentialPrivacyRegistry {
    policies: std::collections::HashMap<String, DifferentialPrivacyPolicy>,
}

impl DifferentialPrivacyRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn register(&mut self, policy: DifferentialPrivacyPolicy) {
        self.policies.insert(policy.table_name.clone(), policy);
    }
    pub fn lookup(&self, table_name: &str) -> Option<&DifferentialPrivacyPolicy> {
        self.policies.get(table_name)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::collections::HashSet;

    #[test]
    fn column_mask_registry_register_lookup() {
        let mut registry = ColumnMaskRegistry::new();
        let mut columns = HashMap::new();
        columns.insert("ssn".to_string(), RedactionMode::Hash);
        columns.insert("card".to_string(), RedactionMode::Truncate(4));
        registry.register(ColumnMaskPolicy {
            table_name: "customers".to_string(),
            columns,
        });

        let policy = registry.lookup("customers").expect("policy registered");
        assert_eq!(policy.table_name, "customers");
        assert_eq!(policy.columns.get("ssn"), Some(&RedactionMode::Hash));
        assert_eq!(
            policy.columns.get("card"),
            Some(&RedactionMode::Truncate(4))
        );
        assert!(registry.lookup("missing").is_none());
    }

    #[test]
    fn redaction_mode_variants_distinct() {
        // Each variant is a distinct value; equality is structural.
        assert_ne!(RedactionMode::Null, RedactionMode::Constant);
        assert_ne!(RedactionMode::Hash, RedactionMode::Null);
        assert_ne!(RedactionMode::Truncate(4), RedactionMode::Truncate(8));
        assert_eq!(RedactionMode::Truncate(4), RedactionMode::Truncate(4));
    }

    #[cfg(feature = "auth-rls-lite")]
    #[test]
    fn column_mask_rewriter_kind_is_column_mask() {
        let rewriter = ColumnMaskRewriter {
            registry: Arc::new(ColumnMaskRegistry::new()),
            actor_role: "analyst".to_string(),
        };
        assert_eq!(rewriter.kind(), PolicyKind::ColumnMask);
        assert_eq!(<ColumnMaskRewriter as PolicyRewriter>::name(&rewriter), "column_mask");
    }

    #[test]
    fn policy_kind_is_hashable_for_dispatch() {
        // PolicyKind being Hash + Eq lets a registry dispatch
        // rewriters by kind in a HashSet/HashMap. Smoke-test the trait
        // bounds by inserting into a HashSet.
        let mut set: HashSet<PolicyKind> = HashSet::new();
        set.insert(PolicyKind::RowFilter);
        set.insert(PolicyKind::ColumnMask);
        set.insert(PolicyKind::RowEncryption);
        set.insert(PolicyKind::DifferentialPrivacy);
        set.insert(PolicyKind::Audit);
        // Inserting a duplicate should be a no-op.
        assert!(!set.insert(PolicyKind::ColumnMask));
        assert_eq!(set.len(), 5);
    }

    #[cfg(feature = "auth-rls-lite")]
    #[test]
    fn column_mask_rewriter_passes_through_for_now() {
        use datafusion::logical_expr::{EmptyRelation, LogicalPlan};
        use std::sync::Arc as StdArc;

        let rewriter = ColumnMaskRewriter {
            registry: Arc::new(ColumnMaskRegistry::new()),
            actor_role: "analyst".to_string(),
        };
        let plan = LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: StdArc::new(datafusion::common::DFSchema::empty()),
        });
        let transformed = rewriter
            .rewrite_plan(plan)
            .expect("rewrite should succeed");
        // Skeleton implementation — should be a no-op until the UDF wrap
        // lands.
        assert!(!transformed.transformed);
    }
}
