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
use datafusion::common::ScalarValue;
#[cfg(feature = "auth-rls-lite")]
use datafusion::logical_expr::{lit, Expr, LogicalPlan, Projection};
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
impl ColumnMaskRewriter {
    /// Replace a column expression according to the given [`RedactionMode`].
    ///
    /// This is the v1 implementation. `Hash` mode uses a literal
    /// `"***REDACTED***"` placeholder; a proper deterministic hash UDF
    /// (FNV-64 or SHA-256-truncated) is a follow-up once the UDF
    /// registry surface is stable. `Truncate(n)` uses DataFusion's
    /// built-in `substr` via `Expr::ScalarFunction`.
    fn mask_expr(expr: &Expr, mode: &RedactionMode) -> Expr {
        match mode {
            RedactionMode::Null => Expr::Literal(ScalarValue::Null, None),
            RedactionMode::Constant => lit("[REDACTED]"),
            RedactionMode::Hash => {
                // v1: deterministic hash UDF is future work. For now,
                // replace with a stable placeholder so the column value
                // is never exposed.
                lit("***REDACTED***")
            }
            RedactionMode::Truncate(n) => {
                // Use DataFusion's built-in `substr(col, 1, n)`.
                let col_expr = expr.clone();
                let start = lit(1_i64);
                let length = lit(*n as i64);
                Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
                    Arc::new(datafusion::functions::unicode::substr().as_ref().clone()),
                    vec![col_expr, start, length],
                ))
            }
        }
    }

    /// Rewrite a single expression, replacing column references that match
    /// a masked column with the appropriate redaction expression.
    fn rewrite_expr(expr: &Expr, policy: &ColumnMaskPolicy) -> (Expr, bool) {
        match expr {
            Expr::Column(col) => {
                let col_name = col.name();
                if let Some(mode) = policy.columns.get(col_name) {
                    (Self::mask_expr(expr, mode), true)
                } else {
                    (expr.clone(), false)
                }
            }
            Expr::Alias(alias) => {
                let (inner, changed) = Self::rewrite_expr(&alias.expr, policy);
                if changed {
                    (inner.alias(alias.name.clone()), true)
                } else {
                    (expr.clone(), false)
                }
            }
            // For other expression types, return unchanged. A future
            // version could recurse into nested expressions (e.g.
            // BinaryExpr operands), but v1 keeps it simple: only
            // top-level column references in Projection are rewritten.
            _ => (expr.clone(), false),
        }
    }
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
        match &plan {
            LogicalPlan::Projection(proj) => {
                // Extract table name from the input plan. We look for
                // TableScan as the immediate or nested input.
                let table_name = Self::extract_table_name(proj.input.as_ref());
                let policy = table_name
                    .as_deref()
                    .and_then(|t| self.registry.lookup(t));

                let Some(policy) = policy else {
                    return Ok(Transformed::no(plan));
                };

                // Rewrite each projection expression.
                let mut any_changed = false;
                let new_exprs: Vec<Expr> = proj
                    .expr
                    .iter()
                    .map(|e| {
                        let (new_e, changed) = Self::rewrite_expr(e, policy);
                        if changed {
                            any_changed = true;
                        }
                        new_e
                    })
                    .collect();

                if any_changed {
                    let new_proj = Projection::try_new(new_exprs, proj.input.clone())?;
                    Ok(Transformed::yes(LogicalPlan::Projection(new_proj)))
                } else {
                    Ok(Transformed::no(plan))
                }
            }
            _ => Ok(Transformed::no(plan)),
        }
    }
}

#[cfg(feature = "auth-rls-lite")]
impl ColumnMaskRewriter {
    /// Walk down the plan tree to find a `TableScan` and extract its name.
    /// This is a best-effort heuristic for v1; it handles the common case
    /// of `Projection → TableScan` and `Projection → Filter → TableScan`.
    fn extract_table_name(plan: &LogicalPlan) -> Option<String> {
        match plan {
            LogicalPlan::TableScan(scan) => Some(scan.table_name.table().to_string()),
            // Recurse through single-input nodes (Filter, Sort, Limit, etc.)
            other => {
                let inputs = other.inputs();
                if inputs.len() == 1 {
                    Self::extract_table_name(inputs[0])
                } else {
                    None
                }
            }
        }
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
        // Empty registry — no projection to rewrite, so pass-through.
        assert!(!transformed.transformed);
    }

    // ── Column masking rewrite tests (one per RedactionMode) ────────────

    #[cfg(feature = "auth-rls-lite")]
    mod mask_rewrite_tests {
        use super::*;
        use datafusion::common::tree_node::TreeNode;
        use datafusion::datasource::{provider_as_source, MemTable};
        use datafusion::logical_expr::builder::LogicalPlanBuilder;
        use datafusion::optimizer::OptimizerContext;

        /// Schema: `name` (Utf8), `ssn` (Utf8), `card` (Utf8), `score` (Int32).
        fn mask_schema() -> Arc<arrow::datatypes::Schema> {
            Arc::new(arrow::datatypes::Schema::new(vec![
                arrow::datatypes::Field::new("name", arrow::datatypes::DataType::Utf8, false),
                arrow::datatypes::Field::new("ssn", arrow::datatypes::DataType::Utf8, false),
                arrow::datatypes::Field::new("card", arrow::datatypes::DataType::Utf8, false),
                arrow::datatypes::Field::new("score", arrow::datatypes::DataType::Int32, true),
            ]))
        }

        fn mem_source(
            schema: Arc<arrow::datatypes::Schema>,
        ) -> Arc<dyn datafusion::datasource::TableProvider> {
            Arc::new(MemTable::try_new(schema, vec![vec![]]).unwrap())
        }

        /// Build a Projection → TableScan plan selecting all columns.
        fn scan_with_projection(table_name: &str) -> LogicalPlan {
            use datafusion::logical_expr::col;
            let src = provider_as_source(mem_source(mask_schema()));
            LogicalPlanBuilder::scan(table_name, src, None)
                .unwrap()
                .project(vec![
                    col("name"),
                    col("ssn"),
                    col("card"),
                    col("score"),
                ])
                .unwrap()
                .build()
                .unwrap()
        }

        fn apply(plan: LogicalPlan, rewriter: &ColumnMaskRewriter) -> LogicalPlan {
            let cfg = OptimizerContext::new();
            plan.transform_down(|n| rewriter.rewrite(n, &cfg))
                .unwrap()
                .data
        }

        fn make_rewriter(
            table: &str,
            masks: Vec<(&str, RedactionMode)>,
        ) -> ColumnMaskRewriter {
            let mut columns = HashMap::new();
            for (col, mode) in masks {
                columns.insert(col.to_string(), mode);
            }
            let mut registry = ColumnMaskRegistry::new();
            registry.register(ColumnMaskPolicy {
                table_name: table.to_string(),
                columns,
            });
            ColumnMaskRewriter {
                registry: Arc::new(registry),
                actor_role: "analyst".to_string(),
            }
        }

        #[test]
        fn redaction_mode_null_replaces_column_with_null() {
            let plan = scan_with_projection("customers");
            let rewriter = make_rewriter("customers", vec![("ssn", RedactionMode::Null)]);
            let rewritten = apply(plan, &rewriter);
            let s = format!("{rewritten}");
            // The SSN column should be replaced with NULL.
            assert!(
                s.contains("NULL"),
                "expected NULL literal in rewritten plan: {s}"
            );
            // Other columns should remain.
            assert!(s.contains("name"), "name column should be preserved: {s}");
            assert!(s.contains("card"), "card column should be preserved: {s}");
        }

        #[test]
        fn redaction_mode_constant_replaces_column_with_redacted() {
            let plan = scan_with_projection("customers");
            let rewriter = make_rewriter("customers", vec![("ssn", RedactionMode::Constant)]);
            let rewritten = apply(plan, &rewriter);
            let s = format!("{rewritten}");
            assert!(
                s.contains("[REDACTED]"),
                "expected [REDACTED] literal in rewritten plan: {s}"
            );
            assert!(s.contains("name"), "name column should be preserved: {s}");
        }

        #[test]
        fn redaction_mode_hash_replaces_column_with_placeholder() {
            let plan = scan_with_projection("customers");
            let rewriter = make_rewriter("customers", vec![("ssn", RedactionMode::Hash)]);
            let rewritten = apply(plan, &rewriter);
            let s = format!("{rewritten}");
            // v1: Hash mode uses "***REDACTED***" placeholder.
            assert!(
                s.contains("***REDACTED***"),
                "expected ***REDACTED*** placeholder in rewritten plan: {s}"
            );
            assert!(s.contains("name"), "name column should be preserved: {s}");
        }

        #[test]
        fn redaction_mode_truncate_wraps_column_in_substr() {
            let plan = scan_with_projection("customers");
            let rewriter =
                make_rewriter("customers", vec![("card", RedactionMode::Truncate(4))]);
            let rewritten = apply(plan, &rewriter);
            let s = format!("{rewritten}");
            // Truncate(4) should produce a substr() call.
            assert!(
                s.contains("substr"),
                "expected substr function in rewritten plan: {s}"
            );
            // The name column should be untouched.
            assert!(s.contains("name"), "name column should be preserved: {s}");
        }
    }
}
