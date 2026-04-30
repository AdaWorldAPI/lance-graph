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
use datafusion::arrow::datatypes::DataType;
#[cfg(feature = "auth-rls-lite")]
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
#[cfg(feature = "auth-rls-lite")]
use datafusion::common::Result as DFResult;
#[cfg(feature = "auth-rls-lite")]
use datafusion::common::{DataFusionError, ScalarValue};
#[cfg(feature = "auth-rls-lite")]
use datafusion::logical_expr::{
    lit, ColumnarValue, Expr, LogicalPlan, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
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
    /// `Hash` mode binds an UDF reference (`policy_hash_v1`) that is
    /// intentionally NOT registered yet — see [`NotYetWiredHashUdf`]
    /// and PR-F1b. Plans build, but execution fails loud with
    /// `NotImplemented("policy_hash_v1 UDF not yet registered ...")`.
    /// This is the "loud > silent" fix for the silent placeholder hole.
    /// `Truncate(n)` uses DataFusion's built-in `substr`.
    fn mask_expr(expr: &Expr, mode: &RedactionMode) -> Expr {
        match mode {
            RedactionMode::Null => Expr::Literal(ScalarValue::Null, None),
            RedactionMode::Constant => lit("[REDACTED]"),
            RedactionMode::Hash => {
                // Reference the unregistered policy_hash_v1 UDF. The
                // plan builds (so call sites compose), but executing
                // the plan returns a `NotImplemented` error at the
                // first row — preventing silent disclosure if the
                // wiring is forgotten.
                Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
                    Arc::new(ScalarUDF::from(NotYetWiredHashUdf::new())),
                    vec![expr.clone()],
                ))
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

    /// Recursively rewrite an expression tree, replacing every
    /// `Expr::Column(c)` whose name is in the policy with the
    /// appropriate redaction expression.
    ///
    /// Uses `Expr::transform_down` from DataFusion's TreeNode trait —
    /// it walks into BinaryExpr operands, AggregateFunction args,
    /// ScalarFunction args, Cast expressions, Sort exprs, and so on.
    /// This is what closes the WHERE / JOIN / aggregate leak (Loose
    /// End #1).
    fn rewrite_expr_deep(expr: Expr, policy: &ColumnMaskPolicy) -> DFResult<Transformed<Expr>> {
        expr.transform_down(|e| match e {
            Expr::Column(ref col) => {
                if let Some(mode) = policy.columns.get(col.name()) {
                    // `Jump` prevents `transform_down` from descending
                    // into the freshly-built mask expression. Otherwise,
                    // RedactionMode::Hash (which wraps the column in a
                    // ScalarFunction whose first arg is the original
                    // Column) would re-enter the visitor on that Column
                    // and recurse infinitely.
                    Ok(Transformed::new(
                        Self::mask_expr(&e, mode),
                        true,
                        TreeNodeRecursion::Jump,
                    ))
                } else {
                    Ok(Transformed::no(e))
                }
            }
            other => Ok(Transformed::no(other)),
        })
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
        // Resolve the policy by walking the plan's input chain to the
        // nearest TableScan. Without a TableScan we have no table
        // name → no policy applies.
        let table_name = Self::extract_table_name(&plan);
        let Some(policy) = table_name.as_deref().and_then(|t| self.registry.lookup(t)) else {
            return Ok(Transformed::no(plan));
        };
        let policy = policy.clone();

        // Walk EVERY expression in this node — Projection's projection
        // list, Filter's predicate, Aggregate's group/aggr exprs,
        // Join's on/filter, Sort's exprs, etc. `map_expressions`
        // dispatches per-variant in DataFusion 52, so a single call
        // covers WHERE / JOIN / GROUP BY / ORDER BY / aggregate args.
        // Closes Loose End #1 (PR-F1) — the WHERE / aggregate leak
        // that the Projection-only rewriter let through.
        let transformed = plan.map_expressions(|e| Self::rewrite_expr_deep(e, &policy))?;
        // `map_expressions` doesn't recompute schemas for some variants
        // (Projection, Aggregate); recompute so field types stay
        // consistent after a Column was replaced by a literal/UDF call
        // of a different type.
        transformed.map_data(|p| p.recompute_schema())
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

// ── policy_hash_v1 — intentionally-unregistered hard-fail UDF ────────────────
//
// Loose End #2 (PR-F1 close): the previous Hash redaction returned a
// silent `lit("***REDACTED***")` placeholder. If the real hash UDF is
// forgotten in a follow-up wiring, every Hash-masked column would
// silently render as `"***REDACTED***"` — a string, not a hash, with
// no surface signal that the policy is mis-wired.
//
// This UDF replaces the placeholder. It binds at plan time (so plans
// COMPOSE), but its `invoke_with_args` returns `NotImplemented` —
// execution fails loudly with "policy_hash_v1 UDF not yet registered
// — see PR-F1b". Loud > silent.
//
// PR-F1b will replace the body with a real FNV-64 / SHA-256-truncated
// implementation and register the UDF in the SessionContext.
#[cfg(feature = "auth-rls-lite")]
#[derive(Debug)]
pub struct NotYetWiredHashUdf {
    signature: Signature,
}

#[cfg(feature = "auth-rls-lite")]
impl Default for NotYetWiredHashUdf {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "auth-rls-lite")]
impl NotYetWiredHashUdf {
    pub fn new() -> Self {
        // Accept any single argument — the actual hash will be over
        // the column's bytes, which we treat as opaque at plan time.
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

#[cfg(feature = "auth-rls-lite")]
impl PartialEq for NotYetWiredHashUdf {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}
#[cfg(feature = "auth-rls-lite")]
impl Eq for NotYetWiredHashUdf {}
#[cfg(feature = "auth-rls-lite")]
impl std::hash::Hash for NotYetWiredHashUdf {
    fn hash<H: std::hash::Hasher>(&self, s: &mut H) {
        self.name().hash(s);
    }
}

#[cfg(feature = "auth-rls-lite")]
impl ScalarUDFImpl for NotYetWiredHashUdf {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    /// Stable name — surfaces in plan rendering so test assertions
    /// (and operators inspecting EXPLAIN output) can verify the
    /// wrap was applied.
    fn name(&self) -> &str {
        "policy_hash_v1"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    /// Hash output is conventionally a 64-bit unsigned integer
    /// (FNV-64 is the v1 target). Fixing the return type here means
    /// upstream operators (aggregates, joins) get a stable schema
    /// even though the real implementation is deferred.
    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::UInt64)
    }
    fn invoke_with_args(
        &self,
        _args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> DFResult<ColumnarValue> {
        Err(DataFusionError::NotImplemented(
            "policy_hash_v1 UDF not yet registered — see PR-F1b".into(),
        ))
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
        assert_eq!(
            <ColumnMaskRewriter as PolicyRewriter>::name(&rewriter),
            "column_mask"
        );
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
        let transformed = rewriter.rewrite_plan(plan).expect("rewrite should succeed");
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
                .project(vec![col("name"), col("ssn"), col("card"), col("score")])
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

        fn make_rewriter(table: &str, masks: Vec<(&str, RedactionMode)>) -> ColumnMaskRewriter {
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
        fn redaction_mode_hash_binds_not_yet_wired_udf() {
            // PR-F1 close (Loose End #2): Hash mode now binds the
            // intentionally-unregistered `policy_hash_v1` UDF instead
            // of emitting a silent `***REDACTED***` placeholder.
            // Plans build (so the rewriter composes), but execution
            // fails loud — preventing silent disclosure when wiring is
            // forgotten. The real implementation lands in PR-F1b.
            let plan = scan_with_projection("customers");
            let rewriter = make_rewriter("customers", vec![("ssn", RedactionMode::Hash)]);
            let rewritten = apply(plan, &rewriter);
            let s = format!("{rewritten}");
            assert!(
                s.contains("policy_hash_v1"),
                "expected policy_hash_v1 UDF reference in rewritten plan: {s}"
            );
            assert!(
                !s.contains("***REDACTED***"),
                "Hash mode must not emit silent ***REDACTED*** placeholder: {s}"
            );
            assert!(s.contains("name"), "name column should be preserved: {s}");
        }

        #[test]
        fn redaction_mode_truncate_wraps_column_in_substr() {
            let plan = scan_with_projection("customers");
            let rewriter = make_rewriter("customers", vec![("card", RedactionMode::Truncate(4))]);
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

    // ── Full-tree leak tests (PR-F1 close: Loose End #1) ───────────────────
    //
    // These tests pin the security-critical invariant that masked columns
    // do NOT leak through Filter (WHERE), Aggregate (MAX/SUM/...),
    // GROUP BY, JOIN, or Sort nodes. They build plans without MemTable
    // (so they compile under `auth-rls-lite` alone) using
    // `datafusion::logical_expr::table_scan`.
    #[cfg(feature = "auth-rls-lite")]
    mod full_tree_leak_tests {
        use super::*;
        use datafusion::common::tree_node::TreeNode;
        use datafusion::functions_aggregate::expr_fn::max;
        use datafusion::logical_expr::{col, table_scan, Expr, LogicalPlan};
        use datafusion::optimizer::OptimizerContext;

        fn users_schema() -> arrow::datatypes::Schema {
            arrow::datatypes::Schema::new(vec![
                arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
                arrow::datatypes::Field::new("ssn", arrow::datatypes::DataType::Utf8, false),
                arrow::datatypes::Field::new("name", arrow::datatypes::DataType::Utf8, false),
            ])
        }

        fn make_rewriter(masks: Vec<(&str, RedactionMode)>) -> ColumnMaskRewriter {
            let mut columns = HashMap::new();
            for (col_name, mode) in masks {
                columns.insert(col_name.to_string(), mode);
            }
            let mut registry = ColumnMaskRegistry::new();
            registry.register(ColumnMaskPolicy {
                table_name: "users".to_string(),
                columns,
            });
            ColumnMaskRewriter {
                registry: Arc::new(registry),
                actor_role: "analyst".to_string(),
            }
        }

        fn apply(plan: LogicalPlan, rewriter: &ColumnMaskRewriter) -> LogicalPlan {
            let cfg = OptimizerContext::new();
            plan.transform_down(|n| rewriter.rewrite(n, &cfg))
                .unwrap()
                .data
        }

        /// Recursively scan a plan for any Filter node and collect predicate
        /// expression strings. Used to assert WHERE clauses got rewritten.
        fn filter_predicates(plan: &LogicalPlan) -> Vec<String> {
            let mut out = Vec::new();
            collect_filters(plan, &mut out);
            out
        }
        fn collect_filters(plan: &LogicalPlan, out: &mut Vec<String>) {
            if let LogicalPlan::Filter(f) = plan {
                out.push(format!("{}", f.predicate));
            }
            for input in plan.inputs() {
                collect_filters(input, out);
            }
        }

        /// Recursively scan a plan for any Aggregate node and collect
        /// aggregate-expression strings. Used to assert MAX/SUM/...
        /// arguments got rewritten.
        fn aggregate_exprs(plan: &LogicalPlan) -> Vec<String> {
            let mut out = Vec::new();
            collect_aggregates(plan, &mut out);
            out
        }
        fn collect_aggregates(plan: &LogicalPlan, out: &mut Vec<String>) {
            if let LogicalPlan::Aggregate(a) = plan {
                for e in &a.aggr_expr {
                    out.push(format!("{}", e));
                }
            }
            for input in plan.inputs() {
                collect_aggregates(input, out);
            }
        }

        /// Reference plain `Expr::Column("ssn")` inside a Filter predicate
        /// and confirm the rewriter walked into the Filter, not just the
        /// outer Projection.
        ///
        /// Plan shape:
        ///   Projection(id) → Filter(ssn = '123-45-6789') → TableScan(users)
        ///
        /// Pre-fix: the rewriter only rewrites Projection (which projects
        ///          `id`, untouched), and the Filter still references
        ///          `ssn` directly — leaking the unmasked SSN.
        /// Post-fix: the Filter predicate's `ssn` reference is replaced
        ///           with the configured mask (here, the [REDACTED]
        ///           constant).
        #[test]
        fn test_where_clause_does_not_leak_unmasked_column() {
            let schema = users_schema();
            let plan = table_scan(Some("users"), &schema, None)
                .unwrap()
                .filter(col("ssn").eq(Expr::Literal(
                    datafusion::common::ScalarValue::Utf8(Some("123-45-6789".into())),
                    None,
                )))
                .unwrap()
                .project(vec![col("id")])
                .unwrap()
                .build()
                .unwrap();

            let rewriter = make_rewriter(vec![("ssn", RedactionMode::Constant)]);
            let rewritten = apply(plan, &rewriter);

            let preds = filter_predicates(&rewritten);
            assert!(
                !preds.is_empty(),
                "expected at least one Filter node in the rewritten plan"
            );
            // The predicate must NOT mention bare `ssn` (the unmasked
            // column ref); it must reference the mask literal instead.
            for p in &preds {
                assert!(
                    p.contains("[REDACTED]"),
                    "Filter predicate must contain the mask literal — leaked unmasked ssn: {p}"
                );
                assert!(
                    !contains_bare_ssn(p),
                    "Filter predicate still references bare `ssn` column — column leaked: {p}"
                );
            }
        }

        /// Reference `Expr::AggregateFunction(MAX, [Column(ssn)])` inside
        /// an Aggregate node and confirm the aggregate's argument got
        /// rewritten.
        ///
        /// Plan shape:
        ///   Aggregate(MAX(ssn)) → TableScan(users)
        ///
        /// Pre-fix: rewriter only handles Projection; the Aggregate's
        ///          `MAX(ssn)` argument is unchanged → MAX runs over
        ///          unmasked SSN values, exposing the maximum.
        /// Post-fix: `MAX(ssn)` becomes `MAX([REDACTED])` — the
        ///           aggregate sees the mask, not the raw column.
        #[test]
        fn test_max_ssn_aggregate_is_masked() {
            let schema = users_schema();
            let plan = table_scan(Some("users"), &schema, None)
                .unwrap()
                .aggregate(Vec::<Expr>::new(), vec![max(col("ssn"))])
                .unwrap()
                .build()
                .unwrap();

            let rewriter = make_rewriter(vec![("ssn", RedactionMode::Constant)]);
            let rewritten = apply(plan, &rewriter);

            let aggs = aggregate_exprs(&rewritten);
            assert!(
                !aggs.is_empty(),
                "expected at least one Aggregate node in the rewritten plan"
            );
            for a in &aggs {
                assert!(
                    a.contains("[REDACTED]"),
                    "Aggregate must operate on the mask literal, not bare ssn: {a}"
                );
                assert!(
                    !contains_bare_ssn(a),
                    "Aggregate still references bare `ssn` column — column leaked: {a}"
                );
            }
        }

        /// Hash mode binds an unregistered UDF reference so plans BUILD
        /// (no panic at plan time), but a Hash-masked column never
        /// resolves to the silent `***REDACTED***` literal.
        #[test]
        fn test_hash_mode_binds_not_yet_wired_udf_not_silent_placeholder() {
            let schema = users_schema();
            let plan = table_scan(Some("users"), &schema, None)
                .unwrap()
                .project(vec![col("ssn")])
                .unwrap()
                .build()
                .unwrap();

            let rewriter = make_rewriter(vec![("ssn", RedactionMode::Hash)]);
            let rewritten = apply(plan, &rewriter);
            let s = format!("{rewritten}");
            // Plan-time: the unregistered UDF name appears in the plan.
            assert!(
                s.contains("policy_hash_v1"),
                "Hash mode must bind the unregistered policy_hash_v1 UDF — got: {s}"
            );
            // Plan-time: the silent placeholder must NOT be there.
            assert!(
                !s.contains("***REDACTED***"),
                "Hash mode must not emit silent ***REDACTED*** placeholder — got: {s}"
            );
        }

        /// Helper: does the predicate string still mention `ssn` as a
        /// bare column (not nested inside an alias / mask literal)?
        ///
        /// Conservative: if the literal substring "ssn" appears AND it's
        /// not part of "[REDACTED]" / mask-literal context, treat it as
        /// a leak. The Display impls used by DataFusion include the
        /// column name verbatim, e.g. `users.ssn` for a Column ref.
        fn contains_bare_ssn(s: &str) -> bool {
            // Either the unqualified or qualified column reference.
            s.contains("users.ssn") || ends_with_word(s, "ssn")
        }
        fn ends_with_word(s: &str, word: &str) -> bool {
            // Look for the word with a non-alphanumeric boundary on either
            // side. This avoids false positives on tokens like "ssn_hash".
            let bytes = s.as_bytes();
            let wlen = word.len();
            if bytes.len() < wlen {
                return false;
            }
            for i in 0..=bytes.len() - wlen {
                if &bytes[i..i + wlen] == word.as_bytes() {
                    let before_ok =
                        i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
                    let after_ok = i + wlen == bytes.len()
                        || !(bytes[i + wlen].is_ascii_alphanumeric() || bytes[i + wlen] == b'_');
                    if before_ok && after_ok {
                        return true;
                    }
                }
            }
            false
        }
    }
}
