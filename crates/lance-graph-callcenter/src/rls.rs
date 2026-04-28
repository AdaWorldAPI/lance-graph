//! DM-7 / LF-3 — Row-Level Security (RLS) rewriter for DataFusion LogicalPlans.
//!
//! Implements a DataFusion `OptimizerRule` that injects per-table RLS
//! predicates into every covered `TableScan` in a logical plan.
//!
//! # Design (UNKNOWN-3 resolved)
//!
//! RLS is implemented at the DataFusion `LogicalPlan` layer, NOT at the
//! pgwire layer. Consequences:
//!
//! - Predicates are injected as an optimizer rule, before physical
//!   planning. Every query path (SQL, DataFrame API, programmatic) gets
//!   the same tenant isolation — no bypass through direct API calls.
//! - The rewriter walks `TableScan` nodes top-down. For each scan whose
//!   table name is registered in the [`RlsPolicyRegistry`], the
//!   configured tenant (and optional actor) predicates are AND-ed onto
//!   the scan's existing `filters` so downstream predicate-pushdown sees
//!   them. Tables NOT in the registry are left untouched — fail-open
//!   for unprivileged-but-non-secret data, fail-closed must be enforced
//!   by registering a policy.
//!
//! # Predicate shape
//!
//! Given `RlsContext { tenant_id: "t1", actor_id: "u1" }` and an
//! `RlsPolicy { tenant_column: Some("tenant_id"), actor_column:
//! Some("actor_id"), .. }`:
//!
//! ```text
//! Before:
//!   TableScan: customers, filters=[]
//!
//! After:
//!   TableScan: customers, filters=[tenant_id = 't1' AND actor_id = 'u1']
//! ```
//!
//! Existing `filters` on the scan are preserved — the new predicate is
//! appended to the `filters` vector (DataFusion semantics: filters are
//! AND-ed during execution).
//!
//! Plan: `.claude/plans/callcenter-membrane-v1.md` § DM-7

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::common::tree_node::Transformed;
use datafusion::common::Result as DFResult;
use datafusion::logical_expr::{col, lit, Expr, LogicalPlan};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};

// ── Public API ────────────────────────────────────────────────────────────────

/// Tenant + actor identity injected by the membrane at session start.
///
/// Mirrors the JWT-derived identity envelope produced by
/// `crate::auth::JwtMiddleware` but uses `String` for both fields so
/// the rewriter is independent of how the upstream identity is shaped.
/// Conversion from `lance_graph_contract::auth::ActorContext` is
/// straightforward (see `From` impl below when the auth-jwt feature is
/// active).
#[derive(Debug, Clone)]
pub struct RlsContext {
    /// Tenant identifier (string form). Compared against the tenant
    /// column of every covered table.
    pub tenant_id: String,
    /// Canonical actor identifier (e.g. JWT `sub`). Compared against
    /// the actor column when the policy demands it.
    pub actor_id: String,
}

impl RlsContext {
    /// Construct from owned strings.
    pub fn new(tenant_id: impl Into<String>, actor_id: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            actor_id: actor_id.into(),
        }
    }
}

/// Per-table RLS policy: which columns must equal which `RlsContext`
/// fields. Both columns are optional so a policy can isolate by tenant
/// only (admin role) or by tenant+actor (non-admin role).
#[derive(Debug, Clone)]
pub struct RlsPolicy {
    /// Table name this policy applies to. Matched verbatim against
    /// `TableScan.table_name.table()`.
    pub table_name: String,
    /// Column whose value must equal `RlsContext.tenant_id`. `None`
    /// disables the tenant predicate (rare; almost always set).
    pub tenant_column: Option<String>,
    /// Column whose value must equal `RlsContext.actor_id`. `None`
    /// disables the actor predicate (used for admin/tenant-wide reads).
    pub actor_column: Option<String>,
}

impl RlsPolicy {
    /// Build a tenant-only policy (admin / tenant-wide read).
    pub fn tenant_only(table_name: impl Into<String>, tenant_column: impl Into<String>) -> Self {
        Self {
            table_name: table_name.into(),
            tenant_column: Some(tenant_column.into()),
            actor_column: None,
        }
    }

    /// Build a tenant+actor policy (non-admin read).
    pub fn tenant_and_actor(
        table_name: impl Into<String>,
        tenant_column: impl Into<String>,
        actor_column: impl Into<String>,
    ) -> Self {
        Self {
            table_name: table_name.into(),
            tenant_column: Some(tenant_column.into()),
            actor_column: Some(actor_column.into()),
        }
    }
}

/// Registry mapping table names to their RLS policy.
///
/// Tables not registered here are passed through unchanged by the
/// rewriter. Construct once at session/membrane start and share via
/// `Arc` across the optimizer rule.
#[derive(Debug, Default, Clone)]
pub struct RlsPolicyRegistry {
    policies: HashMap<String, RlsPolicy>,
}

impl RlsPolicyRegistry {
    /// New empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a policy keyed on its `table_name`. Replaces any
    /// previous policy for the same table.
    pub fn register(&mut self, policy: RlsPolicy) {
        self.policies.insert(policy.table_name.clone(), policy);
    }

    /// Look up a policy by table name.
    pub fn lookup(&self, table_name: &str) -> Option<&RlsPolicy> {
        self.policies.get(table_name)
    }

    /// Number of registered policies.
    pub fn len(&self) -> usize {
        self.policies.len()
    }

    /// `true` if no policies are registered.
    pub fn is_empty(&self) -> bool {
        self.policies.is_empty()
    }
}

/// DataFusion `OptimizerRule` that walks the plan, finds `TableScan`s
/// with a matching policy, and AND-injects
/// `tenant_col = ctx.tenant_id` (and `actor_col = ctx.actor_id` if the
/// policy demands).
///
/// `ApplyOrder::TopDown` is used so each `TableScan` is visited exactly
/// once (the optimizer framework does not recurse into the rewritten
/// node when we return `Transformed::yes`).
#[derive(Debug)]
pub struct RlsRewriter {
    /// Identity envelope to inject. Cloned per scan into literals.
    pub ctx: RlsContext,
    /// Per-table policy registry. Shared via `Arc` so the rule is
    /// cheap to clone if the optimizer demands `Send + Sync` ownership.
    pub registry: Arc<RlsPolicyRegistry>,
}

impl RlsRewriter {
    /// Construct an `RlsRewriter` with a context and a shared registry.
    pub fn new(ctx: RlsContext, registry: Arc<RlsPolicyRegistry>) -> Self {
        Self { ctx, registry }
    }

    /// Build the predicate `Expr` for a given policy.
    ///
    /// Returns `None` when neither `tenant_column` nor `actor_column`
    /// is set (degenerate policy — no predicates to inject).
    fn build_predicate(&self, policy: &RlsPolicy) -> Option<Expr> {
        let tenant_pred = policy
            .tenant_column
            .as_deref()
            .map(|c| col(c).eq(lit(self.ctx.tenant_id.as_str())));
        let actor_pred = policy
            .actor_column
            .as_deref()
            .map(|c| col(c).eq(lit(self.ctx.actor_id.as_str())));

        match (tenant_pred, actor_pred) {
            (Some(t), Some(a)) => Some(t.and(a)),
            (Some(t), None) => Some(t),
            (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }
}

impl OptimizerRule for RlsRewriter {
    fn name(&self) -> &str {
        "rls_rewriter"
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
        match plan {
            LogicalPlan::TableScan(mut scan) => {
                let table_name = scan.table_name.table().to_string();
                let Some(policy) = self.registry.lookup(&table_name) else {
                    // No policy for this table → leave it alone.
                    return Ok(Transformed::no(LogicalPlan::TableScan(scan)));
                };
                let Some(predicate) = self.build_predicate(policy) else {
                    // Degenerate policy with no columns → no-op.
                    return Ok(Transformed::no(LogicalPlan::TableScan(scan)));
                };
                // Append to scan.filters — DataFusion AND-s these
                // during execution and predicate-pushdown sees them as
                // filterable expressions.
                scan.filters.push(predicate);
                Ok(Transformed::yes(LogicalPlan::TableScan(scan)))
            }
            other => Ok(Transformed::no(other)),
        }
    }
}

// ── Bridge from contract::auth::ActorContext (auth-jwt feature only) ─────────

#[cfg(any(feature = "auth-jwt", feature = "auth", feature = "full"))]
impl From<&lance_graph_contract::auth::ActorContext> for RlsContext {
    fn from(actor: &lance_graph_contract::auth::ActorContext) -> Self {
        Self {
            tenant_id: actor.tenant_id.to_string(),
            actor_id: actor.actor_id.clone(),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
//
// We avoid `SessionContext::sql` because it requires the datafusion `sql`
// feature, which is not enabled under `query-lite` (default-features = false +
// unicode_expressions only). Plans are built directly via
// `LogicalPlanBuilder::scan` so the tests run on the smallest feature surface.

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::common::tree_node::TreeNode;
    use datafusion::common::DFSchema;
    use datafusion::datasource::{provider_as_source, MemTable};
    use datafusion::logical_expr::builder::LogicalPlanBuilder;
    use datafusion::logical_expr::JoinType;
    use datafusion::optimizer::OptimizerContext;

    /// RLS-typed schema: `name`, `tenant_id`, `actor_id`, `value`.
    fn rls_schema() -> Arc<arrow::datatypes::Schema> {
        Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("name", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("tenant_id", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("actor_id", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("value", arrow::datatypes::DataType::Int32, true),
        ]))
    }

    /// Public-table schema: just `k` and `v`, no tenant/actor columns.
    fn public_schema() -> Arc<arrow::datatypes::Schema> {
        Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("k", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("v", arrow::datatypes::DataType::Int32, false),
        ]))
    }

    /// Build a `MemTable` source for a given schema.
    fn mem_source(schema: Arc<arrow::datatypes::Schema>) -> Arc<dyn datafusion::datasource::TableProvider> {
        // Empty batches are fine — the rewriter only inspects logical plans.
        Arc::new(MemTable::try_new(schema, vec![vec![]]).unwrap())
    }

    /// Build a `TableScan` plan for the named table with the RLS schema.
    fn rls_scan(name: &str) -> LogicalPlan {
        let src = provider_as_source(mem_source(rls_schema()));
        LogicalPlanBuilder::scan(name, src, None).unwrap().build().unwrap()
    }

    /// Build a `TableScan` plan for the named public table.
    fn public_scan(name: &str) -> LogicalPlan {
        let src = provider_as_source(mem_source(public_schema()));
        LogicalPlanBuilder::scan(name, src, None).unwrap().build().unwrap()
    }

    /// Apply the rewriter top-down across the plan.
    fn apply(plan: LogicalPlan, rewriter: &RlsRewriter) -> LogicalPlan {
        let cfg = OptimizerContext::new();
        plan.transform_down(|n| rewriter.rewrite(n, &cfg))
            .unwrap()
            .data
    }

    fn ps(p: &LogicalPlan) -> String {
        format!("{p}")
    }

    #[test]
    fn single_table_select_star_gets_tenant_filter() {
        let plan = rls_scan("customers");

        let mut reg = RlsPolicyRegistry::new();
        reg.register(RlsPolicy::tenant_only("customers", "tenant_id"));
        let rewriter = RlsRewriter::new(RlsContext::new("t1", "u1"), Arc::new(reg));

        let rewritten = apply(plan, &rewriter);
        let s = ps(&rewritten);
        assert!(s.contains("tenant_id"), "missing tenant filter: {s}");
        assert!(s.contains("t1"), "missing tenant value: {s}");
        // Tenant-only policy: no actor predicate.
        assert!(
            !s.contains("actor_id = Utf8(\"u1\")"),
            "unexpected actor predicate: {s}"
        );
    }

    #[test]
    fn single_table_with_actor_policy_gets_both_filters() {
        let plan = rls_scan("customers");

        let mut reg = RlsPolicyRegistry::new();
        reg.register(RlsPolicy::tenant_and_actor(
            "customers",
            "tenant_id",
            "actor_id",
        ));
        let rewriter = RlsRewriter::new(RlsContext::new("t1", "u1"), Arc::new(reg));

        let rewritten = apply(plan, &rewriter);
        let s = ps(&rewritten);
        assert!(s.contains("tenant_id"), "missing tenant filter: {s}");
        assert!(s.contains("actor_id"), "missing actor filter: {s}");
        assert!(s.contains("t1"), "missing tenant value: {s}");
        assert!(s.contains("u1"), "missing actor value: {s}");
    }

    #[test]
    fn join_two_tables_gets_predicate_on_each() {
        // Build: customers ⋈ orders ON customers.name = orders.name
        let left = rls_scan("customers");
        let right = rls_scan("orders");
        let plan = LogicalPlanBuilder::from(left)
            .join_on(
                right,
                JoinType::Inner,
                vec![col("customers.name").eq(col("orders.name"))],
            )
            .unwrap()
            .build()
            .unwrap();

        let mut reg = RlsPolicyRegistry::new();
        reg.register(RlsPolicy::tenant_only("customers", "tenant_id"));
        reg.register(RlsPolicy::tenant_only("orders", "tenant_id"));
        let rewriter = RlsRewriter::new(RlsContext::new("t1", "u1"), Arc::new(reg));

        let rewritten = apply(plan, &rewriter);
        let s = ps(&rewritten);
        let count = s.matches("tenant_id = Utf8(\"t1\")").count();
        assert!(
            count >= 2,
            "expected tenant filter on both scans, found {count}: {s}"
        );
    }

    #[test]
    fn no_policy_table_is_unmodified() {
        let plan = public_scan("public_lookup");
        let original = ps(&plan);

        let mut reg = RlsPolicyRegistry::new();
        // Policy for a DIFFERENT table — `public_lookup` has none.
        reg.register(RlsPolicy::tenant_only("customers", "tenant_id"));
        let rewriter = RlsRewriter::new(RlsContext::new("t1", "u1"), Arc::new(reg));

        let rewritten = apply(plan, &rewriter);
        let s = ps(&rewritten);
        assert_eq!(s, original, "no-policy table should be untouched: {s}");
        assert!(
            !s.contains("tenant_id"),
            "unexpected tenant filter on public table: {s}"
        );
    }

    #[test]
    fn existing_filter_is_preserved() {
        // Pre-existing WHERE filter (`value > 0`) — must coexist with RLS.
        let plan = LogicalPlanBuilder::from(rls_scan("customers"))
            .filter(col("value").gt(lit(0i32)))
            .unwrap()
            .build()
            .unwrap();

        let mut reg = RlsPolicyRegistry::new();
        reg.register(RlsPolicy::tenant_and_actor(
            "customers",
            "tenant_id",
            "actor_id",
        ));
        let rewriter = RlsRewriter::new(RlsContext::new("t1", "u1"), Arc::new(reg));

        let rewritten = apply(plan, &rewriter);
        let s = ps(&rewritten);
        // User predicate retained.
        assert!(s.contains("value"), "lost user predicate: {s}");
        // RLS predicates injected.
        assert!(s.contains("tenant_id"), "missing tenant filter: {s}");
        assert!(s.contains("actor_id"), "missing actor filter: {s}");
    }

    #[test]
    fn registry_lookup_and_register() {
        let mut reg = RlsPolicyRegistry::new();
        assert!(reg.is_empty());
        reg.register(RlsPolicy::tenant_only("customers", "tenant_id"));
        assert_eq!(reg.len(), 1);
        assert!(reg.lookup("customers").is_some());
        assert!(reg.lookup("missing").is_none());

        // Re-register replaces the prior entry.
        reg.register(RlsPolicy::tenant_and_actor(
            "customers",
            "tenant_id",
            "actor_id",
        ));
        assert_eq!(reg.len(), 1);
        let p = reg.lookup("customers").unwrap();
        assert_eq!(p.actor_column.as_deref(), Some("actor_id"));
    }

    #[test]
    fn rewriter_name_and_apply_order() {
        let reg = Arc::new(RlsPolicyRegistry::new());
        let r = RlsRewriter::new(RlsContext::new("t", "u"), reg);
        assert_eq!(r.name(), "rls_rewriter");
        assert!(matches!(r.apply_order(), Some(ApplyOrder::TopDown)));
    }

    /// Sanity: an empty registry leaves every scan untouched.
    #[test]
    fn empty_registry_is_a_no_op() {
        let plan = rls_scan("customers");
        let original = ps(&plan);
        let rewriter = RlsRewriter::new(
            RlsContext::new("t1", "u1"),
            Arc::new(RlsPolicyRegistry::new()),
        );
        let rewritten = apply(plan, &rewriter);
        assert_eq!(ps(&rewritten), original);
    }

    /// Make sure we still produce a valid `DFSchema`-bearing plan after
    /// rewrite (i.e. the TableScan invariants hold).
    #[test]
    fn rewritten_plan_schema_is_unchanged() {
        let plan = rls_scan("customers");
        let before: DFSchema = plan.schema().as_ref().clone();

        let mut reg = RlsPolicyRegistry::new();
        reg.register(RlsPolicy::tenant_and_actor(
            "customers",
            "tenant_id",
            "actor_id",
        ));
        let rewriter = RlsRewriter::new(RlsContext::new("t1", "u1"), Arc::new(reg));
        let rewritten = apply(plan, &rewriter);
        let after: DFSchema = rewritten.schema().as_ref().clone();
        assert_eq!(before, after, "rewriter must not change scan schema");
    }

    #[cfg(feature = "auth-jwt")]
    #[test]
    fn from_actor_context() {
        use lance_graph_contract::auth::ActorContext;
        let actor = ActorContext::new("user@example.com".into(), 42, vec!["viewer".into()]);
        let ctx: RlsContext = (&actor).into();
        assert_eq!(ctx.tenant_id, "42");
        assert_eq!(ctx.actor_id, "user@example.com");
    }
}
