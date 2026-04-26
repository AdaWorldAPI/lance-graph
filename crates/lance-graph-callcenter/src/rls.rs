//! DM-7 — Row-Level Security (RLS) rewriter for DataFusion LogicalPlans.
//!
//! Implements a DataFusion `OptimizerRule` that injects tenant and actor
//! isolation predicates into every `TableScan` in a logical plan.
//!
//! # Design Decision (UNKNOWN-3 resolved)
//!
//! RLS is implemented at the DataFusion LogicalPlan layer, NOT at the
//! pgwire layer. This means:
//!
//! - Predicates are injected as an optimizer rule, before physical planning.
//! - Every query path (SQL, DataFrame API, programmatic) gets the same
//!   tenant isolation — no bypass through direct API calls.
//! - The rewriter runs after the analyzer (so schema is resolved) but
//!   before other optimizers (so predicate pushdown can push RLS filters
//!   into scans).
//!
//! # Predicate Injection Rules
//!
//! | Role      | Predicates injected                                    |
//! |-----------|--------------------------------------------------------|
//! | `admin`   | `tenant_id = <ctx.tenant_id>` only                     |
//! | non-admin | `tenant_id = <ctx.tenant_id> AND actor_id = <ctx.actor_id>` |
//!
//! Admin actors see all rows within their tenant; non-admin actors see
//! only their own rows.
//!
//! # Column Name Convention
//!
//! The rewriter assumes `tenant_id` and `actor_id` columns exist in
//! every table that needs RLS. Tables without these columns will get
//! a `Filter` node that DataFusion's analyzer will reject at plan
//! validation — this is intentional (fail-closed).
//!
//! Plan: `.claude/plans/callcenter-membrane-v1.md` § DM-7

use datafusion::common::tree_node::Transformed;
use datafusion::common::Result as DfResult;
use datafusion::logical_expr::builder::LogicalPlanBuilder;
use datafusion::logical_expr::{col, lit, LogicalPlan};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};

use lance_graph_contract::auth::ActorContext;

/// Column name for the tenant identifier. Must match the Lance schema.
pub const COL_TENANT_ID: &str = "tenant_id";
/// Column name for the actor identifier. Must match the Lance schema.
pub const COL_ACTOR_ID: &str = "actor_id";

/// Row-Level Security rewriter. Injects `tenant_id` and `actor_id`
/// predicates into every `TableScan` in the `LogicalPlan`.
///
/// # Example
///
/// Given `ActorContext { actor_id: "user@example.com", tenant_id: 42, roles: ["viewer"] }`
/// and a query `SELECT * FROM customers`:
///
/// **Before:**
/// ```text
/// Projection: *
///   TableScan: customers
/// ```
///
/// **After:**
/// ```text
/// Projection: *
///   Filter: tenant_id = 42 AND actor_id = 'user@example.com'
///     TableScan: customers
/// ```
///
/// For an admin role, only the tenant filter is injected (no actor filter).
#[derive(Debug)]
pub struct RlsRewriter {
    actor: ActorContext,
}

impl RlsRewriter {
    /// Create a new RLS rewriter for the given actor context.
    pub fn new(actor: ActorContext) -> Self {
        Self { actor }
    }

    /// Build the predicate expression for this actor.
    ///
    /// - Always: `tenant_id = <tenant_id>`
    /// - Non-admin: `AND actor_id = '<actor_id>'`
    fn build_predicate(&self) -> datafusion::logical_expr::Expr {
        let tenant_pred = col(COL_TENANT_ID).eq(lit(self.actor.tenant_id));

        if self.actor.is_admin() {
            tenant_pred
        } else {
            tenant_pred.and(col(COL_ACTOR_ID).eq(lit(self.actor.actor_id.as_str())))
        }
    }
}

impl OptimizerRule for RlsRewriter {
    fn name(&self) -> &str {
        "rls_tenant_filter"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        // BottomUp: inject filters at the leaf (TableScan) level first,
        // so predicate pushdown and other optimizers see them.
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> DfResult<Transformed<LogicalPlan>> {
        match plan {
            LogicalPlan::TableScan(ref _scan) => {
                // Wrap the TableScan in a Filter node with tenant/actor predicates.
                let predicate = self.build_predicate();
                let filtered = LogicalPlanBuilder::new(plan)
                    .filter(predicate)?
                    .build()?;
                Ok(Transformed::yes(filtered))
            }
            _ => Ok(Transformed::no(plan)),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "query"))]
mod tests {
    use super::*;
    use datafusion::prelude::*;
    use std::sync::Arc;

    /// Helper: create a SessionContext with a simple in-memory table
    /// that has `tenant_id` and `actor_id` columns alongside a `name` column.
    async fn make_ctx() -> SessionContext {
        let ctx = SessionContext::new();
        // Register a simple CSV-like table via DataFrame API
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("name", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("tenant_id", arrow::datatypes::DataType::UInt64, false),
            arrow::datatypes::Field::new("actor_id", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("value", arrow::datatypes::DataType::Int32, true),
        ]));
        let batch = arrow::array::RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["alice", "bob", "carol"])),
                Arc::new(arrow::array::UInt64Array::from(vec![1, 1, 2])),
                Arc::new(arrow::array::StringArray::from(vec!["a@t.com", "b@t.com", "c@t.com"])),
                Arc::new(arrow::array::Int32Array::from(vec![10, 20, 30])),
            ],
        ).unwrap();
        let mem_table = datafusion::datasource::MemTable::try_new(
            schema,
            vec![vec![batch]],
        ).unwrap();
        ctx.register_table("customers", Arc::new(mem_table)).unwrap();
        ctx
    }

    /// Helper: produce a LogicalPlan for `SELECT * FROM customers`.
    async fn simple_scan_plan() -> LogicalPlan {
        let ctx = make_ctx().await;
        let df = ctx.sql("SELECT * FROM customers").await.unwrap();
        df.logical_plan().clone()
    }

    /// Helper: apply the RLS rewriter to a plan.
    fn apply_rls(plan: LogicalPlan, actor: ActorContext) -> DfResult<LogicalPlan> {
        use datafusion::optimizer::OptimizerContext;

        let rewriter = RlsRewriter::new(actor);
        let config = OptimizerContext::new();
        // Use transform_down to apply recursively (matching what the
        // optimizer framework would do with ApplyOrder::BottomUp, but
        // for test purposes transform_down also finds nested TableScans).
        use datafusion::common::tree_node::TreeNode;
        plan.transform_down(|node| rewriter.rewrite(node, &config))
            .map(|t| t.data)
    }

    /// Format a plan for assertion.
    fn plan_str(plan: &LogicalPlan) -> String {
        format!("{plan}")
    }

    #[tokio::test]
    async fn non_admin_gets_tenant_and_actor_filter() {
        let plan = simple_scan_plan().await;
        let actor = ActorContext::new(
            "user@example.com".into(),
            42,
            vec!["viewer".into()],
        );
        let rewritten = apply_rls(plan, actor).unwrap();
        let s = plan_str(&rewritten);
        assert!(s.contains("tenant_id"), "should contain tenant_id filter: {s}");
        assert!(s.contains("actor_id"), "should contain actor_id filter: {s}");
        assert!(s.contains("42"), "should contain tenant_id value 42: {s}");
        assert!(s.contains("user@example.com"), "should contain actor_id value: {s}");
    }

    #[tokio::test]
    async fn admin_gets_only_tenant_filter() {
        let plan = simple_scan_plan().await;
        let actor = ActorContext::new(
            "admin@example.com".into(),
            42,
            vec!["admin".into()],
        );
        let rewritten = apply_rls(plan, actor).unwrap();
        let s = plan_str(&rewritten);
        assert!(s.contains("tenant_id"), "should contain tenant_id filter: {s}");
        assert!(s.contains("42"), "should contain tenant_id value 42: {s}");
        // Admin should NOT get actor_id filter
        assert!(!s.contains("actor_id"), "admin should NOT have actor_id filter: {s}");
    }

    #[tokio::test]
    async fn subquery_gets_filters_at_every_table_scan() {
        let ctx = make_ctx().await;
        // Register a second table for a subquery
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("order_name", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("tenant_id", arrow::datatypes::DataType::UInt64, false),
            arrow::datatypes::Field::new("actor_id", arrow::datatypes::DataType::Utf8, false),
        ]));
        let batch = arrow::array::RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["order1"])),
                Arc::new(arrow::array::UInt64Array::from(vec![1u64])),
                Arc::new(arrow::array::StringArray::from(vec!["a@t.com"])),
            ],
        ).unwrap();
        let mem_table = datafusion::datasource::MemTable::try_new(
            schema,
            vec![vec![batch]],
        ).unwrap();
        ctx.register_table("orders", Arc::new(mem_table)).unwrap();

        // Query with a join (two TableScans)
        let df = ctx.sql(
            "SELECT c.name, o.order_name FROM customers c JOIN orders o ON c.name = o.order_name"
        ).await.unwrap();
        let plan = df.logical_plan().clone();

        let actor = ActorContext::new("u@t.com".into(), 7, vec!["viewer".into()]);
        let rewritten = apply_rls(plan, actor).unwrap();
        let s = plan_str(&rewritten);

        // Count occurrences of the tenant filter — should appear at least twice
        // (one per TableScan).
        let tenant_count = s.matches("tenant_id = UInt64(7)").count();
        assert!(
            tenant_count >= 2,
            "expected tenant_id filter on both tables, found {tenant_count} in: {s}"
        );
    }

    #[tokio::test]
    async fn empty_roles_is_non_admin() {
        let plan = simple_scan_plan().await;
        let actor = ActorContext::new("u@t.com".into(), 1, vec![]);
        let rewritten = apply_rls(plan, actor).unwrap();
        let s = plan_str(&rewritten);
        assert!(s.contains("actor_id"), "empty roles should produce actor_id filter: {s}");
    }

    #[test]
    fn rls_rewriter_name() {
        let r = RlsRewriter::new(ActorContext::new("x".into(), 0, vec![]));
        assert_eq!(r.name(), "rls_tenant_filter");
    }
}
