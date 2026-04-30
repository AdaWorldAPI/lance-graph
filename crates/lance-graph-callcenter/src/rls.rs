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
//!   them.
//! - Tables NOT in the registry are handled per the registry's
//!   [`RegistryMode`]. The default ([`RegistryMode::Sealed`]) rejects
//!   the plan with a `DataFusionError::Plan` — this is the
//!   deny-by-default contract from `foundry-roadmap.md` § 42.
//!   Legacy / public data must opt in via
//!   [`RlsPolicyRegistry::fail_open`] with an audit reason.
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
use datafusion::common::{DataFusionError, Result as DFResult};
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

/// Errors produced by the strict [`RlsContext::new`] constructor.
///
/// Hand-rolled (no `thiserror` dependency) so the crate keeps its
/// minimal feature surface under `auth-rls-lite`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RlsError {
    /// `tenant_id` was empty — RLS predicate would compare against
    /// `''` and silently match nothing, which is worse than failing.
    EmptyTenantId,
    /// `actor_id` was empty — same hazard. Use [`RlsContext::new_unchecked`]
    /// for legitimate system-actor cases (and audit-log the call site).
    EmptyActorId,
}

impl std::fmt::Display for RlsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTenantId => f.write_str("tenant_id must not be empty"),
            Self::EmptyActorId => f.write_str("actor_id must not be empty"),
        }
    }
}

impl std::error::Error for RlsError {}

impl RlsContext {
    /// Construct from owned strings, validating that neither id is
    /// empty. Empty ids would otherwise produce predicates of the
    /// form `tenant_id = ''` which silently match nothing — a
    /// failure mode that hides bugs instead of surfacing them.
    pub fn new(
        tenant_id: impl Into<String>,
        actor_id: impl Into<String>,
    ) -> Result<Self, RlsError> {
        let tenant_id = tenant_id.into();
        let actor_id = actor_id.into();
        if tenant_id.is_empty() {
            return Err(RlsError::EmptyTenantId);
        }
        if actor_id.is_empty() {
            return Err(RlsError::EmptyActorId);
        }
        Ok(Self {
            tenant_id,
            actor_id,
        })
    }

    /// Legacy-permissive constructor — allows empty ids (e.g. for
    /// "system" contexts that operate without an actor). MUST be
    /// paired with an audit-log entry showing why the empty id is
    /// safe in this call site.
    pub fn new_unchecked(tenant_id: impl Into<String>, actor_id: impl Into<String>) -> Self {
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

/// Operating mode of an [`RlsPolicyRegistry`].
///
/// The default is `Sealed`: any `TableScan` of an unregistered table
/// is rejected with a `DataFusionError::Plan`. This matches the
/// deny-by-default contract in `foundry-roadmap.md` § 42 — RLS is
/// useless if the easy path is "forget to register a policy and read
/// every tenant's data."
///
/// `FailOpen` is an explicit opt-in for legacy / non-tenanted data
/// (public lookup tables, system catalogs). The `reason` field exists
/// so the audit trail is grep-able: every `FailOpen { reason: "..." }`
/// site documents *why* fail-open is the right choice there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RegistryMode {
    /// Default: any `TableScan` of an unregistered table → `DataFusionError`.
    /// Matches the deny-by-default contract in `foundry-roadmap.md` §42.
    #[default]
    Sealed,
    /// Explicit opt-in for legacy / non-tenanted data. Requires an
    /// audit trail (the `reason` is logged / grep-able).
    FailOpen {
        /// Human-readable justification for opening this registry.
        /// E.g. `"legacy public lookup"`. Static lifetime so the
        /// reason cannot be silently overwritten at runtime.
        reason: &'static str,
    },
}

/// Registry mapping table names to their RLS policy.
///
/// In `Sealed` mode (the default), tables not registered here cause
/// the rewriter to fail-closed: the plan is rejected. In `FailOpen`
/// mode, unregistered tables are passed through unchanged. Construct
/// once at session/membrane start and share via `Arc` across the
/// optimizer rule.
#[derive(Debug, Default, Clone)]
pub struct RlsPolicyRegistry {
    policies: HashMap<String, RlsPolicy>,
    mode: RegistryMode,
}

impl RlsPolicyRegistry {
    /// New empty registry in [`RegistryMode::Sealed`] mode (default).
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a registry that fails closed on unregistered tables.
    /// Equivalent to `RlsPolicyRegistry::default()` but spells the
    /// intent at the call site.
    pub fn sealed() -> Self {
        Self {
            mode: RegistryMode::Sealed,
            ..Default::default()
        }
    }

    /// Build a registry that passes unregistered tables through. The
    /// `reason` is recorded in the [`RegistryMode::FailOpen`] variant
    /// for audit grep — pick something specific
    /// (`"legacy public lookup"`, `"bootstrap migration window"`).
    pub fn fail_open(reason: &'static str) -> Self {
        Self {
            mode: RegistryMode::FailOpen { reason },
            ..Default::default()
        }
    }

    /// Current mode.
    pub fn mode(&self) -> RegistryMode {
        self.mode
    }

    /// Builder-style mode override.
    pub fn with_mode(mut self, mode: RegistryMode) -> Self {
        self.mode = mode;
        self
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
    /// Private — read via [`RlsRewriter::ctx`] to keep callers from
    /// mutating live RLS state behind the rewriter's back.
    ctx: RlsContext,
    /// Per-table policy registry. Shared via `Arc` so the rule is
    /// cheap to clone if the optimizer demands `Send + Sync` ownership.
    /// Private — read via [`RlsRewriter::registry`].
    registry: Arc<RlsPolicyRegistry>,
}

impl RlsRewriter {
    /// Construct an `RlsRewriter` with a context and a shared registry.
    pub fn new(ctx: RlsContext, registry: Arc<RlsPolicyRegistry>) -> Self {
        Self { ctx, registry }
    }

    /// Borrow the [`RlsContext`] this rewriter injects.
    pub fn ctx(&self) -> &RlsContext {
        &self.ctx
    }

    /// Borrow the shared [`RlsPolicyRegistry`].
    pub fn registry(&self) -> &Arc<RlsPolicyRegistry> {
        &self.registry
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
                    // No policy for this table — branch on the
                    // registry mode. Sealed (default) refuses to
                    // execute the plan; FailOpen passes through.
                    return match self.registry.mode() {
                        RegistryMode::Sealed => Err(DataFusionError::Plan(format!(
                            "RLS sealed registry: no policy for table '{}'. \
                             Register a policy or use \
                             RlsPolicyRegistry::fail_open(...) explicitly.",
                            table_name
                        ))),
                        RegistryMode::FailOpen { reason: _ } => {
                            // Pass-through preserved exactly for
                            // FailOpen mode. (Logging hook would land
                            // here once a tracing facility is wired.)
                            Ok(Transformed::no(LogicalPlan::TableScan(scan)))
                        }
                    };
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
    /// Lossy on purpose — `ActorContext` may carry fields RLS does
    /// not consume. Uses [`RlsContext::new_unchecked`] because the
    /// ActorContext invariants already enforce non-empty fields
    /// upstream (see `lance_graph_contract::auth`).
    fn from(actor: &lance_graph_contract::auth::ActorContext) -> Self {
        Self::new_unchecked(actor.tenant_id.to_string(), actor.actor_id.clone())
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
    fn mem_source(
        schema: Arc<arrow::datatypes::Schema>,
    ) -> Arc<dyn datafusion::datasource::TableProvider> {
        // Empty batches are fine — the rewriter only inspects logical plans.
        Arc::new(MemTable::try_new(schema, vec![vec![]]).unwrap())
    }

    /// Build a `TableScan` plan for the named table with the RLS schema.
    fn rls_scan(name: &str) -> LogicalPlan {
        let src = provider_as_source(mem_source(rls_schema()));
        LogicalPlanBuilder::scan(name, src, None)
            .unwrap()
            .build()
            .unwrap()
    }

    /// Build a `TableScan` plan for the named public table.
    fn public_scan(name: &str) -> LogicalPlan {
        let src = provider_as_source(mem_source(public_schema()));
        LogicalPlanBuilder::scan(name, src, None)
            .unwrap()
            .build()
            .unwrap()
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
        let rewriter = RlsRewriter::new(
            RlsContext::new("t1", "u1").expect("non-empty ids"),
            Arc::new(reg),
        );

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
        let rewriter = RlsRewriter::new(
            RlsContext::new("t1", "u1").expect("non-empty ids"),
            Arc::new(reg),
        );

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
        let rewriter = RlsRewriter::new(
            RlsContext::new("t1", "u1").expect("non-empty ids"),
            Arc::new(reg),
        );

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

        // Opt into FailOpen: this test exercises the legacy
        // pass-through path on an unregistered public table.
        let mut reg = RlsPolicyRegistry::fail_open("legacy public lookup");
        // Policy for a DIFFERENT table — `public_lookup` has none.
        reg.register(RlsPolicy::tenant_only("customers", "tenant_id"));
        let rewriter = RlsRewriter::new(
            RlsContext::new("t1", "u1").expect("non-empty ids"),
            Arc::new(reg),
        );

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
        let rewriter = RlsRewriter::new(
            RlsContext::new("t1", "u1").expect("non-empty ids"),
            Arc::new(reg),
        );

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
        let r = RlsRewriter::new(RlsContext::new("t", "u").expect("non-empty ids"), reg);
        assert_eq!(r.name(), "rls_rewriter");
        assert!(matches!(r.apply_order(), Some(ApplyOrder::TopDown)));
    }

    /// Sanity: an empty FailOpen registry leaves every scan untouched.
    /// (An empty Sealed registry would error; see
    /// `sealed_registry_errors_on_unregistered_table`.)
    #[test]
    fn empty_registry_is_a_no_op() {
        let plan = rls_scan("customers");
        let original = ps(&plan);
        let rewriter = RlsRewriter::new(
            RlsContext::new("t1", "u1").expect("non-empty ids"),
            Arc::new(RlsPolicyRegistry::fail_open("test fixture")),
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
        let rewriter = RlsRewriter::new(
            RlsContext::new("t1", "u1").expect("non-empty ids"),
            Arc::new(reg),
        );
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

    // ── Round-2 hardening tests ──────────────────────────────────────────

    /// Sealed registry (the default) must reject any TableScan whose
    /// table is not registered. This is the deny-by-default contract.
    #[test]
    fn sealed_registry_errors_on_unregistered_table() {
        let plan = rls_scan("customers");
        // Default (Sealed) registry, no policies — `customers` is
        // unregistered.
        let rewriter = RlsRewriter::new(
            RlsContext::new("t1", "u1").expect("non-empty ids"),
            Arc::new(RlsPolicyRegistry::new()),
        );
        let cfg = OptimizerContext::new();
        let result = plan.transform_down(|n| rewriter.rewrite(n, &cfg));
        assert!(
            matches!(result, Err(DataFusionError::Plan(_))),
            "expected DataFusionError::Plan from sealed registry on unregistered scan, got {result:?}"
        );
    }

    /// FailOpen mode is the explicit opt-in for legacy / public
    /// data — unregistered scans pass through unchanged.
    #[test]
    fn fail_open_registry_passes_through_unregistered() {
        let plan = public_scan("public_lookup");
        let original = ps(&plan);

        let reg = RlsPolicyRegistry::fail_open("legacy public lookup");
        assert!(matches!(
            reg.mode(),
            RegistryMode::FailOpen {
                reason: "legacy public lookup"
            }
        ));
        let rewriter = RlsRewriter::new(
            RlsContext::new("t1", "u1").expect("non-empty ids"),
            Arc::new(reg),
        );

        let rewritten = apply(plan, &rewriter);
        assert_eq!(ps(&rewritten), original);
    }

    /// `RlsContext::new` rejects empty tenant_id and empty actor_id.
    /// Both branches must surface as distinct `RlsError` variants.
    #[test]
    fn rls_context_new_rejects_empty() {
        assert_eq!(
            RlsContext::new("", "u1").err(),
            Some(RlsError::EmptyTenantId)
        );
        assert_eq!(
            RlsContext::new("t1", "").err(),
            Some(RlsError::EmptyActorId)
        );
        // Both empty: tenant is checked first.
        assert_eq!(RlsContext::new("", "").err(), Some(RlsError::EmptyTenantId));
        // Non-empty pair succeeds.
        let ok = RlsContext::new("t1", "u1").expect("valid");
        assert_eq!(ok.tenant_id, "t1");
        assert_eq!(ok.actor_id, "u1");
        // new_unchecked deliberately bypasses validation.
        let unchecked = RlsContext::new_unchecked("", "");
        assert!(unchecked.tenant_id.is_empty());
        assert!(unchecked.actor_id.is_empty());
    }

    /// A degenerate policy with neither tenant_column nor actor_column
    /// produces no predicate — the scan is left untouched.
    #[test]
    fn degenerate_policy_both_columns_none() {
        let plan = rls_scan("customers");
        let original = ps(&plan);

        let mut reg = RlsPolicyRegistry::new();
        reg.register(RlsPolicy {
            table_name: "customers".to_string(),
            tenant_column: None,
            actor_column: None,
        });
        let rewriter = RlsRewriter::new(
            RlsContext::new("t1", "u1").expect("non-empty ids"),
            Arc::new(reg),
        );
        let rewritten = apply(plan, &rewriter);
        let s = ps(&rewritten);
        assert_eq!(s, original, "degenerate policy should be a no-op");
        assert!(
            !s.contains("tenant_id ="),
            "no tenant predicate expected: {s}"
        );
        assert!(
            !s.contains("actor_id ="),
            "no actor predicate expected: {s}"
        );
    }

    /// Registry mode default + builder helpers.
    #[test]
    fn registry_mode_defaults_to_sealed() {
        assert_eq!(RegistryMode::default(), RegistryMode::Sealed);
        let r = RlsPolicyRegistry::new();
        assert_eq!(r.mode(), RegistryMode::Sealed);
        let r2 = RlsPolicyRegistry::sealed();
        assert_eq!(r2.mode(), RegistryMode::Sealed);
        let r3 = RlsPolicyRegistry::fail_open("audit reason");
        assert!(matches!(
            r3.mode(),
            RegistryMode::FailOpen {
                reason: "audit reason"
            }
        ));
        let r4 = RlsPolicyRegistry::new().with_mode(RegistryMode::FailOpen {
            reason: "via with_mode",
        });
        assert!(matches!(
            r4.mode(),
            RegistryMode::FailOpen {
                reason: "via with_mode"
            }
        ));
    }

    /// The privatised fields are still readable through accessors.
    #[test]
    fn rewriter_accessors_expose_ctx_and_registry() {
        let reg = Arc::new(RlsPolicyRegistry::sealed());
        let ctx = RlsContext::new("tenant-x", "actor-x").expect("valid");
        let r = RlsRewriter::new(ctx, Arc::clone(&reg));
        assert_eq!(r.ctx().tenant_id, "tenant-x");
        assert_eq!(r.ctx().actor_id, "actor-x");
        assert_eq!(r.registry().mode(), RegistryMode::Sealed);
        // Same Arc instance.
        assert!(Arc::ptr_eq(r.registry(), &reg));
    }
}
