//! Node table provider — Glue #2 (plan §5 `TikvNodeTableProvider`).
//!
//! Implements DataFusion's `TableProvider` for graph **node** shapes stored
//! in TiKV. Each node type maps to a contiguous key range:
//! `<node_type>/<node_id>` → encoded property columns.  Filter pushdown
//! translates DataFusion predicates into TiKV key-range bounds; the
//! result is streamed as Arrow `RecordBatch` by [`crate::scan::TikvScanExec`].
//!
//! See `integration-plan.md §5` for the full API sketch and Sprint 1 scope.

use std::any::Any;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::Result as DfResult;
use datafusion::datasource::TableProvider;
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_plan::ExecutionPlan;

use lance_graph_contract::provider::{BackendId, MvccProvider, TikvBackedProvider};

use crate::error::Error;
use crate::scan::TikvScanExec;

// ---------------------------------------------------------------------------
// Shape placeholder types
//
// `NodeShape` and `EdgeShape` are declared in `lance-graph-catalog` as part
// of Sprint 3 codegen (plan §7 Sprint 3).  Until that crate exposes them we
// define local newtypes so the provider API compiles as a stub.  Replace with
// the catalog import once `lance_graph_catalog::{NodeShape, EdgeShape}` land.
// ---------------------------------------------------------------------------

/// Placeholder for `lance_graph_catalog::NodeShape` (available in Sprint 3).
///
/// See plan §5 API sketch and §7 Sprint 3 for the real type.
#[derive(Clone, Debug)]
pub struct NodeShape {
    /// Arrow schema derived from the node type's property definitions.
    pub schema: SchemaRef,
    /// Node type label (e.g. `"Person"`).
    pub label: String,
}

impl NodeShape {
    /// Return the Arrow schema for this node shape.
    ///
    /// Per plan §5: "Shape from lance-graph-catalog. Drives schema + range encoding."
    pub fn arrow_schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

// ---------------------------------------------------------------------------
// TikvNodeTableProvider
// ---------------------------------------------------------------------------

/// DataFusion `TableProvider` backed by TiKV node ranges.
///
/// Per plan §5: holds a TiKV `TransactionClient`, the `NodeShape` that
/// drives schema and range encoding, and an optional MVCC snapshot timestamp.
/// `None` snapshot means "read latest at scan time".
///
/// Implements:
/// - `datafusion::datasource::TableProvider` — DataFusion integration.
/// - `lance_graph_contract::provider::MvccProvider` — snapshot-consistent
///   read semantics shared with Lance and surrealdb providers.
/// - `lance_graph_contract::provider::TikvBackedProvider` — TiKV marker so
///   the federated planner can detect and compose TiKV providers.
///
/// Sprint 1 TODO: wire `scan()` to `TikvScanExec::new(...)`.
pub struct TikvNodeTableProvider {
    /// TiKV transactional client (plan §5: `Arc<tikv_client::TransactionClient>`).
    ///
    /// Sprint 1 TODO: replace `Arc<()>` with `Arc<tikv_client::TransactionClient>`.
    pub client: Arc<()>,

    /// Node shape from lance-graph-catalog. Drives schema + range key encoding.
    pub shape: NodeShape,

    /// Optional MVCC snapshot. `None` = read latest (plan §5).
    pub snapshot_ts: Option<u64>,
}

impl TikvNodeTableProvider {
    /// Construct a new node provider bound to a TiKV client and shape.
    ///
    /// Per plan §5 `TikvNodeTableProvider::new`. Snapshot defaults to `None`
    /// (read latest). Use [`Self::with_snapshot`] to pin to a specific HLC ts.
    ///
    /// Sprint 1 TODO: accept `Arc<tikv_client::TransactionClient>`.
    pub async fn new(
        client: Arc<()>,
        shape: NodeShape,
    ) -> Result<Self, Error> {
        // Sprint 1 TODO: validate client connectivity against the cluster.
        Ok(Self { client, shape, snapshot_ts: None })
    }

    /// Pin this provider to a specific MVCC snapshot timestamp.
    ///
    /// Per plan §5 `with_snapshot(ts: u64) -> Self`. The `ts` is the same
    /// HLC number used by surrealdb-core's `version` column and by Lance
    /// dataset versions — **one clock, all storage targets**.
    pub fn with_snapshot(mut self, ts: u64) -> Self {
        self.snapshot_ts = Some(ts);
        self
    }
}

// ---------------------------------------------------------------------------
// DataFusion TableProvider impl
// ---------------------------------------------------------------------------

#[async_trait]
impl TableProvider for TikvNodeTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Return the Arrow schema for this node shape.
    ///
    /// Per plan §5: `self.shape.arrow_schema()`.
    fn schema(&self) -> SchemaRef {
        self.shape.arrow_schema()
    }

    /// Report as a base table (not a view or temporary).
    fn table_type(&self) -> TableType {
        TableType::Base
    }

    /// Produce a physical `ExecutionPlan` that streams node rows from TiKV.
    ///
    /// Per plan §5 scan outline:
    /// 1. Translate `filters` into a TiKV key range via the shape.
    /// 2. Snapshot-read at `snapshot_ts` (or current HLC if `None`).
    /// 3. Stream rows as Arrow `RecordBatch` via `TikvScanExec`.
    ///
    /// Sprint 1 TODO: implement steps 1-3; wire to `TikvScanExec::new(...)`.
    async fn scan(
        &self,
        _state: &dyn Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        unimplemented!(
            "LG-1 stub — implement TikvNodeTableProvider::scan in Sprint 1 \
             per integration-plan.md §5"
        )
    }
}

// ---------------------------------------------------------------------------
// MvccProvider + TikvBackedProvider marker impls
// ---------------------------------------------------------------------------

/// Marker impl: `MvccProvider` for snapshot-consistent cross-engine reads.
///
/// Per plan §5 "Snapshot integration": the `snapshot_ts` here is the same
/// `u64` that surrealdb-core, TiKV, and Lance all agree on.
impl MvccProvider for TikvNodeTableProvider {
    fn backend(&self) -> BackendId {
        BackendId::Tikv
    }

    fn snapshot_ts(&self) -> Option<u64> {
        self.snapshot_ts
    }
}

/// Marker impl: `TikvBackedProvider` so the planner detects TiKV providers
/// without downcasting through DataFusion's `dyn TableProvider` objects.
///
/// Per plan §5 "Marker impl from lance-graph-contract::provider."
impl TikvBackedProvider for TikvNodeTableProvider {}
