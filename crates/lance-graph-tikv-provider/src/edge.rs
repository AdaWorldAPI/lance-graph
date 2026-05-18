//! Edge table provider — Glue #2 (plan §5 "Edge tables").
//!
//! Mirror of [`crate::node`] for graph **edge** shapes stored in TiKV.
//! Edges are stored as `(src_node_id, edge_type, dst_node_id) → edge_props`
//! so outbound expansion (MATCH `(a)-[:KNOWS]->(b)`) is a **single range scan**
//! starting at `src_node_id` — no secondary index needed.
//!
//! See `integration-plan.md §5` "Edge tables" and Sprint 1 scope in §7.

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
// EdgeShape placeholder
//
// See node.rs for the rationale.  Replace with
// `lance_graph_catalog::EdgeShape` once Sprint 3 lands.
// ---------------------------------------------------------------------------

/// Placeholder for `lance_graph_catalog::EdgeShape` (available in Sprint 3).
///
/// Edges are encoded as `(src_id, edge_type, dst_id) → props` in TiKV.
/// The `src_type` and `dst_type` labels drive schema validation; the
/// `label` identifies the edge relationship (e.g. `"KNOWS"`).
#[derive(Clone, Debug)]
pub struct EdgeShape {
    /// Arrow schema for this edge type's properties.
    pub schema: SchemaRef,
    /// Edge relationship label (e.g. `"KNOWS"`).
    pub label: String,
    /// Source node type label (e.g. `"Person"`).
    pub src_type: String,
    /// Destination node type label (e.g. `"Person"`).
    pub dst_type: String,
}

impl EdgeShape {
    /// Return the Arrow schema for this edge shape.
    ///
    /// Per plan §5 "Edge tables": schema drives column layout during decode.
    pub fn arrow_schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

// ---------------------------------------------------------------------------
// TikvEdgeTableProvider
// ---------------------------------------------------------------------------

/// DataFusion `TableProvider` backed by TiKV edge ranges.
///
/// Mirror of [`crate::node::TikvNodeTableProvider`] for edge shapes.
///
/// Per plan §5 "Edge tables": edges are keyed by `(src_node_id, edge_type,
/// dst_node_id)`, so outbound neighbour expansion is a single range scan —
/// the same efficiency JanusGraph achieved with Cassandra wide rows, but with
/// TiKV's Percolator ACID and native MVCC.
///
/// Implements:
/// - `datafusion::datasource::TableProvider` — DataFusion integration.
/// - `lance_graph_contract::provider::MvccProvider` — snapshot-consistent reads.
/// - `lance_graph_contract::provider::TikvBackedProvider` — TiKV marker.
///
/// Sprint 1 TODO: wire `scan()` to `TikvScanExec::new(...)` with edge-range encoding.
pub struct TikvEdgeTableProvider {
    /// TiKV transactional client.
    ///
    /// Sprint 1 TODO: replace `Arc<()>` with `Arc<tikv_client::TransactionClient>`.
    pub client: Arc<()>,

    /// Edge shape from lance-graph-catalog. Drives schema + range key encoding.
    pub shape: EdgeShape,

    /// Optional MVCC snapshot. `None` = read latest (plan §5).
    pub snapshot_ts: Option<u64>,
}

impl TikvEdgeTableProvider {
    /// Construct a new edge provider bound to a TiKV client and shape.
    ///
    /// Per plan §5 "Edge tables". Snapshot defaults to `None` (read latest).
    /// Use [`Self::with_snapshot`] to pin to a specific HLC timestamp.
    ///
    /// Sprint 1 TODO: accept `Arc<tikv_client::TransactionClient>`.
    pub async fn new(
        client: Arc<()>,
        shape: EdgeShape,
    ) -> Result<Self, Error> {
        // Sprint 1 TODO: validate client connectivity.
        Ok(Self { client, shape, snapshot_ts: None })
    }

    /// Pin this provider to a specific MVCC snapshot timestamp.
    ///
    /// Same semantics as `TikvNodeTableProvider::with_snapshot` — the `ts`
    /// is the shared HLC number across all storage targets (plan §5).
    pub fn with_snapshot(mut self, ts: u64) -> Self {
        self.snapshot_ts = Some(ts);
        self
    }
}

// ---------------------------------------------------------------------------
// Debug — required by TableProvider trait bound
// ---------------------------------------------------------------------------

impl std::fmt::Debug for TikvEdgeTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TikvEdgeTableProvider {{ shape: {:?} }}", self.shape)
    }
}

// ---------------------------------------------------------------------------
// DataFusion TableProvider impl
// ---------------------------------------------------------------------------

#[async_trait]
impl TableProvider for TikvEdgeTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Return the Arrow schema for this edge shape.
    ///
    /// Per plan §5 "Edge tables": schema includes `src_id`, `dst_id`, and
    /// all edge property columns declared in the catalog.
    fn schema(&self) -> SchemaRef {
        self.shape.arrow_schema()
    }

    /// Report as a base table.
    fn table_type(&self) -> TableType {
        TableType::Base
    }

    /// Produce a physical `ExecutionPlan` that streams edge rows from TiKV.
    ///
    /// Per plan §5 "Edge tables": range scan starts at `src_node_id` so
    /// outbound expansion is a single TiKV range scan, not a scatter-gather.
    ///
    /// Sprint 1 TODO: implement edge key encoding and wire to `TikvScanExec`.
    async fn scan(
        &self,
        _state: &dyn Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        unimplemented!(
            "LG-1 stub — implement TikvEdgeTableProvider::scan in Sprint 1 \
             per integration-plan.md §5 'Edge tables'"
        )
    }
}

// ---------------------------------------------------------------------------
// MvccProvider + TikvBackedProvider marker impls
// ---------------------------------------------------------------------------

/// Marker impl: `MvccProvider` for snapshot-consistent cross-engine reads.
///
/// Per plan §5: `snapshot_ts` is the shared HLC number across TiKV,
/// surrealdb-core, and Lance dataset versions.
impl MvccProvider for TikvEdgeTableProvider {
    fn backend(&self) -> BackendId {
        BackendId::Tikv
    }

    fn snapshot_ts(&self) -> Option<u64> {
        self.snapshot_ts
    }
}

/// Marker impl: `TikvBackedProvider` — TiKV marker for the federated planner.
impl TikvBackedProvider for TikvEdgeTableProvider {}
