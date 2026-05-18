//! Physical execution plan stub for TiKV range scans — Glue #2 (plan §5).
//!
//! `TikvScanExec` is the `ExecutionPlan` produced by
//! [`crate::node::TikvNodeTableProvider`] and
//! [`crate::edge::TikvEdgeTableProvider`] during DataFusion's physical
//! planning phase.  It owns the TiKV snapshot handle, the key-range bounds
//! derived from pushed-down filters, and the projected schema to decode.
//!
//! Per plan §5 scan outline:
//! 1. Translate DataFusion predicates into a TiKV key range (done in the
//!    provider's `scan()` call before constructing `TikvScanExec`).
//! 2. Snapshot-read at `snapshot_ts` (or current HLC if `None`).
//! 3. Stream rows back as Arrow `RecordBatch` via `batch_scan` + decode.
//!
//! Sprint 1 TODO: implement `execute()` with the real TiKV snapshot +
//! batch scan; replace all `unimplemented!()` bodies.

use std::any::Any;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion::error::Result as DfResult;
use datafusion::execution::context::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion::physical_plan::execution_plan::{EmissionType, Boundedness};

// ---------------------------------------------------------------------------
// TikvScanExec
// ---------------------------------------------------------------------------

/// DataFusion `ExecutionPlan` that streams Arrow `RecordBatch` rows from a
/// TiKV range scan.
///
/// Per plan §5: constructed by `TikvNodeTableProvider::scan` (or the edge
/// mirror) and handed to the DataFusion physical planner.  The planner then
/// calls `execute(partition, ctx)` on each partition to get a
/// `RecordBatchStream`.
///
/// This struct is `pub(crate)` — DataFusion only ever holds it as
/// `Arc<dyn ExecutionPlan>`.  The constructors are called from `node.rs`
/// and `edge.rs` exclusively.
///
/// Sprint 1 TODO: add `snapshot: tikv_client::Snapshot` and
/// `key_range: (Vec<u8>, Vec<u8>)` fields once the real client is wired.
pub(crate) struct TikvScanExec {
    /// Projected Arrow schema that this exec will produce.
    ///
    /// Per plan §5: derived from `NodeShape::arrow_schema()` or
    /// `EdgeShape::arrow_schema()` with projection applied.
    schema: SchemaRef,

    /// Cached plan properties (partition count, equivalences, exec mode).
    ///
    /// DataFusion requires `ExecutionPlan::properties()` to return an
    /// `Arc<PlanProperties>` reference; we cache it at construction time.
    properties: Arc<PlanProperties>,
}

impl TikvScanExec {
    /// Construct a new scan exec stub.
    ///
    /// Per plan §5: called from `TikvNodeTableProvider::scan` (and the edge
    /// mirror) with the projected schema, key-range bounds, and optional
    /// snapshot timestamp.
    ///
    /// Sprint 1 TODO: accept `snapshot`, `key_range`, `shape`, and `limit`
    /// parameters; validate partition count against TiKV region topology.
    pub(crate) fn new(schema: SchemaRef) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            datafusion::physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self { schema, properties }
    }
}

// ---------------------------------------------------------------------------
// Debug — required by ExecutionPlan trait bound
// ---------------------------------------------------------------------------

impl std::fmt::Debug for TikvScanExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TikvScanExec {{ schema: {:?} }}", self.schema)
    }
}

// ---------------------------------------------------------------------------
// DisplayAs — required for EXPLAIN output
// ---------------------------------------------------------------------------

impl DisplayAs for TikvScanExec {
    fn fmt_as(
        &self,
        _t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "TikvScanExec(stub — Sprint 1)")
    }
}

// ---------------------------------------------------------------------------
// ExecutionPlan impl
// ---------------------------------------------------------------------------

impl ExecutionPlan for TikvScanExec {
    fn name(&self) -> &str {
        "TikvScanExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Return the schema this exec produces.
    ///
    /// Per plan §5: matches the projected `NodeShape` or `EdgeShape` schema.
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// Return cached plan properties (partition count, equivalences, mode).
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    /// No children — this is a leaf scan operator.
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    /// Leaf nodes return a clone of themselves when asked to re-parent.
    ///
    /// Sprint 1 TODO: propagate new children if the planner wraps this exec.
    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    /// Execute a partition and return a stream of `RecordBatch` rows.
    ///
    /// Per plan §5 steps 2-3: open a TiKV snapshot at `snapshot_ts`,
    /// batch-scan the key range, decode bytes into Arrow columns.
    ///
    /// Sprint 1 TODO: implement with `tikv_client::Snapshot::scan` +
    /// Arrow decoder.  For now returns `unimplemented!()`.
    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        unimplemented!(
            "LG-1 stub — implement TikvScanExec::execute in Sprint 1 \
             per integration-plan.md §5"
        )
    }
}
