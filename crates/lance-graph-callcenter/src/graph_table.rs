//! `graph_table` — DataFusion node/edge adapter over a [`GraphSnapshot`].
//!
//! The DataFusion path for the OSINT/Gotham + FMA graph (contrast: the pure-Rust
//! Gremlin/SurrealQL traversal in [`crate::graph_gremlin`]). Projects the
//! `GraphSnapshot` that [`lance_graph_contract::soa_graph`] builds from the
//! 32-byte node head into two MemTable-backed [`TableProvider`]s — `nodes` and
//! `edges` — so a consumer (q2 cockpit) can run Cypher→SQL / GQL / SQL directly:
//!
//! ```sql
//! SELECT n.kind, count(*) FROM nodes n GROUP BY n.kind;
//! SELECT e.label, count(*) FROM edges e GROUP BY e.label;
//! SELECT * FROM edges WHERE source = '...' ;   -- one-hop, the SQL "out()"
//! ```
//!
//! Round-1 is MemTable-backed (mirrors `transcode::ontology_table`); a
//! Lance-dataset-backed scan is the next round. Feature-gated on `query-lite`.

use std::sync::Arc;

use arrow::array::{Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::datasource::{MemTable, TableProvider};
use datafusion::error::Result as DfResult;

use lance_graph_contract::graph_render::GraphSnapshot;

/// Arrow schema of the `nodes` table: `id, label, kind, confidence`.
pub fn nodes_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("label", DataType::Utf8, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("confidence", DataType::Float32, false),
    ]))
}

/// Arrow schema of the `edges` table: `source, target, label, frequency, confidence`.
pub fn edges_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("source", DataType::Utf8, false),
        Field::new("target", DataType::Utf8, false),
        Field::new("label", DataType::Utf8, false),
        Field::new("frequency", DataType::Float32, false),
        Field::new("confidence", DataType::Float32, false),
    ]))
}

/// One `RecordBatch` holding all snapshot nodes.
pub fn nodes_batch(snap: &GraphSnapshot) -> Result<RecordBatch, arrow::error::ArrowError> {
    let ids: Vec<&str> = snap.nodes.iter().map(|n| n.id.as_str()).collect();
    let labels: Vec<&str> = snap.nodes.iter().map(|n| n.label.as_str()).collect();
    let kinds: Vec<&str> = snap.nodes.iter().map(|n| n.kind.as_str()).collect();
    let confs: Vec<f32> = snap.nodes.iter().map(|n| n.confidence).collect();
    RecordBatch::try_new(
        nodes_schema(),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(labels)),
            Arc::new(StringArray::from(kinds)),
            Arc::new(Float32Array::from(confs)),
        ],
    )
}

/// One `RecordBatch` holding all snapshot edges.
pub fn edges_batch(snap: &GraphSnapshot) -> Result<RecordBatch, arrow::error::ArrowError> {
    let src: Vec<&str> = snap.edges.iter().map(|e| e.source.as_str()).collect();
    let tgt: Vec<&str> = snap.edges.iter().map(|e| e.target.as_str()).collect();
    let lbl: Vec<&str> = snap.edges.iter().map(|e| e.label.as_str()).collect();
    let freq: Vec<f32> = snap.edges.iter().map(|e| e.frequency).collect();
    let conf: Vec<f32> = snap.edges.iter().map(|e| e.confidence).collect();
    RecordBatch::try_new(
        edges_schema(),
        vec![
            Arc::new(StringArray::from(src)),
            Arc::new(StringArray::from(tgt)),
            Arc::new(StringArray::from(lbl)),
            Arc::new(Float32Array::from(freq)),
            Arc::new(Float32Array::from(conf)),
        ],
    )
}

/// MemTable-backed `TableProvider` for the snapshot nodes.
pub fn nodes_table(snap: &GraphSnapshot) -> DfResult<MemTable> {
    MemTable::try_new(nodes_schema(), vec![vec![nodes_batch(snap)?]])
}

/// MemTable-backed `TableProvider` for the snapshot edges.
pub fn edges_table(snap: &GraphSnapshot) -> DfResult<MemTable> {
    MemTable::try_new(edges_schema(), vec![vec![edges_batch(snap)?]])
}

/// The `(nodes, edges)` providers as trait objects, ready to register.
pub fn graph_tables(
    snap: &GraphSnapshot,
) -> DfResult<(Arc<dyn TableProvider>, Arc<dyn TableProvider>)> {
    Ok((
        Arc::new(nodes_table(snap)?),
        Arc::new(edges_table(snap)?),
    ))
}

/// Register `nodes` + `edges` into a DataFusion `SessionContext`, so a consumer
/// can `ctx.sql("SELECT * FROM nodes")` / `edges`. Requires the full `query`
/// feature (a `SessionContext` is the executable surface).
#[cfg(feature = "query")]
pub fn register_graph(
    ctx: &datafusion::prelude::SessionContext,
    snap: &GraphSnapshot,
) -> DfResult<()> {
    ctx.register_table("nodes", Arc::new(nodes_table(snap)?))?;
    ctx.register_table("edges", Arc::new(edges_table(snap)?))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::canonical_node::{EdgeBlock, NodeRow};
    use lance_graph_contract::soa_graph::{project_snapshot, OSINT_GOTHAM};
    use lance_graph_contract::NodeGuid;

    /// Two OSINT members in families 0xA, 0xB; the family-0xA member carries an
    /// out-of-family adapter byte 0x0B → family 0xB. project_snapshot →
    /// GraphSnapshot → arrow tables. End-to-end head → DataFusion.
    fn sample_snapshot() -> GraphSnapshot {
        let mut a_edges = EdgeBlock::default();
        a_edges.out_family[0] = 0x0B;
        let rows = [
            NodeRow {
                key: NodeGuid::new(NodeGuid::CLASSID_OSINT, 1, 0, 0, 0xA, 1),
                edges: a_edges,
                value: [0u8; 480],
            },
            NodeRow {
                key: NodeGuid::new(NodeGuid::CLASSID_OSINT, 2, 0, 0, 0xB, 1),
                edges: EdgeBlock::default(),
                value: [0u8; 480],
            },
        ];
        project_snapshot(&rows, &OSINT_GOTHAM)
    }

    #[test]
    fn node_and_edge_batches_match_schema_and_counts() {
        let snap = sample_snapshot();
        let nb = nodes_batch(&snap).unwrap();
        let eb = edges_batch(&snap).unwrap();
        // 2 members + 2 family nodes = 4 node rows.
        assert_eq!(nb.num_rows(), snap.nodes.len());
        assert_eq!(nb.num_rows(), 4);
        assert_eq!(nb.schema().field(0).name(), "id");
        // 2 member-of edges + 1 out-of-family ("references") edge = 3.
        assert_eq!(eb.num_rows(), snap.edges.len());
        assert_eq!(eb.num_rows(), 3);
        assert_eq!(eb.schema().field(0).name(), "source");
    }

    #[test]
    fn tables_build_as_providers() {
        let snap = sample_snapshot();
        let (nodes, edges) = graph_tables(&snap).unwrap();
        use datafusion::datasource::TableType;
        assert_eq!(nodes.table_type(), TableType::Base);
        assert_eq!(edges.table_type(), TableType::Base);
        assert_eq!(nodes.schema().field(2).name(), "kind");
    }

    #[cfg(feature = "query")]
    #[tokio::test]
    async fn sql_over_registered_graph() {
        use datafusion::prelude::SessionContext;
        let snap = sample_snapshot();
        let ctx = SessionContext::new();
        register_graph(&ctx, &snap).unwrap();

        // SQL "out()": one-hop from the family-0xA member to family:00000b.
        let member_a = snap
            .nodes
            .iter()
            .find(|n| n.kind == "OSINT/Gotham" && n.props.iter().any(|(k, v)| k == "family" && v == "00000a"))
            .unwrap()
            .id
            .clone();
        let df = ctx
            .sql(&format!(
                "SELECT target FROM edges WHERE source = '{member_a}' AND label = 'references'"
            ))
            .await
            .unwrap();
        let batches = df.collect().await.unwrap();
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 1, "the out-of-family adapter edge is queryable via SQL");

        // GROUP BY over node kinds: 2 OSINT members + 2 family nodes.
        let df = ctx
            .sql("SELECT count(*) AS n FROM nodes")
            .await
            .unwrap();
        let batches = df.collect().await.unwrap();
        assert_eq!(batches[0].num_rows(), 1);
    }
}
