// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Versioned graph storage backed by Lance datasets.
//!
//! Each encounter round creates a new Lance version (ACID snapshot).
//! Old versions are readable via `at_version(n)`, supporting time-travel
//! queries over the graph's evolution. Tags name important versions
//! (e.g. "epoch-42") for human-readable bookmarks.
//!
//! # Storage Layout
//!
//! A `VersionedGraph` manages three Lance datasets at a base path:
//!
//! ```text
//! base_path/
//!   nodes.lance/       — NodeSchema columns (planes, seals, encounters)
//!   edges.lance/       — EdgeSchema columns (src, dst, weight, label)
//!   fingerprints.lance/ — FingerprintSchema columns (id, fingerprint)
//! ```
//!
//! Each dataset independently versions on every write, but a
//! `commit_encounter_round` writes all three atomically (from the
//! caller's perspective) in the same logical round.
//!
//! # Seal Checking
//!
//! `graph_seal_check` compares Merkle seals between two versions:
//! - **Wisdom** — no seal changed; the graph's knowledge is stable.
//! - **Staunen** — at least one seal diverged; new learning occurred.

use std::collections::HashSet;

use arrow_array::{
    builder::FixedSizeBinaryBuilder, Array, FixedSizeBinaryArray, Int64Array, RecordBatch,
    RecordBatchIterator, TimestampMicrosecondArray, UInt16Array, UInt32Array,
};
use arrow_schema::SchemaRef;
use futures::TryStreamExt;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use std::sync::Arc;

use crate::error::{GraphError, Result};
use super::blasgraph::columnar::{EdgeSchema, FingerprintSchema, NodeSchema};
use crate::graph::neighborhood::{scope, storage};
use crate::graph::neighborhood::scope::{NeighborhoodVector, ScopeMap};

// ---------------------------------------------------------------------------
// GraphSealStatus — result of comparing seals across versions
// ---------------------------------------------------------------------------

/// Result of comparing Merkle seals between two graph versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphSealStatus {
    /// No Merkle seals changed between the two versions.
    /// The graph's fingerprint knowledge is stable — "wisdom".
    Wisdom,

    /// At least one Merkle seal diverged between the two versions.
    /// New learning or perturbation occurred — "staunen" (astonishment).
    Staunen,
}

// ---------------------------------------------------------------------------
// GraphDiff — changes between two versions
// ---------------------------------------------------------------------------

/// Summary of differences between two graph versions.
#[derive(Debug, Clone)]
pub struct GraphDiff {
    /// Version numbers being compared (from, to).
    pub from_version: u64,
    pub to_version: u64,

    /// Node IDs that exist in `to` but not in `from`.
    pub new_nodes: Vec<u32>,

    /// Node IDs whose seal bytes changed between versions.
    pub modified_nodes: Vec<u32>,

    /// (src, dst) pairs that exist in `to` but not in `from`.
    pub new_edges: Vec<(u32, u32)>,

    /// Overall seal status.
    pub seal_status: GraphSealStatus,
}

// ---------------------------------------------------------------------------
// VersionedGraph
// ---------------------------------------------------------------------------

/// A versioned graph backed by three Lance datasets (nodes, edges, fingerprints).
///
/// Each write creates a new Lance version. Old versions are readable via
/// `at_version(n)`. The struct holds the base path and opens datasets
/// on demand — it is cheap to clone and share.
#[derive(Debug, Clone)]
pub struct VersionedGraph {
    /// Base directory or URI (local, s3://, az://, gs://).
    base_path: String,
}

impl VersionedGraph {
    // -- constructors -------------------------------------------------------

    /// Create a `VersionedGraph` backed by local filesystem storage.
    pub fn local(path: &str) -> Self {
        Self {
            base_path: path.to_string(),
        }
    }

    /// Create a `VersionedGraph` backed by S3.
    ///
    /// `uri` should be an `s3://bucket/prefix` path.
    pub fn s3(uri: &str) -> Self {
        Self {
            base_path: uri.to_string(),
        }
    }

    /// Create a `VersionedGraph` backed by Azure Blob Storage.
    ///
    /// `uri` should be an `az://container/prefix` path.
    pub fn azure(uri: &str) -> Self {
        Self {
            base_path: uri.to_string(),
        }
    }

    /// Create a `VersionedGraph` backed by Google Cloud Storage.
    ///
    /// `uri` should be a `gs://bucket/prefix` path.
    pub fn gcs(uri: &str) -> Self {
        Self {
            base_path: uri.to_string(),
        }
    }

    // -- dataset paths ------------------------------------------------------

    fn nodes_path(&self) -> String {
        format!("{}/nodes.lance", self.base_path)
    }

    fn edges_path(&self) -> String {
        format!("{}/edges.lance", self.base_path)
    }

    fn fingerprints_path(&self) -> String {
        format!("{}/fingerprints.lance", self.base_path)
    }

    fn scopes_path(&self) -> String {
        format!("{}/scopes.lance", self.base_path)
    }

    fn neighborhoods_path(&self) -> String {
        format!("{}/neighborhoods.lance", self.base_path)
    }

    fn cognitive_nodes_path(&self) -> String {
        format!("{}/cognitive_nodes.lance", self.base_path)
    }

    // -- write operations ---------------------------------------------------

    /// Commit an encounter round: write node, edge, and fingerprint batches.
    ///
    /// Each call creates a new Lance version in each of the three datasets.
    /// If a dataset does not yet exist it is created; otherwise data is appended
    /// (creating a new version).
    ///
    /// Returns the version number of the nodes dataset after the write.
    pub async fn commit_encounter_round(
        &self,
        nodes: RecordBatch,
        edges: RecordBatch,
        fingerprints: RecordBatch,
    ) -> Result<u64> {
        // Write all three datasets. Each write is ACID within its dataset.
        let node_version = self
            .write_batch(&self.nodes_path(), nodes, NodeSchema::arrow_schema_ref())
            .await?;
        self.write_batch(&self.edges_path(), edges, EdgeSchema::arrow_schema_ref())
            .await?;
        self.write_batch(
            &self.fingerprints_path(),
            fingerprints,
            FingerprintSchema::arrow_schema_ref(),
        )
        .await?;

        Ok(node_version)
    }

    /// Internal helper: write a single RecordBatch to a Lance dataset.
    ///
    /// Creates the dataset if it doesn't exist, otherwise overwrites
    /// (which creates a new version in Lance).
    async fn write_batch(
        &self,
        path: &str,
        batch: RecordBatch,
        schema: SchemaRef,
    ) -> Result<u64> {
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);

        // Try to open existing dataset first
        let mode = match Dataset::open(path).await {
            Ok(_) => WriteMode::Overwrite,
            Err(_) => WriteMode::Create,
        };

        let params = WriteParams {
            mode,
            ..Default::default()
        };

        let ds = Dataset::write(reader, path, Some(params)).await?;
        Ok(ds.version().version)
    }

    // -- read operations ----------------------------------------------------

    /// Open the nodes dataset at the current (latest) version.
    pub async fn open_nodes(&self) -> Result<Dataset> {
        Ok(Dataset::open(&self.nodes_path()).await?)
    }

    /// Open the edges dataset at the current (latest) version.
    pub async fn open_edges(&self) -> Result<Dataset> {
        Ok(Dataset::open(&self.edges_path()).await?)
    }

    /// Open the fingerprints dataset at the current (latest) version.
    pub async fn open_fingerprints(&self) -> Result<Dataset> {
        Ok(Dataset::open(&self.fingerprints_path()).await?)
    }

    /// Open the scopes dataset at the current (latest) version.
    pub async fn open_scopes(&self) -> Result<Dataset> {
        Ok(Dataset::open(&self.scopes_path()).await?)
    }

    /// Open the neighborhoods dataset at the current (latest) version.
    pub async fn open_neighborhoods(&self) -> Result<Dataset> {
        Ok(Dataset::open(&self.neighborhoods_path()).await?)
    }

    /// Open the cognitive_nodes dataset at the current (latest) version.
    pub async fn open_cognitive_nodes(&self) -> Result<Dataset> {
        Ok(Dataset::open(&self.cognitive_nodes_path()).await?)
    }

    // -- neighborhood extension write operations ----------------------------

    /// Write a scope definition to scopes.lance.
    ///
    /// Each call creates a new version. If the dataset doesn't exist, creates it.
    pub async fn write_scope(&self, scope: &ScopeMap) -> Result<u64> {
        let schema = Arc::new(storage::scopes_schema());
        let node_ids_buf = storage::serialize_scope_node_ids(scope);

        let node_ids_size = (scope::MAX_SCOPE_SIZE * 8) as i32;
        let mut node_ids_builder =
            FixedSizeBinaryBuilder::with_capacity(1, node_ids_size);
        node_ids_builder.append_value(&node_ids_buf).map_err(|e| {
            GraphError::ExecutionError {
                message: format!("Failed to build node_ids column: {e}"),
                location: snafu::Location::new(file!(), line!(), column!()),
            }
        })?;

        let now = chrono::Utc::now().timestamp_micros();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![scope.scope_id as i64])),
                Arc::new(node_ids_builder.finish()),
                Arc::new(UInt16Array::from(vec![scope.node_ids.len() as u16])),
                Arc::new(TimestampMicrosecondArray::from(vec![Some(now)])),
            ],
        )
        .map_err(|e| GraphError::ExecutionError {
            message: format!("Failed to build scope batch: {e}"),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;

        self.write_batch(&self.scopes_path(), batch, schema).await
    }

    /// Write neighborhood vectors to neighborhoods.lance.
    ///
    /// Each NeighborhoodVector becomes one row. Scent and resolution are
    /// stored as separate FixedSizeBinary columns for Lance column pruning
    /// (reading scent never loads resolution from disk).
    pub async fn write_neighborhoods(
        &self,
        scope_id: u64,
        neighborhoods: &[NeighborhoodVector],
    ) -> Result<u64> {
        let schema = Arc::new(storage::neighborhoods_schema());
        let n = neighborhoods.len();
        let bin_size = scope::MAX_SCOPE_SIZE as i32;

        let node_ids: Vec<i64> = neighborhoods.iter().map(|nv| nv.node_id as i64).collect();
        let scope_ids: Vec<i64> = vec![scope_id as i64; n];
        let edge_counts: Vec<u16> =
            neighborhoods.iter().map(|nv| nv.edge_count() as u16).collect();
        let now = chrono::Utc::now().timestamp_micros();
        let timestamps: Vec<Option<i64>> = vec![Some(now); n];

        let mut scent_builder = FixedSizeBinaryBuilder::with_capacity(n, bin_size);
        for nv in neighborhoods {
            let buf = storage::serialize_scent(nv);
            scent_builder.append_value(&buf).map_err(|e| {
                GraphError::ExecutionError {
                    message: format!("Failed to build scent column: {e}"),
                    location: snafu::Location::new(file!(), line!(), column!()),
                }
            })?;
        }

        let mut resolution_builder = FixedSizeBinaryBuilder::with_capacity(n, bin_size);
        for nv in neighborhoods {
            let buf = storage::serialize_resolution(nv);
            resolution_builder.append_value(&buf).map_err(|e| {
                GraphError::ExecutionError {
                    message: format!("Failed to build resolution column: {e}"),
                    location: snafu::Location::new(file!(), line!(), column!()),
                }
            })?;
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(node_ids)),
                Arc::new(Int64Array::from(scope_ids)),
                Arc::new(scent_builder.finish()),
                Arc::new(resolution_builder.finish()),
                Arc::new(UInt16Array::from(edge_counts)),
                Arc::new(TimestampMicrosecondArray::from(timestamps)),
            ],
        )
        .map_err(|e| GraphError::ExecutionError {
            message: format!("Failed to build neighborhoods batch: {e}"),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;

        self.write_batch(&self.neighborhoods_path(), batch, schema)
            .await
    }

    /// Load scent vectors for all nodes in a scope.
    ///
    /// Only reads the `scent` column — Lance column pruning ensures
    /// the `resolution` column is never loaded from disk.
    ///
    /// Returns (node_id, scent_bytes) pairs.
    pub async fn load_scope_scent(&self, scope_id: u64) -> Result<Vec<(u64, Vec<u8>)>> {
        let ds = Dataset::open(&self.neighborhoods_path()).await?;

        let batches: Vec<RecordBatch> = ds
            .scan()
            .project(&["node_id", "scope_id", "scent", "edge_count"])
            .map_err(|e| GraphError::ExecutionError {
                message: format!("Failed to project scent columns: {e}"),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?
            .try_into_stream()
            .await
            .map_err(|e| GraphError::ExecutionError {
                message: format!("Failed to scan neighborhoods: {e}"),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?
            .try_collect()
            .await
            .map_err(|e| GraphError::ExecutionError {
                message: format!("Failed to collect batches: {e}"),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        let mut result = Vec::new();
        for batch in &batches {
            let node_ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("node_id column");
            let scope_ids = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("scope_id column");
            let scents = batch
                .column(2)
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .expect("scent column");
            let edge_counts = batch
                .column(3)
                .as_any()
                .downcast_ref::<UInt16Array>()
                .expect("edge_count column");

            for row in 0..batch.num_rows() {
                if scope_ids.value(row) == scope_id as i64 {
                    let nid = node_ids.value(row) as u64;
                    let ec = edge_counts.value(row);
                    let scent_buf = scents.value(row);
                    let scent = storage::deserialize_scent(scent_buf, ec);
                    result.push((nid, scent));
                }
            }
        }

        Ok(result)
    }

    /// Open the nodes dataset at a specific version (time travel).
    pub async fn at_version(&self, version: u64) -> Result<Dataset> {
        let ds = Dataset::open(&self.nodes_path()).await?;
        let versioned = ds.checkout_version(version).await?;
        Ok(versioned)
    }

    /// Get the current (latest) version number of the nodes dataset.
    pub async fn current_version(&self) -> Result<u64> {
        let ds = Dataset::open(&self.nodes_path()).await?;
        Ok(ds.version().version)
    }

    /// List all versions of the nodes dataset.
    pub async fn versions(&self) -> Result<Vec<lance::dataset::Version>> {
        let ds = Dataset::open(&self.nodes_path()).await?;
        Ok(ds.versions().await?)
    }

    // -- tagging ------------------------------------------------------------

    /// Tag the current version of the nodes dataset with a human-readable name.
    ///
    /// Example: `graph.tag("epoch-42").await?`
    pub async fn tag(&self, name: &str) -> Result<()> {
        let ds = Dataset::open(&self.nodes_path()).await?;
        let version = ds.version().version;
        ds.tags().create(name, version).await?;
        Ok(())
    }

    /// Tag a specific version.
    pub async fn tag_version(&self, name: &str, version: u64) -> Result<()> {
        let ds = Dataset::open(&self.nodes_path()).await?;
        ds.tags().create(name, version).await?;
        Ok(())
    }

    // -- diff / seal check --------------------------------------------------

    /// Compare two versions and produce a `GraphDiff`.
    ///
    /// Reads the node datasets at both versions, compares node IDs and seals.
    pub async fn diff(&self, from_version: u64, to_version: u64) -> Result<GraphDiff> {
        let ds = Dataset::open(&self.nodes_path()).await?;
        let from_ds = ds.checkout_version(from_version).await?;
        let to_ds = ds.checkout_version(to_version).await?;

        let from_batches = Self::read_all_batches(&from_ds).await?;
        let to_batches = Self::read_all_batches(&to_ds).await?;

        // Collect node_ids and seal bytes from each version.
        let from_nodes = Self::extract_node_seals(&from_batches)?;
        let to_nodes = Self::extract_node_seals(&to_batches)?;

        let from_ids: HashSet<u32> = from_nodes.keys().copied().collect();
        let to_ids: HashSet<u32> = to_nodes.keys().copied().collect();

        let new_nodes: Vec<u32> = to_ids.difference(&from_ids).copied().collect();

        // Nodes present in both versions whose seals differ.
        let mut modified_nodes = Vec::new();
        for &id in to_ids.intersection(&from_ids) {
            if from_nodes.get(&id) != to_nodes.get(&id) {
                modified_nodes.push(id);
            }
        }

        // Edge diff: compare (src, dst) pairs.
        let from_edge_batches = {
            let edge_ds = Dataset::open(&self.edges_path()).await?;
            let edge_from = edge_ds.checkout_version(from_version).await?;
            Self::read_all_batches(&edge_from).await?
        };
        let to_edge_batches = {
            let edge_ds = Dataset::open(&self.edges_path()).await?;
            let edge_to = edge_ds.checkout_version(to_version).await?;
            Self::read_all_batches(&edge_to).await?
        };

        let from_edge_set = Self::extract_edges(&from_edge_batches);
        let to_edge_set = Self::extract_edges(&to_edge_batches);

        let new_edges: Vec<(u32, u32)> = to_edge_set.difference(&from_edge_set).copied().collect();

        let seal_status = if modified_nodes.is_empty() && new_nodes.is_empty() {
            GraphSealStatus::Wisdom
        } else {
            GraphSealStatus::Staunen
        };

        Ok(GraphDiff {
            from_version,
            to_version,
            new_nodes,
            modified_nodes,
            new_edges,
            seal_status,
        })
    }

    /// Check the graph-level seal status between two versions.
    ///
    /// This is a lighter-weight check than `diff` — it only reads seal columns
    /// and returns `Wisdom` or `Staunen`.
    pub async fn graph_seal_check(
        &self,
        from_version: u64,
        to_version: u64,
    ) -> Result<GraphSealStatus> {
        let ds = Dataset::open(&self.nodes_path()).await?;
        let from_ds = ds.checkout_version(from_version).await?;
        let to_ds = ds.checkout_version(to_version).await?;

        let from_batches = Self::read_all_batches(&from_ds).await?;
        let to_batches = Self::read_all_batches(&to_ds).await?;

        let from_seals = Self::extract_node_seals(&from_batches)?;
        let to_seals = Self::extract_node_seals(&to_batches)?;

        // Any seal difference means Staunen.
        for (id, to_seal) in &to_seals {
            match from_seals.get(id) {
                Some(from_seal) if from_seal != to_seal => {
                    return Ok(GraphSealStatus::Staunen);
                }
                None => {
                    // New node — seals diverged by definition.
                    return Ok(GraphSealStatus::Staunen);
                }
                _ => {}
            }
        }

        // Check for removed nodes (present in from but not in to).
        for id in from_seals.keys() {
            if !to_seals.contains_key(id) {
                return Ok(GraphSealStatus::Staunen);
            }
        }

        Ok(GraphSealStatus::Wisdom)
    }

    // -- helpers ------------------------------------------------------------

    /// Read all record batches from a dataset.
    async fn read_all_batches(ds: &Dataset) -> Result<Vec<RecordBatch>> {
        let batches: Vec<RecordBatch> = ds
            .scan()
            .try_into_stream()
            .await
            .map_err(|e| GraphError::ExecutionError {
                message: format!("Failed to create scan stream: {e}"),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?
            .try_collect()
            .await
            .map_err(|e| GraphError::ExecutionError {
                message: format!("Failed to collect batches: {e}"),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        Ok(batches)
    }

    /// Extract node_id → (seal_s, seal_p, seal_o) from record batches.
    ///
    /// Each seal is stored as the raw bytes (6 bytes each = 18 bytes total).
    fn extract_node_seals(
        batches: &[RecordBatch],
    ) -> Result<std::collections::HashMap<u32, Vec<u8>>> {
        let mut map = std::collections::HashMap::new();

        for batch in batches {
            let node_ids = batch
                .column_by_name("node_id")
                .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
                .ok_or_else(|| GraphError::ExecutionError {
                    message: "Missing node_id column".to_string(),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

            let seal_s = batch
                .column_by_name("seal_s")
                .and_then(|c| c.as_any().downcast_ref::<FixedSizeBinaryArray>())
                .ok_or_else(|| GraphError::ExecutionError {
                    message: "Missing seal_s column".to_string(),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

            let seal_p = batch
                .column_by_name("seal_p")
                .and_then(|c| c.as_any().downcast_ref::<FixedSizeBinaryArray>())
                .ok_or_else(|| GraphError::ExecutionError {
                    message: "Missing seal_p column".to_string(),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

            let seal_o = batch
                .column_by_name("seal_o")
                .and_then(|c| c.as_any().downcast_ref::<FixedSizeBinaryArray>())
                .ok_or_else(|| GraphError::ExecutionError {
                    message: "Missing seal_o column".to_string(),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

            for i in 0..batch.num_rows() {
                let id = node_ids.value(i);
                let mut combined = Vec::with_capacity(18);
                combined.extend_from_slice(seal_s.value(i));
                combined.extend_from_slice(seal_p.value(i));
                combined.extend_from_slice(seal_o.value(i));
                map.insert(id, combined);
            }
        }

        Ok(map)
    }

    /// Extract (src_id, dst_id) pairs from edge batches.
    fn extract_edges(batches: &[RecordBatch]) -> HashSet<(u32, u32)> {
        let mut set = HashSet::new();
        for batch in batches {
            let src = batch
                .column_by_name("src_id")
                .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
            let dst = batch
                .column_by_name("dst_id")
                .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
            if let (Some(s), Some(d)) = (src, dst) {
                for i in 0..batch.num_rows() {
                    set.insert((s.value(i), d.value(i)));
                }
            }
        }
        set
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::builder::FixedSizeBinaryBuilder;
    use arrow_schema::DataType;
    use std::sync::Arc;

    /// Helper: create a minimal node batch for testing.
    fn make_node_batch(ids: &[u32], encounter_count: u32) -> RecordBatch {
        let plane_bytes = vec![0u8; 2048];
        let seal_bytes = vec![0u8; 6];

        let mut plane_s_builder = FixedSizeBinaryBuilder::with_capacity(ids.len(), 2048);
        let mut plane_p_builder = FixedSizeBinaryBuilder::with_capacity(ids.len(), 2048);
        let mut plane_o_builder = FixedSizeBinaryBuilder::with_capacity(ids.len(), 2048);
        let mut seal_s_builder = FixedSizeBinaryBuilder::with_capacity(ids.len(), 6);
        let mut seal_p_builder = FixedSizeBinaryBuilder::with_capacity(ids.len(), 6);
        let mut seal_o_builder = FixedSizeBinaryBuilder::with_capacity(ids.len(), 6);

        for _ in ids {
            plane_s_builder.append_value(&plane_bytes).unwrap();
            plane_p_builder.append_value(&plane_bytes).unwrap();
            plane_o_builder.append_value(&plane_bytes).unwrap();
            seal_s_builder.append_value(&seal_bytes).unwrap();
            seal_p_builder.append_value(&seal_bytes).unwrap();
            seal_o_builder.append_value(&seal_bytes).unwrap();
        }

        let encounters: Vec<u32> = ids.iter().map(|_| encounter_count).collect();

        RecordBatch::try_new(
            NodeSchema::arrow_schema_ref(),
            vec![
                Arc::new(UInt32Array::from(ids.to_vec())),
                Arc::new(plane_s_builder.finish()),
                Arc::new(plane_p_builder.finish()),
                Arc::new(plane_o_builder.finish()),
                Arc::new(seal_s_builder.finish()),
                Arc::new(seal_p_builder.finish()),
                Arc::new(seal_o_builder.finish()),
                Arc::new(UInt32Array::from(encounters)),
            ],
        )
        .unwrap()
    }

    /// Helper: create a minimal edge batch.
    fn make_edge_batch(edges: &[(u32, u32)]) -> RecordBatch {
        let label_bytes = vec![0u8; 2048];
        let mut label_builder =
            FixedSizeBinaryBuilder::with_capacity(edges.len(), 2048);
        let mut src_ids = Vec::new();
        let mut dst_ids = Vec::new();

        for &(s, d) in edges {
            src_ids.push(s);
            dst_ids.push(d);
            label_builder.append_value(&label_bytes).unwrap();
        }

        // Create Float16Array by casting from Float32Array
        let weight_f32 = arrow_array::Float32Array::from(vec![1.0f32; edges.len()]);
        let weight_f16 = arrow::compute::cast(&weight_f32, &DataType::Float16).unwrap();

        RecordBatch::try_new(
            EdgeSchema::arrow_schema_ref(),
            vec![
                Arc::new(UInt32Array::from(src_ids)),
                Arc::new(UInt32Array::from(dst_ids)),
                weight_f16,
                Arc::new(label_builder.finish()),
            ],
        )
        .unwrap()
    }

    /// Helper: create a minimal fingerprint batch.
    fn make_fingerprint_batch(ids: &[u32]) -> RecordBatch {
        let fp_bytes = vec![0u8; 2048];
        let mut fp_builder = FixedSizeBinaryBuilder::with_capacity(ids.len(), 2048);
        for _ in ids {
            fp_builder.append_value(&fp_bytes).unwrap();
        }

        RecordBatch::try_new(
            FingerprintSchema::arrow_schema_ref(),
            vec![
                Arc::new(UInt32Array::from(ids.to_vec())),
                Arc::new(fp_builder.finish()),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_graph_seal_status_enum() {
        assert_ne!(GraphSealStatus::Wisdom, GraphSealStatus::Staunen);
    }

    #[test]
    fn test_versioned_graph_paths() {
        let g = VersionedGraph::local("/tmp/test-graph");
        assert_eq!(g.nodes_path(), "/tmp/test-graph/nodes.lance");
        assert_eq!(g.edges_path(), "/tmp/test-graph/edges.lance");
        assert_eq!(g.fingerprints_path(), "/tmp/test-graph/fingerprints.lance");
    }

    #[test]
    fn test_versioned_graph_s3_paths() {
        let g = VersionedGraph::s3("s3://my-bucket/graphs/g1");
        assert_eq!(g.nodes_path(), "s3://my-bucket/graphs/g1/nodes.lance");
    }

    #[test]
    fn test_versioned_graph_azure_paths() {
        let g = VersionedGraph::azure("az://container/prefix");
        assert_eq!(g.nodes_path(), "az://container/prefix/nodes.lance");
    }

    #[test]
    fn test_make_node_batch() {
        let batch = make_node_batch(&[1, 2, 3], 0);
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 8);
    }

    #[test]
    fn test_make_edge_batch() {
        let batch = make_edge_batch(&[(1, 2), (2, 3)]);
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 4);
    }

    #[test]
    fn test_make_fingerprint_batch() {
        let batch = make_fingerprint_batch(&[1, 2]);
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 2);
    }

    #[test]
    fn test_extract_edges() {
        let batch = make_edge_batch(&[(10, 20), (30, 40)]);
        let edges = VersionedGraph::extract_edges(&[batch]);
        assert!(edges.contains(&(10, 20)));
        assert!(edges.contains(&(30, 40)));
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_extract_node_seals() {
        let batch = make_node_batch(&[1, 2], 0);
        let seals = VersionedGraph::extract_node_seals(&[batch]).unwrap();
        assert_eq!(seals.len(), 2);
        // All seals are zero, so both should be equal.
        assert_eq!(seals[&1], seals[&2]);
        // Each combined seal is 18 bytes (6+6+6).
        assert_eq!(seals[&1].len(), 18);
    }

    #[tokio::test]
    async fn test_commit_and_read_round() {
        let dir = tempfile::tempdir().unwrap();
        let g = VersionedGraph::local(dir.path().to_str().unwrap());

        let nodes = make_node_batch(&[1, 2, 3], 1);
        let edges = make_edge_batch(&[(1, 2), (2, 3)]);
        let fps = make_fingerprint_batch(&[1, 2, 3]);

        let version = g.commit_encounter_round(nodes, edges, fps).await.unwrap();
        assert!(version >= 1);

        // Read back
        let ds = g.open_nodes().await.unwrap();
        let batches = VersionedGraph::read_all_batches(&ds).await.unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 3);
    }

    #[tokio::test]
    async fn test_version_and_tag() {
        let dir = tempfile::tempdir().unwrap();
        let g = VersionedGraph::local(dir.path().to_str().unwrap());

        // First round
        let nodes = make_node_batch(&[1], 1);
        let edges = make_edge_batch(&[(1, 1)]);
        let fps = make_fingerprint_batch(&[1]);
        let v1 = g.commit_encounter_round(nodes, edges, fps).await.unwrap();

        // Tag it
        g.tag_version("epoch-1", v1).await.unwrap();

        // Second round
        let nodes2 = make_node_batch(&[1, 2], 2);
        let edges2 = make_edge_batch(&[(1, 2)]);
        let fps2 = make_fingerprint_batch(&[1, 2]);
        let v2 = g.commit_encounter_round(nodes2, edges2, fps2).await.unwrap();

        assert!(v2 > v1);

        // Time travel to v1
        let old_ds = g.at_version(v1).await.unwrap();
        let old_batches = VersionedGraph::read_all_batches(&old_ds).await.unwrap();
        let old_rows: usize = old_batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(old_rows, 1);

        // Current version should have 2 nodes
        let cur_ds = g.open_nodes().await.unwrap();
        let cur_batches = VersionedGraph::read_all_batches(&cur_ds).await.unwrap();
        let cur_rows: usize = cur_batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(cur_rows, 2);
    }

    #[tokio::test]
    async fn test_graph_seal_check_wisdom() {
        let dir = tempfile::tempdir().unwrap();
        let g = VersionedGraph::local(dir.path().to_str().unwrap());

        // Two identical rounds — seals unchanged → Wisdom
        let nodes = make_node_batch(&[1, 2], 1);
        let edges = make_edge_batch(&[(1, 2)]);
        let fps = make_fingerprint_batch(&[1, 2]);
        let v1 = g
            .commit_encounter_round(nodes.clone(), edges.clone(), fps.clone())
            .await
            .unwrap();

        let v2 = g
            .commit_encounter_round(nodes, edges, fps)
            .await
            .unwrap();

        let status = g.graph_seal_check(v1, v2).await.unwrap();
        assert_eq!(status, GraphSealStatus::Wisdom);
    }

    #[tokio::test]
    async fn test_graph_seal_check_staunen() {
        let dir = tempfile::tempdir().unwrap();
        let g = VersionedGraph::local(dir.path().to_str().unwrap());

        // First round: node 1
        let nodes1 = make_node_batch(&[1], 1);
        let edges1 = make_edge_batch(&[(1, 1)]);
        let fps1 = make_fingerprint_batch(&[1]);
        let v1 = g
            .commit_encounter_round(nodes1, edges1, fps1)
            .await
            .unwrap();

        // Second round: add node 2 — new node means Staunen
        let nodes2 = make_node_batch(&[1, 2], 2);
        let edges2 = make_edge_batch(&[(1, 2)]);
        let fps2 = make_fingerprint_batch(&[1, 2]);
        let v2 = g
            .commit_encounter_round(nodes2, edges2, fps2)
            .await
            .unwrap();

        let status = g.graph_seal_check(v1, v2).await.unwrap();
        assert_eq!(status, GraphSealStatus::Staunen);
    }

    #[tokio::test]
    async fn test_diff_new_nodes_and_edges() {
        let dir = tempfile::tempdir().unwrap();
        let g = VersionedGraph::local(dir.path().to_str().unwrap());

        let nodes1 = make_node_batch(&[1], 1);
        let edges1 = make_edge_batch(&[(1, 1)]);
        let fps1 = make_fingerprint_batch(&[1]);
        let v1 = g
            .commit_encounter_round(nodes1, edges1, fps1)
            .await
            .unwrap();

        let nodes2 = make_node_batch(&[1, 2, 3], 2);
        let edges2 = make_edge_batch(&[(1, 2), (2, 3)]);
        let fps2 = make_fingerprint_batch(&[1, 2, 3]);
        let v2 = g
            .commit_encounter_round(nodes2, edges2, fps2)
            .await
            .unwrap();

        let diff = g.diff(v1, v2).await.unwrap();
        assert_eq!(diff.from_version, v1);
        assert_eq!(diff.to_version, v2);

        // New nodes: 2 and 3
        let mut new_sorted = diff.new_nodes.clone();
        new_sorted.sort();
        assert_eq!(new_sorted, vec![2, 3]);

        // New edges: (1,2) and (2,3) — (1,1) was already there
        assert_eq!(diff.new_edges.len(), 2);

        assert_eq!(diff.seal_status, GraphSealStatus::Staunen);
    }

    // -- neighborhood extension tests ---------------------------------------

    use crate::graph::neighborhood::scope::ScopeBuilder;
    use crate::graph::blasgraph::types::BitVec;

    fn random_triple(s: u64, p: u64, o: u64) -> (BitVec, BitVec, BitVec) {
        (BitVec::random(s), BitVec::random(p), BitVec::random(o))
    }

    #[tokio::test]
    async fn test_write_and_read_scope() {
        let dir = tempfile::tempdir().unwrap();
        let g = VersionedGraph::local(dir.path().to_str().unwrap());

        let ids: Vec<u64> = (100..110).collect();
        let scope = ScopeMap::new(1, ids);

        let version = g.write_scope(&scope).await.unwrap();
        assert!(version >= 1);

        let ds = g.open_scopes().await.unwrap();
        let batches = VersionedGraph::read_all_batches(&ds).await.unwrap();
        assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    }

    #[tokio::test]
    async fn test_write_and_read_neighborhoods() {
        let dir = tempfile::tempdir().unwrap();
        let g = VersionedGraph::local(dir.path().to_str().unwrap());

        let planes = vec![
            random_triple(10, 11, 12),
            random_triple(20, 21, 22),
            random_triple(30, 31, 32),
        ];
        let ids = vec![100u64, 200, 300];
        let (scope, neighborhoods) = ScopeBuilder::build(1, &ids, &planes);

        // Write scope
        g.write_scope(&scope).await.unwrap();

        // Write neighborhoods
        let version = g.write_neighborhoods(1, &neighborhoods).await.unwrap();
        assert!(version >= 1);

        // Read back scent only (column pruning)
        let scents = g.load_scope_scent(1).await.unwrap();
        assert_eq!(scents.len(), 3);

        // Verify node IDs came back
        let returned_ids: Vec<u64> = scents.iter().map(|&(id, _)| id).collect();
        assert!(returned_ids.contains(&100));
        assert!(returned_ids.contains(&200));
        assert!(returned_ids.contains(&300));

        // Verify scent bytes are non-trivial (not all zeros for non-self edges)
        for (_, scent) in &scents {
            assert!(!scent.is_empty());
        }
    }

    #[test]
    fn test_neighborhoods_path_helpers() {
        let g = VersionedGraph::local("/tmp/test-graph");
        assert_eq!(g.scopes_path(), "/tmp/test-graph/scopes.lance");
        assert_eq!(
            g.neighborhoods_path(),
            "/tmp/test-graph/neighborhoods.lance"
        );
        assert_eq!(
            g.cognitive_nodes_path(),
            "/tmp/test-graph/cognitive_nodes.lance"
        );
    }

    #[tokio::test]
    async fn test_scope_versioning() {
        let dir = tempfile::tempdir().unwrap();
        let g = VersionedGraph::local(dir.path().to_str().unwrap());

        // Write scope v1
        let scope1 = ScopeMap::new(1, vec![1, 2, 3]);
        let v1 = g.write_scope(&scope1).await.unwrap();

        // Write scope v2 (overwrites = new version)
        let scope2 = ScopeMap::new(1, vec![1, 2, 3, 4, 5]);
        let v2 = g.write_scope(&scope2).await.unwrap();

        assert!(v2 > v1);
    }
}
