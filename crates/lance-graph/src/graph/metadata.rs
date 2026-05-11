// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Metadata graph store — the cold path skeleton.
//!
//! Stores graph nodes and edges as Arrow RecordBatches. Provides CRUD
//! operations and Cypher query access via DataFusion.
//!
//! This is the "skeleton" layer: small, sparse, structural.
//! Node properties, labels, and edge types live here.
//! The "flesh" (enrichment planes, fingerprints) lives in SPO3D.
//!
//! # Architecture
//!
//! ```text
//! MetadataStore
//!   ├── nodes: RecordBatch  { node_id: u32, label: utf8, properties... }
//!   ├── edges: RecordBatch  { source: u32, target: u32, edge_type: utf8, properties... }
//!   └── config: GraphConfig { mappings for CypherQuery }
//!
//! Query routing:
//!   "MATCH (p:Person) WHERE p.age > 30 RETURN p.name"
//!     → CypherQuery::new(cypher).with_config(config).execute(datasets)
//! ```
//!
//! # Examples
//!
//! ```rust
//! use lance_graph::graph::metadata::{MetadataStore, NodeRecord, EdgeRecord};
//!
//! # #[tokio::main]
//! # async fn main() -> lance_graph::Result<()> {
//! let mut store = MetadataStore::new();
//!
//! // Add nodes
//! store.add_node(NodeRecord::new(1, "Person").with_prop("name", "Alice"));
//! store.add_node(NodeRecord::new(2, "Person").with_prop("name", "Bob"));
//!
//! // Add edge
//! store.add_edge(EdgeRecord::new(1, 2, "KNOWS"));
//!
//! // Query via Cypher
//! let result = store.query("MATCH (p:Person) RETURN p.name").await?;
//! assert_eq!(result.num_rows(), 2);
//! # Ok(())
//! # }
//! ```

use arrow_array::{RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use std::collections::HashMap;
use std::sync::Arc;

use crate::config::GraphConfig;
use crate::error::Result;
use crate::query::CypherQuery;

/// A node record for the metadata skeleton.
#[derive(Debug, Clone)]
pub struct NodeRecord {
    /// Unique node identifier.
    pub node_id: u32,
    /// Node label (e.g., "Person", "Company").
    pub label: String,
    /// Property key-value pairs.
    pub properties: HashMap<String, String>,
}

impl NodeRecord {
    pub fn new(node_id: u32, label: impl Into<String>) -> Self {
        Self {
            node_id,
            label: label.into(),
            properties: HashMap::new(),
        }
    }

    pub fn with_prop(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }
}

/// An edge record for the metadata skeleton.
#[derive(Debug, Clone)]
pub struct EdgeRecord {
    /// Source node ID.
    pub source: u32,
    /// Target node ID.
    pub target: u32,
    /// Edge type (e.g., "KNOWS", "WORKS_FOR").
    pub edge_type: String,
    /// Property key-value pairs.
    pub properties: HashMap<String, String>,
}

impl EdgeRecord {
    pub fn new(source: u32, target: u32, edge_type: impl Into<String>) -> Self {
        Self {
            source,
            target,
            edge_type: edge_type.into(),
            properties: HashMap::new(),
        }
    }

    pub fn with_prop(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }
}

/// Cold-path metadata store for graph nodes and edges.
///
/// Stores the graph skeleton as Arrow RecordBatches. Queries go through
/// lance-graph's CypherQuery + DataFusion planner.
///
/// For persistence, the node and edge batches can be written to Lance
/// datasets directly — each write creates a new ACID version.
pub struct MetadataStore {
    nodes: Vec<NodeRecord>,
    edges: Vec<EdgeRecord>,
    /// Collected property keys across all nodes (for schema building).
    node_prop_keys: Vec<String>,
    /// Collected property keys across all edges.
    edge_prop_keys: Vec<String>,
}

impl MetadataStore {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            node_prop_keys: Vec::new(),
            edge_prop_keys: Vec::new(),
        }
    }

    /// Add a node to the store.
    pub fn add_node(&mut self, node: NodeRecord) {
        for key in node.properties.keys() {
            if !self.node_prop_keys.contains(key) {
                self.node_prop_keys.push(key.clone());
            }
        }
        self.nodes.push(node);
    }

    /// Add an edge to the store.
    pub fn add_edge(&mut self, edge: EdgeRecord) {
        for key in edge.properties.keys() {
            if !self.edge_prop_keys.contains(key) {
                self.edge_prop_keys.push(key.clone());
            }
        }
        self.edges.push(edge);
    }

    /// Get a node by ID.
    pub fn get_node(&self, node_id: u32) -> Option<&NodeRecord> {
        self.nodes.iter().find(|n| n.node_id == node_id)
    }

    /// Get all edges from a source node.
    pub fn get_edges_from(&self, source: u32) -> Vec<&EdgeRecord> {
        self.edges.iter().filter(|e| e.source == source).collect()
    }

    /// Get all edges to a target node.
    pub fn get_edges_to(&self, target: u32) -> Vec<&EdgeRecord> {
        self.edges.iter().filter(|e| e.target == target).collect()
    }

    /// Remove a node by ID (and all its edges).
    pub fn remove_node(&mut self, node_id: u32) {
        self.nodes.retain(|n| n.node_id != node_id);
        self.edges
            .retain(|e| e.source != node_id && e.target != node_id);
    }

    /// Remove edges between source and target with a given type.
    pub fn remove_edge(&mut self, source: u32, target: u32, edge_type: &str) {
        self.edges
            .retain(|e| !(e.source == source && e.target == target && e.edge_type == edge_type));
    }

    /// Get all distinct labels in the store.
    pub fn labels(&self) -> Vec<String> {
        let mut labels: Vec<String> = self
            .nodes
            .iter()
            .map(|n| n.label.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        labels.sort();
        labels
    }

    /// Get all distinct edge types in the store.
    pub fn edge_types(&self) -> Vec<String> {
        let mut types: Vec<String> = self
            .edges
            .iter()
            .map(|e| e.edge_type.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        types.sort();
        types
    }

    /// Build the GraphConfig for CypherQuery execution.
    ///
    /// Automatically maps all labels and edge types found in the store.
    fn build_config(&self) -> Result<GraphConfig> {
        let mut builder = GraphConfig::builder().with_default_node_id_field("node_id");

        for label in self.labels() {
            builder = builder.with_node_label(label, "node_id".to_string());
        }

        for edge_type in self.edge_types() {
            builder =
                builder.with_relationship(edge_type, "source".to_string(), "target".to_string());
        }

        builder.build()
    }

    /// Build a RecordBatch for nodes of a given label.
    fn nodes_batch(&self, label: &str) -> RecordBatch {
        let filtered: Vec<&NodeRecord> = self
            .nodes
            .iter()
            .filter(|n| n.label.eq_ignore_ascii_case(label))
            .collect();

        let mut fields = vec![
            Field::new("node_id", DataType::UInt32, false),
            Field::new("label", DataType::Utf8, false),
        ];

        // Add property columns
        let sorted_keys = {
            let mut keys = self.node_prop_keys.clone();
            keys.sort();
            keys
        };
        for key in &sorted_keys {
            fields.push(Field::new(key, DataType::Utf8, true));
        }

        let schema = Arc::new(Schema::new(fields));

        let ids: Vec<u32> = filtered.iter().map(|n| n.node_id).collect();
        let labels: Vec<&str> = filtered.iter().map(|n| n.label.as_str()).collect();

        let mut columns: Vec<Arc<dyn arrow_array::Array>> = vec![
            Arc::new(UInt32Array::from(ids)),
            Arc::new(StringArray::from(labels)),
        ];

        // Add property columns
        for key in &sorted_keys {
            let values: Vec<Option<&str>> = filtered
                .iter()
                .map(|n| n.properties.get(key).map(|v| v.as_str()))
                .collect();
            columns.push(Arc::new(StringArray::from(values)));
        }

        RecordBatch::try_new(schema, columns).unwrap()
    }

    /// Build a RecordBatch for edges of a given type.
    fn edges_batch(&self, edge_type: &str) -> RecordBatch {
        let filtered: Vec<&EdgeRecord> = self
            .edges
            .iter()
            .filter(|e| e.edge_type.eq_ignore_ascii_case(edge_type))
            .collect();

        let mut fields = vec![
            Field::new("source", DataType::UInt32, false),
            Field::new("target", DataType::UInt32, false),
            Field::new("edge_type", DataType::Utf8, false),
        ];

        let sorted_keys = {
            let mut keys = self.edge_prop_keys.clone();
            keys.sort();
            keys
        };
        for key in &sorted_keys {
            fields.push(Field::new(key, DataType::Utf8, true));
        }

        let schema = Arc::new(Schema::new(fields));

        let sources: Vec<u32> = filtered.iter().map(|e| e.source).collect();
        let targets: Vec<u32> = filtered.iter().map(|e| e.target).collect();
        let types: Vec<&str> = filtered.iter().map(|e| e.edge_type.as_str()).collect();

        let mut columns: Vec<Arc<dyn arrow_array::Array>> = vec![
            Arc::new(UInt32Array::from(sources)),
            Arc::new(UInt32Array::from(targets)),
            Arc::new(StringArray::from(types)),
        ];

        for key in &sorted_keys {
            let values: Vec<Option<&str>> = filtered
                .iter()
                .map(|e| e.properties.get(key).map(|v| v.as_str()))
                .collect();
            columns.push(Arc::new(StringArray::from(values)));
        }

        RecordBatch::try_new(schema, columns).unwrap()
    }

    /// Build all datasets (HashMap<String, RecordBatch>) for CypherQuery execution.
    pub fn to_datasets(&self) -> HashMap<String, RecordBatch> {
        let mut datasets = HashMap::new();

        for label in self.labels() {
            datasets.insert(label.clone(), self.nodes_batch(&label));
        }

        for edge_type in self.edge_types() {
            datasets.insert(edge_type.clone(), self.edges_batch(&edge_type));
        }

        datasets
    }

    /// Execute a Cypher query against this metadata store.
    ///
    /// Uses lance-graph's DataFusion Cypher planner for query execution.
    pub async fn query(&self, cypher: &str) -> Result<RecordBatch> {
        let config = self.build_config()?;
        let datasets = self.to_datasets();

        let query = CypherQuery::new(cypher)?.with_config(config);
        query.execute(datasets, None).await
    }

    /// Execute a parameterized Cypher query.
    pub async fn query_with_params(
        &self,
        cypher: &str,
        params: HashMap<String, serde_json::Value>,
    ) -> Result<RecordBatch> {
        let config = self.build_config()?;
        let datasets = self.to_datasets();

        let mut query = CypherQuery::new(cypher)?.with_config(config);
        for (key, value) in params {
            query = query.with_parameter(key, value);
        }
        query.execute(datasets, None).await
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

impl Default for MetadataStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn social_graph() -> MetadataStore {
        let mut store = MetadataStore::new();

        store.add_node(
            NodeRecord::new(1, "Person")
                .with_prop("name", "Alice")
                .with_prop("age", "25"),
        );
        store.add_node(
            NodeRecord::new(2, "Person")
                .with_prop("name", "Bob")
                .with_prop("age", "35"),
        );
        store.add_node(
            NodeRecord::new(3, "Person")
                .with_prop("name", "Charlie")
                .with_prop("age", "30"),
        );
        store.add_node(NodeRecord::new(10, "Company").with_prop("name", "Acme"));

        store.add_edge(EdgeRecord::new(1, 2, "KNOWS").with_prop("since", "2020"));
        store.add_edge(EdgeRecord::new(2, 3, "KNOWS").with_prop("since", "2019"));
        store.add_edge(EdgeRecord::new(1, 10, "WORKS_FOR"));

        store
    }

    #[test]
    fn test_node_crud() {
        let mut store = MetadataStore::new();
        store.add_node(NodeRecord::new(1, "Person").with_prop("name", "Alice"));
        store.add_node(NodeRecord::new(2, "Person").with_prop("name", "Bob"));

        assert_eq!(store.node_count(), 2);
        assert!(store.get_node(1).is_some());
        assert_eq!(store.get_node(1).unwrap().properties["name"], "Alice");

        store.remove_node(1);
        assert_eq!(store.node_count(), 1);
        assert!(store.get_node(1).is_none());
    }

    #[test]
    fn test_edge_crud() {
        let mut store = MetadataStore::new();
        store.add_node(NodeRecord::new(1, "Person"));
        store.add_node(NodeRecord::new(2, "Person"));
        store.add_edge(EdgeRecord::new(1, 2, "KNOWS"));
        store.add_edge(EdgeRecord::new(2, 1, "KNOWS"));

        assert_eq!(store.edge_count(), 2);
        assert_eq!(store.get_edges_from(1).len(), 1);
        assert_eq!(store.get_edges_to(1).len(), 1);

        store.remove_edge(1, 2, "KNOWS");
        assert_eq!(store.edge_count(), 1);
    }

    #[test]
    fn test_remove_node_cascades_edges() {
        let mut store = social_graph();
        assert_eq!(store.edge_count(), 3);

        store.remove_node(1); // Alice — has 2 outgoing edges
        assert_eq!(store.edge_count(), 1); // Only Bob→Charlie remains
    }

    #[test]
    fn test_labels_and_types() {
        let store = social_graph();
        assert_eq!(store.labels(), vec!["Company", "Person"]);
        assert_eq!(store.edge_types(), vec!["KNOWS", "WORKS_FOR"]);
    }

    #[test]
    fn test_to_datasets() {
        let store = social_graph();
        let datasets = store.to_datasets();

        // Should have datasets for each label and edge type
        assert!(datasets.contains_key("Person"));
        assert!(datasets.contains_key("Company"));
        assert!(datasets.contains_key("KNOWS"));
        assert!(datasets.contains_key("WORKS_FOR"));

        // Person dataset should have 3 rows
        assert_eq!(datasets["Person"].num_rows(), 3);
        // Company dataset should have 1 row
        assert_eq!(datasets["Company"].num_rows(), 1);
        // KNOWS edges should have 2 rows
        assert_eq!(datasets["KNOWS"].num_rows(), 2);
    }

    #[tokio::test]
    async fn test_cypher_query_match_all_persons() {
        let store = social_graph();
        let result = store.query("MATCH (p:Person) RETURN p.name").await.unwrap();
        assert_eq!(result.num_rows(), 3);
    }

    #[tokio::test]
    async fn test_cypher_query_match_with_filter() {
        let store = social_graph();
        let result = store
            .query("MATCH (p:Person) WHERE p.age > '30' RETURN p.name")
            .await
            .unwrap();
        // Bob (35) matches; age is string comparison here
        assert!(result.num_rows() >= 1);
    }

    #[tokio::test]
    async fn test_cypher_query_relationship_traversal() {
        let store = social_graph();
        let result = store
            .query("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
            .await
            .unwrap();
        // Alice→Bob, Bob→Charlie
        assert_eq!(result.num_rows(), 2);
    }

    #[tokio::test]
    async fn test_cypher_query_cross_label_traversal() {
        let store = social_graph();
        let result = store
            .query("MATCH (p:Person)-[:WORKS_FOR]->(c:Company) RETURN p.name, c.name")
            .await
            .unwrap();
        // Alice→Acme
        assert_eq!(result.num_rows(), 1);
    }

    #[test]
    fn test_config_builds_correctly() {
        let store = social_graph();
        let config = store.build_config().unwrap();

        assert!(config.get_node_mapping("Person").is_some());
        assert!(config.get_node_mapping("Company").is_some());
        assert!(config.get_relationship_mapping("KNOWS").is_some());
        assert!(config.get_relationship_mapping("WORKS_FOR").is_some());
    }
}
