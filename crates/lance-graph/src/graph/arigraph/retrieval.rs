// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Combined retrieval over triplet graph and episodic memory.
//!
//! Merges BFS graph expansion with fingerprint-based episode recall
//! to produce unified context for LLM agent prompts.

use std::collections::{HashMap, HashSet};

use super::episodic::EpisodicMemory;
use super::triplet_graph::TripletGraph;

/// Prompt template for extracting triplets from an observation.
pub const EXTRACTION_PROMPT: &str = r#"You are a knowledge extraction system. Given an observation, extract factual triplets.

Output triplets in the format: subject, relation, object
Separate multiple triplets with semicolons.

Observation: {observation}

Triplets:"#;

/// Prompt template for identifying outdated facts to refine.
pub const REFINING_PROMPT: &str = r#"You are a knowledge refinement system. Given existing graph triplets and a new observation, identify which existing triplets are now outdated or contradicted.

Existing triplets:
{existing_triplets}

New observation: {observation}

List outdated triplets (subject, relation, object) separated by semicolons, or "none" if all are still valid:"#;

/// Prompt template for generating a plan from retrieved context.
pub const PLAN_PROMPT: &str = r#"You are a planning system. Given the current context from a knowledge graph and episodic memory, generate a plan.

Graph context (known facts):
{graph_context}

Episodic context (past experiences):
{episodic_context}

Current observation: {observation}

Plan:"#;

/// Configuration for the retrieval pipeline.
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Maximum BFS depth for graph association retrieval.
    pub max_depth: usize,
    /// Maximum number of graph triplets to return.
    pub top_k: usize,
    /// Minimum truth expectation for including a triplet.
    pub threshold: f32,
    /// Number of episodic memory entries to retrieve.
    pub episodic_k: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            max_depth: 2,
            top_k: 20,
            threshold: 0.4,
            episodic_k: 5,
        }
    }
}

/// Result of a combined retrieval query.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// Triplet strings from the knowledge graph.
    pub graph_context: Vec<String>,
    /// Observation strings from episodic memory.
    pub episodic_context: Vec<String>,
    /// All entities found during retrieval.
    pub entities: HashSet<String>,
}

/// Combines triplet graph BFS with episodic memory retrieval.
///
/// Given seed entities (from a query or observation), expands through the
/// knowledge graph via BFS and retrieves similar episodes by fingerprint
/// distance.
pub struct OsintRetriever<'a> {
    /// Reference to the knowledge graph.
    graph: &'a TripletGraph,
    /// Reference to the episodic memory.
    memory: &'a EpisodicMemory,
    /// Retrieval configuration.
    config: RetrievalConfig,
}

impl<'a> OsintRetriever<'a> {
    /// Create a new retriever with default configuration.
    pub fn new(graph: &'a TripletGraph, memory: &'a EpisodicMemory) -> Self {
        Self {
            graph,
            memory,
            config: RetrievalConfig::default(),
        }
    }

    /// Create a new retriever with custom configuration.
    pub fn with_config(
        graph: &'a TripletGraph,
        memory: &'a EpisodicMemory,
        config: RetrievalConfig,
    ) -> Self {
        Self {
            graph,
            memory,
            config,
        }
    }

    /// Retrieve combined context from graph and episodic memory.
    ///
    /// 1. Expands seed entities through the graph via BFS up to `max_depth`.
    /// 2. Filters triplets by truth expectation threshold.
    /// 3. Retrieves top-k similar episodes from memory.
    /// 4. Returns the merged result.
    pub fn retrieve(
        &self,
        seed_entities: &HashSet<String>,
        query: &str,
    ) -> RetrievalResult {
        // Graph BFS retrieval
        let associated = self.graph.get_associated(seed_entities, self.config.max_depth);

        let mut graph_context: Vec<String> = Vec::new();
        let mut entities: HashSet<String> = seed_entities.clone();

        for triplet in &associated {
            if triplet.truth.expectation() < self.config.threshold {
                continue;
            }
            graph_context.push(triplet.to_string_repr());
            entities.insert(triplet.subject.clone());
            entities.insert(triplet.object.clone());

            if graph_context.len() >= self.config.top_k {
                break;
            }
        }

        // Episodic memory retrieval
        let episodes = self.memory.top_k(query, self.config.episodic_k);
        let episodic_context: Vec<String> = episodes
            .into_iter()
            .map(|ep| ep.observation.clone())
            .collect();

        RetrievalResult {
            graph_context,
            episodic_context,
            entities,
        }
    }

    /// Retrieve with per-entity depth tracking.
    ///
    /// Returns the standard retrieval result plus a map from each entity
    /// to the BFS depth at which it was first encountered.
    pub fn retrieve_with_depths(
        &self,
        seed_entities: &HashSet<String>,
        query: &str,
    ) -> (RetrievalResult, HashMap<String, usize>) {
        let mut depth_map: HashMap<String, usize> = HashMap::new();

        // Track depths during manual BFS
        let mut current: HashSet<String> =
            seed_entities.iter().map(|e| e.to_lowercase()).collect();
        for entity in &current {
            depth_map.insert(entity.clone(), 0);
        }

        let mut all_graph_context: Vec<String> = Vec::new();
        let mut all_entities: HashSet<String> = seed_entities.clone();
        let mut seen_indices: HashSet<usize> = HashSet::new();

        for depth in 0..self.config.max_depth {
            let mut next: HashSet<String> = HashSet::new();

            for entity in &current {
                if let Some(indices) = self.graph.entity_index.get(entity) {
                    for &idx in indices {
                        if seen_indices.contains(&idx) {
                            continue;
                        }
                        let triplet = &self.graph.triplets[idx];
                        if triplet.is_deleted() {
                            continue;
                        }
                        if triplet.truth.expectation() < self.config.threshold {
                            continue;
                        }
                        seen_indices.insert(idx);

                        all_graph_context.push(triplet.to_string_repr());
                        all_entities.insert(triplet.subject.clone());
                        all_entities.insert(triplet.object.clone());

                        let subj_lower = triplet.subject.to_lowercase();
                        let obj_lower = triplet.object.to_lowercase();
                        let other = if subj_lower == *entity {
                            &obj_lower
                        } else {
                            &subj_lower
                        };

                        if other != "itself" && !depth_map.contains_key(other) {
                            depth_map.insert(other.clone(), depth + 1);
                            next.insert(other.clone());
                        }
                    }
                }
            }

            current = next;

            if all_graph_context.len() >= self.config.top_k {
                all_graph_context.truncate(self.config.top_k);
                break;
            }
        }

        // Episodic memory retrieval
        let episodes = self.memory.top_k(query, self.config.episodic_k);
        let episodic_context: Vec<String> = episodes
            .into_iter()
            .map(|ep| ep.observation.clone())
            .collect();

        let result = RetrievalResult {
            graph_context: all_graph_context,
            episodic_context,
            entities: all_entities,
        };

        (result, depth_map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::arigraph::triplet_graph::Triplet;

    fn setup() -> (TripletGraph, EpisodicMemory) {
        let mut graph = TripletGraph::new();
        graph.add_triplets(&[
            Triplet::new("alice", "bob", "knows", 1),
            Triplet::new("bob", "carol", "knows", 2),
            Triplet::new("carol", "dave", "knows", 3),
        ]);

        let mut memory = EpisodicMemory::new(10);
        memory.add("Alice met Bob at the park", &["alice - knows - bob".to_string()], 1);
        memory.add("Carol introduced Dave to the group", &["carol - knows - dave".to_string()], 2);

        (graph, memory)
    }

    #[test]
    fn test_combined_retrieval() {
        let (graph, memory) = setup();
        let retriever = OsintRetriever::new(&graph, &memory);

        let seed: HashSet<String> = ["alice".to_string()].into_iter().collect();
        let result = retriever.retrieve(&seed, "alice and friends");

        assert!(!result.graph_context.is_empty());
        assert!(!result.episodic_context.is_empty());
        assert!(result.entities.contains("alice"));
        assert!(result.entities.contains("bob"));
    }

    #[test]
    fn test_empty_graph_retrieval() {
        let graph = TripletGraph::new();
        let memory = EpisodicMemory::new(10);
        let retriever = OsintRetriever::new(&graph, &memory);

        let seed: HashSet<String> = ["nobody".to_string()].into_iter().collect();
        let result = retriever.retrieve(&seed, "test");

        assert!(result.graph_context.is_empty());
        assert!(result.episodic_context.is_empty());
    }

    #[test]
    fn test_config_defaults() {
        let config = RetrievalConfig::default();
        assert_eq!(config.max_depth, 2);
        assert_eq!(config.top_k, 20);
        assert!((config.threshold - 0.4).abs() < f32::EPSILON);
        assert_eq!(config.episodic_k, 5);
    }

    #[test]
    fn test_retrieve_with_depths() {
        let (graph, memory) = setup();
        let retriever = OsintRetriever::new(&graph, &memory);

        let seed: HashSet<String> = ["alice".to_string()].into_iter().collect();
        let (result, depths) = retriever.retrieve_with_depths(&seed, "alice network");

        assert!(!result.graph_context.is_empty());
        assert_eq!(*depths.get("alice").unwrap(), 0);
        // bob is 1 hop from alice
        assert_eq!(*depths.get("bob").unwrap(), 1);
    }

    #[test]
    fn test_custom_config() {
        let (graph, memory) = setup();
        let config = RetrievalConfig {
            max_depth: 1,
            top_k: 1,
            threshold: 0.0,
            episodic_k: 1,
        };
        let retriever = OsintRetriever::with_config(&graph, &memory, config);

        let seed: HashSet<String> = ["alice".to_string()].into_iter().collect();
        let result = retriever.retrieve(&seed, "alice");

        // top_k=1 limits graph context
        assert!(result.graph_context.len() <= 1);
        // episodic_k=1 limits episodic context
        assert!(result.episodic_context.len() <= 1);
    }
}
