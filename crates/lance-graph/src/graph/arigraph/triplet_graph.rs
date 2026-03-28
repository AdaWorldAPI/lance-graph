// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Triplet-based knowledge graph for OSINT entity tracking.
//!
//! Stores subject-relation-object triplets with NARS truth values,
//! supports BFS association retrieval and spatial path finding.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::graph::fingerprint::{label_fp, Fingerprint};
use crate::graph::spo::truth::TruthValue;

/// A single knowledge triplet: subject -[relation]-> object.
#[derive(Debug, Clone)]
pub struct Triplet {
    /// The subject entity name.
    pub subject: String,
    /// The object entity name.
    pub object: String,
    /// The relation label connecting subject to object.
    pub relation: String,
    /// NARS truth value indicating confidence in this triplet.
    pub truth: TruthValue,
    /// Logical timestamp when this triplet was recorded.
    pub timestamp: u64,
}

impl Triplet {
    /// Create a new triplet with certain truth.
    pub fn new(subject: &str, object: &str, relation: &str, timestamp: u64) -> Self {
        Self {
            subject: subject.to_string(),
            object: object.to_string(),
            relation: relation.to_string(),
            truth: TruthValue::certain(),
            timestamp,
        }
    }

    /// Create a triplet with an explicit truth value.
    pub fn with_truth(
        subject: &str,
        object: &str,
        relation: &str,
        truth: TruthValue,
        timestamp: u64,
    ) -> Self {
        Self {
            subject: subject.to_string(),
            object: object.to_string(),
            relation: relation.to_string(),
            truth,
            timestamp,
        }
    }

    /// Format this triplet as "subject - relation - object".
    pub fn to_string_repr(&self) -> String {
        format!("{} - {} - {}", self.subject, self.relation, self.object)
    }

    /// Returns true if this triplet's truth is effectively unknown (soft-deleted).
    pub fn is_deleted(&self) -> bool {
        self.truth.confidence == 0.0
    }
}

/// A knowledge graph built from triplets with entity indexing and spatial edges.
///
/// Supports BFS-based association retrieval (multi-hop entity expansion)
/// and shortest-path finding over spatial edges.
#[derive(Debug, Clone)]
pub struct TripletGraph {
    /// All stored triplets.
    pub triplets: Vec<Triplet>,
    /// Maps entity name to indices in `triplets` where it appears as subject or object.
    pub entity_index: HashMap<String, Vec<usize>>,
    /// Spatial adjacency: location -> list of (neighbor, direction) pairs.
    pub spatial_edges: HashMap<String, Vec<(String, String)>>,
}

impl TripletGraph {
    /// Create an empty triplet graph.
    pub fn new() -> Self {
        Self {
            triplets: Vec::new(),
            entity_index: HashMap::new(),
            spatial_edges: HashMap::new(),
        }
    }

    /// Add triplets to the graph and update the entity index.
    pub fn add_triplets(&mut self, triplets: &[Triplet]) {
        for triplet in triplets {
            if triplet.relation == "free" {
                continue;
            }
            // Check for duplicate (same subject, object, relation)
            let is_dup = self.triplets.iter().any(|t| {
                t.subject == triplet.subject
                    && t.object == triplet.object
                    && t.relation == triplet.relation
            });
            if is_dup {
                continue;
            }

            let idx = self.triplets.len();
            self.triplets.push(triplet.clone());

            self.entity_index
                .entry(triplet.subject.clone())
                .or_default()
                .push(idx);
            self.entity_index
                .entry(triplet.object.clone())
                .or_default()
                .push(idx);
        }
    }

    /// Mark triplets matching (subject, relation, object) patterns as unknown (soft-delete).
    ///
    /// Matching uses exact string comparison. Matched triplets have their truth
    /// set to `TruthValue::unknown()` rather than being removed, preserving indices.
    pub fn delete_triplets(&mut self, patterns: &[(String, String, String)]) {
        for (subj, rel, obj) in patterns {
            for triplet in self.triplets.iter_mut() {
                if triplet.subject == *subj
                    && triplet.relation == *rel
                    && triplet.object == *obj
                {
                    triplet.truth = TruthValue::unknown();
                }
            }
        }
    }

    /// BFS association retrieval: expand from a seed set of entities for `steps` hops.
    ///
    /// Returns all non-deleted triplets reachable within `steps` BFS expansions
    /// from the initial entity set.
    pub fn get_associated(&self, entities: &HashSet<String>, steps: usize) -> Vec<&Triplet> {
        let mut current_entities: HashSet<String> =
            entities.iter().map(|e| e.to_lowercase()).collect();
        let mut seen_indices: HashSet<usize> = HashSet::new();
        let mut result: Vec<&Triplet> = Vec::new();

        for _step in 0..steps {
            let mut next_entities: HashSet<String> = HashSet::new();

            for entity in &current_entities {
                if let Some(indices) = self.entity_index.get(entity) {
                    for &idx in indices {
                        if seen_indices.contains(&idx) {
                            continue;
                        }
                        let triplet = &self.triplets[idx];
                        if triplet.is_deleted() {
                            continue;
                        }
                        seen_indices.insert(idx);
                        result.push(triplet);

                        // Expand to the other end of this triplet
                        let subj_lower = triplet.subject.to_lowercase();
                        let obj_lower = triplet.object.to_lowercase();
                        if subj_lower == *entity && obj_lower != "itself" {
                            next_entities.insert(obj_lower);
                        } else if obj_lower == *entity && subj_lower != "itself" {
                            next_entities.insert(subj_lower);
                        }
                    }
                }
            }

            current_entities = next_entities;
        }

        result
    }

    /// Add a spatial (navigational) edge between two locations.
    pub fn add_spatial_edge(&mut self, from: &str, to: &str, direction: &str) {
        self.spatial_edges
            .entry(from.to_string())
            .or_default()
            .push((to.to_string(), direction.to_string()));
    }

    /// BFS shortest path on spatial edges from `from` to `to`.
    ///
    /// Returns the sequence of location names forming the path, or `None`
    /// if no path exists.
    pub fn find_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        if from == to {
            return Some(vec![from.to_string()]);
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<Vec<String>> = VecDeque::new();

        visited.insert(from.to_string());
        queue.push_back(vec![from.to_string()]);

        while let Some(path) = queue.pop_front() {
            let current = path.last().unwrap();

            if let Some(neighbors) = self.spatial_edges.get(current) {
                for (neighbor, _direction) in neighbors {
                    if neighbor == to {
                        let mut full_path = path.clone();
                        full_path.push(neighbor.clone());
                        return Some(full_path);
                    }
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        let mut new_path = path.clone();
                        new_path.push(neighbor.clone());
                        queue.push_back(new_path);
                    }
                }
            }
        }

        None
    }

    /// Format all non-deleted triplets as "subject - relation - object" strings.
    pub fn triplets_to_strings(&self) -> Vec<String> {
        self.triplets
            .iter()
            .filter(|t| !t.is_deleted())
            .map(|t| t.to_string_repr())
            .collect()
    }

    /// Collect all unique entity names (subjects and objects) from non-deleted triplets.
    pub fn entities(&self) -> HashSet<String> {
        let mut ents = HashSet::new();
        for t in &self.triplets {
            if t.is_deleted() {
                continue;
            }
            ents.insert(t.subject.clone());
            ents.insert(t.object.clone());
        }
        ents
    }

    /// Number of triplets (including soft-deleted).
    pub fn len(&self) -> usize {
        self.triplets.len()
    }

    /// True if there are no triplets.
    pub fn is_empty(&self) -> bool {
        self.triplets.is_empty()
    }

    /// Remove all triplets, indices, and spatial edges.
    pub fn clear(&mut self) {
        self.triplets.clear();
        self.entity_index.clear();
        self.spatial_edges.clear();
    }

    /// Convert all non-deleted triplets to fingerprint triples (subject_fp, predicate_fp, object_fp).
    pub fn to_fingerprints(&self) -> Vec<(Fingerprint, Fingerprint, Fingerprint)> {
        self.triplets
            .iter()
            .filter(|t| !t.is_deleted())
            .map(|t| {
                (
                    label_fp(&t.subject),
                    label_fp(&t.relation),
                    label_fp(&t.object),
                )
            })
            .collect()
    }
}

impl Default for TripletGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triplet(s: &str, o: &str, r: &str) -> Triplet {
        Triplet::new(s, o, r, 0)
    }

    #[test]
    fn test_add_triplets() {
        let mut g = TripletGraph::new();
        let triplets = vec![
            make_triplet("alice", "bob", "knows"),
            make_triplet("bob", "carol", "knows"),
        ];
        g.add_triplets(&triplets);
        assert_eq!(g.len(), 2);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_add_skips_duplicates() {
        let mut g = TripletGraph::new();
        let t = make_triplet("alice", "bob", "knows");
        g.add_triplets(&[t.clone(), t.clone()]);
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn test_add_skips_free_relation() {
        let mut g = TripletGraph::new();
        let t = make_triplet("alice", "bob", "free");
        g.add_triplets(&[t]);
        assert_eq!(g.len(), 0);
    }

    #[test]
    fn test_delete_triplets() {
        let mut g = TripletGraph::new();
        g.add_triplets(&[make_triplet("alice", "bob", "knows")]);
        assert_eq!(g.triplets_to_strings().len(), 1);

        g.delete_triplets(&[(
            "alice".to_string(),
            "knows".to_string(),
            "bob".to_string(),
        )]);
        // Still in storage but filtered out from active triplets
        assert_eq!(g.len(), 1);
        assert_eq!(g.triplets_to_strings().len(), 0);
        assert!(g.triplets[0].is_deleted());
    }

    #[test]
    fn test_bfs_association_one_hop() {
        let mut g = TripletGraph::new();
        g.add_triplets(&[
            make_triplet("alice", "bob", "knows"),
            make_triplet("bob", "carol", "knows"),
            make_triplet("dave", "eve", "knows"),
        ]);

        let seed: HashSet<String> = ["alice".to_string()].into_iter().collect();
        let assoc = g.get_associated(&seed, 1);
        // One hop from alice: finds alice-bob
        assert_eq!(assoc.len(), 1);
        assert_eq!(assoc[0].subject, "alice");
    }

    #[test]
    fn test_bfs_association_two_hop() {
        let mut g = TripletGraph::new();
        g.add_triplets(&[
            make_triplet("alice", "bob", "knows"),
            make_triplet("bob", "carol", "knows"),
            make_triplet("dave", "eve", "knows"),
        ]);

        let seed: HashSet<String> = ["alice".to_string()].into_iter().collect();
        let assoc = g.get_associated(&seed, 2);
        // Two hops from alice: alice-bob, then bob-carol
        assert_eq!(assoc.len(), 2);
    }

    #[test]
    fn test_bfs_skips_deleted() {
        let mut g = TripletGraph::new();
        g.add_triplets(&[
            make_triplet("alice", "bob", "knows"),
            make_triplet("bob", "carol", "knows"),
        ]);
        g.delete_triplets(&[(
            "alice".to_string(),
            "knows".to_string(),
            "bob".to_string(),
        )]);

        let seed: HashSet<String> = ["alice".to_string()].into_iter().collect();
        let assoc = g.get_associated(&seed, 2);
        // alice-bob is deleted, so we cannot reach bob-carol either
        assert_eq!(assoc.len(), 0);
    }

    #[test]
    fn test_find_path() {
        let mut g = TripletGraph::new();
        g.add_spatial_edge("room1", "room2", "east");
        g.add_spatial_edge("room2", "room3", "east");
        g.add_spatial_edge("room1", "room4", "south");

        let path = g.find_path("room1", "room3");
        assert_eq!(path, Some(vec!["room1".into(), "room2".into(), "room3".into()]));
    }

    #[test]
    fn test_find_path_no_route() {
        let mut g = TripletGraph::new();
        g.add_spatial_edge("room1", "room2", "east");

        assert!(g.find_path("room1", "room99").is_none());
    }

    #[test]
    fn test_find_path_same_node() {
        let g = TripletGraph::new();
        let path = g.find_path("room1", "room1");
        assert_eq!(path, Some(vec!["room1".to_string()]));
    }

    #[test]
    fn test_entities() {
        let mut g = TripletGraph::new();
        g.add_triplets(&[
            make_triplet("alice", "bob", "knows"),
            make_triplet("bob", "carol", "knows"),
        ]);
        let ents = g.entities();
        assert!(ents.contains("alice"));
        assert!(ents.contains("bob"));
        assert!(ents.contains("carol"));
        assert_eq!(ents.len(), 3);
    }

    #[test]
    fn test_triplets_to_strings() {
        let mut g = TripletGraph::new();
        g.add_triplets(&[make_triplet("alice", "bob", "knows")]);
        let strings = g.triplets_to_strings();
        assert_eq!(strings, vec!["alice - knows - bob"]);
    }

    #[test]
    fn test_fingerprint_determinism() {
        let mut g = TripletGraph::new();
        g.add_triplets(&[make_triplet("alice", "bob", "knows")]);

        let fps1 = g.to_fingerprints();
        let fps2 = g.to_fingerprints();
        assert_eq!(fps1.len(), 1);
        assert_eq!(fps1[0].0, fps2[0].0); // subject fp
        assert_eq!(fps1[0].1, fps2[0].1); // relation fp
        assert_eq!(fps1[0].2, fps2[0].2); // object fp
    }

    #[test]
    fn test_clear() {
        let mut g = TripletGraph::new();
        g.add_triplets(&[make_triplet("a", "b", "r")]);
        g.add_spatial_edge("x", "y", "east");
        g.clear();
        assert!(g.is_empty());
        assert!(g.entity_index.is_empty());
        assert!(g.spatial_edges.is_empty());
    }
}
