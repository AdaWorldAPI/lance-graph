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
                if triplet.subject == *subj && triplet.relation == *rel && triplet.object == *obj {
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

// ============================================================================
// Missing Python transcode: normalization, directions, spatial, exclude
// Ported from AriGraph utils.py + parent_graph.py
// ============================================================================

/// Normalize a triplet: alias "I"→"inventory", "P"→"player", lowercase, strip punctuation.
///
/// Transcoded from Python `clear_triplet()` (utils.py:163-172).
pub fn clear_triplet(subject: &str, relation: &str, object: &str) -> (String, String, String) {
    fn normalize(s: &str) -> String {
        let s = match s.trim() {
            "I" => "inventory",
            "P" => "player",
            other => other,
        };
        s.to_lowercase()
            .trim_matches(|c: char| "\"'. `;:".contains(c))
            .to_string()
    }
    (normalize(subject), normalize(relation), normalize(object))
}

/// Extract cardinal direction from an action string.
///
/// Transcoded from Python `find_direction()` (utils.py:61-70).
pub fn find_direction(action: &str) -> &'static str {
    let a = action.to_lowercase();
    if a.contains("north") {
        "is north of"
    } else if a.contains("east") {
        "is east of"
    } else if a.contains("south") {
        "is south of"
    } else if a.contains("west") {
        "is west of"
    } else {
        "can be achieved from"
    }
}

/// Extract the opposite cardinal direction.
///
/// Transcoded from Python `find_opposite_direction()` (utils.py:72-81).
pub fn find_opposite_direction(action: &str) -> &'static str {
    let a = action.to_lowercase();
    if a.contains("north") {
        "is south of"
    } else if a.contains("east") {
        "is west of"
    } else if a.contains("south") {
        "is north of"
    } else if a.contains("west") {
        "is east of"
    } else {
        "can be achieved from"
    }
}

/// Check if a relation string denotes a spatial connection (north/south/east/west).
///
/// Transcoded from Python `check_conn()` (utils.py:160-161).
pub fn is_spatial_connection(relation: &str) -> bool {
    let r = relation.to_lowercase();
    r.contains("north") || r.contains("south") || r.contains("east") || r.contains("west")
}

/// Parse raw triplet text from LLM output.
///
/// Transcoded from Python `process_triplets()` (utils.py:26-40).
/// Handles: semicolon-separated triplets, leading digit strip, colon-delimited subjects.
pub fn process_triplets(raw: &str, timestamp: u64) -> Vec<Triplet> {
    raw.split(';')
        .filter_map(|part| {
            let parts: Vec<&str> = part.split(',').collect();
            if parts.len() != 3 {
                return None;
            }
            let mut subj = parts[0].trim().to_string();
            // Strip leading digit (e.g., "1. subject" → "subject")
            if subj.starts_with(|c: char| c.is_ascii_digit()) {
                if let Some(rest) = subj.get(2..) {
                    subj = rest.to_string();
                }
            }
            // Handle colon-delimited subjects (e.g., "Step 1: subject")
            if let Some(pos) = subj.rfind(':') {
                subj = subj[pos + 1..].to_string();
            }
            let subj = subj
                .trim_matches(|c: char| " '\n\"".contains(c))
                .to_string();
            let rel = parts[1]
                .trim_matches(|c: char| " '\n\"".contains(c))
                .to_string();
            let obj = parts[2]
                .trim_matches(|c: char| " '\n\"".contains(c))
                .to_string();
            if subj.is_empty() || rel.is_empty() || obj.is_empty() {
                return None;
            }
            let (s, r, o) = clear_triplet(&subj, &rel, &obj);
            Some(Triplet::new(&s, &o, &r, timestamp))
        })
        .collect()
}

/// Parse outdated triplet patterns from LLM refinement output.
///
/// Transcoded from Python `parse_triplets_removing()` (utils.py:83-98).
/// Handles the `[[old -> new], [old2 -> new2]]` format.
pub fn parse_triplets_removing(text: &str) -> Vec<(String, String, String)> {
    let text = if text.contains("[[") {
        text.split("[[").last().unwrap_or(text)
    } else if text.contains("[\n[") {
        text.split("[\n[").last().unwrap_or(text)
    } else {
        text
    };
    let text = text.replace('[', "");
    let text = text.trim_end_matches(']');
    text.split("],")
        .filter_map(|pair| {
            let parts: Vec<&str> = pair.split("->").collect();
            if parts.len() != 2 {
                return None;
            }
            let triplet_parts: Vec<&str> = parts[0].split(',').collect();
            if triplet_parts.len() != 3 {
                return None;
            }
            let subj = triplet_parts[0]
                .trim_matches(|c: char| " '\"\n".contains(c))
                .to_string();
            let rel = triplet_parts[1]
                .trim_matches(|c: char| " '\"\n".contains(c))
                .to_string();
            let obj = triplet_parts[2]
                .trim_matches(|c: char| " '\"\n".contains(c))
                .to_string();
            Some((subj, rel, obj))
        })
        .collect()
}

/// Find unexplored exits from a location by scanning triplets.
///
/// Transcoded from Python `find_unexplored_exits()` (utils.py:316-359).
pub fn find_unexplored_exits(location: &str, triplet_strings: &[String]) -> Vec<String> {
    let mut exits: HashSet<String> = HashSet::new();
    let mut explored: HashSet<String> = HashSet::new();
    let loc_lower = location.to_lowercase();

    for ts in triplet_strings {
        let parts: Vec<&str> = ts.split(", ").collect();
        if parts.len() < 3 {
            continue;
        }
        let subj = parts[0].to_lowercase();
        let rel = parts[1].to_lowercase();
        let obj = parts[2].to_lowercase();

        // First pass: exits FROM this location
        if subj == loc_lower && rel.contains("has exit") {
            exits.insert(obj.clone());
        } else if subj == loc_lower
            && (rel.contains("exit")
                || rel.contains("lead")
                || rel.contains("entr")
                || rel.contains("path"))
        {
            for dir in &["north", "south", "east", "west"] {
                if rel.contains(dir) || obj.contains(dir) {
                    exits.insert(dir.to_string());
                }
            }
        }

        // Second pass: explored directions (where we've come FROM)
        if obj == loc_lower && parts[1].split(' ').count() >= 2 {
            let direction = parts[1].split(' ').nth(1).unwrap_or("");
            if exits.contains(direction) {
                explored.insert(direction.to_string());
            }
        }
    }

    exits.difference(&explored).cloned().collect()
}

impl TripletGraph {
    /// Auto-extract spatial edges from triplets that contain directional relations.
    ///
    /// Transcoded from Python `compute_spatial_graph()` (parent_graph.py:131-154).
    pub fn compute_spatial_graph(&mut self) {
        self.spatial_edges.clear();
        let edges: Vec<(String, String, String)> = self
            .triplets
            .iter()
            .filter(|t| !t.is_deleted() && is_spatial_connection(&t.relation))
            .map(|t| {
                (
                    t.subject.to_lowercase(),
                    t.object.to_lowercase(),
                    t.relation.clone(),
                )
            })
            .collect();
        for (from, to, dir) in edges {
            self.add_spatial_edge(&from, &to, &dir);
        }
    }

    /// Find path returning movement directions instead of location names.
    ///
    /// Transcoded from Python `find_path()` (parent_graph.py:157-203).
    pub fn find_path_directions(&self, from: &str, to: &str) -> Option<Vec<String>> {
        if from == to {
            return Some(Vec::new());
        }
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, Vec<String>)> = VecDeque::new();
        visited.insert(from.to_string());
        queue.push_back((from.to_string(), Vec::new()));

        while let Some((current, directions)) = queue.pop_front() {
            if let Some(neighbors) = self.spatial_edges.get(&current) {
                for (neighbor, direction) in neighbors {
                    if neighbor == to {
                        let mut dirs = directions.clone();
                        // Convert "is north of" → "go north"
                        let movement = direction.replace("is ", "go ").replace(" of", "");
                        dirs.push(movement);
                        return Some(dirs);
                    }
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        let mut dirs = directions.clone();
                        let movement = direction.replace("is ", "go ").replace(" of", "");
                        dirs.push(movement);
                        queue.push_back((neighbor.clone(), dirs));
                    }
                }
            }
        }
        None
    }

    /// Filter new triplets against existing graph — return only truly new ones.
    ///
    /// Transcoded from Python `exclude()` (parent_graph.py:121-128).
    pub fn exclude(&self, candidates: &[Triplet]) -> Vec<Triplet> {
        candidates
            .iter()
            .filter(|t| {
                !self.triplets.iter().any(|existing| {
                    !existing.is_deleted()
                        && existing.subject == t.subject
                        && existing.object == t.object
                        && existing.relation == t.relation
                })
            })
            .cloned()
            .collect()
    }

    /// Delete outdated triplets with location protection.
    ///
    /// Won't delete triplets where both subject and object are known locations.
    /// Transcoded from Python `delete_triplets()` (parent_graph.py:89-94).
    pub fn delete_with_location_protection(
        &mut self,
        patterns: &[(String, String, String)],
        locations: &HashSet<String>,
    ) {
        for (subj, rel, obj) in patterns {
            // Protect spatial triplets
            if locations.contains(&subj.to_lowercase()) && locations.contains(&obj.to_lowercase()) {
                continue;
            }
            for triplet in self.triplets.iter_mut() {
                if triplet.subject == *subj && triplet.relation == *rel && triplet.object == *obj {
                    triplet.truth = TruthValue::unknown();
                }
            }
        }
    }

    /// Full OSINT update cycle: extract → exclude → refine → delete → spatial.
    ///
    /// Transcoded from Python `ContrieverGraph.update()` (contriever_graph.py:26-84).
    /// The LLM calls are replaced by pre-parsed inputs (the caller handles HTTP).
    pub fn update(
        &mut self,
        new_triplets_raw: &[Triplet],
        outdated_patterns: &[(String, String, String)],
        locations: &mut HashSet<String>,
        current_location: Option<&str>,
        previous_location: Option<&str>,
        action: Option<&str>,
    ) {
        // 1. Filter out already-known triplets
        let new_triplets = self.exclude(new_triplets_raw);

        // 2. Delete outdated triplets (with location protection)
        self.delete_with_location_protection(outdated_patterns, locations);

        // 3. Add spatial edges if the agent moved
        if let (Some(curr), Some(prev), Some(act)) = (current_location, previous_location, action) {
            if curr != prev && !act.contains("go to") {
                let dir = find_direction(act);
                let opp = find_opposite_direction(act);
                let curr_l = curr.to_lowercase();
                let prev_l = prev.to_lowercase();

                // Add bidirectional spatial triplets
                let spatial_fwd = Triplet::new(&curr_l, &prev_l, dir, 0);
                let spatial_rev = Triplet::new(&prev_l, &curr_l, opp, 0);
                self.add_triplets(&[spatial_fwd, spatial_rev]);
                locations.insert(curr_l);
            }
        }

        // 4. Add the new triplets
        self.add_triplets(&new_triplets);

        // 5. Rebuild spatial graph from all triplets
        self.compute_spatial_graph();
    }
}

// ============================================================================
// NARS inference integration (from adaworldapi/ndarray hpc/nars.rs)
// ============================================================================

impl TripletGraph {
    /// NARS deduction over 2-hop chains: A→B and B→C yields A→C.
    ///
    /// Scans all pairs of triplets sharing an entity and produces inferred
    /// triplets with deduced truth values: f = f1 * f2, c = f1 * f2 * c1 * c2.
    pub fn infer_deductions(&self) -> Vec<Triplet> {
        let mut inferred = Vec::new();

        for t1 in &self.triplets {
            if t1.is_deleted() {
                continue;
            }
            // Look for triplets where t1.object == t2.subject
            if let Some(indices) = self.entity_index.get(&t1.object.to_lowercase()) {
                for &idx in indices {
                    let t2 = &self.triplets[idx];
                    if t2.is_deleted() {
                        continue;
                    }
                    if t2.subject.to_lowercase() != t1.object.to_lowercase() {
                        continue;
                    }
                    // Don't create self-loops
                    if t1.subject == t2.object {
                        continue;
                    }
                    // Deduction: f = f1 * f2, c = f1 * f2 * c1 * c2
                    let f = t1.truth.frequency * t2.truth.frequency;
                    let c = t1.truth.frequency
                        * t2.truth.frequency
                        * t1.truth.confidence
                        * t2.truth.confidence;
                    let combined_rel = format!("{} (via {})", t2.relation, t1.object);
                    let triplet = Triplet::with_truth(
                        &t1.subject,
                        &t2.object,
                        &combined_rel,
                        TruthValue::new(f, c),
                        t2.timestamp.max(t1.timestamp),
                    );
                    // Only keep inferences with meaningful confidence
                    if triplet.truth.confidence > 0.1 {
                        inferred.push(triplet);
                    }
                }
            }
        }
        inferred
    }

    /// NARS contradiction detection: find triplets that conflict.
    ///
    /// Two triplets contradict if they share subject+object but have
    /// different relations and both have high confidence.
    /// Returns pairs of (triplet_index_a, triplet_index_b).
    pub fn detect_contradictions(&self, confidence_threshold: f32) -> Vec<(usize, usize)> {
        let mut contradictions = Vec::new();
        for (i, t1) in self.triplets.iter().enumerate() {
            if t1.is_deleted() || t1.truth.confidence < confidence_threshold {
                continue;
            }
            for (j, t2) in self.triplets.iter().enumerate().skip(i + 1) {
                if t2.is_deleted() || t2.truth.confidence < confidence_threshold {
                    continue;
                }
                // Same subject and object but different relation
                if t1.subject == t2.subject && t1.object == t2.object && t1.relation != t2.relation
                {
                    contradictions.push((i, j));
                }
            }
        }
        contradictions
    }

    /// NARS revision: when the same triplet is observed again, increase confidence.
    ///
    /// Finds existing triplets matching subject+relation+object and revises
    /// their truth value with new evidence.
    pub fn revise_with_evidence(&mut self, observation: &Triplet) {
        for triplet in self.triplets.iter_mut() {
            if !triplet.is_deleted()
                && triplet.subject == observation.subject
                && triplet.relation == observation.relation
                && triplet.object == observation.object
            {
                triplet.truth = triplet.truth.revision(&observation.truth);
                triplet.timestamp = observation.timestamp.max(triplet.timestamp);
                return;
            }
        }
        // Not found — add as new
        self.add_triplets(&[observation.clone()]);
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;

    #[test]
    fn test_clear_triplet_normalization() {
        let (s, r, o) = clear_triplet("I", "is in", "kitchen");
        assert_eq!(s, "inventory");
        assert_eq!(r, "is in");
        assert_eq!(o, "kitchen");

        let (s, _, _) = clear_triplet("P", "has", "sword");
        assert_eq!(s, "player");
    }

    #[test]
    fn test_find_direction() {
        assert_eq!(find_direction("go north"), "is north of");
        assert_eq!(find_direction("go east"), "is east of");
        assert_eq!(find_direction("go south"), "is south of");
        assert_eq!(find_direction("go west"), "is west of");
        assert_eq!(find_direction("teleport"), "can be achieved from");
    }

    #[test]
    fn test_find_opposite_direction() {
        assert_eq!(find_opposite_direction("go north"), "is south of");
        assert_eq!(find_opposite_direction("go east"), "is west of");
    }

    #[test]
    fn test_process_triplets() {
        let triplets = process_triplets(
            "alice, knows, bob; 1.charlie, works at, company; Step 1: dave, lives in, city",
            1,
        );
        assert_eq!(triplets.len(), 3);
        assert_eq!(triplets[0].subject, "alice");
        assert_eq!(triplets[0].object, "bob");
        assert_eq!(triplets[2].subject, "dave");
    }

    #[test]
    fn test_parse_triplets_removing() {
        let patterns = parse_triplets_removing(
            "[[alice, knows, bob -> alice, loves, bob], [cat, is in, box -> cat, is in, bag]]",
        );
        assert_eq!(patterns.len(), 2);
        assert_eq!(patterns[0].0, "alice");
    }

    #[test]
    fn test_compute_spatial_graph() {
        let mut g = TripletGraph::new();
        g.add_triplets(&[
            Triplet::new("room1", "room2", "is north of", 1),
            Triplet::new("room2", "room3", "is east of", 2),
            Triplet::new("alice", "bob", "knows", 3),
        ]);
        g.compute_spatial_graph();
        assert!(g.spatial_edges.contains_key("room1"));
        assert!(g.spatial_edges.contains_key("room2"));
        assert!(!g.spatial_edges.contains_key("alice"));
    }

    #[test]
    fn test_find_path_directions() {
        let mut g = TripletGraph::new();
        g.add_spatial_edge("room1", "room2", "is north of");
        g.add_spatial_edge("room2", "room3", "is east of");
        let dirs = g.find_path_directions("room1", "room3").unwrap();
        assert_eq!(dirs.len(), 2);
        assert!(dirs[0].contains("north"));
        assert!(dirs[1].contains("east"));
    }

    #[test]
    fn test_exclude_filters_known() {
        let mut g = TripletGraph::new();
        g.add_triplets(&[Triplet::new("alice", "bob", "knows", 1)]);
        let candidates = vec![
            Triplet::new("alice", "bob", "knows", 2),
            Triplet::new("carol", "dave", "knows", 2),
        ];
        let new = g.exclude(&candidates);
        assert_eq!(new.len(), 1);
        assert_eq!(new[0].subject, "carol");
    }

    #[test]
    fn test_delete_with_location_protection() {
        let mut g = TripletGraph::new();
        g.add_triplets(&[
            Triplet::new("room1", "room2", "is north of", 1),
            Triplet::new("alice", "bob", "knows", 2),
        ]);
        let locations: HashSet<String> = ["room1", "room2"].into_iter().map(String::from).collect();
        g.delete_with_location_protection(
            &[
                ("room1".into(), "is north of".into(), "room2".into()),
                ("alice".into(), "knows".into(), "bob".into()),
            ],
            &locations,
        );
        // room1→room2 protected (both are locations)
        assert!(!g.triplets[0].is_deleted());
        // alice→bob not protected
        assert!(g.triplets[1].is_deleted());
    }

    #[test]
    fn test_full_update_cycle() {
        let mut g = TripletGraph::new();
        let mut locations: HashSet<String> = ["kitchen"].into_iter().map(String::from).collect();

        let new_triplets = vec![
            Triplet::new("apple", "table", "is on", 1),
            Triplet::new("knife", "kitchen", "is in", 1),
        ];
        g.update(
            &new_triplets,
            &[],
            &mut locations,
            Some("bedroom"),
            Some("kitchen"),
            Some("go north"),
        );

        assert!(g.len() >= 2);
        assert!(locations.contains("bedroom"));
        assert!(!g.spatial_edges.is_empty());
    }

    #[test]
    fn test_infer_deductions() {
        let mut g = TripletGraph::new();
        g.add_triplets(&[
            Triplet::new("alice", "bob", "knows", 1),
            Triplet::new("bob", "carol", "knows", 2),
        ]);
        let inferred = g.infer_deductions();
        assert!(!inferred.is_empty());
        assert_eq!(inferred[0].subject, "alice");
        assert_eq!(inferred[0].object, "carol");
        assert!(inferred[0].truth.confidence > 0.0);
    }

    #[test]
    fn test_detect_contradictions() {
        let mut g = TripletGraph::new();
        g.add_triplets(&[
            Triplet::new("apple", "table", "is on", 1),
            Triplet::new("apple", "table", "is under", 2),
        ]);
        let contradictions = g.detect_contradictions(0.5);
        assert_eq!(contradictions.len(), 1);
    }

    #[test]
    fn test_revise_with_evidence() {
        let mut g = TripletGraph::new();
        g.add_triplets(&[Triplet::new("alice", "bob", "knows", 1)]);
        let c_before = g.triplets[0].truth.confidence;

        g.revise_with_evidence(&Triplet::new("alice", "bob", "knows", 2));
        // Revision should increase confidence
        assert!(g.triplets[0].truth.confidence >= c_before);
        // Should NOT create a duplicate
        assert_eq!(g.triplets.iter().filter(|t| !t.is_deleted()).count(), 1);
    }

    #[test]
    fn test_find_unexplored_exits() {
        let triplets = vec![
            "kitchen, has exit, north".to_string(),
            "kitchen, has exit, east".to_string(),
            "bedroom, is north of, kitchen".to_string(),
        ];
        let unexplored = find_unexplored_exits("kitchen", &triplets);
        // north is explored (bedroom is north of kitchen), east is not
        assert!(unexplored.contains(&"east".to_string()));
        assert!(!unexplored.contains(&"north".to_string()));
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

        g.delete_triplets(&[("alice".to_string(), "knows".to_string(), "bob".to_string())]);
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
        g.delete_triplets(&[("alice".to_string(), "knows".to_string(), "bob".to_string())]);

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
        assert_eq!(
            path,
            Some(vec!["room1".into(), "room2".into(), "room3".into()])
        );
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
