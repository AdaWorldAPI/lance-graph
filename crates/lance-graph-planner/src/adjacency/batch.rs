//! Batch adjacency operations — the vectorized traversal primitive.

/// Result of batch_adjacent(): flat vector + offsets (Arrow ListArray pattern).
#[derive(Debug, Clone)]
pub struct AdjacencyBatch {
    /// Source node IDs (one per batch entry).
    pub source_ids: Vec<u64>,
    /// Offsets into targets/edge_ids (length = source_ids.len() + 1).
    pub offsets: Vec<u64>,
    /// Packed target node_ids (sorted per source for intersection).
    pub targets: Vec<u64>,
    /// Edge IDs corresponding to each target.
    pub edge_ids: Vec<u64>,
}

impl AdjacencyBatch {
    /// Get the target slice for the i-th source in the batch.
    pub fn targets_for(&self, batch_idx: usize) -> &[u64] {
        let start = self.offsets[batch_idx] as usize;
        let end = self.offsets[batch_idx + 1] as usize;
        &self.targets[start..end]
    }

    /// Get the edge_id slice for the i-th source in the batch.
    pub fn edge_ids_for(&self, batch_idx: usize) -> &[u64] {
        let start = self.offsets[batch_idx] as usize;
        let end = self.offsets[batch_idx + 1] as usize;
        &self.edge_ids[start..end]
    }

    /// Total number of adjacency entries across all sources.
    pub fn total_targets(&self) -> usize {
        self.targets.len()
    }

    /// Number of sources in the batch.
    pub fn num_sources(&self) -> usize {
        self.source_ids.len()
    }

    /// Intersect two adjacency batches — Kuzu's worst-case optimal join primitive.
    /// For each source pair where both have the same target, produce the intersection.
    ///
    /// A→B→C = intersect(adjacent(A), adjacent(C).reverse())
    /// Both target lists must be sorted (guaranteed by CSR construction).
    pub fn intersect(&self, other: &AdjacencyBatch) -> IntersectionResult {
        let mut matched_sources_left = Vec::new();
        let mut matched_sources_right = Vec::new();
        let mut matched_targets = Vec::new();

        // For each source in self, intersect its targets with each source in other
        for i in 0..self.num_sources() {
            let left_targets = self.targets_for(i);
            for j in 0..other.num_sources() {
                let right_targets = other.targets_for(j);
                // Sorted merge intersection
                let mut li = 0;
                let mut ri = 0;
                while li < left_targets.len() && ri < right_targets.len() {
                    match left_targets[li].cmp(&right_targets[ri]) {
                        std::cmp::Ordering::Less => li += 1,
                        std::cmp::Ordering::Greater => ri += 1,
                        std::cmp::Ordering::Equal => {
                            matched_sources_left.push(self.source_ids[i]);
                            matched_sources_right.push(other.source_ids[j]);
                            matched_targets.push(left_targets[li]);
                            li += 1;
                            ri += 1;
                        }
                    }
                }
            }
        }

        IntersectionResult {
            left_sources: matched_sources_left,
            right_sources: matched_sources_right,
            shared_targets: matched_targets,
        }
    }
}

/// Result of intersecting two adjacency batches.
#[derive(Debug, Clone)]
pub struct IntersectionResult {
    /// Source node from the left batch.
    pub left_sources: Vec<u64>,
    /// Source node from the right batch.
    pub right_sources: Vec<u64>,
    /// The shared target node (intersection).
    pub shared_targets: Vec<u64>,
}

impl IntersectionResult {
    pub fn len(&self) -> usize {
        self.shared_targets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shared_targets.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersection() {
        // Left batch: node 0 → [1, 2, 3], node 4 → [2, 5]
        let left = AdjacencyBatch {
            source_ids: vec![0, 4],
            offsets: vec![0, 3, 5],
            targets: vec![1, 2, 3, 2, 5],
            edge_ids: vec![0, 1, 2, 3, 4],
        };

        // Right batch: node 10 → [2, 3, 7]
        let right = AdjacencyBatch {
            source_ids: vec![10],
            offsets: vec![0, 3],
            targets: vec![2, 3, 7],
            edge_ids: vec![10, 11, 12],
        };

        let result = left.intersect(&right);

        // Node 0 shares targets 2, 3 with node 10
        // Node 4 shares target 2 with node 10
        assert_eq!(result.len(), 3);
        assert_eq!(result.shared_targets, vec![2, 3, 2]);
    }
}
