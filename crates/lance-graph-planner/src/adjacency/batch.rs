//! Batch adjacency operations — the vectorized traversal primitive.
//!
//! **Zero-copy by construction (2026-07-27).** `AdjacencyBatch` is a BORROWED
//! VIEW over an [`AdjacencyStore`], not an owned re-packing of it. The previous
//! shape copied three vectors (`source_ids.to_vec()` plus `extend_from_slice`
//! of `targets` and `edge_ids`) that the store already holds contiguously and
//! already lends out zero-copy via `adjacent()` / `edge_ids()` — a duplicate
//! representation of resident state, which the universal zero-copy rule
//! forbids (primer §11/§15).
//!
//! The owned struct was REPLACED, not supplemented: keeping both would retain
//! the alternate owned representation the rule is about. Per-source slices are
//! delegated to the store on access, so `targets_for(i)` returns exactly the
//! bytes the old concatenation would have held at that offset range.

use super::csr::AdjacencyStore;

/// Borrowed view of adjacency for a batch of sources (Arrow ListArray pattern,
/// without the packing step).
///
/// Holds no adjacency data of its own — every accessor reads through to the
/// store's resident CSR arrays.
// NOT `Clone`, NOT `Copy` — operator-ruled 2026-07-29: *"copies are forbidden,
// borrows are only for the same mailbox"*. Same ruling that stripped the derive
// from `lance_graph_contract::witness_fabric::WitnessLens` (`b3515ba`); this is
// the same shape — a struct whose every field is a borrow.
//
// A `Copy` borrow duplicates itself silently: it can be handed to a second
// holder, stored beside the first, and carried out of the compartment that owns
// the CSR arrays with no move, no diagnostic, and nothing in a review to point
// at. That is exactly the escape the mailbox rule closes. Without the derive the
// view must be passed by reference (every call site here already does), so its
// reach is bounded by the borrow it was built from; anything that wants the
// adjacency elsewhere takes the source ids as ADDRESSES and re-derives the view
// against the store there — never a second view of the same bytes.
#[derive(Debug)]
pub struct AdjacencyBatch<'a> {
    /// The store the view reads through.
    store: &'a AdjacencyStore,
    /// Source node IDs (one per batch entry) — borrowed from the caller.
    pub source_ids: &'a [u64],
}

impl<'a> AdjacencyBatch<'a> {
    /// Construct a view over `source_ids` against `store`. No allocation.
    #[inline]
    #[must_use]
    pub fn new(store: &'a AdjacencyStore, source_ids: &'a [u64]) -> Self {
        Self { store, source_ids }
    }

    /// Get the target slice for the i-th source in the batch.
    ///
    /// Zero-copy: borrows straight out of the store's CSR targets.
    #[inline]
    #[must_use]
    pub fn targets_for(&self, batch_idx: usize) -> &'a [u64] {
        self.store.adjacent(self.source_ids[batch_idx])
    }

    /// Get the edge_id slice for the i-th source in the batch.
    ///
    /// Zero-copy: borrows straight out of the store's CSR edge ids.
    #[inline]
    #[must_use]
    pub fn edge_ids_for(&self, batch_idx: usize) -> &'a [u64] {
        self.store.edge_ids(self.source_ids[batch_idx])
    }

    /// Total number of adjacency entries across all sources.
    ///
    /// Summed from out-degrees rather than read off a packed length — the
    /// packed vector no longer exists.
    #[must_use]
    pub fn total_targets(&self) -> usize {
        self.source_ids
            .iter()
            .map(|&src| self.store.out_degree(src) as usize)
            .sum()
    }

    /// Number of sources in the batch.
    #[inline]
    #[must_use]
    pub fn num_sources(&self) -> usize {
        self.source_ids.len()
    }

    /// Whether the batch has no sources.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.source_ids.is_empty()
    }

    /// Intersect two adjacency batches — Kuzu's worst-case optimal join primitive.
    /// For each source pair where both have the same target, produce the intersection.
    ///
    /// A→B→C = intersect(adjacent(A), adjacent(C).reverse())
    /// Both target lists must be sorted (guaranteed by CSR construction).
    ///
    /// The returned [`IntersectionResult`] IS owned — it is a derived join
    /// product (matched pairs that existed in neither input), not a re-packing
    /// of resident state.
    #[must_use]
    pub fn intersect(&self, other: &AdjacencyBatch<'_>) -> IntersectionResult {
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
    #[must_use]
    pub fn len(&self) -> usize {
        self.shared_targets.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shared_targets.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the same graph the previous hand-packed literals encoded:
    /// 0 → [1, 2, 3], 4 → [2, 5], 10 → [2, 3, 7].
    fn store() -> AdjacencyStore {
        AdjacencyStore::from_edges(
            "t".to_string(),
            11,
            &[
                (0, 1),
                (0, 2),
                (0, 3),
                (4, 2),
                (4, 5),
                (10, 2),
                (10, 3),
                (10, 7),
            ],
        )
    }

    #[test]
    fn test_intersection() {
        let s = store();
        let left = AdjacencyBatch::new(&s, &[0, 4]);
        let right = AdjacencyBatch::new(&s, &[10]);

        let result = left.intersect(&right);

        // Node 0 shares targets 2, 3 with node 10; node 4 shares target 2.
        assert_eq!(result.len(), 3);
        assert_eq!(result.shared_targets, vec![2, 3, 2]);
    }

    /// The view must report exactly what the store holds — this is the
    /// semantic-identity check that licensed replacing the owned packing.
    #[test]
    fn view_matches_store_slices() {
        let s = store();
        let b = AdjacencyBatch::new(&s, &[0, 4, 10]);

        assert_eq!(b.num_sources(), 3);
        assert_eq!(b.targets_for(0), s.adjacent(0));
        assert_eq!(b.targets_for(1), s.adjacent(4));
        assert_eq!(b.targets_for(2), s.adjacent(10));
        assert_eq!(b.edge_ids_for(0), s.edge_ids(0));
        // 3 + 2 + 3 — summed from out-degrees, not a packed length.
        assert_eq!(b.total_targets(), 8);
    }

    /// A repeated source must yield the same slice at both positions (the old
    /// packed form duplicated those bytes; the view must not diverge from it).
    #[test]
    fn repeated_source_yields_same_slice() {
        let s = store();
        let b = AdjacencyBatch::new(&s, &[0, 0]);
        assert_eq!(b.targets_for(0), b.targets_for(1));
        assert_eq!(b.total_targets(), 6);
    }

    #[test]
    fn empty_batch_is_empty() {
        let s = store();
        let b = AdjacencyBatch::new(&s, &[]);
        assert!(b.is_empty());
        assert_eq!(b.total_targets(), 0);
    }
}
