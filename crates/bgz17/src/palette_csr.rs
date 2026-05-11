//! PaletteCsr: archetype-based compressed scope representation.
//!
//! Instead of O(N²) pairwise distances between all nodes, assign each node
//! to one of k palette archetypes. Build a CLAM-style tree on the k archetypes
//! (O(k²) = O(256²) = 65K operations), then map search results back to nodes.
//!
//! The archetype assignment comes from the palette: each node's combined
//! S+P+O pattern is encoded to a palette index, and that index IS the archetype.
//!
//! Edge topology is extracted from container W16-31 inline edges, mapped
//! through palette assignments to produce archetype-level graph structure.

use crate::distance_matrix::SpoDistanceMatrices;
use crate::container::{CONTAINER_WORDS, W_INLINE_EDGES_START, W_INLINE_EDGES_END};
use crate::palette::PaletteEdge;
use crate::scope::Bgz17Scope;

/// Archetype-based compressed scope CSR.
#[derive(Clone, Debug)]
pub struct PaletteCsr {
    /// Per-plane distance matrices.
    pub distances: SpoDistanceMatrices,
    /// Node → archetype assignment (palette index of dominant plane or combined).
    pub assignments: Vec<u8>,
    /// Per-node actual palette indices (s_idx, p_idx, o_idx).
    pub palette_indices: Vec<PaletteEdge>,
    /// Archetype → list of member node indices.
    pub archetype_members: Vec<Vec<usize>>,
    /// Per-archetype edge topology: `[(verb_palette, target_palette)]`.
    pub edge_topology: Vec<Vec<(u8, u8)>>,
    /// Number of archetypes (= palette size k).
    pub k: usize,
}

/// A node in the archetype tree (simplified CLAM ball-tree node).
#[derive(Clone, Debug)]
pub struct ArchetypeNode {
    /// Archetype index (palette index).
    pub archetype: u8,
    /// Number of member nodes.
    pub member_count: usize,
    /// Radius: max distance from centroid to any member (in palette space).
    pub radius: u32,
    /// Left child archetype (if internal node).
    pub left: Option<Box<ArchetypeNode>>,
    /// Right child archetype (if internal node).
    pub right: Option<Box<ArchetypeNode>>,
}

/// Archetype tree for O(k²) search instead of O(N²).
#[derive(Clone, Debug)]
pub struct ArchetypeTree {
    pub root: Option<ArchetypeNode>,
    pub k: usize,
}

impl PaletteCsr {
    /// Build from a scope and its containers (for inline edge extraction).
    ///
    /// `containers`: the `[u64; 256]` BitVec for each node in the scope.
    /// Inline edges are read from W16-31 of each container.
    pub fn from_scope_with_edges(
        scope: &Bgz17Scope,
        containers: &[[u64; CONTAINER_WORDS]],
    ) -> Self {
        let n = scope.edge_count;
        let k = scope.palette_s.len().max(scope.palette_p.len()).max(scope.palette_o.len());

        // Assign each node to an archetype: use subject palette index as primary.
        let assignments: Vec<u8> = scope.palette_indices.iter()
            .map(|pe| pe.s_idx)
            .collect();

        // Build archetype membership lists
        let mut archetype_members = vec![Vec::new(); k];
        for (node_idx, &arch) in assignments.iter().enumerate() {
            if (arch as usize) < k {
                archetype_members[arch as usize].push(node_idx);
            }
        }

        // Extract edge topology from containers W16-31
        let mut edge_topology = vec![Vec::new(); k];
        for (node_idx, container) in containers.iter().enumerate().take(n) {
            let arch = assignments[node_idx];
            let edges = extract_inline_edges(container);
            for (verb, target) in edges {
                if (target as usize) < n {
                    let target_arch = assignments[target as usize];
                    edge_topology[arch as usize].push((verb, target_arch));
                }
            }
        }

        // Deduplicate edge topology per archetype
        for topo in &mut edge_topology {
            topo.sort();
            topo.dedup();
        }

        PaletteCsr {
            distances: scope.matrices.clone(),
            assignments,
            palette_indices: scope.palette_indices.clone(),
            archetype_members,
            edge_topology,
            k,
        }
    }

    /// Build an archetype tree for hierarchical search.
    ///
    /// Uses a simple bisecting approach: split archetypes by farthest pair,
    /// recurse on each half. O(k²) total.
    pub fn build_archetype_tree(&self) -> ArchetypeTree {
        let active: Vec<u8> = (0..self.k as u8)
            .filter(|&a| !self.archetype_members[a as usize].is_empty())
            .collect();

        let root = if active.is_empty() {
            None
        } else {
            Some(self.build_tree_node(&active))
        };

        ArchetypeTree { root, k: self.k }
    }

    fn build_tree_node(&self, archetypes: &[u8]) -> ArchetypeNode {
        if archetypes.len() <= 1 {
            let arch = archetypes[0];
            return ArchetypeNode {
                archetype: arch,
                member_count: self.archetype_members[arch as usize].len(),
                radius: 0,
                left: None,
                right: None,
            };
        }

        // Find farthest pair for splitting
        let (pivot_a, pivot_b) = self.farthest_pair(archetypes);

        // Partition: each archetype goes to nearer pivot
        let mut left_set = Vec::new();
        let mut right_set = Vec::new();
        for &arch in archetypes {
            let da = self.distances.subject.distance(arch, pivot_a) as u32;
            let db = self.distances.subject.distance(arch, pivot_b) as u32;
            if da <= db {
                left_set.push(arch);
            } else {
                right_set.push(arch);
            }
        }

        // Ensure non-empty partitions
        if left_set.is_empty() {
            left_set.push(right_set.pop().unwrap());
        }
        if right_set.is_empty() {
            right_set.push(left_set.pop().unwrap());
        }

        let left = self.build_tree_node(&left_set);
        let right = self.build_tree_node(&right_set);

        let member_count: usize = archetypes.iter()
            .map(|&a| self.archetype_members[a as usize].len())
            .sum();

        // Radius: max distance between any two archetypes in this subtree
        let radius = self.max_radius(archetypes);

        ArchetypeNode {
            archetype: archetypes[0], // representative
            member_count,
            radius,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        }
    }

    fn farthest_pair(&self, archetypes: &[u8]) -> (u8, u8) {
        let mut best_dist = 0u32;
        let mut best = (archetypes[0], archetypes[0]);
        for i in 0..archetypes.len() {
            for j in (i + 1)..archetypes.len() {
                let d = self.distances.subject.distance(archetypes[i], archetypes[j]) as u32;
                if d > best_dist {
                    best_dist = d;
                    best = (archetypes[i], archetypes[j]);
                }
            }
        }
        best
    }

    fn max_radius(&self, archetypes: &[u8]) -> u32 {
        let mut max_d = 0u32;
        for i in 0..archetypes.len() {
            for j in (i + 1)..archetypes.len() {
                let d = self.distances.subject.distance(archetypes[i], archetypes[j]) as u32;
                if d > max_d { max_d = d; }
            }
        }
        max_d
    }

    /// Search: find top-k nearest nodes to a query.
    ///
    /// Uses archetype-level distances to prune, then refines within archetypes.
    pub fn search(
        &self,
        query_pe: &PaletteEdge,
        k_results: usize,
    ) -> Vec<(usize, u32)> {
        // Score each archetype
        let mut arch_scores: Vec<(u8, u32)> = (0..self.k as u8)
            .filter(|&a| !self.archetype_members[a as usize].is_empty())
            .map(|a| {
                let d = self.distances.spo_distance(
                    query_pe.s_idx, query_pe.p_idx, query_pe.o_idx,
                    a, a, a,  // archetype centroid (simplified: same index for all planes)
                );
                (a, d)
            })
            .collect();
        arch_scores.sort_by_key(|&(_, d)| d);

        // Collect candidates from closest archetypes
        let mut results: Vec<(usize, u32)> = Vec::new();
        for &(arch, _) in &arch_scores {
            for &node_idx in &self.archetype_members[arch as usize] {
                // Use the node's actual palette indices (s, p, o) for distance
                let node_pe = &self.palette_indices[node_idx];
                let d = self.distances.spo_distance(
                    query_pe.s_idx, query_pe.p_idx, query_pe.o_idx,
                    node_pe.s_idx, node_pe.p_idx, node_pe.o_idx,
                );
                results.push((node_idx, d));
            }
            if results.len() >= k_results * 4 {
                break; // enough candidates
            }
        }

        results.sort_by_key(|&(_, d)| d);
        results.truncate(k_results);
        results
    }
}

/// Extract inline edges from container W16-31.
///
/// Each word packs 4 edges: verb(8) + target(8) per edge.
/// Returns (verb, target) pairs, stopping at first zero edge.
fn extract_inline_edges(container: &[u64; CONTAINER_WORDS]) -> Vec<(u8, u8)> {
    let mut edges = Vec::new();
    for &word in &container[W_INLINE_EDGES_START..=W_INLINE_EDGES_END] {
        if word == 0 { break; }
        for shift in (0..64).step_by(16) {
            let verb = ((word >> shift) & 0xFF) as u8;
            let target = ((word >> (shift + 8)) & 0xFF) as u8;
            if verb == 0 && target == 0 { return edges; }
            edges.push((verb, target));
        }
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{CONTAINER_WORDS, InlineEdge};
    use crate::scope::Bgz17Scope;

    fn random_plane(seed: u64) -> Vec<i8> {
        let mut v = vec![0i8; 16384];
        let mut s = seed;
        for x in v.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *x = (s >> 33) as i8;
        }
        v
    }

    fn make_containers_with_edges(n: usize) -> Vec<[u64; CONTAINER_WORDS]> {
        let mut containers = vec![[0u64; CONTAINER_WORDS]; n];
        // Pack some synthetic inline edges into W16-31
        for (i, container) in containers.iter_mut().enumerate() {
            // Each node has 2-3 edges to nearby nodes
            let e1 = InlineEdge { verb: 1, target: ((i + 1) % n) as u8 };
            let e2 = InlineEdge { verb: 2, target: ((i + 2) % n) as u8 };
            let quad = [e1, e2, InlineEdge::default(), InlineEdge::default()];
            container[W_INLINE_EDGES_START] = InlineEdge::pack4(&quad);
        }
        containers
    }

    #[test]
    fn test_extract_inline_edges() {
        let mut container = [0u64; CONTAINER_WORDS];
        let e1 = InlineEdge { verb: 5, target: 10 };
        let e2 = InlineEdge { verb: 7, target: 20 };
        let quad = [e1, e2, InlineEdge::default(), InlineEdge::default()];
        container[W_INLINE_EDGES_START] = InlineEdge::pack4(&quad);

        let edges = extract_inline_edges(&container);
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0], (5, 10));
        assert_eq!(edges[1], (7, 20));
    }

    #[test]
    fn test_from_scope_with_edges() {
        let n: usize = 20;
        let planes: Vec<(Vec<i8>, Vec<i8>, Vec<i8>)> = (0..n)
            .map(|i| {
                let s = i as u64;
                (random_plane(s * 3), random_plane(s * 3 + 1), random_plane(s * 3 + 2))
            })
            .collect();

        let scope = Bgz17Scope::build(1, &planes, 8);
        let containers = make_containers_with_edges(n);
        let pcsr = PaletteCsr::from_scope_with_edges(&scope, &containers);

        assert_eq!(pcsr.assignments.len(), n);
        assert_eq!(pcsr.k, 8);

        // Each node should be assigned to an archetype
        for &a in &pcsr.assignments {
            assert!((a as usize) < pcsr.k);
        }

        // At least some archetypes should have members
        let non_empty: usize = pcsr.archetype_members.iter()
            .filter(|m| !m.is_empty())
            .count();
        assert!(non_empty > 0);
    }

    #[test]
    fn test_build_archetype_tree() {
        let n: usize = 30;
        let planes: Vec<(Vec<i8>, Vec<i8>, Vec<i8>)> = (0..n)
            .map(|i| {
                let s = i as u64;
                (random_plane(s * 3 + 100), random_plane(s * 3 + 101), random_plane(s * 3 + 102))
            })
            .collect();

        let scope = Bgz17Scope::build(2, &planes, 8);
        let containers = make_containers_with_edges(n);
        let pcsr = PaletteCsr::from_scope_with_edges(&scope, &containers);
        let tree = pcsr.build_archetype_tree();

        assert_eq!(tree.k, 8);
        assert!(tree.root.is_some());
    }

    #[test]
    fn test_search_returns_ordered() {
        let n: usize = 30;
        let planes: Vec<(Vec<i8>, Vec<i8>, Vec<i8>)> = (0..n)
            .map(|i| {
                let s = i as u64;
                (random_plane(s * 3 + 200), random_plane(s * 3 + 201), random_plane(s * 3 + 202))
            })
            .collect();

        let scope = Bgz17Scope::build(3, &planes, 8);
        let containers = make_containers_with_edges(n);
        let pcsr = PaletteCsr::from_scope_with_edges(&scope, &containers);

        let query = PaletteEdge { s_idx: 0, p_idx: 0, o_idx: 0 };
        let results = pcsr.search(&query, 5);

        // Results should be sorted by distance
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }
}
