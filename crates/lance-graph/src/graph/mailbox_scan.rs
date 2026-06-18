// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! MailboxSoA scan — `Backend::MailboxSoa` (`cypher-kanban-ast-unification-v1` Inc 0).
//!
//! A Cypher `MATCH` routed over the canonical GUID-keyed substrate via the
//! zero-dep [`MailboxSoaView`] contract, instead of the index-built
//! [`TypedGraph`](crate::graph::blasgraph::typed_graph::TypedGraph) the other
//! backends use. The thesis (`E-GUID-IS-THE-GRAPH`): the substrate **is** the
//! graph — a node is its GUID key, and `MATCH (n:Label)` is a **classid
//! prefix-route**, resolved off the key/class column with **zero value decode**
//! (it never touches the 480 B value slab: `energy` / `meta` / fingerprints).
//!
//! ## Scope of this increment (the verified-safe half)
//!
//! This lands the **node-match** half — `MATCH (n:Label)` → the set of rows whose
//! class discriminator equals the queried class. That is the half that is correct
//! *without* the boundary the 5+3 council said to pin first.
//!
//! The **edge-traversal** half (`(a)-[r]->(b)`) is deliberately **deferred**, for
//! two grounded reasons (verdict §4b):
//!
//! 1. **Edge-representation is not yet pinned.** `EdgeBlock` (12+4 one-byte
//!    *adjacency* slots → neighbor `local_key`) and `CausalEdge64` (an **SPO
//!    triple** of s/p/o palette indices, the `edges_raw` column) are NOT
//!    interchangeable. A relationship-type must bind to one via the class's
//!    `EdgeCodecFlavor` — the router must not guess by availability.
//! 2. **The View exposes only `edges_raw` (`CausalEdge64`/SPO), not `EdgeBlock`
//!    adjacency.** `CausalEdge64` carries s/p/o palette indices, not a row→row
//!    pointer, so it cannot be dereferenced to a neighbor row without the
//!    adjacency accessor (a follow-on contract addition).
//!
//! So this module does the classid prefix-route and the `local_key`→row point
//! lookup (via [`MailboxSoaView::row_for_local_key`]); edge dispatch lands once
//! the representation boundary is resolved.

use lance_graph_contract::canonical_node::EdgeCodecFlavor;
use lance_graph_contract::hhtl::NiblePath;
use lance_graph_contract::soa_view::MailboxSoaView;

use crate::graph::graph_router::Backend;

/// A node matched by a MailboxSoA scan: the row index plus the backend tag.
///
/// Distinct from `GraphHit` (an *edge* with source/target) — a node match has no
/// target until the edge-traversal half lands. Kept minimal and honest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeMatch {
    /// The matched row index within the mailbox.
    pub row: usize,
    /// Always [`Backend::MailboxSoa`] — names the route that produced this match.
    pub backend: Backend,
}

/// `MATCH (n:Label)` → the rows whose class discriminator equals `class_id`.
///
/// The classid prefix-route over the `MailboxSoaView`. **Zero value decode by
/// construction:** the only column read is `class_id()` (which aliases the
/// `entity_type` u16 slot — the Cognitive-RISC N1 class hook); the 480 B value
/// slab (`energy` / `meta` / fingerprints) is never touched. This is the
/// substrate-is-the-graph node-selection half of `Backend::MailboxSoa`.
pub fn match_nodes_by_class<V: MailboxSoaView>(view: &V, class_id: u16) -> Vec<NodeMatch> {
    let classes = view.class_id();
    classes
        .iter()
        .enumerate()
        .filter_map(|(row, &c)| {
            (c == class_id).then_some(NodeMatch {
                row,
                backend: Backend::MailboxSoa,
            })
        })
        .collect()
}

/// Point lookup: resolve a canonical [`NodeGuid::local_key`] to a single row,
/// the GUID-keyed address half of `Backend::MailboxSoa`.
///
/// [`NodeGuid::local_key`]: lance_graph_contract::canonical_node::NodeGuid::local_key
///
/// Returns `None` when the view has not materialized a key index (the
/// deferred-binding default of [`MailboxSoaView::row_for_local_key`]) — the
/// caller then falls back to the positional `(mailbox_id, row)` address, never a
/// wrong row.
pub fn match_node_by_local_key<V: MailboxSoaView>(view: &V, local_key: u64) -> Option<NodeMatch> {
    view.row_for_local_key(local_key).map(|row| NodeMatch {
        row,
        backend: Backend::MailboxSoa,
    })
}

/// **CLAM containment** — the rows in `query`'s subtree: every row whose HHTL
/// path is a descendant-or-equal of `query` (`query.is_ancestor_of(path)`).
///
/// This is the `panCAKES ≡ radix trie ≡ HHTL` neighborhood (`E-CLAM-IS-THE-MANIFOLD-ENGINE`
/// / `E-PANCAKES-IS-RADIX-IS-HHTL`): the CLAM cluster is the radix-trie subtree
/// under the query prefix. Pure key arithmetic — **zero value decode**. Rows with
/// no materialized HHTL path (`hhtl_path_at == None`) are skipped.
pub fn clam_contained<V: MailboxSoaView>(view: &V, query: NiblePath) -> Vec<NodeMatch> {
    (0..view.n_rows())
        .filter(|&row| view.hhtl_path_at(row).is_some_and(|p| query.is_ancestor_of(p)))
        .map(|row| NodeMatch {
            row,
            backend: Backend::MailboxSoa,
        })
        .collect()
}

/// **CAKES nearest** — the `k` rows nearest `query` by longest-common-prefix
/// depth (descending), the radix-trie nearest-neighbor over the HHTL paths.
///
/// Returns `(NodeMatch, shared_depth)`; deeper shared prefix ⇒ nearer (same deeper
/// CLAM cluster). Ties keep ascending row order (stable). Pure key arithmetic —
/// **zero value decode**; rows without a materialized HHTL path are skipped. This
/// is CAKES "attraction" expressed as `NiblePath::common_prefix_depth`
/// (`E-CLAM-IS-THE-MANIFOLD-ENGINE`).
pub fn cakes_nearest<V: MailboxSoaView>(
    view: &V,
    query: NiblePath,
    k: usize,
) -> Vec<(NodeMatch, u8)> {
    let mut scored: Vec<(NodeMatch, u8)> = (0..view.n_rows())
        .filter_map(|row| {
            view.hhtl_path_at(row).map(|p| {
                (
                    NodeMatch {
                        row,
                        backend: Backend::MailboxSoa,
                    },
                    query.common_prefix_depth(p),
                )
            })
        })
        .collect();
    // Descending by shared depth; stable sort preserves ascending row order on ties.
    scored.sort_by_key(|&(_, depth)| core::cmp::Reverse(depth));
    scored.truncate(k);
    scored
}

/// The explicit typed edges of a node under the `CoarseOnly` flavor: the
/// **populated** (non-zero) slot references, split family-internal vs external.
///
/// Canon: an `EdgeBlock` slot is "zeroed when unused", so a zero byte is an empty
/// slot and a non-zero byte is a basin-local edge reference. The 12 `in_family`
/// slots are intra-basin edges; the 4 `external` slots are cross-family interface
/// references (`E-ADJACENCY-IS-KEY-AND-EDGECODEC`). The refs are returned raw —
/// resolving a ref → neighbor row needs the basin-local-index→row convention
/// (the next encoding decision, analogous to `row_for_local_key`); this facet
/// lands the *structure* (which slots are edges, family vs external), not the
/// row-resolution, and never fakes it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EdgeNeighbors {
    /// Populated in-family (intra-basin) edge slot references (non-zero bytes).
    pub in_family: Vec<u8>,
    /// Populated out-of-family (cross-basin interface) edge slot references.
    pub external: Vec<u8>,
}

/// Decode a node's explicit typed edges under the **`CoarseOnly`** flavor —
/// `(a)-[r]->…` as the populated 12-family/4-external slot references.
///
/// Returns `None` when the view has no edge block for `row`
/// ([`MailboxSoaView::edge_block_at`] default), or when the class's `flavor` is
/// not `CoarseOnly` (the adjacency reading) — `Pq32x4` is **turbovec residue**,
/// not adjacency, and `CoarseResidue` is coarse+residue; both are a different
/// read handled elsewhere, never coerced into slot adjacency
/// (`E-ADJACENCY-IS-KEY-AND-EDGECODEC` boundary §4b: classid-resolved, not
/// query-guessed). **Zero value decode** — the `EdgeBlock` is bytes 16..32, the
/// edge region, never the 480 B value slab.
pub fn edge_slots_coarse<V: MailboxSoaView>(
    view: &V,
    row: usize,
    flavor: EdgeCodecFlavor,
) -> Option<EdgeNeighbors> {
    if flavor != EdgeCodecFlavor::CoarseOnly {
        return None;
    }
    let block = view.edge_block_at(row)?;
    Some(EdgeNeighbors {
        in_family: block.in_family.iter().copied().filter(|&b| b != 0).collect(),
        external: block.out_family.iter().copied().filter(|&b| b != 0).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::canonical_node::EdgeBlock;
    use lance_graph_contract::kanban::KanbanColumn;

    /// A minimal view over fixed columns. The value-side columns
    /// (`energy`/`meta`/fingerprints) PANIC on access so the zero-value-decode
    /// gate (F2) is proven structurally: if the scan ever touches them, the test
    /// fails loudly.
    struct GuardedSoa {
        class_ids: Vec<u16>,
        edges: Vec<u64>,
        keyed_rows: Vec<(u64, usize)>,
        paths: Vec<Option<NiblePath>>,
        blocks: Vec<Option<EdgeBlock>>,
    }

    impl MailboxSoaView for GuardedSoa {
        fn mailbox_id(&self) -> u32 {
            0
        }
        fn n_rows(&self) -> usize {
            self.class_ids.len()
        }
        fn w_slot(&self) -> u8 {
            0
        }
        fn current_cycle(&self) -> u32 {
            0
        }
        fn phase(&self) -> KanbanColumn {
            KanbanColumn::Planning
        }
        // ── value slab — must NEVER be touched by a classid route (F2 guard) ──
        fn energy(&self) -> &[f32] {
            panic!("F2 violated: classid node-match touched the energy value column");
        }
        fn edges_raw(&self) -> &[u64] {
            // edges are key/causal side, not value slab — allowed, but the
            // node-match half does not use them; returned for trait completeness.
            &self.edges
        }
        fn meta_raw(&self) -> &[u32] {
            // meta is value-slab adjacent; the node-match must not read it.
            panic!("F2 violated: classid node-match touched the meta value column");
        }
        fn entity_type(&self) -> &[u16] {
            // class_id() aliases entity_type — this IS the class hook, allowed.
            &self.class_ids
        }
        fn row_for_local_key(&self, local_key: u64) -> Option<usize> {
            self.keyed_rows
                .iter()
                .find(|(k, _)| *k == local_key)
                .map(|(_, r)| *r)
        }
        fn hhtl_path_at(&self, row: usize) -> Option<NiblePath> {
            self.paths.get(row).copied().flatten()
        }
        fn edge_block_at(&self, row: usize) -> Option<EdgeBlock> {
            self.blocks.get(row).copied().flatten()
        }
    }

    fn sample() -> GuardedSoa {
        GuardedSoa {
            // rows: 0=A(7) 1=B(9) 2=C(7) 3=D(9) 4=E(7)
            class_ids: vec![7, 9, 7, 9, 7],
            edges: vec![0; 5],
            keyed_rows: vec![(0xABCD, 3), (0x1234, 0)],
            // HHTL radix-trie paths (root basin 1):
            //   row0: 1·2·3   row1: 1·2·4   row2: 1·2   row3: 1·5   row4: 9 (other basin)
            paths: vec![
                Some(NiblePath::root(1).child(2).child(3)),
                Some(NiblePath::root(1).child(2).child(4)),
                Some(NiblePath::root(1).child(2)),
                Some(NiblePath::root(1).child(5)),
                Some(NiblePath::root(9)),
            ],
            // row0 has in-family edges to refs 2,5 and one external ref 1; rest empty.
            blocks: vec![
                Some(EdgeBlock {
                    in_family: [2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    out_family: [1, 0, 0, 0],
                }),
                Some(EdgeBlock::default()),
                None,
                None,
                None,
            ],
        }
    }

    #[test]
    fn match_nodes_by_class_routes_on_classid_only() {
        let soa = sample();
        let hits = match_nodes_by_class(&soa, 7);
        let rows: Vec<usize> = hits.iter().map(|h| h.row).collect();
        assert_eq!(rows, vec![0, 2, 4], "all class-7 rows, in order");
        assert!(hits.iter().all(|h| h.backend == Backend::MailboxSoa));
        // parity: the matched set equals the reference classid filter.
        let reference: Vec<usize> = soa
            .class_ids
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == 7)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(rows, reference);
    }

    #[test]
    fn match_nodes_by_class_empty_when_no_match() {
        let soa = sample();
        assert!(match_nodes_by_class(&soa, 42).is_empty());
    }

    #[test]
    fn match_node_by_local_key_resolves_via_key_index() {
        let soa = sample();
        assert_eq!(
            match_node_by_local_key(&soa, 0xABCD),
            Some(NodeMatch {
                row: 3,
                backend: Backend::MailboxSoa
            })
        );
        // unknown key → None (caller falls back to positional address).
        assert_eq!(match_node_by_local_key(&soa, 0xDEAD), None);
    }

    #[test]
    fn f2_zero_value_decode_the_scan_never_panics_on_value_columns() {
        // The GuardedSoa panics if energy()/meta_raw() are read. If this test
        // completes, the classid node-match + CLAM/CAKES touched ONLY the
        // class/HHTL key columns, never the value slab.
        let soa = sample();
        let _ = match_nodes_by_class(&soa, 7);
        let _ = match_node_by_local_key(&soa, 0x1234);
        let _ = clam_contained(&soa, NiblePath::root(1).child(2));
        let _ = cakes_nearest(&soa, NiblePath::root(1).child(2).child(3), 3);
        let _ = edge_slots_coarse(&soa, 0, EdgeCodecFlavor::CoarseOnly);
    }

    #[test]
    fn edge_slots_coarse_decodes_populated_family_and_external() {
        let soa = sample();
        let n = edge_slots_coarse(&soa, 0, EdgeCodecFlavor::CoarseOnly).unwrap();
        assert_eq!(n.in_family, vec![2, 5], "non-zero in-family slots only");
        assert_eq!(n.external, vec![1], "non-zero external slot");
        // an all-zero block ⇒ no edges (zeroed = unused).
        let empty = edge_slots_coarse(&soa, 1, EdgeCodecFlavor::CoarseOnly).unwrap();
        assert!(empty.in_family.is_empty() && empty.external.is_empty());
        // no edge block materialized ⇒ None (deferred-binding fallback).
        assert!(edge_slots_coarse(&soa, 2, EdgeCodecFlavor::CoarseOnly).is_none());
    }

    #[test]
    fn edge_slots_coarse_refuses_non_coarse_flavors() {
        // Pq32x4 = turbovec residue, NOT adjacency; CoarseResidue likewise.
        // The classid-resolved flavor gates the read — never coerced to slots.
        let soa = sample();
        assert!(edge_slots_coarse(&soa, 0, EdgeCodecFlavor::Pq32x4).is_none());
        assert!(edge_slots_coarse(&soa, 0, EdgeCodecFlavor::CoarseResidue).is_none());
    }

    #[test]
    fn clam_contained_is_the_radix_subtree() {
        // query = 1·2 ⇒ its CLAM cluster = the radix subtree under 1·2:
        // rows 0 (1·2·3), 1 (1·2·4), 2 (1·2 itself). NOT 3 (1·5) or 4 (other basin 9).
        let soa = sample();
        let rows: Vec<usize> = clam_contained(&soa, NiblePath::root(1).child(2))
            .iter()
            .map(|m| m.row)
            .collect();
        assert_eq!(rows, vec![0, 1, 2]);
        // a deeper query narrows the subtree to the exact leaf.
        let leaf: Vec<usize> = clam_contained(&soa, NiblePath::root(1).child(2).child(3))
            .iter()
            .map(|m| m.row)
            .collect();
        assert_eq!(leaf, vec![0]);
    }

    #[test]
    fn cakes_nearest_ranks_by_longest_common_prefix() {
        // query = 1·2·3 (row 0). Shared-prefix depths:
        //   row0 1·2·3 →3, row1 1·2·4 →2, row2 1·2 →2, row3 1·5 →1, row4 9 →0.
        let soa = sample();
        let near = cakes_nearest(&soa, NiblePath::root(1).child(2).child(3), 3);
        let ranked: Vec<(usize, u8)> = near.iter().map(|(m, d)| (m.row, *d)).collect();
        assert_eq!(ranked, vec![(0, 3), (1, 2), (2, 2)], "nearest-3 by shared depth");
        assert!(near.iter().all(|(m, _)| m.backend == Backend::MailboxSoa));
        // k larger than n returns all rows, still depth-sorted descending.
        let all = cakes_nearest(&soa, NiblePath::root(1).child(2).child(3), 99);
        let depths: Vec<u8> = all.iter().map(|(_, d)| *d).collect();
        assert_eq!(depths, vec![3, 2, 2, 1, 0]);
    }

    #[test]
    fn clam_cakes_skip_rows_with_no_materialized_path() {
        // A view with all-None hhtl paths (the deferred-binding default) yields
        // nothing — the consumer falls back to a coarser facet, never a wrong row.
        let mut soa = sample();
        soa.paths = vec![None; 5];
        assert!(clam_contained(&soa, NiblePath::root(1)).is_empty());
        assert!(cakes_nearest(&soa, NiblePath::root(1), 5).is_empty());
    }
}
