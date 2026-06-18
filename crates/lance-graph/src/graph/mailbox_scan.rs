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

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::kanban::KanbanColumn;

    /// A minimal view over fixed columns. The value-side columns
    /// (`energy`/`meta`/fingerprints) PANIC on access so the zero-value-decode
    /// gate (F2) is proven structurally: if the scan ever touches them, the test
    /// fails loudly.
    struct GuardedSoa {
        class_ids: Vec<u16>,
        edges: Vec<u64>,
        keyed_rows: Vec<(u64, usize)>,
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
    }

    fn sample() -> GuardedSoa {
        GuardedSoa {
            // rows: 0=A(7) 1=B(9) 2=C(7) 3=D(9) 4=E(7)
            class_ids: vec![7, 9, 7, 9, 7],
            edges: vec![0; 5],
            keyed_rows: vec![(0xABCD, 3), (0x1234, 0)],
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
        // completes, the classid node-match touched ONLY the class column.
        let soa = sample();
        let _ = match_nodes_by_class(&soa, 7);
        let _ = match_node_by_local_key(&soa, 0x1234);
    }
}
