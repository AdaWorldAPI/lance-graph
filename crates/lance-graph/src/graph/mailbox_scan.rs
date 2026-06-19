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
//! ## Facets landed (the dispatch table — all key-resident, zero value decode)
//!
//! The substrate IS the graph, so a query routes to the cheapest facet that
//! answers it, off the GUID key, never touching the 480 B value slab
//! (`energy` / `meta` / fingerprints — the F2 invariant):
//!
//! - **classid node-match** ([`match_nodes_by_class`]) — `MATCH (n:Label)` as a
//!   classid prefix-route; [`match_node_by_local_key`] for the `local_key`→row
//!   point lookup.
//! - **CLAM/CAKES neighborhood** ([`clam_contained`] / [`cakes_nearest`]) —
//!   `panCAKES ≡ radix trie ≡ HHTL`: the CLAM cluster tree IS the radix trie of
//!   the HHTL nibble paths in the keys, so containment = `is_ancestor_of` and
//!   nearest = `NiblePath::common_prefix_depth` (`E-PANCAKES-IS-RADIX-IS-HHTL`).
//! - **EdgeBlock typed edges** ([`edge_slots_coarse`]) — `(a)-[r]->…` under the
//!   classid-resolved [`EdgeCodecFlavor`]; `EdgeBlock` is bytes 16..32 (the edge
//!   region, not the value slab). `CoarseOnly` = 12-family/4-external slot
//!   structure; `Pq32x4` (turbovec residue) / `CoarseResidue` are refused, not
//!   coerced to adjacency (`E-ADJACENCY-IS-KEY-AND-EDGECODEC` §4b).
//!
//! ## Deliberately deferred (different cost tier / open encoding decision)
//!
//! - **EdgeBlock slot-byte → neighbor-row resolution** — needs the basin-local-
//!   index convention (zero = unused; 1-based vs basin-table), the next encoding
//!   decision, analogous to `row_for_local_key`. This module lands the edge
//!   *structure*, never fakes the row resolution.
//! - **Helix `Signed360` exact-location, CHAODA anomaly, CausalEdge64 SPO** —
//!   the costed tier: a value-slab decode (helix), the metric-space `ClamTree`
//!   (CHAODA), or cross-crate `causal-edge` accessors (SPO). They break the
//!   zero-value-decode invariant or need other-crate work, so they land
//!   separately with their own cost gates (`E-HELIX-IS-EXACT-LOCATION`,
//!   `E-CLAM-IS-THE-MANIFOLD-ENGINE`).

use lance_graph_contract::canonical_node::EdgeCodecFlavor;
use lance_graph_contract::hhtl::NiblePath;
use lance_graph_contract::soa_view::{IdentityPlane, MailboxSoaView};

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
///
/// **Clamped to `n_rows()`** — the real `MailboxSoA<N>` reports `n_rows() ==
/// populated` while `class_id()`/`entity_type()` borrow the full backing
/// capacity (zero-padded). Iterating the raw slice would surface phantom padding
/// rows (e.g. `MATCH` class 0 hitting the zeroed tail, or stale padding after a
/// logical shrink); the scan must stop at the logical row count.
pub fn match_nodes_by_class<V: MailboxSoaView>(view: &V, class_id: u16) -> Vec<NodeMatch> {
    let classes = view.class_id();
    classes
        .iter()
        .take(view.n_rows())
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
        .filter(|&row| {
            view.hhtl_path_at(row)
                .is_some_and(|p| query.is_ancestor_of(p))
        })
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
        in_family: block
            .in_family
            .iter()
            .copied()
            .filter(|&b| b != 0)
            .collect(),
        external: block
            .out_family
            .iter()
            .copied()
            .filter(|&b| b != 0)
            .collect(),
    })
}

/// The **means** by which an edge/node distance is measured over the GUID-keyed
/// substrate — "edge distance over different means, while the GUID is the node"
/// (`E-GUID-IS-THE-GRAPH`). One node IS its GUID; the separation between two
/// nodes is computed by a *selectable* metric, all anchored on the key. The
/// key-resident means are **zero value decode**; the value means (named, not yet
/// wired) read the 480 B value slab and are a separate cost tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMeans {
    /// **CLAM/HHTL radix tree-hop distance** — `(depth_a − cpd) + (depth_b − cpd)`
    /// where `cpd = common_prefix_depth(a, b)`: the steps up to the shared
    /// ancestor and back down. A genuine metric (`d(x,x)=0`, symmetric, tree
    /// triangle inequality). Key-only, zero value decode
    /// (`E-PANCAKES-IS-RADIX-IS-HHTL`).
    PrefixDepth,
    /// **Hamming over an identity plane** (`IdentityPlane`: content / topic /
    /// angle) — the popcount of the XOR of the two nodes' fingerprint planes.
    /// This is the **costed tier**: it reads the value-side plane
    /// (`MailboxSoaView::identity_plane_at`), so — unlike `PrefixDepth` — it is
    /// **NOT zero value decode** (`E-TENANT-ANGLE-RANK-IS-CAM-PQ-ADC`). The right
    /// use of popcount: homogeneous 16K-bit fingerprint bits, not the
    /// heterogeneous GUID key.
    Hamming(IdentityPlane),
    // ── further value means (named; wired as they land, same costed tier):
    //    HelixAngular — Signed360 exact-orthogonal-location distance
    //                   (`E-HELIX-IS-EXACT-LOCATION`);
    //    PqAdc        — CAM-PQ asymmetric distance (IVF probe + tables).
}

/// Distance between two nodes (rows, each a GUID) under the chosen `means`.
///
/// `PrefixDepth`: the radix tree-hop distance `(depth_a − cpd) + (depth_b − cpd)`
/// — `0` for the same leaf, growing as the two GUIDs' `NiblePath`s diverge nearer
/// the root (different basin = farthest). `None` if either row has no
/// materialized HHTL path (deferred-binding fallback). Key-only, **zero value
/// decode**.
///
/// The value-decode means (`DistanceMeans` doc) return `None` here until wired —
/// they land on the costed branch, never silently in the zero-decode path.
pub fn node_distance<V: MailboxSoaView>(
    view: &V,
    a: usize,
    b: usize,
    means: DistanceMeans,
) -> Option<u32> {
    match means {
        DistanceMeans::PrefixDepth => {
            let pa = view.hhtl_path_at(a)?;
            let pb = view.hhtl_path_at(b)?;
            let cpd = pa.common_prefix_depth(pb);
            Some(u32::from((pa.depth() - cpd) + (pb.depth() - cpd)))
        }
        // COSTED tier: reads the value-side plane (NOT zero value decode).
        DistanceMeans::Hamming(plane) => {
            let fa = view.identity_plane_at(a, plane)?;
            let fb = view.identity_plane_at(b, plane)?;
            Some(fa.iter().zip(fb).map(|(x, y)| (x ^ y).count_ones()).sum())
        }
    }
}

/// **`members`** (one-to-many) — the direct child nodes of a basin: the rows
/// exactly one HHTL tier deeper whose path the basin's path is an ancestor of.
///
/// This is the `basin-IS-a-node` navigation (`E-BASIN-IS-A-NODE`): a basin is a
/// node, its members are the next-tier-down rows in the radix trie. The tree is
/// **virtual** — no ownership, no SoA restructure — `members` is pure key
/// arithmetic (`is_ancestor_of` + a depth check), **zero value decode**. Inverse
/// of [`memberof`]: every `r` in `members(basin)` has `memberof(r) == basin`.
/// Rows with no materialized HHTL path are skipped. Returns empty if the basin
/// row itself has no path.
pub fn members<V: MailboxSoaView>(view: &V, basin_row: usize) -> Vec<NodeMatch> {
    let Some(bp) = view.hhtl_path_at(basin_row) else {
        return Vec::new();
    };
    let child_depth = bp.depth() + 1;
    (0..view.n_rows())
        .filter(|&row| {
            view.hhtl_path_at(row)
                .is_some_and(|p| p.depth() == child_depth && bp.is_ancestor_of(p))
        })
        .map(|row| NodeMatch {
            row,
            backend: Backend::MailboxSoa,
        })
        .collect()
}

/// The resolution of a [`memberof`] query: the parent basin is materialized in
/// this mailbox (`Local`), lives in another shard addressed by its HHTL prefix
/// (`Route`), or the node IS a top-tier basin with no parent (`Top`).
///
/// The GUID self-routes (`E-GUID-SELF-ROUTES-THE-BASIN-TREE`): the parent's HHTL
/// prefix **is** the shard/route key (`E-COARSE-QUANTIZER-IS-SCALE-FREE-ROUTER`
/// — the prefix is simultaneously the CLAM cluster key, the IVF cell, and the
/// shard key). So an unmaterialized parent is a **`Route`, not an absence** — no
/// separate coarse-fingerprint table is consulted; the prefix routes directly.
///
/// `Top` is the genuine "no parent" case (the DOLCE top facet). It is **distinct
/// from [`memberof`] returning `None`**, which means the node's own HHTL path is
/// not materialized (the deferred-binding default of
/// [`MailboxSoaView::hhtl_path_at`]) — i.e. *unresolved, fall back to a coarser
/// facet*, NOT *no parent*. Conflating the two would silently stop routing a row
/// whose path simply has not been bound yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BasinOf {
    /// The parent basin-node row, materialized in this mailbox.
    Local(NodeMatch),
    /// The parent lives in another shard; this is its HHTL prefix — route it to
    /// the shard that owns the prefix (the coarse router keys on exactly this).
    /// Zero value decode; the route key derives from the child's own GUID.
    Route(NiblePath),
    /// The node is a top-tier basin (`NiblePath::parent() == None`) — the DOLCE
    /// top facet, genuinely no parent. Distinct from `memberof` returning `None`
    /// (path not materialized; unresolved).
    Top,
}

/// **`memberof`** (many-to-one) — the basin a node belongs to: the parent path
/// (one HHTL tier shallower), via `NiblePath::parent`.
///
/// The many-to-one half of `basin-IS-a-node` (`E-BASIN-IS-A-NODE`), inverse of
/// [`members`]. Pure key arithmetic — **zero value decode**. The parent prefix
/// is the GUID's HHTL surfaced via [`MailboxSoaView::hhtl_path_at`] (the View
/// populates it through `NiblePath::from_guid_prefix`), so it is GUID-derived by
/// construction (`E-GUID-SELF-ROUTES-THE-BASIN-TREE`).
///
/// Returns:
/// - `Some(BasinOf::Local(row))` — the parent basin-node is in this mailbox;
/// - `Some(BasinOf::Route(prefix))` — the parent lives in another shard, addressed
///   by its HHTL prefix (route it; **never an absence for a node that has a parent**);
/// - `Some(BasinOf::Top)` — the node is a genuine top-tier basin: a **depth-1**
///   root (`NiblePath::root(0..16)`) whose `parent()` is `None`;
/// - `None` — **unresolved**: either the node's HHTL path is not materialized (the
///   deferred-binding default of [`MailboxSoaView::hhtl_path_at`]) OR it is the
///   **depth-0 `NiblePath::EMPTY` "no route" sentinel**. Both mean *fall back to a
///   coarser facet*, NOT "no parent". Kept distinct from `Top` so a yet-to-be-bound
///   row — or an explicit no-route sentinel — is not mistaken for a root and
///   silently dropped from routing.
pub fn memberof<V: MailboxSoaView>(view: &V, member_row: usize) -> Option<BasinOf> {
    // `None` here = path not materialized (deferred-binding) → unresolved.
    let path = view.hhtl_path_at(member_row)?;
    // The depth-0 `EMPTY` "no route" sentinel is ALSO unresolved — its `parent()`
    // is `None` like a real root, but it is not a top-tier basin (it has no basin
    // at all). Distinguish by depth before classifying `Top` (codex #550 P2).
    if path.depth() == 0 {
        return None;
    }
    // A real top-tier basin: a depth-1 root, path exists but has no parent.
    let Some(parent) = path.parent() else {
        return Some(BasinOf::Top);
    };
    Some(
        (0..view.n_rows())
            .find(|&row| view.hhtl_path_at(row) == Some(parent))
            .map_or(BasinOf::Route(parent), |row| {
                BasinOf::Local(NodeMatch {
                    row,
                    backend: Backend::MailboxSoa,
                })
            }),
    )
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
        content_planes: Vec<Option<Vec<u64>>>,
        /// Logical populated rows; `None` ⇒ the full `class_ids` length. Set
        /// smaller to model the real `MailboxSoA<N>` (zero-padded capacity with
        /// `n_rows() == populated < entity_type().len()`).
        logical_n: Option<usize>,
    }

    impl MailboxSoaView for GuardedSoa {
        fn mailbox_id(&self) -> u32 {
            0
        }
        fn n_rows(&self) -> usize {
            self.logical_n.unwrap_or(self.class_ids.len())
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
        fn identity_plane_at(&self, row: usize, plane: IdentityPlane) -> Option<&[u64]> {
            // Only the Content plane is materialized in this fake.
            match plane {
                IdentityPlane::Content => self.content_planes.get(row).and_then(|p| p.as_deref()),
                IdentityPlane::Topic | IdentityPlane::Angle => None,
            }
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
            // content planes (2 u64 each) for the Hamming means; row2 unmaterialized.
            content_planes: vec![
                Some(vec![0b1011, 0]),
                Some(vec![0b1101, 0]),
                None,
                Some(vec![0, 0]),
                Some(vec![u64::MAX, u64::MAX]),
            ],
            logical_n: None,
        }
    }

    #[test]
    fn match_nodes_by_class_clamps_to_n_rows_ignoring_padding() {
        // Model the real MailboxSoA<N>: class_id() borrows the full capacity
        // (zero-padded), but n_rows() reports only the populated prefix.
        let mut soa = sample();
        soa.class_ids = vec![7, 7, 0, 0, 0]; // 2 populated, 3 zero-padding
        soa.logical_n = Some(2);
        // class 7 ⇒ only the 2 populated rows, never the padding.
        let sevens: Vec<usize> = match_nodes_by_class(&soa, 7)
            .iter()
            .map(|m| m.row)
            .collect();
        assert_eq!(sevens, vec![0, 1]);
        // class 0 ⇒ EMPTY — the zeroed padding tail must NOT surface as phantom
        // matches (the codex P1 regression).
        assert!(
            match_nodes_by_class(&soa, 0).is_empty(),
            "padding rows beyond n_rows() must not match class 0"
        );
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
        assert_eq!(
            ranked,
            vec![(0, 3), (1, 2), (2, 2)],
            "nearest-3 by shared depth"
        );
        assert!(near.iter().all(|(m, _)| m.backend == Backend::MailboxSoa));
        // k larger than n returns all rows, still depth-sorted descending.
        let all = cakes_nearest(&soa, NiblePath::root(1).child(2).child(3), 99);
        let depths: Vec<u8> = all.iter().map(|(_, d)| *d).collect();
        assert_eq!(depths, vec![3, 2, 2, 1, 0]);
    }

    #[test]
    fn node_distance_prefix_depth_is_the_tree_hop_metric() {
        // rows: 0=1·2·3 (d3) 1=1·2·4 (d3) 2=1·2 (d2) 3=1·5 (d2) 4=9 (d1).
        let soa = sample();
        let d = |a, b| node_distance(&soa, a, b, DistanceMeans::PrefixDepth);
        // metric: d(x,x) = 0.
        assert_eq!(d(0, 0), Some(0));
        // 1·2·3 vs 1·2·4: cpd 2 ⇒ (3−2)+(3−2) = 2 (siblings).
        assert_eq!(d(0, 1), Some(2));
        // 1·2·3 vs 1·2 (its ancestor): cpd 2 ⇒ (3−2)+(2−2) = 1.
        assert_eq!(d(0, 2), Some(1));
        // 1·2·3 vs 1·5: cpd 1 ⇒ (3−1)+(2−1) = 3.
        assert_eq!(d(0, 3), Some(3));
        // 1·2·3 vs 9 (different basin): cpd 0 ⇒ (3−0)+(1−0) = 4 (farthest).
        assert_eq!(d(0, 4), Some(4));
        // symmetric + monotone.
        assert_eq!(d(0, 4), d(4, 0));
        assert!(d(0, 1).unwrap() < d(0, 4).unwrap());
    }

    #[test]
    fn node_distance_hamming_plane_is_popcount_xor_over_the_value_plane() {
        let soa = sample();
        // row0 0b1011, row1 0b1101 → XOR 0b0110 → popcount 2.
        assert_eq!(
            node_distance(&soa, 0, 1, DistanceMeans::Hamming(IdentityPlane::Content)),
            Some(2)
        );
        // self-distance = 0 (metric).
        assert_eq!(
            node_distance(&soa, 0, 0, DistanceMeans::Hamming(IdentityPlane::Content)),
            Some(0)
        );
        // all-zero vs all-ones over 2×u64 = 128 bits.
        assert_eq!(
            node_distance(&soa, 3, 4, DistanceMeans::Hamming(IdentityPlane::Content)),
            Some(128)
        );
        // unmaterialized plane (row2) ⇒ None (costed-tier fallback).
        assert_eq!(
            node_distance(&soa, 0, 2, DistanceMeans::Hamming(IdentityPlane::Content)),
            None
        );
        // a plane this fake doesn't carry ⇒ None (Topic/Angle not materialized).
        assert_eq!(
            node_distance(&soa, 0, 1, DistanceMeans::Hamming(IdentityPlane::Angle)),
            None
        );
    }

    #[test]
    fn node_distance_none_without_materialized_path() {
        let mut soa = sample();
        soa.paths[1] = None;
        assert_eq!(node_distance(&soa, 0, 1, DistanceMeans::PrefixDepth), None);
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

    #[test]
    fn members_are_the_direct_children_one_tier_down() {
        let soa = sample();
        // basin row2 = 1·2 ; direct children at depth 3 = row0 (1·2·3), row1 (1·2·4).
        let kids: Vec<usize> = members(&soa, 2).into_iter().map(|m| m.row).collect();
        assert_eq!(kids, vec![0, 1]);
        // row0 (1·2·3) is a leaf here — no depth-4 rows — so no members.
        assert!(members(&soa, 0).is_empty());
        // row4 (9) has no materialized children in this mailbox.
        assert!(members(&soa, 4).is_empty());
    }

    fn local_row(b: Option<BasinOf>) -> Option<usize> {
        match b {
            Some(BasinOf::Local(m)) => Some(m.row),
            _ => None,
        }
    }

    #[test]
    fn memberof_is_the_parent_basin_and_inverts_members() {
        let soa = sample();
        // row0/row1 (1·2·3, 1·2·4) belong to basin 1·2 = row2, materialized here.
        assert_eq!(local_row(memberof(&soa, 0)), Some(2));
        assert_eq!(local_row(memberof(&soa, 1)), Some(2));
        // Inverse property: every member's memberof is the basin.
        for m in members(&soa, 2) {
            assert_eq!(local_row(memberof(&soa, m.row)), Some(2));
        }
        // row4 (9) is a top-tier basin (depth 1) → parent() is None → Top,
        // NOT None (None is reserved for an unmaterialized path).
        assert_eq!(memberof(&soa, 4), Some(BasinOf::Top));
    }

    #[test]
    fn memberof_routes_when_parent_lives_in_another_shard() {
        let soa = sample();
        // row2 (1·2) parent is basin 1, not materialized in this mailbox → Route,
        // NOT None: the HHTL prefix IS the shard key (E-GUID-SELF-ROUTES-THE-BASIN-TREE).
        assert_eq!(memberof(&soa, 2), Some(BasinOf::Route(NiblePath::root(1))));
    }

    #[test]
    fn memberof_unmaterialized_path_is_none_not_top() {
        // Codex #549 P2: a deferred-binding row (hhtl_path_at == None) must return
        // None (unresolved, fall back) — DISTINCT from a real top-tier basin,
        // which returns Some(Top). Conflating them silently stops routing a
        // not-yet-bound row.
        let mut soa = sample();
        soa.paths[0] = None; // row0's path not materialized
        assert_eq!(
            memberof(&soa, 0),
            None,
            "unmaterialized path ⇒ None (fall back)"
        );
        // row4 still a genuine top-tier basin ⇒ Some(Top), not None.
        assert_eq!(memberof(&soa, 4), Some(BasinOf::Top));
    }

    #[test]
    fn memberof_empty_sentinel_is_none_not_top() {
        // Codex #550 P2: NiblePath::EMPTY (depth 0, the "no route" sentinel) has
        // parent() == None like a real root, but it is NOT a top-tier basin — it
        // has no basin at all. It must read as None (unresolved), not Some(Top),
        // so the no-route fallback is preserved.
        let mut soa = sample();
        soa.paths[4] = Some(NiblePath::EMPTY); // depth-0 sentinel
        assert_eq!(
            memberof(&soa, 4),
            None,
            "EMPTY (depth 0) ⇒ None (unresolved), never Top"
        );
        // root(>=16) also yields EMPTY → same no-route classification.
        soa.paths[4] = Some(NiblePath::root(16));
        assert_eq!(memberof(&soa, 4), None);
    }

    #[test]
    fn members_memberof_are_key_only_no_value_decode() {
        // F2: navigating the virtual basin tree must never touch the value slab.
        let soa = sample();
        let _ = members(&soa, 2);
        let _ = memberof(&soa, 0);
    }
}
