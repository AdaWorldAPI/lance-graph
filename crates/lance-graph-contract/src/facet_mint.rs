//! `facet_mint` — the deterministic `(part_of:is_a)` rank-minter.
//!
//! "Brick 2" of `.claude/knowledge/ast-as-partof-isa-address.md`: turn a finite
//! AST's two structural hierarchies — **`part_of`** (membership:
//! `namespace → class → member`) and **`is_a`** (typing: `root → base → kind`) —
//! into [`FacetCascade`] addresses, **exact and roundtrip-lossless** (deterministic
//! sibling ranking, *not* learned PQ centroids).
//!
//! The carrier is the SHIPPED [`crate::facet::FacetCascade`] (#613/#614): each of
//! its 6 tiers is an 8:8 tile whose `hi` byte is the `part_of` rank at that depth
//! and `lo` byte the `is_a` rank — so [`FacetCascade::hi_chain`] is the part_of
//! hierarchy and [`FacetCascade::lo_chain`] the is_a hierarchy, both
//! prefix-routable (siblings share a coarse prefix). This minter writes into those
//! existing tiers; it invents **no** new type and no new layout.
//!
//! **`I-VSA-IDENTITIES` clean:** it encodes *identity positions* (which sibling,
//! which level), never bundles content — no superposition, no PQ codes.
//!
//! **Bounds (the finite-AST contract).** One rank byte addresses ≤ [`MAX_FANOUT`]
//! (256) siblings per parent; [`MAX_TIERS`] (6) tiers address ≤ 6 levels of depth.
//! Beyond either, the mint returns a [`MintError`] (the doc's ref-escape /
//! escalation is future work) — **exact-or-error, never silent aliasing.**
//!
//! The `facet_classid` (row 0) is a *parameter*: the classid `(part_of:is_a)`
//! half-order is an orthogonal, still-open decision (the operator's Canon:Custom
//! correction), so the per-tier minter does not bake one in.

use crate::facet::{FacetCascade, FacetTier};
use std::collections::HashMap;

/// The number of addressable tiers — the [`FacetCascade`] depth.
pub const MAX_TIERS: usize = 6;
/// Max siblings under one parent that a single rank byte can address.
pub const MAX_FANOUT: usize = 256;

/// One AST node: a stable id plus its parent in each orthogonal hierarchy.
/// A `None` parent marks a root of that hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeDecl {
    /// Stable corpus id (the harvest's interned node id).
    pub id: u32,
    /// `part_of` parent (membership) — `None` for a membership root.
    pub part_of_parent: Option<u32>,
    /// `is_a` parent (typing) — `None` for a type root.
    pub is_a_parent: Option<u32>,
}

impl NodeDecl {
    /// A node that is a root of *both* hierarchies.
    #[inline]
    pub const fn root(id: u32) -> Self {
        Self {
            id,
            part_of_parent: None,
            is_a_parent: None,
        }
    }

    /// A node with explicit `part_of` / `is_a` parents.
    #[inline]
    pub const fn new(id: u32, part_of_parent: Option<u32>, is_a_parent: Option<u32>) -> Self {
        Self {
            id,
            part_of_parent,
            is_a_parent,
        }
    }
}

/// Why a mint could not be exact — the finite-AST bounds, surfaced as errors so a
/// caller never gets a *lossy* address (the iron-rule alternative to aliasing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MintError {
    /// Two [`NodeDecl`]s share an id.
    DuplicateId(u32),
    /// A child names a parent absent from the corpus.
    UnknownParent { id: u32, parent: u32 },
    /// A hierarchy path is deeper than [`MAX_TIERS`] — needs ref-escape (future).
    DepthOverflow { id: u32, depth: usize },
    /// A parent has more than [`MAX_FANOUT`] children — one rank byte cannot index them.
    FanoutOverflow { parent: Option<u32>, count: usize },
    /// A parent chain cycles (not a finite tree).
    Cycle { id: u32 },
}

/// Mint a deterministic [`FacetCascade`] address for every node, in input order.
///
/// Deterministic and **order-independent**: sibling ranks are assigned by sorted
/// id, so the same corpus mints the identical facets regardless of how the slice
/// is ordered. Exact + roundtrip-lossless for any corpus within the
/// [`MAX_TIERS`]/[`MAX_FANOUT`] bounds; otherwise a [`MintError`].
pub fn mint_facets(nodes: &[NodeDecl], facet_classid: u32) -> Result<Vec<FacetCascade>, MintError> {
    // 1. id → index, with duplicate detection.
    let mut index: HashMap<u32, usize> = HashMap::with_capacity(nodes.len());
    for (i, n) in nodes.iter().enumerate() {
        if index.insert(n.id, i).is_some() {
            return Err(MintError::DuplicateId(n.id));
        }
    }
    // 2. validate that every named parent is present.
    for n in nodes {
        for p in [n.part_of_parent, n.is_a_parent].into_iter().flatten() {
            if !index.contains_key(&p) {
                return Err(MintError::UnknownParent {
                    id: n.id,
                    parent: p,
                });
            }
        }
    }
    // 3. own sibling-rank per hierarchy (index within parent group, by sorted id).
    let po_rank = sibling_ranks(nodes, |n| n.part_of_parent)?;
    let ia_rank = sibling_ranks(nodes, |n| n.is_a_parent)?;
    // 4. per node: walk root→node in each hierarchy, pack the coarse→fine ranks
    //    into the 6 tiers (hi = part_of, lo = is_a).
    let mut out = Vec::with_capacity(nodes.len());
    for n in nodes {
        let po = path_chain(n.id, &index, nodes, &po_rank, |x| x.part_of_parent)?;
        let ia = path_chain(n.id, &index, nodes, &ia_rank, |x| x.is_a_parent)?;
        let mut tiers = [FacetTier::default(); MAX_TIERS];
        for (t, tier) in tiers.iter_mut().enumerate() {
            *tier = FacetTier {
                hi: po.get(t).copied().unwrap_or(0),
                lo: ia.get(t).copied().unwrap_or(0),
            };
        }
        out.push(FacetCascade {
            facet_classid,
            tiers,
        });
    }
    Ok(out)
}

/// Assign each node its 0-based sibling rank within its parent group (all roots of
/// the hierarchy share one group). Deterministic: each group is sorted by id.
/// Errors if any group exceeds [`MAX_FANOUT`].
fn sibling_ranks(
    nodes: &[NodeDecl],
    parent_of: impl Fn(&NodeDecl) -> Option<u32>,
) -> Result<Vec<u8>, MintError> {
    let mut groups: HashMap<Option<u32>, Vec<(u32, usize)>> = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        groups.entry(parent_of(n)).or_default().push((n.id, i));
    }
    let mut rank = vec![0u8; nodes.len()];
    for (parent, mut sibs) in groups {
        if sibs.len() > MAX_FANOUT {
            return Err(MintError::FanoutOverflow {
                parent,
                count: sibs.len(),
            });
        }
        sibs.sort_unstable_by_key(|&(id, _)| id);
        for (r, &(_, idx)) in sibs.iter().enumerate() {
            rank[idx] = r as u8;
        }
    }
    Ok(rank)
}

/// The coarse→fine chain of own-ranks along the root→node path (≤ [`MAX_TIERS`]).
fn path_chain(
    id: u32,
    index: &HashMap<u32, usize>,
    nodes: &[NodeDecl],
    own_rank: &[u8],
    parent_of: impl Fn(&NodeDecl) -> Option<u32>,
) -> Result<Vec<u8>, MintError> {
    let mut chain = Vec::with_capacity(MAX_TIERS);
    let mut cur = Some(id);
    let mut guard = 0usize;
    while let Some(c) = cur {
        if guard > nodes.len() {
            return Err(MintError::Cycle { id });
        }
        let i = index[&c];
        chain.push(own_rank[i]);
        cur = parent_of(&nodes[i]);
        guard += 1;
    }
    if chain.len() > MAX_TIERS {
        return Err(MintError::DepthOverflow {
            id,
            depth: chain.len(),
        });
    }
    chain.reverse(); // root (coarse) → node (fine)
    Ok(chain)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A realistic AST slice where the two hierarchies genuinely CROSS — a member's
    // part_of class is independent of its is_a column type, so orthogonality is
    // demonstrable (not an accident of a parallel tree):
    //   1 = Models       (namespace)     part_of root,    is_a root
    //   2 = DbBase       (entity base)   part_of root,    is_a root
    //   3 = Patient (cls) part_of Models, is_a DbBase
    //   4 = Doctor  (cls) part_of Models, is_a DbBase
    //   5 = IntCol  (type)part_of root,   is_a root
    //   6 = StrCol  (type)part_of root,   is_a root
    //   7 = Patient.id    part_of Patient, is_a IntCol
    //   8 = Patient.name  part_of Patient, is_a StrCol   ← part_of-sibling of 7, is_a-distinct
    //   9 = Doctor.id     part_of Doctor,  is_a IntCol   ← is_a-sibling of 7, part_of-distinct
    fn corpus() -> Vec<NodeDecl> {
        vec![
            NodeDecl::root(1),
            NodeDecl::root(2),
            NodeDecl::new(3, Some(1), Some(2)),
            NodeDecl::new(4, Some(1), Some(2)),
            NodeDecl::root(5),
            NodeDecl::root(6),
            NodeDecl::new(7, Some(3), Some(5)),
            NodeDecl::new(8, Some(3), Some(6)),
            NodeDecl::new(9, Some(4), Some(5)),
        ]
    }

    /// The minted facet for a node id (mint preserves input order, so this is
    /// robust to how the corpus slice is ordered).
    fn facet_of(nodes: &[NodeDecl], facets: &[FacetCascade], id: u32) -> FacetCascade {
        let i = nodes.iter().position(|n| n.id == id).expect("id present");
        facets[i]
    }

    #[test]
    fn mint_is_deterministic_and_order_independent() {
        let nodes = corpus();
        let a = mint_facets(&nodes, 0x1000_0700).unwrap();
        let b = mint_facets(&nodes, 0x1000_0700).unwrap();
        assert_eq!(a, b, "same corpus mints identical facets");

        // Reversing the input order must not change ANY node's address.
        let mut shuffled = corpus();
        shuffled.reverse();
        let c = mint_facets(&shuffled, 0x1000_0700).unwrap();
        for n in &nodes {
            assert_eq!(
                facet_of(&nodes, &a, n.id),
                facet_of(&shuffled, &c, n.id),
                "node {} mints the same facet regardless of input order",
                n.id
            );
        }
    }

    #[test]
    fn part_of_and_is_a_chains_are_orthogonal() {
        let nodes = corpus();
        let f = mint_facets(&nodes, 0).unwrap();
        let p_id = facet_of(&nodes, &f, 7); // Patient.id
        let p_name = facet_of(&nodes, &f, 8); // Patient.name
        let d_id = facet_of(&nodes, &f, 9); // Doctor.id

        // part_of-siblings (both members of Patient): share namespace + class
        // tiers, diverge only at the member tier — and this holds INDEPENDENTLY of
        // their is_a types.
        assert_eq!(p_id.hi_chain()[0], p_name.hi_chain()[0], "same namespace");
        assert_eq!(
            p_id.hi_chain()[1],
            p_name.hi_chain()[1],
            "same class (Patient)"
        );
        assert_ne!(p_id.hi_chain()[2], p_name.hi_chain()[2], "distinct member");
        // …yet their is_a chains diverge at the very first tier (IntCol vs StrCol):
        assert_ne!(
            p_id.lo_chain()[0],
            p_name.lo_chain()[0],
            "different is_a type root"
        );

        // Cross the other way: Patient.id vs Doctor.id are part_of-DIFFERENT
        // (different class) but is_a-SAME (both IntCol). The two axes move
        // independently on the SAME facet.
        assert_eq!(p_id.hi_chain()[0], d_id.hi_chain()[0], "same namespace");
        assert_ne!(p_id.hi_chain()[1], d_id.hi_chain()[1], "different class");
        assert_eq!(
            p_id.lo_chain()[0],
            d_id.lo_chain()[0],
            "same is_a type root (IntCol)"
        );
        assert_ne!(
            p_id.lo_chain()[1],
            d_id.lo_chain()[1],
            "distinct instance under IntCol"
        );
    }

    #[test]
    fn part_of_siblings_share_the_coarse_prefix() {
        let nodes = corpus();
        let f = mint_facets(&nodes, 0).unwrap();
        let p_id = facet_of(&nodes, &f, 7); // Patient.id
        let p_name = facet_of(&nodes, &f, 8); // Patient.name (part_of-sibling)
        assert!(p_id.hi_distance(p_name) > 0, "they differ somewhere");
        assert_eq!(
            p_id.hi_chain()[..2],
            p_name.hi_chain()[..2],
            "namespace + class tiers shared"
        );
    }

    #[test]
    fn minted_facet_roundtrips_bytes_and_carries_the_classid() {
        for c in mint_facets(&corpus(), 0xDEAD_BEEF).unwrap() {
            assert_eq!(
                FacetCascade::from_bytes(&c.to_bytes()),
                c,
                "byte round-trip"
            );
            assert_eq!(c.facet_classid, 0xDEAD_BEEF, "row-0 classid preserved");
        }
    }

    #[test]
    fn depth_overflow_is_an_error_not_aliasing() {
        // A 7-deep part_of chain: 0←1←2←3←4←5←6 (depth 7 > MAX_TIERS).
        let mut v = vec![NodeDecl::root(0)];
        for i in 1..=6u32 {
            v.push(NodeDecl::new(i, Some(i - 1), None));
        }
        match mint_facets(&v, 0) {
            Err(MintError::DepthOverflow { id: 6, depth: 7 }) => {}
            other => panic!("expected DepthOverflow on the depth-7 node, got {other:?}"),
        }
        // The depth-6 node (id 5) is exactly at the limit and mints fine.
        let mut ok = vec![NodeDecl::root(0)];
        for i in 1..=5u32 {
            ok.push(NodeDecl::new(i, Some(i - 1), None));
        }
        assert!(mint_facets(&ok, 0).is_ok(), "depth 6 is within bounds");
    }

    #[test]
    fn fanout_overflow_is_an_error() {
        // 300 children under one parent — one rank byte cannot index > 256.
        let mut v = vec![NodeDecl::root(0)];
        for i in 1..=300u32 {
            v.push(NodeDecl::new(i, Some(0), Some(0)));
        }
        assert!(
            matches!(mint_facets(&v, 0), Err(MintError::FanoutOverflow { .. })),
            "300 siblings must overflow the one-byte rank"
        );
    }

    #[test]
    fn unknown_parent_and_duplicate_id_are_errors() {
        assert_eq!(
            mint_facets(&[NodeDecl::new(1, Some(99), None)], 0),
            Err(MintError::UnknownParent { id: 1, parent: 99 })
        );
        assert_eq!(
            mint_facets(&[NodeDecl::root(1), NodeDecl::root(1)], 0),
            Err(MintError::DuplicateId(1))
        );
    }

    #[test]
    fn empty_corpus_mints_empty() {
        assert_eq!(mint_facets(&[], 0).unwrap(), Vec::<FacetCascade>::new());
    }
}
