//! `soa_graph` — project the canonical SoA head into the Gotham graph surface.
//!
//! The bridge from the **canonical node head** (128-bit [`NodeGuid`] + 128-bit
//! [`EdgeBlock`], `key(16)+edges(16)`, bytes 0..32 of a [`NodeRow`]) to the
//! existing [`graph_render`](crate::graph_render) Neo4j/Palantir-Gotham surface
//! ([`GraphSnapshot`] / [`RenderNode`] / [`RenderEdge`]). **Zero value decode:**
//! every node, edge, family, and anchor here is read from the 32-byte head —
//! the 480-byte value slab is never touched (`E-GUID-IS-THE-GRAPH`; the same
//! falsifiable invariant `symbiont::key_render` proves by 0xFF-poisoning).
//!
//! **Rendering lives in q2.** This module produces the *structural* snapshot;
//! the q2 `cockpit-server` cockpit (vis-network / Neo4j-Browser-style UI) lays
//! it out and draws it. What lance-graph owns is "the basic domain + SoA as a
//! graph"; q2 owns the pixels.
//!
//! ## Two head axes, two graph roles
//!
//! The canonical key carries two orthogonal grouping axes, both in the head:
//!
//! - **family** (`u24`, bytes 10..13) — the *basin leaf*. [`project_snapshot`]
//!   groups member nodes by `family` and emits one **family node** per distinct
//!   family (the "use family nodes" requirement). A family node is an **anchor**
//!   when its id is in [`DomainSpec::anchor_families`] (FMA *bones* / OSINT *key
//!   entities* — the stability anchors layout hangs off).
//! - **HHTL path** (`classid_lo·HEEL·HIP·TWIG`, via
//!   [`NiblePath::from_guid_prefix`]) — the *Abstammung tree*. [`nearest_anchor`]
//!   ranks every node against the anchor families by
//!   [`NiblePath::family_hop_count`] (CLAM tree distance) — the "HHTL CLAM via
//!   family-nodes hop count as adjacency" metric.
//!
//! ## Edge resolution (the `EdgeBlock` reading)
//!
//! `EdgeCodecFlavor::CoarseOnly` (the read-mode both registered domains use):
//! each non-zero edge byte is a one-byte basin-local neighbour index.
//! - `in_family[k]` → the same-family member whose `identity & 0xFF` equals the
//!   byte (an intra-basin adjacency edge, [`DomainSpec::in_family_edge`]).
//! - `out_family[k]` → the family node whose `family & 0xFF` equals the byte (a
//!   cross-basin link to another family, [`DomainSpec::out_family_edge`]).
//!
//! Unresolved bytes are skipped (a dangling 1-byte index, never a wrong edge).
//!
//! Two domains ship registered: [`OSINT_GOTHAM`] (classid
//! [`NodeGuid::CLASSID_OSINT`]) and [`FMA_ANATOMY`] (classid
//! [`NodeGuid::CLASSID_FMA`]). New domains are just another `DomainSpec` —
//! the projector is domain-agnostic.

use crate::canonical_node::{NodeGuid, NodeRow};
use crate::graph_render::{GraphSnapshot, RenderEdge, RenderNode};
use crate::hhtl::NiblePath;
use std::collections::HashMap;

/// A graph domain: how a class of SoA nodes is labelled and which families are
/// stability anchors. Domain-agnostic data (no behaviour) — the projector reads
/// it. `&'static` so domains can be `const` (see [`OSINT_GOTHAM`], [`FMA_ANATOMY`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomainSpec {
    /// OGAR classid this domain occupies (the GUID routing prefix).
    pub classid: u32,
    /// Human name, used as the member node `kind` (e.g. "OSINT/Gotham").
    pub name: &'static str,
    /// Families that are **stability anchors** (FMA bones / OSINT key entities).
    /// Family nodes in this set render as `kind = "Anchor"` and are the targets
    /// [`nearest_anchor`] measures hop distance to.
    pub anchor_families: &'static [u32],
    /// Edge label for intra-family adjacency (`in_family` slots).
    pub in_family_edge: &'static str,
    /// Edge label for cross-family links (`out_family` slots).
    pub out_family_edge: &'static str,
    /// Edge label for the member → family-node containment edge.
    pub member_edge: &'static str,
}

/// The **OSINT / Palantir-Gotham** domain (classid [`NodeGuid::CLASSID_OSINT`]):
/// a neo4j-emulation entity graph. Anchor families are caller-supplied (the key
/// entities of an investigation); the default declares none.
pub const OSINT_GOTHAM: DomainSpec = DomainSpec {
    classid: NodeGuid::CLASSID_OSINT,
    name: "OSINT/Gotham",
    anchor_families: &[],
    in_family_edge: "linked",
    out_family_edge: "references",
    member_edge: "member-of",
};

/// The **FMA anatomy** domain (classid [`NodeGuid::CLASSID_FMA`]): ~70k
/// structural entities, family = body region, `out_family` = part-of. Anchor
/// families are the *bones* (the skeleton the soft tissue hangs off); the
/// default declares none — a caller supplies the bone families.
pub const FMA_ANATOMY: DomainSpec = DomainSpec {
    classid: NodeGuid::CLASSID_FMA,
    name: "FMA-Anatomy",
    anchor_families: &[],
    in_family_edge: "adjacent-to",
    out_family_edge: "part-of",
    member_edge: "part-of",
};

/// The synthetic id of a family node in the snapshot (`"family:RRGGBB"` hex).
#[inline]
fn family_node_id(family: u32) -> String {
    format!("family:{family:06x}")
}

/// HHTL routing path of a GUID, via the canonical [`NiblePath::from_guid_prefix`]
/// lowering (`classid_lo·HEEL·HIP·TWIG`). Falls back to [`NiblePath::EMPTY`] for
/// the (canon-reserved) case of a non-zero high `classid` u16.
#[inline]
fn hhtl_path(guid: &NodeGuid) -> NiblePath {
    NiblePath::from_guid_prefix(guid).unwrap_or(NiblePath::EMPTY)
}

/// Project a board-set into a [`GraphSnapshot`] for the Gotham/neo4j surface —
/// member nodes + family nodes + (member→family, in-family, out-of-family)
/// edges. Touches ONLY the 32-byte head of each row (`key` + `edges`); never the
/// value slab.
pub fn project_snapshot(rows: &[NodeRow], domain: &DomainSpec) -> GraphSnapshot {
    // family → its members as (identity_low_byte, guid)
    let mut by_family: HashMap<u32, Vec<(u8, NodeGuid)>> = HashMap::new();
    // family_low_byte → a family id (first seen) for out-of-family resolution
    let mut family_by_low: HashMap<u8, u32> = HashMap::new();
    for row in rows {
        let g = row.key;
        let fam = g.family();
        by_family
            .entry(fam)
            .or_default()
            .push(((g.identity() & 0xFF) as u8, g));
        family_by_low.entry((fam & 0xFF) as u8).or_insert(fam);
    }

    let mut nodes: Vec<RenderNode> = Vec::with_capacity(rows.len() + by_family.len());
    let mut edges: Vec<RenderEdge> = Vec::new();

    // One family node per distinct family (the "use family nodes" surface).
    // Sorted for deterministic output regardless of HashMap iteration order.
    let mut families: Vec<(&u32, &Vec<(u8, NodeGuid)>)> = by_family.iter().collect();
    families.sort_by_key(|(fam, _)| **fam);
    for (&fam, members) in families {
        let is_anchor = domain.anchor_families.contains(&fam);
        nodes.push(RenderNode {
            id: family_node_id(fam),
            label: format!("{} family {fam:06x}", domain.name),
            kind: if is_anchor { "Anchor" } else { "Family" }.to_string(),
            confidence: 1.0,
            props: vec![
                ("family".to_string(), format!("{fam:06x}")),
                ("members".to_string(), members.len().to_string()),
                ("anchor".to_string(), is_anchor.to_string()),
            ],
        });
    }

    // Member nodes + their edges (all head-only).
    for row in rows {
        let g = row.key;
        let fam = g.family();
        nodes.push(RenderNode {
            id: g.to_string(),
            label: format!("{:06x}", g.identity()),
            kind: domain.name.to_string(),
            confidence: 1.0,
            props: vec![
                ("classid".to_string(), format!("{:08x}", g.classid())),
                ("family".to_string(), format!("{fam:06x}")),
                ("hhtl_depth".to_string(), hhtl_path(&g).depth().to_string()),
            ],
        });
        // member → family containment
        edges.push(RenderEdge {
            source: g.to_string(),
            target: family_node_id(fam),
            label: domain.member_edge.to_string(),
            frequency: 1.0,
            confidence: 1.0,
            inferred: false,
        });
        let eb = row.edges;
        // in-family adjacency: byte = same-family member's identity low byte
        if let Some(members) = by_family.get(&fam) {
            for &b in eb.in_family.iter().filter(|&&b| b != 0) {
                if let Some(&(_, target)) =
                    members.iter().find(|(lb, t)| *lb == b && *t != g)
                {
                    edges.push(RenderEdge {
                        source: g.to_string(),
                        target: target.to_string(),
                        label: domain.in_family_edge.to_string(),
                        frequency: 1.0,
                        confidence: 1.0,
                        inferred: false,
                    });
                }
            }
        }
        // out-of-family links: byte = target family's low byte → its family node
        for &b in eb.out_family.iter().filter(|&&b| b != 0) {
            if let Some(&target_fam) = family_by_low.get(&b) {
                if target_fam != fam {
                    edges.push(RenderEdge {
                        source: g.to_string(),
                        target: family_node_id(target_fam),
                        label: domain.out_family_edge.to_string(),
                        frequency: 1.0,
                        confidence: 1.0,
                        inferred: false,
                    });
                }
            }
        }
    }

    GraphSnapshot {
        nodes,
        edges,
        inferences: Vec::new(),
        contradictions: Vec::new(),
        timestamp: 0,
    }
}

/// A node's CLAM hop distance to its nearest stability anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnchorHop {
    /// The node measured.
    pub node: NodeGuid,
    /// The family id of the nearest anchor (`u32::MAX` if the domain declares
    /// none, or none is reachable).
    pub anchor_family: u32,
    /// HHTL CLAM hop count to that anchor's representative path (`u8::MAX` when
    /// no anchor exists).
    pub hops: u8,
}

/// For each node, the nearest stability-anchor family by **HHTL CLAM hop count**
/// ([`NiblePath::family_hop_count`] over the GUIDs' HHTL paths) — the "bones as
/// stability anchor" layout signal: each node hangs off its closest anchor, and
/// the hop count is the adjacency weight the q2 layout uses (anchors fixed, soft
/// tissue positioned by distance). Anchors are the families in
/// [`DomainSpec::anchor_families`]; their representative path is the first member
/// seen. Pure head arithmetic, zero value decode. O(rows × anchors).
///
/// The canonical lowering is fixed-depth-16, so `hops = 2·(16 − lcp)` (`lcp` =
/// shared-prefix nibble count) — a monotone prefix distance, not a variable-depth
/// tree walk: smaller hops ⇔ deeper shared `classid_lo·HEEL·HIP·TWIG` prefix.
/// Ranking (nearest anchor) is what callers use; the absolute value is even.
pub fn nearest_anchor(rows: &[NodeRow], domain: &DomainSpec) -> Vec<AnchorHop> {
    // Representative HHTL path per anchor family (first member encountered).
    let mut anchor_paths: Vec<(u32, NiblePath)> = Vec::new();
    for row in rows {
        let fam = row.key.family();
        if domain.anchor_families.contains(&fam)
            && !anchor_paths.iter().any(|(f, _)| *f == fam)
        {
            anchor_paths.push((fam, hhtl_path(&row.key)));
        }
    }
    rows.iter()
        .map(|row| {
            let g = row.key;
            let p = hhtl_path(&g);
            let mut anchor_family = u32::MAX;
            let mut hops = u8::MAX;
            for &(fam, ap) in &anchor_paths {
                let h = p.family_hop_count(ap);
                if h < hops {
                    hops = h;
                    anchor_family = fam;
                }
            }
            AnchorHop {
                node: g,
                anchor_family,
                hops,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical_node::EdgeBlock;

    /// Build a node in a domain: `classid` from the domain, hierarchy in the HHT
    /// tiers, family = basin leaf, identity = leaf. Edges optional.
    fn node(
        domain: &DomainSpec,
        heel: u16,
        hip: u16,
        twig: u16,
        family: u32,
        identity: u32,
        in_fam: &[u8],
        out_fam: &[u8],
    ) -> NodeRow {
        let mut edges = EdgeBlock::default();
        for (i, &b) in in_fam.iter().enumerate().take(12) {
            edges.in_family[i] = b;
        }
        for (i, &b) in out_fam.iter().enumerate().take(4) {
            edges.out_family[i] = b;
        }
        NodeRow {
            key: NodeGuid::new(domain.classid, heel, hip, twig, family, identity),
            edges,
            value: [0u8; 480],
        }
    }

    #[test]
    fn project_emits_family_nodes_and_member_edges() {
        // Two families (0xA, 0xB), two members each. Member 1 in family A points
        // at member 2 (identity low byte 2) via in_family; member 1 also links
        // out to family B (low byte 0xB) via out_family.
        let rows = [
            node(&OSINT_GOTHAM, 1, 0, 0, 0xA, 1, &[2], &[0xB]),
            node(&OSINT_GOTHAM, 1, 0, 0, 0xA, 2, &[], &[]),
            node(&OSINT_GOTHAM, 2, 0, 0, 0xB, 1, &[], &[]),
            node(&OSINT_GOTHAM, 2, 0, 0, 0xB, 2, &[], &[]),
        ];
        let snap = project_snapshot(&rows, &OSINT_GOTHAM);
        // 4 member nodes + 2 family nodes
        assert_eq!(snap.nodes.len(), 6);
        let family_nodes = snap.nodes.iter().filter(|n| n.kind == "Family").count();
        assert_eq!(family_nodes, 2);
        // every member has a member-of edge → 4 of them
        let member_of = snap.edges.iter().filter(|e| e.label == "member-of").count();
        assert_eq!(member_of, 4);
        // the in-family adjacency edge member1 → member2
        assert!(snap.edges.iter().any(|e| e.label == "linked"
            && e.target.ends_with("000a000002")));
        // the out-of-family link member1 → family:00000b
        assert!(snap
            .edges
            .iter()
            .any(|e| e.label == "references" && e.target == "family:00000b"));
    }

    #[test]
    fn anchor_families_render_as_anchor_kind() {
        // FMA: family 0x01 is a "bone" anchor; 0x02 is soft tissue.
        let fma_bones = DomainSpec {
            anchor_families: &[0x01],
            ..FMA_ANATOMY
        };
        let rows = [
            node(&fma_bones, 0x1, 0, 0, 0x01, 1, &[], &[]), // bone
            node(&fma_bones, 0x2, 0, 0, 0x02, 1, &[], &[]), // tissue
        ];
        let snap = project_snapshot(&rows, &fma_bones);
        let anchor = snap.nodes.iter().find(|n| n.id == "family:000001").unwrap();
        assert_eq!(anchor.kind, "Anchor");
        let tissue = snap.nodes.iter().find(|n| n.id == "family:000002").unwrap();
        assert_eq!(tissue.kind, "Family");
    }

    #[test]
    fn nearest_anchor_ranks_by_hhtl_hop_count() {
        // The canonical lowering is fixed-depth-16, so family_hop_count = 2·(16 −
        // lcp): the deeper the shared prefix, the fewer hops. Anchor family 0x01
        // sits at heel=0x1000. Same path ⇒ 0; a node differing in the last HEEL
        // nibble (lcp=7) ⇒ 18; a node differing in the first HEEL nibble (lcp=4)
        // ⇒ 24. What matters is the ordering (closer prefix ⇒ smaller hops).
        let fma_bones = DomainSpec {
            anchor_families: &[0x01],
            ..FMA_ANATOMY
        };
        let rows = [
            node(&fma_bones, 0x1000, 0, 0, 0x01, 1, &[], &[]), // the anchor itself
            node(&fma_bones, 0x1000, 0, 0, 0x02, 1, &[], &[]), // same HHT path
            node(&fma_bones, 0x1009, 0, 0, 0x03, 1, &[], &[]), // diverges late (lcp 7)
            node(&fma_bones, 0xF000, 0, 0, 0x04, 1, &[], &[]), // diverges early (lcp 4)
        ];
        let hops = nearest_anchor(&rows, &fma_bones);
        assert_eq!(hops.len(), 4);
        assert_eq!(hops[0].hops, 0);
        assert_eq!(hops[0].anchor_family, 0x01);
        assert_eq!(hops[1].hops, 0, "same HHT path as the anchor ⇒ 0 hops");
        assert!(
            hops[1].hops < hops[2].hops && hops[2].hops < hops[3].hops,
            "monotone: closer shared prefix ⇒ fewer hops ({} < {} < {})",
            hops[1].hops,
            hops[2].hops,
            hops[3].hops
        );
        // The exact fixed-depth-16 values: 2·(16−7)=18 and 2·(16−4)=24.
        assert_eq!(hops[2].hops, 18);
        assert_eq!(hops[3].hops, 24);
    }

    #[test]
    fn nearest_anchor_with_no_anchors_is_unreachable() {
        // Default OSINT declares no anchor families ⇒ every node is unreachable.
        let rows = [node(&OSINT_GOTHAM, 1, 0, 0, 0xA, 1, &[], &[])];
        let hops = nearest_anchor(&rows, &OSINT_GOTHAM);
        assert_eq!(hops[0].hops, u8::MAX);
        assert_eq!(hops[0].anchor_family, u32::MAX);
    }

    #[test]
    fn projection_is_head_only_zero_value_decode() {
        // Poison the value slab; the snapshot must be byte-identical (the
        // E-GUID-IS-THE-GRAPH / zero-value-decode invariant, falsifiable).
        let clean = [
            node(&OSINT_GOTHAM, 1, 0, 0, 0xA, 1, &[2], &[0xB]),
            node(&OSINT_GOTHAM, 2, 0, 0, 0xB, 2, &[], &[]),
        ];
        let mut poisoned = clean;
        for row in &mut poisoned {
            row.value = [0xFFu8; 480];
        }
        let a = project_snapshot(&clean, &OSINT_GOTHAM);
        let b = project_snapshot(&poisoned, &OSINT_GOTHAM);
        // GraphSnapshot isn't PartialEq; compare the structural projection.
        let key = |s: &GraphSnapshot| {
            (
                s.nodes.iter().map(|n| (n.id.clone(), n.kind.clone())).collect::<Vec<_>>(),
                s.edges
                    .iter()
                    .map(|e| (e.source.clone(), e.target.clone(), e.label.clone()))
                    .collect::<Vec<_>>(),
            )
        };
        assert_eq!(key(&a), key(&b));
    }
}
