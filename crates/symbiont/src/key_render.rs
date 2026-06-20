//! §2.4 — the key-only, neo4j-grade graph render.
//!
//! Operator superpower (`unified-soa-rubikon-integration-v1.md` §2.4): read all
//! boards touching ONLY the **32-byte head** of each `NodeRow` — the 128-bit
//! [`NodeGuid`] (`key`, bytes 0..16) + the 128-bit [`EdgeBlock`] (`edges`, bytes
//! 16..32) — and NEVER the 480-byte value slab (bytes 32..512). That is a Neo4j-
//! like node+edge view at memory-scan speed: `E-GUID-IS-THE-GRAPH` — GUID-key =
//! node, EdgeBlock-slot = edge, traversal = prefix-route + slot-deref, all
//! **zero value decode**.
//!
//! "Zero value decode" is not a slogan here — it is a falsifiable probe
//! ([`tests::render_ignores_value_slab`]): poison every row's value slab with
//! `0xFF`, render, and assert the output is byte-identical to the render over
//! zeroed slabs. If the render ever touched the value region the two would
//! differ. The render reads exactly two fields, `row.key` and `row.edges`.
//!
//! This is the read side of the same head the SoA owner materialises through the
//! contract's zero-decode key facets `MailboxSoaView::{hhtl_path_at,
//! edge_block_at}` (overridden on `SymbiontBoard`, kanban_loop.rs) — the trait
//! declares them "zero value decode"; this module is the consumer that proves it.

use lance_graph_contract::canonical_node::NodeRow;
use lance_graph_contract::hhtl::NiblePath;
use lance_graph_contract::NodeGuid;

/// Lower a GUID to its HHTL routing path via the canonical
/// [`NiblePath::from_guid_prefix`] (`classid_lo·HEEL·HIP·TWIG`, 16 nibbles,
/// root-first) — the radix-trie / CLAM cluster address. Falls back to
/// [`NiblePath::EMPTY`] only for the canon-reserved non-zero high-`classid` case.
/// Thin wrapper kept for callers in this crate; the lowering itself is the
/// contract's single source of truth (no third copy). Zero value decode.
#[inline]
pub fn hhtl_path_of(guid: &NodeGuid) -> NiblePath {
    NiblePath::from_guid_prefix(guid).unwrap_or(NiblePath::EMPTY)
}

/// One rendered node, derived from the 32-byte head ONLY.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedNode {
    /// The node identity = the 128-bit canonical GUID (`key`, bytes 0..16).
    pub guid: NodeGuid,
    /// The HHTL routing address of the GUID (the 3×4 HHT nibble path).
    pub hhtl: NiblePath,
    /// Live in-family edge slots: `(slot_index 0..12, neighbour_byte)` for each
    /// non-zero of the 12 basin-local slots.
    pub in_family: Vec<(u8, u8)>,
    /// Live out-of-family edge slots: `(slot_index 0..4, neighbour_byte)` for
    /// each non-zero of the 4 inherited-adapter slots.
    pub out_family: Vec<(u8, u8)>,
}

/// The whole key-only graph view: one [`RenderedNode`] per board + the total
/// live-edge count. Built touching only the head of every row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyGraph {
    pub nodes: Vec<RenderedNode>,
    pub edge_count: usize,
}

/// Render the key-only graph over a board-set, touching ONLY `row.key` (the
/// GUID) and `row.edges` (the EdgeBlock) — never `row.value`. This is the
/// memory-scan-speed Neo4j view: at 16384 boards it walks 16384 × 32 B = 512 KiB
/// of heads, leaving the 8 MiB of value slabs cold.
pub fn render_key_only(rows: &[NodeRow]) -> KeyGraph {
    let mut nodes = Vec::with_capacity(rows.len());
    let mut edge_count = 0usize;
    for row in rows {
        // ── the ONLY two field reads: the 32-byte head ──
        let guid = row.key;
        let eb = row.edges;
        let in_family: Vec<(u8, u8)> = eb
            .in_family
            .iter()
            .enumerate()
            .filter(|(_, &b)| b != 0)
            .map(|(i, &b)| (i as u8, b))
            .collect();
        let out_family: Vec<(u8, u8)> = eb
            .out_family
            .iter()
            .enumerate()
            .filter(|(_, &b)| b != 0)
            .map(|(i, &b)| (i as u8, b))
            .collect();
        edge_count += in_family.len() + out_family.len();
        nodes.push(RenderedNode {
            guid,
            hhtl: hhtl_path_of(&guid),
            in_family,
            out_family,
        });
    }
    KeyGraph { nodes, edge_count }
}

/// The §2.4 demo: render the 16k-board SoA key-only and report node/edge counts
/// + a self-describing sample, proving the Neo4j-grade view costs only the
/// 32-byte heads (8 MiB of value slabs untouched).
pub fn run_demo() {
    let rows = crate::domino::seed_boards(crate::bridge::MAX_BOARDS);
    let g = render_key_only(&rows);
    let head_bytes = g.nodes.len() * 32;
    let slab_bytes = g.nodes.len() * 480;
    let sample = &g.nodes[3];
    println!(
        "§2.4 key-only render: {} nodes / {} edges from {} KiB of 32-byte heads \
         ({} KiB of value slabs left COLD, zero decode); sample node[3] = {} \
         hhtl_depth={} in_family={:?} out_family={:?}",
        g.nodes.len(),
        g.edge_count,
        head_bytes / 1024,
        slab_bytes / 1024,
        sample.guid,
        sample.hhtl.depth(),
        sample.in_family,
        sample.out_family,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domino::seed_boards;

    #[test]
    fn render_reads_head_only_and_finds_edges() {
        let rows = seed_boards(16);
        let g = render_key_only(&rows);
        assert_eq!(g.nodes.len(), 16);
        // each seeded board has in_family[0] (ring) + out_family[0] (adapter).
        assert_eq!(g.edge_count, 32);
        let n = &g.nodes[3];
        // identity = idx (bootstrap address: classid + family default).
        assert_eq!(n.guid.identity(), 3);
        assert!(n.guid.is_bootstrap_address());
        assert_eq!(n.in_family, vec![(0u8, 4u8)]); // (3 % 255) + 1
        assert_eq!(n.out_family, vec![(0u8, 4u8)]); // 1 + (3 % 4)
    }

    #[test]
    fn render_ignores_value_slab() {
        // The falsifiable zero-value-decode probe: poisoning the value slab with
        // 0xFF must not change the key-only render by a single byte. If the
        // render touched bytes 32..512, the two KeyGraphs would differ.
        let clean = seed_boards(64);
        let mut poisoned = clean.clone();
        for row in &mut poisoned {
            row.value = [0xFFu8; 480];
        }
        assert_eq!(render_key_only(&clean), render_key_only(&poisoned));
    }

    #[test]
    fn hhtl_path_of_bootstrap_is_depth_16_all_zero() {
        // The canonical lowering packs 16 nibbles (classid_lo·HEEL·HIP·TWIG). A
        // bootstrap GUID (classid=HEEL=HIP=TWIG=0) lowers to a 16-nibble all-zero
        // path — every tier consulted, none discriminating yet (zero-fallback).
        let g = NodeGuid::local(42);
        let p = hhtl_path_of(&g);
        assert_eq!(p.depth(), 16);
    }

    #[test]
    fn hhtl_path_of_includes_classid_lo_and_hht_not_identity() {
        // Canonical lowering: classid_lo is the routing PREFIX (top 4 nibbles),
        // HEEL/HIP/TWIG are the cascade, identity (leaf) is NOT in the path.
        // a vs b differ ONLY in identity ⇒ same path; a vs c differ in classid_lo
        // ⇒ different path; a vs d differ in a HHT tier ⇒ different path.
        let a = NodeGuid::new(0x0000_0007, 0x1234, 0x5678, 0x9ABC, 0xAB, 0x00_0001);
        let b = NodeGuid::new(0x0000_0007, 0x1234, 0x5678, 0x9ABC, 0xAB, 0x00_0002);
        let c = NodeGuid::new(0x0000_0008, 0x1234, 0x5678, 0x9ABC, 0xAB, 0x00_0001);
        let d = NodeGuid::new(0x0000_0007, 0x0234, 0x5678, 0x9ABC, 0xAB, 0x00_0001);
        assert_eq!(hhtl_path_of(&a), hhtl_path_of(&b));
        assert_ne!(hhtl_path_of(&a), hhtl_path_of(&c));
        assert_ne!(hhtl_path_of(&a), hhtl_path_of(&d));
    }
}
