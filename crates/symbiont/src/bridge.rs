//! D1 — the first real runtime edge: perturbation-sim `Grid` → canonical SoA `NodeRow`.
//!
//! Runs the DC-power-flow cascade over a small in-tree lattice and encodes each
//! bus's final perturbation magnitude (`|θ_final − θ_base|`) into one canonical
//! `NodeRow` (key = `NodeGuid::local(bus)`, `value[0..8]` = f64 little-endian).
//!
//! This is the degenerate first case of the Spain-grid acceptance gate: real
//! nodes, on the actual SoA substrate, NaN-free. It turns the golden image from
//! a link-only probe into a harness that exercises one genuine runtime edge
//! between two of the five crates (perturbation-sim ↔ lance-graph-contract).
//!
//! `NodeRow` carries no live simulation state — the cascade owns its `Vec<f64>`
//! working buffers; we encode only the *result* into the SoA. One-directional.

use lance_graph_contract::canonical_node::NodeRowPacket;
use lance_graph_contract::soa_envelope::SoaEnvelope; // brings `as_le_bytes` into scope
use lance_graph_contract::{EdgeBlock, NodeGuid, NodeRow};
use perturbation_sim::{simulate_outage, CascadeConfig, Edge, Grid};

/// Stride of one canonical node on the wire / in the SoA backing store.
const NODE_ROW_STRIDE: usize = 512;
/// Slab offset for the encoded perturbation magnitude (raw f64, not the named
/// `ValueTenant::Energy` carve — honest and decode-trivial).
const NODE_FIELD_OFFSET: usize = 0;

/// Build a `rows × cols` lattice with a corner-to-corner dipole injection.
/// Returns the grid plus a balanced injection vector `p` (Σ = 0, one bus per
/// node). Pure in-tree — no network, no files.
pub fn build_demo_grid(rows: usize, cols: usize) -> (Grid, Vec<f64>) {
    let n = rows * cols;
    let idx = |r: usize, c: usize| r * cols + c;
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            if c + 1 < cols {
                edges.push(Edge::new(idx(r, c), idx(r, c + 1), 1.0, 5.0));
            }
            if r + 1 < rows {
                edges.push(Edge::new(idx(r, c), idx(r + 1, c), 1.0, 5.0));
            }
        }
    }
    let grid = Grid::new(n, edges);
    let mut p = vec![0.0_f64; n];
    p[0] = 1.0; // source at one corner
    p[n - 1] = -1.0; // sink at the opposite corner
    (grid, p)
}

/// Run the cascade over `grid` and encode the result into one canonical
/// `NodeRow` per bus: identity = bus index, `value[0..8]` = `node_field[i]` f64 LE.
pub fn grid_to_noderows(grid: &Grid, p: &[f64], seed_line: usize) -> Vec<NodeRow> {
    let result = simulate_outage(grid, p, seed_line, CascadeConfig::default());
    (0..grid.n)
        .map(|i| {
            let mut value = [0u8; 480];
            value[NODE_FIELD_OFFSET..NODE_FIELD_OFFSET + 8]
                .copy_from_slice(&result.shape.node_field[i].to_le_bytes());
            NodeRow {
                key: NodeGuid::local(i as u32),
                edges: EdgeBlock::default(),
                value,
            }
        })
        .collect()
}

/// Decode a bus's perturbation magnitude back out of its `NodeRow` value slab.
#[inline]
pub fn decode_node_field(row: &NodeRow) -> f64 {
    let bytes: [u8; 8] = row.value[NODE_FIELD_OFFSET..NODE_FIELD_OFFSET + 8]
        .try_into()
        .expect("8 bytes");
    f64::from_le_bytes(bytes)
}

/// The acceptance-gate demo (first real instance): build → cascade → encode →
/// assert finite → report. Returns the encoded rows so callers can sweep them.
pub fn run_demo() -> Vec<NodeRow> {
    let (grid, p) = build_demo_grid(8, 8); // 64 buses
    let rows = grid_to_noderows(&grid, &p, 0);

    // The acceptance-gate invariant, at SoA scale: no NaN reaches a node.
    assert!(
        rows.iter().all(|r| decode_node_field(r).is_finite()),
        "NaN escaped into a NodeRow value slab"
    );

    // Prove the zero-copy SoA stride (512 B/row) without re-serializing.
    let packet = NodeRowPacket::new(&rows, 0);
    let bytes = packet.as_le_bytes().len();

    let max = rows
        .iter()
        .map(decode_node_field)
        .fold(0.0_f64, f64::max);
    println!(
        "D1 bridge: {} buses → {} NodeRows (key=NodeGuid::local), all node_field finite; \
         SoA packet {bytes} bytes ({} B/row); max |perturbation| = {max:.6}",
        grid.n,
        rows.len(),
        bytes / rows.len().max(1),
    );
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe: every bus encodes a finite f64 that round-trips bit-exactly
    /// through the canonical SoA value slab. (Phase-A finiteness + B-series
    /// over-the-SoA, from the battle-test plan.)
    #[test]
    fn grid_to_noderows_is_always_finite_and_roundtrips() {
        let (grid, p) = build_demo_grid(6, 6);
        let result = simulate_outage(&grid, &p, 0, CascadeConfig::default());
        let rows = grid_to_noderows(&grid, &p, 0);
        assert_eq!(rows.len(), grid.n);
        for (i, row) in rows.iter().enumerate() {
            let decoded = decode_node_field(row);
            // bit-exact round-trip through the SoA value slab
            assert_eq!(decoded.to_bits(), result.shape.node_field[i].to_bits());
            assert!(decoded.is_finite());
        }
    }

    /// The SoA backing store is a flat 512-B stride — zero-copy to Lance.
    #[test]
    fn soa_packet_stride_is_512() {
        let (grid, p) = build_demo_grid(5, 5);
        let rows = grid_to_noderows(&grid, &p, 0);
        let packet = NodeRowPacket::new(&rows, 0);
        assert_eq!(packet.as_le_bytes().len(), rows.len() * NODE_ROW_STRIDE);
    }
}
