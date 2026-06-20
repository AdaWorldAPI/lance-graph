//! D1 — the first real runtime edge, to the operator's architecture (2026-06-20).
//!
//! **Each perturbation-grid node becomes ONE canonical SoA node** (`NodeRow`) —
//! not a scalar in a shared slab. The Spanish grid = up to 16384 `NodeRow`s =
//! ~8 MiB (16384 × 512 B) = up to 16384 **kanban boards**.
//!
//! **Each external f64 → ONE internal typed `ValueTenant`.** A bus's perturbation
//! magnitude (`|θ_final − θ_base|`) maps to `ValueTenant::Energy` — the
//! substrate's spatio-temporal accumulator. (`Energy` is an `f32` column; the
//! f64→f32 narrowing is the SoA's deliberate accumulator precision. A true-f64
//! tenant would be a canon EXTENSION — operator-gated, not done here.) One
//! external quantity ⇒ one typed internal column the SIMD sweep / planner reads —
//! never raw `value[0..8]` bytes.
//!
//! **The perturbation cascade IS a thinking-style cascade** — a deterministic
//! per-node field update, the physics instance of the cognitive cascade skeleton
//! (CLAUDE.md "the shader can't resist the thinking").
//!
//! Up to 16384 boards are driven by **surrealdb + ractor + lance-graph-planner**
//! via a **Lance subscription hook** (D2: the planner SoA reacts to Lance
//! versions). `NodeRow` carries no live state during compute — the cascade owns
//! its `Vec<f64>` buffers; only the *result* lands in the `Energy` tenant.

use lance_graph_contract::canonical_node::{NodeRowPacket, ValueTenant};
use lance_graph_contract::soa_envelope::SoaEnvelope;
use lance_graph_contract::{EdgeBlock, NodeGuid, NodeRow};
use perturbation_sim::{simulate_outage, CascadeConfig, Edge, Grid};

/// Canonical node stride (key 16 + edges 16 + value 480).
const NODE_ROW_STRIDE: usize = 512;
/// The substrate's board ceiling: 16384 × 512 B = 8 MiB.
pub const MAX_BOARDS: usize = 16_384;

/// Write one external f64 into a node's `Energy` tenant (the typed SoA column),
/// narrowing to the accumulator's `f32` precision. One external ⇒ one tenant.
#[inline]
fn set_energy(row: &mut NodeRow, value: f64) {
    let off = ValueTenant::Energy.value_offset();
    let len = ValueTenant::Energy.byte_len(); // 4 (F32)
    row.value[off..off + len].copy_from_slice(&(value as f32).to_le_bytes());
}

/// Read a node's `Energy` tenant back out as `f32`.
#[inline]
pub fn energy(row: &NodeRow) -> f32 {
    let off = ValueTenant::Energy.value_offset();
    let len = ValueTenant::Energy.byte_len();
    let bytes: [u8; 4] = row.value[off..off + len].try_into().expect("4 bytes");
    f32::from_le_bytes(bytes)
}

/// One bare SoA board (a `NodeRow`) at the bootstrap address `identity = idx`.
#[inline]
fn board(idx: usize) -> NodeRow {
    NodeRow {
        key: NodeGuid::local(idx as u32),
        edges: EdgeBlock::default(),
        value: [0u8; 480],
    }
}

/// Build a `rows × cols` lattice with a corner-to-corner dipole injection.
/// Pure in-tree — no network, no files.
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
    p[0] = 1.0; // source corner
    p[n - 1] = -1.0; // sink corner
    (grid, p)
}

/// Run the cascade (a thinking-style cascade) over `grid` and turn EACH bus into
/// ONE canonical SoA `NodeRow`: identity = bus index, `Energy` tenant = its
/// perturbation magnitude.
pub fn grid_to_noderows(grid: &Grid, p: &[f64], seed_line: usize) -> Vec<NodeRow> {
    let result = simulate_outage(grid, p, seed_line, CascadeConfig::default());
    (0..grid.n)
        .map(|i| {
            let mut row = board(i);
            set_energy(&mut row, result.shape.node_field[i]);
            row
        })
        .collect()
}

/// The acceptance-gate demo (first real instance, 64 buses → 64 SoA boards).
pub fn run_demo() -> Vec<NodeRow> {
    let (grid, p) = build_demo_grid(8, 8);
    let rows = grid_to_noderows(&grid, &p, 0);
    assert!(
        rows.iter().all(|r| energy(r).is_finite()),
        "NaN reached an Energy tenant"
    );
    let max = rows.iter().map(energy).fold(0.0_f32, f32::max);
    println!(
        "D1 bridge: {} buses → {} SoA NodeRows (1 kanban board each); each bus's f64 \
         perturbation lives in its Energy(f32) tenant, all finite; max |perturbation| = {max:.6}",
        grid.n,
        rows.len(),
    );
    rows
}

/// The scale probe (E2): allocate up to `n` SoA boards and sweep their `Energy`
/// tenants — proves the 16k-board / 8-MiB substrate scale + the flat 512-B
/// zero-copy stride (the "16384 AGI cores" storage footprint, made concrete).
pub fn run_scale_demo(n: usize) -> usize {
    let n = n.min(MAX_BOARDS);
    let rows: Vec<NodeRow> = (0..n)
        .map(|i| {
            let mut row = board(i);
            set_energy(&mut row, (i as f64).sqrt()); // synthetic, non-trivial sweep
            row
        })
        .collect();
    let packet = NodeRowPacket::new(&rows, 0);
    let bytes = packet.as_le_bytes().len();
    assert_eq!(bytes, n * NODE_ROW_STRIDE);
    let finite = rows.iter().filter(|r| energy(r).is_finite()).count();
    assert_eq!(finite, n);
    println!(
        "SoA scale: {n} kanban boards = {} MiB ({} B/row), zero-copy; all {finite} Energy \
         tenants finite",
        bytes / (1024 * 1024),
        bytes / n.max(1),
    );
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each bus is ONE SoA node whose external f64 lives in its Energy tenant,
    /// finite, exact at the accumulator's f32 precision.
    #[test]
    fn each_bus_is_one_soa_node_with_finite_energy_tenant() {
        let (grid, p) = build_demo_grid(6, 6);
        let result = simulate_outage(&grid, &p, 0, CascadeConfig::default());
        let rows = grid_to_noderows(&grid, &p, 0);
        assert_eq!(rows.len(), grid.n);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(energy(row), result.shape.node_field[i] as f32);
            assert!(energy(row).is_finite());
        }
    }

    /// The 16k-board ceiling is exactly 8 MiB of zero-copy SoA.
    #[test]
    fn scale_to_16k_boards_is_8_mib_zero_copy() {
        let bytes = run_scale_demo(MAX_BOARDS);
        assert_eq!(bytes, 16_384 * NODE_ROW_STRIDE);
        assert_eq!(bytes, 8 * 1024 * 1024); // 8 MiB
    }
}
