//! Deterministic HHTL topology key — map each bus to its `(HEEL, HIP, TWIG)`
//! cascade-tier address by recursive Cheeger bisection of the Laplacian.
//!
//! This makes "topology IS the key" concrete (the HHTL-OGAR correction): the key
//! is a **pure, deterministic function of the graph spectrum** — Cheeger/Fiedler
//! is deterministic given the Laplacian, so the same grid always yields the same
//! HHTL grid. Value members (the study factors, helix residues) then hang off this
//! key, orthogonal to it by the key/value split.
//!
//! Tiers here are produced by **binary** Cheeger splits (one bit per level → the
//! HHTL depth-3 tree = 8 leaf basins), the same compartmentalization the
//! `resilience` example uses. The OGAR production form widens each tier to a 16-ary
//! / 256-centroid tile (`FAN_OUT=16`); this is the spectral-bisection instance of
//! the same address, not that full encoding (kept honest).

use crate::graph::{Edge, Grid};
use crate::{cheeger_sweep, symmetric_eigen};
use std::collections::HashMap;

/// A node's HHTL cascade address: the path through the recursive bisection tree.
/// `heel` = top tier, `hip` = mid, `twig` = leaf. (u16 to match the OGAR key
/// layout; the binary-Cheeger instance only fills the low bits per tier.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HhtlKey {
    pub heel: u16,
    pub hip: u16,
    pub twig: u16,
}

/// Induced sub-grid on `members` (reindexed 0..k), edges with both endpoints kept.
fn induced(grid: &Grid, members: &[usize]) -> Grid {
    let mut remap = HashMap::new();
    for (i, &m) in members.iter().enumerate() {
        remap.insert(m, i);
    }
    let edges = grid
        .edges
        .iter()
        .filter_map(|e| match (remap.get(&e.from), remap.get(&e.to)) {
            (Some(&a), Some(&b)) => Some(Edge::new(a, b, e.susceptance, e.limit)),
            _ => None,
        })
        .collect();
    Grid::new(members.len(), edges)
}

/// One deterministic binary Cheeger split of `members` into (bit-0 side, bit-1
/// side). A basin too small / too sparse to split returns everything on bit 0.
fn split(grid: &Grid, members: &[usize]) -> (Vec<usize>, Vec<usize>) {
    if members.len() < 4 {
        return (members.to_vec(), Vec::new());
    }
    let sub = induced(grid, members);
    if sub.edges.is_empty() {
        return (members.to_vec(), Vec::new());
    }
    let part = cheeger_sweep(&sub, &vec![true; sub.edges.len()]).partition;
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (i, &m) in members.iter().enumerate() {
        if part[i] {
            a.push(m);
        } else {
            b.push(m);
        }
    }
    if a.is_empty() || b.is_empty() {
        (members.to_vec(), Vec::new())
    } else {
        (a, b)
    }
}

/// Assign every node its `(HEEL, HIP, TWIG)` key by three nested binary Cheeger
/// splits. Deterministic: a pure function of the grid topology.
pub fn hhtl_keys(grid: &Grid) -> Vec<HhtlKey> {
    let mut keys = vec![
        HhtlKey {
            heel: 0,
            hip: 0,
            twig: 0
        };
        grid.n
    ];
    let all: Vec<usize> = (0..grid.n).collect();
    let (h0, h1) = split(grid, &all);
    for (heel, side) in [h0, h1].into_iter().enumerate() {
        for &n in &side {
            keys[n].heel = heel as u16;
        }
        let (p0, p1) = split(grid, &side);
        for (hip, mid) in [p0, p1].into_iter().enumerate() {
            for &n in &mid {
                keys[n].hip = hip as u16;
            }
            let (t0, t1) = split(grid, &mid);
            for (twig, leaf) in [t0, t1].into_iter().enumerate() {
                for &n in &leaf {
                    keys[n].twig = twig as u16;
                }
            }
        }
    }
    keys
}

/// Per-leaf-basin algebraic connectivity `λ₂` keyed by HHTL address — the topology
/// "value" the key indexes (read once from the spectrum, deterministic).
pub fn basin_lambda2(grid: &Grid, keys: &[HhtlKey]) -> HashMap<HhtlKey, f64> {
    assert_eq!(
        keys.len(),
        grid.n,
        "basin_lambda2 requires exactly one HHTL key per grid node (got {} keys for {} nodes)",
        keys.len(),
        grid.n
    );
    let mut groups: HashMap<HhtlKey, Vec<usize>> = HashMap::new();
    for (n, k) in keys.iter().enumerate() {
        groups.entry(*k).or_default().push(n);
    }
    let mut out = HashMap::new();
    for (k, members) in groups {
        let sub = induced(grid, &members);
        let l2 = if sub.edges.is_empty() {
            0.0
        } else {
            symmetric_eigen(&sub.laplacian_of(&vec![true; sub.edges.len()]), sub.n)
                .values
                .get(1)
                .copied()
                .unwrap_or(0.0)
        };
        out.insert(k, l2);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_2x2_blocks() -> Grid {
        // Two 4-cliques weakly joined by one bridge — a clean 2-basin topology.
        let mut e = Vec::new();
        for (a, b) in [(0, 1), (0, 2), (1, 3), (2, 3)] {
            e.push(Edge::new(a, b, 1.0, 1.0));
        }
        for (a, b) in [(4, 5), (4, 6), (5, 7), (6, 7)] {
            e.push(Edge::new(a, b, 1.0, 1.0));
        }
        e.push(Edge::new(3, 4, 0.01, 1.0)); // weak bridge
        Grid::new(8, e)
    }

    #[test]
    fn key_is_deterministic_function_of_topology() {
        let g = grid_2x2_blocks();
        assert_eq!(hhtl_keys(&g), hhtl_keys(&g), "same grid ⇒ same HHTL grid");
    }

    #[test]
    fn the_two_blocks_get_distinct_heel_tiers() {
        let g = grid_2x2_blocks();
        let k = hhtl_keys(&g);
        // The weak bridge is the top Cheeger cut ⇒ block {0..3} and {4..7} split on HEEL.
        let heel_a = k[0].heel;
        assert!(k[1].heel == heel_a && k[2].heel == heel_a && k[3].heel == heel_a);
        assert!(
            k[4].heel != heel_a,
            "the other block lands on the other HEEL tier"
        );
    }

    #[test]
    fn keys_partition_the_nodes() {
        let g = grid_2x2_blocks();
        let k = hhtl_keys(&g);
        // Every node has a key; nodes sharing a full key are in one leaf basin.
        let l2 = basin_lambda2(&g, &k);
        assert!(!l2.is_empty(), "at least one keyed basin");
        assert_eq!(k.len(), g.n);
    }
}
