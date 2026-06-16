//! Cascade simulation: the **edge-propagation** half. Trip a line, redistribute
//! DC flows, trip whatever now exceeds its limit, repeat. The accumulated
//! per-node and per-edge deviation fields are the **perturbation shape** — the
//! footprint that lights up red on the topology view.

use crate::eigen::symmetric_eigen;
use crate::flow::dc_flows;
use crate::graph::Grid;
use crate::perturbation::{spectral_perturbation, SpectralPerturbation};

/// Tunables for the cascade.
#[derive(Debug, Clone, Copy)]
pub struct CascadeConfig {
    /// A line trips when `|flow| > overload_factor · limit`.
    pub overload_factor: f64,
    /// Hard cap on cascade rounds.
    pub max_rounds: usize,
    /// Relative tolerance for the pseudo-inverse / zero-eigenvalue test.
    pub rel_tol: f64,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            overload_factor: 1.0,
            max_rounds: 64,
            rel_tol: 1e-9,
        }
    }
}

/// The perturbation shape: deviation fields plus the trip footprint.
#[derive(Debug, Clone)]
pub struct PerturbationShape {
    /// Per-bus angle deviation `|θ_final − θ_base|` — the spectral/flow
    /// perturbation magnitude at each node.
    pub node_field: Vec<f64>,
    /// Per-line flow shift. For a surviving line: `|f_final − f_base|`. For a
    /// tripped line: the flow it was carrying when it tripped (its
    /// contribution to the cascade).
    pub edge_field: Vec<f64>,
    /// Whether each line ended up tripped.
    pub tripped: Vec<bool>,
    /// Cascade round in which each line tripped (`-1` if it survived; `0` is
    /// the seed trip).
    pub trip_round: Vec<i32>,
}

impl PerturbationShape {
    /// Lines still overloaded at the end (only nonempty if `max_rounds` was hit
    /// before convergence).
    pub fn n_tripped(&self) -> usize {
        self.tripped.iter().filter(|&&t| t).count()
    }

    /// The `k` buses with the largest perturbation, descending — the epicentre
    /// of the shape. Returns `(bus, magnitude)`.
    pub fn epicentre(&self, k: usize) -> Vec<(usize, f64)> {
        let mut v: Vec<(usize, f64)> = self.node_field.iter().copied().enumerate().collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v.truncate(k);
        v
    }
}

/// Full outage simulation seeded by tripping `seed_line`.
#[derive(Debug, Clone)]
pub struct CascadeResult {
    pub shape: PerturbationShape,
    /// Rank-1 spectral analysis (Weyl / Davis–Kahan / Fiedler) of the seed trip.
    pub spectral: SpectralPerturbation,
    /// Number of cascade rounds executed.
    pub rounds: usize,
    /// Fraction of lines that ended up tripped.
    pub fraction_tripped: f64,
    /// True if the surviving network fragmented into ≥2 components.
    pub islanded: bool,
    /// Connected-component count of the final surviving network.
    pub components_final: usize,
}

/// Simulate a cascading outage on `grid` under balanced injections `p`
/// (`∑ p ≈ 0`), seeded by tripping `seed_line`.
///
/// Each round recomputes the DC flows on the *surviving* network from scratch
/// (an exact recompute, robust where iterated single-line LODF would drift),
/// trips every line now over its limit, and stops when no new line trips,
/// `max_rounds` is reached, or the network islands.
pub fn simulate_outage(
    grid: &Grid,
    p: &[f64],
    seed_line: usize,
    cfg: CascadeConfig,
) -> CascadeResult {
    let n = grid.n;
    let m = grid.edges.len();
    assert_eq!(p.len(), n, "injection vector must have one entry per bus");
    assert!(seed_line < m, "seed line out of range");

    let all_alive = vec![true; m];

    // Base state (all lines in service).
    let eig0 = symmetric_eigen(&grid.laplacian_of(&all_alive), n);
    let theta_base = eig0.pseudo_apply(p, cfg.rel_tol);
    let flow_base = dc_flows(grid, &all_alive, &theta_base);

    // Rank-1 spectral analysis of the seed trip (before vs after).
    let spectral = spectral_perturbation(grid, &all_alive, seed_line);

    // Seed the cascade.
    let mut alive = all_alive.clone();
    alive[seed_line] = false;
    let mut trip_round = vec![-1i32; m];
    trip_round[seed_line] = 0;

    // Assigned on every loop iteration before any break (the loop body always
    // runs at least once), so no initializer is needed.
    let mut theta: Vec<f64>;
    let mut flow: Vec<f64>;
    let mut islanded = false;
    let mut components_final: usize;
    let mut rounds = 0usize;

    loop {
        rounds += 1;
        let eig = symmetric_eigen(&grid.laplacian_of(&alive), n);
        components_final = eig.nullity(cfg.rel_tol);

        theta = eig.pseudo_apply(p, cfg.rel_tol);
        flow = dc_flows(grid, &alive, &theta);

        if components_final > 1 {
            // Network fragmented: injections no longer balance per island, so
            // the DC solution is only the least-norm proxy. Treat as terminal
            // (the blackout has split the grid) rather than fabricate flows.
            islanded = true;
            break;
        }

        let mut new_trips = Vec::new();
        for (e, edge) in grid.edges.iter().enumerate() {
            if alive[e] && flow[e].abs() > cfg.overload_factor * edge.limit {
                new_trips.push(e);
            }
        }
        if new_trips.is_empty() || rounds >= cfg.max_rounds {
            break;
        }
        for e in new_trips {
            alive[e] = false;
            trip_round[e] = rounds as i32;
        }
    }

    // Build the shape.
    let node_field: Vec<f64> = (0..n).map(|i| (theta[i] - theta_base[i]).abs()).collect();
    let edge_field: Vec<f64> = (0..m)
        .map(|e| {
            if alive[e] {
                (flow[e] - flow_base[e]).abs()
            } else {
                flow_base[e].abs()
            }
        })
        .collect();
    let tripped: Vec<bool> = alive.iter().map(|&a| !a).collect();
    let n_tripped = tripped.iter().filter(|&&t| t).count();

    CascadeResult {
        shape: PerturbationShape {
            node_field,
            edge_field,
            tripped,
            trip_round,
        },
        spectral,
        rounds,
        fraction_tripped: n_tripped as f64 / m as f64,
        islanded,
        components_final,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Edge;

    #[test]
    fn no_overload_means_only_the_seed_trips() {
        // Generous limits => nothing cascades beyond the seed.
        let g = Grid::new(
            4,
            vec![
                Edge::new(0, 1, 1.0, 1e6),
                Edge::new(1, 2, 1.0, 1e6),
                Edge::new(2, 3, 1.0, 1e6),
                Edge::new(3, 0, 1.0, 1e6),
            ],
        );
        let p = vec![1.0, 0.0, -1.0, 0.0];
        let r = simulate_outage(&g, &p, 0, CascadeConfig::default());
        assert_eq!(r.shape.n_tripped(), 1);
        assert!(!r.islanded);
        assert!(r.spectral.weyl_satisfied);
    }

    #[test]
    fn tight_limits_produce_a_multi_line_cascade() {
        // A 4-cycle carrying flow on every line; seed-trip forces all flow onto
        // the remaining path, and tight limits make a neighbour trip too.
        let g = Grid::new(
            4,
            vec![
                Edge::new(0, 1, 1.0, 0.6),
                Edge::new(1, 2, 1.0, 0.6),
                Edge::new(2, 3, 1.0, 0.6),
                Edge::new(3, 0, 1.0, 0.6),
            ],
        );
        let p = vec![1.0, 0.0, -1.0, 0.0];
        let r = simulate_outage(&g, &p, 0, CascadeConfig::default());
        assert!(
            r.shape.n_tripped() >= 2,
            "expected a cascade, only {} tripped",
            r.shape.n_tripped()
        );
        // The perturbation shape must be non-trivial somewhere.
        assert!(r.shape.node_field.iter().any(|&x| x > 1e-9));
    }

    #[test]
    fn islanding_is_flagged() {
        // Two triangles + one bridge; tripping the bridge islands the grid.
        let g = Grid::new(
            6,
            vec![
                Edge::new(0, 1, 1.0, 1e6),
                Edge::new(1, 2, 1.0, 1e6),
                Edge::new(2, 0, 1.0, 1e6),
                Edge::new(3, 4, 1.0, 1e6),
                Edge::new(4, 5, 1.0, 1e6),
                Edge::new(5, 3, 1.0, 1e6),
                Edge::new(2, 3, 1.0, 1e6),
            ],
        );
        // Inject across the bridge so it carries flow.
        let p = vec![1.0, 0.0, 0.0, 0.0, 0.0, -1.0];
        let r = simulate_outage(&g, &p, 6, CascadeConfig::default());
        assert!(r.islanded, "bridge trip must island the network");
        assert_eq!(r.components_final, 2);
    }
}
