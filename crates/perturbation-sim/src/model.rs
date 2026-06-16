//! Data-shaped scoping — run the simulation on whatever data exists *now*,
//! compute only what that data supports, and model unknown variables as
//! **uniform constants** (never as noise).
//!
//! The governing principle:
//!
//! > A missing per-asset variable modeled as ONE uniform constant injects **no
//! > spurious heterogeneity** — so the *relative* perturbation shape and the
//! > contingency ranking stay clean. Only genuine, data-backed heterogeneity
//! > should bend the shape. "We don't know each line's age, so assume the whole
//! > network is equally (uniformly) outdated" is therefore the *correct* null
//! > modeling choice, not a cop-out.
//!
//! Two regimes, kept honest:
//!
//! - **Uniform susceptance scale is provably free** ([`scale_susceptance`] +
//!   its invariance test): scaling every `b_e` by a constant `c` leaves the DC
//!   flows, the cascade, and `connectivity_loss` *exactly* invariant (only
//!   absolute angles scale, and `λ₂ → c·λ₂` cancels in the ratio). So a uniform
//!   conductor-age / material assumption costs nothing for relative analysis.
//! - **Uniform derate is a global stress knob** ([`with_uniform_derate`]): it
//!   shifts every line's margin together. It is NOT invariant (thresholds
//!   move), but being uniform it adds no false structure — sweep it as a single
//!   sensitivity parameter, disclose it.
//!
//! [`assess_capability`] reports which outputs the data on hand actually
//! supports (relative shape vs absolute MW vs loss gradient vs tech-debt
//! differential), so a caller never over-claims an uncalibrated number.

use crate::graph::{Edge, Grid};
use crate::ingest::PypsaImport;

/// What the available data supports. Lower levels still run the sim — they just
/// validate fewer *kinds* of output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataLevel {
    /// Buses + lines + voltage only (the PyPSA-Eur/OSM base case): reactance and
    /// limits are uniform-constant estimates.
    TopologyOnly,
    /// Per-line reactance present (`x` column) — electrical distance is real.
    WithReactance,
    /// Per-line resistance present — the loss gradient is computable.
    WithLosses,
    /// Per-asset heterogeneity present (age/condition/relay) — tech-debt
    /// differentials are real, not a uniform assumption.
    WithHeterogeneousAssets,
}

/// Which outputs are valid at the current data level (so callers disclose,
/// never over-claim).
#[derive(Debug, Clone)]
pub struct Capability {
    pub level: DataLevel,
    /// Relative perturbation *shape* + contingency *ranking* (scale-free).
    pub relative_shape: bool,
    /// Absolute MW flows / energy-not-served.
    pub absolute_mw: bool,
    /// The `I²R` loss-gradient field.
    pub loss_gradient: bool,
    /// Old-vs-new asset *differential* effects (vs a uniform assumption).
    pub tech_debt_differential: bool,
    pub notes: Vec<&'static str>,
}

/// Infer the [`Capability`] from how much the import had to estimate.
/// `has_resistance` / `has_asset_heterogeneity` are caller-supplied (the base
/// PyPSA-Eur CSV has neither).
pub fn assess_capability(
    import: &PypsaImport,
    has_resistance: bool,
    has_asset_heterogeneity: bool,
) -> Capability {
    let all_reactance_estimated = import.n_estimated_reactance >= import.grid.edges.len();

    let level = if has_asset_heterogeneity {
        DataLevel::WithHeterogeneousAssets
    } else if has_resistance {
        DataLevel::WithLosses
    } else if !all_reactance_estimated {
        DataLevel::WithReactance
    } else {
        DataLevel::TopologyOnly
    };

    let mut notes = Vec::new();
    if level == DataLevel::TopologyOnly {
        notes.push("reactance & limits are uniform-constant estimates; absolute MW invalid");
        notes.push("relative shape + ranking valid (uniform constants inject no heterogeneity)");
    }
    if !has_asset_heterogeneity {
        notes.push(
            "tech-debt modeled as a uniform constant — sweep it, do not claim a differential",
        );
    }

    Capability {
        level,
        relative_shape: true, // always: the sim runs and the shape is meaningful
        absolute_mw: matches!(
            level,
            DataLevel::WithLosses | DataLevel::WithHeterogeneousAssets
        ),
        loss_gradient: matches!(
            level,
            DataLevel::WithLosses | DataLevel::WithHeterogeneousAssets
        ),
        tech_debt_differential: has_asset_heterogeneity,
        notes,
    }
}

/// Scale every line susceptance by `c` (a uniform conductor-material / age
/// assumption). **Relative-invariant** — see the test.
pub fn scale_susceptance(grid: &Grid, c: f64) -> Grid {
    Grid::new(
        grid.n,
        grid.edges
            .iter()
            .map(|e| Edge::new(e.from, e.to, e.susceptance * c, e.limit))
            .collect(),
    )
}

/// Apply a uniform thermal derate (the "equally outdated network" stress knob):
/// every line `limit` is multiplied by `d ∈ (0,1]`. NOT relative-invariant —
/// sweep `d` as a single sensitivity parameter.
pub fn with_uniform_derate(grid: &Grid, d: f64) -> Grid {
    Grid::new(
        grid.n,
        grid.edges
            .iter()
            .map(|e| Edge::new(e.from, e.to, e.susceptance, e.limit * d))
            .collect(),
    )
}

// ── Age modulators: null vs Gegenhypothese vs modernization-spend ────────────

/// How to assign per-line *age* (`0` = new, `1` = oldest), i.e. which
/// hypothesis about the network's condition heterogeneity to model.
#[derive(Debug, Clone)]
pub enum AgeModel {
    /// **Null hypothesis** — the whole network is uniformly aged at `age`.
    /// Injects no heterogeneity (relative shape unchanged; acts as a uniform
    /// derate stress knob via [`apply_aging`]).
    Uniform(f64),
    /// **Gegenhypothese** — sparse, low-connectivity areas (fewer lines / fewer
    /// substations) are older. Derived from the topology ALONE (degree as the
    /// connectivity-density proxy), so it is computable on today's data and is a
    /// *genuine* heterogeneity that legitimately bends the shape.
    DensityProxy,
    /// **Data-driven** — per-bus "newness" `∈ [0,1]` (1 = freshly modernized)
    /// from a list of modernization projects / money spent per area;
    /// `age = 1 − newness`.
    ModernizationSpend(Vec<f64>),
}

/// Per-line age factors `∈ [0,1]` under the chosen [`AgeModel`].
pub fn edge_age_factors(grid: &Grid, alive: &[bool], model: &AgeModel) -> Vec<f64> {
    let n = grid.n;
    match model {
        AgeModel::Uniform(a) => vec![a.clamp(0.0, 1.0); grid.edges.len()],
        AgeModel::DensityProxy => {
            // Degree = local connectivity density proxy.
            let mut deg = vec![0.0f64; n];
            for (idx, e) in grid.edges.iter().enumerate() {
                if alive[idx] {
                    deg[e.from] += 1.0;
                    deg[e.to] += 1.0;
                }
            }
            let dens: Vec<f64> = grid
                .edges
                .iter()
                .map(|e| 0.5 * (deg[e.from] + deg[e.to]))
                .collect();
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for &d in &dens {
                lo = lo.min(d);
                hi = hi.max(d);
            }
            // Low density → old (age → 1). Flat network → neutral 0.5.
            if hi - lo < 1e-9 {
                vec![0.5; grid.edges.len()]
            } else {
                dens.iter().map(|&d| (hi - d) / (hi - lo)).collect()
            }
        }
        AgeModel::ModernizationSpend(newness) => grid
            .edges
            .iter()
            .map(|e| {
                let nn = 0.5 * (get(newness, e.from) + get(newness, e.to));
                (1.0 - nn).clamp(0.0, 1.0)
            })
            .collect(),
    }
}

fn get(v: &[f64], i: usize) -> f64 {
    v.get(i).copied().unwrap_or(0.0)
}

/// Apply age factors as a thermal derate: `limit *= lerp(1, oldest_derate, age)`
/// (older lines run closer to their limit). `oldest_derate ∈ (0,1]`.
pub fn apply_aging(grid: &Grid, age: &[f64], oldest_derate: f64) -> Grid {
    Grid::new(
        grid.n,
        grid.edges
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let a = age.get(i).copied().unwrap_or(0.0);
                let factor = 1.0 + a * (oldest_derate - 1.0);
                Edge::new(e.from, e.to, e.susceptance, e.limit * factor)
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cascade::{simulate_outage, CascadeConfig};

    fn mesh() -> Grid {
        // A 4-cycle with a chord, every line loaded, tight-ish limits.
        Grid::new(
            4,
            vec![
                Edge::new(0, 1, 1.0, 0.7),
                Edge::new(1, 2, 1.0, 0.7),
                Edge::new(2, 3, 1.0, 0.7),
                Edge::new(3, 0, 1.0, 0.7),
                Edge::new(0, 2, 1.0, 0.7),
            ],
        )
    }

    #[test]
    fn uniform_susceptance_scale_is_relative_invariant() {
        // The "Spain is uniformly outdated" hypothesis, done right: scaling every
        // b by a constant leaves flows, the cascade, and connectivity_loss
        // exactly invariant. Only absolute angles change.
        let g = mesh();
        let p = vec![1.0, 0.0, -1.0, 0.0];
        let cfg = CascadeConfig::default();
        let base = simulate_outage(&g, &p, 0, cfg);

        let scaled = scale_susceptance(&g, 7.3); // arbitrary uniform constant
        let after = simulate_outage(&scaled, &p, 0, cfg);

        assert_eq!(
            base.shape.tripped, after.shape.tripped,
            "cascade set invariant"
        );
        assert!((base.fraction_tripped - after.fraction_tripped).abs() < 1e-12);
        assert!(
            (base.spectral.connectivity_loss() - after.spectral.connectivity_loss()).abs() < 1e-9,
            "connectivity_loss invariant under uniform susceptance scale"
        );
    }

    #[test]
    fn density_proxy_ages_sparse_areas_more() {
        // Dense hub (0-1-2-3) + a sparse low-degree pair (4-5). The sparse edge
        // must come out older than a hub edge (the Gegenhypothese).
        let g = Grid::new(
            6,
            vec![
                Edge::new(0, 1, 1.0, 1.0),
                Edge::new(0, 2, 1.0, 1.0),
                Edge::new(0, 3, 1.0, 1.0),
                Edge::new(1, 2, 1.0, 1.0),
                Edge::new(2, 3, 1.0, 1.0),
                Edge::new(4, 5, 1.0, 1.0), // sparse tail
            ],
        );
        let alive = vec![true; g.edges.len()];
        let age = edge_age_factors(&g, &alive, &AgeModel::DensityProxy);
        assert!(
            age[5] > age[1],
            "sparse edge {} older than hub edge {}",
            age[5],
            age[1]
        );
        assert!((age[5] - 1.0).abs() < 1e-9, "sparsest edge is the oldest");
    }

    #[test]
    fn modernization_spend_lowers_age_where_money_went() {
        let g = mesh();
        // Bus 0 freshly modernized (newness 1.0), rest untouched (0.0).
        let newness = vec![1.0, 0.0, 0.0, 0.0];
        let age = edge_age_factors(
            &g,
            &vec![true; g.edges.len()],
            &AgeModel::ModernizationSpend(newness),
        );
        // Edge 0 (0-1) touches the modernized bus → younger than edge 1 (1-2).
        assert!(
            age[0] < age[1],
            "modernized-adjacent edge younger: {} < {}",
            age[0],
            age[1]
        );
    }

    #[test]
    fn apply_aging_derates_older_lines_more() {
        let g = Grid::new(2, vec![Edge::new(0, 1, 1.0, 100.0)]);
        let aged = apply_aging(&g, &[1.0], 0.5); // oldest → ×0.5
        assert!((aged.edges[0].limit - 50.0).abs() < 1e-9);
        let new = apply_aging(&g, &[0.0], 0.5); // new → ×1.0
        assert!((new.edges[0].limit - 100.0).abs() < 1e-9);
    }

    #[test]
    fn uniform_derate_is_a_stress_knob_not_invariant() {
        // A uniform derate is allowed to change the cascade extent (it tightens
        // every margin together) — that's why it's a swept sensitivity knob, not
        // a free assumption. Tightening must not REDUCE the trip count.
        let g = mesh();
        let p = vec![1.0, 0.0, -1.0, 0.0];
        let cfg = CascadeConfig::default();
        let loose = simulate_outage(&with_uniform_derate(&g, 1.0), &p, 0, cfg)
            .shape
            .n_tripped();
        let tight = simulate_outage(&with_uniform_derate(&g, 0.5), &p, 0, cfg)
            .shape
            .n_tripped();
        assert!(
            tight >= loose,
            "a uniform derate can only widen the cascade"
        );
    }
}
