//! DC power flow and Line Outage Distribution Factors (LODF).
//!
//! DC model: with bus angles `θ = L⁺ p` (balanced injections `∑ p = 0`), the
//! flow on line `e = (a,b)` is `f_e = b_e (θ_a − θ_b)`. A line outage
//! redistributes flow; the closed-form redistribution is the LODF.

use crate::eigen::Eigen;
use crate::graph::Grid;

/// Per-line DC flows for a given angle vector `theta`. Dead lines carry 0.
pub fn dc_flows(grid: &Grid, alive: &[bool], theta: &[f64]) -> Vec<f64> {
    assert_eq!(alive.len(), grid.edges.len());
    assert_eq!(theta.len(), grid.n);
    grid.edges
        .iter()
        .enumerate()
        .map(|(idx, e)| {
            if alive[idx] {
                e.susceptance * (theta[e.from] - theta[e.to])
            } else {
                0.0
            }
        })
        .collect()
}

/// Power-Transfer Distribution Factor of line `e` for an injection that pushes
/// one unit from bus `c` to bus `d`, given the dense pseudo-inverse `x = L⁺`.
fn ptdf_edge(x: &[f64], n: usize, e: &crate::graph::Edge, c: usize, d: usize) -> f64 {
    let a = e.from;
    let b = e.to;
    e.susceptance * (x[a * n + c] - x[a * n + d] - x[b * n + c] + x[b * n + d])
}

/// Line Outage Distribution Factor `LODF[e,k]`: the fraction of line `k`'s
/// pre-outage flow that shifts onto line `e` when line `k` trips.
///
/// `f_e(after) ≈ f_e(before) + LODF[e,k]·f_k(before)`. Returns `None` if line
/// `k`'s self-PTDF is ≈ 1 (the outage would island its endpoints — no finite
/// redistribution). `eig` is the decomposition of the *pre-outage* Laplacian.
pub fn lodf(grid: &Grid, eig: &Eigen, e: usize, k: usize, rel_tol: f64) -> Option<f64> {
    let n = grid.n;
    let x = eig.pseudo_inverse(rel_tol);
    let ek = &grid.edges[k];
    let ptdf_kk = ptdf_edge(&x, n, ek, ek.from, ek.to);
    let denom = 1.0 - ptdf_kk;
    if denom.abs() < 1e-9 {
        return None;
    }
    let ee = &grid.edges[e];
    let ptdf_ek = ptdf_edge(&x, n, ee, ek.from, ek.to);
    Some(ptdf_ek / denom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eigen::symmetric_eigen;
    use crate::graph::Edge;

    #[test]
    fn lodf_matches_full_recompute_single_trip() {
        // Triangle: redistributing line (0,1)'s flow when it trips must match a
        // full angle recompute on the surviving two-line network.
        let g = Grid::new(
            3,
            vec![
                Edge::new(0, 1, 1.0, 100.0),
                Edge::new(1, 2, 1.0, 100.0),
                Edge::new(0, 2, 1.0, 100.0),
            ],
        );
        let p = vec![1.0, 0.0, -1.0]; // inject at 0, withdraw at 2
        let tol = 1e-9;

        let alive_all = vec![true; 3];
        let eig0 = symmetric_eigen(&g.laplacian_of(&alive_all), 3);
        let theta0 = eig0.pseudo_apply(&p, tol);
        let f0 = dc_flows(&g, &alive_all, &theta0);

        let k = 0usize; // trip line (0,1)
        let l = lodf(&g, &eig0, 1, k, tol).expect("finite lodf");
        let predicted_f1 = f0[1] + l * f0[k];

        // Full recompute with line 0 dead.
        let mut alive = alive_all.clone();
        alive[k] = false;
        let eig1 = symmetric_eigen(&g.laplacian_of(&alive), 3);
        let theta1 = eig1.pseudo_apply(&p, tol);
        let f1 = dc_flows(&g, &alive, &theta1);

        assert!(
            (predicted_f1 - f1[1]).abs() < 1e-7,
            "LODF predicted {predicted_f1}, recompute {}",
            f1[1]
        );
    }
}
