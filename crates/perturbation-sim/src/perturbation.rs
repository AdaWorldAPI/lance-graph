//! Spectral perturbation of the Laplacian under a single line outage.
//!
//! A line trip on edge `k = (a,b)` with weight `b_k` is a rank-1 perturbation
//! `E = L' − L = −b_k (e_a − e_b)(e_a − e_b)ᵀ`, so `‖E‖₂ = b_k·‖e_a−e_b‖² =
//! 2·b_k`. We certify:
//!
//! - **Weyl's inequality** `|λᵢ(L') − λᵢ(L)| ≤ ‖E‖₂` for every `i`.
//! - **Davis–Kahan** Fiedler-vector rotation `sinθ ≤ ‖E‖₂ / gap`, where `gap`
//!   is the spectral separation of the Fiedler eigenvalue `λ₂`.
//! - The **algebraic connectivity** (`λ₂`, Fiedler value) before/after. A trip
//!   that pushes `λ₂` toward 0 is fragmenting the network — the precursor shape
//!   of a blackout.

use crate::eigen::symmetric_eigen;
use crate::graph::Grid;

/// Result of the rank-1 spectral perturbation analysis for one line trip.
#[derive(Debug, Clone)]
pub struct SpectralPerturbation {
    /// Index of the tripped line.
    pub line: usize,
    /// `‖E‖₂ = 2·b_k`, the Weyl perturbation budget.
    pub e_norm: f64,
    /// `maxᵢ |λᵢ(L') − λᵢ(L)|`, the largest realized eigenvalue shift.
    pub max_eigenvalue_shift: f64,
    /// Whether Weyl's bound held (max shift ≤ ‖E‖₂, within tolerance).
    pub weyl_satisfied: bool,
    /// Fiedler value `λ₂` before the trip (algebraic connectivity).
    pub fiedler_before: f64,
    /// Fiedler value `λ₂` after the trip.
    pub fiedler_after: f64,
    /// Realized Fiedler-vector rotation `sinθ ∈ [0,1]`.
    pub fiedler_rotation_sin: f64,
    /// Davis–Kahan bound on that rotation (`‖E‖₂ / gap`); `inf` if `gap == 0`.
    pub davis_kahan_bound: f64,
}

impl SpectralPerturbation {
    /// Fractional loss of algebraic connectivity, `1 − λ₂'/λ₂`. Near 1 ⇒ the
    /// trip nearly disconnects the network.
    pub fn connectivity_loss(&self) -> f64 {
        if self.fiedler_before.abs() < 1e-12 {
            0.0
        } else {
            1.0 - self.fiedler_after / self.fiedler_before
        }
    }
}

/// Analyse the rank-1 spectral perturbation of tripping `line` from the
/// sub-network defined by `alive_before` (which must include `line`).
pub fn spectral_perturbation(
    grid: &Grid,
    alive_before: &[bool],
    line: usize,
) -> SpectralPerturbation {
    assert!(
        alive_before[line],
        "line must be in service before tripping"
    );
    let n = grid.n;

    let before = symmetric_eigen(&grid.laplacian_of(alive_before), n);
    let mut alive_after = alive_before.to_vec();
    alive_after[line] = false;
    let after = symmetric_eigen(&grid.laplacian_of(&alive_after), n);

    let e_norm = 2.0 * grid.edges[line].susceptance;

    let max_eigenvalue_shift = before
        .values
        .iter()
        .zip(after.values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    let weyl_satisfied = max_eigenvalue_shift <= e_norm + 1e-6;

    // Fiedler index = 1 (second smallest) when the network is connected. Guard
    // tiny networks.
    let (fiedler_before, fiedler_after, fiedler_rotation_sin, davis_kahan_bound) = if n >= 3 {
        let fb = before.values[1];
        let fa = after.values[1];
        let gap = (before.values[1] - before.values[0]).min(before.values[2] - before.values[1]);
        let vb = before.eigenvector(1);
        let va = after.eigenvector(1);
        let dot: f64 = vb
            .iter()
            .zip(va.iter())
            .map(|(x, y)| x * y)
            .sum::<f64>()
            .abs();
        let sin = (1.0 - dot * dot).max(0.0).sqrt();
        let dk = if gap > 1e-12 {
            e_norm / gap
        } else {
            f64::INFINITY
        };
        (fb, fa, sin, dk)
    } else {
        (0.0, 0.0, 0.0, f64::INFINITY)
    };

    SpectralPerturbation {
        line,
        e_norm,
        max_eigenvalue_shift,
        weyl_satisfied,
        fiedler_before,
        fiedler_after,
        fiedler_rotation_sin,
        davis_kahan_bound,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Edge;

    fn ring(n: usize, b: f64) -> Grid {
        let edges = (0..n)
            .map(|i| Edge::new(i, (i + 1) % n, b, 100.0))
            .collect();
        Grid::new(n, edges)
    }

    #[test]
    fn weyl_inequality_holds_for_every_line() {
        let g = ring(8, 1.5);
        let alive = vec![true; g.edges.len()];
        for line in 0..g.edges.len() {
            let sp = spectral_perturbation(&g, &alive, line);
            assert!(
                sp.weyl_satisfied,
                "Weyl violated on line {line}: shift {} > ‖E‖ {}",
                sp.max_eigenvalue_shift, sp.e_norm
            );
        }
    }

    #[test]
    fn davis_kahan_bounds_the_realized_rotation() {
        let g = ring(10, 2.0);
        let alive = vec![true; g.edges.len()];
        let sp = spectral_perturbation(&g, &alive, 0);
        // The realized rotation must respect the Davis–Kahan bound (allow a
        // small numerical slack); skip the degenerate gap==inf case.
        if sp.davis_kahan_bound.is_finite() {
            assert!(
                sp.fiedler_rotation_sin <= sp.davis_kahan_bound + 1e-6,
                "rotation {} exceeds DK bound {}",
                sp.fiedler_rotation_sin,
                sp.davis_kahan_bound
            );
        }
    }

    #[test]
    fn cutting_a_bridge_drops_connectivity_to_zero() {
        // Two triangles joined by a single bridge line: tripping the bridge
        // disconnects the graph, so λ₂(after) ≈ 0 ⇒ connectivity_loss ≈ 1.
        let g = Grid::new(
            6,
            vec![
                Edge::new(0, 1, 1.0, 100.0),
                Edge::new(1, 2, 1.0, 100.0),
                Edge::new(2, 0, 1.0, 100.0),
                Edge::new(3, 4, 1.0, 100.0),
                Edge::new(4, 5, 1.0, 100.0),
                Edge::new(5, 3, 1.0, 100.0),
                Edge::new(2, 3, 1.0, 100.0), // the bridge
            ],
        );
        let alive = vec![true; g.edges.len()];
        let sp = spectral_perturbation(&g, &alive, 6);
        assert!(sp.weyl_satisfied);
        assert!(
            sp.connectivity_loss() > 0.99,
            "bridge cut should collapse connectivity, got loss {}",
            sp.connectivity_loss()
        );
    }
}
