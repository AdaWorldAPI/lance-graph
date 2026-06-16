//! Perturbation-agnostic resilience certificate — read the field ONCE in the
//! self-inverse `L⁺` reference, never replay a perturbation.
//!
//! The methodological pivot: instead of predicting the cascade of *one specific*
//! trip (which overfits a contingency set and goes circular — see PAPER §4.9.1),
//! we read **resilience** straight off the spectrum. Because the Laplacian
//! pseudoinverse `L⁺` is the inverse map injection → angle-field, its spectral
//! invariants integrate the system's response over the WHOLE perturbation
//! ensemble at once:
//!
//! - **algebraic connectivity** `λ₂` — the worst-case margin (smallest non-trivial
//!   mode); the larger, the more any perturbation is absorbed.
//! - **Kirchhoff index** `Kf = n · Σ_{k≥2} 1/λ_k = n · trace(L⁺)` — the total
//!   effective resistance, i.e. the mean squared angle response to a *random*
//!   balanced injection. The self-inverse reference in one scalar: `1/λ_k` are the
//!   eigenvalues of `L⁺`, summed.
//!
//! Both are read once from one eigensolve — there is no "run the same perturbation
//! again". Raising `λ₂` (lowering `Kf`) hardens the system against the NEXT,
//! unknown perturbation by construction, not against the last one.
//!
//! Honest scope: `λ₂`/`Kf` are exact spectral invariants [G]; whether raising the
//! worst-case margin reduces *operational* cascades is the Braess caveat (PAPER
//! §4.4) — adding connectivity can worsen a specific flow cascade even as it
//! raises `λ₂`, so margin and cascade must be co-designed.

/// Algebraic connectivity `λ₂` — the second-smallest eigenvalue (worst-case
/// resilience margin). `eigenvalues` must be ascending (as [`crate::symmetric_eigen`]
/// returns). Returns 0 for a disconnected/trivial graph.
pub fn algebraic_connectivity(eigenvalues: &[f64]) -> f64 {
    eigenvalues.get(1).copied().unwrap_or(0.0)
}

/// Kirchhoff index `Kf = n · Σ_{λ_k > tol} 1/λ_k = n · trace(L⁺)` — total
/// effective resistance, the response integrated over all balanced injections.
/// The `1/λ_k` are exactly the non-trivial eigenvalues of the self-inverse
/// reference `L⁺`. Lower = more resilient. `tol` drops the trivial zero mode (and
/// any near-zero mode of a near-disconnected graph).
pub fn kirchhoff_index(eigenvalues: &[f64], tol: f64) -> f64 {
    let n = eigenvalues.len();
    if n == 0 {
        return 0.0;
    }
    let inv_sum: f64 = eigenvalues
        .iter()
        .filter(|&&l| l > tol)
        .map(|&l| 1.0 / l)
        .sum();
    n as f64 * inv_sum
}

/// A compartment's perturbation-agnostic resilience certificate.
#[derive(Debug, Clone, Copy)]
pub struct Resilience {
    /// Number of nodes in the compartment.
    pub n: usize,
    /// Worst-case margin `λ₂`.
    pub lambda2: f64,
    /// Total effective resistance `Kf` (lower = more resilient).
    pub kirchhoff: f64,
    /// Number of non-trivial modes counted (connectivity sanity).
    pub modes: usize,
}

impl Resilience {
    /// Build from an ascending eigenvalue list (one eigensolve, no perturbation).
    pub fn from_eigenvalues(eigenvalues: &[f64], tol: f64) -> Self {
        Self {
            n: eigenvalues.len(),
            lambda2: algebraic_connectivity(eigenvalues),
            kirchhoff: kirchhoff_index(eigenvalues, tol),
            modes: eigenvalues.iter().filter(|&&l| l > tol).count(),
        }
    }

    /// Mean effective resistance per node-pair `Kf / C(n,2)` — a size-normalized
    /// resilience density comparable across compartments of different sizes.
    pub fn mean_resistance(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        let pairs = (self.n * (self.n - 1)) as f64 / 2.0;
        self.kirchhoff / pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{symmetric_eigen, Edge, Grid};

    /// Complete graph K_n: eigenvalues are [0, n, n, …, n] ⇒ Kf = n·(n−1)/n = n−1.
    fn complete(n: usize) -> Grid {
        let mut e = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                e.push(Edge::new(i, j, 1.0, 1.0));
            }
        }
        Grid::new(n, e)
    }

    #[test]
    fn kirchhoff_of_complete_graph_is_n_minus_1() {
        for n in [3usize, 5, 8] {
            let g = complete(n);
            let eig = symmetric_eigen(&g.laplacian_of(&vec![true; g.edges.len()]), g.n);
            let kf = kirchhoff_index(&eig.values, 1e-9);
            assert!(
                (kf - (n as f64 - 1.0)).abs() < 1e-6,
                "Kf(K_{n}) = {kf}, expected {}",
                n - 1
            );
            // λ₂ of K_n is n.
            assert!((algebraic_connectivity(&eig.values) - n as f64).abs() < 1e-6);
        }
    }

    #[test]
    fn adding_an_edge_lowers_kirchhoff_and_raises_lambda2() {
        // Path 0–1–2–3 vs the same with a chord 0–3 (more connected = more resilient).
        let path = Grid::new(
            4,
            vec![
                Edge::new(0, 1, 1.0, 1.0),
                Edge::new(1, 2, 1.0, 1.0),
                Edge::new(2, 3, 1.0, 1.0),
            ],
        );
        let mut chorded = path.edges.clone();
        chorded.push(Edge::new(0, 3, 1.0, 1.0));
        let chorded = Grid::new(4, chorded);

        let r_path = Resilience::from_eigenvalues(
            &symmetric_eigen(&path.laplacian_of(&[true; 3]), 4).values,
            1e-9,
        );
        let r_chord = Resilience::from_eigenvalues(
            &symmetric_eigen(&chorded.laplacian_of(&[true; 4]), 4).values,
            1e-9,
        );
        assert!(
            r_chord.kirchhoff < r_path.kirchhoff,
            "more edges ⇒ lower Kf: {} < {}",
            r_chord.kirchhoff,
            r_path.kirchhoff
        );
        assert!(
            r_chord.lambda2 > r_path.lambda2,
            "more edges ⇒ larger λ₂ margin"
        );
    }
}
