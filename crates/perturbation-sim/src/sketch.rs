//! Fast-sketch synergy (**PROTOTYPE**) — the VSA/Hamming machinery applied to
//! the field tier:
//!
//! - [`resistance_sketch`] — **Spielman–Srivastava** (2008): effective
//!   resistance from `k = O(log n / ε²)` random ±1 sign-projections of the
//!   incidence-weighted Laplacian. `R_eff(u,v) ≈ ‖z_u − z_v‖²` where `z` is the
//!   per-node sketch. The random ±1 rows *are* a `vsa_bundle` of sign
//!   fingerprints; `‖z_u−z_v‖²` is the JL distance readout. Exact in
//!   expectation (`Mᵀ M = L`), JL-concentrated.
//! - [`walsh_pyramid_energy`] — the Morton/Walsh pyramid screen: the
//!   Walsh–Hadamard transform of a node field, split into coarse (low-sequency
//!   = global / **Raumgewinn** / collapse) vs fine (high-sequency = local /
//!   **infight**) dyadic levels. One `O(N log N)` pass (sign side = XOR/`bind`).
//!
//! **Honesty (PROTOTYPE):** this uses the dense `L⁺` (eigensolve), so it is a
//! *correctness/accuracy* demonstration of the sketch, **not** the asymptotic
//! speed win — that needs a fast Laplacian solver (Spielman–Teng), which this
//! crate does not have. The value is at continental scale where the exact
//! `O(n³)` eigensolve dies; at the demo `n` the exact path is faster. The
//! Walsh basis equals the graph eigenbasis only on hypercube-structured graphs,
//! so the pyramid energy is a *screen*, not an exact partition.

use crate::basin::laplacian_pinv;
use crate::graph::Grid;

/// Deterministic ±1 generator (SplitMix64).
struct Signs(u64);
impl Signs {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        if z & 1 == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

/// Per-node resistance sketch: `z` is `dim×n` row-major; node `u`'s embedding is
/// column `u`. `resistance(u,v) ≈ R_eff(u,v)`.
#[derive(Debug, Clone)]
pub struct ResistanceSketch {
    pub dim: usize,
    pub n: usize,
    pub z: Vec<f64>,
}

impl ResistanceSketch {
    /// JL estimate of the effective resistance between buses `u` and `v`.
    pub fn resistance(&self, u: usize, v: usize) -> f64 {
        (0..self.dim)
            .map(|i| {
                let d = self.z[i * self.n + u] - self.z[i * self.n + v];
                d * d
            })
            .sum()
    }
}

/// Build the Spielman–Srivastava resistance sketch with `k` random ±1
/// projections (deterministic from `seed`). `Z = Q · W^{1/2}B · L⁺`, so
/// `‖z_u − z_v‖²` is an unbiased JL estimate of `(e_u−e_v)ᵀ L⁺ (e_u−e_v)`.
pub fn resistance_sketch(
    grid: &Grid,
    alive: &[bool],
    k: usize,
    seed: u64,
    rel_tol: f64,
) -> ResistanceSketch {
    let n = grid.n;
    let l_plus = laplacian_pinv(grid, alive, rel_tol);
    let mut z = vec![0.0; k * n];
    let scale = 1.0 / (k as f64).sqrt();
    let mut rng = Signs(seed);
    for i in 0..k {
        for (idx, e) in grid.edges.iter().enumerate() {
            if !alive[idx] {
                continue;
            }
            let coef = rng.next() * e.susceptance.sqrt() * scale;
            let (a, b) = (e.from, e.to);
            let row = i * n;
            for col in 0..n {
                z[row + col] += coef * (l_plus[a * n + col] - l_plus[b * n + col]);
            }
        }
    }
    ResistanceSketch { dim: k, n, z }
}

// ── Walsh / Morton pyramid screen ────────────────────────────────────────────

/// In-place fast Walsh–Hadamard transform (length must be a power of two).
pub fn fwht(a: &mut [f64]) {
    let n = a.len();
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let (x, y) = (a[j], a[j + h]);
                a[j] = x + y;
                a[j + h] = x - y;
            }
            i += 2 * h;
        }
        h *= 2;
    }
}

/// Walsh energy per dyadic (pyramid) level + the coarse fraction.
#[derive(Debug, Clone)]
pub struct WalshEnergy {
    /// Energy in each dyadic level (`per_level[0]` = DC).
    pub per_level: Vec<f64>,
    /// Fraction of energy in the coarse (low-sequency) half of the levels —
    /// high ⇒ a global/field (Raumgewinn) perturbation; low ⇒ local (infight).
    pub coarse_fraction: f64,
}

/// Walsh–Hadamard pyramid energy of a per-node scalar field. Pads to the next
/// power of two; groups coefficients into dyadic levels (the Morton/quadtree
/// pyramid levels) by the highest set bit of the coefficient index.
pub fn walsh_pyramid_energy(field: &[f64]) -> WalshEnergy {
    let mut n = 1usize;
    while n < field.len().max(1) {
        n <<= 1;
    }
    let mut a = vec![0.0; n];
    a[..field.len()].copy_from_slice(field);
    fwht(&mut a);

    let levels = (n as f64).log2() as usize + 1;
    let mut per = vec![0.0; levels];
    for (i, &coef) in a.iter().enumerate() {
        let lvl = if i == 0 {
            0
        } else {
            (usize::BITS - i.leading_zeros()) as usize
        };
        per[lvl] += coef * coef;
    }
    let total: f64 = per.iter().sum();
    let half = levels / 2;
    let coarse: f64 = per[..=half].iter().sum();
    WalshEnergy {
        per_level: per,
        coarse_fraction: if total > 0.0 { coarse / total } else { 0.0 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basin::{effective_resistance, laplacian_pinv};
    use crate::graph::{Edge, Grid};

    fn triangle_pair_bridge() -> Grid {
        Grid::new(
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
        )
    }

    #[test]
    fn sketch_matches_exact_effective_resistance() {
        let g = triangle_pair_bridge();
        let alive = vec![true; g.edges.len()];
        let exact = laplacian_pinv(&g, &alive, 1e-12);
        let sk = resistance_sketch(&g, &alive, 6000, 0xC0FFEE, 1e-12);
        // JL: relative error ~ √(2/k) ≈ 0.018 at k=6000; allow a safe margin.
        for (u, v) in [(0usize, 5usize), (2, 3), (0, 3), (1, 4)] {
            let r_exact = effective_resistance(&exact, g.n, u, v);
            let r_sketch = sk.resistance(u, v);
            let rel = (r_sketch - r_exact).abs() / r_exact;
            assert!(
                rel < 0.12,
                "R({u},{v}) sketch {r_sketch} vs exact {r_exact} (rel {rel})"
            );
        }
    }

    #[test]
    fn fwht_is_an_involution_up_to_scale() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig = a.clone();
        fwht(&mut a);
        fwht(&mut a);
        for (x, y) in a.iter().zip(orig.iter()) {
            assert!((x / 8.0 - y).abs() < 1e-9, "fwht∘fwht = N·I");
        }
    }

    #[test]
    fn smooth_field_is_coarse_spike_is_fine() {
        // A monotone ramp concentrates Walsh energy at coarse (low) levels;
        // a single-node spike spreads it to fine levels.
        let ramp: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let mut spike = vec![0.0; 8];
        spike[3] = 1.0;
        let cr = walsh_pyramid_energy(&ramp).coarse_fraction;
        let cs = walsh_pyramid_energy(&spike).coarse_fraction;
        assert!(cr > cs, "ramp coarse {cr} should exceed spike coarse {cs}");
    }
}
