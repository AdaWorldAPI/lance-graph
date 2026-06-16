//! Basin / HHTL tiering — the **field** half (Raumgewinn) and the bridge to the
//! **local** half (infight) already in [`crate::cascade`].
//!
//! Four named tools, kept deliberately distinct (see `METHODS.md` — do not
//! conflate them):
//!
//! 1. **Kron reduction** ([`kron_reduce`]) — the Schur complement of the
//!    Laplacian. Eliminates a basin's interior buses to a boundary-only
//!    equivalent that *exactly preserves effective resistance* between the
//!    boundary (Dörfler–Bullo 2013). This is "a basin as one super-node with
//!    ports" — the HEEL/HIP tiering operator and the cross-border model.
//! 2. **Effective resistance** ([`effective_resistance`]) — `R_ij =
//!    (eᵢ−eⱼ)ᵀ L⁺ (eᵢ−eⱼ)`, the genuine *electrical* distance metric. The
//!    [`spectral_embedding`] from low eigenvectors gives electrical coordinates
//!    — the correct substrate for a Morton/HHTL tiling (geography is NOT).
//! 3. **Cheeger sweep** ([`cheeger_sweep`]) — the normalized spectral gap `μ₂`
//!    and the Fiedler sweep-cut conductance `φ`, satisfying `μ₂/2 ≤ h(G) ≤ φ ≤
//!    √(2μ₂)`. This is the exchange rate between the field eigenvalue
//!    (Raumgewinn) and the cut (infight).
//! 4. **Go-meta score** ([`infight_vs_raumgewinn`]) — runs the cascade and
//!    classifies a contingency as *infight* (local collapse dominates) vs
//!    *Raumgewinn* (global connectivity collapse dominates).
//!
//! The combinatorial Fiedler value `λ₂` (in [`crate::perturbation`]) is the
//! object Weyl/Davis–Kahan perturb; the **normalized** `μ₂` here is the object
//! Cheeger bounds. They are different eigenvalues of different operators — the
//! `METHODS.md` anti-dilution note spells out why both exist.

use crate::cascade::{simulate_outage, CascadeConfig};
use crate::eigen::symmetric_eigen;
use crate::graph::Grid;

// ── Kron reduction (Schur complement) ───────────────────────────────────────

/// A Kron-reduced (boundary-only) network: a valid loopy Laplacian on the
/// boundary buses, electrically equivalent to the original.
#[derive(Debug, Clone)]
pub struct KronReduced {
    /// Original bus indices, in the order they appear in `l_red`.
    pub boundary: Vec<usize>,
    /// Number of boundary buses (= `boundary.len()`).
    pub n: usize,
    /// Reduced Laplacian, row-major `n×n`.
    pub l_red: Vec<f64>,
}

impl KronReduced {
    /// Position of an original bus index within the reduced network.
    pub fn pos(&self, original: usize) -> Option<usize> {
        self.boundary.iter().position(|&b| b == original)
    }
}

/// Kron-reduce `grid` (over `alive` edges) onto the `boundary` bus set:
/// `L_red = L_BB − L_BI · L_II⁻¹ · L_IB`. Interior buses are everything not in
/// `boundary`. `L_II` is positive-definite for a connected network with a
/// non-empty boundary, so the inverse exists.
pub fn kron_reduce(grid: &Grid, alive: &[bool], boundary: &[usize]) -> KronReduced {
    let n = grid.n;
    let l = grid.laplacian_of(alive);
    let mut is_b = vec![false; n];
    for &b in boundary {
        is_b[b] = true;
    }
    let bidx: Vec<usize> = (0..n).filter(|&x| is_b[x]).collect();
    let iidx: Vec<usize> = (0..n).filter(|&x| !is_b[x]).collect();
    let (nb, ni) = (bidx.len(), iidx.len());

    let l_bb = submatrix(&l, n, &bidx, &bidx);
    let l_red = if ni == 0 {
        l_bb
    } else {
        let l_ii = submatrix(&l, n, &iidx, &iidx);
        let l_ib = submatrix(&l, n, &iidx, &bidx);
        let l_bi = submatrix(&l, n, &bidx, &iidx);
        let l_ii_inv = symmetric_eigen(&l_ii, ni).pseudo_inverse(1e-12);
        let m = matmul(&l_bi, nb, ni, &l_ii_inv, ni, ni); // nb×ni
        let mii = matmul(&m, nb, ni, &l_ib, ni, nb); // nb×nb
        l_bb.iter().zip(mii.iter()).map(|(a, b)| a - b).collect()
    };
    KronReduced {
        boundary: bidx,
        n: nb,
        l_red,
    }
}

// ── Effective resistance + spectral (electrical) embedding ───────────────────

/// Dense pseudo-inverse `L⁺` of the network Laplacian (row-major `n×n`).
pub fn laplacian_pinv(grid: &Grid, alive: &[bool], rel_tol: f64) -> Vec<f64> {
    symmetric_eigen(&grid.laplacian_of(alive), grid.n).pseudo_inverse(rel_tol)
}

/// Effective-resistance distance `R_ij = L⁺_ii + L⁺_jj − 2·L⁺_ij` (a metric).
pub fn effective_resistance(l_plus: &[f64], n: usize, i: usize, j: usize) -> f64 {
    l_plus[i * n + i] + l_plus[j * n + j] - 2.0 * l_plus[i * n + j]
}

/// Electrical coordinates: `dims` low Laplacian eigenvectors (skipping the
/// constant `λ₀`). Z-order/Morton-tiling buses by these coordinates gives an
/// HHTL address that respects electrical — not geographic — distance.
pub fn spectral_embedding(grid: &Grid, alive: &[bool], dims: usize) -> Vec<Vec<f64>> {
    let n = grid.n;
    let eig = symmetric_eigen(&grid.laplacian_of(alive), n);
    let d = dims.min(n.saturating_sub(1));
    (0..n)
        .map(|node| (1..=d).map(|k| eig.vectors[node * n + k]).collect())
        .collect()
}

/// Algebraic connectivity `λ₂` of the subgraph induced by `members` (edges with
/// both endpoints in `members`). Returns 0 for `< 2` members. The per-tier
/// "field eigenvalue" of a basin; Cauchy interlacing relates it to the parent.
pub fn basin_lambda2(grid: &Grid, alive: &[bool], members: &[usize]) -> f64 {
    let k = members.len();
    if k < 2 {
        return 0.0;
    }
    let mut remap = std::collections::HashMap::new();
    for (new, &old) in members.iter().enumerate() {
        remap.insert(old, new);
    }
    let mut sub = vec![0.0; k * k];
    for (idx, e) in grid.edges.iter().enumerate() {
        if !alive[idx] {
            continue;
        }
        if let (Some(&a), Some(&b)) = (remap.get(&e.from), remap.get(&e.to)) {
            let w = e.susceptance;
            sub[a * k + a] += w;
            sub[b * k + b] += w;
            sub[a * k + b] -= w;
            sub[b * k + a] -= w;
        }
    }
    symmetric_eigen(&sub, k)
        .values
        .get(1)
        .copied()
        .unwrap_or(0.0)
}

// ── Cheeger: the field-vs-cut exchange rate ──────────────────────────────────

/// Normalized spectral gap `μ₂`, the Fiedler sweep-cut conductance `φ`, and the
/// Cheeger sandwich bounds.
#[derive(Debug, Clone)]
pub struct Cheeger {
    /// Second eigenvalue of the normalized Laplacian `D^{-1/2} L D^{-1/2}`.
    pub mu2: f64,
    /// Best sweep-cut conductance `φ` (an upper bound on `h(G)`).
    pub conductance: f64,
    /// Cheeger lower bound `μ₂/2` (`≤ h(G)`).
    pub lower: f64,
    /// Cheeger upper bound `√(2·μ₂)` (`≥ φ`).
    pub upper: f64,
    /// The sweep partition (`true` = on the small side of the best cut).
    pub partition: Vec<bool>,
}

/// Compute the normalized gap `μ₂` and the Fiedler sweep cut.
pub fn cheeger_sweep(grid: &Grid, alive: &[bool]) -> Cheeger {
    let n = grid.n;
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (idx, e) in grid.edges.iter().enumerate() {
        if alive[idx] {
            adj[e.from].push((e.to, e.susceptance));
            adj[e.to].push((e.from, e.susceptance));
        }
    }
    let deg: Vec<f64> = (0..n)
        .map(|i| adj[i].iter().map(|&(_, w)| w).sum())
        .collect();

    // Normalized Laplacian.
    let l = grid.laplacian_of(alive);
    let mut ln = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let dij = (deg[i] * deg[j]).sqrt();
            ln[i * n + j] = if dij > 0.0 {
                l[i * n + j] / dij
            } else if i == j {
                1.0
            } else {
                0.0
            };
        }
    }
    let eig = symmetric_eigen(&ln, n);
    let mu2 = eig.values.get(1).copied().unwrap_or(0.0).max(0.0);

    // Generalized Fiedler x = D^{-1/2} u₂, sweep over its ordering.
    let x: Vec<f64> = (0..n)
        .map(|i| {
            let u = eig.vectors[i * n + 1.min(n.saturating_sub(1))];
            if deg[i] > 0.0 {
                u / deg[i].sqrt()
            } else {
                u
            }
        })
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));

    let total_vol: f64 = deg.iter().sum();
    let mut in_s = vec![false; n];
    let (mut cut, mut vol) = (0.0_f64, 0.0_f64);
    let (mut best, mut best_k) = (f64::INFINITY, 0usize);
    for (k, &v) in order.iter().enumerate().take(n.saturating_sub(1)) {
        in_s[v] = true;
        vol += deg[v];
        for &(nb, w) in &adj[v] {
            if in_s[nb] {
                cut -= w;
            } else {
                cut += w;
            }
        }
        let denom = vol.min(total_vol - vol);
        if denom > 0.0 {
            let phi = cut / denom;
            if phi < best {
                best = phi;
                best_k = k;
            }
        }
    }
    let conductance = if best.is_finite() { best } else { 0.0 };
    let partition: Vec<bool> = {
        let mut p = vec![false; n];
        for &v in order.iter().take(best_k + 1) {
            p[v] = true;
        }
        p
    };

    Cheeger {
        mu2,
        conductance,
        lower: mu2 / 2.0,
        upper: (2.0 * mu2).sqrt(),
        partition,
    }
}

// ── The Go-meta: infight (local collapse) vs Raumgewinn (field) ──────────────

/// Which value system a contingency moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    /// Local collapse dominates: the cascade trips many lines but global
    /// connectivity holds (a contained tactical fight).
    Infight,
    /// Field collapse dominates: few trips but algebraic connectivity drops
    /// sharply — a global cut materializes (territory lost).
    Raumgewinn,
    /// Comparable on both scales.
    Balanced,
}

/// The two-tier value of a contingency.
#[derive(Debug, Clone)]
pub struct GoScore {
    /// Fraction of lines tripped by the cascade — the *infight* magnitude.
    pub infight: f64,
    /// Algebraic-connectivity loss `1 − λ₂'/λ₂` — the *Raumgewinn* magnitude.
    pub raumgewinn: f64,
    pub regime: Regime,
}

/// Score a seed contingency on both scales by running the cascade once.
pub fn infight_vs_raumgewinn(
    grid: &Grid,
    p: &[f64],
    seed_line: usize,
    cfg: CascadeConfig,
) -> GoScore {
    let r = simulate_outage(grid, p, seed_line, cfg);
    let infight = r.fraction_tripped;
    let raumgewinn = r.spectral.connectivity_loss().clamp(0.0, 1.0);
    let regime = if raumgewinn > infight + 0.05 {
        Regime::Raumgewinn
    } else if infight > raumgewinn + 0.05 {
        Regime::Infight
    } else {
        Regime::Balanced
    };
    GoScore {
        infight,
        raumgewinn,
        regime,
    }
}

// ── contingency feature vector (the 4 methods as measurement variables) ──────

/// One contingency's reading on each method — the variables a statistician
/// feeds to ICC / Pearson / Spearman / Cronbach (see `METHODS.md` §Statistics).
/// Each field is a *different property of the same Laplacian operator*, so they
/// double as mutual control variables in partial correlation.
#[derive(Debug, Clone, Copy)]
pub struct ContingencyFeatures {
    /// Weyl: `|Δλ₂|`, the field-eigenvalue (algebraic-connectivity) shift.
    pub d_lambda2: f64,
    /// Davis–Kahan: Fiedler-vector rotation `sinθ` (how the partition turned).
    pub dk_rotation: f64,
    /// Cheeger: `Δφ = φ_after − φ_before`, the change in sweep-cut conductance.
    pub d_conductance: f64,
    /// Cascade: fraction of lines tripped — the *infight* (local collapse).
    pub infight: f64,
    /// `1 − λ₂'/λ₂`, the *Raumgewinn* (global connectivity collapse).
    pub raumgewinn: f64,
}

impl ContingencyFeatures {
    /// As a `[f64; 5]` row, for stacking into a feature matrix.
    pub fn as_row(&self) -> [f64; 5] {
        [
            self.d_lambda2,
            self.dk_rotation,
            self.d_conductance,
            self.infight,
            self.raumgewinn,
        ]
    }
}

/// Extract all four methods' properties for one seed contingency. Deterministic.
pub fn contingency_features(
    grid: &Grid,
    p: &[f64],
    seed_line: usize,
    cfg: CascadeConfig,
) -> ContingencyFeatures {
    let all = vec![true; grid.edges.len()];
    let c_before = cheeger_sweep(grid, &all).conductance;
    let mut after = all.clone();
    after[seed_line] = false;
    let c_after = cheeger_sweep(grid, &after).conductance;

    let r = simulate_outage(grid, p, seed_line, cfg);
    let s = &r.spectral;
    ContingencyFeatures {
        d_lambda2: (s.fiedler_before - s.fiedler_after).abs(),
        dk_rotation: s.fiedler_rotation_sin,
        d_conductance: c_after - c_before,
        infight: r.fraction_tripped,
        raumgewinn: s.connectivity_loss().clamp(0.0, 1.0),
    }
}

// ── small dense-matrix helpers ───────────────────────────────────────────────

fn submatrix(mat: &[f64], n: usize, rows: &[usize], cols: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0; rows.len() * cols.len()];
    for (ri, &r) in rows.iter().enumerate() {
        for (ci, &c) in cols.iter().enumerate() {
            out[ri * cols.len() + ci] = mat[r * n + c];
        }
    }
    out
}

fn matmul(a: &[f64], ar: usize, ac: usize, b: &[f64], br: usize, bc: usize) -> Vec<f64> {
    assert_eq!(ac, br);
    let mut out = vec![0.0; ar * bc];
    for i in 0..ar {
        for k in 0..ac {
            let aik = a[i * ac + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..bc {
                out[i * bc + j] += aik * b[k * bc + j];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Edge;

    fn triangle_pair_bridge() -> Grid {
        // Two triangles (0,1,2) & (3,4,5) joined by bridge (2,3).
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
    fn kron_reduction_is_a_valid_laplacian() {
        let g = triangle_pair_bridge();
        let alive = vec![true; g.edges.len()];
        let kr = kron_reduce(&g, &alive, &[0, 1, 3, 4]); // keep 4 boundary buses
                                                         // Row sums ≈ 0 and PSD.
        for i in 0..kr.n {
            let s: f64 = (0..kr.n).map(|j| kr.l_red[i * kr.n + j]).sum();
            assert!(s.abs() < 1e-8, "reduced row {i} sum {s}");
        }
        let lam = symmetric_eigen(&kr.l_red, kr.n).values;
        assert!(
            lam.iter().all(|&l| l > -1e-8),
            "reduced Laplacian must be PSD"
        );
    }

    #[test]
    fn kron_preserves_effective_resistance() {
        // Dörfler–Bullo: R between boundary buses is identical before/after.
        let g = triangle_pair_bridge();
        let alive = vec![true; g.edges.len()];
        let boundary = [0usize, 1, 3, 4];
        let kr = kron_reduce(&g, &alive, &boundary);

        let full_pinv = laplacian_pinv(&g, &alive, 1e-12);
        let red_pinv = symmetric_eigen(&kr.l_red, kr.n).pseudo_inverse(1e-12);

        for &i in &boundary {
            for &j in &boundary {
                if i >= j {
                    continue;
                }
                let r_full = effective_resistance(&full_pinv, g.n, i, j);
                let (pi, pj) = (kr.pos(i).unwrap(), kr.pos(j).unwrap());
                let r_red = effective_resistance(&red_pinv, kr.n, pi, pj);
                assert!(
                    (r_full - r_red).abs() < 1e-7,
                    "R({i},{j}) full {r_full} != reduced {r_red}"
                );
            }
        }
    }

    #[test]
    fn cauchy_interlacing_of_the_interior_block() {
        // Eigenvalues of a principal submatrix (the grounded interior block)
        // interlace the full Laplacian's: λ_k(L) ≤ λ_k(L_II) ≤ λ_{k+(n-m)}(L).
        let g = triangle_pair_bridge();
        let n = g.n;
        let l = g.laplacian();
        let interior = [1usize, 2, 3, 4]; // m = 4
        let m = interior.len();
        let l_full = symmetric_eigen(&l, n).values;
        let l_sub = symmetric_eigen(&submatrix(&l, n, &interior, &interior), m).values;
        for k in 0..m {
            assert!(l_full[k] <= l_sub[k] + 1e-9, "lower interlacing at {k}");
            assert!(
                l_sub[k] <= l_full[k + (n - m)] + 1e-9,
                "upper interlacing at {k}"
            );
        }
    }

    #[test]
    fn effective_resistance_is_a_metric() {
        let g = triangle_pair_bridge();
        let alive = vec![true; g.edges.len()];
        let x = laplacian_pinv(&g, &alive, 1e-12);
        // Triangle inequality across the bridge.
        let (i, j, k) = (0usize, 5usize, 2usize);
        let rij = effective_resistance(&x, g.n, i, j);
        let rik = effective_resistance(&x, g.n, i, k);
        let rkj = effective_resistance(&x, g.n, k, j);
        assert!(rij <= rik + rkj + 1e-9, "R metric triangle inequality");
        assert!(rij > 0.0);
    }

    #[test]
    fn cheeger_sandwich_holds() {
        let ring6 = Grid::new(
            6,
            (0..6)
                .map(|i| Edge::new(i, (i + 1) % 6, 1.0, 1e6))
                .collect(),
        );
        for g in [triangle_pair_bridge(), ring6] {
            let alive = vec![true; g.edges.len()];
            let c = cheeger_sweep(&g, &alive);
            assert!(
                c.lower <= c.conductance + 1e-9,
                "μ₂/2 ≤ φ: {} ≤ {}",
                c.lower,
                c.conductance
            );
            assert!(
                c.conductance <= c.upper + 1e-9,
                "φ ≤ √(2μ₂): {} ≤ {}",
                c.conductance,
                c.upper
            );
        }
    }

    #[test]
    fn bridge_cut_is_raumgewinn_not_infight() {
        // Tripping the single bridge collapses global connectivity (territory)
        // while tripping almost nothing locally → Raumgewinn regime.
        let g = triangle_pair_bridge();
        let p = vec![1.0, 0.0, 0.0, 0.0, 0.0, -1.0];
        let score = infight_vs_raumgewinn(&g, &p, 6, CascadeConfig::default());
        assert_eq!(score.regime, Regime::Raumgewinn, "{score:?}");
        assert!(score.raumgewinn > 0.9);
    }

    #[test]
    fn contingency_features_are_sane() {
        let g = triangle_pair_bridge();
        let p = vec![1.0, 0.0, 0.0, 0.0, 0.0, -1.0];
        let f = contingency_features(&g, &p, 6, CascadeConfig::default());
        assert!(f.raumgewinn > 0.9, "bridge trip is a territorial cut");
        assert_eq!(f.as_row().len(), 5);
        assert!(f.as_row().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn basin_lambda2_positive_for_connected_basin() {
        let g = triangle_pair_bridge();
        let alive = vec![true; g.edges.len()];
        // One triangle is a connected basin → λ₂ > 0.
        assert!(basin_lambda2(&g, &alive, &[0, 1, 2]) > 1e-6);
        // Two non-adjacent buses → no internal edge → λ₂ = 0.
        assert_eq!(basin_lambda2(&g, &alive, &[0, 5]), 0.0);
    }
}
