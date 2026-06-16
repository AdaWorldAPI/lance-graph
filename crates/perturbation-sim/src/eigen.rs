//! Cyclic-Jacobi eigensolver for real **symmetric** matrices.
//!
//! Deterministic, zero-dep, accurate to ~1e-10 for the modest dense matrices
//! (`n` up to a few hundred) this crate targets. The same Jacobi approach is
//! used by `ndarray::hpc::pillar::cov_high_d` for SPD log-maps, so results are
//! consistent with the workspace's existing certification harness.

/// Eigendecomposition of a symmetric matrix: `A = V · diag(values) · Vᵀ`.
///
/// `values` are ascending. `vectors[k*n + j]` is component `k` of the `j`-th
/// eigenvector (i.e. eigenvectors are stored as columns).
#[derive(Debug, Clone)]
pub struct Eigen {
    pub n: usize,
    pub values: Vec<f64>,
    pub vectors: Vec<f64>,
}

impl Eigen {
    /// Column `j` (the `j`-th eigenvector) as an owned vector.
    pub fn eigenvector(&self, j: usize) -> Vec<f64> {
        (0..self.n).map(|k| self.vectors[k * self.n + j]).collect()
    }

    /// Number of (numerically) zero eigenvalues = number of connected
    /// components of the graph whose Laplacian this decomposes.
    pub fn nullity(&self, rel_tol: f64) -> usize {
        let scale = self.values.last().copied().unwrap_or(0.0).abs().max(1.0);
        self.values
            .iter()
            .filter(|&&l| l.abs() < rel_tol * scale)
            .count()
    }

    /// Apply the Moore–Penrose pseudo-inverse to `p`: returns `L⁺ p`, summing
    /// only over eigenpairs with `|λ| ≥ rel_tol·λ_max`. For a Laplacian and a
    /// balanced injection `p` (∑ p = 0) this is the least-norm (mean-zero)
    /// angle solution `θ`.
    pub fn pseudo_apply(&self, p: &[f64], rel_tol: f64) -> Vec<f64> {
        assert_eq!(p.len(), self.n);
        let scale = self.values.last().copied().unwrap_or(0.0).abs().max(1.0);
        let cutoff = rel_tol * scale;
        let mut out = vec![0.0; self.n];
        for j in 0..self.n {
            let lambda = self.values[j];
            if lambda.abs() < cutoff {
                continue;
            }
            // coeff = (v_j · p) / λ_j
            let mut dot = 0.0;
            for (k, &pk) in p.iter().enumerate() {
                dot += self.vectors[k * self.n + j] * pk;
            }
            let coeff = dot / lambda;
            for (k, ok) in out.iter_mut().enumerate() {
                *ok += coeff * self.vectors[k * self.n + j];
            }
        }
        out
    }

    /// Dense Moore–Penrose pseudo-inverse `L⁺` (row-major `n×n`). Used by the
    /// analytic LODF path; the cascade itself uses [`Eigen::pseudo_apply`].
    pub fn pseudo_inverse(&self, rel_tol: f64) -> Vec<f64> {
        let n = self.n;
        let scale = self.values.last().copied().unwrap_or(0.0).abs().max(1.0);
        let cutoff = rel_tol * scale;
        let mut x = vec![0.0; n * n];
        for j in 0..n {
            let lambda = self.values[j];
            if lambda.abs() < cutoff {
                continue;
            }
            let inv = 1.0 / lambda;
            for a in 0..n {
                let va = self.vectors[a * n + j];
                if va == 0.0 {
                    continue;
                }
                for b in 0..n {
                    x[a * n + b] += inv * va * self.vectors[b * n + j];
                }
            }
        }
        x
    }
}

/// Decompose a symmetric `n×n` matrix (row-major) via cyclic Jacobi rotations.
///
/// # Panics
/// If `mat.len() != n*n`.
pub fn symmetric_eigen(mat: &[f64], n: usize) -> Eigen {
    assert_eq!(mat.len(), n * n, "matrix length must be n*n");
    if n == 0 {
        return Eigen {
            n,
            values: vec![],
            vectors: vec![],
        };
    }
    let mut a = mat.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_sweeps = 100;
    for _ in 0..max_sweeps {
        // Off-diagonal Frobenius mass.
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off <= 1e-28 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    let sign = if theta > 0.0 { 1.0 } else { -1.0 };
                    sign / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                // A ← Jᵀ A J : rotate columns p,q then rows p,q.
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                // Accumulate eigenvectors: V ← V J.
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let raw: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        raw[i]
            .partial_cmp(&raw[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let values: Vec<f64> = idx.iter().map(|&i| raw[i]).collect();
    let mut vectors = vec![0.0; n * n];
    for (new_j, &old_j) in idx.iter().enumerate() {
        for k in 0..n {
            vectors[k * n + new_j] = v[k * n + old_j];
        }
    }
    Eigen { n, values, vectors }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reconstruct(e: &Eigen) -> Vec<f64> {
        let n = e.n;
        let mut out = vec![0.0; n * n];
        for j in 0..n {
            for a in 0..n {
                for b in 0..n {
                    out[a * n + b] += e.values[j] * e.vectors[a * n + j] * e.vectors[b * n + j];
                }
            }
        }
        out
    }

    #[test]
    fn diagonal_matrix_eigenvalues() {
        // diag(3,1,2) -> ascending 1,2,3
        let m = vec![3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        let e = symmetric_eigen(&m, 3);
        assert!((e.values[0] - 1.0).abs() < 1e-10);
        assert!((e.values[1] - 2.0).abs() < 1e-10);
        assert!((e.values[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn reconstructs_symmetric_matrix() {
        let m = vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
        let e = symmetric_eigen(&m, 3);
        let r = reconstruct(&e);
        for (x, y) in m.iter().zip(r.iter()) {
            assert!((x - y).abs() < 1e-9, "reconstruction mismatch {x} vs {y}");
        }
    }

    #[test]
    fn path_laplacian_has_one_zero_eigenvalue() {
        // L of a 3-path: connected => exactly one zero eigenvalue, λ2>0.
        let l = vec![1.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 1.0];
        let e = symmetric_eigen(&l, 3);
        assert_eq!(e.nullity(1e-8), 1);
        assert!(e.values[1] > 1e-6, "Fiedler value should be positive");
    }

    #[test]
    fn pseudo_inverse_satisfies_l_lplus_l() {
        // L L⁺ L == L for the Laplacian (Penrose condition 1).
        let l = vec![1.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 1.0];
        let e = symmetric_eigen(&l, 3);
        let x = e.pseudo_inverse(1e-9);
        let n = 3;
        // M = L * X
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    m[i * n + j] += l[i * n + k] * x[k * n + j];
                }
            }
        }
        // R = M * L  should equal L
        let mut r = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    r[i * n + j] += m[i * n + k] * l[k * n + j];
                }
            }
        }
        for (a, b) in l.iter().zip(r.iter()) {
            assert!((a - b).abs() < 1e-8, "L L⁺ L != L: {a} vs {b}");
        }
    }
}
