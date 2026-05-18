//! Pillar 7 (3D analogue): Σ-Push-Forward as EWA-Sandwich along multi-hop
//! edge paths — symmetric 3×3 SPD covariances.
//!
//! # Mathematical claim
//!
//! In 3DGS (3D Gaussian Splatting), world-space covariance matrices Σ ∈ ℝ³ˣ³
//! are pushed forward to image-space via the projection Jacobian J:
//!
//!   Σ_image = J · W · Σ · Wᵀ · Jᵀ
//!
//! This is the **J·W·Σ·Wᵀ·Jᵀ EWA sandwich** — a double affine push-forward
//! of a 3×3 SPD covariance. In `ndarray::hpc::splat3d`, the rendering pipeline
//! requires this sandwich to preserve positive-semidefiniteness exactly, since
//! a non-PSD Σ_image produces degenerate Gaussian splats with negative variances.
//!
//! This probe is the **3D analogue** of Pillar 6 (`ewa_sandwich.rs`), which
//! certifies the same sandwich push-forward claim for 2×2 SPD matrices. Pillar 7
//! extends that certification to the 3×3 case — the actual covariance shape
//! consumed by `ndarray::hpc::splat3d` in the J·W·Σ·Wᵀ·Jᵀ projection.
//!
//! For multi-hop edge propagation, the analogous operation along path
//! A → B → C is:
//!
//!   Σ_path = J_BC · Σ_AB · J_BCᵀ
//!
//! Iterating along a path of length n:
//!
//!   Σ_n = M_n · M_{n-1} · ... · M_1 · Σ_0 · M_1ᵀ · ... · M_{n-1}ᵀ · M_nᵀ
//!
//! where M_k = sqrt(Σ_k) is the step-Jacobian of the k-th edge.
//!
//! # Two claims certified simultaneously
//!
//! 1. **PSD-preservation**: Σ_n stays SPD for all n (sandwich preserves PSD)
//! 2. **Convergence rate**: ‖log(Σ_n)‖_F² concentrates across MC paths with
//!    rate consistent with a log-normal-corrected KS bound in 3D.
//!
//! # Eigendecomposition
//!
//! The probe uses Smith 1961 closed-form eigendecomposition for 3×3 symmetric
//! matrices (three closed-form eigenvalues via the characteristic polynomial,
//! eigenvectors via cross-product of row pairs, with Gram-Schmidt fallback for
//! degenerate eigenspaces). This is the same algorithm implemented in f32 SIMD
//! in `ndarray::hpc::splat3d::spd3`; here it is self-contained in f64 for
//! mathematical certification without any dependency on the graphics crate.
//!
//! # CV prediction formula (3D extension)
//!
//! Pillar 6 uses σ_eff ≈ σ_step · √(2n) (2 eigenvalues per step).
//! For 3D, the natural extension is σ_eff ≈ σ_step · √(3n) (3 eigenvalues),
//! giving: CV_predicted = √(2/n) · √(1 + 3·σ²·n)
//!
//! # PASS criteria (identical to Pillar 6)
//!
//! - PSD preservation rate >= 0.999
//! - CV tightness (measured / predicted) ≤ 1.75
//!
//! # Probe setup
//!
//! - 1000 paths × 10 hops, σ_step = 0.2, PSD eps = 1e-12
//! - Same splitmix64/rand_uniform/rand_normal RNG declared locally
//! - SEED = 0xEDA_5A_DC_5A_DD (one byte higher than Pillar 6's 0xEDA_5A_DC_5A_DC)
//! - Σ_0 = I; each hop: step ~ from_scale_quat(exp-normal scales, uniform quat)
//! - M = sqrt(step); Σ_{n+1} = M · Σ_n · Mᵀ

use crate::PillarResult;

const N_PATHS: usize = 1_000;
const PATH_LENGTH: usize = 10;
const SEED: u64 = 0xEDA_5A_DC_5A_DD;

// ════════════════════════════════════════════════════════════════════════════
// Deterministic RNG (consistent with other pillars)
// ════════════════════════════════════════════════════════════════════════════

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand_uniform(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn rand_normal(state: &mut u64) -> f64 {
    let u1 = rand_uniform(state).max(1e-300);
    let u2 = rand_uniform(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ════════════════════════════════════════════════════════════════════════════
// 3×3 SPD matrix — self-contained f64, mirroring Spd2 in ewa_sandwich.rs.
// Stored as upper triangle: {a11, a12, a13, a22, a23, a33}.
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
struct Spd3 {
    a11: f64, a12: f64, a13: f64,
               a22: f64, a23: f64,
                          a33: f64,
}

impl Spd3 {
    const I: Self = Self { a11: 1.0, a12: 0.0, a13: 0.0, a22: 1.0, a23: 0.0, a33: 1.0 };

    // ── Smith 1961 closed-form eigendecomposition ──────────────────────────
    //
    // Returns (λ1, λ2, λ3, V) where λ1 ≥ λ2 ≥ λ3 and V is column-orthonormal.
    // V[i] = i-th eigenvector (row of the returned array for convenience).
    fn eig(&self) -> (f64, f64, f64, [[f64; 3]; 3]) {
        let eps_diag = 1e-12_f64;

        // Off-diagonal contribution
        let p1 = self.a12 * self.a12 + self.a13 * self.a13 + self.a23 * self.a23;

        if p1 < eps_diag {
            // Diagonal matrix — fast path
            let vals = [self.a11, self.a22, self.a33];
            let mut idx = [0usize, 1, 2];
            // Sort descending
            if vals[idx[0]] < vals[idx[1]] { idx.swap(0, 1); }
            if vals[idx[1]] < vals[idx[2]] { idx.swap(1, 2); }
            if vals[idx[0]] < vals[idx[1]] { idx.swap(0, 1); }
            let lam = [vals[idx[0]], vals[idx[1]], vals[idx[2]]];
            let mut vecs = [[0.0f64; 3]; 3];
            for i in 0..3 {
                vecs[i][idx[i]] = 1.0;
            }
            return (lam[0], lam[1], lam[2], vecs);
        }

        let q = (self.a11 + self.a22 + self.a33) / 3.0;
        let b11 = self.a11 - q;
        let b22 = self.a22 - q;
        let b33 = self.a33 - q;
        let p2 = b11 * b11 + b22 * b22 + b33 * b33 + 2.0 * p1;
        let p = (p2 / 6.0).sqrt();

        // B = (A - q·I) / p
        let inv_p = 1.0 / p.max(1e-300);
        let b = Spd3 {
            a11: b11 * inv_p, a12: self.a12 * inv_p, a13: self.a13 * inv_p,
            a22: b22 * inv_p, a23: self.a23 * inv_p,
            a33: b33 * inv_p,
        };

        // det(B)/2, clamped to [-1, 1]
        let r = (b.det() / 2.0).clamp(-1.0, 1.0);
        let phi = r.acos() / 3.0;

        let two_pi_3 = 2.0 * std::f64::consts::PI / 3.0;
        let lam1 = q + 2.0 * p * phi.cos();
        let lam3 = q + 2.0 * p * (phi + two_pi_3).cos();
        let lam2 = 3.0 * q - lam1 - lam3;

        // Eigenvectors via cross-product of two rows of (A - λᵢ·I)
        let vecs = [
            eigvec_for(self, lam1),
            eigvec_for(self, lam2),
            eigvec_for(self, lam3),
        ];
        (lam1, lam2, lam3, vecs)
    }

    fn det(&self) -> f64 {
        self.a11 * (self.a22 * self.a33 - self.a23 * self.a23)
            - self.a12 * (self.a12 * self.a33 - self.a23 * self.a13)
            + self.a13 * (self.a12 * self.a23 - self.a22 * self.a13)
    }

    /// Σ^t via spectral lift; eigenvalues clamped to EPS_EIGVAL before powf.
    fn pow(&self, t: f64) -> Self {
        const EPS_EIGVAL: f64 = 1e-300;
        let (l1, l2, l3, vecs) = self.eig();
        let l1t = l1.max(EPS_EIGVAL).powf(t);
        let l2t = l2.max(EPS_EIGVAL).powf(t);
        let l3t = l3.max(EPS_EIGVAL).powf(t);
        spectral_reconstruct(l1t, l2t, l3t, &vecs)
    }

    fn sqrt(&self) -> Self { self.pow(0.5) }

    /// V · diag(ln λᵢ) · Vᵀ
    fn log_spd(&self) -> Self {
        const EPS_EIGVAL: f64 = 1e-300;
        let (l1, l2, l3, vecs) = self.eig();
        let l1l = l1.max(EPS_EIGVAL).ln();
        let l2l = l2.max(EPS_EIGVAL).ln();
        let l3l = l3.max(EPS_EIGVAL).ln();
        spectral_reconstruct(l1l, l2l, l3l, &vecs)
    }

    /// Frobenius norm squared (off-diagonals counted twice).
    fn frobenius_sq(&self) -> f64 {
        self.a11 * self.a11
            + self.a22 * self.a22
            + self.a33 * self.a33
            + 2.0 * (self.a12 * self.a12 + self.a13 * self.a13 + self.a23 * self.a23)
    }

    /// Sylvester's criterion + eigenvalue check.
    fn is_spd(&self, eps: f64) -> bool {
        // Leading minors
        if self.a11 <= eps { return false; }
        let m2 = self.a11 * self.a22 - self.a12 * self.a12;
        if m2 <= eps { return false; }
        if self.det() <= eps { return false; }
        // Eigenvalue check
        let (l1, l2, l3, _) = self.eig();
        l1 > eps && l2 > eps && l3 > eps
    }

    /// 3DGS canonical form: Σ = R · diag(s²) · Rᵀ where R is from quaternion q.
    /// q = [w, x, y, z], normalized internally.
    fn from_scale_quat(s: [f64; 3], q: [f64; 4]) -> Self {
        let norm = (q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]).sqrt().max(1e-300);
        let (w, x, y, z) = (q[0]/norm, q[1]/norm, q[2]/norm, q[3]/norm);

        // Rotation matrix columns from quaternion
        let r = [
            [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z),       2.0*(x*z + w*y)      ],
            [2.0*(x*y + w*z),        1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x)      ],
            [2.0*(x*z - w*y),        2.0*(y*z + w*x),        1.0 - 2.0*(x*x + y*y)],
        ];

        let s2 = [s[0]*s[0], s[1]*s[1], s[2]*s[2]];

        // Σ = R · diag(s²) · Rᵀ — upper triangle
        let a11 = r[0][0]*r[0][0]*s2[0] + r[0][1]*r[0][1]*s2[1] + r[0][2]*r[0][2]*s2[2];
        let a12 = r[0][0]*r[1][0]*s2[0] + r[0][1]*r[1][1]*s2[1] + r[0][2]*r[1][2]*s2[2];
        let a13 = r[0][0]*r[2][0]*s2[0] + r[0][1]*r[2][1]*s2[1] + r[0][2]*r[2][2]*s2[2];
        let a22 = r[1][0]*r[1][0]*s2[0] + r[1][1]*r[1][1]*s2[1] + r[1][2]*r[1][2]*s2[2];
        let a23 = r[1][0]*r[2][0]*s2[0] + r[1][1]*r[2][1]*s2[1] + r[1][2]*r[2][2]*s2[2];
        let a33 = r[2][0]*r[2][0]*s2[0] + r[2][1]*r[2][1]*s2[1] + r[2][2]*r[2][2]*s2[2];

        Self { a11, a12, a13, a22, a23, a33 }
    }
}

// ── Eigenvector recovery via cross-product of two rows ──────────────────────

/// Compute (A - lam·I) · row_i cross-product pairs to get the eigenvector.
/// Falls back to Gram-Schmidt if the cross-products are near-zero.
fn eigvec_for(a: &Spd3, lam: f64) -> [f64; 3] {
    // Rows of (A - lam·I)
    let r0 = [a.a11 - lam, a.a12,        a.a13       ];
    let r1 = [a.a12,        a.a22 - lam, a.a23       ];
    let r2 = [a.a13,        a.a23,        a.a33 - lam];

    // Cross products of all row pairs
    let c01 = cross3(r0, r1);
    let c02 = cross3(r0, r2);
    let c12 = cross3(r1, r2);

    // Pick the one with the largest norm
    let n01 = dot3(c01, c01);
    let n02 = dot3(c02, c02);
    let n12 = dot3(c12, c12);

    let best = if n01 >= n02 && n01 >= n12 { c01 }
               else if n02 >= n12 { c02 }
               else { c12 };

    let norm = dot3(best, best).sqrt();
    if norm > 1e-14 {
        [best[0]/norm, best[1]/norm, best[2]/norm]
    } else {
        // Gram-Schmidt fallback — pick an arbitrary non-collinear vector
        gram_schmidt_fallback(best)
    }
}

fn gram_schmidt_fallback(v: [f64; 3]) -> [f64; 3] {
    // Try canonical basis vectors and pick least-collinear
    let bases: [[f64; 3]; 3] = [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]];
    let mut best = bases[0];
    let mut best_cross_norm = 0.0f64;
    for b in &bases {
        let c = cross3(v, *b);
        let n = dot3(c, c).sqrt();
        if n > best_cross_norm { best_cross_norm = n; best = c; }
    }
    let norm = dot3(best, best).sqrt().max(1e-300);
    [best[0]/norm, best[1]/norm, best[2]/norm]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

/// Reconstruct Σ = V · diag(vals) · Vᵀ from eigenvectors and scalar values.
fn spectral_reconstruct(v1: f64, v2: f64, v3: f64, vecs: &[[f64; 3]; 3]) -> Spd3 {
    // Σ = Σᵢ vᵢ · eᵢ · eᵢᵀ
    let (e0, e1, e2) = (vecs[0], vecs[1], vecs[2]);
    Spd3 {
        a11: v1*e0[0]*e0[0] + v2*e1[0]*e1[0] + v3*e2[0]*e2[0],
        a12: v1*e0[0]*e0[1] + v2*e1[0]*e1[1] + v3*e2[0]*e2[1],
        a13: v1*e0[0]*e0[2] + v2*e1[0]*e1[2] + v3*e2[0]*e2[2],
        a22: v1*e0[1]*e0[1] + v2*e1[1]*e1[1] + v3*e2[1]*e2[1],
        a23: v1*e0[1]*e0[2] + v2*e1[1]*e1[2] + v3*e2[1]*e2[2],
        a33: v1*e0[2]*e0[2] + v2*e1[2]*e1[2] + v3*e2[2]*e2[2],
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Sandwich product M · N · Mᵀ for symmetric M, N.
// Off-diagonal elements are averaged to enforce symmetry.
// ════════════════════════════════════════════════════════════════════════════

fn sandwich(m: &Spd3, n: &Spd3) -> Spd3 {
    // P = M · N  (full 3×3 product, using symmetry of N)
    // M is [[a11,a12,a13],[a12,a22,a23],[a13,a23,a33]]
    // N is [[b11,b12,b13],[b12,b22,b23],[b13,b23,b33]]
    let p00 = m.a11*n.a11 + m.a12*n.a12 + m.a13*n.a13;
    let p01 = m.a11*n.a12 + m.a12*n.a22 + m.a13*n.a23;
    let p02 = m.a11*n.a13 + m.a12*n.a23 + m.a13*n.a33;
    let p10 = m.a12*n.a11 + m.a22*n.a12 + m.a23*n.a13;
    let p11 = m.a12*n.a12 + m.a22*n.a22 + m.a23*n.a23;
    let p12 = m.a12*n.a13 + m.a22*n.a23 + m.a23*n.a33;
    let p20 = m.a13*n.a11 + m.a23*n.a12 + m.a33*n.a13;
    let p21 = m.a13*n.a12 + m.a23*n.a22 + m.a33*n.a23;
    let p22 = m.a13*n.a13 + m.a23*n.a23 + m.a33*n.a33;

    // R = P · Mᵀ = P · M (M symmetric)
    let r00 = p00*m.a11 + p01*m.a12 + p02*m.a13;
    let r01 = p00*m.a12 + p01*m.a22 + p02*m.a23;
    let r02 = p00*m.a13 + p01*m.a23 + p02*m.a33;
    let r10 = p10*m.a11 + p11*m.a12 + p12*m.a13;
    let r11 = p10*m.a12 + p11*m.a22 + p12*m.a23;
    let r12 = p10*m.a13 + p11*m.a23 + p12*m.a33;
    let r20 = p20*m.a11 + p21*m.a12 + p22*m.a13;
    let r21 = p20*m.a12 + p21*m.a22 + p22*m.a23;
    let r22 = p20*m.a13 + p21*m.a23 + p22*m.a33;

    Spd3 {
        a11: r00,
        a12: 0.5 * (r01 + r10),
        a13: 0.5 * (r02 + r20),
        a22: r11,
        a23: 0.5 * (r12 + r21),
        a33: r22,
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Edge sampler — 3D analogue of Pillar 6's sample_step_sigma.
// Orientation from a uniformly-sampled unit quaternion (4 normals, normalized).
// Log-scales from σ_step · N(0,1) for each of the 3 axes.
// ════════════════════════════════════════════════════════════════════════════

fn sample_step_sigma(state: &mut u64, sigma_step: f64) -> Spd3 {
    // Uniform random unit quaternion via Gaussian sampling
    let qw = rand_normal(state);
    let qx = rand_normal(state);
    let qy = rand_normal(state);
    let qz = rand_normal(state);

    // Log-scales for each axis
    let n1 = rand_normal(state) * sigma_step;
    let n2 = rand_normal(state) * sigma_step;
    let n3 = rand_normal(state) * sigma_step;
    let s = [n1.exp(), n2.exp(), n3.exp()];

    Spd3::from_scale_quat(s, [qw, qx, qy, qz])
}

// ════════════════════════════════════════════════════════════════════════════
// Path propagation — mirror of Pillar 6's propagate_path
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
struct PathResult {
    final_sigma: Spd3,
    log_norm_sq: f64,
    psd_hops: usize,
}

fn propagate_path(state: &mut u64, length: usize, sigma_step: f64, eps: f64) -> PathResult {
    let mut sigma = Spd3::I;
    let mut psd_hops = 0;
    for _ in 0..length {
        let step = sample_step_sigma(state, sigma_step);
        let m = step.sqrt();
        sigma = sandwich(&m, &sigma);
        if sigma.is_spd(eps) {
            psd_hops += 1;
        }
    }
    let log_sigma = sigma.log_spd();
    let log_norm_sq = log_sigma.frobenius_sq();
    PathResult { final_sigma: sigma, log_norm_sq, psd_hops }
}

// ════════════════════════════════════════════════════════════════════════════
// prove() — mirrors Pillar 6's prove() body with 3D adjustments
// ════════════════════════════════════════════════════════════════════════════

pub fn prove() -> PillarResult {
    let mut state = SEED;
    let sigma_step = 0.2_f64;
    let psd_eps = 1e-12_f64;

    let mut log_norms_sq = Vec::with_capacity(N_PATHS);
    let mut total_psd_hops = 0usize;
    let mut max_psd_violations_in_one_path = 0usize;

    for _ in 0..N_PATHS {
        let r = propagate_path(&mut state, PATH_LENGTH, sigma_step, psd_eps);
        log_norms_sq.push(r.log_norm_sq);
        total_psd_hops += r.psd_hops;
        let violations = PATH_LENGTH - r.psd_hops;
        if violations > max_psd_violations_in_one_path {
            max_psd_violations_in_one_path = violations;
        }
    }

    let total_hops = N_PATHS * PATH_LENGTH;
    let psd_rate = total_psd_hops as f64 / total_hops as f64;

    let mean_log_norm_sq: f64 = log_norms_sq.iter().sum::<f64>() / N_PATHS as f64;
    let var_log_norm_sq: f64 = log_norms_sq
        .iter()
        .map(|x| (x - mean_log_norm_sq).powi(2))
        .sum::<f64>()
        / N_PATHS as f64;
    let std_log_norm_sq = var_log_norm_sq.sqrt();

    // Concentration prediction — 3D analogue of Pillar 6's formula.
    //
    // Pillar 6 (2D): σ_eff ≈ σ_step · √(2n), CV_predicted = √(2/n) · √(1 + 2σ²n)
    //
    // For 3D: there are 3 eigenvalues per step (not 2), so the natural extension
    // is σ_eff ≈ σ_step · √(3n), giving:
    //
    //   CV_predicted = √(2/n) · √(1 + 3·σ_step²·n)
    //
    // For n=10, σ_step=0.2: CV_predicted ≈ 0.4472 · √(2.2) ≈ 0.663.
    // The 3D case has more degrees of freedom, so the effective σ_eff is
    // slightly larger, but the path-length averaging still concentrates the
    // distribution. PASS if measured CV ≤ predicted CV × 1.75.
    let cv_measured = if mean_log_norm_sq > 1e-300 {
        std_log_norm_sq / mean_log_norm_sq
    } else {
        f64::INFINITY
    };
    let n = PATH_LENGTH as f64;
    let cv_predicted = (2.0 / n).sqrt() * (1.0 + 3.0 * sigma_step * sigma_step * n).sqrt();
    let cv_tightness = cv_measured / cv_predicted;

    let psd_pass = psd_rate >= 0.999;
    let cv_pass = cv_tightness <= 1.75;
    let pass = psd_pass && cv_pass;

    let detail = format!(
        "3D analogue of Pillar 6 EWA-Sandwich, certifying symmetric 3×3 SPD covariances \
         as consumed by ndarray::hpc::splat3d in the J·W·Σ·Wᵀ·Jᵀ projection. \
         Smith 1961 closed-form eigendecomp (cross-product row pairs + Gram-Schmidt fallback). \
         n_paths={N_PATHS}, path_length={PATH_LENGTH}, σ_step={sigma_step}. \
         PSD-preservation rate = {psd_rate:.6} ({total_psd_hops}/{total_hops} hops kept SPD; \
         worst path had {max_psd_violations_in_one_path} non-SPD intermediate states). \
         Concentration: mean ‖log(Σ_n)‖²_F = {mean_log_norm_sq:.4}, std = {std_log_norm_sq:.4}, \
         CV = {cv_measured:.4} (predicted CV ≤ √(2/n)·√(1+3σ²n) = {cv_predicted:.4} \
         from 3D log-normal-corrected KS bound [3 eigvals/step vs 2 in 2D], \
         tightness = {cv_tightness:.3}× — PASS if ≤ 1.75). \
         psd_pass={psd_pass}, cv_pass={cv_pass}."
    );

    PillarResult {
        name: "EWA-Sandwich 3D: Σ-push-forward on symmetric 3×3 SPD covariances",
        pass,
        measured: cv_measured,
        predicted: cv_predicted,
        detail,
        runtime_ms: 0,
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(x: f64, y: f64, tol: f64) -> bool {
        (x - y).abs() < tol
    }

    #[test]
    fn identity_sandwich_is_identity() {
        let r = sandwich(&Spd3::I, &Spd3::I);
        assert!(approx(r.a11, 1.0, 1e-12));
        assert!(approx(r.a12, 0.0, 1e-12));
        assert!(approx(r.a13, 0.0, 1e-12));
        assert!(approx(r.a22, 1.0, 1e-12));
        assert!(approx(r.a23, 0.0, 1e-12));
        assert!(approx(r.a33, 1.0, 1e-12));
    }

    #[test]
    fn sandwich_preserves_spd() {
        let mut state = 0xCAFEu64;
        for _ in 0..1000 {
            let m = sample_step_sigma(&mut state, 0.3);
            let n = sample_step_sigma(&mut state, 0.3);
            let m_sqrt = m.sqrt();
            let result = sandwich(&m_sqrt, &n);
            assert!(result.is_spd(1e-10),
                "sandwich produced non-SPD: m={m:?}, n={n:?}, result={result:?}");
        }
    }

    #[test]
    fn sandwich_with_identity_returns_input() {
        // sqrt(Σ) · I · sqrt(Σ)ᵀ = Σ
        let sigma = Spd3 { a11: 2.0, a12: 0.3, a13: 0.1, a22: 1.5, a23: 0.2, a33: 1.2 };
        let sigma_sqrt = sigma.sqrt();
        let result = sandwich(&sigma_sqrt, &Spd3::I);
        assert!(approx(result.a11, sigma.a11, 1e-9), "a11: {} vs {}", result.a11, sigma.a11);
        assert!(approx(result.a12, sigma.a12, 1e-9), "a12: {} vs {}", result.a12, sigma.a12);
        assert!(approx(result.a13, sigma.a13, 1e-9), "a13: {} vs {}", result.a13, sigma.a13);
        assert!(approx(result.a22, sigma.a22, 1e-9), "a22: {} vs {}", result.a22, sigma.a22);
        assert!(approx(result.a23, sigma.a23, 1e-9), "a23: {} vs {}", result.a23, sigma.a23);
        assert!(approx(result.a33, sigma.a33, 1e-9), "a33: {} vs {}", result.a33, sigma.a33);
    }

    #[test]
    fn path_propagation_returns_finite_results() {
        let mut state = 0x1234u64;
        let r = propagate_path(&mut state, 20, 0.2, 1e-12);
        assert!(r.log_norm_sq.is_finite(), "log_norm_sq should be finite, got {}", r.log_norm_sq);
        assert!(r.final_sigma.is_spd(1e-10), "final sigma should be SPD: {:?}", r.final_sigma);
    }

    #[test]
    fn long_paths_dont_explode() {
        let mut state = 0xDEADu64;
        let mut max_log_norm_sq = 0.0f64;
        for _ in 0..100 {
            let r = propagate_path(&mut state, 50, 0.1, 1e-12);
            if r.log_norm_sq > max_log_norm_sq {
                max_log_norm_sq = r.log_norm_sq;
            }
        }
        assert!(max_log_norm_sq < 1000.0,
            "long paths exploded: max ‖log Σ‖² = {max_log_norm_sq}");
    }

    #[test]
    fn deterministic_with_fixed_seed() {
        let mut s1 = 0xABCDu64;
        let mut s2 = 0xABCDu64;
        for _ in 0..50 {
            let r1 = propagate_path(&mut s1, 5, 0.2, 1e-12);
            let r2 = propagate_path(&mut s2, 5, 0.2, 1e-12);
            assert!(approx(r1.log_norm_sq, r2.log_norm_sq, 1e-12));
            assert_eq!(r1.psd_hops, r2.psd_hops);
        }
    }

    #[test]
    fn pillar_passes() {
        let r = prove();
        assert!(r.pass, "Pillar 7 (EWA-sandwich 3D) failed: {}", r.detail);
    }

    #[test]
    fn eigendecomp_diagonal_fast_path() {
        // p1 < eps_diag → fast diagonal branch
        let a = Spd3 { a11: 3.0, a12: 0.0, a13: 0.0, a22: 2.0, a23: 0.0, a33: 1.0 };
        let (l1, l2, l3, vecs) = a.eig();
        // Should return sorted descending: 3, 2, 1
        assert!(approx(l1, 3.0, 1e-12), "l1={l1}");
        assert!(approx(l2, 2.0, 1e-12), "l2={l2}");
        assert!(approx(l3, 1.0, 1e-12), "l3={l3}");
        // Eigenvectors should be canonical basis (in some permutation)
        for v in &vecs {
            let norm_sq = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
            assert!(approx(norm_sq, 1.0, 1e-12), "eigvec not unit: {v:?}");
        }
    }

    #[test]
    fn from_scale_quat_unit_rotation_is_diag_scale_sq() {
        // Identity quaternion [1,0,0,0] → Σ = diag(s²)
        let s = [2.0, 3.0, 0.5];
        let q = [1.0, 0.0, 0.0, 0.0]; // identity rotation
        let sigma = Spd3::from_scale_quat(s, q);
        assert!(approx(sigma.a11, 4.0, 1e-12), "a11={}", sigma.a11);  // 2²
        assert!(approx(sigma.a22, 9.0, 1e-12), "a22={}", sigma.a22);  // 3²
        assert!(approx(sigma.a33, 0.25, 1e-12), "a33={}", sigma.a33); // 0.5²
        assert!(approx(sigma.a12, 0.0, 1e-12), "a12={}", sigma.a12);
        assert!(approx(sigma.a13, 0.0, 1e-12), "a13={}", sigma.a13);
        assert!(approx(sigma.a23, 0.0, 1e-12), "a23={}", sigma.a23);
    }
}
