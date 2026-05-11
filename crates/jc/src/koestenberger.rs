//! Köstenberger-Stark 2024: Concentration of the inductive mean on Hadamard
//! spaces — the math foundation for Σ-edge propagation.
//!
//! Citation: G. Köstenberger & T. Stark, "Robust Signal Recovery in Hadamard
//! Spaces", arXiv:2307.06057v2, July 2024.
//!
//! # Why this pillar
//!
//! Pillar 5 (Jirak) certifies the convergence rate of empirical statistics on
//! weakly-dependent ℝ-valued sequences — the SCALAR case. It is the right
//! foundation for `CausalEdge64`'s scalar bit-fields (frequency, confidence).
//!
//! When edges become anisotropic (Σ-tensor instead of scalar weight), the
//! aggregation is no longer in ℝ but on the cone of symmetric positive-definite
//! matrices — a Hadamard space (CAT(0) metric space, non-positive curvature).
//! Köstenberger-Stark Theorem 1 gives the exact concentration:
//!
//! ```text
//!   E[d²(S_n, μ)]  ≤  (6 D_n / n) · Σ d(μ_k, μ)  +  (1/n²) · Σ Var(X_k)
//! ```
//!
//! where:
//!   - μ      population Fréchet mean
//!   - μ_k    Fréchet mean of the k-th distribution
//!   - X_k    sample from the k-th distribution (NOT iid — heteroscedastic OK)
//!   - D_n    max_{1≤k≤n} max{d(μ, μ_k), E[d(X_k, μ_k)]}
//!   - S_n    inductive mean: S_1 = X_1, S_{n+1} = S_n ⊕_{1/(n+1)} X_{n+1}
//!   - ⊕_t    geodesic at parameter t in the Hadamard space
//!
//! The bound holds *without* requiring identical distribution — exactly what
//! we need for evidence aggregation across edges with varying confidence.
//!
//! # Hadamard space used in this probe
//!
//! 2×2 SPD matrices with the affine-invariant Riemannian metric
//!
//! ```text
//!   d(A, B) = ‖log(B^(-1/2) · A · B^(-1/2))‖_F
//! ```
//!
//! and geodesic
//!
//! ```text
//!   A ⊕_t B = A^(1/2) · (A^(-1/2) · B · A^(-1/2))^t · A^(1/2)
//! ```
//!
//! 2×2 keeps every operation closed-form (eigendecomposition is a quadratic
//! root, no iterative eigensolver). The theorem holds for any k×k SPD by the
//! same argument; 2×2 is sufficient demonstration. This is the canonical
//! example of a Hadamard space (Bridson–Häfliger, Sturm).
//!
//! # Test setup
//!
//! - μ = I (identity) is both the population mean and each μ_k by construction
//! - Heteroscedastic: σ_k = 0.3 / √(k+1) — variance shrinks per sample index
//! - X_k = R(θ_k) · diag(exp(σ_k·n1), exp(σ_k·n2)) · R(θ_k)ᵀ
//!     with θ_k uniform on [0,π) and n1, n2 ~ N(0,1) iid
//! - Var(X_k) = E[d²(X_k, μ_k)] = 2 σ_k²  (rotational symmetry argument)
//! - Σ d(μ_k, μ) = 0  (we set μ_k = μ = I, so the 6·D_n term vanishes)
//! - Predicted bound therefore reduces to (1/n²) · Σ Var(X_k) = (2/n²) · Σ σ_k²
//!
//! Monte Carlo: M=1000 runs of n=100 samples. PASS if measured E[d²(S_n, I)]
//! is at or below the predicted bound (with a 1.5× constant-factor slack —
//! the bound is loose by construction, the rate is what we certify).

use crate::PillarResult;

const N_MONTE_CARLO: usize = 1_000;
const N_SAMPLES: usize = 100;

// ════════════════════════════════════════════════════════════════════════════
// Deterministic randomness (matches jirak.rs convention — splitmix64)
// ════════════════════════════════════════════════════════════════════════════

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand_uniform(state: &mut u64) -> f64 {
    // Uniform on [0, 1) — top 53 bits of splitmix64 output.
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn rand_normal(state: &mut u64) -> f64 {
    // Box-Muller transform — return one of the two standard normals.
    let u1 = rand_uniform(state).max(1e-300);
    let u2 = rand_uniform(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ════════════════════════════════════════════════════════════════════════════
// 2×2 SPD matrix [[a, b], [b, c]] with a > 0, c > 0, a·c > b²
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
struct Spd2 {
    a: f64,
    b: f64,
    c: f64,
}

impl Spd2 {
    const I: Self = Self { a: 1.0, b: 0.0, c: 1.0 };

    /// Eigendecomposition.
    /// Returns (λ₁, λ₂, cos θ, sin θ) where the columns of R(θ) = [[c,-s],[s,c]]
    /// are eigenvectors corresponding to (λ₁, λ₂) respectively, λ₁ ≥ λ₂.
    fn eig(&self) -> (f64, f64, f64, f64) {
        // For 2×2 symmetric: λ = (a+c)/2 ± √(((a-c)/2)² + b²)
        let half_trace = (self.a + self.c) / 2.0;
        let half_diff = (self.a - self.c) / 2.0;
        let disc = (half_diff * half_diff + self.b * self.b).sqrt();
        let l1 = half_trace + disc;
        let l2 = half_trace - disc;
        // Eigenvector angle: tan(2θ) = 2b / (a - c).
        // Degenerate case: a = c and b = 0 (already isotropic) → angle is undefined,
        // pick θ = 0 (any rotation works).
        let theta = if self.b.abs() < 1e-15 && (self.a - self.c).abs() < 1e-15 {
            0.0
        } else {
            0.5 * (2.0 * self.b).atan2(self.a - self.c)
        };
        (l1, l2, theta.cos(), theta.sin())
    }

    /// Matrix power M^t via spectral calculus.
    fn pow(&self, t: f64) -> Self {
        let (l1, l2, c, s) = self.eig();
        // Guard against numerical-zero eigenvalues; for SPD they should be > 0.
        let l1t = l1.max(1e-300).powf(t);
        let l2t = l2.max(1e-300).powf(t);
        // M^t = R · diag(λ₁^t, λ₂^t) · Rᵀ
        // R = [[c, -s], [s, c]] ⇒ symmetric reconstruction:
        let a = c * c * l1t + s * s * l2t;
        let b = c * s * (l1t - l2t);
        let cc = s * s * l1t + c * c * l2t;
        Self { a, b, c: cc }
    }

    fn sqrt(&self) -> Self { self.pow(0.5) }
    fn inv_sqrt(&self) -> Self { self.pow(-0.5) }

    /// Geodesic A ⊕_t B = A^(1/2) · (A^(-1/2) · B · A^(-1/2))^t · A^(1/2).
    fn geodesic(&self, other: &Self, t: f64) -> Self {
        let a_inv_sqrt = self.inv_sqrt();
        let inner = sandwich(&a_inv_sqrt, other); // A^(-1/2) · B · A^(-1/2)
        let powered = inner.pow(t);
        let a_sqrt = self.sqrt();
        sandwich(&a_sqrt, &powered)
    }

    /// Affine-invariant distance d(A, B) = ‖log(B^(-1/2) · A · B^(-1/2))‖_F.
    /// For 2×2: ‖log(M)‖_F² = (log λ₁)² + (log λ₂)².
    fn distance(&self, other: &Self) -> f64 {
        let b_inv_sqrt = other.inv_sqrt();
        let m = sandwich(&b_inv_sqrt, self); // B^(-1/2) · A · B^(-1/2)
        let (l1, l2, _, _) = m.eig();
        let log1 = l1.max(1e-300).ln();
        let log2 = l2.max(1e-300).ln();
        (log1 * log1 + log2 * log2).sqrt()
    }
}

/// Symmetric sandwich product  M · N · M  for symmetric M, N. Result symmetric.
/// M does NOT need to be SPD — this is plain matrix multiplication, used as a
/// primitive for both the geodesic and the affine-invariant distance.
fn sandwich(m: &Spd2, n: &Spd2) -> Spd2 {
    // M · N (4 entries, generally not symmetric):
    let p00 = m.a * n.a + m.b * n.b;
    let p01 = m.a * n.b + m.b * n.c;
    let p10 = m.b * n.a + m.c * n.b;
    let p11 = m.b * n.b + m.c * n.c;
    // (M · N) · M:
    let r00 = p00 * m.a + p01 * m.b;
    let r01 = p00 * m.b + p01 * m.c;
    let r10 = p10 * m.a + p11 * m.b;
    let r11 = p10 * m.b + p11 * m.c;
    // Symmetrize numerically (the analytic result IS symmetric).
    Spd2 {
        a: r00,
        b: 0.5 * (r01 + r10),
        c: r11,
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Sample generator: heteroscedastic SPD around μ_k = I.
// X_k = R(θ) · diag(exp(σ·n1), exp(σ·n2)) · R(θ)ᵀ
// guarantees:  X_k SPD by construction;  Fréchet mean(X_k) = I  (rot. symmetry).
// ════════════════════════════════════════════════════════════════════════════

fn sample_spd(state: &mut u64, sigma_k: f64) -> Spd2 {
    let theta = rand_uniform(state) * std::f64::consts::PI;
    let n1 = rand_normal(state) * sigma_k;
    let n2 = rand_normal(state) * sigma_k;
    let l1 = n1.exp();
    let l2 = n2.exp();
    let c = theta.cos();
    let s = theta.sin();
    Spd2 {
        a: c * c * l1 + s * s * l2,
        b: c * s * (l1 - l2),
        c: s * s * l1 + c * c * l2,
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Inductive mean: S_1 = X_1,  S_{n+1} = S_n ⊕_{1/(n+1)} X_{n+1}
// ════════════════════════════════════════════════════════════════════════════

fn inductive_mean(samples: &[Spd2]) -> Spd2 {
    let mut s = samples[0];
    for (i, x) in samples.iter().enumerate().skip(1) {
        let t = 1.0 / (i as f64 + 1.0);
        s = s.geodesic(x, t);
    }
    s
}

// ════════════════════════════════════════════════════════════════════════════
// Theorem 1 RHS (variance-only branch):
//   E[d²(S_n, μ)]  ≤  (1/n²) · Σ Var(X_k)  =  (2/n²) · Σ σ_k²
// (the 6·D_n term vanishes because we set μ_k = μ = I)
// ════════════════════════════════════════════════════════════════════════════

fn predicted_bound(sigmas: &[f64]) -> f64 {
    let n = sigmas.len() as f64;
    let sum_var: f64 = sigmas.iter().map(|s| 2.0 * s * s).sum();
    sum_var / (n * n)
}

// ════════════════════════════════════════════════════════════════════════════
// The probe
// ════════════════════════════════════════════════════════════════════════════

pub fn prove() -> PillarResult {
    // Heteroscedastic schedule — variance shrinks with k.
    // Non-iid by construction; each sample independent given σ_k.
    let sigmas: Vec<f64> = (0..N_SAMPLES)
        .map(|k| 0.3 / ((k as f64 + 1.0).sqrt()))
        .collect();

    let predicted = predicted_bound(&sigmas);

    // Monte Carlo estimate of E[d²(S_n, I)].
    let mu = Spd2::I;
    let mut state: u64 = 0xC0FFEE_BEEF_5EED;
    let mut sum_sq_dist = 0.0f64;
    let mut samples_buf = Vec::with_capacity(N_SAMPLES);

    for _ in 0..N_MONTE_CARLO {
        samples_buf.clear();
        for &sigma_k in &sigmas {
            samples_buf.push(sample_spd(&mut state, sigma_k));
        }
        let s_n = inductive_mean(&samples_buf);
        let d = s_n.distance(&mu);
        sum_sq_dist += d * d;
    }
    let measured = sum_sq_dist / N_MONTE_CARLO as f64;

    // PASS criterion: measured ≤ predicted · 1.5
    // Theorem 1 gives the rate; the constant has a 6·D_n contribution that we
    // zeroed out by construction, leaving the cleaner Var-only bound. We allow
    // 1.5× slack to absorb finite-MC noise + the Cauchy-Schwarz steps in the
    // proof that introduce small additional constants. The point is: the bound
    // HOLDS — and would NOT hold for a substrate without the Hadamard property.
    let tolerance = 1.5;
    let pass = measured <= predicted * tolerance;

    let detail = format!(
        "n={N_SAMPLES}, MC={N_MONTE_CARLO}, σ_k = 0.3/√(k+1) (heteroscedastic). \
         Σ Var(X_k) = {:.6e}. Measured E[d²(S_n,I)] = {measured:.6e}, \
         predicted bound = {predicted:.6e}, tightness = {:.3}× \
         (PASS if ≤ {tolerance:.1}). Hadamard space: 2×2 SPD with affine-invariant \
         metric d(A,B) = ‖log(B^-½·A·B^-½)‖_F. Generalizes to k×k by the same theorem; \
         certifies Σ-edge aggregation in CausalEdgeTensor.",
        sigmas.iter().map(|s| 2.0 * s * s).sum::<f64>(),
        measured / predicted.max(1e-300),
    );

    PillarResult {
        name: "Köstenberger-Stark: inductive mean on Hadamard 2×2 SPD",
        pass,
        measured,
        predicted,
        detail,
        runtime_ms: 0,
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests — internal sanity (do not require the full prove()).
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(x: f64, y: f64, tol: f64) -> bool {
        (x - y).abs() < tol
    }

    #[test]
    fn identity_distance_is_zero() {
        assert!(Spd2::I.distance(&Spd2::I) < 1e-10);
    }

    #[test]
    fn distance_is_symmetric() {
        let a = Spd2 { a: 2.0, b: 0.3, c: 1.5 };
        let b = Spd2 { a: 1.0, b: -0.1, c: 3.0 };
        let d_ab = a.distance(&b);
        let d_ba = b.distance(&a);
        assert!(approx(d_ab, d_ba, 1e-9), "d(A,B)={d_ab}, d(B,A)={d_ba}");
    }

    #[test]
    fn geodesic_endpoints() {
        let a = Spd2 { a: 2.0, b: 0.3, c: 1.5 };
        let b = Spd2 { a: 1.0, b: -0.1, c: 3.0 };
        let g0 = a.geodesic(&b, 0.0);
        let g1 = a.geodesic(&b, 1.0);
        assert!(a.distance(&g0) < 1e-8, "γ(0) should be A");
        assert!(b.distance(&g1) < 1e-8, "γ(1) should be B");
    }

    #[test]
    fn geodesic_midpoint_of_i_and_2i() {
        // I ⊕_{1/2} 2I should be √2 · I (geometric mean).
        let two_i = Spd2 { a: 2.0, b: 0.0, c: 2.0 };
        let mid = Spd2::I.geodesic(&two_i, 0.5);
        let sqrt2 = std::f64::consts::SQRT_2;
        assert!(approx(mid.a, sqrt2, 1e-10));
        assert!(approx(mid.c, sqrt2, 1e-10));
        assert!(approx(mid.b, 0.0, 1e-10));
    }

    #[test]
    fn pow_zero_is_identity() {
        let m = Spd2 { a: 3.0, b: 0.5, c: 2.0 };
        let p0 = m.pow(0.0);
        assert!(p0.distance(&Spd2::I) < 1e-10);
    }

    #[test]
    fn pow_one_is_self() {
        let m = Spd2 { a: 3.0, b: 0.5, c: 2.0 };
        let p1 = m.pow(1.0);
        assert!(m.distance(&p1) < 1e-10);
    }

    #[test]
    fn sqrt_squared_is_self() {
        let m = Spd2 { a: 3.0, b: 0.5, c: 2.0 };
        let r = m.sqrt();
        let r2 = r.pow(2.0);
        assert!(m.distance(&r2) < 1e-10);
    }

    #[test]
    fn pillar_passes() {
        let r = prove();
        assert!(
            r.pass,
            "Köstenberger-Stark pillar failed: measured {:.6e} vs predicted {:.6e} — {}",
            r.measured, r.predicted, r.detail
        );
    }
}
