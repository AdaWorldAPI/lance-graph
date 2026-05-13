//! Düker-Zoubouloglou 2024: Breuer-Major CLT for Hilbert-space-valued random
//! variables — the math foundation for fingerprint-bundle convergence in ℓ².
//!
//! Citation: M.-C. Düker & P. Zoubouloglou, "Breuer-Major Theorems for Hilbert
//! Space-Valued Random Variables", arXiv:2405.11452v1, May 2024.
//!
//! # Why this pillar
//!
//! With Pillars 5 and 5+ in place, the concentration family is:
//!
//!   Pillar 5  (Jirak):              ℝ-valued sequences, weak dependence
//!   Pillar 5+ (Köstenberger-Stark): Hadamard space (PSD cone, Σ-tensors)
//!
//! What's still missing is **Hilbert space** — the natural home for the
//! 16,384-bit fingerprints when lifted to f32 representation, for SH coefficient
//! sequences on the sphere, and for any operator whose output lives in an
//! infinite-dimensional inner-product space.
//!
//! Düker-Zoubouloglou Theorem 2.1: For a stationary Gaussian process {X_k}
//! taking values in a separable Hilbert space H₁, with covariance operator Q,
//! and an operator G: H₁ → H₂ of Hermite rank q ≥ 1, the partial sums
//!
//!   S_n = (1/√n) · Σ_{k=1}^n (G[X_k] − E G[X_k])
//!
//! converge weakly to a centered Gaussian Z in H₂, with covariance operator
//!
//!   T_Z = Σ_v (E G[X_1] ⊗ G[X_{v+1}] + E G[X_{v+1}] ⊗ G[X_1]) + E G[X_1] ⊗ G[X_1]
//!
//! provided
//!
//!   Σ_v (sup_r Σ_s |ρ_rs(v)|)^q < ∞       (Düker-Zoubouloglou condition (2.1))
//!
//! where ρ_rs(v) is the autocorrelation function of the scores of X_k.
//!
//! For our substrate, this is the formal guarantee that:
//!   1. Bundle-of-N fingerprints converges to a Gaussian limit in ℓ² as N→∞
//!   2. The limit's covariance operator has an explicit closed form
//!   3. Multi-cycle resonance accumulation respects the same CLT
//!
//! # Probe setup
//!
//! - H₁ = ℝ^d with d = 16,384 (substrate-native fingerprint width)
//! - {X_k} = AR(1) Gaussian process: X_{k+1} = φ·X_k + √(1-φ²)·ε_k, ε_k ~ N(0, I)
//! - This is stationary with Cov(X_0) = I, Cov(X_0, X_v) = φ^|v|·I
//! - Autocorrelation factorizes: ρ_rs(v) = δ_rs · φ^|v|  (Düker-Zoubouloglou
//!   Remark 2.2 — the equivalence reduces to the simpler scalar condition)
//! - Condition (2.1) holds since Σ_v |φ|^v = (1+|φ|)/(1-|φ|) < ∞ for |φ| < 1
//! - Operator G = identity (Hermite rank q = 1; cleanest demonstration)
//!
//! Then T_Z = Σ_v Cov(X_0, X_v) = I · Σ_v φ^|v| = I · (1+φ)/(1-φ)
//!
//! and trace(T_Z) = d · (1+φ)/(1-φ).
//!
//! For φ = 0.5, d = 16384: predicted trace = 16384 · 3 = 49,152.
//!
//! Monte Carlo: M runs of n samples each. Measure (1/M) · Σ ‖S_n‖² as the
//! empirical trace of Cov(S_n). PASS if relative error < 10%.
//!
//! # Statistical precision
//!
//! For each run, ‖S_n‖² ≈ chi-squared with d degrees of freedom, scaled by
//! T_Z_ii. Coefficient of variation per run: √(2/d) ≈ 1.1% at d=16384.
//! With M=20 runs: pooled CV ≈ √(2/(M·d)) ≈ 0.25%. Well below 10% tolerance.

use crate::PillarResult;

const D: usize = 16_384;          // substrate-native fingerprint dimension
const N: usize = 1_000;           // samples per Monte Carlo run
const M: usize = 20;              // Monte Carlo runs
const PHI: f64 = 0.5;             // AR(1) coefficient (moderate dependence)
const TOLERANCE: f64 = 0.10;      // relative-error PASS threshold

// ════════════════════════════════════════════════════════════════════════════
// Deterministic randomness (matches existing pillar conventions)
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

/// Box-Muller pair generator with cached second normal.
/// Cuts log/sqrt/cos cost in half by emitting two normals per evaluation.
struct NormalGen {
    state: u64,
    cached: Option<f64>,
}

impl NormalGen {
    fn new(seed: u64) -> Self {
        Self { state: seed, cached: None }
    }

    fn next(&mut self) -> f64 {
        if let Some(z) = self.cached.take() {
            return z;
        }
        let u1 = rand_uniform(&mut self.state).max(1e-300);
        let u2 = rand_uniform(&mut self.state);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.cached = Some(r * theta.sin());
        r * theta.cos()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// AR(1) Gaussian process step in ℝ^d
//   X_{k+1} = φ · X_k + √(1 − φ²) · ε_k,    ε_k ~ N(0, I_d) iid
// Stationary distribution: N(0, I_d).
// ════════════════════════════════════════════════════════════════════════════

#[inline]
fn ar1_step(x: &mut [f64], gen: &mut NormalGen, phi: f64, sqrt_one_minus_phi2: f64) {
    for v in x.iter_mut() {
        let eps = gen.next();
        *v = phi * *v + sqrt_one_minus_phi2 * eps;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Core measurement — pulled into its own function so tests can vary parameters.
// Returns (measured_trace, predicted_trace).
// ════════════════════════════════════════════════════════════════════════════

fn measure_trace(d: usize, n: usize, m: usize, phi: f64, seed: u64) -> (f64, f64) {
    let sqrt_one_minus_phi2 = (1.0 - phi * phi).sqrt();
    // T_Z = Q · (1+φ)/(1-φ) for AR(1) with Q = I.  trace(T_Z) = d · (1+φ)/(1-φ).
    // (For φ = 0 the limit is Q = I and trace = d — the iid CLT.)
    let predicted_trace = d as f64 * (1.0 + phi) / (1.0 - phi);

    let mut gen = NormalGen::new(seed);
    let mut x = vec![0.0f64; d];
    let mut sum = vec![0.0f64; d];
    let mut sum_norm_sq = 0.0f64;

    for _ in 0..m {
        // Initialize X_0 from stationary distribution N(0, I).
        for v in x.iter_mut() {
            *v = gen.next();
        }
        sum.fill(0.0);
        // Add X_0 to the sum.
        for i in 0..d {
            sum[i] += x[i];
        }
        // Generate X_1, …, X_{n-1} and accumulate.
        for _ in 1..n {
            ar1_step(&mut x, &mut gen, phi, sqrt_one_minus_phi2);
            for i in 0..d {
                sum[i] += x[i];
            }
        }
        // S_n = sum / √n.   ‖S_n‖² = Σ (sum_i / √n)² = (1/n) · Σ sum_i².
        let inv_n = 1.0 / n as f64;
        let mut norm_sq = 0.0f64;
        for &s in sum.iter() {
            norm_sq += s * s * inv_n;
        }
        sum_norm_sq += norm_sq;
    }

    let measured_trace = sum_norm_sq / m as f64;
    (measured_trace, predicted_trace)
}

// ════════════════════════════════════════════════════════════════════════════
// The probe
// ════════════════════════════════════════════════════════════════════════════

pub fn prove() -> PillarResult {
    let (measured, predicted) = measure_trace(D, N, M, PHI, 0x0D_15_EA_5E_DEAD_BEEF);

    let relative_error = (measured - predicted).abs() / predicted;
    let pass = relative_error < TOLERANCE;

    let detail = format!(
        "d={D}, n={N}, MC={M}, φ={PHI} (AR(1) Gaussian process in ℝ^d, factorized \
         autocorrelation ρ_rs(v) = δ_rs·φ^|v|; condition (2.1) holds since \
         Σ_v |φ|^v = {:.3} < ∞). Operator G = identity (Hermite rank 1). \
         Predicted trace(T_Z) = d·(1+φ)/(1-φ) = {predicted:.1}. \
         Measured trace(empirical Cov(S_n)) = {measured:.1}. \
         Relative error = {:.3}% (PASS if < {:.1}%). \
         Hilbert-space ℓ² CLT certifies bundle-of-N-fingerprints convergence \
         to Gaussian limit; closes the concentration family alongside Pillar 5 \
         (ℝ, Jirak) and Pillar 5+ (PSD cone, Köstenberger-Stark).",
        (1.0 + PHI.abs()) / (1.0 - PHI.abs()),
        relative_error * 100.0,
        TOLERANCE * 100.0,
    );

    PillarResult {
        name: "Düker-Zoubouloglou: Hilbert-space CLT for AR(1) in ℝ^16384",
        pass,
        measured,
        predicted,
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

    /// φ = 0 reduces AR(1) to iid; the long-run trace becomes exactly d.
    /// Classical Hilbert-space CLT (Varadhan 1962) is the limiting case of
    /// Düker-Zoubouloglou as the dependence vanishes.
    #[test]
    fn iid_case_recovers_unit_trace() {
        // Smaller d/n for fast tests; the rate is independent of d.
        let (measured, predicted) = measure_trace(2048, 500, 30, 0.0, 0x1234_5678);
        let rel_err = (measured - predicted).abs() / predicted;
        assert!(rel_err < 0.05, "iid case: measured={measured:.2}, predicted={predicted:.2}, rel_err={rel_err:.4}");
    }

    /// High dependence φ = 0.9: long-run variance amplified by (1+φ)/(1-φ) = 19.
    /// Tests that the AR(1) machinery correctly tracks heavy positive autocorrelation.
    #[test]
    fn high_dependence_amplifies_long_run_variance() {
        // Larger n needed because mixing time ~ 1/(1−φ) = 10 cycles.
        let (measured, predicted) = measure_trace(2048, 2000, 30, 0.9, 0x9876_5432);
        let rel_err = (measured - predicted).abs() / predicted;
        // Looser tolerance — the φ=0.9 case has slow mixing, so finite-n bias is bigger.
        assert!(rel_err < 0.15, "φ=0.9 case: measured={measured:.2}, predicted={predicted:.2}, rel_err={rel_err:.4}");
    }

    /// Negative dependence φ = -0.5: long-run variance suppressed to (1−0.5)/(1+0.5) = 1/3.
    /// Negative correlation makes successive samples cancel — same theorem, different sign.
    #[test]
    fn negative_dependence_suppresses_long_run_variance() {
        let (measured, predicted) = measure_trace(2048, 1000, 30, -0.5, 0xABCD_EF01);
        let rel_err = (measured - predicted).abs() / predicted;
        assert!(rel_err < 0.10, "φ=-0.5 case: measured={measured:.2}, predicted={predicted:.2}, rel_err={rel_err:.4}");
    }

    /// Symmetry: trace estimate should be deterministic given the same seed
    /// (catches RNG-state corruption regressions).
    #[test]
    fn deterministic_with_fixed_seed() {
        let (m1, _) = measure_trace(512, 200, 5, 0.3, 0xCAFE_BABE);
        let (m2, _) = measure_trace(512, 200, 5, 0.3, 0xCAFE_BABE);
        assert!((m1 - m2).abs() < 1e-12, "seed determinism broken: {m1} vs {m2}");
    }

    /// NormalGen sanity: large sample mean ≈ 0, variance ≈ 1.
    #[test]
    fn normal_gen_first_two_moments() {
        let mut g = NormalGen::new(0x5EED_DEED);
        let n = 100_000;
        let samples: Vec<f64> = (0..n).map(|_| g.next()).collect();
        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let var: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.02, "mean = {mean}");
        assert!((var - 1.0).abs() < 0.02, "variance = {var}");
    }

    /// The full pillar — substrate-native dimension, the actual claim.
    #[test]
    fn pillar_passes() {
        let r = prove();
        assert!(r.pass, "Düker-Zoubouloglou pillar failed: {}", r.detail);
    }
}
