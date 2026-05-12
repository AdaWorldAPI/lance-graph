//! Σ-Codebook Viability Probe — empirical measurement, NOT a pillar.
//!
//! # Mission
//!
//! Decide between encoding strategies for white-matter Σ-edges:
//!
//!   ρ ≥ 0.99   →   256-entry codebook viable; 1-byte sidecar suffices
//!   ρ ∈ [0.95, 0.99) → marginal; consider 2-byte index or hybrid scheme
//!   ρ < 0.95   →   codebook insufficient; separate 7-float storage needed
//!
//! No theorem to prove — this is a fitness check for an architectural choice.
//! Run it BEFORE writing any production CausalEdgeTensor code so the encoding
//! decision rests on measured cluster quality of plausible Σ-distributions
//! rather than on hope.
//!
//! # What is measured
//!
//! Synthesize N_EDGES = 10,000 plausible edges with realistic field
//! distributions (Beta-shaped frequency/confidence, uniform discrete fields).
//! Map each edge to a 2×2 SPD Σ-tensor via a deterministic, semantically
//! reasoned mapping (strength from evidence, anisotropy from confidence,
//! orientation from direction). Run Lloyd's k-Means (k=256) in
//! **log-Euclidean space** — the standard linearization of the affine-
//! invariant Riemannian metric on the SPD cone. Compute R² in log space.
//!
//! ```text
//!   R² = 1 − Σ d²(Σ_i, codebook[assignment_i])  /  Σ d²(Σ_i, Σ_global_mean)
//! ```
//!
//! where d is the Frobenius distance on log-Σ (a Hilbert-space inner-product
//! distance — clean target for k-Means).
//!
//! # Limitations explicitly stated
//!
//! - 2×2 SPD chosen for closed-form eigendecomp; full anisotropic Σ would
//!   typically be 3×3 (3 spatial axes) or higher. Result generalizes
//!   monotonically: more dimensions = harder to cluster, lower R².
//! - The synthetic distribution is *plausible*, not measured from production.
//!   Real CausalEdge stream may cluster better (more structure) or worse
//!   (heavier tails). To re-run with a different distribution, change the
//!   constants in synthesize_edge.
//! - Log-Euclidean ≠ affine-invariant. The two distances agree to first
//!   order around any point, diverge in the tails. For codebook viability
//!   the difference is negligible; for production lookup, use whichever
//!   metric the production code uses.

use crate::PillarResult;

const N_EDGES: usize = 10_000;
const K_CODEBOOK: usize = 256;
const N_KMEANS_ITER: usize = 100;
const SEED: u64 = 0xC0DE_BABE_5EED_F00D;

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

// ════════════════════════════════════════════════════════════════════════════
// 2×2 symmetric matrix (used for both SPD Σ and log-Σ which is symmetric).
// Slimmed-down mirror of the Spd2 type from koestenberger.rs — duplicated
// intentionally to keep the probe self-contained. If a third pillar/probe
// needs SPD math, this should be promoted to a shared `hadamard` module.
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
struct Sym2 {
    a: f64,
    b: f64,
    c: f64,
}

#[allow(dead_code)] // exp_sym + frobenius_* exercised in tests; kept for round-trip sanity.
impl Sym2 {
    /// Eigendecomposition of 2×2 symmetric matrix. Returns (λ₁, λ₂, cos θ, sin θ)
    /// where columns of R(θ) = [[c,-s],[s,c]] are eigenvectors for (λ₁, λ₂).
    fn eig(&self) -> (f64, f64, f64, f64) {
        let half_trace = (self.a + self.c) / 2.0;
        let half_diff = (self.a - self.c) / 2.0;
        let disc = (half_diff * half_diff + self.b * self.b).sqrt();
        let l1 = half_trace + disc;
        let l2 = half_trace - disc;
        let theta = if self.b.abs() < 1e-15 && (self.a - self.c).abs() < 1e-15 {
            0.0
        } else {
            0.5 * (2.0 * self.b).atan2(self.a - self.c)
        };
        (l1, l2, theta.cos(), theta.sin())
    }

    /// Compose 2×2 symmetric from eigenvalues + rotation.
    fn from_eig(l1: f64, l2: f64, c: f64, s: f64) -> Self {
        Self {
            a: c * c * l1 + s * s * l2,
            b: c * s * (l1 - l2),
            c: s * s * l1 + c * c * l2,
        }
    }

    /// log of an SPD matrix — eigendecomp + log of eigenvalues.
    /// Result is symmetric (NOT SPD in general — log can have negative eigenvalues).
    fn log_spd(&self) -> Self {
        let (l1, l2, c, s) = self.eig();
        Self::from_eig(l1.max(1e-300).ln(), l2.max(1e-300).ln(), c, s)
    }

    /// exp of a symmetric matrix — eigendecomp + exp of eigenvalues.
    /// Result is SPD (exp eigenvalues are always positive).
    fn exp_sym(&self) -> Self {
        let (l1, l2, c, s) = self.eig();
        Self::from_eig(l1.exp(), l2.exp(), c, s)
    }

    /// Frobenius norm: ‖M‖_F² = a² + 2b² + c² (off-diagonals counted twice).
    fn frobenius_sq(&self) -> f64 {
        self.a * self.a + 2.0 * self.b * self.b + self.c * self.c
    }

    /// Frobenius distance between two symmetric matrices.
    fn frobenius_distance_sq(&self, other: &Self) -> f64 {
        let d = Self {
            a: self.a - other.a,
            b: self.b - other.b,
            c: self.c - other.c,
        };
        d.frobenius_sq()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Edge synthesizer — Beta-shaped frequency/confidence, uniform discrete fields
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
struct EdgeFields {
    /// Aggregated evidence frequency, [0, 1]. Beta-shaped, biased low (most
    /// edges have weak evidence; confident edges are rare).
    frequency: f64,
    /// Confidence in the frequency estimate, [0, 1]. Beta-shaped, biased
    /// high (when evidence exists, it's usually decisive).
    confidence: f64,
    /// Discrete direction, 0..8 (3-bit field in CausalEdge64).
    direction: u8,
}

fn synthesize_edge(state: &mut u64) -> EdgeFields {
    // Frequency: u² gives PDF ∝ 1/√x for x ∈ (0, 1]; mean ≈ 0.33, biased low.
    let frequency = rand_uniform(state).powi(2);
    // Confidence: 1 − (1−u)² gives PDF ∝ 1/√(1−x); mean ≈ 0.67, biased high.
    let confidence = {
        let u = rand_uniform(state);
        1.0 - (1.0 - u).powi(2)
    };
    let direction = (splitmix64(state) & 0b111) as u8; // 8 directions
    EdgeFields { frequency, confidence, direction }
}

/// Map an edge's bit-fields to an implicit 2×2 SPD Σ-tensor.
///
/// Reasoning:
///   - Strength (overall scale of Σ) ~ frequency · confidence
///     (strong evidence + high confidence → big-magnitude Σ)
///   - Anisotropy (eigenvalue ratio) ~ confidence
///     (high confidence → narrow major axis, big λ₁/λ₂ ratio)
///   - Orientation (rotation angle) = direction · π/8
///     (8 discrete directions evenly spaced on [0, π))
fn edge_to_sigma(edge: &EdgeFields) -> Sym2 {
    let strength = (edge.frequency * edge.confidence + 0.05).max(0.01);
    // Spread parameter: high confidence → small spread → high anisotropy
    let spread = 0.3 + 0.7 * (1.0 - edge.confidence);
    let l1 = strength * (1.0 + spread);
    let l2 = strength * (1.0 - 0.5 * spread).max(0.01);
    let theta = edge.direction as f64 * std::f64::consts::PI / 8.0;
    Sym2::from_eig(l1, l2, theta.cos(), theta.sin())
}

// ════════════════════════════════════════════════════════════════════════════
// Lloyd's k-Means in 3D log-Euclidean space
// ════════════════════════════════════════════════════════════════════════════

#[inline]
fn sq_dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Run Lloyd's k-Means on 3D vectors. Returns (centroids, assignments, iters_run).
/// Empty clusters keep their previous centroid.
fn kmeans_3d(
    data: &[[f64; 3]],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> (Vec<[f64; 3]>, Vec<usize>, usize) {
    let n = data.len();
    let mut state = seed;
    // Initialize: k random samples (k-means++ would be better but uniform is fine
    // for codebook viability assessment — k-means++ would only improve, not change
    // the qualitative answer).
    let mut centroids: Vec<[f64; 3]> = Vec::with_capacity(k);
    for _ in 0..k {
        let idx = (splitmix64(&mut state) as usize) % n;
        centroids.push(data[idx]);
    }

    let mut assignments = vec![0usize; n];
    let mut iters_run = 0;

    for it in 0..max_iter {
        iters_run = it + 1;
        // Assignment step
        let mut changed = 0usize;
        for (i, point) in data.iter().enumerate() {
            let mut best_idx = 0usize;
            let mut best_dist = f64::INFINITY;
            for (j, centroid) in centroids.iter().enumerate() {
                let d = sq_dist3(point, centroid);
                if d < best_dist {
                    best_dist = d;
                    best_idx = j;
                }
            }
            if assignments[i] != best_idx {
                changed += 1;
                assignments[i] = best_idx;
            }
        }
        // Update step
        let mut sums = vec![[0.0f64; 3]; k];
        let mut counts = vec![0usize; k];
        for (i, &assignment) in assignments.iter().enumerate() {
            sums[assignment][0] += data[i][0];
            sums[assignment][1] += data[i][1];
            sums[assignment][2] += data[i][2];
            counts[assignment] += 1;
        }
        for j in 0..k {
            if counts[j] > 0 {
                let cnt = counts[j] as f64;
                centroids[j] = [
                    sums[j][0] / cnt,
                    sums[j][1] / cnt,
                    sums[j][2] / cnt,
                ];
            }
            // else: keep old centroid (empty cluster)
        }
        if changed == 0 {
            break;
        }
    }
    (centroids, assignments, iters_run)
}

// ════════════════════════════════════════════════════════════════════════════
// The probe
// ════════════════════════════════════════════════════════════════════════════

pub fn prove() -> PillarResult {
    let mut state = SEED;

    // 1. Synthesize edges
    let edges: Vec<EdgeFields> = (0..N_EDGES).map(|_| synthesize_edge(&mut state)).collect();

    // 2. Map each edge to Σ
    let sigmas: Vec<Sym2> = edges.iter().map(edge_to_sigma).collect();

    // 3. Convert each Σ to log-Σ, pack as 3-vector for k-Means
    let log_vecs: Vec<[f64; 3]> = sigmas
        .iter()
        .map(|s| {
            let l = s.log_spd();
            [l.a, l.b, l.c]
        })
        .collect();

    // 4. Compute global log-mean (for R² denominator)
    let n = log_vecs.len() as f64;
    let mut log_mean = [0.0f64; 3];
    for v in &log_vecs {
        log_mean[0] += v[0];
        log_mean[1] += v[1];
        log_mean[2] += v[2];
    }
    log_mean[0] /= n;
    log_mean[1] /= n;
    log_mean[2] /= n;

    // 5. SST (total sum of squares from log-mean)
    let sst: f64 = log_vecs.iter().map(|v| sq_dist3(v, &log_mean)).sum();

    // 6. Run k-Means
    let (centroids, assignments, iters_run) =
        kmeans_3d(&log_vecs, K_CODEBOOK, N_KMEANS_ITER, SEED ^ 0xA5A5_A5A5);

    // 7. SSE (sum of squares from assigned centroid)
    let sse: f64 = log_vecs
        .iter()
        .enumerate()
        .map(|(i, v)| sq_dist3(v, &centroids[assignments[i]]))
        .sum();

    // 8. R² in log-Euclidean space
    let r_squared = if sst > 1e-300 { 1.0 - sse / sst } else { 0.0 };

    // 9. Diagnostic stats
    let used_clusters = {
        let mut seen = vec![false; K_CODEBOOK];
        for &a in &assignments {
            seen[a] = true;
        }
        seen.iter().filter(|&&x| x).count()
    };

    let max_eigenvalue_ratio = sigmas
        .iter()
        .map(|s| {
            let (l1, l2, _, _) = s.eig();
            (l1.abs() / l2.abs().max(1e-300)).max(l2.abs() / l1.abs().max(1e-300))
        })
        .fold(1.0f64, f64::max);

    // 10. PASS criteria — graduated
    //   ≥ 0.99: clearly viable (1-byte sidecar)
    //   ≥ 0.95: marginal — caller decides
    //   <  0.95: codebook insufficient
    let pass = r_squared >= 0.99;

    let recommendation = if r_squared >= 0.99 {
        "CODEBOOK VIABLE — 256-entry codebook + 1-byte index per edge captures \
         the Σ-distribution at high fidelity. Recommend Option A (Σ-Codebook \
         in bgz17-Stil) or Option C (SchemaSidecar Block 14/15)."
    } else if r_squared >= 0.95 {
        "MARGINAL — codebook captures most variance but tails are lossy. \
         Recommend either k=4096 (12-bit index, 1.5 bytes per edge) for \
         tighter fit, or hybrid scheme (codebook for common Σ, sidecar for \
         outliers above some quantile)."
    } else {
        "CODEBOOK INSUFFICIENT — Σ-distribution does not cluster well at k=256. \
         Recommend separate Lance storage (full 7-float Σ per edge as side-table) \
         instead of in-container quantization. Container layout stays unchanged."
    };

    let detail = format!(
        "Edges synthesized: {N_EDGES}. Codebook size: k={K_CODEBOOK}. k-Means iters: {iters_run}/{N_KMEANS_ITER}. \
         Used clusters: {used_clusters}/{K_CODEBOOK} (collapsed-cluster check). \
         R² (log-Euclidean) = {r_squared:.6} (PASS if ≥ 0.99). \
         Σ eigenvalue-ratio max across dataset = {max_eigenvalue_ratio:.2}× \
         (anisotropy spread indicator; higher = more elongated ellipses). \
         Synthesis: Beta-shaped freq (biased low), Beta-shaped conf (biased high), \
         8 discrete directions, σ-derivation: strength=freq·conf, anisotropy=conf, rotation=direction·π/8. \
         {recommendation}"
    );

    PillarResult {
        name: "Σ-Codebook Viability Probe (k=256, 10k edges, log-Euclidean)",
        pass,
        measured: r_squared,
        predicted: 0.99,
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

    #[test]
    fn sym2_log_exp_round_trip() {
        let m = Sym2 { a: 2.0, b: 0.3, c: 1.5 };
        let logm = m.log_spd();
        let back = logm.exp_sym();
        // Round-trip should give back the original SPD.
        assert!(m.frobenius_distance_sq(&back).sqrt() < 1e-9,
            "log/exp round-trip failed: {m:?} → {logm:?} → {back:?}");
    }

    #[test]
    fn identity_log_is_zero() {
        let i = Sym2 { a: 1.0, b: 0.0, c: 1.0 };
        let l = i.log_spd();
        assert!(l.frobenius_sq().sqrt() < 1e-12, "log(I) should be 0, got {l:?}");
    }

    #[test]
    fn edge_synthesizer_is_deterministic() {
        let mut s1 = 42u64;
        let mut s2 = 42u64;
        for _ in 0..100 {
            let e1 = synthesize_edge(&mut s1);
            let e2 = synthesize_edge(&mut s2);
            assert_eq!(e1.direction, e2.direction);
            assert!((e1.frequency - e2.frequency).abs() < 1e-15);
            assert!((e1.confidence - e2.confidence).abs() < 1e-15);
        }
    }

    #[test]
    fn edge_to_sigma_produces_spd() {
        let mut state = 0xABCDu64;
        for _ in 0..1000 {
            let edge = synthesize_edge(&mut state);
            let sigma = edge_to_sigma(&edge);
            // Check positive determinant (necessary condition for SPD given a > 0)
            let det = sigma.a * sigma.c - sigma.b * sigma.b;
            assert!(sigma.a > 0.0, "a not positive: {sigma:?}");
            assert!(sigma.c > 0.0, "c not positive: {sigma:?}");
            assert!(det > 0.0, "determinant not positive: {sigma:?}");
        }
    }

    #[test]
    fn kmeans_converges_on_separable_data() {
        // Three well-separated clusters; k=3 should partition perfectly.
        let mut data = Vec::new();
        let mut state = 1u64;
        for _ in 0..100 {
            data.push([rand_uniform(&mut state) * 0.1, rand_uniform(&mut state) * 0.1, 0.0]);
            data.push([10.0 + rand_uniform(&mut state) * 0.1, 0.0, 0.0]);
            data.push([0.0, 10.0 + rand_uniform(&mut state) * 0.1, 0.0]);
        }
        let (centroids, assignments, iters) = kmeans_3d(&data, 3, 100, 42);
        assert!(iters < 100, "should converge well before max_iter, got {iters}");
        assert_eq!(centroids.len(), 3);
        let used = {
            let mut seen = vec![false; 3];
            for &a in &assignments {
                seen[a] = true;
            }
            seen.iter().filter(|&&x| x).count()
        };
        assert_eq!(used, 3, "all 3 clusters should be used");
    }

    #[test]
    fn probe_runs_and_reports_meaningful_result() {
        let r = prove();
        // r.pass might be true or false — we don't assert that. We only assert
        // that the measurement returned a non-trivial result.
        assert!(r.measured.is_finite(), "R² should be finite");
        assert!(r.measured >= 0.0 && r.measured <= 1.0, "R² out of [0, 1]: {}", r.measured);
        assert!(!r.detail.is_empty());
    }
}
