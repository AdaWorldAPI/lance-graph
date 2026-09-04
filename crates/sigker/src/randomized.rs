//! Randomized signatures — the practical bridge from infinite-dimensional
//! signature space to a fixed-width carrier comparable to Vsa16k.
//!
//! Citation: C. Cuchiero, L. Gonon, L. Grigoryeva, J.-P. Ortega, J. Teichmann,
//! "Discrete-time signatures and randomness in reservoir computing",
//! IEEE Transactions on Neural Networks and Learning Systems, 2021.
//! Also: Cuchiero-Schmocker-Teichmann, "Global universal approximation of
//! functional input maps on weighted spaces", 2023, arXiv:2306.03303.
//!
//! # The construction
//!
//! Given a path X = (x₁, …, x_T) in ℝ^d and a target dimension k, the
//! randomized signature evolves a state z ∈ ℝ^k by
//!
//!   z_{t+1} = z_t + Σ_{i=1}^d σ(A_i · z_t + b_i) · Δx_t^(i)
//!
//! where:
//!   - A_i ∈ ℝ^{k×k} are random matrices with entries from N(0, 1/k)
//!   - b_i ∈ ℝ^k are random bias vectors from N(0, 1/k)
//!   - σ is a non-polynomial activation (here: tanh)
//!
//! The Cuchiero et al. theorem states that the map X ↦ z_T is a *universal
//! approximator* of continuous functions on path space — i.e., random
//! features suffice to recover the expressive power of the full signature.
//!
//! # Why this matters for lance-graph
//!
//! - **Fixed width**: z_T ∈ ℝ^k regardless of path length, comparable to
//!   the Vsa16kF32 carrier. After bf16 packing this is ~k·2 bytes.
//! - **Cheap to compute**: O(T · k²) flops, no tensor algebra materialization.
//! - **Stable to perturbation**: the randomness is *fixed* per encoder
//!   instance (seeded), so two paths produce comparable encodings.
//! - **Index-regime classification preserved**: the universality theorem
//!   guarantees information-preservation up to the approximation rate
//!   k^(-1/(2d)) — sharper than CAM-PQ codebook quantization on average.
//!
//! # Performance envelope
//!
//! For k=4096, d=8, T=64 (typical OSINT sub-path length):
//!   - 64 · 8 · 4096² ≈ 8.6 GFLOPS per path.
//!   - At 100 GFLOPS sustained on a modern core: ~85 ms per path.
//!   - Embarrassingly parallel across paths.
//!
//! For comparison Vsa16k bind+bundle on the same path is ~10 µs but is
//! NOT lossless and accumulates Jirak-bounded noise. Sigker's randomized
//! signature is the trade: ~10000× more compute for guaranteed information
//! preservation.

use ndarray::hpc::randomized_signature::randomized_signature_sweep;
use std::f64::consts::PI;

// ════════════════════════════════════════════════════════════════════════════
// Builder + encoder
// ════════════════════════════════════════════════════════════════════════════

/// Builder that materializes the random projections once per encoder
/// instance, then encodes many paths.
pub struct RandomizedSignatureBuilder {
    pub path_dim: usize,
    pub state_dim: usize,
    /// `state_dim · state_dim · path_dim` entries: A[i] flattened row-major.
    matrices: Vec<f64>,
    /// `state_dim · path_dim` entries: b[i] concatenated.
    biases: Vec<f64>,
}

impl RandomizedSignatureBuilder {
    /// Create a new builder with given dimensions and a deterministic seed.
    pub fn new(path_dim: usize, state_dim: usize, seed: u64) -> Self {
        assert!(path_dim > 0, "path_dim must be > 0");
        assert!(state_dim > 0, "state_dim must be > 0");
        let scale = (state_dim as f64).recip().sqrt();
        let mut rng = SplitMix64::new(seed);

        let n_mat = state_dim * state_dim * path_dim;
        let mut matrices = Vec::with_capacity(n_mat);
        for _ in 0..n_mat {
            matrices.push(rng.normal() * scale);
        }

        let n_bias = state_dim * path_dim;
        let mut biases = Vec::with_capacity(n_bias);
        for _ in 0..n_bias {
            biases.push(rng.normal() * scale);
        }

        RandomizedSignatureBuilder {
            path_dim,
            state_dim,
            matrices,
            biases,
        }
    }

    /// Encode a path and return its randomized signature.
    ///
    /// # Implementation
    ///
    /// The recurrence `z ← z + Σ_i tanh(A_i · z + b_i) · Δx^(i)` is solved by
    /// [`ndarray::hpc::randomized_signature::randomized_signature_sweep`] —
    /// the SIMD GEMV sweep that replaced this function's original hand-rolled
    /// row-major scalar loop (`TD-PILLAR11-SCIENTIFIC-LOOPS-BYPASS-NDARRAY-SIMD-1`:
    /// numeric/computational code in the Ada stack is written against
    /// `ndarray::simd::method()`, never a bespoke scalar loop). The buffer
    /// layout is identical (`matrices[i*k*k + row*k + col]`, `biases`
    /// concatenated per-axis) and the same `|Δx_i| < 1e-15` skip applies, so
    /// the delegation carries sigker's numerics unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `path` is empty or `path[0].len() != self.path_dim` — this
    /// crate's original, stricter contract, unchanged by the delegation.
    /// The delegated primitive additionally asserts (via `assert_eq!`, so it
    /// fires in release builds too) that *every* point in `path` shares
    /// `path[0]`'s dimension — a ragged-path guard this crate did not
    /// previously enforce, so a ragged path that used to silently produce a
    /// wrong (partially-truncated) signature now panics with "ragged path"
    /// instead. This is a strict improvement, not a behavior-preserving
    /// no-op — see the `ragged_path_now_panics` test below.
    pub fn encode(&self, path: &[Vec<f64>]) -> RandomizedSignature {
        assert!(!path.is_empty(), "path must have ≥1 point");
        assert_eq!(path[0].len(), self.path_dim, "path point dim mismatch");

        let z = randomized_signature_sweep(path, &self.matrices, &self.biases, self.state_dim);

        RandomizedSignature {
            path_dim: self.path_dim,
            state: z,
        }
    }
}

/// The output of a randomized-signature encode pass.
#[derive(Clone, Debug)]
pub struct RandomizedSignature {
    pub path_dim: usize,
    pub state: Vec<f64>,
}

impl RandomizedSignature {
    pub fn dim(&self) -> usize {
        self.state.len()
    }

    /// L² inner product — natural similarity for randomized signatures.
    pub fn dot(&self, other: &Self) -> f64 {
        debug_assert_eq!(self.state.len(), other.state.len());
        self.state
            .iter()
            .zip(other.state.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Cosine similarity in [-1, 1].
    pub fn cosine(&self, other: &Self) -> f64 {
        let na = self.state.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb = other.state.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na < 1e-12 || nb < 1e-12 {
            return 0.0;
        }
        self.dot(other) / (na * nb)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Minimal deterministic RNG — same constitution as jc (zero deps).
// SplitMix64 + Box-Muller for normals.
// ════════════════════════════════════════════════════════════════════════════

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        // Top 53 bits → uniform in [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        // Box-Muller. Avoid u = 0.
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinism_under_same_seed() {
        let b1 = RandomizedSignatureBuilder::new(3, 32, 0xDEAD_BEEF);
        let b2 = RandomizedSignatureBuilder::new(3, 32, 0xDEAD_BEEF);
        let path = vec![vec![0.0, 0.0, 0.0], vec![1.0, 0.5, -0.2]];
        let s1 = b1.encode(&path);
        let s2 = b2.encode(&path);
        assert_eq!(s1.state, s2.state);
    }

    #[test]
    fn different_seeds_give_different_encodings() {
        let b1 = RandomizedSignatureBuilder::new(2, 16, 1);
        let b2 = RandomizedSignatureBuilder::new(2, 16, 2);
        let path = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let s1 = b1.encode(&path);
        let s2 = b2.encode(&path);
        assert_ne!(s1.state, s2.state);
    }

    #[test]
    fn identical_paths_have_cosine_one() {
        let b = RandomizedSignatureBuilder::new(2, 32, 42);
        let path = vec![vec![0.0, 0.0], vec![1.0, 2.0], vec![3.0, 1.0]];
        let s1 = b.encode(&path);
        let s2 = b.encode(&path);
        assert!((s1.cosine(&s2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn similar_paths_have_high_cosine() {
        let b = RandomizedSignatureBuilder::new(2, 64, 7);
        let p1 = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 1.5]];
        let p2 = vec![vec![0.0, 0.0], vec![1.0, 1.05], vec![2.0, 1.5]];
        let s1 = b.encode(&p1);
        let s2 = b.encode(&p2);
        let cos = s1.cosine(&s2);
        assert!(cos > 0.95, "expected high cosine, got {cos}");
    }

    #[test]
    fn output_dim_matches_state_dim() {
        let b = RandomizedSignatureBuilder::new(3, 128, 0);
        let path = vec![vec![0.0; 3], vec![1.0, 2.0, 3.0]];
        let s = b.encode(&path);
        assert_eq!(s.dim(), 128);
    }

    /// A small scalar oracle mirroring the ORIGINAL hand-rolled recurrence
    /// this module used before delegating to
    /// `ndarray::hpc::randomized_signature::randomized_signature_sweep`.
    /// This test pins that `encode`'s delegated output still matches that
    /// recurrence within a relative tolerance — it would fail if the
    /// delegation silently changed the recurrence, the skip epsilon, the
    /// activation, or the buffer-layout convention (row/col vs i/j swap).
    /// The tolerance is NOT bit-equality: the SIMD GEMV reduces partial
    /// sums and fuses products in a different order than this scalar loop,
    /// so exact bit-equality is not guaranteed by construction.
    fn scalar_reference(
        path: &[Vec<f64>],
        matrices: &[f64],
        biases: &[f64],
        state_dim: usize,
        path_dim: usize,
    ) -> Vec<f64> {
        let k = state_dim;
        let d = path_dim;
        let mut z = vec![0.0f64; k];
        for window in path.windows(2) {
            let delta_x: Vec<f64> = window[1]
                .iter()
                .zip(window[0].iter())
                .map(|(a, b)| a - b)
                .collect();
            let mut z_next = z.clone();
            let mut activated = vec![0.0f64; k];
            for (i, &dx_i) in delta_x.iter().enumerate().take(d) {
                if dx_i.abs() < 1e-15 {
                    continue;
                }
                let a_offset = i * k * k;
                let b_offset = i * k;
                for row in 0..k {
                    let mut sum = biases[b_offset + row];
                    let row_off = a_offset + row * k;
                    for col in 0..k {
                        sum += matrices[row_off + col] * z[col];
                    }
                    activated[row] = sum.tanh();
                }
                for row in 0..k {
                    z_next[row] += activated[row] * dx_i;
                }
            }
            z = z_next;
        }
        z
    }

    #[test]
    fn encode_matches_scalar_reference_within_tolerance() {
        let b = RandomizedSignatureBuilder::new(3, 24, 0x5EED);
        let path = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, -0.5, 0.25],
            vec![0.7, 0.3, -1.1],
            vec![2.0, 2.0, 2.0],
        ];
        let s = b.encode(&path);
        let expected = scalar_reference(&path, &b.matrices, &b.biases, b.state_dim, b.path_dim);
        assert_eq!(s.state.len(), expected.len());
        for (got, want) in s.state.iter().zip(expected.iter()) {
            let scale = want.abs().max(1.0);
            assert!(
                (got - want).abs() / scale < 1e-9,
                "delegated encode diverged from scalar reference: got {got}, want {want}"
            );
        }
    }

    /// Establishes that `encode` now enforces the ragged-path guard the
    /// delegated ndarray primitive carries (`assert_eq!`, so it fires in
    /// release builds too) — a guard this crate's own asserts (which only
    /// check `path[0]`) did not previously provide. Before the delegation
    /// this exact input would NOT have panicked; it would have silently
    /// read out-of-bounds-safe but semantically wrong deltas for the
    /// shorter/longer point instead. This test discriminates the new
    /// behavior from the old one, not merely restating the delegation.
    #[test]
    #[should_panic(expected = "ragged path")]
    fn ragged_path_now_panics() {
        let b = RandomizedSignatureBuilder::new(2, 8, 1);
        let ragged = vec![vec![0.0, 0.0], vec![1.0, 1.0, 1.0]];
        let _ = b.encode(&ragged);
    }
}
