//! Lyons-Victoir cubature on Wiener space — framework for precomputed-basis
//! "splat hydration" of path signatures.
//!
//! Citation: T. Lyons & N. Victoir, "Cubature on Wiener space",
//! Proc. R. Soc. Lond. A 460, 169-198 (2004).
//!
//! # The idea
//!
//! A *cubature formula of degree N* on Wiener space in dimension d is a
//! finite collection of paths {γ₁, …, γ_M} with weights {λ₁, …, λ_M} such
//! that for **every** signature element ℓ of degree ≤ N:
//!
//! ```text
//!   E[ℓ(B)]  =  Σ_k  λ_k · ℓ(γ_k)
//! ```
//!
//! where B is standard d-dimensional Brownian motion. Equivalently, the
//! finite measure δ-supported on the cubature paths integrates the
//! truncated tensor algebra exactly.
//!
//! For the practitioner this is the splat-hydration analog: precompute
//! the cubature basis once (per (d, N)), then any depth-N signature
//! computation reduces to a small linear combination over the basis.
//!
//! # Why this matters past depth 8
//!
//! Naive `signature_truncated` is O(d^(2N)) per pair. For d=4, N=8 that's
//! 4^16 ≈ 4.3 × 10⁹ flops per multiply. Cubature shrinks this to O(M · N)
//! where M is the cubature path count — sub-exponential in N for fixed d.
//!
//! # What this module provides
//!
//! - `CubatureBasis` — the (paths, weights, dim, degree) container.
//! - `validate_basics()` — necessary conditions every cubature must satisfy.
//! - `trivial_constant_cubature()` — the degree-0 cubature (one constant
//!   path with weight 1). Trivially correct; serves as the framework's
//!   correctness anchor.
//! - `hydrate_signature()` — the projection API surface (currently
//!   identity, see "What this does NOT provide" below).
//!
//! # What this module does NOT provide
//!
//! **Concrete non-trivial cubatures are deferred.** Constructing a
//! degree-N cubature for d ≥ 2 is a research task per (d, N) pair —
//! Lyons-Victoir 2004 give examples for low (d, N), Gyurkó-Lyons 2010
//! extend them, and modern symbolic-computation pipelines (Gröbner-basis
//! moment matching) generate them per request. None of these are
//! one-line constructions.
//!
//! In particular, the obvious "out-and-back along coordinate axes"
//! construction is **NOT** a cubature, because by Hambly-Lyons 2010
//! tree-like equivalence such paths have **zero signature at every
//! level** — they cannot match the non-zero Brownian moments E[B^i B^j].
//! This is the same uniqueness theorem that justifies sigker's Index-
//! regime classification (see `codec.rs`); it precludes the lazy
//! construction.
//!
//! Real Lyons-Victoir cubatures use **non-tree-like** paths — typically
//! straight-line steps in carefully chosen directions, weighted by
//! solving a moment-matching system. The d=2, N=3 example from Lyons-
//! Victoir 2004 §3 uses 3 such paths; d=3, N=3 needs 7; d=2, N=5 needs
//! ~5; etc. We document the construction recipe and ship the framework,
//! leaving the concrete cubatures as a follow-up populated from the
//! literature on demand.
//!
//! # Activation path
//!
//! When a downstream consumer (lance-graph-osint, AriGraph traversal)
//! actually needs cubature acceleration:
//!
//! 1. Identify the (d, N) pair from the consumer's path width and
//!    desired accuracy.
//! 2. Look up the cubature in Lyons-Victoir 2004 §3, Gyurkó-Lyons 2010,
//!    or Bayer-Lyons-Schoutens 2008 — or commission a symbolic
//!    construction.
//! 3. Encode the (paths, weights) as a `CubatureBasis` and verify with
//!    `validate_basics()` plus a moment-matching test against
//!    `signature_truncated` of each path.
//! 4. Replace the identity `hydrate_signature()` with the projection
//!    `Σ_k λ_k · 〈sig(query), sig(γ_k)〉 · sig(γ_k)`.
//!
//! Until then, this module ships as the *spine* of the architecture
//! with the same DEFERRED discipline as jc Pillar 11 (Hambly-Lyons).

use crate::signature::{Signature, signature_truncated};

// ════════════════════════════════════════════════════════════════════════════
// Core types
// ════════════════════════════════════════════════════════════════════════════

/// A cubature basis: paths and matching weights, with the (dim, degree)
/// metadata that identifies which moment-matching guarantee they satisfy.
#[derive(Clone, Debug)]
pub struct CubatureBasis {
    /// Spatial dimension of the underlying Wiener space.
    pub dim: usize,
    /// Cubature degree — moments up to this order are matched exactly.
    pub degree: usize,
    /// Cubature paths. Each is a Vec<Vec<f64>> with consistent dim.
    pub paths: Vec<Vec<Vec<f64>>>,
    /// Cubature weights. `weights.len() == paths.len()`. Sum to 1.
    pub weights: Vec<f64>,
}

impl CubatureBasis {
    /// Number of cubature paths.
    pub fn cardinality(&self) -> usize {
        self.paths.len()
    }

    /// Validate the necessary conditions for a cubature on Wiener space:
    ///
    /// - weights sum to 1 (probability measure)
    /// - all paths have consistent dim
    /// - level-1 cubature-weighted average is zero (matches E[B(1)] = 0)
    ///
    /// **This is a necessary, not sufficient, condition.** Full validation
    /// requires moment matching at every multi-index of length ≤ degree
    /// against the closed-form Brownian moments — see Lyons-Victoir 2004
    /// Lemma 3.3 for the full system.
    pub fn validate_basics(&self) -> Result<(), String> {
        if self.paths.is_empty() {
            return Err("cubature has zero paths".into());
        }
        let weight_sum: f64 = self.weights.iter().sum();
        if (weight_sum - 1.0).abs() > 1e-10 {
            return Err(format!(
                "weights sum to {weight_sum}, expected 1.0 (tolerance 1e-10)"
            ));
        }
        if self.weights.len() != self.paths.len() {
            return Err(format!(
                "weights ({}) and paths ({}) length mismatch",
                self.weights.len(),
                self.paths.len()
            ));
        }
        let dim = self.dim;
        for (i, path) in self.paths.iter().enumerate() {
            if path.is_empty() {
                return Err(format!("path {i} has zero points"));
            }
            for (j, point) in path.iter().enumerate() {
                if point.len() != dim {
                    return Err(format!(
                        "path {i} point {j} has dim {} but basis declares {dim}",
                        point.len()
                    ));
                }
            }
        }
        // Level-1 moment: Σ_k λ_k · (γ_k(end) − γ_k(start)) = E[B(1)] = 0.
        let mut level1_sum = vec![0.0f64; dim];
        for (path, &w) in self.paths.iter().zip(self.weights.iter()) {
            let start = &path[0];
            let end = &path[path.len() - 1];
            for i in 0..dim {
                level1_sum[i] += w * (end[i] - start[i]);
            }
        }
        for (i, &m) in level1_sum.iter().enumerate() {
            if m.abs() > 1e-9 {
                return Err(format!(
                    "level-1 moment {i} = {m}, expected 0 (tolerance 1e-9)"
                ));
            }
        }
        Ok(())
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Trivial degree-0 cubature.
//
// A single constant path with weight 1 trivially satisfies the cubature
// property at degree 0: it integrates the constant signature element
// (S^0 ≡ 1) exactly to E[1] = 1, and trivially satisfies degree-1 because
// a constant path has zero displacement.
//
// This is the framework's correctness anchor: the smallest possible
// non-empty cubature, sufficient to verify the validation pipeline, the
// hydration API surface, and the signature computation contract.
// ════════════════════════════════════════════════════════════════════════════

/// The degree-0 cubature: one constant path at the origin, weight 1.
/// Trivially correct at degree 0; the framework's correctness anchor.
pub fn trivial_constant_cubature(dim: usize) -> CubatureBasis {
    let constant_path: Vec<Vec<f64>> = vec![vec![0.0; dim]; 2];
    CubatureBasis {
        dim,
        degree: 0,
        paths: vec![constant_path],
        weights: vec![1.0],
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Hydration — the API surface.
//
// Currently identity: returns the query path's truncated signature. The
// production hydration projects the query onto the cubature span:
//
//   sig_hydrated(query)  =  Σ_k  λ_k · 〈sig(query), sig(γ_k)〉 · sig(γ_k)
//
// which reduces to a basis lookup once `sig(γ_k)` is precomputed.
// Activation requires a non-trivial cubature; until then the identity
// hydration documents the surface and lets downstream consumers wire
// against the API.
// ════════════════════════════════════════════════════════════════════════════

/// Compute the cubature-projected signature of the query path.
///
/// **Currently identity** — returns `signature_truncated(query, basis.degree)`
/// directly. Production hydration (basis-projection lookup) is gated on a
/// non-trivial `CubatureBasis` being supplied; the trivial degree-0 cubature
/// makes the identity hydration trivially correct because at degree 0 the
/// query's signature reduces to the single coefficient 1.
pub fn hydrate_signature(query: &[Vec<f64>], basis: &CubatureBasis) -> Signature {
    assert_eq!(
        query[0].len(),
        basis.dim,
        "query path dim must match basis dim"
    );
    signature_truncated(query, basis.degree)
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivial_cubature_validates() {
        for dim in 1..=4 {
            let basis = trivial_constant_cubature(dim);
            basis.validate_basics().expect("trivial cubature must validate");
            assert_eq!(basis.cardinality(), 1);
            assert_eq!(basis.degree, 0);
            assert_eq!(basis.dim, dim);
        }
    }

    #[test]
    fn validate_rejects_unnormalized_weights() {
        let basis = CubatureBasis {
            dim: 2,
            degree: 0,
            paths: vec![vec![vec![0.0, 0.0], vec![0.0, 0.0]]],
            weights: vec![0.5], // sums to 0.5, not 1
        };
        assert!(basis.validate_basics().is_err());
    }

    #[test]
    fn validate_rejects_inconsistent_dim() {
        let basis = CubatureBasis {
            dim: 2,
            degree: 0,
            paths: vec![vec![vec![0.0, 0.0], vec![0.0, 0.0, 0.0]]], // mixed dim
            weights: vec![1.0],
        };
        assert!(basis.validate_basics().is_err());
    }

    #[test]
    fn validate_rejects_nonzero_level1() {
        // Single open path with weight 1 has nonzero level-1 → not a cubature.
        let basis = CubatureBasis {
            dim: 2,
            degree: 1,
            paths: vec![vec![vec![0.0, 0.0], vec![1.0, 0.5]]],
            weights: vec![1.0],
        };
        let err = basis.validate_basics().expect_err("nonzero level-1 must reject");
        assert!(err.contains("level-1"), "error should mention level-1: {err}");
    }

    #[test]
    fn validate_rejects_length_mismatch() {
        let basis = CubatureBasis {
            dim: 1,
            degree: 0,
            paths: vec![vec![vec![0.0], vec![0.0]]],
            weights: vec![0.5, 0.5], // 2 weights, 1 path
        };
        assert!(basis.validate_basics().is_err());
    }

    #[test]
    fn hydration_returns_correct_dim_and_depth() {
        let basis = trivial_constant_cubature(3);
        let query = vec![vec![0.0, 0.0, 0.0], vec![1.0, 0.5, -0.2]];
        let sig = hydrate_signature(&query, &basis);
        assert_eq!(sig.dim, 3);
        assert_eq!(sig.depth, basis.degree);
    }

    #[test]
    fn hydration_at_degree_zero_is_constant_one() {
        // Degree-0 signature is the scalar 1, regardless of query.
        let basis = trivial_constant_cubature(2);
        let query = vec![vec![0.0, 0.0], vec![3.0, -1.0], vec![1.0, 4.0]];
        let sig = hydrate_signature(&query, &basis);
        // Signature has only level 0 at depth 0.
        assert_eq!(sig.levels.len(), 1);
        assert_eq!(sig.levels[0], vec![1.0]);
    }
}
