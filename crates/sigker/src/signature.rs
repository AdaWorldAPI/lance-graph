//! Truncated path signatures S_N(X) = (1, ∫dX, ∫∫dX⊗dX, …, ∫…∫dX⊗…⊗dX)
//! up to depth N.
//!
//! Citation chain:
//!   Chen 1957:    "Integration of paths, geometric invariants and a
//!                  generalized Baker-Hausdorff formula", Annals of Math 65.
//!   Lyons 1998:   "Differential equations driven by rough signals",
//!                  Revista Matemática Iberoamericana 14(2).
//!   Hambly-Lyons: "Uniqueness for the signature of a path of bounded
//!                  variation and the reduced path group", Annals of Math 171.
//!
//! # Conventions
//!
//! - A path X is a finite sequence of f64 vectors of common dimension d.
//! - The signature is computed on linear interpolations between consecutive
//!   points (the natural choice for sampled time series).
//! - Truncation depth N = 2 is implemented end-to-end; depths 3+ are
//!   structurally supported via the recursive `level_n_increment` helper
//!   but require k-tensor storage that we keep flat (Vec<f64> of length d^N).
//! - For depth 0 the signature is always the constant 1.

use std::fmt;

// ════════════════════════════════════════════════════════════════════════════
// Signature — flat-storage truncated tensor.
//
// Layout for a path in d dimensions truncated at depth N:
//
//   level[0] : 1 scalar (= 1 by convention)
//   level[1] : d entries  (∫dX^i)
//   level[2] : d² entries (∫∫dX^i ⊗ dX^j) — flat row-major (i, j)
//   level[k] : d^k entries
//
// Total length = (d^(N+1) − 1) / (d − 1) for d > 1, or N+1 for d = 1.
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct Signature {
    /// Path dimension.
    pub dim: usize,
    /// Truncation depth.
    pub depth: usize,
    /// Per-level flat storage. `levels[k]` has length d^k.
    pub levels: Vec<Vec<f64>>,
}

impl Signature {
    /// Identity signature (corresponds to a constant path / empty integral).
    pub fn identity(dim: usize, depth: usize) -> Self {
        let mut levels = Vec::with_capacity(depth + 1);
        levels.push(vec![1.0]); // S^0 = 1
        for k in 1..=depth {
            let len = pow_usize(dim, k);
            levels.push(vec![0.0; len]);
        }
        Signature { dim, depth, levels }
    }

    /// Total number of stored coefficients.
    pub fn len(&self) -> usize {
        self.levels.iter().map(|l| l.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }
}

impl fmt::Display for Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sig(dim={}, depth={})", self.dim, self.depth)?;
        for (k, level) in self.levels.iter().enumerate() {
            write!(f, " | L{}: {} coeffs", k, level.len())?;
        }
        Ok(())
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Truncated signature computation via Chen's identity:
//   S(X concat Y) = S(X) ⊗ S(Y)
// where ⊗ is the tensor product on the truncated tensor algebra.
//
// For piecewise-linear paths with N segments, we accumulate by computing
// the signature of each linear segment in closed form, then multiplying.
// For a linear segment from x to y, the increment Δ = y − x gives:
//
//   S_segment[0] = 1
//   S_segment[1][i] = Δ[i]
//   S_segment[2][i][j] = Δ[i] · Δ[j] / 2!
//   S_segment[k][i₁…iₖ] = Δ[i₁] · … · Δ[iₖ] / k!
// ════════════════════════════════════════════════════════════════════════════

/// Compute the truncated signature of a piecewise-linear path of points,
/// each point being a Vec<f64> of length `dim`.
pub fn signature_truncated(path: &[Vec<f64>], depth: usize) -> Signature {
    assert!(!path.is_empty(), "path must have at least one point");
    let dim = path[0].len();
    assert!(
        path.iter().all(|p| p.len() == dim),
        "all points must share dimension {dim}"
    );

    if path.len() == 1 || depth == 0 {
        return Signature::identity(dim, depth);
    }

    let mut acc = Signature::identity(dim, depth);
    for window in path.windows(2) {
        let delta: Vec<f64> = window[1]
            .iter()
            .zip(window[0].iter())
            .map(|(a, b)| a - b)
            .collect();
        let seg = segment_signature(&delta, depth);
        acc = tensor_multiply(&acc, &seg);
    }
    acc
}

/// Closed-form signature of a single linear segment with increment Δ.
fn segment_signature(delta: &[f64], depth: usize) -> Signature {
    let dim = delta.len();
    let mut levels = Vec::with_capacity(depth + 1);
    levels.push(vec![1.0]);
    let mut factorial = 1.0f64;
    for k in 1..=depth {
        factorial *= k as f64;
        let len = pow_usize(dim, k);
        let mut level = vec![0.0; len];
        // Outer-product expansion of Δ ⊗ ⋯ ⊗ Δ (k times) divided by k!
        // Index mapping: flat_idx = i₁ · d^(k-1) + i₂ · d^(k-2) + … + iₖ.
        for flat in 0..len {
            let mut idx = flat;
            let mut prod = 1.0;
            for _ in 0..k {
                let ax = idx % dim;
                idx /= dim;
                prod *= delta[ax];
            }
            level[flat] = prod / factorial;
        }
        levels.push(level);
    }
    Signature { dim, depth, levels }
}

/// Tensor multiplication on the truncated tensor algebra:
///   (S * T)[k] = Σ_{i+j=k} S[i] ⊗ T[j]
fn tensor_multiply(a: &Signature, b: &Signature) -> Signature {
    debug_assert_eq!(a.dim, b.dim);
    debug_assert_eq!(a.depth, b.depth);
    let dim = a.dim;
    let depth = a.depth;

    let mut out = Signature::identity(dim, depth);
    // Reset L0 — identity already sets it to 1, which is correct since
    // S^0 * T^0 = 1 * 1 = 1.
    out.levels[0][0] = a.levels[0][0] * b.levels[0][0];

    for k in 1..=depth {
        let len_k = pow_usize(dim, k);
        let mut level_k = vec![0.0; len_k];
        // Sum over splits i + j = k.
        for i in 0..=k {
            let j = k - i;
            let len_i = pow_usize(dim, i);
            let len_j = pow_usize(dim, j);
            // Outer product a.levels[i] ⊗ b.levels[j] → flat index space of size d^k.
            for ai in 0..len_i {
                let av = a.levels[i][ai];
                if av == 0.0 {
                    continue;
                }
                for bj in 0..len_j {
                    let bv = b.levels[j][bj];
                    if bv == 0.0 {
                        continue;
                    }
                    // Concatenate the two multi-indices: (ai, bj) → flat in d^k space.
                    // ai is the high-order block (i factors of d), bj is the low.
                    let flat = ai * len_j + bj;
                    level_k[flat] += av * bv;
                }
            }
        }
        out.levels[k] = level_k;
    }
    out
}

// ════════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════════

fn pow_usize(base: usize, exp: usize) -> usize {
    let mut acc = 1usize;
    for _ in 0..exp {
        acc *= base;
    }
    acc
}

// ════════════════════════════════════════════════════════════════════════════
// Tests — Chen's identity, level-1 = increment, depth-0 = identity.
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn identity_signature_shape() {
        let s = Signature::identity(3, 2);
        assert_eq!(s.dim, 3);
        assert_eq!(s.depth, 2);
        assert_eq!(s.levels.len(), 3);
        assert_eq!(s.levels[0], vec![1.0]);
        assert_eq!(s.levels[1].len(), 3);
        assert_eq!(s.levels[2].len(), 9);
    }

    #[test]
    fn single_point_path_is_identity() {
        let path = vec![vec![1.0, 2.0]];
        let s = signature_truncated(&path, 2);
        assert_eq!(s.levels[0][0], 1.0);
        assert!(s.levels[1].iter().all(|&x| x == 0.0));
        assert!(s.levels[2].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn level_1_equals_total_increment() {
        // For a piecewise-linear path the level-1 signature equals
        // the total increment (last − first).
        let path = vec![
            vec![0.0, 0.0],
            vec![1.0, 2.0],
            vec![3.0, 1.0],
            vec![5.0, 4.0],
        ];
        let s = signature_truncated(&path, 1);
        assert!(approx_eq(s.levels[1][0], 5.0 - 0.0, 1e-12));
        assert!(approx_eq(s.levels[1][1], 4.0 - 0.0, 1e-12));
    }

    #[test]
    fn chens_identity_two_segments() {
        // S(X concat Y) should equal tensor_multiply(S(X), S(Y)).
        let x = vec![vec![0.0, 0.0], vec![1.0, 2.0]];
        let y = vec![vec![1.0, 2.0], vec![3.0, 5.0]];
        let xy = vec![vec![0.0, 0.0], vec![1.0, 2.0], vec![3.0, 5.0]];

        let s_x = signature_truncated(&x, 2);
        let s_y = signature_truncated(&y, 2);
        let s_xy = signature_truncated(&xy, 2);
        let s_combined = tensor_multiply(&s_x, &s_y);

        for k in 0..=2 {
            for i in 0..s_xy.levels[k].len() {
                assert!(
                    approx_eq(s_xy.levels[k][i], s_combined.levels[k][i], 1e-10),
                    "Chen's identity failed at level {k}, idx {i}: {} vs {}",
                    s_xy.levels[k][i],
                    s_combined.levels[k][i],
                );
            }
        }
    }

    #[test]
    fn segment_signature_level_2_is_outer_div_2() {
        // For a single segment with Δ = (a, b), S^2 = (Δ⊗Δ)/2 =
        //   [a²/2, ab/2, ab/2, b²/2]   (flat row-major: (0,0),(0,1),(1,0),(1,1))
        let s = segment_signature(&[3.0, 5.0], 2);
        assert!(approx_eq(s.levels[2][0], 9.0 / 2.0, 1e-12)); // (0,0): a²/2
        assert!(approx_eq(s.levels[2][1], 15.0 / 2.0, 1e-12)); // (0,1): ab/2
        assert!(approx_eq(s.levels[2][2], 15.0 / 2.0, 1e-12)); // (1,0): ab/2
        assert!(approx_eq(s.levels[2][3], 25.0 / 2.0, 1e-12)); // (1,1): b²/2
    }

    #[test]
    fn signature_total_length_matches_geometric_sum() {
        // For dim=3, depth=2: 1 + 3 + 9 = 13 entries.
        let s = Signature::identity(3, 2);
        assert_eq!(s.len(), 13);
        // dim=2, depth=3: 1 + 2 + 4 + 8 = 15.
        let s = Signature::identity(2, 3);
        assert_eq!(s.len(), 15);
    }
}
