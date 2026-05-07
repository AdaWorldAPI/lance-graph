//! Signature kernel — inner product 〈S(X), S(Y)〉 in path space.
//!
//! Citation: C. Salvi, T. Cass, J. Foster, T. Lyons, M. Lemercier,
//! "The Signature Kernel is the solution of a Goursat PDE", SIAM Journal
//! on Mathematics of Data Science 3.3 (2021), arXiv:2006.14794.
//!
//! # The truncated form (this implementation)
//!
//! For paths X (length T₁), Y (length T₂) in ℝ^d, the truncated signature
//! kernel of depth N is
//!
//!   K_N(X, Y) = Σ_{k=0}^N 〈S^k(X), S^k(Y)〉
//!
//! For depth 2 with piecewise-linear paths this reduces (after summing the
//! per-segment factorial-corrected outer products) to
//!
//!   K_0 = 1
//!   K_1 = 〈ΔX_total, ΔY_total〉
//!   K_2 = (1/2) [〈ΔX_total, ΔY_total〉² + Σ_segs (cross terms)]
//!
//! We implement K_N(X, Y) directly by computing both signatures (via
//! `signature_truncated`) and taking the level-wise dot product. This is
//! O(d^N · max(T₁, T₂)) per pair — adequate for OSINT-scale paths
//! (d ≤ 8, T ≤ 64, N ≤ 3) and matches the `bgz17` envelope.
//!
//! # The Goursat PDE form (production extension)
//!
//! The full (untruncated) signature kernel satisfies the hyperbolic PDE
//!
//!   ∂²K(s, t) / ∂s ∂t  =  〈Ẋ(s), Ẏ(t)〉 · K(s, t)
//!
//! with boundary K(0, t) = K(s, 0) = 1, and K(X, Y) = K(T₁, T₂). This avoids
//! signature materialization entirely and runs in O(T₁ · T₂) flops on the
//! grid. We expose the API surface (`signature_kernel_pde`) but defer the
//! solver to a follow-up; the truncated form is correct and sufficient for
//! the certification pillar (jc pillar 11) to validate sigker classification.
//!
//! # Universality
//!
//! Salvi et al. prove the signature kernel is *universal* in the
//! Christmann-Steinwart sense on weighted path spaces — i.e., RKHS dense in
//! continuous functions on path space. This is the rigorous form of the
//! claim that sigker preserves all extractable information from a path,
//! and the basis for sigker's Index-regime classification in `codec.rs`.

use crate::signature::signature_truncated;

/// Truncated signature kernel of depth N.
pub fn signature_kernel(x: &[Vec<f64>], y: &[Vec<f64>], depth: usize) -> f64 {
    let s_x = signature_truncated(x, depth);
    let s_y = signature_truncated(y, depth);
    debug_assert_eq!(s_x.dim, s_y.dim);
    debug_assert_eq!(s_x.depth, s_y.depth);

    let mut k = 0.0f64;
    for level in 0..=depth {
        let lx = &s_x.levels[level];
        let ly = &s_y.levels[level];
        debug_assert_eq!(lx.len(), ly.len());
        for i in 0..lx.len() {
            k += lx[i] * ly[i];
        }
    }
    k
}

/// Normalized signature kernel — cosine in feature space.
pub fn signature_kernel_normalized(x: &[Vec<f64>], y: &[Vec<f64>], depth: usize) -> f64 {
    let kxx = signature_kernel(x, x, depth);
    let kyy = signature_kernel(y, y, depth);
    let kxy = signature_kernel(x, y, depth);
    let denom = (kxx * kyy).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    kxy / denom
}

/// Goursat-PDE signature kernel — full (untruncated) form.
///
/// Currently delegates to a high-depth truncated computation as an
/// approximation; the proper hyperbolic PDE solver is the planned
/// production path (PR target).
pub fn signature_kernel_pde(x: &[Vec<f64>], y: &[Vec<f64>]) -> f64 {
    // Stand-in: truncated kernel at depth 4. The full PDE solver (Salvi
    // et al. 2020, Algorithm 1) is the production path; this surface
    // exists so consumers can already wire against it.
    signature_kernel(x, y, 4)
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn self_kernel_positive() {
        let x = vec![vec![0.0, 0.0], vec![1.0, 2.0], vec![3.0, 1.0]];
        let k = signature_kernel(&x, &x, 2);
        assert!(k > 0.0, "self-kernel should be > 0, got {k}");
    }

    #[test]
    fn kernel_symmetric() {
        let x = vec![vec![0.0, 0.0], vec![1.0, 2.0]];
        let y = vec![vec![0.0, 0.0], vec![3.0, -1.0]];
        let kxy = signature_kernel(&x, &y, 2);
        let kyx = signature_kernel(&y, &x, 2);
        assert!(approx(kxy, kyx, 1e-10));
    }

    #[test]
    fn normalized_self_is_one() {
        let x = vec![vec![0.0, 0.0], vec![1.0, 2.0], vec![3.0, 4.0]];
        let k = signature_kernel_normalized(&x, &x, 2);
        assert!(approx(k, 1.0, 1e-10), "got {k}");
    }

    #[test]
    fn cauchy_schwarz() {
        // |K(X,Y)|² ≤ K(X,X) · K(Y,Y).
        let x = vec![vec![0.0, 0.0], vec![1.0, 2.0], vec![3.0, 1.0]];
        let y = vec![vec![0.0, 0.0], vec![2.0, 1.0], vec![1.0, 4.0]];
        let kxx = signature_kernel(&x, &x, 2);
        let kyy = signature_kernel(&y, &y, 2);
        let kxy = signature_kernel(&x, &y, 2);
        assert!(
            kxy * kxy <= kxx * kyy + 1e-9,
            "Cauchy-Schwarz violated: {} > {}",
            kxy * kxy,
            kxx * kyy
        );
    }

    #[test]
    fn level_zero_contributes_one() {
        // Two completely orthogonal paths: kernel ≥ 1 (level-0 always = 1).
        let x = vec![vec![1.0, 0.0], vec![1.0, 0.0]]; // constant
        let y = vec![vec![0.0, 1.0], vec![0.0, 1.0]]; // constant orthogonal
        let k = signature_kernel(&x, &y, 2);
        assert!(approx(k, 1.0, 1e-10), "got {k}");
    }
}
