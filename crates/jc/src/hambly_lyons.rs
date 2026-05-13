//! Hambly-Lyons 2010: Signature uniqueness for paths of bounded variation —
//! the math foundation that certifies sigker's Index-regime classification.
//!
//! Citation: B. Hambly & T. Lyons, "Uniqueness for the signature of a path
//! of bounded variation and the reduced path group", Annals of Mathematics,
//! Vol. 171, No. 1 (2010), 109-167.
//!
//! # What this pillar certifies
//!
//! Hambly-Lyons 2010 Theorem 4: for paths X, Y of bounded variation taking
//! values in ℝ^d:
//!
//!   S(X) = S(Y)   ⟺   X and Y are equal modulo tree-like equivalence
//!
//! where tree-like equivalence is the smallest equivalence relation generated
//! by identifying any sub-path with its concatenated reverse (a detour-and-
//! return collapses to its start point).
//!
//! # Operational consequence in lance-graph
//!
//! Sigker (in `crates/sigker/`) declares `CodecRoute::Sigker` with **Index
//! regime** — the encoding is asserted lossless on the natural quotient
//! (tree-like equivalence). For graph traversal, a detour-and-return that
//! visits node X and returns conveys no information beyond visiting the
//! start point; the signature respects that.
//!
//! # Activation gate
//!
//! Active under `--features hambly-lyons` (default: off, JC stays zero-dep).
//! When active, the probe runs against `sigker::signature_truncated` at
//! depth 2.
//!
//! # Probe design (`hambly-lyons` feature)
//!
//! Two complementary tests against `sigker::signature_truncated` at depth 2:
//!
//! **Forward (tree-equivalence preserves signature):**
//! 1. Generate `N` random piecewise-linear segments `[p₀, p₁]` in ℝ³.
//! 2. For each, build the out-and-back path `[p₀, p₁, p₀]`.
//! 3. Verify `‖S([p₀, p₁, p₀]) − S_identity‖_F < ε`.
//!
//! Out-and-back is the canonical generator of tree-like equivalence: by
//! Chen's identity the forward signature and reverse signature concatenate
//! to identity (= signature of a constant path).
//!
//! **Converse (non-tree perturbation distinguishes signatures):**
//! 1. For each base segment, build the triangle loop `[p₀, p₁, p₂, p₀]`
//!    where p₂ is chosen so the three points are not collinear.
//! 2. Verify `‖S(triangle) − S_identity‖_F > δ`.
//!
//! A triangle has non-zero level-2 signature components (signed area along
//! each coordinate pair); these are *measurable* even at depth-2 truncation.
//! Tree-quotient class is non-trivial.
//!
//! # Pass criteria (`hambly-lyons` feature active)
//!
//! Across `N_PAIRS = 100` random pairs in d = 3:
//! - Forward: max `‖S(out-and-back) − S_identity‖` < ε (1e-9)
//! - Converse: min `‖S(triangle) − S_identity‖` > δ (0.05)
//! - Discrimination ratio (min-converse / max-forward) > 1e6
//!
//! # Why this avoids the `signature_kernel_pde` math bug
//!
//! `sigker::kernel::signature_kernel_pde` ships a Goursat-PDE form that
//! diverges from the true signature kernel `I₀(2·√⟨u, v⟩)` at moderate
//! inner products (PR #350 documents the corrected form). This probe uses
//! `signature_truncated` (the tensor-algebra path, untouched by the PDE
//! correction) for both the forward and converse legs — the Hambly-Lyons
//! certification is independent of any PR-#350 outcome.

use crate::PillarResult;

#[cfg(feature = "hambly-lyons")]
mod active {
    use super::*;

    use std::time::Instant;

    use sigker::signature::Signature;
    use sigker::signature_truncated;

    const N_PAIRS: usize = 100;
    const DEPTH: usize = 2;
    const DIM: usize = 3;

    const FORWARD_TOLERANCE: f64 = 1e-9;
    const CONVERSE_THRESHOLD: f64 = 0.05;
    const DISCRIMINATION_RATIO_MIN: f64 = 1.0e6;

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn rand_in(state: &mut u64, lo: f64, hi: f64) -> f64 {
        let u = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64;
        lo + u * (hi - lo)
    }

    fn random_point(state: &mut u64, dim: usize) -> Vec<f64> {
        (0..dim).map(|_| rand_in(state, -1.0, 1.0)).collect()
    }

    /// Frobenius distance across all signature levels.
    fn signature_distance(a: &Signature, b: &Signature) -> f64 {
        assert_eq!(a.dim, b.dim);
        assert_eq!(a.depth, b.depth);
        let mut acc = 0.0_f64;
        for (la, lb) in a.levels.iter().zip(b.levels.iter()) {
            for (xa, xb) in la.iter().zip(lb.iter()) {
                let d = xa - xb;
                acc += d * d;
            }
        }
        acc.sqrt()
    }

    /// Out-and-back: `[p₀, p₁, p₀]`. Tree-equivalent to constant path `[p₀]`.
    fn out_and_back(p0: &[f64], p1: &[f64]) -> Vec<Vec<f64>> {
        vec![p0.to_vec(), p1.to_vec(), p0.to_vec()]
    }

    /// Triangle loop: `[p₀, p₁, p₂, p₀]`. Encloses non-zero signed area in
    /// any coordinate plane where `p₀, p₁, p₂` are not collinear.
    fn triangle_loop(p0: &[f64], p1: &[f64], p2: &[f64]) -> Vec<Vec<f64>> {
        vec![p0.to_vec(), p1.to_vec(), p2.to_vec(), p0.to_vec()]
    }

    pub fn prove() -> PillarResult {
        let t0 = Instant::now();

        let identity = Signature::identity(DIM, DEPTH);
        let mut state: u64 = 0xCAFE_BABE_DEAD_BEEFu64;

        let mut max_forward_dist = 0.0_f64;
        let mut min_converse_dist = f64::INFINITY;
        let mut forward_pairs_pass = 0u64;
        let mut converse_pairs_pass = 0u64;

        for _ in 0..N_PAIRS {
            let p0 = random_point(&mut state, DIM);
            let p1 = random_point(&mut state, DIM);
            let p2 = random_point(&mut state, DIM);

            // Forward leg: out-and-back ≈ identity
            let oab = out_and_back(&p0, &p1);
            let s_oab = signature_truncated(&oab, DEPTH);
            let d_forward = signature_distance(&s_oab, &identity);
            if d_forward > max_forward_dist {
                max_forward_dist = d_forward;
            }
            if d_forward < FORWARD_TOLERANCE {
                forward_pairs_pass += 1;
            }

            // Converse leg: triangle ≠ identity
            let tri = triangle_loop(&p0, &p1, &p2);
            let s_tri = signature_truncated(&tri, DEPTH);
            let d_converse = signature_distance(&s_tri, &identity);
            if d_converse < min_converse_dist {
                min_converse_dist = d_converse;
            }
            if d_converse > CONVERSE_THRESHOLD {
                converse_pairs_pass += 1;
            }
        }

        let runtime_ms = t0.elapsed().as_millis() as u64;

        let discrimination_ratio = if max_forward_dist > 0.0 {
            min_converse_dist / max_forward_dist
        } else {
            f64::INFINITY
        };

        let pass = forward_pairs_pass == N_PAIRS as u64
            && converse_pairs_pass == N_PAIRS as u64
            && discrimination_ratio >= DISCRIMINATION_RATIO_MIN;

        let detail = format!(
            "N={} pairs, dim={}, depth={}. \
             Forward (tree-equivalence): max ‖S(out-and-back) − S_identity‖ = {:.3e} \
             (pass if < {:.0e}); {}/{} pairs within tolerance. \
             Converse (non-tree): min ‖S(triangle) − S_identity‖ = {:.4} \
             (pass if > {:.2}); {}/{} pairs above threshold. \
             Discrimination ratio (min-converse / max-forward) = {:.3e} \
             (pass if ≥ {:.0e}). \
             Pillar uses sigker::signature_truncated (tensor-algebra path), \
             not signature_kernel_pde — independent of PR #350 PDE-form correction.",
            N_PAIRS, DIM, DEPTH,
            max_forward_dist, FORWARD_TOLERANCE,
            forward_pairs_pass, N_PAIRS,
            min_converse_dist, CONVERSE_THRESHOLD,
            converse_pairs_pass, N_PAIRS,
            discrimination_ratio, DISCRIMINATION_RATIO_MIN,
        );

        PillarResult {
            name: "Hambly-Lyons: signature uniqueness on tree-quotient",
            pass,
            measured: discrimination_ratio,
            predicted: DISCRIMINATION_RATIO_MIN,
            detail,
            runtime_ms,
        }
    }
}

#[cfg(feature = "hambly-lyons")]
pub fn prove() -> PillarResult {
    active::prove()
}

#[cfg(not(feature = "hambly-lyons"))]
pub fn prove() -> PillarResult {
    PillarResult::deferred(
        "Hambly-Lyons: signature uniqueness on tree-quotient",
        "build with --features hambly-lyons to activate the probe \
         (pulls in the sigker workspace sibling). Default JC build stays \
         zero-dep per the standalone-crate constitution.",
    )
}

#[cfg(all(test, feature = "hambly-lyons"))]
mod tests {
    use super::*;

    #[test]
    fn pillar_passes() {
        let r = prove();
        assert!(r.pass, "Hambly-Lyons probe must pass: {}", r.detail);
    }
}
