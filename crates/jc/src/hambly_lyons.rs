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
//! # Depth-infinity leg (W2)
//!
//! Hambly-Lyons is a depth-∞ theorem; the two legs above certify it at
//! DEPTH-2 TRUNCATION. The depth-∞ leg reaches the full statement through the
//! Goursat PDE kernel, which needs no signature materialization — but it is a
//! DISCRETIZED object, and that changes what an honest gate can assert.
//!
//! Statistic: `dev(x) = |K(x,y)/√(K(x,x)K(y,y)) − 1|` against the constant
//! base `y`. A constant path has zero increments, so `K(x,y) = K(y,y) = 1`
//! and this reduces to `|1/√(K(x,x)) − 1|` — zero iff x's depth-∞ signature
//! is the identity.
//!
//! **Two things had to be measured before any threshold was written.**
//!
//! 1. *Refinement* (`examples/w2_refinement_sweep.rs`). A raw 3-point
//!    out-and-back is NOT tree-invariant under the first-order scheme:
//!    `K(x,x) = 1 + a²` for increment norm² `a`. Resampling kills it at
//!    SECOND order (measured ratio 4.00 per doubling), while the triangle's
//!    deviation converges to a constant — the two legs separate exactly as
//!    the theorem requires. At 1536 points/segment: forward 1.11e-5 over 100
//!    pairs, converse floor 1.340e-2, stable to four digits from 64 upward.
//!
//! 2. *The converse's real edge* (`examples/w2_area_edge.rs`). Taking a
//!    random minimum was the wrong instrument — it moved from 1.53e-2 at 25
//!    pairs to 1.03e-1 at 12, because fewer draws find fewer near-degenerate
//!    triangles: a gate that gets EASIER with a smaller sample. Measured on a
//!    controlled family (apex offset `h` off the base midpoint, enclosed area
//!    `h/2`), the deviation obeys
//!
//!    ```text
//!      dev  ≈  2 · area²        (measured dev/area²: 1.9923, 1.9985,
//!                                2.0012, 2.0051 as area shrinks)
//!    ```
//!
//!    with an artifact floor of 2.119e-7 at exactly `h = 0`.
//!
//! So the leg certifies the FUNCTIONAL FORM and its boundary, not a sampled
//! minimum:
//!
//! - forward: max deviation over out-and-backs < 5e-5
//! - converse law: `|dev/area² − 2| < 0.05` over the shrinking family
//! - edge: area 2.5e-3 IS distinguished (dev > 1e-5, ~60x the floor)
//! - below edge: area 2.5e-4 is NOT (dev < 1e-6, at the floor) — the
//!   can-stay-silent half. A converse gate with no boundary would be
//!   satisfied by any input; this one has a measured edge and says where.
//!
//! # Why this uses `signature_truncated`, not the PDE kernel — history
//!
//! An earlier `signature_kernel_pde` diverged from the closed form
//! `I₀(2·√⟨u, v⟩)` at moderate inner products; PR #350 fixed it. The
//! CURRENT solver is measured sound (2026-08-31, D-SK arc: rel err
//! 6.25e-5 at d=3 / 4.53e-4 at d=24 on N=256 linear-path anchors), so the
//! old warning that stood here — steering readers away from the PDE form
//! outright — is retired as pre-#350 history, not deleted. This probe
//! still uses `signature_truncated` for both legs: depth-2 truncation is
//! the DELIBERATE scope of these gates, and the tensor-algebra path keeps
//! the certification independent of solver discretization entirely. The
//! depth-∞ PDE leg is its own wave (W2 of
//! `pillar11-signature-certification-unification-v1`, with
//! refinement-swept tolerances — a raw 3-point loop is NOT tree-invariant
//! under the first-order scheme).

use crate::PillarResult;

#[cfg(feature = "hambly-lyons")]
mod active {
    use super::*;

    use std::time::Instant;

    use sigker::signature::Signature;
    use sigker::signature_kernel_pde;
    use sigker::signature_truncated;

    const N_PAIRS: usize = 100;
    const DEPTH: usize = 2;
    const DIM: usize = 3;

    const FORWARD_TOLERANCE: f64 = 1e-9;
    const CONVERSE_THRESHOLD: f64 = 0.05;
    const DISCRIMINATION_RATIO_MIN: f64 = 1.0e6;

    // ── W2: the depth-infinity leg ──────────────────────────────────────────
    // Resolution and every threshold are pre-registered from the two committed
    // sweeps: `examples/w2_refinement_sweep.rs` (convergence) and
    // `examples/w2_area_edge.rs` (the converse law and its edge).
    const PER_SEG: usize = 1536;
    /// Points in the longest path this leg constructs: the converse triangle
    /// is three resampled segments plus the closing point. Exported so the W5
    /// trigger check DERIVES the in-tree maximum instead of restating it —
    /// a measured value retyped into a second file is one that goes stale
    /// silently, and the trigger's whole job is to notice when it has not.
    pub const LONGEST_PATH_POINTS: usize = 3 * PER_SEG + 1;
    /// Forward pairs. The forward statistic is population-stable — the
    /// refinement sweep measured max deviation 5.005e-6 at both 12 and 25
    /// pairs — so a small sample is a faithful one here.
    const N_PAIRS_PDE: usize = 8;
    /// eps(N) at PER_SEG. The forward artifact is O(h^2); the sweep measured
    /// 1.108764e-5 over 100 pairs at this resolution, and the exactly
    /// tree-like fixture bottoms out at 2.119e-7. Pinned with ~4x margin.
    const PDE_FORWARD_EPS: f64 = 5.0e-5;
    /// The converse law: deviation / area^2 for the controlled triangle
    /// family. Measured 1.9923, 1.9985, 2.0012, 2.0051 as area shrinks —
    /// the deviation is QUADRATIC IN THE ENCLOSED LEVY AREA, and that
    /// functional form is what this leg certifies, not a random minimum.
    const PDE_AREA_LAW: f64 = 2.0;
    const PDE_AREA_LAW_TOL: f64 = 0.05;
    /// The near-degenerate case that must still be distinguished: area
    /// 2.5e-3, measured deviation 1.262e-5 — 60x the 2.1e-7 artifact floor.
    const PDE_EDGE_H: f64 = 0.005;
    const PDE_EDGE_MIN_DEV: f64 = 1.0e-5;
    /// Far below the edge the deviation IS the floor and the loop is NOT
    /// distinguishable. Stating that boundary is what keeps the gate above
    /// from being vacuous — see `depth_infinity_edge_is_real`.
    const PDE_BELOW_EDGE_H: f64 = 0.0005;
    const PDE_FLOOR: f64 = 1.0e-6;

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

    /// Resample a polyline so each segment carries `per_seg` sub-intervals.
    ///
    /// Mandatory for the depth-infinity leg: a raw 3-point out-and-back is NOT
    /// tree-invariant under the first-order Goursat scheme (for increment `u`
    /// with `a = ‖u‖²` the recurrence gives `K(x,x) = 1 + a²` against 1 for
    /// the constant base). That is discretization, not a uniqueness failure,
    /// and it vanishes at second order under refinement.
    fn resample(corners: &[Vec<f64>], per_seg: usize) -> Vec<Vec<f64>> {
        let dim = corners[0].len();
        let mut out = vec![corners[0].clone()];
        for w in corners.windows(2) {
            for s in 1..=per_seg {
                let t = s as f64 / per_seg as f64;
                out.push(
                    (0..dim)
                        .map(|a| w[0][a] + t * (w[1][a] - w[0][a]))
                        .collect(),
                );
            }
        }
        out
    }

    /// Deviation of the normalized depth-infinity kernel from the constant
    /// base: `|K(x,y)/√(K(x,x)K(y,y)) − 1|` with `y` constant. A constant path
    /// has zero increments, so every cell coefficient vanishes and
    /// `K(x,y) = K(y,y) = 1`, leaving `|1/√(K(x,x)) − 1|`. Zero iff the path's
    /// depth-infinity signature is the identity.
    fn pde_deviation_from_constant(path: &[Vec<f64>]) -> f64 {
        let kxx = signature_kernel_pde(path, path);
        (1.0 / kxx.sqrt() - 1.0).abs()
    }

    /// A triangle `[p0, p1, p2(h), p0]` whose apex sits `h` off the base
    /// midpoint along a unit normal, so the enclosed area is exactly
    /// `‖p1−p0‖·h/2` and `h` tunes degeneracy directly. `h = 0` is tree-like.
    fn controlled_triangle(h: f64) -> (Vec<Vec<f64>>, f64) {
        let p0 = vec![0.0, 0.0, 0.0];
        let p1 = vec![1.0, 0.0, 0.0];
        let p2 = vec![0.5, h, 0.0];
        let area = h / 2.0; // base length is 1
        (resample(&[p0.clone(), p1, p2, p0], PER_SEG), area)
    }

    /// The depth-infinity leg's three measurements.
    struct PdeLeg {
        /// max |deviation| over tree-like (out-and-back) paths
        forward_max: f64,
        /// worst |deviation/area^2 − 2| over the controlled family
        area_law_err: f64,
        /// deviation at the near-degenerate edge that must be distinguished
        edge_dev: f64,
        /// deviation far below the edge — must sit at the artifact floor
        below_edge_dev: f64,
    }

    fn depth_infinity_leg() -> PdeLeg {
        let mut state: u64 = 0xCAFE_BABE_DEAD_BEEF;
        let mut forward_max = 0.0f64;
        for _ in 0..N_PAIRS_PDE {
            let p0 = random_point(&mut state, DIM);
            let p1 = random_point(&mut state, DIM);
            let oab = resample(&[p0.clone(), p1, p0], PER_SEG);
            forward_max = forward_max.max(pde_deviation_from_constant(&oab));
        }

        // The converse LAW, over the shrinking-area family.
        let mut area_law_err = 0.0f64;
        for &h in &[0.1f64, 0.05, 0.02, 0.01] {
            let (tri, area) = controlled_triangle(h);
            let q = pde_deviation_from_constant(&tri) / (area * area);
            area_law_err = area_law_err.max((q - PDE_AREA_LAW).abs());
        }

        let (edge_tri, _) = controlled_triangle(PDE_EDGE_H);
        let (below_tri, _) = controlled_triangle(PDE_BELOW_EDGE_H);

        PdeLeg {
            forward_max,
            area_law_err,
            edge_dev: pde_deviation_from_constant(&edge_tri),
            below_edge_dev: pde_deviation_from_constant(&below_tri),
        }
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

        // ── W2: depth-infinity leg (M-2) ────────────────────────────────
        let pde = depth_infinity_leg();
        let pde_pass = pde.forward_max < PDE_FORWARD_EPS
            && pde.area_law_err < PDE_AREA_LAW_TOL
            && pde.edge_dev > PDE_EDGE_MIN_DEV
            && pde.below_edge_dev < PDE_FLOOR;

        let runtime_ms = t0.elapsed().as_millis() as u64;

        let discrimination_ratio = if max_forward_dist > 0.0 {
            min_converse_dist / max_forward_dist
        } else {
            f64::INFINITY
        };

        let pass = forward_pairs_pass == N_PAIRS as u64
            && converse_pairs_pass == N_PAIRS as u64
            && discrimination_ratio >= DISCRIMINATION_RATIO_MIN
            && pde_pass;

        let detail = format!(
            "N={} pairs, dim={}, depth={}. \
             Forward (tree-equivalence): max ‖S(out-and-back) − S_identity‖ = {:.3e} \
             (pass if < {:.0e}); {}/{} pairs within tolerance. \
             Converse (non-tree): min ‖S(triangle) − S_identity‖ = {:.4} \
             (pass if > {:.2}); {}/{} pairs above threshold. \
             Discrimination ratio (min-converse / max-forward) = {:.3e} \
             (pass if ≥ {:.0e}). \
             DEPTH-INFINITY leg (W2, resampled to {} pts/segment, {} forward pairs): \
             max forward deviation = {:.3e} (pass if < {:.0e}); \
             converse law |dev/area² − 2| = {:.4} (pass if < {:.2}); \
             edge (area {:.4}) deviation = {:.3e} (pass if > {:.0e}); \
             below-edge (area {:.5}) deviation = {:.3e} (pass if < {:.0e}, \
             i.e. NOT distinguishable — the gate's real boundary). \
             The depth-2 legs use sigker::signature_truncated (tensor algebra, \
             exact, so their forward distance is exactly 0). The depth-infinity \
             leg uses signature_kernel_pde, where the forward deviation is an \
             O(h²) discretization artifact rather than a uniqueness failure — \
             so that leg certifies the converse FUNCTIONAL FORM (deviation is \
             quadratic in enclosed Lévy area) and its own resolution boundary, \
             not a sampled ratio.",
            N_PAIRS,
            DIM,
            DEPTH,
            max_forward_dist,
            FORWARD_TOLERANCE,
            forward_pairs_pass,
            N_PAIRS,
            min_converse_dist,
            CONVERSE_THRESHOLD,
            converse_pairs_pass,
            N_PAIRS,
            discrimination_ratio,
            DISCRIMINATION_RATIO_MIN,
            PER_SEG,
            N_PAIRS_PDE,
            pde.forward_max,
            PDE_FORWARD_EPS,
            pde.area_law_err,
            PDE_AREA_LAW_TOL,
            PDE_EDGE_H / 2.0,
            pde.edge_dev,
            PDE_EDGE_MIN_DEV,
            PDE_BELOW_EDGE_H / 2.0,
            pde.below_edge_dev,
            PDE_FLOOR,
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

/// Points in the longest path this module constructs — see the constant's
/// own doc in the gated implementation.
#[cfg(feature = "hambly-lyons")]
pub use active::LONGEST_PATH_POINTS;

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
