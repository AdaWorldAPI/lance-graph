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
//! # Status — GREEN for lattice walks, length-parameterized (2026-09-01, W6)
//!
//! The pillar was red because the uniqueness theorem is a depth-∞ statement
//! and the substrate truncates. The finite-depth certificate is in the SAME
//! paper, §2.4 — cited from the PUBLISHED text, Annals of Mathematics 171(1)
//! 2010, pp. 109–167 (the constant below was verified against that PDF):
//!
//!   **Theorem 5** (Annals numbering; Theorem 1 in the introduction restates
//!   it). A path of length L on the 2-d integer lattice whose first
//!   ⌊2e·log(1+√2)·L⌋ GL(2,C)-iterated integrals vanish is tree-like and
//!   its reduced word is trivial. (2e·ln(1+√2) = 4.7916…)
//!
//!   **Theorem 6.** In the d-dimensional lattice the depth is
//!   ⌊(2⌈log₃(d/2)⌉ + 3)·2e·log(1+√2)·L⌋.
//!
//! ⚠ Version trap, recorded so it is not re-walked: arXiv math/0507536v2
//! (Dec 2006) states the same results as Theorems 2/3 with coefficient `e`,
//! and its proof applies Lemma 2.4(2) with `x = log(1+√2)·L` while the sum
//! it bounds runs over ODD degrees `2k−1` — the index `k` counts pairs of
//! degrees, so "first N terms" there means degree up to ~2N. The published
//! version takes `x = 2·log(1+√2)·L` and states `2e`; that is the corrected
//! form and the one this pillar implements. A review bot caught the
//! discrepancy against the arXiv-only reading (lance-graph #1133).
//!
//! The GL(2,C) integrals are a projection of the tensor-algebra ones
//! (fn. 2: "a priori contain less information"), so vanishing of the FULL
//! truncated signature to that depth implies the hypothesis a fortiori.
//! Because the truncated signature is a homomorphism into the free
//! nilpotent group, the pair form is
//!
//!   S^(N)(X) = S^(N)(Y)  ⟺  X ∼ Y     for N ≥ ⌊c(d)·(|X|+|Y|)⌋, c(2) = 2e·ln(1+√2),
//!
//! i.e. the Index regime is LENGTH-PARAMETERIZED: a consumer must carry a
//! walk-length budget and escalate depth (or refuse) beyond it. Depth 2 is
//! a NECESSARY condition only — the paper's own §1.6 figure-of-8 has
//! S¹ = S² = 0 and is not tree-like; the W6 leg below finds 64 such reduced
//! words at length 8 and separates every one by depth 3.
//!
//! Preconditions, both pinned executably in the W6 leg:
//!   * `d ≥ 2` — in d = 1 the reduced-path group is Z (Diehl-Ebrahimi-Fard-
//!     Tapia Rem. 1.4: every closed 1-d path is tree-like), so a single
//!     `u8:u8` rail read as ONE scalar axis carries only its net increment;
//!   * unit basis-aligned steps on the integer lattice (p.8:
//!     ‖x_k − x_{k+1}‖ = 1, x_k ∈ Z^{|A|}). Arbitrary quantized step vectors
//!     are OUTSIDE Theorem 2; for those the applicable statement is
//!     Theorem 9 (piecewise-linear, bound in the smallest angle and the
//!     shortest edge) which gives non-triviality but no explicit depth.
//!
//! What stays honest: the depth-2 legs certify a necessary condition and
//! the Lévy-area functional form; the depth-∞ PDE leg reads, to leading
//! order, the same Lévy area (dev ≈ ½‖S²‖²) — so it is a second resolution
//! of the same measurement, not independent evidence. Unnormalized
//! truncated features are not characteristic (Chevyrev-Oberhauser 2022
//! Thm 21 / Rem. 4); the discrimination RATIO below is scale-dependent and
//! is kept as a regression guard, not a discrimination bound. Ledger:
//! `.claude/knowledge/literature-harvest-2026-09-01-post-1132.md` (D1–D8).
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

    // ── W6: the Theorem 5 lattice leg (the finite-depth certificate) ────────
    //
    // Hambly-Lyons, Annals 171 (2010) §2.4, Theorem 5 (= arXiv v2 Theorem 2
    // with the published `2e` coefficient; verified against the Annals PDF
    // 2026-09-01): a lattice path of length L on Z² whose first
    // ⌊2e·log(1+√2)·L⌋ GL(2,C)-iterated integrals vanish is tree-like and its
    // reduced word is trivial. The GL(2,C) integrals are a projection of the
    // tensor-algebra ones ("a priori contain less information", fn. 2), so
    // vanishing of the FULL truncated signature to that depth implies it a
    // fortiori. The truncated signature is a homomorphism into the free
    // nilpotent group, so for two words w, v the pair statement is
    // S^(N)(w) = S^(N)(v) ⟺ S^(N)(w·v⁻¹) = 1 with N = ⌊c·(|w|+|v|)⌋ — this
    // leg therefore tests single words z = w·v⁻¹ of the combined length.
    //
    // Hypotheses that make the certificate apply, both pinned below:
    //   * unit basis-aligned steps on the integer lattice (Def. of lattice
    //     path, p.8: ‖x_k − x_{k+1}‖ = 1, x_k ∈ Z^{|A|});
    //   * d ≥ 2 — in d = 1 the reduced-path group is Z (net increment only,
    //     every closed path is tree-like), so the quotient carries nothing.
    /// `2e · ln(1 + √2)` — Theorem 5's constant (Annals), computed rather
    /// than retyped.
    pub(super) fn hl_theorem2_constant() -> f64 {
        2.0 * std::f64::consts::E * (1.0 + 2f64.sqrt()).ln()
    }
    /// The depth Theorem 5 needs for a word of length `l` (floor, as stated).
    pub(super) fn hl_theorem2_depth(l: usize) -> usize {
        (hl_theorem2_constant() * l as f64).floor() as usize
    }
    /// Longest word the exhaustive theorem arm enumerates. Depth ⌊c·3⌋ = 14
    /// in d = 2 is 32768 coefficients — cheap enough for a debug test.
    const LATTICE_L_MAX: usize = 3;
    /// Length at which the depth-2 false-merge search runs. The paper's own
    /// §1.6 figure-of-8 (two equal, opposite lobes) lives here: ⌊c·8⌋ = 38
    /// (never materialized — the search escalates depth and stops at 3).
    const LATTICE_FALSE_MERGE_L: usize = 8;
    const LATTICE_N_TREELIKE: usize = 64;
    /// Depth for the tree-like arm (identity holds at every depth; fixed, cheap).
    const LATTICE_TREELIKE_DEPTH: usize = 12;
    /// Exactness tolerance: signatures of lattice words are rationals with
    /// k! denominators; f64 products of a handful of exponentials round at
    /// ~1e-15, and a genuinely nonzero coefficient is ≥ 1/k! ≥ 1/19!.
    pub(super) const LATTICE_EPS: f64 = 1e-12;

    /// Letters a, b, a⁻¹, b⁻¹ as 0..4; inverse is `(l + 2) % 4`. Unit step of a letter.
    fn letter_step(l: u8) -> [f64; 2] {
        match l {
            0 => [1.0, 0.0],
            1 => [0.0, 1.0],
            2 => [-1.0, 0.0],
            _ => [0.0, -1.0],
        }
    }
    /// The lattice path of a word: unit steps from the origin, one per letter.
    fn lattice_path(word: &[u8]) -> Vec<Vec<f64>> {
        let mut p = vec![vec![0.0, 0.0]];
        for &l in word {
            let s = letter_step(l);
            let last = p.last().unwrap();
            p.push(vec![last[0] + s[0], last[1] + s[1]]);
        }
        p
    }
    /// Freely reduced: no adjacent `x x⁻¹`.
    pub(super) fn is_reduced(word: &[u8]) -> bool {
        word.windows(2).all(|w| (w[0] + 2) % 4 != w[1])
    }
    /// `‖S^(depth)(word) − 1‖_F` via `sigker::signature_truncated`.
    pub(super) fn distance_from_identity(word: &[u8], depth: usize) -> f64 {
        let s = signature_truncated(&lattice_path(word), depth);
        signature_distance(&s, &Signature::identity(2, depth))
    }
    /// Every word of exactly `len` letters over the 4-letter alphabet.
    fn for_each_word(len: usize, mut f: impl FnMut(&[u8])) {
        let total = 4usize.pow(len as u32);
        let mut w = vec![0u8; len];
        for code in 0..total {
            let mut c = code;
            for slot in w.iter_mut() {
                *slot = (c % 4) as u8;
                c /= 4;
            }
            f(&w);
        }
    }
    /// A tree-like word: grow from empty by inserting `c c⁻¹` at random
    /// positions — the generator of tree-like equivalence (Def. 2.1).
    fn treelike_word(state: &mut u64, len: usize) -> Vec<u8> {
        let mut w: Vec<u8> = Vec::with_capacity(len);
        while w.len() + 2 <= len {
            let c = (splitmix64(state) % 4) as u8;
            let pos = (splitmix64(state) as usize) % (w.len() + 1);
            w.insert(pos, c);
            w.insert(pos + 1, (c + 2) % 4);
        }
        w
    }

    struct LatticeLeg {
        /// reduced non-empty words of length ≤ L_MAX checked at ⌊c·L⌋
        reduced_checked: usize,
        /// how many of them the theorem depth FAILED to separate (must be 0)
        reduced_merged: usize,
        /// min ‖S − 1‖ over those words at the theorem depth
        reduced_min_dist: f64,
        /// tree-like words checked; max ‖S − 1‖ (must sit at f64 rounding)
        treelike_checked: usize,
        treelike_max_dist: f64,
        /// reduced words of length FALSE_MERGE_L that depth 2 cannot separate
        depth2_false_merges: usize,
        /// the largest depth any of those needed to separate (≤ ⌊c·L⌋)
        false_merge_max_sep_depth: usize,
        /// how many of them stayed merged even at the theorem depth (must be 0)
        false_merge_unresolved: usize,
        /// d = 1 fence: distinct signature classes among all 2^6 words on
        /// {a, a⁻¹} of length 6 — must be exactly 7 (net increment −6..6)
        d1_classes: usize,
    }

    fn lattice_leg() -> LatticeLeg {
        // Arm 1 — the theorem: reduced ⟹ separated at ⌊c·L⌋.
        let mut reduced_checked = 0usize;
        let mut reduced_merged = 0usize;
        let mut reduced_min_dist = f64::INFINITY;
        for len in 1..=LATTICE_L_MAX {
            let depth = hl_theorem2_depth(len);
            for_each_word(len, |w| {
                if !is_reduced(w) {
                    return;
                }
                let d = distance_from_identity(w, depth);
                reduced_checked += 1;
                if d < LATTICE_EPS {
                    reduced_merged += 1;
                }
                reduced_min_dist = reduced_min_dist.min(d);
            });
        }

        // Arm 2 — tree-like words collapse to the identity at the same depth.
        let mut state: u64 = 0x5EED_1A77_1CE0_0001;
        let mut treelike_max_dist = 0.0f64;
        let mut treelike_checked = 0usize;
        // Tree-like words are the identity at EVERY depth (Cor. 6.4), so this
        // arm needs no theorem depth — a fixed one keeps the length-6 words
        // off the 2^29-coefficient tensors the doubled constant would demand.
        for i in 0..LATTICE_N_TREELIKE {
            let len = 2 + 2 * (i % 3); // 2, 4, 6
            let w = treelike_word(&mut state, len);
            treelike_max_dist =
                treelike_max_dist.max(distance_from_identity(&w, LATTICE_TREELIKE_DEPTH));
            treelike_checked += 1;
        }

        // Arm 3 — depth 2 is NOT the Index regime: find reduced words of
        // length FALSE_MERGE_L with S^(2) = 1, then show each separates by
        // the theorem depth (and record how deep it had to go).
        let theorem_depth = hl_theorem2_depth(LATTICE_FALSE_MERGE_L);
        let mut depth2_false_merges = 0usize;
        let mut false_merge_max_sep_depth = 0usize;
        let mut false_merge_unresolved = 0usize;
        for_each_word(LATTICE_FALSE_MERGE_L, |w| {
            if !is_reduced(w) || distance_from_identity(w, 2) >= LATTICE_EPS {
                return;
            }
            depth2_false_merges += 1;
            let mut separated_at = None;
            for depth in 3..=theorem_depth {
                if distance_from_identity(w, depth) >= LATTICE_EPS {
                    separated_at = Some(depth);
                    break;
                }
            }
            match separated_at {
                Some(d) => false_merge_max_sep_depth = false_merge_max_sep_depth.max(d),
                None => false_merge_unresolved += 1,
            }
        });

        // Arm 4 — the d = 1 fence: 2^6 words on {a, a⁻¹}, signatures keyed
        // by their level-1..3 coefficients rounded to 1e-9.
        let mut classes: Vec<[i64; 3]> = Vec::new();
        for code in 0..64u32 {
            let mut p = vec![vec![0.0f64]];
            for bit in 0..6 {
                let step = if (code >> bit) & 1 == 0 { 1.0 } else { -1.0 };
                let last = p.last().unwrap()[0];
                p.push(vec![last + step]);
            }
            let s = signature_truncated(&p, 3);
            let key = [
                (s.levels[1][0] * 1e9).round() as i64,
                (s.levels[2][0] * 1e9).round() as i64,
                (s.levels[3][0] * 1e9).round() as i64,
            ];
            if !classes.contains(&key) {
                classes.push(key);
            }
        }

        LatticeLeg {
            reduced_checked,
            reduced_merged,
            reduced_min_dist,
            treelike_checked,
            treelike_max_dist,
            depth2_false_merges,
            false_merge_max_sep_depth,
            false_merge_unresolved,
            d1_classes: classes.len(),
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

        // ── W6: Theorem 2 lattice leg ───────────────────────────────────
        let lat = lattice_leg();
        let lattice_pass = lat.reduced_merged == 0
            && lat.treelike_max_dist < LATTICE_EPS
            && lat.depth2_false_merges >= 1
            && lat.false_merge_unresolved == 0
            && lat.false_merge_max_sep_depth <= hl_theorem2_depth(LATTICE_FALSE_MERGE_L)
            && lat.d1_classes == 7;

        let runtime_ms = t0.elapsed().as_millis() as u64;

        let discrimination_ratio = if max_forward_dist > 0.0 {
            min_converse_dist / max_forward_dist
        } else {
            f64::INFINITY
        };

        let pass = forward_pairs_pass == N_PAIRS as u64
            && converse_pairs_pass == N_PAIRS as u64
            && discrimination_ratio >= DISCRIMINATION_RATIO_MIN
            && pde_pass
            && lattice_pass;

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
             not a sampled ratio. \
             THEOREM 5 LATTICE leg (W6, Annals numbering, c = 2e·ln(1+√2) = {:.4}): {} reduced \
             words of length ≤ {} at depth ⌊c·L⌋, {} merged with the identity \
             (pass if 0), min ‖S − 1‖ = {:.3e}; {} tree-like words, max \
             ‖S − 1‖ = {:.1e} (pass if < {:.0e}); depth-2 false merges among \
             reduced words of length {} = {} (pass if ≥ 1 — depth 2 is NOT \
             the Index regime), all separated by depth {} ≤ ⌊c·{}⌋ = {}, {} \
             unresolved (pass if 0); d = 1 signature classes over the 64 \
             length-6 words = {} (pass if exactly 7: net increment only — \
             the d ≥ 2 precondition).",
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
            hl_theorem2_constant(),
            lat.reduced_checked,
            LATTICE_L_MAX,
            lat.reduced_merged,
            lat.reduced_min_dist,
            lat.treelike_checked,
            lat.treelike_max_dist,
            LATTICE_EPS,
            LATTICE_FALSE_MERGE_L,
            lat.depth2_false_merges,
            lat.false_merge_max_sep_depth,
            LATTICE_FALSE_MERGE_L,
            hl_theorem2_depth(LATTICE_FALSE_MERGE_L),
            lat.false_merge_unresolved,
            lat.d1_classes,
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

    // ── W6 lattice helpers, tested directly (not only through prove()) ──────

    #[test]
    fn theorem2_depth_is_the_paper_floor() {
        // ⌊2e·ln(1+√2)·L⌋, Annals 171 Theorem 5, L = 1..16
        let expect = [
            4usize, 9, 14, 19, 23, 28, 33, 38, 43, 47, 52, 57, 62, 67, 71, 76,
        ];
        for (i, &e) in expect.iter().enumerate() {
            assert_eq!(active::hl_theorem2_depth(i + 1), e, "L={}", i + 1);
        }
        assert!((active::hl_theorem2_constant() - 4.7916).abs() < 1e-3);
    }

    #[test]
    fn is_reduced_rejects_exactly_adjacent_inverse_pairs() {
        assert!(active::is_reduced(&[0, 1, 2, 3]));
        assert!(!active::is_reduced(&[0, 2]));
        assert!(!active::is_reduced(&[1, 0, 2, 3]));
        assert!(active::is_reduced(&[0, 0, 0]));
        assert!(active::is_reduced(&[]));
    }

    #[test]
    fn the_figure_of_eight_is_a_depth_2_false_merge_separated_at_depth_3() {
        // a b a⁻¹ b⁻¹ · b⁻¹ a⁻¹ b a — the paper's §1.6 counterexample
        let w = [0u8, 1, 2, 3, 3, 2, 1, 0];
        assert!(active::is_reduced(&w));
        assert!(active::distance_from_identity(&w, 2) < active::LATTICE_EPS);
        assert!(active::distance_from_identity(&w, 3) > 0.1);
        // and the genuine out-and-back is the identity at every depth
        for depth in 1..=6 {
            assert!(active::distance_from_identity(&[0, 2], depth) < active::LATTICE_EPS);
        }
    }

    #[test]
    fn pillar_passes() {
        let r = prove();
        assert!(r.pass, "Hambly-Lyons probe must pass: {}", r.detail);
    }
}
