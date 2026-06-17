//! Spectral perturbation of the Laplacian under a single line outage.
//!
//! A line trip on edge `k = (a,b)` with weight `b_k` is a rank-1 perturbation
//! `E = L' − L = −b_k (e_a − e_b)(e_a − e_b)ᵀ`, so `‖E‖₂ = b_k·‖e_a−e_b‖² =
//! 2·b_k`. We certify:
//!
//! - **Weyl's inequality** `|λᵢ(L') − λᵢ(L)| ≤ ‖E‖₂` for every `i`.
//! - **Davis–Kahan** Fiedler-vector rotation `sinθ ≤ ‖E‖₂ / gap`, where `gap`
//!   is the spectral separation of the Fiedler eigenvalue `λ₂`.
//! - The **algebraic connectivity** (`λ₂`, Fiedler value) before/after. A trip
//!   that pushes `λ₂` toward 0 is fragmenting the network — the precursor shape
//!   of a blackout.

use crate::eigen::symmetric_eigen;
use crate::graph::Grid;

/// Relative floor for the **Fiedler spectral gap** in the Davis–Kahan bound,
/// applied as a fraction of the spectral scale `λ_max`.
///
/// The Davis–Kahan rotation bound is `sinθ ≤ ‖E‖₂ / gap`, where
/// `gap = min(λ₂−λ₁, λ₃−λ₂)` is the separation of the Fiedler eigenvalue from
/// its neighbours. This is the model's intrinsic NaN/divergence landmine: at the
/// blackout precursor (`λ₂ → 0`, a fragmenting network) the spectrum becomes
/// degenerate, `gap → 0`, and the bound diverges — *in the exact regime the model
/// exists to capture*.
///
/// Two failure modes the naked `gap > ε` test let through:
/// 1. a tiny-but-positive `gap` (a near-degenerate cluster) passes the test and
///    produces an astronomically large *finite* bound presented as if it were a
///    real, trustworthy rotation bound;
/// 2. cyclic-Jacobi numerical noise on a degenerate cluster can make a subtracted
///    `gap` slightly negative, which (if it slipped past the test) would yield a
///    *negative* rotation bound — mathematically meaningless.
///
/// The fix mirrors `ndarray::hpc::entropy_ladder::residue_surprise`: floor the
/// denominator **after** the subtract-and-`min` that builds it, then divide
/// unconditionally. A `gap` at or below `SPECTRAL_GAP_FLOOR · λ_max` is treated as
/// *spectral degeneracy* — the documented "the Fiedler mode is no longer separated
/// from its neighbours" condition — and the bound is reported as the
/// [`FRAGMENTATION_SENTINEL`] divergence signal rather than a noisy finite number.
///
/// Scaled by `λ_max` so the floor tracks the spectrum's magnitude (the same
/// relative-tolerance convention `eigen.rs` uses for its zero-eigenvalue cutoff).
pub const SPECTRAL_GAP_FLOOR: f64 = 1e-12;

/// The Davis–Kahan bound when the Fiedler gap collapses (spectral degeneracy /
/// fragmentation): the bound has *diverged*, so there is no finite rotation
/// guarantee — the Fiedler vector may rotate arbitrarily.
///
/// **Mathematical decision (not a swallow):** `gap = 0` is a *real result*, not an
/// error — it means the network is fragmenting and `sinθ ≤ ‖E‖₂ / 0 = +∞`. We
/// surface that as `f64::INFINITY`, the workspace's established divergence sentinel
/// (cf. [`crate::weyl_over_fiedler`], [`crate::kirchhoff_index`],
/// [`crate::collapse_number`]). It is **finiteness-checkable** — downstream code
/// branches on `davis_kahan_bound.is_finite()` (see the
/// `davis_kahan_bounds_the_realized_rotation` test) — and it is **never `NaN`**.
/// The one invariant the gate guarantees: the output is never `NaN` and never a
/// silently-wrong finite number; it is either a trustworthy finite bound or the
/// explicit `+∞` divergence signal.
pub const FRAGMENTATION_SENTINEL: f64 = f64::INFINITY;

/// Result of the rank-1 spectral perturbation analysis for one line trip.
#[derive(Debug, Clone)]
pub struct SpectralPerturbation {
    /// Index of the tripped line.
    pub line: usize,
    /// `‖E‖₂ = 2·b_k`, the Weyl perturbation budget.
    pub e_norm: f64,
    /// `maxᵢ |λᵢ(L') − λᵢ(L)|`, the largest realized eigenvalue shift.
    pub max_eigenvalue_shift: f64,
    /// Whether Weyl's bound held (max shift ≤ ‖E‖₂, within tolerance).
    pub weyl_satisfied: bool,
    /// Fiedler value `λ₂` before the trip (algebraic connectivity).
    pub fiedler_before: f64,
    /// Fiedler value `λ₂` after the trip.
    pub fiedler_after: f64,
    /// Realized Fiedler-vector rotation `sinθ ∈ [0,1]`.
    pub fiedler_rotation_sin: f64,
    /// Davis–Kahan bound on that rotation (`‖E‖₂ / gap`); `inf` if `gap == 0`.
    pub davis_kahan_bound: f64,
}

impl SpectralPerturbation {
    /// Fractional loss of algebraic connectivity, `1 − λ₂'/λ₂`. Near 1 ⇒ the
    /// trip nearly disconnects the network.
    ///
    /// **Degenerate denominator (`λ₂ → 0` before the trip):** an already-
    /// fragmented / barely-connected network has `fiedler_before ≈ 0`, so the
    /// ratio `λ₂'/λ₂` is `0/0`. That is not a loss — there was no connectivity to
    /// lose — so the documented decision is `0.0`, gated by the absolute floor
    /// (the Fiedler value is itself an eigenvalue near 0 here, so an absolute
    /// floor, not a relative one, is the right test). The result is then clamped
    /// to `[0, 1]` so a tiny noisy `fiedler_before` can never produce a spurious
    /// loss outside the fraction's domain — never `NaN`, never out of range.
    pub fn connectivity_loss(&self) -> f64 {
        if self.fiedler_before.abs() < SPECTRAL_GAP_FLOOR {
            0.0
        } else {
            (1.0 - self.fiedler_after / self.fiedler_before).clamp(0.0, 1.0)
        }
    }
}

/// Analyse the rank-1 spectral perturbation of tripping `line` from the
/// sub-network defined by `alive_before` (which must include `line`).
pub fn spectral_perturbation(
    grid: &Grid,
    alive_before: &[bool],
    line: usize,
) -> SpectralPerturbation {
    assert!(
        alive_before[line],
        "line must be in service before tripping"
    );
    let n = grid.n;

    let before = symmetric_eigen(&grid.laplacian_of(alive_before), n);
    let mut alive_after = alive_before.to_vec();
    alive_after[line] = false;
    let after = symmetric_eigen(&grid.laplacian_of(&alive_after), n);

    let e_norm = 2.0 * grid.edges[line].susceptance;

    let max_eigenvalue_shift = before
        .values
        .iter()
        .zip(after.values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    let weyl_satisfied = max_eigenvalue_shift <= e_norm + 1e-6;

    // Fiedler index = 1 (second smallest) when the network is connected. Guard
    // tiny networks.
    let (fiedler_before, fiedler_after, fiedler_rotation_sin, davis_kahan_bound) = if n >= 3 {
        let fb = before.values[1];
        let fa = after.values[1];
        // The Fiedler spectral gap = separation of λ₂ from its neighbours. Both
        // terms are subtractions (ascending eigenvalues, but cyclic-Jacobi noise
        // on a degenerate cluster can perturb either), so the gap is built first…
        let gap = (before.values[1] - before.values[0]).min(before.values[2] - before.values[1]);
        let vb = before.eigenvector(1);
        let va = after.eigenvector(1);
        let dot: f64 = vb
            .iter()
            .zip(va.iter())
            .map(|(x, y)| x * y)
            .sum::<f64>()
            .abs();
        let sin = (1.0 - dot * dot).max(0.0).sqrt();
        // …then FLOORED after the subtract-and-`min` (the entropy_ladder pattern),
        // relative to the spectral scale λ_max. A gap at or below the relative
        // floor means spectral degeneracy / fragmentation — the Davis–Kahan bound
        // diverges and we surface the FRAGMENTATION_SENTINEL signal rather than a
        // noisy finite number or a NaN. The divide is then unconditional and can
        // never produce 0/0.
        let lambda_max = before.values.last().copied().unwrap_or(0.0).abs().max(1.0);
        let gap_floor = SPECTRAL_GAP_FLOOR * lambda_max;
        let dk = if gap > gap_floor {
            e_norm / gap
        } else {
            FRAGMENTATION_SENTINEL
        };
        (fb, fa, sin, dk)
    } else {
        // Networks too small for a separated Fiedler mode: no finite rotation
        // bound exists (the gap is undefined), so the sentinel is the honest
        // answer — never a fabricated finite number.
        (0.0, 0.0, 0.0, FRAGMENTATION_SENTINEL)
    };

    SpectralPerturbation {
        line,
        e_norm,
        max_eigenvalue_shift,
        weyl_satisfied,
        fiedler_before,
        fiedler_after,
        fiedler_rotation_sin,
        davis_kahan_bound,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Edge;

    fn ring(n: usize, b: f64) -> Grid {
        let edges = (0..n)
            .map(|i| Edge::new(i, (i + 1) % n, b, 100.0))
            .collect();
        Grid::new(n, edges)
    }

    #[test]
    fn weyl_inequality_holds_for_every_line() {
        let g = ring(8, 1.5);
        let alive = vec![true; g.edges.len()];
        for line in 0..g.edges.len() {
            let sp = spectral_perturbation(&g, &alive, line);
            assert!(
                sp.weyl_satisfied,
                "Weyl violated on line {line}: shift {} > ‖E‖ {}",
                sp.max_eigenvalue_shift, sp.e_norm
            );
        }
    }

    #[test]
    fn davis_kahan_bounds_the_realized_rotation() {
        let g = ring(10, 2.0);
        let alive = vec![true; g.edges.len()];
        let sp = spectral_perturbation(&g, &alive, 0);
        // The realized rotation must respect the Davis–Kahan bound (allow a
        // small numerical slack); skip the degenerate gap==inf case.
        if sp.davis_kahan_bound.is_finite() {
            assert!(
                sp.fiedler_rotation_sin <= sp.davis_kahan_bound + 1e-6,
                "rotation {} exceeds DK bound {}",
                sp.fiedler_rotation_sin,
                sp.davis_kahan_bound
            );
        }
    }

    /// **The blackout-precursor finiteness gate (B1).** Drive the model into the
    /// exact regime it exists to capture — `λ₂ → 0` / `gap → 0` / degenerate or
    /// fragmenting spectra — and assert that NO output is ever `NaN`, the
    /// degenerate Davis–Kahan bound surfaces as the documented divergence sentinel
    /// (not a silently-wrong finite number), and `connectivity_loss` stays in
    /// `[0, 1]`. This is the spectral-gap NaN landmine
    /// (`sinθ ≤ ‖E‖₂ / gap`, `gap → 0` at fragmentation) guarded by the
    /// floored-denominator pattern.
    #[test]
    fn blackout_precursor_regime_is_never_nan() {
        // A near-disconnected grid: two triangles joined by a vanishingly-weak
        // bridge. λ₂ is tiny and the Fiedler gap is near-degenerate — the
        // fragmentation precursor.
        let eps = 1e-13; // bridge weight far below the spectral floor
        let weak = Grid::new(
            6,
            vec![
                Edge::new(0, 1, 1.0, 100.0),
                Edge::new(1, 2, 1.0, 100.0),
                Edge::new(2, 0, 1.0, 100.0),
                Edge::new(3, 4, 1.0, 100.0),
                Edge::new(4, 5, 1.0, 100.0),
                Edge::new(5, 3, 1.0, 100.0),
                Edge::new(2, 3, eps, 100.0), // near-zero bridge → λ₂ ≈ 0
            ],
        );
        let alive = vec![true; weak.edges.len()];
        for line in 0..weak.edges.len() {
            let sp = spectral_perturbation(&weak, &alive, line);
            // Every reported scalar is finite-or-sentinel, NEVER NaN.
            assert!(!sp.e_norm.is_nan(), "e_norm NaN on line {line}");
            assert!(
                !sp.max_eigenvalue_shift.is_nan(),
                "max_eigenvalue_shift NaN on line {line}"
            );
            assert!(!sp.fiedler_before.is_nan(), "fiedler_before NaN on {line}");
            assert!(!sp.fiedler_after.is_nan(), "fiedler_after NaN on {line}");
            assert!(
                !sp.fiedler_rotation_sin.is_nan(),
                "rotation_sin NaN on {line}"
            );
            // The DK bound is either a trustworthy finite value or the explicit
            // +∞ divergence sentinel — never NaN.
            assert!(
                !sp.davis_kahan_bound.is_nan(),
                "davis_kahan_bound NaN on line {line}"
            );
            // connectivity_loss is a fraction: finite and in [0,1], never NaN.
            let loss = sp.connectivity_loss();
            assert!(loss.is_finite(), "connectivity_loss not finite on {line}");
            assert!(
                (0.0..=1.0).contains(&loss),
                "connectivity_loss {loss} out of [0,1] on line {line}"
            );
        }

        // Tripping the bridge itself fully fragments → the gap collapses → the DK
        // bound MUST be the sentinel (the documented divergence signal), not a
        // huge noisy finite number masquerading as a real bound.
        let sp_bridge = spectral_perturbation(&weak, &alive, 6);
        assert_eq!(
            sp_bridge.davis_kahan_bound, FRAGMENTATION_SENTINEL,
            "fragmenting trip must surface the divergence sentinel, got {}",
            sp_bridge.davis_kahan_bound
        );
        assert!(
            !sp_bridge.davis_kahan_bound.is_finite(),
            "the sentinel is the explicit divergence signal (is_finite() == false)"
        );
    }

    /// A fully disconnected graph (no bridge at all): two independent components,
    /// so the spectrum has two zero modes and λ₂ ≈ 0. Every output stays finite-
    /// or-sentinel and NaN-free.
    #[test]
    fn disconnected_graph_spectral_outputs_are_nan_free() {
        let split = Grid::new(
            6,
            vec![
                Edge::new(0, 1, 1.0, 100.0),
                Edge::new(1, 2, 1.0, 100.0),
                Edge::new(2, 0, 1.0, 100.0),
                Edge::new(3, 4, 1.0, 100.0),
                Edge::new(4, 5, 1.0, 100.0),
                Edge::new(5, 3, 1.0, 100.0),
                // no bridge: {0,1,2} ⫫ {3,4,5}
            ],
        );
        let alive = vec![true; split.edges.len()];
        let sp = spectral_perturbation(&split, &alive, 0);
        assert!(
            !sp.davis_kahan_bound.is_nan(),
            "DK bound NaN on disconnected"
        );
        assert!(
            !sp.connectivity_loss().is_nan(),
            "connectivity_loss NaN on disconnected"
        );
        // λ₂ ≈ 0 (two zero modes) and a finite trip ⇒ the gap collapses ⇒ sentinel.
        assert_eq!(
            sp.davis_kahan_bound, FRAGMENTATION_SENTINEL,
            "λ₂≈0 ⇒ gap collapses ⇒ divergence sentinel"
        );
    }

    /// **Normal-regime parity.** The floor gates ONLY the degenerate path. On a
    /// grid with a genuinely *separated* Fiedler eigenvalue (asymmetric weights, so
    /// `λ₂ ≠ λ₃` — unlike a symmetric ring, whose Fiedler mode is degenerate and
    /// correctly yields the sentinel), the Davis–Kahan bound is finite, positive,
    /// and bounds the realized rotation — the gate must not perturb healthy
    /// numbers. (The pre-existing `davis_kahan_bounds_the_realized_rotation` test
    /// covers the degenerate-ring sentinel path; this is the separated-gap path.)
    #[test]
    fn healthy_grid_davis_kahan_is_finite_and_bounds_rotation() {
        // An asymmetric weighted path 0–1–2–3–4: distinct edge weights break the
        // spectral symmetry, so λ₂ is separated from λ₃ ⇒ a real finite gap.
        let g = Grid::new(
            5,
            vec![
                Edge::new(0, 1, 1.0, 100.0),
                Edge::new(1, 2, 2.3, 100.0),
                Edge::new(2, 3, 0.7, 100.0),
                Edge::new(3, 4, 1.7, 100.0),
            ],
        );
        let alive = vec![true; g.edges.len()];
        let mut saw_finite = false;
        for line in 0..g.edges.len() {
            let sp = spectral_perturbation(&g, &alive, line);
            // The DK bound is never NaN in any case.
            assert!(!sp.davis_kahan_bound.is_nan(), "DK NaN on line {line}");
            if sp.davis_kahan_bound.is_finite() {
                saw_finite = true;
                assert!(
                    sp.davis_kahan_bound > 0.0,
                    "a real separated gap ⇒ a positive bound on line {line}"
                );
                assert!(
                    sp.fiedler_rotation_sin <= sp.davis_kahan_bound + 1e-6,
                    "realized rotation {} must respect the (unchanged) DK bound {} on line {line}",
                    sp.fiedler_rotation_sin,
                    sp.davis_kahan_bound
                );
            }
            // connectivity_loss on a healthy trip is a finite fraction in [0,1].
            let loss = sp.connectivity_loss();
            assert!(loss.is_finite() && (0.0..=1.0).contains(&loss));
        }
        assert!(
            saw_finite,
            "a separated-gap grid must produce at least one finite DK bound \
             (else the parity claim is vacuous)"
        );
    }

    #[test]
    fn cutting_a_bridge_drops_connectivity_to_zero() {
        // Two triangles joined by a single bridge line: tripping the bridge
        // disconnects the graph, so λ₂(after) ≈ 0 ⇒ connectivity_loss ≈ 1.
        let g = Grid::new(
            6,
            vec![
                Edge::new(0, 1, 1.0, 100.0),
                Edge::new(1, 2, 1.0, 100.0),
                Edge::new(2, 0, 1.0, 100.0),
                Edge::new(3, 4, 1.0, 100.0),
                Edge::new(4, 5, 1.0, 100.0),
                Edge::new(5, 3, 1.0, 100.0),
                Edge::new(2, 3, 1.0, 100.0), // the bridge
            ],
        );
        let alive = vec![true; g.edges.len()];
        let sp = spectral_perturbation(&g, &alive, 6);
        assert!(sp.weyl_satisfied);
        assert!(
            sp.connectivity_loss() > 0.99,
            "bridge cut should collapse connectivity, got loss {}",
            sp.connectivity_loss()
        );
    }
}
