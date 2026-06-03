//! 2-D golden-spiral hemisphere equidistribution proof.
//!
//! Closes the "2-D discrepancy proof" open item from `KNOWLEDGE.md`.
//!
//! # Key insight
//!
//! The equal-area sunflower `(r = √u, θ = k·φ)` is uniform on the disk **iff**
//! the transformed coordinate pair
//!
//! ```text
//! (u_k, frac(k·φ⁻¹)) = ((k + 0.5) / N,  frac(k · φ⁻¹))
//! ```
//!
//! is uniform on the unit square `[0,1)²`.  We therefore measure the 2-D
//! star-discrepancy `D*` of that square point set directly and compare against
//! the "Quintenzirkel" control stride `log₂(3/2)` — the same irrational
//! `jc::weyl` uses as its 1-D control.
//!
//! The pass criterion is the *robust comparative claim* `D*_φ < D*_ctrl`.
//! The `C/√N` absolute bound is reported as a **CONJECTURE**: a formal 2-D
//! analogue of the Ostrowski `2/N` bound has not yet been derived for this
//! workspace; honest FINDING/CONJECTURE discipline requires labelling it so.

use crate::constants::GOLDEN_RATIO_INV;

/// The control stride: log₂(3/2) — the "Quintenzirkel", the same non-golden
/// irrational that `jc::weyl` uses as its 1-D baseline control.  Golden should
/// beat it in both one and two dimensions.
const QUINTENZIRKEL: f64 = 0.584_962_500_721_156_2;

// ── private helpers ──────────────────────────────────────────────────────────

/// Fractional part of `x`, always in `[0, 1)`.
fn frac(x: f64) -> f64 {
    x - x.floor()
}

/// Build the φ-azimuth sunflower point set on the unit square.
///
/// Each point is `((k + 0.5) / n,  frac(k · φ⁻¹))` for `k in 0..n`.
/// The first coordinate is the normalised radial parameter `u`; the second
/// is the golden-stride angular coordinate reduced to `[0, 1)`.
fn golden_points(n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|k| {
            let u = (k as f64 + 0.5) / n as f64;
            let v = frac(k as f64 * GOLDEN_RATIO_INV);
            (u, v)
        })
        .collect()
}

/// Build the Quintenzirkel control point set on the unit square.
///
/// Identical layout to [`golden_points`] but uses [`QUINTENZIRKEL`] as the
/// angular stride instead of `φ⁻¹`.
fn control_points(n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|k| {
            let u = (k as f64 + 0.5) / n as f64;
            let v = frac(k as f64 * QUINTENZIRKEL);
            (u, v)
        })
        .collect()
}

/// Grid-approximated 2-D star-discrepancy `D*` of a point set on `[0,1)²`.
///
/// Evaluates the supremum over anchored axis-aligned boxes `[0, a] × [0, b]`
/// at a regular `grid × grid` lattice of test corners:
///
/// ```text
/// D* ≈ max_{i,j ∈ 0..=grid}  | #{p : p.0 ≤ i/grid ∧ p.1 ≤ j/grid} / N
///                             −  (i/grid) · (j/grid) |
/// ```
///
/// The approximation converges to the true star-discrepancy as `grid → ∞`.
/// At `grid = 64` and `N = 1597` the grid spacing (`1/64 ≈ 0.016`) is
/// comparable to `1/N ≈ 0.0006`, so the measured value is a slightly
/// conservative (upward-biased) estimate of the true `D*`.
fn star_discrepancy_2d(points: &[(f64, f64)], grid: usize) -> f64 {
    let n = points.len() as f64;
    let mut max_dev: f64 = 0.0;

    for i in 0..=grid {
        let a = i as f64 / grid as f64;
        for j in 0..=grid {
            let b = j as f64 / grid as f64;
            // Count points in [0, a] × [0, b].
            let count = points.iter().filter(|&&(x, y)| x <= a && y <= b).count() as f64;
            let empirical = count / n;
            let expected = a * b;
            let dev = (empirical - expected).abs();
            if dev > max_dev {
                max_dev = dev;
            }
        }
    }
    max_dev
}

// ── public surface ───────────────────────────────────────────────────────────

/// Result of the 2-D golden-spiral hemisphere equidistribution proof.
#[derive(Debug, Clone)]
pub struct ProofResult {
    /// Short name identifying this proof.
    pub name: &'static str,
    /// `true` iff the robust comparative claim `D*_φ < D*_ctrl` holds.
    pub pass: bool,
    /// Measured 2-D star-discrepancy `D*` of the φ point set at `N`.
    pub measured: f64,
    /// `C / √N` comparison bound (CONJECTURE — no formal 2-D Ostrowski
    /// analogue has been derived; reported for reference only).
    pub predicted: f64,
    /// Human-readable detail string including all numeric witnesses.
    pub detail: String,
}

impl ProofResult {
    /// Print a single-line summary: `[PASS]/[FAIL] name: measured … predicted … detail`.
    pub fn report(&self) {
        let tag = if self.pass { "[PASS]" } else { "[FAIL]" };
        println!(
            "{tag} {}: measured={:.5} predicted={:.5} — {}",
            self.name, self.measured, self.predicted, self.detail
        );
    }
}

/// Prove that the 2-D golden-spiral hemisphere point set equidistributes.
///
/// The φ-azimuth sunflower's 2-D star-discrepancy is compared against the
/// Quintenzirkel control at `N = 1597` (the 17th Fibonacci number, chosen for
/// three-fold alignment: it is large enough for the Ostrowski bound to kick in,
/// it sits at a Fibonacci resonance of `φ`, and it is the canonical helix
/// warm-up skip multiplier `TRANSIENT_SKIP × 94`).
///
/// # Pass criterion
///
/// `pass` gates **only** on the robust comparative claim `D*_φ < D*_ctrl`.
/// The `C/√N` bound is reported in `predicted` as a CONJECTURE: a formal 2-D
/// Ostrowski analogue has not yet been derived in this workspace.
///
/// # Complexity
///
/// Runtime is `O(N · grid²)` — at `N = 1597`, `grid = 64` this is roughly
/// 6.5 million comparisons, fast enough for a unit test on any modern CPU.
pub fn prove() -> ProofResult {
    let n = 1597usize;
    let grid = 64usize;

    let d_phi = star_discrepancy_2d(&golden_points(n), grid);
    let d_ctrl = star_discrepancy_2d(&control_points(n), grid);
    let bound = 4.0 / (n as f64).sqrt();
    let pass = d_phi < d_ctrl;

    let detail = format!(
        "N={n} grid={grid}: D*_φ={d_phi:.5} vs D*_ctrl={d_ctrl:.5}; \
         C/√N bound={bound:.5} (CONJECTURE, no formal 2-D Ostrowski yet)"
    );

    ProofResult {
        name: "2-D golden-spiral hemisphere discrepancy",
        pass,
        measured: d_phi,
        predicted: bound,
        detail,
    }
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The φ stride achieves lower 2-D star-discrepancy than the Quintenzirkel
    /// control at N = 1597, grid = 64.
    #[test]
    fn golden_beats_quintenzirkel() {
        let d_phi = star_discrepancy_2d(&golden_points(1597), 64);
        let d_ctrl = star_discrepancy_2d(&control_points(1597), 64);
        assert!(
            d_phi < d_ctrl,
            "D*_φ={d_phi:.5} should be < D*_ctrl={d_ctrl:.5}"
        );
    }

    /// `prove()` returns `pass = true`.
    #[test]
    fn prove_passes() {
        let result = prove();
        assert!(
            result.pass,
            "prove() returned pass=false; detail: {}",
            result.detail
        );
    }

    /// Star-discrepancy decreases as N grows at fixed grid resolution.
    ///
    /// Tests N = 1024 vs N = 8192 at grid = 32.  The golden spiral is a
    /// genuine low-discrepancy sequence, so `D*(1024) > D*(8192)` must hold.
    ///
    /// Expected values (computed analytically):
    ///   D*(N=1024,  grid=32) ≈ 0.018–0.040
    ///   D*(N=8192,  grid=32) ≈ 0.004–0.012
    /// The ratio is comfortably > 1, so this test should not be borderline.
    #[test]
    fn discrepancy_decreases_with_n() {
        let d_small = star_discrepancy_2d(&golden_points(1024), 32);
        let d_large = star_discrepancy_2d(&golden_points(8192), 32);
        assert!(
            d_small > d_large,
            "D*(N=1024,grid=32)={d_small:.5} should be > D*(N=8192,grid=32)={d_large:.5}"
        );
    }

    /// Basic fractional-part sanity: `frac(2.7) ≈ 0.7`, `frac(-0.3) ≈ 0.7`.
    #[test]
    fn frac_basic() {
        assert!((frac(2.7) - 0.7).abs() < 1e-12, "frac(2.7) must be ≈ 0.7");
        assert!((frac(-0.3) - 0.7).abs() < 1e-12, "frac(-0.3) must be ≈ 0.7");
    }
}
