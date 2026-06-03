//! Stage 1 — tomato-rose equal-area hemisphere placement.
//!
//! A flat φ-spiral disk slice curled up into a dome. Spherical equal-AREA via
//! `r = √u` / `Y = √(1 − r²)`, NOT hyperbolic — the hyperbolic-depth feeling is
//! supplied separately by the Fisher-Z step, not by this placement.
//!
//! The midpoint rule `u = (n + 0.5) / N` gives each residue index `n` a
//! fractional area slice of width `1/N` centred at `(n + 0.5)/N`, which removes
//! edge bias: neither pole (u = 0) nor rim (u = 1) is ever exactly reached.
//!
//! # Coordinate convention
//!
//! - `r ∈ [0, 1)` — equal-area disk radius (zero at pole, approaches 1 at rim).
//! - `y ∈ (0, 1]` — hemispheric lift, pole-distance (`Y` in the contract).
//! - `azimuth` — golden-angle stride `n · φ` in radians (unbounded; caller wraps
//!   mod 2π if a canonical angle is needed).
//! - Cartesian `(X, Z, Y)` = `(r·sin(azimuth), r·cos(azimuth), y)`.
//!
//! The rim coordinate `r` feeds the Fisher-Z step: `ρ = 2·arctanh(r)` gives the
//! hyperbolic depth. Late/fine residues (large `n`) sit near the rim (`r → 1`)
//! and thus acquire the largest hyperbolic depth.

use crate::constants::GOLDEN_RATIO;

/// A single residue index lifted onto the equal-area hemisphere.
///
/// All three fields are computed in one call to [`HemispherePoint::lift`]; they
/// are public so downstream stages can destructure without extra method calls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HemispherePoint {
    /// Equal-area disk radius `r = √u` ∈ [0, 1).
    ///
    /// `u = (n + 0.5) / N` (midpoint rule). Because `u < 1` always holds, `r`
    /// never reaches 1 exactly, keeping `arctanh(r)` finite in the Fisher-Z
    /// step.
    pub r: f64,

    /// Hemispheric lift (pole distance) `Y = √(1 − u)` ∈ (0, 1].
    ///
    /// Equivalently `√(1 − r²)` because `u = r²`. The pole (`y = 1`) is
    /// approached by early indices; the equator (`y → 0`) is approached by late
    /// indices.
    pub y: f64,

    /// Golden-angle azimuth `n · φ` in radians (unbounded).
    ///
    /// Using the raw stride `n · GOLDEN_RATIO` (not the golden angle in radians)
    /// matches the Weyl / low-discrepancy literature convention: wrap mod 2π only
    /// when a canonical angle is required for display or trigonometry.
    pub azimuth: f64,
}

impl HemispherePoint {
    /// Lift residue index `n` of `total` (= N) onto the equal-area hemisphere.
    ///
    /// Uses the midpoint rule `u = (n + 0.5) / N` to avoid edge bias. `total`
    /// is clamped to `max(1, total)` so that `lift(n, 0)` is defined (returns a
    /// valid point rather than panicking or producing NaN).
    ///
    /// # Arguments
    ///
    /// * `n` — zero-based residue index (0 ≤ n < total is the intended range;
    ///   larger values are accepted and produce `r > 1`, which the Fisher-Z step
    ///   will clip or reject).
    /// * `total` — the total number of residue slots N. Clamped to 1 if zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use helix::placement::HemispherePoint;
    ///
    /// let p = HemispherePoint::lift(0, 4);
    /// // u = 0.5/4 = 0.125  →  r = √0.125,  y = √0.875
    /// assert!((p.r * p.r + p.y * p.y - 1.0).abs() < 1e-12);
    /// ```
    pub fn lift(n: usize, total: usize) -> Self {
        let total = total.max(1);
        let u = (n as f64 + 0.5) / total as f64;
        let r = u.sqrt();
        let y = (1.0 - u).sqrt();
        let azimuth = n as f64 * GOLDEN_RATIO;
        Self { r, y, azimuth }
    }

    /// Cartesian coordinates `(X, Z, Y)` on the unit hemisphere.
    ///
    /// - `X = r · sin(azimuth)` — east component.
    /// - `Z = r · cos(azimuth)` — north component.
    /// - `Y = self.y` — lift (height above the equatorial plane).
    ///
    /// The tuple order `(X, Z, Y)` matches the contract so that the third
    /// element is always the vertical lift; callers that want the more usual
    /// `(X, Y, Z)` convention must reorder.
    ///
    /// # Examples
    ///
    /// ```
    /// use helix::placement::HemispherePoint;
    ///
    /// let p = HemispherePoint::lift(0, 1);
    /// let (x, z, y) = p.cartesian();
    /// // r² + y² = 1  and  x² + z² = r²
    /// assert!((x * x + z * z + y * y - 1.0).abs() < 1e-12);
    /// ```
    pub fn cartesian(&self) -> (f64, f64, f64) {
        let x = self.r * self.azimuth.sin();
        let z = self.r * self.azimuth.cos();
        (x, z, self.y)
    }

    /// The bounded rim coordinate fed to the Fisher-Z step.
    ///
    /// Returns `self.r` ∈ [0, 1). The hyperbolic depth is `ρ = 2 · arctanh(r)`;
    /// late/fine residues (large `n`) sit near the rim (`r → 1`) and acquire the
    /// largest hyperbolic depth. Rim (r → 1) = late / fine detail.
    ///
    /// # Examples
    ///
    /// ```
    /// use helix::placement::HemispherePoint;
    ///
    /// let p = HemispherePoint::lift(3, 10);
    /// assert_eq!(p.rim(), p.r);
    /// ```
    pub fn rim(&self) -> f64 {
        self.r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── midpoint rule ────────────────────────────────────────────────────────

    #[test]
    fn midpoint_rule_lift_0_of_2() {
        // u = (0 + 0.5) / 2 = 0.25  →  r = √0.25 = 0.5,  y = √0.75
        let p = HemispherePoint::lift(0, 2);
        assert!(
            (p.r - 0.5_f64).abs() < 1e-15,
            "r should be 0.5, got {}",
            p.r
        );
        let expected_y = 0.75_f64.sqrt();
        assert!(
            (p.y - expected_y).abs() < 1e-15,
            "y should be √0.75, got {}",
            p.y
        );
    }

    #[test]
    fn midpoint_rule_lift_1_of_2() {
        // u = (1 + 0.5) / 2 = 0.75  →  r = √0.75,  y = √0.25 = 0.5
        let p = HemispherePoint::lift(1, 2);
        let expected_r = 0.75_f64.sqrt();
        assert!(
            (p.r - expected_r).abs() < 1e-15,
            "r should be √0.75, got {}",
            p.r
        );
        assert!(
            (p.y - 0.5_f64).abs() < 1e-15,
            "y should be 0.5, got {}",
            p.y
        );
    }

    // ── unit-sphere identity r² + y² = 1 ────────────────────────────────────

    #[test]
    fn unit_sphere_identity_various() {
        let cases: &[(usize, usize)] = &[
            (0, 1),
            (0, 10),
            (5, 10),
            (9, 10),
            (0, 1000),
            (499, 1000),
            (999, 1000),
            (7, 17),
        ];
        for &(n, total) in cases {
            let p = HemispherePoint::lift(n, total);
            let sum = p.r * p.r + p.y * p.y;
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "r²+y² should be 1 for ({n},{total}), got {sum}"
            );
        }
    }

    // ── strict monotonicity (fixed N = 20) ──────────────────────────────────

    #[test]
    fn r_strictly_increasing_y_strictly_decreasing() {
        let n = 20usize;
        let mut prev = HemispherePoint::lift(0, n);
        for i in 1..n {
            let cur = HemispherePoint::lift(i, n);
            assert!(
                cur.r > prev.r,
                "r should be strictly increasing: r[{i}]={} ≤ r[{}]={}",
                cur.r,
                i - 1,
                prev.r
            );
            assert!(
                cur.y < prev.y,
                "y should be strictly decreasing: y[{i}]={} ≥ y[{}]={}",
                cur.y,
                i - 1,
                prev.y
            );
            prev = cur;
        }
    }

    // ── azimuth formula ──────────────────────────────────────────────────────

    #[test]
    fn azimuth_is_n_times_golden_ratio() {
        for n in [0usize, 1, 2, 10, 99, 255] {
            let p = HemispherePoint::lift(n, 300);
            let expected = n as f64 * GOLDEN_RATIO;
            assert_eq!(p.azimuth, expected, "azimuth should equal n*φ for n={n}");
        }
    }

    // ── rim accessor ─────────────────────────────────────────────────────────

    #[test]
    fn rim_equals_r() {
        for n in [0usize, 3, 17, 100] {
            let p = HemispherePoint::lift(n, 200);
            assert_eq!(p.rim(), p.r, "rim() should equal r for n={n}");
        }
    }

    // ── total = 0 guard (no panic, no NaN) ──────────────────────────────────

    #[test]
    fn lift_with_total_zero_does_not_panic() {
        let p = HemispherePoint::lift(0, 0);
        // total clamped to 1  →  u = 0.5  →  r = √0.5,  y = √0.5
        assert!(p.r.is_finite(), "r must be finite when total=0");
        assert!(p.y.is_finite(), "y must be finite when total=0");
        assert!(p.azimuth.is_finite(), "azimuth must be finite when total=0");
        // sanity: unit-sphere identity still holds
        let sum = p.r * p.r + p.y * p.y;
        assert!((sum - 1.0).abs() < 1e-12, "r²+y² should be 1, got {sum}");
    }

    // ── cartesian: x² + z² = r² and overall unit sphere ────────────────────

    #[test]
    fn cartesian_lies_on_unit_sphere() {
        for &(n, total) in &[(0usize, 1usize), (3, 10), (7, 17), (500, 1000)] {
            let p = HemispherePoint::lift(n, total);
            let (x, z, y) = p.cartesian();
            // x² + z² = r²
            let radial = x * x + z * z;
            assert!(
                (radial - p.r * p.r).abs() < 1e-12,
                "x²+z² should equal r² for ({n},{total})"
            );
            // x² + z² + y² = 1
            let total_sq = radial + y * y;
            assert!(
                (total_sq - 1.0).abs() < 1e-12,
                "x²+z²+y² should be 1 for ({n},{total}), got {total_sq}"
            );
        }
    }

    // ── equal-area equidistribution sanity (N = 1000) ───────────────────────
    //
    // For N points placed with the midpoint rule, the fraction of points with
    // r ≤ √(m/N) (i.e. u ≤ m/N) should be ≈ m/N (equal-area property).
    // We check m = 100 and m = 500 with a generous ±5% tolerance.

    #[test]
    fn equal_area_equidistribution_n1000() {
        let big_n = 1000usize;
        let points: Vec<HemispherePoint> = (0..big_n)
            .map(|i| HemispherePoint::lift(i, big_n))
            .collect();

        // m = 100: expect ~10% of points inside r ≤ √(100/1000) = √0.1
        let threshold_100 = (100.0_f64 / big_n as f64).sqrt();
        let count_100 = points.iter().filter(|p| p.r <= threshold_100).count();
        let frac_100 = count_100 as f64 / big_n as f64;
        assert!(
            (frac_100 - 0.1).abs() < 0.05,
            "equal-area: expected ~10% inside r≤√0.1, got {:.1}%",
            frac_100 * 100.0
        );

        // m = 500: expect ~50% of points inside r ≤ √(500/1000) = √0.5
        let threshold_500 = (500.0_f64 / big_n as f64).sqrt();
        let count_500 = points.iter().filter(|p| p.r <= threshold_500).count();
        let frac_500 = count_500 as f64 / big_n as f64;
        assert!(
            (frac_500 - 0.5).abs() < 0.05,
            "equal-area: expected ~50% inside r≤√0.5, got {:.1}%",
            frac_500 * 100.0
        );
    }
}
