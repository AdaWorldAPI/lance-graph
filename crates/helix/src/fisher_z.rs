//! Stage 3 — Fisher-Z / hyperbolic-depth alignment.
//!
//! `z = arctanh(s)` is identical to the hyperbolic depth `ρ = 2·arctanh(r)` up to
//! the factor 2 (geometry keeps the 2 as arc length ∫2/(1−t²); statistics drops it
//! for variance stabilisation). The Poincaré rim-densified depth thus arrives as a
//! by-product of the Fisher-Z alignment, with NO separate hyperbolic geometry in
//! the hot path. Map a bounded [−1,1] similarity onto an unbounded scale so equal
//! steps mean equal amounts — stretching rim-near differences before quantisation.
//!
//! # Two meanings, one `arctanh` core
//!
//! | Method | Formula | Meaning |
//! |---|---|---|
//! | [`Similarity::fisher_z`] | `½·(ln(1+s) − ln(1−s))` | Variance-stabilising z-score |
//! | [`Similarity::hyperbolic_depth`] | `2·arctanh(s)` | Poincaré-disk arc-length depth |
//!
//! Both are computed from the same `ln`-form expression so the hot path mirrors the
//! documented `simd_ln_f32` route used in the SIMD codec layers upstream.

/// A cosine (or other) similarity value, nominally in `[−1, 1]`.
///
/// The newtype enforces that callers think about the domain before calling
/// [`fisher_z`](Similarity::fisher_z) or
/// [`hyperbolic_depth`](Similarity::hyperbolic_depth); raw `f64` silently accepts
/// out-of-range values and produces `±∞` or `NaN`.  `Similarity` clamps internally
/// (see [`CLAMP_EPS`](Similarity::CLAMP_EPS)) so results are always finite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Similarity(pub f64);

impl Similarity {
    /// Clamp guard ε: inputs are clamped to `[−1+ε, 1−ε]` so `arctanh` stays finite.
    ///
    /// At `s = 1 − 1e-9` the Fisher-Z value is `≈ 10.36`, well within `f64` range.
    /// The guard is applied symmetrically so `fisher_z(−1)` and `fisher_z(2)` are
    /// both finite.
    pub const CLAMP_EPS: f64 = 1e-9;

    /// Fisher-Z transform `z = arctanh(s) = ½·(ln(1+s) − ln(1−s))`.
    ///
    /// The computation uses the explicit `ln` form rather than `f64::atanh` so that
    /// it mirrors the `simd_ln_f32` hot path used in the upstream codec layers.
    /// The input is clamped to `[−1+ε, 1−ε]` (where ε = [`CLAMP_EPS`](Self::CLAMP_EPS))
    /// before the transform, guaranteeing a finite result for every `f64` input.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use helix::fisher_z::Similarity;
    ///
    /// assert_eq!(Similarity(0.0).fisher_z(), 0.0);
    /// // Rim clamping: exact ±1 and out-of-range values are finite.
    /// assert!(Similarity(1.0).fisher_z().is_finite());
    /// assert!(Similarity(-1.0).fisher_z().is_finite());
    /// ```
    pub fn fisher_z(self) -> f64 {
        let s = self.0.clamp(-1.0 + Self::CLAMP_EPS, 1.0 - Self::CLAMP_EPS);
        0.5 * ((1.0 + s).ln() - (1.0 - s).ln())
    }

    /// Hyperbolic depth `ρ = 2·arctanh(r)` — exactly twice [`fisher_z`](Self::fisher_z).
    ///
    /// In the Poincaré-disk model the geodesic arc length from the centre to a point
    /// at Euclidean radius `r` is `2·arctanh(r)`. The factor-of-2 is the arc-length
    /// integral `∫₀ʳ 2/(1−t²) dt`; the Fisher-Z statistic drops it for variance
    /// stabilisation. Both share the same `ln`-form core.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use helix::fisher_z::Similarity;
    ///
    /// let s = Similarity(0.5);
    /// let ratio = s.hyperbolic_depth() / s.fisher_z();
    /// assert!((ratio - 2.0).abs() < 1e-15);
    /// ```
    pub fn hyperbolic_depth(self) -> f64 {
        2.0 * self.fisher_z()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── basic identities ──────────────────────────────────────────────────────

    #[test]
    fn zero_maps_to_zero() {
        assert_eq!(
            Similarity(0.0).fisher_z(),
            0.0,
            "arctanh(0) must be exactly 0"
        );
    }

    #[test]
    fn odd_symmetry() {
        // arctanh is an odd function: z(−s) = −z(s).
        for &s in &[0.1_f64, 0.3, 0.5, 0.7, 0.9, 0.99] {
            let pos = Similarity(s).fisher_z();
            let neg = Similarity(-s).fisher_z();
            assert!(
                (pos + neg).abs() < 1e-14,
                "odd-symmetry failed at s={s}: z(s)={pos}, z(-s)={neg}"
            );
        }
    }

    #[test]
    fn hyperbolic_depth_is_double_fisher_z() {
        for &s in &[-0.9_f64, -0.5, 0.0, 0.3, 0.7, 0.937, 0.982] {
            let sim = Similarity(s);
            let fz = sim.fisher_z();
            let hd = sim.hyperbolic_depth();
            assert!(
                (hd - 2.0 * fz).abs() < 1e-15,
                "hyperbolic_depth ≠ 2·fisher_z at s={s}: hd={hd}, 2·fz={}",
                2.0 * fz
            );
        }
    }

    // ── clamp safety ──────────────────────────────────────────────────────────

    #[test]
    fn exact_rim_values_are_finite() {
        assert!(
            Similarity(1.0).fisher_z().is_finite(),
            "fisher_z(1.0) must be finite (clamped)"
        );
        assert!(
            Similarity(-1.0).fisher_z().is_finite(),
            "fisher_z(-1.0) must be finite (clamped)"
        );
    }

    #[test]
    fn out_of_range_values_are_finite() {
        assert!(
            Similarity(2.0).fisher_z().is_finite(),
            "fisher_z(2.0) must be finite (clamped to 1−ε)"
        );
        assert!(
            Similarity(-2.0).fisher_z().is_finite(),
            "fisher_z(-2.0) must be finite (clamped to −1+ε)"
        );
    }

    #[test]
    fn exact_rim_hyperbolic_depth_is_finite() {
        assert!(Similarity(1.0).hyperbolic_depth().is_finite());
        assert!(Similarity(-1.0).hyperbolic_depth().is_finite());
    }

    // ── parity with std::f64::atanh ───────────────────────────────────────────

    /// The ln-form and `f64::atanh` must agree to < 1e-9 across the informative range.
    /// This confirms the `simd_ln_f32` mirror is numerically equivalent.
    #[test]
    fn ln_form_matches_std_atanh() {
        let samples = [-0.9_f64, -0.5, 0.0, 0.3, 0.7, 0.937, 0.982];
        for &s in &samples {
            let ln_form = Similarity(s).fisher_z();
            let std_form = s.atanh();
            assert!(
                (ln_form - std_form).abs() < 1e-9,
                "ln-form vs std::atanh disagreement at s={s}: ln={ln_form}, std={std_form}, diff={}",
                (ln_form - std_form).abs()
            );
        }
    }

    // ── monotonicity ──────────────────────────────────────────────────────────

    /// fisher_z must be strictly increasing: a larger similarity must map to a
    /// larger z-score.
    #[test]
    fn strictly_increasing() {
        let samples = [-0.99_f64, -0.7, -0.3, 0.0, 0.3, 0.7, 0.99];
        for window in samples.windows(2) {
            let (a, b) = (window[0], window[1]);
            let za = Similarity(a).fisher_z();
            let zb = Similarity(b).fisher_z();
            assert!(
                zb > za,
                "monotonicity violated: fisher_z({b})={zb} ≤ fisher_z({a})={za}"
            );
        }
    }

    // ── roundtrip ─────────────────────────────────────────────────────────────

    /// For any z, `fisher_z(tanh(z)) ≈ z` (inverse relationship).
    /// Tolerance 1e-6: the clamping at 1−ε introduces a tiny offset for large |z|.
    #[test]
    fn roundtrip_tanh_fisher_z() {
        let z_values = [-2.0_f64, -0.5, 0.5, 1.5, 3.0];
        for &z in &z_values {
            let recovered = Similarity(z.tanh()).fisher_z();
            assert!(
                (recovered - z).abs() < 1e-6,
                "roundtrip failed for z={z}: fisher_z(tanh(z))={recovered}, diff={}",
                (recovered - z).abs()
            );
        }
    }
}
