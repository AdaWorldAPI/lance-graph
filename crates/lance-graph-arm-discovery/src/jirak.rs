// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Jirak-bound Stage-A significance floor (D-ARM-7, `I-NOISE-FLOOR-JIRAK`).
//!
//! The classical ARM floors ([`crate::rule::CandidateRule::passes`]) gate on
//! hand-tuned `min_support`/`min_confidence`. They say nothing about whether an
//! observed support is *statistically distinguishable from noise* at the given
//! window size — and for this substrate the items are **weakly dependent by
//! construction** (shared categorical encodings, partial-order co-occurrence),
//! so the classical IID Berry-Esseen concentration is the WRONG yardstick
//! (iron rule `I-NOISE-FLOOR-JIRAK`). Without this floor the proposer leaks
//! thin-but-frequent noise into the SPO store faster than NARS revision can
//! down-weight it (ISSUE `ARM-JIRAK-FLOOR`; plan §4 "this is not optional").
//!
//! ## The rate (and a spec correction)
//!
//! Jirak 2016 (arXiv 1606.01617, Ann. Probab. 44(3) 2024–2063) gives the
//! Berry-Esseen error under weak dependence as `n^{-(p/2-1)}` for dependence
//! moments `p ∈ (2, 3]`, saturating at the classical `n^{-1/2}` for `p ≥ 4`
//! (and capped there for all `p ≥ 3`). At `p = 2.5` that is `n^{-0.25}` —
//! much *slower* than IID, so the significance floor stays *higher*: stricter.
//!
//! **Spec correction (documented deviation):** plan
//! `streaming-arm-nars-discovery-v1.md` §4 spells the default threshold as
//! `n^{-1/(p/2-1)}` and claims "p ≈ 3.0 … giving n^{-1} decay". Both are
//! internally inconsistent with the plan's own examples (`p=2.5 → n^{-0.25}`,
//! `p=4 → n^{-1/2}`), with the iron rule's stated rate, and with the
//! empirical pillar (`jc::jirak`, which measures `n^{-0.25}` at `p=2.5`).
//! This module implements the rate all three agree on:
//! `exponent = min(p/2 - 1, 1/2)`.
//!
//! ## The construction (honesty per the iron rule)
//!
//! `threshold(n, p, α) = z(α) · n^{-min(p/2-1, 1/2)}`
//!
//! where `z(α)` is the upper-tail standard-normal quantile (Abramowitz &
//! Stegun 26.2.23 rational approximation, |ε| < 4.5e-4 — deterministic, no
//! tables). This is a *bound-shaped conservative composite* — the Jirak rate
//! replaces the IID `n^{-1/2}` concentration and the quantile carries the
//! significance level — not the theorem's exact constant (which is
//! data-dependent). `I-NOISE-FLOOR-JIRAK` requires thresholds to cite the
//! Jirak rate and to say when they are construction-level rather than
//! constant-exact: this one is construction-level, monotone in all three
//! arguments, and reduces to the one-sided classical `1.645/√n` at the
//! `p = 3` boundary.
//!
//! ## Relation to `jc::jirak` (declared — not a duplicate)
//!
//! `jc::jirak` is the **pillar prover**: it measures the empirical
//! Berry-Esseen error on simulated weakly-dependent fingerprints and verifies
//! the Jirak rate beats the classical citation. This module is the **gate
//! function** that applies the proven rate as a Stage-A floor. Prover vs
//! gate — same name, two roles, relation declared here per the census
//! discipline.
//!
//! ## Float discipline
//!
//! The crate's decision path is float-free. `f32` appears here only in the
//! one-time threshold *derivation* ([`jirak_significance_threshold`]); the
//! decision path compares integers via [`jirak_floor_ppm`] (parts-per-million,
//! same fixed-point scale as [`crate::rule::PPM`]). Derive once per window,
//! compare in ppm.

use crate::rule::PPM;

/// Default dependence moment `p` for ARM items (shared categorical encoding +
/// partial-order co-occurrence — the canonical weak-dependence pattern).
/// Plan §4 default. At `p = 3.0` the rate sits exactly at the classical
/// `n^{-1/2}` boundary; lower it toward 2 when the feed is known to be more
/// strongly dependent (stricter floor).
pub const DEFAULT_P_MOMENT: f32 = 3.0;

/// Default one-sided significance level α (plan §4 default).
pub const DEFAULT_ALPHA: f32 = 0.05;

/// Upper-tail standard-normal quantile `z(α)` via Abramowitz & Stegun
/// 26.2.23 (rational approximation, |ε| < 4.5e-4). `α` is clamped to
/// `[1e-6, 0.5]`; `z(0.5) = 0`, `z(0.05) ≈ 1.6449`, `z(0.025) ≈ 1.9600`.
fn probit_upper(alpha: f32) -> f32 {
    let a = f64::from(alpha).clamp(1e-6, 0.5);
    let t = (-2.0 * a.ln()).sqrt();
    let z = t - (2.30753 + 0.27061 * t) / (1.0 + 0.99229 * t + 0.04481 * t * t);
    z.max(0.0) as f32
}

/// The Jirak-bound Stage-A significance threshold on observed support, as a
/// fraction in `(0, 1]` — the spec'd D-ARM-7 surface (plan §4, with the rate
/// correction documented in the module docs).
///
/// `threshold = z(confidence_alpha) · window_size^{-min(p_moment/2 - 1, 1/2)}`
///
/// Guards (all yield the admit-nothing threshold `1.0`):
/// - `window_size == 0` — no evidence, nothing is significant;
/// - `p_moment <= 2.0` — outside Jirak's `p ∈ (2, 3]` regime, no CLT rate to
///   stand on;
/// - any non-finite input.
///
/// The result is clamped to `(0, 1]`; on small windows the floor legitimately
/// saturates at 1.0 (no support level is certifiable).
#[must_use]
pub fn jirak_significance_threshold(window_size: u32, p_moment: f32, confidence_alpha: f32) -> f32 {
    if window_size == 0 || !p_moment.is_finite() || p_moment <= 2.0 {
        return 1.0;
    }
    // Jirak rate exponent: n^{-(p/2-1)} for p ∈ (2,3], saturating at the
    // classical n^{-1/2} for p ≥ 3 (and staying there per Jirak's p ≥ 4 L^q
    // result). Conservative: never decays faster than 1/√n.
    let exponent = f64::from(p_moment / 2.0 - 1.0).min(0.5);
    let z = f64::from(probit_upper(confidence_alpha));
    let thr = z * f64::from(window_size).powf(-exponent);
    thr.clamp(f64::MIN_POSITIVE, 1.0) as f32
}

/// Integer edge of [`jirak_significance_threshold`]: the floor in
/// parts-per-million (ceiling-rounded, clamped to `PPM`). Derive once per
/// window; the per-rule comparison `support_ppm >= floor_ppm` is then pure
/// integer — the crate's decision path never touches the float.
#[must_use]
pub fn jirak_floor_ppm(window_size: u32, p_moment: f32, confidence_alpha: f32) -> u32 {
    let thr = f64::from(jirak_significance_threshold(
        window_size,
        p_moment,
        confidence_alpha,
    ));
    let ppm = (thr * PPM as f64).ceil();
    if ppm >= PPM as f64 {
        PPM as u32
    } else {
        ppm as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── probit (A&S 26.2.23) ────────────────────────────────────────────────

    #[test]
    fn probit_matches_standard_quantiles() {
        assert!((probit_upper(0.05) - 1.6449).abs() < 2e-3, "z(0.05)");
        assert!((probit_upper(0.025) - 1.9600).abs() < 2e-3, "z(0.025)");
        assert!((probit_upper(0.01) - 2.3263).abs() < 2e-3, "z(0.01)");
        assert!(probit_upper(0.5).abs() < 2e-3, "z(0.5) = 0");
    }

    // ── threshold values (hand-computed against the corrected rate) ─────────

    #[test]
    fn threshold_at_p3_is_classical_one_sided() {
        // p = 3 → exponent 0.5 → z(0.05)/√n. n = 10_000 → ≈ 0.016449.
        let thr = jirak_significance_threshold(10_000, 3.0, 0.05);
        assert!((0.0160..=0.0169).contains(&thr), "got {thr}");
        // n = 100 → ≈ 0.16449.
        let thr = jirak_significance_threshold(100, 3.0, 0.05);
        assert!((0.160..=0.169).contains(&thr), "got {thr}");
    }

    #[test]
    fn lower_p_moment_is_strictly_stricter() {
        // p = 2.5 → exponent 0.25 → z·n^{-1/4}: at n = 10_000 that is
        // ≈ 0.16449 — a 10× higher floor than p = 3 at the same window.
        // (This is the plan's own §4 example, p=2.5 → n^{-0.25}.)
        let strict = jirak_significance_threshold(10_000, 2.5, 0.05);
        let classical = jirak_significance_threshold(10_000, 3.0, 0.05);
        assert!((0.160..=0.169).contains(&strict), "got {strict}");
        assert!(strict > classical * 9.0, "p=2.5 must dominate p=3");
    }

    #[test]
    fn p_at_or_above_3_saturates_at_the_classical_rate() {
        // Jirak: p ≥ 4 stays n^{-1/2}; the exponent is capped so p = 3, 4, 8
        // all gate identically (never decays faster than 1/√n).
        let p3 = jirak_significance_threshold(10_000, 3.0, 0.05);
        let p4 = jirak_significance_threshold(10_000, 4.0, 0.05);
        let p8 = jirak_significance_threshold(10_000, 8.0, 0.05);
        assert_eq!(p3, p4);
        assert_eq!(p4, p8);
    }

    #[test]
    fn monotone_in_window_p_and_alpha() {
        // Larger window → lower floor.
        assert!(
            jirak_significance_threshold(100_000, 3.0, 0.05)
                < jirak_significance_threshold(1_000, 3.0, 0.05)
        );
        // Stronger dependence (lower p) → higher floor.
        assert!(
            jirak_significance_threshold(10_000, 2.2, 0.05)
                > jirak_significance_threshold(10_000, 2.8, 0.05)
        );
        // Stricter significance (lower α) → higher floor.
        assert!(
            jirak_significance_threshold(10_000, 3.0, 0.01)
                > jirak_significance_threshold(10_000, 3.0, 0.10)
        );
    }

    #[test]
    fn guards_admit_nothing() {
        assert_eq!(jirak_significance_threshold(0, 3.0, 0.05), 1.0);
        assert_eq!(jirak_significance_threshold(1_000, 2.0, 0.05), 1.0);
        assert_eq!(jirak_significance_threshold(1_000, 1.5, 0.05), 1.0);
        assert_eq!(jirak_significance_threshold(1_000, f32::NAN, 0.05), 1.0);
        // Tiny window: z·n^{-e} > 1 → clamped to 1.0 (nothing certifiable).
        assert_eq!(jirak_significance_threshold(1, 3.0, 0.05), 1.0);
        assert_eq!(jirak_significance_threshold(2, 2.1, 0.05), 1.0);
    }

    // ── integer edge ─────────────────────────────────────────────────────────

    #[test]
    fn floor_ppm_is_the_ceil_of_the_threshold() {
        // n = 10_000, p = 3, α = 0.05 → ≈ 0.016449 → ≈ 16_445 ppm.
        let ppm = jirak_floor_ppm(10_000, 3.0, 0.05);
        assert!((16_000..=17_000).contains(&ppm), "got {ppm}");
        // n = 600 (the extract fixture size) → 1.6449/√600 ≈ 6.72% ≈ 67_157.
        let ppm = jirak_floor_ppm(600, 3.0, 0.05);
        assert!((66_000..=68_000).contains(&ppm), "got {ppm}");
        // Saturated threshold → exactly PPM, never above.
        assert_eq!(jirak_floor_ppm(0, 3.0, 0.05), PPM as u32);
        assert_eq!(jirak_floor_ppm(1, 3.0, 0.05), PPM as u32);
    }

    #[test]
    fn floor_ppm_defaults_are_the_plan_defaults() {
        assert_eq!(
            jirak_floor_ppm(5_000, DEFAULT_P_MOMENT, DEFAULT_ALPHA),
            jirak_floor_ppm(5_000, 3.0, 0.05)
        );
    }
}
