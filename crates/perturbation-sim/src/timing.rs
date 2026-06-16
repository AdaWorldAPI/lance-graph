//! Time, inertia, and the cascade clock — the mediator between structure and
//! collapse.
//!
//! The cascade is **hops × time-per-hop**: `simulate_outage` returns `rounds`
//! (the edge-propagation hop count); this module supplies the **clock**. Time
//! per hop is set by grid **inertia** (the swing equation: `RoCoF = f₀·ΔP/2H` —
//! low inertia ⇒ fast frequency collapse) plus relay reaction. **Inertia
//! *mediates* the structural perturbation (HH / Raumgewinn) → realized cascade
//! (TL / infight):** more inertia = slower clock = fewer hops complete before
//! protection/operators arrest it; less inertia (more renewables displacing
//! synchronous machines) = faster clock = a bigger cascade in the same window.
//!
//! The total timescale **fingerprints the mechanism** (the "tell"): seconds ⇒
//! electromechanical / frequency-voltage (low-inertia, RoCoF-driven); minutes ⇒
//! thermal-overload; hours ⇒ market/dispatch. The 28 Apr 2025 Iberian event
//! collapsed in **~27 s** → unmistakably the electromechanical, low-inertia,
//! voltage/frequency regime (consistent with ENTSO-E), NOT a slow thermal one.
//!
//! **Scope (honest):** this is a *first-order timescale estimator* (swing-
//! equation RoCoF + a relay band), not a transient-stability ODE integrator —
//! that is the dynamic fork. It converts a hop count to a wall-clock and back,
//! and classifies the mechanism by timescale.

/// European nominal frequency (Hz).
pub const F0_HZ: f64 = 50.0;

/// Rate of change of frequency (Hz/s) just after losing `delta_p_fraction` of
/// system power, given the aggregate inertia constant `inertia_h` (seconds).
/// Swing equation: `RoCoF = f₀·ΔP / (2·H)`. Low `H` ⇒ steep RoCoF.
pub fn rocof_hz_per_s(delta_p_fraction: f64, inertia_h: f64) -> f64 {
    if inertia_h <= 0.0 {
        return f64::INFINITY;
    }
    F0_HZ * delta_p_fraction / (2.0 * inertia_h)
}

/// Per-hop time: relay reaction `relay_s` plus the time for frequency to cross a
/// protection band `df_band` (Hz) at the current RoCoF. Higher inertia ⇒ smaller
/// RoCoF ⇒ longer per hop (slower clock).
pub fn per_hop_time(relay_s: f64, inertia_h: f64, delta_p_fraction: f64, df_band: f64) -> f64 {
    let rocof = rocof_hz_per_s(delta_p_fraction, inertia_h).max(1e-9);
    relay_s + df_band / rocof
}

/// Wall-clock duration of a `hops`-round cascade at `dt_per_hop` seconds.
pub fn cascade_wall_time(hops: usize, dt_per_hop: f64) -> f64 {
    hops as f64 * dt_per_hop
}

/// Inverse: the per-hop time implied by an observed total over `hops` rounds.
pub fn implied_dt_per_hop(total_seconds: f64, hops: usize) -> f64 {
    if hops == 0 {
        return total_seconds;
    }
    total_seconds / hops as f64
}

/// Mechanism fingerprint from the total cascade timescale — the diagnostic tell.
pub fn mechanism_from_timescale(seconds: f64) -> &'static str {
    if seconds < 60.0 {
        "electromechanical / frequency-voltage (low-inertia, RoCoF-driven)"
    } else if seconds < 3600.0 {
        "thermal-overload (conductor time constants)"
    } else {
        "market / dispatch / maintenance"
    }
}

/// HHTL per-tier weights `(w_raumgewinn, w_infight)` for HEEL→HIP→TWIG→LEAF,
/// each summing to 5 — coarse tiers weight **Raumgewinn** (4:1), fine tiers
/// weight **infight** (1:4). Encodes that the two theorems' relative importance
/// (and their coupling, which is scale-dependent — orthogonal globally, coupled
/// per leaf basin) shifts across the tiers.
pub const HHTL_WEIGHTS: [(f64, f64); 4] = [(4.0, 1.0), (3.0, 2.0), (2.0, 3.0), (1.0, 4.0)];

/// Per-tier composite risk `(w_R·raumgewinn + w_I·infight)/(w_R+w_I)` — blends
/// the two residents with the tier's weights (HEEL favours Raumgewinn, LEAF
/// favours infight). Inputs should be comparably scaled (e.g. z-scored).
pub fn tier_composite(tier: usize, raumgewinn: f64, infight: f64) -> f64 {
    let (wr, wi) = HHTL_WEIGHTS[tier.min(3)];
    (wr * raumgewinn + wi * infight) / (wr + wi)
}

/// Dimensionless **collapse number** `Π = (raumgewinn · spread) / (infight ·
/// inertia)`. The numerator `raumgewinn · spread ≈ time · distance` (the field
/// perturbation is a space-time front — how far × how fast); the denominator is
/// the local fight damped by inertia. **High Π ⇒ fast, wide spread
/// (blackout-prone); inertia and infight damp it (inverse correlation).**
/// CONJECTURE [H]: a proposed scaling law — needs a probe against observed
/// cascade size / the 27 s timescale before promotion to FINDING.
pub fn collapse_number(raumgewinn: f64, spread: f64, infight: f64, inertia: f64) -> f64 {
    let denom = infight * inertia;
    if denom.abs() < 1e-12 {
        f64::INFINITY
    } else {
        (raumgewinn * spread) / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocof_is_inverse_in_inertia() {
        // Halving inertia doubles RoCoF (the renewable-displacement effect).
        let hi = rocof_hz_per_s(0.1, 6.0);
        let lo = rocof_hz_per_s(0.1, 3.0);
        assert!((lo - 2.0 * hi).abs() < 1e-12, "RoCoF ∝ 1/H");
    }

    #[test]
    fn lower_inertia_speeds_the_clock() {
        // Less inertia ⇒ shorter per-hop time ⇒ faster cascade.
        let high_h = per_hop_time(0.2, 6.0, 0.1, 0.2);
        let low_h = per_hop_time(0.2, 2.0, 0.1, 0.2);
        assert!(
            low_h < high_h,
            "low inertia = faster clock: {low_h} < {high_h}"
        );
    }

    #[test]
    fn the_27_second_tell() {
        // ~27 s over a handful of hops ⇒ electromechanical, and the per-hop time
        // is a few seconds — the Iberian-event regime.
        assert!(mechanism_from_timescale(27.0).starts_with("electromechanical"));
        assert!(mechanism_from_timescale(600.0).starts_with("thermal"));
        let dt = implied_dt_per_hop(27.0, 7);
        assert!(dt > 1.0 && dt < 10.0, "≈{dt:.1} s/hop in the fast phase");
        // round-trip
        assert!((cascade_wall_time(7, dt) - 27.0).abs() < 1e-9);
    }

    #[test]
    fn tier_weights_shift_coarse_to_fine() {
        // HEEL favours Raumgewinn; LEAF favours infight (R=1, I=0 probe).
        let heel = tier_composite(0, 1.0, 0.0);
        let leaf = tier_composite(3, 1.0, 0.0);
        assert!(heel > leaf, "HEEL weights Raumgewinn more: {heel} > {leaf}");
        assert!((tier_composite(0, 1.0, 0.0) - 0.8).abs() < 1e-12); // 4/5
        assert!((tier_composite(3, 1.0, 0.0) - 0.2).abs() < 1e-12); // 1/5
    }

    #[test]
    fn collapse_number_inverse_in_inertia_and_infight() {
        let base = collapse_number(2.0, 3.0, 1.0, 1.0);
        assert!(collapse_number(2.0, 3.0, 1.0, 2.0) < base, "↓ with inertia");
        assert!(collapse_number(2.0, 3.0, 2.0, 1.0) < base, "↓ with infight");
        assert!(
            collapse_number(4.0, 3.0, 1.0, 1.0) > base,
            "↑ with Raumgewinn"
        );
        assert!(collapse_number(2.0, 6.0, 1.0, 1.0) > base, "↑ with spread");
    }
}
