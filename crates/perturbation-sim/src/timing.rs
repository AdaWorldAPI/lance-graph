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

/// **Meta-hop cascade** — the tier-as-hop simplification. Instead of an
/// N-line cascade, treat each HHTL tier as ONE hop and let tier `i` MODIFY
/// tier `i+1`: the perturbation amplitude entering the next tier is scaled by
/// that tier's **pass-through gain** `gᵢ = infightᵢ · (1 − raumgewinnᵢ)` (raw
/// per-tier `raumgewinn`/`infight` are min-max normalized to `[0,1]` internally,
/// so `gᵢ ∈ [0,1]`). A tier with strong local fight and weak field connectivity
/// passes the perturbation on (deep penetration); a well-connected tier absorbs
/// it. Returns `(per-tier amplitude incl. the seed, meta_hops)` where
/// `meta_hops` = how many tiers the amplitude stays `≥ threshold` (the
/// penetration depth, 0..=tiers). Total cascade time ≈ `meta_hops · Δt`
/// (the inertia clock), so the whole event is ≤ 4 meta-hops — easy to model.
pub fn meta_cascade(raumgewinn: &[f64], infight: &[f64], threshold: f64) -> (Vec<f64>, usize) {
    let n = raumgewinn.len().min(infight.len());
    if n == 0 {
        return (vec![1.0], 0);
    }
    let norm = |xs: &[f64]| -> Vec<f64> {
        let lo = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if (hi - lo).abs() < 1e-30 {
            vec![0.5; xs.len()]
        } else {
            xs.iter().map(|&x| (x - lo) / (hi - lo)).collect()
        }
    };
    let rn = norm(&raumgewinn[..n]);
    let inn = norm(&infight[..n]);

    let mut amps = vec![1.0]; // seed at the first tier
    let mut hops = 0;
    for i in 0..n {
        if amps[i] >= threshold {
            hops += 1;
        }
        let g = (inn[i] * (1.0 - rn[i])).clamp(0.0, 1.0); // pass-through gain
        amps.push(amps[i] * g);
    }
    (amps, hops)
}

/// One between-level (tier→tier) meta-hop in the **inertia × phase** cascade.
#[derive(Debug, Clone, Copy)]
pub struct MetaHop {
    /// Tier index (0 = HEEL … 3 = LEAF).
    pub tier: usize,
    /// Signed contribution entering this tier = phase (±) × magnitude. The sign
    /// is the bipolar Walsh phase carried down from the level above.
    pub signed_amp: f64,
    /// Running **interference field** `Σ signed contributions` — the bundled
    /// perturbation as seen at this tier. Aligned phases add (constructive,
    /// the field grows ⇒ deeper cascade); opposed phases cancel (destructive,
    /// the field self-arrests ⇒ shallow cascade).
    pub field: f64,
    /// Inertia clock for this hop (s) — the swing-equation per-hop time
    /// ([`per_hop_time`]); higher inertia ⇒ longer dt ⇒ slower descent.
    pub dt: f64,
    /// Cumulative wall-clock when this hop fires (s).
    pub t: f64,
}

/// **Inertia × phase between-level perturbation cascade** — the meta-hop model
/// refined with the two things a chained product misses: a **clock** and a
/// **sign**.
///
/// Each HHTL tier is one meta-hop (tier `i` modifies tier `i+1`). Two quantities
/// propagate between levels, on the workspace's two algebras
/// (cf. the OGAR bipolar-phase pyramid: *sign side = multiply/XOR, magnitude
/// side = bundle/add*):
///
/// - **Phase** (sign, ±1): composes multiplicatively — `phase_{i+1} =
///   phase_i · phase_seed[i]` (a sign multiply = XOR of sign bits). This is the
///   between-level *interference* channel.
/// - **Magnitude**: the pass-through gain `gᵢ = infightᵢ·(1−raumgewinnᵢ)` (the
///   plain [`meta_cascade`] law; raw `raumgewinn`/`infight` min-max normalized).
///
/// The realized perturbation at tier `i` is the **bundle** (running sum) of the
/// signed contributions `field_k = Σ_{i≤k} phaseᵢ·magnitudeᵢ` — so when the
/// per-tier phases alternate the field cancels and the cascade dies in the upper
/// tiers; when they align it reinforces and reaches the leaves. **Inertia** sets
/// the per-hop clock `dtᵢ` (via [`per_hop_time`] with `inertia_h[i]`), so the
/// cumulative time at the penetration depth is the event's wall-clock — the 27 s
/// tell falls out of low inertia (short `dt`) over a few deep hops.
///
/// Returns `(per-tier MetaHop trace, penetration_depth)` where
/// `penetration_depth` = the number of tiers the **arriving** contribution
/// `|signed_ampᵢ|` stays `≥ threshold` — the front reach. It is **gain-driven and
/// hence phase-independent** (`|±x| = x`): phase governs the `field` interference
/// (constructive grows, alternating cancels — read it off the `MetaHop.field`
/// column), magnitude/gain governs how deep the front propagates. Counting depth
/// from the cumulative `field` would overstate the reach once a tier absorbs the
/// front (Codex #509 P2). CONJECTURE [H]: a modeling refinement of `meta_cascade`
/// (it adds the clock + the interference field); needs a probe against an observed
/// multi-tier cascade before promotion to FINDING.
#[allow(clippy::too_many_arguments)]
pub fn meta_cascade_phase(
    raumgewinn: &[f64],
    infight: &[f64],
    phase_seed: &[i8],
    inertia_h: &[f64],
    delta_p_fraction: f64,
    relay_s: f64,
    df_band: f64,
    threshold: f64,
) -> (Vec<MetaHop>, usize) {
    let n = raumgewinn.len().min(infight.len());
    if n == 0 {
        return (Vec::new(), 0);
    }
    let norm = |xs: &[f64]| -> Vec<f64> {
        let lo = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if (hi - lo).abs() < 1e-30 {
            vec![0.5; xs.len()]
        } else {
            xs.iter().map(|&x| (x - lo) / (hi - lo)).collect()
        }
    };
    let rn = norm(&raumgewinn[..n]);
    let inn = norm(&infight[..n]);

    let mut hops = Vec::with_capacity(n);
    let mut signed_amp = 1.0_f64; // seed: phase +, unit magnitude
    let mut field = 0.0_f64; // running interference bundle
    let mut t = 0.0_f64; // cumulative wall-clock
    let mut depth = 0usize;

    for i in 0..n {
        // Inertia sets this hop's clock; higher H ⇒ slower descent.
        let h = inertia_h.get(i).copied().unwrap_or(1.0);
        let dt = per_hop_time(relay_s, h, delta_p_fraction, df_band);
        t += dt;

        // Deposit the signed contribution into the interference field (bundle).
        field += signed_amp;
        hops.push(MetaHop {
            tier: i,
            signed_amp,
            field,
            dt,
            t,
        });
        // Penetration depth follows the ARRIVING contribution `|signed_amp|` (the
        // front), NOT the cumulative `field`. Once a tier absorbs (g→0) the front
        // dies and no deeper tier is reached, even though the bundled `field`
        // retains the earlier seed — counting from `field` would overstate the
        // reach (Codex #509 P2). The front magnitude is phase-independent (|±x|=x),
        // so phase governs the `field` interference, not the depth.
        if signed_amp.abs() >= threshold {
            depth += 1;
        }

        // Propagate to the next tier: magnitude × gain, phase × seed sign.
        let g = (inn[i] * (1.0 - rn[i])).clamp(0.0, 1.0);
        let phase = if phase_seed.get(i).copied().unwrap_or(1) < 0 {
            -1.0
        } else {
            1.0
        };
        signed_amp = signed_amp * g * phase;
    }
    (hops, depth)
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

    // Cross-tier gradients: min-max normalization is *relative across tiers*,
    // so a constant vector collapses to 0.5 everywhere (meaningless). These
    // probes use opposed gradients — high infight where raumgewinn is low —
    // to drive the gain to ≈1 in the upper tiers and ≈0 at the absorbing tier.
    const PASS_RAUM: [f64; 4] = [0.0, 0.0, 0.0, 1.0]; // → rn = [0,0,0,1]
    const PASS_FIGHT: [f64; 4] = [1.0, 1.0, 1.0, 0.0]; // → inn = [1,1,1,0]

    #[test]
    fn meta_cascade_penetrates_when_field_passes_through() {
        // gains g = inn·(1−rn) = [1,1,1,0] ⇒ amplitude survives the first three
        // tiers, absorbed at the well-connected leaf.
        let (amps, hops) = meta_cascade(&PASS_RAUM, &PASS_FIGHT, 0.5);
        assert_eq!(hops, 4, "amplitude=1 holds through three pass-tiers + seed");
        assert_eq!(amps.len(), 5, "seed + one per tier");
        // Inverse: a well-connected first tier (high raumgewinn, low infight)
        // absorbs the seed immediately.
        let (_, shallow) = meta_cascade(&[1.0, 1.0, 1.0, 0.0], &[0.0, 0.0, 0.0, 1.0], 0.5);
        assert_eq!(shallow, 1, "first tier holds the seed, then it dies");
    }

    #[test]
    fn phase_governs_the_field_not_the_front_depth() {
        // Same magnitudes (gains [1,1,1,0]); only the phase pattern differs. The
        // arriving front |signed_amp| is identical (phase = ±1 ⇒ |·| unchanged), so
        // penetration depth is EQUAL — phase does NOT change the front reach. What
        // phase DOES change is the bundled `field`: aligned phases reinforce
        // (1→2→3→4), alternating phases cancel (1→2→1→0). (Post Codex #509 P2.)
        let h = [3.0, 3.0, 3.0, 3.0];
        let (con, depth_con) = meta_cascade_phase(
            &PASS_RAUM,
            &PASS_FIGHT,
            &[1, 1, 1, 1],
            &h,
            0.1,
            0.2,
            0.2,
            0.5,
        );
        let (alt, depth_alt) = meta_cascade_phase(
            &PASS_RAUM,
            &PASS_FIGHT,
            &[1, -1, 1, -1],
            &h,
            0.1,
            0.2,
            0.2,
            0.5,
        );
        assert_eq!(
            depth_con, depth_alt,
            "front reach is phase-independent: {depth_con} == {depth_alt}"
        );
        let con_max = con.iter().map(|hp| hp.field.abs()).fold(0.0, f64::max);
        let alt_max = alt.iter().map(|hp| hp.field.abs()).fold(0.0, f64::max);
        assert!(
            alt_max < con_max,
            "alternating phase self-arrests the FIELD: peak {alt_max} < {con_max}"
        );
    }

    #[test]
    fn inertia_sets_the_meta_hop_clock() {
        // Lower inertia ⇒ shorter cumulative wall-clock over the same tiers.
        let ph = [1, 1, 1, 1];
        let (slow, _) =
            meta_cascade_phase(&PASS_RAUM, &PASS_FIGHT, &ph, &[6.0; 4], 0.1, 0.2, 0.2, 0.5);
        let (fast, _) =
            meta_cascade_phase(&PASS_RAUM, &PASS_FIGHT, &ph, &[2.0; 4], 0.1, 0.2, 0.2, 0.5);
        assert!(
            fast.last().unwrap().t < slow.last().unwrap().t,
            "low inertia = faster total descent"
        );
        // Times accumulate monotonically.
        assert!(slow.windows(2).all(|w| w[1].t > w[0].t));
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
