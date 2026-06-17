//! The 4-D rolling floor — L1..L4 HHTL tiers as an HDR popcount-stacking,
//! early-exit, Belichtungsmesser cascade with preheated confidence-interval
//! thresholds.
//!
//! This is the `perturbation-sim` mirror of `ndarray::hpc::cascade::Cascade`
//! (the "Belichtungsmesser"): a stateful exposure meter that calibrates a
//! `mu + k·sigma` rejection floor (Welford online), sorts each reading into
//! quality **bands**, and **recalibrates** (rolls) the floor on drift. There the
//! metered quantity is Hamming distance; here it is the per-tier **mode-
//! instability modifier** (below). The mapping the operator asked for:
//!
//! | HDR cascade (ndarray) | Rolling floor (here) |
//! |---|---|
//! | resolution strokes (coarse popcount → fine) | the 4 HHTL tiers L1..L4 = Bardioc Mode 1 Global → Mode 4 (eigenmode multi-scale zoom) |
//! | partial-popcount **stacking** with early reject | per-tier intensity **stacking** (coarse→fine), early-exit at the first Alarm |
//! | `mu + 3σ` calibrated threshold | per-tier `mu + k·σ` **confidence-interval** floor |
//! | `expose()` → Foveal/Near/Good/Weak/Reject | `band()` → Stable/Watch/Concern/Warning/**Alarm** |
//! | `observe()`/`recalibrate()` on drift | the floor **rolls** as tiers stream; coarse tiers **preheat** the fine floors |
//!
//! The "fluid-dynamics perturbation theory" framing (Bardioc ZPN pillars 03/04):
//! perturbation theory is the *multi-scale zoom* (the tiers), the flow front is
//! the stacked intensity rolling down the tiers; the floor is the early-warning
//! trip wire ("monitor mode stability for early warning").
//!
//! Honest scope: the σ here is from a small, weakly-dependent tier sample, so the
//! nominal CI is approximate — significance is the Jirak `n^(p/2−1)` rate, not a
//! clean Gaussian tail. The floor is an operating threshold, not a proof.

/// Early-warning band — the Belichtungsmesser exposure classes, renamed for the
/// grid-stability meter. `Alarm` is the `Reject`-equivalent: the reading has
/// crossed the floor and the cascade early-exits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloorBand {
    /// `< t/4` — deep inside the safe envelope.
    Stable,
    /// `< t/2`.
    Watch,
    /// `< 3t/4`.
    Concern,
    /// `≤ t` — at the confidence-interval edge.
    Warning,
    /// `> t` — crossed the floor; early-exit trigger.
    Alarm,
}

/// The **mode-instability modifier**: the Weyl eigenvalue perturbation amplified
/// by inverse Fiedler connectivity — `Δλ × (1 / λ₂)`. Dimensionless. High when a
/// perturbation shifts the spectrum a lot *relative to* an already weakly-
/// connected mode (small `λ₂` = near-disconnected regional mode = unstable). This
/// is the per-tier early-warning signal the Bardioc "monitor mode stability"
/// slide calls for; `1/λ₂` is the regional-mode susceptibility (Mode 2 = Fiedler).
pub fn weyl_over_fiedler(weyl_dlambda: f64, fiedler_lambda2: f64) -> f64 {
    // `1/λ₂` is the regional-mode susceptibility, and `λ₂ → 0` is the blackout
    // precursor — the same NaN/divergence landmine as the Davis–Kahan `/ gap`. We
    // floor the Fiedler denominator (absolute floor: λ₂ is itself an eigenvalue
    // near 0) and surface `λ₂ ≤ floor` as the FRAGMENTATION_SENTINEL divergence
    // (an unbounded instability ratio — the near-disconnected mode is maximally
    // susceptible), never a `NaN`. `weyl_dlambda.max(0.0)` keeps the numerator
    // non-negative so a sign-noisy Δλ cannot flip the ratio negative.
    if fiedler_lambda2.abs() < crate::perturbation::SPECTRAL_GAP_FLOOR {
        crate::perturbation::FRAGMENTATION_SENTINEL
    } else {
        weyl_dlambda.max(0.0) / fiedler_lambda2.abs()
    }
}

/// One tier's rolling floor: a Welford online `mu`/`sigma` and a `mu + k·σ`
/// confidence-interval threshold. Mirrors `ndarray::hpc::cascade::Cascade`'s
/// `mu`/`sigma`/`threshold` + `observe`/`recalibrate`, generalized to an `f64`
/// intensity.
#[derive(Debug, Clone, Copy)]
pub struct RollingFloor {
    mu: f64,
    sigma: f64,
    /// σ-multiplier for the CI floor (2.0 ≈ 97.7% one-sided Gaussian; see scope).
    pub k: f64,
    n: usize,
}

impl RollingFloor {
    /// New empty floor with σ-multiplier `k`.
    pub fn new(k: f64) -> Self {
        Self {
            mu: 0.0,
            sigma: 0.0,
            k,
            n: 0,
        }
    }

    /// Calibrate (mean/σ/threshold) from a batch — the camera's initial metering.
    /// This is also how a coarse tier **preheats** the next floor (warm start).
    pub fn preheat(&mut self, samples: &[f64]) {
        let n = samples.len();
        if n == 0 {
            return;
        }
        let mu = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / n as f64;
        self.mu = mu;
        self.sigma = var.sqrt();
        self.n = n;
    }

    /// Welford online update; returns `true` if `x` crossed the current floor
    /// (i.e. `band(x) == Alarm`). Updates the floor AFTER testing, so the test
    /// uses the floor as it stood — the rolling property.
    pub fn observe(&mut self, x: f64) -> bool {
        let crossed = self.band(x) == FloorBand::Alarm;
        self.n += 1;
        if self.n == 1 {
            self.mu = x;
            self.sigma = 0.0;
            return crossed;
        }
        let delta = x - self.mu;
        self.mu += delta / self.n as f64;
        let delta2 = x - self.mu;
        let m2 = self.sigma * self.sigma * (self.n - 1) as f64 + delta * delta2;
        self.sigma = (m2 / self.n as f64).sqrt();
        crossed
    }

    /// The confidence-interval floor `mu + k·σ`.
    pub fn threshold(&self) -> f64 {
        self.mu + self.k * self.sigma
    }

    /// Standardized exceedance `(x − mu)/σ` — how many σ above the mean (the
    /// Jirak-honest "noise-floor units"; significance via `n^(p/2−1)`, not IID).
    pub fn z(&self, x: f64) -> f64 {
        if self.sigma < 1e-12 {
            0.0
        } else {
            (x - self.mu) / self.sigma
        }
    }

    /// Belichtungsmesser band by quarters of the threshold (mirrors `Cascade::expose`).
    pub fn band(&self, x: f64) -> FloorBand {
        let t = self.threshold();
        if t <= 0.0 {
            // Un-calibrated: anything strictly positive is already an alarm.
            return if x > 0.0 {
                FloorBand::Alarm
            } else {
                FloorBand::Stable
            };
        }
        if x <= t / 4.0 {
            FloorBand::Stable
        } else if x <= t / 2.0 {
            FloorBand::Watch
        } else if x <= t * 3.0 / 4.0 {
            FloorBand::Concern
        } else if x <= t {
            FloorBand::Warning
        } else {
            FloorBand::Alarm
        }
    }

    /// Current mean / σ / observation count.
    pub fn mu(&self) -> f64 {
        self.mu
    }
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
    pub fn observations(&self) -> usize {
        self.n
    }
}

/// Outcome of a coarse→fine stacked, early-exit pass over the four tiers.
#[derive(Debug, Clone)]
pub struct StackResult {
    /// Tier index where the cascade exited (0..=3); `3` = ran to the leaf.
    pub exit_tier: usize,
    /// Cumulative stacked intensity at the exit tier.
    pub stacked: f64,
    /// Band of the stacked value at the exit tier.
    pub band: FloorBand,
    /// Whether the exit was EARLY (crossed the floor before the leaf tier).
    pub early: bool,
    /// Per-tier standardized exceedance up to the exit.
    pub z: Vec<f64>,
}

/// The 4-D rolling floor over L1..L4 — one [`RollingFloor`] per eigenmode tier.
#[derive(Debug, Clone)]
pub struct TierFloors {
    /// HEEL(L1) → HIP(L2) → TWIG(L3) → LEAF(L4) floors.
    pub floors: [RollingFloor; 4],
}

impl TierFloors {
    /// Four empty floors sharing σ-multiplier `k`.
    pub fn new(k: f64) -> Self {
        Self {
            floors: [RollingFloor::new(k); 4],
        }
    }

    /// Preheat each tier's floor from a per-tier calibration sample (e.g. the
    /// per-basin modifier distribution at that tier). `samples[t]` calibrates
    /// tier `t`; a coarse tier with no sample inherits nothing (stays cold).
    pub fn preheat(&mut self, samples: &[Vec<f64>]) {
        for (t, floor) in self.floors.iter_mut().enumerate() {
            if let Some(s) = samples.get(t) {
                floor.preheat(s);
            }
        }
    }

    /// Coarse→fine **popcount-stacking** with early-exit. The per-tier
    /// `intensity` (e.g. [`weyl_over_fiedler`] per tier) is accumulated; at each
    /// tier the *stacked* value is metered against that tier's rolling floor.
    /// The pass **exits at the first tier whose stacked band is `Alarm`** — the
    /// decision is confident, the finer tiers need not be computed (the HDR
    /// early-reject). Coarser tiers also **preheat** the next floor when it is
    /// still cold (zero threshold), so the warm start rolls down the tiers.
    /// Floors are updated (rolled) as the stack passes through.
    pub fn stack_early_exit(&mut self, intensity: [f64; 4]) -> StackResult {
        let mut stacked = 0.0;
        let mut z = Vec::with_capacity(4);
        for (t, &inc) in intensity.iter().enumerate() {
            stacked += inc;
            // Preheat a cold finer floor from the coarser floor we just used.
            if t + 1 < 4 && self.floors[t + 1].threshold() <= 0.0 {
                let warm = self.floors[t];
                self.floors[t + 1].mu = warm.mu;
                self.floors[t + 1].sigma = warm.sigma;
                self.floors[t + 1].n = warm.n.max(1);
            }
            let band = self.floors[t].band(stacked);
            z.push(self.floors[t].z(stacked));
            let crossed = self.floors[t].observe(stacked);
            if crossed || band == FloorBand::Alarm {
                return StackResult {
                    exit_tier: t,
                    stacked,
                    band: FloorBand::Alarm,
                    early: t < 3,
                    z,
                };
            }
        }
        StackResult {
            exit_tier: 3,
            stacked,
            band: self.floors[3].band(stacked),
            early: false,
            z,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weyl_over_fiedler_amplifies_weak_connectivity() {
        // Same perturbation, weaker mode (smaller λ₂) ⇒ larger instability ratio.
        let strong = weyl_over_fiedler(1e-6, 1e-2);
        let weak = weyl_over_fiedler(1e-6, 1e-4);
        assert!(
            weak > strong,
            "1/Fiedler amplifies the weakly-connected mode"
        );
        assert!(
            weyl_over_fiedler(1.0, 0.0).is_infinite(),
            "λ₂→0 ⇒ unbounded"
        );
    }

    /// B1 finiteness gate: the `1/λ₂` instability ratio is NaN-free across the
    /// blackout-precursor regime (λ₂ → 0, exactly 0, tiny-negative noise), and a
    /// degenerate λ₂ surfaces the divergence sentinel — never a noisy finite or a
    /// NaN. Normal-regime values are unchanged (parity).
    #[test]
    fn weyl_over_fiedler_is_nan_free_at_the_precursor() {
        for &l2 in &[0.0, 1e-14, -1e-14, 1e-300, 1e-9, 1e-3, 1.0, 5.0] {
            let r = weyl_over_fiedler(1e-6, l2);
            assert!(!r.is_nan(), "weyl_over_fiedler NaN at λ₂={l2}");
            assert!(
                r >= 0.0,
                "instability ratio must be non-negative at λ₂={l2}"
            );
        }
        // Below the spectral floor ⇒ divergence sentinel (fragmenting mode).
        assert!(weyl_over_fiedler(1e-6, 1e-14).is_infinite());
        // Parity: a healthy λ₂ gives the plain, unchanged ratio.
        assert!((weyl_over_fiedler(2.0, 4.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn floor_bands_partition_by_quarters() {
        let mut f = RollingFloor::new(2.0);
        f.preheat(&[1.0, 1.0, 1.0, 1.0, 2.0, 0.0]); // mu=1, σ≈0.577 → t≈2.15
        let t = f.threshold();
        assert!(f.band(t * 0.1) == FloorBand::Stable);
        assert!(f.band(t * 0.9) == FloorBand::Warning);
        assert!(f.band(t * 1.5) == FloorBand::Alarm);
    }

    #[test]
    fn preheat_then_observe_rolls_the_floor() {
        let mut f = RollingFloor::new(2.0);
        f.preheat(&[1.0; 8]); // mu=1, σ=0
        let t0 = f.threshold();
        // A run of larger readings lifts the floor (it rolls up).
        for _ in 0..20 {
            f.observe(3.0);
        }
        assert!(f.threshold() > t0, "floor rolls up under sustained load");
    }

    #[test]
    fn early_exit_when_coarse_tier_already_alarms() {
        let mut tf = TierFloors::new(2.0);
        // Preheat only L1 with a calm baseline; a huge L1 reading must alarm at
        // tier 0 without consulting the finer tiers.
        tf.preheat(&[vec![0.1, 0.1, 0.1, 0.1]]);
        let r = tf.stack_early_exit([10.0, 0.0, 0.0, 0.0]);
        assert_eq!(r.exit_tier, 0, "coarse alarm exits at L1");
        assert!(r.early, "exit before the leaf");
        assert_eq!(r.band, FloorBand::Alarm);
    }

    #[test]
    fn calm_stack_runs_to_the_leaf() {
        let mut tf = TierFloors::new(3.0);
        tf.preheat(&[
            vec![1.0, 1.0, 1.2, 0.8],
            vec![1.0, 1.0, 1.2, 0.8],
            vec![1.0, 1.0, 1.2, 0.8],
            vec![1.0, 1.0, 1.2, 0.8],
        ]);
        // Small per-tier increments: stacked stays under each rolling floor.
        let r = tf.stack_early_exit([0.2, 0.2, 0.2, 0.2]);
        assert_eq!(r.exit_tier, 3, "calm stack reaches the leaf");
        assert!(!r.early);
        assert_eq!(r.z.len(), 4, "a z per tier visited");
    }
}
