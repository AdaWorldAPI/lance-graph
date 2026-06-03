//! Rolling-floor 256-bucket quantiser — Stage 4 tail of the helix codec pipeline.
//!
//! # Responsibilities
//!
//! [`RollingFloor`] maps an `f64` value to a `u8` bucket in `0..=255` over a
//! live `[lo, hi]` window that can slide (roll) when the observed distribution
//! drifts too far from uniform.  The 256 buckets are simultaneously their own
//! monitoring instrument: the `occupancy` array is the empirical distribution
//! estimate.  No separate histogram is needed.
//!
//! # Quantisation cost (honest accounting)
//!
//! This is **NOT lossless**.  Each bucket covers `(hi − lo) / 256` of the span.
//! Uniform error is ± ½ bucket = ± 0.195% of span in the informative range.
//! The outermost `1/256` = 0.390625% of span is wider than the ±3σ tail
//! (0.27%), so values outside `[lo, hi]` saturate into the two rim buckets
//! (0 and 255).  This is controlled saturation, not information loss — the
//! floor is calibrated so that the tail occupancy stays small.
//!
//! # Versioning contract
//!
//! "Same value → same `u8`" holds **only within a stable floor version**.
//! Every [`RollingFloor::roll`] call that actually moves the bounds bumps the
//! wrapping [`RollingFloor::version`] counter.  Callers that need consistent
//! mapping (e.g. distance LUTs) must embed the version stamp alongside the
//! quantised byte and invalidate cached LUTs on version change.
//!
//! # Compute / calibration split (workspace P0)
//!
//! | Method | Receiver | Why |
//! |---|---|---|
//! | [`quantize`](RollingFloor::quantize) | `&self` | **COMPUTE** — pure read, called in hot paths; no state mutation. |
//! | [`drift_score`](RollingFloor::drift_score) | `&self` | **COMPUTE** — pure read, measures drift without mutating. |
//! | [`observe`](RollingFloor::observe) | `&mut self` | **CALIBRATION** — accumulates occupancy. |
//! | [`roll`](RollingFloor::roll) | `&mut self` | **CALIBRATION/BUILDER** — adapts bounds, resets counters, bumps version. |
//!
//! This mirrors the workspace-wide rule from `data-flow.md`: engines never
//! `&mut self` while computing; mutation is gated to explicit calibration paths.

use crate::constants::PALETTE_SIZE;

/// A rolling-floor 256-bucket quantiser with occupancy-drift detection.
///
/// Maintains a live `[lo, hi]` window, an occupancy array (one `u32` per
/// bucket), a sample counter, and a `version` stamp that increments on every
/// successful [`roll`](RollingFloor::roll).
///
/// See the [module-level documentation](self) for the full contract.
#[derive(Debug, Clone)]
pub struct RollingFloor {
    lo: f64,
    hi: f64,
    occupancy: [u32; PALETTE_SIZE],
    samples: u64,
    version: u8,
    /// Drift threshold in multinomial-SD units.  A `drift_score()` above this
    /// value causes [`roll`](RollingFloor::roll) to move the bounds.
    drift_sigma: f64,
    /// Glide rate α ∈ (0, 1].  The new bound is `(1 − α)·old + α·estimated`.
    /// Smaller values glide more slowly, reducing churn on noisy signals.
    inertia: f64,
}

impl RollingFloor {
    /// Construct a uniform floor over `[lo, hi]` with default parameters:
    /// `drift_sigma = 3.0`, `inertia = 0.1`, `version = 0`, empty occupancy.
    ///
    /// # Panics
    ///
    /// Does not panic; degenerate bounds (`hi <= lo`) are handled gracefully
    /// by [`quantize`](Self::quantize) (returns 0).
    pub fn uniform(lo: f64, hi: f64) -> Self {
        Self::with_params(lo, hi, 3.0, 0.1)
    }

    /// Construct a floor with explicit drift / inertia parameters.
    ///
    /// - `drift_sigma` — k-SD threshold; typical range 2.0–4.0.
    /// - `inertia` — α glide rate; typical range 0.05–0.3.
    pub fn with_params(lo: f64, hi: f64, drift_sigma: f64, inertia: f64) -> Self {
        Self {
            lo,
            hi,
            occupancy: [0u32; PALETTE_SIZE],
            samples: 0,
            version: 0,
            drift_sigma,
            inertia,
        }
    }

    /// **COMPUTE — `&self`.**  Map `value` to a bucket in `0..=255`.
    ///
    /// - Returns `0` if `hi <= lo` (degenerate bounds guard).
    /// - Saturates into bucket 0 for `value <= lo`.
    /// - Saturates into bucket 255 for `value >= hi`.
    /// - Interior values: `idx = floor(t × 256)` where `t = (value − lo) / (hi − lo)`,
    ///   clamped to `[0, 255]`.
    pub fn quantize(&self, value: f64) -> u8 {
        if self.hi <= self.lo {
            return 0;
        }
        let t = (value - self.lo) / (self.hi - self.lo);
        let idx = (t * 256.0).floor();
        // Clamp to [0.0, 255.0] then cast — no negative or out-of-range values.
        let idx = idx.clamp(0.0, 255.0);
        idx as u8
    }

    /// **CALIBRATION — `&mut self`.**  Record one observation.
    ///
    /// Increments `occupancy[quantize(value)]` and `samples`.
    pub fn observe(&mut self, value: f64) {
        let b = self.quantize(value);
        self.occupancy[b as usize] = self.occupancy[b as usize].saturating_add(1);
        self.samples = self.samples.saturating_add(1);
    }

    /// **COMPUTE — `&self`.**  Drift score: maximum per-bucket deviation from the
    /// expected uniform occupancy, in multinomial-SD units.
    ///
    /// Formula:
    /// ```text
    /// p   = 1 / 256
    /// mu  = samples × p
    /// sd  = sqrt(samples × p × (1 − p))
    /// score = max over i of |occupancy[i] − mu| / sd
    /// ```
    ///
    /// Returns `0.0` when `samples == 0` or `sd == 0`.
    pub fn drift_score(&self) -> f64 {
        if self.samples == 0 {
            return 0.0;
        }
        let p = 1.0 / PALETTE_SIZE as f64; // 1/256
        let n = self.samples as f64;
        let mu = n * p;
        let sd = (n * p * (1.0 - p)).sqrt();
        if sd == 0.0 {
            return 0.0;
        }
        self.occupancy
            .iter()
            .map(|&occ| (occ as f64 - mu).abs() / sd)
            .fold(0.0_f64, f64::max)
    }

    /// **CALIBRATION — `&mut self`.**  Glide the floor toward the empirical
    /// distribution when drift exceeds the threshold.
    ///
    /// Returns `true` and updates state when `drift_score() > drift_sigma`:
    ///
    /// 1. Walk cumulative occupancy to find the bucket where the cumulative sum
    ///    first reaches `0.004 × samples` → `est_lo` bucket.
    /// 2. Walk cumulative occupancy to find the bucket where the cumulative sum
    ///    first reaches `0.996 × samples` → `est_hi` bucket.
    /// 3. Convert bucket indices to values: `lo + (b / 256) × (hi − lo)`.
    /// 4. Glide: `new_lo = (1 − α)·lo + α·est_lo_val`;
    ///    `new_hi = (1 − α)·hi + α·est_hi_val`.
    /// 5. If `new_hi <= new_lo`, leave bounds unchanged (avoids degenerate state).
    /// 6. Reset `occupancy` to all-zeros, `samples` to 0, bump `version`
    ///    (wrapping), and return `true`.
    ///
    /// Returns `false` (no state change) when `drift_score() <= drift_sigma`.
    pub fn roll(&mut self) -> bool {
        if self.drift_score() <= self.drift_sigma {
            return false;
        }

        let lo_target = (0.004 * self.samples as f64).ceil() as u64;
        let hi_target = (0.996 * self.samples as f64).floor() as u64;

        // Walk cumulative to find est_lo bucket.
        let mut cum: u64 = 0;
        let mut est_lo_bucket: usize = 0;
        for (i, &occ) in self.occupancy.iter().enumerate() {
            cum += occ as u64;
            if cum >= lo_target {
                est_lo_bucket = i;
                break;
            }
        }

        // Walk cumulative to find est_hi bucket.
        cum = 0;
        let mut est_hi_bucket: usize = PALETTE_SIZE - 1;
        for (i, &occ) in self.occupancy.iter().enumerate() {
            cum += occ as u64;
            if cum >= hi_target {
                est_hi_bucket = i;
                break;
            }
        }

        // Convert bucket indices to float values.
        let span = self.hi - self.lo;
        let est_lo_val = self.lo + (est_lo_bucket as f64 / PALETTE_SIZE as f64) * span;
        let est_hi_val = self.lo + (est_hi_bucket as f64 / PALETTE_SIZE as f64) * span;

        // Glide bounds.
        let alpha = self.inertia;
        let new_lo = (1.0 - alpha) * self.lo + alpha * est_lo_val;
        let new_hi = (1.0 - alpha) * self.hi + alpha * est_hi_val;

        // Guard: do not allow degenerate bounds.
        if new_hi <= new_lo {
            return false;
        }

        self.lo = new_lo;
        self.hi = new_hi;
        self.occupancy = [0u32; PALETTE_SIZE];
        self.samples = 0;
        self.version = self.version.wrapping_add(1);
        true
    }

    /// The floor-version stamp.  Wraps at 255 → 0.  Callers should treat a
    /// changed version as a signal to invalidate any cached distance LUTs built
    /// from [`bucket_center`](Self::bucket_center).
    pub fn version(&self) -> u8 {
        self.version
    }

    /// Per-bucket occupancy counts (length 256).
    pub fn occupancy(&self) -> &[u32; PALETTE_SIZE] {
        &self.occupancy
    }

    /// Total number of observations since the last [`roll`](Self::roll)
    /// (or since construction).
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Current `(lo, hi)` bounds.
    pub fn bounds(&self) -> (f64, f64) {
        (self.lo, self.hi)
    }

    /// **COMPUTE — `&self`.**  The representative center value of bucket `b`.
    ///
    /// Formula: `lo + ((b + 0.5) / 256) × (hi − lo)`.
    ///
    /// Used to build distance LUTs from the current floor without decompressing
    /// individual observations.  All returned values lie strictly inside
    /// `(lo, hi)` and are monotonically increasing in `b`.
    pub fn bucket_center(&self, b: u8) -> f64 {
        self.lo + ((b as f64 + 0.5) / PALETTE_SIZE as f64) * (self.hi - self.lo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn floor_0_100() -> RollingFloor {
        RollingFloor::uniform(0.0, 100.0)
    }

    // ── uniform-bounds quantization ──────────────────────────────────────────

    #[test]
    fn quantize_lo_maps_to_zero() {
        let f = floor_0_100();
        assert_eq!(f.quantize(0.0), 0);
    }

    #[test]
    fn quantize_just_below_hi_maps_to_255() {
        let f = floor_0_100();
        // hi - tiny should land in the top bucket
        assert_eq!(f.quantize(99.999), 255);
    }

    #[test]
    fn quantize_midpoint_is_near_128() {
        let f = floor_0_100();
        let mid = f.quantize(50.0);
        // 50.0 / 100.0 * 256 = 128.0 → floor → 128; allow ±1 for float rounding
        assert!((127..=129).contains(&mid), "midpoint bucket was {mid}");
    }

    // ── saturation ────────────────────────────────────────────────────────────

    #[test]
    fn saturation_below_lo() {
        let f = floor_0_100();
        assert_eq!(f.quantize(-100.0), 0);
    }

    #[test]
    fn saturation_above_hi() {
        let f = floor_0_100();
        assert_eq!(f.quantize(200.0), 255);
    }

    // ── determinism within a version ─────────────────────────────────────────

    #[test]
    fn same_value_same_bucket_repeated() {
        let f = floor_0_100();
        let v = 42.7;
        let b0 = f.quantize(v);
        for _ in 0..1000 {
            assert_eq!(f.quantize(v), b0);
        }
    }

    // ── observe increments occupancy + samples ────────────────────────────────

    #[test]
    fn observe_increments_occupancy_and_samples() {
        let mut f = floor_0_100();
        assert_eq!(f.samples(), 0);
        f.observe(50.0);
        assert_eq!(f.samples(), 1);
        let bucket = RollingFloor::uniform(0.0, 100.0).quantize(50.0) as usize;
        assert_eq!(f.occupancy()[bucket], 1);
        f.observe(50.0);
        assert_eq!(f.samples(), 2);
        assert_eq!(f.occupancy()[bucket], 2);
    }

    // ── no-drift case ─────────────────────────────────────────────────────────

    #[test]
    fn no_drift_uniform_observations_no_roll() {
        let mut f = floor_0_100();
        // Observe 2560 values spread uniformly across [0, 100) — 10 per bucket.
        for i in 0..2560u32 {
            let v = (i as f64 / 2560.0) * 100.0;
            f.observe(v);
        }
        let version_before = f.version();
        let rolled = f.roll();
        assert!(!rolled, "uniform observations should not trigger roll");
        assert_eq!(f.version(), version_before);
    }

    // ── drift case ────────────────────────────────────────────────────────────

    #[test]
    fn drift_concentrated_observations_triggers_roll() {
        let mut f = floor_0_100();
        // Observe 10000 values all at 50.0 — extreme concentration.
        for _ in 0..10_000 {
            f.observe(50.0);
        }
        let score = f.drift_score();
        // With 10000 samples all in one bucket, drift_score should be enormous.
        assert!(
            score > f.drift_sigma,
            "expected large drift score, got {score}"
        );
        let rolled = f.roll();
        assert!(rolled, "concentrated observations should trigger roll");
        assert_eq!(f.version(), 1);
        // Occupancy and samples must be reset.
        assert_eq!(f.samples(), 0);
        for &occ in f.occupancy().iter() {
            assert_eq!(occ, 0);
        }
    }

    // ── bucket_center ─────────────────────────────────────────────────────────

    #[test]
    fn bucket_center_within_bounds() {
        let f = floor_0_100();
        for b in 0u8..=255 {
            let c = f.bucket_center(b);
            assert!(
                c > 0.0 && c < 100.0,
                "bucket_center({b}) = {c} should be in (0, 100)"
            );
        }
    }

    #[test]
    fn bucket_center_monotonic() {
        let f = floor_0_100();
        let mut prev = f.bucket_center(0);
        for b in 1u8..=255 {
            let cur = f.bucket_center(b);
            assert!(
                cur > prev,
                "bucket_center should be monotonic: center({b}) = {cur} <= center({}) = {prev}",
                b - 1
            );
            prev = cur;
        }
    }

    // ── degenerate-bounds guard ───────────────────────────────────────────────

    #[test]
    fn degenerate_bounds_quantize_returns_zero_no_panic() {
        let f = RollingFloor::uniform(5.0, 5.0); // hi == lo
        assert_eq!(f.quantize(5.0), 0);
        assert_eq!(f.quantize(0.0), 0);
        assert_eq!(f.quantize(100.0), 0);

        let f2 = RollingFloor::uniform(10.0, 1.0); // hi < lo
        assert_eq!(f2.quantize(5.0), 0);
    }

    // ── drift_score zero when no samples ─────────────────────────────────────

    #[test]
    fn drift_score_zero_with_no_samples() {
        let f = floor_0_100();
        assert_eq!(f.drift_score(), 0.0);
    }

    // ── version stamp ─────────────────────────────────────────────────────────

    #[test]
    fn version_starts_at_zero_and_wraps() {
        let mut f = RollingFloor::with_params(0.0, 100.0, 0.0001, 0.5);
        assert_eq!(f.version(), 0);
        // One concentrated batch → roll → version 1
        for _ in 0..1000 {
            f.observe(50.0);
        }
        let rolled = f.roll();
        assert!(rolled);
        assert_eq!(f.version(), 1);
    }

    #[test]
    fn version_wraps_at_255() {
        // Spread a drifting batch over a sub-band so the re-estimated bounds stay
        // valid (non-degenerate) and the roll actually commits + bumps version.
        // (A single concentrated value collapses the bounds and hits roll()'s
        // documented degenerate guard, which deliberately does NOT bump version.)
        let mut f = RollingFloor::with_params(0.0, 100.0, 3.0, 0.25);
        for k in 0..4000u32 {
            f.observe(20.0 + (k % 200) as f64 * 0.2); // values in [20, 60)
        }
        assert!(f.drift_score() > 3.0, "concentrated sub-band should drift");
        // Force the counter to its max; one committing roll must wrap 255 → 0.
        f.version = 255;
        assert!(f.roll(), "non-degenerate drift ⇒ roll commits");
        assert_eq!(f.version(), 0, "version should wrap 255 → 0");
    }

    // ── bounds accessor ───────────────────────────────────────────────────────

    #[test]
    fn bounds_returns_current_lo_hi() {
        let f = RollingFloor::uniform(-1.0, 1.0);
        assert_eq!(f.bounds(), (-1.0, 1.0));
    }
}
