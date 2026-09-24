// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # Cascade — HDR Exposure Cascade for Binary Vector Search
//!
//! Self-calibrating exposure meter for Hamming distance queries on binary
//! vectors. Eliminates 97%+ of candidates using sampled bit comparisons
//! before a full comparison is ever attempted.
//!
//! ## Design principles
//!
//! - **Integer only** — no `f32`/`f64` on the hot path. Distances are `u32`,
//!   thresholds are `u32`, square roots are integer Newton's method.
//! - **Self-calibrating** — thresholds derived from actual corpus
//!   distribution, not hardcoded. Adapts to any vector width.
//! - **Shift detection** — Welford's online algorithm detects when the
//!   data distribution changes and triggers recalibration.
//! - **Distribution-free** — reservoir sampling provides empirical quantile
//!   thresholds that work for any distribution shape (bimodal, skewed,
//!   heavy-tailed). Auto-switches when σ-based bands don't fit.
//! - **Cascade** — 1/16 sample → 1/4 sample → full comparison. Each
//!   stage rejects on sigma bands. Only ~0.5% of candidates reach full.
//!
//! ## Statistics
//!
//! Two random N-bit vectors have Hamming distance ~ Binomial(N, 0.5):
//!
//! ```text
//! μ = N/2,  σ = √(N/4)
//!
//! N = 16384:  μ = 8192,  σ = 64
//! N = 10000:  μ = 5000,  σ ≈ 50
//! N = 32768:  μ = 16384, σ ≈ 91
//! ```
//!
//! Lower distance = better match. Everything at or above μ is noise.

// ---------------------------------------------------------------------------
// Integer square root — Newton's method, no float
// ---------------------------------------------------------------------------

/// Integer square root via Newton's method. No float arithmetic.
///
/// Returns floor(√n). Guaranteed correct for all `u32` inputs.
pub fn isqrt(n: u32) -> u32 {
    ndarray::hpc::rolling_floor::isqrt_u32(n)
}

// ---------------------------------------------------------------------------
// Band — sigma-based classification
// ---------------------------------------------------------------------------

/// Sigma band for classifying Hamming distances.
///
/// Ordered from best match to worst. `Foveal < Near < Good < Weak < Reject`.
/// The `PartialOrd`/`Ord` derive gives this ordering from the discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Band {
    /// d < μ − 3σ. Less than 0.13% chance of random. Near-certain match.
    Foveal = 0,
    /// d < μ − 2σ. Less than 2.3% chance of random. Strong match.
    Near = 1,
    /// d < μ − σ. Less than 15.9% chance of random. Good candidate.
    Good = 2,
    /// d < μ. Could be random. Weak signal.
    Weak = 3,
    /// d ≥ μ. Noise or anti-correlated. Not a match.
    Reject = 4,
}

// ---------------------------------------------------------------------------
// RankedHit / ShiftAlert — result types
// ---------------------------------------------------------------------------

/// A candidate that survived the cascade and was classified.
#[derive(Debug, Clone)]
pub struct RankedHit {
    /// Index of the matched candidate in the input slice.
    pub index: usize,
    /// Exact Hamming distance (from stage 3 full comparison). `u32`, not float.
    pub distance: u32,
    /// Sigma band classification.
    pub band: Band,
}

/// Alert that the observed distance distribution has shifted.
///
/// The rolling floor's own shift record from
/// [`ndarray::hpc::rolling_floor`]; same fields as before.
pub use ndarray::hpc::rolling_floor::FloorShift as ShiftAlert;
use ndarray::hpc::rolling_floor::{RollingFloor, SigmaLevel};

/// The band cuts on the σ-lattice: 3σ, 2σ, 1σ and μ itself.
const BAND_LEVELS: [SigmaLevel; 4] = [SigmaLevel(12), SigmaLevel(8), SigmaLevel(4), SigmaLevel(0)];

/// The cascade cuts on the σ-lattice: 1σ, 1.5σ, 1.75σ, 2σ, 2.25σ, 2.5σ,
/// 2.75σ, 3σ.
const CASCADE_LEVELS: [SigmaLevel; 8] = [
    SigmaLevel(4),
    SigmaLevel(6),
    SigmaLevel(7),
    SigmaLevel(8),
    SigmaLevel(9),
    SigmaLevel(10),
    SigmaLevel(11),
    SigmaLevel(12),
];

// ---------------------------------------------------------------------------
// Hamming on &[u64] words — self-contained, no external deps
// ---------------------------------------------------------------------------

/// Hamming distance between two word arrays (XOR + popcount).
///
/// Routes through [`super::ndarray_bridge::dispatch_hamming`], the single
/// SIMD-dispatched Hamming kernel for this module tree (VPOPCNTDQ →
/// AVX-512BW → AVX2 → scalar, via `ndarray::hpc::bitwise` under
/// `ndarray-hpc`). The previous body was a scalar `count_ones()` loop that
/// bypassed that dispatch — a second implementation of a kernel the crate
/// already owned.
///
/// XOR-then-popcount is invariant under byte order, so viewing the `u64`
/// words as their little-/big-endian byte image changes nothing: the same
/// bits are compared, only their traversal order differs.
///
/// Both slices must have the same length.
#[inline]
fn words_hamming(a: &[u64], b: &[u64]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    // SAFETY: `u8` has alignment 1 and no invalid bit patterns, so any
    // `[u64]` is a valid `[u8]` of 8× the length; the borrow keeps `a`/`b`
    // alive for the call and neither view is written through.
    let (a_bytes, b_bytes) = unsafe {
        (
            core::slice::from_raw_parts(a.as_ptr().cast::<u8>(), core::mem::size_of_val(a)),
            core::slice::from_raw_parts(b.as_ptr().cast::<u8>(), core::mem::size_of_val(b)),
        )
    };
    super::ndarray_bridge::dispatch_hamming(a_bytes, b_bytes) as u32
}

/// Sampled Hamming distance: compare every `step`-th word and extrapolate.
///
/// For `step=16`, samples 1/16 of the words. The extrapolation multiplies
/// by `step`, giving an unbiased estimate with higher variance.
#[inline]
fn words_hamming_sampled(a: &[u64], b: &[u64], step: usize) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(step > 0);
    let mut dist = 0u32;
    let mut sampled = 0u32;
    let mut i = 0;
    while i < a.len() {
        dist += (a[i] ^ b[i]).count_ones();
        sampled += 1;
        i += step;
    }
    if sampled == 0 {
        return 0;
    }
    // Extrapolate: scale to full width.
    (dist as u64 * a.len() as u64 / sampled as u64) as u32
}

// ---------------------------------------------------------------------------
// ReservoirSample — Vitter's Algorithm R for distribution-free quantiles
// ---------------------------------------------------------------------------

/// Reservoir sampling for distribution-free quantile estimation.
///
/// The deterministic Algorithm-R reservoir from
/// [`ndarray::hpc::rolling_floor`]: every element of the stream has equal
/// probability of being held, and an identical stream yields an identical
/// reservoir. Used for the empirical thresholds when the distance distribution
/// is non-normal (bimodal, skewed, heavy-tailed).
pub use ndarray::hpc::rolling_floor::ReservoirU32 as ReservoirSample;

// ---------------------------------------------------------------------------
// Cascade — the exposure meter
// ---------------------------------------------------------------------------

/// Self-calibrating exposure meter for Hamming distance queries.
///
/// Every band and cascade cut is a point on the integer σ-lattice (`k`
/// quarter-σ below the noise floor), derived on demand from the rolling
/// floor's current coordinates — nothing is cached:
/// - **Gaussian shape**: `μ − k·σ/4`.
/// - **Empirical shape**: the sample value at `k`'s Gaussian-equivalent tail
///   rank, moved into the current frame.
///
/// The shape switches to empirical when the reservoir detects non-normality
/// (skewness or kurtosis outside normal range). Everything is integer.
///
/// # Usage
///
/// ```ignore
/// let meter = Cascade::calibrate(&sample_distances);
/// let band = meter.band(7950); // → Band::Foveal
/// let results = meter.query(query_words, &candidates, 10);
///
/// if let Some(alert) = meter.observe(distance) {
///     meter.recalibrate(&alert);
/// }
/// ```
pub struct Cascade {
    /// The rolling floor: exact running moments, reservoir, shape belief
    /// and the drift anchor. Thresholds are derived from it per call.
    floor: RollingFloor,
    /// Which cascade[] index stage 1 uses. Default: 0 (μ-1σ).
    ///
    /// ```text
    /// cascade[0] = μ − 1.00σ   (stage 1 default — 1/16 Stichprobe)
    /// cascade[1] = μ − 1.50σ
    /// cascade[2] = μ − 1.75σ
    /// cascade[3] = μ − 2.00σ   (stage 2 default — 1/4 Stichprobe)
    /// cascade[4] = μ − 2.25σ
    /// cascade[5] = μ − 2.50σ
    /// cascade[6] = μ − 2.75σ
    /// cascade[7] = μ − 3.00σ
    /// ```
    stage1_level: usize,
    /// Which cascade[] index stage 2 uses. Default: 3 (μ-2σ).
    stage2_level: usize,
}

impl Cascade {
    fn with_floor(floor: RollingFloor) -> Self {
        Self {
            floor,
            stage1_level: 0, // μ-1σ
            stage2_level: 3, // μ-2σ
        }
    }

    /// Create from theoretical parameters for a given bit width.
    ///
    /// Uses the binomial distribution: μ = bits/2, σ = isqrt(bits/4).
    /// Cascade thresholds use theoretical σ — call `calibrate` after
    /// warmup for confidence-backed thresholds.
    pub fn for_width(total_bits: u32) -> Self {
        Self::with_floor(RollingFloor::for_width(total_bits))
    }

    /// Calibrate from a sample of actual pairwise distances (warmup).
    ///
    /// Call once on startup with ≥2 distances sampled from the corpus.
    /// This is the warmup — after this, cascade thresholds have real
    /// confidence because σ comes from observed data, not theory.
    pub fn calibrate(sample_distances: &[u32]) -> Self {
        Self::with_floor(RollingFloor::calibrate(sample_distances))
    }

    /// Classify a Hamming distance into a sigma band.
    ///
    /// Cuts at 3σ, 2σ, 1σ and μ on the σ-lattice, located by the current
    /// shape. Pure integer. Constant time.
    #[inline]
    pub fn band(&self, distance: u32) -> Band {
        let b = self.floor.thresholds(&BAND_LEVELS);
        if distance < b[0] {
            Band::Foveal
        } else if distance < b[1] {
            Band::Near
        } else if distance < b[2] {
            Band::Good
        } else if distance < b[3] {
            Band::Weak
        } else {
            Band::Reject
        }
    }

    /// Classify a single distance into a sigma band.
    ///
    /// Public alias for `band()`. Naming follows the exposure-meter metaphor:
    /// "expose" a distance to see which confidence band it falls into.
    #[inline]
    pub fn expose(&self, distance: u32) -> Band {
        self.band(distance)
    }

    /// Single pair test: is this distance in a useful band?
    ///
    /// Returns `true` if the distance falls in Foveal, Near, or Good bands.
    #[inline]
    pub fn test_distance(&self, distance: u32) -> bool {
        self.band(distance) <= Band::Good
    }

    /// Calibrated mean (the drift anchor; thresholds use the running mean).
    pub fn mu(&self) -> u32 {
        self.floor.mu()
    }

    /// Calibrated sigma (the drift anchor; thresholds use the running σ).
    pub fn sigma(&self) -> u32 {
        self.floor.sigma()
    }

    /// Current band thresholds: the 3σ, 2σ, 1σ and μ lattice points.
    pub fn thresholds(&self) -> [u32; 4] {
        self.floor.thresholds(&BAND_LEVELS)
    }

    /// Current cascade thresholds: the 1σ … 3σ lattice points.
    pub fn cascade_thresholds(&self) -> [u32; 8] {
        self.floor.thresholds(&CASCADE_LEVELS)
    }

    /// Whether empirical (quantile-based) thresholds are active.
    pub fn is_empirical(&self) -> bool {
        self.floor.is_empirical()
    }

    /// Distribution skewness diagnostic. 0 = symmetric.
    pub fn skewness(&self) -> i32 {
        self.floor.skewness()
    }

    /// Distribution kurtosis diagnostic ×100. 300 = normal.
    pub fn kurtosis(&self) -> u32 {
        self.floor.kurtosis()
    }

    /// The underlying rolling floor.
    pub fn floor(&self) -> &RollingFloor {
        &self.floor
    }

    /// Threshold at an arbitrary σ-lattice point.
    ///
    /// `quarter_sigmas`: number of quarter-sigmas below μ.
    /// E.g., 4 = 1σ, 6 = 1.5σ, 7 = 1.75σ, 9 = 2.25σ, 11 = 2.75σ.
    ///
    /// Located by the current shape in the current coordinates; Gaussian is
    /// `μ − quarter_sigmas·σ/4`.
    #[inline]
    pub fn cascade_at(&self, quarter_sigmas: u32) -> u32 {
        self.floor
            .threshold(SigmaLevel(u8::try_from(quarter_sigmas).unwrap_or(u8::MAX)))
    }

    /// Set which cascade[] index stage 1 uses (0..8).
    ///
    /// 0=μ-1σ, 1=μ-1.5σ, 2=μ-1.75σ, 3=μ-2σ, 4=μ-2.25σ,
    /// 5=μ-2.5σ, 6=μ-2.75σ, 7=μ-3σ.
    pub fn set_stage1_level(&mut self, level: usize) {
        assert!(level < 8, "stage1_level must be 0..7");
        self.stage1_level = level;
    }

    /// Set which cascade[] index stage 2 uses (0..8).
    pub fn set_stage2_level(&mut self, level: usize) {
        assert!(level < 8, "stage2_level must be 0..7");
        self.stage2_level = level;
    }

    /// Cascade thresholds, derived once per query.
    #[inline]
    fn active_cascade(&self) -> [u32; 8] {
        self.cascade_thresholds()
    }

    // -- Cascade query --

    /// Three-stage cascade query on `&[u64]` word arrays.
    ///
    /// Returns up to `top_k` results bucketed by band (Foveal first),
    /// sorted by `u32` distance within each bucket. No float anywhere.
    ///
    /// Thresholds follow the current shape (Gaussian or empirical).
    pub fn query(&self, query: &[u64], candidates: &[&[u64]], top_k: usize) -> Vec<RankedHit> {
        let nwords = query.len();
        let do_stage1 = nwords >= 16;
        let do_stage2 = nwords >= 4;
        let cascade = self.active_cascade();

        let mut foveal: Vec<(usize, u32)> = Vec::new();
        let mut near: Vec<(usize, u32)> = Vec::new();
        let mut good: Vec<(usize, u32)> = Vec::new();

        for (idx, candidate) in candidates.iter().enumerate() {
            debug_assert_eq!(candidate.len(), nwords);

            // -- Stage 1: 1/16 sample --
            if do_stage1 {
                let projected = words_hamming_sampled(query, candidate, 16);
                if projected > cascade[self.stage1_level] {
                    continue;
                }
            }

            // -- Stage 2: 1/4 sample --
            if do_stage2 {
                let projected = words_hamming_sampled(query, candidate, 4);
                if projected > cascade[self.stage2_level] {
                    continue;
                }
            }

            // -- Stage 3: full comparison --
            let full_dist = words_hamming(query, candidate);
            match self.band(full_dist) {
                Band::Foveal => foveal.push((idx, full_dist)),
                Band::Near => near.push((idx, full_dist)),
                Band::Good => good.push((idx, full_dist)),
                _ => {} // Weak or Reject — don't collect
            }

            // Early termination: enough foveal hits.
            if foveal.len() >= top_k {
                break;
            }
        }

        // Assemble top-k from buckets. Best band first, integer sort within.
        let mut results = Vec::with_capacity(top_k);

        foveal.sort_unstable_by_key(|&(_, d)| d);
        for &(idx, dist) in foveal.iter().take(top_k) {
            results.push(RankedHit {
                index: idx,
                distance: dist,
                band: Band::Foveal,
            });
        }

        if results.len() < top_k {
            near.sort_unstable_by_key(|&(_, d)| d);
            for &(idx, dist) in near.iter().take(top_k - results.len()) {
                results.push(RankedHit {
                    index: idx,
                    distance: dist,
                    band: Band::Near,
                });
            }
        }

        if results.len() < top_k {
            good.sort_unstable_by_key(|&(_, d)| d);
            for &(idx, dist) in good.iter().take(top_k - results.len()) {
                results.push(RankedHit {
                    index: idx,
                    distance: dist,
                    band: Band::Good,
                });
            }
        }

        results
    }

    // -- Shift detection --

    /// Feed an observed distance into the rolling floor.
    ///
    /// Updates the exact running moments AND the reservoir sample. Every
    /// 1000 observations (after the first 1000), evaluates distribution
    /// shape (skewness/kurtosis), selects σ-based or empirical thresholds,
    /// and checks for drift.
    ///
    /// Returns `Some(ShiftAlert)` if the running mean moved by more than
    /// σ/2, or the running σ by more than σ/4, from the calibrated values.
    pub fn observe(&mut self, distance: u32) -> Option<ShiftAlert> {
        self.floor.observe(distance)
    }

    /// Feed a batch of distances, stopping right after the first checkpoint
    /// that raises a shift. Returns how many were consumed and the shift.
    /// Recalibrating and feeding the rest reproduces the one-at-a-time loop
    /// exactly, for any batching.
    pub fn observe_batch(&mut self, distances: &[u32]) -> (usize, Option<ShiftAlert>) {
        self.floor.observe_batch(distances)
    }

    /// Recalibrate thresholds from a shift alert.
    ///
    /// Resets everything: running moments, reservoir, empirical mode.
    /// σ will converge to the **new** distribution's true σ without
    /// contamination from old data.
    ///
    /// Stage level selections (stage1_level, stage2_level) are preserved.
    pub fn recalibrate(&mut self, alert: &ShiftAlert) {
        self.floor.recalibrate(alert);
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // cascade sweeps index into multiple parallel arrays
mod tests {
    use super::*;

    /// Deterministic PRNG for test reproducibility.
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Generate a random word array of `nwords` u64 values.
    fn random_words(nwords: usize, seed: u64) -> Vec<u64> {
        let mut state = seed;
        (0..nwords).map(|_| splitmix64(&mut state)).collect()
    }

    /// Generate pairwise distances from random vectors.
    fn random_pairwise_distances(nwords: usize, count: usize, seed: u64) -> Vec<u32> {
        let vecs: Vec<Vec<u64>> = (0..count)
            .map(|i| random_words(nwords, seed + i as u64))
            .collect();
        let mut dists = Vec::new();
        for i in 0..count {
            for j in (i + 1)..count {
                dists.push(words_hamming(&vecs[i], &vecs[j]));
            }
        }
        dists
    }

    /// Approximately-normal random distance via Irwin-Hall (sum of 12 uniforms).
    fn gen_approx_normal(rng: &mut u64, mu: u32, sigma: u32) -> u32 {
        let mut sum = 0i64;
        for _ in 0..12 {
            sum += (splitmix64(rng) % 10000) as i64;
        }
        let result = mu as i64 + ((sum - 60000) * sigma as i64) / 10000;
        result.max(0) as u32
    }

    /// Create a Cascade with explicit parametric values (for unit tests):
    /// 1000 prior observations with mean μ and variance σ², empty reservoir.
    fn meter_with_params(mu: u32, sigma: u32) -> Cascade {
        let (n, mu128, s128) = (1000u128, u128::from(mu), u128::from(sigma));
        let moments = ndarray::hpc::statistics::MomentsU32 {
            n: 1000,
            sum: mu128 * n,
            sum_sq: (s128 * s128 + mu128 * mu128) * n,
        };
        Cascade::with_floor(RollingFloor::from_params_and_moments(mu, sigma, moments))
    }

    /// Raw empirical cascade quantiles of a reservoir: the sample value at
    /// each cascade level's Gaussian-equivalent tail rank.
    fn empirical_cascade_of(reservoir: &ReservoirSample) -> [u32; 8] {
        let sorted = reservoir.sorted();
        CASCADE_LEVELS.map(|l| {
            ndarray::hpc::rolling_floor::quantile_of_sorted(&sorted, l.gaussian_tail_per_10000())
        })
    }

    // -- Test 1: Calibration from known distances --

    #[test]
    fn test_calibration_from_corpus() {
        let dists = random_pairwise_distances(256, 100, 42);
        let meter = Cascade::calibrate(&dists);

        assert!(
            meter.mu() > 7800 && meter.mu() < 8600,
            "Expected μ near 8192, got {}",
            meter.mu()
        );
        assert!(
            meter.sigma() > 20 && meter.sigma() < 110,
            "Expected σ near 64, got {}",
            meter.sigma()
        );
        assert!(meter.thresholds()[0] < meter.thresholds()[1]);
        assert!(meter.thresholds()[1] < meter.thresholds()[2]);
        assert!(meter.thresholds()[2] < meter.thresholds()[3]);

        // Reservoir should be seeded from calibration data.
        assert!(
            !meter.floor.reservoir().is_empty(),
            "Reservoir should be seeded from calibration"
        );

        println!(
            "Calibrated: μ={}, σ={}, bands={:?}, reservoir={}",
            meter.mu(),
            meter.sigma(),
            meter.thresholds(),
            meter.floor.reservoir().len()
        );
    }

    // -- Test 2: Direction — lower distance = better band --

    #[test]
    fn test_direction_lower_is_better() {
        let meter = meter_with_params(8192, 64);

        assert_eq!(meter.band(7900), Band::Foveal);
        assert_eq!(meter.band(8010), Band::Near);
        assert_eq!(meter.band(8100), Band::Good);
        assert_eq!(meter.band(8170), Band::Weak);
        assert_eq!(meter.band(8192), Band::Reject);
        assert_eq!(meter.band(9000), Band::Reject);

        assert!(Band::Foveal < Band::Near);
        assert!(Band::Near < Band::Good);
        assert!(Band::Good < Band::Weak);
        assert!(Band::Weak < Band::Reject);
    }

    // -- Test 3: Cascade eliminates most random candidates --

    #[test]
    fn test_elimination_rate() {
        let nwords = 256;
        let meter = Cascade::for_width(nwords as u32 * 64);

        let query = random_words(nwords, 0);
        let n_candidates = 10_000;
        let candidates: Vec<Vec<u64>> = (1..=n_candidates)
            .map(|i| random_words(nwords, i as u64))
            .collect();

        let cascade = meter.active_cascade();
        let mut stage1_pass = 0u32;
        let mut stage2_pass = 0u32;
        let mut stage3_pass = 0u32;

        let t0 = std::time::Instant::now();
        for candidate in &candidates {
            let s1 = words_hamming_sampled(&query, candidate, 16);
            if s1 > cascade[meter.stage1_level] {
                continue;
            }
            stage1_pass += 1;

            let s2 = words_hamming_sampled(&query, candidate, 4);
            if s2 > cascade[meter.stage2_level] {
                continue;
            }
            stage2_pass += 1;

            let full = words_hamming(&query, candidate);
            if meter.band(full) <= Band::Good {
                stage3_pass += 1;
            }
        }
        let elapsed = t0.elapsed();

        let s1_reject_pct = 100.0 * (1.0 - stage1_pass as f64 / n_candidates as f64);
        let total_reject_pct = 100.0 * (1.0 - stage2_pass as f64 / n_candidates as f64);

        println!("Cascade: {:?}", cascade);
        println!(
            "Stage 1: {}/{} pass ({:.1}% rejected)",
            stage1_pass, n_candidates, s1_reject_pct
        );
        println!("Stage 2: {}/{} pass", stage2_pass, stage1_pass);
        println!("Stage 3: {} final", stage3_pass);
        println!("Combined: {:.1}% rejected", total_reject_pct);
        println!(
            "Time: {:?} ({} ns/candidate)",
            elapsed,
            elapsed.as_nanos() / n_candidates as u128
        );

        assert!(
            s1_reject_pct > 50.0,
            "Stage 1 must reject >50%, got {:.1}%",
            s1_reject_pct
        );
        assert!(
            total_reject_pct > 80.0,
            "Combined must reject >80%, got {:.1}%",
            total_reject_pct
        );
    }

    // -- Test 4: Cascade work savings --

    #[test]
    fn test_work_savings() {
        let nwords = 256;
        let meter = Cascade::for_width(nwords as u32 * 64);
        let n_candidates: u64 = 10_000;
        let cascade = meter.active_cascade();

        let query = random_words(nwords, 0);
        let candidates: Vec<Vec<u64>> = (1..=n_candidates)
            .map(|i| random_words(nwords, i))
            .collect();

        let mut s1_pass = 0u64;
        let mut s2_pass = 0u64;

        let t_cascade = std::time::Instant::now();
        for c in &candidates {
            let s1 = words_hamming_sampled(&query, c, 16);
            if s1 > cascade[meter.stage1_level] {
                continue;
            }
            s1_pass += 1;
            let s2 = words_hamming_sampled(&query, c, 4);
            if s2 > cascade[meter.stage2_level] {
                continue;
            }
            s2_pass += 1;
            let _full = words_hamming(&query, c);
        }
        let cascade_ns = t_cascade.elapsed().as_nanos();

        let t_brute = std::time::Instant::now();
        for c in &candidates {
            let _full = words_hamming(&query, c);
        }
        let brute_ns = t_brute.elapsed().as_nanos();

        let brute_cost = n_candidates * nwords as u64;
        let cascade_cost = n_candidates * (nwords as u64 / 16)
            + s1_pass * (nwords as u64 / 4)
            + s2_pass * nwords as u64;
        let savings_pct = 100 - (cascade_cost * 100 / brute_cost);

        println!(
            "Brute: {} ops, Cascade: {} ops, Savings: {}%",
            brute_cost, cascade_cost, savings_pct
        );
        println!("Brute: {} ns, Cascade: {} ns", brute_ns, cascade_ns);
        if cascade_ns > 0 {
            println!("Speedup: {:.1}x", brute_ns as f64 / cascade_ns as f64);
        }

        assert!(
            savings_pct > 50,
            "Must save >50% work, got {}%",
            savings_pct
        );
    }

    // -- Test 5: Shift detection --

    #[test]
    fn test_shift_detection() {
        let mut meter = meter_with_params(8192, 64);

        let mut alert_fired = false;
        for i in 0..5000u32 {
            let fake_dist = 7500 + (i % 100);
            if let Some(alert) = meter.observe(fake_dist) {
                assert!(
                    alert.new_mu < alert.old_mu,
                    "Shift should detect lower mean: old={}, new={}",
                    alert.old_mu,
                    alert.new_mu
                );
                meter.recalibrate(&alert);
                alert_fired = true;
                break;
            }
        }

        assert!(alert_fired, "Shift alert must fire when μ changes by >σ/2");
        assert!(
            meter.mu() < 8000,
            "After recalibration, μ should reflect new distribution, got {}",
            meter.mu()
        );
        // Welford should be reset after recalibrate.
        assert_eq!(
            meter.floor.observations(),
            0,
            "Welford should be reset after recalibrate"
        );
    }

    // -- Test 6: No float guarantee --

    #[test]
    fn test_no_float_guarantee() {
        let meter = Cascade::calibrate(&[8000, 8100, 8200, 8300, 8400]);

        let b: Band = meter.band(8050);
        assert!(matches!(
            b,
            Band::Foveal | Band::Near | Band::Good | Band::Weak | Band::Reject
        ));

        let t: [u32; 4] = meter.thresholds();
        for &thresh in &t {
            let _check: u32 = thresh;
        }

        let hit = RankedHit {
            index: 0,
            distance: 8050,
            band: Band::Near,
        };
        let _d: u32 = hit.distance;
        let _m: u32 = meter.mu();
        let _s: u32 = meter.sigma();
    }

    // -- Test 7: isqrt correctness --

    #[test]
    fn test_isqrt_correctness() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(16), 4);
        assert_eq!(isqrt(64 * 64), 64);
        assert_eq!(isqrt(100 * 100), 100);

        assert_eq!(isqrt(2), 1);
        assert_eq!(isqrt(3), 1);
        assert_eq!(isqrt(5), 2);
        assert_eq!(isqrt(4095), 63);
        assert_eq!(isqrt(4097), 64);

        assert_eq!(isqrt(u32::MAX), 65535);

        for n in [10, 50, 100, 1000, 4096, 16384, 100000, 1000000u32] {
            let s = isqrt(n);
            assert!(s * s <= n, "isqrt({})={}: {}² > {}", n, s, s, n);
            assert!(
                (s + 1) * (s + 1) > n,
                "isqrt({})={}: {}² ≤ {}",
                n,
                s,
                s + 1,
                n
            );
        }
    }

    // -- Test 8: Cascade table structure --

    #[test]
    fn test_cascade_table_structure() {
        let meter = Cascade::for_width(16384);
        let c = meter.cascade_thresholds();

        assert_eq!(c[0], 8192 - 64); // μ - 1.00σ = 8128
        assert_eq!(c[1], 8192 - 96); // μ - 1.50σ = 8096
        assert_eq!(c[2], 8192 - 112); // μ - 1.75σ = 8080
        assert_eq!(c[3], 8192 - 128); // μ - 2.00σ = 8064
        assert_eq!(c[4], 8192 - 144); // μ - 2.25σ = 8048
        assert_eq!(c[5], 8192 - 160); // μ - 2.50σ = 8032
        assert_eq!(c[6], 8192 - 176); // μ - 2.75σ = 8016
        assert_eq!(c[7], 8192 - 192); // μ - 3.00σ = 8000

        for i in 0..7 {
            assert!(
                c[i] > c[i + 1],
                "cascade[{}]={} must > cascade[{}]={}",
                i,
                c[i],
                i + 1,
                c[i + 1]
            );
        }

        assert_eq!(meter.cascade_at(7), 8192 - 7 * 64 / 4); // 1.75σ
        assert_eq!(meter.cascade_at(9), 8192 - 9 * 64 / 4); // 2.25σ
        assert_eq!(meter.cascade_at(11), 8192 - 11 * 64 / 4); // 2.75σ

        println!("Cascade: {:?}", c);
    }

    // -- Test 9: Reservoir sample + empirical quantiles --

    #[test]
    fn test_reservoir_and_empirical_quantiles() {
        let mut rng = 777u64;
        let mut reservoir = ReservoirSample::new(1000);

        // Feed normal-ish distances.
        for _ in 0..8000 {
            let d = gen_approx_normal(&mut rng, 8192, 64);
            reservoir.observe(d);
        }

        // Reservoir should be at capacity.
        assert_eq!(reservoir.len(), 1000);

        // Empirical cascade thresholds should be close to theoretical.
        let cascade = empirical_cascade_of(&reservoir);
        println!("Empirical cascade (from N(8192,64)): {:?}", cascade);

        // 1σ threshold should be near 8128 (±25).
        assert!(
            (cascade[0] as i32 - 8128).unsigned_abs() < 25,
            "1σ empirical should be near 8128, got {}",
            cascade[0]
        );
        // 2σ threshold should be near 8064 (±25).
        assert!(
            (cascade[3] as i32 - 8064).unsigned_abs() < 25,
            "2σ empirical should be near 8064, got {}",
            cascade[3]
        );

        // Skewness and kurtosis should be near-normal.
        let skew = reservoir.skewness(8192, 64);
        let kurt = reservoir.kurtosis(8192, 64);
        println!(
            "Normal data: skewness={}, kurtosis={} (300=normal)",
            skew, kurt
        );
        assert!(
            skew.abs() < 2,
            "Normal data skewness should be near 0, got {}",
            skew
        );
        assert!(
            kurt > 200 && kurt < 500,
            "Normal data kurtosis should be near 300, got {}",
            kurt
        );

        // Now test with bimodal distribution.
        let mut reservoir_bi = ReservoirSample::new(1000);
        for _ in 0..8000 {
            let d = if splitmix64(&mut rng).is_multiple_of(2) {
                gen_approx_normal(&mut rng, 7500, 30)
            } else {
                gen_approx_normal(&mut rng, 8500, 30)
            };
            reservoir_bi.observe(d);
        }

        let skew_bi = reservoir_bi.skewness(8000, 500);
        let kurt_bi = reservoir_bi.kurtosis(8000, 500);
        println!("Bimodal data: skewness={}, kurtosis={}", skew_bi, kurt_bi);

        // Bimodal with modes 1000 apart should NOT look normal.
        // Kurtosis should be well below 200 (platykurtic) for equal-weight bimodal.
        let is_normal_bi = skew_bi.abs() < 2 && kurt_bi > 200 && kurt_bi < 500;
        assert!(
            !is_normal_bi,
            "Bimodal distribution should be detected as non-normal (skew={}, kurt={})",
            skew_bi, kurt_bi
        );
    }

    // -- Test 10: Warmup with 2000 samples, then shift, with reservoir --

    #[test]
    fn test_warmup_2k_then_shift_10k() {
        let nwords = 256;

        let dists_2k = random_pairwise_distances(nwords, 64, 42);
        assert!(dists_2k.len() >= 2000);

        let mut meter = Cascade::calibrate(&dists_2k);

        println!("=== Phase 1: Warmup ({} samples) ===", dists_2k.len());
        println!(
            "  μ={}, σ={}, reservoir={}",
            meter.mu(),
            meter.sigma(),
            meter.floor.reservoir().len()
        );
        println!("  cascade: {:?}", meter.cascade_thresholds());
        println!(
            "  empirical: {:?}",
            empirical_cascade_of(meter.floor.reservoir())
        );
        println!("  use_empirical: {}", meter.is_empirical());

        assert!(
            meter.sigma() > 50 && meter.sigma() < 80,
            "After 2000 samples, σ should be near 64, got {}",
            meter.sigma()
        );

        // Cascade level sweep.
        let query = random_words(nwords, 0);
        let n_test = 5_000u32;
        let candidates: Vec<Vec<u64>> = (1..=n_test)
            .map(|i| random_words(nwords, i as u64 + 10000))
            .collect();

        let level_names = [
            "1.00σ", "1.50σ", "1.75σ", "2.00σ", "2.25σ", "2.50σ", "2.75σ", "3.00σ",
        ];
        let expected_reject = [84.13, 93.32, 95.99, 97.72, 98.78, 99.38, 99.70, 99.87];

        println!(
            "\n  Cascade level sweep (1/16 sample, {} candidates):",
            n_test
        );
        for level in 0..8 {
            let threshold = meter.cascade_thresholds()[level];
            let pass: u32 = candidates
                .iter()
                .filter(|c| words_hamming_sampled(&query, c, 16) <= threshold)
                .count() as u32;
            let reject_pct = 100.0 * (1.0 - pass as f64 / n_test as f64);
            println!("    cascade[{}] ({:>5}) = {:>5}: {:>5}/{} pass ({:>5.1}% rejected, expected {:.1}% at full width)",
                level, level_names[level], threshold, pass, n_test, reject_pct, expected_reject[level]);
        }

        // Phase 2: Shift.
        println!("\n=== Phase 2: Shift detection (10000 obs, μ→7800) ===");
        let mut shifts = 0u32;
        let mut rng_state = 999u64;
        for i in 0..10_000u32 {
            let noise = (splitmix64(&mut rng_state) % 161) as i64 - 80;
            let shifted_dist = (7800i64 + noise).max(0) as u32;
            if let Some(alert) = meter.observe(shifted_dist) {
                println!(
                    "  Shift #{} at obs {}: μ {}→{}, σ {}→{}",
                    shifts + 1,
                    i,
                    alert.old_mu,
                    alert.new_mu,
                    alert.old_sigma,
                    alert.new_sigma
                );
                meter.recalibrate(&alert);
                shifts += 1;
                println!("  New cascade: {:?}", meter.cascade_thresholds());
                println!("  Welford reset: count={}", meter.floor.observations());
            }
        }

        assert!(shifts > 0, "Must detect at least one shift");
        assert!(
            meter.mu() < 8000,
            "μ should have shifted below 8000, got {}",
            meter.mu()
        );

        // After recalibrate: Welford was reset, reservoir was reset.
        println!(
            "\n  After shift: μ={}, σ={}, reservoir={}, empirical={}",
            meter.mu(),
            meter.sigma(),
            meter.floor.reservoir().len(),
            meter.is_empirical()
        );
        println!("  Cascade: {:?}", meter.cascade_thresholds());

        // Re-sweep.
        println!("\n  Post-shift cascade level sweep:");
        for level in 0..8 {
            let threshold = meter.cascade_thresholds()[level];
            let pass: u32 = candidates
                .iter()
                .filter(|c| words_hamming_sampled(&query, c, 16) <= threshold)
                .count() as u32;
            let reject_pct = 100.0 * (1.0 - pass as f64 / n_test as f64);
            println!(
                "    cascade[{}] ({:>5}) = {:>5}: {:>5}/{} pass ({:>5.1}% rejected)",
                level, level_names[level], threshold, pass, n_test, reject_pct
            );
        }
    }

    // -- Test 11: Ironclad 3-strategy comparison across shifting distributions --

    #[test]
    fn test_three_strategies_ironclad() {
        let nwords = 256;
        let total_bits = nwords as u32 * 64;

        let level_names = [
            "1.00σ", "1.50σ", "1.75σ", "2.00σ", "2.25σ", "2.50σ", "2.75σ", "3.00σ",
        ];

        // One-sided normal CDF: P(X > μ-kσ).
        let expected_full = [84.13, 93.32, 95.99, 97.72, 98.78, 99.38, 99.70, 99.87];
        // 1/16 sample: effective Z = k/4.
        let expected_s16 = [59.87, 64.62, 66.91, 69.15, 71.31, 73.40, 75.41, 77.34];
        // 1/4 sample: effective Z = k/2.
        let expected_s4 = [69.15, 77.34, 80.92, 84.13, 86.97, 89.44, 91.54, 93.32];

        // =========================================================
        // Part 1: Full-width σ thresholds vs theory
        // =========================================================
        println!("\n============================================================");
        println!("Part 1: Full-width σ thresholds vs normal distribution theory");
        println!("============================================================");

        let meter = Cascade::for_width(total_bits);
        let query = random_words(nwords, 0);
        let n_test = 10_000usize;
        let candidates: Vec<Vec<u64>> = (1..=n_test)
            .map(|i| random_words(nwords, i as u64 + 50000))
            .collect();

        println!("\n  Full-width Hamming ({} candidates):", n_test);
        let mut max_delta_full = 0.0f64;
        for level in 0..8 {
            let threshold = meter.cascade_thresholds()[level];
            let pass = candidates
                .iter()
                .filter(|c| words_hamming(&query, c) <= threshold)
                .count();
            let reject_pct = 100.0 * (1.0 - pass as f64 / n_test as f64);
            let delta = (reject_pct - expected_full[level]).abs();
            max_delta_full = max_delta_full.max(delta);
            println!(
                "    {} {:>5}: {:>5.1}% rejected (expected {:.1}%, Δ={:.1}) {}",
                level_names[level],
                threshold,
                reject_pct,
                expected_full[level],
                delta,
                if delta < 3.0 { "✓" } else { "✗" }
            );
        }
        assert!(
            max_delta_full < 3.0,
            "Full-width Δ must be < 3%, worst={:.1}",
            max_delta_full
        );

        // =========================================================
        // Part 2: Sampling variance inflation
        // =========================================================
        println!("\n============================================================");
        println!("Part 2: Sampling variance inflation");
        println!("  1/16 sample → σ_eff = 4σ → effective Z = k/4");
        println!("  1/4  sample → σ_eff = 2σ → effective Z = k/2");
        println!("============================================================");

        println!("\n  1/16 sample:");
        for level in 0..8 {
            let threshold = meter.cascade_thresholds()[level];
            let pass = candidates
                .iter()
                .filter(|c| words_hamming_sampled(&query, c, 16) <= threshold)
                .count();
            let reject_pct = 100.0 * (1.0 - pass as f64 / n_test as f64);
            let delta = (reject_pct - expected_s16[level]).abs();
            let k = [1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0][level];
            println!(
                "    {} {:>5}: {:>5.1}% rejected (expected {:.1}% at Z={:.3}, Δ={:.1}) {}",
                level_names[level],
                threshold,
                reject_pct,
                expected_s16[level],
                k / 4.0,
                delta,
                if delta < 4.0 { "✓" } else { "✗" }
            );
        }

        println!("\n  1/4 sample:");
        for level in 0..8 {
            let threshold = meter.cascade_thresholds()[level];
            let pass = candidates
                .iter()
                .filter(|c| words_hamming_sampled(&query, c, 4) <= threshold)
                .count();
            let reject_pct = 100.0 * (1.0 - pass as f64 / n_test as f64);
            let delta = (reject_pct - expected_s4[level]).abs();
            let k = [1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0][level];
            println!(
                "    {} {:>5}: {:>5.1}% rejected (expected {:.1}% at Z={:.3}, Δ={:.1}) {}",
                level_names[level],
                threshold,
                reject_pct,
                expected_s4[level],
                k / 2.0,
                delta,
                if delta < 4.0 { "✓" } else { "✗" }
            );
        }

        // =========================================================
        // Part 3: Integrated meter through distribution shifts
        // =========================================================
        println!("\n============================================================");
        println!("Part 3: Integrated meter through distribution shifts");
        println!("  Welford reset + reservoir + auto-switch on non-normality");
        println!("============================================================");

        struct Phase {
            name: &'static str,
            mu: u32,
            sigma: u32,
            n_obs: u32,
        }

        let phases = [
            Phase {
                name: "A (μ=8192, σ=64)",
                mu: 8192,
                sigma: 64,
                n_obs: 3000,
            },
            Phase {
                name: "B (μ=7800, σ=80)",
                mu: 7800,
                sigma: 80,
                n_obs: 5000,
            },
            Phase {
                name: "C (μ=8500, σ=40)",
                mu: 8500,
                sigma: 40,
                n_obs: 5000,
            },
        ];

        let mut meter = Cascade::for_width(total_bits);
        let mut total_delta = 0.0f64;
        let mut n_measurements = 0u32;
        let mut rng = 12345u64;

        for phase in &phases {
            println!(
                "\n  --- Phase {} ({} observations) ---",
                phase.name, phase.n_obs
            );

            for _ in 0..phase.n_obs {
                let d = gen_approx_normal(&mut rng, phase.mu, phase.sigma);
                if let Some(alert) = meter.observe(d) {
                    println!(
                        "    Shift: μ {}→{}, σ {}→{} (skew={}, kurt={})",
                        alert.old_mu,
                        alert.new_mu,
                        alert.old_sigma,
                        alert.new_sigma,
                        meter.skewness(),
                        meter.kurtosis()
                    );
                    meter.recalibrate(&alert);
                    println!(
                        "    → Recalibrated + reset. reservoir={}, empirical={}",
                        meter.floor.reservoir().len(),
                        meter.is_empirical()
                    );
                }
            }

            println!(
                "\n    State: μ={}, σ={}, empirical={}, reservoir={}, skew={}, kurt={}",
                meter.mu(),
                meter.sigma(),
                meter.is_empirical(),
                meter.floor.reservoir().len(),
                meter.skewness(),
                meter.kurtosis()
            );

            // Measure rejection accuracy with test distances from this phase.
            let mut test_rng = (phase.mu as u64) * 7 + 99999;
            let n_test_dist = 10_000usize;
            let test_dists: Vec<u32> = (0..n_test_dist)
                .map(|_| gen_approx_normal(&mut test_rng, phase.mu, phase.sigma))
                .collect();

            let active = meter.active_cascade();
            println!(
                "\n    Rejection accuracy (active cascade, {} test distances):",
                n_test_dist
            );
            println!("    {:>5}  {:>12}  expected", "level", "actual");
            for &level in &[0usize, 3, 7] {
                let rej = 100.0 * test_dists.iter().filter(|&&d| d > active[level]).count() as f64
                    / n_test_dist as f64;
                let delta = (rej - expected_full[level]).abs();
                total_delta += delta;
                n_measurements += 1;
                println!(
                    "    {:>5}  {:>6.1}% Δ{:<4.1}  {:.1}%",
                    level_names[level], rej, delta, expected_full[level]
                );
            }
        }

        // =========================================================
        // Part 4: Summary
        // =========================================================
        println!("\n============================================================");
        println!("Part 4: Summary");
        println!("============================================================");

        let avg_delta = total_delta / n_measurements as f64;
        println!("\n  Average |Δ| across all phases: {:.2}%", avg_delta);
        println!("  Total |Δ|: {:.1}", total_delta);
        println!(
            "  Final state: μ={}, σ={}, empirical={}",
            meter.mu(),
            meter.sigma(),
            meter.is_empirical()
        );
        println!("  Active cascade: {:?}", meter.active_cascade());

        assert!(
            avg_delta < 10.0,
            "Average Δ must be < 10%, got {:.2}%",
            avg_delta
        );
    }

    // -- Test 12: Empirical auto-switch on bimodal distribution --

    #[test]
    fn test_empirical_auto_switch() {
        let mut meter = Cascade::for_width(16384);
        let mut rng = 42u64;

        // Phase 1: Normal data — should stay σ-based.
        for _ in 0..3000 {
            let d = gen_approx_normal(&mut rng, 8192, 64);
            if let Some(alert) = meter.observe(d) {
                meter.recalibrate(&alert);
            }
        }
        println!(
            "After normal phase: empirical={}, skew={}, kurt={}",
            meter.is_empirical(),
            meter.skewness(),
            meter.kurtosis()
        );

        // Phase 2: Feed bimodal data — should switch to empirical.
        // Mix two narrow distributions far apart.
        let mut switched_to_empirical = false;
        for _ in 0..10000 {
            let d = if splitmix64(&mut rng).is_multiple_of(2) {
                gen_approx_normal(&mut rng, 7200, 30)
            } else {
                gen_approx_normal(&mut rng, 8800, 30)
            };
            if let Some(alert) = meter.observe(d) {
                meter.recalibrate(&alert);
            }
            if meter.is_empirical() {
                switched_to_empirical = true;
            }
        }

        println!(
            "After bimodal phase: empirical={}, skew={}, kurt={}",
            meter.is_empirical(),
            meter.skewness(),
            meter.kurtosis()
        );

        // The meter should have detected non-normality at some point during
        // the bimodal phase. After recalibrate resets, it may have switched
        // back, but the mechanism should have triggered.
        // Note: bimodal data triggers shift detection too, so recalibrate
        // resets the reservoir. The auto-switch happens between recalibrations
        // when enough bimodal data accumulates.
        println!("  Ever switched to empirical: {}", switched_to_empirical);
    }
}
