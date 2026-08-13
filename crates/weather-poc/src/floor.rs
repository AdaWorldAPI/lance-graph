//! `D-WXS-3` — the shared canonical floor calibration + the linear 256-level
//! quantiser (`.claude/plans/weather-soa-bake-v1.md` §4 W1, bar B2).
//!
//! Two halves, one genuinely new, one a documented re-expression:
//!
//! 1. **Calibration (new work).** [`calibrate`] derives a floor window
//!    `[lo, hi]` from a sample of `f64` values at the robust
//!    **0.4th – 99.6th percentile** ([`LO_PERCENTILE`] / [`HI_PERCENTILE`])
//!    — a window that deliberately excludes the extreme tails so a single
//!    outlier sample cannot collapse `[lo, hi]` toward a degenerate span —
//!    and stamps the result with a [`CalibratedFloor::floor_version`]
//!    identifying the calibration epoch that produced it.
//! 2. **The linear floor (a re-expression, not a re-derivation).**
//!    [`CalibratedFloor::quantize`] and [`CalibratedFloor::bucket_center`]
//!    are byte-for-byte the same arithmetic as
//!    `helix::quantize::RollingFloor::{quantize, bucket_center}`
//!    (`crates/helix/src/quantize.rs:99-108`, `:248-250`): 256 uniform
//!    levels over `[lo, hi]`, clip-saturating outside the window, decoding
//!    to the bucket centre. This crate is zero-dep BY CONSTRUCTION (see
//!    `weather-poc/Cargo.toml`'s ZERO-DEP-BY-CONSTRUCTION note — an
//!    *optional path* dep on `helix` would still be read at manifest
//!    resolution and break a clean checkout), so the formula is
//!    re-expressed here rather than imported, and the two implementations
//!    must be kept in lockstep by inspection, not by a shared dependency
//!    edge — the same shape as `D-WXS-12` (`jc` ↔ `ndarray::hpc`
//!    agreement): parity is a measured comparison, not a dependency.
//!
//! [`CalibratedFloor::occupancy`] **is** the histogram — no separate
//! instrument is kept, mirroring the property
//! `helix::quantize::RollingFloor`'s own module docs state
//! (`crates/helix/src/quantize.rs:5-9`).

/// Number of quantisation buckets. Mirrors `helix::constants::PALETTE_SIZE`
/// (`= 256`), duplicated here rather than imported — see the module docs.
const BUCKETS: usize = 256;

/// Lower percentile of the robust calibration window (0.4th percentile).
///
/// Chosen — per the deliverable spec — to exclude the extreme tail so a
/// single outlier sample cannot collapse `[lo, hi]` to a degenerate or
/// near-degenerate span. See [`HI_PERCENTILE`] for the upper bound and the
/// module docs for the full window.
pub const LO_PERCENTILE: f64 = 0.4;

/// Upper percentile of the robust calibration window (99.6th percentile).
/// See [`LO_PERCENTILE`].
pub const HI_PERCENTILE: f64 = 99.6;

/// The result of scoring an external population against a floor
/// ([`CalibratedFloor::saturation_of`]).
///
/// Carries the non-finite count alongside the fraction **by design**: see
/// [`CalibratedFloor::saturation_of`] for why folding non-finite values into
/// the fraction turns "no data at all" into "completely saturated", and why
/// silently dropping them is the other half of the same mistake.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaturationScore {
    /// Fraction of the **finite** values landing in the two rim buckets.
    /// `0.0` when `finite == 0` — check `finite` before reading this.
    pub fraction: f64,
    /// How many values were finite, i.e. the denominator of `fraction`.
    pub finite: usize,
    /// How many values were non-finite (`NaN`, `±inf`) and therefore
    /// excluded. Non-zero means the population was partly or wholly absent
    /// — a fact about the data, not about the floor.
    pub non_finite: usize,
}

/// A calibrated, versioned linear 256-bucket floor over `[lo, hi]`.
///
/// Produced only by [`calibrate`] — there is no public constructor that
/// bypasses calibration, because [`Self::floor_version`] is a promise about
/// *how* `(lo, hi)` was derived, and a hand-built floor would break that
/// promise silently.
///
/// The quantise/dequantise arithmetic ([`Self::quantize`],
/// [`Self::bucket_center`]) is the re-expressed half of `D-WXS-3` — see the
/// module docs for the exact `helix::quantize::RollingFloor` line
/// references it mirrors.
#[derive(Debug, Clone)]
pub struct CalibratedFloor {
    lo: f64,
    hi: f64,
    floor_version: u64,
    occupancy: [u32; BUCKETS],
    samples: u64,
}

impl CalibratedFloor {
    /// **COMPUTE — `&self`.** Map `value` to a bucket in `0..=255`.
    ///
    /// Identical formula to `helix::quantize::RollingFloor::quantize`
    /// (`crates/helix/src/quantize.rs:99-108`):
    ///
    /// - Returns `0` for degenerate bounds (`hi <= lo`).
    /// - Saturates to bucket `0` for `value <= lo` and to bucket `255` for
    ///   `value >= hi`.
    /// - Otherwise `idx = floor(((value - lo) / (hi - lo)) * 256)`, clamped
    ///   to `[0, 255]`.
    pub fn quantize(&self, value: f64) -> u8 {
        if self.hi <= self.lo {
            return 0;
        }
        let t = (value - self.lo) / (self.hi - self.lo);
        let idx = (t * 256.0).floor();
        let idx = idx.clamp(0.0, 255.0);
        idx as u8
    }

    /// **COMPUTE — `&self`.** The representative centre value of bucket `b`.
    ///
    /// Identical formula to `helix::quantize::RollingFloor::bucket_center`
    /// (`crates/helix/src/quantize.rs:248-250`):
    /// `lo + ((b + 0.5) / 256) * (hi - lo)`.
    pub fn bucket_center(&self, b: u8) -> f64 {
        self.lo + ((b as f64 + 0.5) / BUCKETS as f64) * (self.hi - self.lo)
    }

    /// Decode bucket `b`, but only when `floor_version` matches this
    /// floor's own [`Self::floor_version`].
    ///
    /// Returns `None` on a version mismatch instead of silently returning a
    /// bucket centre computed from the WRONG `[lo, hi]` window.
    /// `helix::quantize::RollingFloor`'s module docs state the same
    /// contract for its own version stamp (`crates/helix/src/quantize.rs:
    /// 20-26`): "same value → same `u8`" holds only within a stable floor
    /// version, so decoding across a version boundary without checking is
    /// exactly the silent-mis-dequantisation failure this method closes.
    pub fn decode(&self, b: u8, floor_version: u64) -> Option<f64> {
        if floor_version != self.floor_version {
            return None;
        }
        Some(self.bucket_center(b))
    }

    /// The calibration-epoch stamp.
    ///
    /// A deterministic FNV-1a-64 hash (see the private `fnv1a_64` helper in
    /// this module) of: `lo`, `hi` (the calibration OUTPUT), the
    /// calibration sample's finite-value count, and [`LO_PERCENTILE`] /
    /// [`HI_PERCENTILE`] (the calibration PARAMETERS). Because `(lo, hi)`
    /// is itself a deterministic function of the input sample and the
    /// percentiles, this satisfies "same sample + same percentiles ⇒ same
    /// version": re-calibrating on a byte-identical sample with the same
    /// percentile window reproduces the same `lo`, `hi`, and count, hence
    /// the same hash.
    ///
    /// It does NOT guarantee the converse — two different samples that
    /// happen to land on byte-identical `(lo, hi, count)` would share a
    /// version — but that is correct rather than a gap: `(lo, hi)` is the
    /// only state [`Self::quantize`] and [`Self::bucket_center`] ever
    /// consult, so two floors with the same `(lo, hi)` really are the same
    /// calibration epoch as far as decoding is concerned.
    ///
    /// `std::collections::hash_map::DefaultHasher` (SipHash) was
    /// deliberately NOT used here: its seed is randomised per process, so
    /// two runs of the identical calibration would produce two different
    /// stamps — exactly the property this stamp must not have.
    pub fn floor_version(&self) -> u64 {
        self.floor_version
    }

    /// Current `(lo, hi)` calibration bounds.
    pub fn bounds(&self) -> (f64, f64) {
        (self.lo, self.hi)
    }

    /// Per-bucket occupancy counts over the calibration sample, length 256.
    /// This **is** the histogram — no separate instrument is kept. See the
    /// module docs.
    pub fn occupancy(&self) -> &[u32; BUCKETS] {
        &self.occupancy
    }

    /// Number of finite values the calibration sample contributed to
    /// [`Self::occupancy`] (non-finite values are excluded — see
    /// [`calibrate`]).
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// **COMPUTE — `&self`.** Fraction of the CALIBRATION SAMPLE landing in
    /// the two rim buckets (`0` and `255`), read straight off
    /// [`Self::occupancy`]. Returns `0.0` when [`Self::samples`] is `0`.
    pub fn saturation(&self) -> f64 {
        if self.samples == 0 {
            return 0.0;
        }
        let rim = self.occupancy[0] as u64 + self.occupancy[BUCKETS - 1] as u64;
        rim as f64 / self.samples as f64
    }

    /// **COMPUTE — `&self`.** Fraction of an ARBITRARY external population
    /// `values` that lands in the two rim buckets under this floor.
    ///
    /// Unlike [`Self::saturation`] (which scores the calibration sample's
    /// own recorded [`Self::occupancy`]), this re-quantises `values`
    /// against this floor's `[lo, hi]` on the fly, without mutating
    /// anything. Both directions bar B2 needs are "this floor scored
    /// against a population it was not calibrated on": a narrow floor
    /// scored globally, and a global floor scored on one narrow box.
    ///
    /// # Non-finite values are EXCLUDED and COUNTED, never folded in
    ///
    /// The `fraction` is computed over the **finite** subset only, and
    /// [`SaturationScore::non_finite`] reports how many were set aside. This
    /// is not fussiness — it closes the sibling of the PR #948 `lane.rs`
    /// finding, in the more dangerous position:
    ///
    /// * [`Self::quantize`] maps non-finite input to a rim bucket silently
    ///   (`NaN` → `0`, `-inf` → `0`, `+inf` → `255`; see its own docs). Rim
    ///   is exactly what this function counts.
    /// * So a folded-in `NaN` reads as **saturation**, and an all-`NaN`
    ///   population — *valid ARCO-ERA5 store semantics for a missing chunk*,
    ///   per `probes/weather-p1/README.md` §1 — would score `1.0`:
    ///   "completely saturated" when the truth is "no data at all". Those
    ///   are opposite findings and the bare fraction cannot tell them apart.
    /// * This function is bar B2's **instrument**. A corrupted stored value
    ///   is bad; a corrupted measurement is worse, because every conclusion
    ///   downstream inherits it silently.
    ///
    /// Silently *dropping* the non-finite values would be the other half of
    /// the same mistake — the caller would not learn that a population was
    /// partly or wholly absent. Hence a reported count, matching this
    /// crate's standing rule that a degenerate case is **reported**, never
    /// folded (the same shape as [`calibrate`] and [`Self::decode`]
    /// returning `None`).
    ///
    /// On an empty slice, or one with no finite values at all, `fraction` is
    /// `0.0` and `finite` is `0` — check `finite` before reading `fraction`.
    pub fn saturation_of(&self, values: &[f64]) -> SaturationScore {
        let mut finite = 0usize;
        let mut non_finite = 0usize;
        let mut rim = 0usize;

        for &v in values {
            if !v.is_finite() {
                non_finite += 1;
                continue;
            }
            finite += 1;
            let b = self.quantize(v);
            if b == 0 || b == (BUCKETS - 1) as u8 {
                rim += 1;
            }
        }

        let fraction = if finite == 0 {
            0.0
        } else {
            rim as f64 / finite as f64
        };

        SaturationScore {
            fraction,
            finite,
            non_finite,
        }
    }
}

/// Deterministic FNV-1a 64-bit hash over raw bytes. See
/// [`CalibratedFloor::floor_version`] for why this, and not
/// `std::collections::hash_map::DefaultHasher`, backs the version stamp.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Linear-interpolated percentile (the "R-7" / `PERCENTILE.INC` method)
/// over an already-sorted, non-empty, all-finite slice.
///
/// `p` is expected in `[0, 100]`. `rank = (p / 100) * (n - 1)`;
/// interpolates between the two adjacent order statistics when `rank` is
/// not an integer.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0) * (n as f64 - 1.0);
    let lo_idx = (rank.floor() as usize).min(n - 1);
    let hi_idx = (rank.ceil() as usize).min(n - 1);
    if lo_idx == hi_idx {
        sorted[lo_idx]
    } else {
        let frac = rank - lo_idx as f64;
        sorted[lo_idx] * (1.0 - frac) + sorted[hi_idx] * frac
    }
}

/// Calibrate a [`CalibratedFloor`] from a sample of `f64` values.
///
/// Derives `[lo, hi]` at the [`LO_PERCENTILE`]–[`HI_PERCENTILE`] robust
/// window (see the module docs), then re-scans the sample under the
/// resulting floor to build [`CalibratedFloor::occupancy`] and
/// [`CalibratedFloor::samples`] — the "genuinely new" half of `D-WXS-3`;
/// [`CalibratedFloor::quantize`] / [`CalibratedFloor::bucket_center`] are
/// the re-expressed half.
///
/// Non-finite (`NaN` / `±inf`) values in `sample` are excluded from both
/// the percentile computation and the occupancy scan — they carry no
/// ordering information a percentile window can use.
///
/// Returns `None` when `sample` contains no finite values (there is
/// nothing to calibrate a window from).
pub fn calibrate(sample: &[f64]) -> Option<CalibratedFloor> {
    let mut finite: Vec<f64> = sample.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return None;
    }
    finite.sort_by(|a, b| a.total_cmp(b));

    let lo = percentile(&finite, LO_PERCENTILE);
    let hi = percentile(&finite, HI_PERCENTILE);

    let mut version_bytes: Vec<u8> = Vec::with_capacity(8 * 5);
    version_bytes.extend_from_slice(&lo.to_le_bytes());
    version_bytes.extend_from_slice(&hi.to_le_bytes());
    version_bytes.extend_from_slice(&(finite.len() as u64).to_le_bytes());
    version_bytes.extend_from_slice(&LO_PERCENTILE.to_le_bytes());
    version_bytes.extend_from_slice(&HI_PERCENTILE.to_le_bytes());
    let floor_version = fnv1a_64(&version_bytes);

    let mut floor = CalibratedFloor {
        lo,
        hi,
        floor_version,
        occupancy: [0u32; BUCKETS],
        samples: 0,
    };

    for &v in &finite {
        let b = floor.quantize(v);
        floor.occupancy[b as usize] = floor.occupancy[b as usize].saturating_add(1);
        floor.samples = floor.samples.saturating_add(1);
    }

    Some(floor)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── fixture helper ───────────────────────────────────────────────────

    /// Evenly spaced points spanning `[lo, hi]` inclusive — deterministic,
    /// no RNG, so every calibration and every measured saturation number in
    /// these tests is exactly reproducible.
    fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
        assert!(n >= 2, "linspace needs at least 2 points");
        (0..n)
            .map(|i| lo + (hi - lo) * i as f64 / (n as f64 - 1.0))
            .collect()
    }

    // ── B2 primary: well-covered sample, zero empty buckets, low saturation ─

    #[test]
    fn primary_well_covered_sample_has_zero_empty_buckets_and_low_saturation() {
        // Pre-registered epsilon, worked out from the construction BEFORE
        // running: the 0.4/99.6 percentile window trims ~0.4% of the
        // 102_400-point evenly-spaced fixture off each end (~409-410
        // points per side), and each rim bucket (0 and 255) ALSO collects
        // its own ~1/256 share of the interior mass (~397 points) because
        // the trimmed points and the interior points nearest lo/hi both
        // land in the same rim bucket. That puts expected rim occupancy
        // near (409 + 397 + 410 + 397) / 102_400 ≈ 1.57%. epsilon = 0.03
        // leaves roughly 2x margin above that expectation without being so
        // loose that a real quantisation bug (e.g. an inverted saturation
        // branch, which would push saturation toward 1.0) would slip past.
        const EPSILON: f64 = 0.03;

        let sample = linspace(-1000.0, 1000.0, 102_400);
        let floor = calibrate(&sample).expect("non-empty sample calibrates");

        let empty_buckets = floor.occupancy().iter().filter(|&&c| c == 0).count();
        assert_eq!(
            empty_buckets, 0,
            "expected every one of the 256 buckets occupied, found {empty_buckets} empty"
        );

        let sat = floor.saturation();
        assert!(
            sat < EPSILON,
            "saturation {sat} did not clear the pre-registered epsilon {EPSILON}"
        );
    }

    // ── B2 control that can lose: narrow floor applied globally saturates ──

    #[test]
    fn control_narrow_floor_applied_globally_shows_high_saturation() {
        // This is `GEO-DEGENERATE` re-homed (plan §4 bar B2): a floor
        // calibrated on a narrow slice, applied to a much wider
        // independent population, must show HIGH saturation. Known
        // losable — the arc measured 72-97% on real data for exactly this
        // construction.
        let narrow_sample = linspace(-10.0, 10.0, 2_000);
        let narrow_floor = calibrate(&narrow_sample).expect("non-empty sample calibrates");

        // Anti-vacuity: confirm the fixture really IS narrow before
        // scoring it — otherwise a passing assertion below would prove
        // nothing about the narrow-vs-wide mechanism.
        let (n_lo, n_hi) = narrow_floor.bounds();
        assert!(
            n_hi - n_lo < 25.0,
            "fixture is not actually narrow: bounds [{n_lo}, {n_hi}]"
        );

        let wide_population = linspace(-1000.0, 1000.0, 102_400);
        let sat = narrow_floor.saturation_of(&wide_population);
        assert_eq!(sat.non_finite, 0, "fixture population is all finite");
        assert_eq!(sat.finite, wide_population.len());
        assert!(
            sat.fraction > 0.9,
            "expected high saturation applying a narrow floor globally, got {}",
            sat.fraction
        );
    }

    // ── B2 stay-silent twin: global floor on a non-trivial sub-window ──────

    #[test]
    fn stay_silent_global_floor_on_a_narrow_sub_window_does_not_saturate() {
        // The global floor scored against a non-trivial sub-window of the
        // SAME population it was calibrated on must NOT show high
        // saturation. The sub-window here is the population's own upper
        // third — real values, not an empty set — chosen specifically
        // because it brushes against (without mostly exceeding) the
        // calibration window's own trimmed edge, which is a genuinely
        // testable case rather than a trivially-safe interior slice.
        let wide_population = linspace(-1000.0, 1000.0, 102_400);
        let global_floor = calibrate(&wide_population).expect("non-empty sample calibrates");

        let n = wide_population.len();
        let sub_window = &wide_population[(2 * n / 3)..n];
        assert!(
            sub_window.len() > 1_000,
            "sub-window must be non-trivial, got {} points",
            sub_window.len()
        );

        let sat = global_floor.saturation_of(sub_window);
        assert_eq!(sat.non_finite, 0, "fixture sub-window is all finite");
        assert_eq!(sat.finite, sub_window.len());
        assert!(
            sat.fraction < 0.1,
            "global floor should not saturate heavily on an interior-adjacent sub-window, got {}",
            sat.fraction
        );
    }

    // ── non-finite values are excluded and counted, never folded ─────────

    /// A non-finite value must NOT be counted as saturation.
    ///
    /// This is the sibling of the PR #948 `lane.rs` finding, in the more
    /// dangerous position: `saturation_of` is bar B2's INSTRUMENT, so folding
    /// `NaN` into the fraction corrupts a measurement rather than a stored
    /// value, and every conclusion downstream inherits it silently.
    ///
    /// The first block is the sharp case: an ALL-`NaN` population is valid
    /// ARCO-ERA5 store semantics for a missing chunk
    /// (`probes/weather-p1/README.md` §1), and the pre-fix code would have
    /// scored it `1.0` — "completely saturated" — when the truth is "no data
    /// at all".
    #[test]
    fn non_finite_values_are_excluded_and_counted_never_scored_as_saturation() {
        let population = linspace(-1000.0, 1000.0, 10_240);
        let floor = calibrate(&population).expect("non-empty sample calibrates");

        // ── the sharp case: an entirely absent field ──
        let all_nan = vec![f64::NAN; 4_096];
        let sat = floor.saturation_of(&all_nan);
        assert_eq!(
            sat.non_finite, 4_096,
            "every value must be counted as absent"
        );
        assert_eq!(sat.finite, 0, "none of them is a measurement");
        assert_eq!(
            sat.fraction, 0.0,
            "an absent field must not read as saturation — pre-fix this was 1.0"
        );

        // ── mixed: NaN must not inflate a genuinely low fraction ──
        // Interior values (the population's middle third) do not saturate.
        let n = population.len();
        let interior = &population[(n / 3)..(2 * n / 3)];
        let clean = floor.saturation_of(interior);
        assert_eq!(clean.non_finite, 0);
        assert!(
            clean.fraction < 0.1,
            "interior values must not saturate, got {}",
            clean.fraction
        );

        let mut poisoned: Vec<f64> = interior.to_vec();
        poisoned.extend(std::iter::repeat_n(f64::NAN, interior.len()));
        let mixed = floor.saturation_of(&poisoned);
        assert_eq!(mixed.non_finite, interior.len(), "the NaN half is counted");
        assert_eq!(
            mixed.finite,
            interior.len(),
            "the real half is the denominator"
        );
        assert_eq!(
            mixed.fraction, clean.fraction,
            "adding absent values must not move the fraction at all"
        );

        // ── +/-inf too, not just NaN: they land on the RIM buckets ──
        for (name, bad) in [("+inf", f64::INFINITY), ("-inf", f64::NEG_INFINITY)] {
            let mut with_inf: Vec<f64> = interior.to_vec();
            with_inf.extend(std::iter::repeat_n(bad, interior.len()));
            let scored = floor.saturation_of(&with_inf);
            assert_eq!(
                scored.non_finite,
                interior.len(),
                "{name} must be counted as absent"
            );
            assert_eq!(
                scored.fraction, clean.fraction,
                "{name} must not inflate saturation (it quantises to a rim bucket)"
            );
        }

        // ── stay-silent twin: an all-finite population is unaffected ──
        let sat_finite = floor.saturation_of(&population);
        assert_eq!(sat_finite.non_finite, 0, "no false positives on clean data");
        assert_eq!(sat_finite.finite, population.len());
    }

    // ── round-trip bound ─────────────────────────────────────────────────

    #[test]
    fn round_trip_bucket_center_is_within_half_bucket_of_source_value() {
        let sample = linspace(-500.0, 1500.0, 50_000);
        let floor = calibrate(&sample).expect("non-empty sample calibrates");
        let (lo, hi) = floor.bounds();
        assert!(
            hi > lo,
            "calibration must be non-degenerate for this fixture"
        );

        let half_bucket = (hi - lo) / (2.0 * BUCKETS as f64);
        let probe_count = 5_000usize;
        for i in 0..probe_count {
            // Offset by half a probe-step so we never land exactly on a
            // window endpoint, which would legitimately saturate.
            let t = (i as f64 + 0.5) / probe_count as f64;
            let v = lo + t * (hi - lo);
            let b = floor.quantize(v);
            let center = floor.bucket_center(b);
            let delta = (center - v).abs();
            assert!(
                delta <= half_bucket + 1e-9,
                "round-trip error {delta} exceeds the ±half-bucket bound {half_bucket} at v={v}"
            );
        }
    }

    // ── version stamp is load-bearing ───────────────────────────────────

    #[test]
    fn version_stamp_mismatch_is_detected_and_matching_version_decodes() {
        let sample = linspace(-50.0, 50.0, 5_000);
        let floor = calibrate(&sample).expect("non-empty sample calibrates");

        let real_version = floor.floor_version();
        let bogus_version = real_version.wrapping_add(1);
        assert_ne!(
            bogus_version, real_version,
            "fixture must construct a genuinely different version"
        );

        // Wrong version: must be detected, never silently mis-dequantized.
        assert_eq!(floor.decode(10, bogus_version), None);

        // Matching version: must decode, to exactly what bucket_center gives.
        let expected = floor.bucket_center(10);
        assert_eq!(floor.decode(10, real_version), Some(expected));
    }

    // ── small direct-risk sanity checks on this file's own new code ────────

    #[test]
    fn calibrate_on_a_sample_with_no_finite_values_returns_none() {
        let sample = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        assert!(calibrate(&sample).is_none());
    }

    #[test]
    fn quantize_degenerate_bounds_returns_zero_matching_helix_guard() {
        // An all-identical sample collapses the percentile window to
        // lo == hi; `quantize` must not panic and must saturate every
        // input to bucket 0 — the same degenerate-bounds guard
        // `helix::quantize::RollingFloor::quantize` documents
        // (`crates/helix/src/quantize.rs:100-102`).
        let sample = vec![42.0_f64; 500];
        let floor = calibrate(&sample).expect("non-empty sample calibrates");
        let (lo, hi) = floor.bounds();
        assert!(
            hi <= lo,
            "an all-identical sample should collapse the window, got [{lo}, {hi}]"
        );
        assert_eq!(floor.quantize(0.0), 0);
        assert_eq!(floor.quantize(42.0), 0);
        assert_eq!(floor.quantize(1000.0), 0);
    }
}
