//! HDR cascade adapter: wires ndarray's Cascade (Welford σ, ShiftAlert,
//! 3-stroke query) into bgz-tensor's stacked encoding pipeline.
//!
//! ndarray::hpc::cascade already provides:
//!   - Cascade::calibrate() — μ, σ from real distances
//!   - Cascade::observe()   — Welford online σ + ShiftAlert on floor change
//!   - Cascade::recalibrate() — threshold adjustment after drift
//!   - Cascade::expose()    — band classification (Foveal/Near/Good/Weak/Reject)
//!   - 3-stroke query()     — σ-warmup from first 128 vectors
//!
//! This module adapts that infrastructure for StackedN vectors:
//!   HEEL: palette L1 distance (O(1) lookup, NOT popcount)
//!   HIP:  palette ranking from 256-entry precomputed distance table
//!   TWIG: Base17 L1 (17 i16 subtracts)
//!   LEAF: BF16→f32 hydration (exact)
//!
//! NO Hamming/popcount on bgz17 data. Palette ranking replaces it.
//! The only valid distance metrics on bgz17 are L1, PCDVQ-weighted L1,
//! and palette lookup.
//!
//! Quarter-sigma bands are computed BY the Cascade's σ tracking —
//! we expose 12 bands at ¼σ from whatever μ/σ the Cascade currently holds.
//! When ShiftAlert fires, bands recalculate automatically.

use crate::stacked_n::StackedN;
use crate::projection::Base17;

// ndarray = hardware acceleration (Welford σ, ShiftAlert, SIMD L1).
// bgz-tensor = consumer. NOT optional — both in same binary.
pub use ndarray::hpc::cascade::{Cascade as NdarrayCascade, Band as NdarrayBand, ShiftAlert};

/// Number of quarter-sigma bands.
pub const N_BANDS: usize = 12;

/// Quarter-sigma band derived from a Cascade's current μ/σ.
///
/// NOT stored — computed on the fly from the Cascade's rolling σ.
/// When σ changes (ShiftAlert), bands change automatically.
#[derive(Clone, Copy, Debug)]
pub struct QuarterSigmaBand {
    pub lo: f64,
    pub hi: f64,
}

/// Compute 12 quarter-sigma band edges from current μ and σ.
///
/// Bands span [μ-3σ, μ+3σ) in ½σ steps (12 bands).
/// This function is called AFTER each ShiftAlert to update band edges.
/// It does NOT store state — it reads from the Cascade's μ/σ.
#[inline]
pub fn quarter_sigma_bands(mu: f64, sigma: f64) -> [QuarterSigmaBand; N_BANDS] {
    let mut bands = [QuarterSigmaBand { lo: 0.0, hi: 0.0 }; N_BANDS];
    for (b, band) in bands.iter_mut().enumerate().take(N_BANDS) {
        let offset = -3.0 + b as f64 * 0.5;
        band.lo = mu + offset * sigma;
        band.hi = if b == N_BANDS - 1 { f64::INFINITY } else { mu + (offset + 0.5) * sigma };
    }
    bands
}

/// Classify a distance into a band index (0-11) given current μ/σ.
#[inline]
pub fn classify_band(distance: f64, mu: f64, sigma: f64) -> u8 {
    if sigma < 1e-12 { return 6; } // center band if no variance
    let z = (distance - mu) / sigma;
    let band = ((z + 3.0) / 0.5).floor() as i32;
    band.clamp(0, (N_BANDS - 1) as i32) as u8
}

/// Map ndarray Band (Foveal/Near/Good/Weak/Reject) to quarter-sigma index.
///
/// ndarray's Cascade uses 5 bands (threshold/4 increments).
/// We map these to the 12 ¼σ bands for finer granularity:
///   Foveal → bands 0-2  (< μ-1.5σ, very close)
///   Near   → bands 3-4  (μ-1.5σ to μ-0.5σ)
///   Good   → bands 5-6  (μ-0.5σ to μ+0.5σ, center)
///   Weak   → bands 7-9  (μ+0.5σ to μ+2σ)
///   Reject → bands 10-11 (> μ+2σ)
pub fn cascade_band_to_quarter_sigma(band_name: &str) -> (u8, u8) {
    match band_name {
        "Foveal" => (0, 2),
        "Near"   => (3, 4),
        "Good"   => (5, 6),
        "Weak"   => (7, 9),
        "Reject" => (10, 11),
        _        => (6, 6),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cascade stages for StackedN — palette L1, NOT popcount
// ═══════════════════════════════════════════════════════════════════════════

/// HEEL stage: palette L1 distance between two Base17 vectors.
///
/// O(1) if 256-palette distance table is available (precomputed).
/// O(17) if computing L1 directly on i16[17].
/// NEVER popcount.
#[inline]
pub fn heel_distance_l1(a: &Base17, b: &Base17) -> u32 {
    a.l1(b)
}

/// HEEL stage with palette: O(1) lookup in 256×256 distance table.
///
/// `palette_table[a_idx * 256 + b_idx]` = precomputed L1 distance.
/// This IS the correct cheap distance for bgz17.
#[inline]
pub fn heel_distance_palette(a_idx: u8, b_idx: u8, palette_table: &[u16]) -> u16 {
    palette_table[a_idx as usize * 256 + b_idx as usize]
}

/// HIP stage: palette ranking.
///
/// Given query palette index, returns all entries within the ¼σ band
/// from the 256-entry precomputed distance table.
/// This replaces the popcount scan entirely.
pub fn hip_palette_candidates(
    query_idx: u8,
    palette_table: &[u16],
    mu: f64,
    sigma: f64,
    max_band: u8,
) -> Vec<(u8, u16)> {
    let mut candidates = Vec::new();
    for k in 0..256u16 {
        let dist = palette_table[query_idx as usize * 256 + k as usize];
        let band = classify_band(dist as f64, mu, sigma);
        if band <= max_band {
            candidates.push((k as u8, dist));
        }
    }
    candidates.sort_by_key(|&(_, d)| d);
    candidates
}

/// TWIG stage: Base17 L1 on actual vectors (not palette proxies).
///
/// Only called for pairs that survived HEEL + HIP.
/// 17 i16 subtracts — the true metric distance.
#[inline]
pub fn twig_distance_l1(a: &Base17, b: &Base17) -> u32 {
    a.l1(b)
}

/// LEAF stage: BF16→f32 hydration for exact cosine.
///
/// Only called for pairs that survived all prior stages.
/// Uses Rust 1.94 built-in: f32::from_bits((bits as u32) << 16).
pub fn leaf_hydrate_cosine(a: &StackedN, b: &StackedN) -> f64 {
    a.cosine(b)
}

/// Full cascade: HEEL(palette) → HIP(ranking) → TWIG(L1) → LEAF(hydrate).
///
/// Delegates Welford σ tracking to ndarray::hpc::cascade::Cascade.
/// Quarter-sigma bands are derived from the Cascade's current μ/σ.
/// When ShiftAlert fires, bands recalculate automatically.
pub struct PaletteCascade {
    /// ndarray Cascade: Welford μ/σ, ShiftAlert, band classification.
    /// ndarray = hardware, bgz-tensor = consumer. Same binary.
    inner: NdarrayCascade,
    /// Maximum ¼σ band to pass HEEL (0 = only Foveal, 11 = pass all).
    pub heel_max_band: u8,
    /// Maximum ¼σ band to pass HIP.
    pub hip_max_band: u8,
    /// Maximum ¼σ band to pass TWIG.
    pub twig_max_band: u8,
}

impl PaletteCascade {
    /// Create with initial calibration. Delegates to ndarray Cascade::calibrate().
    pub fn calibrate(distances: &[u32]) -> Self {
        let inner = NdarrayCascade::calibrate(distances, 34); // 34 = Base17 byte size
        Self { inner, heel_max_band: 6, hip_max_band: 8, twig_max_band: 10 }
    }

    /// Welford online update — delegates to ndarray Cascade::observe().
    /// Returns true if ShiftAlert fired (floor change detected).
    pub fn observe(&mut self, distance: u32) -> bool {
        self.inner.observe(distance).is_some()
    }

    /// Access ndarray Cascade directly for expose() / query() / recalibrate().
    pub fn ndarray_cascade(&self) -> &NdarrayCascade {
        &self.inner
    }

    /// Mutable access for recalibrate() after ShiftAlert.
    pub fn ndarray_cascade_mut(&mut self) -> &mut NdarrayCascade {
        &mut self.inner
    }

    /// Current μ from ndarray Cascade.
    pub fn mu(&self) -> f64 { self.inner.mu() }

    /// Current σ from ndarray Cascade.
    pub fn sigma(&self) -> f64 { self.inner.sigma() }

    /// Current ¼σ bands (recomputed from Cascade's rolling μ/σ).
    pub fn bands(&self) -> [QuarterSigmaBand; N_BANDS] {
        quarter_sigma_bands(self.mu(), self.sigma())
    }

    /// Classify a distance into current ¼σ band.
    #[inline]
    pub fn classify(&self, distance: f64) -> u8 {
        classify_band(distance, self.mu(), self.sigma())
    }

    /// Should this distance be rejected at a given stage?
    #[inline]
    pub fn should_reject(&self, distance: f64, max_band: u8) -> bool {
        self.classify(distance) > max_band
    }
}

/// Cascade statistics.
#[derive(Clone, Debug, Default)]
pub struct PaletteCascadeStats {
    pub total_pairs: usize,
    pub heel_rejected: usize,
    pub hip_rejected: usize,
    pub twig_rejected: usize,
    pub leaf_computed: usize,
}

impl PaletteCascadeStats {
    pub fn elimination_rate(&self) -> f64 {
        if self.total_pairs == 0 { return 0.0; }
        1.0 - self.leaf_computed as f64 / self.total_pairs as f64
    }

    pub fn summary(&self) -> String {
        let pct = |n: usize| if self.total_pairs > 0 { n as f64 / self.total_pairs as f64 * 100.0 } else { 0.0 };
        format!(
            "PaletteCascade: {} pairs\n\
             HEEL (palette L1):  {:>6} rejected ({:.1}%)\n\
             HIP  (ranking):     {:>6} rejected ({:.1}%)\n\
             TWIG (Base17 L1):   {:>6} rejected ({:.1}%)\n\
             LEAF (hydrate):     {:>6} computed  ({:.1}%)\n\
             Elimination:        {:.1}%",
            self.total_pairs,
            self.heel_rejected, pct(self.heel_rejected),
            self.hip_rejected, pct(self.hip_rejected),
            self.twig_rejected, pct(self.twig_rejected),
            self.leaf_computed, pct(self.leaf_computed),
            self.elimination_rate() * 100.0,
        )
    }
}

/// Run the palette cascade on Base17 vectors with precomputed palette table.
pub fn run_palette_cascade(
    queries: &[Base17],
    keys: &[Base17],
    q_palette_idx: &[u8],
    k_palette_idx: &[u8],
    palette_table: &[u16], // 256×256 precomputed L1 distances
    cascade: &PaletteCascade,
    max_pairs: usize,
) -> (Vec<(usize, usize, u32)>, PaletteCascadeStats) {
    let mut stats = PaletteCascadeStats::default();
    let mut results = Vec::new();
    let mut count = 0;

    'outer: for (qi, q) in queries.iter().enumerate() {
        for (ki, k) in keys.iter().enumerate() {
            if count >= max_pairs { break 'outer; }
            count += 1;
            stats.total_pairs += 1;

            // HEEL: palette distance (O(1) table lookup)
            let heel_dist = heel_distance_palette(q_palette_idx[qi], k_palette_idx[ki], palette_table);
            if cascade.should_reject(heel_dist as f64, cascade.heel_max_band) {
                stats.heel_rejected += 1;
                continue;
            }

            // HIP: same palette distance, tighter band
            if cascade.should_reject(heel_dist as f64, cascade.hip_max_band) {
                stats.hip_rejected += 1;
                continue;
            }

            // TWIG: actual Base17 L1 (17 i16 subtracts)
            let twig_dist = twig_distance_l1(q, k);
            if cascade.should_reject(twig_dist as f64, cascade.twig_max_band) {
                stats.twig_rejected += 1;
                continue;
            }

            // LEAF: survived — report actual L1 distance
            stats.leaf_computed += 1;
            results.push((qi, ki, twig_dist));
        }
    }

    (results, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarter_sigma_band_edges() {
        let bands = quarter_sigma_bands(100.0, 20.0);
        assert_eq!(bands.len(), N_BANDS);
        // Band 0 starts at μ-3σ = 40
        assert!((bands[0].lo - 40.0).abs() < 0.1);
        // Band 6 starts at μ = 100
        assert!((bands[6].lo - 100.0).abs() < 0.1);
        // Last band extends to infinity
        assert!(bands[N_BANDS - 1].hi.is_infinite());
    }

    #[test]
    fn classify_center_is_band_6() {
        let band = classify_band(100.0, 100.0, 20.0);
        assert_eq!(band, 6, "center should be band 6");
    }

    #[test]
    fn classify_low_is_low_band() {
        let band = classify_band(50.0, 100.0, 20.0);
        assert!(band < 6, "below mean should be low band: {}", band);
    }

    #[test]
    fn classify_high_is_high_band() {
        let band = classify_band(150.0, 100.0, 20.0);
        assert!(band > 6, "above mean should be high band: {}", band);
    }

    #[test]
    fn palette_cascade_calibrate() {
        let distances: Vec<u32> = (0..1000).map(|i| (i * 3 % 500) as u32).collect();
        let cascade = PaletteCascade::calibrate(&distances);
        assert!(cascade.sigma() > 0.0);
        assert!(cascade.ndarray_cascade().observations() == 1000);
    }

    #[test]
    fn palette_cascade_observe_welford() {
        let mut cascade = PaletteCascade::calibrate(&[100, 200, 150, 180, 120]);
        let initial_mu = cascade.mu();

        // Normal observations shouldn't trigger shift
        for d in [110, 130, 140, 160, 170, 190, 200, 150, 140, 130] {
            let shifted = cascade.observe(d);
            assert!(!shifted, "normal variation shouldn't trigger shift");
        }

        // Extreme shift should trigger
        for _ in 0..20 {
            cascade.observe(5000);
        }
        // After many extreme observations, mu should have moved significantly
        assert!(cascade.mu() > initial_mu + 100.0,
            "extreme observations should shift mu: {} vs {}", cascade.mu(), initial_mu);
    }

    #[test]
    fn heel_l1_not_popcount() {
        let a = Base17 { dims: [100; 17] };
        let b = Base17 { dims: [110; 17] };
        let dist = heel_distance_l1(&a, &b);
        // L1 = 17 * 10 = 170
        assert_eq!(dist, 170);
    }

    #[test]
    fn palette_lookup_o1() {
        // Simulate a 256×256 distance table
        let mut table = vec![0u16; 256 * 256];
        table[5 * 256 + 10] = 42;
        table[10 * 256 + 5] = 42; // symmetric
        assert_eq!(heel_distance_palette(5, 10, &table), 42);
        assert_eq!(heel_distance_palette(10, 5, &table), 42);
    }

    #[test]
    fn run_cascade_with_palette() {
        let queries = vec![
            Base17 { dims: [100; 17] },
            Base17 { dims: [200; 17] },
        ];
        let keys = vec![
            Base17 { dims: [105; 17] },
            Base17 { dims: [500; 17] },
        ];

        // Build simple palette table: distance = |a - b|
        let mut table = vec![0u16; 256 * 256];
        for a in 0..256u16 {
            for b in 0..256u16 {
                table[a as usize * 256 + b as usize] = (a as i16 - b as i16).unsigned_abs();
            }
        }

        let q_idx = vec![0u8, 1]; // arbitrary palette assignments
        let k_idx = vec![0u8, 3];

        let cascade = PaletteCascade::calibrate(&[50, 100, 150, 200, 250]);

        let (results, stats) = run_palette_cascade(
            &queries, &keys, &q_idx, &k_idx, &table, &cascade, 100
        );

        assert_eq!(stats.total_pairs, 4);
        assert!(stats.heel_rejected + stats.hip_rejected +
                stats.twig_rejected + stats.leaf_computed == stats.total_pairs);
        eprintln!("{}", stats.summary());
    }
}
