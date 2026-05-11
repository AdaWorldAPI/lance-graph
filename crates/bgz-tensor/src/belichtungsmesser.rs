//! Belichtungsmesser: HDR cascade search via quarter-sigma L1 bands.
//!
//! Named after the German term for "light meter" — measures the exposure
//! (distance distribution) of Base17 vectors and bins them into 12 bands
//! at quarter-sigma intervals.
//!
//! 3-stroke early exit: if two vectors land in the same band at all three
//! strokes (S/P/O planes), they are likely similar without computing full L1.
//!
//! Calibrated from real pairwise L1 distributions of model weight vectors.

use crate::projection::Base17;

/// Number of quarter-sigma bands.
pub const N_BANDS: usize = 12;

/// A single band boundary in the L1 distribution.
#[derive(Clone, Copy, Debug)]
pub struct Band {
    /// Lower L1 threshold (inclusive).
    pub lo: u32,
    /// Upper L1 threshold (exclusive).
    pub hi: u32,
    /// Fraction of all pairs in this band (from calibration).
    pub density: f32,
}

/// HDR cascade configured from real L1 distribution.
#[derive(Clone, Debug)]
pub struct Belichtungsmesser {
    /// 12 bands at quarter-sigma intervals.
    pub bands: [Band; N_BANDS],
    /// Distribution mean.
    pub mean: f64,
    /// Distribution standard deviation.
    pub sigma: f64,
    /// Total pairs used for calibration.
    pub n_calibration: usize,
}

impl Belichtungsmesser {
    /// Calibrate from a sample of pairwise L1 distances.
    ///
    /// Computes mean and sigma, then creates 12 bands at:
    /// [0, μ-2.5σ), [μ-2.5σ, μ-2σ), ..., [μ+0σ, μ+0.25σ), ..., [μ+2.5σ, ∞)
    pub fn calibrate(l1_distances: &[u32]) -> Self {
        let n = l1_distances.len();
        if n == 0 {
            return Self::default_bands();
        }

        let mean = l1_distances.iter().map(|&d| d as f64).sum::<f64>() / n as f64;
        let variance = l1_distances
            .iter()
            .map(|&d| {
                let diff = d as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / n as f64;
        let sigma = variance.sqrt().max(1.0);

        // 12 quarter-sigma bands centered on mean
        // Band edges: μ - 3σ, μ - 2.5σ, μ - 2σ, ..., μ + 2.5σ, μ + 3σ
        let mut edges = [0u32; N_BANDS + 1];
        for (i, edge) in edges.iter_mut().enumerate().take(N_BANDS + 1) {
            let offset = -3.0 + i as f64 * 0.5; // -3σ to +3σ in 0.5σ steps
            let val = (mean + offset * sigma).max(0.0);
            *edge = val as u32;
        }
        // Last edge extends to max
        edges[N_BANDS] = u32::MAX;

        // Count pairs in each band
        let mut counts = [0u64; N_BANDS];
        for &d in l1_distances {
            for b in 0..N_BANDS {
                if d >= edges[b] && d < edges[b + 1] {
                    counts[b] += 1;
                    break;
                }
            }
        }

        let mut bands = [Band {
            lo: 0,
            hi: 0,
            density: 0.0,
        }; N_BANDS];
        for (b, band) in bands.iter_mut().enumerate().take(N_BANDS) {
            *band = Band {
                lo: edges[b],
                hi: edges[b + 1],
                density: counts[b] as f32 / n as f32,
            };
        }

        Belichtungsmesser {
            bands,
            mean,
            sigma,
            n_calibration: n,
        }
    }

    /// Default bands when no calibration data is available.
    fn default_bands() -> Self {
        let mut bands = [Band {
            lo: 0,
            hi: 0,
            density: 0.0,
        }; N_BANDS];
        for (b, band) in bands.iter_mut().enumerate().take(N_BANDS) {
            *band = Band {
                lo: b as u32 * 1000,
                hi: (b as u32 + 1) * 1000,
                density: 1.0 / N_BANDS as f32,
            };
        }
        bands[N_BANDS - 1].hi = u32::MAX;
        Belichtungsmesser {
            bands,
            mean: 6000.0,
            sigma: 2000.0,
            n_calibration: 0,
        }
    }

    /// Classify an L1 distance into a band index (0..12).
    #[inline]
    pub fn classify(&self, l1: u32) -> u8 {
        for b in 0..N_BANDS {
            if l1 < self.bands[b].hi {
                return b as u8;
            }
        }
        (N_BANDS - 1) as u8
    }

    /// 3-stroke early exit: classify across S/P/O planes.
    ///
    /// Splits the 17 Base17 dims into 3 planes (S: 0-5, P: 6-11, O: 12-16)
    /// and classifies each plane's L1 independently.
    ///
    /// If all three planes land in the same band, the pair is confidently
    /// classified without computing full L1. Returns (band, confident).
    pub fn three_stroke(&self, a: &Base17, b: &Base17) -> (u8, bool) {
        let l1_s = plane_l1(a, b, 0, 6);
        let l1_p = plane_l1(a, b, 6, 12);
        let l1_o = plane_l1(a, b, 12, 17);

        let band_s = self.classify(l1_s);
        let band_p = self.classify(l1_p);
        let band_o = self.classify(l1_o);

        if band_s == band_p && band_p == band_o {
            (band_s, true) // Confident: all planes agree
        } else {
            // Compute full L1 for disambiguation
            let full_l1 = a.l1(b);
            (self.classify(full_l1), false)
        }
    }

    /// Validate cascade: compute false negative rate.
    ///
    /// A false negative is when two vectors that should be classified as "similar"
    /// (cosine > threshold) are placed in a high-distance band.
    ///
    /// Returns (false_negative_rate, band_agreement_rate).
    pub fn validate(
        &self,
        pairs: &[(Base17, Base17, f64)], // (a, b, ground_truth_cosine)
        similarity_threshold: f64,
        band_threshold: u8, // pairs below this band are "similar"
    ) -> (f64, f64) {
        if pairs.is_empty() {
            return (0.0, 1.0);
        }

        let mut false_negatives = 0;
        let mut band_agreements = 0;
        let mut total_similar = 0;

        for (a, b, cosine) in pairs {
            let (band, confident) = self.three_stroke(a, b);
            let is_similar = *cosine > similarity_threshold;
            let classified_similar = band < band_threshold;

            if is_similar {
                total_similar += 1;
                if !classified_similar {
                    false_negatives += 1;
                }
            }
            if confident {
                band_agreements += 1;
            }
        }

        let fn_rate = if total_similar > 0 {
            false_negatives as f64 / total_similar as f64
        } else {
            0.0
        };
        let agreement_rate = band_agreements as f64 / pairs.len() as f64;

        (fn_rate, agreement_rate)
    }

    /// Human-readable summary of the cascade bands.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "Belichtungsmesser: μ={:.1}, σ={:.1}, n={}\n",
            self.mean, self.sigma, self.n_calibration
        );
        for (i, band) in self.bands.iter().enumerate() {
            s.push_str(&format!(
                "  Band {:2}: [{:6}, {:6}) density={:.3}\n",
                i,
                band.lo,
                if band.hi == u32::MAX { 999999 } else { band.hi },
                band.density
            ));
        }
        s
    }
}

/// L1 distance across a subset of Base17 dimensions.
#[inline]
fn plane_l1(a: &Base17, b: &Base17, start: usize, end: usize) -> u32 {
    let mut d = 0u32;
    for i in start..end {
        d += (a.dims[i] as i32 - b.dims[i] as i32).unsigned_abs();
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_distances(n: usize, mean: f64, sigma: f64) -> Vec<u32> {
        // Pseudo-normal distribution via Box-Muller approximation
        (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                let z = (2.0 * std::f64::consts::PI * t).sin() * 2.0; // crude approx
                let val = mean + z * sigma;
                val.max(0.0) as u32
            })
            .collect()
    }

    #[test]
    fn calibrate_basic() {
        let dists = make_distances(10000, 5000.0, 1500.0);
        let bel = Belichtungsmesser::calibrate(&dists);
        assert_eq!(bel.bands.len(), N_BANDS);
        assert!(bel.sigma > 0.0);
        // Total density should sum to ≈ 1.0
        let total: f32 = bel.bands.iter().map(|b| b.density).sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "densities should sum to 1.0: {}",
            total
        );
    }

    #[test]
    fn classify_low_distance_low_band() {
        let dists = make_distances(1000, 5000.0, 1500.0);
        let bel = Belichtungsmesser::calibrate(&dists);
        let low_band = bel.classify(100);
        let high_band = bel.classify(9000);
        assert!(low_band < high_band, "low L1 should be in lower band");
    }

    #[test]
    fn three_stroke_self() {
        let dists = make_distances(1000, 5000.0, 1500.0);
        let bel = Belichtungsmesser::calibrate(&dists);
        let a = Base17 { dims: [1000; 17] };
        let (band, confident) = bel.three_stroke(&a, &a);
        // Self-comparison has L1=0, which falls in the lowest band
        assert!(
            band <= 1,
            "self-comparison should be in lowest bands, got band {}",
            band
        );
        assert!(
            confident,
            "self-comparison should be confident (all planes agree)"
        );
    }

    #[test]
    fn bands_are_ordered() {
        let dists = make_distances(1000, 5000.0, 1500.0);
        let bel = Belichtungsmesser::calibrate(&dists);
        for i in 1..N_BANDS {
            assert!(
                bel.bands[i].lo >= bel.bands[i - 1].lo,
                "band {} lo ({}) < band {} lo ({})",
                i,
                bel.bands[i].lo,
                i - 1,
                bel.bands[i - 1].lo
            );
        }
    }

    #[test]
    fn validate_no_false_negatives_for_identical() {
        let dists = make_distances(1000, 5000.0, 1500.0);
        let bel = Belichtungsmesser::calibrate(&dists);

        // Identical pairs = cosine 1.0, L1 0
        let a = Base17 { dims: [500; 17] };
        let pairs = vec![(a.clone(), a.clone(), 1.0)];
        let (fn_rate, _) = bel.validate(&pairs, 0.5, 6);
        assert_eq!(
            fn_rate, 0.0,
            "identical pairs should never be false negatives"
        );
    }
}
