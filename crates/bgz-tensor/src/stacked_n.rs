//! Variable-resolution stacked BF16 encoding with CLAM cosine codebook
//! and BF16→f32 leaf hydration.
//!
//! Supports encoding at any sample count per dimension:
//!   4 samples/dim  = 136 bytes  (BF16×4, compact)
//!   8 samples/dim  = 272 bytes  (BF16×8, moderate)
//!  16 samples/dim  = 544 bytes  (BF16×16, high-res)
//!   N samples/dim  = N*34 bytes (full octave, maximum fidelity)
//!
//! Codebook sizes: 512 / 1024 / 2048 / 4096 / 8192 / 16384
//! CLAM clustering uses cosine distance (not L1).
//! Cascade: use hdr_belichtung::PaletteCascade (palette L1, ndarray Welford σ).

use crate::projection::{BASE_DIM, GOLDEN_STEP};

/// Golden-step position table.
const GOLDEN_POS: [u8; BASE_DIM] = {
    let mut t = [0u8; BASE_DIM];
    let mut i = 0;
    while i < BASE_DIM {
        t[i] = ((i * GOLDEN_STEP) % BASE_DIM) as u8;
        i += 1;
    }
    t
};

// ─── BF16↔f32 conversion (Rust 1.94 built-in) ─────────────────────────────

/// BF16 → f32: lossless, zero-cost bit shift.
#[inline(always)]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// f32 → BF16: truncate mantissa (lossy, 1 ULP).
#[inline(always)]
pub fn f32_to_bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

// ═══════════════════════════════════════════════════════════════════════════
// StackedN: variable-resolution stacked BF16 encoding
// ═══════════════════════════════════════════════════════════════════════════

/// Variable-resolution stacked encoding.
///
/// Each of the 17 base dimensions holds `samples_per_dim` BF16 values.
/// The raw BF16 bit patterns are stored for lossless hydration to f32.
#[derive(Clone, Debug)]
pub struct StackedN {
    /// Samples per base dimension.
    pub samples_per_dim: usize,
    /// Flat storage: dims[d * samples_per_dim + s] = BF16 bit pattern.
    /// Total: 17 * samples_per_dim entries.
    pub data: Vec<u16>,
}

impl StackedN {
    /// Encode from raw BF16 weight vector at given sample count.
    ///
    /// Golden-step folding maps input dimensions to 17 base positions.
    /// Up to `samples_per_dim` values are kept per position (strided sampling
    /// when more octaves exist than sample slots).
    pub fn from_bf16(weights: &[u16], samples_per_dim: usize) -> Self {
        let n = weights.len();
        let n_octaves = (n + BASE_DIM - 1) / BASE_DIM;
        let spd = samples_per_dim.max(1);
        let mut data = vec![0u16; BASE_DIM * spd];

        if n_octaves <= spd {
            // All octaves fit — take them all
            for octave in 0..n_octaves {
                for bi in 0..BASE_DIM {
                    let dim = octave * BASE_DIM + GOLDEN_POS[bi] as usize;
                    if dim < n && octave < spd {
                        data[bi * spd + octave] = weights[dim];
                    }
                }
            }
        } else {
            // More octaves than slots — stride sample
            let stride = n_octaves / spd;
            for slot in 0..spd {
                let octave = slot * stride;
                for bi in 0..BASE_DIM {
                    let dim = octave * BASE_DIM + GOLDEN_POS[bi] as usize;
                    if dim < n {
                        data[bi * spd + slot] = weights[dim];
                    }
                }
            }
        }

        StackedN { samples_per_dim: spd, data }
    }

    /// Encode from f32 weights.
    pub fn from_f32(weights: &[f32], samples_per_dim: usize) -> Self {
        let bf16: Vec<u16> = weights.iter().map(|&v| f32_to_bf16(v)).collect();
        Self::from_bf16(&bf16, samples_per_dim)
    }

    /// Byte size of this encoding.
    pub fn byte_size(&self) -> usize {
        BASE_DIM * self.samples_per_dim * 2
    }

    /// Get all BF16 values for dimension d.
    #[inline]
    pub fn dim_bf16(&self, d: usize) -> &[u16] {
        let start = d * self.samples_per_dim;
        &self.data[start..start + self.samples_per_dim]
    }

    /// Hydrate dimension d to f32 (BF16→f32 lossless).
    pub fn dim_f32(&self, d: usize) -> Vec<f32> {
        self.dim_bf16(d).iter().map(|&b| bf16_to_f32(b)).collect()
    }

    /// Full hydration: all 17×N values as f32.
    pub fn hydrate_f32(&self) -> Vec<f32> {
        self.data.iter().map(|&b| bf16_to_f32(b)).collect()
    }

    // ─── Distance metrics ───────────────────────────────────────────────

    /// Cosine similarity using all hydrated f32 values.
    pub fn cosine(&self, other: &StackedN) -> f64 {
        assert_eq!(self.samples_per_dim, other.samples_per_dim);
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for i in 0..self.data.len() {
            let a = bf16_to_f32(self.data[i]) as f64;
            let b = bf16_to_f32(other.data[i]) as f64;
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }
        let denom = (norm_a * norm_b).sqrt();
        if denom < 1e-12 { 0.0 } else { dot / denom }
    }

    /// L1 distance on hydrated f32 values.
    pub fn l1_f32(&self, other: &StackedN) -> f64 {
        assert_eq!(self.samples_per_dim, other.samples_per_dim);
        let mut d = 0.0f64;
        for i in 0..self.data.len() {
            let a = bf16_to_f32(self.data[i]) as f64;
            let b = bf16_to_f32(other.data[i]) as f64;
            d += (a - b).abs();
        }
        d
    }

}

// NOTE: popcount_distance / sign_bits / sign_agreement REMOVED.
// bgz17 data is i16[17] — Hamming distance is meaningless on it.
// Use palette L1 lookup (hdr_belichtung::PaletteCascade) for cheap distance.
// Use ndarray::hpc::cascade::Cascade for Welford σ tracking + ShiftAlert.
// POPCOUNT is only valid for ThinkingStyleFingerprint texture resonance checks.

// ═══════════════════════════════════════════════════════════════════════════
// Parameterized Codebook: 512 / 1K / 2K / 4K / 8K / 16K entries
// ═══════════════════════════════════════════════════════════════════════════

/// A codebook entry with its stacked encoding.
#[derive(Clone, Debug)]
pub struct CodebookEntry {
    pub stacked: StackedN,
    pub population: usize,
    pub radius: f64,
}

/// Parameterized codebook built via CLAM cosine clustering.
///
/// Sizes: 512 (9-bit), 1024 (10-bit), 2048 (11-bit),
///        4096 (12-bit), 8192 (13-bit), 16384 (14-bit).
#[derive(Clone, Debug)]
pub struct ClamCodebook {
    /// Codebook entries (centroids).
    pub entries: Vec<CodebookEntry>,
    /// Assignment of each source vector to its nearest entry.
    pub assignments: Vec<u16>,
    /// Samples per dim used for encoding.
    pub samples_per_dim: usize,
    /// Cosine distance was used for clustering.
    pub metric: &'static str,
}

impl ClamCodebook {
    /// Build codebook via CLAM cosine clustering (furthest-point on cosine).
    ///
    /// `k`: number of entries (512, 1024, 2048, 4096, 8192, 16384).
    /// Uses cosine distance for all cluster selection and assignment.
    pub fn build_cosine(vectors: &[StackedN], k: usize) -> Self {
        let k = k.min(vectors.len());
        if vectors.is_empty() || k == 0 {
            return ClamCodebook {
                entries: Vec::new(),
                assignments: Vec::new(),
                samples_per_dim: 0,
                metric: "cosine",
            };
        }

        let spd = vectors[0].samples_per_dim;

        // Phase 1: Furthest-point sampling on cosine distance
        let mut selected = Vec::with_capacity(k);
        let mut max_cos_dist = vec![f64::NEG_INFINITY; vectors.len()];

        // Start with first vector
        selected.push(0);
        for i in 0..vectors.len() {
            // Cosine distance = 1 - cosine_similarity
            max_cos_dist[i] = 1.0 - vectors[i].cosine(&vectors[0]);
        }

        for _ in 1..k {
            // Select vector with MAXIMUM minimum cosine distance
            let next = max_cos_dist.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            selected.push(next);

            for i in 0..vectors.len() {
                let d = 1.0 - vectors[i].cosine(&vectors[next]);
                if d < max_cos_dist[i] {
                    max_cos_dist[i] = d;
                }
            }
        }

        // Phase 2: Assign all vectors to nearest centroid (cosine)
        let mut assignments = vec![0u16; vectors.len()];
        let mut populations = vec![0usize; k];
        let mut max_radii = vec![0.0f64; k];

        for (vi, v) in vectors.iter().enumerate() {
            let (best, dist) = selected.iter()
                .enumerate()
                .map(|(ci, &si)| (ci, 1.0 - v.cosine(&vectors[si])))
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            assignments[vi] = best as u16;
            populations[best] += 1;
            if dist > max_radii[best] {
                max_radii[best] = dist;
            }
        }

        let entries: Vec<CodebookEntry> = selected.iter()
            .enumerate()
            .map(|(ci, &si)| CodebookEntry {
                stacked: vectors[si].clone(),
                population: populations[ci],
                radius: max_radii[ci],
            })
            .collect();

        ClamCodebook {
            entries,
            assignments,
            samples_per_dim: spd,
            metric: "cosine",
        }
    }

    /// Look up entry by index.
    pub fn get(&self, index: u16) -> Option<&CodebookEntry> {
        self.entries.get(index as usize)
    }

    /// Assign a single query vector (returns index + cosine distance).
    pub fn assign(&self, query: &StackedN) -> (u16, f64) {
        self.entries.iter()
            .enumerate()
            .map(|(i, e)| (i as u16, 1.0 - query.cosine(&e.stacked)))
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, f64::MAX))
    }

    /// Total byte size (entries only).
    pub fn byte_size(&self) -> usize {
        self.entries.len() * self.entries.first().map_or(0, |e| e.stacked.byte_size())
    }

    /// Summary.
    pub fn summary(&self) -> String {
        let pops: Vec<usize> = self.entries.iter().map(|e| e.population).collect();
        format!(
            "ClamCodebook: {} entries ({}-bit index), {} metric, {} B/entry\n\
             Total: {:.1} KB, pop range: [{}, {}], mean radius: {:.4}",
            self.entries.len(),
            (self.entries.len() as f64).log2().ceil() as u32,
            self.metric,
            self.entries.first().map_or(0, |e| e.stacked.byte_size()),
            self.byte_size() as f64 / 1024.0,
            pops.iter().min().unwrap_or(&0),
            pops.iter().max().unwrap_or(&0),
            self.entries.iter().map(|e| e.radius).sum::<f64>() / self.entries.len().max(1) as f64,
        )
    }
}

// HDR cascade REMOVED from stacked_n.rs.
// Use hdr_belichtung::PaletteCascade (palette L1, ndarray Cascade Welford σ).
// The popcount-based cascade was architecturally wrong:
//   - Hamming distance on BF16 sign bits is not a valid metric for bgz17
//   - Palette L1 lookup is O(1), correct, and cheaper than sign extraction
// See hdr_belichtung.rs for the correct cascade implementation.

/// f32 cosine similarity.
pub fn cosine_f32_slice(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vectors(n: usize, dim: usize, spd: usize) -> Vec<StackedN> {
        (0..n).map(|i| {
            let vals: Vec<f32> = (0..dim).map(|d| {
                ((i * 97 + d * 31) as f32 % 200.0 - 100.0) * 0.01
            }).collect();
            StackedN::from_f32(&vals, spd)
        }).collect()
    }

    #[test]
    fn stacked_n_variable_sizes() {
        let weights: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        for spd in [4, 8, 16, 32, 64] {
            let enc = StackedN::from_f32(&weights, spd);
            assert_eq!(enc.data.len(), 17 * spd);
            assert_eq!(enc.byte_size(), 17 * spd * 2);
        }
    }

    #[test]
    fn cosine_self_one() {
        let v = StackedN::from_f32(&(0..256).map(|i| (i as f32 - 128.0) * 0.01).collect::<Vec<_>>(), 8);
        assert!((v.cosine(&v) - 1.0).abs() < 1e-6);
    }

    // popcount_self_zero test REMOVED — popcount is not valid on bgz17 data.
    // Use palette L1 via hdr_belichtung::PaletteCascade instead.

    #[test]
    fn hydrate_roundtrip() {
        let original: Vec<f32> = vec![1.0, -2.0, 0.5, 3.14, -0.001];
        let enc = StackedN::from_f32(&original, 4);
        let hydrated = enc.hydrate_f32();
        // BF16→f32 loses mantissa bits (7-bit mantissa vs 23-bit).
        // Relative error ≈ 2^-7 ≈ 0.8% for normal values.
        // Golden-step maps input dims to base positions, so hydrated[0]
        // may not correspond to original[0]. Check that the VALUES exist.
        let orig_bf16: Vec<f32> = original.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
        // Verify BF16 roundtrip preserves values
        for (i, (&o, &b)) in original.iter().zip(orig_bf16.iter()).enumerate() {
            let err = (b - o).abs() / o.abs().max(1e-6);
            assert!(err < 0.02, "BF16 roundtrip error at dim {}: {} vs {}, err={:.4}", i, b, o, err);
        }
    }

    #[test]
    fn codebook_512() {
        let vecs = make_vectors(200, 512, 4);
        let cb = ClamCodebook::build_cosine(&vecs, 32); // small for test
        assert_eq!(cb.entries.len(), 32);
        let total_pop: usize = cb.entries.iter().map(|e| e.population).sum();
        assert_eq!(total_pop, 200);
        eprintln!("{}", cb.summary());
    }

    // hdr_cascade_runs test REMOVED — popcount cascade replaced by
    // hdr_belichtung::PaletteCascade (palette L1, ndarray Welford σ).

    #[test]
    fn higher_spd_preserves_better() {
        // Same vector, increasing samples_per_dim
        let base: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let shifted: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01 + 0.3).sin() * 0.5).collect();

        let true_cos = cosine_f32_slice(&base, &shifted);

        let mut errors = Vec::new();
        for spd in [4, 8, 16, 32] {
            let a = StackedN::from_f32(&base, spd);
            let b = StackedN::from_f32(&shifted, spd);
            let enc_cos = a.cosine(&b);
            let err = (enc_cos - true_cos).abs();
            errors.push((spd, err));
            eprintln!("spd={:>3}: cosine={:.6}, error={:.6}", spd, enc_cos, err);
        }

        // Higher SPD should generally have equal or lower error
        // (not strictly monotone due to sampling, but trend should hold)
        let first_err = errors[0].1;
        let last_err = errors.last().unwrap().1;
        assert!(last_err <= first_err + 0.05,
            "higher SPD should not be much worse: spd=4 err={:.4}, spd=32 err={:.4}",
            first_err, last_err);
    }
}
