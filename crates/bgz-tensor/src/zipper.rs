//! Zipper codec — phase + magnitude φ-multiplexed in a single container.
//!
//! Per `.claude/board/IDEAS.md` 2026-04-19 "Zipper codec" + existing
//! `.claude/knowledge/phi-spiral-reconstruction.md` "family zipper" concept.
//!
//! Design:
//! - Phase stream sampled at stride = round(N/φ), extracting sign bits
//!   from a Hadamard-rotated row → `PHASE_ACTIVE_BITS` active bits.
//! - Magnitude stream sampled at stride = round(N/φ²), extracting
//!   i8 quantized coefficients → `MAG_ACTIVE_SAMPLES` samples.
//! - Both strides are maximally-irrational → anti-moiré against the
//!   Hadamard butterfly (X-Trans sensor principle).
//! - Non-collision is mathematical: Zeckendorf non-adjacent Fibonacci
//!   decomposition property guarantees that positions visited by
//!   round(N/φ) and round(N/φ²) do not periodically overlap.
//!
//! Matryoshka truncation: `cosine_phase_only` < `cosine_zipper_full`.
//! Same descriptor serves both truncation levels.

use ndarray::hpc::fft::wht_f32;

/// Active phase bits in the zipper container. bgz17's design places
/// ~48-64 discriminative bits in the 16,384-bit halo; we lock the
/// high end of that range (64) as the explicit phase signal width.
pub const PHASE_ACTIVE_BITS: usize = 64;

/// Active magnitude samples in the zipper container. 56 i8 samples
/// = 448 bits, fitting in the halo alongside the phase bits without
/// stride collisions at the φ² offset.
pub const MAG_ACTIVE_SAMPLES: usize = 56;

/// Total wire size for the zipper descriptor.
/// = 64 bits phase + 56 × 8 bits magnitude = 8 + 56 = 64 bytes.
pub const ZIPPER_BYTES: usize = (PHASE_ACTIVE_BITS / 8) + MAG_ACTIVE_SAMPLES;

/// Golden ratio φ = 1.618033988749...
const PHI: f64 = 1.618_033_988_749_895;
/// φ² = φ + 1 = 2.618...
const PHI_SQ: f64 = 2.618_033_988_749_895;
/// Circle of Fifths stride: log₂(3/2) ≈ 0.58496...
/// Irrational rotation giving harmonic-proximity ordering.
const QUINT_STRIDE: f64 = 0.584_962_500_721_156;

/// μ-law companding parameter (same as telephony/audio μ-law).
/// μ=255 concentrates quantization levels near zero where argmax
/// decisions happen; coarsens at extremes where the answer is obvious.
const MU_LAW: f32 = 255.0;

/// Zipper descriptor: single-container phase + magnitude encoding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZipperDescriptor {
    /// `PHASE_ACTIVE_BITS` sign bits, packed little-endian.
    pub phase_bits: u64,
    /// `MAG_ACTIVE_SAMPLES` i8 quantized magnitude samples.
    pub mag_samples: [i8; MAG_ACTIVE_SAMPLES],
}

impl ZipperDescriptor {
    pub const SIZE_BYTES: usize = ZIPPER_BYTES;

    /// Encode a row into the zipper descriptor:
    /// 1. wht_f32 for orthogonal projection (anti-moiré basis).
    /// 2. Phase stream: PHASE_ACTIVE_BITS sign bits at stride round(N/φ).
    /// 3. Magnitude stream: MAG_ACTIVE_SAMPLES i8 samples at stride round(N/φ²).
    ///    Quantized against the row's own max-abs for per-row i8 range.
    pub fn encode(row: &[f32]) -> Self {
        let n = row.len();
        assert!(
            n.is_power_of_two() && n >= 128,
            "row length must be power of 2 ≥ 128 (phase + mag streams need room), got {n}"
        );

        // Orthogonal basis projection.
        let mut rotated = row.to_vec();
        wht_f32(&mut rotated);

        let phase_stride = (n as f64 / PHI).round() as usize;
        let mag_stride = (n as f64 / PHI_SQ).round() as usize;

        // Phase stream: PHASE_ACTIVE_BITS sign bits, stride-indexed modulo N.
        let mut phase_bits: u64 = 0;
        let mut pos: usize = 0;
        for i in 0..PHASE_ACTIVE_BITS {
            pos = (pos + phase_stride) % n;
            if rotated[pos] >= 0.0 {
                phase_bits |= 1u64 << i;
            }
        }

        // Magnitude stream: MAG_ACTIVE_SAMPLES i8 samples at φ²-stride.
        // Per-row max-abs normalizes magnitudes into [-127, 127].
        let max_abs = rotated.iter().fold(0.0_f32, |m, &x| m.max(x.abs())).max(1e-20);
        let scale = 127.0 / max_abs;

        let mut mag_samples = [0i8; MAG_ACTIVE_SAMPLES];
        let mut mpos: usize = 0;
        for i in 0..MAG_ACTIVE_SAMPLES {
            mpos = (mpos + mag_stride) % n;
            let q = (rotated[mpos] * scale).round().clamp(-127.0, 127.0);
            mag_samples[i] = q as i8;
        }

        Self { phase_bits, mag_samples }
    }

    /// Phase-only similarity — matryoshka truncation level 0.
    /// Hamming agreement between phase bits mapped to [−1, 1].
    pub fn cosine_phase_only(&self, other: &Self) -> f32 {
        let agree = (!(self.phase_bits ^ other.phase_bits)).count_ones() as i32;
        let disagree = PHASE_ACTIVE_BITS as i32 - agree;
        (agree - disagree) as f32 / PHASE_ACTIVE_BITS as f32
    }

    /// Magnitude-only similarity — sum-of-products normalized (cosine).
    pub fn cosine_magnitude_only(&self, other: &Self) -> f32 {
        let mut dot = 0.0_f32;
        let mut na = 0.0_f32;
        let mut nb = 0.0_f32;
        for i in 0..MAG_ACTIVE_SAMPLES {
            let a = self.mag_samples[i] as f32;
            let b = other.mag_samples[i] as f32;
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        let d = (na * nb).sqrt();
        if d < 1e-15 { 0.0 } else { dot / d }
    }

    /// Full zipper similarity — matryoshka truncation level 1.
    /// Weighted sum of phase-agreement + magnitude-cosine.
    /// Weight 0.5/0.5 since both streams carry independent φ-properties.
    pub fn cosine_zipper_full(&self, other: &Self) -> f32 {
        0.5 * self.cosine_phase_only(other) + 0.5 * self.cosine_magnitude_only(other)
    }

}

// ─────────────────────────────────────────────────────────────────────────
// I8 zipper — magnitude+sign per sample instead of sign-only
// ─────────────────────────────────────────────────────────────────────────

/// I8 zipper descriptor: K i8 samples at φ-stride positions.
/// Each sample carries sign AND magnitude → 8× info density vs sign-only.
/// Supports both φ-stride (anti-moiré) and Quintenzirkel-stride (harmonic).
#[derive(Debug, Clone)]
pub struct ZipperI8Descriptor {
    pub samples: Vec<i8>,
    pub stride_kind: StrideKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrideKind {
    Phi,
    Quintenzirkel,
}

impl ZipperI8Descriptor {
    /// Encode K i8 samples from a row at the given stride.
    /// Applies μ-law companding for gamma-corrected quantization:
    /// concentrates precision near zero where argmax discrimination happens.
    pub fn encode(row: &[f32], k: usize, stride_kind: StrideKind) -> Self {
        let n = row.len();
        assert!(n.is_power_of_two() && n >= 64);

        let mut rotated = row.to_vec();
        wht_f32(&mut rotated);

        let stride_frac = match stride_kind {
            StrideKind::Phi => 1.0 / PHI,
            StrideKind::Quintenzirkel => QUINT_STRIDE,
        };

        // Per-row max-abs for normalization to [-1, 1].
        let max_abs = rotated.iter().fold(0.0_f32, |m, &x| m.max(x.abs())).max(1e-20);

        let mut samples = Vec::with_capacity(k);
        for i in 0..k {
            let frac = ((i + 1) as f64 * stride_frac) % 1.0;
            let pos = (frac * n as f64) as usize % n;
            let x = rotated[pos] / max_abs; // normalized to [-1, 1]
            // μ-law companding (gamma correction).
            let compressed = mu_law_encode(x);
            samples.push(compressed);
        }

        Self { samples, stride_kind }
    }

    /// Cosine similarity between two I8 descriptors.
    pub fn cosine(&self, other: &Self) -> f32 {
        let k = self.samples.len().min(other.samples.len());
        let mut dot = 0.0_f32;
        let mut na = 0.0_f32;
        let mut nb = 0.0_f32;
        for i in 0..k {
            let a = self.samples[i] as f32;
            let b = other.samples[i] as f32;
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        let d = (na * nb).sqrt();
        if d < 1e-15 { 0.0 } else { dot / d }
    }

    pub fn bytes_per_row(&self) -> usize { self.samples.len() }
}

/// μ-law encode: x ∈ [-1, 1] → i8 with gamma-concentrated precision.
/// sign(x) * log(1 + μ|x|) / log(1 + μ) → scale to [-127, 127].
fn mu_law_encode(x: f32) -> i8 {
    let sign = if x >= 0.0 { 1.0_f32 } else { -1.0 };
    let compressed = sign * (1.0 + MU_LAW * x.abs()).ln() / (1.0 + MU_LAW).ln();
    (compressed * 127.0).round().clamp(-127.0, 127.0) as i8
}

/// μ-law decode: i8 → f32 ∈ [-1, 1], inverse of mu_law_encode.
#[allow(dead_code)]
fn mu_law_decode(q: i8) -> f32 {
    let y = q as f32 / 127.0;
    let sign = if y >= 0.0 { 1.0_f32 } else { -1.0 };
    sign * (1.0 / MU_LAW) * ((1.0 + MU_LAW).powf(y.abs()) - 1.0)
}

// ─────────────────────────────────────────────────────────────────────────
// 5-level bipolar zipper — Structured5x5 alignment, negative cancellation
// ─────────────────────────────────────────────────────────────────────────

/// 5-level bipolar zipper descriptor. Each sample ∈ {-2, -1, 0, +1, +2}.
/// Uses GLOBAL (population-wide) scale, not per-row max-abs — critical fix
/// for the inter-row magnitude preservation that per-row i8 μ-law destroyed.
///
/// Samples packed 3 bits each; 21 samples → 63 bits → 8 B; 42 → 128 bits → 16 B.
///
/// Bipolar cells support VSA-style bundling with negative cancellation
/// (noise cancels, signal accumulates) when superposing multiple rows.
#[derive(Debug, Clone)]
pub struct Zipper5LevelDescriptor {
    /// Values in {-2, -1, 0, +1, +2}, packed 3 bits each in `packed`.
    pub samples: Vec<i8>,
    pub stride_kind: StrideKind,
}

impl Zipper5LevelDescriptor {
    /// Encode K 5-level samples at given stride using a POPULATION-GLOBAL
    /// scale. This preserves inter-row magnitude relationships — unlike
    /// per-row max-abs normalization which collapses them.
    pub fn encode(row: &[f32], k: usize, stride_kind: StrideKind, global_scale: f32) -> Self {
        let n = row.len();
        assert!(n.is_power_of_two() && n >= 64);
        assert!(global_scale > 0.0, "global_scale must be positive");

        let mut rotated = row.to_vec();
        wht_f32(&mut rotated);

        let stride_frac = match stride_kind {
            StrideKind::Phi => 1.0 / PHI,
            StrideKind::Quintenzirkel => QUINT_STRIDE,
        };

        let mut samples = Vec::with_capacity(k);
        for i in 0..k {
            let frac = ((i + 1) as f64 * stride_frac) % 1.0;
            let pos = (frac * n as f64) as usize % n;
            let normalized = rotated[pos] / global_scale;
            // 5-level signed quantization via thresholds at {-1.5, -0.5, 0.5, 1.5}
            let q = if normalized < -1.5 { -2 }
                    else if normalized < -0.5 { -1 }
                    else if normalized <= 0.5 { 0 }
                    else if normalized <= 1.5 { 1 }
                    else { 2 };
            samples.push(q as i8);
        }

        Self { samples, stride_kind }
    }

    /// Compute population-global scale: median of per-row max-abs,
    /// scaled so ~70% of coefficients land in the middle 3 levels.
    pub fn compute_global_scale(rows: &[Vec<f32>]) -> f32 {
        let mut all_abs: Vec<f32> = Vec::with_capacity(rows.len() * rows[0].len());
        for row in rows {
            let mut rotated = row.clone();
            let n = rotated.len();
            // Pad to pow2 if needed
            if !n.is_power_of_two() {
                let mut p = 1usize;
                while p < n { p <<= 1; }
                rotated.resize(p, 0.0);
            }
            wht_f32(&mut rotated);
            for &c in rotated.iter() {
                all_abs.push(c.abs());
            }
        }
        // Median gives a robust scale; 1.0 × median places 50% of coefs
        // at |normalized| ≤ 1 (the 5 middle levels).
        all_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = all_abs[all_abs.len() / 2];
        med.max(1e-20)
    }

    /// Cosine similarity in the 5-level signed space.
    pub fn cosine(&self, other: &Self) -> f32 {
        let k = self.samples.len().min(other.samples.len());
        let mut dot = 0.0_f32;
        let mut na = 0.0_f32;
        let mut nb = 0.0_f32;
        for i in 0..k {
            let a = self.samples[i] as f32;
            let b = other.samples[i] as f32;
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        let d = (na * nb).sqrt();
        if d < 1e-15 { 0.0 } else { dot / d }
    }

    /// VSA-style bundle with signed accumulation and saturation at ±2.
    /// Noise cancels (opposite signs → 0); signal accumulates.
    pub fn bundle(&self, other: &Self) -> Self {
        let k = self.samples.len().min(other.samples.len());
        let mut samples = Vec::with_capacity(k);
        for i in 0..k {
            let sum = self.samples[i] as i16 + other.samples[i] as i16;
            samples.push(sum.clamp(-2, 2) as i8);
        }
        Self { samples, stride_kind: self.stride_kind }
    }

    pub fn bytes_per_row(k: usize) -> usize {
        // 3 bits per sample, rounded up to bytes.
        (k * 3 + 7) / 8
    }
}

// ─────────────────────────────────────────────────────────────────────────
// 7-level bipolar zipper — 7^7 = 823,543 states per 7-sample tuple
// ─────────────────────────────────────────────────────────────────────────

/// 7-level bipolar descriptor. Values ∈ {-3, -2, -1, 0, +1, +2, +3}.
/// 7 samples = 7^7 states ≈ 20 bits; 21 samples = 3 × 7^7 ≈ 60 bits = 8 B.
/// Finer magnitude discrimination than 5-level; deeper bundling cancellation.
#[derive(Debug, Clone)]
pub struct Zipper7LevelDescriptor {
    pub samples: Vec<i8>,
    pub stride_kind: StrideKind,
}

impl Zipper7LevelDescriptor {
    pub fn encode(row: &[f32], k: usize, stride_kind: StrideKind, global_scale: f32) -> Self {
        let n = row.len();
        assert!(n.is_power_of_two() && n >= 64);
        assert!(global_scale > 0.0);

        let mut rotated = row.to_vec();
        wht_f32(&mut rotated);

        let stride_frac = match stride_kind {
            StrideKind::Phi => 1.0 / PHI,
            StrideKind::Quintenzirkel => QUINT_STRIDE,
        };

        let mut samples = Vec::with_capacity(k);
        for i in 0..k {
            let frac = ((i + 1) as f64 * stride_frac) % 1.0;
            let pos = (frac * n as f64) as usize % n;
            let normalized = rotated[pos] / global_scale;
            // 7-level signed quantization: thresholds at {±0.5, ±1.5, ±2.5}
            let q = if normalized < -2.5 { -3 }
                    else if normalized < -1.5 { -2 }
                    else if normalized < -0.5 { -1 }
                    else if normalized <= 0.5 { 0 }
                    else if normalized <= 1.5 { 1 }
                    else if normalized <= 2.5 { 2 }
                    else { 3 };
            samples.push(q as i8);
        }

        Self { samples, stride_kind }
    }

    pub fn cosine(&self, other: &Self) -> f32 {
        let k = self.samples.len().min(other.samples.len());
        let mut dot = 0.0_f32;
        let mut na = 0.0_f32;
        let mut nb = 0.0_f32;
        for i in 0..k {
            let a = self.samples[i] as f32;
            let b = other.samples[i] as f32;
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        let d = (na * nb).sqrt();
        if d < 1e-15 { 0.0 } else { dot / d }
    }

    pub fn bundle(&self, other: &Self) -> Self {
        let k = self.samples.len().min(other.samples.len());
        let mut samples = Vec::with_capacity(k);
        for i in 0..k {
            let sum = self.samples[i] as i16 + other.samples[i] as i16;
            samples.push(sum.clamp(-3, 3) as i8);
        }
        Self { samples, stride_kind: self.stride_kind }
    }
}

impl ZipperDescriptor {
    pub fn pack(&self) -> [u8; ZIPPER_BYTES] {
        let mut out = [0u8; ZIPPER_BYTES];
        out[0..8].copy_from_slice(&self.phase_bits.to_le_bytes());
        for i in 0..MAG_ACTIVE_SAMPLES {
            out[8 + i] = self.mag_samples[i] as u8;
        }
        out
    }

    pub fn unpack(bytes: [u8; ZIPPER_BYTES]) -> Self {
        let phase_bits = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let mut mag_samples = [0i8; MAG_ACTIVE_SAMPLES];
        for i in 0..MAG_ACTIVE_SAMPLES {
            mag_samples[i] = bytes[8 + i] as i8;
        }
        Self { phase_bits, mag_samples }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 32) as i32 as f32) / i32::MAX as f32 * scale
            })
            .collect()
    }

    #[test]
    fn constants_are_explicit() {
        assert_eq!(PHASE_ACTIVE_BITS, 64);
        assert_eq!(MAG_ACTIVE_SAMPLES, 56);
        assert_eq!(ZIPPER_BYTES, 64);
    }

    #[test]
    fn encode_pack_roundtrip() {
        let row = make_row(1024, 0xABCD, 1.0);
        let d = ZipperDescriptor::encode(&row);
        let bytes = d.pack();
        let d2 = ZipperDescriptor::unpack(bytes);
        assert_eq!(d, d2);
    }

    #[test]
    fn self_similarity_unity() {
        let row = make_row(1024, 0xBEEF, 1.0);
        let d = ZipperDescriptor::encode(&row);
        assert!((d.cosine_phase_only(&d) - 1.0).abs() < 1e-5);
        assert!((d.cosine_magnitude_only(&d) - 1.0).abs() < 1e-5);
        assert!((d.cosine_zipper_full(&d) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn different_rows_lower_similarity() {
        let a = make_row(1024, 1111, 1.0);
        let b = make_row(1024, 2222, 1.0);
        let da = ZipperDescriptor::encode(&a);
        let db = ZipperDescriptor::encode(&b);
        let sim = da.cosine_zipper_full(&db);
        // Independent random rows should not agree strongly.
        assert!(sim.abs() < 0.9, "random rows too similar: {sim}");
    }

    #[test]
    fn sign_flip_inverts_both_streams() {
        // Hadamard is linear: Wx = y → W(-x) = -y. Both streams sign-flip.
        let a = make_row(1024, 7777, 1.0);
        let b: Vec<f32> = a.iter().map(|&x| -x).collect();
        let da = ZipperDescriptor::encode(&a);
        let db = ZipperDescriptor::encode(&b);
        let phase = da.cosine_phase_only(&db);
        let mag = da.cosine_magnitude_only(&db);
        // Phase bits all flip → agreement → -1.
        assert!(phase < -0.95, "flipped row should give ~-1 phase: {phase}");
        // Magnitude samples all negate → cosine → -1 (sign-inverted).
        assert!(mag < -0.95, "flipped row should give ~-1 magnitude cosine: {mag}");
    }

    #[test]
    fn positive_scaling_preserves_both() {
        // Scaling by positive constant → magnitudes scale, signs preserved → cosines 1.
        let a = make_row(1024, 9999, 1.0);
        let b: Vec<f32> = a.iter().map(|&x| x * 2.5).collect();
        let da = ZipperDescriptor::encode(&a);
        let db = ZipperDescriptor::encode(&b);
        assert!(da.cosine_phase_only(&db) > 0.99);
        assert!(da.cosine_magnitude_only(&db) > 0.99);
    }
}
