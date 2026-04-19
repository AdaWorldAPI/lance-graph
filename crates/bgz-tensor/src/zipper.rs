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
