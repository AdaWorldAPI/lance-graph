//! BF16 band packing and weighted distance.
//!
//! Each of 24 Bark-scale critical bands is packed as one BF16 value (16 bits).
//! BF16 layout: 1 sign + 8 exponent + 7 mantissa.
//! The natural bit weighting IS psychoacoustic:
//!   - Sign error (phase flip) = catastrophic = weight 8
//!   - Exponent error (energy scale) = loud = weight 4 per bit
//!   - Mantissa error (fine detail) = subtle = weight 1 per bit

use crate::BARK_BANDS;

/// Pack f32 to BF16: truncate lower 16 bits of IEEE 754.
#[inline]
pub fn f32_to_bf16(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

/// Unpack BF16 to f32: pad lower 16 bits with zeros.
#[inline]
pub fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

/// Pack 24 f32 band energies into 24 BF16 values.
pub fn pack_bands(energies: &[f32; BARK_BANDS]) -> [u16; BARK_BANDS] {
    let mut packed = [0u16; BARK_BANDS];
    for i in 0..BARK_BANDS {
        packed[i] = f32_to_bf16(energies[i]);
    }
    packed
}

/// Unpack 24 BF16 values back to f32.
pub fn unpack_bands(packed: &[u16; BARK_BANDS]) -> [f32; BARK_BANDS] {
    let mut energies = [0.0f32; BARK_BANDS];
    for i in 0..BARK_BANDS {
        energies[i] = bf16_to_f32(packed[i]);
    }
    energies
}

/// Weighted BF16 Hamming distance across 24 bands.
///
/// For each BF16 value, weights differing bits by significance:
///   sign (bit 15):       8× weight (phase error = perceptually catastrophic)
///   exponent (bits 7-14): 4× weight per bit (energy error = loud)
///   mantissa (bits 0-6):  1× weight per bit (detail error = subtle)
///
/// Returns total weighted distance. Maximum = 24 × (8 + 8×4 + 7×1) = 24 × 47 = 1128.
pub fn bf16_band_distance(a: &[u16; BARK_BANDS], b: &[u16; BARK_BANDS]) -> u32 {
    let mut dist = 0u32;
    for i in 0..BARK_BANDS {
        let xor = a[i] ^ b[i];
        // Sign bit (bit 15): weight 8
        dist += ((xor >> 15) & 1) as u32 * 8;
        // Exponent (bits 7-14): weight 4 per bit
        dist += ((xor >> 7) & 0xFF).count_ones() * 4;
        // Mantissa (bits 0-6): weight 1 per bit
        dist += (xor & 0x7F).count_ones();
    }
    dist
}

/// Raw (unweighted) BF16 Hamming distance. For comparison with weighted.
pub fn bf16_band_hamming_raw(a: &[u16; BARK_BANDS], b: &[u16; BARK_BANDS]) -> u32 {
    let mut dist = 0u32;
    for i in 0..BARK_BANDS {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Extract the sign+exponent (9 bits) from each BF16 band.
/// This is the "gain" component (spectral envelope shape).
/// Ignoring mantissa = ~7 bits less precision but captures the shape.
pub fn extract_gain(bands: &[u16; BARK_BANDS]) -> [u16; BARK_BANDS] {
    let mut gain = [0u16; BARK_BANDS];
    for i in 0..BARK_BANDS {
        gain[i] = bands[i] & 0xFF80; // sign + exponent, mantissa zeroed
    }
    gain
}

/// Extract the mantissa (7 bits) from each BF16 band.
/// This is the "shape" component (fine spectral detail).
pub fn extract_shape(bands: &[u16; BARK_BANDS]) -> [u16; BARK_BANDS] {
    let mut shape = [0u16; BARK_BANDS];
    for i in 0..BARK_BANDS {
        shape[i] = bands[i] & 0x007F; // mantissa only
    }
    shape
}

/// Total bytes for one frame's band representation.
pub const FRAME_BYTES: usize = BARK_BANDS * 2; // 24 × 2 = 48 bytes

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_roundtrip() {
        for &val in &[0.0f32, 1.0, -1.0, 0.5, 100.0, 0.001, -42.5] {
            let bf16 = f32_to_bf16(val);
            let back = bf16_to_f32(bf16);
            // BF16 has 7-bit mantissa = ~2 decimal digits precision
            let rel_err = if val.abs() > 1e-10 {
                ((back - val) / val).abs()
            } else {
                (back - val).abs()
            };
            assert!(rel_err < 0.01, "BF16 roundtrip error for {}: got {}, err {}", val, back, rel_err);
        }
    }

    #[test]
    fn test_weighted_distance_sign_most_expensive() {
        let a = [0x3F80u16; BARK_BANDS]; // 1.0 in BF16
        let mut b = a;
        
        // Flip only sign bit of band 0
        b[0] ^= 0x8000;
        let d_sign = bf16_band_distance(&a, &b);

        // Flip only one exponent bit of band 0
        b = a;
        b[0] ^= 0x0100;
        let d_exp = bf16_band_distance(&a, &b);

        // Flip only one mantissa bit of band 0
        b = a;
        b[0] ^= 0x0001;
        let d_mant = bf16_band_distance(&a, &b);

        assert!(d_sign > d_exp, "Sign error should cost more than exponent");
        assert!(d_exp > d_mant, "Exponent error should cost more than mantissa");
        assert_eq!(d_sign, 8);
        assert_eq!(d_exp, 4);
        assert_eq!(d_mant, 1);
    }

    #[test]
    fn test_gain_shape_separation() {
        let bands: [u16; BARK_BANDS] = core::array::from_fn(|i| 0x4020 + i as u16);
        let gain = extract_gain(&bands);
        let shape = extract_shape(&bands);

        // Gain should have mantissa zeroed
        for g in &gain {
            assert_eq!(g & 0x007F, 0);
        }
        // Shape should have sign+exp zeroed
        for s in &shape {
            assert_eq!(s & 0xFF80, 0);
        }
        // Recombine should equal original
        for i in 0..BARK_BANDS {
            assert_eq!(gain[i] | shape[i], bands[i]);
        }
    }
}
