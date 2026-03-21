//! i16[17] base pattern: the fundamental representation.
//!
//! Each accumulator plane i8[16384] is folded into 17 base dimensions
//! via golden-step traversal (step=11, covers all 17 positions).
//! Stored as i16 fixed-point (×256) for sub-unit precision.

use crate::{BASE_DIM, FULL_DIM, FP_SCALE, GOLDEN_STEP};

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

/// Number of octaves.
const N_OCTAVES: usize = (FULL_DIM + BASE_DIM - 1) / BASE_DIM;

/// 17-dimensional base pattern. 34 bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Base17 {
    pub dims: [i16; BASE_DIM],
}

impl Base17 {
    pub const BYTE_SIZE: usize = BASE_DIM * 2; // 34

    /// Encode i8[16384] → Base17.
    pub fn encode(acc: &[i8]) -> Self {
        assert!(acc.len() >= FULL_DIM);
        let mut sum = [0i64; BASE_DIM];
        let mut count = [0u32; BASE_DIM];

        for octave in 0..N_OCTAVES {
            for bi in 0..BASE_DIM {
                let dim = octave * BASE_DIM + GOLDEN_POS[bi] as usize;
                if dim < FULL_DIM {
                    sum[bi] += acc[dim] as i64;
                    count[bi] += 1;
                }
            }
        }

        let mut dims = [0i16; BASE_DIM];
        for d in 0..BASE_DIM {
            if count[d] > 0 {
                let mean = sum[d] as f64 / count[d] as f64;
                dims[d] = (mean * FP_SCALE).round().clamp(-32768.0, 32767.0) as i16;
            }
        }
        Base17 { dims }
    }

    /// L1 (Manhattan) distance. The core metric, matches ZeckF64 pipeline.
    #[inline]
    pub fn l1(&self, other: &Base17) -> u32 {
        let mut d = 0u32;
        for i in 0..BASE_DIM {
            d += (self.dims[i] as i32 - other.dims[i] as i32).unsigned_abs();
        }
        d
    }

    /// Sign-bit agreement (out of 17).
    #[inline]
    pub fn sign_agreement(&self, other: &Base17) -> u32 {
        let mut a = 0u32;
        for i in 0..BASE_DIM {
            if (self.dims[i] >= 0) == (other.dims[i] >= 0) {
                a += 1;
            }
        }
        a
    }

    /// All-zero pattern.
    pub fn zero() -> Self {
        Base17 { dims: [0i16; BASE_DIM] }
    }

    /// Serialize to 34 bytes.
    pub fn to_bytes(&self) -> [u8; Self::BYTE_SIZE] {
        let mut buf = [0u8; Self::BYTE_SIZE];
        for i in 0..BASE_DIM {
            let b = self.dims[i].to_le_bytes();
            buf[i * 2] = b[0];
            buf[i * 2 + 1] = b[1];
        }
        buf
    }

    /// Deserialize from 34 bytes.
    pub fn from_bytes(buf: &[u8]) -> Self {
        assert!(buf.len() >= Self::BYTE_SIZE);
        let mut dims = [0i16; BASE_DIM];
        for i in 0..BASE_DIM {
            dims[i] = i16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
        }
        Base17 { dims }
    }
}

/// A complete S/P/O edge as three Base17 patterns. 102 bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpoBase17 {
    pub subject: Base17,
    pub predicate: Base17,
    pub object: Base17,
}

impl SpoBase17 {
    pub const BYTE_SIZE: usize = Base17::BYTE_SIZE * 3; // 102

    /// Encode three i8[16384] planes.
    pub fn encode(s: &[i8], p: &[i8], o: &[i8]) -> Self {
        SpoBase17 {
            subject: Base17::encode(s),
            predicate: Base17::encode(p),
            object: Base17::encode(o),
        }
    }

    /// Combined L1 distance (sum of three planes).
    #[inline]
    pub fn l1(&self, other: &SpoBase17) -> u32 {
        self.subject.l1(&other.subject)
            + self.predicate.l1(&other.predicate)
            + self.object.l1(&other.object)
    }

    /// Per-plane L1 distances.
    #[inline]
    pub fn l1_per_plane(&self, other: &SpoBase17) -> (u32, u32, u32) {
        (
            self.subject.l1(&other.subject),
            self.predicate.l1(&other.predicate),
            self.object.l1(&other.object),
        )
    }

    /// Compute ZeckF64 scent byte from base patterns.
    pub fn scent(&self, other: &SpoBase17) -> u8 {
        let (ds, dp, d_o) = self.l1_per_plane(other);
        let max_l1 = BASE_DIM as u32 * u16::MAX as u32;
        let threshold = max_l1 / 2;

        let sc = (ds < threshold) as u8;
        let pc = (dp < threshold) as u8;
        let oc = (d_o < threshold) as u8;
        sc | (pc << 1) | (oc << 2)
            | ((sc & pc) << 3) | ((sc & oc) << 4)
            | ((pc & oc) << 5) | ((sc & pc & oc) << 6)
    }

    /// Compute full ZeckF64 u64 from base patterns.
    pub fn zeckf64(&self, other: &SpoBase17) -> u64 {
        let (ds, dp, d_o) = self.l1_per_plane(other);
        let max = BASE_DIM as u64 * u16::MAX as u64;
        let threshold = (max / 2) as u32;

        let sc = (ds < threshold) as u8;
        let pc = (dp < threshold) as u8;
        let oc = (d_o < threshold) as u8;
        let byte0 = sc | (pc << 1) | (oc << 2)
            | ((sc & pc) << 3) | ((sc & oc) << 4)
            | ((pc & oc) << 5) | ((sc & pc & oc) << 6);

        let q1 = |d: u32| ((d as u64 * 255) / max).min(255) as u8;
        let q2 = |a: u32, b: u32| (((a as u64 + b as u64) * 255) / (2 * max)).min(255) as u8;
        let q3 = |a: u32, b: u32, c: u32| (((a as u64 + b as u64 + c as u64) * 255) / (3 * max)).min(255) as u8;

        (byte0 as u64)
            | ((q3(ds, dp, d_o) as u64) << 8)
            | ((q2(dp, d_o) as u64) << 16)
            | ((q2(ds, d_o) as u64) << 24)
            | ((q2(ds, dp) as u64) << 32)
            | ((q1(d_o) as u64) << 40)
            | ((q1(dp) as u64) << 48)
            | ((q1(ds) as u64) << 56)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_coverage() {
        let mut seen = [false; BASE_DIM];
        for &p in &GOLDEN_POS { seen[p as usize] = true; }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn test_l1_self_zero() {
        let a = Base17 { dims: [100, -50, 0, 127, -128, 1, -1, 50, 25, -25, 0, 0, 0, 0, 0, 0, 0] };
        assert_eq!(a.l1(&a), 0);
    }

    #[test]
    fn test_l1_symmetric() {
        let a = Base17 { dims: [100; BASE_DIM] };
        let b = Base17 { dims: [-100; BASE_DIM] };
        assert_eq!(a.l1(&b), b.l1(&a));
    }

    #[test]
    fn test_spo_self_scent() {
        let edge = SpoBase17 {
            subject: Base17 { dims: [100; BASE_DIM] },
            predicate: Base17 { dims: [-50; BASE_DIM] },
            object: Base17 { dims: [25; BASE_DIM] },
        };
        assert_eq!(edge.scent(&edge) & 0x7F, 0x7F);
    }

    #[test]
    fn test_byte_roundtrip() {
        let a = Base17 { dims: [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17] };
        let bytes = a.to_bytes();
        let b = Base17::from_bytes(&bytes);
        assert_eq!(a, b);
    }
}
