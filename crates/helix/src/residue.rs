//! The residue carrier — [`ResidueEdge`] (the 3-byte endpoint pair) and
//! [`ResidueEncoder`] (the four-stage pipeline).
//!
//! The object speaks for itself: `encoder.encode(place, n)` runs placement →
//! coupling → Fisher-Z → Euler hand-off → quantise and returns the edge that
//! *is* the residue. `encode` is the compute path (`&self`, read-only); `observe`
//! / `roll` are the calibration paths (`&mut self`) — honouring "no `&mut self`
//! during computation".
use crate::constants::{EULER_GAMMA, GOLDEN_RATIO, LN_17, MODULUS, STRIDE};
use crate::curve_ruler::CurveRuler;
use crate::distance::DistanceLut;
use crate::fisher_z::Similarity;
use crate::placement::{HemispherePoint, Sign};
use crate::quantize::RollingFloor;

/// A residue edge: the `(start, end)` endpoint pair on the φ-spiral curve-ruler,
/// plus the floor-version stamp under which it was quantised. 3 bytes total — the
/// whole curve regenerates from these endpoints (the curve-ruler principle).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidueEdge {
    /// Where the arc BEGINS on the ruler — the quantised PLACE anchor.
    pub start_idx: u8,
    /// Where the arc ENDS — the quantised, hemisphere-placed, Fisher-Z-aligned residue.
    pub end_idx: u8,
    /// Floor-version stamp: "same value → same int8" holds only within one version.
    pub floor_version: u8,
}

impl ResidueEdge {
    /// Serialise to 3 bytes `[start, end, floor_version]`.
    pub fn to_bytes(self) -> [u8; 3] {
        [self.start_idx, self.end_idx, self.floor_version]
    }

    /// Deserialise from 3 bytes.
    pub fn from_bytes(b: [u8; 3]) -> Self {
        Self {
            start_idx: b[0],
            end_idx: b[1],
            floor_version: b[2],
        }
    }

    /// **Metric-safe** distance: L1 over the linear endpoint index order (via the
    /// 256×256 LUT). The triangle inequality holds by construction → safe for
    /// CAKES / CLAM pruning bounds.
    pub fn distance_adaptive(&self, other: &Self, lut: &DistanceLut) -> u32 {
        lut.distance(self.start_idx, other.start_idx) as u32
            + lut.distance(self.end_idx, other.end_idx) as u32
    }

    /// **Heuristic only** byte-Hamming pre-filter on the raw endpoint bytes,
    /// returning `(distance, is_below_threshold)`. NOT a metric (same failure mode
    /// as Scent's 7-bit lattice / the raw-azimuth 2π wrap) — NEVER use for CAKES
    /// bounds; HEEL-stage candidate selection only.
    pub fn distance_heuristic(&self, other: &Self) -> (u32, bool) {
        let d = (self.start_idx ^ other.start_idx).count_ones()
            + (self.end_idx ^ other.end_idx).count_ones();
        (d, d <= 3)
    }
}

/// Signed full-sphere residue — the 24-bit hemisphere [`ResidueEdge`] **doubled
/// to 48 bit (6 bytes)**. Maps a signed magnitude to the FULL sphere: the
/// unsigned hemisphere `rim` edge (rim radius + place anchor via the existing
/// pipeline), the signed `polar` byte (the equal-area lift `y = sign·√(1 − u)`
/// quantised, centred at 128 — `> 128` upper hemisphere, `< 128` lower, so the
/// hemisphere sign is recoverable), and the 16-bit `azimuth` (`n·φ` wrapped to
/// `[0, 2π)` over the full **360°**). Wire layout (LE):
/// `[rim.start, rim.end, rim.floor_version, polar, azimuth_lo, azimuth_hi]`.
///
/// This is the codec the contract `HelixResidue` value-tenant reserves 6 bytes
/// for; the producer writes [`to_bytes`](Signed360::to_bytes). The contract
/// itself is zero-dep and only reserves the bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Signed360 {
    /// Unsigned hemisphere edge (rim radius + place anchor). 3 bytes.
    pub rim: ResidueEdge,
    /// Signed equal-area lift `y` quantised, centred at 128 (128 = equator,
    /// `> 128` = upper hemisphere, `< 128` = lower). 1 byte.
    pub polar: u8,
    /// Golden azimuth `n·φ mod 2π` mapped to `[0, 65536)` over the full 360°. 2 bytes.
    pub azimuth: u16,
}

impl Signed360 {
    /// Serialise to 6 bytes (LE):
    /// `[rim.start, rim.end, rim.floor_version, polar, azimuth_lo, azimuth_hi]`.
    pub fn to_bytes(self) -> [u8; 6] {
        let r = self.rim.to_bytes();
        let a = self.azimuth.to_le_bytes();
        [r[0], r[1], r[2], self.polar, a[0], a[1]]
    }

    /// Deserialise from 6 bytes.
    pub fn from_bytes(b: [u8; 6]) -> Self {
        Self {
            rim: ResidueEdge::from_bytes([b[0], b[1], b[2]]),
            polar: b[3],
            azimuth: u16::from_le_bytes([b[4], b[5]]),
        }
    }

    /// Which hemisphere this residue sits in — recovered from the `polar` byte
    /// (`>= 128` ⇒ upper [`Sign::Pos`], `< 128` ⇒ lower [`Sign::Neg`]).
    pub fn sign(&self) -> Sign {
        if self.polar >= 128 {
            Sign::Pos
        } else {
            Sign::Neg
        }
    }
}

/// The four-stage residue encoder: total residue count `N` + the rolling
/// 256-palette floor.
#[derive(Debug, Clone)]
pub struct ResidueEncoder {
    total: usize,
    floor: RollingFloor,
}

impl ResidueEncoder {
    /// New encoder for `N = total` residues, with the floor auto-seeded so the
    /// bulk lands in-range and the top ~1% rim saturates (the intended
    /// controlled-saturation tail).
    pub fn new(total: usize) -> Self {
        let total = total.max(1);
        let lo = Self::aligned_for_residue(0, total);
        let hi_n = (((total as f64) * 0.99) as usize).min(total - 1);
        let hi = Self::aligned_for_residue(hi_n, total);
        let (lo, hi) = if hi > lo { (lo, hi) } else { (lo, lo + 1.0) };
        let pad = 0.05 * (hi - lo);
        Self {
            total,
            floor: RollingFloor::uniform(lo - pad, hi + pad),
        }
    }

    /// Stage 1+3+4(pre-quant) for residue point `n`: hemisphere rim → Fisher-Z →
    /// Euler hand-off. Floor-independent (pure), so it can seed the bounds.
    ///
    /// Operation order FIXED per `KNOWLEDGE.md` Open Item #2:
    /// `aligned = (z × STRIDE) + γ·(rank/N − ln 17)`.
    fn aligned_for_residue(n: usize, total: usize) -> f64 {
        let p = HemispherePoint::lift(n, total);
        let z = Similarity(p.rim()).fisher_z(); // arctanh(r), r = √u
        let rank_frac = n as f64 / total as f64;
        z * STRIDE as f64 + EULER_GAMMA * (rank_frac - LN_17)
    }

    /// The PLACE anchor's aligned value — where the arc begins on the ruler.
    fn aligned_for_place(place: u64) -> f64 {
        let start = CurveRuler::from_place(place).start_offset();
        let r0 = start as f64 / MODULUS as f64;
        let z = Similarity(r0).fisher_z();
        z * STRIDE as f64 + EULER_GAMMA * (r0 - LN_17)
    }

    /// Encode `(place, n)` into a 3-byte [`ResidueEdge`] — the full pipeline.
    pub fn encode(&self, place: u64, n: usize) -> ResidueEdge {
        let n = n.min(self.total - 1);
        ResidueEdge {
            start_idx: self.floor.quantize(Self::aligned_for_place(place)),
            end_idx: self
                .floor
                .quantize(Self::aligned_for_residue(n, self.total)),
            floor_version: self.floor.version(),
        }
    }

    /// Encode `(place, n, sign)` into a 6-byte [`Signed360`] — the signed
    /// full-sphere residue (the doubled-hemisphere companion to
    /// [`encode`](Self::encode)). The `rim` reuses the unsigned hemisphere
    /// pipeline; `polar` carries the signed equal-area lift `y = sign·√(1 − u)`
    /// (centred at 128, so the hemisphere sign is recoverable via
    /// [`Signed360::sign`]); `azimuth` is the golden angle `n·φ` over the full 360°.
    pub fn encode_signed(&self, place: u64, n: usize, sign: Sign) -> Signed360 {
        let n = n.min(self.total - 1);
        let rim = self.encode(place, n);
        // Signed equal-area lift y ∈ [−1, 1] → byte centred at 128.
        let p = HemispherePoint::signed_lift(n, self.total, sign);
        let polar = (128.0 + p.y * 127.0).round().clamp(0.0, 255.0) as u8;
        // Golden azimuth n·φ wrapped to [0, 2π) → u16 over the full 360°.
        let az = (n as f64 * GOLDEN_RATIO).rem_euclid(core::f64::consts::TAU);
        let azimuth = ((az / core::f64::consts::TAU) * 65536.0) as u16;
        Signed360 {
            rim,
            polar,
            azimuth,
        }
    }

    /// Calibration: feed an observation through the floor's occupancy monitor.
    pub fn observe(&mut self, place: u64, n: usize) {
        let n = n.min(self.total - 1);
        self.floor.observe(Self::aligned_for_place(place));
        self.floor.observe(Self::aligned_for_residue(n, self.total));
    }

    /// Calibration: roll the floor if occupancy drift exceeds the threshold.
    pub fn roll(&mut self) -> bool {
        self.floor.roll()
    }

    /// Borrow the rolling floor (e.g. to build `DistanceLut::from_floor`).
    pub fn floor(&self) -> &RollingFloor {
        &self.floor
    }

    /// The total residue count `N`.
    pub fn total(&self) -> usize {
        self.total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_byte_roundtrip() {
        let e = ResidueEdge {
            start_idx: 21,
            end_idx: 200,
            floor_version: 3,
        };
        assert_eq!(ResidueEdge::from_bytes(e.to_bytes()), e);
    }

    #[test]
    fn encode_is_deterministic() {
        let enc = ResidueEncoder::new(4096);
        assert_eq!(enc.encode(0x1234, 1700), enc.encode(0x1234, 1700));
    }

    #[test]
    fn same_place_same_start() {
        let enc = ResidueEncoder::new(4096);
        let a = enc.encode(0x1234, 100);
        let b = enc.encode(0x1234, 3000);
        assert_eq!(a.start_idx, b.start_idx, "same place ⇒ same arc start");
    }

    #[test]
    fn end_idx_monotonic_in_n() {
        let enc = ResidueEncoder::new(4096);
        let mut prev = 0u8;
        for n in (0..4096).step_by(64) {
            let e = enc.encode(0x1234, n);
            assert!(e.end_idx >= prev, "end_idx must be non-decreasing in n");
            prev = e.end_idx;
        }
    }

    #[test]
    fn adjacent_no_farther_than_distant() {
        let enc = ResidueEncoder::new(4096);
        let lut = DistanceLut::linear();
        let base = enc.encode(0x1234, 1700);
        let near = base.distance_adaptive(&enc.encode(0x1234, 1701), &lut);
        let far = base.distance_adaptive(&enc.encode(0x1234, 4000), &lut);
        assert!(near <= far);
    }

    #[test]
    fn distance_is_symmetric_and_zero_on_self() {
        let enc = ResidueEncoder::new(1024);
        let lut = DistanceLut::linear();
        let a = enc.encode(7, 100);
        let b = enc.encode(9, 800);
        assert_eq!(a.distance_adaptive(&a, &lut), 0);
        assert_eq!(a.distance_adaptive(&b, &lut), b.distance_adaptive(&a, &lut));
    }

    #[test]
    fn encode_clamps_out_of_range_n() {
        let enc = ResidueEncoder::new(256);
        // n >= total must not panic; clamps to total-1.
        assert_eq!(enc.encode(3, 9999), enc.encode(3, 255));
    }

    #[test]
    fn heuristic_returns_flag() {
        let enc = ResidueEncoder::new(1024);
        let a = enc.encode(1, 10);
        let (_d, _below) = a.distance_heuristic(&a);
        assert_eq!(a.distance_heuristic(&a).0, 0);
    }

    // ── Signed360 (signed full-sphere, 48-bit) ───────────────────────────────

    #[test]
    fn signed360_byte_roundtrip_is_6_bytes() {
        let enc = ResidueEncoder::new(4096);
        let s = enc.encode_signed(0x1234, 1700, Sign::Neg);
        assert_eq!(Signed360::from_bytes(s.to_bytes()), s);
        assert_eq!(
            s.to_bytes().len(),
            6,
            "Signed360 is exactly 6 bytes (48 bit)"
        );
    }

    #[test]
    fn signed360_rim_matches_unsigned_encode() {
        let enc = ResidueEncoder::new(4096);
        // The rim edge is the existing unsigned hemisphere encode (sign-independent).
        let rim = enc.encode(0x1234, 1700);
        assert_eq!(enc.encode_signed(0x1234, 1700, Sign::Pos).rim, rim);
        assert_eq!(enc.encode_signed(0x1234, 1700, Sign::Neg).rim, rim);
    }

    #[test]
    fn signed360_sign_recoverable_from_polar() {
        let enc = ResidueEncoder::new(4096);
        for n in [1usize, 100, 1700, 4000] {
            let pos = enc.encode_signed(7, n, Sign::Pos);
            let neg = enc.encode_signed(7, n, Sign::Neg);
            assert_eq!(
                pos.sign(),
                Sign::Pos,
                "Pos ⇒ upper hemisphere (polar ≥ 128)"
            );
            assert_eq!(
                neg.sign(),
                Sign::Neg,
                "Neg ⇒ lower hemisphere (polar < 128)"
            );
            assert!(pos.polar >= 128 && neg.polar < 128);
        }
    }

    #[test]
    fn signed360_azimuth_varies_with_n() {
        let enc = ResidueEncoder::new(4096);
        let a = enc.encode_signed(7, 100, Sign::Pos).azimuth;
        let b = enc.encode_signed(7, 101, Sign::Pos).azimuth;
        assert_ne!(a, b, "consecutive residues get distinct golden azimuths");
    }

    #[test]
    fn signed360_is_deterministic() {
        let enc = ResidueEncoder::new(4096);
        assert_eq!(
            enc.encode_signed(0x99, 2000, Sign::Neg),
            enc.encode_signed(0x99, 2000, Sign::Neg)
        );
    }
}
