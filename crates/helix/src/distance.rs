//! The metric-safe distance surface: a 256×256 L1 lookup table over the linear
//! endpoint index order.
//!
//! **Layer discipline (inherited from `bgz17/KNOWLEDGE.md`).** L1 on a linear
//! index order IS a metric — the triangle inequality `d(a,c) ≤ d(a,b)+d(b,c)`
//! holds by construction — so it is safe for CAKES / CLAM pruning bounds. This is
//! the property Scent does NOT have (Hamming on a 7-bit lattice) and Palette /
//! Base17 / this table DO have (L1 on a linear order). The raw-azimuth angular
//! pre-filter ([`crate::ResidueEdge::distance_heuristic`]) is NOT a metric (the
//! 2π wrap) and must never feed CAKES bounds.
//!
//! 256×256 × 2 bytes = 128 KB — fits L1/L2 cache, O(1) lookup, `U8x64`-friendly.
use crate::constants::PALETTE_SIZE;
use crate::quantize::RollingFloor;

/// A 256×256 distance lookup table over the palette index order.
#[derive(Debug, Clone)]
pub struct DistanceLut {
    table: Vec<u16>,
}

impl DistanceLut {
    /// Linear-order L1 table: `distance(a, b) = |a − b|`. A metric — safe for
    /// CAKES/CLAM pruning bounds. This is the canonical residue distance.
    pub fn linear() -> Self {
        let mut table = vec![0u16; PALETTE_SIZE * PALETTE_SIZE];
        for a in 0..PALETTE_SIZE {
            for b in 0..PALETTE_SIZE {
                table[a * PALETTE_SIZE + b] = (a as i32 - b as i32).unsigned_abs() as u16;
            }
        }
        Self { table }
    }

    /// L1 over the floor's real bucket centers, scaled to the same `[0, 255]`
    /// integer range as [`Self::linear`]. Still a metric (L1 on a linear real
    /// order), but it reflects the (possibly non-uniform after rolling) bucket
    /// spacing rather than assuming uniform buckets.
    pub fn from_floor(floor: &RollingFloor) -> Self {
        let (lo, hi) = floor.bounds();
        let span = (hi - lo).abs().max(f64::MIN_POSITIVE);
        let mut table = vec![0u16; PALETTE_SIZE * PALETTE_SIZE];
        for a in 0..PALETTE_SIZE {
            for b in 0..PALETTE_SIZE {
                let d = (floor.bucket_center(a as u8) - floor.bucket_center(b as u8)).abs();
                table[a * PALETTE_SIZE + b] = (d / span * 255.0).round().min(65535.0) as u16;
            }
        }
        Self { table }
    }

    /// **Circular** (ring) order table: `distance(a, b) = min(|a − b|, 256 − |a − b|)`
    /// — the cycle-graph geodesic on `Z_256`. Also a metric (verified exhaustively
    /// over all 256³ triples by `circular_satisfies_triangle_inequality`), so it
    /// IS safe for CAKES / CLAM pruning bounds.
    ///
    /// **Use this for any quantity whose range WRAPS** — a bearing, a phase, an
    /// angle — where [`Self::linear`] is simply the wrong table: it puts index
    /// 255 and index 0 at *maximum* distance when they are in fact adjacent
    /// (`d_linear(255,0) = 255` vs `d_circular(255,0) = 1`). That failure, not
    /// any property of angles themselves, is what the module note above means by
    /// *"the raw-azimuth angular pre-filter is NOT a metric (the 2π wrap)"* —
    /// [`crate::ResidueEdge::distance_heuristic`] compares raw bytes with no
    /// table at all. A wrapping quantity **quantised into the 256-palette and
    /// given THIS table** is metric-safe and stays in the index domain.
    ///
    /// Same amortization as every other constructor: the `[a, b]` normalization
    /// is folded in ONCE at build time, so each subsequent comparison is a pure
    /// index lookup — no bounds, no division, no per-element normalization.
    pub fn circular() -> Self {
        let mut table = vec![0u16; PALETTE_SIZE * PALETTE_SIZE];
        let n = PALETTE_SIZE as i32;
        for a in 0..PALETTE_SIZE {
            for b in 0..PALETTE_SIZE {
                let raw = (a as i32 - b as i32).abs();
                table[a * PALETTE_SIZE + b] = raw.min(n - raw) as u16;
            }
        }
        Self { table }
    }

    /// O(1) metric-safe distance lookup.
    #[inline]
    pub fn distance(&self, a: u8, b: u8) -> u16 {
        self.table[a as usize * PALETTE_SIZE + b as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_is_abs_difference() {
        let lut = DistanceLut::linear();
        assert_eq!(lut.distance(5, 5), 0);
        assert_eq!(lut.distance(0, 255), 255);
        assert_eq!(lut.distance(255, 0), 255);
        assert_eq!(lut.distance(10, 7), 3);
    }

    #[test]
    fn linear_is_symmetric_zero_diagonal() {
        let lut = DistanceLut::linear();
        for a in (0..=255u16).step_by(17) {
            assert_eq!(lut.distance(a as u8, a as u8), 0);
            for b in (0..=255u16).step_by(17) {
                assert_eq!(
                    lut.distance(a as u8, b as u8),
                    lut.distance(b as u8, a as u8)
                );
            }
        }
    }

    /// The wrap is the whole point: on a ring, 255 and 0 are ADJACENT.
    /// `linear()` calls them maximally distant — this is the concrete failure
    /// the module note warns about, and the reason a bearing needs `circular()`.
    #[test]
    fn circular_wrap_is_adjacent_where_linear_is_maximal() {
        let circ = DistanceLut::circular();
        let lin = DistanceLut::linear();
        assert_eq!(circ.distance(255, 0), 1, "255 and 0 are adjacent on a ring");
        assert_eq!(
            lin.distance(255, 0),
            255,
            "...but maximal under the linear table"
        );
        // and the far side of the ring is the true maximum, at half a turn
        assert_eq!(circ.distance(0, 128), 128);
        assert_eq!(circ.distance(64, 192), 128);
    }

    /// EXHAUSTIVE metric proof — all 256³ = 16 777 216 triples, not a stride
    /// sample. A circular table is cheap enough to prove outright, and its
    /// metric-safety is exactly the claim that licenses CAKES/CLAM pruning.
    #[test]
    fn circular_satisfies_triangle_inequality() {
        let lut = DistanceLut::circular();
        let mut violations = 0u64;
        for a in 0..PALETTE_SIZE {
            for b in 0..PALETTE_SIZE {
                let dab = lut.distance(a as u8, b as u8) as u32;
                for c in 0..PALETTE_SIZE {
                    let dac = lut.distance(a as u8, c as u8) as u32;
                    let dbc = lut.distance(b as u8, c as u8) as u32;
                    if dac > dab + dbc {
                        violations += 1;
                    }
                }
            }
        }
        assert_eq!(violations, 0, "circular table must be a metric");
    }

    /// Symmetry, identity, and positivity — the rest of the metric axioms.
    #[test]
    fn circular_is_symmetric_with_zero_diagonal_and_positive_off_diagonal() {
        let lut = DistanceLut::circular();
        for a in 0..PALETTE_SIZE {
            assert_eq!(lut.distance(a as u8, a as u8), 0);
            for b in 0..PALETTE_SIZE {
                assert_eq!(
                    lut.distance(a as u8, b as u8),
                    lut.distance(b as u8, a as u8)
                );
                if a != b {
                    assert!(lut.distance(a as u8, b as u8) > 0, "d(a,b)>0 for a!=b");
                }
            }
        }
    }

    #[test]
    fn linear_satisfies_triangle_inequality() {
        // Metric-safety regression (mirrors bgz17's test_palette_triangle_inequality):
        // sample the index cube on a stride-8 grid and assert zero violations.
        let lut = DistanceLut::linear();
        let mut violations = 0u64;
        for a in (0..=255u16).step_by(8) {
            for b in (0..=255u16).step_by(8) {
                for c in (0..=255u16).step_by(8) {
                    let dac = lut.distance(a as u8, c as u8) as u32;
                    let dab = lut.distance(a as u8, b as u8) as u32;
                    let dbc = lut.distance(b as u8, c as u8) as u32;
                    if dac > dab + dbc {
                        violations += 1;
                    }
                }
            }
        }
        assert_eq!(violations, 0, "L1 on a linear order must be a metric");
    }

    #[test]
    fn from_floor_uniform_matches_linear_shape() {
        // A uniform floor's bucket centers are evenly spaced, so from_floor ≈ linear.
        let floor = RollingFloor::uniform(-2.0, 11.0);
        let lut = DistanceLut::from_floor(&floor);
        assert_eq!(lut.distance(7, 7), 0);
        // monotone: farther indices are at least as far in value
        assert!(lut.distance(10, 0) >= lut.distance(5, 0));
    }

    #[test]
    fn from_floor_is_a_metric() {
        let floor = RollingFloor::uniform(-2.0, 11.0);
        let lut = DistanceLut::from_floor(&floor);
        let mut violations = 0u64;
        for a in (0..=255u16).step_by(16) {
            for b in (0..=255u16).step_by(16) {
                for c in (0..=255u16).step_by(16) {
                    let dac = lut.distance(a as u8, c as u8) as u32;
                    let dab = lut.distance(a as u8, b as u8) as u32;
                    let dbc = lut.distance(b as u8, c as u8) as u32;
                    if dac > dab + dbc {
                        violations += 1;
                    }
                }
            }
        }
        assert_eq!(violations, 0);
    }
}
