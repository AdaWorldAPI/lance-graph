//! Local geometry helpers for `Crystal4096` — the L1 layer of the three-tier model.
//!
//! ## Three-tier model
//!
//! ```text
//! L0  ABI          signed_crystal.rs
//!                    Crystal4096 = compact 12-bit coordinate
//!                    SignedOffset4 = 4-bit signed distance (-7..+7 + overflow)
//!
//! L1  local geometry  crystal_neighborhood.rs   ← THIS FILE
//!                    neighbors_4096(center, radius, metric)
//!                    perturbation tile expansion
//!                    splat candidates
//!
//! L2  graph / DP     blasgraph (future v2)
//!                    frontier propagation over sentence sequence
//!                    basin continuity scoring
//!                    inverse/right-context Pika pass
//! ```
//!
//! L1 computes neighbourhood masks, splats, and traversal around the local
//! signed lattice. It does **not** own the semantic meaning of the coordinates;
//! it only knows the geometry.
//!
//! ## No floats
//!
//! All computations use integer nibble arithmetic. The weights for
//! `LaneCompatible` filtering are `u8` activation levels. L2 (blasgraph) will
//! use `u16` transition costs when it arrives. No f32 in this file.
//!
//! ## Neighbourhood sizes at radius = 1
//!
//! | Metric | Max cells (incl. center) |
//! |--------|--------------------------|
//! | Manhattan | 7 (center + 6 axis-aligned faces) |
//! | Chebyshev | 27 (3^3 = all nibble combinations within ±1 per axis) |
//! | LaneCompatible | ≤ 27 (Chebyshev, then excludes morphologically incompatible) |
//!
//! At radius = 2, Chebyshev gives up to 5^3 = 125 cells, but valid nibbles are
//! 0-14 so overflow cells (nibble 15) are always excluded.

use crate::signed_crystal::{Crystal4096, SignedOffset4};

// ── Neighbourhood metric ──────────────────────────────────────────────────────

/// Distance metric for `Crystal4096` neighbourhood queries.
///
/// All metrics use **nibble-level** distance — each axis is one nibble (0-14
/// valid, 15 = overflow/excluded). No float arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NeighborhoodMetric {
    /// Axis-aligned only: |dx| + |dy| + |dz| ≤ radius.
    /// Radius 1 → 7 cells (center + 6 faces).
    Manhattan,
    /// Cube neighbourhood: max(|dx|, |dy|, |dz|) ≤ radius.
    /// Radius 1 → up to 27 cells (3^3).
    #[default]
    Chebyshev,
    /// Chebyshev filtered by morphological and clause compatibility.
    ///
    /// A neighbor is included only if the X axis (sentence offset) delta
    /// is ≤ the `lane_compat_x_limit` and neither Y nor Z axis is overflow.
    /// This approximates "valid reading transitions" without a full grammar table.
    /// v1 implementation; a full compatibility table is a v2 concern.
    LaneCompatible,
}

// ── Crystal4096Neighbourhood ──────────────────────────────────────────────────

/// Fixed-capacity neighbourhood result for `Crystal4096` queries.
///
/// Holds up to 27 cells (Chebyshev radius 1 in 3D). Stack-allocated, no heap.
pub struct Crystal4096Neighbourhood {
    buf: [Crystal4096; 27],
    len: usize,
}

impl Crystal4096Neighbourhood {
    fn new() -> Self {
        Self {
            buf: [Crystal4096(0); 27],
            len: 0,
        }
    }

    fn push(&mut self, c: Crystal4096) {
        if self.len < 27 {
            self.buf[self.len] = c;
            self.len += 1;
        }
    }

    /// Iterate the neighbourhood cells (including center).
    pub fn iter(&self) -> &[Crystal4096] {
        &self.buf[..self.len]
    }

    /// Total cells including center.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if only the center is present.
    pub fn is_singleton(&self) -> bool {
        self.len == 1
    }
}

// ── Neighbour query ───────────────────────────────────────────────────────────

/// Compute the neighbourhood of `center` within `radius` using `metric`.
///
/// Overflow cells (any nibble = 15) are always excluded. The center is always
/// included as the first element (distance 0). `radius` is clamped to 7
/// (the maximum valid signed offset).
///
/// ```
/// use crate::deepnsm::crystal_neighborhood::{neighbors_4096, NeighborhoodMetric};
/// use crate::deepnsm::signed_crystal::{Crystal4096, SignedOffset4};
///
/// let center = Crystal4096::new(
///     SignedOffset4::ZERO, SignedOffset4::ZERO, SignedOffset4::ZERO,
/// );
/// let nb = neighbors_4096(center, 1, NeighborhoodMetric::Manhattan);
/// assert_eq!(nb.len(), 7); // center + 6 face neighbors
/// ```
pub fn neighbors_4096(
    center: Crystal4096,
    radius: u8,
    metric: NeighborhoodMetric,
) -> Crystal4096Neighbourhood {
    let r = radius.min(7) as i8;
    let mut out = Crystal4096Neighbourhood::new();

    // Decode center nibbles as signed offsets.
    let cx = center.x();
    let cy = center.y();
    let cz = center.z();

    // If center itself has an overflow axis, return it alone.
    if cx.is_overflow() || cy.is_overflow() || cz.is_overflow() {
        out.push(center);
        return out;
    }

    let cx_off = cx.to_offset().unwrap();
    let cy_off = cy.to_offset().unwrap();
    let cz_off = cz.to_offset().unwrap();

    // Center is always first.
    out.push(center);

    for dx in -r..=r {
        for dy in -r..=r {
            for dz in -r..=r {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                } // already pushed

                // Metric filter.
                let in_metric = match metric {
                    NeighborhoodMetric::Manhattan => dx.abs() + dy.abs() + dz.abs() <= r,
                    NeighborhoodMetric::Chebyshev => dx.abs().max(dy.abs()).max(dz.abs()) <= r,
                    NeighborhoodMetric::LaneCompatible => dx.abs().max(dy.abs()).max(dz.abs()) <= r,
                };
                if !in_metric {
                    continue;
                }

                let nx = cx_off + dx;
                let ny = cy_off + dy;
                let nz = cz_off + dz;

                // Clip to valid range (-7..+7); skip overflow cells.
                let sx = SignedOffset4::from_offset(nx);
                let sy = SignedOffset4::from_offset(ny);
                let sz = SignedOffset4::from_offset(nz);

                if sx.is_overflow() || sy.is_overflow() || sz.is_overflow() {
                    continue;
                }

                // LaneCompatible: sentence axis (X) delta ≤ 1.
                if matches!(metric, NeighborhoodMetric::LaneCompatible) && dx.abs() > 1 {
                    continue;
                }

                out.push(Crystal4096::new(sx, sy, sz));
            }
        }
    }
    out
}

/// Chebyshev distance between two `Crystal4096` coordinates.
///
/// Returns `None` if either coordinate has an overflow axis.
/// Returns `Some(distance)` where distance = max(|dx|, |dy|, |dz|) over signed axes.
pub fn chebyshev_distance(a: Crystal4096, b: Crystal4096) -> Option<u8> {
    let (ax, ay, az) = (a.x(), a.y(), a.z());
    let (bx, by, bz) = (b.x(), b.y(), b.z());
    if ax.is_overflow()
        || ay.is_overflow()
        || az.is_overflow()
        || bx.is_overflow()
        || by.is_overflow()
        || bz.is_overflow()
    {
        return None;
    }
    let dx = (ax.to_offset().unwrap() - bx.to_offset().unwrap()).unsigned_abs();
    let dy = (ay.to_offset().unwrap() - by.to_offset().unwrap()).unsigned_abs();
    let dz = (az.to_offset().unwrap() - bz.to_offset().unwrap()).unsigned_abs();
    Some(dx.max(dy).max(dz))
}

/// Manhattan distance between two `Crystal4096` coordinates.
///
/// Returns `None` if either coordinate has an overflow axis.
pub fn manhattan_distance(a: Crystal4096, b: Crystal4096) -> Option<u8> {
    let (ax, ay, az) = (a.x(), a.y(), a.z());
    let (bx, by, bz) = (b.x(), b.y(), b.z());
    if ax.is_overflow()
        || ay.is_overflow()
        || az.is_overflow()
        || bx.is_overflow()
        || by.is_overflow()
        || bz.is_overflow()
    {
        return None;
    }
    let dx = (ax.to_offset().unwrap() - bx.to_offset().unwrap()).unsigned_abs();
    let dy = (ay.to_offset().unwrap() - by.to_offset().unwrap()).unsigned_abs();
    let dz = (az.to_offset().unwrap() - bz.to_offset().unwrap()).unsigned_abs();
    Some(dx + dy + dz)
}

/// Enumerate all valid `Crystal4096` cells — those with no overflow axis.
///
/// 15^3 = 3375 valid cells (nibbles 0-14 on each axis).
/// Returns a fixed-capacity buffer; useful for debug and codebook construction.
pub fn all_valid_cells() -> impl Iterator<Item = Crystal4096> {
    (0u8..15).flat_map(move |x| {
        (0u8..15).flat_map(move |y| {
            (0u8..15).map(move |z| {
                Crystal4096::new(SignedOffset4(x), SignedOffset4(y), SignedOffset4(z))
            })
        })
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signed_crystal::{Crystal4096, SignedOffset4};

    fn zero() -> Crystal4096 {
        Crystal4096::new(
            SignedOffset4::ZERO,
            SignedOffset4::ZERO,
            SignedOffset4::ZERO,
        )
    }

    fn at(x: i8, y: i8, z: i8) -> Crystal4096 {
        Crystal4096::new(
            SignedOffset4::from_offset(x),
            SignedOffset4::from_offset(y),
            SignedOffset4::from_offset(z),
        )
    }

    #[test]
    fn manhattan_radius1_gives_7_cells() {
        let nb = neighbors_4096(zero(), 1, NeighborhoodMetric::Manhattan);
        // center (0,0,0) + 6 face neighbors = 7
        assert_eq!(nb.len(), 7);
    }

    #[test]
    fn chebyshev_radius1_gives_27_cells_at_center() {
        // Center is (0,0,0) — all 26 neighbours + center are within ±7.
        let nb = neighbors_4096(zero(), 1, NeighborhoodMetric::Chebyshev);
        assert_eq!(nb.len(), 27);
    }

    #[test]
    fn chebyshev_radius1_near_boundary_clips_overflow() {
        // Center at (+7,+7,+7) — neighbours in +direction would overflow.
        let c = at(7, 7, 7);
        let nb = neighbors_4096(c, 1, NeighborhoodMetric::Chebyshev);
        // Only cells with x,y,z ∈ {+6,+7} are valid (not +8 which overflows).
        // 2^3 = 8 cells.
        assert_eq!(nb.len(), 8);
        for cell in nb.iter() {
            assert!(!cell.has_overflow());
        }
    }

    #[test]
    fn chebyshev_radius0_gives_singleton() {
        let nb = neighbors_4096(zero(), 0, NeighborhoodMetric::Chebyshev);
        assert_eq!(nb.len(), 1);
        assert!(nb.is_singleton());
    }

    #[test]
    fn lane_compatible_limits_x_axis_to_1() {
        // LaneCompatible: |dx| ≤ 1, |dy|/|dz| ≤ r.
        let nb = neighbors_4096(zero(), 2, NeighborhoodMetric::LaneCompatible);
        for cell in nb.iter() {
            let x_off = cell.x().to_offset().unwrap_or(99);
            assert!(x_off >= -1 && x_off <= 1, "X offset {x_off} exceeds ±1");
        }
    }

    #[test]
    fn overflow_center_returns_singleton() {
        let overflow = Crystal4096::new(
            SignedOffset4::OVERFLOW,
            SignedOffset4::ZERO,
            SignedOffset4::ZERO,
        );
        let nb = neighbors_4096(overflow, 1, NeighborhoodMetric::Chebyshev);
        assert_eq!(nb.len(), 1);
        assert_eq!(nb.iter()[0], overflow);
    }

    #[test]
    fn chebyshev_distance_same_is_zero() {
        let c = at(1, 2, 3);
        assert_eq!(chebyshev_distance(c, c), Some(0));
    }

    #[test]
    fn chebyshev_distance_one_axis() {
        let a = at(0, 0, 0);
        let b = at(3, 0, 0);
        assert_eq!(chebyshev_distance(a, b), Some(3));
    }

    #[test]
    fn chebyshev_distance_multi_axis_is_max() {
        let a = at(0, 0, 0);
        let b = at(2, 3, 1);
        assert_eq!(chebyshev_distance(a, b), Some(3));
    }

    #[test]
    fn manhattan_distance_same_is_zero() {
        let c = at(0, 0, 0);
        assert_eq!(manhattan_distance(c, c), Some(0));
    }

    #[test]
    fn manhattan_distance_multi_axis_is_sum() {
        let a = at(0, 0, 0);
        let b = at(1, 2, 3);
        assert_eq!(manhattan_distance(a, b), Some(6));
    }

    #[test]
    fn overflow_distance_returns_none() {
        let a = at(0, 0, 0);
        let b = Crystal4096::new(
            SignedOffset4::OVERFLOW,
            SignedOffset4::ZERO,
            SignedOffset4::ZERO,
        );
        assert_eq!(chebyshev_distance(a, b), None);
        assert_eq!(manhattan_distance(a, b), None);
    }

    #[test]
    fn all_valid_cells_count() {
        // 15 values per axis (0..14), 3 axes → 15^3 = 3375
        assert_eq!(all_valid_cells().count(), 15 * 15 * 15);
    }

    #[test]
    fn all_valid_cells_no_overflow() {
        for c in all_valid_cells() {
            assert!(!c.has_overflow(), "unexpected overflow cell: {c:?}");
        }
    }

    #[test]
    fn center_is_always_first_in_neighbourhood() {
        let c = at(2, -3, 1);
        let nb = neighbors_4096(c, 1, NeighborhoodMetric::Chebyshev);
        assert_eq!(nb.iter()[0], c);
    }
}
