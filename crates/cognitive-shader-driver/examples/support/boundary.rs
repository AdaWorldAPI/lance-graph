//! The D-CTX-4 strongest-boundary operator, shared with D-CTX-6.
//!
//! Read-only over a rendered [`Field`]: central-difference gradient, argmax
//! over the inner region, ties to the smaller Morton code. Returns a `Copy`
//! witness and nothing else. See `boundary_measure_probe.rs` for why the
//! inner region is `5..=10`.

#![allow(dead_code)]

use std::ops::RangeInclusive;

use lance_graph_contract::morton8x8::Morton8x8;

use crate::ewa::Field;
use crate::virtual_surfel::PIXELS;

/// Pixels whose value and central difference do not see the tile edge.
pub const INNER: RangeInclusive<u8> = 5..=10;

/// What the operator returns: where the surface is steepest, and how.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryWitness {
    pub at: Morton8x8,
    pub magnitude: f64,
    pub gx: f64,
    pub gy: f64,
}

#[inline]
fn at(f: &Field, p: Morton8x8, dx: i8, dy: i8) -> f64 {
    let n = p
        .checked_offset(dx, dy)
        .expect("inner pixels have all four neighbours");
    f[n.code() as usize]
}

/// One pass in Morton order, one witness register.
pub fn strongest_boundary(f: &Field) -> Option<BoundaryWitness> {
    strongest_boundary_where(f, |_| true)
}

/// As [`strongest_boundary`], over the inner pixels `keep` admits. `keep` is a
/// predicate on the address only; it never sees the field.
pub fn strongest_boundary_where(
    f: &Field,
    keep: impl Fn(Morton8x8) -> bool,
) -> Option<BoundaryWitness> {
    let mut best: Option<BoundaryWitness> = None;
    for code in 0..PIXELS as u16 {
        let p = Morton8x8::from_code(code);
        if !INNER.contains(&p.x()) || !INNER.contains(&p.y()) || !keep(p) {
            continue;
        }
        let gx = (at(f, p, 1, 0) - at(f, p, -1, 0)) / 2.0;
        let gy = (at(f, p, 0, 1) - at(f, p, 0, -1)) / 2.0;
        let magnitude = (gx * gx + gy * gy).sqrt();
        if best.is_none_or(|b| magnitude > b.magnitude) {
            best = Some(BoundaryWitness {
                at: p,
                magnitude,
                gx,
                gy,
            });
        }
    }
    best
}
