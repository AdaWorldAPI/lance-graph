//! Shared support for the D-CTX rendering probes (included with `#[path]`,
//! after `fisher_relation` and `virtual_surfel`): the isotropic EWA law
//! proven in D-CTX-2 (`ewa_render_probe`). Probe-local by decision: `Spd2` and
//! `ewa_sandwich` keep their certified meaning and are only used here. One
//! copy, so later rounds render exactly what D-CTX-2 tested.
#![allow(dead_code)]

use std::f64::consts::PI;

use lance_graph_contract::morton8x8::Morton8x8;
use lance_graph_contract::sigma_propagation::{ewa_sandwich, Spd2};

use crate::fisher_relation::PairwiseFisherZ;
use crate::virtual_surfel::{
    for_each_surfel, neighbor, SurfelReading, Tile, IDENTITY_WEIGHT, PIXELS,
};

/// Footprint window radius: 7 x 7 offsets around the center.
pub const RADIUS: i8 = 3;
/// Largest relation mass a surfel can have: 8 neighbours at 254.
pub const MAX_MASS: f64 = 8.0 * 254.0;

/// The transient rendered surface, one `f64` per pixel, lane = Morton code.
pub type Field = [f64; PIXELS];

/// Isotropic scale of a reading: `(Σxx + Σyy) / 2W`. `None` when `W = 0`.
#[inline]
pub fn isotropic_scale(r: &SurfelReading) -> Option<f64> {
    (r.weight > 0).then(|| f64::from(r.sxx + r.syy) / (2.0 * f64::from(r.weight)))
}

/// The footprint covariance through the contract's sandwich: `√s·I · I · √s·I`.
#[inline]
pub fn footprint_sigma(s: f64) -> Spd2 {
    let m = Spd2 {
        a: s.sqrt(),
        b: 0.0,
        c: s.sqrt(),
    };
    ewa_sandwich(&m, &Spd2::I)
}

/// Splat amplitude: activation times the surfel's relation mass fraction.
#[inline]
pub fn amplitude(r: &SurfelReading) -> f64 {
    f64::from(r.amplitude) * f64::from(r.weight) / MAX_MASS
}

/// The normalized 2-D Gaussian density of `Σ` at offset `(dx, dy)`.
#[inline]
pub fn footprint(sigma: &Spd2, dx: f64, dy: f64) -> f64 {
    let det = sigma.det();
    let q = (sigma.c * dx * dx - 2.0 * sigma.b * dx * dy + sigma.a * dy * dy) / det;
    (-0.5 * q).exp() / (2.0 * PI * det.sqrt())
}

/// Scatter one footprint into the field over the clipped window.
#[inline]
pub fn splat(field: &mut Field, center: Morton8x8, sigma: &Spd2, amp: f64) {
    for dy in -RADIUS..=RADIUS {
        for dx in -RADIUS..=RADIUS {
            if let Some(p) = neighbor(center, dx, dy) {
                field[p.code() as usize] += amp * footprint(sigma, f64::from(dx), f64::from(dy));
            }
        }
    }
}

/// Read every activated surfel and accumulate its footprint at once.
pub fn render_isotropic(tile: &Tile, act: &Tile, law: &PairwiseFisherZ<'_>, field: &mut Field) {
    for_each_surfel(tile, act, law, IDENTITY_WEIGHT, |r| {
        if let Some(s) = isotropic_scale(&r) {
            splat(field, r.center, &footprint_sigma(s), amplitude(&r));
        }
    });
}

pub fn max_abs(f: &Field) -> f64 {
    f.iter().fold(0.0, |m, v| m.max(v.abs()))
}
