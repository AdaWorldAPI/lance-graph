//! Shared support for the D-CTX probes (included with `#[path]`, after
//! `fisher_relation`): the 16 x 16 Morton tile and the virtual surfel reading
//! proven in D-CTX-1 (`virtual_surfel_probe`). One copy, so the rounds that
//! render from it read exactly what D-CTX-1 tested.
#![allow(dead_code)]

use cognitive_shader_driver::palette_perturbation::PaletteState;
use lance_graph_contract::morton8x8::Morton8x8;
use lance_graph_contract::sigma_propagation::Spd2;

use crate::fisher_relation::{PairwiseFisherZ, Relation, MOORE};

// ── geometry ───────────────────────────────────────────────────────────────

/// Grid side; the tile is the 256-code Morton prefix.
pub const SIDE: u8 = 16;
/// Pixels in the tile.
pub const PIXELS: usize = 256;

/// A 16 x 16 tile: one palette byte per pixel, lane = Morton code.
pub type Tile = [u8; PIXELS];

/// The neighbour of `center` at `(dx, dy)` inside the tile, or `None`.
#[inline(always)]
pub fn neighbor(center: Morton8x8, dx: i8, dy: i8) -> Option<Morton8x8> {
    center
        .checked_offset(dx, dy)
        .filter(|n| n.code() < PIXELS as u16)
}

// ── the reading ────────────────────────────────────────────────────────────

/// One virtual surfel, as read. Copy, register-sized, never stored by B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfelReading {
    /// The pixel address. Geometry comes from here and nowhere else.
    pub center: Morton8x8,
    /// Activation at the center (the orthogonal reading).
    pub amplitude: u8,
    /// Σ w over the present neighbours.
    pub weight: u32,
    /// Second moment of the neighbour offsets, weighted: Σ w·dx², Σ w·dx·dy, Σ w·dy².
    pub sxx: u32,
    pub sxy: i32,
    pub syy: u32,
}

impl SurfelReading {
    /// The weighted second moment as the contract's SPD carrier, for the
    /// rendering rounds. `None` while it is not strictly positive definite.
    pub fn sigma(&self) -> Option<Spd2> {
        let s = Spd2 {
            a: f64::from(self.sxx),
            b: f64::from(self.sxy),
            c: f64::from(self.syy),
        };
        s.is_spd(1e-9).then_some(s)
    }
}

/// The weight one relation contributes. `identity_weight` answers the
/// needle case; the table is never asked for it.
#[inline(always)]
pub fn weight_of(rel: Relation, identity_weight: u16) -> u32 {
    match rel {
        Relation::Identity => u32::from(identity_weight),
        Relation::Pair(r) => (i32::from(r) + 127) as u32,
    }
}

/// Read one surfel: eight checked neighbour reads, folded in registers.
#[inline]
pub fn read_surfel(
    tile: &Tile,
    center: Morton8x8,
    amplitude: u8,
    law: &PairwiseFisherZ<'_>,
    identity_weight: u16,
) -> SurfelReading {
    let c = PaletteState(tile[center.code() as usize]);
    let mut s = SurfelReading {
        center,
        amplitude,
        weight: 0,
        sxx: 0,
        sxy: 0,
        syy: 0,
    };
    for &(dx, dy) in &MOORE {
        let Some(n) = neighbor(center, dx, dy) else {
            continue;
        };
        let w = weight_of(
            law.relation(c, PaletteState(tile[n.code() as usize])),
            identity_weight,
        );
        let (dx, dy) = (i32::from(dx), i32::from(dy));
        s.weight += w;
        s.sxx += w * (dx * dx) as u32;
        s.sxy += w as i32 * dx * dy;
        s.syy += w * (dy * dy) as u32;
    }
    s
}

/// B: hand every activated pixel's surfel to `sink` the moment it is read.
/// Lanes in Morton code order; activation `0` produces no surfel.
pub fn for_each_surfel(
    tile: &Tile,
    activation: &Tile,
    law: &PairwiseFisherZ<'_>,
    identity_weight: u16,
    mut sink: impl FnMut(SurfelReading),
) {
    for code in 0..PIXELS as u16 {
        let amp = activation[code as usize];
        if amp == 0 {
            continue;
        }
        sink(read_surfel(
            tile,
            Morton8x8::from_code(code),
            amp,
            law,
            identity_weight,
        ));
    }
}

// ── fixtures ───────────────────────────────────────────────────────────────

/// What an identity neighbour (same palette byte) weighs: the ceiling of the
/// calibrated range, i.e. as much as the most related distinct pair can.
///
/// DECISION (operator, 2026-10-06): 254. SCOPE: the D-CTX probes. BASIS: with
/// 254 a stripe of one material orients its surfels along itself; with 0 they
/// turn across it (pinned in `virtual_surfel_probe`). REVISIT WHEN: a
/// calibrated identity term exists in the Fisher-Z family itself.
pub const IDENTITY_WEIGHT: u16 = 254;

/// Every palette ordinal exactly once (167 is odd, so `i·167 + 13` is a
/// permutation of `0..256`). No two pixels share material.
pub fn permutation_tile() -> Tile {
    core::array::from_fn(|i| (i as u32 * 167 + 13) as u8)
}

/// A few materials, so identity neighbours occur.
pub fn repeated_tile() -> Tile {
    let mut t = [0u8; PIXELS];
    for y in 0..SIDE {
        for x in 0..SIDE {
            t[Morton8x8::from_xy(x, y).code() as usize] =
                [7, 42, 200, 99][usize::from((x / 3 + y / 5) % 4)];
        }
    }
    t
}

/// Activation with zeros (one lane in five gated).
pub fn activation_fixture() -> Tile {
    core::array::from_fn(|i| ((i * 29 + 7) % 5) as u8)
}
