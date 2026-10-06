//! D-CTX-4: one read-only measurement over the rendered surface.
//!
//! Claim under test: a cognitive operator can read the transient field that
//! D-CTX-2 renders and return a small, fixed witness, without producing a
//! second field and without writing anything back.
//!
//! # The operator: strongest boundary
//!
//! ```text
//! g(p)  = ((f(x+1) − f(x−1)) / 2, (f(y+1) − f(y−1)) / 2)     central differences
//! |g|   = √(gx² + gy²)
//! out   = argmax |g| over the inner region → BoundaryWitness { at, |g|, gx, gy }
//! ```
//!
//! Ties go to the smaller Morton code. The witness is `Copy`, 32 bytes, and
//! is all that leaves the operator: no gradient field, no list of candidates.
//! The field is borrowed immutably; nothing in the operator can reach the
//! palette tile, the activation, the law or any evidence.
//!
//! # The inner region, and why it is not a free parameter
//!
//! A pixel's rendered value depends on the surfels within the footprint radius
//! (3), and a surfel's weight on its Moore neighbours (1): four pixels in
//! total. A pixel closer than 4 to the tile edge therefore sees missing
//! neighbours and clipped footprints, which read as a boundary that is not in
//! the material. The gradient needs one more pixel on each side, so the
//! operator reads `x, y ∈ 5..=10`. The silence test shows this region reads a
//! uniform tile as flat, and a disable run shows a wider region does not.
//!
//! # Two implementations
//!
//! - **B, streaming:** Morton order over the tile, one witness register.
//! - **A, materialized:** row-major, every interior gradient into a `Vec`, then
//!   an explicit argmax with the same tie-break.
//!
//! Run: `cargo run -p cognitive-shader-driver --example boundary_measure_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example boundary_measure_probe`

use std::mem::size_of;
use std::ops::RangeInclusive;

use bgz_tensor::fisher_z::FisherZTable;
use lance_graph_contract::morton8x8::Morton8x8;

#[path = "support/fisher_relation.rs"]
mod fisher_relation;
use fisher_relation::{allocations_during, representatives, PairwiseFisherZ};

#[path = "support/virtual_surfel.rs"]
mod virtual_surfel;
use virtual_surfel::{Tile, PIXELS};

#[path = "support/ewa.rs"]
mod ewa;
use ewa::{max_abs, render_isotropic, Field};

/// Pixels whose value and central difference do not see the tile edge.
const INNER: RangeInclusive<u8> = 5..=10;

// ── the operator ───────────────────────────────────────────────────────────

/// What the operator returns: where the surface is steepest, and how.
#[derive(Clone, Copy, Debug, PartialEq)]
struct BoundaryWitness {
    at: Morton8x8,
    magnitude: f64,
    gx: f64,
    gy: f64,
}

#[inline]
fn at(f: &Field, p: Morton8x8, dx: i8, dy: i8) -> f64 {
    let n = p
        .checked_offset(dx, dy)
        .expect("inner pixels have all four neighbours");
    f[n.code() as usize]
}

/// B: one pass in Morton order, one witness register.
fn strongest_boundary(f: &Field) -> Option<BoundaryWitness> {
    let mut best: Option<BoundaryWitness> = None;
    for code in 0..PIXELS as u16 {
        let p = Morton8x8::from_code(code);
        if !INNER.contains(&p.x()) || !INNER.contains(&p.y()) {
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

// ── A: materialized ────────────────────────────────────────────────────────

/// Row-major: every interior gradient into a `Vec`, then argmax with the
/// smaller Morton code winning ties. Returns the witness and the bytes built.
fn strongest_boundary_materialized(f: &Field) -> (Option<BoundaryWitness>, usize) {
    let v = |x: u8, y: u8| f[Morton8x8::from_xy(x, y).code() as usize];
    let mut all: Vec<BoundaryWitness> = Vec::new();
    for y in INNER {
        for x in INNER {
            let gx = (v(x + 1, y) - v(x - 1, y)) / 2.0;
            let gy = (v(x, y + 1) - v(x, y - 1)) / 2.0;
            all.push(BoundaryWitness {
                at: Morton8x8::from_xy(x, y),
                magnitude: (gx * gx + gy * gy).sqrt(),
                gx,
                gy,
            });
        }
    }
    let bytes = all.len() * size_of::<BoundaryWitness>();
    let best = all.into_iter().reduce(|a, b| {
        if b.magnitude > a.magnitude || (b.magnitude == a.magnitude && b.at.code() < a.at.code()) {
            b
        } else {
            a
        }
    });
    (best, bytes)
}

// ── fixtures ───────────────────────────────────────────────────────────────

/// Material `left` for `x < 8` (or `y < 8`), `right` for the rest.
fn split(left: u8, right: u8, vertical: bool) -> Tile {
    core::array::from_fn(|i| {
        let p = Morton8x8::from_code(i as u16);
        let first = if vertical { p.x() < 8 } else { p.y() < 8 };
        if first {
            left
        } else {
            right
        }
    })
}

fn render(tile: &Tile, law: &PairwiseFisherZ<'_>) -> Field {
    let mut f = [0.0; PIXELS];
    render_isotropic(tile, &[255; PIXELS], law, &mut f);
    f
}

fn main() {
    let table = FisherZTable::build(&representatives(1), 256);
    let law = PairwiseFisherZ::borrow(&table);
    let (weak_s, weak_b) = law.pair_with_code(-100..=-80);
    let (strong_s, strong_b) = law.pair_with_code(80..=100);

    let weak = render(&split(weak_s, weak_b, true), &law);
    let (w, n_alloc, n_bytes) = allocations_during(|| strongest_boundary(&weak));
    let w = w.expect("non-empty inner region");
    let (a, a_bytes) = strongest_boundary_materialized(&weak);
    let strong = strongest_boundary(&render(&split(strong_s, strong_b, true), &law)).unwrap();
    let flat = strongest_boundary(&render(&[weak_s; PIXELS], &law)).unwrap();
    let flat_field = render(&[weak_s; PIXELS], &law);

    println!(
        "D-CTX-4 strongest-boundary measurement over the D-CTX-2 surface, inner region {INNER:?}"
    );
    println!("  law generation             : {:#018x}", law.generation);
    println!(
        "  weak boundary (code -100..=-80) : at ({}, {})  |g| {:.4}  gx {:+.4}  gy {:+.4}",
        w.at.x(),
        w.at.y(),
        w.magnitude,
        w.gx,
        w.gy
    );
    println!(
        "  strong boundary (code 80..=100) : at ({}, {})  |g| {:.4}",
        strong.at.x(),
        strong.at.y(),
        strong.magnitude
    );
    println!(
        "  uniform tile                    : |g| {:.3e} against field max {:.3e}",
        flat.magnitude,
        max_abs(&flat_field)
    );
    println!("  streaming == materialized : {}", Some(w) == a);
    println!(
        "  witness                   : {} B, returned by value",
        size_of::<BoundaryWitness>()
    );
    println!("  streaming heap            : {n_alloc} allocations, {n_bytes} B");
    println!("  materialized gradients    : {a_bytes} B");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> FisherZTable {
        FisherZTable::build(&representatives(1), 256)
    }

    /// FAILS IF: streaming and materialized disagree on any rendered fixture
    /// or on a synthetic field, or either returns nothing.
    #[test]
    fn streaming_equals_materialized() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let mut fields = Vec::new();
        for range in [-100..=-80, -40..=-20, 80..=100] {
            let (s, b) = law.pair_with_code(range);
            fields.push(render(&split(s, b, true), &law));
            fields.push(render(&split(s, b, false), &law));
        }
        fields.push(core::array::from_fn(|i| ((i * 37 + 11) % 101) as f64));
        for f in &fields {
            let b = strongest_boundary(f);
            assert!(b.is_some());
            assert_eq!(b, strongest_boundary_materialized(f).0);
        }
    }

    /// FAILS IF: the streaming operator allocates, or the counter is blind.
    #[test]
    fn the_operator_allocates_nothing() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let (s, b) = law.pair_with_code(-100..=-80);
        let f = render(&split(s, b, true), &law);
        let (_, n, bytes) = allocations_during(|| strongest_boundary(&f));
        assert_eq!((n, bytes), (0, 0));
        let (_, n_a, _) = allocations_during(|| strongest_boundary_materialized(&f));
        assert!(n_a >= 1, "counter is blind");
    }

    /// FAILS IF: a material boundary is not found within one pixel of where
    /// the palette changes, does not point across it, or does not turn when
    /// the boundary turns.
    #[test]
    fn the_boundary_is_found_where_the_material_changes() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let (s, b) = law.pair_with_code(-100..=-80);
        let v = strongest_boundary(&render(&split(s, b, true), &law)).unwrap();
        assert!(
            (6..=9).contains(&v.at.x()),
            "vertical boundary at x {}",
            v.at.x()
        );
        assert!(v.gx.abs() > 10.0 * v.gy.abs(), "points across x: {v:?}");
        let h = strongest_boundary(&render(&split(s, b, false), &law)).unwrap();
        assert!(
            (6..=9).contains(&h.at.y()),
            "horizontal boundary at y {}",
            h.at.y()
        );
        assert!(h.gy.abs() > 10.0 * h.gx.abs(), "points across y: {h:?}");
        assert!((v.magnitude - h.magnitude).abs() <= 1e-12 * v.magnitude);
    }

    /// FAILS IF: a uniform tile, read over the inner region, is not flat. The
    /// field must be non-trivial, so "flat" is a statement about the region,
    /// not about an empty render.
    #[test]
    fn a_uniform_tile_reads_flat() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        for m in [7u8, 42, 200] {
            let f = render(&[m; PIXELS], &law);
            let scale = max_abs(&f);
            assert!(scale > 1.0);
            let w = strongest_boundary(&f).unwrap();
            assert!(w.magnitude <= 1e-12 * scale, "material {m}: {w:?}");
        }
    }

    /// FAILS IF: the measured contrast does not follow the law: a weakly
    /// related pair must read as a stronger boundary than a strongly related
    /// one, and identical material must read as none.
    #[test]
    fn the_contrast_follows_the_pair_law() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let measure = |range: RangeInclusive<i8>| {
            let (s, b) = law.pair_with_code(range);
            strongest_boundary(&render(&split(s, b, true), &law))
                .unwrap()
                .magnitude
        };
        let (weak, mid, strong) = (measure(-100..=-80), measure(-40..=-20), measure(80..=100));
        assert!(weak > mid && mid > strong, "{weak} {mid} {strong}");
        assert!(strong > 0.0);
    }

    /// FAILS IF: the tie-break is not "smaller Morton code wins". Two equal
    /// spikes make several pixels share the maximum |g|; the fixture is only
    /// meaningful if row-major order would pick a different one of them.
    #[test]
    fn ties_go_to_the_smaller_morton_code() {
        let mut f: Field = [0.0; PIXELS];
        for (x, y) in [(6u8, 7u8), (10, 6)] {
            f[Morton8x8::from_xy(x, y).code() as usize] = 4.0;
        }
        // The tied set, computed by a plain loop over the inner region.
        let mag = |x: u8, y: u8| -> f64 {
            let v = |x: u8, y: u8| f[Morton8x8::from_xy(x, y).code() as usize];
            let (gx, gy) = (
                (v(x + 1, y) - v(x - 1, y)) / 2.0,
                (v(x, y + 1) - v(x, y - 1)) / 2.0,
            );
            (gx * gx + gy * gy).sqrt()
        };
        let max = INNER
            .flat_map(|y| INNER.map(move |x| (x, y)))
            .map(|(x, y)| mag(x, y))
            .fold(0.0, f64::max);
        let tied: Vec<(u8, u8)> = INNER
            .flat_map(|y| INNER.map(move |x| (x, y)))
            .filter(|&(x, y)| mag(x, y) == max)
            .collect();
        assert!(tied.len() >= 2, "no tie: {tied:?}");
        let morton_first = tied
            .iter()
            .map(|&(x, y)| Morton8x8::from_xy(x, y))
            .min()
            .unwrap();
        let row_major_first = tied[0];
        assert_ne!(
            Morton8x8::from_xy(row_major_first.0, row_major_first.1),
            morton_first,
            "the two orders must pick different winners, or the test proves nothing"
        );
        assert_eq!(strongest_boundary(&f).unwrap().at, morton_first);
        assert_eq!(
            strongest_boundary_materialized(&f).0.unwrap().at,
            morton_first
        );
    }

    /// FAILS IF: measuring changes the field it reads, or the witness grows
    /// beyond a fixed 32 bytes.
    #[test]
    fn measuring_is_read_only_and_bounded() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let (s, b) = law.pair_with_code(-100..=-80);
        let f = render(&split(s, b, true), &law);
        let before = f;
        let _ = strongest_boundary(&f);
        assert_eq!(f, before);
        assert_eq!(size_of::<BoundaryWitness>(), 32);
    }
}
