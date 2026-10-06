//! D-CTX-5: Morton / HHTL visit order versus row-major, for the D-CTX-2 render.
//!
//! The D-CTX probes already *address* pixels by [`Morton8x8`] code (lane =
//! code; a 16 x 16 tile is codes `0..256`). This probe asks the remaining
//! question: does the order in which surfels are *visited* change the
//! rendered texture?
//!
//! # Four orders over the same 256 pixels
//!
//! - **morton**: code order `0..256`, i.e. depth-first over the 2-level 16-ary
//!   nibble trie (coarse nibble = 4 x 4 block, fine nibble = pixel in block).
//!   This is the HHTL order and the order D-CTX-2 already uses.
//! - **row-major**: `y` then `x`.
//! - **tiled**: 4 x 4 blocks in row-major order, row-major inside each block.
//! - **reversed**: code order `255..=0`.
//!
//! # The finding the probe pins
//!
//! The D-CTX-2 render accumulates with `f64` scatter. Floating-point addition
//! is not associative, so the visit order changes the last bits of some
//! pixels: the order is part of the replay identity of that render. Two ways
//! out, both measured here:
//!
//! - **pin the order**: Morton order reproduces D-CTX-2 bit for bit;
//! - **make the accumulator order-free**: each contribution is rounded once to
//!   `i64` at 2⁻³² and summed as integers. Integer addition is associative, so
//!   every order gives the same bits, within `49 · 2⁻³³` of the `f64` render.
//!
//! Addressing stays external to `PaletteState`: the palette byte is never read
//! as a coordinate, and the Morton arithmetic is the contract's (`D-MORTON-0`),
//! checked against `FacetTier::morton`.
//!
//! Run: `cargo run -p cognitive-shader-driver --example morton_order_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example morton_order_probe`

use std::mem::size_of;

use bgz_tensor::fisher_z::FisherZTable;
use lance_graph_contract::morton8x8::Morton8x8;

#[path = "support/fisher_relation.rs"]
mod fisher_relation;
use fisher_relation::{allocations_during, representatives, PairwiseFisherZ};

#[path = "support/virtual_surfel.rs"]
mod virtual_surfel;
use virtual_surfel::{
    activation_fixture, neighbor, read_surfel, repeated_tile, Tile, IDENTITY_WEIGHT, PIXELS,
};

#[path = "support/ewa.rs"]
mod ewa;
use ewa::{
    amplitude, footprint, footprint_sigma, isotropic_scale, max_abs, render_isotropic, Field,
    RADIUS,
};

/// Fixed-point scale of the order-free accumulator: 2³² per unit.
const FIXED_ONE: f64 = 4_294_967_296.0;

/// A field accumulated in `i64` at [`FIXED_ONE`].
type FixedField = [i64; PIXELS];

// ── orders ─────────────────────────────────────────────────────────────────

/// A visit order: every pixel's Morton code exactly once.
type Order = [u16; PIXELS];

fn morton_order() -> Order {
    core::array::from_fn(|i| i as u16)
}

fn row_major_order() -> Order {
    core::array::from_fn(|i| Morton8x8::from_xy((i % 16) as u8, (i / 16) as u8).code())
}

fn tiled_order() -> Order {
    core::array::from_fn(|i| {
        let (block, inner) = (i / 16, i % 16);
        let (bx, by) = (block % 4, block / 4);
        let (ix, iy) = (inner % 4, inner / 4);
        Morton8x8::from_xy((bx * 4 + ix) as u8, (by * 4 + iy) as u8).code()
    })
}

fn reversed_order() -> Order {
    core::array::from_fn(|i| (PIXELS - 1 - i) as u16)
}

fn orders() -> [(&'static str, Order); 4] {
    [
        ("morton", morton_order()),
        ("row-major", row_major_order()),
        ("tiled", tiled_order()),
        ("reversed", reversed_order()),
    ]
}

/// Mean and maximum trie ascent (`nibble_climb`) between consecutive visits.
fn climb_stats(order: &Order) -> (f64, u8) {
    let (mut sum, mut max) = (0u32, 0u8);
    for w in order.windows(2) {
        let c = Morton8x8::from_code(w[0]).nibble_climb(Morton8x8::from_code(w[1]));
        sum += u32::from(c);
        max = max.max(c);
    }
    (f64::from(sum) / (PIXELS - 1) as f64, max)
}

// ── renders in a given order ───────────────────────────────────────────────

/// The D-CTX-2 render with surfels visited in `order`, `f64` scatter.
fn render_f64(tile: &Tile, act: &Tile, law: &PairwiseFisherZ<'_>, order: &Order) -> Field {
    let mut field = [0.0; PIXELS];
    for &code in order {
        let a = act[code as usize];
        if a == 0 {
            continue;
        }
        let r = read_surfel(tile, Morton8x8::from_code(code), a, law, IDENTITY_WEIGHT);
        if let Some(s) = isotropic_scale(&r) {
            ewa::splat(&mut field, r.center, &footprint_sigma(s), amplitude(&r));
        }
    }
    field
}

/// The same render, each contribution rounded once to `i64` at 2⁻³².
fn render_fixed(tile: &Tile, act: &Tile, law: &PairwiseFisherZ<'_>, order: &Order) -> FixedField {
    let mut field = [0i64; PIXELS];
    for &code in order {
        let a = act[code as usize];
        if a == 0 {
            continue;
        }
        let r = read_surfel(tile, Morton8x8::from_code(code), a, law, IDENTITY_WEIGHT);
        let Some(s) = isotropic_scale(&r) else {
            continue;
        };
        let (sigma, amp) = (footprint_sigma(s), amplitude(&r));
        for dy in -RADIUS..=RADIUS {
            for dx in -RADIUS..=RADIUS {
                if let Some(p) = neighbor(r.center, dx, dy) {
                    let v = amp * footprint(&sigma, f64::from(dx), f64::from(dy));
                    field[p.code() as usize] += (v * FIXED_ONE).round() as i64;
                }
            }
        }
    }
    field
}

/// Pixels that differ, and the largest difference.
fn diff(a: &Field, b: &Field) -> (usize, f64) {
    (0..PIXELS).fold((0, 0.0f64), |(n, m), i| {
        let d = (a[i] - b[i]).abs();
        (n + usize::from(a[i].to_bits() != b[i].to_bits()), m.max(d))
    })
}

fn main() {
    let table = FisherZTable::build(&representatives(1), 256);
    let law = PairwiseFisherZ::borrow(&table);
    let (tile, act) = (repeated_tile(), activation_fixture());
    let reference = render_f64(&tile, &act, &law, &morton_order());
    let fixed_ref = render_fixed(&tile, &act, &law, &morton_order());

    println!("D-CTX-5 visit order versus the D-CTX-2 render, 16 x 16 Morton tile");
    println!("  law generation : {:#018x}", law.generation);
    println!(
        "  order        mean climb  max climb  f64 pixels changed  max |Δ| f64   fixed == morton"
    );
    for (name, order) in orders() {
        let (mean, max) = climb_stats(&order);
        let (n, d) = diff(&render_f64(&tile, &act, &law, &order), &reference);
        let fixed_same = render_fixed(&tile, &act, &law, &order) == fixed_ref;
        println!("  {name:<10}   {mean:>9.4}  {max:>9}  {n:>18}  {d:>11.3e}   {fixed_same}");
    }
    let mut d2 = [0.0; PIXELS];
    render_isotropic(&tile, &act, &law, &mut d2);
    println!("  morton f64 == D-CTX-2 render : {}", reference == d2);
    let worst = (0..PIXELS)
        .map(|i| (fixed_ref[i] as f64 / FIXED_ONE - reference[i]).abs())
        .fold(0.0, f64::max);
    println!(
        "  fixed vs f64 : max |Δ| {worst:.3e} (field max {:.3e})",
        max_abs(&reference)
    );
    let (_, n_alloc, _) =
        allocations_during(|| render_fixed(&tile, &act, &law, &row_major_order()));
    println!(
        "  fixed render heap : {n_alloc} allocations; accumulator {} B",
        size_of::<FixedField>()
    );
}

#[cfg(test)]
mod tests {
    use super::virtual_surfel::permutation_tile;
    use super::*;
    use lance_graph_contract::facet::FacetTier;

    fn table() -> FisherZTable {
        FisherZTable::build(&representatives(1), 256)
    }

    fn tiles() -> Vec<Tile> {
        let mut t = vec![permutation_tile(), repeated_tile()];
        let mut s: u32 = 0xDEAD_BEEF;
        for _ in 0..8 {
            t.push(core::array::from_fn(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 24) as u8 % 5 * 50
            }));
        }
        t
    }

    /// FAILS IF: an order misses or repeats a pixel, two orders coincide, or
    /// the Morton code disagrees with the contract's `FacetTier::morton`.
    #[test]
    fn every_order_is_a_distinct_permutation_of_the_tile() {
        let all = orders();
        for (name, order) in &all {
            let mut seen = [false; PIXELS];
            for &c in order {
                assert!(!seen[c as usize], "{name} visits {c} twice");
                seen[c as usize] = true;
            }
            assert!(seen.iter().all(|&s| s), "{name} misses a pixel");
        }
        for i in 0..all.len() {
            for j in i + 1..all.len() {
                assert_ne!(all[i].1, all[j].1, "{} == {}", all[i].0, all[j].0);
            }
        }
        for y in 0..16u8 {
            for x in 0..16u8 {
                assert_eq!(
                    Morton8x8::from_xy(x, y).code(),
                    FacetTier { lo: x, hi: y }.morton()
                );
            }
        }
    }

    /// FAILS IF: a 7 x 7 footprint neighbour reached through the Morton code
    /// differs from row-major geometry anywhere on the tile.
    #[test]
    fn footprint_neighbours_match_geometry() {
        let mut visits = 0;
        for y in 0..16u8 {
            for x in 0..16u8 {
                let c = Morton8x8::from_xy(x, y);
                for dy in -RADIUS..=RADIUS {
                    for dx in -RADIUS..=RADIUS {
                        let (nx, ny) = (i16::from(x) + i16::from(dx), i16::from(y) + i16::from(dy));
                        let on = (0..16).contains(&nx) && (0..16).contains(&ny);
                        assert_eq!(
                            neighbor(c, dx, dy),
                            on.then(|| Morton8x8::from_xy(nx as u8, ny as u8))
                        );
                        visits += usize::from(on);
                    }
                }
            }
        }
        // Σ over x of the clipped window width, squared: (Σ_x |[x-3, x+3] ∩ [0, 15]|)².
        let per_axis: usize = (0..16i32)
            .map(|x| ((x + 3).min(15) - (x - 3).max(0) + 1) as usize)
            .sum();
        assert_eq!(visits, per_axis * per_axis);
    }

    /// FAILS IF: Morton order does not reproduce the D-CTX-2 render bit for
    /// bit (it is the order D-CTX-2 uses).
    #[test]
    fn morton_order_is_the_d_ctx_2_render() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let act = activation_fixture();
        for tile in tiles() {
            let mut d2 = [0.0; PIXELS];
            render_isotropic(&tile, &act, &law, &mut d2);
            assert_eq!(render_f64(&tile, &act, &law, &morton_order()), d2);
        }
    }

    /// FAILS IF: changing the visit order changes the `f64` render by more
    /// than rounding, or never changes a single bit (then the fixed-point
    /// accumulator would solve a problem that does not exist).
    #[test]
    fn f64_scatter_depends_on_the_order_only_at_the_last_bits() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let act = activation_fixture();
        let mut changed = 0;
        for tile in tiles() {
            let reference = render_f64(&tile, &act, &law, &morton_order());
            let scale = max_abs(&reference);
            for (name, order) in orders() {
                let (n, d) = diff(&render_f64(&tile, &act, &law, &order), &reference);
                assert!(d <= 1e-13 * scale, "{name}: max |Δ| {d}");
                changed += n;
            }
        }
        assert!(changed > 0, "no order changed any bit");
    }

    /// FAILS IF: the fixed-point render differs between any two orders, or
    /// strays from the `f64` render by more than 49 rounding steps.
    #[test]
    fn the_fixed_point_accumulator_is_order_free() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let act = activation_fixture();
        for tile in tiles() {
            let reference = render_fixed(&tile, &act, &law, &morton_order());
            let f = render_f64(&tile, &act, &law, &morton_order());
            for (name, order) in orders() {
                assert_eq!(render_fixed(&tile, &act, &law, &order), reference, "{name}");
            }
            for i in 0..PIXELS {
                let d = (reference[i] as f64 / FIXED_ONE - f[i]).abs();
                assert!(d <= 49.0 * 0.5 / FIXED_ONE + 1e-12, "pixel {i}: {d}");
            }
        }
    }

    /// FAILS IF: the trie-ascent locality of the four orders moves. Pinned as
    /// measured: Morton, tiled and reversed all climb 1.0588 nibbles per step
    /// on average (inside a 4 x 4 block only the fine nibble moves, so any
    /// order that finishes a block before leaving it climbs the same);
    /// row-major leaves its block every 4 pixels and climbs 1.2471.
    #[test]
    fn trie_locality_is_pinned() {
        let stats: Vec<(f64, u8)> = orders().iter().map(|(_, o)| climb_stats(o)).collect();
        let (morton, row, tiled, rev) = (stats[0], stats[1], stats[2], stats[3]);
        assert_eq!(morton, tiled, "block-finishing orders climb alike");
        assert_eq!(morton, rev, "reversing an order keeps its climbs");
        assert!((morton.0 - 270.0 / 255.0).abs() < 1e-12, "{morton:?}");
        assert!(row.0 > morton.0 + 0.15, "{row:?}");
        assert_eq!((morton.1, row.1), (2, 2));
    }

    /// FAILS IF: the order-free render allocates.
    #[test]
    fn the_fixed_render_allocates_nothing() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let (tile, act) = (repeated_tile(), activation_fixture());
        let order = tiled_order();
        let (_, n, bytes) = allocations_during(|| render_fixed(&tile, &act, &law, &order));
        assert_eq!((n, bytes), (0, 0));
    }
}
