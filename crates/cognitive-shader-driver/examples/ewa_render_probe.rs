//! D-CTX-2: isotropic Gaussian / EWA accumulation from virtual surfels.
//!
//! Claim under test: rendering the 16 x 16 Palette256 tile needs no surfel or
//! Gaussian population. Each virtual surfel (D-CTX-1) is read, turned into an
//! isotropic footprint and accumulated into a transient field in one pass; a
//! materialized `Vec<Surfel> -> Vec<Gaussian> -> field` pipeline gives the same
//! field bit for bit.
//!
//! # The law, isotropic case only
//!
//! For a surfel with weight `W > 0` and second moment `Σ_c`:
//!
//! ```text
//! s     = (Σxx + Σyy) / (2W)                 isotropic part of Σ_c / W
//! Σ_fp  = ewa_sandwich(M = √s·I, Σ₀ = I)     = s·I   (contract kernel)
//! A     = activation · W / (8 · 254)         relation mass × activation
//! field(p) += A · g_Σfp(p − c)               over a 7 x 7 window, clipped
//! g_Σ(d) = exp(−½ dᵀΣ⁻¹d) / (2π √det Σ)
//! ```
//!
//! The footprint evaluator `g` is probe-local (decision C): `Spd2` and
//! `ewa_sandwich` keep their certified meaning, covariance propagation, and are
//! only *used* here. Identity neighbours weigh [`IDENTITY_WEIGHT`] = 254
//! (operator decision 2026-10-06).
//!
//! The field is the one thing that materializes: a 2 KB `f64` accumulator, a
//! transient working surface. It is interpretation, not evidence: the probe
//! shows rendering only redistributes the surfels' mass (never creates any) and
//! never touches the tile or the activation (borrowed immutably).
//!
//! # Three implementations
//!
//! - **B, fused:** virtual reading → footprint → accumulate, nothing stored.
//! - **A, materialized:** `Vec<SurfelReading>`, then `Vec<Gaussian>`, then the
//!   same scatter. Must equal B exactly (same summation order).
//! - **C, independent gather:** per pixel, sum over the Gaussians with the
//!   closed-form isotropic density `exp(−r²/2s) / (2πs)` and its own amplitude
//!   formula. Different order, so equal within `1e-12 × max|field|`.
//!
//! Run: `cargo run -p cognitive-shader-driver --example ewa_render_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example ewa_render_probe`

use std::f64::consts::PI;
use std::mem::size_of;

use bgz_tensor::fisher_z::FisherZTable;
use lance_graph_contract::morton8x8::Morton8x8;
use lance_graph_contract::sigma_propagation::{ewa_sandwich, Spd2};

#[path = "support/fisher_relation.rs"]
mod fisher_relation;
use fisher_relation::{allocations_during, representatives, PairwiseFisherZ};

#[path = "support/virtual_surfel.rs"]
mod virtual_surfel;
use virtual_surfel::{
    activation_fixture, for_each_surfel, neighbor, permutation_tile, repeated_tile, SurfelReading,
    Tile, IDENTITY_WEIGHT, PIXELS,
};

/// Footprint window radius: 7 x 7 offsets around the center.
const RADIUS: i8 = 3;
/// Largest relation mass a surfel can have: 8 neighbours at 254.
const MAX_MASS: f64 = 8.0 * 254.0;

/// The transient rendered surface, one `f64` per pixel, lane = Morton code.
type Field = [f64; PIXELS];

// ── the law, probe-local ───────────────────────────────────────────────────

/// Isotropic scale of a reading: `(Σxx + Σyy) / 2W`. `None` when `W = 0`.
#[inline]
fn isotropic_scale(r: &SurfelReading) -> Option<f64> {
    (r.weight > 0).then(|| f64::from(r.sxx + r.syy) / (2.0 * f64::from(r.weight)))
}

/// The footprint covariance through the contract's sandwich: `√s·I · I · √s·I`.
#[inline]
fn footprint_sigma(s: f64) -> Spd2 {
    let m = Spd2 {
        a: s.sqrt(),
        b: 0.0,
        c: s.sqrt(),
    };
    ewa_sandwich(&m, &Spd2::I)
}

/// Splat amplitude: activation times the surfel's relation mass fraction.
#[inline]
fn amplitude(r: &SurfelReading) -> f64 {
    f64::from(r.amplitude) * f64::from(r.weight) / MAX_MASS
}

/// The normalized 2-D Gaussian density of `Σ` at offset `(dx, dy)`.
#[inline]
fn footprint(sigma: &Spd2, dx: f64, dy: f64) -> f64 {
    let det = sigma.det();
    let q = (sigma.c * dx * dx - 2.0 * sigma.b * dx * dy + sigma.a * dy * dy) / det;
    (-0.5 * q).exp() / (2.0 * PI * det.sqrt())
}

/// Scatter one footprint into the field over the clipped window.
#[inline]
fn splat(field: &mut Field, center: Morton8x8, sigma: &Spd2, amp: f64) {
    for dy in -RADIUS..=RADIUS {
        for dx in -RADIUS..=RADIUS {
            if let Some(p) = neighbor(center, dx, dy) {
                field[p.code() as usize] += amp * footprint(sigma, f64::from(dx), f64::from(dy));
            }
        }
    }
}

// ── B: fused ───────────────────────────────────────────────────────────────

/// Read every activated surfel and accumulate its footprint at once.
fn render_fused(tile: &Tile, act: &Tile, law: &PairwiseFisherZ<'_>, field: &mut Field) {
    for_each_surfel(tile, act, law, IDENTITY_WEIGHT, |r| {
        if let Some(s) = isotropic_scale(&r) {
            splat(field, r.center, &footprint_sigma(s), amplitude(&r));
        }
    });
}

// ── A: materialized ────────────────────────────────────────────────────────

/// A materialized footprint.
#[derive(Clone, Copy, Debug)]
struct Gaussian {
    center: Morton8x8,
    sigma: Spd2,
    amp: f64,
}

/// What A built, in bytes.
#[derive(Clone, Copy, Debug, Default)]
struct Inventory {
    surfel_bytes: usize,
    gaussian_bytes: usize,
}

/// `Vec<SurfelReading>`, then `Vec<Gaussian>`, then the same scatter.
fn materialize(tile: &Tile, act: &Tile, law: &PairwiseFisherZ<'_>) -> (Vec<Gaussian>, Inventory) {
    let mut surfels: Vec<SurfelReading> = Vec::new();
    for_each_surfel(tile, act, law, IDENTITY_WEIGHT, |r| surfels.push(r));
    let gaussians: Vec<Gaussian> = surfels
        .iter()
        .filter_map(|r| {
            isotropic_scale(r).map(|s| Gaussian {
                center: r.center,
                sigma: footprint_sigma(s),
                amp: amplitude(r),
            })
        })
        .collect();
    let inv = Inventory {
        surfel_bytes: surfels.len() * size_of::<SurfelReading>(),
        gaussian_bytes: gaussians.len() * size_of::<Gaussian>(),
    };
    (gaussians, inv)
}

fn render_materialized(tile: &Tile, act: &Tile, law: &PairwiseFisherZ<'_>) -> (Field, Inventory) {
    let (gaussians, inv) = materialize(tile, act, law);
    let mut field = [0.0; PIXELS];
    for g in &gaussians {
        splat(&mut field, g.center, &g.sigma, g.amp);
    }
    (field, inv)
}

// ── C: independent gather ──────────────────────────────────────────────────

/// Per pixel, sum every surfel's closed-form isotropic density, with its own
/// scale and amplitude arithmetic and row-major geometry.
fn render_gather(tile: &Tile, act: &Tile, law: &PairwiseFisherZ<'_>) -> Field {
    let mut surfels: Vec<(i32, i32, f64, f64)> = Vec::new(); // (x, y, s, amp)
    for_each_surfel(tile, act, law, IDENTITY_WEIGHT, |r| {
        if r.weight > 0 {
            let s = (f64::from(r.sxx) + f64::from(r.syy)) / f64::from(r.weight) / 2.0;
            let amp = f64::from(r.amplitude) * (f64::from(r.weight) / 2032.0);
            surfels.push((i32::from(r.center.x()), i32::from(r.center.y()), s, amp));
        }
    });
    let mut field = [0.0; PIXELS];
    for y in 0..16i32 {
        for x in 0..16i32 {
            let mut sum = 0.0;
            for &(cx, cy, s, amp) in &surfels {
                let (dx, dy) = (x - cx, y - cy);
                if dx.abs() <= i32::from(RADIUS) && dy.abs() <= i32::from(RADIUS) {
                    let r2 = f64::from(dx * dx + dy * dy);
                    sum += amp * (-r2 / (2.0 * s)).exp() / (2.0 * PI * s);
                }
            }
            field[Morton8x8::from_xy(x as u8, y as u8).code() as usize] = sum;
        }
    }
    field
}

fn max_abs(f: &Field) -> f64 {
    f.iter().fold(0.0, |m, v| m.max(v.abs()))
}

fn main() {
    let table = FisherZTable::build(&representatives(1), 256);
    let law = PairwiseFisherZ::borrow(&table);
    let (tile, act) = (repeated_tile(), activation_fixture());

    let mut b = [0.0; PIXELS];
    let ((), n_alloc, n_bytes) = allocations_during(|| render_fused(&tile, &act, &law, &mut b));
    let (a, inv) = render_materialized(&tile, &act, &law);
    let c = render_gather(&tile, &act, &law);
    let worst = (0..PIXELS).map(|i| (b[i] - c[i]).abs()).fold(0.0, f64::max);

    let (mut s_min, mut s_max) = (f64::INFINITY, 0.0f64);
    for_each_surfel(
        &permutation_tile(),
        &[1; PIXELS],
        &law,
        IDENTITY_WEIGHT,
        |r| {
            if let Some(s) = isotropic_scale(&r) {
                s_min = s_min.min(s);
                s_max = s_max.max(s);
            }
        },
    );

    println!("D-CTX-2 isotropic EWA accumulation, 16 x 16 Morton tile, identity weight {IDENTITY_WEIGHT}");
    println!("  law generation             : {:#018x}", law.generation);
    println!("  fused == materialized      : {}", a == b);
    println!(
        "  fused vs gather (closed)   : max |Δ| {worst:.3e} against max field {:.3e}",
        max_abs(&b)
    );
    println!(
        "  field sum / max            : {:.4} / {:.4}",
        b.iter().sum::<f64>(),
        max_abs(&b)
    );
    println!("  isotropic scale s (perm.)  : {s_min:.4} ..= {s_max:.4}");
    println!(
        "  transient field            : {} B (the only materialization)",
        size_of::<Field>()
    );
    println!("  fused heap                 : {n_alloc} allocations, {n_bytes} B (surfel and Gaussian population 0 B)");
    println!(
        "  materialized               : surfels {} B, gaussians {} B",
        inv.surfel_bytes, inv.gaussian_bytes
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> FisherZTable {
        FisherZTable::build(&representatives(1), 256)
    }

    fn fused(tile: &Tile, act: &Tile, law: &PairwiseFisherZ<'_>) -> Field {
        let mut f = [0.0; PIXELS];
        render_fused(tile, act, law, &mut f);
        f
    }

    fn tiles() -> Vec<Tile> {
        let mut t = vec![permutation_tile(), repeated_tile()];
        let mut s: u32 = 0x2545_F491;
        for _ in 0..16 {
            t.push(core::array::from_fn(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 24) as u8 % 5 * 50
            }));
        }
        t
    }

    /// FAILS IF: removing the virtual/materialized distinction changes a
    /// single bit of the field, on 18 tiles.
    #[test]
    fn fused_equals_materialized_bit_for_bit() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let act = activation_fixture();
        for tile in tiles() {
            let (a, _) = render_materialized(&tile, &act, &law);
            assert_eq!(fused(&tile, &act, &law), a);
        }
    }

    /// FAILS IF: the fused field disagrees with the independent gather (closed
    /// form, row-major, own amplitude) by more than `1e-12 × max|field|`, or
    /// the field is empty (which would make the bound trivial).
    #[test]
    fn fused_equals_the_independent_closed_form_gather() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let act = activation_fixture();
        for tile in tiles() {
            let (b, c) = (fused(&tile, &act, &law), render_gather(&tile, &act, &law));
            let scale = max_abs(&b);
            assert!(scale > 1.0, "field is empty: {scale}");
            for i in 0..PIXELS {
                assert!(
                    (b[i] - c[i]).abs() <= 1e-12 * scale,
                    "pixel {i}: {} vs {}",
                    b[i],
                    c[i]
                );
            }
        }
    }

    /// FAILS IF: the fused path allocates, or the counter is blind.
    #[test]
    fn fused_rendering_allocates_nothing() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let (tile, act) = (repeated_tile(), activation_fixture());
        let mut f = [0.0; PIXELS];
        let ((), n, bytes) = allocations_during(|| render_fused(&tile, &act, &law, &mut f));
        assert_eq!((n, bytes), (0, 0));
        let (_, n_a, _) = allocations_during(|| render_materialized(&tile, &act, &law));
        assert!(n_a >= 2, "counter is blind");
    }

    /// FAILS IF: the contract sandwich `√s·I · I · √s·I` is not `s·I` and
    /// SPD, or the scaling law `√s·I · Σ · √s·I = s·Σ` fails on an
    /// anisotropic Σ.
    #[test]
    fn the_isotropic_sandwich_is_scalar_and_psd() {
        let aniso = Spd2 {
            a: 2.0,
            b: 0.7,
            c: 1.1,
        };
        for k in 1..=40 {
            let s = f64::from(k) * 0.05;
            let fp = footprint_sigma(s);
            assert!(
                (fp.a - s).abs() < 1e-15 && fp.b == 0.0 && (fp.c - s).abs() < 1e-15,
                "{fp:?}"
            );
            assert!(fp.is_spd(1e-12));
            let m = Spd2 {
                a: s.sqrt(),
                b: 0.0,
                c: s.sqrt(),
            };
            let out = ewa_sandwich(&m, &aniso);
            assert!((out.a - s * aniso.a).abs() < 1e-12);
            assert!((out.b - s * aniso.b).abs() < 1e-12);
            assert!((out.c - s * aniso.c).abs() < 1e-12);
            assert!(out.is_spd(1e-12));
        }
    }

    /// FAILS IF: the general footprint evaluator on `s·I` differs from the
    /// closed-form isotropic Gaussian at any window offset.
    #[test]
    fn the_footprint_matches_the_isotropic_closed_form() {
        for k in 2..=30 {
            let s = f64::from(k) * 0.05;
            let sigma = footprint_sigma(s);
            for dy in -RADIUS..=RADIUS {
                for dx in -RADIUS..=RADIUS {
                    let r2 = f64::from(dx * dx + dy * dy);
                    let closed = (-r2 / (2.0 * s)).exp() / (2.0 * PI * s);
                    let general = footprint(&sigma, f64::from(dx), f64::from(dy));
                    assert!(
                        (general - closed).abs() <= 1e-14 * closed.max(1e-300),
                        "s {s} ({dx},{dy})"
                    );
                }
            }
        }
    }

    /// FAILS IF: rendering creates mass. The field's total must equal the sum
    /// of each surfel's amplitude times its own clipped window mass, every
    /// window mass must be at most 1 (up to sampling, 1e-3), and an interior
    /// window must hold at least 0.99 of it (so the 7 x 7 window is not
    /// silently truncating).
    #[test]
    fn rendering_redistributes_mass_and_creates_none() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let (tile, act) = (repeated_tile(), activation_fixture());
        let mut expected = 0.0;
        let mut interior = 0;
        for_each_surfel(&tile, &act, &law, IDENTITY_WEIGHT, |r| {
            let Some(s) = isotropic_scale(&r) else { return };
            let sigma = footprint_sigma(s);
            let mut mass = 0.0;
            for dy in -RADIUS..=RADIUS {
                for dx in -RADIUS..=RADIUS {
                    if neighbor(r.center, dx, dy).is_some() {
                        mass += footprint(&sigma, f64::from(dx), f64::from(dy));
                    }
                }
            }
            assert!(mass <= 1.001, "window mass {mass}");
            let (x, y) = (r.center.x(), r.center.y());
            if (3..13).contains(&x) && (3..13).contains(&y) {
                assert!(mass >= 0.99, "interior window mass {mass} at ({x},{y})");
                interior += 1;
            }
            expected += amplitude(&r) * mass;
        });
        assert!(interior > 50);
        let total: f64 = fused(&tile, &act, &law).iter().sum();
        assert!(
            (total - expected).abs() <= 1e-9 * expected,
            "{total} vs {expected}"
        );
        let input: f64 = act.iter().map(|&a| f64::from(a)).sum();
        assert!(
            total <= input * 1.001,
            "rendered {total} exceeds input amplitude {input}"
        );
    }

    /// FAILS IF: an all-zero activation renders anything (no surfel, no ink),
    /// or a non-zero one renders nothing.
    #[test]
    fn no_activation_renders_nothing() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let tile = repeated_tile();
        assert_eq!(fused(&tile, &[0; PIXELS], &law), [0.0; PIXELS]);
        assert!(max_abs(&fused(&tile, &activation_fixture(), &law)) > 0.0);
    }

    /// FAILS IF: the same tile and law generation do not replay bit for bit,
    /// or a different law generation or a palette swap leaves the field
    /// unchanged.
    #[test]
    fn replay_is_exact_and_material_and_law_matter() {
        let (t1, t2, t3) = (
            table(),
            table(),
            FisherZTable::build(&representatives(2), 256),
        );
        let (l1, l2, l3) = (
            PairwiseFisherZ::borrow(&t1),
            PairwiseFisherZ::borrow(&t2),
            PairwiseFisherZ::borrow(&t3),
        );
        let (tile, act) = (permutation_tile(), activation_fixture());
        assert_eq!(fused(&tile, &act, &l1), fused(&tile, &act, &l2));
        assert_ne!(fused(&tile, &act, &l1), fused(&tile, &act, &l3));
        let mut swapped = tile;
        swapped.swap(
            Morton8x8::from_xy(5, 5).code() as usize,
            Morton8x8::from_xy(10, 7).code() as usize,
        );
        assert_ne!(fused(&tile, &act, &l1), fused(&swapped, &act, &l1));
    }
}
