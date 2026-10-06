//! D-CTX-3: anisotropic Gaussian / EWA accumulation from virtual surfels.
//!
//! D-CTX-2 rendered each virtual surfel with an isotropic footprint `s·I`. Here
//! the footprint is the surfel's full normalised second moment, so orientation
//! read from the Fisher-Z law (D-CTX-1) reaches the rendered surface:
//!
//! ```text
//! Σ̂    = Σ_c / W                              normalised second moment
//! Σ̂'   = V · diag(max(λᵢ, ½)) · Vᵀ             eigenvalue floor (below)
//! Σ_fp = ewa_sandwich(M = Σ̂'^{1/2}, Σ₀ = I)    = Σ̂'  (contract kernel)
//! field(p) += A · g_Σfp(p − c)                 A and window as in D-CTX-2
//! ```
//!
//! # The floor, and why it is not a free parameter
//!
//! A footprint whose minor eigenvalue is far below one pixel, sampled on the
//! integer grid, sums to more than 1: the renderer would create mass that no
//! surfel carried, i.e. manufacture evidence. The probe measures this on a
//! stripe (window mass > 1.3 without the floor) and on the degenerate case the
//! law can produce (a cross relation of code −127 gives a singular Σ̂).
//!
//! The floor ½ comes from the law, not from tuning: every Moore offset has
//! `|d|² ≥ 1`, so `tr Σ̂ ≥ 1` and the isotropic scale of D-CTX-2,
//! `s = tr Σ̂ / 2`, is never below ½. The floor therefore never binds on an
//! isotropic reading, which is what makes the isotropic case collapse onto
//! D-CTX-2 (tested). At variance ½ the lattice sum of a Gaussian exceeds its
//! integral by at most ~2·10⁻⁴ per axis, which is the mass bound tested below.
//!
//! What stays out of scope, stated: BF16 is not used anywhere (f64 transient
//! accumulator only); `PaletteState` is not widened; no Σ codebook is needed,
//! the orientation is fully recoverable from resident palette + law.
//!
//! Run: `cargo run -p cognitive-shader-driver --example ewa_anisotropic_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example ewa_anisotropic_probe`

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
    activation_fixture, for_each_surfel, repeated_tile, SurfelReading, Tile, IDENTITY_WEIGHT,
    PIXELS,
};

#[path = "support/ewa.rs"]
mod ewa;
use ewa::{amplitude, max_abs, render_isotropic, splat, Field};

/// Smallest footprint eigenvalue: the smallest isotropic scale the law can
/// produce (`tr Σ̂ ≥ 1` because every Moore offset has `|d|² ≥ 1`).
const MIN_FOOTPRINT_EIGENVALUE: f64 = 0.5;

// ── the anisotropic law ────────────────────────────────────────────────────

/// `Σ̂ = Σ_c / W`, unfloored. `None` when `W = 0`.
#[inline]
fn normalized_moment(r: &SurfelReading) -> Option<Spd2> {
    (r.weight > 0).then(|| {
        let w = f64::from(r.weight);
        Spd2 {
            a: f64::from(r.sxx) / w,
            b: f64::from(r.sxy) / w,
            c: f64::from(r.syy) / w,
        }
    })
}

/// Raise every eigenvalue of a symmetric 2 x 2 to at least `floor`.
#[inline]
fn floor_eigenvalues(n: &Spd2, floor: f64) -> Spd2 {
    let (l1, l2, c, s) = n.eig();
    let (a, b) = (l1.max(floor), l2.max(floor));
    Spd2 {
        a: c * c * a + s * s * b,
        b: c * s * (a - b),
        c: s * s * a + c * c * b,
    }
}

/// The anisotropic footprint covariance through the contract's sandwich.
#[inline]
fn anisotropic_sigma(r: &SurfelReading) -> Option<Spd2> {
    normalized_moment(r).map(|n| {
        let m = floor_eigenvalues(&n, MIN_FOOTPRINT_EIGENVALUE).sqrt();
        ewa_sandwich(&m, &Spd2::I)
    })
}

/// B: read every activated surfel, accumulate its anisotropic footprint.
fn render_anisotropic(tile: &Tile, act: &Tile, law: &PairwiseFisherZ<'_>, field: &mut Field) {
    for_each_surfel(tile, act, law, IDENTITY_WEIGHT, |r| {
        if let Some(sigma) = anisotropic_sigma(&r) {
            splat(field, r.center, &sigma, amplitude(&r));
        }
    });
}

// ── A: materialized ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Gaussian {
    center: Morton8x8,
    sigma: Spd2,
    amp: f64,
}

fn render_materialized(tile: &Tile, act: &Tile, law: &PairwiseFisherZ<'_>) -> (Field, usize) {
    let mut surfels: Vec<SurfelReading> = Vec::new();
    for_each_surfel(tile, act, law, IDENTITY_WEIGHT, |r| surfels.push(r));
    let gaussians: Vec<Gaussian> = surfels
        .iter()
        .filter_map(|r| {
            anisotropic_sigma(r).map(|sigma| Gaussian {
                center: r.center,
                sigma,
                amp: amplitude(r),
            })
        })
        .collect();
    let mut field = [0.0; PIXELS];
    for g in &gaussians {
        splat(&mut field, g.center, &g.sigma, g.amp);
    }
    let bytes =
        surfels.len() * size_of::<SurfelReading>() + gaussians.len() * size_of::<Gaussian>();
    (field, bytes)
}

// ── C: independent gather (own eigen-decomposition, rotated closed form) ────

/// Per pixel, sum every surfel's rotated Gaussian, computed from its own
/// eigen-decomposition and floor, with row-major geometry.
fn render_gather(tile: &Tile, act: &Tile, law: &PairwiseFisherZ<'_>) -> Field {
    // (x, y, λ_major, λ_minor, cos φ, sin φ, amp)
    let mut g: Vec<(i32, i32, f64, f64, f64, f64, f64)> = Vec::new();
    for_each_surfel(tile, act, law, IDENTITY_WEIGHT, |r| {
        if r.weight == 0 {
            return;
        }
        let w = f64::from(r.weight);
        let (a, b, c) = (
            f64::from(r.sxx) / w,
            f64::from(r.sxy) / w,
            f64::from(r.syy) / w,
        );
        let mid = (a + c) / 2.0;
        let rad = (((a - c) / 2.0).powi(2) + b * b).sqrt();
        let phi = 0.5 * (2.0 * b).atan2(a - c);
        let (l1, l2) = ((mid + rad).max(0.5), (mid - rad).max(0.5));
        let amp = f64::from(r.amplitude) * w / 2032.0;
        g.push((
            i32::from(r.center.x()),
            i32::from(r.center.y()),
            l1,
            l2,
            phi.cos(),
            phi.sin(),
            amp,
        ));
    });
    let mut field = [0.0; PIXELS];
    for y in 0..16i32 {
        for x in 0..16i32 {
            let mut sum = 0.0;
            for &(cx, cy, l1, l2, cs, sn, amp) in &g {
                let (dx, dy) = (x - cx, y - cy);
                if dx.abs() <= 3 && dy.abs() <= 3 {
                    let (dx, dy) = (f64::from(dx), f64::from(dy));
                    let (u, v) = (cs * dx + sn * dy, -sn * dx + cs * dy);
                    sum += amp * (-0.5 * (u * u / l1 + v * v / l2)).exp()
                        / (2.0 * PI * (l1 * l2).sqrt());
                }
            }
            field[Morton8x8::from_xy(x as u8, y as u8).code() as usize] = sum;
        }
    }
    field
}

// ── fixtures ───────────────────────────────────────────────────────────────

/// A stripe of material `s` through background `b`, horizontal or vertical.
fn stripe(s: u8, b: u8, horizontal: bool) -> Tile {
    let mut t = [b; PIXELS];
    for i in 0..16u8 {
        let (x, y) = if horizontal { (i, 8) } else { (8, i) };
        t[Morton8x8::from_xy(x, y).code() as usize] = s;
    }
    t
}

/// First material pair whose Fisher-Z code lies in `range`.
fn pair_with_code(law: &PairwiseFisherZ<'_>, range: std::ops::RangeInclusive<i8>) -> (u8, u8) {
    use cognitive_shader_driver::palette_perturbation::PaletteState;
    use fisher_relation::Relation;
    for s in 0..=255u8 {
        for b in 0..=255u8 {
            if s != b {
                if let Relation::Pair(r) = law.relation(PaletteState(s), PaletteState(b)) {
                    if range.contains(&r) {
                        return (s, b);
                    }
                }
            }
        }
    }
    panic!("no pair with code in {range:?}");
}

/// Only pixel `(x, y)` activated.
fn single(x: u8, y: u8) -> Tile {
    let mut a = [0u8; PIXELS];
    a[Morton8x8::from_xy(x, y).code() as usize] = 255;
    a
}

/// Mass, x-variance and y-variance of a field around `(cx, cy)`.
fn field_moments(f: &Field, cx: u8, cy: u8) -> (f64, f64, f64) {
    let (mut m, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for code in 0..PIXELS as u16 {
        let p = Morton8x8::from_code(code);
        let (dx, dy) = (
            f64::from(p.x()) - f64::from(cx),
            f64::from(p.y()) - f64::from(cy),
        );
        let v = f[code as usize];
        m += v;
        vx += v * dx * dx;
        vy += v * dy * dy;
    }
    (m, vx / m, vy / m)
}

fn main() {
    let table = FisherZTable::build(&representatives(1), 256);
    let law = PairwiseFisherZ::borrow(&table);
    let (tile, act) = (repeated_tile(), activation_fixture());

    let mut b = [0.0; PIXELS];
    let ((), n_alloc, n_bytes) =
        allocations_during(|| render_anisotropic(&tile, &act, &law, &mut b));
    let (a, mat_bytes) = render_materialized(&tile, &act, &law);
    let c = render_gather(&tile, &act, &law);
    let worst = (0..PIXELS).map(|i| (b[i] - c[i]).abs()).fold(0.0, f64::max);

    let (s, bg) = pair_with_code(&law, -100..=-80);
    let (stripe_h, one) = (stripe(s, bg, true), single(8, 8));
    let (mut iso, mut aniso) = ([0.0; PIXELS], [0.0; PIXELS]);
    render_isotropic(&stripe_h, &one, &law, &mut iso);
    render_anisotropic(&stripe_h, &one, &law, &mut aniso);
    let (mi, xi, yi) = field_moments(&iso, 8, 8);
    let (ma, xa, ya) = field_moments(&aniso, 8, 8);

    println!("D-CTX-3 anisotropic EWA accumulation, 16 x 16 Morton tile, eigenvalue floor {MIN_FOOTPRINT_EIGENVALUE}");
    println!("  law generation               : {:#018x}", law.generation);
    println!("  fused == materialized        : {}", a == b);
    println!(
        "  fused vs gather (rotated)    : max |Δ| {worst:.3e} against max field {:.3e}",
        max_abs(&b)
    );
    println!("  horizontal stripe, 1 surfel  : isotropic  mass {mi:.4}  var x {xi:.4}  var y {yi:.4}  peak {:.4}", iso[Morton8x8::from_xy(8, 8).code() as usize]);
    println!("                                 anisotropic mass {ma:.4}  var x {xa:.4}  var y {ya:.4}  peak {:.4}", aniso[Morton8x8::from_xy(8, 8).code() as usize]);
    println!("  fused heap                   : {n_alloc} allocations, {n_bytes} B");
    println!("  materialized                 : surfels + gaussians {mat_bytes} B");
}

#[cfg(test)]
mod tests {
    use super::ewa::{footprint, footprint_sigma, isotropic_scale, RADIUS};
    use super::virtual_surfel::{neighbor, permutation_tile, read_surfel};
    use super::*;

    fn table() -> FisherZTable {
        FisherZTable::build(&representatives(1), 256)
    }

    fn aniso(tile: &Tile, act: &Tile, law: &PairwiseFisherZ<'_>) -> Field {
        let mut f = [0.0; PIXELS];
        render_anisotropic(tile, act, law, &mut f);
        f
    }

    fn iso(tile: &Tile, act: &Tile, law: &PairwiseFisherZ<'_>) -> Field {
        let mut f = [0.0; PIXELS];
        render_isotropic(tile, act, law, &mut f);
        f
    }

    fn tiles(law: &PairwiseFisherZ<'_>) -> Vec<Tile> {
        let (s, b) = pair_with_code(law, -100..=-80);
        let mut t = vec![
            permutation_tile(),
            repeated_tile(),
            stripe(s, b, true),
            stripe(s, b, false),
        ];
        let mut x: u32 = 0x7F4A_7C15;
        for _ in 0..12 {
            t.push(core::array::from_fn(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x >> 24) as u8 % 4 * 60
            }));
        }
        t
    }

    /// Window mass of one footprint at `center`.
    fn window_mass(center: Morton8x8, sigma: &Spd2) -> f64 {
        let mut m = 0.0;
        for dy in -RADIUS..=RADIUS {
            for dx in -RADIUS..=RADIUS {
                if neighbor(center, dx, dy).is_some() {
                    m += footprint(sigma, f64::from(dx), f64::from(dy));
                }
            }
        }
        m
    }

    /// FAILS IF: the fused anisotropic field differs by a bit from the
    /// materialized `Vec<Surfel> → Vec<Gaussian>` field, on 16 tiles.
    #[test]
    fn fused_equals_materialized_bit_for_bit() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let act = activation_fixture();
        for tile in tiles(&law) {
            assert_eq!(
                aniso(&tile, &act, &law),
                render_materialized(&tile, &act, &law).0
            );
        }
    }

    /// FAILS IF: the fused field disagrees with the independent rotated
    /// closed-form gather (own eigen-decomposition and floor) by more than
    /// `1e-12 × max|field|`.
    #[test]
    fn fused_equals_the_independent_rotated_gather() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let act = activation_fixture();
        for tile in tiles(&law) {
            let (b, c) = (aniso(&tile, &act, &law), render_gather(&tile, &act, &law));
            let scale = max_abs(&b);
            assert!(scale > 1.0);
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

    /// FAILS IF: an isotropic reading's anisotropic footprint differs from
    /// D-CTX-2's `s·I`, or a tile whose activated readings are all isotropic
    /// renders differently from D-CTX-2. The second half needs a non-trivial
    /// number of isotropic surfels to mean anything.
    #[test]
    fn the_isotropic_case_collapses_onto_d_ctx_2() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let uniform = [42u8; PIXELS];
        let mut act = [0u8; PIXELS];
        let mut n = 0;
        for y in 1..15u8 {
            for x in 1..15u8 {
                let c = Morton8x8::from_xy(x, y);
                let r = read_surfel(&uniform, c, 1, &law, IDENTITY_WEIGHT);
                assert_eq!(
                    (r.sxx, r.sxy),
                    (r.syy, 0),
                    "interior of a uniform tile is isotropic"
                );
                let (a, i) = (
                    anisotropic_sigma(&r).unwrap(),
                    footprint_sigma(isotropic_scale(&r).unwrap()),
                );
                for (x, y) in [(0.0, 0.0), (1.0, 0.0), (2.0, -1.0), (3.0, 3.0)] {
                    let (fa, fi) = (footprint(&a, x, y), footprint(&i, x, y));
                    assert!((fa - fi).abs() <= 1e-14 * fi, "{fa} vs {fi}");
                }
                act[c.code() as usize] = (x + y) % 7 + 1;
                n += 1;
            }
        }
        assert_eq!(n, 196);
        let (a, i) = (aniso(&uniform, &act, &law), iso(&uniform, &act, &law));
        let scale = max_abs(&i);
        for p in 0..PIXELS {
            assert!((a[p] - i[p]).abs() <= 1e-12 * scale, "pixel {p}");
        }
    }

    /// FAILS IF: on a stripe the anisotropic render does not differ from the
    /// isotropic one, does not spread along the stripe, does not turn with
    /// it, or its peak is not exactly `A / (2π √det Σ_fp)`.
    ///
    /// The peak is not ordered against the isotropic one: without the floor
    /// `det Σ̂ ≤ s²` would make it higher, but on this stripe the minor variance
    /// (~0.30) is raised to the floor 0.5, and the peak drops (pinned).
    #[test]
    fn orientation_reaches_the_rendered_surface() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let (s, b) = pair_with_code(&law, -100..=-80);
        let one = single(8, 8);
        let center = Morton8x8::from_xy(8, 8).code() as usize;
        let (h, v) = (stripe(s, b, true), stripe(s, b, false));

        let (_, ix, iy) = field_moments(&iso(&h, &one, &law), 8, 8);
        assert!(
            (ix - iy).abs() < 1e-12,
            "isotropic render is round: {ix} {iy}"
        );

        let (fh, fv) = (aniso(&h, &one, &law), aniso(&v, &one, &law));
        let (_, hx, hy) = field_moments(&fh, 8, 8);
        let (_, vx, vy) = field_moments(&fv, 8, 8);
        assert!(
            hx > 1.5 * hy,
            "horizontal stripe spreads along x: {hx} {hy}"
        );
        assert!(vy > 1.5 * vx, "vertical stripe spreads along y: {vx} {vy}");
        assert!(
            (hx - vy).abs() < 1e-12 && (hy - vx).abs() < 1e-12,
            "turning the stripe turns the render"
        );

        assert_ne!(fh, iso(&h, &one, &law));
        let r = read_surfel(&h, Morton8x8::from_xy(8, 8), 255, &law, IDENTITY_WEIGHT);
        let raw = normalized_moment(&r).unwrap();
        assert!(
            raw.eig().1 < MIN_FOOTPRINT_EIGENVALUE,
            "the floor binds on this stripe"
        );
        let sigma = anisotropic_sigma(&r).unwrap();
        let peak = amplitude(&r) / (2.0 * PI * sigma.det().sqrt());
        assert!(
            (fh[center] - peak).abs() <= 1e-12 * peak,
            "{} vs {peak}",
            fh[center]
        );
        assert!(
            fh[center] < iso(&h, &one, &law)[center],
            "floored minor axis lowers the peak"
        );
    }

    /// FAILS IF: without the floor the law cannot produce a footprint that
    /// manufactures mass (then the floor would be decoration), or with it any
    /// footprint holds more than 1.001 of its mass, or the law's degenerate
    /// case (cross code −127) is not singular before the floor.
    #[test]
    fn the_floor_prevents_manufactured_mass() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let center = Morton8x8::from_xy(8, 8);

        // Thin, not degenerate: a weak but non-zero cross relation.
        let (s, b) = pair_with_code(&law, -126..=-120);
        let r = read_surfel(&stripe(s, b, true), center, 255, &law, IDENTITY_WEIGHT);
        let raw = normalized_moment(&r).unwrap();
        assert!(raw.is_spd(1e-12));
        let unfloored = window_mass(center, &ewa_sandwich(&raw.sqrt(), &Spd2::I));
        assert!(unfloored > 1.3, "unfloored thin footprint mass {unfloored}");
        let floored = window_mass(center, &anisotropic_sigma(&r).unwrap());
        assert!(floored <= 1.001, "floored mass {floored}");

        // Degenerate: the cross relation at the bottom of the family range.
        let (s, b) = pair_with_code(&law, -127..=-127);
        let r = read_surfel(&stripe(s, b, true), center, 255, &law, IDENTITY_WEIGHT);
        assert!(
            !normalized_moment(&r).unwrap().is_spd(1e-12),
            "code -127 gives a singular moment"
        );
        let m = window_mass(center, &anisotropic_sigma(&r).unwrap());
        assert!(m.is_finite() && m <= 1.001, "degenerate floored mass {m}");

        // Every footprint on every fixture stays within the bound.
        for tile in tiles(&law) {
            for_each_surfel(&tile, &[1; PIXELS], &law, IDENTITY_WEIGHT, |r| {
                if let Some(sigma) = anisotropic_sigma(&r) {
                    let m = window_mass(r.center, &sigma);
                    assert!(m <= 1.001, "{m} at {:?}", r.center);
                }
            });
        }
    }

    /// FAILS IF: the floor ever binds on an isotropic scale (it must not, or
    /// the collapse onto D-CTX-2 would be an accident of the fixtures).
    #[test]
    fn the_law_never_produces_an_isotropic_scale_below_the_floor() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        for tile in tiles(&law) {
            for_each_surfel(&tile, &[1; PIXELS], &law, IDENTITY_WEIGHT, |r| {
                if let Some(s) = isotropic_scale(&r) {
                    assert!(s >= MIN_FOOTPRINT_EIGENVALUE, "s {s}");
                }
            });
        }
    }

    /// FAILS IF: anisotropic rendering creates mass overall, or renders
    /// anything without activation, or the fused path allocates.
    #[test]
    fn anisotropy_shapes_influence_but_creates_no_evidence() {
        let table = table();
        let law = PairwiseFisherZ::borrow(&table);
        let act = activation_fixture();
        for tile in tiles(&law) {
            let total: f64 = aniso(&tile, &act, &law).iter().sum();
            let input: f64 = act.iter().map(|&a| f64::from(a)).sum();
            assert!(total <= input * 1.001, "{total} > {input}");
            assert_eq!(aniso(&tile, &[0; PIXELS], &law), [0.0; PIXELS]);
        }
        let tile = repeated_tile();
        let mut f = [0.0; PIXELS];
        let ((), n, bytes) = allocations_during(|| render_anisotropic(&tile, &act, &law, &mut f));
        assert_eq!((n, bytes), (0, 0));
    }

    /// Cosine of two representative rows (calibration-side reference only).
    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            d += f64::from(*x) * f64::from(*y);
            na += f64::from(*x).powi(2);
            nb += f64::from(*y).powi(2);
        }
        d / (na.sqrt() * nb.sqrt())
    }

    /// FAILS IF: the i8 Fisher-Z quantization moves a surfel's orientation by
    /// more than 1.5° or an eigenvalue by more than 2 % against the same
    /// surfel built from the unquantized cosines (Fisher-Z in f64, same family
    /// range). The orientation check only counts surfels that are clearly
    /// anisotropic (λ₁ / λ₂ > 1.2), and needs at least 100 of them.
    ///
    /// Measured, then pinned: worst 1.11° and 1.27 % over 2,672 anisotropic
    /// surfels on the 16 fixture tiles. One i8 step is 1/254 of the family's z
    /// range, so small weights carry the largest relative error.
    #[test]
    fn i8_quantization_is_sufficient_for_the_footprint() {
        let reps = representatives(1);
        let table = FisherZTable::build(&reps, 256);
        let law = PairwiseFisherZ::borrow(&table);
        let (z_min, z_range) = (f64::from(table.gamma.z_min), f64::from(table.gamma.z_range));
        let true_w = |a: u8, b: u8| -> f64 {
            if a == b {
                return f64::from(IDENTITY_WEIGHT);
            }
            let c = cosine(&reps[a as usize], &reps[b as usize]).clamp(-0.9999, 0.9999);
            (c.atanh() - z_min) / z_range * 254.0
        };
        let (mut worst_angle, mut worst_eig, mut checked) = (0.0f64, 0.0f64, 0);
        for tile in tiles(&law) {
            for code in 0..PIXELS as u16 {
                let center = Morton8x8::from_code(code);
                let q = read_surfel(&tile, center, 1, &law, IDENTITY_WEIGHT);
                let Some(nq) = normalized_moment(&q) else {
                    continue;
                };
                let (mut w, mut a, mut b, mut c) = (0.0, 0.0, 0.0, 0.0);
                for &(dx, dy) in &fisher_relation::MOORE {
                    if let Some(n) = neighbor(center, dx, dy) {
                        let wt = true_w(tile[code as usize], tile[n.code() as usize]);
                        let (dx, dy) = (f64::from(dx), f64::from(dy));
                        w += wt;
                        a += wt * dx * dx;
                        b += wt * dx * dy;
                        c += wt * dy * dy;
                    }
                }
                let nt = Spd2 {
                    a: a / w,
                    b: b / w,
                    c: c / w,
                };
                let (q1, q2, qc, qs) = nq.eig();
                let (t1, t2, tc, ts) = nt.eig();
                worst_eig = worst_eig
                    .max(((q1 - t1) / t1).abs())
                    .max(((q2 - t2) / t2.max(1e-9)).abs().min(1.0));
                if t1 / t2.max(1e-12) > 1.2 {
                    let dot = (qc * tc + qs * ts).abs().min(1.0);
                    worst_angle = worst_angle.max(dot.acos().to_degrees());
                    checked += 1;
                }
            }
        }
        eprintln!("MEASURE angle {worst_angle} eig {worst_eig} checked {checked}");
        assert!(checked >= 100, "only {checked} anisotropic surfels");
        assert!(worst_angle <= 1.5, "worst orientation error {worst_angle}°");
        assert!(worst_eig <= 0.02, "worst eigenvalue error {worst_eig}");
    }
}
