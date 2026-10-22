//! P-D3: flag out-of-support input by distance to the codebook, not by
//! readout magnitude (pre-registered in
//! `.claude/board/entries/2026-10-22-coresearch-pal256-evidence-probes.md`).
//!
//! A bounded readout says nothing about support: for a piecewise-linear
//! score, scaling the input pushes confidence to the bound (Hein et al.
//! 1812.05720, Thm 3.1), and cosine to the nearest entry does not move at
//! all under scaling. Distance to the nearest entry does (Thm 3.2). This
//! probe compares the two on the production `WeightPalette::assign`
//! (distortion) and the production `FamilyGamma` readout.
//!
//! Data is synthetic: 16 clusters in Base17 space, a training set the
//! palette is built from, an in-support hold-out from the same clusters,
//! and two far-field sets. It is a mechanism probe, not a calibration
//! claim on real weights.
//!
//! Pre-registered conditions (on the ×4 far-field set):
//! - kill: distance AUROC < 0.90, or readout-magnitude AUROC within 0.05 of
//!   the distance AUROC;
//! - silent: the radius flag (`distortion > radii[index]`) fires on at most
//!   5% of the in-support hold-out;
//! - clamp counter: zero on in-range cosines, non-zero on saturating ones.

use bgz_tensor::fisher_z::FamilyGamma;
use bgz_tensor::palette::WeightPalette;
use bgz_tensor::projection::{Base17, BASE_DIM};

/// Deterministic LCG, so the probe is reproducible without a dependency.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    /// Uniform in [-r, r].
    fn sym(&mut self, r: i32) -> i32 {
        (self.next() % (2 * r as u64 + 1)) as i32 - r
    }
}

fn point(rng: &mut Lcg, center: &Base17, noise: i32) -> Base17 {
    let mut dims = [0i16; BASE_DIM];
    for (d, c) in dims.iter_mut().zip(center.dims.iter()) {
        *d = (i32::from(*c) + rng.sym(noise)).clamp(-32768, 32767) as i16;
    }
    Base17 { dims }
}

fn scaled(b: &Base17, s: i32) -> Base17 {
    let mut dims = [0i16; BASE_DIM];
    for (d, x) in dims.iter_mut().zip(b.dims.iter()) {
        *d = (i32::from(*x) * s).clamp(-32768, 32767) as i16;
    }
    Base17 { dims }
}

struct World {
    palette: WeightPalette,
    gamma: FamilyGamma,
    held_out: Vec<Base17>,
    far_scaled: Vec<Base17>,
    far_random: Vec<Base17>,
}

fn world() -> World {
    let mut rng = Lcg(0x005E_EDD3);
    let centers: Vec<Base17> = (0..16)
        .map(|_| {
            let mut dims = [0i16; BASE_DIM];
            for d in dims.iter_mut() {
                *d = rng.sym(1500) as i16;
            }
            Base17 { dims }
        })
        .collect();
    let draw = |rng: &mut Lcg, n: usize| -> Vec<Base17> {
        (0..n)
            .map(|i| point(rng, &centers[i % centers.len()], 120))
            .collect()
    };
    let train = draw(&mut rng, 16 * 40);
    let held_out = draw(&mut rng, 16 * 25);
    let palette = WeightPalette::build(&train, 16);
    let mut cosines = Vec::new();
    for (i, a) in palette.entries.iter().enumerate() {
        for b in &palette.entries[i + 1..] {
            cosines.push(a.cosine(b) as f32);
        }
    }
    let gamma = FamilyGamma::from_cosines(&cosines);
    let far_scaled = held_out.iter().map(|b| scaled(b, 4)).collect();
    let far_random = (0..held_out.len())
        .map(|_| {
            let mut dims = [0i16; BASE_DIM];
            for d in dims.iter_mut() {
                *d = rng.sym(1500) as i16;
            }
            Base17 { dims }
        })
        .collect();
    World {
        palette,
        gamma,
        held_out,
        far_scaled,
        far_random,
    }
}

/// OOD score from distance: larger distortion means further from support.
fn distance_score(w: &World, b: &Base17) -> f64 {
    f64::from(w.palette.assign(b).distortion)
}

/// OOD score from the bounded readout: a lower code toward the nearest
/// entry means less confident, so the score is its negation.
fn magnitude_score(w: &World, b: &Base17) -> f64 {
    let a = w.palette.assign(b);
    let cos = b.cosine(&w.palette.entries[a.index as usize]) as f32;
    -f64::from(w.gamma.encode(cos))
}

/// P(score(ood) > score(in)), ties counted as half.
fn auroc(inside: &[f64], ood: &[f64]) -> f64 {
    let mut wins = 0.0;
    for &o in ood {
        for &i in inside {
            wins += if o > i {
                1.0
            } else if o == i {
                0.5
            } else {
                0.0
            };
        }
    }
    wins / (inside.len() * ood.len()) as f64
}

fn radius_flag(w: &World, b: &Base17) -> bool {
    let a = w.palette.assign(b);
    a.distortion > w.palette.radii[a.index as usize]
}

#[test]
fn distance_detects_scaled_far_field_and_magnitude_does_not() {
    let w = world();
    let score = |f: fn(&World, &Base17) -> f64, set: &[Base17]| -> Vec<f64> {
        set.iter().map(|b| f(&w, b)).collect()
    };
    let d_in = score(distance_score, &w.held_out);
    let m_in = score(magnitude_score, &w.held_out);
    for (name, far) in [("x4", &w.far_scaled), ("random", &w.far_random)] {
        let d = auroc(&d_in, &score(distance_score, far));
        let m = auroc(&m_in, &score(magnitude_score, far));
        println!("{name}: distance AUROC {d:.3}, magnitude AUROC {m:.3}");
        if name == "x4" {
            assert!(d >= 0.90, "kill: distance AUROC {d}");
            assert!(m < d - 0.05, "kill: magnitude {m} within 0.05 of {d}");
        }
    }
}

#[test]
fn radius_flag_stays_quiet_in_support_and_fires_far_out() {
    let w = world();
    let rate = |set: &[Base17]| -> f64 {
        set.iter().filter(|b| radius_flag(&w, b)).count() as f64 / set.len() as f64
    };
    let (fpr, tpr) = (rate(&w.held_out), rate(&w.far_scaled));
    println!("radius flag: in-support {fpr:.3}, x4 far field {tpr:.3}");
    assert!(fpr <= 0.05, "silent arm: flag fires on {fpr} of in-support");
    assert!(tpr >= 0.90, "flag misses the far field: {tpr}");
}

/// Inputs `FamilyGamma::encode` clamps silently: outside the family's z
/// range, or inside `atanh_clamp`'s ±0.9999 saturation.
fn clamp_count(g: &FamilyGamma, cosines: &[f32]) -> usize {
    cosines
        .iter()
        .filter(|&&c| {
            let z = c.clamp(-0.9999, 0.9999).atanh();
            c.abs() >= 0.9999 || z < g.z_min || z > g.z_min + g.z_range
        })
        .count()
}

#[test]
fn clamp_counter_is_zero_in_range_and_counts_saturation() {
    let w = world();
    let mut in_range = Vec::new();
    for (i, a) in w.palette.entries.iter().enumerate() {
        for b in &w.palette.entries[i + 1..] {
            in_range.push(a.cosine(b) as f32);
        }
    }
    let saturating: Vec<f32> = w
        .held_out
        .iter()
        .map(|b| b.cosine(&scaled(b, 4)) as f32)
        .chain([1.0, -1.0, 0.99995])
        .collect();
    let (n_in, n_sat) = (
        clamp_count(&w.gamma, &in_range),
        clamp_count(&w.gamma, &saturating),
    );
    println!(
        "clamp count: in-range {n_in}/{}, saturating {n_sat}/{}",
        in_range.len(),
        saturating.len()
    );
    assert_eq!(n_in, 0);
    assert!(n_sat > 0);
    // The production encoder gives saturated inputs the same code as the
    // in-range extreme: nothing in its output marks them.
    assert_eq!(w.gamma.encode(1.0), w.gamma.encode(0.99995));
}
