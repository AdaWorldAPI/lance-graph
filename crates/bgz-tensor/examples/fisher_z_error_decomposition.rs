//! Fisher-Z -> i8 -> tanh: where the error comes from (lab probe).
//!
//! For every off-diagonal pair cosine `c` of a family, the round trip
//! `FamilyGamma::decode(FamilyGamma::encode(c))` is split into stages
//! that telescope exactly to the total error:
//!
//! | stage | what changes | error measured against |
//! |---|---|---|
//! | calibration | z clamped to the codable range `[z_min - zr/254, z_min + zr]` and the input to +-0.9999 | the true cosine |
//! | quantization | z rounded to the nearest code (round half to even), decoded in f64 | the calibration-only value |
//! | truncation | the codec's `as i8` truncation instead of rounding | the rounded value |
//! | f32 decode | the codec's f32 decode with `f32::tanh` | the same code decoded in f64 |
//! | tanh kernel | `ndarray::hpc::vml::vstanh` instead of `f32::tanh` on the same f32 z | the codec's f32 decode |
//!
//! `vstanh` is never on the codec's path today; the last row says what it
//! would cost if the decode were batched through it.
//!
//! Families are synthetic and seeded. Real GGUF families are listed in the
//! validation manifest and NOT RUN here.
//!
//! ```text
//! cargo run --release -p bgz-tensor --example fisher_z_error_decomposition
//! ```

use bgz_tensor::fisher_z::FamilyGamma;
use ndarray::hpc::vml::vstanh;
use ndarray::Array1;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// Rows `shared * s + (1 - shared) * noise`: `shared = 0` gives cosines
/// centred on zero, larger values give an all-positive family.
fn family(seed: u64, k: usize, d: usize, shared: f64) -> Vec<Vec<f32>> {
    let mut r = Rng(seed);
    let s: Vec<f64> = (0..d).map(|_| r.normal()).collect();
    (0..k)
        .map(|_| {
            (0..d)
                .map(|j| (shared * s[j] + (1.0 - shared) * r.normal()) as f32)
                .collect()
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa * bb).sqrt()
}

fn pair_cosines(rows: &[Vec<f32>]) -> Vec<f64> {
    let mut out = Vec::new();
    for i in 0..rows.len() {
        for j in i + 1..rows.len() {
            out.push(cosine(&rows[i], &rows[j]));
        }
    }
    out
}

#[derive(Default)]
struct Stat {
    n: usize,
    sum: f64,
    sum_abs: f64,
    max_abs: f64,
}
impl Stat {
    fn add(&mut self, e: f64) {
        self.n += 1;
        self.sum += e;
        self.sum_abs += e.abs();
        self.max_abs = self.max_abs.max(e.abs());
    }
}

fn ulp_distance(a: f32, b: f32) -> u32 {
    let key = |x: f32| {
        let i = x.to_bits() as i32;
        if i < 0 {
            i32::MIN - i
        } else {
            i
        }
    };
    (key(a) as i64 - key(b) as i64).unsigned_abs() as u32
}

/// `fit` calibrates the gamma, `eval` is the cosine set measured.
fn decompose(name: &str, fit: &[f64], eval: &[f64]) {
    let fit_f32: Vec<f32> = fit.iter().map(|&c| c as f32).collect();
    let g = FamilyGamma::from_cosines(&fit_f32);
    let (z_min, zr) = (g.z_min as f64, g.z_range as f64);
    let z_lo = z_min - zr / 254.0;
    let z_hi = z_min + zr;

    // The codec's f32 z for each pair, decoded once by f32::tanh (scalar,
    // the codec) and once by vstanh (batched).
    let codes: Vec<i8> = eval.iter().map(|&c| g.encode(c as f32)).collect();
    let z_f32 = Array1::from_iter(codes.iter().map(|&q| {
        let normalized = (q as f32 + 127.0) / 254.0;
        normalized * g.z_range + g.z_min
    }));
    let mut vs = Array1::<f32>::zeros(z_f32.len());
    vstanh(z_f32.view(), vs.view_mut());

    let mut st: [Stat; 6] = Default::default();
    let mut max_tanh_ulp = 0u32;
    let mut code_min = 0u8;
    let mut clamped = 0usize;
    for (i, &c) in eval.iter().enumerate() {
        let cf = c as f32 as f64;
        let z = cf.clamp(-0.9999, 0.9999).atanh();
        if !(z_lo..=z_hi).contains(&z) {
            clamped += 1;
        }
        let s_cal = z.clamp(z_lo, z_hi).tanh();
        let ideal = (z.clamp(z_lo, z_hi) - z_min) / zr * 254.0 - 127.0;
        let rounded = ideal.round_ties_even().clamp(-128.0, 127.0);
        let s_round = ((rounded + 127.0) / 254.0 * zr + z_min).tanh();
        let q = codes[i];
        if q == -128 {
            code_min += 1;
        }
        let s_trunc = ((q as f64 + 127.0) / 254.0 * zr + z_min).tanh();
        let s_codec = g.decode(q) as f64;
        let s_vs = vs[i] as f64;
        max_tanh_ulp = max_tanh_ulp.max(ulp_distance(vs[i], g.decode(q)));

        st[0].add(s_cal - c);
        st[1].add(s_round - s_cal);
        st[2].add(s_trunc - s_round);
        st[3].add(s_codec - s_trunc);
        st[4].add(s_vs - s_codec);
        st[5].add(s_codec - c);
        // Telescoping check: the stages sum to the codec's total error.
        let parts = (s_cal - c) + (s_round - s_cal) + (s_trunc - s_round) + (s_codec - s_trunc);
        assert!((parts - (s_codec - c)).abs() < 1e-12);
    }

    let lo = eval.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = eval.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "\n{name}: n={} cosine range [{lo:+.4}, {hi:+.4}]  z_min={:+.4} z_range={:.4}  outside codable z: {clamped}  code -128: {code_min}",
        eval.len(),
        g.z_min,
        g.z_range
    );
    println!(
        "  {:<14} {:>12} {:>12} {:>12}",
        "stage", "bias", "mean |e|", "max |e|"
    );
    let names = [
        "calibration",
        "quantization",
        "truncation",
        "f32 decode",
        "tanh kernel",
        "TOTAL codec",
    ];
    for (n, s) in names.iter().zip(&st) {
        println!(
            "  {:<14} {:>+12.3e} {:>12.3e} {:>12.3e}",
            n,
            s.sum / s.n as f64,
            s.sum_abs / s.n as f64,
            s.max_abs
        );
    }
    println!("  vstanh vs f32::tanh on the codec's z: max {max_tanh_ulp} ULP");
}

fn main() {
    println!(
        "fisher_z_error_decomposition  avx512f={}",
        cfg!(target_feature = "avx512f")
    );
    let (k, d) = (256, 64);

    let centred = pair_cosines(&family(1, k, d, 0.0));
    decompose("centred family (shared=0), in-sample", &centred, &centred);

    let positive = pair_cosines(&family(2, k, d, 0.6));
    decompose(
        "all-positive family (shared=0.6), in-sample",
        &positive,
        &positive,
    );

    // Held out: the gamma is fitted on one draw and applied to another draw
    // of the same generator, so some pairs fall outside the fitted range.
    let fit = pair_cosines(&family(3, 64, d, 0.6));
    let held = pair_cosines(&family(4, k, d, 0.6));
    decompose(
        "all-positive, gamma fitted on 64 rows, eval on 256 held-out rows",
        &fit,
        &held,
    );

    // Cosines close to 1, where atanh is steep.
    let near = pair_cosines(&family(5, k, d, 0.9));
    decompose(
        "high-similarity family (shared=0.9), in-sample",
        &near,
        &near,
    );

    // Near-duplicate rows: every cosine exceeds the 0.9999 input clamp, so
    // the whole family collapses onto one z and z_range hits its 1e-10 floor.
    let dup = pair_cosines(&family(6, k, d, 0.995));
    decompose(
        "near-duplicate family (shared=0.995), in-sample",
        &dup,
        &dup,
    );
}
