//! Symmetric signed-i8 quantization: mathematical probe (lab only).
//!
//! Proposed code space: `-127..=127` are values, `-128` is reserved as the
//! invalid / NaN sentinel. Nothing here changes `FamilyGamma` or any wire
//! format; the existing codec is only measured.
//!
//! Hypotheses kept apart:
//! - H1 representational symmetry: zero is the exact centre of the valid
//!   code range.
//! - H2 quantization symmetry: `encode(-x) == -encode(x)` under one
//!   zero-centred scale and one rounding rule.
//! - H3 distributional symmetry: the data has median zero. H1 does not imply it.
//! - H4 a symmetric code removes the integer half-bin offset; it does not
//!   remove calibration offsets that asymmetric data needs.
//!
//! Run with `--nocapture` to print the comparison tables.

use bgz_tensor::fisher_z::{FamilyGamma, FisherZTable};

// ── the proposed codec ─────────────────────────────────────────────────────

mod sym {
    pub const SENTINEL: i8 = i8::MIN;

    /// Round-half-to-even on `x / scale`, saturating at ±127. NaN gives the
    /// sentinel; ±inf saturates.
    pub fn encode(x: f32, scale: f32) -> i8 {
        if x.is_nan() {
            return SENTINEL;
        }
        (x / scale).round_ties_even().clamp(-127.0, 127.0) as i8
    }

    pub fn decode(c: i8, scale: f32) -> f32 {
        if c == SENTINEL {
            f32::NAN
        } else {
            c as f32 * scale
        }
    }

    pub fn neg(c: i8) -> i8 {
        if c == SENTINEL {
            SENTINEL
        } else {
            -c
        }
    }

    /// Saturating add; either sentinel gives the sentinel.
    pub fn add(a: i8, b: i8) -> i8 {
        if a == SENTINEL || b == SENTINEL {
            SENTINEL
        } else {
            (a as i16 + b as i16).clamp(-127, 127) as i8
        }
    }
}

// ── deterministic generator ────────────────────────────────────────────────

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

// ── B2: exhaustive code-space tests ────────────────────────────────────────

#[test]
fn proposed_code_space_has_255_values_one_sentinel_and_one_zero() {
    let scale = 0.01f32;
    let all: Vec<i8> = (i8::MIN..=i8::MAX).collect();
    assert_eq!(all.len(), 256);
    let valid: Vec<i8> = all
        .iter()
        .copied()
        .filter(|&c| c != sym::SENTINEL)
        .collect();
    assert_eq!(valid.len(), 255);
    assert!(sym::decode(sym::SENTINEL, scale).is_nan());
    assert_eq!(
        valid
            .iter()
            .filter(|&&c| sym::decode(c, scale) == 0.0)
            .count(),
        1
    );
    // H1: as many negative as positive valid codes, zero in the middle.
    assert_eq!(valid.iter().filter(|&&c| c < 0).count(), 127);
    assert_eq!(valid.iter().filter(|&&c| c > 0).count(), 127);
    for &c in &valid {
        let n = sym::neg(c);
        assert_ne!(
            n,
            sym::SENTINEL,
            "negating a valid code never hits the sentinel"
        );
        assert_eq!(sym::neg(n), c);
        assert_eq!(
            sym::decode(n, scale),
            -sym::decode(c, scale),
            "decoded symmetry at {c}"
        );
        assert_eq!(
            sym::encode(sym::decode(c, scale), scale),
            c,
            "decode/encode round trip at {c}"
        );
    }
    assert_eq!(sym::neg(sym::SENTINEL), sym::SENTINEL);

    // Conventional i8 is not symmetric: 128 negative codes, 127 positive,
    // and -128 has no negation.
    assert_eq!(all.iter().filter(|&&c| c < 0).count(), 128);
    assert_eq!(i8::MIN.checked_neg(), None);
    assert_eq!(i8::MIN.wrapping_neg(), i8::MIN);
    assert_eq!(i8::MIN.saturating_neg(), i8::MAX);
    assert_eq!(i8::MIN.unsigned_abs(), 128u8, "abs(-128) does not fit i8");
}

#[test]
fn proposed_add_is_odd_and_commutative_over_all_256x256_pairs() {
    let mut conventional_odd_failures = 0u32;
    for a in i8::MIN..=i8::MAX {
        for b in i8::MIN..=i8::MAX {
            let s = sym::add(a, b);
            assert_eq!(s, sym::add(b, a));
            if a == sym::SENTINEL || b == sym::SENTINEL {
                assert_eq!(s, sym::SENTINEL, "sentinel propagates");
                continue;
            }
            assert_ne!(s, sym::SENTINEL, "valid inputs never produce the sentinel");
            assert_eq!(
                sym::add(sym::neg(a), sym::neg(b)),
                sym::neg(s),
                "odd at ({a}, {b})"
            );
            // Conventional saturating i8 add: -a only exists when a != -128.
            if let (Some(na), Some(nb)) = (a.checked_neg(), b.checked_neg()) {
                if na.saturating_add(nb).checked_neg() != Some(a.saturating_add(b)) {
                    conventional_odd_failures += 1;
                }
            }
        }
    }
    // e.g. (-1) + (-127) saturates to -128, but 1 + 127 saturates to 127.
    assert!(conventional_odd_failures > 0);
    println!("conventional saturating_add oddness failures among 255x255 negatable pairs: {conventional_odd_failures}");
}

#[test]
fn proposed_rounding_is_symmetric_with_deterministic_ties() {
    let scale = 0.01f32;
    // H2: encode(-x) == -encode(x) for every tested x, including exact half codes.
    let mut rng = Rng(7);
    for k in -200i32..=200 {
        for x in [
            (k as f32 + 0.5) * scale,
            k as f32 * scale,
            (rng.uniform() as f32 - 0.5) * 3.0,
        ] {
            assert_eq!(
                sym::encode(-x, scale),
                sym::neg(sym::encode(x, scale)),
                "x = {x}"
            );
        }
    }
    // Ties go to the even code, so ±0.5, ±2.5 land on 0 and ±2.
    let tie = |v: f32| sym::encode(v * scale, scale);
    assert_eq!(
        [
            tie(0.5),
            tie(-0.5),
            tie(1.5),
            tie(-1.5),
            tie(2.5),
            tie(-2.5)
        ],
        [0, 0, 2, -2, 2, -2]
    );
    assert_eq!(sym::encode(f32::INFINITY, scale), 127);
    assert_eq!(sym::encode(f32::NEG_INFINITY, scale), -127);
    assert_eq!(sym::encode(f32::NAN, scale), sym::SENTINEL);
    assert_eq!(sym::encode(-0.0, scale), 0);
    // The common `floor(x + 0.5)` rule is not symmetric: +0.5 -> 1, -0.5 -> 0.
    let floor_half = |v: f32| (v + 0.5).floor();
    assert_ne!(floor_half(-0.5), -floor_half(0.5));
}

// ── B4: the existing FamilyGamma contract, measured ────────────────────────

#[test]
fn family_gamma_emits_minus_128_only_below_its_calibrated_range() {
    let g = FamilyGamma {
        z_min: -0.5,
        z_range: 1.0,
    };
    // In range the code space is -127..=127.
    for i in 0..=10_000 {
        let z = g.z_min + g.z_range * (i as f32 / 10_000.0);
        let c = g.encode(z.tanh());
        assert!(c >= -127, "in-range z = {z} encoded to {c}");
    }
    // One full code below z_min clamps to -128, which decodes as a value
    // (z_min - z_range/254), not as an invalid marker.
    let below = (g.z_min - 2.0 * g.z_range / 254.0).tanh();
    assert_eq!(g.encode(below), -128);
    assert!(g.decode(-128).is_finite());
    // NaN input encodes to code 0, the family's mid-z.
    assert_eq!(g.encode(f32::NAN), 0);
}

#[test]
fn fisher_z_table_build_never_stores_minus_128() {
    // build() fits z_min over the table's own off-diagonal cosines and the
    // diagonal is the maximum, so every stored z is >= z_min: no entry can
    // clamp below -127. Property-checked over random tables.
    let mut rng = Rng(1);
    let mut min_seen = i8::MAX;
    for trial in 0..300 {
        let k = 2 + (rng.next_u64() % 40) as usize;
        let d = 2 + (rng.next_u64() % 31) as usize;
        let rows: Vec<Vec<f32>> = (0..k)
            .map(|_| (0..d).map(|_| rng.normal() as f32).collect())
            .collect();
        let t = FisherZTable::build(&rows, k);
        let m = *t.entries.iter().min().unwrap();
        min_seen = min_seen.min(m);
        assert!(m >= -127, "trial {trial}: k={k} d={d} stored {m}");
    }
    assert_eq!(min_seen, -127, "the lowest off-diagonal reaches -127");
}

#[test]
fn family_gamma_truncation_biases_codes_toward_the_mid_code() {
    // encode uses `as i8`, which truncates toward zero, so every code moves
    // toward code 0 (the family's mid-z) by up to one whole bin; decode adds
    // no half-bin back. Measured on the exact real-valued code.
    let g = FamilyGamma {
        z_min: -1.0,
        z_range: 2.0,
    };
    let n = 200_000;
    let (mut pos_sum, mut pos_n, mut neg_sum, mut neg_n) = (0.0f64, 0u32, 0.0f64, 0u32);
    let (mut round_sum, mut round_n) = (0.0f64, 0u32);
    for i in 0..n {
        let z = g.z_min + g.z_range * ((i as f64 + 0.5) / n as f64) as f32;
        let ideal = ((z - g.z_min) / g.z_range) as f64 * 254.0 - 127.0;
        let got = g.encode(z.tanh()) as f64;
        let err = got - ideal;
        if ideal > 0.5 {
            pos_sum += err;
            pos_n += 1;
        } else if ideal < -0.5 {
            neg_sum += err;
            neg_n += 1;
        }
        round_sum += ideal.round_ties_even() - ideal;
        round_n += 1;
    }
    let (pos, neg, round) = (
        pos_sum / pos_n as f64,
        neg_sum / neg_n as f64,
        round_sum / round_n as f64,
    );
    println!("FamilyGamma mean code error: ideal>0 {pos:+.4}, ideal<0 {neg:+.4}; round-to-nearest {round:+.4}");
    assert!(
        (pos + 0.5).abs() < 0.02,
        "positive side biased down by half a bin: {pos}"
    );
    assert!(
        (neg - 0.5).abs() < 0.02,
        "negative side biased up by half a bin: {neg}"
    );
    assert!(round.abs() < 0.01);
}

#[test]
fn family_gamma_code_zero_is_the_mid_z_not_cosine_zero() {
    // H1 does not give H3: for a family whose cosines are all positive,
    // code 0 (also what lookup_i8 returns for an unknown centroid) decodes
    // to a clearly positive cosine.
    let g = FamilyGamma::from_cosines(&[0.1, 0.3, 0.5, 0.7, 0.9]);
    let c0 = g.decode(0);
    println!(
        "all-positive family: code 0 decodes to cosine {c0:.4}; cosine 0 encodes to {}",
        g.encode(0.0)
    );
    assert!(c0 > 0.4);
    // Cosine 0 lies below this family's calibrated range, so the clamp emits
    // -128: an ordinary input lands on the code a sentinel scheme would reserve.
    assert_eq!(g.encode(0.0), -128);
}

// ── B3: scheme comparison over synthetic distributions ─────────────────────

#[derive(Clone, Copy, Debug)]
enum Scheme {
    /// -127..=127, scale = max|x|/127, round half to even.
    Symmetric,
    /// -128..=127 affine min-max, integer zero point, round half to even.
    Affine,
    /// 0..=127 affine min-max (7-bit unsigned).
    Unsigned7,
    /// FamilyGamma's mapping applied to the values: [min,max] -> -127..127
    /// by truncation, clamped to -128..127.
    LegacyTrunc,
}

fn quantize(scheme: Scheme, xs: &[f64]) -> (Vec<f64>, Vec<i32>) {
    let lo = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let amax = xs.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-30);
    let range = (hi - lo).max(1e-30);
    let mut dec = Vec::with_capacity(xs.len());
    let mut codes = Vec::with_capacity(xs.len());
    for &x in xs {
        let (c, d) = match scheme {
            Scheme::Symmetric => {
                let s = amax / 127.0;
                let c = (x / s).round_ties_even().clamp(-127.0, 127.0);
                (c as i32, c * s)
            }
            Scheme::Affine => {
                let s = range / 255.0;
                let zp = (-lo / s).round_ties_even() - 128.0;
                let c = ((x / s).round_ties_even() + zp).clamp(-128.0, 127.0);
                (c as i32, (c - zp) * s)
            }
            Scheme::Unsigned7 => {
                let s = range / 127.0;
                let c = ((x - lo) / s).round_ties_even().clamp(0.0, 127.0);
                (c as i32, c * s + lo)
            }
            Scheme::LegacyTrunc => {
                let v = ((x - lo) / range * 254.0 - 127.0)
                    .clamp(-128.0, 127.0)
                    .trunc();
                (v as i32, (v + 127.0) / 254.0 * range + lo)
            }
        };
        codes.push(c);
        dec.push(d);
    }
    (dec, codes)
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0;
        for &k in &idx[i..=j] {
            r[k] = avg;
        }
        i = j + 1;
    }
    r
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = a.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        c += (ra[i] - ma) * (rb[i] - mb);
        va += (ra[i] - ma).powi(2);
        vb += (rb[i] - mb).powi(2);
    }
    c / (va * vb).sqrt()
}

#[derive(Debug, Clone, Copy)]
struct Metrics {
    bias: f64,
    median: f64,
    mae: f64,
    rmse: f64,
    max_abs: f64,
    zero_kept: f64,
    sat: f64,
    bins: usize,
    entropy: f64,
    rho: f64,
    near_zero_mae: f64,
}

fn measure(scheme: Scheme, xs: &[f64]) -> Metrics {
    let (dec, codes) = quantize(scheme, xs);
    let n = xs.len() as f64;
    let mut errs: Vec<f64> = dec.iter().zip(xs).map(|(d, x)| d - x).collect();
    let bias = errs.iter().sum::<f64>() / n;
    let mae = errs.iter().map(|e| e.abs()).sum::<f64>() / n;
    let rmse = (errs.iter().map(|e| e * e).sum::<f64>() / n).sqrt();
    let max_abs = errs.iter().map(|e| e.abs()).fold(0.0, f64::max);
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = errs[errs.len() / 2];
    let zeros: Vec<usize> = (0..xs.len()).filter(|&i| xs[i] == 0.0).collect();
    let zero_kept = if zeros.is_empty() {
        f64::NAN
    } else {
        zeros.iter().filter(|&&i| dec[i] == 0.0).count() as f64 / zeros.len() as f64
    };
    let (cmin, cmax) = match scheme {
        Scheme::Symmetric => (-127, 127),
        Scheme::Affine | Scheme::LegacyTrunc => (-128, 127),
        Scheme::Unsigned7 => (0, 127),
    };
    let sat = codes.iter().filter(|&&c| c == cmin || c == cmax).count() as f64 / n;
    let mut hist = std::collections::HashMap::new();
    for &c in &codes {
        *hist.entry(c).or_insert(0u32) += 1;
    }
    let entropy = hist
        .values()
        .map(|&k| {
            let p = k as f64 / n;
            -p * p.log2()
        })
        .sum();
    let amax = xs.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let near: Vec<f64> = (0..xs.len())
        .filter(|&i| xs[i].abs() < 0.05 * amax)
        .map(|i| (dec[i] - xs[i]).abs())
        .collect();
    let near_zero_mae = if near.is_empty() {
        f64::NAN
    } else {
        near.iter().sum::<f64>() / near.len() as f64
    };
    Metrics {
        bias,
        median,
        mae,
        rmse,
        max_abs,
        zero_kept,
        sat,
        bins: hist.len(),
        entropy,
        rho: spearman(xs, &dec),
        near_zero_mae,
    }
}

fn distributions() -> Vec<(&'static str, Vec<f64>)> {
    let n = 20_000;
    let mut r = Rng(20_261_024);
    let mut gen =
        |f: &mut dyn FnMut(&mut Rng) -> f64| (0..n).map(|_| f(&mut r)).collect::<Vec<f64>>();
    let sparse = |p0: f64| move |r: &mut Rng| if r.uniform() < p0 { 0.0 } else { r.normal() };
    vec![
        ("uniform symmetric", gen(&mut |r| r.uniform() * 2.0 - 1.0)),
        ("gaussian centred", gen(&mut |r| r.normal())),
        ("gaussian shifted +4σ", gen(&mut |r| r.normal() + 4.0)),
        (
            "laplace",
            gen(&mut |r| {
                let u = r.uniform() - 0.5;
                -u.signum() * (1.0 - 2.0 * u.abs()).ln()
            }),
        ),
        (
            "student-t ν=3",
            gen(&mut |r| {
                let z = r.normal();
                let chi = (0..3).map(|_| r.normal().powi(2)).sum::<f64>();
                z / (chi / 3.0).sqrt()
            }),
        ),
        ("exponential (skewed)", gen(&mut |r| -r.uniform().ln())),
        (
            "bimodal -2 / +3",
            gen(&mut |r| {
                if r.uniform() < 0.5 {
                    -2.0 + 0.4 * r.normal()
                } else {
                    3.0 + 0.4 * r.normal()
                }
            }),
        ),
        (
            "multimodal 4",
            gen(&mut |r| [-3.0, -1.0, 0.5, 2.5][(r.next_u64() % 4) as usize] + 0.2 * r.normal()),
        ),
        ("sparse 90% zero", gen(&mut sparse(0.90))),
        ("sparse 99% zero", gen(&mut sparse(0.99))),
        ("sparse 99.9% zero", gen(&mut sparse(0.999))),
        (
            "sparse asym (+exp / -tiny)",
            gen(&mut |r| {
                let u = r.uniform();
                if u < 0.90 {
                    0.0
                } else if u < 0.97 {
                    -r.uniform().ln()
                } else {
                    -0.05 * r.uniform()
                }
            }),
        ),
        (
            "on symmetric half-codes",
            gen(&mut |r| {
                let k = (r.next_u64() % 254) as f64 - 127.0;
                (k + 0.5).clamp(-127.0, 127.0) / 127.0
            }),
        ),
    ]
}

#[test]
fn scheme_comparison_over_synthetic_distributions() {
    let schemes = [
        Scheme::Symmetric,
        Scheme::Affine,
        Scheme::Unsigned7,
        Scheme::LegacyTrunc,
    ];
    let mut table = std::collections::HashMap::new();
    println!(
        "{:<28} {:<11} {:>10} {:>10} {:>9} {:>9} {:>9} {:>6} {:>6} {:>4} {:>6} {:>8} {:>9}",
        "distribution",
        "scheme",
        "bias",
        "median",
        "MAE",
        "RMSE",
        "max|e|",
        "zero",
        "sat",
        "bins",
        "H",
        "rho",
        "nearzero"
    );
    for (name, xs) in distributions() {
        for s in schemes {
            let m = measure(s, &xs);
            println!(
                "{:<28} {:<11} {:>+10.2e} {:>+10.2e} {:>9.2e} {:>9.2e} {:>9.2e} {:>6.3} {:>6.4} {:>4} {:>6.3} {:>8.6} {:>9.2e}",
                name, format!("{s:?}"), m.bias, m.median, m.mae, m.rmse, m.max_abs, m.zero_kept, m.sat, m.bins, m.entropy, m.rho, m.near_zero_mae
            );
            table.insert((name, format!("{s:?}")), m);
        }
    }
    let get = |d: &'static str, s: &str| table[&(d, s.to_string())];

    // Symmetric keeps exact zeros; legacy truncation and 7-bit unsigned do not.
    for d in ["sparse 90% zero", "sparse 99% zero", "sparse 99.9% zero"] {
        assert_eq!(get(d, "Symmetric").zero_kept, 1.0, "{d}");
        assert!(get(d, "LegacyTrunc").zero_kept < 1.0, "{d}");
    }
    // Zero-centred data: symmetric costs at most the one reserved code
    // against the full affine range (255 vs 256 levels over a symmetric range).
    let (s, a) = (
        get("gaussian centred", "Symmetric").rmse,
        get("gaussian centred", "Affine").rmse,
    );
    assert!(
        s <= a * 1.10,
        "centred gaussian: symmetric {s} vs affine {a}"
    );
    // Falsifier for "symmetric is universally better": shifted data uses
    // under half the symmetric code space, and affine wins clearly.
    let (s, a) = (
        get("gaussian shifted +4σ", "Symmetric"),
        get("gaussian shifted +4σ", "Affine"),
    );
    assert!(
        a.rmse * 1.5 < s.rmse,
        "shifted: affine {} vs symmetric {}",
        a.rmse,
        s.rmse
    );
    assert!(
        s.bins < 128,
        "shifted data occupies {} symmetric bins",
        s.bins
    );
    let (s, a) = (
        get("exponential (skewed)", "Symmetric"),
        get("exponential (skewed)", "Affine"),
    );
    assert!(
        a.rmse * 1.5 < s.rmse,
        "exponential: affine {} vs symmetric {}",
        a.rmse,
        s.rmse
    );
    // Legacy truncation carries a signed bias the rounding schemes do not.
    let lt = get("uniform symmetric", "LegacyTrunc");
    let sy = get("uniform symmetric", "Symmetric");
    assert!(
        lt.mae > sy.mae * 1.5,
        "truncation MAE {} vs symmetric {}",
        lt.mae,
        sy.mae
    );
}
