//! W4 (D-NALP-4) measurement: how exactly each candidate realization of the
//! per-row truth laws reproduces the exact law, over EVERY input
//! (`(f1, c1, f2, c2)` ∈ 256⁴ = 2³² rows per law).
//!
//! Shapes measured against `tests/common/truth_law.rs` (exact integers, one
//! half-up rounding):
//! - `table`: 256×256 `u8` laws composed, rounding at every stage
//!   (`mul[a][b] = round(a·b/255)`, `w2c[x] = round(255·x/(x+255))`);
//! - `f32-dec`: `causal_edge::isa::truth`'s arithmetic, codes decoded to
//!   `x/255` in `f32`, packed with `(v·255).round()`;
//! - `f32-int`: the exact integer formula evaluated in `f32` (products of
//!   codes, one division by the constant, one rounding);
//! - `f64-int`: the same in `f64`.
//!
//! Run: `cargo run --release --example w4_truth_law_probe`.

#[path = "../tests/common/truth_law.rs"]
mod truth_law;

use std::thread;

type Shape = fn(&str, u8, u8, u8, u8) -> (u8, u8);

fn mul_table() -> Vec<u8> {
    let mut t = vec![0u8; 256 * 256];
    for a in 0..256u64 {
        for b in 0..256u64 {
            t[(a * 256 + b) as usize] = truth_law::round_div(a * b, 255);
        }
    }
    t
}

fn w2c_table() -> Vec<u8> {
    (0..256u64)
        .map(|x| truth_law::round_div(255 * x, x + 255))
        .collect()
}

thread_local! {
    static MUL: Vec<u8> = mul_table();
    static W2C: Vec<u8> = w2c_table();
}

fn table(name: &str, f1: u8, c1: u8, f2: u8, c2: u8) -> (u8, u8) {
    MUL.with(|m| {
        W2C.with(|w| {
            let mul = |a: u8, b: u8| m[usize::from(a) * 256 + usize::from(b)];
            match name {
                "deduction" => {
                    let f = mul(f1, f2);
                    (f, mul(f, mul(c1, c2)))
                }
                "analogy" => (mul(f1, f2), mul(mul(c1, c2), f2)),
                "induction" => (f2, w[usize::from(mul(mul(f1, c1), c2))]),
                "abduction" => (f1, w[usize::from(mul(mul(f2, c1), c2))]),
                _ => unreachable!(),
            }
        })
    })
}

fn pack(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn f32_dec(name: &str, f1: u8, c1: u8, f2: u8, c2: u8) -> (u8, u8) {
    use causal_edge::isa::truth;
    let d = |x: u8| f32::from(x) / 255.0;
    let (f, c) = match name {
        "deduction" => truth::deduction(d(f1), d(c1), d(f2), d(c2)),
        "induction" => truth::induction(d(f1), d(c1), d(f2), d(c2)),
        "abduction" => truth::abduction(d(f1), d(c1), d(f2), d(c2)),
        "analogy" => (d(f1) * d(f2), d(c1) * d(c2) * d(f2)),
        _ => unreachable!(),
    };
    (pack(f), pack(c))
}

macro_rules! int_shape {
    ($fname:ident, $t:ty) => {
        fn $fname(name: &str, f1: u8, c1: u8, f2: u8, c2: u8) -> (u8, u8) {
            let x = |v: u8| <$t>::from(v);
            let r = |p: $t, q: $t| (p / q + 0.5).floor() as u8;
            let (k1, k2, k3): ($t, $t, $t) = (255.0, 65025.0, 16581375.0);
            match name {
                "deduction" => {
                    let ff = x(f1) * x(f2);
                    (r(ff, k1), r(ff * x(c1) * x(c2), k3))
                }
                "analogy" => (r(x(f1) * x(f2), k1), r(x(c1) * x(c2) * x(f2), k2)),
                "induction" => {
                    let w = x(f1) * x(c1) * x(c2);
                    (f2, r(255.0 * w, w + k3))
                }
                "abduction" => {
                    let w = x(f2) * x(c1) * x(c2);
                    (f1, r(255.0 * w, w + k3))
                }
                _ => unreachable!(),
            }
        }
    };
}
int_shape!(f32_int, f32);
int_shape!(f64_int, f64);

const SHAPES: [(&str, Shape); 4] = [
    ("table", table),
    ("f32-dec", f32_dec),
    ("f32-int", f32_int),
    ("f64-int", f64_int),
];

#[derive(Default, Clone, Copy)]
struct Tally {
    rows_off: u64,
    max_df: u8,
    max_dc: u8,
}

fn measure(name: &'static str, law: truth_law::Law) -> [Tally; 4] {
    let threads = 4u16;
    let parts: Vec<[Tally; 4]> = thread::scope(|s| {
        (0..threads)
            .map(|t| {
                s.spawn(move || {
                    let mut tally = [Tally::default(); 4];
                    let mut f1 = t;
                    while f1 < 256 {
                        let f1b = f1 as u8;
                        for c1 in 0..=255u8 {
                            for f2 in 0..=255u8 {
                                for c2 in 0..=255u8 {
                                    let (ef, ec) = law(f1b, c1, f2, c2);
                                    for (i, (_, shape)) in SHAPES.iter().enumerate() {
                                        let (sf, sc) = shape(name, f1b, c1, f2, c2);
                                        let df = ef.abs_diff(sf);
                                        let dc = ec.abs_diff(sc);
                                        let tl = &mut tally[i];
                                        tl.rows_off += u64::from(df != 0 || dc != 0);
                                        tl.max_df = tl.max_df.max(df);
                                        tl.max_dc = tl.max_dc.max(dc);
                                    }
                                }
                            }
                        }
                        f1 += threads;
                    }
                    tally
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("worker"))
            .collect()
    });
    let mut out = [Tally::default(); 4];
    for p in parts {
        for i in 0..4 {
            out[i].rows_off += p[i].rows_off;
            out[i].max_df = out[i].max_df.max(p[i].max_df);
            out[i].max_dc = out[i].max_dc.max(p[i].max_dc);
        }
    }
    out
}

fn main() {
    println!(
        "tier: avx512f={} avx2={} (scalar probe; exactness, not speed)",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2")
    );
    println!("rows per law: 2^32 = {}", 1u64 << 32);
    println!(
        "{:<10} {:<8} {:>14} {:>8} {:>8}",
        "law", "shape", "rows off", "max df", "max dc"
    );
    for (name, law) in truth_law::LAWS {
        let t = measure(name, law);
        for (i, (shape, _)) in SHAPES.iter().enumerate() {
            println!(
                "{:<10} {:<8} {:>14} {:>8} {:>8}",
                name, shape, t[i].rows_off, t[i].max_df, t[i].max_dc
            );
        }
    }
}
