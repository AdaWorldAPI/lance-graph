//! Softmax precision arms over a Palette256 codebook, using ndarray's own FMA.
//!
//! stdin, per case (5 lines):
//!   id beta_hex n d
//!   x[0..256] as f64 hex bits         (codebook representatives, any order)
//!   codes[0..n]                        (u8)
//!   v[0..n*d] as f64 hex bits          (row-major, element-major)
//!   vint[0..n*d]                       (i32; the same values when integral)
//! stdout, per case:
//!   A id S N_0..N_{d-1}                (exact integers; F24 table)
//!   B id y_0..                         (F24 table, f64 sum, mul_add_f64 dot)
//!   M id y_0..                         (same, separate multiply + add)
//!   U id y_0..                         (unary f64 table e^{beta x[q]}, no row max)
//!   C id y_0..                         (per-element f64 exp, Neumaier sum, FMA dot)
//!   PB/PU/PC id p_0..                  (materialized probabilities)
//! `--bench`: ns/element per arm.
use ndarray::simd::mul_add_f64;
use std::hint::black_box;
use std::io::{self, BufRead, Write};
use std::time::Instant;

const F: i32 = 24;

fn hx(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).unwrap())
}

/// F24 weight of code a relative to the row maximum m, prepared once per
/// (codebook, beta): round(2^24 * exp(beta * (x[a] - x[m]))).
fn pair_table(x: &[f64; 256], beta: f64) -> Vec<u32> {
    let mut t = vec![0u32; 256 * 256];
    for m in 0..256 {
        for a in 0..256 {
            if x[a] <= x[m] {
                t[m * 256 + a] = ((beta * (x[a] - x[m])).exp() * (1u64 << F) as f64).round() as u32;
            }
        }
    }
    t
}

/// Neumaier compensated sum.
fn nsum(it: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for v in it {
        let t = s + v;
        c += if s.abs() >= v.abs() { (s - t) + v } else { (v - t) + s };
        s = t;
    }
    s + c
}

struct Case {
    beta: f64,
    x: [f64; 256],
    q: Vec<usize>,
    v: Vec<f64>,
    vi: Vec<i64>,
    d: usize,
}

fn row_max(c: &Case) -> usize {
    *c.q.iter().max_by(|&&a, &&b| c.x[a].partial_cmp(&c.x[b]).unwrap()).unwrap()
}

fn run(id: &str, c: &Case, out: &mut impl Write) {
    let n = c.q.len();
    let m = row_max(c);
    let pt = pair_table(&c.x, c.beta);
    let w: Vec<u32> = c.q.iter().map(|&a| pt[m * 256 + a]).collect();
    // A: exact integers.
    let s: u64 = w.iter().map(|&v| v as u64).sum();
    let nn: Vec<i128> = (0..c.d)
        .map(|k| (0..n).map(|i| w[i] as i128 * c.vi[i * c.d + k] as i128).sum())
        .collect();
    write!(out, "A {id} {s}").unwrap();
    for v in &nn { write!(out, " {v}").unwrap(); }
    writeln!(out).unwrap();
    // B / M: F24 weights as f64 (exact), sequential sum; FMA vs separate.
    let wf: Vec<f64> = w.iter().map(|&v| v as f64).collect();
    let sf: f64 = wf.iter().sum();
    let line = |tag: &str, fma: bool, out: &mut dyn Write| {
        write!(out, "{tag} {id}").unwrap();
        for k in 0..c.d {
            let mut acc = 0.0f64;
            for i in 0..n {
                acc = if fma { mul_add_f64(wf[i], c.v[i * c.d + k], acc) } else { wf[i] * c.v[i * c.d + k] + acc };
            }
            write!(out, " {:x}", (acc / sf).to_bits()).unwrap();
        }
        writeln!(out).unwrap();
    };
    line("B", true, out);
    line("M", false, out);
    write!(out, "PB {id}").unwrap();
    for &v in &wf { write!(out, " {:x}", (v / sf).to_bits()).unwrap(); }
    writeln!(out).unwrap();
    // U: unary f64 table relative to the codebook maximum (no row max).
    let xmax = c.x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = c.x.iter().map(|&xv| (c.beta * (xv - xmax)).exp()).collect();
    let wu: Vec<f64> = c.q.iter().map(|&a| e[a]).collect();
    let su: f64 = wu.iter().sum();
    write!(out, "U {id}").unwrap();
    for k in 0..c.d {
        let mut acc = 0.0f64;
        for i in 0..n { acc = mul_add_f64(wu[i], c.v[i * c.d + k], acc); }
        write!(out, " {:x}", (acc / su).to_bits()).unwrap();
    }
    writeln!(out).unwrap();
    write!(out, "PU {id}").unwrap();
    for &v in &wu { write!(out, " {:x}", (v / su).to_bits()).unwrap(); }
    writeln!(out).unwrap();
    // C: per-element f64 exp against the row max, compensated sums.
    let wc: Vec<f64> = c.q.iter().map(|&a| (c.beta * (c.x[a] - c.x[m])).exp()).collect();
    let sc = nsum(wc.iter().cloned());
    write!(out, "C {id}").unwrap();
    for k in 0..c.d {
        let yk = nsum((0..n).map(|i| wc[i] * c.v[i * c.d + k]));
        write!(out, " {:x}", (yk / sc).to_bits()).unwrap();
    }
    writeln!(out).unwrap();
    write!(out, "PC {id}").unwrap();
    for &v in &wc { write!(out, " {:x}", (v / sc).to_bits()).unwrap(); }
    writeln!(out).unwrap();
}

fn bench() {
    let n = 4096usize;
    let reps = 4000;
    let mut x = [0.0f64; 256];
    for (i, v) in x.iter_mut().enumerate() { *v = -1.0 + 2.0 * i as f64 / 255.0; }
    let beta = 4.0;
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let q: Vec<u8> = (0..n).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 56) as u8 }).collect();
    let v: Vec<f64> = (0..n).map(|i| ((i * 37 % 255) as f64 - 127.0) / 64.0).collect();
    let vi: Vec<i64> = (0..n).map(|i| (i * 37 % 255) as i64 - 127).collect();
    let pt = pair_table(&x, beta);
    let diff: Vec<u32> = (0..256).map(|d| ((beta * (x[1] - x[0]) * -(d as f64)).exp() * (1u64 << F) as f64).round() as u32).collect();
    let e: Vec<f64> = x.iter().map(|&xv| (beta * (xv - 1.0)).exp()).collect();
    let mut report = |name: &str, f: &mut dyn FnMut() -> f64| {
        for _ in 0..50 { black_box(f()); }
        let t = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps { acc += f(); }
        black_box(acc);
        println!("{name:<44} {:>7.3} ns/elem", t.elapsed().as_nanos() as f64 / (reps * n) as f64);
    };
    report("A diff table (affine): max+sub+gather+u64/i128", &mut || {
        let m = *q.iter().max().unwrap() as usize;
        let (mut s, mut nn) = (0u64, 0i128);
        for i in 0..n { let w = diff[m - q[i] as usize]; s += w as u64; nn += w as i128 * vi[i] as i128; }
        nn as f64 / s as f64
    });
    report("A pair table (any palette): gather u64/i128", &mut || {
        let m = *q.iter().max_by(|&&a, &&b| x[a as usize].partial_cmp(&x[b as usize]).unwrap()).unwrap() as usize;
        let (mut s, mut nn) = (0u64, 0i128);
        for i in 0..n { let w = pt[m * 256 + q[i] as usize]; s += w as u64; nn += w as i128 * vi[i] as i128; }
        nn as f64 / s as f64
    });
    report("B F24 table + f64 mul_add_f64", &mut || {
        let m = *q.iter().max().unwrap() as usize;
        let (mut s, mut acc) = (0.0f64, 0.0f64);
        for i in 0..n { let w = diff[m - q[i] as usize] as f64; s += w; acc = mul_add_f64(w, v[i], acc); }
        acc / s
    });
    report("U unary f64 table + mul_add_f64 (no max)", &mut || {
        let (mut s, mut acc) = (0.0f64, 0.0f64);
        for i in 0..n { let w = e[q[i] as usize]; s += w; acc = mul_add_f64(w, v[i], acc); }
        acc / s
    });
    report("C per-element f64 exp + Neumaier", &mut || {
        let m = q.iter().map(|&a| x[a as usize]).fold(f64::NEG_INFINITY, f64::max);
        let w = q.iter().map(|&a| (beta * (x[a as usize] - m)).exp());
        let ws: Vec<f64> = w.collect();
        // y = N / S (was S / N: inverted in the Amendment 5 run; timing unaffected)
        nsum((0..n).map(|i| ws[i] * v[i])) / nsum(ws.iter().cloned())
    });
}

fn main() {
    if std::env::args().any(|a| a == "--bench") { bench(); return; }
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines().map(|l| l.unwrap());
    let mut out = io::BufWriter::new(io::stdout());
    while let Some(h) = lines.next() {
        let h: Vec<String> = h.split_whitespace().map(String::from).collect();
        if h.is_empty() { continue; }
        let (id, beta, n, d) = (h[0].clone(), hx(&h[1]), h[2].parse::<usize>().unwrap(), h[3].parse::<usize>().unwrap());
        let xs: Vec<f64> = lines.next().unwrap().split_whitespace().map(hx).collect();
        let mut x = [0.0; 256];
        x.copy_from_slice(&xs);
        let q: Vec<usize> = lines.next().unwrap().split_whitespace().map(|s| s.parse().unwrap()).collect();
        let v: Vec<f64> = lines.next().unwrap().split_whitespace().map(hx).collect();
        let vi: Vec<i64> = lines.next().unwrap().split_whitespace().map(|s| s.parse().unwrap()).collect();
        assert_eq!((q.len(), v.len(), vi.len()), (n, n * d, n * d));
        run(&id, &Case { beta, x, q, v, vi, d }, &mut out);
    }
}
