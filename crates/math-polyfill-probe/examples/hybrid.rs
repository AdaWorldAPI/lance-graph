//! D-MPF-3 (lab): dense/sparse hybrid masked sum, and one mask reused across
//! several reductions. Reported, not gated. Nothing here changes ndarray or
//! R2IL; a winning body is a proposal for those owners.
//!
//! Part 1, the hybrid sum. Per 64-bit mask word: skip an empty word, walk the
//! set bits (`trailing_zeros`, the current `ndarray::simd::masked_sum_i32`
//! body) when the word holds at most `th` bits, else sum all 64 lanes
//! branch-free. `th = 64` is the walk alone, `th = 0` is branch-free alone.
//! Masks are drawn two ways at the same overall density: `uniform` (each bit
//! independent) and `clustered` (whole words set or clear), because a
//! per-word threshold can only help when density varies between words.
//!
//! Part 2, mask reuse. count, sum, min and max of `x > t`:
//! `recompute` builds the mask once per reduction, `reuse` builds it once,
//! `reuse+hybrid` also uses the hybrid sum, `fused scalar` does all four in
//! one row loop. Every arm is checked against an `i128` oracle before timing.

use std::hint::black_box;
use std::time::Instant;

use ndarray::simd::{
    gt_i32_to_mask, masked_max_i32, masked_min_i32, masked_sum_i32, popcount_batch_u64,
};

fn med(reps: usize, iters: usize, mut f: impl FnMut() -> i64) -> f64 {
    let mut v: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..iters {
                black_box(f());
            }
            t.elapsed().as_nanos() as f64 / iters as f64
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// SplitMix64: deterministic, no dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// True with probability `p_permille / 1000`.
    fn hit(&mut self, p_permille: u64) -> bool {
        self.next() % 1000 < p_permille
    }
}

fn mask_uniform(n: usize, p: u64, r: &mut Rng) -> Vec<u64> {
    let mut m = vec![0u64; n.div_ceil(64)];
    for i in 0..n {
        if r.hit(p) {
            m[i / 64] |= 1 << (i % 64);
        }
    }
    m
}

fn mask_clustered(n: usize, p: u64, r: &mut Rng) -> Vec<u64> {
    (0..n.div_ceil(64))
        .map(|_| if r.hit(p) { u64::MAX } else { 0 })
        .collect()
}

fn oracle_sum(x: &[i32], m: &[u64]) -> i64 {
    let s: i128 = (0..x.len())
        .filter(|&i| m[i / 64] >> (i % 64) & 1 == 1)
        .map(|i| i128::from(x[i]))
        .sum();
    i64::try_from(s).unwrap()
}

/// The proposed body. `x.len()` must be a multiple of 64 (true for every size
/// used here); the tail case is the existing walk's business.
#[inline(never)]
fn hybrid_sum(x: &[i32], m: &[u64], th: u32) -> i64 {
    let mut acc = 0i64;
    let (chunks, _) = x.as_chunks::<64>();
    for (w, chunk) in chunks.iter().enumerate() {
        let word = m[w];
        let c = word.count_ones();
        if c == 0 {
            continue;
        }
        if c > th {
            let mut part = 0i64;
            for (j, &v) in chunk.iter().enumerate() {
                part += i64::from(v) & -(((word >> j) & 1) as i64);
            }
            acc += part;
        } else {
            let mut bits = word;
            while bits != 0 {
                acc += i64::from(chunk[bits.trailing_zeros() as usize]);
                bits &= bits - 1;
            }
        }
    }
    acc
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct Four {
    count: i64,
    sum: i64,
    min: Option<i32>,
    max: Option<i32>,
}

fn four_oracle(x: &[i32], t: i32) -> Four {
    let sel: Vec<i32> = x.iter().copied().filter(|&v| v > t).collect();
    Four {
        count: sel.len() as i64,
        sum: i64::try_from(sel.iter().map(|&v| i128::from(v)).sum::<i128>()).unwrap(),
        min: sel.iter().copied().min(),
        max: sel.iter().copied().max(),
    }
}

#[inline(never)]
fn four_recompute(x: &[i32], t: i32, m: &mut [u64]) -> Four {
    gt_i32_to_mask(x, t, m);
    let count = popcount_batch_u64(m) as i64;
    gt_i32_to_mask(x, t, m);
    let sum = masked_sum_i32(x, m);
    gt_i32_to_mask(x, t, m);
    let min = masked_min_i32(x, m);
    gt_i32_to_mask(x, t, m);
    let max = masked_max_i32(x, m);
    Four {
        count,
        sum,
        min,
        max,
    }
}

#[inline(never)]
fn four_reuse(x: &[i32], t: i32, m: &mut [u64], hybrid: Option<u32>) -> Four {
    gt_i32_to_mask(x, t, m);
    Four {
        count: popcount_batch_u64(m) as i64,
        sum: match hybrid {
            Some(th) => hybrid_sum(x, m, th),
            None => masked_sum_i32(x, m),
        },
        min: masked_min_i32(x, m),
        max: masked_max_i32(x, m),
    }
}

#[inline(never)]
fn four_fused_scalar(x: &[i32], t: i32) -> Four {
    let (mut count, mut sum, mut min, mut max) = (0i64, 0i64, i32::MAX, i32::MIN);
    for &v in x {
        let s = v > t;
        count += i64::from(s);
        sum += if s { i64::from(v) } else { 0 };
        min = if s && v < min { v } else { min };
        max = if s && v > max { v } else { max };
    }
    Four {
        count,
        sum,
        min: (count > 0).then_some(min),
        max: (count > 0).then_some(max),
    }
}

fn pack(f: Four) -> i64 {
    f.count ^ f.sum ^ i64::from(f.min.unwrap_or(0)) ^ i64::from(f.max.unwrap_or(0))
}

fn main() {
    let reps = 21;
    let n = 65_536;
    let mut r = Rng(0x5EED);
    let x: Vec<i32> = (0..n).map(|_| (r.next() % 2001) as i32 - 1000).collect();

    println!("## Part 1: hybrid masked sum, {n} rows, ns per call (median of {reps})\n");
    let ths = [64u32, 48, 32, 16, 8, 0];
    print!("| mask | density | walk (ndarray) |");
    for th in ths {
        print!(" th={th} |");
    }
    println!();
    println!("|---|---:|---:|{}", "---:|".repeat(ths.len()));
    for (kind, gen) in [
        (
            "uniform",
            mask_uniform as fn(usize, u64, &mut Rng) -> Vec<u64>,
        ),
        ("clustered", mask_clustered),
    ] {
        for p in [0u64, 10, 100, 250, 500, 750, 1000] {
            let m = gen(n, p, &mut r);
            let want = oracle_sum(&x, &m);
            assert_eq!(masked_sum_i32(&x, &m), want);
            let walk = med(reps, 200, || masked_sum_i32(black_box(&x), black_box(&m)));
            print!("| {kind} | {:.1} % | {walk:.0} |", p as f64 / 10.0);
            for th in ths {
                assert_eq!(hybrid_sum(&x, &m, th), want, "{kind} p={p} th={th}");
                let t = med(reps, 200, || hybrid_sum(black_box(&x), black_box(&m), th));
                print!(" {t:.0} |");
            }
            println!();
        }
    }

    println!(
        "\n## Part 2: one mask, four reductions (count, sum, min, max of x > t), ns per call\n"
    );
    println!("| rows | t | selected | recompute ×4 | reuse | reuse + hybrid(16) | fused scalar |");
    println!("|---:|---:|---:|---:|---:|---:|---:|");
    for nn in [64usize, 4_096, 65_536] {
        let xs = &x[..nn];
        let mut m = vec![0u64; nn.div_ceil(64)];
        let iters = (200 * 65_536 / nn).max(200);
        for t in [900, 0, -900] {
            let want = four_oracle(xs, t);
            assert_eq!(four_recompute(xs, t, &mut m), want);
            assert_eq!(four_reuse(xs, t, &mut m, None), want);
            assert_eq!(four_reuse(xs, t, &mut m, Some(16)), want);
            assert_eq!(four_fused_scalar(xs, t), want);
            let a = med(reps, iters, || {
                pack(four_recompute(black_box(xs), black_box(t), &mut m))
            });
            let b = med(reps, iters, || {
                pack(four_reuse(black_box(xs), black_box(t), &mut m, None))
            });
            let c = med(reps, iters, || {
                pack(four_reuse(black_box(xs), black_box(t), &mut m, Some(16)))
            });
            let d = med(reps, iters, || {
                pack(four_fused_scalar(black_box(xs), black_box(t)))
            });
            println!(
                "| {nn} | {t} | {} | {a:.0} | {b:.0} | {c:.0} | {d:.0} |",
                want.count
            );
        }
    }
    println!("\nReported, not gated.");
}
