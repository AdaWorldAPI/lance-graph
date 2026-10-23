//! D-MPF-2 follow-up: split the ndarray floor (`gt_i32_to_mask` then
//! `masked_sum_i32`) into its two calls and sweep selectivity. Reported, not
//! gated. Selectivity is set by the threshold over a lane in `[-48, 48]`:
//! `t = 100` selects nothing, `t = 0` about half, `t = -100` every row.

use std::hint::black_box;
use std::time::Instant;

fn med(reps: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    let mut v: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..iters {
                f();
            }
            t.elapsed().as_nanos() as f64 / iters as f64
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The proposed `masked_sum_i32` body (see `codegen_probe`).
#[inline(never)]
fn masked_sum_branchless(x: &[i32], m: &[u64]) -> i64 {
    let mut acc = 0i64;
    for (w, chunk) in x.chunks(64).enumerate() {
        let word = m[w];
        let mut part = 0i64;
        for (j, &v) in chunk.iter().enumerate() {
            part += i64::from(v) & -(((word >> j) & 1) as i64);
        }
        acc += part;
    }
    acc
}

#[inline(never)]
fn scalar(x: &[i32], t: i32) -> i64 {
    x.iter()
        .map(|&v| if v > t { i64::from(v) } else { 0 })
        .sum()
}

fn main() {
    let n = 65_536;
    let x: Vec<i32> = (0..n).map(|i| ((i * 13) % 97) as i32 - 48).collect();
    let mut m = vec![0u64; n / 64];
    println!("| t | selected | compare | masked sum | branch-free sum | scalar | ns/row compare | ns/row sum | ns/row branch-free |");
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for t in [100, 0, -100] {
        ndarray::simd::gt_i32_to_mask(&x, t, &mut m);
        let sel: u32 = m.iter().map(|w| w.count_ones()).sum();
        assert_eq!(ndarray::simd::masked_sum_i32(&x, &m), scalar(&x, t));
        let c = med(21, 200, || {
            ndarray::simd::gt_i32_to_mask(black_box(&x), t, black_box(&mut m))
        });
        ndarray::simd::gt_i32_to_mask(&x, t, &mut m);
        let s = med(21, 200, || {
            black_box(ndarray::simd::masked_sum_i32(black_box(&x), black_box(&m)));
        });
        assert_eq!(masked_sum_branchless(&x, &m), scalar(&x, t));
        let bf = med(21, 200, || {
            black_box(masked_sum_branchless(black_box(&x), black_box(&m)));
        });
        let sc = med(21, 200, || {
            black_box(scalar(black_box(&x), black_box(t)));
        });
        println!(
            "| {t} | {sel} | {c:.0} | {s:.0} | {bf:.0} | {sc:.0} | {:.3} | {:.3} | {:.3} |",
            c / n as f64,
            s / n as f64,
            bf / n as f64
        );
    }
}
