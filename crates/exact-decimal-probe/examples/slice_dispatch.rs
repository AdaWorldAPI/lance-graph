//! D-AED-5 measurement: a decimal threshold, COUNT and SUM, dispatched once
//! per sealed slice at the slice's own exponent, against one pass over a
//! column held at a single exponent. Run:
//! `cargo run --release --example slice_dispatch`.
//!
//! The column: 1 048 576 amounts, every 9th absent. Slices alternate
//! between scale 2 and scale 3; the single-exponent column holds the same
//! values at scale 3. Width-specific kernels (PM9, PM17) do not exist yet
//! (ndarray #438), so every slice runs on its i32 lane.

use std::hint::black_box;
use std::time::Instant;

use exact_decimal_probe::{slice_count_sum_where, Exp, Op};
use lance_graph_mask_risc::words_for;

fn best(reps: usize, mut f: impl FnMut()) -> f64 {
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as f64
        })
        .fold(f64::MAX, f64::min)
}

fn main() {
    let n = 1usize << 20;
    let lanes64 = n as f64 / 64.0;
    // value i in thousandths: a deterministic spread over [-5, 5).
    let milli = |i: usize| ((i as i64 * 7919) % 10_000 - 5_000) as i32;
    let mut plane = vec![0u64; words_for(n)];
    for i in (0..n).filter(|i| i % 9 != 0) {
        plane[i / 64] |= 1 << (i % 64);
    }
    let at3: Vec<i32> = (0..n)
        .map(|i| if i % 9 == 0 { 0 } else { milli(i) })
        .collect();
    println!("rows {n}; ns per 64 lanes, best of 30; dispatch = (k slices − 1 slice) / k");
    for (label, op, literal) in [("dense", Op::Ge, "-4.5"), ("sparse", Op::Ge, "4.99")] {
        // Baseline: one slice at scale 3.
        let one =
            slice_count_sum_where(&at3, &plane, &[0, n], &[Exp::scale(3)], op, literal).unwrap();
        let t1 = best(30, || {
            black_box(
                slice_count_sum_where(&at3, &plane, &[0, n], &[Exp::scale(3)], op, literal)
                    .unwrap(),
            );
        });
        for k in [16usize, 256] {
            let bounds: Vec<usize> = (0..=k).map(|j| j * n / k).collect();
            let exps: Vec<Exp> = (0..k)
                .map(|j| Exp::scale(if j % 2 == 0 { 2 } else { 3 }))
                .collect();
            // Scale-2 slices hold the value rounded to hundredths exactly:
            // build them from multiples of 10 so both columns hold one value set.
            let coef: Vec<i32> = (0..n)
                .map(|i| {
                    let j = bounds.partition_point(|&b| b <= i) - 1;
                    if i % 9 == 0 {
                        0
                    } else if exps[j] == Exp::scale(2) {
                        milli(i) / 10
                    } else {
                        milli(i)
                    }
                })
                .collect();
            let same: Vec<i32> = (0..n)
                .map(|i| {
                    let j = bounds.partition_point(|&b| b <= i) - 1;
                    if i % 9 != 0 && exps[j] == Exp::scale(2) {
                        (milli(i) / 10) * 10
                    } else {
                        at3[i]
                    }
                })
                .collect();
            let want = slice_count_sum_where(&same, &plane, &[0, n], &[Exp::scale(3)], op, literal)
                .unwrap();
            let got = slice_count_sum_where(&coef, &plane, &bounds, &exps, op, literal).unwrap();
            assert_eq!(got, want, "{label} k={k}: per-slice equals one exponent");
            let tk = best(30, || {
                black_box(
                    slice_count_sum_where(&coef, &plane, &bounds, &exps, op, literal).unwrap(),
                );
            });
            println!(
                "{label:<6} (kept {:>7})  1 slice {:>6.2} ns/64   {k:>3} slices {:>6.2} ns/64   dispatch {:>7.0} ns/slice",
                want.0,
                t1 / lanes64,
                tk / lanes64,
                (tk - t1) / k as f64,
            );
        }
        let _ = one;
    }
}
