//! D-MPF-1 cost split, reported not gated. For each row count it times the
//! stages separately, each as the median over repetitions of a batch:
//!
//! - `bind`: `FoldDialect::new` (allocates the scratch once).
//! - `count`: selection + `COUNT` on a reused dialect (one fold).
//! - `expr`: the whole expression on a reused dialect (two folds + scalar ops).
//! - `count+bind` and `keep+bind`: a fresh dialect each time; their
//!   difference is the cost of materializing the selection into a mask.
//! - `relalg`: `relalg::run` of the same selection count (normalize + lower +
//!   one fold), fresh dialect inside.
//! - `exec`: mask-risc `execute_into` of the count program relalg lowered,
//!   prebuilt, over a reused scratch buffer: the fold alone, no lowering.
//! - `oracle`: the scalar row loop.
//!
//! `MPF_REPS` (default 21) and `MPF_ITERS` (default 2000) set the sampling.

use std::hint::black_box;
use std::time::Instant;

use lance_graph_mask_risc::{execute_into, words_for, Foreign, LaneRef, Out, Planes, Scratch};
use lance_graph_r2il::procs::{and_cmp_i32, count, is_not_null, keep, Order};
use lance_graph_r2il::FoldDialect;
use lance_graph_relalg::{run, run_with_program, Cmp, Measure, Query, Rel};
use math_polyfill_probe::{eval, oracle, Expr, Layout};

const L: Layout = Layout {
    x: 0,
    y: 1,
    v: 2,
    valid: 0,
};
const E: Expr = Expr {
    lo: -5,
    hi: 20,
    k: 3,
};

fn env(name: &str, d: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// Median ns per call over `reps` batches of `iters` calls.
fn time(reps: usize, iters: usize, mut f: impl FnMut()) -> f64 {
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

fn main() {
    let reps = env("MPF_REPS", 21);
    let iters0 = env("MPF_ITERS", 2000);
    println!(
        "| rows | bind | exec | count | expr | count+bind | keep+bind | relalg | oracle | expr ns/row |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for n in [64usize, 512, 4_096, 65_536] {
        let iters = (iters0 * 64 / n).max(20);
        let x: Vec<i32> = (0..n).map(|i| (i % 50) as i32 - 20).collect();
        let y: Vec<i32> = (0..n).map(|i| ((i * 7) % 31) as i32).collect();
        let v: Vec<i32> = (0..n).map(|i| ((i * 13) % 97) as i32 - 48).collect();
        let mut valid = vec![0u64; words_for(n)];
        for r in (0..n).filter(|r| r % 7 != 3) {
            valid[r / 64] |= 1 << (r % 64);
        }
        let lanes = [LaneRef::I32(&x), LaneRef::I32(&y), LaneRef::I32(&v)];
        let masks: [&[u64]; 1] = [&valid];
        let p = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &lanes,
        };

        // Correctness first: a timing of a wrong answer is not a measurement.
        let want = oracle(&x, &y, &v, &valid, E).1;
        assert_eq!(
            eval(&mut FoldDialect::new(&p, &Foreign::NONE), L, E).ok(),
            want
        );

        let sel = |d: &mut FoldDialect<'_>, s: &mut Vec<_>| {
            is_not_null(d, s, L.valid).unwrap();
            and_cmp_i32(d, s, L.x, Order::Ge, E.lo).unwrap();
            and_cmp_i32(d, s, L.y, Order::Lt, E.hi).unwrap();
        };
        let bind = time(reps, iters, || {
            black_box(FoldDialect::new(black_box(&p), &Foreign::NONE));
        });
        let mut d = FoldDialect::new(&p, &Foreign::NONE);
        let mut s = Vec::with_capacity(8);
        let cnt = time(reps, iters, || {
            sel(&mut d, &mut s);
            count(&mut d, &mut s).unwrap();
            black_box(s.pop());
        });
        let expr = time(reps, iters, || {
            black_box(eval(&mut d, L, E).unwrap());
        });
        let cnt_bind = time(reps, iters, || {
            let mut d = FoldDialect::new(&p, &Foreign::NONE);
            let mut s = Vec::with_capacity(8);
            sel(&mut d, &mut s);
            count(&mut d, &mut s).unwrap();
            black_box(s.pop());
        });
        let mut sink = vec![0u64; words_for(n)];
        let keep_bind = time(reps, iters, || {
            let mut d = FoldDialect::with_keep(&p, &Foreign::NONE, &mut sink);
            let mut s = Vec::with_capacity(8);
            sel(&mut d, &mut s);
            keep(&mut d, &mut s).unwrap();
        });
        black_box(&sink);
        let q = Query {
            rel: Rel::And(vec![
                Rel::Plane(L.valid),
                Rel::i32(L.x, Cmp::Ge, E.lo),
                Rel::i32(L.y, Cmp::Lt, E.hi),
            ]),
            measure: Measure::Count,
        };
        let rel = time(reps, iters, || {
            black_box(run(black_box(&q), &p, None).unwrap());
        });
        let (_, prog) = run_with_program(&q, &p, None).unwrap();
        let prog = prog.expect("the selection is not symbolic");
        let mut buf = vec![0u64; 1 << 16];
        let exec = time(reps, iters, || {
            let mut sc = Scratch::over_for_program(&mut buf, &prog, n).unwrap();
            black_box(execute_into(&prog, &p, &Foreign::NONE, &mut sc, Out::None).unwrap());
        });
        let orc = time(reps, iters, || {
            black_box(oracle(black_box(&x), &y, &v, &valid, E));
        });
        println!(
            "| {n} | {bind:.0} | {exec:.0} | {cnt:.0} | {expr:.0} | {cnt_bind:.0} | {keep_bind:.0} | {rel:.0} | {orc:.0} | {:.2} |",
            expr / n as f64
        );
    }
    println!("\nAll figures ns per call (median). Reported, not gated.");
}
