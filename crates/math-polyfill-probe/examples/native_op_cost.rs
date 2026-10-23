//! D-MPF-1b cost split for the native `SumWhere` op, reported not gated.
//!
//! - `bind`: build `Planes` and `FoldDialect::new` (the setup a caller pays
//!   once per population).
//! - `hot`: `SumWhere::eval` on a reused dialect and a reused stack.
//! - `hot allocs`: heap allocations per hot call, counted by a global
//!   allocator wrapper (the hot path's allocation budget).
//! - `onnx wrapper`: `reduce_sum_where_greater`, which binds on every call.
//! - `oracle`: the `i128` row loop.
//!
//! `MPF_REPS` (default 21) and `MPF_ITERS` (default 2000) set the sampling.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use lance_graph_mask_risc::{Foreign, LaneRef, Planes};
use lance_graph_r2il::FoldDialect;
use math_polyfill_probe::native::{sum_where_oracle, Pred, SumWhere};
use math_polyfill_probe::onnx_compat::reduce_sum_where_greater;

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards every call to `System` unchanged; only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn env(name: &str, d: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

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
    let op = SumWhere {
        lane: 0,
        present: None,
        pred: Pred::Gt(0),
    };
    println!("| rows | bind | hot | hot allocs | onnx wrapper | oracle | hot ns/row |");
    println!("|---:|---:|---:|---:|---:|---:|---:|");
    for n in [64usize, 4_096, 65_536] {
        let iters = (iters0 * 64 / n).max(20);
        let x: Vec<i32> = (0..n).map(|i| ((i * 13) % 97) as i32 - 48).collect();
        let lanes = [LaneRef::I32(&x)];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };

        let want = sum_where_oracle(&x, None, op.pred);
        let mut d = FoldDialect::new(&planes, &Foreign::NONE);
        let mut s = Vec::with_capacity(8);
        assert_eq!(op.eval(&mut d, &mut s).map(i128::from), Ok(want));
        let w = reduce_sum_where_greater(&x, &[n], 0, &[], None, 0, 0).unwrap();
        assert_eq!(i128::from(w.value), want);

        let bind = time(reps, iters, || {
            let lanes = [LaneRef::I32(black_box(&x))];
            let p = Planes {
                n_rows: n,
                masks: &[],
                lanes: &lanes,
            };
            black_box(FoldDialect::new(&p, &Foreign::NONE));
        });
        let hot = time(reps, iters, || {
            black_box(op.eval(&mut d, &mut s).unwrap());
        });
        let before = ALLOCS.load(Ordering::Relaxed);
        for _ in 0..100 {
            black_box(op.eval(&mut d, &mut s).unwrap());
        }
        let allocs = (ALLOCS.load(Ordering::Relaxed) - before) as f64 / 100.0;
        let wrap = time(reps, iters, || {
            black_box(reduce_sum_where_greater(black_box(&x), &[n], 0, &[], None, 0, 0).unwrap());
        });
        let orc = time(reps, iters, || {
            black_box(sum_where_oracle(black_box(&x), None, op.pred));
        });
        println!(
            "| {n} | {bind:.0} | {hot:.0} | {allocs:.1} | {wrap:.0} | {orc:.0} | {:.2} |",
            hot / n as f64
        );
    }
    println!("\nns per call, median. Reported, not gated.");
}
