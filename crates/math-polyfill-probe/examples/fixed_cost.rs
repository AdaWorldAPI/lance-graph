//! D-MPF-2: where the fixed cost of one native fold goes. Reported, not gated.
//!
//! One question, `Σ { x[r] : x[r] > t }` over a resident `i32` lane, answered
//! by six arms that add one layer each, all on the same `&[i32]`:
//!
//! - `scalar`: a branch-free `i64` row loop (the compiler may vectorize it),
//!   threshold passed through `black_box` so it is not specialised.
//! - `nd floor`: `ndarray::simd::gt_i32_to_mask` + `masked_sum_i32` into a
//!   reused mask buffer, the two calls mask-risc delegates to.
//! - `mr compiled`: mask-risc `execute_compiled` on a program recognised once,
//!   reused scratch.
//! - `mr into`: `execute_into` on a prebuilt program (adds recognition).
//! - `mr build`: `Program::new` + `Scratch::over` + `execute_into` per call,
//!   what the dialect's fold boundary does.
//! - `dialect`: `SumWhere::eval` on a reused `FoldDialect`.
//!
//! Every arm is checked against an `i128` oracle at every size before timing.
//! Allocations per call are counted by a global allocator wrapper.
//! `MPF_REPS` (default 21) and `MPF_ITERS` (default 4000) set the sampling.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_compiled, execute_into, tile_words_for, Scratch};
use lance_graph_mask_risc::ir::{
    CompiledProgram, MaskOp, Operand, Pred as MPred, Program, Terminal,
};
use lance_graph_mask_risc::{words_for, Foreign, LaneRef, Out, Planes, Value};
use lance_graph_r2il::FoldDialect;
use math_polyfill_probe::native::{sum_where_oracle, Pred, SumWhere};

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

/// Median ns per call, plus heap allocations per call.
fn measure(reps: usize, iters: usize, mut f: impl FnMut()) -> (f64, f64) {
    let before = ALLOCS.load(Ordering::Relaxed);
    for _ in 0..100 {
        f();
    }
    let allocs = (ALLOCS.load(Ordering::Relaxed) - before) as f64 / 100.0;
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
    (v[v.len() / 2], allocs)
}

const T: i32 = 0;

fn program() -> Program {
    Program::new(
        vec![MaskOp::Pred {
            pred: MPred::GtI32 { lane: 0, t: T },
            under: None,
            dst: 0,
        }],
        Terminal::MaskedSumI32 {
            mask: Operand::Scratch(0),
            lane: 0,
        },
    )
}

fn sum_of(v: Value) -> i64 {
    match v {
        Value::SumI64(s) => s,
        other => panic!("MaskedSumI32 yields SumI64, got {other:?}"),
    }
}

#[inline(never)]
fn scalar(x: &[i32], t: i32) -> i64 {
    x.iter()
        .map(|&v| if v > t { i64::from(v) } else { 0 })
        .sum()
}

#[inline(never)]
fn nd_floor(x: &[i32], m: &mut [u64]) -> i64 {
    ndarray::simd::gt_i32_to_mask(x, T, m);
    ndarray::simd::masked_sum_i32(x, m)
}

fn main() {
    let reps = env("MPF_REPS", 21);
    let iters0 = env("MPF_ITERS", 4000);
    let op = SumWhere {
        lane: 0,
        present: None,
        pred: Pred::Gt(T),
    };
    let arms = [
        "scalar",
        "nd floor",
        "mr compiled",
        "mr into",
        "mr build",
        "dialect",
    ];
    print!("| rows |");
    for a in arms {
        print!(" {a} |");
    }
    println!(" allocs/call (per arm) |");
    println!("|---:|{}---|", "---:|".repeat(arms.len()));
    for n in [64usize, 512, 4_096, 16_384, 65_536] {
        let iters = (iters0 * 64 / n).max(50);
        let x: Vec<i32> = (0..n).map(|i| ((i * 13) % 97) as i32 - 48).collect();
        let lanes = [LaneRef::I32(&x)];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let want = sum_where_oracle(&x, None, op.pred);

        let mut m = vec![0u64; words_for(n)];
        let prog = program();
        let cp = CompiledProgram::new(program());
        let mut buf = vec![0u64; tile_words_for(n) * 4];
        let mut d = FoldDialect::new(&planes, &Foreign::NONE);
        let mut stack = Vec::with_capacity(8);

        // Equal work, checked: every arm answers the oracle at this size.
        let run_compiled = |buf: &mut [u64]| {
            let mut s = Scratch::over_for_program(buf, cp.program(), n).unwrap();
            sum_of(
                execute_compiled(
                    &cp.compiled(),
                    &planes,
                    &Foreign::NONE,
                    &mut s,
                    Out::None,
                    0..n,
                )
                .unwrap(),
            )
        };
        let run_into = |buf: &mut [u64]| {
            let mut s = Scratch::over_for_program(buf, &prog, n).unwrap();
            sum_of(execute_into(&prog, &planes, &Foreign::NONE, &mut s, Out::None).unwrap())
        };
        let run_build = |buf: &mut [u64]| {
            let p = program();
            let mut s = Scratch::over_for_program(buf, &p, n).unwrap();
            sum_of(execute_into(&p, &planes, &Foreign::NONE, &mut s, Out::None).unwrap())
        };
        assert_eq!(i128::from(scalar(&x, T)), want);
        assert_eq!(i128::from(nd_floor(&x, &mut m)), want);
        assert_eq!(i128::from(run_compiled(&mut buf)), want);
        assert_eq!(i128::from(run_into(&mut buf)), want);
        assert_eq!(i128::from(run_build(&mut buf)), want);
        assert_eq!(op.eval(&mut d, &mut stack).map(i128::from), Ok(want));

        let r = [
            measure(reps, iters, || {
                black_box(scalar(black_box(&x), black_box(T)));
            }),
            measure(reps, iters, || {
                black_box(nd_floor(black_box(&x), &mut m));
            }),
            measure(reps, iters, || {
                black_box(run_compiled(&mut buf));
            }),
            measure(reps, iters, || {
                black_box(run_into(&mut buf));
            }),
            measure(reps, iters, || {
                black_box(run_build(&mut buf));
            }),
            measure(reps, iters, || {
                black_box(op.eval(&mut d, &mut stack).unwrap());
            }),
        ];
        print!("| {n} |");
        for (t, _) in r {
            print!(" {t:.0} |");
        }
        let a: Vec<String> = r.iter().map(|(_, a)| format!("{a:.1}")).collect();
        println!(" {} |", a.join(" / "));
    }
    println!("\nns per call, median of MPF_REPS. Reported, not gated.");
}
