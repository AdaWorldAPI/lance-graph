//! The layer benchmark: every op through ndarray primitives, then the
//! lance-graph mask-risc program, then the R2IL polyfill, on the same
//! resident `&[i32]` lane. Reported, not gated.
//!
//! Each op is `<fold> { x[r] : x[r] > t }` for fold in count, exists, sum,
//! min, max. Layers:
//!
//! - `ndarray`: `gt_i32_to_mask` into a reused mask, then the matching
//!   reducer (`popcount_batch_u64`, `mask_any`, `masked_sum_i32`,
//!   `masked_min_i32`, `masked_max_i32`).
//! - `mask-risc`: one `Program` (`Pred::GtI32` + the matching terminal),
//!   recognised once, run with `execute_compiled` over reused scratch.
//! - `r2il`: the same question lowered onto a reused `FoldDialect` with R2IL
//!   calls (`LOAD`, `IntSLess`) and the `procs` fold (`count`, `exists`,
//!   `sum`, `min`, `max`).
//!
//! "before r2il" is the mask-risc column, "after" the r2il column; `Δ` is
//! each layer minus the one below. Every layer is checked against an `i128`
//! oracle at every size before timing. `scalar` is an equal-work row loop
//! for reference. `MPF_REPS` (default 31), `MPF_ITERS` (default 4000).

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_compiled, tile_words_for, Scratch};
use lance_graph_mask_risc::ir::{CompiledProgram, MaskOp, Operand, Pred, Program, Terminal};
use lance_graph_mask_risc::{words_for, Foreign, LaneRef, Out, Planes, Value};
use lance_graph_r2il::procs;
use lance_graph_r2il::{FoldDialect, FoldError, Val, INT_S_LESS, LOAD};

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

/// Median ns per call and heap allocations per call.
fn measure(reps: usize, iters: usize, mut f: impl FnMut() -> Option<i64>) -> (f64, f64) {
    let before = ALLOCS.load(Ordering::Relaxed);
    for _ in 0..100 {
        black_box(f());
    }
    let allocs = (ALLOCS.load(Ordering::Relaxed) - before) as f64 / 100.0;
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
    (v[v.len() / 2], allocs)
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Count,
    Exists,
    Sum,
    Min,
    Max,
}

const OPS: [Op; 5] = [Op::Count, Op::Exists, Op::Sum, Op::Min, Op::Max];

/// The answer every layer must give. `None` is an empty MIN/MAX.
fn oracle(x: &[i32], t: i32, op: Op) -> Option<i64> {
    let sel = x.iter().filter(|&&v| v > t).map(|&v| i128::from(v));
    let v = match op {
        Op::Count => Some(sel.count() as i128),
        Op::Exists => Some(i128::from(x.iter().any(|&v| v > t))),
        Op::Sum => Some(sel.sum()),
        Op::Min => sel.min(),
        Op::Max => sel.max(),
    };
    v.map(|v| i64::try_from(v).expect("fits"))
}

#[inline(never)]
fn scalar(x: &[i32], t: i32, op: Op) -> Option<i64> {
    let it = x.iter().copied();
    match op {
        Op::Count => Some(it.filter(|&v| v > t).count() as i64),
        Op::Exists => Some(i64::from(it.clone().any(|v| v > t))),
        Op::Sum => Some(it.map(|v| if v > t { i64::from(v) } else { 0 }).sum()),
        Op::Min => it.filter(|&v| v > t).min().map(i64::from),
        Op::Max => it.filter(|&v| v > t).max().map(i64::from),
    }
}

#[inline(never)]
fn nd(x: &[i32], t: i32, op: Op, m: &mut [u64]) -> Option<i64> {
    use ndarray::simd::{
        gt_i32_to_mask, mask_any, masked_max_i32, masked_min_i32, masked_sum_i32,
        popcount_batch_u64,
    };
    gt_i32_to_mask(x, t, m);
    match op {
        Op::Count => Some(popcount_batch_u64(m) as i64),
        Op::Exists => Some(i64::from(mask_any(m))),
        Op::Sum => Some(masked_sum_i32(x, m)),
        Op::Min => masked_min_i32(x, m).map(i64::from),
        Op::Max => masked_max_i32(x, m).map(i64::from),
    }
}

fn program(t: i32, op: Op) -> Program {
    let mask = Operand::Scratch(0);
    let terminal = match op {
        Op::Count => Terminal::Count { mask },
        Op::Exists => Terminal::Any { mask },
        Op::Sum => Terminal::MaskedSumI32 { mask, lane: 0 },
        Op::Min => Terminal::MaskedMinI32 { mask, lane: 0 },
        Op::Max => Terminal::MaskedMaxI32 { mask, lane: 0 },
    };
    Program::new(
        vec![MaskOp::Pred {
            pred: Pred::GtI32 { lane: 0, t },
            under: None,
            dst: 0,
        }],
        terminal,
    )
}

fn value(v: Value) -> Option<i64> {
    match v {
        Value::Count(n) => Some(n as i64),
        Value::Bool(b) => Some(i64::from(b)),
        Value::SumI64(s) => Some(s),
        Value::OptI32(o) => o.map(i64::from),
        other => panic!("scalar terminal, got {other:?}"),
    }
}

/// `x > t` as `t < x` (`IntSLess` is the only ordered compare), then the fold.
fn r2il(
    d: &mut FoldDialect<'_>,
    s: &mut Vec<Val>,
    t: i32,
    op: Op,
) -> Result<Option<i64>, FoldError> {
    s.clear();
    s.push(Val::Scalar(i64::from(t)));
    s.push(Val::Scalar(0));
    d.call(LOAD, [1, 0, 0], s)?;
    d.call(INT_S_LESS, [0; 3], s)?;
    match op {
        Op::Count => procs::count(d, s)?,
        Op::Exists => procs::exists(d, s)?,
        Op::Sum => procs::sum(d, s, 0)?,
        Op::Min => procs::min(d, s, 0)?,
        Op::Max => procs::max(d, s, 0)?,
    }
    let v = match s.pop() {
        Some(Val::Scalar(v)) => v,
        _ => return Err(FoldError::WrongOperandKind(INT_S_LESS)),
    };
    // `procs::min`/`max` push i64::MAX / i64::MIN for an empty mask.
    Ok(match (op, v) {
        (Op::Min, i64::MAX) | (Op::Max, i64::MIN) => None,
        _ => Some(v),
    })
}

fn main() {
    let reps = env("MPF_REPS", 31);
    let iters0 = env("MPF_ITERS", 4000);
    let t = 0;
    println!("| rows | op | scalar | ndarray | mask-risc (before r2il) | Δ mask-risc | r2il (after) | Δ r2il | allocs nd/mr/r2il |");
    println!("|---:|---|---:|---:|---:|---:|---:|---:|---|");
    for n in [64usize, 4_096, 65_536] {
        let iters = (iters0 * 64 / n).max(50);
        let x: Vec<i32> = (0..n).map(|i| ((i * 13) % 97) as i32 - 48).collect();
        let lanes = [LaneRef::I32(&x)];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let mut m = vec![0u64; words_for(n)];
        let mut buf = vec![0u64; tile_words_for(n) * 4];
        let mut d = FoldDialect::new(&planes, &Foreign::NONE);
        let mut s = Vec::with_capacity(8);
        for op in OPS {
            let want = oracle(&x, t, op);
            let cp = CompiledProgram::new(program(t, op));
            let mr = |buf: &mut [u64]| {
                let mut sc = Scratch::over_for_program(buf, cp.program(), n).unwrap();
                value(
                    execute_compiled(
                        &cp.compiled(),
                        &planes,
                        &Foreign::NONE,
                        &mut sc,
                        Out::None,
                        0..n,
                    )
                    .unwrap(),
                )
            };
            assert_eq!(scalar(&x, t, op), want, "scalar {op:?} n={n}");
            assert_eq!(nd(&x, t, op, &mut m), want, "ndarray {op:?} n={n}");
            assert_eq!(mr(&mut buf), want, "mask-risc {op:?} n={n}");
            assert_eq!(r2il(&mut d, &mut s, t, op), Ok(want), "r2il {op:?} n={n}");

            let (sc, _) = measure(reps, iters, || scalar(black_box(&x), black_box(t), op));
            let (a, aa) = measure(reps, iters, || nd(black_box(&x), black_box(t), op, &mut m));
            let (b, ba) = measure(reps, iters, || mr(&mut buf));
            let (c, ca) = measure(reps, iters, || {
                r2il(&mut d, &mut s, black_box(t), op).unwrap()
            });
            println!(
                "| {n} | {op:?} | {sc:.0} | {a:.0} | {b:.0} | {:+.0} | {c:.0} | {:+.0} | {aa:.0}/{ba:.0}/{ca:.0} |",
                b - a,
                c - b
            );
        }
    }
    println!("\nns per call, median of MPF_REPS; t = 0 selects about half the rows. Reported, not gated.");
}
