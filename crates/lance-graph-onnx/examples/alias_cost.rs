//! D-ONNX-NATIVE-0: what the facade costs over the native call. Reported,
//! not gated.
//!
//! Arms, same inputs, outputs and CPU configuration:
//! - `direct`: the native function, called by its ndarray name.
//! - `onnx`: the facade under its ONNX name (the same symbol for an ALIAS).
//! - `wrapper (#1752)`: diagnostic baseline only. It reproduces the #1752
//!   `onnx_compat::reduce_sum_where_greater` flow: attributes checked and a
//!   `FoldDialect` built on every call.
//!
//! Every arm is checked against the oracle before timing. Allocations per call
//! are counted by a global allocator wrapper. `ONNX_REPS` (default 21) and
//! `ONNX_ITERS` (default 4000) set the sampling. The `probe_*` functions are
//! out of line so their assembly can be read on its own:
//! `cargo rustc --release -p lance-graph-onnx --example alias_cost -- --emit=asm`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use lance_graph_mask_risc::{Foreign, LaneRef, Planes};
use lance_graph_onnx as onnx;
use lance_graph_r2il::{FoldDialect, Val, INT_S_LESS, LOAD};
use ndarray::Array1;

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
    for _ in 0..100 {
        f();
    }
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

// ── out-of-line call shapes for the assembly reading ──────────────────────

#[inline(never)]
pub fn probe_direct_add(a: &Array1<f32>, b: &Array1<f32>) -> Array1<f32> {
    a + b
}

#[inline(never)]
pub fn probe_onnx_add(a: &Array1<f32>, b: &Array1<f32>) -> Result<Array1<f32>, onnx::Refusal> {
    onnx::add(a, b)
}

#[inline(never)]
pub fn probe_direct_sum_where(x: &[i32], t: i32, m: &mut [u64]) -> i64 {
    ndarray::simd::gt_i32_to_mask(x, t, m);
    ndarray::simd::masked_sum_i32(x, m)
}

#[inline(never)]
pub fn probe_onnx_sum_where(
    rs: onnx::ReduceSum,
    x: &[i32],
    t: i32,
    m: &mut [u64],
) -> Result<onnx::Reduced, onnx::Refusal> {
    rs.sum_where_greater_i32(x, t, m)
}

/// The #1752 wrapper's work per call: attribute checks, a dialect, the
/// procedure, the range check. Diagnostic only.
fn wrapper_1752(x: &[i32], t: i32, keepdims: i64) -> Option<i32> {
    if keepdims != 0 && keepdims != 1 {
        return None;
    }
    let lanes = [LaneRef::I32(x)];
    let planes = Planes {
        n_rows: x.len(),
        masks: &[],
        lanes: &lanes,
    };
    let mut d = FoldDialect::new(&planes, &Foreign::NONE);
    let mut stack = vec![Val::Scalar(i64::from(t)), Val::Scalar(0)];
    d.call(LOAD, [1, 0, 0], &mut stack).ok()?;
    d.call(INT_S_LESS, [0; 3], &mut stack).ok()?;
    lance_graph_r2il::procs::sum(&mut d, &mut stack, 0).ok()?;
    match stack.pop() {
        Some(Val::Scalar(s)) => i32::try_from(s).ok(),
        _ => None,
    }
}

fn main() {
    let reps = env("ONNX_REPS", 21);
    let iters0 = env("ONNX_ITERS", 4000);
    println!(
        "alias identity: greater_i32 == gt_i32_to_mask: {}",
        std::ptr::fn_addr_eq(
            onnx::greater_i32 as fn(&[i32], i32, &mut [u64]),
            ndarray::simd::gt_i32_to_mask as fn(&[i32], i32, &mut [u64])
        )
    );
    println!(
        "avx512f compile-time: {}\n",
        cfg!(target_feature = "avx512f")
    );
    println!("| rows | op | direct | onnx | wrapper (#1752) | allocs direct/onnx/wrapper |");
    println!("|---:|---|---:|---:|---:|---|");
    for n in [64usize, 4_096, 65_536] {
        let iters = (iters0 * 64 / n).max(50);
        let x: Vec<i32> = (0..n).map(|i| ((i * 13) % 97) as i32 - 48).collect();
        let t = 0;
        let want: i64 = x.iter().filter(|&&v| v > t).map(|&v| i64::from(v)).sum();
        let mut m = vec![0u64; onnx::mask_words(n)];
        let rs = onnx::ReduceSum::new(None, 0, 0).unwrap();
        assert_eq!(probe_direct_sum_where(&x, t, &mut m), want);
        assert_eq!(
            probe_onnx_sum_where(rs, &x, t, &mut m).unwrap().value as i64,
            want
        );
        assert_eq!(wrapper_1752(&x, t, 0).map(i64::from), Some(want));

        // Greater: an ALIAS, so both arms call one symbol.
        let g = [
            measure(reps, iters, || {
                ndarray::simd::gt_i32_to_mask(black_box(&x), black_box(t), &mut m)
            }),
            measure(reps, iters, || {
                onnx::greater_i32(black_box(&x), black_box(t), &mut m)
            }),
        ];
        println!(
            "| {n} | Greater (ALIAS) | {:.1} | {:.1} | — | {:.0}/{:.0}/— |",
            g[0].0, g[1].0, g[0].1, g[1].1
        );

        let s = [
            measure(reps, iters, || {
                black_box(probe_direct_sum_where(black_box(&x), black_box(t), &mut m));
            }),
            measure(reps, iters, || {
                let _ = black_box(probe_onnx_sum_where(
                    rs,
                    black_box(&x),
                    black_box(t),
                    &mut m,
                ));
            }),
            measure(reps, iters, || {
                black_box(wrapper_1752(black_box(&x), black_box(t), 0));
            }),
        ];
        println!(
            "| {n} | ReduceSum(Where(Greater)) (INLINE) | {:.1} | {:.1} | {:.1} | {:.0}/{:.0}/{:.0} |",
            s[0].0, s[1].0, s[2].0, s[0].1, s[1].1, s[2].1
        );

        let a = Array1::from_iter((0..n).map(|i| i as f32 * 0.5));
        let b = Array1::from_iter((0..n).map(|i| 1.0 - i as f32));
        assert_eq!(probe_onnx_add(&a, &b).unwrap(), probe_direct_add(&a, &b));
        let ad = [
            measure(reps, iters, || {
                black_box(probe_direct_add(black_box(&a), black_box(&b)));
            }),
            measure(reps, iters, || {
                let _ = black_box(probe_onnx_add(black_box(&a), black_box(&b)));
            }),
        ];
        println!(
            "| {n} | Add f32 (INLINE) | {:.1} | {:.1} | — | {:.0}/{:.0}/— |",
            ad[0].0, ad[1].0, ad[0].1, ad[1].1
        );
    }
    println!("\nns per call, median of ONNX_REPS. Reported, not gated.");
}
