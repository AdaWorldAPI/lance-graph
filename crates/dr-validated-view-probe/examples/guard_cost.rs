//! D-NFR-6 cost split. Reported, not gated.
//!
//! Arms, on the same table, the same five operations and the same kernels:
//!
//! - **A direct**: the `ndarray::simd` call.
//! - **B prebound**: the `fn` pointer resolved once in `KERNELS`, no guard.
//! - **C bind per op**: `bind_dr64` (the whole existing validator), then the kernel.
//! - **C' guard per op**: what the #1752 ClassId route runs per call:
//!   `FrameStore::check` plus focus and scope equality, then the kernel.
//! - **D view**: one `validate`, then the view's operations (no check).
//! - **D' shared view**: as D, plus one grant-epoch load per operation.
//! - **E**: re-validation after a change; it must refuse.
//!
//! Then the guard itself, taken apart. `cold` costs (the first `validate`,
//! the `LazyLock` initialisation) are reported separately and not mixed into
//! warm per-operation figures. Allocations per call are counted by a global
//! allocator wrapper. The `probe_*` functions are out of line for reading the
//! assembly: `cargo rustc --release --example guard_cost -- --emit=asm`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::Instant;

use dr_math_binding_probe::{
    FrameStore, Guards, MethodSignature, Operand, Refusal, Scope, Truth, bind_dr64,
};
use dr_validated_view_probe::fixture::{data, revisions, setup};
use dr_validated_view_probe::{
    Focus, KERNELS, SharedGrants, View, ViewRefusal, validate, validate_shared,
};

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

/// Median ns per call, and allocations per call.
fn measure(reps: usize, iters: usize, mut f: impl FnMut()) -> (f64, f64) {
    for _ in 0..200 {
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

// ── out-of-line shapes for the assembly reading ────────────────────────────

#[inline(never)]
pub fn probe_direct_sum(values: &[i32], words: &[u64]) -> i64 {
    ndarray::simd::masked_sum_i32(values, words)
}

#[inline(never)]
pub fn probe_view_sum(view: &View<'_>) -> i64 {
    view.sum()
}

#[inline(never)]
pub fn probe_shared_sum(
    view: &dr_validated_view_probe::SharedView<'_>,
) -> Result<i64, ViewRefusal> {
    view.sum()
}

#[inline(never)]
pub fn probe_bind_sum(
    store: &FrameStore,
    m: MethodSignature,
    guards: &Guards,
    now: (Focus, Scope),
    rows: &[u64; 64],
    values: &[i32],
) -> Result<i64, Refusal> {
    let b = bind_dr64(
        store,
        m,
        m,
        guards,
        now,
        Operand {
            rows,
            validity: None,
        },
    )?;
    Ok(ndarray::simd::masked_sum_i32(
        values,
        &b.rows[..values.len().div_ceil(64)],
    ))
}

#[inline(never)]
pub fn probe_guard_sum(
    store: &FrameStore,
    guards: &Guards,
    now: (Focus, Scope),
    rows: &[u64; 64],
    values: &[i32],
) -> Result<i64, Refusal> {
    store.check(&guards.frame)?;
    if now.0 != guards.focus {
        return Err(Refusal::FocusChanged);
    }
    if now.1 != guards.scope {
        return Err(Refusal::ScopeChanged);
    }
    Ok(ndarray::simd::masked_sum_i32(
        values,
        &rows[..values.len().div_ceil(64)],
    ))
}

#[inline(never)]
pub fn probe_check(store: &FrameStore, guards: &Guards) -> Result<(), Refusal> {
    store.check(&guards.frame)
}

const OPS: [&str; 5] = ["count", "sum", "min", "max", "count_gt"];

fn main() {
    let reps = env("NFR6_REPS", 21);
    let iters = env("NFR6_ITERS", 20_000);
    println!(
        "avx512f compile-time: {}\n",
        cfg!(target_feature = "avx512f")
    );

    // Cold costs, once each, before anything warms them.
    let t0 = Instant::now();
    let k = *KERNELS;
    let lazy_cold = t0.elapsed().as_nanos();
    let (t, g, m, guards) = setup(64);
    let t0 = Instant::now();
    let v = validate(&t, &g, m, m, &guards).unwrap();
    let validate_cold = t0.elapsed().as_nanos();
    black_box(v.count());
    println!(
        "cold: LazyLock init {lazy_cold} ns, first validate {validate_cold} ns (single samples)\n"
    );

    for n in [64usize, 4096] {
        let (t, g, m, guards) = setup(n);
        let sg = SharedGrants::new(Scope { id: 3 });
        let w = n.div_ceil(64);
        let now = (t.focus(), guards.scope);
        let d = data(n, 0);
        let (rows, values) = (&d.rows, &d.values[..]);
        let words = &rows[..w];
        let mut scratch = vec![0u64; 2 * w];
        let mut scratch_c = vec![0u64; 2 * w];
        let mut scratch_c2 = vec![0u64; 2 * w];
        let mut cmp = vec![0u64; w];
        let mut both = vec![0u64; w];
        let view = validate(&t, &g, m, m, &guards).unwrap();
        let shared = validate_shared(&t, &sg, m, m, &guards).unwrap();

        // Identical answers before timing.
        assert_eq!(view.sum(), ndarray::simd::masked_sum_i32(values, words));
        assert_eq!(
            probe_bind_sum(t.store(), m, &guards, now, rows, values),
            Ok(view.sum())
        );
        assert_eq!(
            probe_guard_sum(t.store(), &guards, now, rows, values),
            Ok(view.sum())
        );
        assert_eq!(shared.sum(), Ok(view.sum()));
        // The out-of-line probes, so they are emitted for the assembly reading.
        assert_eq!(probe_direct_sum(values, words), view.sum());
        assert_eq!(probe_view_sum(&view), view.sum());
        assert_eq!(probe_shared_sum(&shared), Ok(view.sum()));

        println!("rows = {n}: ns per operation (allocations per call in brackets)\n");
        println!(
            "| op | A direct | B prebound | C bind_dr64 per op | C' guard per op | D view | D' shared view |"
        );
        println!("|---|---:|---:|---:|---:|---:|---:|");
        for op in OPS {
            let kernel_direct =
                |s: &mut [u64]| -> i64 {
                    match op {
                        "count" => ndarray::simd::popcount_batch_u64(black_box(words)) as i64,
                        "sum" => ndarray::simd::masked_sum_i32(black_box(values), words),
                        "min" => ndarray::simd::masked_min_i32(black_box(values), words)
                            .unwrap_or(0) as i64,
                        "max" => ndarray::simd::masked_max_i32(black_box(values), words)
                            .unwrap_or(0) as i64,
                        _ => {
                            let (c, rest) = s.split_at_mut(w);
                            ndarray::simd::gt_i32_to_mask(black_box(values), 0, c);
                            ndarray::simd::mask_and(c, words, &mut rest[..w]);
                            ndarray::simd::popcount_batch_u64(&rest[..w]) as i64
                        }
                    }
                };
            let kernel_pre = |s: &mut [u64]| -> i64 {
                match op {
                    "count" => (k.count)(black_box(words)) as i64,
                    "sum" => (k.sum)(black_box(values), words),
                    "min" => (k.min)(black_box(values), words).unwrap_or(0) as i64,
                    "max" => (k.max)(black_box(values), words).unwrap_or(0) as i64,
                    _ => {
                        let (c, rest) = s.split_at_mut(w);
                        (k.gt)(black_box(values), 0, c);
                        (k.and)(c, words, &mut rest[..w]);
                        (k.count)(&rest[..w]) as i64
                    }
                }
            };
            let view_op = |v: &View<'_>, s: &mut [u64]| -> i64 {
                match op {
                    "count" => v.count() as i64,
                    "sum" => v.sum(),
                    "min" => v.min().unwrap_or(0) as i64,
                    "max" => v.max().unwrap_or(0) as i64,
                    _ => v.count_gt(0, s) as i64,
                }
            };
            let a = measure(reps, iters, || {
                black_box(kernel_direct(&mut scratch));
            });
            let b = measure(reps, iters, || {
                black_box(kernel_pre(&mut scratch));
            });
            let c = measure(reps, iters, || {
                let bound = bind_dr64(
                    t.store(),
                    m,
                    m,
                    black_box(&guards),
                    now,
                    Operand {
                        rows,
                        validity: None,
                    },
                )
                .unwrap();
                black_box(&bound);
                black_box(kernel_pre(&mut scratch_c));
            });
            let c2 = measure(reps, iters, || {
                t.store().check(black_box(&guards.frame)).unwrap();
                assert!(black_box(now.0) == guards.focus && black_box(now.1) == guards.scope);
                black_box(kernel_pre(&mut scratch_c2));
            });
            let dd = measure(reps, iters, || {
                black_box(view_op(black_box(&view), &mut scratch));
            });
            let ds = measure(reps, iters, || {
                let r = match op {
                    "count" => shared.count().map(|x| x as i64),
                    "sum" => shared.sum(),
                    "min" => shared.min().map(|x| x.unwrap_or(0) as i64),
                    "max" => shared.max().map(|x| x.unwrap_or(0) as i64),
                    _ => shared.count_gt(0, &mut scratch).map(|x| x as i64),
                };
                black_box(r.unwrap());
            });
            let f = |x: (f64, f64)| format!("{:.1} [{:.0}]", x.0, x.1);
            println!(
                "| {op} | {} | {} | {} | {} | {} | {} |",
                f(a),
                f(b),
                f(c),
                f(c2),
                f(dd),
                f(ds)
            );
        }

        // Five operations under one validation vs five validations.
        let five_view = measure(reps, iters / 5, || {
            let v = validate(&t, &g, m, m, black_box(&guards)).unwrap();
            black_box((
                v.count(),
                v.sum(),
                v.min(),
                v.max(),
                v.count_gt(0, &mut scratch),
            ));
        });
        let five_bind = measure(reps, iters / 5, || {
            let mut out = [0i64; 5];
            for (i, slot) in out.iter_mut().enumerate() {
                let bound = bind_dr64(
                    t.store(),
                    m,
                    m,
                    black_box(&guards),
                    now,
                    Operand {
                        rows,
                        validity: None,
                    },
                )
                .unwrap();
                black_box(&bound);
                *slot = match i {
                    0 => (k.count)(words) as i64,
                    1 => (k.sum)(values, words),
                    2 => (k.min)(values, words).unwrap_or(0) as i64,
                    3 => (k.max)(values, words).unwrap_or(0) as i64,
                    _ => {
                        (k.gt)(values, 0, &mut cmp);
                        (k.and)(&cmp, words, &mut both);
                        (k.count)(&both) as i64
                    }
                };
            }
            black_box(out);
        });
        let five_direct = measure(reps, iters / 5, || {
            (k.gt)(values, 0, &mut cmp);
            (k.and)(&cmp, words, &mut both);
            black_box((
                ndarray::simd::popcount_batch_u64(black_box(words)),
                ndarray::simd::masked_sum_i32(values, words),
                ndarray::simd::masked_min_i32(values, words),
                ndarray::simd::masked_max_i32(values, words),
                ndarray::simd::popcount_batch_u64(&both),
            ));
        });
        println!(
            "\nfive ops: direct {:.1} [{:.0}] · one validate + five view ops {:.1} [{:.0}] · bind_dr64 before each {:.1} [{:.0}]\n",
            five_direct.0, five_direct.1, five_view.0, five_view.1, five_bind.0, five_bind.1
        );
    }

    // ── the guard, taken apart (rows = 64; independent of data size) ───────
    let (t, g, m, guards) = setup(64);
    let now = (t.focus(), guards.scope);
    let rows = data(64, 0).rows;
    let reps_b = reps;
    let it = iters * 5;
    let pieces: Vec<(&str, (f64, f64))> = vec![
        (
            "method + axis + semantics compares (3 tuple compares)",
            measure(reps_b, it, || {
                let (a, b) = (black_box(m), black_box(m));
                black_box(
                    (a.owner_class, a.method_ordinal) == (b.owner_class, b.method_ordinal)
                        && (a.row, a.column) == (b.row, b.column)
                        && (a.truth, a.population) == (b.truth, b.population),
                );
            }),
        ),
        (
            "validity / truth match",
            measure(reps_b, it, || {
                let v: Option<&[u64; 64]> = black_box(None);
                black_box(matches!((black_box(m).truth, v), (Truth::TwoValued, None)));
            }),
        ),
        (
            "focus + scope equality",
            measure(reps_b, it, || {
                black_box(black_box(now.0) == guards.focus && black_box(now.1) == guards.scope);
            }),
        ),
        (
            "FrameStore::check (real)",
            measure(reps_b, it, || {
                black_box(probe_check(t.store(), black_box(&guards))).unwrap();
            }),
        ),
        (
            "Frame::clone + drop (real; bind_dr64 clones the frame)",
            measure(reps_b, it, || {
                black_box(black_box(&guards.frame).clone());
            }),
        ),
        ("Arc::downgrade + drop of the Weak (proxy, x1)", {
            let arc = Arc::new(0u64);
            measure(reps_b, it, || {
                black_box(Arc::downgrade(black_box(&arc)));
            })
        }),
        ("Weak::ptr_eq (proxy, x1)", {
            let arc = Arc::new(0u64);
            let (w1, w2) = (Arc::downgrade(&arc), Arc::downgrade(&arc));
            measure(reps_b, it, || {
                black_box(Weak::ptr_eq(black_box(&w1), black_box(&w2)));
            })
        }),
        ("Weak::as_ptr == Arc::as_ptr (proxy, x1; no refcount)", {
            let arc = Arc::new(0u64);
            let w1 = Arc::downgrade(&arc);
            measure(reps_b, it, || {
                black_box(std::ptr::eq(
                    Weak::as_ptr(black_box(&w1)),
                    Arc::as_ptr(black_box(&arc)),
                ));
            })
        }),
        (
            "bind_dr64 total (real)",
            measure(reps_b, it, || {
                black_box(
                    bind_dr64(
                        t.store(),
                        m,
                        m,
                        black_box(&guards),
                        now,
                        Operand {
                            rows: &rows,
                            validity: None,
                        },
                    )
                    .unwrap(),
                );
            }),
        ),
        (
            "validate total (real; bind_dr64 + grant read)",
            measure(reps_b, it, || {
                black_box(validate(&t, &g, m, m, black_box(&guards)).unwrap());
            }),
        ),
        ("grant epoch load (D' per op)", {
            let sg = SharedGrants::new(Scope { id: 3 });
            let sv = validate_shared(&t, &sg, m, m, &guards).unwrap();
            measure(reps_b, it, || {
                black_box(black_box(&sv).count().unwrap());
            })
        }),
        ("count via fn pointer, rows = 64 (B)", {
            let w = &rows[..1];
            measure(reps_b, it, || {
                black_box((KERNELS.count)(black_box(w)));
            })
        }),
        ("count direct, rows = 64 (A)", {
            let w = &rows[..1];
            measure(reps_b, it, || {
                black_box(ndarray::simd::popcount_batch_u64(black_box(w)));
            })
        }),
    ];
    println!("guard pieces, ns per call [allocations]:\n");
    println!("| piece | ns | allocs |");
    println!("|---|---:|---:|");
    for (name, (ns, al)) in &pieces {
        println!("| {name} | {ns:.2} | {al:.0} |");
    }

    // ── E: a change after the borrow is refused at acquisition ─────────────
    let (mut t, g, m, guards) = setup(64);
    t.advance(revisions(2), data(64, 1));
    let e = measure(reps, iters, || {
        black_box(validate(&t, &g, m, m, black_box(&guards)).err());
    });
    assert_eq!(
        validate(&t, &g, m, m, &guards).err(),
        Some(ViewRefusal::Bind(Refusal::SourceChanged))
    );
    println!(
        "\nE: re-validation after advance is refused (SourceChanged), {:.1} ns [{:.0}]",
        e.0, e.1
    );
    println!("\nns per call, median of NFR6_REPS. Reported, not gated.");
}
