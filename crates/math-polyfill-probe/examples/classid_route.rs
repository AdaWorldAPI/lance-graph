//! D-MPF-4: ClassId → LazyLock → prebound native callable → direct kernel.
//! Lab probe, reported not gated. Adds no ABI, no interpreter, no dispatcher
//! to any production crate: everything below lives in this example.
//!
//! The question: what does R2IL routing cost when the route is resolved ONCE
//! per ClassId and the resolved callable is stored in the binding, against
//! calling the same `ndarray::simd` kernels directly?
//!
//! - One `LazyLock` per ClassId resolves the route (DIRECT / COMPOSE /
//!   LOWER / REFUSE) and its contract. `INITS[c]` counts resolutions;
//!   `ROUTE_EVALS` counts every route evaluation.
//! - `bind` reads the route once and copies the callable into [`Bound`].
//!   The hot path never touches the `LazyLock`, a registry, a `match` on
//!   the backend, or a capability probe.
//! - Frame, focus and scope stay checked on every run, by the DR lab's own
//!   `FrameStore::check` plus focus/scope equality. That check is
//!   validation of a versioned identity, not route selection, and it is
//!   timed as its own arm.
//!
//! Arms, all on the same `&[i32]`, `SUM { x : x > t }`:
//!   direct nd     `gt_i32_to_mask` + `masked_sum_i32`
//!   route         prebound callable + frame/focus/scope check
//!   route nochk   prebound callable alone (splits the check's cost)
//!   lazy/call     `LazyLock` deref + `match` per call (simd_runtime shape)
//!   mr prepared   mask-risc `execute_compiled` on a hand-built program
//!   r2il prepared FoldDialect run ONCE, its `last_program` kept as a
//!                 `CompiledProgram`, then `execute_compiled`
//!   dialect       `SumWhere::eval` per call (the existing fold reading)
//!   onnx wrapper  diagnostic only; rebinds every call; returns i32
//! Every arm is oracle-checked before timing. `MPF_REPS`, `MPF_ITERS`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;
use std::time::Instant;

use dr_math_binding_probe::{Focus, FrameRevisions, FrameStore, Guards, Refusal, Scope};
use lance_graph_mask_risc::exec::{execute_compiled, tile_words_for, Scratch};
use lance_graph_mask_risc::ir::{
    CompiledProgram, MaskOp, Operand, Pred as MPred, Program, Terminal,
};
use lance_graph_mask_risc::{words_for, Foreign, LaneRef, Out, Planes, Value};
use lance_graph_r2il::FoldDialect;
use math_polyfill_probe::native::{sum_where_oracle, Pred, SumWhere};
use math_polyfill_probe::onnx_compat::reduce_sum_where_greater;
use ndarray::simd as nd;

// ── counting allocator ─────────────────────────────────────────────────────

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

// ── the route table ────────────────────────────────────────────────────────

/// A class of mathematical operation. The probe's ids are local; a real
/// router would key on the ClassView classid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ClassId(usize);

const COUNT_GT: ClassId = ClassId(0);
const EXISTS_GT: ClassId = ClassId(1);
const SUM_GT: ClassId = ClassId(2);
const MIN_GT: ClassId = ClassId(3);
const SUM_OVER_MASK: ClassId = ClassId(4);
const INT_CARRY: ClassId = ClassId(5);
const F32_SUM_ON_I32: ClassId = ClassId(6);
const N_CLASSES: usize = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Answer {
    Count(u64),
    Bool(bool),
    Sum(i64),
    Opt(Option<i32>),
}

/// One native callable. `m` is caller-owned scratch, or for
/// [`SUM_OVER_MASK`] the resident mask itself.
type Kernel = fn(&[i32], i32, &mut [u64]) -> Answer;

/// What a ClassId keeps through routing: op identity, dtype, shape,
/// exactness and the capability the route was resolved under.
#[derive(Clone, Copy, Debug)]
struct Contract {
    op: &'static str,
    dtype: &'static str,
    shape: &'static str,
    exact: bool,
    caps: &'static str,
}

#[derive(Clone, Copy, Debug)]
enum Route {
    /// One existing native entrypoint.
    Direct(Kernel),
    /// Several existing native entrypoints, composed in Rust.
    Compose(Kernel),
    /// Needs R2Sleigh's machine reading; names its entrypoint.
    Lower(&'static str),
    /// No admitted native method.
    Refuse(&'static str),
}

#[derive(Debug)]
struct Resolved {
    route: Route,
    contract: Contract,
}

static INITS: [AtomicUsize; N_CLASSES] = [const { AtomicUsize::new(0) }; N_CLASSES];
static ROUTE_EVALS: AtomicUsize = AtomicUsize::new(0);

fn caps() -> &'static str {
    if cfg!(target_feature = "avx512f") {
        "avx512f (compile-time)"
    } else if cfg!(target_feature = "avx2") {
        "avx2 (compile-time)"
    } else {
        "scalar or other (compile-time)"
    }
}

fn k_count(x: &[i32], t: i32, m: &mut [u64]) -> Answer {
    nd::gt_i32_to_mask(x, t, m);
    Answer::Count(nd::popcount_batch_u64(m))
}
fn k_exists(x: &[i32], t: i32, m: &mut [u64]) -> Answer {
    nd::gt_i32_to_mask(x, t, m);
    Answer::Bool(nd::mask_any(m))
}
fn k_sum(x: &[i32], t: i32, m: &mut [u64]) -> Answer {
    nd::gt_i32_to_mask(x, t, m);
    Answer::Sum(nd::masked_sum_i32(x, m))
}
fn k_min(x: &[i32], t: i32, m: &mut [u64]) -> Answer {
    nd::gt_i32_to_mask(x, t, m);
    Answer::Opt(nd::masked_min_i32(x, m))
}
fn k_sum_over_mask(x: &[i32], _t: i32, m: &mut [u64]) -> Answer {
    Answer::Sum(nd::masked_sum_i32(x, m))
}

/// The one place a route is chosen. Runs once per ClassId, from its
/// `LazyLock`.
fn resolve(c: ClassId) -> Resolved {
    INITS[c.0].fetch_add(1, Ordering::Relaxed);
    ROUTE_EVALS.fetch_add(1, Ordering::Relaxed);
    let exact = |op| Contract {
        op,
        dtype: "i32",
        shape: "rank-1 lane",
        exact: true,
        caps: caps(),
    };
    let (route, contract) = match c {
        COUNT_GT => (Route::Compose(k_count), exact("count(x > t)")),
        EXISTS_GT => (Route::Compose(k_exists), exact("exists(x > t)")),
        SUM_GT => (Route::Compose(k_sum), exact("sum(x | x > t)")),
        MIN_GT => (Route::Compose(k_min), exact("min(x | x > t)")),
        SUM_OVER_MASK => (
            Route::Direct(k_sum_over_mask),
            exact("sum(x | resident mask)"),
        ),
        INT_CARRY => (
            Route::Lower("r2conc::SlabState::step(IntCarry)"),
            exact("carry-out of a machine-width add"),
        ),
        F32_SUM_ON_I32 => (
            Route::Refuse("f32 reduction requested over an i32 lane"),
            Contract {
                op: "sum_f32",
                dtype: "f32 requested, i32 bound",
                shape: "rank-1 lane",
                exact: false,
                caps: caps(),
            },
        ),
        _ => (Route::Refuse("unknown class"), exact("unknown")),
    };
    Resolved { route, contract }
}

static ROUTES: [LazyLock<Resolved>; N_CLASSES] = [
    LazyLock::new(|| resolve(ClassId(0))),
    LazyLock::new(|| resolve(ClassId(1))),
    LazyLock::new(|| resolve(ClassId(2))),
    LazyLock::new(|| resolve(ClassId(3))),
    LazyLock::new(|| resolve(ClassId(4))),
    LazyLock::new(|| resolve(ClassId(5))),
    LazyLock::new(|| resolve(ClassId(6))),
];

// ── binding ────────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq)]
enum BindRefusal {
    Lower(&'static str),
    Refused(&'static str),
    Guard(Refusal),
}

/// A route bound to one lane under one frame, focus and scope. Holds the
/// callable itself, so running it never reads the route table.
struct Bound<'a> {
    class: ClassId,
    kernel: Kernel,
    contract: Contract,
    guards: Guards,
    x: &'a [i32],
    t: i32,
}

fn bind<'a>(
    class: ClassId,
    store: &FrameStore,
    guards: Guards,
    x: &'a [i32],
    t: i32,
) -> Result<Bound<'a>, BindRefusal> {
    store.check(&guards.frame).map_err(BindRefusal::Guard)?;
    let r = &*ROUTES[class.0];
    let kernel = match r.route {
        Route::Direct(k) | Route::Compose(k) => k,
        Route::Lower(e) => return Err(BindRefusal::Lower(e)),
        Route::Refuse(why) => return Err(BindRefusal::Refused(why)),
    };
    Ok(Bound {
        class,
        kernel,
        contract: r.contract,
        guards,
        x,
        t,
    })
}

impl Bound<'_> {
    /// Validate the versioned identity, then call the callable.
    #[inline]
    fn run(
        &self,
        store: &FrameStore,
        focus: Focus,
        scope: Scope,
        m: &mut [u64],
    ) -> Result<Answer, Refusal> {
        store.check(&self.guards.frame)?;
        if focus != self.guards.focus {
            return Err(Refusal::FocusChanged);
        }
        if scope != self.guards.scope {
            return Err(Refusal::ScopeChanged);
        }
        Ok((self.kernel)(self.x, self.t, m))
    }
}

// The three steady-state call shapes, kept out of line so their assembly
// can be read on its own (`--emit=asm`, symbols `probe_*`).

#[inline(never)]
pub fn probe_direct_sum(x: &[i32], t: i32, m: &mut [u64]) -> i64 {
    nd::gt_i32_to_mask(x, t, m);
    nd::masked_sum_i32(x, m)
}

#[inline(never)]
fn probe_route_run(
    b: &Bound<'_>,
    store: &FrameStore,
    focus: Focus,
    scope: Scope,
    m: &mut [u64],
) -> Result<Answer, Refusal> {
    b.run(store, focus, scope, m)
}

#[inline(never)]
fn probe_lazy_per_call(x: &[i32], t: i32, m: &mut [u64]) -> Answer {
    match ROUTES[SUM_GT.0].route {
        Route::Direct(k) | Route::Compose(k) => k(x, t, m),
        _ => Answer::Bool(false),
    }
}

// ── measurement ────────────────────────────────────────────────────────────

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

fn revs(source: u64) -> FrameRevisions {
    FrameRevisions {
        source,
        relation_map: 1,
        coordinates: 1,
    }
}

fn sum_of(v: Value) -> i64 {
    match v {
        Value::SumI64(s) => s,
        other => panic!("MaskedSumI32 yields SumI64, got {other:?}"),
    }
}

fn main() {
    let reps = env("MPF_REPS", 21);
    let iters0 = env("MPF_ITERS", 4000);
    let t: i32 = 0;
    let focus = Focus { revision: 7 };
    let scope = Scope { id: 3 };
    let mut store = FrameStore::new(revs(1));
    let g = |s: &FrameStore| Guards {
        frame: s.seal(),
        focus,
        scope,
    };

    // ── correctness across classes, before any timing ──────────────────────
    let x: Vec<i32> = (0..1_000i32).map(|i| (i * 13) % 97 - 48).collect();
    let mut m = vec![0u64; words_for(x.len())];
    let sel: Vec<i32> = x.iter().copied().filter(|&v| v > t).collect();
    let want = [
        (COUNT_GT, Answer::Count(sel.len() as u64)),
        (EXISTS_GT, Answer::Bool(!sel.is_empty())),
        (SUM_GT, Answer::Sum(sel.iter().map(|&v| i64::from(v)).sum())),
        (MIN_GT, Answer::Opt(sel.iter().copied().min())),
    ];
    for (c, w) in want {
        let b = bind(c, &store, g(&store), &x, t).expect("COMPOSE classes bind");
        assert_eq!(b.run(&store, focus, scope, &mut m), Ok(w), "{c:?}");
        // An empty selection: the contract's empty answers, not a guess.
        let b = bind(c, &store, g(&store), &x, i32::MAX).unwrap();
        let empty = b.run(&store, focus, scope, &mut m).unwrap();
        assert!(matches!(
            empty,
            Answer::Count(0) | Answer::Bool(false) | Answer::Sum(0) | Answer::Opt(None)
        ));
    }
    // DIRECT over a resident mask: the mask is the binding's input.
    nd::gt_i32_to_mask(&x, t, &mut m);
    let b = bind(SUM_OVER_MASK, &store, g(&store), &x, 0).unwrap();
    assert_eq!(
        b.run(&store, focus, scope, &mut m),
        Ok(Answer::Sum(sel.iter().map(|&v| i64::from(v)).sum()))
    );
    // LOWER and REFUSE are refused at bind, never at run.
    assert_eq!(
        bind(INT_CARRY, &store, g(&store), &x, t).err(),
        Some(BindRefusal::Lower("r2conc::SlabState::step(IntCarry)"))
    );
    assert!(matches!(
        bind(F32_SUM_ON_I32, &store, g(&store), &x, t).err(),
        Some(BindRefusal::Refused(_))
    ));

    // ── refusal and rebinding on incompatible changes ──────────────────────
    let b = bind(SUM_GT, &store, g(&store), &x, t).unwrap();
    assert_eq!(
        b.run(&store, Focus { revision: 8 }, scope, &mut m),
        Err(Refusal::FocusChanged)
    );
    assert_eq!(
        b.run(&store, focus, Scope { id: 4 }, &mut m),
        Err(Refusal::ScopeChanged)
    );
    drop(b);
    let inits_before = INITS[SUM_GT.0].load(Ordering::Relaxed);
    let stale_guards = g(&store);
    store.advance(revs(2));
    assert_eq!(
        bind(SUM_GT, &store, stale_guards.clone(), &x, t).err(),
        Some(BindRefusal::Guard(Refusal::SourceChanged))
    );
    let b = bind(SUM_GT, &store, g(&store), &x, t).expect("rebind under the new frame");
    assert!(b.run(&store, focus, scope, &mut m).is_ok());
    assert_eq!(
        INITS[SUM_GT.0].load(Ordering::Relaxed),
        inits_before,
        "rebinding must not re-resolve the route"
    );
    let stale = Bound {
        guards: stale_guards,
        ..b
    };
    assert_eq!(
        stale.run(&store, focus, scope, &mut m),
        Err(Refusal::SourceChanged),
        "a binding sealed before the advance refuses at run"
    );
    let c = stale.contract;
    println!(
        "contract kept through routing: {:?} → op {}, dtype {}, shape {}, exact {}, caps {}",
        stale.class, c.op, c.dtype, c.shape, c.exact, c.caps
    );

    // ── steady state ───────────────────────────────────────────────────────
    let arms = [
        "direct nd",
        "route",
        "route nochk",
        "lazy/call",
        "mr prepared",
        "r2il prepared",
        "dialect",
        "onnx wrapper",
    ];
    print!("\n| rows |");
    for a in arms {
        print!(" {a} |");
    }
    println!(" allocs/call |");
    println!("|---:|{}---|", "---:|".repeat(arms.len()));
    let evals_before = ROUTE_EVALS.load(Ordering::Relaxed);
    for n in [64usize, 512, 4_096, 65_536] {
        let iters = (iters0 * 64 / n).max(50);
        let x: Vec<i32> = (0..n).map(|i| ((i * 13) % 97) as i32 - 48).collect();
        let mut m = vec![0u64; words_for(n)];
        let lanes = [LaneRef::I32(&x)];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let want = sum_where_oracle(&x, None, Pred::Gt(t)) as i64;

        let b = bind(SUM_GT, &store, g(&store), &x, t).unwrap();
        let cp = CompiledProgram::new(Program::new(
            vec![MaskOp::Pred {
                pred: MPred::GtI32 { lane: 0, t },
                under: None,
                dst: 0,
            }],
            Terminal::MaskedSumI32 {
                mask: Operand::Scratch(0),
                lane: 0,
            },
        ));
        // The existing facility: run the fold reading once, keep its program.
        let op = SumWhere {
            lane: 0,
            present: None,
            pred: Pred::Gt(t),
        };
        let mut d = FoldDialect::new(&planes, &Foreign::NONE);
        let mut stack = Vec::with_capacity(8);
        assert_eq!(op.eval(&mut d, &mut stack), Ok(want));
        let r2 = CompiledProgram::new(d.last_program().expect("one fold ran").clone());
        let mut buf = vec![0u64; tile_words_for(n) * 4];
        let run_cp = |cp: &CompiledProgram, buf: &mut [u64]| {
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

        assert_eq!(probe_direct_sum(&x, t, &mut m), want);
        assert_eq!(
            probe_route_run(&b, &store, focus, scope, &mut m),
            Ok(Answer::Sum(want))
        );
        assert_eq!((b.kernel)(&x, t, &mut m), Answer::Sum(want));
        assert_eq!(probe_lazy_per_call(&x, t, &mut m), Answer::Sum(want));
        assert_eq!(run_cp(&cp, &mut buf), want);
        assert_eq!(run_cp(&r2, &mut buf), want);
        let wrapped =
            reduce_sum_where_greater(&x, &[n], t, &[], None, 0, 0).map(|o| i64::from(o.value));
        assert_eq!(wrapped, Ok(want));

        let r = [
            measure(reps, iters, || {
                black_box(probe_direct_sum(black_box(&x), black_box(t), &mut m));
            }),
            measure(reps, iters, || {
                let _ = black_box(probe_route_run(black_box(&b), &store, focus, scope, &mut m));
            }),
            measure(reps, iters, || {
                black_box((black_box(b.kernel))(black_box(&x), t, &mut m));
            }),
            measure(reps, iters, || {
                black_box(probe_lazy_per_call(black_box(&x), black_box(t), &mut m));
            }),
            measure(reps, iters, || {
                black_box(run_cp(&cp, &mut buf));
            }),
            measure(reps, iters, || {
                black_box(run_cp(&r2, &mut buf));
            }),
            measure(reps, iters, || {
                let _ = black_box(op.eval(&mut d, &mut stack));
            }),
            measure(reps, iters, || {
                let _ = black_box(reduce_sum_where_greater(
                    black_box(&x),
                    &[n],
                    black_box(t),
                    &[],
                    None,
                    0,
                    0,
                ));
            }),
        ];
        print!("| {n} |");
        for (ns, _) in r {
            print!(" {ns:.1} |");
        }
        let a: Vec<String> = r.iter().map(|(_, a)| format!("{a:.0}")).collect();
        println!(" {} |", a.join("/"));
    }
    let evals_after = ROUTE_EVALS.load(Ordering::Relaxed);

    // ── one binding, one frame, several operations ─────────────────────────
    let n = 4_096;
    let x: Vec<i32> = (0..n).map(|i| ((i * 13) % 97) as i32 - 48).collect();
    let mut m = vec![0u64; words_for(n)];
    let classes = [COUNT_GT, SUM_GT, MIN_GT, EXISTS_GT];
    let bound: Vec<Bound<'_>> = classes
        .iter()
        .map(|&c| bind(c, &store, g(&store), &x, t).unwrap())
        .collect();
    let iters = 2_000;
    let multi = [
        (
            "rebind per op (4 binds + 4 runs)",
            measure(reps, iters, || {
                for &c in &classes {
                    let b = bind(c, &store, g(&store), &x, t).unwrap();
                    black_box(b.run(&store, focus, scope, &mut m).unwrap());
                }
            }),
        ),
        (
            "bind once, check per run (4 runs)",
            measure(reps, iters, || {
                for b in &bound {
                    black_box(b.run(&store, focus, scope, &mut m).unwrap());
                }
            }),
        ),
        (
            "bind once, check once, 4 callables",
            measure(reps, iters, || {
                store.check(&bound[0].guards.frame).unwrap();
                for b in &bound {
                    black_box((b.kernel)(&x, t, &mut m));
                }
            }),
        ),
        (
            "direct nd, one mask, 4 consumers",
            measure(reps, iters, || {
                nd::gt_i32_to_mask(&x, t, &mut m);
                black_box(nd::popcount_batch_u64(&m));
                black_box(nd::masked_sum_i32(&x, &m));
                black_box(nd::masked_min_i32(&x, &m));
                black_box(nd::mask_any(&m));
            }),
        ),
    ];
    println!("\n| {n} rows, 4 ops | ns per answer set | allocs |");
    println!("|---|---:|---:|");
    for (name, (ns, a)) in multi {
        println!("| {name} | {ns:.0} | {a:.0} |");
    }
    let evals_end = ROUTE_EVALS.load(Ordering::Relaxed);

    // The run-time guard on its own: frame seal, focus, scope.
    let (chk, chk_allocs) = measure(reps, 100_000, || {
        let _ = black_box(store.check(black_box(&bound[0].guards.frame)));
        black_box(black_box(focus) == bound[0].guards.focus);
        black_box(black_box(scope) == bound[0].guards.scope);
    });
    println!("\nframe/focus/scope check alone: {chk:.1} ns, {chk_allocs:.0} allocs");

    let inits: Vec<usize> = INITS.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    println!("\nresolutions per ClassId: {inits:?}");
    assert!(
        inits.iter().all(|&i| i <= 1),
        "one init per ClassId at most"
    );
    println!(
        "route evaluations during steady state: {} (single-op), {} (multi-op)",
        evals_after - evals_before,
        evals_end - evals_after
    );
    assert_eq!(evals_after, evals_before);
    assert_eq!(evals_end, evals_after);
    println!("ns per call, median of MPF_REPS. Reported, not gated.");
}
