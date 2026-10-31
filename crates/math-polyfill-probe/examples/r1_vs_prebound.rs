//! R1 against the prebound route. Lab probe, reported not gated. Adds nothing
//! to any production crate: everything below lives in this example.
//!
//! Operation, identical in every arm: `SUM { a[r] : a[r] < t }` over one i32
//! lane, i64 result, t = 5. Every arm is checked against `sum_where_oracle`
//! before timing, at every row count. All preparation happens before the
//! timed loop.
//!
//! Arms:
//!   1 direct nd       `lt_i32_to_mask` + `masked_sum_i32` into a reused mask
//!   2 prebound        ClassId -> LazyLock -> resolved once -> stored fn pointer
//!   3 prebound+guard  arm 2 plus `Guards::check(frames, (focus, scope))`
//!   4 mr prepared     hand-built mask-risc program, `execute_compiled`
//!   5 r2il prepared   FoldDialect run once, `last_program` kept, `execute_compiled`
//!   6 dialect         `SumWhere::eval` (FoldDialect fold) per call
//!   7 R1 run          `binding::prepare` once, `binding::run` per call
//!
//! `MPF_REPS`, `MPF_ITERS` as in `classid_route`. ns per call, median.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;
use std::time::Instant;

use lance_graph_contract::class_view::{ClassId as ViewClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{Focus, FrameRevisions, FrameStore, Guards, Scope};
use lance_graph_dt64::{
    AlgebraRef, Arg, BindStore, ClassSchema, Effect, Extent, MethodBinding, MethodId, MethodRef,
    SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mask_risc::exec::{execute_compiled, tile_words_for, Scratch};
use lance_graph_mask_risc::ir::{
    CompiledProgram, MaskOp, Operand, Pred as MPred, Program, Terminal,
};
use lance_graph_mask_risc::{words_for, Foreign, LaneRef, Out, Planes, Value};
use lance_graph_r2il::binding::{
    prepare, run, ExecPermit, FieldLanes, R2ilLens, Receipt, Resolution,
};
use lance_graph_r2il::FoldDialect;
use math_polyfill_probe::native::{sum_where_oracle, Pred, SumWhere};
use ndarray::simd as nd;
use ogar_loco::{FnIndex, Vocabulary};
use ogar_r2il::{R2ILVocabulary, CONCEPT_R2IL_FOLD};

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

// ── prebound route (arms 2, 3) ─────────────────────────────────────────────

type Kernel = fn(&[i32], i32, &mut [u64]) -> i64;

#[derive(Clone, Copy)]
struct ClassId(usize);
const SUM_LT: ClassId = ClassId(0);

static INITS: AtomicUsize = AtomicUsize::new(0);

fn k_sum_lt(x: &[i32], t: i32, m: &mut [u64]) -> i64 {
    nd::lt_i32_to_mask(x, t, m);
    nd::masked_sum_i32(x, m)
}

static ROUTES: [LazyLock<Kernel>; 1] = [LazyLock::new(|| {
    INITS.fetch_add(1, Ordering::Relaxed);
    k_sum_lt as Kernel
})];

struct Prebound {
    kernel: Kernel,
    guards: Guards,
}

fn bind(c: ClassId, guards: Guards) -> Prebound {
    Prebound {
        kernel: *ROUTES[c.0],
        guards,
    }
}

#[inline(never)]
fn probe_direct(x: &[i32], t: i32, m: &mut [u64]) -> i64 {
    nd::lt_i32_to_mask(x, t, m);
    nd::masked_sum_i32(x, m)
}

#[inline(never)]
fn probe_prebound(b: &Prebound, x: &[i32], t: i32, m: &mut [u64]) -> i64 {
    (b.kernel)(x, t, m)
}

#[inline(never)]
fn probe_guarded(
    b: &Prebound,
    frames: &FrameStore,
    now: (Focus, Scope),
    x: &[i32],
    t: i32,
    m: &mut [u64],
) -> Option<i64> {
    b.guards.check(frames, now).ok()?;
    Some((b.kernel)(x, t, m))
}

// ── R1 fixture (after lance-graph-r2il/tests/integration/binding.rs) ───────

const CLASS: u32 = 0x7F0B_0001;
const CLASS_VIEW: ViewClassId = 0x7F0B;
const LENS: u32 = (CONCEPT_R2IL_FOLD as u32) << 16;
const REV: SchemaRev = SchemaRev(1);
const R2IL: R2ilLens = R2ilLens {
    lens: LENS,
    schema_rev: REV,
};
const FIELDS: [&str; 3] = ["a", "m", "s"];
/// Field `a` (0) is resident lane 0; `m` and `s` are computed.
const LANES: [Option<u16>; 3] = [Some(0), None, None];
/// The lens's words; only the label binds a word to an opcode.
const WORDS: [&str; 2] = ["IntSLess", "SUM"];
const W_LT: u8 = 0;
const W_SUM: u8 = 1;
const F_A: u8 = 0;
const F_M: u8 = 1;
const F_S: u8 = 2;
const METHOD: u32 = 1;
const POSITION: u16 = 3;

struct Views {
    class: Vec<FieldRef>,
    lens: Vec<FieldRef>,
}

fn refs(labels: &[&str], ns: &str) -> Vec<FieldRef> {
    labels
        .iter()
        .map(|l| FieldRef::new(format!("lab:{ns}/{l}"), (*l).to_string()))
        .collect()
}

impl ClassView for Views {
    fn fields(&self, class: ViewClassId) -> &[FieldRef] {
        match class {
            CLASS_VIEW => &self.class,
            c if c == CONCEPT_R2IL_FOLD => &self.lens,
            _ => &[],
        }
    }
    fn template(&self, _class: ViewClassId) -> DisplayTemplate {
        DisplayTemplate::Card
    }
    fn dolce_category_id(&self, _class: ViewClassId) -> u8 {
        0
    }
}

fn ogar_arities() -> Vec<u8> {
    WORDS
        .iter()
        .map(|w| {
            (0..=u8::MAX)
                .map(FnIndex)
                .find(|&f| R2ILVocabulary.name(f) == Some(*w))
                .and_then(|f| R2ILVocabulary.stack_arity(f))
                .unwrap_or(1)
        })
        .collect()
}

fn step(target: u8, word: u8, args: Vec<Arg>) -> Effect {
    Effect {
        target,
        algebra: AlgebraRef { lens: LENS, word },
        args,
    }
}

fn masked_sum_method(t: i32) -> MethodBinding {
    let reads = WideFieldMask::from_positions(&[F_A]);
    MethodBinding {
        id: MethodId(METHOD),
        sig_rev: SigRev(1),
        name: "sum_where_lt".to_string(),
        on: reads.clone(),
        when: None,
        reads,
        effects: vec![
            step(
                F_M,
                W_LT,
                vec![Arg::Field(F_A), Arg::Const(i64::from(t))],
            ),
            step(F_S, W_SUM, vec![Arg::Field(F_M), Arg::Field(F_A)]),
        ],
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

// ── measurement ────────────────────────────────────────────────────────────

fn env(name: &str, d: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

struct Measured {
    ns: f64,
    allocs: f64,
    /// Every call `measure` made, warm-up included.
    calls: usize,
}

/// Median ns per call, plus heap allocations per call.
fn measure(reps: usize, iters: usize, mut f: impl FnMut()) -> Measured {
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
    Measured {
        ns: v[v.len() / 2],
        allocs,
        calls: 200 + reps * iters,
    }
}

fn sum_of(v: Value) -> i64 {
    match v {
        Value::SumI64(s) => s,
        other => panic!("MaskedSumI32 yields SumI64, got {other:?}"),
    }
}

fn answer(r: &Receipt) -> i64 {
    r.output
        .values
        .iter()
        .find(|(f, _)| *f == F_S)
        .map(|&(_, v)| v)
        .expect("R1 receipt carries the SUM field")
}

fn main() {
    let reps = env("MPF_REPS", 21);
    let iters0 = env("MPF_ITERS", 4000);
    let t: i32 = 5;
    let focus = Focus { revision: 1 };
    let scope = Scope { id: 1 };
    let now = (focus, scope);
    let frames = FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 1,
    });

    // R1 preparation: cold, once, outside every timer.
    let views = Views {
        class: refs(&FIELDS, "class"),
        lens: refs(&WORDS, "r2il"),
    };
    let method = masked_sum_method(t);
    let mut bstore = BindStore::new();
    bstore
        .declare_algebra(LENS, REV, Extent::DR64_CUBE, ogar_arities())
        .unwrap();
    bstore
        .register(
            CLASS,
            ClassSchema {
                schema_rev: REV,
                field_count: FIELDS.len(),
                extent: Extent::DR64_CUBE,
            },
            vec![method.clone()],
        )
        .unwrap();
    let route_version = bstore.route(POSITION, CLASS).unwrap();
    let mref = MethodRef::of(CLASS, &method);
    let r1_guards = Guards {
        frame: frames.seal(),
        focus,
        scope,
    };
    let plan = prepare(
        &bstore,
        &views,
        R2IL,
        &Resolution {
            method: &mref,
            schema_rev: REV,
            route: (route_version, POSITION),
            guards: &r1_guards,
        },
        &frames,
        now,
    )
    .expect("R1 prepare");
    let permit = ExecPermit {
        methods: vec![mref.clone()],
    };

    let prebound2 = bind(
        SUM_LT,
        Guards {
            frame: frames.seal(),
            focus,
            scope,
        },
    );
    let prebound3 = bind(
        SUM_LT,
        Guards {
            frame: frames.seal(),
            focus,
            scope,
        },
    );

    let arms = [
        "direct nd",
        "prebound",
        "prebound+guard",
        "mr prepared",
        "r2il prepared",
        "dialect",
        "R1 run",
    ];
    let mut time_rows = Vec::new();
    let mut alloc_rows = Vec::new();
    let mut prog_rows = Vec::new();

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
        let want = sum_where_oracle(&x, None, Pred::Lt(t)) as i64;

        // arm 4: hand-built mask-risc program
        let cp = CompiledProgram::new(Program::new(
            vec![MaskOp::Pred {
                pred: MPred::LtI32 { lane: 0, t },
                under: None,
                dst: 0,
            }],
            Terminal::MaskedSumI32 {
                mask: Operand::Scratch(0),
                lane: 0,
            },
        ));
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

        // arm 5: FoldDialect run once, program kept
        let op = SumWhere {
            lane: 0,
            present: None,
            pred: Pred::Lt(t),
        };
        let mut d5 = FoldDialect::new(&planes, &Foreign::NONE);
        let mut stack5 = Vec::with_capacity(8);
        assert_eq!(op.eval(&mut d5, &mut stack5), Ok(want));
        let r2 = CompiledProgram::new(d5.last_program().expect("one fold ran").clone());

        // arm 6: the interpreted reading, per call
        let mut d6 = FoldDialect::new(&planes, &Foreign::NONE);
        let mut stack6 = Vec::with_capacity(8);

        // arm 7: R1 run, dialect lent once
        let mut d7 = FoldDialect::new(&planes, &Foreign::NONE);
        let fields = FieldLanes { lanes: &LANES };

        // oracle check, every arm, before timing
        assert!(want != 0, "the fixture must select rows");
        assert_eq!(probe_direct(&x, t, &mut m), want, "direct nd, n={n}");
        assert_eq!(
            probe_prebound(&prebound2, &x, t, &mut m),
            want,
            "prebound, n={n}"
        );
        assert_eq!(
            probe_guarded(&prebound3, &frames, now, &x, t, &mut m),
            Some(want),
            "prebound+guard, n={n}"
        );
        assert_eq!(run_cp(&cp, &mut buf), want, "mr prepared, n={n}");
        assert_eq!(run_cp(&r2, &mut buf), want, "r2il prepared, n={n}");
        assert_eq!(op.eval(&mut d6, &mut stack6), Ok(want), "dialect, n={n}");
        let r1 = run(&plan, &bstore, &frames, now, &permit, fields, &mut d7).expect("R1 run");
        assert_eq!(answer(&r1), want, "R1 run, n={n}");

        let p5 = d5.programs_run();
        let (p6, p7) = (d6.programs_run(), d7.programs_run());
        let mut res = Vec::with_capacity(arms.len());
        res.push(measure(reps, iters, || {
            black_box(probe_direct(black_box(&x), black_box(t), &mut m));
        }));
        res.push(measure(reps, iters, || {
            black_box(probe_prebound(
                black_box(&prebound2),
                black_box(&x),
                black_box(t),
                &mut m,
            ));
        }));
        res.push(measure(reps, iters, || {
            black_box(probe_guarded(
                black_box(&prebound3),
                &frames,
                now,
                black_box(&x),
                black_box(t),
                &mut m,
            ));
        }));
        res.push(measure(reps, iters, || {
            black_box(run_cp(&cp, &mut buf));
        }));
        res.push(measure(reps, iters, || {
            black_box(run_cp(&r2, &mut buf));
        }));
        res.push(measure(reps, iters, || {
            let _ = black_box(op.eval(&mut d6, &mut stack6));
        }));
        res.push(measure(reps, iters, || {
            if let Ok(r) = run(&plan, &bstore, &frames, now, &permit, fields, &mut d7) {
                black_box(answer(&r));
            }
        }));

        // arm 5 never touches its dialect after the first fold.
        let progs = [
            (d5.programs_run() - p5) as f64 / res[4].calls as f64,
            (d6.programs_run() - p6) as f64 / res[5].calls as f64,
            (d7.programs_run() - p7) as f64 / res[6].calls as f64,
        ];

        // the timed loops must still produce the oracle's answer
        assert_eq!(op.eval(&mut d6, &mut stack6), Ok(want));
        let again = run(&plan, &bstore, &frames, now, &permit, fields, &mut d7).expect("R1 run");
        assert_eq!(answer(&again), want, "R1 run after timing, n={n}");

        time_rows.push((n, res.iter().map(|r| r.ns).collect::<Vec<_>>()));
        alloc_rows.push((n, res.iter().map(|r| r.allocs).collect::<Vec<_>>()));
        prog_rows.push((n, progs));
    }

    let resolutions = INITS.load(Ordering::Relaxed);
    assert_eq!(resolutions, 1, "one resolution per ClassId");

    println!("\nns per call, median of MPF_REPS");
    print!("\n| rows |");
    for a in arms {
        print!(" {a} |");
    }
    println!();
    println!("|---:|{}", "---:|".repeat(arms.len()));
    for (n, v) in &time_rows {
        print!("| {n} |");
        for ns in v {
            print!(" {ns:.1} |");
        }
        println!();
    }

    println!("\nallocations per call");
    print!("\n| rows |");
    for a in arms {
        print!(" {a} |");
    }
    println!();
    println!("|---:|{}", "---:|".repeat(arms.len()));
    for (n, v) in &alloc_rows {
        print!("| {n} |");
        for a in v {
            print!(" {a:.1} |");
        }
        println!();
    }

    println!("\nFoldDialect::programs_run() per call (the dialect each arm owns)");
    println!("\n| rows | r2il prepared | dialect | R1 run |");
    println!("|---:|---:|---:|---:|");
    for (n, p) in &prog_rows {
        println!("| {n} | {:.2} | {:.2} | {:.2} |", p[0], p[1], p[2]);
    }

    println!("\nresolutions of the prebound ClassId: {resolutions}");
    println!("Reported, not gated.");
}
