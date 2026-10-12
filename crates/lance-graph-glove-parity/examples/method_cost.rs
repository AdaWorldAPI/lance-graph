//! Experiment E: what the declarative method interface costs.
//!
//! Measures, on one SAP CATS batch, separately:
//!
//! 1. cold: `MethodTable::prepare` (binding) and `resolve_call` (text → key);
//! 2. warm: `MethodTable::call`, Quack and R2IL frontends;
//! 3. direct Quack: the same query, lowered once, executed;
//! 4. scalar reference: a row loop over the same lanes.
//!
//! Every timed path is asserted to give the same answer before it is timed.
//! Allocations are counted by a wrapping global allocator.
//!
//! Run: `cargo run --release --example method_cost -- [rows] [reps]`
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use lance_graph_glove_parity::basis::{canonical_fields, FieldMap, Grade, SAP_FIELDS};
use lance_graph_glove_parity::call::resolve_call;
use lance_graph_glove_parity::method::{
    Answer, Cond, EffectClass, Fold, FoldValue, Frontend, Glove, Law, MethodBody, MethodKey,
    MethodSpec, MethodTable, ReadingId, SchemaSeal, Universe,
};
use lance_graph_mask_risc::{execute_into, Foreign, LaneRef, Out, Planes, Scratch, Value};
use lance_graph_ogar::ogar_vocab::class_ids::BILLABLE_WORK_ENTRY;
use lance_graph_quack::bind::{table, Binder};
use lance_graph_quack::{lower, Agg, Query};
use lance_graph_sap::bind::{CatsBatch, BILLABLE, BILLING, HOURS};
use lance_graph_sap::binder::{live_words, CatsBinder, TABLE};
use lance_graph_sap::schema::{CatsSchema, FIELD_COUNT};

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

// SAFETY: forwards every call to `System` unchanged; only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(l.size() as u64, Relaxed);
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

const INDICATORS: [&str; 4] = ["Billable", "Non-Billable", "Internal Cost", "Unknown"];
const HOURS_TEXT: [&str; 3] = ["8.50", "0.125", "2"];

fn batch(n: usize) -> CatsBatch {
    let base: Vec<Option<&str>> = include_str!("../../lance-graph-sap/fixtures/cats.txt")
        .lines()
        .map(|v| if v == "\\N" { None } else { Some(v) })
        .collect();
    let mut input: [Vec<Option<&str>>; FIELD_COUNT] = std::array::from_fn(|i| vec![base[i]; n]);
    for i in 0..n {
        input[BILLING][i] = Some(INDICATORS[(i * 7 + i / 3) % 4]);
        input[HOURS][i] = Some(HOURS_TEXT[i % 3]);
    }
    CatsBatch::bind(
        CatsSchema::new(42, 0),
        std::array::from_fn(|i| input[i].as_slice()),
    )
    .unwrap()
}

/// Median of `reps` timings of `f`, in nanoseconds, plus allocations per call.
fn time<T>(reps: usize, mut f: impl FnMut() -> T) -> (u128, f64, f64) {
    let mut v = Vec::with_capacity(reps);
    let (a0, b0) = (ALLOCS.load(Relaxed), BYTES.load(Relaxed));
    for _ in 0..reps {
        let t = Instant::now();
        black_box(f());
        v.push(t.elapsed().as_nanos());
    }
    let (a1, b1) = (ALLOCS.load(Relaxed), BYTES.load(Relaxed));
    v.sort_unstable();
    (
        v[reps / 2],
        (a1 - a0) as f64 / reps as f64,
        (b1 - b0) as f64 / reps as f64,
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().map_or(1 << 20, |s| s.parse().unwrap());
    let reps: usize = args.next().map_or(51, |s| s.parse().unwrap());

    let batch = batch(n);
    let cats = CatsBinder::new(&batch);
    let basis = canonical_fields();
    let claims: Vec<FieldMap> = SAP_FIELDS
        .iter()
        .map(|m| FieldMap {
            grade: if m.canonical == "duration" {
                Grade::Exact
            } else {
                m.grade
            },
            ..*m
        })
        .collect();
    let spec = MethodSpec {
        key: MethodKey([0xB2; 16]),
        concept: BILLABLE_WORK_ENTRY,
        schema: SchemaSeal::of(&basis),
        reading: ReadingId(1),
        effect: EffectClass::Pure,
        body: MethodBody::Query {
            filter: &[Cond::Eq {
                field: "billable",
                literal: "true",
            }],
            folds: &[Fold::Sum("duration")],
            law: Law::Values,
        },
    };
    let glove = Glove {
        concept: BILLABLE_WORK_ENTRY,
        basis: &basis,
        claims: &claims,
        binder: &cats,
        table: TABLE,
        reading: ReadingId(1),
        universe: Universe(42),
    };
    let specs = [("billable_hours", spec)];
    let sql = "SELECT billable_work_entry::billable_hours(cats) FROM cats";

    let live = live_words(n);
    let lanes = batch.lanes();
    let planes = Planes {
        n_rows: n,
        masks: &[live.as_slice()],
        lanes: &lanes,
    };

    // The answer every path must give.
    let (LaneRef::U32(bill), LaneRef::I32(hours)) = (lanes[BILLABLE], lanes[HOURS]) else {
        panic!("lane kinds")
    };
    let scalar = || -> i64 {
        bill.iter()
            .zip(hours)
            .filter(|(b, _)| **b == 1)
            .map(|(_, h)| i64::from(*h))
            .sum()
    };
    let want = scalar();

    let t = MethodTable::prepare(&glove, &specs);
    let key = resolve_call(sql, &t).unwrap().method;
    let expect = Answer::Values(vec![FoldValue::Scalar(want)]);
    for f in [Frontend::Quack, Frontend::R2il] {
        assert_eq!(t.call(key, &planes, Universe(42), f).unwrap(), expect);
    }

    let tid = cats.table(TABLE).unwrap();
    let hours_col = cats.field(tid, "hours_logged").unwrap().col;
    let direct = Query {
        filter: table(TABLE)
            .where_eq("billable", "true")
            .count()
            .bind(&cats)
            .unwrap()
            .filter,
        agg: Agg::SumI32(hours_col),
    };
    let program = lower(&direct).unwrap();
    // The method compiles to the very program a direct query does.
    let prepared = t.get(key).unwrap().as_ref().unwrap();
    let method_q = prepared.queries();
    // A call executes the program lowered at preparation, and it is the
    // very program the direct query lowers to.
    assert_eq!(
        prepared.programs(),
        std::slice::from_ref(&program),
        "same lowered program"
    );
    let mut scratch = Scratch::for_program(&program, n).unwrap();
    let mut run_direct = || match execute_into(
        &program,
        &planes,
        &Foreign::NONE,
        &mut scratch,
        Out::None,
    )
    .unwrap()
    {
        Value::SumI64(s) => s,
        v => panic!("{v:?}"),
    };
    assert_eq!(run_direct(), want);

    println!("rows {n}, reps {reps} (median), answer {want}");
    println!(
        "lowered program: {} mask ops + terminal, {} scratch slots",
        program.ops.len(),
        program.scratch_slots
    );
    let row = |name: &str, (ns, a, b): (u128, f64, f64)| {
        println!(
            "{name:<34} {:>12.3} µs  {:>8.2} ns/row  {:>7.1} allocs  {:>11.0} B alloc",
            ns as f64 / 1e3,
            ns as f64 / n as f64,
            a,
            b
        );
    };
    row(
        "cold: prepare (bind 1 method)",
        time(reps, || MethodTable::prepare(&glove, &specs)),
    );
    row(
        "cold: resolve_call (text → key)",
        time(reps, || resolve_call(sql, &t).unwrap()),
    );
    row(
        "cold: lower (bound query → program)",
        time(reps, || lower(&method_q[0]).unwrap()),
    );
    // Hot paths interleaved: each rep runs all four in a rotating order,
    // so machine drift lands on every path alike. Median and minimum.
    let mut samples: [Vec<u128>; 4] = Default::default();
    for r in 0..reps {
        for k in 0..4 {
            let path = (r + k) % 4;
            let t0 = Instant::now();
            match path {
                0 => {
                    black_box(t.call(key, &planes, Universe(42), Frontend::Quack).unwrap());
                }
                1 => {
                    black_box(t.call(key, &planes, Universe(42), Frontend::R2il).unwrap());
                }
                2 => {
                    black_box(run_direct());
                }
                _ => {
                    black_box(scalar());
                }
            }
            samples[path].push(t0.elapsed().as_nanos());
        }
    }
    let names = [
        "warm: method call, Quack",
        "warm: method call, R2IL",
        "direct Quack (lowered once)",
        "scalar reference (row loop)",
    ];
    for (name, v) in names.iter().zip(samples.iter_mut()) {
        v.sort_unstable();
        println!(
            "{name:<34} median {:>8.1} µs  min {:>8.1} µs  {:>6.3} ns/row (min)",
            v[v.len() / 2] as f64 / 1e3,
            v[0] as f64 / 1e3,
            v[0] as f64 / n as f64
        );
    }
    let (_, a, b) = time(reps, || {
        t.call(key, &planes, Universe(42), Frontend::Quack).unwrap()
    });
    println!("allocs per warm Quack call:  {a:.1} ({b:.0} B)");
    let (_, a, b) = time(reps, || {
        t.call(key, &planes, Universe(42), Frontend::R2il).unwrap()
    });
    println!("allocs per warm R2IL call:   {a:.1} ({b:.0} B)");
    let (_, a, b) = time(reps, &mut run_direct);
    println!("allocs per direct call:      {a:.1} ({b:.0} B)");
    println!(
        "lane bytes read per call: billable {} + hours {} + live {}",
        n * 4,
        n * 4,
        live.len() * 8
    );
}
