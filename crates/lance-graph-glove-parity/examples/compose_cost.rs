//! D-CMH-G1 cost: what loading a composed method from text costs, cold, and
//! what calling it costs, warm, against the same query run directly.
//!
//! Every timed path is asserted to give the same answer before it is timed.
//! Run: `cargo run --release --example compose_cost -- [rows] [reps]`
use std::hint::black_box;
use std::time::Instant;

use lance_graph_glove_parity::basis::{canonical_fields, FieldMap, Grade, SAP_FIELDS};
use lance_graph_glove_parity::decl::parse_decls;
use lance_graph_glove_parity::method::{
    Answer, FoldValue, Frontend, Glove, MethodTable, ReadingId, Universe,
};
use lance_graph_mask_risc::{execute_into, Foreign, LaneRef, Out, Planes, Scratch, Value};
use lance_graph_ogar::ogar_vocab::class_ids::BILLABLE_WORK_ENTRY;
use lance_graph_sap::bind::{CatsBatch, BILLABLE, BILLING, HOURS};
use lance_graph_sap::binder::{live_words, CatsBinder, TABLE};
use lance_graph_sap::schema::{CatsSchema, FIELD_COUNT};

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

fn median_min(mut v: Vec<u128>) -> (f64, f64) {
    v.sort_unstable();
    (v[v.len() / 2] as f64 / 1e3, v[0] as f64 / 1e3)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().map_or(1 << 20, |s| s.parse().unwrap());
    let reps: usize = args.next().map_or(101, |s| s.parse().unwrap());
    let text = include_str!("../tests/fixtures/lego.methods");

    let batch = batch(n);
    let cats = CatsBinder::new(&batch);
    let basis = canonical_fields();
    // FIXTURE: proves SAP's hypothesized `duration` claim, as the tests do.
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
    let glove = Glove {
        concept: BILLABLE_WORK_ENTRY,
        basis: &basis,
        claims: &claims,
        binder: &cats,
        table: TABLE,
        reading: ReadingId(1),
        universe: Universe(42),
        relations: &[],
    };
    let live = live_words(n);
    let lanes = batch.lanes();
    let planes = Planes {
        n_rows: n,
        masks: &[live.as_slice()],
        lanes: &lanes,
    };
    let (LaneRef::U32(bill), LaneRef::I32(hours)) = (lanes[BILLABLE], lanes[HOURS]) else {
        panic!("lane kinds")
    };
    let want: i64 = bill
        .iter()
        .zip(hours)
        .filter(|(b, _)| **b == 1)
        .map(|(_, h)| i64::from(*h))
        .sum();

    // Cold: text -> declarations -> prepared table (3 methods, 2 references).
    let mut parse = Vec::new();
    let mut prep = Vec::new();
    for _ in 0..reps {
        let t0 = Instant::now();
        let d = black_box(parse_decls(text).unwrap());
        parse.push(t0.elapsed().as_nanos());
        let t1 = Instant::now();
        black_box(MethodTable::prepare_with(&glove, &[], &d));
        prep.push(t1.elapsed().as_nanos());
    }
    let decls = parse_decls(text).unwrap();
    let t = MethodTable::prepare_with(&glove, &[], &decls);
    let key = t.resolve("billable_hours").unwrap();
    let expect = Answer::Values(vec![FoldValue::Scalar(want)]);
    for f in [Frontend::Quack, Frontend::R2il] {
        assert_eq!(t.call(key, &planes, Universe(42), f).unwrap(), expect);
    }
    // Direct: the prepared method's own program, run without the method
    // layer. The composition test pins that this is also the program a
    // direct Quack query lowers to.
    let program = t.get(key).unwrap().as_ref().unwrap().programs()[0].clone();
    let mut scratch = Scratch::for_program(&program, n).unwrap();
    let mut direct = || match execute_into(
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
    assert_eq!(direct(), want);

    let mut warm: [Vec<u128>; 3] = Default::default();
    for r in 0..reps {
        for k in 0..3 {
            let path = (r + k) % 3;
            let t0 = Instant::now();
            match path {
                0 => {
                    black_box(t.call(key, &planes, Universe(42), Frontend::Quack).unwrap());
                }
                1 => {
                    black_box(t.call(key, &planes, Universe(42), Frontend::R2il).unwrap());
                }
                _ => {
                    black_box(direct());
                }
            }
            warm[path].push(t0.elapsed().as_nanos());
        }
    }
    println!(
        "tier: avx512f={} avx2={} (compiled target features)",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2")
    );
    println!("rows {n}, reps {reps}, answer {want}");
    let row = |name: &str, (med, min): (f64, f64)| {
        println!("{name:<40} median {med:>10.2} µs  min {min:>10.2} µs");
    };
    row("cold: parse 3 declarations", median_min(parse));
    row(
        "cold: prepare_with (resolve+inline+lower)",
        median_min(prep),
    );
    let [q, r, d] = warm;
    row("warm: composed billable_hours, Quack", median_min(q));
    row("warm: composed billable_hours, R2IL", median_min(r));
    row("warm: direct program (same Program)", median_min(d));
}
