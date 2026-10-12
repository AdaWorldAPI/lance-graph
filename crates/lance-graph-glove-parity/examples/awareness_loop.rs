//! D-SAG-1, the loop run once: project prepared methods into the
//! software-awareness graph, ask it which programs are H1 fusion candidates,
//! time each candidate against a fused scalar reference on THIS machine, and
//! record the timings as tier-labelled evidence.
//!
//! ```text
//! cargo run --release --example awareness_loop [rows] [reps]
//! ```
//!
//! The scalar loop is a reference for what one pass costs, not a fused
//! mask-risc terminal: no such terminal exists (`FusedFold` is `Count`/`Any`),
//! so the graph records the candidate's obligations open and the gap as a
//! measured ratio, never as an achieved speed-up.
use std::hint::black_box;
use std::time::Instant;

use lance_graph_glove_parity::awareness::{CpuTier, Measurement, SoftwareGraph};
use lance_graph_glove_parity::basis::{canonical_fields, FieldMap, Grade, SAP_FIELDS};
use lance_graph_glove_parity::method::{
    Cond, EffectClass, Fold, Glove, Law, MethodBody, MethodKey, MethodSpec, MethodTable, ReadingId,
    SchemaSeal, Universe,
};
use lance_graph_mask_risc::{execute_into, Foreign, LaneRef, Out, Planes, Program, Scratch, Value};
use lance_graph_ogar::ogar_vocab::class_ids::BILLABLE_WORK_ENTRY;
use lance_graph_sap::bind::{CatsBatch, BILLABLE, BILLING, HOURS};
use lance_graph_sap::binder::{live_words, CatsBinder, TABLE};
use lance_graph_sap::schema::{CatsSchema, FIELD_COUNT};

const INDICATORS: [&str; 4] = ["Billable", "Non-Billable", "Internal Cost", "Unknown"];
const HOURS_TEXT: [&str; 3] = ["8.50", "0.125", "2"];
const BILLABLE_TRUE: Cond = Cond::Eq {
    field: "billable",
    literal: "true",
};

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

fn tier() -> CpuTier {
    CpuTier {
        target: format!(
            "compiled avx512f={} avx2={}",
            cfg!(target_feature = "avx512f"),
            cfg!(target_feature = "avx2")
        ),
        #[cfg(target_arch = "x86_64")]
        avx512f: std::is_x86_feature_detected!("avx512f"),
        #[cfg(not(target_arch = "x86_64"))]
        avx512f: false,
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().map_or(1 << 20, |s| s.parse().unwrap());
    let reps: usize = args.next().map_or(51, |s| s.parse().unwrap());
    let rev = std::env::var("SAG_REV").unwrap_or_else(|_| "worktree".into());

    let batch = batch(n);
    let cats = CatsBinder::new(&batch);
    let basis = canonical_fields();
    // FIXTURE (as in `method_cost.rs`): SAP's real claim on `duration` is
    // `Hypothesized`; under it `billable_hours` is refused, and the graph
    // records that refusal instead of a program.
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
    let spec = |key, folds: &'static [Fold<'static>]| MethodSpec {
        key: MethodKey(key),
        concept: BILLABLE_WORK_ENTRY,
        schema: SchemaSeal::of(&basis),
        reading: ReadingId(1),
        effect: EffectClass::Pure,
        body: MethodBody::Query {
            filter: &[BILLABLE_TRUE],
            folds,
            law: Law::Values,
        },
    };
    let specs = [
        ("count_billable", spec([0xB1; 16], &[Fold::Count])),
        ("billable_hours", spec([0xB2; 16], &[Fold::Sum("duration")])),
    ];
    let glove = Glove {
        concept: BILLABLE_WORK_ENTRY,
        basis: &basis,
        claims: &claims,
        binder: &cats,
        table: TABLE,
        reading: ReadingId(1),
        universe: Universe(42),
    };
    let table = MethodTable::prepare(&glove, &specs);
    let mut g = SoftwareGraph::new();
    g.project(
        &table,
        &specs,
        &claims,
        "crates/lance-graph-glove-parity/examples/awareness_loop.rs",
        &rev,
    );
    println!(
        "graph: {} nodes, {} edges, rev {rev}",
        g.nodes().len(),
        g.edges().len()
    );
    println!(
        "methods using MaskedSumI32 (Quack over the edge table): {:?}",
        g.methods_using("MaskedSumI32").unwrap()
    );

    let mut programs: Vec<(String, Program)> = Vec::new();
    for (_, s) in &specs {
        if let Some(Ok(p)) = table.get(s.key) {
            for (i, q) in p.programs().iter().enumerate() {
                let hex: String = s.key.0.iter().map(|b| format!("{b:02x}")).collect();
                programs.push((format!("program:{hex}#{i}"), q.clone()));
            }
        }
    }
    let refs: Vec<(String, &Program)> = programs.iter().map(|(i, p)| (i.clone(), p)).collect();
    let candidates = g.fusion_candidates(&refs);
    println!("H1 candidates: {}", candidates.len());

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
    let scalar = || -> i64 {
        bill.iter()
            .zip(hours)
            .filter(|(b, _)| **b == 1)
            .map(|(_, h)| i64::from(*h))
            .sum()
    };
    let t = tier();
    for c in &candidates {
        let lance_graph_glove_parity::awareness::CandidateKind::FusePredicateIntoSum { program } =
            &c.kind
        else {
            continue;
        };
        let p = &programs.iter().find(|(i, _)| i == program).unwrap().1;
        let mut scratch = Scratch::for_program(p, n).unwrap();
        let mut run =
            || match execute_into(p, &planes, &Foreign::NONE, &mut scratch, Out::None).unwrap() {
                Value::SumI64(s) => s,
                v => panic!("{v:?}"),
            };
        // Both paths must give the same answer before either is timed.
        assert_eq!(run(), scalar(), "the reference computes another answer");
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for r in 0..reps {
            for k in 0..2 {
                let t0 = Instant::now();
                if (r + k) % 2 == 0 {
                    black_box(run());
                    a.push(t0.elapsed().as_nanos() as f64);
                } else {
                    black_box(scalar());
                    b.push(t0.elapsed().as_nanos() as f64);
                }
            }
        }
        a.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        let (ma, mb) = (a[a.len() / 2], b[b.len() / 2]);
        g.measure(Measurement {
            program: program.clone(),
            rows: n,
            tier: t.clone(),
            median_ns: ma,
        });
        g.measure(Measurement {
            program: format!("reference:scalar_fused:{program}"),
            rows: n,
            tier: t.clone(),
            median_ns: mb,
        });
        println!("candidate {program} of {:?}", c.methods);
        for o in &c.obligations {
            println!(
                "  obligation: {} — {}",
                o.what,
                o.discharged.as_deref().unwrap_or("OPEN")
            );
        }
        println!(
            "  tier {:?}; rows {n}; reps {reps}; median program {:.1} µs, scalar reference {:.1} µs, ratio {:.2}",
            t,
            ma / 1e3,
            mb / 1e3,
            ma / mb
        );
        println!("  recorded cost on this tier: {:?}", g.cost(program, n, &t));
        let other = CpuTier {
            target: "x86-64-v3".into(),
            avx512f: t.avx512f,
        };
        if other != t {
            println!(
                "  asked for another tier ({}): {:?}",
                other.target,
                g.cost(program, n, &other)
            );
        }
    }
}
