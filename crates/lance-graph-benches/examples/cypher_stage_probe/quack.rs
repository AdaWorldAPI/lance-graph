//! The Quack arm: the same corpus, HAND-lowered to `lance_graph_quack::Query`
//! values and executed by mask-risc over borrowed lanes of the same data.
//!
//! This is a measurement instrument, not a frontend: each query's lowering is
//! written out here so its stage costs (lower, execute) can be compared with
//! DataFusion's. Each row says which existing Quack carrier answers it, or
//! names the gap that stops it. A gap is reported, never worked around with a
//! shape the substrate forbids (a `Keep` mask fed to another program's
//! `Semijoin`, a scattered mask consumed as an intermediate).
//!
//! Tables (`person_id` == row ordinal, so an id column IS a row-ordinal fk):
//! - Person: col 0 = age (I32), col 1 = person_id (U32); plane 0 = alpha.
//! - FRIEND_OF, stored in `dst` order: col 0 = src (U32), col 1 = dst (U32);
//!   plane 0 = alpha. Foreign plane 0 = Person alpha.

use lance_graph_mask_risc::{
    execute_into, materialize_rows, words_for, Foreign, ForeignPlane, LaneRef, Out, Planes,
    Program, Scratch, Value,
};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, ForeignPlane as FP, GroupAddr, GroupAgg, Mask, Query};

use crate::common::{fmt_us, measure, median, Answer, Data, PROBE_ID};

const AGE: Col = Col(0);
const PID: Col = Col(1);
const SRC: Col = Col(0);
const DST: Col = Col(1);
const ALPHA: Mask = Mask(0);
const PERSON: FP = FP(0);

#[derive(Clone, Copy, PartialEq)]
enum Table {
    Person,
    Edges,
}

/// One executable step: a query over one table, and the sink its terminal needs.
struct Step {
    table: Table,
    query: Query,
    sink: Sink,
}

#[derive(Clone, Copy)]
enum Sink {
    None,
    /// `Out::Mask` over the step's own table (`Agg::Rows`).
    KeepMask,
    /// `Out::I64` of length n_people (group fold keyed by a Person ordinal).
    PerPerson,
}

/// How a corpus query maps onto Quack today.
enum Plan {
    /// One program answers it.
    One(Step),
    /// Two group-count programs and a host-side O(n) dot product. Not one
    /// program: the walk count needs the per-node count of hop 1 summed over
    /// hop 2, i.e. "sum of a foreign value", one of #1311's unbuilt gaps.
    TwoFoldsAndDot(Step, Step),
    /// No existing carrier: the named gap.
    Gap(&'static str),
}

fn both_ends() -> Filter {
    Filter::and([
        Filter::plane(ALPHA),
        Filter::semijoin(SRC, PERSON),
        Filter::semijoin(DST, PERSON),
    ])
}

fn plan(id: &str) -> Plan {
    let person = |filter, agg| Step { table: Table::Person, query: Query { filter, agg }, sink: Sink::None };
    let edges = |filter, agg| Step { table: Table::Edges, query: Query { filter, agg }, sink: Sink::None };
    let gt = |k| Filter::and([Filter::plane(ALPHA), Filter::cmp(AGE, Cmp::GtI32(k))]);
    match id {
        "Q0" => Plan::One(person(Filter::plane(ALPHA), Agg::Count)),
        "Q1" => Plan::One(person(gt(50), Agg::Count)),
        "Q2" => Plan::One(person(gt(50), Agg::SumI32(AGE))),
        // one hop re-anchors on the edge table: one edge row is one path
        "Q3" => Plan::One(edges(both_ends(), Agg::Count)),
        "Q4" => Plan::TwoFoldsAndDot(
            Step {
                table: Table::Edges,
                query: Query {
                    filter: both_ends(),
                    agg: Agg::GroupReduce { key: GroupAddr::Local(DST), agg: GroupAgg::Count },
                },
                sink: Sink::PerPerson,
            },
            Step {
                table: Table::Edges,
                query: Query {
                    filter: both_ends(),
                    agg: Agg::GroupReduce { key: GroupAddr::Local(SRC), agg: GroupAgg::Count },
                },
                sink: Sink::PerPerson,
            },
        ),
        "Q5" => Plan::One(Step {
            table: Table::Person,
            query: Query {
                filter: Filter::and([Filter::plane(ALPHA), Filter::cmp(PID, Cmp::EqU32(PROBE_ID as u32))]),
                agg: Agg::Rows,
            },
            sink: Sink::KeepMask,
        }),
        // edge table is stored in dst order — the ordered-distinct precondition
        "Q6" | "Q6b" => Plan::One(edges(both_ends(), Agg::CountDistinctOrderedU32 { key: DST })),
        "Q7" | "Q8" => Plan::Gap(
            "ordered compare through an fk (a.age > k read via src): only EqU32Via exists; \
             a Person Keep mask fed to the edge Semijoin is the forbidden shape",
        ),
        "Q9" => Plan::One(person(gt(78), Agg::Any)),
        "QV" => Plan::Gap("sum of a foreign value (hop-2 walks = sum over src=0 edges of out[dst])"),
        "QVd" | "Q10" => Plan::Gap(
            "hop chain over a computed frontier: the hop-1 target mask would be a scattered \
             intermediate (RF-CHAIN)",
        ),
        _ => unreachable!(),
    }
}

struct World<'a> {
    d: &'a Data,
    pid: Vec<u32>,
    alpha_p: Vec<u64>,
    alpha_e: Vec<u64>,
}

impl<'a> World<'a> {
    fn new(d: &'a Data) -> Self {
        let ones = |n: usize| {
            let mut w = vec![u64::MAX; words_for(n)];
            if n % 64 != 0 {
                *w.last_mut().unwrap() = (1u64 << (n % 64)) - 1;
            }
            w
        };
        World {
            d,
            pid: (0..d.n_people as u32).collect(),
            alpha_p: ones(d.n_people),
            alpha_e: ones(d.src.len()),
        }
    }

    fn run(&self, step: &Step, program: &Program, out: Out<'_>) -> Value {
        let (n_rows, lanes, alpha) = match step.table {
            Table::Person => (
                self.d.n_people,
                vec![LaneRef::I32(&self.d.age), LaneRef::U32(&self.pid)],
                &self.alpha_p,
            ),
            Table::Edges => (
                self.d.src.len(),
                vec![LaneRef::U32(&self.d.src), LaneRef::U32(&self.d.dst)],
                &self.alpha_e,
            ),
        };
        let masks: [&[u64]; 1] = [alpha];
        let planes = Planes { n_rows, masks: &masks, lanes: &lanes };
        let fplanes = [ForeignPlane { words: &self.alpha_p, rows: self.d.n_people }];
        let foreign = Foreign { planes: &fplanes, lanes: &[] };
        let mut scratch = Scratch::for_program(program, n_rows).expect("scratch carves");
        execute_into(program, &planes, &foreign, &mut scratch, out).expect("executes")
    }

    /// Execute one step and turn its value into a corpus answer.
    fn answer(&self, id: &str, step: &Step, program: &Program) -> Answer {
        match step.sink {
            Sink::None => match self.run(step, program, Out::None) {
                Value::Count(c) => Answer::Int(c as i64),
                Value::SumI64(s) => Answer::Int(s),
                Value::Bool(b) if id == "Q9" => Answer::Exists(b),
                v => panic!("{id}: unexpected {v:?}"),
            },
            Sink::KeepMask => {
                let mut m = vec![0u64; words_for(self.d.n_people)];
                self.run(step, program, Out::Mask(&mut m));
                // the one materialiser: rows, then the caller's string column
                let rows = materialize_rows(&m, self.d.n_people);
                Answer::Strings(rows.into_iter().map(|r| self.d.name[r].clone()).collect())
            }
            Sink::PerPerson => unreachable!("handled by the two-fold plan"),
        }
    }

    fn fold(&self, step: &Step, program: &Program) -> Vec<i64> {
        let mut out = vec![0i64; self.d.n_people];
        match self.run(step, program, Out::I64(&mut out)) {
            Value::GroupReduced => out,
            v => panic!("unexpected {v:?}"),
        }
    }
}

/// Print one row per (size, query): verdict against the oracle, the lowering
/// cost, and the median execution cost over `reps` runs of the SAME program
/// (Quack's prepared shape: lower once, execute many).
pub fn suite(sizes: &[usize], reps_for: impl Fn(usize) -> usize) {
    println!("# specimen = fork-quack (hand-lowered)");
    println!("size\tq\tverdict\tlower_us\texec_us\tlower_allocs\texec_allocs\tplan");
    for &n in sizes {
        let d = crate::common::fixture(n);
        let w = World::new(&d);
        let reps = reps_for(n);
        for q in crate::common::corpus() {
            let want = crate::common::oracle(q.id, &d);
            match plan(q.id) {
                Plan::Gap(why) => println!("{n}\t{}\tGAP\t-\t-\t-\t-\t{why}", q.id),
                Plan::One(step) => {
                    let (prog, lc) = measure(|| lower(&step.query).expect("lowers"));
                    let mut ts = Vec::with_capacity(reps);
                    let mut got = None;
                    let mut ec = Default::default();
                    for _ in 0..reps {
                        let (a, c) = measure(|| w.answer(q.id, &step, &prog));
                        ts.push(c.ns);
                        ec = c;
                        got = Some(a);
                    }
                    let got = got.unwrap();
                    let verdict = if got == want { "OK".to_string() } else { format!("WRONG(got={got:?},want={want:?})") };
                    let ec: crate::common::Cost = ec;
                    println!(
                        "{n}\t{}\t{verdict}\t{}\t{}\t{}\t{}\tone program",
                        q.id,
                        fmt_us(lc.ns),
                        fmt_us(median(ts)),
                        lc.allocs,
                        ec.allocs
                    );
                }
                Plan::TwoFoldsAndDot(a, b) => {
                    let ((pa, pb), lc) = measure(|| (lower(&a.query).expect("lowers"), lower(&b.query).expect("lowers")));
                    let mut ts = Vec::with_capacity(reps);
                    let mut got = 0i64;
                    let mut ec = crate::common::Cost::default();
                    for _ in 0..reps {
                        let (v, c) = measure(|| {
                            let indeg = w.fold(&a, &pa);
                            let outdeg = w.fold(&b, &pb);
                            indeg.iter().zip(&outdeg).map(|(i, o)| i * o).sum::<i64>()
                        });
                        ts.push(c.ns);
                        ec = c;
                        got = v;
                    }
                    let got = Answer::Int(got);
                    let verdict = if got == want { "OK".to_string() } else { format!("WRONG(got={got:?},want={want:?})") };
                    println!(
                        "{n}\t{}\t{verdict}\t{}\t{}\t{}\t{}\ttwo folds + host dot (potential: needs foreign-value sum)",
                        q.id,
                        fmt_us(lc.ns),
                        fmt_us(median(ts)),
                        lc.allocs,
                        ec.allocs
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------- prepared

/// The smallest prepared-Cypher falsifier, on Q5 (`WHERE n.person_id = $id
/// RETURN n.name`): lower ONCE with a sentinel, locate the single op that
/// carries it (the parameter slot), then for every probe value patch only
/// that op and execute. Two checks per value:
/// - the patched program `==` a program freshly lowered for that value
///   (so the patch is exactly what re-lowering would produce);
/// - the answer equals the oracle's.
///
/// Returns the median patch+execute cost and the number of values checked.
pub fn prepared_param_probe(n: usize, values: &[u32]) -> (u128, usize) {
    use lance_graph_mask_risc::{MaskOp, Pred};
    let d = crate::common::fixture(n);
    let w = World::new(&d);
    let q = |v: u32| Query {
        filter: Filter::and([Filter::plane(ALPHA), Filter::cmp(PID, Cmp::EqU32(v))]),
        agg: Agg::Rows,
    };
    const SENTINEL: u32 = 0xDEAD_BEEF;
    let mut program = lower(&q(SENTINEL)).expect("lowers");
    let slots: Vec<usize> = program
        .ops
        .iter()
        .enumerate()
        .filter(|(_, op)| matches!(op, MaskOp::Pred { pred: Pred::EqU32 { v, .. }, .. } if *v == SENTINEL))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(slots.len(), 1, "the sentinel must occupy exactly one slot");
    let slot = slots[0];
    let step = Step {
        table: Table::Person,
        query: q(SENTINEL),
        sink: Sink::KeepMask,
    };
    let mut ts = Vec::with_capacity(values.len());
    for &v in values {
        let t = std::time::Instant::now();
        if let MaskOp::Pred { pred: Pred::EqU32 { v: slot_v, .. }, .. } = &mut program.ops[slot] {
            *slot_v = v;
        }
        let got = w.answer("Q5", &step, &program);
        ts.push(t.elapsed().as_nanos());
        assert_eq!(program, lower(&q(v)).expect("lowers"), "patched program == fresh lowering for {v}");
        let want = if (v as usize) < d.n_people {
            Answer::Strings(vec![d.name[v as usize].clone()])
        } else {
            Answer::Strings(vec![])
        };
        assert_eq!(got, want, "answer for id {v}");
    }
    (median(ts), values.len())
}
