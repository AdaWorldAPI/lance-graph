//! D-SAG-1: the software-awareness graph over #1504's methods.
//!
//! The methods are #1504's own declarations (`tests/method.rs`), projected
//! with the table that prepared them. Every structural verdict below is the
//! compiler's (`MethodTable::prepare`, `class_view::compute_dag_is_acyclic`);
//! the graph records it and the queries read it back through Quack.
use lance_graph_contract::class_view::ComputeEdge;
use lance_graph_glove_parity::awareness::{
    description_length, field_ident, method_ident, verdict, Candidate, CandidateKind, CapStatus,
    CertError, Certificate, CostError, CpuTier, Measurement, Rel, SoftwareGraph, Tier, Verdict,
    Workload,
};
use lance_graph_glove_parity::basis::{canonical_fields, FieldMap, Grade, SAP_FIELDS};
use lance_graph_glove_parity::method::{
    Answer, Cond, EffectClass, Fold, FoldValue, Frontend, Glove, Law, MethodBody, MethodKey,
    MethodRef, MethodSpec, MethodTable, ReadingId, SchemaSeal, Universe,
};
use lance_graph_mask_risc::{LaneRef, Planes, Program};
use lance_graph_ogar::ogar_vocab::class_ids::{BILLABLE_WORK_ENTRY, DOCUMENT};
use lance_graph_quack::bind::{Binder, BoundField, FieldKind, TableId};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, Mask, Query};
use lance_graph_sap::bind::{billable_code, CatsBatch, BILLING, HOURS};
use lance_graph_sap::binder::{live_words, CatsBinder, TABLE};
use lance_graph_sap::schema::{CatsSchema, FIELD_COUNT};

const COUNT_BILLABLE: MethodKey = MethodKey([0xB1; 16]);
const COUNT_BY_DURATION: MethodKey = MethodKey([0xD0; 16]);
const NOT_BILLABLE: MethodKey = MethodKey([0xB0; 16]);
const BILLABLE_HOURS: MethodKey = MethodKey([0xB2; 16]);
const HOURS_BY_BILLABLE: MethodKey = MethodKey([0xB3; 16]);
const READING: ReadingId = ReadingId(1);
const UNIVERSE: Universe = Universe(42);
const SOURCE: &str = "crates/lance-graph-glove-parity/tests/awareness.rs";
const REV: &str = "worktree";

const BILLABLE_TRUE: Cond = Cond::Eq {
    field: "billable",
    literal: "true",
};

const fn query(
    filter: &'static [Cond<'static>],
    folds: &'static [Fold<'static>],
) -> MethodBody<'static> {
    MethodBody::Query {
        filter,
        folds,
        law: Law::Values,
    }
}

fn seal() -> SchemaSeal {
    SchemaSeal::of(&canonical_fields())
}

fn pure(key: MethodKey, body: MethodBody<'static>) -> MethodSpec<'static> {
    MethodSpec {
        key,
        concept: BILLABLE_WORK_ENTRY,
        schema: seal(),
        reading: READING,
        effect: EffectClass::Pure,
        body,
    }
}

/// #1504's methods (`tests/method.rs::specs`), minus the action.
fn specs() -> Vec<(&'static str, MethodSpec<'static>)> {
    vec![
        (
            "count_billable",
            pure(COUNT_BILLABLE, query(&[BILLABLE_TRUE], &[Fold::Count])),
        ),
        (
            "count_by_duration",
            pure(
                COUNT_BY_DURATION,
                query(
                    &[Cond::Eq {
                        field: "duration",
                        literal: "1",
                    }],
                    &[Fold::Count],
                ),
            ),
        ),
        (
            "count_not_billable",
            pure(
                NOT_BILLABLE,
                query(
                    &[Cond::Ne {
                        field: "billable",
                        literal: "true",
                    }],
                    &[Fold::Count],
                ),
            ),
        ),
        (
            "billable_hours",
            pure(
                BILLABLE_HOURS,
                query(&[BILLABLE_TRUE], &[Fold::Count, Fold::Sum("duration")]),
            ),
        ),
        (
            "hours_by_billable",
            pure(
                HOURS_BY_BILLABLE,
                query(
                    &[],
                    &[Fold::GroupSum {
                        key: "billable",
                        val: "duration",
                        groups: 2,
                    }],
                ),
            ),
        ),
    ]
}

const INDICATORS: [&str; 4] = ["Billable", "Non-Billable", "Internal Cost", "Unknown"];
const HOURS_TEXT: [&str; 3] = ["8.50", "0.125", "2"];

fn indicator(i: usize) -> &'static str {
    INDICATORS[(i * 7 + i / 3) % 4]
}

fn batch(n: usize) -> CatsBatch {
    let base: Vec<Option<&str>> = include_str!("../../lance-graph-sap/fixtures/cats.txt")
        .lines()
        .map(|v| if v == "\\N" { None } else { Some(v) })
        .collect();
    let mut input: [Vec<Option<&str>>; FIELD_COUNT] = std::array::from_fn(|i| vec![base[i]; n]);
    for i in 0..n {
        input[BILLING][i] = Some(indicator(i));
        input[HOURS][i] = Some(HOURS_TEXT[i % 3]);
    }
    CatsBatch::bind(
        CatsSchema::new(42, 0),
        std::array::from_fn(|i| input[i].as_slice()),
    )
    .unwrap()
}

/// FIXTURE (as in `tests/method.rs`): SAP's real claim on `duration` is
/// `Hypothesized`.
fn with_duration() -> Vec<FieldMap> {
    SAP_FIELDS
        .iter()
        .map(|m| FieldMap {
            grade: if m.canonical == "duration" {
                Grade::Exact
            } else {
                m.grade
            },
            ..*m
        })
        .collect()
}

fn glove<'a>(basis: &'a [String], binder: &'a dyn Binder, claims: &'a [FieldMap]) -> Glove<'a> {
    Glove {
        concept: BILLABLE_WORK_ENTRY,
        basis,
        claims,
        binder,
        table: TABLE,
        reading: READING,
        universe: UNIVERSE,
    }
}

struct World {
    table: MethodTable,
    graph: SoftwareGraph,
}

fn world(batch: &CatsBatch, claims: &[FieldMap], specs: &[(&str, MethodSpec<'_>)]) -> World {
    let basis = canonical_fields();
    let cats = CatsBinder::new(batch);
    let table = MethodTable::prepare(&glove(&basis, &cats, claims), specs);
    let mut graph = SoftwareGraph::new();
    graph.project(&table, specs, claims, SOURCE, REV);
    World { table, graph }
}

fn programs(t: &MethodTable, key: MethodKey) -> Vec<Program> {
    t.get(key).unwrap().as_ref().unwrap().programs().to_vec()
}

fn hours_ident() -> String {
    method_ident(BILLABLE_HOURS)
}

/// FAILS IF: `billable_hours` is not projected with its class, fields,
/// conditions, folds, law, two programs, their primitives, its claims and
/// provenance — each at the tier that knows it.
#[test]
fn billable_hours_is_projected_with_tiers_and_provenance() {
    let b = batch(65);
    let w = world(&b, &with_duration(), &specs());
    let g = &w.graph;
    let m = hours_ident();
    assert_eq!(
        g.out(&format!("class:{BILLABLE_WORK_ENTRY:#06x}"), Rel::HasMethod)
            .iter()
            .filter(|(d, _)| *d == m)
            .count(),
        1
    );
    let billable = field_ident(BILLABLE_WORK_ENTRY, seal(), "billable");
    let duration = field_ident(BILLABLE_WORK_ENTRY, seal(), "duration");
    let mut reads: Vec<_> = g.out(&m, Rel::ReadsField);
    reads.sort();
    assert_eq!(
        reads,
        vec![
            (billable.as_str(), Tier::Declared),
            (duration.as_str(), Tier::Declared)
        ]
    );
    assert_eq!(
        g.out(&m, Rel::Filters),
        vec![(format!("cond:{billable}=true@r1").as_str(), Tier::Declared)]
    );
    assert_eq!(
        g.out(&m, Rel::FinishedBy),
        vec![("law:values", Tier::Declared)]
    );
    // Two folds, two programs: the compiler emits the selection once per fold.
    let progs = g.out(&m, Rel::LowersTo);
    assert_eq!(progs.len(), 2);
    assert!(progs.iter().all(|(_, t)| *t == Tier::Compiled));
    // The primitives are read from the programs, not asserted.
    let p1 = progs[1].0;
    let prims: Vec<&str> = g.out(p1, Rel::UsesPrimitive).iter().map(|x| x.0).collect();
    let real = &programs(&w.table, BILLABLE_HOURS)[1];
    assert!(prims.contains(&"prim:MaskedSumI32"));
    assert_eq!(
        prims.len(),
        {
            let mut n: Vec<String> = real
                .ops
                .iter()
                .map(lance_graph_glove_parity::awareness::op_name)
                .collect();
            n.push("MaskedSumI32".into());
            n.sort();
            n.dedup();
            n.len()
        },
        "one primitive node per distinct op kind of the real program"
    );
    // The claim behind `duration` is a derived node; with SAP's real claim
    // it is hypothesized.
    assert_eq!(
        g.out(&duration, Rel::DependsOnClaim),
        vec![("claim:hours_logged:exact", Tier::Derived)]
    );
    // Provenance names the declaring file, revision, seal and reading.
    let e = g
        .edges()
        .iter()
        .find(|e| e.rel == Rel::LowersTo && g.nodes()[e.src as usize].ident == m)
        .unwrap();
    let p = g.provenance(e);
    assert_eq!((p.source.as_str(), p.rev.as_str()), (SOURCE, REV));
    assert_eq!((p.seal, p.reading), (seal(), READING));
}

/// Query A. FAILS IF: the dependency query does not return exactly the
/// methods whose compiled programs contain the primitive, or answers for a
/// primitive no program uses.
#[test]
fn dependency_query_runs_through_quack_and_matches_the_programs() {
    let b = batch(65);
    let w = world(&b, &with_duration(), &specs());
    // Oracle: the programs themselves.
    let using = |terminal: &str| -> Vec<String> {
        let mut v: Vec<String> = specs()
            .iter()
            .filter(|(_, s)| {
                w.table.get(s.key).unwrap().as_ref().is_ok_and(|p| {
                    p.programs().iter().any(|q| {
                        lance_graph_glove_parity::awareness::terminal_name(&q.terminal) == terminal
                    })
                })
            })
            .map(|(_, s)| method_ident(s.key))
            .collect();
        v.sort();
        v
    };
    for prim in ["MaskedSumI32", "Count", "GroupSumI32"] {
        let mut got: Vec<String> = w
            .graph
            .methods_using(prim)
            .unwrap()
            .into_iter()
            .map(str::to_owned)
            .collect();
        got.sort();
        assert_eq!(got, using(prim), "{prim}");
        assert!(!got.is_empty(), "{prim}: non-vacuous");
    }
    assert_eq!(
        w.graph.methods_using("MaskedSumI32").unwrap(),
        vec![hours_ident().as_str()]
    );
    // Silence twin: a primitive no program uses is not a node, so it does
    // not bind (nothing is minted).
    assert!(w.graph.methods_using("CountKeyRunsU32").is_err());
}

/// Query B, the claim behind a refusal. FAILS IF: under SAP's real claims
/// `billable_hours` is not recorded as refused, or the field it needs does not
/// lead to the hypothesized claim that blocks it.
#[test]
fn a_hypothesized_claim_refuses_the_method_and_the_graph_says_why() {
    let b = batch(65);
    let w = world(&b, SAP_FIELDS, &specs());
    let refused = w.graph.refused_for("UnmappedRequired").unwrap();
    assert!(refused.contains(&hours_ident().as_str()));
    let duration = field_ident(BILLABLE_WORK_ENTRY, seal(), "duration");
    assert_eq!(
        w.graph.out(&duration, Rel::DependsOnClaim),
        vec![("claim:hours_logged:hypothesized", Tier::Derived)]
    );
    // count_billable reads only `billable` and is not refused.
    assert!(!refused.contains(&method_ident(COUNT_BILLABLE).as_str()));
    assert!(w.graph.out(&hours_ident(), Rel::LowersTo).is_empty());
}

/// Falsifier 1. FAILS IF: two concepts sharing a field label share a field
/// node, or a spec of another concept is not refused by the compiler.
#[test]
fn the_same_field_label_in_two_classes_is_two_fields() {
    let b = batch(65);
    let mut s = specs();
    s.push((
        "document_billable",
        MethodSpec {
            concept: DOCUMENT,
            ..pure(
                MethodKey([0xC0; 16]),
                query(&[BILLABLE_TRUE], &[Fold::Count]),
            )
        },
    ));
    let w = world(&b, &with_duration(), &s);
    let a = field_ident(BILLABLE_WORK_ENTRY, seal(), "billable");
    let d = field_ident(DOCUMENT, seal(), "billable");
    assert_ne!(a, d);
    assert!(w.graph.node(&a).is_some() && w.graph.node(&d).is_some());
    assert_eq!(
        w.graph.refused_for("WrongConcept").unwrap(),
        vec![method_ident(MethodKey([0xC0; 16])).as_str()]
    );
    // Its condition is not the same node as count_billable's.
    let same = w.graph.out(&method_ident(COUNT_BILLABLE), Rel::Filters);
    let other = w
        .graph
        .out(&method_ident(MethodKey([0xC0; 16])), Rel::Filters);
    assert_ne!(same, other);
}

/// Falsifier 2. FAILS IF: a cyclic compute DAG is not recorded as the
/// contract's refusal, or an acyclic one is.
#[test]
fn a_compute_cycle_is_the_contracts_refusal() {
    let basis = canonical_fields();
    let cyclic = [
        ComputeEdge {
            target: 2,
            inputs: &[3],
        },
        ComputeEdge {
            target: 3,
            inputs: &[2],
        },
    ];
    let acyclic = [ComputeEdge {
        target: 2,
        inputs: &[3],
    }];
    for (edges, refused) in [(&cyclic[..], true), (&acyclic[..], false)] {
        let mut g = SoftwareGraph::new();
        let p = g.note(SOURCE, REV);
        g.project_compute(BILLABLE_WORK_ENTRY, seal(), &basis, edges, p);
        let got = g.refused_for("ComputeCycle").map(|v| v.len()).unwrap_or(0);
        assert_eq!(got == 1, refused, "{edges:?}");
        let projected = g
            .edges()
            .iter()
            .filter(|e| e.rel == Rel::ComputesFrom)
            .count();
        assert_eq!(
            projected,
            edges.iter().map(|e| e.inputs.len()).sum::<usize>()
        );
    }
}

/// Falsifier 3. FAILS IF: the same fold under another reading is merged with
/// it, or the compiler does not refuse the foreign reading.
#[test]
fn identical_folds_under_two_readings_stay_apart() {
    let b = batch(65);
    let mut s = specs();
    s.push((
        "billable_hours_r2",
        MethodSpec {
            reading: ReadingId(2),
            ..pure(
                MethodKey([0xC1; 16]),
                query(&[BILLABLE_TRUE], &[Fold::Count, Fold::Sum("duration")]),
            )
        },
    ));
    let w = world(&b, &with_duration(), &s);
    let r1 = w.graph.out(&hours_ident(), Rel::Folds);
    let r2 = w
        .graph
        .out(&method_ident(MethodKey([0xC1; 16])), Rel::Folds);
    assert_eq!(r1.len(), 2);
    assert!(r1.iter().all(|x| !r2.contains(x)));
    assert!(w
        .graph
        .refused_for("ReadingMismatch")
        .unwrap()
        .contains(&method_ident(MethodKey([0xC1; 16])).as_str()));
    // The shared-predicate query does not pair them.
    for c in w.graph.shared_predicates() {
        assert!(!c.methods.contains(&method_ident(MethodKey([0xC1; 16]))));
    }
}

fn run(t: &MethodTable, b: &CatsBatch, key: MethodKey) -> Answer {
    let n = b.len();
    let live = live_words(n);
    let lanes = b.lanes();
    let planes = Planes {
        n_rows: n,
        masks: &[live.as_slice()],
        lanes: &lanes,
    };
    t.call(key, &planes, UNIVERSE, Frontend::Quack).unwrap()
}

/// Falsifier 4. FAILS IF: a certificate issued for one body or schema is
/// accepted after either changed.
#[test]
fn a_certificate_goes_stale_when_the_body_or_seal_changes() {
    let b = batch(4097);
    let w = world(&b, &with_duration(), &specs());
    // The oracle actually runs: row loop over the indicator text.
    let LaneRef::I32(hours) = b.lanes()[HOURS] else {
        panic!("hours is i32")
    };
    let rows = (0..b.len()).filter(|&i| billable_code(indicator(i)) == 1);
    let (n, s) = rows.fold((0i64, 0i64), |(n, s), i| (n + 1, s + i64::from(hours[i])));
    assert_eq!(
        run(&w.table, &b, BILLABLE_HOURS),
        Answer::Values(vec![FoldValue::Scalar(n), FoldValue::Scalar(s)])
    );
    let cert = Certificate {
        method: BILLABLE_HOURS,
        digest: w.graph.digest(BILLABLE_HOURS).unwrap(),
        seal: seal(),
        test: "awareness.rs::a_certificate_goes_stale…".into(),
    };
    let mut g = w.graph.clone();
    let p = g.note(SOURCE, REV);
    g.certify(&cert, p).unwrap();
    assert_eq!(g.out(&hours_ident(), Rel::VerifiedBy).len(), 1);

    // The body changes: the literal is now `false`.
    let mut changed = specs();
    changed[3].1.body = query(
        &[Cond::Eq {
            field: "billable",
            literal: "false",
        }],
        &[Fold::Count, Fold::Sum("duration")],
    );
    let w2 = world(&b, &with_duration(), &changed);
    let mut g2 = w2.graph;
    assert_eq!(
        g2.certify(&cert, p),
        Err(CertError::StaleBody {
            method: BILLABLE_HOURS
        })
    );
    // The seal changes, the body text does not.
    let mut g3 = w.graph.clone();
    let stale = Certificate {
        seal: SchemaSeal(seal().0 ^ 1),
        ..cert.clone()
    };
    assert_eq!(
        g3.certify(&stale, p),
        Err(CertError::StaleSeal {
            method: BILLABLE_HOURS
        })
    );
}

/// Falsifier 5. FAILS IF: a bag count and a set count over the same edges
/// are treated as one primitive. Two Quack queries over one filter: the
/// number of traversers (`COUNT(*)`) and the number of distinct targets
/// (`COUNT(DISTINCT dst)`, ordered).
#[test]
fn a_bag_count_and_a_set_count_are_different_primitives() {
    let filter = Filter::and([Filter::plane(Mask(0)), Filter::cmp(Col(0), Cmp::EqU32(1))]);
    let bag = lower(&Query {
        filter: filter.clone(),
        agg: Agg::Count,
    })
    .unwrap();
    let set = lower(&Query {
        filter,
        agg: Agg::CountDistinctOrderedU32 { key: Col(1) },
    })
    .unwrap();
    let mut g = SoftwareGraph::new();
    let p = g.note(SOURCE, REV);
    let bag_m = g.add_method("method:bag");
    g.add_program(bag_m, "program:bag", &bag, p);
    let set_m = g.add_method("method:set");
    g.add_program(set_m, "program:set", &set, p);
    let bag_t: Vec<_> = g.out("program:bag", Rel::UsesPrimitive);
    let set_t: Vec<_> = g.out("program:set", Rel::UsesPrimitive);
    assert_ne!(bag_t, set_t);
    assert_eq!(g.methods_using("Count").unwrap(), vec!["method:bag"]);
    // The ops before the terminal are identical: a label-level reading
    // ("both count over the same filter") would merge them.
    assert_eq!(bag.ops, set.ops);
}

/// H1 and falsifier 6. FAILS IF: `billable_hours`'s sum program is not a
/// fusion candidate, or a candidate whose program ANDs a validity plane
/// claims its NULL obligation discharged.
#[test]
fn the_fusion_candidate_keeps_its_null_obligation_open_when_a_validity_plane_gates() {
    let b = batch(65);
    let w = world(&b, &with_duration(), &specs());
    let progs = programs(&w.table, BILLABLE_HOURS);
    let id = |i| format!("program:{}#{i}", "b2".repeat(16));
    let inputs: Vec<(String, &Program)> = vec![(id(0), &progs[0]), (id(1), &progs[1])];
    let c = w.graph.fusion_candidates(&inputs);
    assert_eq!(c.len(), 1, "only the MaskedSumI32 program");
    assert_eq!(
        c[0].kind,
        CandidateKind::FusePredicateIntoSum { program: id(1) }
    );
    assert_eq!(c[0].methods, vec![hours_ident()]);
    let null = |c: &lance_graph_glove_parity::awareness::Candidate| {
        c.obligations
            .iter()
            .find(|o| o.what.starts_with("NULL"))
            .unwrap()
            .discharged
            .clone()
    };
    assert!(null(&c[0]).is_some());
    assert!(c[0].obligations.iter().any(|o| o.discharged.is_none()));
    // The same sum, bound three-valued over a nullable lane (validity plane 1).
    let nullable = Query {
        filter: Filter::and([Filter::plane(Mask(0)), Filter::cmp(Col(0), Cmp::EqU32(1))])
            .sql_where(&[(Col(0), Mask(1))]),
        agg: Agg::SumI32(Col(2)),
    };
    let p = lower(&nullable).unwrap();
    let c = w
        .graph
        .fusion_candidates(&[("program:nullable".into(), &p)]);
    assert_eq!(c.len(), 1);
    assert_eq!(null(&c[0]), None);
}

/// Falsifier 7. FAILS IF: a transformation faster on one workload and slower
/// on another is reported as an improvement. FIXTURE numbers: they test the
/// rule, they are not measurements.
#[test]
fn a_mixed_result_is_not_an_improvement() {
    let faster = Workload {
        before_ns: 600.0,
        after_ns: 366.0,
    };
    let slower = Workload {
        before_ns: 56.0,
        after_ns: 70.0,
    };
    assert_eq!(verdict(&[faster, slower]), Verdict::Mixed);
    assert_eq!(verdict(&[faster, faster]), Verdict::Improves);
    assert_eq!(verdict(&[slower]), Verdict::Regresses);
    let tie = Workload {
        before_ns: 1.0,
        after_ns: 1.0,
    };
    assert_eq!(verdict(&[faster, tie]), Verdict::Mixed);
    assert_eq!(verdict(&[]), Verdict::Unmeasured);
}

/// Falsifier 8. FAILS IF: a primitive claimed for a method that none of its
/// compiled programs contains goes unreported, or a true claim is reported.
#[test]
fn a_claimed_primitive_must_be_in_the_lowered_program() {
    let b = batch(65);
    let mut w = world(&b, &with_duration(), &specs());
    let p = w.graph.note(SOURCE, REV);
    let m = hours_ident();
    w.graph
        .claim(
            &m,
            Rel::UsesPrimitive,
            "prim:MaskedSumI32",
            Tier::Declared,
            p,
        )
        .unwrap();
    assert!(w.graph.primitive_claim_mismatches().is_empty());
    w.graph
        .claim(
            &m,
            Rel::UsesPrimitive,
            "prim:GroupSumI32",
            Tier::Declared,
            p,
        )
        .unwrap();
    let bad = w.graph.primitive_claim_mismatches();
    assert_eq!(bad.len(), 1);
    assert_eq!(
        (bad[0].method.as_str(), bad[0].primitive.as_str()),
        (m.as_str(), "prim:GroupSumI32")
    );
}

/// FIXTURE: the NARS evidence glove of `tests/polyfill.rs`, reduced to what
/// preparation reads.
const NARS: u16 = 0xFFF0;
const EVIDENCE: &str = "evidence";

struct EvidenceBinder;

impl Binder for EvidenceBinder {
    fn table(&self, name: &str) -> Option<TableId> {
        (name == EVIDENCE).then_some(TableId(0))
    }
    fn live(&self, _: TableId) -> Mask {
        Mask(0)
    }
    fn field(&self, _: TableId, name: &str) -> Option<BoundField> {
        let (col, kind) = match name {
            "statement" => (0, FieldKind::Code),
            "stamp" => (1, FieldKind::Code),
            "independent" => (2, FieldKind::Code),
            "w" => (3, FieldKind::I32),
            "wf" => (4, FieldKind::I32),
            _ => return None,
        };
        Some(BoundField {
            col: Col(col),
            kind,
            validity: None,
        })
    }
    fn code(&self, _: TableId, col: Col, literal: &str) -> Option<u32> {
        (col.0 == 2)
            .then_some(match literal {
                "true" => Some(1),
                "false" => Some(0),
                _ => None,
            })
            .flatten()
    }
}

fn nars_table(
    filter: &'static [Cond<'static>],
) -> (MethodTable, Vec<(&'static str, MethodSpec<'static>)>) {
    let basis: Vec<String> = ["statement", "stamp", "independent", "w", "wf"]
        .map(String::from)
        .to_vec();
    let claims: Vec<FieldMap> = ["statement", "stamp", "independent", "w", "wf"]
        .map(|f| FieldMap {
            native: f,
            canonical: f,
            grade: Grade::Exact,
            note: "fixture",
        })
        .to_vec();
    const FOLDS: &[Fold] = &[
        Fold::GroupSum {
            key: "statement",
            val: "w",
            groups: 4,
        },
        Fold::GroupSum {
            key: "statement",
            val: "wf",
            groups: 4,
        },
    ];
    let specs = vec![(
        "revision",
        MethodSpec {
            key: MethodKey([0xEE; 16]),
            concept: NARS,
            schema: SchemaSeal::of(&basis),
            reading: READING,
            effect: EffectClass::Pure,
            body: MethodBody::Query {
                filter,
                folds: FOLDS,
                law: Law::NarsRevision {
                    independent: "independent",
                    scale: 1000,
                    horizon: 1,
                },
            },
        },
    )];
    let t = MethodTable::prepare(
        &Glove {
            concept: NARS,
            basis: &basis,
            claims: &claims,
            binder: &EvidenceBinder,
            table: EVIDENCE,
            reading: READING,
            universe: UNIVERSE,
        },
        &specs,
    );
    (t, specs)
}

/// Falsifier 9 and query C. FAILS IF: a capability is shipped because a
/// method declares it, a gap is "demonstrated" without a compiler refusal,
/// or a primitive a program uses is not shipped.
#[test]
fn a_declared_need_never_ships_a_capability() {
    let b = batch(65);
    let mut w = world(&b, &with_duration(), &specs());
    let p = w.graph.note(
        ".claude/board/entries/2026-10-12-class-method-polyfill-ledger.md",
        REV,
    );
    let g = &mut w.graph;
    assert_eq!(g.capability("prim:MaskedSumI32"), CapStatus::Shipped);
    // Hypotheses from other domains: recorded, never promoted.
    g.claim(
        "method:iam_reconcile",
        Rel::MayRequire,
        "cap:FirstOfRun",
        Tier::Hypothesis,
        p,
    )
    .unwrap();
    g.claim(
        "method:sap_import",
        Rel::MayRequire,
        "cap:FirstOfRun",
        Tier::Hypothesis,
        p,
    )
    .unwrap();
    // Even a method that says it USES it does not make it shipped.
    g.claim(
        &hours_ident(),
        Rel::UsesPrimitive,
        "cap:FirstOfRun",
        Tier::Declared,
        p,
    )
    .unwrap();
    assert!(matches!(g.capability("cap:FirstOfRun"), CapStatus::Claimed { by } if by.len() == 3));
    // An identity whose prefix names no kind is refused, not guessed.
    assert_eq!(
        g.claim(
            "method:x",
            Rel::MayRequire,
            "FirstOfRun",
            Tier::Hypothesis,
            p
        ),
        Err("FirstOfRun".into())
    );
    // A gap cannot be demonstrated by a refusal nobody compiled: not by an
    // absent one, and not by one a person only noted.
    assert_eq!(
        g.blocked_by("refusal:EvidenceNotIndependent", "cap:FirstOfRun", p),
        Err("refusal:EvidenceNotIndependent".into())
    );
    g.claim(
        "method:a_note",
        Rel::RefusedFor,
        "refusal:EvidenceNotIndependent",
        Tier::Hypothesis,
        p,
    )
    .unwrap();
    assert_eq!(
        g.blocked_by("refusal:EvidenceNotIndependent", "cap:FirstOfRun", p),
        Err("refusal:EvidenceNotIndependent".into())
    );
    // Revision without the payload's independence mark: the compiler refuses
    // it. FirstOfRun is what would compute that mark.
    let (t, s) = nars_table(&[]);
    g.project(&t, &s, &[], SOURCE, REV);
    g.blocked_by("refusal:EvidenceNotIndependent", "cap:FirstOfRun", p)
        .unwrap();
    assert_eq!(
        g.capability("cap:FirstOfRun"),
        CapStatus::Demonstrated {
            refusals: vec!["refusal:EvidenceNotIndependent".into()]
        }
    );
    // Silence twin: with the mark in the filter, revision prepares and no
    // such refusal exists to demonstrate anything.
    const INDEP: &[Cond] = &[Cond::Eq {
        field: "independent",
        literal: "true",
    }];
    let (t, s) = nars_table(INDEP);
    let mut clean = SoftwareGraph::new();
    clean.project(&t, &s, &[], SOURCE, REV);
    let q = clean.note(SOURCE, REV);
    assert!(clean
        .blocked_by("refusal:EvidenceNotIndependent", "cap:FirstOfRun", q)
        .is_err());
    assert_eq!(
        clean.methods_using("GroupSumI32").unwrap(),
        vec![method_ident(MethodKey([0xEE; 16])).as_str()]
    );
}

/// Falsifier 10. FAILS IF: a cost measured on one CPU tier is reported for
/// another.
#[test]
fn a_measurement_answers_only_for_its_own_tier() {
    let mut g = SoftwareGraph::new();
    let v3 = CpuTier {
        target: "x86-64-v3".into(),
        avx512f: true,
    };
    let v4 = CpuTier {
        target: "x86-64-v4".into(),
        avx512f: true,
    };
    let prog = format!("program:{}#1", "b2".repeat(16));
    // The #1504 ledger's recorded median (`method_cost.rs`, 1,048,576 rows).
    g.measure(Measurement {
        program: prog.clone(),
        rows: 1 << 20,
        tier: v3.clone(),
        median_ns: 600_000.0,
    });
    assert_eq!(g.cost(&prog, 1 << 20, &v3), Ok(600_000.0));
    assert_eq!(
        g.cost(&prog, 1 << 20, &v4),
        Err(CostError::TierMismatch { measured: v3 })
    );
    assert_eq!(g.cost(&prog, 1 << 16, &v4), Err(CostError::Unmeasured));
}

/// Query D. FAILS IF: the shared condition is not found across the methods
/// that carry it, or a method's per-fold rescans are not reported.
#[test]
fn repeated_structure_is_found_by_identity() {
    let b = batch(65);
    let w = world(&b, &with_duration(), &specs());
    let shared = w.graph.shared_predicates();
    let billable = field_ident(BILLABLE_WORK_ENTRY, seal(), "billable");
    let want = Candidate {
        kind: CandidateKind::SharedPredicate {
            conds: vec![format!("cond:{billable}=true@r1")],
        },
        methods: vec![method_ident(COUNT_BILLABLE), hours_ident()],
        obligations: shared[0].obligations.clone(),
    };
    assert_eq!(shared, vec![want]);
    // `!= true` is another condition, not a repeat of `= true`.
    assert!(!shared[0].methods.contains(&method_ident(NOT_BILLABLE)));
    // billable_hours scans its selection once per fold.
    let sel = w.graph.shared_selections(&w.table);
    assert_eq!(sel.len(), 1);
    assert_eq!(sel[0].methods, vec![hours_ident()]);
    assert!(sel[0].obligations.iter().all(|o| o.discharged.is_none()));
}

/// The MDL experiment (plan §5). FAILS IF: factoring a condition set shared
/// by two methods does not shorten the description when the set has two
/// conditions, shortens it when it has one, or a label-only reading counts
/// a cross-class repeat that typed identities do not.
#[test]
fn description_length_rewards_only_typed_multi_condition_reuse() {
    let s = |v: &[&str]| v.iter().map(|x| (*x).to_owned()).collect::<Vec<_>>();
    // One condition, shared by two methods (count_billable, billable_hours).
    let one = [
        s(&["c:billable=true", "f:count", "l:values"]),
        s(&["c:billable=true", "f:count", "f:sum", "l:values"]),
    ];
    let base = description_length(&one, &[]);
    let factored = description_length(&one, &[s(&["c:billable=true"])]);
    assert!(factored.total() >= base.total(), "{factored:?} vs {base:?}");
    // Two conditions, shared by three methods.
    let two = [
        s(&[
            "c:billable=true",
            "c:status=approved",
            "f:count",
            "l:values",
        ]),
        s(&["c:billable=true", "c:status=approved", "f:sum", "l:values"]),
        s(&[
            "c:billable=true",
            "c:status=approved",
            "f:group_sum",
            "l:values",
        ]),
    ];
    let base = description_length(&two, &[]);
    let factored = description_length(&two, &[s(&["c:billable=true", "c:status=approved"])]);
    assert!(factored.total() < base.total(), "{factored:?} vs {base:?}");
    // Counterexample: the same LABELS in two classes. Read by label, they
    // repeat and factoring "saves" bits; read by typed identity, they do not
    // repeat and factoring saves nothing.
    // The case above, read twice: by label (it repeats) and by identity,
    // where each method's conditions belong to another class (no repeat).
    let by_label = two.clone();
    let by_identity = [
        s(&[
            "c:0x0103/billable=true",
            "c:0x0103/status=approved",
            "f:count",
            "l:values",
        ]),
        s(&[
            "c:0x0201/billable=true",
            "c:0x0201/status=approved",
            "f:sum",
            "l:values",
        ]),
        s(&[
            "c:0x0202/billable=true",
            "c:0x0202/status=approved",
            "f:group_sum",
            "l:values",
        ]),
    ];
    let label_gain = description_length(&by_label, &[]).total()
        - description_length(&by_label, &[s(&["c:billable=true", "c:status=approved"])]).total();
    let typed_gain = description_length(&by_identity, &[]).total()
        - description_length(
            &by_identity,
            &[s(&["c:0x0103/billable=true", "c:0x0103/status=approved"])],
        )
        .total();
    assert!(label_gain > 0.0, "{label_gain}");
    assert!(typed_gain <= 0.0, "{typed_gain}");
}

/// The graph's own bodies feed the MDL measure by identity. FAILS IF the
/// declared tokens of a projected method are not its condition, fold and law
/// identities.
#[test]
fn the_graph_supplies_typed_bodies() {
    let b = batch(65);
    let w = world(&b, &with_duration(), &specs());
    let bodies = w.graph.bodies();
    let (_, hours) = bodies.iter().find(|(m, _)| *m == hours_ident()).unwrap();
    assert_eq!(hours.len(), 4, "{hours:?}");
    assert!(hours.iter().any(|t| t.starts_with("cond:field:0x0103/")));
    assert!(hours.contains(&"law:values".to_owned()));
}

/// D-SAG-1-5 (G1 merged). FAILS IF: a method reference is not a declared
/// CALLS edge to the method the table resolves, the callers of a predicate
/// are not found through Quack, or a reference cycle is not the compiler's
/// `MethodCycle` recorded on the cycle.
#[test]
fn method_references_are_calls_edges_and_cycles_are_the_compilers() {
    const IS_BILLABLE: MethodKey = MethodKey([0xA1; 16]);
    const HOURS: MethodKey = MethodKey([0xA2; 16]);
    const A: MethodKey = MethodKey([0xA3; 16]);
    const B: MethodKey = MethodKey([0xA4; 16]);
    const USES: &[Cond] = &[Cond::Method(MethodRef::Label("is_billable"))];
    let specs = vec![
        (
            "is_billable",
            pure(
                IS_BILLABLE,
                MethodBody::Select {
                    filter: &[BILLABLE_TRUE],
                },
            ),
        ),
        ("hours", pure(HOURS, query(USES, &[Fold::Sum("duration")]))),
        (
            "a",
            pure(
                A,
                MethodBody::Select {
                    filter: &[Cond::Method(MethodRef::Key(B))],
                },
            ),
        ),
        (
            "b",
            pure(
                B,
                MethodBody::Select {
                    filter: &[Cond::Method(MethodRef::Key(A))],
                },
            ),
        ),
    ];
    let b = batch(65);
    let w = world(&b, &with_duration(), &specs);
    assert_eq!(
        w.graph.out(&method_ident(HOURS), Rel::Calls),
        vec![(method_ident(IS_BILLABLE).as_str(), Tier::Declared)]
    );
    // Who calls is_billable: one Quack program over the edge table.
    let callers = w
        .graph
        .hop_to_ctx(
            &lance_graph_quack::bind::table("edges")
                .where_eq("rel", "calls")
                .where_eq("dst", method_ident(IS_BILLABLE).as_str()),
        )
        .unwrap();
    assert_eq!(callers, vec![method_ident(HOURS).as_str()]);
    // The caller lowered to a program reading the predicate's field.
    assert_eq!(
        w.graph.methods_using("MaskedSumI32").unwrap(),
        vec![method_ident(HOURS).as_str()]
    );
    // The cycle: the graph records the compiler's verdict, it does not
    // detect cycles itself.
    let mut cyc = w.graph.refused_for("MethodCycle").unwrap();
    cyc.sort_unstable();
    let mut want = [method_ident(A), method_ident(B)];
    want.sort();
    assert_eq!(cyc, want.iter().map(String::as_str).collect::<Vec<_>>());
    assert_eq!(w.graph.out(&method_ident(A), Rel::Calls).len(), 1);
}
