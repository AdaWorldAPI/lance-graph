//! D-CMH-1: the class-qualified method as a compiler seam.
//!
//! `SELECT <concept>::<method>(<row>) FROM <table>` resolves through OGAR's
//! codebook to a concept id and a method key, binds once against the glove,
//! and runs as Quack algebra. Its answer is checked against the same query
//! written directly in Quack, against R2IL's procedures, and against a
//! scalar oracle. Then every way the binding could lie is refused by name,
//! and NAL revision is compiled from the same algebra plus one certified
//! law.
//!
//! Every method is declared in this file. The method keys, the reading id,
//! the population id and the NARS concept are FIXTURES.
use lance_graph_glove_parity::basis::{canonical_fields, FieldMap, Grade, SAP_FIELDS};
use lance_graph_glove_parity::call::{resolve_call, CallParseError, MethodCall};
use lance_graph_glove_parity::method::{
    Answer, CallError, Cond, EffectClass, Fold, FoldValue, Frontend, Glove, Law, MethodBody,
    MethodKey, MethodSpec, MethodTable, PrepareError, ReadingId, SchemaSeal, Truth, Universe,
};
use lance_graph_mask_risc::{execute_into, Foreign, LaneRef, Out, Planes, Scratch, Value};
use lance_graph_ogar::ogar_vocab::class_ids::BILLABLE_WORK_ENTRY;
use lance_graph_quack::bind::{table, Binder, BoundField, FieldKind, TableId};
use lance_graph_quack::{lower, Agg, Col, Mask, Query};
use lance_graph_r2il::{procs, FoldDialect, Val};
use lance_graph_sap::bind::{billable_code, CatsBatch, BILLING, HOURS};
use lance_graph_sap::binder::{live_words, CatsBinder, LIVE, TABLE};
use lance_graph_sap::schema::{CatsSchema, FIELD_COUNT};

const READING: ReadingId = ReadingId(1);
const UNIVERSE: Universe = Universe(42);
const BILLABLE_HOURS: MethodKey = MethodKey([0xB2; 16]);

const BILLABLE_TRUE: Cond = Cond::Eq {
    field: "billable",
    literal: "true",
};

fn seal() -> SchemaSeal {
    SchemaSeal::of(&canonical_fields())
}

fn billable_hours() -> MethodSpec<'static> {
    MethodSpec {
        key: BILLABLE_HOURS,
        concept: BILLABLE_WORK_ENTRY,
        schema: seal(),
        reading: READING,
        effect: EffectClass::Pure,
        body: MethodBody::Query {
            filter: &[BILLABLE_TRUE],
            folds: &[Fold::Sum("duration")],
            law: Law::Values,
        },
    }
}

// --- the SAP fixture (as in tests/method.rs) ---------------------------------

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

/// FIXTURE: SAP's real claim on `duration` is `Hypothesized`; this table
/// promotes it so the method has an `I32` quantity to sum.
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

fn hours(batch: &CatsBatch) -> Vec<i32> {
    let LaneRef::I32(h) = batch.lanes()[HOURS] else {
        panic!("hours is an I32 lane")
    };
    h.to_vec()
}

// --- the seam ---------------------------------------------------------------

/// FAILS IF: the class-qualified call stops agreeing with the same query
/// written directly in Quack, with R2IL's procedures, or with a row loop.
#[test]
fn a_class_qualified_call_agrees_with_direct_quack_r2il_and_the_oracle() {
    let sql = "SELECT billable_work_entry::billable_hours(cats) FROM cats";
    for n in [1, 64, 65, 4097] {
        let batch = batch(n);
        let cats = CatsBinder::new(&batch);
        let claims = with_duration();
        let basis = canonical_fields();
        let t = MethodTable::prepare(
            &glove(&basis, &cats, &claims),
            &[("billable_hours", billable_hours())],
        );
        let call = resolve_call(sql, &t).unwrap();
        assert_eq!(
            call,
            MethodCall {
                concept: BILLABLE_WORK_ENTRY,
                method: BILLABLE_HOURS
            }
        );

        let live = live_words(n);
        let lanes = batch.lanes();
        let planes = Planes {
            n_rows: n,
            masks: &[live.as_slice()],
            lanes: &lanes,
        };

        // 1. The oracle: a row loop over the indicator text and the hours.
        let h = hours(&batch);
        let oracle: i64 = (0..n)
            .filter(|&i| billable_code(indicator(i)) == 1)
            .map(|i| i64::from(h[i]))
            .sum();

        // 2. The same query written by hand in Quack, bound from names.
        let tid = cats.table(TABLE).unwrap();
        let hours_col = cats.field(tid, "hours_logged").unwrap().col;
        let by_name = table(TABLE)
            .where_eq("billable", "true")
            .count()
            .bind(&cats)
            .unwrap();
        let direct = Query {
            filter: by_name.filter,
            agg: Agg::SumI32(hours_col),
        };
        let p = lower(&direct).unwrap();
        let mut scratch = Scratch::for_program(&p, n).unwrap();
        let Value::SumI64(direct) =
            execute_into(&p, &planes, &Foreign::NONE, &mut scratch, Out::None).unwrap()
        else {
            panic!()
        };

        // 3. R2IL's procedures, called by hand.
        let billable = cats.field(tid, "billable").unwrap().col;
        let mut d = FoldDialect::new(&planes, &Foreign::NONE);
        let mut stack = Vec::new();
        procs::select_u32(&mut d, &mut stack, LIVE.0, &[(billable.0, 1, false)]).unwrap();
        procs::sum(&mut d, &mut stack, hours_col.0).unwrap();
        let [Val::Scalar(r2il)] = stack.as_slice() else {
            panic!()
        };

        let want = Answer::Values(vec![FoldValue::Scalar(oracle)]);
        assert_eq!(direct, oracle, "direct quack n={n}");
        assert_eq!(*r2il, oracle, "r2il procs n={n}");
        for f in [Frontend::Quack, Frontend::R2il] {
            assert_eq!(
                t.call(call.method, &planes, UNIVERSE, f).unwrap(),
                want,
                "method {f:?} n={n}"
            );
        }
    }
}

/// The call text names a concept through OGAR's codebook, a method through
/// the table's labels, and a row; each part that does not resolve is named.
#[test]
fn a_call_that_does_not_resolve_is_refused_by_name() {
    let batch = batch(2);
    let cats = CatsBinder::new(&batch);
    let claims = with_duration();
    let basis = canonical_fields();
    let t = MethodTable::prepare(
        &glove(&basis, &cats, &claims),
        &[("billable_hours", billable_hours())],
    );
    let cases = [
        (
            "SELECT nars::revision(cats) FROM cats",
            CallParseError::UnknownConcept("nars".into()),
        ),
        (
            "SELECT project::billable_hours(cats) FROM cats",
            CallParseError::WrongConcept {
                called: lance_graph_ogar::ogar_vocab::class_ids::PROJECT,
                table: BILLABLE_WORK_ENTRY,
            },
        ),
        (
            "SELECT billable_work_entry::frobnicate(cats) FROM cats",
            CallParseError::UnknownMethod("frobnicate".into()),
        ),
        (
            "SELECT billable_work_entry::billable_hours(orders) FROM cats",
            CallParseError::PayloadMismatch,
        ),
        (
            "SELECT billable_work_entry::billable_hours(orders) FROM orders",
            CallParseError::WrongTable("orders".into()),
        ),
        (
            "SELECT billable_work_entry::billable_hours(cats) FROM cats WHERE x = 1",
            CallParseError::Unrecognised,
        ),
        (
            "SELECT billable_work_entry::billable_hours(cats); DROP TABLE cats FROM cats",
            CallParseError::Unrecognised,
        ),
    ];
    for (sql, want) in cases {
        assert_eq!(resolve_call(sql, &t), Err(want), "{sql}");
    }
}

// --- refusals ---------------------------------------------------------------

/// FAILS IF: a spec written for one basis binds a reordered one. Swapping two
/// canonical positions keeps every name, so only the seal can tell.
#[test]
fn a_spec_for_another_schema_or_reading_is_refused() {
    let batch = batch(2);
    let cats = CatsBinder::new(&batch);
    let claims = with_duration();
    let basis = canonical_fields();
    let mut swapped = basis.clone();
    swapped.swap(0, 1);
    assert_ne!(SchemaSeal::of(&swapped), seal());
    let t = MethodTable::prepare(&glove(&swapped, &cats, &claims), &[("h", billable_hours())]);
    assert!(matches!(
        t.get(BILLABLE_HOURS),
        Some(Err(PrepareError::SchemaMismatch { .. }))
    ));

    let other = MethodSpec {
        reading: ReadingId(2),
        ..billable_hours()
    };
    let t = MethodTable::prepare(&glove(&basis, &cats, &claims), &[("h", other)]);
    assert_eq!(
        t.get(BILLABLE_HOURS),
        Some(&Err(PrepareError::ReadingMismatch {
            spec: ReadingId(2),
            glove: READING
        }))
    );

    // Twin: the unswapped basis and matching reading prepare.
    let t = MethodTable::prepare(&glove(&basis, &cats, &claims), &[("h", billable_hours())]);
    assert!(t.get(BILLABLE_HOURS).unwrap().is_ok());
}

/// FAILS IF: planes of another population run under a method prepared for
/// this one.
#[test]
fn planes_of_another_population_are_refused() {
    let batch = batch(65);
    let cats = CatsBinder::new(&batch);
    let claims = with_duration();
    let basis = canonical_fields();
    let t = MethodTable::prepare(&glove(&basis, &cats, &claims), &[("h", billable_hours())]);
    let live = live_words(65);
    let lanes = batch.lanes();
    let planes = Planes {
        n_rows: 65,
        masks: &[live.as_slice()],
        lanes: &lanes,
    };
    for f in [Frontend::Quack, Frontend::R2il] {
        assert!(matches!(
            t.call(BILLABLE_HOURS, &planes, Universe(43), f),
            Err(CallError::UniverseMismatch {
                method: UNIVERSE,
                planes: Universe(43)
            })
        ));
        assert!(t.call(BILLABLE_HOURS, &planes, UNIVERSE, f).is_ok());
    }
}

/// A binder that declares `billable` nullable.
struct NullableBillable<'a>(CatsBinder<'a>);

impl Binder for NullableBillable<'_> {
    fn table(&self, name: &str) -> Option<TableId> {
        self.0.table(name)
    }
    fn live(&self, t: TableId) -> Mask {
        self.0.live(t)
    }
    fn field(&self, t: TableId, name: &str) -> Option<BoundField> {
        let mut f = self.0.field(t, name)?;
        if name == "billable" {
            f.validity = Some(Mask(7));
        }
        Some(f)
    }
    fn code(&self, t: TableId, col: Col, literal: &str) -> Option<u32> {
        self.0.code(t, col, literal)
    }
}

/// FAILS IF: a nullable field binds into a two-valued condition (NULL would
/// read as whatever the lane holds there).
#[test]
fn a_nullable_field_is_refused() {
    let batch = batch(2);
    let nb = NullableBillable(CatsBinder::new(&batch));
    let claims = with_duration();
    let basis = canonical_fields();
    let t = MethodTable::prepare(&glove(&basis, &nb, &claims), &[("h", billable_hours())]);
    assert_eq!(
        t.get(BILLABLE_HOURS),
        Some(&Err(PrepareError::Nullable("billable".into())))
    );
}

/// FAILS IF: a basis wider than a `WideFieldMask` position can address is
/// truncated instead of refused.
#[test]
fn a_basis_past_256_positions_is_refused() {
    let batch = batch(2);
    let cats = CatsBinder::new(&batch);
    let claims = with_duration();
    let mut wide = canonical_fields();
    // Pad with names nothing claims, until `billable` sits past position 255
    // only in the second case.
    while wide.len() < 257 {
        wide.push(format!("pad_{}", wide.len()));
    }
    let spec = MethodSpec {
        schema: SchemaSeal::of(&wide),
        ..billable_hours()
    };
    let t = MethodTable::prepare(&glove(&wide, &cats, &claims), &[("h", spec)]);
    assert_eq!(
        t.get(BILLABLE_HOURS),
        Some(&Err(PrepareError::BasisTooWide(257)))
    );
    // Twin: exactly 256 positions prepare.
    wide.pop();
    let spec = MethodSpec {
        schema: SchemaSeal::of(&wide),
        ..billable_hours()
    };
    let t = MethodTable::prepare(&glove(&wide, &cats, &claims), &[("h", spec)]);
    assert!(t.get(BILLABLE_HOURS).unwrap().is_ok());
}

// --- NAL revision -----------------------------------------------------------

/// FIXTURE: no NARS concept is minted in OGAR's codebook yet.
const NARS: u16 = 0xFFF0;
const REVISION: MethodKey = MethodKey([0xEE; 16]);
const RAW_SUMS: MethodKey = MethodKey([0xEF; 16]);
const SCALE: u32 = 1000;
const GROUPS: u32 = 4;

const EVIDENCE: &str = "evidence";
const STATEMENT: usize = 0;
const STAMP: usize = 1;
const INDEPENDENT: usize = 2;
const W: usize = 3;
const WF: usize = 4;

fn nars_basis() -> Vec<String> {
    ["statement", "stamp", "independent", "w", "wf"]
        .map(String::from)
        .to_vec()
}

const NARS_CLAIMS: [FieldMap; 5] = [
    FieldMap {
        native: "statement",
        canonical: "statement",
        grade: Grade::Exact,
        note: "fixture",
    },
    FieldMap {
        native: "stamp",
        canonical: "stamp",
        grade: Grade::Exact,
        note: "fixture",
    },
    FieldMap {
        native: "independent",
        canonical: "independent",
        grade: Grade::Exact,
        note: "fixture",
    },
    FieldMap {
        native: "w",
        canonical: "w",
        grade: Grade::Exact,
        note: "fixture",
    },
    FieldMap {
        native: "wf",
        canonical: "wf",
        grade: Grade::Exact,
        note: "fixture",
    },
];

/// The evidence table's resolution membrane: five non-nullable lanes.
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
            "statement" => (STATEMENT, FieldKind::Code),
            "stamp" => (STAMP, FieldKind::Code),
            "independent" => (INDEPENDENT, FieldKind::Code),
            "w" => (W, FieldKind::I32),
            "wf" => (WF, FieldKind::I32),
            _ => return None,
        };
        Some(BoundField {
            col: Col(col as u16),
            kind,
            validity: None,
        })
    }
    fn code(&self, _: TableId, col: Col, literal: &str) -> Option<u32> {
        if usize::from(col.0) != INDEPENDENT {
            return None;
        }
        match literal {
            "true" => Some(1),
            "false" => Some(0),
            _ => None,
        }
    }
}

const REVISION_FOLDS: &[Fold] = &[
    Fold::GroupSum {
        key: "statement",
        val: "w",
        groups: GROUPS,
    },
    Fold::GroupSum {
        key: "statement",
        val: "wf",
        groups: GROUPS,
    },
];

const REVISION_LAW: Law = Law::NarsRevision {
    independent: "independent",
    scale: SCALE,
    horizon: 1,
};

fn nars_spec(
    key: MethodKey,
    filter: &'static [Cond],
    folds: &'static [Fold],
    law: Law<'static>,
) -> MethodSpec<'static> {
    MethodSpec {
        key,
        concept: NARS,
        schema: SchemaSeal::of(&nars_basis()),
        reading: READING,
        effect: EffectClass::Pure,
        body: MethodBody::Query { filter, folds, law },
    }
}

fn nars_specs() -> Vec<(&'static str, MethodSpec<'static>)> {
    const INDEP: &[Cond] = &[Cond::Eq {
        field: "independent",
        literal: "true",
    }];
    vec![
        (
            "revision",
            nars_spec(REVISION, INDEP, REVISION_FOLDS, REVISION_LAW),
        ),
        // The same sums without the independence condition, as plain values:
        // what revision would read if duplicates were not excluded.
        (
            "raw_sums",
            nars_spec(RAW_SUMS, &[], REVISION_FOLDS, Law::Values),
        ),
    ]
}

/// One piece of evidence: `(statement, stamp, f, c)`.
type Evidence = (u32, u32, f64, f64);

/// The OpenNARS reference revisions (`lance-graph-planner`
/// `tests/nal_revision_oracle.rs`), as evidence per statement, plus one
/// REPEATED stamp on statement 0. Statement 3 has no evidence.
const ROWS: [Evidence; 7] = [
    (0, 10, 1.0, 0.9),
    (0, 11, 0.1, 0.6),
    (1, 20, 0.8, 0.5),
    (1, 21, 0.3, 0.7),
    (2, 30, 0.0, 0.9),
    (2, 31, 1.0, 0.9),
    (0, 10, 1.0, 0.9), // the same stamp again
];

/// OpenNARS's revised truths, to two places.
const REFERENCE: [(f64, f64); 3] = [(0.87, 0.91), (0.45, 0.77), (0.50, 0.95)];

struct Lanes {
    statement: Vec<u32>,
    stamp: Vec<u32>,
    independent: Vec<u32>,
    w: Vec<i32>,
    wf: Vec<i32>,
}

/// The payload's producer: fixed-point weights, and the first-occurrence
/// mark of each `(statement, stamp)` (what the algebra cannot compute yet).
fn produce(rows: &[Evidence]) -> Lanes {
    let mut seen = Vec::new();
    let mut l = Lanes {
        statement: vec![],
        stamp: vec![],
        independent: vec![],
        w: vec![],
        wf: vec![],
    };
    for &(s, stamp, f, c) in rows {
        let w = (f64::from(SCALE) * c / (1.0 - c)).round();
        l.statement.push(s);
        l.stamp.push(stamp);
        l.independent.push(u32::from(!seen.contains(&(s, stamp))));
        seen.push((s, stamp));
        l.w.push(w as i32);
        l.wf.push((w * f).round() as i32);
    }
    l
}

/// The independent oracle: revision as the NAL defines it, in floating
/// point, from `(f, c)` directly, over the independent evidence.
fn oracle(rows: &[Evidence]) -> Vec<Option<Truth>> {
    let mut seen = Vec::new();
    let mut acc = vec![(0.0f64, 0.0f64); GROUPS as usize];
    for &(s, stamp, f, c) in rows {
        if seen.contains(&(s, stamp)) {
            continue;
        }
        seen.push((s, stamp));
        let w = c / (1.0 - c);
        acc[s as usize].0 += w;
        acc[s as usize].1 += w * f;
    }
    acc.iter()
        .map(|&(w, wf)| {
            (w > 0.0).then(|| Truth {
                f: wf / w,
                c: w / (w + 1.0),
            })
        })
        .collect()
}

fn nars_table() -> MethodTable {
    let basis = nars_basis();
    MethodTable::prepare(
        &Glove {
            concept: NARS,
            basis: &basis,
            claims: &NARS_CLAIMS,
            binder: &EvidenceBinder,
            table: EVIDENCE,
            reading: READING,
            universe: UNIVERSE,
        },
        &nars_specs(),
    )
}

fn run_nars(t: &MethodTable, key: MethodKey, l: &Lanes, f: Frontend) -> Answer {
    let n = l.w.len();
    let live = live_words(n);
    let lanes = [
        LaneRef::U32(&l.statement),
        LaneRef::U32(&l.stamp),
        LaneRef::U32(&l.independent),
        LaneRef::I32(&l.w),
        LaneRef::I32(&l.wf),
    ];
    let planes = Planes {
        n_rows: n,
        masks: &[live.as_slice()],
        lanes: &lanes,
    };
    t.call(key, &planes, UNIVERSE, f).unwrap()
}

/// FAILS IF: revision compiled from two grouped sums and the certified law
/// disagrees with the floating-point oracle or with OpenNARS, on either
/// frontend; or counts the repeated stamp.
#[test]
fn nars_revision_compiles_from_grouped_sums_and_one_law() {
    let t = nars_table();
    let lanes = produce(&ROWS);
    let want = oracle(&ROWS);
    for f in [Frontend::Quack, Frontend::R2il] {
        let Answer::Truths(got) = run_nars(&t, REVISION, &lanes, f) else {
            panic!("revision answers truths")
        };
        assert_eq!(got.len(), GROUPS as usize);
        for (s, (g, o)) in got.iter().zip(&want).enumerate() {
            match (g, o) {
                (Some(g), Some(o)) => {
                    // Fixed point at scale 1000 moves the third place at most.
                    assert!((g.f - o.f).abs() < 1e-3, "{f:?} s{s} f {g:?} vs {o:?}");
                    assert!((g.c - o.c).abs() < 1e-3, "{f:?} s{s} c {g:?} vs {o:?}");
                }
                (None, None) => {}
                _ => panic!("{f:?} s{s}: {g:?} vs {o:?}"),
            }
        }
        for (s, &(rf, rc)) in REFERENCE.iter().enumerate() {
            let g = got[s].unwrap();
            assert!(
                (g.f - rf).abs() < 0.005 && (g.c - rc).abs() < 0.005,
                "{f:?} s{s} {g:?}"
            );
        }
        assert_eq!(got[3], None, "no evidence, no truth");
    }
}

/// The counterexample for the missing algebra: the same two sums without the
/// independence condition count the repeated stamp, and the confidence that
/// would follow is not NAL revision.
#[test]
fn a_repeated_stamp_inflates_the_raw_sums() {
    let t = nars_table();
    let lanes = produce(&ROWS);
    let Answer::Values(v) = run_nars(&t, RAW_SUMS, &lanes, Frontend::Quack) else {
        panic!()
    };
    let [FoldValue::Groups(w), _] = v.as_slice() else {
        panic!()
    };
    // Statement 0: 9000 + 1500 counted, plus 9000 again.
    assert_eq!(w[0], 19_500);
    let weight = w[0] as f64 / f64::from(SCALE);
    let inflated = weight / (weight + 1.0);
    assert!((inflated - 0.951).abs() < 1e-3);
    assert!(
        (inflated - REFERENCE[0].1).abs() > 0.03,
        "the repeat must matter"
    );
}

/// FAILS IF: a revision body that does not select independent evidence, or
/// whose folds are not the law's shape, prepares.
#[test]
fn a_revision_body_without_independence_or_in_the_wrong_shape_is_refused() {
    const MIXED_KEYS: &[Fold] = &[
        Fold::GroupSum {
            key: "statement",
            val: "w",
            groups: GROUPS,
        },
        Fold::GroupSum {
            key: "stamp",
            val: "wf",
            groups: GROUPS,
        },
    ];
    const INDEP: &[Cond] = &[Cond::Eq {
        field: "independent",
        literal: "true",
    }];
    let basis = nars_basis();
    let specs = [
        (
            "no_indep",
            nars_spec(MethodKey([1; 16]), &[], REVISION_FOLDS, REVISION_LAW),
        ),
        (
            "mixed",
            nars_spec(MethodKey([2; 16]), INDEP, MIXED_KEYS, REVISION_LAW),
        ),
        (
            "one_fold",
            nars_spec(
                MethodKey([3; 16]),
                INDEP,
                &REVISION_FOLDS[..1],
                REVISION_LAW,
            ),
        ),
    ];
    let t = MethodTable::prepare(
        &Glove {
            concept: NARS,
            basis: &basis,
            claims: &NARS_CLAIMS,
            binder: &EvidenceBinder,
            table: EVIDENCE,
            reading: READING,
            universe: UNIVERSE,
        },
        &specs,
    );
    assert_eq!(
        t.get(MethodKey([1; 16])),
        Some(&Err(PrepareError::EvidenceNotIndependent(
            "independent".into()
        )))
    );
    assert_eq!(
        t.get(MethodKey([2; 16])),
        Some(&Err(PrepareError::LawShape))
    );
    assert_eq!(
        t.get(MethodKey([3; 16])),
        Some(&Err(PrepareError::LawShape))
    );
}

/// KNOWN GAP, pinned: a statement at or past the declared domain is dropped
/// by the grouped-sum terminal, not refused. A method cannot see it.
#[test]
fn evidence_outside_the_declared_domain_is_dropped_silently() {
    let t = nars_table();
    let mut rows = ROWS.to_vec();
    rows.push((GROUPS + 1, 99, 1.0, 0.9));
    let Answer::Truths(got) = run_nars(&t, REVISION, &produce(&rows), Frontend::Quack) else {
        panic!()
    };
    let Answer::Truths(base) = run_nars(&t, REVISION, &produce(&ROWS), Frontend::Quack) else {
        panic!()
    };
    assert_eq!(got, base, "the out-of-domain row left no trace");
}

// --- Experiment C: replay, layers, retraction --------------------------------

/// One delta as it arrives: `(delta_id, layer, statement, stamp, f, c)`.
/// Replay identity (`delta_id`) and evidential identity (`stamp`) are
/// different things: a replayed delta repeats a delta; two deltas can carry
/// the same stamp (one observation, reported in two hypothesis layers).
type Delta = (u32, u8, u32, u32, f64, f64);

/// The payload producer: drops a replayed `delta_id` first, then marks the
/// first occurrence of each `(statement, stamp)` across ALL layers as
/// independent. Both marks are the producer's, because the algebra cannot
/// compute a first occurrence yet (ledger row 1).
fn produce_deltas(deltas: &[Delta]) -> Lanes {
    let mut seen_delta = Vec::new();
    let rows: Vec<Evidence> = deltas
        .iter()
        .filter(|d| {
            let fresh = !seen_delta.contains(&d.0);
            seen_delta.push(d.0);
            fresh
        })
        .map(|&(_, _, s, stamp, f, c)| (s, stamp, f, c))
        .collect();
    produce(&rows)
}

/// FAILS IF: a replayed delta, or the same stamp arriving in a second
/// hypothesis layer, changes the revised truth.
#[test]
fn replay_and_cross_layer_repeats_do_not_change_revision() {
    let t = nars_table();
    let base: Vec<Delta> = ROWS[..6]
        .iter()
        .enumerate()
        .map(|(i, &(s, stamp, f, c))| (i as u32, 0, s, stamp, f, c))
        .collect();
    let mut noisy = base.clone();
    noisy.push(base[0]); // the same delta, replayed
    noisy.push(base[3]); // replayed again, later
    noisy.push((100, 1, 0, 11, 0.1, 0.6)); // stamp 11, seen in layer 1 too
    noisy.push((101, 1, 2, 30, 0.0, 0.9)); // stamp 30, layer 1
    for f in [Frontend::Quack, Frontend::R2il] {
        let a = run_nars(&t, REVISION, &produce_deltas(&base), f);
        let b = run_nars(&t, REVISION, &produce_deltas(&noisy), f);
        assert_eq!(a, b, "{f:?}");
    }
    // Non-vacuity: without the marks, the noisy payload inflates the sums.
    let raw = |d: &[Delta]| {
        let rows: Vec<Evidence> = d.iter().map(|&(_, _, s, st, f, c)| (s, st, f, c)).collect();
        let Answer::Values(v) = run_nars(&t, RAW_SUMS, &produce(&rows), Frontend::Quack) else {
            panic!()
        };
        v
    };
    assert_ne!(raw(&base), raw(&noisy));
}

/// TYPED INSUFFICIENCY, pinned: retraction is not declared by
/// `Law::NarsRevision`. Writing it as a negative-weight delta is legal
/// arithmetic in the sums, and wrong whenever the retracted stamp was never
/// asserted: the sums cannot see which stamps were asserted. Retraction needs
/// a semijoin of retraction stamps onto asserted stamps (ledger).
#[test]
fn retraction_by_negative_weight_is_not_revision() {
    let t = nars_table();
    // Retract stamp 11 from statement 0 (it was asserted): exact.
    let mut lanes = produce(&ROWS[..6]);
    let retract_11 = (0, 11, 0.1, 0.6);
    let w = (f64::from(SCALE) * 0.6 / 0.4).round();
    lanes.statement.push(retract_11.0);
    lanes.stamp.push(retract_11.1);
    lanes.independent.push(1);
    lanes.w.push(-(w as i32));
    lanes.wf.push(-((w * 0.1).round() as i32));
    let Answer::Truths(got) = run_nars(&t, REVISION, &lanes, Frontend::Quack) else {
        panic!()
    };
    let only_10 = oracle(&[(0, 10, 1.0, 0.9)]);
    assert!((got[0].unwrap().c - only_10[0].unwrap().c).abs() < 1e-3);

    // Retract stamp 99 (never asserted): the sums accept it and answer.
    let mut lanes = produce(&ROWS[..6]);
    lanes.statement.push(0);
    lanes.stamp.push(99);
    lanes.independent.push(1);
    lanes.w.push(-(w as i32));
    lanes.wf.push(-((w * 0.1).round() as i32));
    let Answer::Truths(got) = run_nars(&t, REVISION, &lanes, Frontend::Quack) else {
        panic!()
    };
    let unchanged = oracle(&ROWS[..6])[0].unwrap();
    assert!(
        (got[0].unwrap().c - unchanged.c).abs() > 0.01,
        "a phantom retraction changed the truth, and nothing refused it"
    );
}

// --- review fixes: ambiguity and law parameters are refused ------------------

/// FAILS IF: a repeated method key or label is resolved by slice order
/// instead of refused.
#[test]
fn a_repeated_key_or_label_is_refused() {
    let batch = batch(2);
    let cats = CatsBinder::new(&batch);
    let claims = with_duration();
    let basis = canonical_fields();
    let other_key = MethodSpec {
        key: MethodKey([0xC1; 16]),
        ..billable_hours()
    };
    let t = MethodTable::prepare(
        &glove(&basis, &cats, &claims),
        &[
            ("h", billable_hours()),
            ("h2", billable_hours()), // same key, another label
            ("dup", other_key),
            (
                "dup",
                MethodSpec {
                    key: MethodKey([0xC2; 16]),
                    ..billable_hours()
                },
            ),
        ],
    );
    assert_eq!(
        t.get(BILLABLE_HOURS),
        Some(&Err(PrepareError::DuplicateKey(BILLABLE_HOURS)))
    );
    assert_eq!(t.resolve("dup"), None, "an ambiguous label names nothing");
    assert_eq!(
        t.get(MethodKey([0xC1; 16])),
        Some(&Err(PrepareError::DuplicateLabel("dup".into())))
    );
    // Twin: a unique key and label still prepare.
    let t = MethodTable::prepare(&glove(&basis, &cats, &claims), &[("h", billable_hours())]);
    assert!(t.get(BILLABLE_HOURS).unwrap().is_ok());
    assert_eq!(t.resolve("h"), Some(BILLABLE_HOURS));
}

/// FAILS IF: a revision law with a zero scale or zero horizon prepares (a
/// zero scale yields NaN, a zero horizon a confidence of exactly 1).
#[test]
fn a_degenerate_revision_law_is_refused() {
    const INDEP: &[Cond] = &[Cond::Eq {
        field: "independent",
        literal: "true",
    }];
    let law = |scale, horizon| Law::NarsRevision {
        independent: "independent",
        scale,
        horizon,
    };
    let basis = nars_basis();
    let specs = [
        (
            "zero_scale",
            nars_spec(MethodKey([4; 16]), INDEP, REVISION_FOLDS, law(0, 1)),
        ),
        (
            "zero_horizon",
            nars_spec(MethodKey([5; 16]), INDEP, REVISION_FOLDS, law(SCALE, 0)),
        ),
        (
            "ok",
            nars_spec(MethodKey([6; 16]), INDEP, REVISION_FOLDS, law(SCALE, 1)),
        ),
    ];
    let t = MethodTable::prepare(
        &Glove {
            concept: NARS,
            basis: &basis,
            claims: &NARS_CLAIMS,
            binder: &EvidenceBinder,
            table: EVIDENCE,
            reading: READING,
            universe: UNIVERSE,
        },
        &specs,
    );
    assert_eq!(
        t.get(MethodKey([4; 16])),
        Some(&Err(PrepareError::LawParameter))
    );
    assert_eq!(
        t.get(MethodKey([5; 16])),
        Some(&Err(PrepareError::LawParameter))
    );
    assert!(t.get(MethodKey([6; 16])).unwrap().is_ok());
}

/// FAILS IF: two proven claims for one canonical field are resolved by
/// slice order. Reordering the claims must not change which lane binds.
#[test]
fn an_ambiguous_canonical_claim_is_refused() {
    let batch = batch(2);
    let cats = CatsBinder::new(&batch);
    let basis = canonical_fields();
    let mut claims = with_duration();
    // A second proven source for `duration`, from another native field.
    claims.push(FieldMap {
        native: "work_day",
        canonical: "duration",
        grade: Grade::Exact,
        note: "fixture: a conflicting claim",
    });
    let t = MethodTable::prepare(&glove(&basis, &cats, &claims), &[("h", billable_hours())]);
    assert_eq!(
        t.get(BILLABLE_HOURS),
        Some(&Err(PrepareError::AmbiguousClaim("duration".into())))
    );
    // Twin: the same claim repeated for the same native field is not
    // ambiguous.
    let mut claims = with_duration();
    let dup = *claims.iter().find(|c| c.canonical == "duration").unwrap();
    claims.push(dup);
    let t = MethodTable::prepare(&glove(&basis, &cats, &claims), &[("h", billable_hours())]);
    assert!(t.get(BILLABLE_HOURS).unwrap().is_ok());
}
