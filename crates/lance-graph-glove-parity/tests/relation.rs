//! D-CMH-G1-4a: a functional relation carried by the glove.
//!
//! `out(rel)` over a foreign key keeps the anchor (#1311's anchor rule): each
//! anchor row still stands for one traverser, and the target's fields are
//! read through the key. Gremlin drops a traverser whose edge leads nowhere,
//! so `out(rel)` is "the key names a LIVE row of the target": a dangling key
//! and a key to a dead row both fail it.
//!
//! FIXTURE: an ERP-shaped pair (`line -billed_to-> partner`) with synthetic
//! concept ids. Every answer is checked against a row loop that reads the
//! TEXT of each field, never a bound lane or a code.
use lance_graph_glove_parity::basis::{FieldMap, Grade};
use lance_graph_glove_parity::gremlin::{compose, GremlinRefusal, Pred, Step};
use lance_graph_glove_parity::method::{
    Answer, CallError, Cond, EffectClass, Fold, FoldValue, Frontend, Glove, Law, MethodBody,
    MethodKey, MethodRef, MethodSpec, MethodTable, PrepareError, Reach, ReadingId, Relation,
    SchemaSeal, Universe,
};
use lance_graph_mask_risc::{words_for, Foreign, ForeignPlane, LaneRef, Planes};
use lance_graph_quack::bind::{Binder, BoundField, FieldKind, TableId};
use lance_graph_quack::{
    lower, Agg, Cmp, Col, Filter, ForeignLane, ForeignPlane as QPlane, Mask, Query,
};

/// FIXTURE concept ids.
const LINE: u16 = 0xFFE1;
const PARTNER: u16 = 0xFFE2;
const READING: ReadingId = ReadingId(1);
const LINE_U: Universe = Universe(7);
const PARTNER_U: Universe = Universe(8);
const PARTNERS: usize = 37;
const SIZES: [usize; 4] = [1, 64, 65, 4097];

const STATUS: [&str; 3] = ["open", "posted", "void"];
const COUNTRY: [&str; 3] = ["DE", "FR", "US"];

fn status(i: usize) -> &'static str {
    STATUS[(i * 5 + i / 7) % 3]
}
fn amount(i: usize) -> i32 {
    ((i * 37) % 201) as i32 - 100
}
/// Some keys dangle: `PARTNERS..PARTNERS + 4` names no partner.
fn partner_of(i: usize) -> usize {
    (i * 11 + 3) % (PARTNERS + 4)
}
fn line_live(i: usize) -> bool {
    i % 13 != 5
}
fn country(p: usize) -> &'static str {
    COUNTRY[(p * 3 + p / 4) % 3]
}
fn partner_live(p: usize) -> bool {
    p % 7 != 2
}

/// The binder of both tables. Codes are the text's index in its domain.
struct Erp;

impl Binder for Erp {
    fn table(&self, name: &str) -> Option<TableId> {
        match name {
            "line" => Some(TableId(0)),
            "partner" => Some(TableId(1)),
            _ => None,
        }
    }
    fn live(&self, _: TableId) -> Mask {
        Mask(0)
    }
    fn field(&self, t: TableId, name: &str) -> Option<BoundField> {
        let (col, kind) = match (t.0, name) {
            (0, "status") => (0, FieldKind::Code),
            (0, "amount") => (1, FieldKind::I32),
            (0, "partner") => (2, FieldKind::Code),
            (1, "country") => (0, FieldKind::Code),
            (1, "credit") => (1, FieldKind::I32),
            _ => return None,
        };
        Some(BoundField {
            col: Col(col),
            kind,
            validity: None,
        })
    }
    fn code(&self, t: TableId, col: Col, literal: &str) -> Option<u32> {
        let domain: &[&str] = match (t.0, col.0) {
            (0, 0) => &STATUS,
            (1, 0) => &COUNTRY,
            _ => return None,
        };
        domain.iter().position(|d| *d == literal).map(|p| p as u32)
    }
}

fn exact(fields: &[&'static str]) -> Vec<FieldMap> {
    fields
        .iter()
        .map(|f| FieldMap {
            native: f,
            canonical: f,
            grade: Grade::Exact,
            note: "fixture",
        })
        .collect()
}

fn names(fields: &[&str]) -> Vec<String> {
    fields.iter().map(|f| (*f).to_owned()).collect()
}

/// The planes of one size, owned.
struct Data {
    n: usize,
    line_live: Vec<u64>,
    status: Vec<u32>,
    amount: Vec<i32>,
    partner: Vec<u32>,
    partner_live: Vec<u64>,
    country: Vec<u32>,
    credit: Vec<i32>,
}

fn bits(n: usize, f: impl Fn(usize) -> bool) -> Vec<u64> {
    let mut w = vec![0u64; words_for(n)];
    for i in (0..n).filter(|&i| f(i)) {
        w[i / 64] |= 1 << (i % 64);
    }
    w
}

fn data(n: usize) -> Data {
    let code = |d: &[&str], s: &str| d.iter().position(|x| *x == s).unwrap() as u32;
    Data {
        n,
        line_live: bits(n, line_live),
        status: (0..n).map(|i| code(&STATUS, status(i))).collect(),
        amount: (0..n).map(amount).collect(),
        partner: (0..n).map(|i| partner_of(i) as u32).collect(),
        partner_live: bits(PARTNERS, partner_live),
        country: (0..PARTNERS).map(|p| code(&COUNTRY, country(p))).collect(),
        credit: (0..PARTNERS).map(|p| p as i32).collect(),
    }
}

impl Data {
    fn run(
        &self,
        table: &MethodTable,
        key: MethodKey,
        frontend: Frontend,
    ) -> Result<Answer, CallError> {
        self.run_as(table, key, frontend, &[(PARTNER, PARTNER_U)])
    }

    fn run_as(
        &self,
        table: &MethodTable,
        key: MethodKey,
        frontend: Frontend,
        universes: &[(u16, Universe)],
    ) -> Result<Answer, CallError> {
        let masks = [self.line_live.as_slice()];
        let lanes = [
            LaneRef::U32(&self.status),
            LaneRef::I32(&self.amount),
            LaneRef::U32(&self.partner),
        ];
        let planes = Planes {
            n_rows: self.n,
            masks: &masks,
            lanes: &lanes,
        };
        let fplanes = [ForeignPlane {
            words: &self.partner_live,
            rows: PARTNERS,
        }];
        let flanes = [LaneRef::U32(&self.country), LaneRef::I32(&self.credit)];
        let reach = Reach {
            foreign: Foreign {
                planes: &fplanes,
                lanes: &flanes,
            },
            universes,
        };
        table.call_with(key, &planes, LINE_U, &reach, frontend)
    }
}

/// Row loop over the text: rows that are live and pass `keep`.
fn oracle(n: usize, keep: impl Fn(usize) -> bool) -> (i64, i64) {
    let rows: Vec<usize> = (0..n).filter(|&i| line_live(i) && keep(i)).collect();
    (
        rows.len() as i64,
        rows.iter().map(|&i| i64::from(amount(i))).sum(),
    )
}

/// The target row a line reaches, if any: in range and live.
fn reached(i: usize) -> Option<usize> {
    let p = partner_of(i);
    (p < PARTNERS && partner_live(p)).then_some(p)
}

/// A row filter of the oracle.
type Keep = Box<dyn Fn(usize) -> bool>;

struct Fx {
    line_basis: Vec<String>,
    line_claims: Vec<FieldMap>,
    partner_basis: Vec<String>,
    partner_claims: Vec<FieldMap>,
}

impl Fx {
    fn new() -> Self {
        Self {
            line_basis: names(&["status", "amount", "partner"]),
            line_claims: exact(&["status", "amount", "partner"]),
            partner_basis: names(&["country", "credit"]),
            partner_claims: exact(&["country", "credit"]),
        }
    }

    fn spec<'b>(&self, key: u8, body: MethodBody<'b>) -> MethodSpec<'b> {
        MethodSpec {
            key: MethodKey([key; 16]),
            concept: LINE,
            schema: SchemaSeal::of(&self.line_basis),
            reading: READING,
            effect: EffectClass::Pure,
            body,
        }
    }

    /// Prepare `specs` against the line glove with the `billed_to` relation.
    fn table(&self, specs: &[(&str, MethodSpec<'_>)]) -> MethodTable {
        self.table_with(specs, &[("country", 0), ("credit", 1)], &self.line_claims)
    }

    fn table_with(
        &self,
        specs: &[(&str, MethodSpec<'_>)],
        lanes: &[(&str, u16)],
        line_claims: &[FieldMap],
    ) -> MethodTable {
        let partner = Glove {
            concept: PARTNER,
            basis: &self.partner_basis,
            claims: &self.partner_claims,
            binder: &Erp,
            table: "partner",
            reading: READING,
            universe: PARTNER_U,
            relations: &[],
        };
        let rels = [Relation {
            name: "billed_to",
            fk: "partner",
            target: &partner,
            plane: 0,
            lanes,
        }];
        let glove = Glove {
            concept: LINE,
            basis: &self.line_basis,
            claims: line_claims,
            binder: &Erp,
            table: "line",
            reading: READING,
            universe: LINE_U,
            relations: &rels,
        };
        MethodTable::prepare(&glove, specs)
    }
}

fn query(filter: &'static [Cond<'static>], folds: &'static [Fold<'static>]) -> MethodBody<'static> {
    MethodBody::Query {
        filter,
        folds,
        law: Law::Values,
    }
}

const COUNT_SUM: &[Fold] = &[Fold::Count, Fold::Sum("amount")];
const OPEN: Cond = Cond::Eq {
    field: "status",
    literal: "open",
};
const TO_DE: Cond = Cond::Via {
    rel: "billed_to",
    field: "country",
    literal: "DE",
    negate: false,
};
const NOT_DE: Cond = Cond::Via {
    rel: "billed_to",
    field: "country",
    literal: "DE",
    negate: true,
};
const REACHES: Cond = Cond::Reaches { rel: "billed_to" };

fn scalars(a: Answer) -> (i64, i64) {
    match a {
        Answer::Values(v) => match v.as_slice() {
            [FoldValue::Scalar(c), FoldValue::Scalar(s)] => (*c, *s),
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_condition_through_the_relation_matches_the_row_loop() {
    let fx = Fx::new();
    let specs = [
        ("open_to_de", fx.spec(1, query(&[OPEN, TO_DE], COUNT_SUM))),
        ("open_not_de", fx.spec(2, query(&[OPEN, NOT_DE], COUNT_SUM))),
        ("reaches", fx.spec(3, query(&[REACHES], COUNT_SUM))),
    ];
    let table = fx.table(&specs);
    for n in SIZES {
        let d = data(n);
        let open = |i: usize| status(i) == "open";
        let cases: [(u8, Keep); 3] = [
            (
                1,
                Box::new(move |i| open(i) && reached(i).is_some_and(|p| country(p) == "DE")),
            ),
            (
                2,
                Box::new(move |i| open(i) && reached(i).is_some_and(|p| country(p) != "DE")),
            ),
            (3, Box::new(|i| reached(i).is_some())),
        ];
        for (k, keep) in cases {
            let got = scalars(d.run(&table, MethodKey([k; 16]), Frontend::Quack).unwrap());
            assert_eq!(got, oracle(n, keep), "method {k}, n={n}");
        }
    }
    // Anti-vacuity at the largest size: each guard changes the answer.
    let n = 4097;
    let any = oracle(n, |_| true).0;
    let in_range = oracle(n, |i| partner_of(i) < PARTNERS).0;
    let reach = oracle(n, |i| reached(i).is_some()).0;
    assert!(
        reach < in_range && in_range < any,
        "{reach} {in_range} {any}"
    );
    let not_de = oracle(n, |i| reached(i).is_some_and(|p| country(p) != "DE")).0;
    let not_eq_de = oracle(n, |i| {
        let p = partner_of(i);
        !(p < PARTNERS && country(p) == "DE")
    })
    .0;
    assert!(
        not_de < not_eq_de,
        "!= through a relation must require a live target"
    );
}

#[test]
fn the_relation_lowers_to_the_direct_quack_query() {
    let fx = Fx::new();
    let specs = [("open_to_de", fx.spec(1, query(&[OPEN, TO_DE], COUNT_SUM)))];
    let table = fx.table(&specs);
    let m = table.get(MethodKey([1; 16])).unwrap().as_ref().unwrap();
    let filter = || {
        Filter::and([
            Filter::plane(Mask(0)),
            Filter::cmp(Col(0), Cmp::EqU32(0)),
            Filter::semijoin(Col(2), QPlane(0)),
            Filter::eq_u32_via(Col(2), ForeignLane(0), 0),
        ])
    };
    let direct = [
        Query {
            filter: filter(),
            agg: Agg::Count,
        },
        Query {
            filter: filter(),
            agg: Agg::SumI32(Col(1)),
        },
    ];
    assert_eq!(m.queries(), direct);
    let programs: Vec<_> = direct.iter().map(|q| lower(q).unwrap()).collect();
    assert_eq!(m.programs(), programs.as_slice());
    // The key is a field the method reads.
    assert!(m.required.has(2), "the fk is required");
}

#[test]
fn r2il_names_the_missing_gather() {
    let fx = Fx::new();
    let specs = [
        ("to_de", fx.spec(1, query(&[TO_DE], COUNT_SUM))),
        ("open", fx.spec(2, query(&[OPEN], COUNT_SUM))),
    ];
    let table = fx.table(&specs);
    let d = data(65);
    match d.run(&table, MethodKey([1; 16]), Frontend::R2il) {
        Err(CallError::FrontendGap {
            frontend: Frontend::R2il,
            missing: "GATHER",
        }) => {}
        other => panic!("{other:?}"),
    }
    // Silence twin: a method that reads no relation still runs on R2IL.
    let got = scalars(d.run(&table, MethodKey([2; 16]), Frontend::R2il).unwrap());
    assert_eq!(got, oracle(65, |i| status(i) == "open"));
}

#[test]
fn a_predicate_through_the_relation_is_reused_by_reference() {
    let fx = Fx::new();
    let specs = [
        (
            "billed_in_de",
            fx.spec(1, MethodBody::Select { filter: &[TO_DE] }),
        ),
        (
            "open_de",
            fx.spec(
                2,
                query(
                    &[OPEN, Cond::Method(MethodRef::Label("billed_in_de"))],
                    COUNT_SUM,
                ),
            ),
        ),
        ("direct", fx.spec(3, query(&[OPEN, TO_DE], COUNT_SUM))),
    ];
    let table = fx.table(&specs);
    let get = |k: u8| table.get(MethodKey([k; 16])).unwrap().as_ref().unwrap();
    assert_eq!(get(2).programs(), get(3).programs());
    let d = data(4097);
    assert_eq!(
        scalars(d.run(&table, MethodKey([2; 16]), Frontend::Quack).unwrap()),
        oracle(4097, |i| status(i) == "open"
            && reached(i).is_some_and(|p| country(p) == "DE"))
    );
}

#[test]
fn a_relation_that_cannot_bind_is_refused_by_name() {
    let fx = Fx::new();
    let refused = |filter: &'static [Cond<'static>]| {
        let specs = [("m", fx.spec(1, query(filter, COUNT_SUM)))];
        fx.table(&specs)
            .get(MethodKey([1; 16]))
            .unwrap()
            .clone()
            .unwrap_err()
    };
    assert_eq!(
        refused(&[Cond::Reaches { rel: "shipped_to" }]),
        PrepareError::UnknownRelation("shipped_to".into())
    );
    assert_eq!(
        refused(&[Cond::Via {
            rel: "billed_to",
            field: "vat_id",
            literal: "x",
            negate: false,
        }]),
        PrepareError::UnknownField("billed_to.vat_id".into())
    );
    assert_eq!(
        refused(&[Cond::Via {
            rel: "billed_to",
            field: "country",
            literal: "XX",
            negate: false,
        }]),
        PrepareError::UnknownValue {
            field: "billed_to.country".into(),
            literal: "XX".into()
        }
    );
    assert_eq!(
        refused(&[Cond::Via {
            rel: "billed_to",
            field: "credit",
            literal: "1",
            negate: false,
        }]),
        PrepareError::KindMismatch("billed_to.credit".into())
    );
    // A target field the relation supplies no lane for.
    let specs = [("m", fx.spec(1, query(&[TO_DE], COUNT_SUM)))];
    assert_eq!(
        fx.table_with(&specs, &[("credit", 1)], &fx.line_claims)
            .get(MethodKey([1; 16]))
            .unwrap()
            .clone()
            .unwrap_err(),
        PrepareError::NoForeignLane("billed_to.country".into())
    );
    // The key itself needs a proven claim like any field the method reads.
    let unproven: Vec<FieldMap> = fx
        .line_claims
        .iter()
        .map(|c| FieldMap {
            grade: if c.canonical == "partner" {
                Grade::Hypothesized
            } else {
                c.grade
            },
            ..*c
        })
        .collect();
    assert_eq!(
        fx.table_with(&specs, &[("country", 0)], &unproven)
            .get(MethodKey([1; 16]))
            .unwrap()
            .clone()
            .unwrap_err(),
        PrepareError::UnmappedRequired("partner".into())
    );
}

#[test]
fn the_target_population_is_checked_at_the_call() {
    let fx = Fx::new();
    let specs = [("to_de", fx.spec(1, query(&[TO_DE], COUNT_SUM)))];
    let table = fx.table(&specs);
    let d = data(65);
    let key = MethodKey([1; 16]);
    match d.run_as(&table, key, Frontend::Quack, &[]) {
        Err(CallError::ForeignUniverse {
            concept: PARTNER,
            method: PARTNER_U,
            planes: None,
        }) => {}
        other => panic!("{other:?}"),
    }
    match d.run_as(&table, key, Frontend::Quack, &[(PARTNER, Universe(99))]) {
        Err(CallError::ForeignUniverse {
            planes: Some(Universe(99)),
            ..
        }) => {}
        other => panic!("{other:?}"),
    }
    // The plain call supplies no foreign planes.
    let masks = [d.line_live.as_slice()];
    let lanes = [
        LaneRef::U32(&d.status),
        LaneRef::I32(&d.amount),
        LaneRef::U32(&d.partner),
    ];
    let planes = Planes {
        n_rows: d.n,
        masks: &masks,
        lanes: &lanes,
    };
    match table.call(key, &planes, LINE_U, Frontend::Quack) {
        Err(CallError::ForeignUniverse { planes: None, .. }) => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn out_composes_and_keeps_the_anchor() {
    let v = Step::V(LINE);
    let c = compose(
        LINE,
        &[
            v,
            Step::Out("billed_to"),
            Step::Has {
                field: "country",
                pred: Pred::Eq("DE"),
            },
            Step::Count,
        ],
    )
    .unwrap();
    assert_eq!(c.filter(), &[REACHES, TO_DE]);
    assert_eq!(c.folds(), &[Fold::Count]);
    let c = compose(
        LINE,
        &[
            v,
            Step::Out("billed_to"),
            Step::Has {
                field: "country",
                pred: Pred::Neq("DE"),
            },
        ],
    )
    .unwrap();
    assert_eq!(c.filter(), &[REACHES, NOT_DE]);
    let refuse = |steps: &[Step<'static>]| compose(LINE, steps).unwrap_err();
    let out = Step::Out("billed_to");
    assert_eq!(
        refuse(&[v, out, Step::Out("owned_by"), Step::Count]),
        GremlinRefusal::SecondHop("owned_by".into())
    );
    assert_eq!(
        refuse(&[v, out, Step::Values("credit"), Step::Sum]),
        GremlinRefusal::ValueThroughRelation("billed_to".into())
    );
    assert_eq!(
        refuse(&[v, out, Step::Dedup, Step::Count]),
        GremlinRefusal::DedupThroughRelation("billed_to".into())
    );
    assert_eq!(
        refuse(&[v, out, Step::Where(MethodRef::Label("m")), Step::Count]),
        GremlinRefusal::WhereThroughRelation("billed_to".into())
    );
    for hop in [Step::In("billed_to"), Step::Both("billed_to")] {
        assert_eq!(
            refuse(&[v, hop, Step::Count]),
            GremlinRefusal::FanOut("billed_to".into())
        );
    }
    // The composed body prepares and answers like the hand-written one.
    let fx = Fx::new();
    let composed = compose(
        LINE,
        &[
            v,
            Step::Out("billed_to"),
            Step::Has {
                field: "country",
                pred: Pred::Eq("DE"),
            },
            Step::Count,
        ],
    )
    .unwrap();
    let specs = [
        ("composed", fx.spec(1, composed.body())),
        (
            "direct",
            fx.spec(2, query(&[REACHES, TO_DE], &[Fold::Count])),
        ),
    ];
    let table = fx.table(&specs);
    let get = |k: u8| table.get(MethodKey([k; 16])).unwrap().as_ref().unwrap();
    assert_eq!(get(1).programs(), get(2).programs());
}
