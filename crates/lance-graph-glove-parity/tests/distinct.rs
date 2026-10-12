//! D-CMH-G1-4b: an edge table as the anchor, and the distinct targets of a
//! hop.
//!
//! An M:N relation is carried as its own concept: each edge row is one
//! traverser, with a functional relation to each end (`out(by)`,
//! `out(of)`). Both ends are filtered at the source, through the key, so a
//! hop never produces a mask another program reads.
//!
//! - bag: `V(reads)…out(of).count()` counts edge rows (traversers);
//! - set: `V(reads)…out(of).dedup().count()` counts distinct live targets:
//!   `COUNT(DISTINCT key)` over a lane stored in key order
//!   (`Terminal::CountKeyRunsU32`, two words of state). A lane in another
//!   order is refused at run time (`LaneNotOrdered`), never miscounted. A
//!   distinct count over the other end needs the edges stored in that end's
//!   order: a second projection of the same concept.
//!
//! FIXTURE: synthetic `method`, `field` and `reads` tables, with dangling
//! keys and dead rows on every table. Every answer is checked against a row
//! loop over the TEXT of each field.
use lance_graph_glove_parity::awareness::SoftwareGraph;
use lance_graph_glove_parity::basis::{FieldMap, Grade};
use lance_graph_glove_parity::decl::{parse_decls, MethodDecl};
use lance_graph_glove_parity::gremlin::{compose, GremlinRefusal, Pred, Step};
use lance_graph_glove_parity::method::{
    Answer, CallError, Cond, EffectClass, Fold, FoldValue, Frontend, Glove, Law, MethodBody,
    MethodKey, MethodRef, MethodSpec, MethodTable, Reach, ReadingId, Relation, SchemaSeal,
    Universe,
};
use lance_graph_mask_risc::{words_for, ExecError, Foreign, ForeignPlane, LaneRef, Planes};
use lance_graph_ogar::ogar_vocab::class_ids::DOCUMENT;
use lance_graph_quack::bind::{Binder, BoundField, FieldKind, TableId};
use lance_graph_quack::{Agg, Col, Filter, ForeignLane, ForeignPlane as QPlane, Mask, Query};

/// FIXTURE concept ids. The edge concept borrows a minted id so its methods
/// can be declared in text, which names concepts through the codebook.
const METHOD: u16 = 0xFFD1;
const FIELD: u16 = 0xFFD2;
const READS: u16 = DOCUMENT;
const READING: ReadingId = ReadingId(1);
const METHOD_U: Universe = Universe(11);
const FIELD_U: Universe = Universe(12);
const READS_U: Universe = Universe(13);
const METHODS: usize = 23;
const FIELDS: usize = 31;
const SIZES: [usize; 4] = [1, 64, 65, 4097];

const KINDS: [&str; 3] = ["query", "select", "action"];
const GRADES: [&str; 3] = ["exact", "converted", "hypothesized"];

fn kind(m: usize) -> &'static str {
    KINDS[(m * 5 + m / 3) % 3]
}
fn method_live(m: usize) -> bool {
    m % 9 != 4
}
fn grade(f: usize) -> &'static str {
    GRADES[(f * 7 + f / 5) % 3]
}
fn field_live(f: usize) -> bool {
    f % 8 != 6
}
fn edge_live(e: usize) -> bool {
    e % 11 != 3
}

/// Which table order a `reads` projection is stored in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Order {
    ByField,
    ByMethod,
}

/// Edge `e` of `n` as `(method, field)`. The ordering end's key never
/// decreases and runs past the table (dangling keys at the end); the other
/// end is scattered and also dangles.
fn edge(order: Order, n: usize, e: usize) -> (usize, usize) {
    match order {
        Order::ByField => ((e * 7 + e / 5) % (METHODS + 2), e * (FIELDS + 3) / n),
        Order::ByMethod => (e * (METHODS + 2) / n, (e * 13 + e / 4) % (FIELDS + 3)),
    }
}

/// The method an edge reaches: in range and live.
fn by(order: Order, n: usize, e: usize) -> Option<usize> {
    let m = edge(order, n, e).0;
    (m < METHODS && method_live(m)).then_some(m)
}
fn of(order: Order, n: usize, e: usize) -> Option<usize> {
    let f = edge(order, n, e).1;
    (f < FIELDS && field_live(f)).then_some(f)
}

/// The binder of all three tables.
struct Db;

impl Binder for Db {
    fn table(&self, name: &str) -> Option<TableId> {
        match name {
            "reads" => Some(TableId(0)),
            "method" => Some(TableId(1)),
            "field" => Some(TableId(2)),
            _ => None,
        }
    }
    fn live(&self, _: TableId) -> Mask {
        Mask(0)
    }
    fn field(&self, t: TableId, name: &str) -> Option<BoundField> {
        let col = match (t.0, name) {
            (0, "method") | (1, "kind") | (2, "grade") => 0,
            (0, "field") => 1,
            _ => return None,
        };
        Some(BoundField {
            col: Col(col),
            kind: FieldKind::Code,
            validity: None,
        })
    }
    fn code(&self, t: TableId, col: Col, literal: &str) -> Option<u32> {
        let domain: &[&str] = match (t.0, col.0) {
            (1, 0) => &KINDS,
            (2, 0) => &GRADES,
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

fn bits(n: usize, f: impl Fn(usize) -> bool) -> Vec<u64> {
    let mut w = vec![0u64; words_for(n)];
    for i in (0..n).filter(|&i| f(i)) {
        w[i / 64] |= 1 << (i % 64);
    }
    w
}

/// One size of one projection, owned.
struct Data {
    n: usize,
    live: Vec<u64>,
    method: Vec<u32>,
    field: Vec<u32>,
    method_live: Vec<u64>,
    kind: Vec<u32>,
    field_live: Vec<u64>,
    grade: Vec<u32>,
}

fn code(d: &[&str], s: &str) -> u32 {
    d.iter().position(|x| *x == s).unwrap() as u32
}

fn data(order: Order, n: usize) -> Data {
    Data {
        n,
        live: bits(n, edge_live),
        method: (0..n).map(|e| edge(order, n, e).0 as u32).collect(),
        field: (0..n).map(|e| edge(order, n, e).1 as u32).collect(),
        method_live: bits(METHODS, method_live),
        kind: (0..METHODS).map(|m| code(&KINDS, kind(m))).collect(),
        field_live: bits(FIELDS, field_live),
        grade: (0..FIELDS).map(|f| code(&GRADES, grade(f))).collect(),
    }
}

impl Data {
    fn run(&self, t: &MethodTable, key: u8, frontend: Frontend) -> Result<Answer, CallError> {
        let masks = [self.live.as_slice()];
        let lanes = [LaneRef::U32(&self.method), LaneRef::U32(&self.field)];
        let planes = Planes {
            n_rows: self.n,
            masks: &masks,
            lanes: &lanes,
        };
        let fplanes = [
            ForeignPlane {
                words: &self.method_live,
                rows: METHODS,
            },
            ForeignPlane {
                words: &self.field_live,
                rows: FIELDS,
            },
        ];
        let flanes = [LaneRef::U32(&self.kind), LaneRef::U32(&self.grade)];
        let reach = Reach {
            foreign: Foreign {
                planes: &fplanes,
                lanes: &flanes,
            },
            universes: &[(METHOD, METHOD_U), (FIELD, FIELD_U)],
        };
        t.call_with(MethodKey([key; 16]), &planes, READS_U, &reach, frontend)
    }

    fn scalar(&self, t: &MethodTable, key: u8) -> i64 {
        match self.run(t, key, Frontend::Quack).unwrap() {
            Answer::Values(v) => match v.as_slice() {
                [FoldValue::Scalar(c)] => *c,
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
    }
}

struct Fx {
    reads_basis: Vec<String>,
    reads_claims: Vec<FieldMap>,
    method_basis: Vec<String>,
    method_claims: Vec<FieldMap>,
    field_basis: Vec<String>,
    field_claims: Vec<FieldMap>,
}

impl Fx {
    fn new() -> Self {
        Self {
            reads_basis: names(&["method", "field"]),
            reads_claims: exact(&["method", "field"]),
            method_basis: names(&["kind"]),
            method_claims: exact(&["kind"]),
            field_basis: names(&["grade"]),
            field_claims: exact(&["grade"]),
        }
    }

    fn spec<'b>(&self, key: u8, body: MethodBody<'b>) -> MethodSpec<'b> {
        MethodSpec {
            key: MethodKey([key; 16]),
            concept: READS,
            schema: SchemaSeal::of(&self.reads_basis),
            reading: READING,
            effect: EffectClass::Pure,
            body,
        }
    }

    fn table(&self, specs: &[(&str, MethodSpec<'_>)]) -> MethodTable {
        self.table_with(specs, &[])
    }

    fn table_with(&self, specs: &[(&str, MethodSpec<'_>)], decls: &[MethodDecl]) -> MethodTable {
        let glove = |concept, basis, claims, table, universe| Glove {
            concept,
            basis,
            claims,
            binder: &Db,
            table,
            reading: READING,
            universe,
            relations: &[],
        };
        let method = glove(
            METHOD,
            &self.method_basis,
            &self.method_claims,
            "method",
            METHOD_U,
        );
        let field = glove(
            FIELD,
            &self.field_basis,
            &self.field_claims,
            "field",
            FIELD_U,
        );
        let rels = [
            Relation {
                name: "by",
                fk: "method",
                target: &method,
                plane: 0,
                lanes: &[("kind", 0)],
            },
            Relation {
                name: "of",
                fk: "field",
                target: &field,
                plane: 1,
                lanes: &[("grade", 1)],
            },
        ];
        let reads = Glove {
            relations: &rels,
            ..glove(
                READS,
                &self.reads_basis,
                &self.reads_claims,
                "reads",
                READS_U,
            )
        };
        MethodTable::prepare_with(&reads, specs, decls)
    }
}

fn query(filter: &'static [Cond<'static>], folds: &'static [Fold<'static>]) -> MethodBody<'static> {
    MethodBody::Query {
        filter,
        folds,
        law: Law::Values,
    }
}

const BY_QUERY: &[Cond] = &[
    Cond::Reaches { rel: "by" },
    Cond::Via {
        rel: "by",
        field: "kind",
        literal: "query",
        negate: false,
    },
];
const FROM_QUERIES: Cond = Cond::Method(MethodRef::Label("by_query"));
const OF: Cond = Cond::Reaches { rel: "of" };
const OF_EXACT: Cond = Cond::Via {
    rel: "of",
    field: "grade",
    literal: "exact",
    negate: false,
};
const DISTINCT_OF: &[Fold] = &[Fold::Distinct { rel: "of" }];
const DISTINCT_BY: &[Fold] = &[Fold::Distinct { rel: "by" }];

fn specs(fx: &Fx) -> Vec<(&'static str, MethodSpec<'static>)> {
    vec![
        (
            "by_query",
            fx.spec(1, MethodBody::Select { filter: BY_QUERY }),
        ),
        // bag: edges from query methods to live fields.
        (
            "bag",
            fx.spec(2, query(&[FROM_QUERIES, OF], &[Fold::Count])),
        ),
        // set: distinct live fields read by query methods.
        ("set", fx.spec(3, query(&[FROM_QUERIES, OF], DISTINCT_OF))),
        // set, with the target filtered at the source.
        ("exact", fx.spec(4, query(&[OF_EXACT], DISTINCT_OF))),
        // set, with no condition but the hop itself.
        ("all", fx.spec(5, query(&[], DISTINCT_OF))),
        // set over the other end.
        ("methods", fx.spec(6, query(&[OF_EXACT], DISTINCT_BY))),
    ]
}

/// Row-loop oracle: the selected edges of one projection.
fn edges(n: usize, keep: impl Fn(usize) -> bool) -> Vec<usize> {
    (0..n).filter(|&e| edge_live(e) && keep(e)).collect()
}

fn distinct(v: impl Iterator<Item = usize>) -> i64 {
    let mut v: Vec<usize> = v.collect();
    v.sort_unstable();
    v.dedup();
    v.len() as i64
}

/// FAILS IF: a bag or set count over an edge table, with conditions on
/// either end, disagrees with the row loop, or the two are not different
/// answers on this data.
#[test]
fn bag_and_set_counts_over_an_edge_table_match_the_row_loop() {
    let fx = Fx::new();
    let t = fx.table(&specs(&fx));
    let o = Order::ByField;
    let mut differ = false;
    for n in SIZES {
        let d = data(o, n);
        let from_query = |e: usize| by(o, n, e).is_some_and(|m| kind(m) == "query");
        let selected = edges(n, |e| from_query(e) && of(o, n, e).is_some());
        let bag = selected.len() as i64;
        let set = distinct(selected.iter().map(|&e| of(o, n, e).unwrap()));
        assert_eq!(d.scalar(&t, 2), bag, "bag, n={n}");
        assert_eq!(d.scalar(&t, 3), set, "set, n={n}");
        differ |= bag != set;
        let exact = edges(n, |e| of(o, n, e).is_some_and(|f| grade(f) == "exact"));
        assert_eq!(
            d.scalar(&t, 4),
            distinct(exact.iter().map(|&e| of(o, n, e).unwrap())),
            "exact, n={n}"
        );
        let all = edges(n, |e| of(o, n, e).is_some());
        assert_eq!(
            d.scalar(&t, 5),
            distinct(all.iter().map(|&e| of(o, n, e).unwrap())),
            "all, n={n}"
        );
    }
    assert!(differ, "the bag and the set must differ somewhere");
    // Anti-vacuity at the largest size: a dead or dangling target is not a
    // distinct target.
    let n = 4097;
    let in_range = distinct(
        edges(n, |e| edge(o, n, e).1 < FIELDS)
            .iter()
            .map(|&e| edge(o, n, e).1),
    );
    let live = distinct(
        edges(n, |e| of(o, n, e).is_some())
            .iter()
            .map(|&e| edge(o, n, e).1),
    );
    let any = distinct(edges(n, |_| true).iter().map(|&e| edge(o, n, e).1));
    assert!(live < in_range && in_range < any, "{live} {in_range} {any}");
}

/// FAILS IF: a distinct count over a lane that is not in key order returns
/// a number. The other end needs a projection stored in its own order.
#[test]
fn a_distinct_count_needs_the_lane_in_key_order() {
    let fx = Fx::new();
    let t = fx.table(&specs(&fx));
    let n = 4097;
    // Stored by field: distinct methods is refused, not miscounted.
    match data(Order::ByField, n).run(&t, 6, Frontend::Quack) {
        Err(CallError::Exec(ExecError::LaneNotOrdered { .. })) => {}
        other => panic!("{other:?}"),
    }
    // Stored by method, the same method answers.
    let o = Order::ByMethod;
    let d = data(o, n);
    let exact = edges(n, |e| {
        of(o, n, e).is_some_and(|f| grade(f) == "exact") && by(o, n, e).is_some()
    });
    assert_eq!(
        d.scalar(&t, 6),
        distinct(exact.iter().map(|&e| by(o, n, e).unwrap()))
    );
    // ...and distinct fields is refused there.
    assert!(matches!(
        d.run(&t, 5, Frontend::Quack),
        Err(CallError::Exec(ExecError::LaneNotOrdered { .. }))
    ));
}

/// FAILS IF: R2IL runs a distinct hop (its fold band has no `GATHER` to
/// test the target's liveness).
#[test]
fn r2il_names_the_missing_gather_for_a_distinct_hop() {
    let fx = Fx::new();
    let t = fx.table(&specs(&fx));
    let d = data(Order::ByField, 65);
    match d.run(&t, 5, Frontend::R2il) {
        Err(CallError::FrontendGap {
            frontend: Frontend::R2il,
            missing: "GATHER",
        }) => {}
        other => panic!("{other:?}"),
    }
}

/// FAILS IF: `out(rel).dedup().count()` stops composing to the distinct
/// fold, a target filter around `dedup()` changes the body, or `dedup()` on
/// a target without `count()` composes.
#[test]
fn dedup_after_a_hop_composes_to_the_distinct_fold() {
    let v = Step::V(READS);
    let c = compose(
        READS,
        &[
            v,
            Step::Where(MethodRef::Label("by_query")),
            Step::Out("of"),
            Step::Dedup,
            Step::Count,
        ],
    )
    .unwrap();
    assert_eq!(c.filter(), &[FROM_QUERIES, OF]);
    assert_eq!(c.folds(), DISTINCT_OF);
    // A target filter commutes with dedup.
    let has = Step::Has {
        field: "grade",
        pred: Pred::Eq("exact"),
    };
    let before = compose(READS, &[v, Step::Out("of"), has, Step::Dedup, Step::Count]).unwrap();
    let after = compose(READS, &[v, Step::Out("of"), Step::Dedup, has, Step::Count]).unwrap();
    assert_eq!(before, after);
    assert_eq!(before.filter(), &[OF, OF_EXACT]);
    let refuse = |steps: &[Step<'static>]| compose(READS, steps).unwrap_err();
    for tail in [
        &[][..],
        &[Step::Values("grade")][..],
        &[Step::Out("x")][..],
        &[Step::Where(MethodRef::Label("p"))][..],
        &[Step::Dedup][..],
    ] {
        let mut steps = vec![v, Step::Out("of"), Step::Dedup];
        steps.extend_from_slice(tail);
        assert_eq!(
            refuse(&steps),
            GremlinRefusal::DedupThroughRelation("of".into()),
            "{tail:?}"
        );
    }
}

/// FAILS IF: the text form of a distinct hop prepares to other programs
/// than the composed one, or the awareness graph does not record the fold.
#[test]
fn a_distinct_hop_loads_from_text_and_projects() {
    let fx = Fx::new();
    let line = format!(
        "document::t key={} schema={:016x} reading=1 := \
         V(document).where(by_query()).out(of).dedup().count()\n",
        "d7".repeat(16),
        SchemaSeal::of(&fx.reads_basis).0
    );
    let decls = parse_decls(&line).unwrap();
    let s = specs(&fx);
    let t = fx.table_with(&s, &decls);
    let programs = |l: &str| {
        t.get(t.resolve(l).unwrap())
            .unwrap()
            .as_ref()
            .unwrap()
            .programs()
            .to_vec()
    };
    assert_eq!(programs("t"), programs("set"));
    let mut g = SoftwareGraph::new();
    g.project(&t, &s, &fx.reads_claims, "distinct.rs", "worktree");
    let set = lance_graph_glove_parity::awareness::method_ident(MethodKey([3; 16]));
    assert!(
        g.out(&set, lance_graph_glove_parity::awareness::Rel::Folds)
            .iter()
            .any(|(f, _)| f.starts_with("fold:distinct(of)")),
        "{:?}",
        g.out(&set, lance_graph_glove_parity::awareness::Rel::Folds)
    );
}

/// FAILS IF: a distinct hop lowers to anything but the direct Quack query:
/// the selection, the relation's liveness test exactly once (whether or not
/// a condition already named the relation), and `COUNT(DISTINCT key)`.
#[test]
fn a_distinct_hop_lowers_to_the_direct_quack_query() {
    let fx = Fx::new();
    let t = fx.table(&specs(&fx));
    let get = |k: u8| t.get(MethodKey([k; 16])).unwrap().as_ref().unwrap();
    let distinct = |filter| {
        vec![Query {
            filter,
            agg: Agg::CountDistinctOrderedU32 { key: Col(1) },
        }]
    };
    // The conditions name `of`: its test is not emitted a second time.
    assert_eq!(
        get(3).queries(),
        distinct(Filter::and([
            Filter::plane(Mask(0)),
            Filter::semijoin(Col(0), QPlane(0)),
            Filter::eq_u32_via(Col(0), ForeignLane(0), 0),
            Filter::semijoin(Col(1), QPlane(1)),
        ]))
    );
    // No condition names `of`: the fold brings its own test.
    assert_eq!(
        get(5).queries(),
        distinct(Filter::and([
            Filter::plane(Mask(0)),
            Filter::semijoin(Col(1), QPlane(1)),
        ]))
    );
    assert!(get(5).required.has(1), "the key is required");
}
