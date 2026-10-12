//! D-CMH-G1-7: chained hops are folded hops.
//!
//! `out(r1)…out(r2)` over two functional relations keeps the anchor (each
//! line still names at most one partner, each partner at most one region),
//! and lowers to ONE `Filter::Chain` — one `MaskOp::GatherChain` — with every
//! condition on a hop read in place at the row the key reached. No mask over
//! `partner` or `region` exists anywhere in the program.
//!
//! FIXTURE: `line -billed_to-> partner -located_in-> region`. Every answer is
//! checked against a row loop over the TEXT of each field.
use lance_graph_glove_parity::awareness::{method_ident, Rel, SoftwareGraph};
use lance_graph_glove_parity::basis::{FieldMap, Grade};
use lance_graph_glove_parity::decl::parse_decls;
use lance_graph_glove_parity::gremlin::{compose, GremlinRefusal, Pred, Step};
use lance_graph_glove_parity::method::{
    Answer, CallError, Cond, EffectClass, Fold, FoldValue, Frontend, Glove, Law, MethodBody,
    MethodKey, MethodSpec, MethodTable, Path, PrepareError, Reach, ReadingId, Relation, SchemaSeal,
    Universe, MAX_PATH,
};
use lance_graph_mask_risc::{words_for, Foreign, ForeignPlane, LaneRef, MaskOp, Planes};
use lance_graph_ogar::ogar_vocab::class_ids::DOCUMENT;
use lance_graph_quack::bind::{Binder, BoundField, FieldKind, TableId};
use lance_graph_quack::{
    ChainHop, ChainTest, Cmp, Col, Filter, ForeignLane, ForeignPlane as QPlane, Mask,
};

/// The anchor is a minted concept so the text form can name it.
const LINE: u16 = DOCUMENT;
const PARTNER: u16 = 0xFFE2;
const REGION: u16 = 0xFFE3;
const READING: ReadingId = ReadingId(1);
const LINE_U: Universe = Universe(7);
const PARTNER_U: Universe = Universe(8);
const REGION_U: Universe = Universe(9);
const PARTNERS: usize = 37;
const REGIONS: usize = 11;
const SIZES: [usize; 4] = [1, 64, 65, 4097];

const STATUS: [&str; 3] = ["open", "posted", "void"];
const COUNTRY: [&str; 3] = ["DE", "FR", "US"];
const ZONE: [&str; 3] = ["EU", "NA", "APAC"];

fn status(i: usize) -> &'static str {
    STATUS[(i * 5 + i / 7) % 3]
}
fn amount(i: usize) -> i32 {
    ((i * 37) % 201) as i32 - 100
}
fn line_live(i: usize) -> bool {
    i % 13 != 5
}
/// Some keys dangle at depth 1.
fn partner_of(i: usize) -> usize {
    (i * 11 + 3) % (PARTNERS + 4)
}
fn partner_live(p: usize) -> bool {
    p % 7 != 2
}
fn country(p: usize) -> &'static str {
    COUNTRY[(p * 3 + p / 4) % 3]
}
/// Some keys dangle at depth 2.
fn region_of(p: usize) -> usize {
    (p * 5 + 1) % (REGIONS + 3)
}
fn region_live(r: usize) -> bool {
    r % 5 != 3
}
fn zone(r: usize) -> &'static str {
    ZONE[(r * 2 + r / 3) % 3]
}

/// The partner a line reaches: in range and live.
fn reached(i: usize) -> Option<usize> {
    let p = partner_of(i);
    (p < PARTNERS && partner_live(p)).then_some(p)
}
/// The region a line reaches through its partner.
fn reached2(i: usize) -> Option<usize> {
    let r = region_of(reached(i)?);
    (r < REGIONS && region_live(r)).then_some(r)
}

struct Erp;

impl Binder for Erp {
    fn table(&self, name: &str) -> Option<TableId> {
        match name {
            "line" => Some(TableId(0)),
            "partner" => Some(TableId(1)),
            "region" => Some(TableId(2)),
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
            (1, "region") => (1, FieldKind::Code),
            (2, "zone") => (0, FieldKind::Code),
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
            (2, 0) => &ZONE,
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

fn code(d: &[&str], s: &str) -> u32 {
    d.iter().position(|x| *x == s).unwrap() as u32
}

struct Data {
    n: usize,
    line_live: Vec<u64>,
    status: Vec<u32>,
    amount: Vec<i32>,
    partner: Vec<u32>,
    partner_live: Vec<u64>,
    country: Vec<u32>,
    region: Vec<u32>,
    region_live: Vec<u64>,
    zone: Vec<u32>,
}

fn data(n: usize) -> Data {
    Data {
        n,
        line_live: bits(n, line_live),
        status: (0..n).map(|i| code(&STATUS, status(i))).collect(),
        amount: (0..n).map(amount).collect(),
        partner: (0..n).map(|i| partner_of(i) as u32).collect(),
        partner_live: bits(PARTNERS, partner_live),
        country: (0..PARTNERS).map(|p| code(&COUNTRY, country(p))).collect(),
        region: (0..PARTNERS).map(|p| region_of(p) as u32).collect(),
        region_live: bits(REGIONS, region_live),
        zone: (0..REGIONS).map(|r| code(&ZONE, zone(r))).collect(),
    }
}

impl Data {
    fn run(
        &self,
        table: &MethodTable,
        key: MethodKey,
        frontend: Frontend,
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
        let fplanes = [
            ForeignPlane {
                words: &self.partner_live,
                rows: PARTNERS,
            },
            ForeignPlane {
                words: &self.region_live,
                rows: REGIONS,
            },
        ];
        let flanes = [
            LaneRef::U32(&self.country),
            LaneRef::U32(&self.region),
            LaneRef::U32(&self.zone),
        ];
        let reach = Reach {
            foreign: Foreign {
                planes: &fplanes,
                lanes: &flanes,
            },
            universes: &[(PARTNER, PARTNER_U), (REGION, REGION_U)],
        };
        table.call_with(key, &planes, LINE_U, &reach, frontend)
    }
}

/// Row loop over the text: `(count, sum(amount))` of live rows passing `keep`.
fn oracle(n: usize, keep: impl Fn(usize) -> bool) -> (i64, i64) {
    let rows: Vec<usize> = (0..n).filter(|&i| line_live(i) && keep(i)).collect();
    (
        rows.len() as i64,
        rows.iter().map(|&i| i64::from(amount(i))).sum(),
    )
}

fn scalars(a: Answer) -> (i64, i64) {
    match a {
        Answer::Values(v) => match v.as_slice() {
            [FoldValue::Scalar(c), FoldValue::Scalar(s)] => (*c, *s),
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    }
}

struct Fx {
    line_basis: Vec<String>,
    line_claims: Vec<FieldMap>,
    partner_basis: Vec<String>,
    partner_claims: Vec<FieldMap>,
    region_basis: Vec<String>,
    region_claims: Vec<FieldMap>,
}

impl Fx {
    fn new() -> Self {
        Self {
            line_basis: names(&["status", "amount", "partner"]),
            line_claims: exact(&["status", "amount", "partner"]),
            partner_basis: names(&["country", "region"]),
            partner_claims: exact(&["country", "region"]),
            region_basis: names(&["zone"]),
            region_claims: exact(&["zone"]),
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

    fn table(&self, specs: &[(&str, MethodSpec<'_>)]) -> MethodTable {
        self.table_with(specs, &[("country", 0), ("region", 1)])
    }

    /// `partner_lanes` is what a call supplies for the partner table; the
    /// region key must be among them for a second hop to bind.
    fn table_with(
        &self,
        specs: &[(&str, MethodSpec<'_>)],
        partner_lanes: &[(&str, u16)],
    ) -> MethodTable {
        self.build(partner_lanes, |g| MethodTable::prepare(g, specs))
    }

    fn build(
        &self,
        partner_lanes: &[(&str, u16)],
        f: impl FnOnce(&Glove<'_>) -> MethodTable,
    ) -> MethodTable {
        let region = Glove {
            concept: REGION,
            basis: &self.region_basis,
            claims: &self.region_claims,
            binder: &Erp,
            table: "region",
            reading: READING,
            universe: REGION_U,
            relations: &[],
        };
        let partner_rels = [Relation {
            name: "located_in",
            fk: "region",
            target: &region,
            plane: 1,
            lanes: &[("zone", 2)],
        }];
        let partner = Glove {
            concept: PARTNER,
            basis: &self.partner_basis,
            claims: &self.partner_claims,
            binder: &Erp,
            table: "partner",
            reading: READING,
            universe: PARTNER_U,
            relations: &partner_rels,
        };
        let rels = [Relation {
            name: "billed_to",
            fk: "partner",
            target: &partner,
            plane: 0,
            lanes: partner_lanes,
        }];
        let glove = Glove {
            concept: LINE,
            basis: &self.line_basis,
            claims: &self.line_claims,
            binder: &Erp,
            table: "line",
            reading: READING,
            universe: LINE_U,
            relations: &rels,
        };
        f(&glove)
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
const BL: Path = Path::of(&["billed_to", "located_in"]);
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
const REACHES2: Cond = Cond::ReachesPath { path: BL };
const IN_EU: Cond = Cond::ViaPath {
    path: BL,
    field: "zone",
    literal: "EU",
    negate: false,
};
const NOT_EU: Cond = Cond::ViaPath {
    path: BL,
    field: "zone",
    literal: "EU",
    negate: true,
};

/// FAILS IF a two-hop condition disagrees with the row loop at any size —
/// with tests at hop 1, hop 2, both, negated, or with a local condition.
/// Anti-vacuity: each depth rejects rows the other keeps.
#[test]
fn two_hops_match_the_row_loop() {
    let fx = Fx::new();
    let specs = [
        ("reaches2", fx.spec(1, query(&[REACHES2], COUNT_SUM))),
        ("de_eu", fx.spec(2, query(&[TO_DE, IN_EU], COUNT_SUM))),
        ("open_not_eu", fx.spec(3, query(&[OPEN, NOT_EU], COUNT_SUM))),
    ];
    let table = fx.table(&specs);
    for n in SIZES {
        let d = data(n);
        type Keep = Box<dyn Fn(usize) -> bool>;
        let cases: [(u8, Keep); 3] = [
            (1, Box::new(|i| reached2(i).is_some())),
            (
                2,
                Box::new(|i| {
                    reached(i).is_some_and(|p| country(p) == "DE")
                        && reached2(i).is_some_and(|r| zone(r) == "EU")
                }),
            ),
            (
                3,
                Box::new(|i| status(i) == "open" && reached2(i).is_some_and(|r| zone(r) != "EU")),
            ),
        ];
        for (k, keep) in cases {
            let got = scalars(d.run(&table, MethodKey([k; 16]), Frontend::Quack).unwrap());
            assert_eq!(got, oracle(n, keep), "method {k}, n={n}");
        }
    }
    let n = 4097;
    let one = oracle(n, |i| reached(i).is_some()).0;
    let two = oracle(n, |i| reached2(i).is_some()).0;
    let eu = oracle(n, |i| reached2(i).is_some_and(|r| zone(r) == "EU")).0;
    assert!(0 < eu && eu < two && two < one, "{eu} {two} {one}");
    // `!=` at depth 2 still requires a live region.
    let not_eu = oracle(n, |i| reached2(i).is_some_and(|r| zone(r) != "EU")).0;
    let not_eu_loose = oracle(n, |i| {
        reached(i).is_some_and(|p| {
            let r = region_of(p);
            !(r < REGIONS && zone(r) == "EU")
        })
    })
    .0;
    assert!(not_eu < not_eu_loose, "{not_eu} {not_eu_loose}");
}

/// FAILS IF the two hops lower to anything but ONE chain over both, with the
/// hop-1 condition folded into hop 1 (no separate semijoin or `EqU32Via`).
#[test]
fn two_hops_lower_to_one_chain() {
    let fx = Fx::new();
    let specs = [("de_eu", fx.spec(1, query(&[OPEN, TO_DE, IN_EU], COUNT_SUM)))];
    let table = fx.table(&specs);
    let m = table.get(MethodKey([1; 16])).unwrap().as_ref().unwrap();
    let chain = Filter::chain(
        Col(2),
        [
            ChainHop {
                live: QPlane(0),
                tests: vec![ChainTest {
                    lane: ForeignLane(0),
                    value: 0,
                    negate: false,
                }],
                next: Some(ForeignLane(1)),
            },
            ChainHop {
                live: QPlane(1),
                tests: vec![ChainTest {
                    lane: ForeignLane(2),
                    value: 0,
                    negate: false,
                }],
                next: None,
            },
        ],
    );
    let filter = Filter::and([
        Filter::plane(Mask(0)),
        Filter::cmp(Col(0), Cmp::EqU32(0)),
        chain,
    ]);
    let queries = m.queries();
    assert_eq!(queries.len(), 2);
    for q in &queries {
        assert_eq!(q.filter, filter);
    }
    for p in m.programs() {
        let gc = p
            .ops
            .iter()
            .filter(|o| matches!(o, MaskOp::GatherChain { .. }))
            .count();
        assert_eq!(gc, 1, "{:?}", p.ops);
        assert!(
            !p.ops.iter().any(|o| matches!(o, MaskOp::Gather { .. })),
            "no per-hop gather: {:?}",
            p.ops
        );
        assert_eq!(p.chains.len(), 1);
        assert_eq!(p.chains[0].len(), 2);
    }
    // Writing `out(billed_to)` explicitly changes nothing.
    let specs = [
        (
            "explicit",
            fx.spec(
                1,
                query(
                    &[
                        OPEN,
                        Cond::Reaches { rel: "billed_to" },
                        TO_DE,
                        REACHES2,
                        IN_EU,
                    ],
                    COUNT_SUM,
                ),
            ),
        ),
        (
            "implied",
            fx.spec(2, query(&[OPEN, TO_DE, IN_EU], COUNT_SUM)),
        ),
    ];
    let t = fx.table(&specs);
    let get = |k: u8| t.get(MethodKey([k; 16])).unwrap().as_ref().unwrap();
    assert_eq!(get(1).queries(), get(2).queries());
}

/// FAILS IF the composer refuses a second hop, or composes it to other
/// conditions than the written ones.
#[test]
fn gremlin_composes_a_second_hop() {
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
            Step::Out("located_in"),
            Step::Has {
                field: "zone",
                pred: Pred::Neq("EU"),
            },
            Step::Count,
        ],
    )
    .unwrap();
    assert_eq!(
        c.filter(),
        &[Cond::Reaches { rel: "billed_to" }, TO_DE, REACHES2, NOT_EU]
    );
    assert_eq!(c.folds(), &[Fold::Count]);
    let refuse = |steps: &[Step<'static>]| compose(LINE, steps).unwrap_err();
    let two = [v, Step::Out("billed_to"), Step::Out("located_in")];
    let mut s = two.to_vec();
    s.extend([Step::Dedup, Step::Count]);
    assert_eq!(
        refuse(&s),
        GremlinRefusal::DedupThroughRelation("located_in".into())
    );
    let mut s = two.to_vec();
    s.extend([Step::Values("zone"), Step::Sum]);
    assert_eq!(
        refuse(&s),
        GremlinRefusal::ValueThroughRelation("located_in".into())
    );
    let mut deep = vec![v];
    deep.extend(std::iter::repeat_n(Step::Out("r"), MAX_PATH + 1));
    deep.push(Step::Count);
    assert_eq!(refuse(&deep), GremlinRefusal::PathTooDeep("r".into()));
}

/// FAILS IF a second hop binds without the relation, without the next key
/// supplied, or with the R2IL frontend (which has no gather).
#[test]
fn a_second_hop_is_refused_where_it_cannot_bind() {
    let fx = Fx::new();
    const UNKNOWN: Cond = Cond::ReachesPath {
        path: Path::of(&["billed_to", "shipped_to"]),
    };
    let specs = [("x", fx.spec(1, query(&[UNKNOWN], COUNT_SUM)))];
    let t = fx.table(&specs);
    assert_eq!(
        t.get(MethodKey([1; 16])).unwrap().as_ref().unwrap_err(),
        &PrepareError::UnknownRelation("shipped_to".into())
    );
    let specs = [("y", fx.spec(1, query(&[REACHES2], COUNT_SUM)))];
    let t = fx.table_with(&specs, &[("country", 0)]);
    assert_eq!(
        t.get(MethodKey([1; 16])).unwrap().as_ref().unwrap_err(),
        &PrepareError::NoForeignLane("billed_to.region".into())
    );
    let t = fx.table(&specs);
    match data(65).run(&t, MethodKey([1; 16]), Frontend::R2il) {
        Err(CallError::FrontendGap {
            frontend: Frontend::R2il,
            missing: "GATHER",
        }) => {}
        other => panic!("{other:?}"),
    }
    // A call that does not supply the region table's universe is refused.
    let d = data(65);
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
    let reach = Reach {
        foreign: Foreign {
            planes: &[],
            lanes: &[],
        },
        universes: &[(PARTNER, PARTNER_U)],
    };
    match t.call_with(MethodKey([1; 16]), &planes, LINE_U, &reach, Frontend::Quack) {
        Err(CallError::ForeignUniverse { concept, .. }) => assert_eq!(concept, REGION),
        other => panic!("{other:?}"),
    }
}

/// FAILS IF the text form of a two-hop traversal prepares to other programs
/// than the compiled one, or the awareness graph does not record the path.
#[test]
fn a_two_hop_traversal_loads_from_text_and_projects() {
    let fx = Fx::new();
    let line = format!(
        "document::t key={} schema={:016x} reading=1 := \
         V(document).out(billed_to).has(country, DE).out(located_in).has(zone, EU).count()\n",
        "e7".repeat(16),
        SchemaSeal::of(&fx.line_basis).0
    );
    let decls = parse_decls(&line).unwrap();
    let compiled = [(
        "c",
        fx.spec(
            1,
            query(
                &[Cond::Reaches { rel: "billed_to" }, TO_DE, REACHES2, IN_EU],
                &[Fold::Count],
            ),
        ),
    )];
    let t = fx.build(&[("country", 0), ("region", 1)], |g| {
        MethodTable::prepare_with(g, &compiled, &decls)
    });
    let programs = |l: &str| {
        t.get(t.resolve(l).unwrap())
            .unwrap()
            .as_ref()
            .unwrap()
            .programs()
            .to_vec()
    };
    assert_eq!(programs("t"), programs("c"));
    let mut g = SoftwareGraph::new();
    g.project(&t, &compiled, &fx.line_claims, "two_hop.rs", "worktree");
    let filters = g.out(&method_ident(MethodKey([1; 16])), Rel::Filters);
    assert!(
        filters
            .iter()
            .any(|(f, _)| f.starts_with("cond:out(billed_to).out(located_in).zone=EU")),
        "{filters:?}"
    );
}
