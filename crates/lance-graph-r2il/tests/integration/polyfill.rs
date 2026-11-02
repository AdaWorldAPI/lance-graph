//! The hardcoded polyfill lens `0xFFFF :: 0xF000` (`polyfill`): words bind
//! by position, and R2IL, SQL and Quack names are aliases of a position.
//!
//! Methods here are registered by their names in one language and run
//! through the R2IL binding; each answer is checked against a scalar oracle
//! and, for the Quack names, against Quack's own lowering run through
//! mask-risc over the same lanes.

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ogar_codebook::{classid_canon_compat, classid_concept};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{Focus, FrameRevisions, FrameStore, Guards, Scope};
use lance_graph_dt64::{
    AlgebraRef, Arg, BindStore, ClassSchema, Effect, Extent, MethodBinding, MethodId, MethodRef,
    RouteVersion, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mask_risc::{execute, Foreign, LaneRef, Planes, Scratch, Value};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, Query};
use lance_graph_r2il::binding::{
    prepare, run, ExecPermit, ExecRefusal, FieldLanes, Prepared, R2ilLens, Resolution, EXECUTABLE,
};
use lance_graph_r2il::polyfill::{
    polyfill_arities, position_of, HARDCODED, POLYFILL, POLYFILL_LENS,
};
use lance_graph_r2il::FoldDialect;
use ogar_loco::Vocabulary;
use ogar_r2il::R2ILVocabulary;

const CLASS: u32 = 0x7F0B_0001;
const CLASS_VIEW: ClassId = 0x7F0B;
const REV: SchemaRev = SchemaRev(1);
const LENS: R2ilLens = R2ilLens {
    lens: POLYFILL_LENS,
    schema_rev: REV,
};
const FIELDS: [&str; 8] = ["a", "b", "k", "m", "s", "t", "u", "v"];
const LANES: [Option<u16>; 8] = [Some(0), Some(1), Some(2), None, None, None, None, None];
const FOCUS: Focus = Focus { revision: 1 };
const SCOPE: Scope = Scope { id: 1 };
const NOW: (Focus, Scope) = (FOCUS, SCOPE);
const POSITION: u16 = 3;
const REVS: FrameRevisions = FrameRevisions {
    source: 1,
    relation_map: 1,
    coordinates: 1,
};

// ---------------------------------------------------------------- the table

#[test]
fn the_lens_is_hardcoded_and_every_reader_sees_ffff() {
    assert_eq!(POLYFILL_LENS, 0xFFFF_F000);
    assert_eq!(classid_concept(POLYFILL_LENS), HARDCODED);
    // The compat reader misreads a left-empty classid (TD-CLASSID-CANON-
    // COMPAT-MISREADS-LEFT-EMPTY); a 0xFFFF left side is not affected.
    assert_eq!(classid_canon_compat(POLYFILL_LENS), HARDCODED);
}

#[test]
fn the_table_matches_ogar_and_the_executable_subset() {
    let opcodes: Vec<_> = POLYFILL.iter().map(|w| w.opcode).collect();
    assert_eq!(opcodes, EXECUTABLE.to_vec(), "positions follow EXECUTABLE");
    for (p, w) in POLYFILL.iter().enumerate() {
        assert_eq!(
            R2ILVocabulary.name(w.opcode),
            Some(w.aliases[0]),
            "position {p}: the first alias is the R2IL name"
        );
        assert_eq!(R2ILVocabulary.stack_arity(w.opcode), Some(w.arity), "{p}");
        assert_eq!(R2ILVocabulary.pushes_result(w.opcode), Some(true), "{p}");
    }
}

#[test]
fn every_alias_names_exactly_one_position() {
    let mut n = 0;
    for (p, w) in POLYFILL.iter().enumerate() {
        for a in w.aliases {
            assert_eq!(position_of(a), u8::try_from(p).ok(), "alias {a:?}");
            n += 1;
        }
    }
    assert!(n > POLYFILL.len() * 2, "aliases beyond the R2IL names: {n}");
    assert_eq!(position_of("NotAName"), None);
}

// ---------------------------------------------------------------- fixture

/// One view of the polyfill lens: a label for every position.
struct Views {
    class: Vec<FieldRef>,
    lens: Vec<FieldRef>,
}

impl Views {
    /// Label every position with its alias in `lang` (an index into each
    /// position's aliases, falling back to the R2IL name).
    fn language(lang: &str) -> Self {
        let pick = |w: &lance_graph_r2il::polyfill::FrozenWord| -> String {
            let found = match lang {
                "sql" => w
                    .aliases
                    .iter()
                    .skip(1)
                    .find(|a| !a.contains("::"))
                    .copied(),
                "quack" => w.aliases.iter().find(|a| a.contains("::")).copied(),
                _ => None,
            };
            found.unwrap_or(w.aliases[0]).to_string()
        };
        Self::with_labels(POLYFILL.iter().map(pick).collect())
    }
    fn with_labels(labels: Vec<String>) -> Self {
        Self {
            class: FIELDS
                .iter()
                .map(|l| FieldRef::new(format!("lab:class/{l}"), (*l).to_string()))
                .collect(),
            lens: labels
                .into_iter()
                .map(|l| FieldRef::new(format!("lab:polyfill/{l}"), l))
                .collect(),
        }
    }
}

impl ClassView for Views {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        match class {
            CLASS_VIEW => &self.class,
            HARDCODED => &self.lens,
            _ => &[],
        }
    }
    // D-DT64-7 O9: the polyfill lens is a behaviour lens, read by its full
    // classid.
    fn lens_fields(&self, lens: u32) -> Option<&[FieldRef]> {
        (lens == POLYFILL_LENS).then_some(self.lens.as_slice())
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Card
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

fn pos(name: &str) -> u8 {
    position_of(name).unwrap_or_else(|| panic!("{name:?} is no alias"))
}

fn step(target: u8, name: &str, args: Vec<Arg>) -> Effect {
    Effect {
        target,
        algebra: AlgebraRef {
            lens: POLYFILL_LENS,
            word: pos(name),
        },
        args,
    }
}

fn method(id: u32, reads: &[u8], effects: Vec<Effect>) -> MethodBinding {
    MethodBinding {
        id: MethodId(id),
        sig_rev: SigRev(1),
        name: format!("m{id}"),
        on: WideFieldMask::from_positions(reads),
        when: None,
        reads: WideFieldMask::from_positions(reads),
        effects,
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

fn mref(id: u32) -> MethodRef {
    MethodRef {
        class: CLASS,
        method: MethodId(id),
        sig_rev: SigRev(1),
        guard: None,
    }
}

fn store(methods: Vec<MethodBinding>) -> (BindStore, RouteVersion) {
    let mut s = BindStore::new();
    s.declare_algebra(POLYFILL_LENS, REV, Extent::DR64_CUBE, polyfill_arities())
        .unwrap();
    s.register(
        CLASS,
        ClassSchema {
            schema_rev: REV,
            field_count: FIELDS.len(),
            extent: Extent::DR64_CUBE,
        },
        methods,
    )
    .unwrap();
    let v = s.route(POSITION, CLASS).unwrap();
    (s, v)
}

/// `a = i % 37 - 10`, `b = (7i) % 23`, `k = i / 7` (key-ordered).
struct Data {
    a: Vec<i32>,
    b: Vec<i32>,
    k: Vec<u32>,
}

impl Data {
    fn new(n: usize) -> Self {
        Self {
            a: (0..n).map(|i| (i % 37) as i32 - 10).collect(),
            b: (0..n).map(|i| ((7 * i) % 23) as i32).collect(),
            k: (0..n).map(|i| (i / 7) as u32).collect(),
        }
    }
    fn lanes(&self) -> [LaneRef<'_>; 3] {
        [
            LaneRef::I32(&self.a),
            LaneRef::I32(&self.b),
            LaneRef::U32(&self.k),
        ]
    }
}

struct Run {
    store: BindStore,
    version: RouteVersion,
    frames: FrameStore,
}

impl Run {
    fn new(methods: Vec<MethodBinding>) -> Self {
        let (store, version) = store(methods);
        Self {
            store,
            version,
            frames: FrameStore::new(REVS),
        }
    }
    fn prepare(&self, views: &Views, id: u32) -> Result<Prepared, ExecRefusal> {
        let r = mref(id);
        let g = Guards {
            frame: self.frames.seal(),
            focus: FOCUS,
            scope: SCOPE,
        };
        prepare(
            &self.store,
            views,
            LENS,
            &Resolution {
                method: &r,
                schema_rev: REV,
                route: (self.version, POSITION),
                guards: &g,
            },
            &self.frames,
            NOW,
        )
    }
    /// The value the plan leaves in field 4 (`s`).
    fn answer(&self, plan: &Prepared, data: &Data) -> i64 {
        let lanes = data.lanes();
        let planes = Planes {
            n_rows: data.a.len(),
            masks: &[],
            lanes: &lanes,
        };
        let foreign = Foreign::NONE;
        let mut d = FoldDialect::new(&planes, &foreign);
        let receipt = run(
            plan,
            &self.store,
            &self.frames,
            NOW,
            &ExecPermit {
                methods: vec![plan.method.clone()],
            },
            FieldLanes { lanes: &LANES },
            &mut d,
        )
        .expect("runs");
        receipt
            .output
            .values
            .iter()
            .find(|(f, _)| *f == 4)
            .map(|(_, v)| *v)
            .expect("field 4 written")
    }
}

// ------------------------------------------- position binds, labels alias

const SUM_BELOW: u32 = 1; // s = SUM(b) where a < 5

fn sum_below() -> MethodBinding {
    method(
        SUM_BELOW,
        &[0, 1],
        vec![
            step(3, "IntSLess", vec![Arg::Field(0), Arg::Const(5)]),
            step(4, "SUM", vec![Arg::Field(3), Arg::Field(1)]),
        ],
    )
}

#[test]
fn three_languages_bind_the_same_plan_and_answer() {
    let run = Run::new(vec![sum_below()]);
    let data = Data::new(1000);
    let oracle: i64 = data
        .a
        .iter()
        .zip(&data.b)
        .filter(|(a, _)| **a < 5)
        .map(|(_, b)| i64::from(*b))
        .sum();
    let mut plans = Vec::new();
    for lang in ["r2il", "sql", "quack"] {
        let views = Views::language(lang);
        let plan = run.prepare(&views, SUM_BELOW).expect(lang);
        assert_eq!(run.answer(&plan, &data), oracle, "{lang}");
        plans.push(plan);
    }
    // The languages differ in their labels (`IntSLess`, `<`, `Cmp::LtI32`)
    // and nowhere else: the bound plan, and its identity, are the same.
    assert_eq!(
        Views::language("sql").lens[pos("IntSLess") as usize].label,
        "<"
    );
    for p in &plans[1..] {
        assert_eq!(p.steps, plans[0].steps);
        assert_eq!(p.plan_id, plans[0].plan_id);
    }
}

#[test]
fn a_label_from_another_position_is_refused_not_followed() {
    let run = Run::new(vec![sum_below()]);
    // The view calls the SUM position "MIN": a real alias, of another word.
    let mut labels: Vec<String> = POLYFILL.iter().map(|w| w.aliases[0].to_string()).collect();
    labels[pos("SUM") as usize] = "MIN".to_string();
    let err = run
        .prepare(&Views::with_labels(labels), SUM_BELOW)
        .unwrap_err();
    assert_eq!(
        err,
        ExecRefusal::NotAnAlias {
            effect: 1,
            word: AlgebraRef {
                lens: POLYFILL_LENS,
                word: pos("SUM"),
            },
            label: "MIN".to_string(),
        }
    );
}

#[test]
fn the_position_binds_where_the_label_binding_would_take_another_opcode() {
    // Under the R2IL fold reading the label binds, so a lens that put "MIN"
    // at the SUM position would run MIN. Under the polyfill lens the same
    // label at that position is refused (above), and the R2IL name binds
    // the position's own opcode even when another position comes first in
    // the view's order.
    let run = Run::new(vec![sum_below()]);
    let views = Views::language("r2il");
    let plan = run.prepare(&views, SUM_BELOW).unwrap();
    assert_eq!(plan.steps[1].opcode, POLYFILL[pos("SUM") as usize].opcode);
    assert_ne!(plan.steps[1].opcode, POLYFILL[pos("MIN") as usize].opcode);
}

// ---------------------------------------------- Quack names, Quack answers

/// Quack's answer for `q` over `data`, as the fold dialect reports it.
fn quack(q: &Query, data: &Data, agg: Agg) -> i64 {
    let p = lower(q).expect("lowers");
    let lanes = data.lanes();
    let planes = Planes {
        n_rows: data.a.len(),
        masks: &[],
        lanes: &lanes,
    };
    let mut scratch = Scratch::for_program(&p, data.a.len()).expect("scratch");
    match (execute(&p, &planes, &mut scratch, None).expect("runs"), agg) {
        (Value::Count(n), _) => i64::try_from(n).unwrap(),
        (Value::Bool(b), _) => i64::from(b),
        (Value::SumI64(s), _) => s,
        (Value::OptI32(v), Agg::MinI32(_)) => v.map_or(i64::MAX, i64::from),
        (Value::OptI32(v), _) => v.map_or(i64::MIN, i64::from),
        (other, _) => panic!("unexpected {other:?}"),
    }
}

/// A method registered by Quack names: `m = a < t`, then the aggregate.
fn by_quack_names(id: u32, t: i64, agg: Agg) -> MethodBinding {
    let filter = step(3, "Cmp::LtI32", vec![Arg::Field(0), Arg::Const(t)]);
    let fold = match agg {
        Agg::Count => step(4, "Agg::Count", vec![Arg::Field(3)]),
        Agg::Any => step(4, "Agg::Any", vec![Arg::Field(3)]),
        Agg::All => step(4, "Agg::All", vec![Arg::Field(3)]),
        Agg::SumI32(_) => step(4, "Agg::SumI32", vec![Arg::Field(3), Arg::Field(1)]),
        Agg::MinI32(_) => step(4, "Agg::MinI32", vec![Arg::Field(3), Arg::Field(1)]),
        Agg::MaxI32(_) => step(4, "Agg::MaxI32", vec![Arg::Field(3), Arg::Field(1)]),
        Agg::CountDistinctOrderedU32 { .. } => step(
            4,
            "Agg::CountDistinctOrderedU32",
            vec![Arg::Field(3), Arg::Field(2)],
        ),
        other => panic!("not covered: {other:?}"),
    };
    method(id, &[0, 1, 2], vec![filter, fold])
}

fn aggs() -> [Agg; 7] {
    [
        Agg::Count,
        Agg::Any,
        Agg::All,
        Agg::SumI32(Col(1)),
        Agg::MinI32(Col(1)),
        Agg::MaxI32(Col(1)),
        Agg::CountDistinctOrderedU32 { key: Col(2) },
    ]
}

#[test]
fn quack_names_register_methods_that_answer_as_quack_does() {
    let data = Data::new(1000);
    // Three thresholds: a partial selection, everything (a <= 26 < 100),
    // and nothing (a >= -10 > -50).
    let thresholds = [5_i64, 100, -50];
    let mut methods = Vec::new();
    let mut cases = Vec::new();
    for (ti, &t) in thresholds.iter().enumerate() {
        for (ai, agg) in aggs().into_iter().enumerate() {
            let id = u32::try_from(100 + ti * 10 + ai).unwrap();
            methods.push(by_quack_names(id, t, agg));
            cases.push((id, t, agg));
        }
    }
    let run = Run::new(methods);
    let views = Views::language("quack");
    let mut distinct = std::collections::BTreeSet::new();
    for (id, t, agg) in cases {
        let plan = run.prepare(&views, id).expect("prepares");
        let got = run.answer(&plan, &data);
        let q = Query {
            filter: Filter::cmp(Col(0), Cmp::LtI32(i32::try_from(t).unwrap())),
            agg,
        };
        assert_eq!(got, quack(&q, &data, agg), "{agg:?} where a < {t}");
        distinct.insert(got);
    }
    // Anti-vacuity: the 21 cases do not all collapse to a few values.
    assert!(distinct.len() >= 10, "answers: {distinct:?}");
}

#[test]
fn a_compound_quack_filter_registers_by_name_and_matches() {
    // COUNT where a < 5 AND NOT a = -3.
    let m = method(
        200,
        &[0],
        vec![
            step(3, "Cmp::LtI32", vec![Arg::Field(0), Arg::Const(5)]),
            step(5, "Cmp::EqI32", vec![Arg::Field(0), Arg::Const(-3)]),
            step(6, "Filter::negate", vec![Arg::Field(5)]),
            step(7, "Filter::and", vec![Arg::Field(3), Arg::Field(6)]),
            step(4, "Agg::Count", vec![Arg::Field(7)]),
        ],
    );
    let run = Run::new(vec![m]);
    let data = Data::new(1000);
    let plan = run.prepare(&Views::language("quack"), 200).unwrap();
    let got = run.answer(&plan, &data);
    let q = Query {
        filter: Filter::and([
            Filter::cmp(Col(0), Cmp::LtI32(5)),
            Filter::negate(Filter::cmp(Col(0), Cmp::EqI32(-3))),
        ]),
        agg: Agg::Count,
    };
    assert_eq!(got, quack(&q, &data, Agg::Count));
    let oracle = data.a.iter().filter(|a| **a < 5 && **a != -3).count() as i64;
    assert_eq!(got, oracle);
    assert!(oracle > 0 && oracle < 1000);
}

#[test]
fn a_wrong_runtime_arity_for_the_lens_is_refused_by_the_table() {
    // The store's declared arity is a runtime input; the table is not. A
    // store that declares every polyfill word unary accepts a one-argument
    // SUM at registration, and the table refuses it at prepare.
    let mut s = BindStore::new();
    s.declare_algebra(
        POLYFILL_LENS,
        REV,
        Extent::DR64_CUBE,
        vec![1; POLYFILL.len()],
    )
    .unwrap();
    s.register(
        CLASS,
        ClassSchema {
            schema_rev: REV,
            field_count: FIELDS.len(),
            extent: Extent::DR64_CUBE,
        },
        vec![method(300, &[0], vec![step(4, "SUM", vec![Arg::Field(0)])])],
    )
    .unwrap();
    let version = s.route(POSITION, CLASS).unwrap();
    let run = Run {
        store: s,
        version,
        frames: FrameStore::new(REVS),
    };
    let err = run.prepare(&Views::language("sql"), 300).unwrap_err();
    assert_eq!(
        err,
        ExecRefusal::Arity {
            effect: 0,
            ogar: None,
            declared: Some(2),
            args: 1,
        }
    );
}
