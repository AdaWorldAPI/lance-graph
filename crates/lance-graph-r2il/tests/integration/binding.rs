//! The optional R2IL reading of a DT64 `MethodBinding` (`binding`).
//!
//! Every answer is checked against a scalar oracle over the same lane slices.
//! Lab classids: `0x7F0B_0001` (the consumer class) and `0x7F0A_0001` (a
//! second, non-R2IL lens), as in DT64's own tests. The R2IL lens uses the
//! concept half OGAR declares for the fold reading (`0xC401`), which OGAR
//! marks provisional: it is never persisted here.

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_algebra::{Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{
    Focus, FrameRefusal, FrameRevisions, FrameStore, GuardRefusal, Guards, Scope,
};
use lance_graph_dt64::{
    AlgebraRef, Arg, BindStore, ClassSchema, Effect, Extent, Guard, MethodBinding, MethodId,
    MethodRef, Refusal, RouteVersion, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mask_risc::{Foreign, LaneRef, Planes};
use lance_graph_r2il::binding::{
    prepare, run, ExecPermit, ExecRefusal, FieldLanes, Prepared, R2ilLens, Receipt, Resolution,
};
use lance_graph_r2il::{FoldDialect, FoldError, INT_S_LESS};
use ogar_loco::{FnIndex, Vocabulary};
use ogar_r2il::{R2ILVocabulary, CONCEPT_R2IL_FOLD};

const CLASS: u32 = 0x7F0B_0001;
const CLASS_VIEW: ClassId = 0x7F0B;
const OTHER_CLASS: u32 = 0x7F0C_0001;
const OTHER_VIEW: ClassId = 0x7F0C;
const THIRD_CLASS: u32 = 0x7F0D_0001;
const THIRD_VIEW: ClassId = 0x7F0D;
const LENS: u32 = (CONCEPT_R2IL_FOLD as u32) << 16;
const OTHER_LENS: u32 = 0x7F0A_0001;
const OTHER_LENS_VIEW: ClassId = 0x7F0A;
const REV: SchemaRev = SchemaRev(1);
const R2IL: R2ilLens = R2ilLens {
    lens: LENS,
    schema_rev: REV,
};

/// The class's fields: two I32 lanes, one U32 lane, then computed fields.
const FIELDS: [&str; 8] = ["a", "b", "k", "m", "s", "t", "u", "free"];
/// Field → resident lane. `free` (7) has none.
const LANES: [Option<u16>; 8] = [Some(0), Some(1), Some(2), None, None, None, None, None];

/// The lens's words, deliberately NOT at their opcode numbers: only the
/// label binds a word to an opcode.
const WORDS: [&str; 10] = [
    "SUM",
    "IntSLess",
    "IntSub",
    "IntAdd",
    "IntMult",
    "PopCount",
    "IntEqual",
    "GROUP_SUM",
    "NotAnOpcode",
    "IntAnd",
];
const W_SUM: u8 = 0;
const W_LT: u8 = 1;
const W_SUB: u8 = 2;
const W_ADD: u8 = 3;
const W_MUL: u8 = 4;
const W_POP: u8 = 5;
const W_GSUM: u8 = 7;
const W_BAD: u8 = 8;
const W_AND: u8 = 9;

const FOCUS: Focus = Focus { revision: 1 };
const SCOPE: Scope = Scope { id: 1 };
const NOW: (Focus, Scope) = (FOCUS, SCOPE);
const POSITION: u16 = 3;

struct Views {
    class: Vec<FieldRef>,
    lens: Vec<FieldRef>,
}

fn refs(labels: &[&str], ns: &str) -> Vec<FieldRef> {
    labels
        .iter()
        .map(|l| FieldRef::new(format!("lab:{ns}/{l}"), (*l).to_string()))
        .collect()
}

impl Views {
    fn new() -> Self {
        Self {
            class: refs(&FIELDS, "class"),
            lens: refs(&WORDS, "r2il"),
        }
    }
}

impl ClassView for Views {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        match class {
            CLASS_VIEW | OTHER_VIEW | THIRD_VIEW => &self.class,
            c if c == CONCEPT_R2IL_FOLD || c == OTHER_LENS_VIEW => &self.lens,
            _ => &[],
        }
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Card
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

/// The lens arities, read from OGAR by label. `NotAnOpcode` has none there,
/// so the lens declares 1 for it.
fn ogar_arities() -> Vec<u8> {
    WORDS
        .iter()
        .map(|w| {
            (0..=u8::MAX)
                .map(FnIndex)
                .find(|&f| R2ILVocabulary.name(f) == Some(*w))
                .and_then(|f| R2ILVocabulary.stack_arity(f))
                .unwrap_or(1)
        })
        .collect()
}

fn w(lens: u32, word: u8) -> AlgebraRef {
    AlgebraRef { lens, word }
}

fn step(target: u8, word: u8, args: Vec<Arg>) -> Effect {
    Effect {
        target,
        algebra: w(LENS, word),
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

/// `a < 5` as a mask in `m`, then `SUM(b)` under it into `s`.
fn masked_sum() -> Vec<Effect> {
    vec![
        step(3, W_LT, vec![Arg::Field(0), Arg::Const(5)]),
        step(4, W_SUM, vec![Arg::Field(3), Arg::Field(1)]),
    ]
}

const ORDER_A: u32 = 1; // t = s + 1, then s = s * 2
const ORDER_B: u32 = 2; // s = s * 2, then t = s + 1
const SUM_A: u32 = 3; // the masked sum over a instead of b
const SUB_RIGHT: u32 = 4; // t = s - 7
const SUB_LEFT: u32 = 5; // t = 7 - s
const WRONG_KIND: u32 = 6; // a signed compare on the U32 lane
const WIDE_CONST: u32 = 7; // a < 2^40
const UNBOUND: u32 = 8; // free < 5
const BAD_WORD: u32 = 9;
const SINK_WORD: u32 = 10; // GROUP_SUM
const GUARDED: u32 = 11;
const THREE_VALUED: u32 = 12;
const COUNT: u32 = 13; // POP_COUNT of the mask
const SHARED_AND: u32 = 14; // m = a < 5; t = a < 10; u = m AND t; s = POP_COUNT(t)
const PRED_ONLY: u32 = 15; // m = a < 5, consumed by no fold

fn methods() -> Vec<MethodBinding> {
    let with = |id, extra: Vec<Effect>| {
        let mut e = masked_sum();
        e.extend(extra);
        method(id, &[0, 1], e)
    };
    let mut guarded = method(GUARDED, &[0, 1], masked_sum());
    guarded.when = Some(Guard::new(SetExpr::Sym(1), Logic::TwoValued));
    let mut three = method(THREE_VALUED, &[0, 1], masked_sum());
    three.truth = Truth::ThreeValued;
    vec![
        with(
            ORDER_A,
            vec![
                step(5, W_ADD, vec![Arg::Field(4), Arg::Const(1)]),
                step(4, W_MUL, vec![Arg::Field(4), Arg::Const(2)]),
            ],
        ),
        with(
            ORDER_B,
            vec![
                step(4, W_MUL, vec![Arg::Field(4), Arg::Const(2)]),
                step(5, W_ADD, vec![Arg::Field(4), Arg::Const(1)]),
            ],
        ),
        method(
            SUM_A,
            &[0],
            vec![
                step(3, W_LT, vec![Arg::Field(0), Arg::Const(5)]),
                step(4, W_SUM, vec![Arg::Field(3), Arg::Field(0)]),
            ],
        ),
        with(
            SUB_RIGHT,
            vec![step(5, W_SUB, vec![Arg::Field(4), Arg::Const(7)])],
        ),
        with(
            SUB_LEFT,
            vec![step(5, W_SUB, vec![Arg::Const(7), Arg::Field(4)])],
        ),
        method(
            WRONG_KIND,
            &[2],
            vec![step(3, W_LT, vec![Arg::Field(2), Arg::Const(5)])],
        ),
        method(
            WIDE_CONST,
            &[0],
            vec![step(3, W_LT, vec![Arg::Field(0), Arg::Const(1 << 40)])],
        ),
        method(
            UNBOUND,
            &[7],
            vec![step(3, W_LT, vec![Arg::Field(7), Arg::Const(5)])],
        ),
        method(BAD_WORD, &[0], vec![step(3, W_BAD, vec![Arg::Field(0)])]),
        method(
            SINK_WORD,
            &[0, 1, 2],
            vec![step(
                3,
                W_GSUM,
                vec![Arg::Field(0), Arg::Field(2), Arg::Field(1)],
            )],
        ),
        guarded,
        three,
        method(
            COUNT,
            &[0],
            vec![
                step(3, W_LT, vec![Arg::Field(0), Arg::Const(5)]),
                step(4, W_POP, vec![Arg::Field(3)]),
            ],
        ),
        method(
            SHARED_AND,
            &[0],
            vec![
                step(3, W_LT, vec![Arg::Field(0), Arg::Const(5)]),
                step(5, W_LT, vec![Arg::Field(0), Arg::Const(10)]),
                step(6, W_AND, vec![Arg::Field(3), Arg::Field(5)]),
                step(4, W_POP, vec![Arg::Field(5)]),
            ],
        ),
        method(
            PRED_ONLY,
            &[0],
            vec![step(3, W_LT, vec![Arg::Field(0), Arg::Const(5)])],
        ),
    ]
}

fn schema() -> ClassSchema {
    ClassSchema {
        schema_rev: REV,
        field_count: FIELDS.len(),
        extent: Extent::DR64_CUBE,
    }
}

/// The store: the R2IL lens, a second lens, the class and a second class.
fn store(arities: Vec<u8>) -> (BindStore, RouteVersion) {
    let mut s = BindStore::new();
    s.declare_algebra(LENS, REV, Extent::DR64_CUBE, arities.clone())
        .unwrap();
    s.declare_algebra(OTHER_LENS, REV, Extent::DR64_CUBE, arities)
        .unwrap();
    s.register(CLASS, schema(), methods()).unwrap();
    s.register(OTHER_CLASS, schema(), vec![]).unwrap();
    let v = s.route(POSITION, CLASS).unwrap();
    (s, v)
}

const REVS: FrameRevisions = FrameRevisions {
    source: 1,
    relation_map: 1,
    coordinates: 1,
};

fn guards(frames: &FrameStore) -> Guards {
    Guards {
        frame: frames.seal(),
        focus: FOCUS,
        scope: SCOPE,
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

/// Lanes of `n` rows: `a = i % 37 - 10`, `b = (7i) % 23`, `k = i % 5`.
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
            k: (0..n).map(|i| (i % 5) as u32).collect(),
        }
    }
    /// The scalar oracle: `SUM(over)` where `a < t`.
    fn masked_sum(&self, over: &[i32], t: i32) -> i64 {
        self.a
            .iter()
            .zip(over)
            .filter(|(a, _)| **a < t)
            .map(|(_, v)| i64::from(*v))
            .sum()
    }
    fn count_below(&self, t: i32) -> i64 {
        self.a.iter().filter(|a| **a < t).count() as i64
    }
}

struct Fixture {
    views: Views,
    store: BindStore,
    version: RouteVersion,
    frames: FrameStore,
}

impl Fixture {
    fn new() -> Self {
        let (store, version) = store(ogar_arities());
        Self {
            views: Views::new(),
            store,
            version,
            frames: FrameStore::new(REVS),
        }
    }
    fn prepare(&self, id: u32) -> Result<Prepared, ExecRefusal> {
        let r = mref(id);
        let g = guards(&self.frames);
        prepare(
            &self.store,
            &self.views,
            R2IL,
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
}

fn permit(ids: &[u32]) -> ExecPermit {
    ExecPermit {
        methods: ids.iter().map(|&i| mref(i)).collect(),
    }
}

fn exec(
    f: &Fixture,
    plan: &Prepared,
    data: &Data,
    permit: &ExecPermit,
    now: (Focus, Scope),
) -> Result<Receipt, ExecRefusal> {
    let lanes = [
        LaneRef::I32(&data.a),
        LaneRef::I32(&data.b),
        LaneRef::U32(&data.k),
    ];
    let planes = Planes {
        n_rows: data.a.len(),
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign::NONE;
    let mut d = FoldDialect::new(&planes, &foreign);
    run(
        plan,
        &f.store,
        &f.frames,
        now,
        permit,
        FieldLanes { lanes: &LANES },
        &mut d,
    )
}

fn values(r: &Receipt) -> Vec<(u8, i64)> {
    r.output.values.clone()
}

const N: usize = 1_000;

#[test]
fn the_binding_determines_what_executes() {
    let f = Fixture::new();
    let data = Data::new(N);
    let plan = f.prepare(ORDER_A).unwrap();
    let r = exec(&f, &plan, &data, &permit(&[ORDER_A]), NOW).unwrap();
    let s = data.masked_sum(&data.b, 5);
    assert!(s > 0, "the fixture must select rows");
    assert_eq!(values(&r), vec![(4, 2 * s), (5, s + 1)]);
    assert_eq!(r.output.intermediates, vec![3], "the mask is not an answer");
    assert_eq!(r.plan_id, plan.plan_id);
    assert_eq!(r.method, mref(ORDER_A));
    assert_eq!(r.frame, REVS);
}

#[test]
fn changing_the_recipe_changes_the_answer() {
    let f = Fixture::new();
    let data = Data::new(N);
    let over_b = data.masked_sum(&data.b, 5);
    let over_a = data.masked_sum(&data.a, 5);
    assert_ne!(over_a, over_b, "the fixture must tell the recipes apart");
    let a = exec(
        &f,
        &f.prepare(SUM_A).unwrap(),
        &data,
        &permit(&[SUM_A]),
        NOW,
    )
    .unwrap();
    assert_eq!(values(&a), vec![(4, over_a)]);
    let count = exec(
        &f,
        &f.prepare(COUNT).unwrap(),
        &data,
        &permit(&[COUNT]),
        NOW,
    )
    .unwrap();
    assert_eq!(values(&count), vec![(4, data.count_below(5))]);
    assert_ne!(
        f.prepare(SUM_A).unwrap().plan_id,
        f.prepare(ORDER_A).unwrap().plan_id
    );
}

#[test]
fn reordering_effects_changes_the_answer_as_the_oracle_predicts() {
    let f = Fixture::new();
    let data = Data::new(N);
    let s = data.masked_sum(&data.b, 5);
    let a = exec(
        &f,
        &f.prepare(ORDER_A).unwrap(),
        &data,
        &permit(&[ORDER_A]),
        NOW,
    )
    .unwrap();
    let b = exec(
        &f,
        &f.prepare(ORDER_B).unwrap(),
        &data,
        &permit(&[ORDER_B]),
        NOW,
    )
    .unwrap();
    assert_eq!(values(&a), vec![(4, 2 * s), (5, s + 1)]);
    assert_eq!(values(&b), vec![(4, 2 * s), (5, 2 * s + 1)]);
    let right = exec(
        &f,
        &f.prepare(SUB_RIGHT).unwrap(),
        &data,
        &permit(&[SUB_RIGHT]),
        NOW,
    )
    .unwrap();
    let left = exec(
        &f,
        &f.prepare(SUB_LEFT).unwrap(),
        &data,
        &permit(&[SUB_LEFT]),
        NOW,
    )
    .unwrap();
    assert_eq!(values(&right), vec![(4, s), (5, s - 7)]);
    assert_eq!(values(&left), vec![(4, s), (5, 7 - s)]);
}

#[test]
fn a_word_binds_by_its_label_not_its_number() {
    // `SUM` sits at lens position 0, far from its opcode number; the plan
    // still binds the opcode OGAR names `SUM`.
    let f = Fixture::new();
    let plan = f.prepare(ORDER_A).unwrap();
    assert_eq!(plan.steps[0].opcode, INT_S_LESS);
    assert_eq!(R2ILVocabulary.name(plan.steps[1].opcode), Some("SUM"));
    assert_ne!(plan.steps[1].opcode.0, W_SUM);
}

#[test]
fn a_prepared_plan_runs_without_the_class_view() {
    let f = Fixture::new();
    let plan = f.prepare(ORDER_A).unwrap();
    // `run` takes no ClassView: the views could be dropped here.
    let data = Data::new(N);
    assert!(exec(&f, &plan, &data, &permit(&[ORDER_A]), NOW).is_ok());
}

#[test]
fn execution_needs_a_permit_beyond_admission() {
    let f = Fixture::new();
    let data = Data::new(N);
    let plan = f.prepare(ORDER_A).unwrap();
    assert!(f.store.admit(&mref(ORDER_A)).is_ok());
    assert_eq!(
        exec(&f, &plan, &data, &ExecPermit::default(), NOW),
        Err(ExecRefusal::NotPermitted)
    );
    assert_eq!(
        exec(&f, &plan, &data, &permit(&[ORDER_B]), NOW),
        Err(ExecRefusal::NotPermitted)
    );
}

#[test]
fn a_moved_frame_focus_or_scope_is_never_reused() {
    let mut f = Fixture::new();
    let data = Data::new(N);
    let plan = f.prepare(ORDER_A).unwrap();
    let p = permit(&[ORDER_A]);
    assert_eq!(
        exec(&f, &plan, &data, &p, (Focus { revision: 2 }, SCOPE)),
        Err(ExecRefusal::Guards(GuardRefusal::FocusChanged))
    );
    assert_eq!(
        exec(&f, &plan, &data, &p, (FOCUS, Scope { id: 2 })),
        Err(ExecRefusal::Guards(GuardRefusal::ScopeChanged))
    );
    f.frames.advance(FrameRevisions {
        coordinates: 2,
        ..REVS
    });
    assert_eq!(
        exec(&f, &plan, &data, &p, NOW),
        Err(ExecRefusal::Guards(GuardRefusal::Frame(
            FrameRefusal::CoordinatesChanged
        )))
    );
}

#[test]
fn a_changed_route_is_never_reused() {
    let mut f = Fixture::new();
    let data = Data::new(N);
    let plan = f.prepare(ORDER_A).unwrap();
    f.store.route(POSITION, OTHER_CLASS).unwrap();
    assert_eq!(
        exec(&f, &plan, &data, &permit(&[ORDER_A]), NOW),
        Err(ExecRefusal::RouteChanged {
            position: POSITION,
            captured: Some(CLASS),
            now: Some(OTHER_CLASS),
            class: CLASS,
        })
    );
}

#[test]
fn a_wrong_signature_or_dropped_guard_refuses() {
    let f = Fixture::new();
    let g = guards(&f.frames);
    let res = |r: &MethodRef| {
        prepare(
            &f.store,
            &f.views,
            R2IL,
            &Resolution {
                method: r,
                schema_rev: REV,
                route: (f.version, POSITION),
                guards: &g,
            },
            &f.frames,
            NOW,
        )
        .map(|_| ())
    };
    let mut wrong_sig = mref(ORDER_A);
    wrong_sig.sig_rev = SigRev(9);
    assert!(matches!(
        res(&wrong_sig),
        Err(ExecRefusal::Binding(
            Refusal::SignatureRevisionMismatch { .. }
        ))
    ));
    assert!(matches!(
        res(&mref(GUARDED)),
        Err(ExecRefusal::Binding(Refusal::GuardDropped { .. }))
    ));
}

#[test]
fn what_the_reading_cannot_honour_is_refused_at_prepare() {
    let f = Fixture::new();
    // The guard is kept (the reference witnesses it), so admit passes, and
    // the reading still refuses: it does not evaluate WHEN.
    let guarded = MethodRef::of(CLASS, &methods()[10]);
    assert_eq!(guarded.method, MethodId(GUARDED));
    let g = guards(&f.frames);
    let res = Resolution {
        method: &guarded,
        schema_rev: REV,
        route: (f.version, POSITION),
        guards: &g,
    };
    assert!(f.store.admit(&guarded).is_ok());
    assert_eq!(
        prepare(&f.store, &f.views, R2IL, &res, &f.frames, NOW).map(|_| ()),
        Err(ExecRefusal::GuardNotExecutable)
    );
    assert_eq!(
        f.prepare(THREE_VALUED).map(|_| ()),
        Err(ExecRefusal::ThreeValuedNotExecutable)
    );
    assert!(matches!(
        f.prepare(BAD_WORD),
        Err(ExecRefusal::UnknownWord { effect: 0, .. })
    ));
    assert!(matches!(
        f.prepare(SINK_WORD),
        Err(ExecRefusal::NotExecutable { effect: 0, .. })
    ));
}

#[test]
fn a_word_under_another_lens_or_reading_refuses() {
    let views = Views::new();
    let (mut store, _) = store(ogar_arities());
    let other = method(
        1,
        &[0],
        vec![Effect {
            target: 3,
            algebra: w(OTHER_LENS, W_LT),
            args: vec![Arg::Field(0), Arg::Const(5)],
        }],
    );
    let class = THIRD_CLASS;
    store
        .register(class, schema(), vec![other.clone()])
        .unwrap();
    let v = store.route(9, class).unwrap();
    let frames = FrameStore::new(REVS);
    let g = guards(&frames);
    let r = MethodRef::of(class, &other);
    let res = Resolution {
        method: &r,
        schema_rev: REV,
        route: (v, 9),
        guards: &g,
    };
    assert_eq!(
        prepare(&store, &views, R2IL, &res, &frames, NOW).map(|_| ()),
        Err(ExecRefusal::WrongLens {
            effect: 0,
            lens: OTHER_LENS
        })
    );
    let not_r2il = R2ilLens {
        lens: OTHER_LENS,
        schema_rev: REV,
    };
    assert_eq!(
        prepare(&store, &views, not_r2il, &res, &frames, NOW).map(|_| ()),
        Err(ExecRefusal::NotR2ilReading { lens: OTHER_LENS })
    );
}

#[test]
fn an_arity_the_lens_declares_against_ogar_refuses() {
    // The lens claims SUM takes one operand; OGAR says two.
    let mut arities = ogar_arities();
    arities[usize::from(W_SUM)] = 1;
    let views = Views::new();
    let mut store = BindStore::new();
    store
        .declare_algebra(LENS, REV, Extent::DR64_CUBE, arities)
        .unwrap();
    let m = method(1, &[1], vec![step(4, W_SUM, vec![Arg::Field(1)])]);
    store.register(CLASS, schema(), vec![m.clone()]).unwrap();
    let v = store.route(POSITION, CLASS).unwrap();
    let frames = FrameStore::new(REVS);
    let g = guards(&frames);
    let r = MethodRef::of(CLASS, &m);
    let res = Resolution {
        method: &r,
        schema_rev: REV,
        route: (v, POSITION),
        guards: &g,
    };
    assert_eq!(
        prepare(&store, &views, R2IL, &res, &frames, NOW).map(|_| ()),
        Err(ExecRefusal::Arity {
            effect: 0,
            ogar: Some(2),
            declared: Some(1),
            args: 1
        })
    );
}

#[test]
fn operands_that_do_not_translate_faithfully_refuse_at_run() {
    let f = Fixture::new();
    let data = Data::new(N);
    let go = |id| exec(&f, &f.prepare(id).unwrap(), &data, &permit(&[id]), NOW);
    assert!(matches!(
        go(WRONG_KIND),
        Err(ExecRefusal::Fold {
            effect: 0,
            error: FoldError::WrongOperandKind(_)
        })
    ));
    assert_eq!(
        go(WIDE_CONST),
        Err(ExecRefusal::ConstOutOfRange {
            effect: 0,
            value: 1 << 40
        })
    );
    assert_eq!(
        go(UNBOUND),
        Err(ExecRefusal::FieldUnbound {
            effect: 0,
            field: 7
        })
    );
}

#[test]
fn work_per_run_does_not_grow_with_rows() {
    let f = Fixture::new();
    let plan = f.prepare(ORDER_A).unwrap();
    let p = permit(&[ORDER_A]);
    let small = exec(&f, &plan, &Data::new(200), &p, NOW).unwrap();
    let large = exec(&f, &plan, &Data::new(200_000), &p, NOW).unwrap();
    // One mask-risc program per fold step (SUM), whatever the row count.
    assert_eq!(small.programs_run, 1);
    assert_eq!(large.programs_run, 1);
    assert_eq!(small.steps_executed, large.steps_executed);
    let s = Data::new(200_000);
    assert_eq!(
        values(&large),
        vec![
            (4, 2 * s.masked_sum(&s.b, 5)),
            (5, s.masked_sum(&s.b, 5) + 1)
        ]
    );
}

#[test]
fn an_and_operand_read_again_keeps_its_own_mask() {
    // `t` is the right operand of `u = m AND t` and is read again by
    // `POP_COUNT(t)`. Gating `t`'s predicate under `m` in place would make the
    // count `count_below(5)` instead of `count_below(10)`.
    let f = Fixture::new();
    let data = Data::new(64);
    assert_ne!(
        data.count_below(5),
        data.count_below(10),
        "the fixture separates the two"
    );
    let plan = f.prepare(SHARED_AND).unwrap();
    let r = exec(&f, &plan, &data, &permit(&[SHARED_AND]), NOW).unwrap();
    assert_eq!(values(&r), vec![(4, data.count_below(10))]);
}

#[test]
fn a_run_leaves_no_pending_predicates_in_the_dialect() {
    // A plan whose predicates no fold consumes must not leave their ops in the
    // caller's dialect: after more such runs than there are scratch slots, a
    // folding plan on the same dialect still runs and counts only its own mask.
    let f = Fixture::new();
    let data = Data::new(64);
    let pred_only = f.prepare(PRED_ONLY).unwrap();
    let count = f.prepare(COUNT).unwrap();
    let ok = permit(&[PRED_ONLY, COUNT]);
    let lanes = [
        LaneRef::I32(&data.a),
        LaneRef::I32(&data.b),
        LaneRef::U32(&data.k),
    ];
    let planes = Planes {
        n_rows: data.a.len(),
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign::NONE;
    let mut d = FoldDialect::new(&planes, &foreign);
    let fl = FieldLanes { lanes: &LANES };
    for _ in 0..=lance_graph_r2il::MAX_SLOTS {
        let r = run(&pred_only, &f.store, &f.frames, NOW, &ok, fl, &mut d).unwrap();
        assert_eq!(r.programs_run, 0);
    }
    let r = run(&count, &f.store, &f.frames, NOW, &ok, fl, &mut d).unwrap();
    assert_eq!(values(&r), vec![(4, data.count_below(5))]);
    assert_eq!(
        d.last_program().unwrap().ops.len(),
        1,
        "only COUNT's own predicate"
    );
}
