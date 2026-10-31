//! The prepared native form of an R1 plan (`binding_native`).
//!
//! Every native answer is compared with what `binding::run` returns for the
//! same plan over the same lanes, and with a scalar oracle. The fixture mirrors
//! `binding.rs`'s (same lab classids, same lens construction), with its own
//! method set.
//!
//! This binary installs a counting global allocator. The count is per thread,
//! so tests running in parallel do not disturb each other.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{Focus, FrameRevisions, FrameStore, GuardRefusal, Guards, Scope};
use lance_graph_dt64::{
    AlgebraRef, Arg, BindStore, ClassSchema, Effect, Extent, MethodBinding, MethodId, MethodRef,
    RouteVersion, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mask_risc::{ExecError, Foreign, LaneRef, Planes};
use lance_graph_r2il::binding::{
    prepare, run, ExecPermit, ExecRefusal, FieldLanes, Prepared, R2ilLens, Receipt, Resolution,
};
use lance_graph_r2il::binding_native::{
    prepare_native, run_native, NativePlan, NativeReceipt, NativeScratch, MAX_NATIVE_OUTPUTS,
};
use lance_graph_r2il::{FoldDialect, FoldError, INT_ADD, INT_AND, INT_S_LESS, POP_COUNT};
use ogar_loco::{FnIndex, Vocabulary};
use ogar_r2il::{R2ILVocabulary, ANY, CONCEPT_R2IL_FOLD, MAX, MIN, SUM};

// ── counting allocator ─────────────────────────────────────────────────────

struct Counting;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: forwards every call to `System` unchanged; only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.with(Cell::get)
}

// ── fixture ────────────────────────────────────────────────────────────────

const CLASS: u32 = 0x7F0B_0001;
const CLASS_VIEW: ClassId = 0x7F0B;
const OTHER_CLASS: u32 = 0x7F0C_0001;
const LENS: u32 = (CONCEPT_R2IL_FOLD as u32) << 16;
const REV: SchemaRev = SchemaRev(1);
const R2IL: R2ilLens = R2ilLens {
    lens: LENS,
    schema_rev: REV,
};

/// Two I32 lanes, one U32 lane, then nine computed fields.
const FIELDS: [&str; 12] = [
    "a", "b", "k", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11",
];
const LANES: [Option<u16>; 12] = [
    Some(0),
    Some(1),
    Some(2),
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
];

/// The lens's words, in this order; only the label binds a word to an opcode.
const W_LT: u8 = 0;
const W_SUM: u8 = 1;
const W_POP: u8 = 2;
const W_ADD: u8 = 3;
const W_MIN: u8 = 4;
const W_MAX: u8 = 5;
const W_ANY: u8 = 6;
const W_AND: u8 = 7;
const WORD_OPS: [FnIndex; 8] = [INT_S_LESS, SUM, POP_COUNT, INT_ADD, MIN, MAX, ANY, INT_AND];

fn words() -> Vec<&'static str> {
    WORD_OPS
        .iter()
        .map(|&f| R2ILVocabulary.name(f).expect("OGAR names the opcode"))
        .collect()
}

const FOCUS: Focus = Focus { revision: 1 };
const SCOPE: Scope = Scope { id: 1 };
const NOW: (Focus, Scope) = (FOCUS, SCOPE);
const POSITION: u16 = 3;
const REVS: FrameRevisions = FrameRevisions {
    source: 1,
    relation_map: 1,
    coordinates: 1,
};

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

impl ClassView for Views {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        match class {
            CLASS_VIEW => &self.class,
            c if c == CONCEPT_R2IL_FOLD => &self.lens,
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

fn ogar_arities() -> Vec<u8> {
    WORD_OPS
        .iter()
        .map(|&f| R2ILVocabulary.stack_arity(f).expect("OGAR gives an arity"))
        .collect()
}

fn step(target: u8, word: u8, args: Vec<Arg>) -> Effect {
    Effect {
        target,
        algebra: AlgebraRef { lens: LENS, word },
        args,
    }
}

fn method(id: u32, effects: Vec<Effect>) -> MethodBinding {
    let reads = WideFieldMask::from_positions(&[0, 1]);
    MethodBinding {
        id: MethodId(id),
        sig_rev: SigRev(1),
        name: format!("m{id}"),
        on: reads.clone(),
        when: None,
        reads,
        effects,
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

fn lt(target: u8, t: i64) -> Effect {
    step(target, W_LT, vec![Arg::Field(0), Arg::Const(t)])
}

const MASKED_SUM: u32 = 1; // SUM(b) where a < 5
const SUM_A: u32 = 2; // SUM(a) where a < 5
const COUNT: u32 = 3; // POP_COUNT(a < 5)
const TWO_FOLD: u32 = 4; // COUNT into 4, then SUM(b) into 6
const TWO_FOLD_REV: u32 = 5; // SUM(b) into 4, then COUNT into 6
const MIN_MAX: u32 = 6; // MIN(b) into 4, MAX(b) into 6, each under a < 5
const EMPTY_MIN: u32 = 7; // MIN(b) under a mask that selects nothing
const ANY_ROW: u32 = 8; // ANY(a < 5)
const ORDER_A: u32 = 9; // a scalar computed from a fold result
const SUB_RIGHT: u32 = 10; // likewise, subtraction
const CONST_AFTER: u32 = 11; // a constant-only scalar after a fold
const FEEDS: u32 = 12; // a compare constant that is a fold result
const FEEDS_OK: u32 = 13; // the same shape with a literal constant
const NINE_OUT: u32 = 14; // nine scalar fields
const EIGHT_OUT: u32 = 15; // eight scalar fields
const SHARED_AND: u32 = 16; // f3 = a < 5; f5 = a < 10; f6 = f3 AND f5; f4 = POP_COUNT(f5)

fn methods() -> Vec<MethodBinding> {
    let masked_sum = || vec![lt(3, 5), step(4, W_SUM, vec![Arg::Field(3), Arg::Field(1)])];
    let consts = |n: u8| -> Vec<Effect> {
        (0..n)
            .map(|i| step(i, W_ADD, vec![Arg::Const(i64::from(i)), Arg::Const(100)]))
            .collect()
    };
    let with = |mut base: Vec<Effect>, extra: Vec<Effect>| {
        base.extend(extra);
        base
    };
    vec![
        method(MASKED_SUM, masked_sum()),
        method(
            SUM_A,
            vec![lt(3, 5), step(4, W_SUM, vec![Arg::Field(3), Arg::Field(0)])],
        ),
        method(COUNT, vec![lt(3, 5), step(4, W_POP, vec![Arg::Field(3)])]),
        method(
            TWO_FOLD,
            vec![
                lt(3, 5),
                step(4, W_POP, vec![Arg::Field(3)]),
                lt(5, 5),
                step(6, W_SUM, vec![Arg::Field(5), Arg::Field(1)]),
            ],
        ),
        method(
            TWO_FOLD_REV,
            vec![
                lt(3, 5),
                step(4, W_SUM, vec![Arg::Field(3), Arg::Field(1)]),
                lt(5, 5),
                step(6, W_POP, vec![Arg::Field(5)]),
            ],
        ),
        method(
            MIN_MAX,
            vec![
                lt(3, 5),
                step(4, W_MIN, vec![Arg::Field(3), Arg::Field(1)]),
                lt(5, 5),
                step(6, W_MAX, vec![Arg::Field(5), Arg::Field(1)]),
            ],
        ),
        method(
            EMPTY_MIN,
            vec![
                lt(3, -1000),
                step(4, W_MIN, vec![Arg::Field(3), Arg::Field(1)]),
            ],
        ),
        method(ANY_ROW, vec![lt(3, 5), step(4, W_ANY, vec![Arg::Field(3)])]),
        method(
            ORDER_A,
            with(
                masked_sum(),
                vec![
                    step(5, W_ADD, vec![Arg::Field(4), Arg::Const(1)]),
                    step(4, W_ADD, vec![Arg::Field(4), Arg::Field(4)]),
                ],
            ),
        ),
        method(
            SUB_RIGHT,
            with(
                masked_sum(),
                vec![step(5, W_ADD, vec![Arg::Const(7), Arg::Field(4)])],
            ),
        ),
        method(
            CONST_AFTER,
            with(
                masked_sum(),
                vec![step(5, W_ADD, vec![Arg::Const(2), Arg::Const(3)])],
            ),
        ),
        method(
            FEEDS,
            vec![
                lt(3, 5),
                step(4, W_POP, vec![Arg::Field(3)]),
                step(5, W_LT, vec![Arg::Field(1), Arg::Field(4)]),
                step(6, W_SUM, vec![Arg::Field(5), Arg::Field(0)]),
            ],
        ),
        method(
            FEEDS_OK,
            vec![
                lt(3, 5),
                step(4, W_POP, vec![Arg::Field(3)]),
                step(5, W_LT, vec![Arg::Field(1), Arg::Const(5)]),
                step(6, W_SUM, vec![Arg::Field(5), Arg::Field(0)]),
            ],
        ),
        method(NINE_OUT, consts(9)),
        method(EIGHT_OUT, consts(8)),
        method(
            SHARED_AND,
            vec![
                lt(3, 5),
                lt(5, 10),
                step(6, W_AND, vec![Arg::Field(3), Arg::Field(5)]),
                step(4, W_POP, vec![Arg::Field(5)]),
            ],
        ),
    ]
}

fn mref(id: u32) -> MethodRef {
    MethodRef {
        class: CLASS,
        method: MethodId(id),
        sig_rev: SigRev(1),
        guard: None,
    }
}

fn permit(ids: &[u32]) -> ExecPermit {
    ExecPermit {
        methods: ids.iter().map(|&i| mref(i)).collect(),
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
    fn lanes(&self) -> [LaneRef<'_>; 3] {
        [
            LaneRef::I32(&self.a),
            LaneRef::I32(&self.b),
            LaneRef::U32(&self.k),
        ]
    }
    fn selected(&self, t: i32) -> impl Iterator<Item = usize> + '_ {
        (0..self.a.len()).filter(move |&i| self.a[i] < t)
    }
    fn sum_b(&self, t: i32) -> i64 {
        self.selected(t).map(|i| i64::from(self.b[i])).sum()
    }
    fn sum_a(&self, t: i32) -> i64 {
        self.selected(t).map(|i| i64::from(self.a[i])).sum()
    }
    fn count(&self, t: i32) -> i64 {
        self.selected(t).count() as i64
    }
    fn min_b(&self, t: i32) -> Option<i64> {
        self.selected(t).map(|i| i64::from(self.b[i])).min()
    }
    fn max_b(&self, t: i32) -> Option<i64> {
        self.selected(t).map(|i| i64::from(self.b[i])).max()
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
        let mut store = BindStore::new();
        store
            .declare_algebra(LENS, REV, Extent::DR64_CUBE, ogar_arities())
            .unwrap();
        let schema = ClassSchema {
            schema_rev: REV,
            field_count: FIELDS.len(),
            extent: Extent::DR64_CUBE,
        };
        store.register(CLASS, schema, methods()).unwrap();
        store.register(OTHER_CLASS, schema, vec![]).unwrap();
        let version = store.route(POSITION, CLASS).unwrap();
        Self {
            views: Views {
                class: refs(&FIELDS, "class"),
                lens: refs(&words(), "r2il"),
            },
            store,
            version,
            frames: FrameStore::new(REVS),
        }
    }

    fn prepare(&self, id: u32) -> Prepared {
        let r = mref(id);
        let g = Guards {
            frame: self.frames.seal(),
            focus: FOCUS,
            scope: SCOPE,
        };
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
        .expect("the fixture's methods prepare")
    }

    fn r1(
        &self,
        plan: &Prepared,
        data: &Data,
        permit: &ExecPermit,
        now: (Focus, Scope),
    ) -> Result<Receipt, ExecRefusal> {
        let lanes = data.lanes();
        let planes = Planes {
            n_rows: data.a.len(),
            masks: &[],
            lanes: &lanes,
        };
        let foreign = Foreign::NONE;
        let mut d = FoldDialect::new(&planes, &foreign);
        run(
            plan,
            &self.store,
            &self.frames,
            now,
            permit,
            FieldLanes { lanes: &LANES },
            &mut d,
        )
    }

    fn native(&self, plan: &Prepared, data: &Data) -> Result<NativePlan, ExecRefusal> {
        let lanes = data.lanes();
        let planes = Planes {
            n_rows: data.a.len(),
            masks: &[],
            lanes: &lanes,
        };
        prepare_native(
            plan,
            &self.store,
            &self.frames,
            NOW,
            FieldLanes { lanes: &LANES },
            &planes,
        )
    }

    fn run_native(
        &self,
        np: &NativePlan,
        data: &Data,
        permit: &ExecPermit,
        now: (Focus, Scope),
        scratch: &mut NativeScratch,
    ) -> Result<NativeReceipt, ExecRefusal> {
        let lanes = data.lanes();
        let planes = Planes {
            n_rows: data.a.len(),
            masks: &[],
            lanes: &lanes,
        };
        run_native(np, &self.store, &self.frames, now, permit, &planes, scratch)
    }
}

const N: usize = 1_000;

/// Prepare natively and run once, over `data`.
fn native_answer(f: &Fixture, id: u32, data: &Data) -> NativeReceipt {
    let plan = f.prepare(id);
    let np = f.native(&plan, data).expect("prepare_native");
    let mut scratch = NativeScratch::for_plan(&np);
    f.run_native(&np, data, &permit(&[id]), NOW, &mut scratch)
        .expect("run_native")
}

// ── agreement with `run` ───────────────────────────────────────────────────

// Fails if `run_native` returns a different (field, value) list, programs_run,
// plan id, frame or intermediate list than `run` for any plan whose folds are
// independent, or if the fixture stops selecting rows (the comparison would
// then hold for an all-zero answer).
#[test]
fn native_agrees_with_run_on_every_runnable_fixture() {
    let f = Fixture::new();
    let data = Data::new(N);
    assert!(data.sum_b(5) > 0 && data.count(5) > 0 && data.min_b(5) != data.max_b(5));
    let ids = [
        MASKED_SUM,
        SUM_A,
        COUNT,
        TWO_FOLD,
        TWO_FOLD_REV,
        MIN_MAX,
        EMPTY_MIN,
        ANY_ROW,
        CONST_AFTER,
        FEEDS_OK,
        EIGHT_OUT,
    ];
    for id in ids {
        let plan = f.prepare(id);
        let want = f.r1(&plan, &data, &permit(&[id]), NOW).expect("run");
        let np = f.native(&plan, &data).expect("prepare_native");
        let mut scratch = NativeScratch::for_plan(&np);
        let got = f
            .run_native(&np, &data, &permit(&[id]), NOW, &mut scratch)
            .expect("run_native");
        assert_eq!(got.values(), want.output.values.as_slice(), "method {id}");
        assert_eq!(got.programs_run, want.programs_run, "method {id}");
        assert_eq!(np.program_count(), want.programs_run, "method {id}");
        assert_eq!(got.plan_id, want.plan_id, "method {id}");
        assert_eq!(got.frame, want.frame, "method {id}");
        assert_eq!(np.intermediates(), want.output.intermediates, "method {id}");
        assert_eq!(np.outputs(), want.output.values.len(), "method {id}");
    }
}

// Fails if the answers are not the oracle's (a native answer equal to `run`'s
// but both wrong would pass the test above).
#[test]
fn native_answers_match_the_oracle() {
    let f = Fixture::new();
    let data = Data::new(N);
    let v = |id| native_answer(&f, id, &data).values().to_vec();
    assert_eq!(v(MASKED_SUM), vec![(4, data.sum_b(5))]);
    assert_eq!(v(SUM_A), vec![(4, data.sum_a(5))]);
    assert_eq!(v(COUNT), vec![(4, data.count(5))]);
    assert_eq!(v(ANY_ROW), vec![(4, 1)]);
    assert_eq!(
        v(MIN_MAX),
        vec![(4, data.min_b(5).unwrap()), (6, data.max_b(5).unwrap())]
    );
    assert_ne!(data.sum_a(5), data.sum_b(5), "the recipes must differ");
}

// Fails if an empty MIN is reported as anything but the dialect's empty value.
#[test]
fn an_empty_min_keeps_the_dialects_empty_value() {
    let f = Fixture::new();
    let data = Data::new(N);
    assert_eq!(data.count(-1000), 0, "the mask must select nothing");
    assert_eq!(
        native_answer(&f, EMPTY_MIN, &data).values(),
        &[(4, i64::MAX)]
    );
}

// Fails if programs do not run in step order or a program's result lands in
// the wrong field: the two plans swap which fold sits where.
#[test]
fn two_folds_run_in_step_order() {
    let f = Fixture::new();
    let data = Data::new(N);
    let (count, sum) = (data.count(5), data.sum_b(5));
    assert_ne!(count, sum);
    let a = native_answer(&f, TWO_FOLD, &data);
    let b = native_answer(&f, TWO_FOLD_REV, &data);
    assert_eq!(a.values(), &[(4, count), (6, sum)]);
    assert_eq!(b.values(), &[(4, sum), (6, count)]);
    assert_eq!((a.programs_run, b.programs_run), (2, 2));
    let np = f.native(&f.prepare(TWO_FOLD), &data).unwrap();
    assert_eq!(np.program_effects(), vec![1, 3]);
    assert_eq!(np.intermediates(), &[3, 5]);
}

// Fails if a constant-only scalar is not carried as a fixed value in its
// field's place, or if it displaces the fold result.
#[test]
fn a_constant_only_scalar_is_a_fixed_output() {
    let f = Fixture::new();
    let data = Data::new(N);
    assert_eq!(
        native_answer(&f, CONST_AFTER, &data).values(),
        &[(4, data.sum_b(5)), (5, 5)]
    );
}

// Fails if the hot path replays prepare-time data: the lanes change, the plan
// is the same, and the answer must follow the lanes.
#[test]
fn a_run_reads_the_lanes_it_is_given() {
    let f = Fixture::new();
    let first = Data::new(N);
    let mut second = Data::new(N);
    second.b.iter_mut().for_each(|x| *x += 1);
    assert_ne!(first.sum_b(5), second.sum_b(5));
    let plan = f.prepare(MASKED_SUM);
    let np = f.native(&plan, &first).unwrap();
    let mut scratch = NativeScratch::for_plan(&np);
    let p = permit(&[MASKED_SUM]);
    let r = f.run_native(&np, &second, &p, NOW, &mut scratch).unwrap();
    assert_eq!(r.values(), &[(4, second.sum_b(5))]);
}

// ── named refusals at prepare ──────────────────────────────────────────────

// Fails unless a scalar computed from a fold result is refused by name, with
// the index of the offending step. Silent twin: CONST_AFTER (a step that does
// not touch the fold result) and plain MASKED_SUM prepare.
#[test]
fn a_scalar_after_a_fold_is_refused() {
    let f = Fixture::new();
    let data = Data::new(N);
    for id in [ORDER_A, SUB_RIGHT] {
        assert!(
            matches!(
                f.native(&f.prepare(id), &data),
                Err(ExecRefusal::ScalarAfterFold { effect: 2 })
            ),
            "method {id}"
        );
    }
    for id in [MASKED_SUM, CONST_AFTER] {
        assert!(f.native(&f.prepare(id), &data).is_ok(), "method {id}");
    }
}

// Fails unless a fold fed by a value computed from an earlier fold is refused
// by name. Silent twin: the same plan with a literal constant prepares and
// matches `run`.
#[test]
fn a_fold_fed_by_a_fold_result_is_refused() {
    let f = Fixture::new();
    let data = Data::new(N);
    assert!(matches!(
        f.native(&f.prepare(FEEDS), &data),
        Err(ExecRefusal::FoldResultFeedsFold { effect: 3 })
    ));
    let plan = f.prepare(FEEDS_OK);
    assert!(f.native(&plan, &data).is_ok());
    // `run` accepts the refused plan: only the native reading cannot.
    assert!(f
        .r1(&f.prepare(FEEDS), &data, &permit(&[FEEDS]), NOW)
        .is_ok());
}

// Fails unless nine scalar fields are refused and eight are not.
#[test]
fn too_many_outputs_are_refused() {
    let f = Fixture::new();
    let data = Data::new(N);
    assert_eq!(MAX_NATIVE_OUTPUTS, 8);
    assert!(matches!(
        f.native(&f.prepare(NINE_OUT), &data),
        Err(ExecRefusal::TooManyOutputs { outputs: 9 })
    ));
    let eight = f.native(&f.prepare(EIGHT_OUT), &data).unwrap();
    assert_eq!(eight.outputs(), 8);
    assert_eq!(eight.program_count(), 0);
}

// ── refusals before the first program ──────────────────────────────────────

// Fails if `run_native` answers a refusal `run` raises before its first step
// with anything other than the same refusal, or checks them in another order.
#[test]
fn run_native_refuses_what_run_refuses_first() {
    let mut f = Fixture::new();
    let data = Data::new(N);
    let id = MASKED_SUM;
    let plan = f.prepare(id);
    let np = f.native(&plan, &data).unwrap();
    let mut s = NativeScratch::for_plan(&np);
    let p = permit(&[id]);
    let none = ExecPermit::default();
    let moved_focus = (Focus { revision: 2 }, SCOPE);
    let moved_scope = (FOCUS, Scope { id: 2 });
    for (now, pm) in [
        (NOW, &none),
        (moved_focus, &p),
        (moved_scope, &p),
        (moved_focus, &none),
    ] {
        let want = f.r1(&plan, &data, pm, now).unwrap_err();
        let got = f.run_native(&np, &data, pm, now, &mut s).unwrap_err();
        assert_eq!(got, want);
    }
    assert_eq!(
        f.run_native(&np, &data, &none, NOW, &mut s),
        Err(ExecRefusal::NotPermitted)
    );
    assert_eq!(
        f.run_native(&np, &data, &p, moved_focus, &mut s),
        Err(ExecRefusal::Guards(GuardRefusal::FocusChanged))
    );
    // Silent twin: with everything in place the same call runs.
    assert!(f.run_native(&np, &data, &p, NOW, &mut s).is_ok());

    let revs = FrameRevisions {
        coordinates: 2,
        ..REVS
    };
    f.frames.advance(revs);
    let want = f.r1(&plan, &data, &p, NOW).unwrap_err();
    assert!(matches!(want, ExecRefusal::Guards(GuardRefusal::Frame(_))));
    assert_eq!(f.run_native(&np, &data, &p, NOW, &mut s), Err(want));
}

// Fails if a route moved after prepare is reused.
#[test]
fn run_native_refuses_a_changed_route() {
    let mut f = Fixture::new();
    let data = Data::new(N);
    let plan = f.prepare(MASKED_SUM);
    let np = f.native(&plan, &data).unwrap();
    let mut s = NativeScratch::for_plan(&np);
    f.store.route(POSITION, OTHER_CLASS).unwrap();
    let p = permit(&[MASKED_SUM]);
    let want = f.r1(&plan, &data, &p, NOW).unwrap_err();
    assert_eq!(
        want,
        ExecRefusal::RouteChanged {
            position: POSITION,
            captured: Some(CLASS),
            now: Some(OTHER_CLASS),
            class: CLASS,
        }
    );
    assert_eq!(f.run_native(&np, &data, &p, NOW, &mut s), Err(want));
}

// Fails unless a change in the planes' shape (rows, or lane count) is
// refused by name. Silent twin: the same shape with other content runs
// (`a_run_reads_the_lanes_it_is_given`).
#[test]
fn run_native_refuses_changed_planes() {
    let f = Fixture::new();
    let data = Data::new(N);
    let np = f.native(&f.prepare(MASKED_SUM), &data).unwrap();
    let mut s = NativeScratch::for_plan(&np);
    let p = permit(&[MASKED_SUM]);
    let shorter = Data::new(N / 2);
    assert_eq!(
        f.run_native(&np, &shorter, &p, NOW, &mut s),
        Err(ExecRefusal::PlanesChanged {
            recorded: (N, 3, 0),
            now: (N / 2, 3, 0),
        })
    );
    let lanes = [LaneRef::I32(&data.a), LaneRef::I32(&data.b)];
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    assert_eq!(
        run_native(&np, &f.store, &f.frames, NOW, &p, &planes, &mut s),
        Err(ExecRefusal::PlanesChanged {
            recorded: (N, 3, 0),
            now: (N, 2, 0),
        })
    );
}

// Fails if a scratch that is too small panics or is accepted instead of being
// reported as the mask-risc scratch refusal at the program's step.
#[test]
fn a_scratch_from_another_plan_is_refused() {
    let f = Fixture::new();
    let data = Data::new(N);
    let np = f.native(&f.prepare(MASKED_SUM), &data).unwrap();
    let no_programs = f.native(&f.prepare(EIGHT_OUT), &data).unwrap();
    let mut small = NativeScratch::for_plan(&no_programs);
    assert!(np.scratch_words() > no_programs.scratch_words());
    let r = f.run_native(&np, &data, &permit(&[MASKED_SUM]), NOW, &mut small);
    assert!(matches!(
        r,
        Err(ExecRefusal::Fold {
            effect: 1,
            error: FoldError::Exec(ExecError::ScratchBufferTooSmall { .. })
        })
    ));
}

// ── allocation ─────────────────────────────────────────────────────────────

// Fails if `run_native` allocates. The twin proves the counter can fire: the
// interpreting `run` allocates on every call.
#[test]
fn run_native_allocates_nothing() {
    let f = Fixture::new();
    let data = Data::new(N);
    let lanes = data.lanes();
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    for id in [MASKED_SUM, TWO_FOLD, MIN_MAX, CONST_AFTER] {
        let plan = f.prepare(id);
        let np = f.native(&plan, &data).unwrap();
        let mut scratch = NativeScratch::for_plan(&np);
        let p = permit(&[id]);
        let want = native_answer(&f, id, &data);
        for _ in 0..16 {
            black_box(
                run_native(&np, &f.store, &f.frames, NOW, &p, &planes, &mut scratch).unwrap(),
            );
        }
        let before = allocs();
        let mut last = None;
        for _ in 0..1_000 {
            last = Some(black_box(run_native(
                black_box(&np),
                &f.store,
                &f.frames,
                NOW,
                &p,
                &planes,
                &mut scratch,
            )));
        }
        let spent = allocs() - before;
        assert_eq!(spent, 0, "method {id} allocated {spent} times");
        assert_eq!(last.unwrap().unwrap(), want, "method {id}");

        let before = allocs();
        f.r1(&plan, &data, &p, NOW).unwrap();
        assert!(allocs() > before, "the counter must see `run` allocate");
    }
}

// Fails if preparing `f6 = f3 AND f5` gates `f5`'s predicate under `f3` in
// place: the later `POP_COUNT(f5)` would then count `a < 5`, not `a < 10`.
#[test]
fn native_keeps_an_and_operand_that_is_read_again() {
    let f = Fixture::new();
    let data = Data::new(N);
    assert_ne!(
        data.count(5),
        data.count(10),
        "the fixture separates the two"
    );
    assert_eq!(
        native_answer(&f, SHARED_AND, &data).values(),
        &[(4, data.count(10))]
    );
}
