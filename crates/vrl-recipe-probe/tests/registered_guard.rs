//! A VRL recipe's aborts as the WHEN guard of a registered DT64 method.
//!
//! `r2il_parity.rs` registers a recipe as DO steps (frozen F000 words, no
//! guard). This is the other half: the aborts are the guard the class is
//! bound with, under Kleene logic, and the caller names only the method and
//! the route. The guard and its logic are read from the admitted binding; its
//! fields are requested as a field mask; lanes and presence planes are lent
//! by one `borrow_all_present`. The rows where the guard is TRUE must be the
//! rows the recipe's own program keeps.
//!
//! Not r2il's `prepare_gated` / `run_gated`: no DO step is run here, only the
//! guard. The evaluation is the one `lance-graph-r2il`'s integration test
//! uses for a hand-written guard (`kleene_true_rows`).

mod common;

use common::{all_fields, bind, witness, Source, View, CLASS, LABELS, VIEW_CLASS};
use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::set_algebra::{Cmp as SetCmp, Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{Frame, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, Guard, LaneRefusal, LaneRequest, MethodBinding,
    MethodId, Refusal as StoreRefusal, RouteAt, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mask_risc::{execute_into, Foreign, Out, Planes, Scratch, Value};
use lance_graph_quack::set::{lower_set, SetLowered};
use lance_graph_quack::{lower, Agg, Mask, Query};
use vrl_recipe_probe::{Cmp, Recipe};

const POSITION: u16 = 9;
const METHOD: MethodId = MethodId(1);
/// The reader scope and row space the fixture's source issues its lanes for.
const READER: Scope = Scope { id: 7 };
const ROWS: CoordSpace = CoordSpace {
    population: 1,
    coordinates: 1,
};

fn set_cmp(c: Cmp) -> SetCmp {
    match c {
        Cmp::Eq => SetCmp::Eq,
        Cmp::Ne => SetCmp::Ne,
        Cmp::Lt => SetCmp::Lt,
        Cmp::Le => SetCmp::Le,
        Cmp::Gt => SetCmp::Gt,
        Cmp::Ge => SetCmp::Ge,
    }
}

/// The rows a recipe keeps, over field positions: no abort term holds.
fn kept_guard(recipe: &Recipe) -> SetExpr {
    SetExpr::And(
        recipe
            .terms()
            .iter()
            .map(|&(f, c, v)| SetExpr::negated(SetExpr::i32(u16::from(f), set_cmp(c), v)))
            .collect(),
    )
}

/// The recipe as a method: its guard is the registration's WHEN, it reads the
/// fields the guard names, and it has no DO step.
fn method(recipe: &Recipe, logic: Logic) -> MethodBinding {
    let fields: Vec<u8> = recipe.terms().iter().map(|t| t.0).collect();
    let reads = WideFieldMask::from_positions(&fields);
    MethodBinding {
        id: METHOD,
        sig_rev: SigRev(1),
        name: "vrl aborts".to_string(),
        on: reads.clone(),
        when: Some(Guard::new(kept_guard(recipe), logic)),
        reads,
        effects: Vec::new(),
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

/// A store with the fixture's class registered together with `methods`.
fn registered(methods: Vec<MethodBinding>) -> (BindStore, View, RouteAt) {
    let mut store = BindStore::new();
    let view = View::new();
    store
        .register(
            &view,
            CLASS,
            ClassSchema {
                schema_rev: SchemaRev(1),
                field_count: LABELS.len(),
                extent: Extent::DR64_CUBE,
            },
            methods,
        )
        .unwrap();
    let version = store.route(POSITION, CLASS).unwrap();
    let route = RouteAt {
        position: POSITION,
        version,
    };
    (store, view, route)
}

/// How many rows the registered guard of `id` is TRUE for, under `frame`.
fn guard_true_rows(
    store: &BindStore,
    view: &View,
    route: RouteAt,
    source: &Source,
    frame: &Frame,
    id: MethodId,
) -> Result<u64, LaneRefusal> {
    let mref = store
        .method_ref(CLASS, id)
        .expect("the method is registered");
    let bound = store
        .admit(&mref)
        .expect("the store admits its own reference");
    let guard = bound.when.as_ref().expect("the method is guarded");
    let SetLowered::Filter(set) = lower_set(&guard.expr, guard.logic).unwrap() else {
        panic!("a guard of comparisons lowers to a filter")
    };
    let fields: Vec<u8> = set.lanes().iter().map(|&l| l as u8).collect();
    let binding = store.bind_lanes(
        view,
        &source.frames,
        source,
        LaneRequest {
            route,
            rev: SchemaRev(1),
            fields: &WideFieldMask::from_positions(&fields),
            frame,
            scope: READER,
            selection: ROWS,
        },
    )?;
    let (mut lanes, mut masks, mut nullable) = (Vec::new(), Vec::new(), Vec::new());
    for lent in binding.borrow_all_present(store, view, &source.frames, source)? {
        let (field, lane, plane) = lent?;
        nullable.push((u16::from(field), Mask(masks.len() as u16)));
        lanes.push(lane);
        masks.push(plane);
    }
    let filter = match guard.logic {
        Logic::Kleene => set.where_true(set.lanes(), &nullable).unwrap(),
        Logic::TwoValued => set.two_valued(set.lanes()).unwrap(),
    };
    let rows = source.lanes[0].len();
    let planes = Planes {
        n_rows: rows,
        masks: &masks,
        lanes: &lanes,
    };
    let program = lower(&Query {
        filter,
        agg: Agg::Count,
    })
    .unwrap();
    let mut scratch = Scratch::for_program(&program, rows).unwrap();
    match execute_into(&program, &planes, &Foreign::NONE, &mut scratch, Out::None).unwrap() {
        Value::Count(n) => Ok(n as u64),
        other => panic!("Count returned {other:?}"),
    }
}

/// About a fifth of status, region and amount absent: presence 0 and value 0.
fn source_with_absent(n: usize) -> Source {
    let mut source = Source::random(n, 31 + n as u64);
    for lane in 1..=3usize {
        source.set_absent(lane, (0..n).filter(|r| (r * 7 + lane * 3) % 5 == 0));
    }
    source
}

const PROGRAMS: &[&str] = &[
    ".a = .properties.amount.value\nif .a < 0 { abort }",
    ".a = .properties.amount.value\nif .properties.status.value != 2 { abort }",
    ".a = .properties.amount.value\nif .properties.region.value == 0 { abort }\nif .a >= 200 { abort }",
    ".r = .properties.region.value\n.a = .properties.amount.value\nif .properties.status.value > 2 { abort }\nif .a <= 10 { abort }\nif .r != 1 { abort }",
    ".a = .properties.amount.value\nif .key < 1010 { abort }\nif .key >= 1050 { abort }",
];

/// The registered Kleene guard is TRUE on exactly the rows the recipe keeps,
/// at widths 1..1000 with absent values, and an UNKNOWN row is in neither.
#[test]
fn the_registered_guard_keeps_the_rows_the_recipe_keeps() {
    let (mut unknown, mut kept, mut differs_two_valued) = (0u64, 0u64, 0u32);
    for &n in &[1usize, 63, 64, 65, 300, 1000] {
        let source = source_with_absent(n);
        let world = bind(&source, &all_fields());
        let w = witness(&world, &source);
        for src in PROGRAMS {
            let recipe = Recipe::compile(src, &View::new(), VIEW_CLASS).unwrap();
            let want = recipe.count(&w).unwrap();
            let frame = source.frames.seal();

            let (store, view, route) = registered(vec![method(&recipe, Logic::Kleene)]);
            let got = guard_true_rows(&store, &view, route, &source, &frame, METHOD).unwrap();
            assert_eq!(got, want, "{src} at {n}");

            // The same guard registered two-valued reads an absent value as 0.
            let (store, view, route) = registered(vec![method(&recipe, Logic::TwoValued)]);
            let flat = guard_true_rows(&store, &view, route, &source, &frame, METHOD).unwrap();
            differs_two_valued += u32::from(flat != want);

            kept += want;
            unknown += recipe.count_unknown(&w).unwrap();
        }
    }
    // Anti-vacuity: rows are kept, rows are unknown, and the logic the class
    // was bound with changes the answer.
    assert!(kept > 500, "kept {kept}");
    assert!(unknown > 100, "unknown {unknown}");
    assert!(
        differs_two_valued >= 10,
        "two-valued differs in {differs_two_valued} cases"
    );
}

/// The guard runs only under the frame its lanes were bound for.
#[test]
fn a_moved_frame_is_refused_before_the_guard_runs() {
    let mut source = source_with_absent(300);
    let recipe = Recipe::compile(PROGRAMS[0], &View::new(), VIEW_CLASS).unwrap();
    let (store, view, route) = registered(vec![method(&recipe, Logic::Kleene)]);
    let stale = source.frames.seal();
    assert!(guard_true_rows(&store, &view, route, &source, &stale, METHOD).is_ok());
    source.advance();
    assert!(guard_true_rows(&store, &view, route, &source, &stale, METHOD).is_err());
    let fresh = source.frames.seal();
    assert!(guard_true_rows(&store, &view, route, &source, &fresh, METHOD).is_ok());
}

/// A program that was never registered has no reference: it stays outside
/// the store, and nothing runs for it.
#[test]
fn an_unregistered_program_has_no_method_reference() {
    let (store, _view, _route) = registered(Vec::new());
    assert!(matches!(
        store.method_ref(CLASS, METHOD),
        Err(StoreRefusal::UnknownMethod { .. })
    ));
}

/// OPEN, pinned as measured: the two presence readings differ on one kind of
/// row. The recipe reads DR512 (EXO-294): a value is absent only with
/// presence 0 AND value 0, so presence 0 with a nonzero value is a value.
/// Quack's `where_true` reads the plane as SQL validity: presence 0 is NULL
/// whatever the lane holds. The main test avoids such rows; this one states
/// what happens on them, so the difference is not found by accident later.
#[test]
fn presence_zero_with_a_nonzero_value_is_read_differently_by_the_two_paths() {
    // Row 0: presence 0, value -4. Row 1: presence 0, value 0 (absent).
    let mut source = Source::from_lanes([vec![1, 2], vec![0; 2], vec![0; 2], vec![-4, 0]]);
    source.presence[3].as_mut().unwrap()[0] &= !0b11;
    let world = bind(&source, &all_fields());
    let w = witness(&world, &source);
    let recipe = Recipe::compile(
        ".a = .properties.amount.value\nif .a > 0 { abort }",
        &View::new(),
        VIEW_CLASS,
    )
    .unwrap();
    let (store, view, route) = registered(vec![method(&recipe, Logic::Kleene)]);
    let frame = source.frames.seal();
    let registered = guard_true_rows(&store, &view, route, &source, &frame, METHOD).unwrap();
    // The recipe keeps row 0 (-4 is not > 0) and cannot decide row 1.
    assert_eq!(
        (recipe.count(&w).unwrap(), recipe.count_unknown(&w).unwrap()),
        (1, 1)
    );
    // The registered guard keeps neither: both rows are NULL to it.
    assert_eq!(registered, 0);
}
