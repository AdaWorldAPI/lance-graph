//! Selection is not execution population.
//!
//! The socket issues an ordered selection with repeats (`issue_concat`), and
//! an address can be resolved through it. That does not make it the
//! population a method runs over: `act` takes no selection and runs over the
//! populated prefix.
//!
//! Fixture: four rows, all satisfying the guard; the selection `[1, 1, 3]`.
//! Three counts, each by hand:
//!
//! | reading | path | count |
//! |---|---|---|
//! | every row of the source | `act` | 4 |
//! | the selection's distinct rows | `act_over`, distinct method | 2 |
//! | the selection as a bag | `act_over`, bag method | 3 |
//!
//! `act` refuses a bag-declared method: it has no selection to weigh.

use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_algebra::{Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{Focus, Scope};
use lance_graph_dt64::{
    view_class, AlgebraRef, Arg, BindStore, ClassSchema, Effect, Extent, Guard, LaneRequest,
    MethodBinding, MethodId, RouteAt, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mailbox_lanes::act::{act, act_over, ActContext, ActRefusal};
use lance_graph_mailbox_lanes::when::{when, WhenRefusal, WhenRequest};
use lance_graph_mailbox_lanes::{Column, Keep, MailboxLanes};
use lance_graph_r2il::binding::{ExecPermit, ExecRefusal, R2ilLens};
use lance_graph_r2il::polyfill::{AliasLanguage, POLYFILL, POLYFILL_LENS};

/// Counts this thread's allocations, so tests running beside it do not
/// disturb a count.
struct Counting;

thread_local! {
    static ALLOCS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

// SAFETY: forwards to the system allocator unchanged; the counter is a
// thread-local `Cell` with a const initialiser, so reading it allocates
// nothing.
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        // SAFETY: same layout, passed through.
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        // SAFETY: `ptr` came from `System.alloc` with this layout.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocations(f: impl FnOnce()) -> u64 {
    let before = ALLOCS.with(std::cell::Cell::get);
    f();
    ALLOCS.with(std::cell::Cell::get) - before
}

const N: usize = 64;
type Owner = MailboxSoA<N>;
const POPULATED: usize = 4;
const CLASS: u32 = 0x0A1A_0009;
const POSITION: u16 = 16;
const REV: SchemaRev = SchemaRev(1);

const META: u8 = 0;
const ACTIVE: u8 = 1;
#[allow(dead_code)]
const SIGMA: u8 = 2;
const MASK: u8 = 3;
/// Names no mailbox column.
const SCRATCH: u8 = 4;
const FIELDS: usize = 5;

/// ON meta; WHEN `meta == 2`; DO `scratch := COUNT(kept rows)`.
const COUNT: MethodId = MethodId(1);
/// The same, declared over a bag.
const COUNT_BAG: MethodId = MethodId(2);
/// The same, with no guard.
const COUNT_ALL: MethodId = MethodId(3);
/// As `COUNT`, written to `meta`.
const COUNT_TO_META: MethodId = MethodId(4);

/// No guard, no read: `scratch := 2 + 3`.
const CONST: MethodId = MethodId(5);
/// The same, written to `meta`.
const CONST_TO_META: MethodId = MethodId(6);

/// WHEN `meta == 2`; the effect mask is `active == 5`; DO `scratch := COUNT`.
/// Declared over a bag.
const BAG_ACTIVE: MethodId = MethodId(7);
/// The same, over the distinct rows.
const DISTINCT_ACTIVE: MethodId = MethodId(8);
/// As `COUNT`, declared over a bag, with an `ANY` fold.
const BAG_ANY: MethodId = MethodId(9);
/// As `COUNT_TO_META`, declared over a bag.
const BAG_TO_META: MethodId = MethodId(10);
/// As `COUNT_ALL` (no guard), declared over a bag.
const BAG_ALL: MethodId = MethodId(11);

const W_ADD: u8 = 0;
const W_EQUAL: u8 = 4;
const W_POP_COUNT: u8 = 10;
const W_ANY: u8 = 14;

/// Every row satisfies the guard (`meta == 2`).
fn meta_of(_r: usize) -> u32 {
    2
}
/// Rows 1 and 3 are marked, so the socket can issue `[1]` and `[1, 3]`.
fn active_of(r: usize) -> u32 {
    [0, 5, 0, 6][r]
}

fn owner() -> Box<Owner> {
    let mut m: Box<Owner> = Box::new(MailboxSoA::new(7, 3, 1.0));
    m.set_populated(POPULATED);
    for r in 0..POPULATED {
        m.set_meta(r, MetaWord(meta_of(r)));
        m.last_active_cycle[r] = active_of(r);
    }
    m
}

struct View {
    class: Vec<FieldRef>,
    lens: Vec<FieldRef>,
}

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == view_class(CLASS) {
            &self.class
        } else {
            &[]
        }
    }
    fn lens_fields(&self, lens: u32) -> Option<&[FieldRef]> {
        (lens == POLYFILL_LENS).then_some(&self.lens[..])
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

fn word(target: u8, word: u8, args: Vec<Arg>) -> Effect {
    Effect {
        target,
        algebra: AlgebraRef {
            lens: POLYFILL_LENS,
            word,
        },
        args,
    }
}

/// `target := COUNT(rows of the population)`: the fold keeps `meta == 2`,
/// which every guard row satisfies.
fn count_into(id: MethodId, target: u8) -> MethodBinding {
    MethodBinding {
        id,
        sig_rev: SigRev(1),
        name: format!("m{}", id.0),
        on: WideFieldMask::from_positions(&[META]),
        when: Some(Guard::new(
            SetExpr::eq_u32(u16::from(META), 2),
            Logic::TwoValued,
        )),
        reads: WideFieldMask::from_positions(&[META]),
        effects: vec![
            word(MASK, W_EQUAL, vec![Arg::Field(META), Arg::Const(2)]),
            word(target, W_POP_COUNT, vec![Arg::Field(MASK)]),
        ],
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

/// WHEN `meta == 2`; the effect mask is `active == 5`, independent of the
/// guard; DO `scratch := COUNT(mask)`.
fn active_into(id: MethodId, population: Population) -> MethodBinding {
    MethodBinding {
        id,
        sig_rev: SigRev(1),
        name: format!("m{}", id.0),
        on: WideFieldMask::from_positions(&[META]),
        when: Some(Guard::new(
            SetExpr::eq_u32(u16::from(META), 2),
            Logic::TwoValued,
        )),
        reads: WideFieldMask::from_positions(&[META, ACTIVE]),
        effects: vec![
            word(MASK, W_EQUAL, vec![Arg::Field(ACTIVE), Arg::Const(5)]),
            word(SCRATCH, W_POP_COUNT, vec![Arg::Field(MASK)]),
        ],
        population,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

/// `target := 2 + 3`: no guard, and no field is read.
fn constant_into(id: MethodId, target: u8) -> MethodBinding {
    MethodBinding {
        id,
        sig_rev: SigRev(1),
        name: format!("m{}", id.0),
        on: WideFieldMask::from_positions(&[META]),
        when: None,
        reads: WideFieldMask::from_positions(&[]),
        effects: vec![word(target, W_ADD, vec![Arg::Const(2), Arg::Const(3)])],
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

struct World {
    store: BindStore,
    view: View,
    route: RouteAt,
    permit: ExecPermit,
}

fn world() -> World {
    let field = |iri: &str| FieldRef::new(iri, iri);
    let view = View {
        class: vec![
            field(Column::Meta.iri()),
            field(Column::LastActiveCycle.iri()),
            field("urn:mailbox:sigma"),
            field("urn:lab:mask"),
            field("urn:lab:scratch"),
        ],
        lens: POLYFILL
            .iter()
            .enumerate()
            .map(|(i, w)| {
                FieldRef::new(format!("urn:polyfill:{i}"), w.name_in(AliasLanguage::R2il))
            })
            .collect(),
    };
    let mut store = BindStore::new();
    store
        .declare_algebra(
            POLYFILL_LENS,
            REV,
            Extent::DR256_SQ_X64,
            POLYFILL.iter().map(|w| w.arity).collect(),
        )
        .unwrap();
    store
        .register(
            &view,
            CLASS,
            ClassSchema {
                schema_rev: REV,
                field_count: FIELDS,
                extent: Extent::DR256_SQ_X64,
            },
            vec![
                count_into(COUNT, SCRATCH),
                MethodBinding {
                    population: Population::Bag,
                    ..count_into(COUNT_BAG, SCRATCH)
                },
                MethodBinding {
                    when: None,
                    ..count_into(COUNT_ALL, SCRATCH)
                },
                count_into(COUNT_TO_META, META),
                constant_into(CONST, SCRATCH),
                constant_into(CONST_TO_META, META),
                active_into(BAG_ACTIVE, Population::Bag),
                active_into(DISTINCT_ACTIVE, Population::Distinct),
                MethodBinding {
                    population: Population::Bag,
                    effects: vec![
                        word(MASK, W_EQUAL, vec![Arg::Field(META), Arg::Const(2)]),
                        word(SCRATCH, W_ANY, vec![Arg::Field(MASK)]),
                    ],
                    ..count_into(BAG_ANY, SCRATCH)
                },
                MethodBinding {
                    population: Population::Bag,
                    ..count_into(BAG_TO_META, META)
                },
                MethodBinding {
                    population: Population::Bag,
                    when: None,
                    ..count_into(BAG_ALL, SCRATCH)
                },
            ],
        )
        .expect("the class registers");
    let version = store.route(POSITION, CLASS).unwrap();
    let permit = ExecPermit {
        methods: [
            COUNT,
            COUNT_BAG,
            COUNT_ALL,
            COUNT_TO_META,
            CONST,
            CONST_TO_META,
            BAG_ACTIVE,
            DISTINCT_ACTIVE,
            BAG_ANY,
            BAG_TO_META,
            BAG_ALL,
        ]
        .iter()
        .map(|&m| store.method_ref(CLASS, m).unwrap())
        .collect(),
    };
    World {
        store,
        view,
        route: RouteAt {
            position: POSITION,
            version,
        },
        permit,
    }
}

fn ctx(w: &World) -> ActContext<'_> {
    ActContext {
        route: w.route,
        rev: REV,
        scope: Scope { id: 3 },
        focus: Focus { revision: 1 },
        lens: R2ilLens {
            lens: POLYFILL_LENS,
            schema_rev: REV,
        },
        permit: &w.permit,
    }
}

const READER: Scope = Scope { id: 3 };

/// `[1, 1, 3]`, as the socket issues it.
fn bag(lanes: &mut MailboxLanes<'_, N>) -> u16 {
    let one = lanes
        .issue_where(Column::LastActiveCycle, Keep::Eq(5))
        .unwrap();
    let both = lanes
        .issue_where(Column::LastActiveCycle, Keep::Ge(5))
        .unwrap();
    lanes.issue_concat(one, both).unwrap()
}

/// The three readings by hand, over the positions the socket lends.
fn oracles(positions: &[u32]) -> (i64, i64, i64) {
    let mut distinct = positions.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    (
        POPULATED as i64,
        distinct.len() as i64,
        positions.len() as i64,
    )
}

fn positions_of(w: &World, lanes: &MailboxLanes<'_, N>, sel: u16) -> Vec<u32> {
    let frame = lanes.seal();
    let binding = w
        .store
        .bind_lanes(
            &w.view,
            lanes.frame_store(),
            lanes,
            LaneRequest {
                route: w.route,
                rev: REV,
                fields: &WideFieldMask::from_positions(&[META]),
                frame: &frame,
                scope: READER,
                selection: lanes.space(),
            },
        )
        .unwrap();
    let (_, positions) = binding
        .selected(&w.store, &w.view, lanes.frame_store(), lanes, META, sel)
        .unwrap();
    positions.to_vec()
}

/// Which population each path computes over.
/// FAILS IF: `act` computes over anything but the whole source, `act_over`
/// of a distinct method over anything but the selection's distinct rows,
/// `act_over` of a bag method returns anything but the bag count, or `act`
/// runs a bag method.
#[test]
fn each_path_names_its_population() {
    let m = owner();
    let w = world();
    let mut lanes = MailboxLanes::new(m.published(), READER);
    let sel = bag(&mut lanes);
    let positions = positions_of(&w, &lanes, sel);
    assert_eq!(positions, [1, 1, 3]);
    let (full, distinct, as_bag) = oracles(&positions);
    assert_eq!((full, distinct, as_bag), (4, 2, 3));

    let count = w.store.method_ref(CLASS, COUNT).unwrap();
    // `act` takes no selection: the whole populated prefix.
    let whole = act(&lanes, &w.store, &w.view, &ctx(&w), &count).unwrap();
    assert_eq!(whole.unwritten, vec![(SCRATCH, full)]);
    assert_eq!(whole.rows, [0, 1, 2, 3]);

    // `act_over`: the guard's rows that are in the selection, each once.
    let over = act_over(&lanes, &w.store, &w.view, &ctx(&w), &count, sel).unwrap();
    assert_eq!(over.unwritten, vec![(SCRATCH, distinct)]);
    assert_eq!(over.rows, [1, 3]);
    assert!(over.intents.is_empty());
    // Which population a result was computed over is part of the result.
    assert_eq!((whole.over, over.over), (None, Some(sel)));
    // Neither of these is the bag reading.
    assert_ne!(whole.unwritten[0].1, as_bag);
    assert_ne!(over.unwritten[0].1, as_bag);

    // The bag reading has one path: `act_over` with a bag-declared method.
    // `act` has no selection and refuses it.
    let count_bag = w.store.method_ref(CLASS, COUNT_BAG).unwrap();
    assert_eq!(
        act(&lanes, &w.store, &w.view, &ctx(&w), &count_bag),
        Err(ActRefusal::BagPopulation { method: COUNT_BAG })
    );
    let bagged = act_over(&lanes, &w.store, &w.view, &ctx(&w), &count_bag, sel).unwrap();
    assert_eq!(bagged.unwritten, vec![(SCRATCH, as_bag)]);
    assert!(bagged.intents.is_empty());
    assert_eq!(bagged.over, Some(sel));
    assert_eq!(bagged.receipt.method, count_bag);
    assert!(bagged.receipt.output.values.contains(&(SCRATCH, 3)));
}

/// The selection narrows; it does not replace the guard.
/// FAILS IF: a selected row the guard rejects is counted.
#[test]
fn a_selection_narrows_the_guard_and_does_not_replace_it() {
    let mut m = owner();
    // Row 3 no longer satisfies the guard.
    m.set_meta(3, MetaWord(7));
    let w = world();
    let mut lanes = MailboxLanes::new(m.published(), READER);
    let sel = bag(&mut lanes);
    assert_eq!(positions_of(&w, &lanes, sel), [1, 1, 3]);
    let count = w.store.method_ref(CLASS, COUNT).unwrap();
    let over = act_over(&lanes, &w.store, &w.view, &ctx(&w), &count, sel).unwrap();
    assert_eq!(over.rows, [1]);
    assert_eq!(over.unwritten, vec![(SCRATCH, 1)]);
    let whole = act(&lanes, &w.store, &w.view, &ctx(&w), &count).unwrap();
    assert_eq!(whole.unwritten, vec![(SCRATCH, 3)]);
}

/// FAILS IF: `act_over` runs a method that writes, a method with no guard
/// (there is no gate to narrow), or a selection the socket did not issue.
#[test]
fn what_act_over_refuses() {
    let m = owner();
    let w = world();
    let mut lanes = MailboxLanes::new(m.published(), READER);
    let sel = bag(&mut lanes);

    let to_meta = w.store.method_ref(CLASS, COUNT_TO_META).unwrap();
    assert_eq!(
        act_over(&lanes, &w.store, &w.view, &ctx(&w), &to_meta, sel),
        Err(ActRefusal::NotReadOnly {
            field: META,
            column: Column::Meta
        })
    );
    let all = w.store.method_ref(CLASS, COUNT_ALL).unwrap();
    assert_eq!(
        act_over(&lanes, &w.store, &w.view, &ctx(&w), &all, sel),
        Err(ActRefusal::SelectionNeedsGuard)
    );
    // Silence twin: unguarded and unselected, it runs over every row.
    let got = act(&lanes, &w.store, &w.view, &ctx(&w), &all).unwrap();
    assert_eq!(got.unwritten, vec![(SCRATCH, 4)]);

    let count = w.store.method_ref(CLASS, COUNT).unwrap();
    assert!(matches!(
        act_over(&lanes, &w.store, &w.view, &ctx(&w), &count, sel + 1),
        Err(ActRefusal::Lane(_))
    ));
}

/// A guard that reads no column still belongs to one reader scope.
/// FAILS IF: a constant guard is evaluated for a scope other than the
/// socket's (no lane is bound for it, so no lane check ever sees the scope).
/// The check is consistency of two caller-asserted values, not a grant.
#[test]
fn a_constant_guard_still_checks_the_reader_scope() {
    let m = owner();
    let w = world();
    let lanes = MailboxLanes::new(m.published(), READER);
    let request = |scope| WhenRequest {
        route: w.route,
        rev: REV,
        scope,
    };
    for expr in [SetExpr::All, SetExpr::Empty] {
        let guard = Guard::new(expr, Logic::TwoValued);
        let other = when(&lanes, &w.store, &w.view, request(Scope { id: 4 }), &guard);
        assert!(
            matches!(other, Err(WhenRefusal::ScopeMismatch { .. })),
            "{other:?}"
        );
        // Silence twin: the socket's own scope.
        let own = when(&lanes, &w.store, &w.view, request(READER), &guard).unwrap();
        assert_eq!(own.rows(), POPULATED);
    }
    assert_eq!(lanes.lends(), 0);
}

/// A method with no guard and no read binds no lane, so only `act`'s own
/// check compares the request's scope with the socket's.
/// FAILS IF: such a method computes a scalar or builds a write intent for a
/// reader scope other than the socket's.
#[test]
fn an_unguarded_method_that_reads_nothing_still_checks_the_reader_scope() {
    let m = owner();
    let w = world();
    let lanes = MailboxLanes::new(m.published(), READER);
    let foreign = ActContext {
        scope: Scope { id: 4 },
        ..ctx(&w)
    };
    for (id, target) in [(CONST, SCRATCH), (CONST_TO_META, META)] {
        let method = w.store.method_ref(CLASS, id).unwrap();
        let refused = act(&lanes, &w.store, &w.view, &foreign, &method);
        assert!(
            matches!(
                refused,
                Err(ActRefusal::When(WhenRefusal::ScopeMismatch { .. }))
            ),
            "{refused:?}"
        );
        // Silence twin: the socket's own scope runs it.
        let own = act(&lanes, &w.store, &w.view, &ctx(&w), &method).unwrap();
        assert_eq!(own.receipt.output.values, vec![(target, 5)]);
        assert_eq!(own.intents.len(), usize::from(target == META));
    }
    assert_eq!(lanes.lends(), 0);
}

/// The three counts by hand, with the guard (`meta == 2`) applied to every
/// row or occurrence. Calls nothing of the implementation.
fn guarded_oracles(meta: &[u32], positions: &[u32]) -> (i64, i64, i64) {
    let holds = |p: u32| meta[p as usize] == 2;
    let whole = (0..meta.len() as u32).filter(|&p| holds(p)).count();
    let mut distinct = positions.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    let distinct = distinct.into_iter().filter(|&p| holds(p)).count();
    let bag = positions.iter().filter(|&&p| holds(p)).count();
    (whole as i64, distinct as i64, bag as i64)
}

/// The bag reading against the two others, over six selections.
/// FAILS IF: `act` or `act_over` returns anything but the whole-source or
/// distinct count, `act` runs a bag-declared method, or the bag path returns
/// anything but the per-occurrence guarded count.
#[test]
fn a_bag_count_weighs_each_occurrence_by_the_guard() {
    // (row 3 fails the guard, selection, positions, whole, distinct, bag)
    type Case = (bool, &'static str, &'static [u32], i64, i64, i64);
    let cases: [Case; 6] = [
        (false, "one+both", &[1, 1, 3], 4, 2, 3),
        (true, "one+both", &[1, 1, 3], 3, 1, 2),
        // Order does not change a count; multiplicity does.
        (false, "both+one", &[1, 3, 1], 4, 2, 3),
        (true, "both+one", &[1, 3, 1], 3, 1, 2),
        (false, "none", &[], 4, 0, 0),
        (false, "one+one", &[1, 1], 4, 1, 2),
    ];
    for (row3_fails, which, expected, whole, distinct, as_bag) in cases {
        let mut m = owner();
        if row3_fails {
            m.set_meta(3, MetaWord(7));
        }
        let meta: Vec<u32> = (0..POPULATED).map(|r| m.meta[r].0).collect();
        let w = world();
        let mut lanes = MailboxLanes::new(m.published(), READER);
        let one = lanes
            .issue_where(Column::LastActiveCycle, Keep::Eq(5))
            .unwrap();
        let both = lanes
            .issue_where(Column::LastActiveCycle, Keep::Ge(5))
            .unwrap();
        let sel = match which {
            "one+both" => lanes.issue_concat(one, both).unwrap(),
            "both+one" => lanes.issue_concat(both, one).unwrap(),
            "one+one" => lanes.issue_concat(one, one).unwrap(),
            _ => lanes
                .issue_where(Column::LastActiveCycle, Keep::Eq(99))
                .unwrap(),
        };
        let positions = positions_of(&w, &lanes, sel);
        assert_eq!(positions, expected, "{which}");
        assert_eq!(
            guarded_oracles(&meta, &positions),
            (whole, distinct, as_bag),
            "{which} row3_fails={row3_fails}"
        );

        let count = w.store.method_ref(CLASS, COUNT).unwrap();
        let got = act(&lanes, &w.store, &w.view, &ctx(&w), &count).unwrap();
        assert_eq!(got.unwritten, vec![(SCRATCH, whole)], "{which}");
        let got = act_over(&lanes, &w.store, &w.view, &ctx(&w), &count, sel).unwrap();
        assert_eq!(got.unwritten, vec![(SCRATCH, distinct)], "{which}");

        let count_bag = w.store.method_ref(CLASS, COUNT_BAG).unwrap();
        let got = act_over(&lanes, &w.store, &w.view, &ctx(&w), &count_bag, sel).unwrap();
        assert_eq!(got.unwritten, vec![(SCRATCH, as_bag)], "{which}");
        assert_eq!(
            act(&lanes, &w.store, &w.view, &ctx(&w), &count_bag),
            Err(ActRefusal::BagPopulation { method: COUNT_BAG }),
            "{which}"
        );
    }
}

/// Both the guard and the effect's own mask decide whether an occurrence
/// counts in a bag.
/// FAILS IF: the bag count ignores the effect mask (`active == 5`), ignores
/// the guard (`meta == 2`), or counts a row the effect mask excludes.
#[test]
fn an_effect_mask_and_the_guard_both_decide_a_bag_occurrence() {
    let run = |m: &Owner, method: MethodId, over: bool| -> i64 {
        let w = world();
        let mut lanes = MailboxLanes::new(m.published(), READER);
        let sel = bag(&mut lanes);
        let r = w.store.method_ref(CLASS, method).unwrap();
        let got = if over {
            act_over(&lanes, &w.store, &w.view, &ctx(&w), &r, sel).unwrap()
        } else {
            act(&lanes, &w.store, &w.view, &ctx(&w), &r).unwrap()
        };
        assert_eq!(got.unwritten.len(), 1);
        got.unwritten[0].1
    };

    // Hand oracle from literals. Selection [1, 1, 3]; meta [2,2,2,2];
    // active [0,5,0,6]. An occurrence counts when meta == 2 AND active == 5:
    // row 1 passes both (twice), row 3 fails the effect mask (6 != 5).
    let meta = [2u32, 2, 2, 2];
    let active = [0u32, 5, 0, 6];
    let selection = [1usize, 1, 3];
    let oracle = selection
        .iter()
        .filter(|&&p| meta[p] == 2 && active[p] == 5)
        .count() as i64;
    assert_eq!(oracle, 2);

    let m = owner();
    assert_eq!(run(&m, BAG_ACTIVE, true), 2);
    assert_eq!(run(&m, DISTINCT_ACTIVE, true), 1);
    // `act` has no selection: the whole prefix, each row once. Only row 1
    // passes both conditions.
    assert_eq!(run(&m, DISTINCT_ACTIVE, false), 1);

    // Row 1 fails the guard (meta 7) although its effect mask holds: the
    // guard excludes it, so both counts are 0.
    let mut m = owner();
    m.set_meta(1, MetaWord(7));
    assert_eq!(run(&m, BAG_ACTIVE, true), 0);
    assert_eq!(run(&m, DISTINCT_ACTIVE, true), 0);

    // Row 3 fails the guard, but it never passed the effect mask (6 != 5),
    // so the bag count is unchanged: the effect mask alone excludes it.
    let mut m = owner();
    m.set_meta(3, MetaWord(7));
    assert_eq!(run(&m, BAG_ACTIVE, true), 2);
}

/// What a bag run refuses, and the silence twin.
/// FAILS IF: a bag method with a non-count fold, a write, no guard, a foreign
/// selection, a foreign scope or no permit still runs, or the admitted bag
/// count is refused.
#[test]
fn what_a_bag_run_refuses() {
    let m = owner();
    let w = world();
    let mut lanes = MailboxLanes::new(m.published(), READER);
    let sel = bag(&mut lanes);

    let any = w.store.method_ref(CLASS, BAG_ANY).unwrap();
    let refused = act_over(&lanes, &w.store, &w.view, &ctx(&w), &any, sel);
    assert!(
        matches!(
            refused,
            Err(ActRefusal::Exec(ExecRefusal::BagFold { effect: 1, .. }))
        ),
        "{refused:?}"
    );

    let to_meta = w.store.method_ref(CLASS, BAG_TO_META).unwrap();
    assert_eq!(
        act_over(&lanes, &w.store, &w.view, &ctx(&w), &to_meta, sel),
        Err(ActRefusal::NotReadOnly {
            field: META,
            column: Column::Meta
        })
    );
    let all = w.store.method_ref(CLASS, BAG_ALL).unwrap();
    assert_eq!(
        act_over(&lanes, &w.store, &w.view, &ctx(&w), &all, sel),
        Err(ActRefusal::SelectionNeedsGuard)
    );

    let count_bag = w.store.method_ref(CLASS, COUNT_BAG).unwrap();
    assert!(matches!(
        act_over(&lanes, &w.store, &w.view, &ctx(&w), &count_bag, sel + 1),
        Err(ActRefusal::Lane(_))
    ));

    let foreign = ActContext {
        scope: Scope { id: 4 },
        ..ctx(&w)
    };
    let other = act_over(&lanes, &w.store, &w.view, &foreign, &count_bag, sel);
    assert!(
        matches!(
            other,
            Err(ActRefusal::When(WhenRefusal::ScopeMismatch { .. }))
        ),
        "{other:?}"
    );

    let empty = ExecPermit { methods: vec![] };
    let unpermitted = ActContext {
        permit: &empty,
        ..ctx(&w)
    };
    assert_eq!(
        act_over(&lanes, &w.store, &w.view, &unpermitted, &count_bag, sel),
        Err(ActRefusal::Exec(ExecRefusal::NotPermitted))
    );

    // Silence twin: the same method, the real context and the issued
    // selection runs and counts 3.
    let got = act_over(&lanes, &w.store, &w.view, &ctx(&w), &count_bag, sel).unwrap();
    assert_eq!(got.unwritten, vec![(SCRATCH, 3)]);
}

/// A bag run reads in place: it writes nothing and copies no column.
/// FAILS IF: a bag run builds a write intent, changes any owner cell, or
/// lends a different number of lanes than its reads need.
#[test]
fn a_bag_run_writes_nothing_and_copies_no_column() {
    let m = owner();
    let snapshot = |m: &Owner| -> Vec<(u32, u64, u32, u32)> {
        (0..POPULATED)
            .map(|r| {
                (
                    m.meta[r].0,
                    m.temporal[r],
                    m.last_active_cycle[r],
                    m.last_write_cycle[r],
                )
            })
            .collect()
    };
    let before = snapshot(&m);
    {
        let w = world();
        let mut lanes = MailboxLanes::new(m.published(), READER);
        let sel = bag(&mut lanes);
        let count_bag = w.store.method_ref(CLASS, COUNT_BAG).unwrap();
        let lends_before = lanes.lends();
        let got = act_over(&lanes, &w.store, &w.view, &ctx(&w), &count_bag, sel).unwrap();
        assert!(got.intents.is_empty());
        // Measured: the guard's `meta` lane, the operand `meta` lane, and
        // the lane `LaneBinding::selected` borrows with the positions.
        assert_eq!(lanes.lends() - lends_before, 3);
    }
    assert_eq!(snapshot(&m), before);
}

/// What one read-only `act` allocates, pinned to the measured count.
/// FAILS IF: `act` allocates more than it did when this was measured, for
/// example by building a vector it does not need for a method that writes
/// nothing.
#[test]
fn a_read_only_act_allocates_what_was_measured() {
    let m = owner();
    let w = world();
    let lanes = MailboxLanes::new(m.published(), READER);
    let count = w.store.method_ref(CLASS, COUNT).unwrap();
    let context = ctx(&w);
    // Warm once: lazily built tables are not part of a call's cost.
    act(&lanes, &w.store, &w.view, &context, &count).unwrap();
    let mut out = None;
    let n = allocations(|| out = Some(act(&lanes, &w.store, &w.view, &context, &count)));
    assert_eq!(out.unwrap().unwrap().unwritten, vec![(SCRATCH, 4)]);
    assert_eq!(
        n, MEASURED_READ_ONLY_ACT_ALLOCATIONS,
        "allocations of one act"
    );
}

/// Measured on this fixture; lower it when `act` gets leaner.
const MEASURED_READ_ONLY_ACT_ALLOCATIONS: u64 = 41;
