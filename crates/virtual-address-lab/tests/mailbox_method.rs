//! D-VAS P1: the lab's constant `Method { selection, orientation }` on the
//! real mailbox path.
//!
//! ```text
//! publication -> socket-issued selection -> const Method -> field + ordinal
//!   -> ON (methods of the routed class on that field) -> WHEN at the
//!   addressed row -> DO only where the guard holds
//! ```
//!
//! Until this test the two halves were proven apart: the address law on the
//! lab `Grid` (`tests/cube_method.rs`), and ON/WHEN/DO on a real
//! `MailboxSoA<64>` from DT64 bindings (`lance-graph-mailbox-lanes`,
//! `tests/e1.rs`), with no `Method` in the second.
//!
//! Addressing oracle: the tile `[field][ordinal]` is built from the owner's
//! arrays and a sort written here, then MOVED (rows reversed, lanes reversed,
//! transposed) into the view each recipe names. It calls no `Recipe` code.
//! Behaviour oracle: the two methods written out by hand.
//!
//! Two address carriers enter the same chain: an ordinal of a selection the
//! socket issued, and an id of DT64's `MapTable` at a sealed version. A
//! `NodeGuid` is not one of them (the mailbox stores no key).

use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_algebra::{Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{Focus, LocalMap, MapTable, Scope, TileId, Width};
use lance_graph_dt64::{
    view_class, AlgebraRef, Arg, BindStore, ClassSchema, Effect, Extent, Guard, LaneBinding,
    LaneRequest, MethodBinding, MethodId, RouteAt, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mailbox_lanes::act::{act, ActContext};
use lance_graph_mailbox_lanes::coords::Coordinates;
use lance_graph_mailbox_lanes::when::{when, WhenRequest};
use lance_graph_mailbox_lanes::{Column, MailboxLanes};
use lance_graph_mask_risc::LaneRef;
use lance_graph_r2il::binding::{ExecPermit, R2ilLens};
use lance_graph_r2il::polyfill::{AliasLanguage, POLYFILL, POLYFILL_LENS};
use virtual_address_lab::method::Method;
use virtual_address_lab::recipe::Recipe;

const N: usize = 64;
type Owner = MailboxSoA<N>;
const POPULATED: usize = 40;

const CLASS: u32 = 0x0A1A_0003;
const POSITION: u16 = 11;
const READER: Scope = Scope { id: 3 };
const FOCUS: Focus = Focus { revision: 1 };
const REV: SchemaRev = SchemaRev(1);

/// Field positions of the class basis. `MASK` names no mailbox column: it is
/// where a method keeps an intermediate mask.
const META: u8 = 0;
const ACTIVE: u8 = 1;
const TEMPORAL: u8 = 2;
const MASK: u8 = 3;
const FIELDS: usize = 4;

/// `tally`: ON meta; WHEN `meta == 2 AND active != 1`;
/// DO `temporal := COUNT(rows the guard keeps where active == 0)`.
const TALLY: MethodId = MethodId(1);
/// `mark`: ON temporal; WHEN `meta == 1`;
/// DO `meta := COUNT(rows the guard keeps where active == 2)`.
const MARK: MethodId = MethodId(2);

/// Polyfill positions (`lance_graph_r2il::polyfill::POLYFILL`).
const W_EQUAL: u8 = 4;
const W_POP_COUNT: u8 = 10;

fn meta_of(r: usize) -> u32 {
    (r as u32 * 7 + 3) % 4
}
fn active_of(r: usize) -> u32 {
    (r as u32 * 5 + 1) % 3
}

fn owner() -> Box<Owner> {
    let mut m: Box<Owner> = Box::new(MailboxSoA::new(7, 3, 1.0));
    m.set_populated(POPULATED);
    for r in 0..POPULATED {
        m.set_meta(r, MetaWord(meta_of(r)));
        m.last_active_cycle[r] = active_of(r);
        m.set_temporal(r, 1_000 + 3 * r as u64);
    }
    m
}

struct View {
    class: Vec<FieldRef>,
    lens: Vec<FieldRef>,
}

impl View {
    fn new() -> View {
        let field = |iri: &str| FieldRef::new(iri, iri);
        View {
            class: vec![
                field(Column::Meta.iri()),
                field(Column::LastActiveCycle.iri()),
                field(Column::Temporal.iri()),
                field("urn:lab:mask"),
            ],
            lens: POLYFILL
                .iter()
                .enumerate()
                .map(|(i, w)| {
                    FieldRef::new(format!("urn:polyfill:{i}"), w.name_in(AliasLanguage::R2il))
                })
                .collect(),
        }
    }
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

/// `target := COUNT(rows where field == v)`, through the scratch mask field.
fn count_where(field: u8, v: i64, target: u8) -> Vec<Effect> {
    vec![
        word(MASK, W_EQUAL, vec![Arg::Field(field), Arg::Const(v)]),
        word(target, W_POP_COUNT, vec![Arg::Field(MASK)]),
    ]
}

fn method(id: MethodId, on: u8, when: SetExpr, effects: Vec<Effect>) -> MethodBinding {
    MethodBinding {
        id,
        sig_rev: SigRev(1),
        name: format!("m{}", id.0),
        on: WideFieldMask::from_positions(&[on]),
        when: Some(Guard::new(when, Logic::TwoValued)),
        reads: WideFieldMask::from_positions(&[ACTIVE]),
        effects,
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

fn tally_guard() -> SetExpr {
    SetExpr::And(vec![
        SetExpr::eq_u32(u16::from(META), 2),
        SetExpr::ne_u32(u16::from(ACTIVE), 1),
    ])
}

struct World {
    store: BindStore,
    view: View,
    route: RouteAt,
    permit: ExecPermit,
}

fn world() -> World {
    let view = View::new();
    let mut store = BindStore::new();
    store
        .declare_algebra(
            POLYFILL_LENS,
            REV,
            Extent::DR256_SQ_X64,
            POLYFILL.iter().map(|w| w.arity).collect(),
        )
        .unwrap();
    let schema = ClassSchema {
        schema_rev: REV,
        field_count: FIELDS,
        extent: Extent::DR256_SQ_X64,
    };
    store
        .register(
            &view,
            CLASS,
            schema,
            vec![
                method(TALLY, META, tally_guard(), count_where(ACTIVE, 0, TEMPORAL)),
                method(
                    MARK,
                    TEMPORAL,
                    SetExpr::eq_u32(u16::from(META), 1),
                    count_where(ACTIVE, 2, META),
                ),
            ],
        )
        .expect("the class registers");
    let version = store.route(POSITION, CLASS).unwrap();
    let permit = ExecPermit {
        methods: [TALLY, MARK]
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
        scope: READER,
        focus: FOCUS,
        lens: R2ilLens {
            lens: POLYFILL_LENS,
            schema_rev: REV,
        },
        permit: &w.permit,
    }
}

/// What a demanded cell is and does.
#[derive(Debug, Clone, PartialEq)]
struct Demanded {
    field: u8,
    row: u32,
    value: u64,
    /// Methods ON the cell's field, ascending.
    candidates: Vec<MethodId>,
    /// Of those, the ones whose guard holds at the row: the column and value
    /// of the write each asks for.
    fired: Vec<(MethodId, Column, u64)>,
}

fn bind(w: &World, lanes: &MailboxLanes<'_, N>) -> LaneBinding {
    let frame = lanes.seal();
    w.store
        .bind_lanes(
            &w.view,
            lanes.frame_store(),
            lanes,
            LaneRequest {
                route: w.route,
                rev: REV,
                fields: &WideFieldMask::from_positions(&[META, ACTIVE, TEMPORAL]),
                frame: &frame,
                scope: READER,
                selection: lanes.space(),
            },
        )
        .unwrap()
}

fn as_u64(lane: LaneRef<'_>, row: u32) -> u64 {
    match lane {
        LaneRef::U32(v) => u64::from(v[row as usize]),
        LaneRef::U64(v) => v[row as usize],
        other => panic!("a mailbox column: {other:?}"),
    }
}

/// ON, then WHEN at `row`, then DO only where the guard holds.
fn behave(w: &World, lanes: &MailboxLanes<'_, N>, field: u8, row: u32, value: u64) -> Demanded {
    let class = w
        .store
        .resolve_at(&w.view, w.route.version, w.route.position, REV)
        .unwrap();
    let candidates: Vec<MethodId> = class.methods_on(field).map(|m| m.id).collect();
    let mut fired = Vec::new();
    for &id in &candidates {
        let reference = w.store.method_ref(CLASS, id).unwrap();
        let admitted = w.store.admit_routed(w.route, &reference).unwrap();
        let guard = admitted.when.as_ref().expect("both methods are guarded");
        let request = WhenRequest {
            route: w.route,
            rev: REV,
            scope: READER,
        };
        let kept = when(lanes, &w.store, &w.view, request, guard).unwrap();
        if kept.holds(&w.store, &w.view, lanes, row).unwrap() {
            let acted = act(lanes, &w.store, &w.view, &ctx(w), &reference).unwrap();
            assert!(acted.rows.contains(&row));
            fired.push((id, acted.intents[0].column(), acted.intents[0].value()));
        }
    }
    Demanded {
        field,
        row,
        value,
        candidates,
        fired,
    }
}

/// The whole path for view cell `(row, col)` of `method`. `None`: the cell
/// has no address (outside the view, or a field that names no column).
fn demand(
    method: Method,
    w: &World,
    lanes: &MailboxLanes<'_, N>,
    binding: &LaneBinding,
    row: usize,
    col: usize,
) -> Option<Demanded> {
    // The field needs no lane and no selection.
    let field = method.field(FIELDS, row, col)?;
    let (lane, positions) = binding
        .selected(
            &w.store,
            &w.view,
            lanes.frame_store(),
            lanes,
            field,
            method.selection,
        )
        .ok()?;
    let cell = method.cell(FIELDS, positions.len(), row, col)?;
    assert_eq!(cell.field, usize::from(field));
    let source_row = positions[cell.ordinal];
    Some(behave(
        w,
        lanes,
        field,
        source_row,
        as_u64(lane, source_row),
    ))
}

/// One cell of the oracle's tile: where it comes from and what it holds.
/// `value` is `None` for a field that names no mailbox column.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Src {
    field: u8,
    row: u32,
    value: Option<u64>,
}

fn reverse_lanes(mut t: Vec<Vec<Src>>) -> Vec<Vec<Src>> {
    t.reverse();
    t
}
fn reverse_positions(mut t: Vec<Vec<Src>>) -> Vec<Vec<Src>> {
    t.iter_mut().for_each(|lane| lane.reverse());
    t
}
fn transpose(t: Vec<Vec<Src>>) -> Vec<Vec<Src>> {
    (0..t[0].len())
        .map(|c| t.iter().map(|lane| lane[c]).collect())
        .collect()
}

/// The view a recipe names, by moving the tile. `Rotate90` is the
/// counter-clockwise quarter turn `view(r, c) = M[c][N-1-r]`.
fn moved(name: &str, t: Vec<Vec<Src>>) -> Vec<Vec<Src>> {
    match name {
        "Identity" => t,
        "MirrorX" => reverse_lanes(t),
        "MirrorY" => reverse_positions(t),
        "Rotate180" => reverse_lanes(reverse_positions(t)),
        "Transpose" => transpose(t),
        "Rotate90" => transpose(reverse_positions(t)),
        "Rotate270" => transpose(reverse_lanes(t)),
        "AntiTranspose" => transpose(reverse_lanes(reverse_positions(t))),
        other => panic!("{other}"),
    }
}

/// The owner's columns, read once, for both oracles.
struct Columns {
    meta: Vec<u32>,
    active: Vec<u32>,
    temporal: Vec<u64>,
}

impl Columns {
    fn of(m: &Owner) -> Columns {
        Columns {
            meta: (0..POPULATED).map(|r| m.meta[r].0).collect(),
            active: (0..POPULATED).map(|r| m.last_active_cycle[r]).collect(),
            temporal: (0..POPULATED).map(|r| m.temporal[r]).collect(),
        }
    }

    /// Rows sorted by `meta`, ties in row order.
    fn by_meta(&self) -> Vec<u32> {
        let mut pairs: Vec<(u32, u32)> = (0..POPULATED as u32)
            .map(|r| (self.meta[r as usize], r))
            .collect();
        pairs.sort_unstable();
        pairs.iter().map(|p| p.1).collect()
    }

    fn tile(&self, order: &[u32]) -> Vec<Vec<Src>> {
        (0..FIELDS as u8)
            .map(|field| {
                order
                    .iter()
                    .map(|&row| Src {
                        field,
                        row,
                        value: match field {
                            META => Some(u64::from(self.meta[row as usize])),
                            ACTIVE => Some(u64::from(self.active[row as usize])),
                            TEMPORAL => Some(self.temporal[row as usize]),
                            _ => None,
                        },
                    })
                    .collect()
            })
            .collect()
    }

    /// The two methods by hand.
    fn behaviour(&self, field: u8, row: u32) -> (Vec<MethodId>, Vec<(MethodId, Column, u64)>) {
        let r = row as usize;
        let rows = 0..POPULATED;
        match field {
            META => {
                let keeps = |r: usize| self.meta[r] == 2 && self.active[r] != 1;
                let value = rows.filter(|&r| keeps(r) && self.active[r] == 0).count() as u64;
                let fired = if keeps(r) {
                    vec![(TALLY, Column::Temporal, value)]
                } else {
                    vec![]
                };
                (vec![TALLY], fired)
            }
            TEMPORAL => {
                let keeps = |r: usize| self.meta[r] == 1;
                let value = rows.filter(|&r| keeps(r) && self.active[r] == 2).count() as u64;
                let fired = if keeps(r) {
                    vec![(MARK, Column::Meta, value)]
                } else {
                    vec![]
                };
                (vec![MARK], fired)
            }
            _ => (vec![], vec![]),
        }
    }

    fn expect(&self, src: Src) -> Option<Demanded> {
        let (candidates, fired) = self.behaviour(src.field, src.row);
        Some(Demanded {
            field: src.field,
            row: src.row,
            value: src.value?,
            candidates,
            fired,
        })
    }
}

/// FAILS IF: for any of the eight orientations and any view cell, the path
/// through the constant `Method` names another field, another source row or
/// another value than the moved tile, offers another method than the ones ON
/// that field, fires one whose guard does not hold at that row, or computes
/// another effect value than the hand-written methods.
#[test]
fn a_constant_method_addresses_and_behaves_on_the_real_mailbox() {
    assert_eq!(std::mem::size_of::<Method>(), 6);
    let m = owner();
    let w = world();
    let cols = Columns::of(&m);
    let order = cols.by_meta();
    assert_ne!(order, (0..POPULATED as u32).collect::<Vec<_>>());

    let mut lanes = MailboxLanes::new(m.published(), READER);
    let sel = lanes.issue_order(Column::Meta).unwrap();
    let binding = bind(&w, &lanes);

    let (mut cells, mut unaddressable, mut with_candidates, mut fired) = (0, 0, 0, 0);
    for (recipe, name) in Recipe::ALL.into_iter().zip(Recipe::NAMES) {
        let method = Method::new(sel, recipe);
        let view = moved(name, cols.tile(&order));
        for (row, line) in view.iter().enumerate() {
            for (col, &src) in line.iter().enumerate() {
                let got = demand(method, &w, &lanes, &binding, row, col);
                assert_eq!(got, cols.expect(src), "{name} ({row}, {col})");
                cells += 1;
                match &got {
                    None => unaddressable += 1,
                    Some(d) => {
                        with_candidates += usize::from(!d.candidates.is_empty());
                        fired += d.fired.len();
                    }
                }
            }
        }
        // One past the view in each direction has no address.
        assert_eq!(demand(method, &w, &lanes, &binding, view.len(), 0), None);
        assert_eq!(demand(method, &w, &lanes, &binding, 0, view[0].len()), None);
    }
    // The sweep is not vacuous: every case occurs, in every orientation.
    assert_eq!(cells, 8 * FIELDS * POPULATED);
    assert_eq!(unaddressable, 8 * POPULATED);
    assert_eq!(with_candidates, 8 * 2 * POPULATED);
    let per_view = cols
        .tile(&order)
        .concat()
        .iter()
        .map(|&s| cols.behaviour(s.field, s.row).1.len())
        .sum::<usize>();
    assert!(per_view > 4 && per_view < 2 * POPULATED, "{per_view}");
    assert_eq!(fired, 8 * per_view);
}

/// FAILS IF: anything is lent before the field and the ON candidates are
/// known, a cell with no candidate runs a guard, or a cell whose guard does
/// not hold runs an effect.
#[test]
fn behaviour_is_resolved_before_anything_is_read() {
    let m = owner();
    let w = world();
    let cols = Columns::of(&m);
    let order = cols.by_meta();
    let mut lanes = MailboxLanes::new(m.published(), READER);
    let sel = lanes.issue_order(Column::Meta).unwrap();
    let binding = bind(&w, &lanes);
    let method = Method::new(sel, Recipe::TRANSPOSE);

    // The field of a view cell, and the methods ON it: no lane lent.
    let before = lanes.lends();
    let field = method.field(FIELDS, 0, usize::from(META)).unwrap();
    let class = w
        .store
        .resolve_at(&w.view, w.route.version, w.route.position, REV)
        .unwrap();
    assert_eq!(field, META);
    assert_eq!(class.methods_on(field).count(), 1);
    assert_eq!(lanes.lends(), before);

    // Under TRANSPOSE view row `o` is ordinal `o`, view column `f` field `f`.
    let cost = |ordinal: usize, field: u8| {
        let before = lanes.lends();
        let got = demand(method, &w, &lanes, &binding, ordinal, usize::from(field)).unwrap();
        (got, lanes.lends() - before)
    };
    // No method is ON `active`: one lane lent, for the one value read.
    let (got, lent) = cost(0, ACTIVE);
    assert!(got.candidates.is_empty());
    assert_eq!(lent, 1);
    // ON `meta`, guard false at the row: the value, then the guard's two
    // fields. No operand, no effect.
    let cold = order
        .iter()
        .position(|&r| cols.behaviour(META, r).1.is_empty())
        .unwrap();
    let (got, lent) = cost(cold, META);
    assert_eq!((got.candidates.len(), got.fired.len()), (1, 0));
    assert_eq!(lent, 1 + 2);
    // Guard true: the effect runs, lending the guard's fields again and the
    // method's one declared read.
    let hot = order
        .iter()
        .position(|&r| !cols.behaviour(META, r).1.is_empty())
        .unwrap();
    let (got, lent) = cost(hot, META);
    assert_eq!(got.fired.len(), 1);
    assert_eq!(lent, 1 + 2 + 2 + 1);
}

/// A real source write and a virtual mapping change are different events.
/// FAILS IF: re-orienting the method touches the owner or invalidates the
/// binding, or a binding survives a write the owner made without a `tick`.
#[test]
fn reorienting_is_not_a_write_and_a_write_is_not_a_reorientation() {
    let mut m = owner();
    let w = world();
    let cols = Columns::of(&m);
    let order = cols.by_meta();

    let binding = {
        let mut lanes = MailboxLanes::new(m.published(), READER);
        let sel = lanes.issue_order(Column::Meta).unwrap();
        let binding = bind(&w, &lanes);
        // Mapping change: the same view cell, seen through one more recipe,
        // is another source row. Nothing was written; the binding holds.
        let entity = Method::new(sel, Recipe::TRANSPOSE);
        let reversed = entity.then(Recipe::MIRROR_X);
        let a = demand(entity, &w, &lanes, &binding, 0, usize::from(META)).unwrap();
        let b = demand(reversed, &w, &lanes, &binding, 0, usize::from(META)).unwrap();
        assert_eq!((a.field, b.field), (META, META));
        assert_eq!(a.row, order[0]);
        assert_eq!(b.row, order[POPULATED - 1]);
        assert_ne!(a.row, b.row);
        binding
    };
    assert_eq!(Columns::of(&m).meta, cols.meta, "no byte moved");

    // Source write, through the owner's setter, with no tick.
    let row = order[0] as usize;
    let cycle = m.current_cycle;
    m.set_meta(row, MetaWord(cols.meta[row] + 100));
    assert_eq!(m.current_cycle, cycle);
    let mut lanes = MailboxLanes::new(m.published(), READER);
    let sel = lanes.issue_order(Column::Meta).unwrap();
    assert!(binding
        .selected(&w.store, &w.view, lanes.frame_store(), &lanes, META, sel)
        .is_err());
    // The same constant under the new publication reads the new byte, at
    // the row the new order puts there.
    let fresh = bind(&w, &lanes);
    let now = Columns::of(&m);
    let last = demand(
        Method::new(sel, Recipe::TRANSPOSE),
        &w,
        &lanes,
        &fresh,
        POPULATED - 1,
        usize::from(META),
    )
    .unwrap();
    assert_eq!(last.row as usize, row);
    assert_eq!(last.value, u64::from(cols.meta[row] + 100));
    assert_eq!(
        Some(last),
        now.expect(Src {
            field: META,
            row: row as u32,
            value: Some(u64::from(now.meta[row])),
        })
    );
}

const TILE: TileId = 7;

/// The id of entity `i`: not a row number.
fn id(i: usize) -> u64 {
    0xA000_0000_0000 + (i as u64 * 7919) % 1_000_000 * 1_000 + i as u64
}
fn assigned_row(i: usize) -> usize {
    (i * 7 + 3) % POPULATED
}

/// Second carrier. FAILS IF: an id of the existing versioned coordinate map
/// enters the chain at another row than the map assigns, or behaves
/// differently there than the same row reached through a selection ordinal.
#[test]
fn a_versioned_coordinate_enters_the_same_chain() {
    let m = owner();
    let w = world();
    let cols = Columns::of(&m);
    let order = cols.by_meta();
    let mut lanes = MailboxLanes::new(m.published(), READER);
    let sel = lanes.issue_order(Column::Meta).unwrap();
    let binding = bind(&w, &lanes);
    let method = Method::new(sel, Recipe::TRANSPOSE);

    let mut globals = vec![0u64; POPULATED];
    for i in 0..POPULATED {
        globals[assigned_row(i)] = id(i);
    }
    let mut table = MapTable::new(7, 0);
    table.advance(
        7,
        0,
        vec![(TILE, Some(LocalMap::prepare(Width::W64, &globals).unwrap()))],
    );
    let at = Coordinates::bind(&table, TILE);

    let mut moved_rows = 0;
    for i in 0..POPULATED {
        let row = at.row_of(&table, id(i), lanes.rows()).unwrap();
        assert_eq!(row as usize, assigned_row(i));
        moved_rows += usize::from(row as usize != i);
        for field in [META, ACTIVE, TEMPORAL] {
            // The field is the method's (view column `field`); the row is
            // the map's.
            assert_eq!(method.field(FIELDS, 0, usize::from(field)), Some(field));
            let lane = binding
                .lane(&w.store, &w.view, lanes.frame_store(), &lanes, field)
                .unwrap();
            let by_id = behave(&w, &lanes, field, row, as_u64(lane, row));
            // The oracle, and the same row reached through its ordinal.
            let src = cols.tile(&order)[usize::from(field)]
                .iter()
                .copied()
                .find(|s| s.row == row)
                .unwrap();
            assert_eq!(Some(by_id.clone()), cols.expect(src));
            let ordinal = order.iter().position(|&r| r == row).unwrap();
            let by_ordinal =
                demand(method, &w, &lanes, &binding, ordinal, usize::from(field)).unwrap();
            assert_eq!(by_id, by_ordinal);
        }
    }
    assert!(moved_rows > POPULATED / 2);
    // A row number is not an id, and an unmapped id has no row.
    assert!(at.row_of(&table, 3, lanes.rows()).is_err());
    assert!(at.row_of(&table, 0xDEAD, lanes.rows()).is_err());
}
