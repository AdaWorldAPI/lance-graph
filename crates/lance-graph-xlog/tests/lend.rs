//! D-XLOG-W7-2 step 4a: a cube relation lent through DT64's lane socket.
//!
//! A relation is two lanes and a live plane; row `i` is `x[i] : y[i]`. The
//! tests take both lanes with ONE `borrow_all_present`, check they are the
//! cube's own, and fold over them.

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{LocalMap, MapTable, Scope, Width};
use lance_graph_dt64::{
    BindStore, ClassSchema, Extent, LaneBinding, LaneRefusal, LaneRequest, RouteAt, SchemaRev,
};
use lance_graph_mask_risc::{execute_into, Foreign, LaneRef, Out, Planes, Scratch};
use lance_graph_quack::{
    lower, normalize_group_sink, Agg, Col, Filter, GroupAddr, GroupAgg, Mask, Query,
};
use lance_graph_xlog::cube::{Cube, CubeLanes};

const TILE: u32 = 5;
const CLASS: u32 = 0x7F0E_0001;
const CLASS_VIEW: ClassId = 0x7F0E;
const REV: SchemaRev = SchemaRev(1);
const POSITION: u16 = 2;
const READER: Scope = Scope { id: 9 };
const X: &str = "lab:edge/x";
const Y: &str = "lab:edge/y";

struct View(Vec<FieldRef>);

impl View {
    /// `y` before `x`: the socket must map the predicate, never the position.
    fn new() -> Self {
        View(vec![FieldRef::new(Y, "y"), FieldRef::new(X, "x")])
    }
}

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == CLASS_VIEW {
            &self.0
        } else {
            &[]
        }
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Card
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

fn table(ids: &[u32]) -> MapTable {
    let mut all: Vec<u64> = ids.iter().map(|&i| u64::from(i)).collect();
    all.sort_unstable();
    all.dedup();
    let mut t = MapTable::new(1, 1);
    t.advance(
        1,
        1,
        vec![(TILE, Some(LocalMap::prepare(Width::W256, &all).unwrap()))],
    );
    t
}

/// 40 edges over ids 100..130, with repeats in both columns and one pair
/// held twice.
fn edges() -> Vec<(u32, u32)> {
    let mut e: Vec<(u32, u32)> = (0..39u32)
        .map(|i| (100 + (i * 7) % 13, 110 + (i * 5) % 19))
        .collect();
    e.push(e[3]);
    e
}

fn ids() -> Vec<u32> {
    (100..130).collect()
}

struct World {
    store: BindStore,
    view: View,
    route: RouteAt,
}

fn world() -> World {
    let mut store = BindStore::new();
    let view = View::new();
    store
        .register(
            &view,
            CLASS,
            ClassSchema {
                schema_rev: REV,
                field_count: 2,
                extent: Extent::DR64_CUBE,
            },
            Vec::new(),
        )
        .unwrap();
    let version = store.route(POSITION, CLASS).unwrap();
    World {
        store,
        view,
        route: RouteAt {
            position: POSITION,
            version,
        },
    }
}

fn bind(w: &World, source: &CubeLanes<'_>, table: &MapTable) -> Result<LaneBinding, LaneRefusal> {
    let frame = table.seal();
    w.store.bind_lanes(
        &w.view,
        table.frames(),
        source,
        LaneRequest {
            route: w.route,
            rev: REV,
            fields: &WideFieldMask::from_positions(&[0, 1]),
            frame: &frame,
            scope: READER,
            selection: source.space(),
        },
    )
}

/// The two lanes and their presence, by predicate: `(x, y, live)`.
type Lent<'s> = (&'s [u32], &'s [u32], &'s [u64]);

fn lent<'s>(
    w: &World,
    binding: &LaneBinding,
    source: &'s CubeLanes<'_>,
    table: &MapTable,
) -> Result<Lent<'s>, LaneRefusal> {
    let (mut x, mut y, mut live) = (None, None, None);
    for item in binding.borrow_all_present(&w.store, &w.view, table.frames(), source)? {
        let (field, lane, plane) = item?;
        let LaneRef::U32(values) = lane else {
            panic!("a relation lane is u32")
        };
        // Field 0 of the view is `y`, field 1 is `x`.
        if field == 0 {
            y = Some(values);
        } else {
            x = Some(values);
        }
        live = Some(plane);
    }
    Ok((x.unwrap(), y.unwrap(), live.unwrap()))
}

/// FAILS IF: the socket copies or rebuilds a lane for a reader, hands the
/// columns out by position instead of by predicate, or lends rows that are
/// not the relation's tuples.
#[test]
fn the_lent_lanes_are_the_cubes_own_tuples() {
    let (t, e) = (table(&ids()), edges());
    let cube = Cube::new(&t, &t.seal(), TILE, &[&e]).unwrap();
    let w = world();
    let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
    let binding = bind(&w, &source, &t).unwrap();

    let (x, y, live) = lent(&w, &binding, &source, &t).unwrap();
    assert_eq!((x.len(), y.len()), (e.len(), e.len()));
    assert!(live.len() * 64 >= e.len());

    // Row i is x[i] : y[i]; decoded through the tile's map it is edge i.
    let frame = t.seal();
    let map = t.map(&frame, TILE).unwrap();
    let decoded: Vec<(u32, u32)> = x
        .iter()
        .zip(y)
        .map(|(&a, &b)| {
            (
                map.global(u8::try_from(a).unwrap()).unwrap() as u32,
                map.global(u8::try_from(b).unwrap()).unwrap() as u32,
            )
        })
        .collect();
    assert_eq!(decoded, e);
    assert_ne!(x, y, "the two columns differ, so a swap would show");

    // A second borrow lends the same words: nothing was copied or rebuilt.
    let (x2, y2, live2) = lent(&w, &binding, &source, &t).unwrap();
    assert!(std::ptr::eq(x.as_ptr(), x2.as_ptr()));
    assert!(std::ptr::eq(y.as_ptr(), y2.as_ptr()));
    assert!(std::ptr::eq(live.as_ptr(), live2.as_ptr()));
}

/// FAILS IF: with a logged edge, `borrow_all` lends lanes that are rebuilt per
/// borrow, still the base's, or lanes and a live plane of different states.
///
/// "In place" names two memories. With an empty log the lent lanes are the
/// base lanes. With a logged edge they are the effective fold, built on the
/// first read and then lent in place by every later `borrow_all` until the
/// relation changes.
#[test]
fn with_a_logged_edge_borrow_all_lends_the_cached_fold_in_place() {
    let (t, e) = (table(&ids()), edges());
    let mut cube = Cube::new(&t, &t.seal(), TILE, &[&e]).unwrap();
    let w = world();

    let base_x = {
        let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
        let binding = bind(&w, &source, &t).unwrap();
        lent(&w, &binding, &source, &t).unwrap().0.as_ptr()
    };

    cube.add_edge(&t, 0, 101, 111).unwrap();
    let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
    let binding = bind(&w, &source, &t).unwrap();
    let (x, y, live) = lent(&w, &binding, &source, &t).unwrap();
    assert_eq!((x.len(), y.len()), (e.len() + 1, e.len() + 1));
    assert!(
        live.len() * 64 >= x.len(),
        "the plane covers the fold's rows"
    );
    assert!(
        !std::ptr::eq(x.as_ptr(), base_x),
        "a logged edge is read through the fold, not the base"
    );

    let (x2, y2, live2) = lent(&w, &binding, &source, &t).unwrap();
    assert!(
        std::ptr::eq(x.as_ptr(), x2.as_ptr()),
        "the fold is built once"
    );
    assert!(std::ptr::eq(y.as_ptr(), y2.as_ptr()));
    assert!(std::ptr::eq(live.as_ptr(), live2.as_ptr()));
}

/// FAILS IF: a fold over the lent lanes disagrees with the cube's own count
/// of the same column.
#[test]
fn a_fold_over_the_lent_lanes_equals_the_cubes_count() {
    let (t, e) = (table(&ids()), edges());
    let mut cube = Cube::new(&t, &t.seal(), TILE, &[&e]).unwrap();
    let scan = cube.scan(0).unwrap();
    let want: [Vec<i64>; 2] = [
        cube.column_counts(&t, scan, 0).unwrap(),
        cube.column_counts(&t, scan, 1).unwrap(),
    ];
    assert!(want[0].iter().filter(|&&c| c > 1).count() > 3, "repeats");
    assert_ne!(want[0], want[1]);

    let w = world();
    let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
    let binding = bind(&w, &source, &t).unwrap();
    let (x, y, live) = lent(&w, &binding, &source, &t).unwrap();

    let lanes = [LaneRef::U32(x), LaneRef::U32(y)];
    let masks = [live];
    let planes = Planes {
        n_rows: x.len(),
        masks: &masks,
        lanes: &lanes,
    };
    for (col, want) in want.iter().enumerate() {
        let program = lower(&Query {
            filter: Filter::plane(Mask(0)),
            agg: Agg::GroupReduce {
                key: GroupAddr::Local(Col(col as u16)),
                agg: GroupAgg::Count,
            },
        })
        .unwrap();
        let mut scratch = Scratch::for_program(&program, x.len()).unwrap();
        let mut sink = vec![0i64; want.len()];
        execute_into(
            &program,
            &planes,
            &Foreign::NONE,
            &mut scratch,
            Out::I64(&mut sink),
        )
        .unwrap();
        normalize_group_sink(GroupAgg::Count.fold(), &mut sink);
        assert_eq!(&sink, want, "column {col}");
    }
}

/// FAILS IF: a binding made before the relation changed is still honoured,
/// or one made before the table advanced.
#[test]
fn a_binding_goes_stale_with_the_relation_and_with_the_table() {
    let (mut t, e) = (table(&ids()), edges());
    let mut cube = Cube::new(&t, &t.seal(), TILE, &[&e]).unwrap();
    let w = world();

    let before = {
        let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
        let binding = bind(&w, &source, &t).unwrap();
        assert!(lent(&w, &binding, &source, &t).is_ok());
        binding
    };

    // An edge is logged: the cube's frame does not advance, the row order does.
    cube.add_edge(&t, 0, 101, 111).unwrap();
    let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
    assert!(lent(&w, &before, &source, &t).is_err(), "stale row order");
    let fresh = bind(&w, &source, &t).unwrap();
    let (x, _, _) = lent(&w, &fresh, &source, &t).unwrap();
    assert_eq!(x.len(), e.len() + 1);

    // The table advances: every earlier frame is refused.
    let map = LocalMap::prepare(
        Width::W256,
        &ids().iter().map(|&i| u64::from(i)).collect::<Vec<_>>(),
    )
    .unwrap();
    t.advance(2, 1, vec![(TILE, Some(map))]);
    let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
    assert!(matches!(
        lent(&w, &fresh, &source, &t),
        Err(LaneRefusal::Frame(_))
    ));
}

/// The relation's tuples as a sorted multiset of `(x, y)` local coordinates.
fn bag(x: &[u32], y: &[u32]) -> Vec<(u32, u32)> {
    let mut b: Vec<(u32, u32)> = x.iter().copied().zip(y.iter().copied()).collect();
    b.sort_unstable();
    b
}

/// FAILS IF: a binding made before a refold is still served after it, a
/// refold changes the relation's bag of tuples (it may only reorder rows), a
/// refold with an empty log invalidates a binding although nothing changed,
/// or a refold computes again the fold a read already cached.
///
/// `refold` is the one operation that replaces the lent lane memory (the old
/// lanes are dropped), so it is the case a zero-copy reader outside Rust's
/// borrow checker would be exposed to; the space check must refuse it here.
#[test]
fn a_binding_goes_stale_when_the_relation_is_refolded() {
    let (t, e) = (table(&ids()), edges());
    let mut cube = Cube::new(&t, &t.seal(), TILE, &[&e]).unwrap();
    let w = world();

    // A refold with an empty log changes nothing: a binding stays valid.
    let unchanged = {
        let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
        bind(&w, &source, &t).unwrap()
    };
    cube.refold(0).unwrap();
    {
        let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
        assert!(
            lent(&w, &unchanged, &source, &t).is_ok(),
            "an empty-log refold must not invalidate a binding"
        );
    }

    // Log one edge, then bind and read the effective bag.
    cube.add_edge(&t, 0, 101, 111).unwrap();
    let (before, bag_before, fold_at) = {
        let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
        let binding = bind(&w, &source, &t).unwrap();
        let (x, y, live) = lent(&w, &binding, &source, &t).unwrap();
        assert_eq!(x.len(), e.len() + 1);
        // Where the cached fold lives; compared as addresses only.
        let at = [
            x.as_ptr() as usize,
            y.as_ptr() as usize,
            live.as_ptr() as usize,
        ];
        (binding, bag(x, y), at)
    };

    cube.refold(0).unwrap();
    assert_eq!(cube.delta_len(0), 0, "refold empties the log");

    let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
    assert!(
        lent(&w, &before, &source, &t).is_err(),
        "a binding made before a refold must be refused after it"
    );
    // The binding from before any edge was logged had an empty log too: only
    // the base generation tells it apart from the refolded relation, whose
    // log is empty again.
    assert!(
        lent(&w, &unchanged, &source, &t).is_err(),
        "an empty-log binding of the old base must be refused after a refold"
    );
    let fresh = bind(&w, &source, &t).unwrap();
    let (x, y, live) = lent(&w, &fresh, &source, &t).unwrap();
    assert_eq!(
        bag(x, y),
        bag_before,
        "a refold may reorder rows but must keep the bag of tuples"
    );
    assert_eq!(
        [
            x.as_ptr() as usize,
            y.as_ptr() as usize,
            live.as_ptr() as usize
        ],
        fold_at,
        "a refold takes the cached fold as the base instead of computing it again"
    );
}

/// FAILS IF: a refold takes a cached fold that an edge logged after the read
/// has made stale, so the new base misses the late edge.
///
/// `refold` takes the cached fold as the new base. That is only sound if every
/// log append drops the cache; this pins the order read, log, refold.
#[test]
fn a_refold_after_a_late_edge_keeps_the_late_edge() {
    let (t, e) = (table(&ids()), edges());
    let mut cube = Cube::new(&t, &t.seal(), TILE, &[&e]).unwrap();
    let w = world();

    cube.add_edge(&t, 0, 101, 111).unwrap();
    let early = {
        let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
        let binding = bind(&w, &source, &t).unwrap();
        let (x, y, _) = lent(&w, &binding, &source, &t).unwrap();
        bag(x, y)
    };

    // The late edge, logged after the fold was cached and never read before
    // the refold; a pair not yet held.
    cube.add_edge(&t, 0, 129, 100).unwrap();
    cube.refold(0).unwrap();

    // The oracle: a cube whose base already holds both logged edges.
    let mut all = e.clone();
    all.extend([(101, 111), (129, 100)]);
    let oracle = Cube::new(&t, &t.seal(), TILE, &[&all]).unwrap();
    let late = {
        let source = CubeLanes::new(&oracle, &t, 0, CLASS, [X, Y], READER);
        let binding = bind(&w, &source, &t).unwrap();
        let (x, y, _) = lent(&w, &binding, &source, &t).unwrap();
        bag(x, y)
    };
    assert_eq!(late.len(), early.len() + 1);

    let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
    let binding = bind(&w, &source, &t).unwrap();
    let (x, y, _) = lent(&w, &binding, &source, &t).unwrap();
    assert_eq!(bag(x, y), late, "the refolded base holds the late edge");
}

/// FAILS IF: another class, another scope or an unknown predicate is served.
#[test]
fn what_the_socket_does_not_hold_is_refused() {
    let (t, e) = (table(&ids()), edges());
    let cube = Cube::new(&t, &t.seal(), TILE, &[&e]).unwrap();
    let w = world();

    let other_class = CubeLanes::new(&cube, &t, 0, CLASS + 1, [X, Y], READER);
    assert!(bind(&w, &other_class, &t).is_err());

    let unknown = CubeLanes::new(&cube, &t, 0, CLASS, [X, "lab:edge/z"], READER);
    assert!(bind(&w, &unknown, &t).is_err());

    let other_scope = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], Scope { id: 10 });
    assert!(bind(&w, &other_scope, &t).is_err());

    let no_relation = CubeLanes::new(&cube, &t, 7, CLASS, [X, Y], READER);
    let binding = bind(&w, &no_relation, &t);
    assert!(binding.is_err() || lent(&w, &binding.unwrap(), &no_relation, &t).is_err());

    let held = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
    assert!(bind(&w, &held, &t).is_ok());
}

/// FAILS IF: a binding made over one cube is honoured over another cube
/// built from the same map table, whose relation is in the same state and
/// holds other tuples.
#[test]
fn a_binding_over_one_cube_is_refused_over_another() {
    let t = table(&ids());
    let e = edges();
    let mut other: Vec<(u32, u32)> = e.iter().map(|&(a, b)| (b - 10, a + 10)).collect();
    other.reverse();
    assert_ne!(other, e);
    let first = Cube::new(&t, &t.seal(), TILE, &[&e]).unwrap();
    let second = Cube::new(&t, &t.seal(), TILE, &[&other]).unwrap();
    assert_eq!(
        (first.base_generation(0), first.delta_len(0)),
        (second.base_generation(0), second.delta_len(0)),
        "same relation state, so only the cube tells them apart"
    );
    let w = world();

    let a = CubeLanes::new(&first, &t, 0, CLASS, [X, Y], READER);
    let b = CubeLanes::new(&second, &t, 0, CLASS, [X, Y], READER);
    assert_ne!(a.space(), b.space());
    let binding = bind(&w, &a, &t).unwrap();
    assert!(lent(&w, &binding, &a, &t).is_ok());
    assert!(lent(&w, &binding, &b, &t).is_err());

    // The second cube's own binding reads the second cube's tuples.
    let own = bind(&w, &b, &t).unwrap();
    let (x, _, _) = lent(&w, &own, &b, &t).unwrap();
    let (xa, _, _) = lent(&w, &binding, &a, &t).unwrap();
    assert_ne!(x, xa);
}
