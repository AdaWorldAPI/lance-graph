//! D-DRDT-SPLIT-S7 over the cube: `BindStore` = `xlog::cube` × DT[`ClassView`]
//! × DR[`WideFieldMask`], kept in the `MapTable` log as a reusable pattern.
//!
//! A relation's binding (which class, which fields, which lanes, which row
//! order, which scope) is kept under the version of the table it was bound
//! at. It is handed out by reference and replayed; it is never replaced or
//! removed. What is kept is the mapping, never the contents (plan
//! `dr-widefieldmask-dt-classview-v1.md` §1, §1b; DECISION EXO-255, EXO-256).

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{FrameRefusal, LocalMap, MapTable, Scope, Width};
use lance_graph_dt64::{
    BindStore, ClassSchema, Extent, KeepRefusal, LaneBinding, LaneRefusal, LaneRequest, RouteAt,
    SchemaRev,
};
use lance_graph_mask_risc::LaneRef;
use lance_graph_xlog::cube::{Cube, CubeLanes};

const TILE: u32 = 5;
const CLASS: u32 = 0x7F30_0001;
const CLASS_VIEW: ClassId = 0x7F30;
const POSITION: u16 = 4;
const READER: Scope = Scope { id: 3 };
const X: &str = "lab:edge/x";
const Y: &str = "lab:edge/y";

struct View(Vec<FieldRef>);

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

fn ids() -> Vec<u64> {
    (100..130).collect()
}

fn table() -> MapTable {
    let mut t = MapTable::new(1, 1);
    t.advance(
        1,
        1,
        vec![(TILE, Some(LocalMap::prepare(Width::W256, &ids()).unwrap()))],
    );
    t
}

/// 30 edges with repeats in both columns and one pair held twice.
fn edges() -> Vec<(u32, u32)> {
    let mut e: Vec<(u32, u32)> = (0..29u32)
        .map(|i| (100 + (i * 7) % 13, 110 + (i * 5) % 19))
        .collect();
    e.push(e[2]);
    e
}

struct World {
    store: BindStore,
    view: View,
    route: RouteAt,
}

fn world() -> World {
    let view = View(vec![FieldRef::new(X, "x"), FieldRef::new(Y, "y")]);
    let mut store = BindStore::new();
    store
        .register(
            &view,
            CLASS,
            ClassSchema {
                schema_rev: SchemaRev(1),
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

/// The cube relation × the class view × the field mask {x, y}.
fn bind(w: &World, cube: &Cube, t: &MapTable) -> Result<LaneBinding, LaneRefusal> {
    let source = CubeLanes::new(cube, t, 0, CLASS, [X, Y], READER);
    let frame = t.seal();
    w.store.bind_lanes(
        &w.view,
        t.frames(),
        &source,
        LaneRequest {
            route: w.route,
            rev: SchemaRev(1),
            fields: &WideFieldMask::from_positions(&[0, 1]),
            frame: &frame,
            scope: READER,
            selection: source.space(),
        },
    )
}

/// The relation read through `binding`: `(x, y)` lanes, in place.
type Lanes<'s> = (&'s [u32], &'s [u32]);

fn read<'s>(
    w: &World,
    binding: &LaneBinding,
    source: &'s CubeLanes<'_>,
    t: &MapTable,
) -> Result<Lanes<'s>, LaneRefusal> {
    let mut cols: [&[u32]; 2] = [&[], &[]];
    for item in binding.borrow_all_present(&w.store, &w.view, t.frames(), source)? {
        let (field, lane, _) = item?;
        let LaneRef::U32(values) = lane else {
            panic!("a relation lane is u32")
        };
        cols[usize::from(field)] = values;
    }
    Ok((cols[0], cols[1]))
}

/// Rows decoded through DR's pointer table: `global(x[i]) : global(y[i])`.
fn decode(t: &MapTable, (x, y): Lanes<'_>) -> Vec<(u64, u64)> {
    let frame = t.seal();
    let map = t.map(&frame, TILE).unwrap();
    let g = |l: u32| map.global(u8::try_from(l).unwrap()).unwrap();
    x.iter().zip(y).map(|(&a, &b)| (g(a), g(b))).collect()
}

/// FAILS IF: a kept mapping, replayed by reference, does not reproduce the
/// assignment of a fresh bind bit for bit (same lane addresses, the cube's
/// own lanes), or the replay set of its version does not list it.
#[test]
fn a_kept_cube_mapping_replays_the_same_assignment() {
    let mut t = table();
    let e = edges();
    let cube = Cube::new(&t, &t.seal(), TILE, &[&e]).unwrap();
    let w = world();

    let binding = bind(&w, &cube, &t).unwrap();
    let id = binding.keep_in(&mut t).unwrap();

    let frame = t.seal();
    let kept = t.kept::<LaneBinding>(&frame, id).unwrap();
    let fresh = bind(&w, &cube, &t).unwrap();
    for field in [0u8, 1] {
        assert_eq!(kept.addr(field), fresh.addr(field), "field {field}");
    }
    let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
    let (kx, ky) = read(&w, kept, &source, &t).unwrap();
    let (fx, fy) = read(&w, &fresh, &source, &t).unwrap();
    assert!(
        std::ptr::eq(kx, fx) && std::ptr::eq(ky, fy),
        "the cube's own lanes"
    );

    let decoded = decode(&t, (kx, ky));
    let want: Vec<(u64, u64)> = e.iter().map(|&(a, b)| (a.into(), b.into())).collect();
    assert_eq!(decoded, want, "the replayed mapping reads the relation");

    let listed: Vec<_> = t
        .kept_under::<LaneBinding>(&frame)
        .unwrap()
        .map(|(k, _)| k)
        .collect();
    assert_eq!(listed, [id]);
}

/// FAILS IF: two replays of one kept mapping over unchanged contents differ.
#[test]
fn two_replays_over_unchanged_contents_are_bit_equal() {
    let mut t = table();
    let cube = Cube::new(&t, &t.seal(), TILE, &[&edges()]).unwrap();
    let w = world();
    let id = bind(&w, &cube, &t).unwrap().keep_in(&mut t).unwrap();

    let frame = t.seal();
    let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
    let first = decode(
        &t,
        read(&w, t.kept::<LaneBinding>(&frame, id).unwrap(), &source, &t).unwrap(),
    );
    let second = decode(
        &t,
        read(&w, t.kept::<LaneBinding>(&frame, id).unwrap(), &source, &t).unwrap(),
    );
    assert_eq!(first, second);
}

/// FAILS IF: a logged edge (row order moved, table version unchanged) lets
/// the kept mapping read the moved rows, or a re-kept mapping replaces the
/// old one instead of being appended.
#[test]
fn a_logged_edge_keeps_the_mapping_but_refuses_to_read_through_it() {
    let mut t = table();
    let mut cube = Cube::new(&t, &t.seal(), TILE, &[&edges()]).unwrap();
    let w = world();
    let old = bind(&w, &cube, &t).unwrap().keep_in(&mut t).unwrap();

    cube.add_edge(&t, 0, 101, 111).unwrap();
    let frame = t.seal();
    let kept = t
        .kept::<LaneBinding>(&frame, old)
        .expect("still the mapping of this version");
    let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
    assert!(
        read(&w, kept, &source, &t).is_err(),
        "the row order the mapping names has moved"
    );

    let new = bind(&w, &cube, &t).unwrap().keep_in(&mut t).unwrap();
    assert_ne!(new, old);
    let frame = t.seal();
    let listed: Vec<_> = t
        .kept_under::<LaneBinding>(&frame)
        .unwrap()
        .map(|(k, _)| k)
        .collect();
    assert_eq!(listed, [old, new], "append-only: the old mapping stays");
    let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
    let (x, _) = read(&w, t.kept::<LaneBinding>(&frame, new).unwrap(), &source, &t).unwrap();
    assert_eq!(x.len(), edges().len() + 1);
}

/// FAILS IF: after the table advances, the old mapping is handed out under
/// the new version, or is no longer the mapping of its own version.
#[test]
fn an_advanced_table_does_not_witness_the_old_mapping() {
    let mut t = table();
    let cube = Cube::new(&t, &t.seal(), TILE, &[&edges()]).unwrap();
    let w = world();
    let before = t.seal();
    let id = bind(&w, &cube, &t).unwrap().keep_in(&mut t).unwrap();

    t.advance(
        2,
        1,
        vec![(TILE, Some(LocalMap::prepare(Width::W256, &ids()).unwrap()))],
    );
    let after = t.seal();
    assert_eq!(
        t.kept::<LaneBinding>(&after, id).err(),
        Some(KeepRefusal::NotWitnessed {
            kept_at: before.version(),
            asked: after.version(),
        })
    );
    assert_eq!(t.kept_under::<LaneBinding>(&after).unwrap().count(), 0);

    // Still the mapping of its own version: replayable as a mapping, but its
    // lanes are not lent under a frame that is no longer current.
    let replayed = t
        .kept::<LaneBinding>(&before, id)
        .expect("the mapping of its version");
    let source = CubeLanes::new(&cube, &t, 0, CLASS, [X, Y], READER);
    assert!(matches!(
        read(&w, replayed, &source, &t),
        Err(LaneRefusal::Frame(_))
    ));
}

/// FAILS IF: a mapping made over another table, or before this table's last
/// advance, is kept in this table's log.
#[test]
fn only_this_tables_current_mappings_are_kept() {
    let mut t = table();
    let other = table();
    let w = world();
    let theirs = Cube::new(&other, &other.seal(), TILE, &[&edges()]).unwrap();
    assert_eq!(
        bind(&w, &theirs, &other).unwrap().keep_in(&mut t).err(),
        Some(KeepRefusal::Frame(FrameRefusal::ForeignFrame))
    );

    let cube = Cube::new(&t, &t.seal(), TILE, &[&edges()]).unwrap();
    let stale = bind(&w, &cube, &t).unwrap();
    t.advance(
        2,
        1,
        vec![(TILE, Some(LocalMap::prepare(Width::W256, &ids()).unwrap()))],
    );
    assert!(matches!(
        stale.keep_in(&mut t).err(),
        Some(KeepRefusal::Frame(
            FrameRefusal::SourceChanged
                | FrameRefusal::MapChanged
                | FrameRefusal::CoordinatesChanged
        ))
    ));
}
