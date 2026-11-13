//! D-VAS-B2: one id, unchanged source bytes, two coordinate versions.
//!
//! The id is `MapTable`'s `u64` global, not a `NodeGuid` (B1 is BLOCKED; see
//! `src/coords.rs`). The test fills the map; no authority does that for a
//! mailbox. Oracle: two permutations written here, over `owner.meta`.

use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{LocalMap, MapRefusal, MapTable, Scope, TileId, Width};
use lance_graph_dt64::{
    view_class, BindStore, ClassSchema, Extent, LaneBinding, LaneRequest, RouteAt, SchemaRev,
};
use lance_graph_mailbox_lanes::coords::{CoordRefusal, Coordinates};
use lance_graph_mailbox_lanes::{Column, MailboxLanes};
use lance_graph_mask_risc::LaneRef;

const N: usize = 64;
type Owner = MailboxSoA<N>;
const ROWS: usize = 48;
const CLASS: u32 = 0x0A1A_0003;
const POSITION: u16 = 11;
const READER: Scope = Scope { id: 3 };
const META: u8 = 0;
const TILE: TileId = 7;

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The id of entity `i`: not a row number.
fn id(i: usize) -> u64 {
    0xA000_0000_0000 + mix(i as u64) % 1_000_000 * 1_000 + i as u64
}
/// Two assignments of entities to rows.
fn row_v1(i: usize) -> usize {
    (i * 7 + 3) % ROWS
}
fn row_v2(i: usize) -> usize {
    (i * 11 + 5) % ROWS
}

fn owner() -> Box<Owner> {
    let mut m: Box<Owner> = Box::new(MailboxSoA::new(7, 3, 1.0));
    m.set_populated(ROWS);
    for r in 0..ROWS {
        m.set_meta(r, MetaWord(mix(9_000 + r as u64) as u32));
    }
    m
}

/// The map for an assignment: position `row` holds the id of the entity
/// assigned to it.
fn map_of(row_of: fn(usize) -> usize) -> LocalMap {
    let mut globals = vec![0u64; ROWS];
    for i in 0..ROWS {
        globals[row_of(i)] = id(i);
    }
    LocalMap::prepare(Width::W64, &globals).unwrap()
}

struct View(Vec<FieldRef>);

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == view_class(CLASS) {
            &self.0
        } else {
            &[]
        }
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

struct World {
    store: BindStore,
    view: View,
    route: RouteAt,
}

fn world() -> World {
    let view = View(vec![FieldRef::new(Column::Meta.iri(), "meta")]);
    let mut store = BindStore::new();
    store
        .register(
            &view,
            CLASS,
            ClassSchema {
                schema_rev: SchemaRev(1),
                field_count: 1,
                extent: Extent::DR256_SQ_X64,
            },
            vec![],
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

fn bind(w: &World, lanes: &MailboxLanes<'_, N>) -> LaneBinding {
    let frame = lanes.seal();
    w.store
        .bind_lanes(
            &w.view,
            lanes.frame_store(),
            lanes,
            LaneRequest {
                route: w.route,
                rev: SchemaRev(1),
                fields: &WideFieldMask::from_positions(&[META]),
                frame: &frame,
                scope: READER,
                selection: lanes.space(),
            },
        )
        .unwrap()
}

/// The property of entity `i`: coordinate first, then one source read.
fn read(
    w: &World,
    lanes: &MailboxLanes<'_, N>,
    b: &LaneBinding,
    table: &MapTable,
    at: &Coordinates,
    i: usize,
) -> Result<u32, CoordRefusal> {
    let row = at.row_of(table, id(i), lanes.rows())?;
    let LaneRef::U32(meta) = b
        .lane(&w.store, &w.view, lanes.frame_store(), lanes, META)
        .unwrap()
    else {
        panic!("a u32 lane")
    };
    Ok(meta[row as usize])
}

/// FAILS IF: a binding of the first coordinate version reads through the
/// second, the two versions read the same rows, or advancing the map touches
/// the mailbox.
#[test]
fn one_id_two_coordinate_versions_same_bytes() {
    let m = owner();
    let w = world();
    let lanes = MailboxLanes::new(m.published(), READER);
    let b = bind(&w, &lanes);
    let before: Vec<u32> = (0..ROWS).map(|r| m.meta[r].0).collect();

    let mut table = MapTable::new(7, 0);
    table.advance(7, 0, vec![(TILE, Some(map_of(row_v1)))]);
    let v1 = Coordinates::bind(&table, TILE);
    let mut moved = 0;
    for i in 0..ROWS {
        assert_eq!(
            read(&w, &lanes, &b, &table, &v1, i),
            Ok(m.meta[row_v1(i)].0)
        );
        moved += usize::from(row_v1(i) != row_v2(i));
    }
    assert!(moved > ROWS / 2, "the two assignments really differ");

    // A new coordinate version. No source byte moves, and the lane binding
    // made before stays valid: the publication did not change.
    table.advance(7, 0, vec![(TILE, Some(map_of(row_v2)))]);
    assert_eq!((0..ROWS).map(|r| m.meta[r].0).collect::<Vec<_>>(), before);
    let v2 = Coordinates::bind(&table, TILE);
    for i in 0..ROWS {
        assert_eq!(
            read(&w, &lanes, &b, &table, &v2, i),
            Ok(m.meta[row_v2(i)].0)
        );
        assert_eq!(
            read(&w, &lanes, &b, &table, &v1, i),
            Err(CoordRefusal::StaleMapping),
            "entity {i}"
        );
    }
    // The old assignment is still history, readable as such.
    assert_eq!(
        table.map(v1.frame(), TILE).unwrap().local(id(4)),
        Ok(row_v1(4) as u8)
    );
    assert_eq!(v1.frame().version().0 + 1, v2.frame().version().0);
}

/// FAILS IF: an id the map does not hold, another table's frame, an unknown
/// tile, a row past the lane or a repeated id resolves.
#[test]
fn what_does_not_resolve() {
    let m = owner();
    let lanes = MailboxLanes::new(m.published(), READER);
    let mut table = MapTable::new(7, 0);
    table.advance(7, 0, vec![(TILE, Some(map_of(row_v1)))]);
    let at = Coordinates::bind(&table, TILE);
    assert_eq!(at.row_of(&table, id(3), lanes.rows()), Ok(row_v1(3) as u32));
    assert_eq!(
        at.row_of(&table, 0xDEAD, lanes.rows()),
        Err(CoordRefusal::Unmapped)
    );
    // A row number is not an id.
    assert_eq!(
        at.row_of(&table, 3, lanes.rows()),
        Err(CoordRefusal::Unmapped)
    );

    let mut twin = MapTable::new(7, 0);
    twin.advance(7, 0, vec![(TILE, Some(map_of(row_v1)))]);
    assert_eq!(
        at.row_of(&twin, id(3), lanes.rows()),
        Err(CoordRefusal::ForeignMap)
    );
    assert_eq!(
        Coordinates::bind(&table, 99).row_of(&table, id(3), lanes.rows()),
        Err(CoordRefusal::NoSuchTile(99))
    );
    // A lane shorter than the map: the row exists in the map, not in the lane.
    let short = row_v1(3);
    assert_eq!(
        at.row_of(&table, id(3), short),
        Err(CoordRefusal::RowOutsideLane {
            row: short as u32,
            rows: short
        })
    );
    // Two entities cannot share an id: the map refuses to be built.
    assert_eq!(
        LocalMap::prepare(Width::W64, &[id(1), id(2), id(1)]).err(),
        Some(MapRefusal::Duplicate)
    );
    // And one map position holds one id, so two ids never share a row.
    let mut rows: Vec<u32> = (0..ROWS)
        .map(|i| at.row_of(&table, id(i), lanes.rows()).unwrap())
        .collect();
    rows.sort_unstable();
    rows.dedup();
    assert_eq!(rows.len(), ROWS);
}
