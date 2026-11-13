//! Coordinate and content are independent (D-VAS, canonical read law).
//!
//! ```text
//! Read(S, a, Vm, Vc) = Interpret_Vc( S[ Resolve_Vm(a) ] )
//! ```
//!
//! `S` is one publication of a real `MailboxSoA<64>`, `a` an id of DT64's
//! `MapTable`, `Vm` a sealed map version (`Coordinates`), `Vc` the class the
//! position routes to. Content changes go through the owner's `set_meta`
//! with NO `tick`: the mailbox's cycle is its own write protocol and plays
//! no part in any check here.
//!
//! Oracle: the assignments and the expected values are written here over
//! plain vectors; nothing of the map, the socket or the store computes them.

use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{LocalMap, MapTable, Scope, TileId, Width};
use lance_graph_dt64::{
    view_class, BindStore, ClassSchema, Extent, LaneBinding, LaneRequest, RouteAt, SchemaRev,
};
use lance_graph_mailbox_lanes::coords::{CoordRefusal, Coordinates};
use lance_graph_mailbox_lanes::{Column, MailboxLanes};
use lance_graph_mask_risc::LaneRef;

const N: usize = 64;
type Owner = MailboxSoA<N>;
const ROWS: usize = 40;
/// Reads field 0 as `meta`.
const AS_META: u32 = 0x0A1A_0007;
/// Reads field 0 as `temporal`: another interpretation of the same address.
/// Another CONCEPT (the high half): a `ClassView` basis is keyed by the
/// concept, so two classids of one concept share one basis.
const AS_TEMPORAL: u32 = 0x0A1B_0008;
const POSITION: u16 = 14;
const OTHER_POSITION: u16 = 15;
const READER: Scope = Scope { id: 3 };
const REV: SchemaRev = SchemaRev(1);
const FIELD: u8 = 0;
const TILE: TileId = 7;

fn meta_of(r: usize) -> u32 {
    1_000 + 7 * r as u32
}
fn temporal_of(r: usize) -> u64 {
    500_000 + 13 * r as u64
}
/// The id of entity `i`: not a row number.
fn id(i: usize) -> u64 {
    0xA000_0000_0000 + (i as u64 * 7919) % 1_000_000 * 1_000 + i as u64
}
/// Two assignments of entities to rows.
fn row_v1(i: usize) -> usize {
    (i * 7 + 3) % ROWS
}
fn row_v2(i: usize) -> usize {
    (i * 11 + 5) % ROWS
}

fn owner(mailbox: u32) -> Box<Owner> {
    let mut m: Box<Owner> = Box::new(MailboxSoA::new(mailbox, 3, 1.0));
    m.set_populated(ROWS);
    for r in 0..ROWS {
        m.set_meta(r, MetaWord(meta_of(r)));
        m.set_temporal(r, temporal_of(r));
    }
    m
}

fn map_of(row_of: fn(usize) -> usize) -> LocalMap {
    let mut globals = vec![0u64; ROWS];
    for i in 0..ROWS {
        globals[row_of(i)] = id(i);
    }
    LocalMap::prepare(Width::W64, &globals).unwrap()
}

struct View {
    meta: Vec<FieldRef>,
    temporal: Vec<FieldRef>,
}

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == view_class(AS_META) {
            &self.meta
        } else if class == view_class(AS_TEMPORAL) {
            &self.temporal
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
    let view = View {
        meta: vec![FieldRef::new(Column::Meta.iri(), "meta")],
        temporal: vec![FieldRef::new(Column::Temporal.iri(), "temporal")],
    };
    let mut store = BindStore::new();
    let schema = ClassSchema {
        schema_rev: REV,
        field_count: 1,
        extent: Extent::DR256_SQ_X64,
    };
    store.register(&view, AS_META, schema, vec![]).unwrap();
    store.register(&view, AS_TEMPORAL, schema, vec![]).unwrap();
    let version = store.route(POSITION, AS_META).unwrap();
    World {
        store,
        view,
        route: RouteAt {
            position: POSITION,
            version,
        },
    }
}

fn bind_at(
    w: &World,
    lanes: &MailboxLanes<'_, N>,
    route: RouteAt,
) -> Result<LaneBinding, lance_graph_dt64::LaneRefusal> {
    let frame = lanes.seal();
    w.store.bind_lanes(
        &w.view,
        lanes.frame_store(),
        lanes,
        LaneRequest {
            route,
            rev: REV,
            fields: &WideFieldMask::from_positions(&[FIELD]),
            frame: &frame,
            scope: READER,
            selection: lanes.space(),
        },
    )
}

/// `Read(S, a, Vm, Vc)`: the coordinate first, then one read of the lane the
/// routed class's field 0 names. Returns the row and the value.
fn read(
    w: &World,
    lanes: &MailboxLanes<'_, N>,
    binding: &LaneBinding,
    table: &MapTable,
    at: &Coordinates,
    entity: usize,
) -> Result<(u32, u64), String> {
    let row = at
        .row_of(table, id(entity), lanes.rows())
        .map_err(|e| format!("{e:?}"))?;
    let lane = binding
        .lane(&w.store, &w.view, lanes.frame_store(), lanes, FIELD)
        .map_err(|e| format!("{e:?}"))?;
    let value = match lane {
        LaneRef::U32(v) => u64::from(v[row as usize]),
        LaneRef::U64(v) => v[row as usize],
        other => panic!("a mailbox column: {other:?}"),
    };
    Ok((row, value))
}

/// The content change of the matrix: every row's `meta` gets another value,
/// through the owner's setter, and the owner's cycle does not move.
fn change_contents(m: &mut Owner) {
    let cycle = m.current_cycle;
    for r in 0..ROWS {
        m.set_meta(r, MetaWord(meta_of(r) + 90_000));
    }
    assert_eq!(m.current_cycle, cycle, "no tick is involved");
}

fn table_v1() -> MapTable {
    let mut table = MapTable::new(7, 0);
    table.advance(7, 0, vec![(TILE, Some(map_of(row_v1)))]);
    table
}

/// The 2 x 2 matrix. FAILS IF: a mapping change moves a source byte or a
/// content change moves a coordinate, or either reading differs from the
/// assignment and the values written above.
#[test]
fn mapping_and_contents_vary_independently() {
    let mut m = owner(7);
    let w = world();
    let mut table = table_v1();
    let bytes_before: Vec<u32> = (0..ROWS).map(|r| m.meta[r].0).collect();

    // Same contents.
    {
        let lanes = MailboxLanes::new(m.published(), READER);
        let binding = bind_at(&w, &lanes, w.route).unwrap();
        let v1 = Coordinates::bind(&table, TILE);
        let mut moved = 0;
        for i in 0..ROWS {
            // Same contents, same mapping: the same coordinate and value,
            // twice.
            let first = read(&w, &lanes, &binding, &table, &v1, i).unwrap();
            assert_eq!(first, (row_v1(i) as u32, u64::from(meta_of(row_v1(i)))));
            assert_eq!(read(&w, &lanes, &binding, &table, &v1, i).unwrap(), first);
        }
        // Same contents, another mapping: the map advances while the
        // publication is live. The lane binding is untouched.
        table.advance(7, 0, vec![(TILE, Some(map_of(row_v2)))]);
        let v2 = Coordinates::bind(&table, TILE);
        for i in 0..ROWS {
            let got = read(&w, &lanes, &binding, &table, &v2, i).unwrap();
            assert_eq!(got, (row_v2(i) as u32, u64::from(meta_of(row_v2(i)))));
            moved += usize::from(row_v1(i) != row_v2(i));
        }
        assert!(moved > ROWS / 2, "the two assignments really differ");
    }
    let bytes_after: Vec<u32> = (0..ROWS).map(|r| m.meta[r].0).collect();
    assert_eq!(bytes_after, bytes_before, "a mapping change moved no byte");

    // Other contents.
    change_contents(&mut m);
    let lanes = MailboxLanes::new(m.published(), READER);
    let binding = bind_at(&w, &lanes, w.route).unwrap();
    // Other contents, same mapping (v2, still current): the coordinate is
    // what it was, the value is the new one.
    let v2 = Coordinates::bind(&table, TILE);
    for i in 0..ROWS {
        let got = read(&w, &lanes, &binding, &table, &v2, i).unwrap();
        let row = row_v2(i);
        assert_eq!(got, (row as u32, u64::from(meta_of(row) + 90_000)));
    }
    // Other contents, other mapping: back to the first assignment, as a new
    // map version. Both changes compose.
    table.advance(7, 0, vec![(TILE, Some(map_of(row_v1)))]);
    let v3 = Coordinates::bind(&table, TILE);
    for i in 0..ROWS {
        let got = read(&w, &lanes, &binding, &table, &v3, i).unwrap();
        let row = row_v1(i);
        assert_eq!(got, (row as u32, u64::from(meta_of(row) + 90_000)));
    }
}

/// FAILS IF: a coordinate binding of an earlier map version resolves against
/// the current map, or against another table.
#[test]
fn an_obsolete_mapping_does_not_resolve() {
    let m = owner(7);
    let mut table = table_v1();
    let lanes = MailboxLanes::new(m.published(), READER);
    let v1 = Coordinates::bind(&table, TILE);
    assert_eq!(v1.row_of(&table, id(3), lanes.rows()), Ok(row_v1(3) as u32));
    table.advance(7, 0, vec![(TILE, Some(map_of(row_v2)))]);
    // Not the new row, not the old row: refused.
    assert_ne!(row_v1(3), row_v2(3));
    assert_eq!(
        v1.row_of(&table, id(3), lanes.rows()),
        Err(CoordRefusal::StaleMapping)
    );
    // Another table with the same content is another map.
    let twin = table_v1();
    assert_eq!(
        Coordinates::bind(&table, TILE).row_of(&twin, id(3), lanes.rows()),
        Err(CoordRefusal::ForeignMap)
    );
}

/// FAILS IF: a lane binding outlives its publication, or an unchanged map
/// version is taken as evidence that a value read earlier is still current.
#[test]
fn an_unchanged_mapping_does_not_validate_an_earlier_value() {
    let mut m = owner(7);
    let w = world();
    let table = table_v1();
    let at = Coordinates::bind(&table, TILE);

    let (old_binding, cached) = {
        let lanes = MailboxLanes::new(m.published(), READER);
        let binding = bind_at(&w, &lanes, w.route).unwrap();
        let cached = read(&w, &lanes, &binding, &table, &at, 3).unwrap();
        (binding, cached)
    };
    change_contents(&mut m);

    let lanes = MailboxLanes::new(m.published(), READER);
    // The mapping is as valid as before and names the same row.
    assert_eq!(at.row_of(&table, id(3), lanes.rows()), Ok(cached.0));
    // The value there is another one: the map version says nothing about it.
    let fresh_binding = bind_at(&w, &lanes, w.route).unwrap();
    let fresh = read(&w, &lanes, &fresh_binding, &table, &at, 3).unwrap();
    assert_eq!(fresh.0, cached.0);
    assert_ne!(fresh.1, cached.1);
    assert_eq!(fresh.1, cached.1 + 90_000);
    // The only witness of the earlier read, its lane binding, is refused.
    assert!(read(&w, &lanes, &old_binding, &table, &at, 3).is_err());
}

/// FAILS IF: after the position is routed to another class, a binding made
/// for the first interpretation still reads, or the new interpretation
/// returns the old one's column.
#[test]
fn another_interpretation_is_not_served_by_the_old_binding() {
    let m = owner(7);
    let mut w = world();
    let table = table_v1();
    let at = Coordinates::bind(&table, TILE);
    let lanes = MailboxLanes::new(m.published(), READER);

    let as_meta = bind_at(&w, &lanes, w.route).unwrap();
    let row = row_v1(3);
    assert_eq!(
        read(&w, &lanes, &as_meta, &table, &at, 3).unwrap(),
        (row as u32, u64::from(meta_of(row)))
    );

    let version = w.store.route(POSITION, AS_TEMPORAL).unwrap();
    // The source and the mapping did not change; the interpretation did.
    assert!(read(&w, &lanes, &as_meta, &table, &at, 3).is_err());
    // The earlier route no longer binds either.
    assert!(bind_at(&w, &lanes, w.route).is_err());
    let now = RouteAt {
        position: POSITION,
        version,
    };
    let as_temporal = bind_at(&w, &lanes, now).unwrap();
    assert_eq!(
        read(&w, &lanes, &as_temporal, &table, &at, 3).unwrap(),
        (row as u32, temporal_of(row))
    );
    assert_ne!(u64::from(meta_of(row)), temporal_of(row));
}

/// FAILS IF: a binding made over one owner reads another owner's columns, or
/// a position nothing was routed to binds.
#[test]
fn a_foreign_source_or_route_is_refused() {
    let a = owner(7);
    let b = owner(8);
    let w = world();
    let table = table_v1();
    let at = Coordinates::bind(&table, TILE);
    let lanes_a = MailboxLanes::new(a.published(), READER);
    let lanes_b = MailboxLanes::new(b.published(), READER);
    let binding_a = bind_at(&w, &lanes_a, w.route).unwrap();
    assert!(read(&w, &lanes_a, &binding_a, &table, &at, 3).is_ok());
    assert!(read(&w, &lanes_b, &binding_a, &table, &at, 3).is_err());
    // A second socket over the SAME owner is another publication.
    let lanes_a2 = MailboxLanes::new(a.published(), READER);
    assert!(read(&w, &lanes_a2, &binding_a, &table, &at, 3).is_err());

    let unrouted = RouteAt {
        position: OTHER_POSITION,
        version: w.route.version,
    };
    assert!(bind_at(&w, &lanes_a, unrouted).is_err());
    // Another reader scope than the socket's.
    let frame = lanes_a.seal();
    assert!(w
        .store
        .bind_lanes(
            &w.view,
            lanes_a.frame_store(),
            &lanes_a,
            LaneRequest {
                route: w.route,
                rev: REV,
                fields: &WideFieldMask::from_positions(&[FIELD]),
                frame: &frame,
                scope: Scope { id: 4 },
                selection: lanes_a.space(),
            },
        )
        .is_err());
}
