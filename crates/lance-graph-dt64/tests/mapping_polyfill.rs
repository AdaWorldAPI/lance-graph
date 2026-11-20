//! D-MAPPING-POLYFILL-1: a PSCustomObject-style field mapping read through the
//! existing DT64 / DR substrate, never through a table of tuples.
//!
//! ```text
//! Source:    DirectoryUsers     (one source owner: its MapTable + its lanes)
//! Selection: AuthorizedUsers    (a position list the source itself holds)
//! Fields:    Identity    <- dir:objectId      (the identity key, MapTable)
//!            Name        <- dir:displayName   (lane)
//!            PrimaryMail <- dir:mail          (lane)
//!            Groups      <- dir:memberOf      (a relation, not a lane: refused here)
//! ```
//!
//! The aliases on the left are display metadata. They are resolved to a field
//! position through the class basis (`predicate_iri`) once, at bind, and play
//! no part in identity, routing or authority.
//!
//! Every read goes through `BindStore::bind_lanes` / `LaneBinding`; identity
//! keys go through `MapTable::locate`. The source returns the table's own
//! frame store (`MapTable::frames`), so a key resolved at a frame and a lane
//! read at that frame are checked by one witness. Values are read only at the
//! terminal, per selected position; the counters below show that nothing on
//! the way there enumerates rows, builds a tuple or allocates.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{
    Frame, FrameRefusal, FrameRevisions, FrameStore, LocalMap, LocateRefusal, MapRefusal, MapTable,
    Scope, TileId,
};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneBinding, LaneRefusal, LaneRequest,
    LaneSource, RouteAt, SchemaRev, SelAddr, SelectionSource, Width,
};

// ---------------------------------------------------------------------------
// Allocation counter (this test binary only; per thread, so parallel tests
// do not see each other's allocations).

struct Counting;

thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|a| a.set(a.get() + 1));
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> u64 {
    ALLOCS.with(Cell::get)
}

// ---------------------------------------------------------------------------
// The class and its basis.

const USER: u32 = 0x0D1A_0001;
const VIEW_CLASS: ClassId = 0x0D1A;
const POSITION: u16 = 9;
const PROPS: usize = 64;
const TILE: usize = 256;

const F_OBJECT_ID: u8 = 0;
const F_NAME: u8 = 1;
const F_MAIL: u8 = 2;
const F_MEMBER_OF: u8 = 3;
/// A field the source serves only to another reader scope.
const F_SALARY: u8 = 10;

const READER: Scope = Scope { id: 7 };
const HR_ONLY: Scope = Scope { id: 99 };

/// 64 fields: the four of the recipe, a restricted one, and filler.
struct View(Vec<FieldRef>);

impl View {
    fn directory() -> View {
        let mut f: Vec<FieldRef> = (0..PROPS)
            .map(|i| FieldRef::new(format!("dir:p{i}"), format!("P{i}")))
            .collect();
        f[0] = FieldRef::new("dir:objectId", "Object id");
        f[1] = FieldRef::new("dir:displayName", "Display name");
        f[2] = FieldRef::new("dir:mail", "Mail");
        f[3] = FieldRef::new("dir:memberOf", "Member of");
        f[10] = FieldRef::new("hr:salary", "Salary");
        View(f)
    }
    /// The same class with name and mail swapped in the basis.
    fn swapped() -> View {
        let mut v = View::directory();
        v.0.swap(1, 2);
        v
    }
}

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == VIEW_CLASS {
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

/// The recipe: alias -> predicate. Display metadata only.
const RECIPE: [(&str, &str); 4] = [
    ("Identity", "dir:objectId"),
    ("Name", "dir:displayName"),
    ("PrimaryMail", "dir:mail"),
    ("Groups", "dir:memberOf"),
];

/// Resolve an alias to a field position through the view's basis.
fn field_of(view: &View, alias: &str) -> Option<u8> {
    let iri = RECIPE.iter().find(|(a, _)| *a == alias)?.1;
    let pos = view
        .fields(VIEW_CLASS)
        .iter()
        .position(|f| f.predicate_iri == iri)?;
    u8::try_from(pos).ok()
}

// ---------------------------------------------------------------------------
// The source owner: one MapTable (identity -> coordinate) and its lanes.

/// One source-owned lane: values plus presence and tombstone bits. A missing
/// value, an explicit zero and a deleted value are three different states.
#[derive(Debug)]
struct Cells {
    values: Vec<u32>,
    present: Vec<u64>,
    deleted: Vec<u64>,
}

#[derive(Debug, PartialEq, Eq)]
enum Read {
    Value(u32),
    Missing,
    Deleted,
}

fn bit(b: &[u64], i: usize) -> bool {
    b[i / 64] >> (i % 64) & 1 == 1
}

fn read(c: &Cells, row: u32) -> Read {
    let r = row as usize;
    if bit(&c.deleted, r) {
        Read::Deleted
    } else if bit(&c.present, r) {
        Read::Value(c.values[r])
    } else {
        Read::Missing
    }
}

/// The identity of user `u` (sparse, not the row number).
fn object_id(u: usize) -> u64 {
    1_000_003 + 7 * u as u64
}

/// The oracle: what the directory holds for user `u`, independent of rows.
fn oracle(field: u8, u: usize) -> Read {
    match field {
        F_NAME => Read::Value(100 + u as u32),
        F_MAIL if u == 3 => Read::Missing,
        F_MAIL if u == 4 => Read::Deleted,
        F_MAIL => Read::Value(3 * u as u32),
        F_SALARY => Read::Value(50_000 + u as u32),
        _ => Read::Value(u as u32),
    }
}

struct Source {
    table: MapTable,
    tiles: Vec<TileId>,
    /// `row_user[row]` = the user stored at that row.
    row_user: Vec<usize>,
    lanes: Vec<Cells>,
    /// Field -> (lane, scope). `dir:memberOf` has no lane.
    map: Vec<(u8, u16, Scope)>,
    selections: Vec<Vec<u32>>,
    lane_of_calls: Cell<u64>,
    borrows: Cell<u64>,
}

impl Source {
    /// `users` users, rows in `order` (a permutation of user numbers).
    fn new(users: usize, order: &[usize]) -> Source {
        let mut table = MapTable::new(1, 1);
        let mut s = Source {
            table: MapTable::new(0, 0),
            tiles: Vec::new(),
            row_user: Vec::new(),
            lanes: Vec::new(),
            map: Vec::new(),
            selections: Vec::new(),
            lane_of_calls: Cell::new(0),
            borrows: Cell::new(0),
        };
        s.load(&mut table, users, order);
        s.table = table;
        s
    }

    fn load(&mut self, table: &mut MapTable, users: usize, order: &[usize]) {
        assert_eq!(order.len(), users);
        let tiles: Vec<TileId> = (0..users.div_ceil(TILE) as u32).collect();
        let changes = tiles
            .iter()
            .map(|&t| {
                let ids: Vec<u64> = order[t as usize * TILE..((t as usize + 1) * TILE).min(users)]
                    .iter()
                    .map(|&u| object_id(u))
                    .collect();
                (t, Some(LocalMap::prepare(Width::W256, &ids).unwrap()))
            })
            .collect();
        table.advance(1, 1, changes);
        let words = users.div_ceil(64);
        let lane = |field: u8| {
            let mut c = Cells {
                values: vec![0; users],
                present: vec![0; words],
                deleted: vec![0; words],
            };
            for (row, &u) in order.iter().enumerate() {
                match oracle(field, u) {
                    Read::Value(v) => {
                        c.values[row] = v;
                        c.present[row / 64] |= 1 << (row % 64);
                    }
                    Read::Deleted => c.deleted[row / 64] |= 1 << (row % 64),
                    Read::Missing => {}
                }
            }
            c
        };
        self.lanes = (0..PROPS as u8).map(lane).collect();
        self.map = (0..PROPS as u8)
            .filter(|&f| f != F_MEMBER_OF)
            .map(|f| {
                (
                    f,
                    u16::from(f),
                    if f == F_SALARY { HR_ONLY } else { READER },
                )
            })
            .collect();
        self.row_user = order.to_vec();
        self.tiles = tiles;
    }

    /// Replace the rows (same users, another order) under a new frame.
    fn reorder(&mut self, order: &[usize]) {
        let mut table = std::mem::replace(&mut self.table, MapTable::new(0, 0));
        self.load(&mut table, order.len(), order);
        self.table = table;
    }

    fn space(&self) -> CoordSpace {
        CoordSpace {
            population: 1,
            coordinates: self.table.current_version().0,
        }
    }

    /// The row of a user identity, through the table (the XVERWEIS key path).
    fn locate(&self, frame: &Frame, id: u64) -> Result<u32, LocateRefusal> {
        let (tile, local) = self.table.locate(frame, &self.tiles, id)?;
        Ok(tile * TILE as u32 + u32::from(local))
    }

    fn map_lookups(&self, frame: &Frame) -> u64 {
        self.tiles
            .iter()
            .map(|&t| self.table.map(frame, t).unwrap().lookups())
            .sum()
    }
}

impl LaneSource for Source {
    type Lane<'a> = &'a Cells;
    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        field: u8,
        _basis: &FieldRef,
        _frame: &Frame,
    ) -> Option<LaneAddr> {
        self.lane_of_calls.set(self.lane_of_calls.get() + 1);
        if class != USER || rev != SchemaRev(1) {
            return None;
        }
        let space = self.space();
        self.map
            .iter()
            .find(|(f, _, _)| *f == field)
            .map(|&(_, lane, scope)| LaneAddr { space, lane, scope })
    }
    fn borrow(&self, addr: &LaneAddr) -> Option<&Cells> {
        self.borrows.set(self.borrows.get() + 1);
        self.lanes.get(usize::from(addr.lane))
    }
    fn frames(&self) -> &FrameStore {
        self.table.frames()
    }
}

impl SelectionSource for Source {
    fn selection_of(&self, sel: u16, _frame: &Frame) -> Option<SelAddr> {
        self.selections.get(usize::from(sel))?;
        Some(SelAddr {
            space: self.space(),
            sel,
            scope: READER,
        })
    }
    fn borrow_positions(&self, addr: &SelAddr) -> Option<&[u32]> {
        self.selections
            .get(usize::from(addr.sel))
            .map(Vec::as_slice)
    }
}

// ---------------------------------------------------------------------------
// World.

struct World {
    store: BindStore,
    view: View,
    source: Source,
    route: RouteAt,
}

fn world(users: usize) -> World {
    let order: Vec<usize> = (0..users).collect();
    let mut store = BindStore::new();
    let schema = ClassSchema {
        schema_rev: SchemaRev(1),
        field_count: PROPS,
        extent: Extent::DR64_CUBE,
    };
    store
        .register(&View::directory(), USER, schema, Vec::new())
        .unwrap();
    let version = store.route(POSITION, USER).unwrap();
    World {
        store,
        view: View::directory(),
        source: Source::new(users, &order),
        route: RouteAt {
            position: POSITION,
            version,
        },
    }
}

fn bind(w: &World, fields: &WideFieldMask, frame: &Frame) -> Result<LaneBinding, LaneRefusal> {
    w.store.bind_lanes(
        &w.view,
        w.source.frames(),
        &w.source,
        LaneRequest {
            route: w.route,
            rev: SchemaRev(1),
            fields,
            frame,
            scope: READER,
            selection: w.source.space(),
        },
    )
}

/// The terminal: (alias, value) per selected position, built only here.
///
/// ONE `borrow_all_selected`: one check, every bound lane and the witnessed
/// positions. Row `i` of alias `a` is `lane(a)[positions[i]]`, read in place.
fn project(
    w: &World,
    b: &LaneBinding,
    sel: u16,
    aliases: &[&str],
) -> Result<Vec<Vec<Read>>, LaneRefusal> {
    let (lanes, rows) =
        b.borrow_all_selected(&w.store, &w.view, w.source.frames(), &w.source, sel)?;
    let mut by_field: [Option<&Cells>; PROPS] = [None; PROPS];
    for item in lanes {
        let (field, lane) = item?;
        by_field[usize::from(field)] = Some(lane);
    }
    let mut out: Vec<Vec<Read>> = Vec::new();
    for alias in aliases {
        let f = field_of(&w.view, alias).expect("alias in recipe");
        let lane = by_field[usize::from(f)].ok_or(LaneRefusal::NotBound { field: f })?;
        out.push(rows.iter().map(|&r| read(lane, r)).collect());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Falsifiers.

/// 1 + 10: five identities, two projected properties, equal to the oracle; the
/// path borrows two lanes, reads no map, builds no tuple and allocates nothing.
#[test]
fn five_identities_two_properties_equal_the_oracle_without_a_join() {
    let mut w = world(512);
    let picked = [5usize, 17, 300, 511, 0];
    w.source.selections = vec![picked.iter().map(|&u| u as u32).collect()];
    let frame = w.source.frames().seal();
    let fields = WideFieldMask::from_positions(&[F_NAME, F_MAIL]);
    let b = bind(&w, &fields, &frame).unwrap();
    assert_eq!(
        w.source.lane_of_calls.get(),
        2,
        "one lane_of per bound field"
    );

    // The hot path: two checked borrows, no map read, no allocation.
    let (lookups_before, borrows_before) = (w.source.map_lookups(&frame), w.source.borrows.get());
    let a0 = allocs();
    let mut seen = 0u64;
    for f in [F_NAME, F_MAIL] {
        let (lane, rows) = b
            .selected(&w.store, &w.view, w.source.frames(), &w.source, f, 0)
            .unwrap();
        for &r in rows {
            if read(lane, r) == oracle(f, w.source.row_user[r as usize]) {
                seen += 1;
            }
        }
    }
    assert_eq!(allocs() - a0, 0, "the projection allocates nothing");
    assert_eq!(seen, 10, "5 identities x 2 fields agree with the oracle");
    assert_eq!(w.source.borrows.get() - borrows_before, 2);
    assert_eq!(
        w.source.map_lookups(&frame),
        lookups_before,
        "no key lookup"
    );

    // The terminal, by alias.
    let got = project(&w, &b, 0, &["Name", "PrimaryMail"]).unwrap();
    for (i, &u) in picked.iter().enumerate() {
        assert_eq!(got[0][i], oracle(F_NAME, u));
        assert_eq!(got[1][i], oracle(F_MAIL, u));
    }
}

/// 2: rows reordered under a new frame. The old binding is refused before a
/// lane is borrowed; a fresh binding gives the same values per identity.
#[test]
fn reordered_rows_refuse_the_old_binding_and_agree_by_identity() {
    let mut w = world(512);
    let frame = w.source.frames().seal();
    let fields = WideFieldMask::from_positions(&[F_MAIL]);
    let old = bind(&w, &fields, &frame).unwrap();
    let reversed: Vec<usize> = (0..512).rev().collect();
    w.source.reorder(&reversed);
    let borrows = w.source.borrows.get();
    let err = old
        .lane(&w.store, &w.view, w.source.frames(), &w.source, F_MAIL)
        .unwrap_err();
    assert!(matches!(err, LaneRefusal::Frame(_)), "{err:?}");
    assert_eq!(w.source.borrows.get(), borrows, "refused before borrowing");

    let now = w.source.frames().seal();
    let b = bind(&w, &fields, &now).unwrap();
    let lane = b
        .lane(&w.store, &w.view, w.source.frames(), &w.source, F_MAIL)
        .unwrap();
    for u in [0usize, 5, 300, 511] {
        let row = w.source.locate(&now, object_id(u)).unwrap();
        assert_eq!(read(lane, row), oracle(F_MAIL, u), "user {u}");
    }
}

/// 3: duplicate keys never resolve to a first match.
#[test]
fn duplicate_and_absent_keys_are_refused_not_picked() {
    let id = object_id(1);
    assert_eq!(
        LocalMap::prepare(Width::W256, &[id, object_id(2), id]).err(),
        Some(MapRefusal::Duplicate)
    );
    let mut table = MapTable::new(1, 1);
    table.advance(
        1,
        1,
        vec![
            (
                0,
                Some(LocalMap::prepare(Width::W256, &[id, object_id(2)]).unwrap()),
            ),
            (
                1,
                Some(LocalMap::prepare(Width::W256, &[object_id(3), id]).unwrap()),
            ),
        ],
    );
    let f = table.seal();
    assert_eq!(table.locate(&f, &[0, 1], id), Err(LocateRefusal::Ambiguous));
    assert_eq!(table.locate(&f, &[0, 1], object_id(3)), Ok((1, 0)));
    assert_eq!(table.locate(&f, &[0, 1], 42), Err(LocateRefusal::Unmapped));
}

/// XVERWEIS by identity: key -> row through the table, then the requested
/// field, under the binding's own frame. A map advance refuses both.
#[test]
fn identity_key_lookup_reads_the_return_field_under_one_frame() {
    let mut w = world(512);
    let frame = w.source.frames().seal();
    let fields = WideFieldMask::from_positions(&[F_NAME]);
    let b = bind(&w, &fields, &frame).unwrap();
    let lane = b
        .lane(&w.store, &w.view, w.source.frames(), &w.source, F_NAME)
        .unwrap();
    let before = w.source.map_lookups(&frame);
    let row = w.source.locate(&frame, object_id(300)).unwrap();
    assert_eq!(read(lane, row), oracle(F_NAME, 300));
    assert_eq!(
        w.source.map_lookups(&frame) - before,
        2,
        "one lookup per listed tile, no scan of the lanes"
    );

    w.source.reorder(&(0..512).rev().collect::<Vec<_>>());
    assert!(matches!(
        b.check(&w.store, &w.view, w.source.frames(), &w.source),
        Err(LaneRefusal::Frame(_))
    ));
    // The old frame still reads its own map (history), never today's lanes.
    assert_eq!(w.source.locate(&frame, object_id(300)), Ok(300));
}

/// 4: a basis that moved `dir:mail` cannot redirect a bound Mail read, and
/// renaming an alias does not move the field.
#[test]
fn a_reordered_basis_cannot_redirect_mail_into_another_lane() {
    let w = world(512);
    let frame = w.source.frames().seal();
    assert_eq!(field_of(&w.view, "Identity"), Some(F_OBJECT_ID));
    let mail = field_of(&w.view, "PrimaryMail").unwrap();
    assert_eq!(mail, F_MAIL);
    let b = bind(&w, &WideFieldMask::from_positions(&[mail]), &frame).unwrap();
    let err = b
        .check(&w.store, &View::swapped(), w.source.frames(), &w.source)
        .unwrap_err();
    assert!(
        matches!(err, LaneRefusal::PredicateChanged { field: F_MAIL, .. })
            || matches!(err, LaneRefusal::Bind(_)),
        "{err:?}"
    );
    // The alias is display metadata: the field is found by predicate.
    assert_eq!(field_of(&View::swapped(), "PrimaryMail"), Some(F_NAME));
}

/// 5: users and groups are two coordinate mappings. The same number is a
/// different coordinate in each, and a user id is unmapped among groups. A
/// membership field is a relation, not a lane: the lane socket refuses it.
#[test]
fn users_and_groups_keep_separate_coordinates_and_membership_is_not_a_lane() {
    let w = world(512);
    let shared = object_id(7);
    let mut groups = MapTable::new(2, 1);
    let gids: Vec<u64> = (0..256)
        .map(|g| 5_000 + g)
        .chain([shared])
        .collect::<Vec<_>>();
    groups.advance(
        2,
        1,
        vec![
            (
                0,
                Some(LocalMap::prepare(Width::W256, &gids[..256]).unwrap()),
            ),
            (
                1,
                Some(LocalMap::prepare(Width::W256, &gids[256..]).unwrap()),
            ),
        ],
    );
    let gf = groups.seal();
    let uf = w.source.frames().seal();
    assert_eq!(w.source.locate(&uf, shared), Ok(7));
    assert_eq!(groups.locate(&gf, &[0, 1], shared), Ok((1, 0)));
    assert_eq!(
        groups.locate(&gf, &[0, 1], object_id(8)),
        Err(LocateRefusal::Unmapped)
    );
    // A frame of one mapping is foreign to the other.
    assert!(groups.map(&uf, 0).is_err());

    let member_of = field_of(&w.view, "Groups").unwrap();
    assert_eq!(
        bind(&w, &WideFieldMask::from_positions(&[member_of]), &uf).err(),
        Some(LaneRefusal::Unbound {
            class: USER,
            field: F_MEMBER_OF
        })
    );
}

/// 6: a selection is a bag; the socket keeps duplicates. A Set reading is the
/// caller's terminal, never the socket's.
#[test]
fn bag_duplicates_survive_the_socket() {
    let mut w = world(512);
    w.source.selections = vec![vec![9, 9, 42]];
    let frame = w.source.frames().seal();
    let b = bind(&w, &WideFieldMask::from_positions(&[F_NAME]), &frame).unwrap();
    let got = project(&w, &b, 0, &["Name"]).unwrap();
    assert_eq!(
        got[0],
        vec![oracle(F_NAME, 9), oracle(F_NAME, 9), oracle(F_NAME, 42)]
    );
    let mut set: Vec<u32> = w.source.selections[0].clone();
    set.sort_unstable();
    set.dedup();
    assert_eq!(set.len(), 2);
}

/// 7: an unbound field, a missing value, an explicit zero and a deleted value
/// are four different answers.
#[test]
fn absent_missing_zero_and_deleted_stay_distinct() {
    let mut w = world(512);
    w.source.selections = vec![vec![0, 3, 4, 6]];
    let frame = w.source.frames().seal();
    let b = bind(&w, &WideFieldMask::from_positions(&[F_MAIL]), &frame).unwrap();
    let got = project(&w, &b, 0, &["PrimaryMail"]).unwrap();
    assert_eq!(
        got[0],
        vec![
            Read::Value(0),
            Read::Missing,
            Read::Deleted,
            Read::Value(18)
        ]
    );
    assert_eq!(
        bind(&w, &WideFieldMask::from_positions(&[F_MEMBER_OF]), &frame).err(),
        Some(LaneRefusal::Unbound {
            class: USER,
            field: F_MEMBER_OF
        })
    );
}

/// 8: foreign source, another reader scope, a field outside the basis and a
/// stale generation refuse before any value is read.
#[test]
fn foreign_scope_basis_and_stale_refuse_before_reading() {
    let mut w = world(512);
    let frame = w.source.frames().seal();
    let name = WideFieldMask::from_positions(&[F_NAME]);

    // Foreign source: another owner's frame store.
    let other = FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 1,
    });
    // A caller that brings its own store AND a frame that store sealed: only
    // the source-owns-its-store check can refuse this.
    let other_frame = other.seal();
    let foreign = w.store.bind_lanes(
        &w.view,
        &other,
        &w.source,
        LaneRequest {
            route: w.route,
            rev: SchemaRev(1),
            fields: &name,
            frame: &other_frame,
            scope: READER,
            selection: w.source.space(),
        },
    );
    assert_eq!(
        foreign.err(),
        Some(LaneRefusal::Frame(FrameRefusal::ForeignFrame))
    );
    // A field the source serves only to another scope.
    assert_eq!(
        bind(&w, &WideFieldMask::from_positions(&[F_SALARY]), &frame).err(),
        Some(LaneRefusal::Scope {
            field: F_SALARY,
            reader: READER,
            lane: HR_ONLY
        })
    );
    // Outside the 64-field basis.
    assert!(matches!(
        bind(&w, &WideFieldMask::from_positions(&[200]), &frame).err(),
        Some(LaneRefusal::FieldNotInBasis { field: 200, .. }) | Some(LaneRefusal::Bind(_))
    ));
    // Stale generation.
    let b = bind(&w, &name, &frame).unwrap();
    w.source.reorder(&(0..512).collect::<Vec<_>>());
    assert!(matches!(
        b.check(&w.store, &w.view, w.source.frames(), &w.source),
        Err(LaneRefusal::Frame(_))
    ));
    assert_eq!(w.source.borrows.get(), 0, "no lane was ever borrowed");
}

/// 9: a 4x larger population (and with it the virtual users x fields space)
/// leaves the work of the same five-identity projection unchanged.
#[test]
fn virtual_space_growth_does_not_grow_projection_work() {
    let mut costs = Vec::new();
    for users in [512usize, 2048] {
        let mut w = world(users);
        w.source.selections = vec![vec![5, 17, 300, 511, 0]];
        let frame = w.source.frames().seal();
        let fields = WideFieldMask::from_positions(&[F_NAME, F_MAIL]);
        let b = bind(&w, &fields, &frame).unwrap();
        let (l0, b0, m0) = (
            w.source.lane_of_calls.get(),
            w.source.borrows.get(),
            w.source.map_lookups(&frame),
        );
        let a0 = allocs();
        let mut reads = 0u64;
        for f in [F_NAME, F_MAIL] {
            let (lane, rows) = b
                .selected(&w.store, &w.view, w.source.frames(), &w.source, f, 0)
                .unwrap();
            for &r in rows {
                let _ = read(lane, r);
                reads += 1;
            }
        }
        costs.push((
            w.source.lane_of_calls.get() - l0,
            w.source.borrows.get() - b0,
            w.source.map_lookups(&frame) - m0,
            allocs() - a0,
            reads,
        ));
    }
    assert_eq!(costs[0], costs[1], "512 vs 2048 users: {costs:?}");
    assert_eq!(costs[0].3, 0, "no allocation");
    assert_eq!(costs[0].4, 10, "exactly the requested cells are read");
}

/// 11 (D-MAPPING-POLYFILL-1 consumer contract): a projection of three
/// properties is ONE check, not one per property. It equals the oracle in
/// all three value states, the read path allocates nothing, and a projection
/// over a stale selection is refused before any lane is borrowed.
#[test]
fn a_projection_checks_once_not_once_per_property() {
    let mut w = world(512);
    // Users 3 and 4 hold a missing and a deleted mail.
    let picked = [3usize, 4, 17, 300, 3];
    w.source.selections = vec![picked.iter().map(|&u| u as u32).collect()];
    let frame = w.source.frames().seal();
    let fields = WideFieldMask::from_positions(&[F_OBJECT_ID, F_NAME, F_MAIL]);
    let b = bind(&w, &fields, &frame).unwrap();

    // Checks: `check` asks the source for each bound lane once.
    w.source.lane_of_calls.set(0);
    let got = project(&w, &b, 0, &["Identity", "Name", "PrimaryMail"]).unwrap();
    assert_eq!(
        w.source.lane_of_calls.get(),
        3,
        "one check over three bound lanes, not three checks"
    );
    for (i, &u) in picked.iter().enumerate() {
        assert_eq!(got[0][i], oracle(F_OBJECT_ID, u), "identity, user {u}");
        assert_eq!(got[1][i], oracle(F_NAME, u), "name, user {u}");
        assert_eq!(got[2][i], oracle(F_MAIL, u), "mail, user {u}");
    }
    assert_eq!(got[2][0], Read::Missing);
    assert_eq!(got[2][1], Read::Deleted);

    // The read path, without the terminal's output vectors: no allocation.
    let a0 = allocs();
    let mut agree = 0u32;
    let (lanes, rows) = b
        .borrow_all_selected(&w.store, &w.view, w.source.frames(), &w.source, 0)
        .unwrap();
    for item in lanes {
        let (field, lane) = item.unwrap();
        for &r in rows {
            if read(lane, r) == oracle(field, w.source.row_user[r as usize]) {
                agree += 1;
            }
        }
    }
    assert_eq!(allocs() - a0, 0, "the read path allocates nothing");
    assert_eq!(agree, 15, "5 positions x 3 fields agree with the oracle");

    // A stale selection: the rows were reordered under a new frame.
    w.source.reorder(&(0..512).rev().collect::<Vec<_>>());
    let borrows = w.source.borrows.get();
    assert!(matches!(
        project(&w, &b, 0, &["Name"]),
        Err(LaneRefusal::Frame(_))
    ));
    assert_eq!(w.source.borrows.get(), borrows, "refused before borrowing");
}
