//! D-LANESOCK-1: the property-to-lane socket. Every refusal fires on a
//! realistic input, and a binding that passes borrows the source's own lane.

use std::cell::Cell;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{Frame, FrameRefusal, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneRefusal, LaneRequest, LaneSource,
    PresenceSource, Refusal, RouteAt, SchemaRev,
};

const HYBRID: u32 = 0x0C1A_0001;
const HYBRID_V2: u32 = 0x0C1A_0002;
const VIEW_CLASS: ClassId = 0x0C1A;
const POSITION: u16 = 5;

/// Fields: 0 department (AD), 1 cost centre (AD), 2 license count (Entra).
struct View(Vec<FieldRef>);

impl View {
    fn three() -> View {
        View(vec![
            FieldRef::new("ad:department", "Department"),
            FieldRef::new("ad:costCenter", "Cost centre"),
            FieldRef::new("entra:licenseCount", "Licenses"),
        ])
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

const AD: CoordSpace = CoordSpace {
    population: 1,
    coordinates: 1,
};
const ENTRA: CoordSpace = CoordSpace {
    population: 2,
    coordinates: 1,
};
const READER: Scope = Scope { id: 7 };

/// A source owner: its lanes, and which lane holds which field.
struct Source {
    lanes: Vec<Vec<i32>>,
    map: Vec<(u8, LaneAddr)>,
    /// The store this source's data lives with (it never takes one from a
    /// caller; see `LaneSource::frames`).
    frames: FrameStore,
    /// Presence plane per lane (by lane number), or none.
    presence: Vec<Option<Vec<u64>>>,
    /// `lane_of` calls, `borrow` calls and `borrow_presence` calls, so far.
    lookups: Cell<usize>,
    borrows: Cell<usize>,
    presence_borrows: Cell<usize>,
}

impl Source {
    fn hybrid() -> Source {
        let at = |space, lane| LaneAddr {
            space,
            lane,
            scope: READER,
        };
        Source {
            lanes: vec![vec![6, 7, 6], vec![100, 200, 300], vec![25, 40]],
            map: vec![(0, at(AD, 0)), (1, at(AD, 1)), (2, at(ENTRA, 2))],
            frames: FrameStore::new(FrameRevisions {
                source: 1,
                relation_map: 1,
                coordinates: 1,
            }),
            // Distinct planes, so a plane handed out for the wrong lane is
            // visible: lane 0 has row 1 absent, lane 1 every row present.
            presence: vec![Some(vec![0b101]), Some(vec![0b111]), Some(vec![0b11])],
            lookups: Cell::new(0),
            borrows: Cell::new(0),
            presence_borrows: Cell::new(0),
        }
    }
}

impl PresenceSource for Source {
    fn borrow_presence(&self, addr: &LaneAddr) -> Option<&[u64]> {
        self.presence_borrows.set(self.presence_borrows.get() + 1);
        self.presence.get(usize::from(addr.lane))?.as_deref()
    }
}

impl LaneSource for Source {
    type Lane<'a> = &'a [i32];
    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        field: u8,
        _basis: &FieldRef,
        _frame: &Frame,
    ) -> Option<LaneAddr> {
        self.lookups.set(self.lookups.get() + 1);
        if class != HYBRID || rev != SchemaRev(1) {
            return None;
        }
        self.map.iter().find(|(f, _)| *f == field).map(|(_, a)| *a)
    }
    fn borrow(&self, addr: &LaneAddr) -> Option<&[i32]> {
        self.borrows.set(self.borrows.get() + 1);
        self.lanes.get(usize::from(addr.lane)).map(Vec::as_slice)
    }
    fn frames(&self) -> &FrameStore {
        &self.frames
    }
}

fn schema(fields: usize) -> ClassSchema {
    ClassSchema {
        schema_rev: SchemaRev(1),
        field_count: fields,
        extent: Extent::DR64_CUBE,
    }
}

struct World {
    store: BindStore,
    view: View,
    frame: Frame,
    source: Source,
    route: RouteAt,
}

fn world() -> World {
    let mut store = BindStore::new();
    store
        .register(&View::three(), HYBRID, schema(3), Vec::new())
        .unwrap();
    let version = store.route(POSITION, HYBRID).unwrap();
    let source = Source::hybrid();
    let frame = source.frames.seal();
    World {
        store,
        view: View::three(),
        frame,
        source,
        route: RouteAt {
            position: POSITION,
            version,
        },
    }
}

fn request<'a>(w: &'a World, fields: &'a WideFieldMask) -> LaneRequest<'a> {
    LaneRequest {
        route: w.route,
        rev: SchemaRev(1),
        fields,
        frame: &w.frame,
        scope: READER,
        selection: AD,
    }
}

fn bind(w: &World, fields: &WideFieldMask) -> Result<lance_graph_dt64::LaneBinding, LaneRefusal> {
    w.store
        .bind_lanes(&w.view, &w.source.frames, &w.source, request(w, fields))
}

#[test]
fn two_fields_of_one_basis_bind_and_borrow_the_source_lanes() {
    let w = world();
    let both = WideFieldMask::from_positions(&[0, 1]);
    let b = bind(&w, &both).unwrap();
    assert_eq!(b.class(), HYBRID);
    assert!(b.fields().has(0) && b.fields().has(1));
    let dept = b
        .lane(&w.store, &w.view, &w.source.frames, &w.source, 0)
        .unwrap();
    let cost = b
        .lane(&w.store, &w.view, &w.source.frames, &w.source, 1)
        .unwrap();
    // Borrowed, not copied: the very slices the source holds.
    assert!(std::ptr::eq(dept, w.source.lanes[0].as_slice()));
    assert!(std::ptr::eq(cost, w.source.lanes[1].as_slice()));
    assert_eq!(
        b.lane(&w.store, &w.view, &w.source.frames, &w.source, 2),
        Err(LaneRefusal::NotBound { field: 2 })
    );
}

#[test]
fn a_lane_in_another_row_order_is_refused() {
    let w = world();
    let mask = WideFieldMask::from_positions(&[0, 2]);
    assert_eq!(
        bind(&w, &mask).err(),
        Some(LaneRefusal::CoordinateSpace {
            field: 2,
            selection: AD,
            lane: ENTRA,
        })
    );
    // The same field binds under its own row order.
    let only = WideFieldMask::from_positions(&[2]);
    let mut req = request(&w, &only);
    req.selection = ENTRA;
    assert!(w
        .store
        .bind_lanes(&w.view, &w.source.frames, &w.source, req)
        .is_ok());
}

#[test]
fn wrong_schema_revision_and_changed_basis_are_refused() {
    let w = world();
    let mask = WideFieldMask::from_positions(&[0]);
    let mut req = request(&w, &mask);
    req.rev = SchemaRev(2);
    assert!(matches!(
        w.store
            .bind_lanes(&w.view, &w.source.frames, &w.source, req),
        Err(LaneRefusal::Bind(Refusal::SchemaRevisionMismatch { .. }))
    ));

    let b = bind(&w, &mask).unwrap();
    let narrower = View(View::three().0[..2].to_vec());
    assert!(matches!(
        b.check(&w.store, &narrower, &w.source.frames, &w.source),
        Err(LaneRefusal::Bind(Refusal::FieldBasisChanged { .. }))
    ));
}

#[test]
fn a_field_outside_the_basis_or_without_a_lane_is_refused() {
    let w = world();
    let outside = WideFieldMask::from_positions(&[0, 9]);
    assert_eq!(
        bind(&w, &outside).err(),
        Some(LaneRefusal::FieldNotInBasis {
            class: HYBRID,
            field: 9,
        })
    );
    let mut w = w;
    w.source.map.retain(|(f, _)| *f != 1);
    let both = WideFieldMask::from_positions(&[0, 1]);
    assert_eq!(
        bind(&w, &both).err(),
        Some(LaneRefusal::Unbound {
            class: HYBRID,
            field: 1,
        })
    );
}

#[test]
fn a_lane_for_another_scope_is_refused() {
    let mut w = world();
    w.source.map[1].1.scope = Scope { id: 8 };
    let both = WideFieldMask::from_positions(&[0, 1]);
    assert_eq!(
        bind(&w, &both).err(),
        Some(LaneRefusal::Scope {
            field: 1,
            reader: READER,
            lane: Scope { id: 8 },
        })
    );
}

#[test]
fn a_changed_mapping_or_a_vanished_lane_is_refused_on_use() {
    let mut w = world();
    let both = WideFieldMask::from_positions(&[0, 1]);
    let b = bind(&w, &both).unwrap();
    w.source.map[0].1.lane = 1;
    assert_eq!(
        b.check(&w.store, &w.view, &w.source.frames, &w.source),
        Err(LaneRefusal::MappingChanged { field: 0 })
    );

    let mut w = world();
    let b = bind(&w, &both).unwrap();
    w.source.lanes.truncate(1);
    assert_eq!(
        b.lane(&w.store, &w.view, &w.source.frames, &w.source, 1),
        Err(LaneRefusal::Vanished { field: 1 })
    );
}

#[test]
fn stale_and_foreign_frames_are_refused() {
    let w = world();
    let mask = WideFieldMask::from_positions(&[0]);
    let twin = FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 1,
    });
    assert_eq!(
        w.store
            .bind_lanes(&w.view, &twin, &w.source, request(&w, &mask))
            .err(),
        Some(LaneRefusal::Frame(FrameRefusal::ForeignFrame))
    );

    let mut w = w;
    let b = bind(&w, &mask).unwrap();
    w.source.frames.advance(FrameRevisions {
        source: 1,
        relation_map: 2,
        coordinates: 1,
    });
    assert_eq!(
        b.check(&w.store, &w.view, &w.source.frames, &w.source),
        Err(LaneRefusal::Frame(FrameRefusal::MapChanged))
    );
}

/// A binding made over one source cannot be checked or read against another
/// source whose lane numbers match (D-LANESOCK, PR B). Before the gate the
/// substitute's values came back; now the pairing is refused before any lane
/// is looked up or borrowed.
#[test]
fn a_substituted_source_is_refused_not_read() {
    let w = world();
    let mask = WideFieldMask::from_positions(&[1]);
    let b = bind(&w, &mask).unwrap();
    assert_eq!(
        b.lane(&w.store, &w.view, &w.source.frames, &w.source, 1),
        Ok(&[100, 200, 300][..]),
        "the bound source reads its own lane"
    );

    // Same map, same lane numbers, different data and its own store.
    let mut other = Source::hybrid();
    other.lanes[1] = vec![30000, 31000, 32000];

    // Caller pairs the substitute with the bound source's store.
    assert_eq!(
        b.lane(&w.store, &w.view, &w.source.frames, &other, 1),
        Err(LaneRefusal::Frame(FrameRefusal::ForeignFrame))
    );
    // Caller pairs the substitute with its own store: the frame is foreign.
    assert_eq!(
        b.lane(&w.store, &w.view, &other.frames, &other, 1),
        Err(LaneRefusal::Frame(FrameRefusal::ForeignFrame))
    );
    // And a fresh bind cannot pair the bound source with another store.
    assert_eq!(
        w.store
            .bind_lanes(&w.view, &other.frames, &w.source, request(&w, &mask))
            .err(),
        Some(LaneRefusal::Frame(FrameRefusal::ForeignFrame))
    );
}

/// The route and the frame are checked as a pair: a re-route alone, with the
/// frame untouched, refuses.
#[test]
fn a_re_route_refuses_even_under_a_current_frame() {
    let mut w = world();
    let mask = WideFieldMask::from_positions(&[0]);
    let b = bind(&w, &mask).unwrap();
    w.store
        .register_derived(&w.view, HYBRID, HYBRID_V2, schema(3), Vec::new())
        .unwrap();
    w.store.route(POSITION, HYBRID_V2).unwrap();
    assert!(
        w.source.frames.check(b.frame()).is_ok(),
        "the frame did not move"
    );
    assert_eq!(
        b.check(&w.store, &w.view, &w.source.frames, &w.source),
        Err(LaneRefusal::RouteChanged {
            position: POSITION,
            bound: HYBRID,
            now: Some(HYBRID_V2),
        })
    );
    // And a bind at the old route version refuses too.
    assert!(matches!(
        bind(&w, &mask),
        Err(LaneRefusal::RouteChanged { .. })
    ));
}

/// Decision J on the hot path: a binding checked against ANOTHER store, which
/// binds the class over another basis, is refused by the recorded witness
/// (one `u128` compare, no hashing).
#[test]
fn a_binding_checked_against_another_store_with_another_basis_refuses() {
    let w = world();
    let mask = WideFieldMask::from_positions(&[0]);
    let b = bind(&w, &mask).unwrap();
    let mut swapped = View::three();
    swapped.0.swap(0, 1);
    let mut other = BindStore::new();
    other
        .register(&swapped, HYBRID, schema(3), Vec::new())
        .unwrap();
    other.route(POSITION, HYBRID).unwrap();
    assert_eq!(
        b.check(&other, &w.view, &w.source.frames, &w.source),
        Err(LaneRefusal::StoreBasisMismatch { class: HYBRID })
    );
    assert!(b
        .check(&w.store, &w.view, &w.source.frames, &w.source)
        .is_ok());
}

/// Closes the L5a FINDING
/// `finding_the_hot_check_does_not_see_a_view_reorder_under_a_positional_source`
/// (plan gap G7): a same-count reorder of the view after the bind, at a bound
/// position, is refused by the hot check even when the source maps by
/// position.
#[test]
fn a_view_reorder_at_a_bound_position_is_refused_under_a_positional_source() {
    let w = world();
    let mask = WideFieldMask::from_positions(&[0]);
    let b = bind(&w, &mask).unwrap();
    let mut swapped = View::three();
    swapped.0.swap(0, 1);
    assert!(w.store.resolve(&swapped, HYBRID, SchemaRev(1)).is_err());
    let refusal = LaneRefusal::PredicateChanged {
        class: HYBRID,
        field: 0,
    };
    assert_eq!(
        b.check(&w.store, &swapped, &w.source.frames, &w.source),
        Err(refusal.clone())
    );
    w.source.borrows.set(0);
    assert_eq!(
        b.lane(&w.store, &swapped, &w.source.frames, &w.source, 0),
        Err(refusal)
    );
    assert_eq!(w.source.borrows.get(), 0, "nothing is borrowed");
}

/// An unchanged view admits every bound lane, and the compare is by content,
/// not by address: a separately built view with equal content passes too.
#[test]
fn an_unchanged_view_admits_every_bound_lane() {
    let w = world();
    let b = bind(&w, &WideFieldMask::from_positions(&[0, 1])).unwrap();
    assert_eq!(
        b.check(&w.store, &w.view, &w.source.frames, &w.source),
        Ok(())
    );
    let rebuilt = View::three();
    assert_eq!(
        b.check(&w.store, &rebuilt, &w.source.frames, &w.source),
        Ok(())
    );
    let n = b
        .borrow_all(&w.store, &rebuilt, &w.source.frames, &w.source)
        .unwrap()
        .count();
    assert_eq!(n, 2);
}

/// A label is not identity (Decision J item 4): renaming a bound field's label
/// leaves the binding admitted.
#[test]
fn a_label_only_rename_at_a_bound_position_admits() {
    let w = world();
    let b = bind(&w, &WideFieldMask::from_positions(&[0])).unwrap();
    let mut renamed = View::three();
    renamed.0[0].label = "Abteilung".to_string();
    assert!(w.store.resolve(&renamed, HYBRID, SchemaRev(1)).is_ok());
    assert_eq!(
        b.check(&w.store, &renamed, &w.source.frames, &w.source),
        Ok(())
    );
}

/// The accepted residual: a reorder that leaves every bound position naming
/// the same predicate is admitted hot. The bound position's meaning is
/// unchanged, so the binding's mapping and interpretation hold; the
/// whole-basis witness is checked at bind and at `resolve` (which refuses
/// this view), never on the hot path.
#[test]
fn a_reorder_outside_the_bound_positions_admits_hot_and_is_refused_cold() {
    let w = world();
    let b = bind(&w, &WideFieldMask::from_positions(&[0])).unwrap();
    let mut swapped = View::three();
    swapped.0.swap(1, 2);
    assert!(w.store.resolve(&swapped, HYBRID, SchemaRev(1)).is_err());
    assert_eq!(
        b.check(&w.store, &swapped, &w.source.frames, &w.source),
        Ok(())
    );
}

/// The refusal names the first bound position whose predicate changed.
#[test]
fn the_refusal_names_the_first_changed_bound_position() {
    let w = world();
    let b = bind(&w, &WideFieldMask::from_positions(&[0, 1])).unwrap();

    let mut replaced = View::three();
    replaced.0[1] = FieldRef::new("ad:manager", "Manager");
    assert_eq!(
        b.check(&w.store, &replaced, &w.source.frames, &w.source),
        Err(LaneRefusal::PredicateChanged {
            class: HYBRID,
            field: 1,
        })
    );

    let mut swapped = View::three();
    swapped.0.swap(0, 1);
    assert_eq!(
        b.check(&w.store, &swapped, &w.source.frames, &w.source),
        Err(LaneRefusal::PredicateChanged {
            class: HYBRID,
            field: 0,
        })
    );
}

/// The very view object the binding was made through, mutated in place, is
/// refused: no address-identity shortcut.
#[test]
fn a_view_mutated_in_place_after_the_bind_is_refused() {
    let mut w = world();
    let b = bind(&w, &WideFieldMask::from_positions(&[0])).unwrap();
    w.view.0.swap(0, 1);
    assert_eq!(
        b.check(&w.store, &w.view, &w.source.frames, &w.source),
        Err(LaneRefusal::PredicateChanged {
            class: HYBRID,
            field: 0,
        })
    );
}

/// `borrow_all` hands out the source's own lanes, in field order, the same
/// ones `lane` hands out one at a time.
#[test]
fn borrow_all_yields_every_bound_lane_in_place_in_field_order() {
    let w = world();
    let b = bind(&w, &WideFieldMask::from_positions(&[1, 0])).unwrap();
    let all = b
        .borrow_all(&w.store, &w.view, &w.source.frames, &w.source)
        .unwrap();
    assert_eq!(all.len(), 2);
    let got: Vec<(u8, &[i32])> = all.map(Result::unwrap).collect();
    assert_eq!(got.iter().map(|(f, _)| *f).collect::<Vec<_>>(), [0, 1]);
    for (field, lane) in got {
        assert!(std::ptr::eq(
            lane,
            w.source.lanes[usize::from(field)].as_slice()
        ));
        assert_eq!(
            b.lane(&w.store, &w.view, &w.source.frames, &w.source, field),
            Ok(lane)
        );
    }
}

/// One check for all lanes: N source lookups where `lane` per field does N².
#[test]
fn borrow_all_checks_once_where_lane_per_field_checks_each_time() {
    let w = world();
    let b = bind(&w, &WideFieldMask::from_positions(&[0, 1])).unwrap();

    w.source.lookups.set(0);
    let n = b
        .borrow_all(&w.store, &w.view, &w.source.frames, &w.source)
        .unwrap()
        .count();
    assert_eq!((n, w.source.lookups.get()), (2, 2));

    w.source.lookups.set(0);
    for f in [0, 1] {
        b.lane(&w.store, &w.view, &w.source.frames, &w.source, f)
            .unwrap();
    }
    assert_eq!(
        w.source.lookups.get(),
        4,
        "lane per field re-checks every lane"
    );
}

/// A refused check returns before any lane is borrowed.
#[test]
fn a_refused_borrow_all_borrows_nothing() {
    let mut w = world();
    let b = bind(&w, &WideFieldMask::from_positions(&[0, 1])).unwrap();
    w.source.frames.advance(FrameRevisions {
        source: 2,
        relation_map: 1,
        coordinates: 1,
    });
    w.source.borrows.set(0);
    assert_eq!(
        b.borrow_all(&w.store, &w.view, &w.source.frames, &w.source)
            .err(),
        Some(LaneRefusal::Frame(FrameRefusal::SourceChanged))
    );
    assert_eq!(w.source.borrows.get(), 0);
}

/// A lane the source no longer holds is reported at its position; the
/// other lanes are still handed out.
#[test]
fn a_vanished_lane_is_reported_in_place() {
    let mut w = world();
    let b = bind(&w, &WideFieldMask::from_positions(&[0, 1])).unwrap();
    w.source.lanes.truncate(1);
    let got: Vec<_> = b
        .borrow_all(&w.store, &w.view, &w.source.frames, &w.source)
        .unwrap()
        .collect();
    assert_eq!(
        got,
        vec![
            Ok((0, w.source.lanes[0].as_slice())),
            Err(LaneRefusal::Vanished { field: 1 })
        ]
    );
}

/// EXO-294: each bound lane comes with the presence plane the source holds
/// for THAT lane, found by the lane address the check admitted, not by the
/// field number. Field 1 is mapped to lane 0 here and field 0 to lane 1, so
/// a plane looked up by field would be the other lane's.
#[test]
fn borrow_all_present_hands_each_lane_its_own_presence_plane() {
    let mut w = world();
    let at = |lane| LaneAddr {
        space: AD,
        lane,
        scope: READER,
    };
    w.source.map = vec![(0, at(1)), (1, at(0)), (2, at(2))];
    let b = bind(&w, &WideFieldMask::from_positions(&[0, 1])).unwrap();
    let got: Vec<_> = b
        .borrow_all_present(&w.store, &w.view, &w.source.frames, &w.source)
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(got.len(), 2);
    for (field, lane, plane) in got {
        let lane_no = if field == 0 { 1 } else { 0 };
        assert!(std::ptr::eq(lane, w.source.lanes[lane_no].as_slice()));
        assert!(std::ptr::eq(
            plane,
            w.source.presence[lane_no].as_deref().unwrap()
        ));
    }
    // The planes really differ, so the pointer checks above discriminate.
    assert_ne!(w.source.presence[0], w.source.presence[1]);
}

/// A lane without a presence plane is refused at its position and never read
/// as "all present"; the other lanes are still handed out with theirs.
#[test]
fn a_lane_without_presence_is_refused_in_place() {
    let mut w = world();
    w.source.presence[1] = None;
    let b = bind(&w, &WideFieldMask::from_positions(&[0, 1])).unwrap();
    let got: Vec<_> = b
        .borrow_all_present(&w.store, &w.view, &w.source.frames, &w.source)
        .unwrap()
        .collect();
    assert_eq!(
        got,
        vec![
            Ok((
                0,
                w.source.lanes[0].as_slice(),
                w.source.presence[0].as_deref().unwrap()
            )),
            Err(LaneRefusal::PresenceUnbound { field: 1 })
        ]
    );
}

/// A refused check returns before any lane or presence plane is borrowed.
#[test]
fn a_refused_borrow_all_present_borrows_nothing() {
    let mut w = world();
    let b = bind(&w, &WideFieldMask::from_positions(&[0, 1])).unwrap();
    w.source.frames.advance(FrameRevisions {
        source: 2,
        relation_map: 1,
        coordinates: 1,
    });
    w.source.borrows.set(0);
    w.source.presence_borrows.set(0);
    assert_eq!(
        b.borrow_all_present(&w.store, &w.view, &w.source.frames, &w.source)
            .err(),
        Some(LaneRefusal::Frame(FrameRefusal::SourceChanged))
    );
    assert_eq!(
        (w.source.borrows.get(), w.source.presence_borrows.get()),
        (0, 0)
    );
}

// ---------------------------------------------------------------------------
// The relation socket (D-XLOG-W7-2 step 2)
// ---------------------------------------------------------------------------

mod relation_socket {
    use lance_graph_dt64::dr::{Axis, MapTable, PairDir, PairPlane, PairRefusal, Width};
    use lance_graph_dt64::{relation_rows, RelNo, RelationRefusal, RelationSource};

    /// An owner holding relation 4 only.
    struct Owner {
        plane: PairPlane,
    }

    impl RelationSource for Owner {
        fn borrow_relation(&self, rel: RelNo) -> Option<&PairPlane> {
            (rel == 4).then_some(&self.plane)
        }
    }

    fn owner(table: &MapTable) -> Owner {
        let f = table.seal();
        let mut plane = PairPlane::new(&f, Axis::new(Width::W64, 2), Axis::new(Width::W256, 1));
        for (x, y) in [(0, 0), (0, 255), (5, 7), (127, 128), (64, 3)] {
            plane.insert(x, y).unwrap();
        }
        Owner { plane }
    }

    /// The lent rows are the owner's own words (same address, no copy), and
    /// each row equals the plane's row in both directions.
    #[test]
    fn the_socket_lends_the_owners_plane_without_copying() {
        let t = MapTable::new(1, 1);
        let o = owner(&t);
        let f = t.seal();
        for dir in [PairDir::Forward, PairDir::Backward] {
            let lent = relation_rows(&o, 4, &f, dir).unwrap();
            let own = o.plane.rows(&f, dir).unwrap();
            assert_eq!(lent.as_words().as_ptr(), own.as_ptr(), "{dir:?}: copied");
            assert_eq!(lent.len(), o.plane.rows_in(dir));
            assert_eq!(lent.words_per_row(), o.plane.words_per_row(dir));
            for at in 0..lent.len() {
                assert_eq!(lent.row(at), Some(o.plane.row(&f, dir, at).unwrap()));
            }
            assert_eq!(lent.row(lent.len()), None);
        }
        let fwd = relation_rows(&o, 4, &f, PairDir::Forward).unwrap();
        assert_eq!(fwd.row(0).unwrap()[3] >> 63, 1, "(0, 255) present");
    }

    /// A relation the owner does not hold is refused, never read as empty;
    /// a frame other than the plane's issue is refused through the socket.
    #[test]
    fn missing_relations_and_foreign_frames_are_refused() {
        let mut t = MapTable::new(1, 1);
        let o = owner(&t);
        let f = t.seal();
        assert_eq!(
            relation_rows(&o, 3, &f, PairDir::Forward).unwrap_err(),
            RelationRefusal::NoRelation(3)
        );
        let other = MapTable::new(1, 1).seal();
        assert_eq!(
            relation_rows(&o, 4, &other, PairDir::Forward).unwrap_err(),
            RelationRefusal::Pair(PairRefusal::ForeignFrame)
        );
        t.advance(1, 2, Vec::new());
        assert_eq!(
            relation_rows(&o, 4, &t.seal(), PairDir::Backward).unwrap_err(),
            RelationRefusal::Pair(PairRefusal::ForeignFrame)
        );
    }
}

mod relation_reach {
    use lance_graph_dt64::dr::{Axis, Frame, MapTable, PairDir, PairPlane, PairRefusal, Width};
    use lance_graph_dt64::{
        reach_path, relation_reach, ReachBuf, ReachPart, ReachRefusal, RelNo, RelationRefusal,
        RelationSource, UNREACHED,
    };

    const P: usize = 512;
    const W: usize = P / 64;

    struct Owner {
        plane: PairPlane,
    }

    impl RelationSource for Owner {
        fn borrow_relation(&self, rel: RelNo) -> Option<&PairPlane> {
            (rel == 0).then_some(&self.plane)
        }
    }

    fn splitmix(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A chain 0→1→…→9, a branch 3→400→401, a cycle 20→21→22→20, an island
    /// 300→301, and `extra` random edges among positions 100..200.
    fn edges(extra: usize) -> Vec<(usize, usize)> {
        let mut e: Vec<(usize, usize)> = (0..9).map(|i| (i, i + 1)).collect();
        e.extend([
            (3, 400),
            (400, 401),
            (20, 21),
            (21, 22),
            (22, 20),
            (300, 301),
        ]);
        let mut s = 7u64;
        for _ in 0..extra {
            let a = 100 + (splitmix(&mut s) % 100) as usize;
            let b = 100 + (splitmix(&mut s) % 100) as usize;
            e.push((a, b));
        }
        e
    }

    fn owner(f: &Frame, edges: &[(usize, usize)]) -> Owner {
        let axis = Axis::new(Width::W256, 2);
        let mut plane = PairPlane::new(f, axis, axis);
        for &(a, b) in edges {
            plane.insert(a, b).unwrap();
        }
        Owner { plane }
    }

    fn mask(bits: &[usize]) -> Vec<u64> {
        let mut m = vec![0u64; W];
        for &b in bits {
            m[b / 64] |= 1 << (b % 64);
        }
        m
    }

    /// Breadth first over the edge list: per position its distance from the
    /// seeds, every step landing in `target`. Shares nothing with the plane.
    fn bfs(
        edges: &[(usize, usize)],
        seeds: &[usize],
        target: Option<&[u64]>,
        fwd: bool,
    ) -> Vec<u16> {
        let mut dist = vec![UNREACHED; P];
        let mut queue = std::collections::VecDeque::new();
        for &s in seeds {
            dist[s] = 0;
            queue.push_back(s);
        }
        while let Some(x) = queue.pop_front() {
            for &(a, b) in edges {
                let (from, to) = if fwd { (a, b) } else { (b, a) };
                let ok = target.is_none_or(|t| t[to / 64] >> (to % 64) & 1 == 1);
                if from == x && ok && dist[to] == UNREACHED {
                    dist[to] = dist[x] + 1;
                    queue.push_back(to);
                }
            }
        }
        dist
    }

    struct Run {
        reached: Vec<u64>,
        hop: Vec<u16>,
        scratch: Vec<u64>,
    }

    fn run() -> Run {
        Run {
            reached: vec![u64::MAX; W],
            hop: vec![7; P],
            scratch: vec![u64::MAX; 2 * W],
        }
    }

    impl Run {
        fn buf(&mut self) -> ReachBuf<'_> {
            ReachBuf {
                reached: &mut self.reached,
                hop: &mut self.hop,
                scratch: &mut self.scratch,
            }
        }
    }

    /// The reached mask and the hop of every position equal the BFS oracle,
    /// forward and backward, with and without a target, from buffers that
    /// start dirty. The answer is a strict, non-empty subset of the axis.
    #[test]
    fn reach_and_hops_equal_bfs() {
        let t = MapTable::new(1, 1);
        let f = t.seal();
        let e = edges(150);
        let o = owner(&f, &e);
        let no_401 = {
            let mut m = vec![u64::MAX; W];
            m[401 / 64] &= !(1 << (401 % 64));
            m
        };
        type Case<'a> = (&'a [usize], Option<&'a [u64]>, PairDir);
        let cases: [Case<'_>; 5] = [
            (&[0], None, PairDir::Forward),
            (&[0, 20, 150], None, PairDir::Forward),
            (&[0], Some(&no_401), PairDir::Forward),
            (&[9, 401], None, PairDir::Backward),
            (&[300], None, PairDir::Forward),
        ];
        for (seeds, target, dir) in cases {
            let mut r = run();
            let stats = relation_reach(&o, 0, &f, dir, &mask(seeds), target, &mut r.buf()).unwrap();
            let want = bfs(&e, seeds, target, dir == PairDir::Forward);
            assert_eq!(r.hop, want, "{seeds:?} {dir:?}");
            let hit: Vec<usize> = (0..P).filter(|&p| want[p] != UNREACHED).collect();
            assert_eq!(r.reached, mask(&hit));
            assert_eq!(stats.reached as usize, hit.len());
            assert!(!hit.is_empty() && hit.len() * 2 < P);
        }
        // The chain: nine hops to position 9, and the branch is two past 3.
        let mut r = run();
        relation_reach(&o, 0, &f, PairDir::Forward, &mask(&[0]), None, &mut r.buf()).unwrap();
        assert_eq!((r.hop[9], r.hop[400], r.hop[401]), (9, 4, 5));
        assert_eq!(r.hop[300], UNREACHED);
    }

    /// A path read back from the witness starts at a seed, ends at the asked
    /// position, has `hop + 1` entries, and every step is an edge.
    #[test]
    fn a_path_is_made_of_held_edges() {
        let t = MapTable::new(1, 1);
        let f = t.seal();
        let e = edges(150);
        let o = owner(&f, &e);
        for dir in [PairDir::Forward, PairDir::Backward] {
            let seeds: &[usize] = if dir == PairDir::Forward {
                &[0, 150]
            } else {
                &[9, 150]
            };
            let mut r = run();
            relation_reach(&o, 0, &f, dir, &mask(seeds), None, &mut r.buf()).unwrap();
            let mut walked = 0;
            for to in (0..P).filter(|&p| r.hop[p] != UNREACHED) {
                let mut path = [u32::MAX; 64];
                let n = reach_path(&o, 0, &f, dir, &r.hop, to, &mut path).unwrap();
                assert_eq!(n, r.hop[to] as usize + 1);
                assert!(seeds.contains(&(path[0] as usize)));
                assert_eq!(path[n - 1] as usize, to);
                for s in path[..n].windows(2) {
                    let (a, b) = (s[0] as usize, s[1] as usize);
                    let pair = if dir == PairDir::Forward {
                        (a, b)
                    } else {
                        (b, a)
                    };
                    assert!(e.contains(&pair), "{pair:?} is not an edge");
                    walked += 1;
                }
            }
            assert!(walked > 20);
        }
    }

    /// A witness is refused once the relation no longer holds its hops, and
    /// an unreached or out-of-axis position is refused by name.
    #[test]
    fn a_stale_witness_is_refused() {
        let t = MapTable::new(1, 1);
        let f = t.seal();
        let e = edges(0);
        let mut o = owner(&f, &e);
        let mut r = run();
        relation_reach(&o, 0, &f, PairDir::Forward, &mask(&[0]), None, &mut r.buf()).unwrap();
        let mut path = [0u32; 16];
        assert_eq!(
            reach_path(&o, 0, &f, PairDir::Forward, &r.hop, 300, &mut path),
            Err(ReachRefusal::NotReached(300))
        );
        assert_eq!(
            reach_path(&o, 0, &f, PairDir::Forward, &r.hop, P, &mut path),
            Err(ReachRefusal::OutOfAxis {
                at: P,
                positions: P
            })
        );
        assert_eq!(
            reach_path(&o, 0, &f, PairDir::Forward, &r.hop, 9, &mut path[..9]),
            Err(ReachRefusal::Length {
                part: ReachPart::Path,
                want: 10,
                got: 9
            })
        );
        assert_eq!(
            reach_path(&o, 0, &f, PairDir::Forward, &r.hop, 9, &mut path),
            Ok(10)
        );
        o.plane.remove(4, 5).unwrap();
        assert_eq!(
            reach_path(&o, 0, &f, PairDir::Forward, &r.hop, 9, &mut path),
            Err(ReachRefusal::WitnessMismatch(5))
        );
    }

    /// A frame the plane was not built under, a missing relation, a plane
    /// whose axes differ and a buffer of the wrong length are each refused,
    /// and a refusal writes nothing.
    #[test]
    fn refusals_write_nothing() {
        let mut t = MapTable::new(1, 1);
        let f = t.seal();
        let o = owner(&f, &edges(0));
        let seed = mask(&[0]);
        let mut r = run();
        assert_eq!(
            relation_reach(&o, 1, &f, PairDir::Forward, &seed, None, &mut r.buf()),
            Err(ReachRefusal::Relation(RelationRefusal::NoRelation(1)))
        );
        t.advance(1, 2, Vec::new());
        let moved = t.seal();
        assert_eq!(
            relation_reach(&o, 0, &moved, PairDir::Forward, &seed, None, &mut r.buf()),
            Err(ReachRefusal::Relation(RelationRefusal::Pair(
                PairRefusal::ForeignFrame
            )))
        );
        assert_eq!(
            reach_path(&o, 0, &moved, PairDir::Forward, &[0; P], 0, &mut [0; 4]),
            Err(ReachRefusal::Relation(RelationRefusal::Pair(
                PairRefusal::ForeignFrame
            )))
        );
        assert_eq!(
            relation_reach(
                &o,
                0,
                &f,
                PairDir::Forward,
                &seed[..W - 1],
                None,
                &mut r.buf()
            ),
            Err(ReachRefusal::Length {
                part: ReachPart::Seed,
                want: W,
                got: W - 1
            })
        );
        assert_eq!(
            relation_reach(
                &o,
                0,
                &f,
                PairDir::Forward,
                &seed,
                Some(&seed[..1]),
                &mut r.buf()
            ),
            Err(ReachRefusal::Length {
                part: ReachPart::Target,
                want: W,
                got: 1
            })
        );
        let mut short = run();
        short.hop.pop();
        assert_eq!(
            relation_reach(&o, 0, &f, PairDir::Forward, &seed, None, &mut short.buf()),
            Err(ReachRefusal::Length {
                part: ReachPart::Hop,
                want: P,
                got: P - 1
            })
        );
        let wide = Owner {
            plane: PairPlane::new(&f, Axis::new(Width::W64, 2), Axis::new(Width::W256, 1)),
        };
        assert_eq!(
            relation_reach(
                &wide,
                0,
                &f,
                PairDir::Forward,
                &mask(&[0])[..4],
                None,
                &mut r.buf()
            ),
            Err(ReachRefusal::NotSquare { src: 128, dst: 256 })
        );
        assert_eq!(r.reached, vec![u64::MAX; W]);
        assert_eq!(r.hop, vec![7; P]);
        assert_eq!(r.scratch, vec![u64::MAX; 2 * W]);
    }
}
