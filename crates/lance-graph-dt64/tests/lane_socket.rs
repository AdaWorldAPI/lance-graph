//! D-LANESOCK-1: the property-to-lane socket. Every refusal fires on a
//! realistic input, and a binding that passes borrows the source's own lane.

use std::cell::Cell;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{Frame, FrameRefusal, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneRefusal, LaneRequest, LaneSource,
    Refusal, RouteAt, SchemaRev,
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
    /// `lane_of` calls, and `borrow` calls, so far.
    lookups: Cell<usize>,
    borrows: Cell<usize>,
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
            lookups: Cell::new(0),
            borrows: Cell::new(0),
        }
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
