//! D-LANESOCK-1: the property-to-lane socket. Every refusal fires on a
//! realistic input, and a binding that passes borrows the source's own lane.

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
        if class != HYBRID || rev != SchemaRev(1) {
            return None;
        }
        self.map.iter().find(|(f, _)| *f == field).map(|(_, a)| *a)
    }
    fn borrow(&self, addr: &LaneAddr) -> Option<&[i32]> {
        self.lanes.get(usize::from(addr.lane)).map(Vec::as_slice)
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
    frames: FrameStore,
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
    let frames = FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 1,
    });
    let frame = frames.seal();
    World {
        store,
        view: View::three(),
        frames,
        frame,
        source: Source::hybrid(),
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
        .bind_lanes(&w.view, &w.frames, &w.source, request(w, fields))
}

#[test]
fn two_fields_of_one_basis_bind_and_borrow_the_source_lanes() {
    let w = world();
    let both = WideFieldMask::from_positions(&[0, 1]);
    let b = bind(&w, &both).unwrap();
    assert_eq!(b.class(), HYBRID);
    assert!(b.fields().has(0) && b.fields().has(1));
    let dept = b.lane(&w.store, &w.view, &w.frames, &w.source, 0).unwrap();
    let cost = b.lane(&w.store, &w.view, &w.frames, &w.source, 1).unwrap();
    // Borrowed, not copied: the very slices the source holds.
    assert!(std::ptr::eq(dept, w.source.lanes[0].as_slice()));
    assert!(std::ptr::eq(cost, w.source.lanes[1].as_slice()));
    assert_eq!(
        b.lane(&w.store, &w.view, &w.frames, &w.source, 2),
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
        .bind_lanes(&w.view, &w.frames, &w.source, req)
        .is_ok());
}

#[test]
fn wrong_schema_revision_and_changed_basis_are_refused() {
    let w = world();
    let mask = WideFieldMask::from_positions(&[0]);
    let mut req = request(&w, &mask);
    req.rev = SchemaRev(2);
    assert!(matches!(
        w.store.bind_lanes(&w.view, &w.frames, &w.source, req),
        Err(LaneRefusal::Bind(Refusal::SchemaRevisionMismatch { .. }))
    ));

    let b = bind(&w, &mask).unwrap();
    let narrower = View(View::three().0[..2].to_vec());
    assert!(matches!(
        b.check(&w.store, &narrower, &w.frames, &w.source),
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
        b.check(&w.store, &w.view, &w.frames, &w.source),
        Err(LaneRefusal::MappingChanged { field: 0 })
    );

    let mut w = world();
    let b = bind(&w, &both).unwrap();
    w.source.lanes.truncate(1);
    assert_eq!(
        b.lane(&w.store, &w.view, &w.frames, &w.source, 1),
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
    w.frames.advance(FrameRevisions {
        source: 1,
        relation_map: 2,
        coordinates: 1,
    });
    assert_eq!(
        b.check(&w.store, &w.view, &w.frames, &w.source),
        Err(LaneRefusal::Frame(FrameRefusal::MapChanged))
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
    assert!(w.frames.check(b.frame()).is_ok(), "the frame did not move");
    assert_eq!(
        b.check(&w.store, &w.view, &w.frames, &w.source),
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
        b.check(&other, &w.view, &w.frames, &w.source),
        Err(LaneRefusal::StoreBasisMismatch { class: HYBRID })
    );
    assert!(b.check(&w.store, &w.view, &w.frames, &w.source).is_ok());
}

/// FINDING (L5d): the hot check does not re-hash the view, so a same-count
/// reorder of the VIEW after the bind passes it when the source maps lanes by
/// position (this fixture's `Source` does). `resolve` through the same view
/// refuses. Re-bind after a basis change; L5d decides whether R2IL must.
#[test]
fn finding_the_hot_check_does_not_see_a_view_reorder_under_a_positional_source() {
    let w = world();
    let mask = WideFieldMask::from_positions(&[0]);
    let b = bind(&w, &mask).unwrap();
    let mut swapped = View::three();
    swapped.0.swap(0, 1);
    assert!(w.store.resolve(&swapped, HYBRID, SchemaRev(1)).is_err());
    assert!(b.check(&w.store, &swapped, &w.frames, &w.source).is_ok());
}
