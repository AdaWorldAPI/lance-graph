//! D-REGC-1 / D-REGC-2: the read by reference. After bind, a read consults
//! no `ClassView`. It checks frame, route, the store's revision and basis
//! digest, every lane's mapping, scope and row order, and nothing else.
//!
//! Each refusal fires on a realistic input. A view whose fields moved after
//! bind is refused by `check` (the view read) and changes nothing that the
//! bound read returns.

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{Frame, FrameRefusal, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneBinding, LaneRefusal, LaneRequest,
    LaneSource, PresenceSource, RouteAt, SchemaRev, SelAddr, SelectionSource,
};

const CLASS: u32 = 0x0D2A_0001;
const OTHER: u32 = 0x0D2B_0001;
const POSITION: u16 = 9;
const READER: Scope = Scope { id: 3 };
const ROWS: CoordSpace = CoordSpace {
    population: 4,
    coordinates: 1,
};
const REFOLDED: CoordSpace = CoordSpace {
    population: 4,
    coordinates: 2,
};

struct View {
    class: ClassId,
    fields: Vec<FieldRef>,
}

impl View {
    fn new(class: u32, names: &[&str]) -> View {
        View {
            class: (class >> 16) as ClassId,
            fields: names
                .iter()
                .map(|n| FieldRef::new(format!("t:{n}"), *n))
                .collect(),
        }
    }
}

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == self.class {
            &self.fields
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

/// Lanes mapped by predicate, as a source owner maps them: the field's
/// identity, never its position in some view.
struct Source {
    names: Vec<&'static str>,
    lanes: Vec<Vec<u32>>,
    presence: Vec<Vec<u64>>,
    frames: FrameStore,
    selections: Vec<(CoordSpace, Vec<u32>)>,
}

impl Source {
    fn new() -> Source {
        Source {
            names: vec!["a", "b", "c"],
            lanes: vec![vec![1, 2, 3], vec![10, 20, 30], vec![7, 0, 9]],
            presence: vec![vec![0b111], vec![0b101], vec![0b101]],
            frames: FrameStore::new(FrameRevisions {
                source: 1,
                relation_map: 1,
                coordinates: 1,
            }),
            selections: vec![(ROWS, vec![2, 0, 2]), (REFOLDED, vec![1])],
        }
    }
}

impl LaneSource for Source {
    type Lane<'a> = &'a [u32];
    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        _field: u8,
        basis: &FieldRef,
        _frame: &Frame,
    ) -> Option<LaneAddr> {
        if class != CLASS || rev != SchemaRev(1) {
            return None;
        }
        let lane = self
            .names
            .iter()
            .position(|n| format!("t:{n}") == basis.predicate_iri)?;
        Some(LaneAddr {
            space: ROWS,
            lane: lane as u16,
            scope: READER,
        })
    }
    fn borrow(&self, addr: &LaneAddr) -> Option<&[u32]> {
        self.lanes.get(usize::from(addr.lane)).map(Vec::as_slice)
    }
    fn frames(&self) -> &FrameStore {
        &self.frames
    }
}

impl PresenceSource for Source {
    fn borrow_presence(&self, addr: &LaneAddr) -> Option<&[u64]> {
        self.presence.get(usize::from(addr.lane)).map(Vec::as_slice)
    }
}

impl SelectionSource for Source {
    fn selection_of(&self, sel: u16, _frame: &Frame) -> Option<SelAddr> {
        let (space, _) = self.selections.get(usize::from(sel))?;
        Some(SelAddr {
            space: *space,
            sel,
            scope: READER,
        })
    }
    fn borrow_positions(&self, addr: &SelAddr) -> Option<&[u32]> {
        self.selections
            .get(usize::from(addr.sel))
            .map(|(_, p)| p.as_slice())
    }
}

fn schema(fields: usize) -> ClassSchema {
    ClassSchema {
        schema_rev: SchemaRev(1),
        field_count: fields,
        extent: Extent::DR64_CUBE,
    }
}

fn store_with(view: &View, fields: usize) -> (BindStore, RouteAt) {
    let mut store = BindStore::new();
    store
        .register(view, CLASS, schema(fields), Vec::new())
        .expect("register");
    let version = store.route(POSITION, CLASS).expect("route");
    (
        store,
        RouteAt {
            position: POSITION,
            version,
        },
    )
}

fn bind(store: &BindStore, view: &View, source: &Source, route: RouteAt) -> LaneBinding {
    let frame = source.frames.seal();
    store
        .bind_lanes(
            view,
            &source.frames,
            source,
            LaneRequest {
                route,
                rev: SchemaRev(1),
                fields: &WideFieldMask::from_positions(&[0, 1, 2]),
                frame: &frame,
                scope: READER,
                selection: ROWS,
            },
        )
        .expect("bind")
}

type Lent<'s> = Vec<(u8, &'s [u32], &'s [u64])>;

fn lent<'s>(it: impl Iterator<Item = Result<(u8, &'s [u32], &'s [u64]), LaneRefusal>>) -> Lent<'s> {
    it.map(|l| l.expect("lane")).collect()
}

/// FAILS IF: the bound read lends other lanes, planes or positions than the
/// view read, or copies them.
#[test]
fn the_bound_read_lends_what_the_view_read_lends() {
    let source = Source::new();
    let view = View::new(CLASS, &["a", "b", "c"]);
    let (store, route) = store_with(&view, 3);
    let b = bind(&store, &view, &source, route);

    let by_view = lent(
        b.borrow_all_present(&store, &view, &source.frames, &source)
            .unwrap(),
    );
    let by_ref = lent(
        b.borrow_all_present_bound(&store, &source.frames, &source)
            .unwrap(),
    );
    assert_eq!(by_view.len(), 3);
    for ((f1, l1, p1), (f2, l2, p2)) in by_view.iter().zip(&by_ref) {
        assert_eq!(f1, f2);
        assert!(std::ptr::eq(*l1, *l2) && std::ptr::eq(*p1, *p2));
        assert!(std::ptr::eq(*l2, source.lanes[usize::from(*f2)].as_slice()));
    }
    let (_, pos) = b
        .borrow_all_selected_present_bound(&store, &source.frames, &source, 0)
        .unwrap();
    assert_eq!(pos, &[2, 0, 2], "a bag in list order");
}

/// FAILS IF: a view whose fields moved after bind changes what the bound
/// read lends, or the view read stops refusing it.
#[test]
fn a_view_changed_after_bind_is_invisible_to_the_bound_read() {
    let source = Source::new();
    let view = View::new(CLASS, &["a", "b", "c"]);
    let (store, route) = store_with(&view, 3);
    let b = bind(&store, &view, &source, route);

    let swapped = View::new(CLASS, &["b", "a", "c"]);
    assert!(
        b.check(&store, &swapped, &source.frames, &source).is_err(),
        "the view read refuses a moved field"
    );
    let lanes = lent(
        b.borrow_all_present_bound(&store, &source.frames, &source)
            .unwrap(),
    );
    assert_eq!(
        lanes.iter().map(|(_, l, _)| l[0]).collect::<Vec<_>>(),
        vec![1, 10, 7],
        "field 0 is still `a`, as it was bound"
    );
}

/// FAILS IF: a store that holds another basis for the class is read through.
#[test]
fn a_store_with_another_basis_is_refused() {
    let source = Source::new();
    let view = View::new(CLASS, &["a", "b", "c"]);
    let (store, route) = store_with(&view, 3);
    let b = bind(&store, &view, &source, route);

    // Another store: same classid, same route position, a fourth field.
    let wider = View::new(CLASS, &["a", "b", "c", "d"]);
    let (other, _) = store_with(&wider, 4);
    assert_eq!(
        b.check_bound(&other, &source.frames, &source),
        Err(LaneRefusal::StoreBasisMismatch { class: CLASS })
    );
    // The same basis in another store reads: the digest names the basis.
    let (twin, _) = store_with(&view, 3);
    assert!(b.check_bound(&twin, &source.frames, &source).is_ok());
}

/// FAILS IF: any of frame, route, mapping, frame store or row order goes
/// unchecked by reference.
#[test]
fn frame_route_mapping_and_row_order_are_refused() {
    let view = View::new(CLASS, &["a", "b", "c"]);

    // A stale frame: the source advanced after bind.
    let mut source = Source::new();
    let (store, route) = store_with(&view, 3);
    let b = bind(&store, &view, &source, route);
    source.frames.advance(FrameRevisions {
        source: 2,
        relation_map: 1,
        coordinates: 1,
    });
    assert!(matches!(
        b.check_bound(&store, &source.frames, &source),
        Err(LaneRefusal::Frame(_))
    ));

    // A foreign frame store.
    let source = Source::new();
    let b = bind(&store, &view, &source, route);
    let foreign = FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 1,
    });
    assert_eq!(
        b.check_bound(&store, &foreign, &source),
        Err(LaneRefusal::Frame(FrameRefusal::ForeignFrame))
    );

    // The position routes to another class now.
    let mut moved = store.clone();
    moved
        .register(&View::new(OTHER, &["x"]), OTHER, schema(1), Vec::new())
        .unwrap();
    let b = bind(&moved, &view, &source, route);
    moved.route(POSITION, OTHER).unwrap();
    assert!(matches!(
        b.check_bound(&moved, &source.frames, &source),
        Err(LaneRefusal::RouteChanged { .. })
    ));

    // The source maps a field to another lane now.
    let mut remapped = Source::new();
    let b = bind(&store, &view, &remapped, route);
    remapped.names.swap(0, 1);
    assert_eq!(
        b.check_bound(&store, &remapped.frames, &remapped),
        Err(LaneRefusal::MappingChanged { field: 0 })
    );

    // A selection issued in another row order.
    let source = Source::new();
    let b = bind(&store, &view, &source, route);
    assert!(matches!(
        b.borrow_all_selected_present_bound(&store, &source.frames, &source, 1),
        Err(LaneRefusal::SelectionSpace { .. })
    ));
}

/// FAILS IF: `borrow_all_at` lends other lanes or planes than the bound
/// read, copies or reorders the caller's positions, or admits positions
/// resolved under a frame that is no longer the source's current seal.
#[test]
fn borrow_all_at_lends_in_place_and_refuses_positions_from_another_frame() {
    let view = View::new(CLASS, &["a", "b", "c"]);
    let (store, route) = store_with(&view, 3);

    // Current frame: lent in place, positions handed back as given.
    let source = Source::new();
    let b = bind(&store, &view, &source, route);
    let positions = [2u32, 0, 2];
    let under = source.frames.seal();
    let (lanes, at) = b
        .borrow_all_at(&store, &source.frames, &source, &positions, &under)
        .expect("current frame");
    assert!(std::ptr::eq(at, positions.as_slice()));
    let by_at = lent(lanes);
    let by_ref = lent(
        b.borrow_all_present_bound(&store, &source.frames, &source)
            .unwrap(),
    );
    assert_eq!(by_at.len(), 3);
    for ((f1, l1, p1), (f2, l2, p2)) in by_at.iter().zip(&by_ref) {
        assert_eq!(f1, f2);
        assert!(std::ptr::eq(*l1, *l2) && std::ptr::eq(*p1, *p2));
    }

    // Positions resolved before an advance, then a binding made after it:
    // the binding is current, the positions are not.
    let mut source = Source::new();
    let old = source.frames.seal();
    source.frames.advance(FrameRevisions {
        source: 2,
        relation_map: 1,
        coordinates: 1,
    });
    let b = bind(&store, &view, &source, route);
    assert!(b.check_bound(&store, &source.frames, &source).is_ok());
    assert!(matches!(
        b.borrow_all_at(&store, &source.frames, &source, &positions, &old),
        Err(LaneRefusal::Frame(FrameRefusal::SourceChanged))
    ));

    // Positions resolved under another store's frame.
    let foreign = FrameStore::new(FrameRevisions {
        source: 2,
        relation_map: 1,
        coordinates: 1,
    });
    assert!(matches!(
        b.borrow_all_at(&store, &source.frames, &source, &positions, &foreign.seal()),
        Err(LaneRefusal::Frame(FrameRefusal::ForeignFrame))
    ));
}
