//! E0 (roadmap stage E, first integration), on a real `MailboxSoA<64>`:
//! publication -> routed data class -> admitted method -> a lane borrowed
//! over `reads` only -> one demanded read.
//!
//! Oracle: the owner's public arrays and the generator functions below. No
//! guard, no effect, no selection, no `NodeGuid`.

use causal_edge::CausalEdge64;
use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_contract::soa_view::MailboxSoaView;
use lance_graph_dt64::dr::{FrameRefusal, Scope};
use lance_graph_dt64::{
    view_class, BindStore, ClassSchema, CoordSpace, Extent, LaneBinding, LaneRefusal, LaneRequest,
    MethodBinding, MethodId, Refusal, RouteAt, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mailbox_lanes::{Column, ColumnRefusal, MailboxLanes};
use lance_graph_mask_risc::LaneRef;

const N: usize = 64;
type Owner = MailboxSoA<N>;
const POPULATED: usize = 40;

/// The data class, a second data class over the same basis, and a class
/// whose basis names what the mailbox cannot lend.
const CLASS: u32 = 0x0A1A_0003;
const OTHER: u32 = 0x0A1A_0004;
const WRONG: u32 = 0x0A1B_0003;
const POSITION: u16 = 11;
const WRONG_POSITION: u16 = 12;
const READER: Scope = Scope { id: 3 };

/// Field positions of the class basis. Deliberately not the column order.
const TEMPORAL: u8 = 0;
const META: u8 = 1;
const EDGES: u8 = 2;
const READ_META: MethodId = MethodId(1);
const READ_EDGES: MethodId = MethodId(2);

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}
fn meta_of(id: u32, r: usize) -> u32 {
    mix(u64::from(id) << 32 | r as u64) as u32 | 1
}
fn edge_of(id: u32, r: usize) -> u64 {
    mix(1_000 + (u64::from(id) << 32 | r as u64)) | 1
}
fn temporal_of(id: u32, r: usize) -> u64 {
    mix(2_000 + (u64::from(id) << 32 | r as u64)) | 1
}

fn owner(id: u32) -> Box<Owner> {
    let mut m: Box<Owner> = Box::new(MailboxSoA::new(id, 3, 1.0));
    m.set_populated(POPULATED);
    for r in 0..POPULATED {
        m.set_meta(r, MetaWord(meta_of(id, r)));
        m.set_edge(r, CausalEdge64(edge_of(id, r)));
        m.set_temporal(r, temporal_of(id, r));
    }
    m
}

struct View {
    basis: Vec<FieldRef>,
    wrong: Vec<FieldRef>,
}

impl View {
    fn new() -> View {
        let field = |iri: &str| FieldRef::new(iri, iri);
        View {
            basis: vec![
                field(Column::Temporal.iri()),
                field(Column::Meta.iri()),
                field(Column::Edges.iri()),
            ],
            wrong: vec![field("urn:mailbox:energy"), field(Column::Meta.iri())],
        }
    }
}

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == view_class(CLASS) {
            &self.basis
        } else if class == view_class(WRONG) {
            &self.wrong
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

fn method(id: MethodId, name: &str, on: &[u8], reads: &[u8]) -> MethodBinding {
    MethodBinding {
        id,
        sig_rev: SigRev(1),
        name: name.into(),
        on: WideFieldMask::from_positions(on),
        when: None,
        reads: WideFieldMask::from_positions(reads),
        effects: vec![],
        population: Population::Bag,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

struct World {
    store: BindStore,
    view: View,
    route: RouteAt,
    wrong_route: RouteAt,
}

fn world() -> World {
    let view = View::new();
    let mut store = BindStore::new();
    let schema = |field_count| ClassSchema {
        schema_rev: SchemaRev(1),
        field_count,
        extent: Extent::DR256_SQ_X64,
    };
    store
        .register(
            &view,
            CLASS,
            schema(3),
            vec![
                method(READ_META, "read_meta", &[EDGES], &[META]),
                method(READ_EDGES, "read_edges", &[META], &[EDGES, TEMPORAL]),
            ],
        )
        .expect("the data class registers");
    store
        .register(
            &view,
            OTHER,
            schema(3),
            vec![method(READ_META, "impostor", &[EDGES], &[META, EDGES])],
        )
        .expect("the second class registers");
    store
        .register(&view, WRONG, schema(2), vec![])
        .expect("the wrong-basis class registers");
    let version = store.route(POSITION, CLASS).unwrap();
    let wrong_version = store.route(WRONG_POSITION, WRONG).unwrap();
    World {
        store,
        view,
        route: RouteAt {
            position: POSITION,
            version,
        },
        wrong_route: RouteAt {
            position: WRONG_POSITION,
            version: wrong_version,
        },
    }
}

fn bind(
    w: &World,
    lanes: &MailboxLanes<'_, N>,
    route: RouteAt,
    fields: &WideFieldMask,
) -> Result<LaneBinding, LaneRefusal> {
    let frame = lanes.seal();
    w.store.bind_lanes(
        &w.view,
        lanes.frame_store(),
        lanes,
        LaneRequest {
            route,
            rev: SchemaRev(1),
            fields,
            frame: &frame,
            scope: READER,
            selection: lanes.space(),
        },
    )
}

fn u32_lane<'s>(
    w: &World,
    b: &LaneBinding,
    lanes: &'s MailboxLanes<'_, N>,
    field: u8,
) -> Result<&'s [u32], LaneRefusal> {
    match b.lane(&w.store, &w.view, lanes.frame_store(), lanes, field)? {
        LaneRef::U32(v) => Ok(v),
        other => panic!("a u32 lane, got {other:?}"),
    }
}

/// E0, positive. FAILS IF: an admitted read differs from the owner's array,
/// the lent slice is not the owner's memory, or the lane is longer than the
/// populated prefix.
#[test]
fn every_admitted_read_is_the_owners_own_value() {
    let m = owner(7);
    let w = world();
    let lanes = MailboxLanes::new(m.published(), READER);

    // ON and route-bound admission, from the address's field. No lane yet.
    let class = w
        .store
        .resolve_at(&w.view, w.route.version, POSITION, SchemaRev(1))
        .unwrap();
    assert_eq!(class.class, CLASS);
    let names: Vec<&str> = class.methods_on(EDGES).map(|x| x.name.as_str()).collect();
    assert_eq!(names, ["read_meta"]);
    let own = w.store.method_ref(CLASS, READ_META).unwrap();
    let admitted = w.store.admit_routed(w.route, &own).unwrap();

    // The lane is bound over the admitted method's `reads` only.
    let b = bind(&w, &lanes, w.route, &admitted.reads).unwrap();
    assert_eq!(b.fields().count(), 1);
    let meta = u32_lane(&w, &b, &lanes, META).unwrap();
    assert_eq!(meta.len(), POPULATED);
    assert_eq!(meta.as_ptr().cast::<u8>(), m.meta.as_ptr().cast::<u8>());
    let mut distinct = std::collections::BTreeSet::new();
    for (r, &value) in meta.iter().enumerate() {
        assert_eq!(value, m.meta[r].0, "row {r}");
        assert_eq!(value, meta_of(7, r), "row {r}");
        distinct.insert(value);
    }
    assert_eq!(distinct.len(), POPULATED, "heterogeneous values");

    // A position past the populated prefix is not in the lane.
    assert_eq!(meta.get(POPULATED), None);
    assert_eq!(meta.get(N - 1), None);

    // The 64-bit columns, through the other method's `reads`.
    let edges_ref = w.store.method_ref(CLASS, READ_EDGES).unwrap();
    let reads = w.store.admit_routed(w.route, &edges_ref).unwrap().reads.clone();
    let b2 = bind(&w, &lanes, w.route, &reads).unwrap();
    for (field, column, oracle) in [
        (
            EDGES,
            m.edges_raw().as_ptr(),
            edge_of as fn(u32, usize) -> u64,
        ),
        (TEMPORAL, m.temporal.as_ptr(), temporal_of),
    ] {
        let LaneRef::U64(v) = b2
            .lane(&w.store, &w.view, lanes.frame_store(), &lanes, field)
            .unwrap()
        else {
            panic!("a u64 lane")
        };
        assert_eq!(v.as_ptr(), column);
        assert_eq!(v.len(), POPULATED);
        for (r, &value) in v.iter().enumerate() {
            assert_eq!(value, oracle(7, r));
        }
    }
    assert_eq!(m.temporal[5], temporal_of(7, 5));
    assert_eq!(m.edges[5].0, edge_of(7, 5));
}

/// FAILS IF: a field outside the admitted `reads`, or outside the basis, can
/// be read through the binding.
#[test]
fn a_field_outside_reads_is_not_readable() {
    let m = owner(7);
    let w = world();
    let lanes = MailboxLanes::new(m.published(), READER);
    let own = w.store.method_ref(CLASS, READ_META).unwrap();
    let admitted = w.store.admit_routed(w.route, &own).unwrap();
    let b = bind(&w, &lanes, w.route, &admitted.reads).unwrap();
    for outside in [EDGES, TEMPORAL] {
        assert_eq!(
            b.lane(&w.store, &w.view, lanes.frame_store(), &lanes, outside)
                .err(),
            Some(LaneRefusal::NotBound { field: outside })
        );
    }
    assert_eq!(
        bind(&w, &lanes, w.route, &WideFieldMask::from_positions(&[3])).err(),
        Some(LaneRefusal::FieldNotInBasis {
            class: CLASS,
            field: 3
        })
    );
}

/// FAILS IF: a method of a class the position does not route to is admitted
/// for it, or the routed class's method survives a re-route.
#[test]
fn only_the_routed_classes_method_is_admitted() {
    let m = owner(7);
    let mut w = world();
    let foreign = w.store.method_ref(OTHER, READ_META).unwrap();
    assert_eq!(w.store.admit(&foreign).unwrap().name, "impostor");
    assert_eq!(
        w.store.admit_routed(w.route, &foreign).err(),
        Some(Refusal::NotRoutedClass {
            position: POSITION,
            captured: Some(CLASS),
            now: Some(CLASS),
            class: OTHER
        })
    );
    let own = w.store.method_ref(CLASS, READ_META).unwrap();
    let reads = w.store.admit_routed(w.route, &own).unwrap().reads.clone();

    let lanes = MailboxLanes::new(m.published(), READER);
    let b = bind(&w, &lanes, w.route, &reads).unwrap();
    assert!(u32_lane(&w, &b, &lanes, META).is_ok());

    // The position is routed elsewhere.
    w.store.route(POSITION, OTHER).unwrap();
    assert_eq!(
        w.store.admit_routed(w.route, &own).err(),
        Some(Refusal::NotRoutedClass {
            position: POSITION,
            captured: Some(CLASS),
            now: Some(OTHER),
            class: CLASS
        })
    );
    assert_eq!(
        u32_lane(&w, &b, &lanes, META).err(),
        Some(LaneRefusal::RouteChanged {
            position: POSITION,
            bound: CLASS,
            now: Some(OTHER)
        })
    );

    // A reference of another store is refused whatever the route says.
    let twin = world();
    let theirs = twin.store.method_ref(CLASS, READ_META).unwrap();
    assert!(matches!(
        world().store.admit_routed(w.route, &theirs),
        Err(Refusal::ForeignStore { class: CLASS })
    ));
}

/// FAILS IF: a class whose basis names a column the mailbox cannot lend
/// binds anyway.
#[test]
fn a_basis_the_mailbox_cannot_lend_does_not_bind() {
    let m = owner(7);
    let w = world();
    let lanes = MailboxLanes::new(m.published(), READER);
    assert_eq!(
        bind(
            &w,
            &lanes,
            w.wrong_route,
            &WideFieldMask::from_positions(&[0, 1])
        )
        .err(),
        Some(LaneRefusal::Unbound {
            class: WRONG,
            field: 0
        })
    );
    assert_eq!(
        Column::from_iri("urn:mailbox:energy"),
        Err(ColumnRefusal::UnsupportedKind("f32"))
    );
    assert_eq!(
        Column::from_iri("urn:mailbox:nothing"),
        Err(ColumnRefusal::Unknown)
    );
    for c in Column::ALL {
        assert_eq!(Column::from_iri(c.iri()), Ok(c));
    }
}

/// A1. FAILS IF: a binding made under one publication reads under another.
/// The write between them moves neither mailbox stamp, so a frame keyed on
/// the cycle would have stayed current.
#[test]
fn a_binding_does_not_survive_its_publication() {
    let mut m = owner(7);
    let w = world();
    let reads = WideFieldMask::from_positions(&[META]);
    let (first, old_frame) = {
        let lanes = MailboxLanes::new(m.published(), READER);
        let b = bind(&w, &lanes, w.route, &reads).unwrap();
        assert_eq!(u32_lane(&w, &b, &lanes, META).unwrap()[9], meta_of(7, 9));
        (b, lanes.seal())
    };

    let stamps = (m.current_cycle(), m.last_write_cycle_at(9));
    m.set_meta(9, MetaWord(0xDEAD_BEEF));
    assert_eq!((m.current_cycle(), m.last_write_cycle_at(9)), stamps);

    let lanes = MailboxLanes::new(m.published(), READER);
    assert_eq!(
        u32_lane(&w, &first, &lanes, META).err(),
        Some(LaneRefusal::Frame(FrameRefusal::ForeignFrame))
    );
    // The old frame cannot seal a new binding either.
    assert_eq!(
        w.store
            .bind_lanes(
                &w.view,
                lanes.frame_store(),
                &lanes,
                LaneRequest {
                    route: w.route,
                    rev: SchemaRev(1),
                    fields: &reads,
                    frame: &old_frame,
                    scope: READER,
                    selection: lanes.space(),
                },
            )
            .err(),
        Some(LaneRefusal::Frame(FrameRefusal::ForeignFrame))
    );
    // A fresh binding reads the new value.
    let fresh = bind(&w, &lanes, w.route, &reads).unwrap();
    assert_eq!(u32_lane(&w, &fresh, &lanes, META).unwrap()[9], 0xDEAD_BEEF);

    // Two sockets over the SAME publication do not share frames either.
    let again = MailboxLanes::new(m.published(), READER);
    assert_eq!(
        u32_lane(&w, &fresh, &again, META).err(),
        Some(LaneRefusal::Frame(FrameRefusal::ForeignFrame))
    );
}

/// FAILS IF: a binding over one mailbox reads another, a caller pairs one
/// source with another's frame store, or the row order or reader scope of
/// another source is accepted.
#[test]
fn another_owner_is_another_source() {
    let (a, b) = (owner(7), owner(8));
    let w = world();
    let reads = WideFieldMask::from_positions(&[META]);
    let (la, lb) = (
        MailboxLanes::new(a.published(), READER),
        MailboxLanes::new(b.published(), READER),
    );
    let bound = bind(&w, &la, w.route, &reads).unwrap();
    assert_eq!(
        u32_lane(&w, &bound, &lb, META).err(),
        Some(LaneRefusal::Frame(FrameRefusal::ForeignFrame))
    );
    // A's frame store with B as the source.
    assert_eq!(
        bound
            .lane(&w.store, &w.view, la.frame_store(), &lb, META)
            .err(),
        Some(LaneRefusal::Frame(FrameRefusal::ForeignFrame))
    );
    assert_ne!(la.space(), lb.space());

    let request = |selection: CoordSpace, scope: Scope| {
        let frame = la.seal();
        w.store.bind_lanes(
            &w.view,
            la.frame_store(),
            &la,
            LaneRequest {
                route: w.route,
                rev: SchemaRev(1),
                fields: &reads,
                frame: &frame,
                scope,
                selection,
            },
        )
    };
    assert_eq!(
        request(lb.space(), READER).err(),
        Some(LaneRefusal::CoordinateSpace {
            field: META,
            selection: lb.space(),
            lane: la.space()
        })
    );
    let stranger = Scope { id: 99 };
    assert_eq!(
        request(la.space(), stranger).err(),
        Some(LaneRefusal::Scope {
            field: META,
            reader: stranger,
            lane: READER
        })
    );
    assert_eq!(a.meta_raw()[3], meta_of(7, 3));
    assert_eq!(b.meta_raw()[3], meta_of(8, 3));
}
