//! D-LANESOCK-2/3: resident `AbiBatch` lanes through DT64's lane socket,
//! folded by Quack / mask-RISC. Every refusal fires on a realistic input.
//!
//! These are unit tests because the only `AbiLanes` constructor,
//! `AbiLanes::unwitnessed`, takes the reader scope from the caller and exists
//! only under `#[cfg(test)]` (see the module docs: no DR scope witness).

use std::sync::Arc;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ogar_codebook::compose_classid;
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{FrameRefusal, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneBinding, LaneRefusal, LaneRequest,
    MethodBinding, MethodId, MethodRef, RouteAt, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mask_risc::{execute_into, Foreign, LaneRef, Out, Planes, Scratch, Value};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, Query};

use super::{AbiLanes, PublishRefusal};
use crate::batch::{AbiBatch, BatchError, Column, LaneData, MAX_COLUMNS};
use crate::boundary::Catalog;
use crate::ids::{FieldId, SourceId};

// ── allocation counter (hot-path falsifier) ──────────────────────────────────

struct CountingAlloc;
thread_local! {
    static ALLOCATIONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}
#[allow(unsafe_code)] // test-only counting allocator; the library forbids unsafe
unsafe impl std::alloc::GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        unsafe { std::alloc::System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;
fn allocations() -> u64 {
    ALLOCATIONS.with(|c| c.get())
}

// ── the class, its ClassView basis, the catalog and the resident batch ───────

/// Concept half `0x0C1C`, custom half `1`, under the codebook's active split.
const CONCEPT: ClassId = 0x0C1C;
fn class() -> u32 {
    compose_classid(CONCEPT, 1)
}
fn other_class() -> u32 {
    compose_classid(CONCEPT, 2)
}
const POSITION: u16 = 9;
const READER: Scope = Scope { id: 41 };
const SOURCE: SourceId = SourceId(7);
const N: usize = 300;

/// Basis positions.
const REVENUE: u8 = 0; // I32 lane
const REGION: u8 = 1; // U32 coordinate lane
const NOT_CATALOGUED: u8 = 2; // name the catalog does not know
const NO_COLUMN: u8 = 3; // catalogued, but the batch has no column
const REFERENCE: u8 = 4; // U64 KV reference lane: not an admitted value kind

const F_REVENUE: FieldId = FieldId(1);
const F_REGION: FieldId = FieldId(2);
const F_NO_COLUMN: FieldId = FieldId(3);
const F_REFERENCE: FieldId = FieldId(4);

struct View(Vec<FieldRef>);
impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == CONCEPT {
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

fn view() -> View {
    View(vec![
        FieldRef::new("rpt:revenue", "Revenue"),
        FieldRef::new("rpt:region", "Region"),
        FieldRef::new("rpt:notCatalogued", "Not catalogued"),
        FieldRef::new("rpt:noColumn", "No column"),
        FieldRef::new("rpt:document", "Document"),
    ])
}

fn catalog() -> Catalog {
    Catalog::default()
        .with("rpt:revenue", F_REVENUE)
        .with("rpt:region", F_REGION)
        .with("rpt:noColumn", F_NO_COLUMN)
        .with("rpt:document", F_REFERENCE)
}

fn revenue(generation: u32) -> Vec<i32> {
    (0..N)
        .map(|i| ((i * 37 + generation as usize * 11) % 1000) as i32 - 200)
        .collect()
}
fn region() -> Vec<u32> {
    (0..N).map(|i| (i % 5) as u32).collect()
}

fn batch(generation: u32) -> Arc<AbiBatch> {
    let refs: Arc<[u64]> = (0..N as u64).collect();
    Arc::new(
        AbiBatch::new(SOURCE, generation, N)
            .with_column(Column::value(
                F_REVENUE,
                LaneData::I32(revenue(generation).into()),
            ))
            .unwrap()
            .with_column(Column::coordinate(F_REGION, region().into(), 5))
            .unwrap()
            .with_column(Column::value(F_REFERENCE, LaneData::U64(refs)))
            .unwrap(),
    )
}

fn test_method(id: u32, reads: &[u8]) -> MethodBinding {
    // A test fold, not a production F### implementation.
    MethodBinding {
        id: MethodId(id),
        sig_rev: SigRev(1),
        name: format!("test-only/fold-{id}"),
        on: WideFieldMask::EMPTY,
        when: None,
        reads: WideFieldMask::from_positions(reads),
        effects: Vec::new(),
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

fn store() -> (BindStore, RouteAt) {
    let mut store = BindStore::new();
    let view = view();
    store
        .register(
            &view,
            class(),
            ClassSchema {
                schema_rev: SchemaRev(1),
                field_count: 5,
                extent: Extent::DR64_CUBE,
            },
            vec![
                test_method(1, &[REVENUE, REGION]),
                test_method(2, &[REGION]),
            ],
        )
        .unwrap();
    let version = store.route(POSITION, class()).unwrap();
    (
        store,
        RouteAt {
            position: POSITION,
            version,
        },
    )
}

fn request<'a>(
    route: RouteAt,
    fields: &'a WideFieldMask,
    frame: &'a lance_graph_dt64::dr::Frame,
    selection: CoordSpace,
) -> LaneRequest<'a> {
    LaneRequest {
        route,
        rev: SchemaRev(1),
        fields,
        frame,
        scope: READER,
        selection,
    }
}

fn bind<V: ClassView>(
    store: &BindStore,
    view: &V,
    lanes: &AbiLanes<'_>,
    route: RouteAt,
    positions: &[u8],
) -> Result<LaneBinding, LaneRefusal> {
    let fields = WideFieldMask::from_positions(positions);
    let frame = lanes.seal();
    store.bind_lanes(
        view,
        lanes.frames(),
        lanes,
        request(route, &fields, &frame, lanes.space()),
    )
}

fn slice_i32(lane: LaneRef<'_>) -> &[i32] {
    match lane {
        LaneRef::I32(v) => v,
        _ => panic!("expected an I32 lane"),
    }
}
fn slice_u32(lane: LaneRef<'_>) -> &[u32] {
    match lane {
        LaneRef::U32(v) => v,
        _ => panic!("expected a U32 lane"),
    }
}

// ── positive: borrow, admit, fold, parity, reuse, zero hot allocations ───────

#[test]
fn bound_lanes_are_the_batch_lanes_and_fold_with_scalar_parity() {
    let (store, route) = store();
    let view = view();
    let catalog = catalog();
    let batch = batch(1);
    let lanes = AbiLanes::unwitnessed(&catalog, Arc::clone(&batch), READER);
    let binding = bind(&store, &view, &lanes, route, &[REVENUE, REGION]).unwrap();

    // Borrowing: the binding hands out the batch's own Arc-backed lanes.
    let rev = binding
        .lane(&store, &view, lanes.frames(), &lanes, REVENUE)
        .unwrap();
    let reg = binding
        .lane(&store, &view, lanes.frames(), &lanes, REGION)
        .unwrap();
    assert_eq!(
        slice_i32(rev).as_ptr() as usize,
        batch.lane_addr(F_REVENUE).unwrap()
    );
    assert_eq!(
        slice_u32(reg).as_ptr() as usize,
        batch.lane_addr(F_REGION).unwrap()
    );

    // Behaviour: DT64 admits each method; its reads lie inside the binding.
    let m1 = MethodRef::of(class(), &test_method(1, &[REVENUE, REGION]));
    let m2 = MethodRef::of(class(), &test_method(2, &[REGION]));
    for m in [&m1, &m2] {
        let admitted = store.admit(m).unwrap();
        assert!(admitted.reads.is_subset_of(binding.fields()));
    }

    // Fold 1 (method 1): Σ revenue where region == 2 ∧ revenue > 100.
    let sum = lower(&Query {
        filter: Filter::and([
            Filter::cmp(Col(1), Cmp::EqU32(2)),
            Filter::cmp(Col(0), Cmp::GtI32(100)),
        ]),
        agg: Agg::SumI32(Col(0)),
    })
    .unwrap();
    // Fold 2 (method 2), same binding: count where region == 3.
    let count = lower(&Query {
        filter: Filter::cmp(Col(1), Cmp::EqU32(3)),
        agg: Agg::Count,
    })
    .unwrap();
    let mut s1 = Scratch::for_program(&sum, N).unwrap();
    let mut s2 = Scratch::for_program(&count, N).unwrap();

    let warm = |s1: &mut Scratch, s2: &mut Scratch| {
        let rev = binding
            .lane(&store, &view, lanes.frames(), &lanes, REVENUE)
            .unwrap();
        let reg = binding
            .lane(&store, &view, lanes.frames(), &lanes, REGION)
            .unwrap();
        let cols = [rev, reg];
        let inputs = Planes {
            n_rows: N,
            masks: &[],
            lanes: &cols,
        };
        let a = execute_into(&sum, &inputs, &Foreign::NONE, s1, Out::None).unwrap();
        let b = execute_into(&count, &inputs, &Foreign::NONE, s2, Out::None).unwrap();
        (a, b)
    };
    let (got_sum, got_count) = warm(&mut s1, &mut s2);

    // Independent scalar oracle over freshly generated values, not the lanes.
    let (want_rev, want_reg) = (revenue(1), region());
    let want_sum: i64 = (0..N)
        .filter(|&i| want_reg[i] == 2 && want_rev[i] > 100)
        .map(|i| i64::from(want_rev[i]))
        .sum();
    let want_count = (0..N).filter(|&i| want_reg[i] == 3).count();
    assert_eq!(got_sum, Value::SumI64(want_sum));
    assert_eq!(got_count, Value::Count(want_count));
    let kept = (0..N)
        .filter(|&i| want_reg[i] == 2 && want_rev[i] > 100)
        .count();
    assert!(kept > 0 && kept < N / 5, "non-trivial filter: {kept}");

    // Reuse: the same binding checks and folds again, with no allocation.
    let before = allocations();
    for _ in 0..3 {
        binding
            .check(&store, &view, lanes.frames(), &lanes)
            .unwrap();
        let (a, b) = warm(&mut s1, &mut s2);
        assert_eq!((a, b), (Value::SumI64(want_sum), Value::Count(want_count)));
    }
    assert_eq!(allocations() - before, 0, "hot path allocated");
}

// ── negative ─────────────────────────────────────────────────────────────────

/// A published replacement batch moves the frame: the old binding refuses,
/// a fresh bind succeeds on the new lanes.
#[test]
fn a_published_replacement_refuses_the_old_binding() {
    let (store, route) = store();
    let view = view();
    let catalog = catalog();
    let mut lanes = AbiLanes::unwitnessed(&catalog, batch(1), READER);
    let old = bind(&store, &view, &lanes, route, &[REVENUE]).unwrap();
    lanes.publish(batch(2)).unwrap();
    assert_eq!(
        old.check(&store, &view, lanes.frames(), &lanes),
        Err(LaneRefusal::Frame(FrameRefusal::CoordinatesChanged))
    );
    assert!(old
        .lane(&store, &view, lanes.frames(), &lanes, REVENUE)
        .is_err());
    let fresh = bind(&store, &view, &lanes, route, &[REVENUE]).unwrap();
    let lane = fresh
        .lane(&store, &view, lanes.frames(), &lanes, REVENUE)
        .unwrap();
    assert_eq!(slice_i32(lane), revenue(2).as_slice());

    // A replacement must be a later generation of the same source.
    assert_eq!(
        lanes.publish(batch(2)).err(),
        Some(PublishRefusal::StaleGeneration {
            current: 2,
            offered: 2
        })
    );
    let foreign = Arc::new(AbiBatch::new(SourceId(8), 9, N));
    assert_eq!(
        lanes.publish(foreign).err(),
        Some(PublishRefusal::ForeignSource {
            bound: SOURCE,
            offered: SourceId(8)
        })
    );
}

/// A frame sealed by another source's store is refused, even with equal
/// revisions.
#[test]
fn a_frame_from_another_store_is_refused() {
    let (store, route) = store();
    let view = view();
    let catalog = catalog();
    let lanes = AbiLanes::unwitnessed(&catalog, batch(1), READER);
    let twin = AbiLanes::unwitnessed(&catalog, batch(1), READER);
    let fields = WideFieldMask::from_positions(&[REVENUE]);
    let foreign = twin.seal();
    assert_eq!(
        store
            .bind_lanes(
                &view,
                lanes.frames(),
                &lanes,
                request(route, &fields, &foreign, lanes.space())
            )
            .err(),
        Some(LaneRefusal::Frame(FrameRefusal::ForeignFrame))
    );
}

/// The position re-routes to another class: the binding refuses.
#[test]
fn a_wrong_route_is_refused() {
    let (mut store, route) = store();
    let view = view();
    let catalog = catalog();
    let lanes = AbiLanes::unwitnessed(&catalog, batch(1), READER);
    let binding = bind(&store, &view, &lanes, route, &[REVENUE]).unwrap();
    store
        .register_derived(
            &view,
            class(),
            other_class(),
            ClassSchema {
                schema_rev: SchemaRev(1),
                field_count: 5,
                extent: Extent::DR64_CUBE,
            },
            Vec::new(),
        )
        .unwrap();
    store.route(POSITION, other_class()).unwrap();
    assert_eq!(
        binding.check(&store, &view, lanes.frames(), &lanes),
        Err(LaneRefusal::RouteChanged {
            position: POSITION,
            bound: class(),
            now: Some(other_class()),
        })
    );
}

/// A selection in another row order (another source, or another generation
/// of this one) is refused: reading through it would be a gather.
#[test]
fn an_incompatible_coordinate_space_is_refused() {
    let (store, route) = store();
    let view = view();
    let catalog = catalog();
    let lanes = AbiLanes::unwitnessed(&catalog, batch(1), READER);
    let fields = WideFieldMask::from_positions(&[REVENUE]);
    let frame = lanes.seal();
    for selection in [
        CoordSpace {
            population: 8,
            coordinates: 1,
        },
        CoordSpace {
            population: 7,
            coordinates: 2,
        },
    ] {
        assert_eq!(
            store
                .bind_lanes(
                    &view,
                    lanes.frames(),
                    &lanes,
                    request(route, &fields, &frame, selection)
                )
                .err(),
            Some(LaneRefusal::CoordinateSpace {
                field: REVENUE,
                selection,
                lane: lanes.space(),
            })
        );
    }
}

/// Unknown name, catalogued-but-absent column, and a U64 reference lane all
/// leave the field unbound; a position outside the basis is refused by DT64.
#[test]
fn unknown_or_unadmitted_fields_are_refused() {
    let (store, route) = store();
    let view = view();
    let catalog = catalog();
    let lanes = AbiLanes::unwitnessed(&catalog, batch(1), READER);
    for field in [NOT_CATALOGUED, NO_COLUMN, REFERENCE] {
        assert_eq!(
            bind(&store, &view, &lanes, route, &[REVENUE, field]).err(),
            Some(LaneRefusal::Unbound {
                class: class(),
                field
            }),
            "field {field}"
        );
    }
    assert_eq!(
        bind(&store, &view, &lanes, route, &[5]).err(),
        Some(LaneRefusal::FieldNotInBasis {
            class: class(),
            field: 5
        })
    );
}

/// A reader in another scope is refused.
#[test]
fn a_foreign_scope_is_refused() {
    let (store, route) = store();
    let view = view();
    let catalog = catalog();
    let lanes = AbiLanes::unwitnessed(&catalog, batch(1), READER);
    let fields = WideFieldMask::from_positions(&[REVENUE]);
    let frame = lanes.seal();
    let mut req = request(route, &fields, &frame, lanes.space());
    req.scope = Scope { id: 42 };
    assert_eq!(
        store.bind_lanes(&view, lanes.frames(), &lanes, req).err(),
        Some(LaneRefusal::Scope {
            field: REVENUE,
            reader: Scope { id: 42 },
            lane: READER,
        })
    );
}

/// After a publish, a selection still in the old generation's row order is
/// refused against a fresh bind.
#[test]
fn the_old_row_order_is_refused_after_a_publish() {
    let (store, route) = store();
    let view = view();
    let catalog = catalog();
    let mut lanes = AbiLanes::unwitnessed(&catalog, batch(1), READER);
    let old_space = lanes.space();
    lanes.publish(batch(2)).unwrap();
    let fields = WideFieldMask::from_positions(&[REVENUE]);
    let frame = lanes.seal();
    assert_eq!(
        store
            .bind_lanes(
                &view,
                lanes.frames(),
                &lanes,
                request(route, &fields, &frame, old_space)
            )
            .err(),
        Some(LaneRefusal::CoordinateSpace {
            field: REVENUE,
            selection: old_space,
            lane: lanes.space(),
        })
    );
}

/// The adapter answers only for frames its own store sealed: another
/// source's store and frame, presented together, pass DT64's frame check
/// but leave every field unbound here.
#[test]
fn a_frame_and_store_of_another_source_bind_nothing() {
    let (store, route) = store();
    let view = view();
    let catalog = catalog();
    let lanes = AbiLanes::unwitnessed(&catalog, batch(1), READER);
    let other = AbiLanes::unwitnessed(&catalog, batch(1), READER);
    let fields = WideFieldMask::from_positions(&[REVENUE]);
    let frame = other.seal();
    assert_eq!(
        store
            .bind_lanes(
                &view,
                other.frames(),
                &lanes,
                request(route, &fields, &frame, lanes.space())
            )
            .err(),
        Some(LaneRefusal::Unbound {
            class: class(),
            field: REVENUE
        })
    );
}

/// A batch holds at most `MAX_COLUMNS` columns, so every lane index the
/// adapter publishes is the column it names (no `u16` wrap onto column 0).
#[test]
fn a_column_past_the_u16_lane_space_is_refused() {
    let mut b = AbiBatch::new(SOURCE, 1, 1);
    for i in 0..MAX_COLUMNS as u32 {
        b = b
            .with_column(Column::value(
                FieldId(i),
                LaneData::I32(vec![i as i32].into()),
            ))
            .unwrap();
    }
    let last = FieldId(MAX_COLUMNS as u32 - 1);
    let (lane, _) = b.column(last).unwrap();
    assert_eq!(usize::from(lane), MAX_COLUMNS - 1);
    assert_eq!(
        b.with_column(Column::value(
            FieldId(MAX_COLUMNS as u32),
            LaneData::I32(vec![0].into())
        ))
        .err(),
        Some(BatchError::TooManyColumns(FieldId(MAX_COLUMNS as u32)))
    );
}

/// DT64 validates ClassView A at bind. Checked later through ClassView B
/// (same class, revision and field count, first two fields swapped), the
/// binding refuses: the source maps the field B now names, which is another
/// lane. Both candidate lanes are nonzero and different. Before D-LANESOCK-3
/// the source read a ClassView of its own and A's revenue field borrowed B's
/// region lane. Since L5a a fresh bind under B is refused as well.
#[test]
fn a_disagreeing_field_basis_never_returns_the_wrong_lane() {
    let (store, route) = store();
    let a = view();
    let mut b = view();
    b.0.swap(usize::from(REVENUE), usize::from(REGION));
    let catalog = catalog();
    let lanes = AbiLanes::unwitnessed(&catalog, batch(1), READER);
    // The two candidate lanes hold different, nonzero values.
    let (rev, reg) = (revenue(1), region());
    assert!(rev.iter().any(|&v| v != 0) && reg.iter().any(|&v| v != 0));
    assert!((0..N).any(|i| i64::from(rev[i]) != i64::from(reg[i])));

    // Bound under A, A's revenue field is the revenue lane.
    let binding = bind(&store, &a, &lanes, route, &[REVENUE]).unwrap();
    let lane = binding
        .lane(&store, &a, lanes.frames(), &lanes, REVENUE)
        .unwrap();
    assert_eq!(slice_i32(lane), revenue(1).as_slice());

    // Through B the same position names region: refused, no lane returned.
    assert_eq!(
        binding.check(&store, &b, lanes.frames(), &lanes),
        Err(LaneRefusal::MappingChanged { field: REVENUE })
    );
    assert_eq!(
        binding
            .lane(&store, &b, lanes.frames(), &lanes, REVENUE)
            .err(),
        Some(LaneRefusal::MappingChanged { field: REVENUE })
    );

    // A fresh bind under B is refused too (D-DT64-7 Decision J, L5a): B has
    // the bound revision and count but another ordered predicate basis, so
    // the class no longer resolves through it. Before L5a this bind succeeded
    // and position 0 borrowed the region lane.
    assert!(matches!(
        bind(&store, &b, &lanes, route, &[REVENUE]),
        Err(LaneRefusal::Bind(
            lance_graph_dt64::Refusal::PredicateBasisChanged { .. }
        ))
    ));
}
