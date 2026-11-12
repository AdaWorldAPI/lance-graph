//! The witnessed fold allocates nothing once the binding exists: the check,
//! both borrows and the fold (D-DR-SELECT-K-SV-2). Own file: the counting
//! allocator is process-global.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneRequest, LaneSource, RouteAt,
    SchemaRev, SelAddr, SelectionSource,
};
use lance_graph_mask_risc::{LaneRef, Value};
use lance_graph_report::{fold_bound_selection, BoundFold, BoundSelRefusal};

static BYTES: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
}

struct Counting;

// SAFETY: forwards to `System` unchanged; it only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if MEASURING.try_with(Cell::get).unwrap_or(false) {
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: same layout, same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn measure(f: impl FnOnce()) -> usize {
    let before = BYTES.load(Ordering::Relaxed);
    MEASURING.with(|m| m.set(true));
    f();
    MEASURING.with(|m| m.set(false));
    BYTES.load(Ordering::Relaxed) - before
}

const CLASS: u32 = 0x0C1A_0001;
const VIEW_CLASS: ClassId = 0x0C1A;
const SPACE: CoordSpace = CoordSpace {
    population: 1,
    coordinates: 1,
};
const READER: Scope = Scope { id: 7 };

struct View([FieldRef; 1]);

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

struct Source {
    frames: FrameStore,
    values: Vec<i32>,
    positions: Vec<u32>,
}

impl LaneSource for Source {
    type Lane<'a> = LaneRef<'a>;
    fn lane_of(
        &self,
        class: u32,
        _: SchemaRev,
        field: u8,
        _: &FieldRef,
        frame: &Frame,
    ) -> Option<LaneAddr> {
        self.frames.check(frame).ok()?;
        (class == CLASS && field == 0).then_some(LaneAddr {
            space: SPACE,
            lane: 0,
            scope: READER,
        })
    }
    fn borrow(&self, _: &LaneAddr) -> Option<LaneRef<'_>> {
        Some(LaneRef::I32(&self.values))
    }
    fn frames(&self) -> &FrameStore {
        &self.frames
    }
}

impl SelectionSource for Source {
    fn selection_of(&self, sel: u16, frame: &Frame) -> Option<SelAddr> {
        self.frames.check(frame).ok()?;
        (sel == 0).then_some(SelAddr {
            space: SPACE,
            sel,
            scope: READER,
        })
    }
    fn borrow_positions(&self, _: &SelAddr) -> Option<&[u32]> {
        Some(&self.positions)
    }
}

/// FAILS IF: the witness, a borrow or the fold allocates, on the accepting
/// path or on a refusing one; or the counter is inert (last assertion).
#[test]
fn a_thousand_witnessed_folds_allocate_nothing() {
    let mut source = Source {
        frames: FrameStore::new(FrameRevisions {
            source: 1,
            relation_map: 1,
            coordinates: 1,
        }),
        values: (0..10_000).map(|i| i % 97 - 40).collect(),
        positions: (0..10_000u32).filter(|p| p.is_multiple_of(37)).collect(),
    };
    let view = View([FieldRef::new("ord:amount", "Amount")]);
    let mut store = BindStore::new();
    store
        .register(
            &view,
            CLASS,
            ClassSchema {
                schema_rev: SchemaRev(1),
                field_count: 1,
                extent: Extent::DR64_CUBE,
            },
            Vec::new(),
        )
        .unwrap();
    let route = RouteAt {
        position: 5,
        version: store.route(5, CLASS).unwrap(),
    };
    let fields = WideFieldMask::from_positions(&[0]);
    let frame = source.frames.seal();
    let binding = store
        .bind_lanes(
            &view,
            &source.frames,
            &source,
            LaneRequest {
                route,
                rev: SchemaRev(1),
                fields: &fields,
                frame: &frame,
                scope: READER,
                selection: SPACE,
            },
        )
        .unwrap();

    let mut seen = 0i64;
    let bytes = measure(|| {
        for t in 0..1000 {
            for fold in [BoundFold::SumI32, BoundFold::CountGtI32 { t: t % 50 }] {
                match fold_bound_selection(
                    &binding,
                    &store,
                    &view,
                    &source.frames,
                    &source,
                    0,
                    0,
                    fold,
                ) {
                    Ok(Value::SumI64(s)) => seen += s,
                    Ok(Value::Count(c)) => seen += c as i64,
                    other => panic!("{other:?}"),
                }
            }
        }
    });
    assert_eq!(bytes, 0, "witnessed folds allocated {bytes} bytes");
    assert_ne!(seen, 0, "the folds ran");

    // The refusing path: the frame moved.
    source.frames.advance(FrameRevisions {
        source: 2,
        relation_map: 1,
        coordinates: 1,
    });
    let mut refused = 0;
    let bytes = measure(|| {
        for _ in 0..1000 {
            if let Err(BoundSelRefusal::Binding(_)) = fold_bound_selection(
                &binding,
                &store,
                &view,
                &source.frames,
                &source,
                0,
                0,
                BoundFold::SumI32,
            ) {
                refused += 1;
            }
        }
    });
    assert_eq!(bytes, 0, "refusals allocated {bytes} bytes");
    assert_eq!(refused, 1000);

    let fired = measure(|| {
        std::hint::black_box(vec![0u8; 4096]);
    });
    assert!(fired >= 4096, "the counter is inert");
}
