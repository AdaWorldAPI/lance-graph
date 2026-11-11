//! Cost of the hot `LaneBinding::check`, per check and per bound lane.
//!
//! Run: `cargo run --release -p lance-graph-dt64 --example lane_check_cost`
//!
//! The "before" figure (no per-position predicate compare) is obtained by
//! applying the disable edit that turns the guard condition in
//! `LaneBinding::check` into `if false {` and re-running. The numbers are a
//! report, not an assertion: the example does not fail on a threshold.

use std::hint::black_box;
use std::time::Instant;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneBinding, LaneRefusal, LaneRequest,
    LaneSource, RouteAt, SchemaRev,
};

const CLASS: u32 = 0x0C1A_0001;
const VIEW_CLASS: ClassId = 0x0C1A;
const POSITION: u16 = 5;
const FIELDS: usize = 16;
const SPACE: CoordSpace = CoordSpace {
    population: 1,
    coordinates: 1,
};
const READER: Scope = Scope { id: 7 };

struct View(Vec<FieldRef>);

impl View {
    fn sixteen() -> View {
        View(
            (0..FIELDS)
                .map(|i| {
                    FieldRef::new(
                        format!("https://example.org/dir/attribute{i:02}"),
                        format!("Attribute {i:02}"),
                    )
                })
                .collect(),
        )
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

/// Positional source: field `n` lives in lane `n`; the basis is ignored.
struct Source {
    map: [LaneAddr; FIELDS],
    lanes: Vec<i32>,
    frames: FrameStore,
}

impl Source {
    fn new() -> Source {
        let map = core::array::from_fn(|i| LaneAddr {
            space: SPACE,
            lane: i as u16,
            scope: READER,
        });
        Source {
            map,
            lanes: vec![1, 2, 3, 4],
            frames: FrameStore::new(FrameRevisions {
                source: 1,
                relation_map: 1,
                coordinates: 1,
            }),
        }
    }
}

impl LaneSource for Source {
    type Lane<'a> = &'a [i32];
    fn lane_of(
        &self,
        _class: u32,
        _rev: SchemaRev,
        field: u8,
        _basis: &FieldRef,
        _frame: &Frame,
    ) -> Option<LaneAddr> {
        Some(self.map[usize::from(field)])
    }
    fn borrow(&self, _addr: &LaneAddr) -> Option<&[i32]> {
        Some(&self.lanes[..])
    }
    fn frames(&self) -> &FrameStore {
        &self.frames
    }
}

fn bind(
    store: &BindStore,
    view: &View,
    source: &Source,
    frame: &Frame,
    route: RouteAt,
    k: usize,
) -> LaneBinding {
    let positions: Vec<u8> = (0..k as u8).collect();
    let mask = WideFieldMask::from_positions(&positions);
    store
        .bind_lanes(
            view,
            source.frames(),
            source,
            LaneRequest {
                route,
                rev: SchemaRev(1),
                fields: &mask,
                frame,
                scope: READER,
                selection: SPACE,
            },
        )
        .expect("bind")
}

/// Minimum over 7 repetitions of 2,000,000 calls, in nanoseconds per call.
fn time<F: FnMut() -> bool>(mut call: F) -> f64 {
    for _ in 0..100_000 {
        black_box(call());
    }
    let mut best = f64::MAX;
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..2_000_000u32 {
            black_box(call());
        }
        let ns = start.elapsed().as_nanos() as f64 / 2_000_000.0;
        if ns < best {
            best = ns;
        }
    }
    best
}

fn main() {
    let view = View::sixteen();
    let mut store = BindStore::new();
    store
        .register(
            &view,
            CLASS,
            ClassSchema {
                schema_rev: SchemaRev(1),
                field_count: FIELDS,
                extent: Extent::DR64_CUBE,
            },
            Vec::new(),
        )
        .expect("register");
    let version = store.route(POSITION, CLASS).expect("route");
    let route = RouteAt {
        position: POSITION,
        version,
    };
    let source = Source::new();
    let frame = source.frames().seal();

    for k in [1usize, 4, 16] {
        let binding = bind(&store, &view, &source, &frame, route, k);
        assert!(binding
            .check(&store, &view, source.frames(), &source)
            .is_ok());
        let ns = time(|| {
            black_box(&binding)
                .check(
                    black_box(&store),
                    black_box(&view),
                    source.frames(),
                    black_box(&source),
                )
                .is_ok()
        });
        println!(
            "lane_check k={k:<2} ns_per_check={ns:.2}  ns_per_lane={:.2}",
            ns / k as f64
        );
    }

    let binding = bind(&store, &view, &source, &frame, route, 1);
    let mut swapped = View::sixteen();
    swapped.0.swap(0, 1);
    let refused = binding.check(&store, &swapped, source.frames(), &source);
    assert!(matches!(refused, Err(LaneRefusal::PredicateChanged { .. })));
    let ns = time(|| {
        black_box(&binding)
            .check(
                black_box(&store),
                black_box(&swapped),
                source.frames(),
                black_box(&source),
            )
            .is_err()
    });
    println!("lane_check k=1 refused ns_per_check={ns:.2}");
}
