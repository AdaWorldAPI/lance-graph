//! A DT64 source owner and its binding, as in `lance-graph-report`'s
//! `bound_sel` tests: lanes are issued by their source for its own frame, and
//! every read goes through one `LaneBinding`.
#![allow(dead_code)]

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneBinding, LaneRequest, LaneSource,
    RouteAt, SchemaRev,
};
use lance_graph_mask_risc::LaneRef;
use vrl_recipe_probe::Witness;

pub const CLASS: u32 = 0x0C1B_0001;
pub const VIEW_CLASS: ClassId = 0x0C1B;
const POSITION: u16 = 3;
pub const LABELS: [&str; 5] = ["key", "status", "region", "amount", "code"];
/// Field 4 (`code`) is a `u32` lane; the others are `i32`.
pub const CODE: u8 = 4;
const REVS: FrameRevisions = FrameRevisions {
    source: 1,
    relation_map: 1,
    coordinates: 1,
};
const ROWS: CoordSpace = CoordSpace {
    population: 1,
    coordinates: 1,
};
const READER: Scope = Scope { id: 7 };

pub struct View(Vec<FieldRef>);

impl View {
    pub fn new() -> View {
        View(
            LABELS
                .iter()
                .map(|l| FieldRef::new(format!("ord:{l}"), *l))
                .collect(),
        )
    }

    /// A view of `class` with the given field labels, for compile-time tests.
    pub fn with_labels(labels: &[String]) -> View {
        View(
            labels
                .iter()
                .map(|l| FieldRef::new(format!("ord:{l}"), l.as_str()))
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

pub struct Source {
    pub frames: FrameStore,
    pub lanes: [Vec<i32>; 4],
    pub code: Vec<u32>,
}

struct Lcg(u64);
impl Lcg {
    fn below(&mut self, k: i32) -> i32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % k as u64) as i32
    }
}

impl Source {
    pub fn random(n: usize, seed: u64) -> Source {
        let mut l = Lcg(seed);
        let key = (0..n as i32).map(|i| 1000 + i).collect();
        let status = (0..n).map(|_| l.below(4)).collect();
        let region = (0..n).map(|_| l.below(3)).collect();
        let amount = (0..n).map(|_| l.below(500) - 100).collect();
        Source::from_lanes([key, status, region, amount])
    }

    pub fn from_lanes(lanes: [Vec<i32>; 4]) -> Source {
        let n = lanes[0].len();
        Source {
            frames: FrameStore::new(REVS),
            lanes,
            code: (0..n as u32).collect(),
        }
    }

    pub fn i32_lanes(&self) -> Vec<&[i32]> {
        self.lanes.iter().map(Vec::as_slice).collect()
    }

    /// A write that moves the source's frame: every earlier seal is stale.
    pub fn advance(&mut self) {
        self.frames.advance(FrameRevisions {
            source: 2,
            relation_map: 1,
            coordinates: 1,
        });
    }
}

impl LaneSource for Source {
    type Lane<'a> = LaneRef<'a>;

    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        field: u8,
        _basis: &FieldRef,
        frame: &Frame,
    ) -> Option<LaneAddr> {
        self.frames.check(frame).ok()?;
        (class == CLASS && rev == SchemaRev(1) && field <= CODE).then_some(LaneAddr {
            space: ROWS,
            lane: u16::from(field),
            scope: READER,
        })
    }

    fn borrow(&self, addr: &LaneAddr) -> Option<LaneRef<'_>> {
        match addr.lane {
            0..=3 => Some(LaneRef::I32(&self.lanes[usize::from(addr.lane)])),
            4 => Some(LaneRef::U32(&self.code)),
            _ => None,
        }
    }

    fn frames(&self) -> &FrameStore {
        &self.frames
    }
}

pub struct World {
    pub store: BindStore,
    pub view: View,
    pub binding: LaneBinding,
}

/// Register the class and bind `fields` of `source` under its current frame.
pub fn bind(source: &Source, fields: &WideFieldMask) -> World {
    let mut store = BindStore::new();
    let view = View::new();
    store
        .register(
            &view,
            CLASS,
            ClassSchema {
                schema_rev: SchemaRev(1),
                field_count: LABELS.len(),
                extent: Extent::DR64_CUBE,
            },
            Vec::new(),
        )
        .unwrap();
    let version = store.route(POSITION, CLASS).unwrap();
    let frame = source.frames.seal();
    let binding = store
        .bind_lanes(
            &view,
            &source.frames,
            source,
            LaneRequest {
                route: RouteAt {
                    position: POSITION,
                    version,
                },
                rev: SchemaRev(1),
                fields,
                frame: &frame,
                scope: READER,
                selection: ROWS,
            },
        )
        .expect("the fixture binds");
    World {
        store,
        view,
        binding,
    }
}

pub fn witness<'a>(w: &'a World, source: &'a Source) -> Witness<'a, View, Source> {
    Witness {
        binding: &w.binding,
        store: &w.store,
        view: &w.view,
        frames: &source.frames,
        source,
    }
}

/// Every field bound, the usual case.
pub fn all_fields() -> WideFieldMask {
    WideFieldMask::from_positions(&[0, 1, 2, 3, 4])
}
