//! The source owner: `lanes` lanes of `len` `i32` positions in one buffer,
//! served one lane per field through the DT64 lane socket.

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneBinding, LaneRefusal, LaneRequest,
    LaneSource, MethodBinding, RouteAt, SchemaRev,
};
use lance_graph_mask_risc::{LaneRef, StridedRef};

/// DT64 class of the fixture (lab-local; nothing is minted).
pub const CLASS: u32 = 0x0A1A_0003;
/// DR position the class is routed at.
pub const POSITION: u16 = 11;
/// A consistency value. It carries no authority.
pub const READER: Scope = Scope { id: 3 };

fn iri(field: usize) -> String {
    format!("urn:virtual-address-lab/f{field}")
}

/// A source owner.
pub struct Grid {
    lanes: usize,
    len: usize,
    /// Lane-major: lane `l` is `cells[l * len..(l + 1) * len]`.
    cells: Vec<i32>,
    frames: FrameStore,
    space: CoordSpace,
    /// `phys[l]` is the source's own lane id of field `l`.
    phys: Vec<u16>,
}

impl Grid {
    /// `lanes` lanes of `len` positions, cell `(lane, pos)` holding
    /// `value(lane, pos)`.
    pub fn new(lanes: usize, len: usize, value: impl Fn(usize, usize) -> i32) -> Grid {
        let mut cells = Vec::with_capacity(lanes * len);
        for l in 0..lanes {
            cells.extend((0..len).map(|p| value(l, p)));
        }
        Grid {
            lanes,
            len,
            cells,
            frames: FrameStore::new(FrameRevisions {
                source: 1,
                relation_map: 1,
                coordinates: 1,
            }),
            space: CoordSpace {
                population: 1,
                coordinates: 1,
            },
            phys: (0..lanes as u16).collect(),
        }
    }

    /// The brief's fixture: `L0 = [1,2,3]`, `L1 = [4,5,6]`, `L2 = [7,8,9]`.
    pub fn three() -> Grid {
        Grid::new(3, 3, |l, p| (l * 3 + p + 1) as i32)
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Bytes the source owns.
    pub fn source_bytes(&self) -> usize {
        self.cells.len() * 4
    }

    pub fn space(&self) -> CoordSpace {
        self.space
    }

    pub fn frame_store(&self) -> &FrameStore {
        &self.frames
    }

    /// The physical cell, for oracles.
    pub fn cell(&self, lane: usize, pos: usize) -> i32 {
        self.cells[lane * self.len + pos]
    }

    /// One lane, with no witness. For oracles and unbound controls.
    pub fn raw_lane(&self, lane: usize) -> &[i32] {
        &self.cells[lane * self.len..(lane + 1) * self.len]
    }

    /// Address of the buffer, to show it did not move.
    pub fn buffer_address(&self) -> usize {
        self.cells.as_ptr() as usize
    }

    /// Position `pos` of every lane as ONE strided operand over the source's
    /// own bytes. No binding issues this: the lane socket hands out one lane
    /// per field and says nothing about lanes sharing a buffer.
    pub fn across(&self, pos: usize) -> LaneRef<'_> {
        LaneRef::Strided(StridedRef {
            bytes: bytemuck::cast_slice(&self.cells),
            first_offset: pos * 4,
            stride: self.len * 4,
            records: self.lanes,
        })
    }

    /// The same lanes under other lane ids. Values and order are unchanged.
    pub fn with_lane_ids(mut self, phys: Vec<u16>) -> Grid {
        assert_eq!(phys.len(), self.lanes);
        self.phys = phys;
        self
    }

    /// The source declares a new storage revision.
    pub fn advance_source(&mut self) {
        let mut revs = self.frames.revisions();
        revs.source += 1;
        self.frames.advance(revs);
    }

    /// A coordinate revision: every lane rotated left by `k`, a new row order
    /// and a new frame.
    pub fn reorder(&mut self, k: usize) {
        for l in 0..self.lanes {
            self.cells[l * self.len..(l + 1) * self.len].rotate_left(k);
        }
        self.space.coordinates += 1;
        let mut revs = self.frames.revisions();
        revs.coordinates += 1;
        self.frames.advance(revs);
    }
}

impl LaneSource for Grid {
    type Lane<'a> = LaneRef<'a>;

    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        _field: u8,
        basis: &FieldRef,
        frame: &Frame,
    ) -> Option<LaneAddr> {
        self.frames.check(frame).ok()?;
        if class != CLASS || rev != SchemaRev(1) {
            return None;
        }
        // Mapped by the field DT64 resolved, never by its position.
        let lane = (0..self.lanes).find(|&l| iri(l) == basis.predicate_iri.as_str())?;
        Some(LaneAddr {
            space: self.space,
            lane: self.phys[lane],
            scope: READER,
        })
    }

    fn borrow(&self, addr: &LaneAddr) -> Option<LaneRef<'_>> {
        if addr.space != self.space {
            return None;
        }
        let lane = self.phys.iter().position(|&id| id == addr.lane)?;
        Some(LaneRef::I32(self.raw_lane(lane)))
    }

    fn frames(&self) -> &FrameStore {
        &self.frames
    }
}

/// The class view: one field per lane.
pub struct View(Vec<FieldRef>);

impl View {
    pub fn new(lanes: usize) -> View {
        View(
            (0..lanes)
                .map(|l| FieldRef::new(iri(l), format!("f{l}")))
                .collect(),
        )
    }
}

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if lance_graph_dt64::view_class(CLASS) == class {
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

/// The DT64 side.
pub struct World {
    pub store: BindStore,
    pub view: View,
    pub route: RouteAt,
    lanes: usize,
}

impl World {
    pub fn new(lanes: usize) -> World {
        World::with(lanes, Vec::new())
    }

    /// A world whose class has `lanes` fields and `methods`.
    pub fn with(lanes: usize, methods: Vec<MethodBinding>) -> World {
        let view = View::new(lanes);
        let mut store = BindStore::new();
        store
            .register(
                &view,
                CLASS,
                ClassSchema {
                    schema_rev: SchemaRev(1),
                    field_count: lanes,
                    extent: Extent::DR256_SQ_X64,
                },
                methods,
            )
            .expect("the fixture class registers");
        let version = store.route(POSITION, CLASS).expect("the fixture routes");
        World {
            store,
            view,
            route: RouteAt {
                position: POSITION,
                version,
            },
            lanes,
        }
    }

    /// Bind every field of `grid`, expecting row order `selection`.
    pub fn bind_as(&self, grid: &Grid, selection: CoordSpace) -> Result<LaneBinding, LaneRefusal> {
        let all: Vec<u8> = (0..self.lanes as u8).collect();
        let fields = WideFieldMask::from_positions(&all);
        let frame = grid.frames.seal();
        self.store.bind_lanes(
            &self.view,
            &grid.frames,
            grid,
            LaneRequest {
                route: self.route,
                rev: SchemaRev(1),
                fields: &fields,
                frame: &frame,
                scope: READER,
                selection,
            },
        )
    }

    pub fn bind(&self, grid: &Grid) -> Result<LaneBinding, LaneRefusal> {
        self.bind_as(grid, grid.space())
    }
}
