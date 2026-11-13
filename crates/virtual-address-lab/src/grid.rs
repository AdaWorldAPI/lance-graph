//! The fixture source owner: `lanes` lanes of `len` `i32` positions, served
//! through the two EXISTING sockets, `LaneSource` and `SelectionSource`.
//!
//! Its selections are the source's own. The ordered one is a
//! `dr256_ternlog_lab::order_store::SealedOrder` built by
//! `OrderIndex::by_i32` over these lanes; `borrow_positions` lends
//! `SealedOrder::rows`. Nothing here is a second order or position owner
//! beside the source, and no lane is moved.

use dr256_ternlog_lab::order_store::{KeyLane, SealedOrder};
use dr256_ternlog_lab::{OrderIndex, Table};
use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneBinding, LaneRefusal, LaneRequest,
    LaneSource, MethodBinding, RouteAt, SchemaRev, SelAddr, SelectionSource,
};
use lance_graph_mask_risc::LaneRef;

/// DT64 class of the fixture (lab-local; nothing is minted).
pub const CLASS: u32 = 0x0A1A_0003;
/// The behaviour lens of [`CLASS`]: same left half, right half `0xF###`. Its
/// basis is read by its full classid.
pub const BEHAVIOUR: u32 = 0x0A1A_F003;
/// The one-word polyfill lens the fixture's effects name.
pub const POLYFILL: u32 = 0x0A1A_F0A0;
/// DR position the class is routed at.
pub const POSITION: u16 = 11;
/// A consistency value. It carries no authority.
pub const READER: Scope = Scope { id: 3 };
/// The population number of an order index's own row order.
const INDEX_POPULATION: u64 = 0x1D;

fn iri(field: usize) -> String {
    format!("urn:virtual-address-lab/f{field}")
}

enum Positions {
    /// An order index of the lab, sealed against these lanes.
    Order(SealedOrder),
    /// A list the source produced itself (a filter, a bag).
    List(Vec<u32>),
}

struct Issued {
    positions: Positions,
    /// The revisions the selection was produced at.
    at: FrameRevisions,
    /// The row order its positions are in.
    space: CoordSpace,
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
    /// The field names, built once: `lane_of` compares, it does not format.
    iris: Vec<String>,
    selections: Vec<Issued>,
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
            iris: (0..lanes).map(iri).collect(),
            selections: Vec::new(),
        }
    }

    /// `L0 = [1,2,3]`, `L1 = [4,5,6]`, `L2 = [7,8,9]`.
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

    /// One lane with no witness, for oracles.
    pub fn raw_lane(&self, lane: usize) -> &[i32] {
        &self.cells[lane * self.len..(lane + 1) * self.len]
    }

    /// The lane matrix as an owned copy, for the materialising oracle.
    pub fn matrix(&self) -> Vec<Vec<i32>> {
        (0..self.lanes).map(|l| self.raw_lane(l).to_vec()).collect()
    }

    fn issue(&mut self, positions: Positions, space: CoordSpace) -> u16 {
        self.selections.push(Issued {
            positions,
            at: self.frames.revisions(),
            space,
        });
        (self.selections.len() - 1) as u16
    }

    /// The source produces the order of `key`: `OrderIndex::by_i32` over its
    /// own lane, sealed (`SealedOrder::seal` checks every cached key against
    /// the lane). Rows are positions of THIS source's row order.
    pub fn issue_order(&mut self, key: u8) -> Option<u16> {
        let sealed = {
            let lane = [(key, LaneRef::I32(self.raw_lane(usize::from(key))))];
            let table = Table::new(
                CLASS,
                self.space.population,
                self.len,
                self.frames.revisions().source,
                READER,
                &lane,
            );
            let index = OrderIndex::by_i32(&table, key, INDEX_POPULATION)?;
            SealedOrder::seal(index, &table, KeyLane::Field(key), 0).ok()?
        };
        Some(self.issue(Positions::Order(sealed), self.space))
    }

    /// The source produces a filter: the rows where `field` lies in
    /// `lo..=hi`, ascending.
    pub fn issue_filter(&mut self, field: usize, lo: i32, hi: i32) -> u16 {
        let rows = (0..self.len as u32)
            .filter(|&r| (lo..=hi).contains(&self.cell(field, r as usize)))
            .collect();
        self.issue(Positions::List(rows), self.space)
    }

    /// The source issues a list as it is: a bag, in list order.
    pub fn issue_list(&mut self, rows: Vec<u32>) -> u16 {
        self.issue(Positions::List(rows), self.space)
    }

    /// The source issues a list whose positions are in ANOTHER row order.
    pub fn issue_list_in(&mut self, rows: Vec<u32>, space: CoordSpace) -> u16 {
        self.issue(Positions::List(rows), space)
    }

    /// The positions of selection `sel`, with no witness. For oracles and to
    /// compare addresses.
    pub fn raw_positions(&self, sel: u16) -> &[u32] {
        match &self.selections[usize::from(sel)].positions {
            Positions::Order(sealed) => sealed.rows(),
            Positions::List(rows) => rows,
        }
    }

    /// The same lanes under other lane ids.
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

    /// Rewrite lane `lane` WITHOUT moving the frame: what another producer of
    /// that lane would have written.
    pub fn rewrite_lane(&mut self, lane: usize, value: impl Fn(usize) -> i32) {
        for p in 0..self.len {
            self.cells[lane * self.len + p] = value(p);
        }
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
        let lane = self
            .iris
            .iter()
            .position(|i| i.as_str() == basis.predicate_iri.as_str())?;
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

impl SelectionSource for Grid {
    fn selection_of(&self, sel: u16, frame: &Frame) -> Option<SelAddr> {
        self.frames.check(frame).ok()?;
        let issued = self.selections.get(usize::from(sel))?;
        // A selection produced at other revisions is not held under this frame.
        if issued.at != frame.revisions() {
            return None;
        }
        Some(SelAddr {
            space: issued.space,
            sel,
            scope: READER,
        })
    }

    fn borrow_positions(&self, addr: &SelAddr) -> Option<&[u32]> {
        let issued = self.selections.get(usize::from(addr.sel))?;
        if issued.space != addr.space {
            return None;
        }
        Some(match &issued.positions {
            Positions::Order(sealed) => sealed.rows(),
            Positions::List(rows) => rows,
        })
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

    /// A class whose fields are named by the caller, in basis order.
    pub fn named(fields: &[(&str, &str)]) -> View {
        View(
            fields
                .iter()
                .map(|(iri, label)| FieldRef::new(*iri, *label))
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
    fn lens_fields(&self, lens: u32) -> Option<&[FieldRef]> {
        (lens == BEHAVIOUR).then_some(self.0.as_slice())
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
        World::over(View::new(lanes), lanes, methods)
    }

    /// A world over a class with the given named fields.
    pub fn named(fields: &[(&str, &str)]) -> World {
        World::over(View::named(fields), fields.len(), Vec::new())
    }

    /// A world whose data class has no methods and whose behaviour lens
    /// [`BEHAVIOUR`] carries `methods`, over a polyfill lens of one word per
    /// entry of `arity`.
    pub fn with_behaviour(lanes: usize, arity: Vec<u8>, methods: Vec<MethodBinding>) -> World {
        let mut world = World::new(lanes);
        world
            .store
            .declare_algebra(POLYFILL, SchemaRev(1), Extent::DR256_SQ_X64, arity)
            .expect("the polyfill lens binds");
        world
            .store
            .register(
                &world.view,
                BEHAVIOUR,
                ClassSchema {
                    schema_rev: SchemaRev(1),
                    field_count: lanes,
                    extent: Extent::DR256_SQ_X64,
                },
                methods,
            )
            .expect("the behaviour lens registers");
        world
    }

    fn over(view: View, lanes: usize, methods: Vec<MethodBinding>) -> World {
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

    /// Bind every field of any source, expecting row order `selection`.
    pub fn bind_source<S: LaneSource>(
        &self,
        source: &S,
        selection: CoordSpace,
    ) -> Result<LaneBinding, LaneRefusal> {
        let all: Vec<u8> = (0..self.lanes as u8).collect();
        let fields = WideFieldMask::from_positions(&all);
        let frame = source.frames().seal();
        self.store.bind_lanes(
            &self.view,
            source.frames(),
            source,
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
}
