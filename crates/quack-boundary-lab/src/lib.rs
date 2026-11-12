//! LAB: where a Quack-lowered `Program` gets its operands from.
//!
//! A `lance_graph_mask_risc::Program` names lanes and planes by index and
//! carries no source, frame or row order. That is kept: the program is
//! reusable algebra. This lab asks the other question — through which
//! existing seam a consumer establishes that the lanes it hands the executor
//! belong to the source, frame and row order the program was lowered for.
//!
//! Two paths run the same program:
//!
//! - [`unbound`]: the caller hands lanes, planes and foreign planes straight
//!   to `execute_into`. Nothing is checked beyond shape. This is the negative
//!   control: it may return a plausible wrong answer.
//! - [`admitted`]: the lanes come from one [`LaneBinding`] check over the
//!   source (`borrow_all`), and the program's frame is checked against the
//!   same source's store. No mask plane and no foreign plane is admitted,
//!   because no owner issues one through the lane socket; a program that
//!   needs one is [`Refused::Blocked`].
//!
//! Nothing here is an owner, a store or an executor. [`Prepared`] is a
//! lab-local pair of a program and the owner-issued [`Frame`] it was lowered
//! under; production has no such pairing (see the plan's list of unsatisfied
//! contracts).
//!
//! Lab convention: the binding binds every field, and field position `i` is
//! lane index `i` in the planes. A production consumer needs a real
//! field-to-lane-index mapping; none is defined here.

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{Frame, FrameRefusal, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneBinding, LaneRefusal, LaneRequest,
    LaneSource, RouteAt, SchemaRev,
};
use lance_graph_mask_risc::{
    execute_into, ExecError, Foreign, LaneRef, Out, Planes, Program, Scratch, Value,
};

pub mod oracle;

/// DT64 class of the fixture.
pub const CLASS: u32 = 0x0C1A_0002;
/// The same class as the `ClassView` sees it.
pub const VIEW_CLASS: ClassId = 0x0C1A;
/// DR position the class is routed at.
pub const POSITION: u16 = 9;

/// Rows per source: three scheduling tiles, last word partial.
pub const ROWS: usize = 40_003;
/// Distinct `region` values (the group universe).
pub const REGIONS: u32 = 8;
/// Rows of the foreign `partners` table.
pub const PARTNERS: usize = 5_003;

/// Field positions, which are also lane indices (lab convention).
pub const ID: u8 = 0;
pub const AMOUNT: u8 = 1;
pub const QTY: u8 = 2;
pub const REGION: u8 = 3;
pub const PARTNER: u8 = 4;
pub const DISC_VALID: u8 = 5;
pub const DISCOUNT: u8 = 6;
/// Number of fields.
pub const FIELDS: usize = 7;

const IRIS: [&str; FIELDS] = [
    "ord:id",
    "ord:amount",
    "ord:quantity",
    "ord:region",
    "ord:partner",
    "ord:discountValid",
    "ord:discount",
];

// Hand-assigned consistency values. They carry no authority.
pub const READER: Scope = Scope { id: 7 };
pub const OTHER_READER: Scope = Scope { id: 8 };

/// One logical row, independent of where a source stores it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Row {
    pub id: i32,
    pub amount: i32,
    pub qty: i32,
    pub region: u32,
    pub partner: u32,
    /// `None` is SQL NULL.
    pub discount: Option<i32>,
}

/// The logical row with id `id`. Both sources hold exactly these rows.
pub fn logical(id: usize) -> Row {
    let i = id as i64;
    let valid = !id.is_multiple_of(5);
    Row {
        id: id as i32,
        amount: ((i * 37) % 401 - 200) as i32,
        qty: ((i * 11) % 17 + 1) as i32,
        region: ((i * 7) % i64::from(REGIONS)) as u32,
        partner: ((i * 13) % PARTNERS as i64) as u32,
        discount: valid.then_some(((i * 3) % 201) as i32),
    }
}

/// What a NULL discount's value lane holds: stale bytes, never zero. Half of
/// them satisfy `discount > 100`, half do not, so an evaluation that reads
/// them is wrong for a predicate and for its negation.
pub fn stale_discount(id: usize) -> i32 {
    if id.is_multiple_of(10) {
        999
    } else {
        -5
    }
}

/// In which order a source stores the logical rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowOrder {
    /// Row `r` holds id `r`.
    ById,
    /// Row `r` holds id `ROWS - 1 - r`.
    Reversed,
}

/// A source owner: its lanes and the frame store they live with.
pub struct Source {
    frames: FrameStore,
    space: CoordSpace,
    scope: Scope,
    id: Vec<i32>,
    amount: Vec<i32>,
    qty: Vec<i32>,
    region: Vec<u32>,
    partner: Vec<u32>,
    disc_valid: Vec<u32>,
    discount: Vec<i32>,
}

impl Source {
    /// A source holding the `ROWS` logical rows in `order`.
    ///
    /// Both orders get the same `CoordSpace` numbers and the same frame
    /// revisions on purpose: hand-assigned numbers can collide, and the lab
    /// must not depend on them being distinct.
    pub fn new(order: RowOrder) -> Source {
        let ids: Vec<usize> = match order {
            RowOrder::ById => (0..ROWS).collect(),
            RowOrder::Reversed => (0..ROWS).rev().collect(),
        };
        let rows: Vec<Row> = ids.iter().map(|&i| logical(i)).collect();
        Source {
            frames: FrameStore::new(FrameRevisions {
                source: 1,
                relation_map: 1,
                coordinates: 1,
            }),
            space: CoordSpace {
                population: 1,
                coordinates: 1,
            },
            scope: READER,
            id: rows.iter().map(|r| r.id).collect(),
            amount: rows.iter().map(|r| r.amount).collect(),
            qty: rows.iter().map(|r| r.qty).collect(),
            region: rows.iter().map(|r| r.region).collect(),
            partner: rows.iter().map(|r| r.partner).collect(),
            disc_valid: rows
                .iter()
                .map(|r| u32::from(r.discount.is_some()))
                .collect(),
            discount: ids
                .iter()
                .zip(&rows)
                .map(|(&i, r)| r.discount.unwrap_or_else(|| stale_discount(i)))
                .collect(),
        }
    }

    /// The row order this source's lanes are in now.
    pub fn space(&self) -> CoordSpace {
        self.space
    }

    /// The store this source's lanes live with.
    pub fn frame_store(&self) -> &FrameStore {
        &self.frames
    }

    /// Serve the lanes to another reader scope (a moved mapping).
    pub fn serve(&mut self, scope: Scope) {
        self.scope = scope;
    }

    /// A coordinate revision: every lane is rotated left by `k` rows, and the
    /// store moves to a new frame with a new coordinate revision.
    pub fn reorder(&mut self, k: usize) {
        self.id.rotate_left(k);
        self.amount.rotate_left(k);
        self.qty.rotate_left(k);
        self.region.rotate_left(k);
        self.partner.rotate_left(k);
        self.disc_valid.rotate_left(k);
        self.discount.rotate_left(k);
        self.space.coordinates += 1;
        let mut revs = self.frames.revisions();
        revs.coordinates += 1;
        self.frames.advance(revs);
    }

    /// The logical row stored at row ordinal `r`.
    pub fn row(&self, r: usize) -> Row {
        Row {
            id: self.id[r],
            amount: self.amount[r],
            qty: self.qty[r],
            region: self.region[r],
            partner: self.partner[r],
            discount: (self.disc_valid[r] == 1).then_some(self.discount[r]),
        }
    }

    /// The raw lanes, in lane-index order, with no witness. This is what a
    /// caller of [`unbound`] holds.
    pub fn raw_lanes(&self) -> [LaneRef<'_>; FIELDS] {
        [
            LaneRef::I32(&self.id),
            LaneRef::I32(&self.amount),
            LaneRef::I32(&self.qty),
            LaneRef::U32(&self.region),
            LaneRef::U32(&self.partner),
            LaneRef::U32(&self.disc_valid),
            LaneRef::I32(&self.discount),
        ]
    }

    /// The discount validity as a mask plane, built by the caller from the
    /// validity lane. It has no witness.
    pub fn validity_plane(&self) -> Vec<u64> {
        let mut words = vec![0u64; ROWS.div_ceil(64)];
        for (r, &v) in self.disc_valid.iter().enumerate() {
            if v == 1 {
                words[r / 64] |= 1u64 << (r % 64);
            }
        }
        words
    }

    /// Start address and length of every lane, for before/after comparison.
    pub fn lane_addresses(&self) -> [(usize, usize); FIELDS] {
        [
            (self.id.as_ptr() as usize, self.id.len()),
            (self.amount.as_ptr() as usize, self.amount.len()),
            (self.qty.as_ptr() as usize, self.qty.len()),
            (self.region.as_ptr() as usize, self.region.len()),
            (self.partner.as_ptr() as usize, self.partner.len()),
            (self.disc_valid.as_ptr() as usize, self.disc_valid.len()),
            (self.discount.as_ptr() as usize, self.discount.len()),
        ]
    }

    /// An order-sensitive checksum over every lane's values.
    pub fn checksum(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0x0100_0000_01b3);
        };
        for r in 0..ROWS {
            mix(self.id[r] as u32 as u64);
            mix(self.amount[r] as u32 as u64);
            mix(self.qty[r] as u32 as u64);
            mix(u64::from(self.region[r]));
            mix(u64::from(self.partner[r]));
            mix(u64::from(self.disc_valid[r]));
            mix(self.discount[r] as u32 as u64);
        }
        h
    }
}

impl LaneSource for Source {
    type Lane<'a> = LaneRef<'a>;

    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        _field: u8,
        basis: &FieldRef,
        frame: &Frame,
    ) -> Option<LaneAddr> {
        // Only this source's current frame addresses its lanes.
        self.frames.check(frame).ok()?;
        if class != CLASS || rev != SchemaRev(1) {
            return None;
        }
        // Mapped by the field DT64 resolved, never by its position.
        let lane = IRIS
            .iter()
            .position(|iri| *iri == basis.predicate_iri.as_str())?;
        Some(LaneAddr {
            space: self.space,
            lane: lane as u16,
            scope: self.scope,
        })
    }

    fn borrow(&self, addr: &LaneAddr) -> Option<LaneRef<'_>> {
        if addr.space != self.space {
            return None;
        }
        self.raw_lanes().get(usize::from(addr.lane)).copied()
    }

    fn frames(&self) -> &FrameStore {
        &self.frames
    }
}

/// The fixture's `ClassView`: the seven fields in a given order.
pub struct View(Vec<FieldRef>);

impl View {
    /// The fields in lane order.
    pub fn standard() -> View {
        View(IRIS.iter().map(|iri| FieldRef::new(*iri, *iri)).collect())
    }

    /// The same fields with positions `a` and `b` exchanged.
    pub fn swapped(a: u8, b: u8) -> View {
        let mut v = View::standard();
        v.0.swap(usize::from(a), usize::from(b));
        v
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

/// The DT64 side: the bind store, the class view it was registered with, and
/// the route the class sits at.
pub struct World {
    pub store: BindStore,
    pub view: View,
    pub route: RouteAt,
}

impl World {
    pub fn new() -> World {
        let mut store = BindStore::new();
        let view = View::standard();
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
            .expect("the fixture class registers");
        let version = store.route(POSITION, CLASS).expect("the fixture routes");
        World {
            store,
            view,
            route: RouteAt {
                position: POSITION,
                version,
            },
        }
    }

    /// Bind every field of `source` under its current frame, for `scope`,
    /// expecting lanes in row order `selection`.
    pub fn bind_as(
        &self,
        source: &Source,
        scope: Scope,
        selection: CoordSpace,
    ) -> Result<LaneBinding, LaneRefusal> {
        let all: Vec<u8> = (0..FIELDS as u8).collect();
        let fields = WideFieldMask::from_positions(&all);
        let frame = source.frames.seal();
        self.store.bind_lanes(
            &self.view,
            &source.frames,
            source,
            LaneRequest {
                route: self.route,
                rev: SchemaRev(1),
                fields: &fields,
                frame: &frame,
                scope,
                selection,
            },
        )
    }

    /// [`World::bind_as`] for the fixture reader in the source's own order.
    pub fn bind(&self, source: &Source) -> Result<LaneBinding, LaneRefusal> {
        self.bind_as(source, READER, source.space())
    }
}

impl Default for World {
    fn default() -> World {
        World::new()
    }
}

/// A program and the owner-issued frame it was lowered under. Lab-local.
///
/// The frame has no public constructor: it comes from the source's own
/// `FrameStore::seal`. The program itself is unchanged and carries nothing.
#[derive(Debug, Clone)]
pub struct Prepared {
    pub program: Program,
    pub frame: Frame,
}

impl Prepared {
    /// Pair `program` with `source`'s current frame.
    pub fn under(program: Program, source: &Source) -> Prepared {
        Prepared {
            program,
            frame: source.frames.seal(),
        }
    }
}

/// Why the admitted path did not execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refused {
    /// The lane binding's joint check failed. Nothing was borrowed.
    Binding(LaneRefusal),
    /// The program was lowered under a frame that is not the source's
    /// current one.
    Program(FrameRefusal),
    /// The program needs an operand no owner issues through the lane socket.
    /// Not executed; no witness is invented for it.
    Blocked(&'static str),
    /// The witness held and the executor refused the operands.
    Exec(ExecError),
}

/// Run `p` over lanes taken from `source` under `binding`'s one check.
///
/// Order: the binding (frame against the source's own store, route, basis,
/// every bound lane, scope, row order), then the program's frame against the
/// same store, then execution with no mask plane and no foreign table.
pub fn admitted(
    p: &Prepared,
    world: &World,
    view: &View,
    binding: &LaneBinding,
    source: &Source,
    scratch: &mut Scratch<'_>,
    out: Out<'_>,
) -> Result<Value, Refused> {
    let mut lanes = [LaneRef::U32(&[]); FIELDS];
    let mut bound = 0usize;
    for item in binding
        .borrow_all(&world.store, view, source.frames(), source)
        .map_err(Refused::Binding)?
    {
        let (field, lane) = item.map_err(Refused::Binding)?;
        lanes[usize::from(field)] = lane;
        bound += 1;
    }
    if bound != FIELDS {
        return Err(Refused::Blocked("the binding does not cover every lane"));
    }
    source.frames().check(&p.frame).map_err(Refused::Program)?;
    let planes = Planes {
        n_rows: lanes[0].len(),
        masks: &[],
        lanes: &lanes,
    };
    execute_into(&p.program, &planes, &Foreign::NONE, scratch, out).map_err(|e| match e {
        ExecError::PlaneOutOfRange(_) => {
            Refused::Blocked("mask plane: the lane socket issues lanes, not planes")
        }
        ExecError::ForeignOutOfRange(_) => {
            Refused::Blocked("foreign plane: no owner-issued residency admission")
        }
        ExecError::ForeignLaneOutOfRange(_) => {
            Refused::Blocked("foreign lane: no admission of the fk's coordinate domain")
        }
        other => Refused::Exec(other),
    })
}

/// Run `program` over whatever the caller hands in. The negative control:
/// shape is validated, origin is not.
pub fn unbound(
    program: &Program,
    lanes: &[LaneRef<'_>],
    masks: &[&[u64]],
    foreign: &Foreign<'_>,
    scratch: &mut Scratch<'_>,
    out: Out<'_>,
) -> Result<Value, ExecError> {
    let planes = Planes {
        n_rows: lanes.first().map_or(0, LaneRef::len),
        masks,
        lanes,
    };
    execute_into(program, &planes, foreign, scratch, out)
}
