//! DR: addressing. Positions, tiles, reservations, the route log, the frame
//! witness and the global → local pointer table.
//!
//! "DR routes, DT interprets." This module knows no method, guard or effect;
//! the binding side of the crate reads it, and it reads nothing back. DR's
//! computations (relation composition, `DR256::WHERE`, membership folds) are
//! not here: they are executions, and execution is Quack's (relalg →
//! mask-risc → `ndarray::simd`), referenced from a binding as a polyfill word.
//!
//! Lifted from `dr-math-binding-probe` (D-DRMC-1: [`FrameStore`], [`Frame`],
//! [`Focus`], [`Scope`], [`Guards`], [`AxisKind`]; D-DRCA-1: [`LocalMap`],
//! [`address`], [`coordinates`], [`rotate_yz`]). Two changes against the probe:
//! - The probe's `Axis { class, kind }` is [`ClassAxis`] here, because [`Axis`]
//!   is the reservation axis.
//! - [`LocalMap`] and [`address`] take a tile [`Width`] instead of a fixed 64,
//!   so a DR256 tile is addressed the same way.
//!
//! Nothing here holds a value. A route maps a position to a classid; a tile
//! is a reservation, claimed when a position in it is routed; a local map
//! holds identity references only.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Weak};

use lance_graph_contract::class_view::ClassId;
use lance_graph_contract::morton8x8::Morton8x8;

/// Positions per axis of a DR tile (D-DRF-1 §1): DR64 or DR256.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Width {
    W64,
    W256,
}

impl Width {
    pub const fn positions(self) -> usize {
        match self {
            Width::W64 => 64,
            Width::W256 => 256,
        }
    }
}

/// One axis of a reservation: `tiles` tiles of `tile` positions each.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Axis {
    pub tile: Width,
    pub tiles: u32,
}

impl Axis {
    pub const fn new(tile: Width, tiles: u32) -> Axis {
        Axis { tile, tiles }
    }

    /// Positions reserved on this axis.
    pub const fn reserved(self) -> usize {
        self.tile.positions() * self.tiles as usize
    }

    /// The tile a position falls in.
    pub const fn tile_of(self, position: usize) -> usize {
        position / self.tile.positions()
    }
}

/// What a class reserves, per axis: Identity × Property × Predicate, each a
/// number of DR64 or DR256 tiles (D-DT64-6 A.1: the binding names DR64 or
/// DR256 and states its cost). Thin provisioning: a reservation costs
/// nothing; only the identity tiles a route actually touches are claimed
/// ([`RouteLog::claimed`]), and nothing is ever allocated here (§7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Extent {
    pub identity: Axis,
    pub property: Axis,
    /// The class's predicates: its methods (ON × WHEN × DO registrations).
    pub predicate: Axis,
}

impl Extent {
    /// DR64³: one 64-wide tile per axis.
    pub const DR64_CUBE: Extent = Extent {
        identity: Axis::new(Width::W64, 1),
        property: Axis::new(Width::W64, 1),
        predicate: Axis::new(Width::W64, 1),
    };
    /// DR256² × 64: one 256 × 256 Identity × Property tile under 64
    /// predicates.
    pub const DR256_SQ_X64: Extent = Extent {
        identity: Axis::new(Width::W256, 1),
        property: Axis::new(Width::W256, 1),
        predicate: Axis::new(Width::W64, 1),
    };
    /// The whole DR256 identity space (65,536 positions, 256 tiles of 256)
    /// reserved over a 256-wide property axis and 64 predicates. Reserving
    /// it claims nothing.
    pub const DR256_GLOBAL_X64: Extent = Extent {
        identity: Axis::new(Width::W256, 256),
        property: Axis::new(Width::W256, 1),
        predicate: Axis::new(Width::W64, 1),
    };

    /// Logical cells reserved.
    pub const fn reserved_cells(self) -> usize {
        self.identity.reserved() * self.property.reserved() * self.predicate.reserved()
    }

    /// What one claimed identity tile would cost as a dense Boolean volume,
    /// in bytes: a statement of cost, not an allocation. DR64³ is 32 KiB
    /// (D-DRF-1 §1).
    pub const fn tile_bytes(self) -> usize {
        self.identity.tile.positions() * self.property.reserved() * self.predicate.reserved() / 8
    }
}

/// What a class has claimed of its reservation at one route version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Claim {
    /// Distinct identity tiles holding at least one position routed to the
    /// class.
    pub identity_tiles: usize,
    /// `identity_tiles × Extent::tile_bytes`.
    pub bytes: usize,
}

/// The widest identity or predicate reservation: route positions are `u16`.
pub const MAX_RESERVED: usize = 1 << 16;

/// A position in the append-only route log. `RouteVersion(0)` is the empty
/// log; each route or retract advances it by one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RouteVersion(pub u64);

/// Why the route log refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteRefusal {
    /// A route to a position outside the class's identity reservation.
    OutsideReservation {
        class: u32,
        position: u16,
        reserved: usize,
    },
    /// The version is past the end of the route log.
    FutureVersion {
        asked: RouteVersion,
        current: RouteVersion,
    },
}

/// DR64 direct routing: the append-only log from a DR position to the
/// classid bound there. Only routing is versioned (D-DT64-6 §3); what a
/// classid means is the binding side's, and does not change.
#[derive(Debug, Clone, Default)]
pub struct RouteLog {
    /// Per position, the route events in version order: `(version, class)`,
    /// `None` for a retract.
    routes: BTreeMap<u16, Vec<(RouteVersion, Option<u32>)>>,
    version: RouteVersion,
}

impl RouteLog {
    pub fn new() -> RouteLog {
        RouteLog::default()
    }

    /// The current end of the log.
    pub fn current_version(&self) -> RouteVersion {
        self.version
    }

    /// Route `position` to `class`, whose identity reservation is `identity`;
    /// returns the new version. A position past the reservation is refused
    /// and moves no version.
    pub fn route(
        &mut self,
        position: u16,
        class: u32,
        identity: Axis,
    ) -> Result<RouteVersion, RouteRefusal> {
        let reserved = identity.reserved();
        if usize::from(position) >= reserved {
            return Err(RouteRefusal::OutsideReservation {
                class,
                position,
                reserved,
            });
        }
        Ok(self.push(position, Some(class)))
    }

    /// Remove `position`'s route; returns the new version.
    pub fn retract(&mut self, position: u16) -> RouteVersion {
        self.push(position, None)
    }

    fn push(&mut self, position: u16, class: Option<u32>) -> RouteVersion {
        self.version = RouteVersion(self.version.0 + 1);
        self.routes
            .entry(position)
            .or_default()
            .push((self.version, class));
        self.version
    }

    /// The classid `position` routed to at `version`, `None` if it was
    /// unrouted then.
    pub fn routed_at(
        &self,
        version: RouteVersion,
        position: u16,
    ) -> Result<Option<u32>, RouteRefusal> {
        if version > self.version {
            return Err(RouteRefusal::FutureVersion {
                asked: version,
                current: self.version,
            });
        }
        let Some(events) = self.routes.get(&position) else {
            return Ok(None);
        };
        // Events are in version order; take the last one at or before `version`.
        let seen = events.partition_point(|(v, _)| *v <= version);
        Ok(seen.checked_sub(1).and_then(|i| events[i].1))
    }

    /// What `class` has claimed of its reservation `extent` at `version`: the
    /// identity tiles that hold a position routed to it then. A reservation
    /// alone claims nothing.
    pub fn claimed(
        &self,
        class: u32,
        extent: Extent,
        version: RouteVersion,
    ) -> Result<Claim, RouteRefusal> {
        let mut tiles = BTreeSet::new();
        for &position in self.routes.keys() {
            if self.routed_at(version, position)? == Some(class) {
                tiles.insert(extent.identity.tile_of(usize::from(position)));
            }
        }
        Ok(Claim {
            identity_tiles: tiles.len(),
            bytes: tiles.len() * extent.tile_bytes(),
        })
    }
}

/// What one numeric coordinate means according to the resolved ClassView.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AxisKind {
    Identity,
    Property,
    Community,
    Capability,
}

/// A typed coordinate axis: which class's ClassView, and which kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClassAxis {
    pub class: ClassId,
    pub kind: AxisKind,
}

/// The revisions that identify a reference: which source, which relation
/// map, which coordinate assignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameRevisions {
    pub source: u64,
    pub relation_map: u64,
    pub coordinates: u64,
}

/// Why a frame was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameRefusal {
    /// Sealed by another store, even at equal revisions.
    ForeignFrame,
    SourceChanged,
    MapChanged,
    CoordinatesChanged,
    /// Re-sealed at equal revisions: still a new frame.
    FrameResealed,
}

/// The owner of a frame. Only it issues [`Frame`]s, and a frame compares by
/// identity of that issue, never by value: two stores at the same numbers do
/// not alias, and re-sealing equal numbers is a new frame.
#[derive(Debug)]
pub struct FrameStore {
    identity: Arc<()>,
    current: Arc<FrameRevisions>,
}

impl FrameStore {
    pub fn new(revisions: FrameRevisions) -> Self {
        Self {
            identity: Arc::new(()),
            current: Arc::new(revisions),
        }
    }

    /// Seal the current frame.
    pub fn seal(&self) -> Frame {
        Frame {
            store: Arc::downgrade(&self.identity),
            seal: Arc::downgrade(&self.current),
            at: *self.current,
        }
    }

    /// Move to new revisions; every frame sealed before is stale from now on.
    pub fn advance(&mut self, revisions: FrameRevisions) {
        self.current = Arc::new(revisions);
    }

    pub fn revisions(&self) -> FrameRevisions {
        *self.current
    }

    /// `Ok` iff `frame` is this store's current seal. A frame of another store
    /// is `ForeignFrame` even at equal numbers; a stale frame of this store
    /// names the first revision that moved, or `FrameResealed` if none did.
    pub fn check(&self, frame: &Frame) -> Result<(), FrameRefusal> {
        if !Weak::ptr_eq(&frame.store, &Arc::downgrade(&self.identity)) {
            return Err(FrameRefusal::ForeignFrame);
        }
        if Weak::ptr_eq(&frame.seal, &Arc::downgrade(&self.current)) {
            return Ok(());
        }
        let (then, now) = (frame.at, *self.current);
        Err(if then.source != now.source {
            FrameRefusal::SourceChanged
        } else if then.relation_map != now.relation_map {
            FrameRefusal::MapChanged
        } else if then.coordinates != now.coordinates {
            FrameRefusal::CoordinatesChanged
        } else {
            FrameRefusal::FrameResealed
        })
    }
}

/// A store-issued seal over the reference revisions. No public constructor: a
/// caller cannot build a frame from numbers, only receive one from
/// [`FrameStore::seal`].
#[derive(Clone, Debug)]
pub struct Frame {
    store: Weak<()>,
    seal: Weak<FrameRevisions>,
    at: FrameRevisions,
}

impl Frame {
    /// The revisions this frame was sealed at (for reporting, never identity).
    pub fn revisions(&self) -> FrameRevisions {
        self.at
    }

    /// Same issue of the same store.
    pub fn same_issue(&self, other: &Frame) -> bool {
        Weak::ptr_eq(&self.store, &other.store) && Weak::ptr_eq(&self.seal, &other.seal)
    }
}

/// Which tile of the frame is read. Motion, not reference: a new focus over
/// the same frame is not a new frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Focus {
    pub revision: u64,
}

/// What the reader is authorized to see. Policy, checked as its own guard;
/// never part of frame identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scope {
    pub id: u64,
}

/// What a read was made under.
#[derive(Clone, Debug)]
pub struct Guards {
    pub frame: Frame,
    pub focus: Focus,
    pub scope: Scope,
}

/// Why a guard set was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuardRefusal {
    Frame(FrameRefusal),
    FocusChanged,
    ScopeChanged,
}

impl Guards {
    /// Each guard on its own: the frame against the store that issued it, then
    /// the focus, then the scope.
    pub fn check(&self, store: &FrameStore, now: (Focus, Scope)) -> Result<(), GuardRefusal> {
        store.check(&self.frame).map_err(GuardRefusal::Frame)?;
        if self.focus != now.0 {
            return Err(GuardRefusal::FocusChanged);
        }
        if self.scope != now.1 {
            return Err(GuardRefusal::ScopeChanged);
        }
        Ok(())
    }
}

/// Why a local mapping or an address was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapRefusal {
    /// More identities than the tile holds.
    TooMany,
    /// The same global id offered twice.
    Duplicate,
    /// A global id this map does not hold.
    Unmapped,
    /// A Morton code outside the tile.
    OutOfTile,
}

/// The prepared global → local pointer table for one tile under one frame
/// (EXO-46). It holds identity references and local positions only, never
/// values. Lookups are counted, so a test can show a projection never calls
/// it.
#[derive(Debug)]
pub struct LocalMap {
    width: Width,
    /// local → global (`globals[local]`), in the order the source supplied.
    globals: Vec<u64>,
    /// Globals sorted, with the local position each maps to.
    sorted: Vec<u64>,
    sorted_local: Vec<u8>,
    lookups: Cell<u64>,
}

impl LocalMap {
    /// Prepare the map for one tile of `width`. Refuses more ids than the
    /// tile holds, and duplicates.
    pub fn prepare(width: Width, globals: &[u64]) -> Result<Self, MapRefusal> {
        if globals.len() > width.positions() {
            return Err(MapRefusal::TooMany);
        }
        // `globals.len() <= 256`, so every local position fits a `u8`.
        let mut order: Vec<u8> = (0..globals.len()).map(|l| l as u8).collect();
        order.sort_unstable_by_key(|&l| globals[l as usize]);
        let sorted: Vec<u64> = order.iter().map(|&l| globals[l as usize]).collect();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(MapRefusal::Duplicate);
        }
        Ok(Self {
            width,
            globals: globals.to_vec(),
            sorted,
            sorted_local: order,
            lookups: Cell::new(0),
        })
    }

    pub fn width(&self) -> Width {
        self.width
    }

    /// global → local. Counted.
    pub fn local(&self, global: u64) -> Result<u8, MapRefusal> {
        self.lookups.set(self.lookups.get() + 1);
        self.sorted
            .binary_search(&global)
            .map(|at| self.sorted_local[at])
            .map_err(|_| MapRefusal::Unmapped)
    }

    /// local → global (the inverse). Not counted: it is the pointer table read.
    pub fn global(&self, local: u8) -> Option<u64> {
        self.globals.get(usize::from(local)).copied()
    }

    pub fn len(&self) -> usize {
        self.globals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.globals.is_empty()
    }

    /// How many global → local lookups this map has served.
    pub fn lookups(&self) -> u64 {
        self.lookups.get()
    }

    /// FNV-1a over the local → global table, in local order. A frame states
    /// which coordinate assignment it was sealed under by carrying this value
    /// as its `coordinates` revision. A digest, not an identity: two maps
    /// over the same ids in the same order are the same assignment.
    pub fn digest(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &g in &self.globals {
            for b in g.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }
}

/// The HHTL16 Cartesian address of cell `(x, y)` of one tile of `width`: the
/// contract's [`Morton8x8`] code, `x` the row coordinate, `y` the column.
#[inline]
pub fn address(width: Width, x: u8, y: u8) -> Result<Morton8x8, MapRefusal> {
    if usize::from(x) >= width.positions() || usize::from(y) >= width.positions() {
        return Err(MapRefusal::OutOfTile);
    }
    Ok(Morton8x8::from_xy(x, y))
}

/// The inverse of [`address`]: the `(x, y)` a code addresses in a tile of
/// `width`.
#[inline]
pub fn coordinates(width: Width, code: Morton8x8) -> Result<(u8, u8), MapRefusal> {
    let (x, y) = (code.x(), code.y());
    if usize::from(x) >= width.positions() || usize::from(y) >= width.positions() {
        return Err(MapRefusal::OutOfTile);
    }
    Ok((x, y))
}

/// YZ rotation: swap the two Cartesian axes of a code. An address map; it
/// moves no stored bits, and it is its own inverse.
#[inline]
pub fn rotate_yz(code: Morton8x8) -> Morton8x8 {
    Morton8x8::from_xy(code.y(), code.x())
}
