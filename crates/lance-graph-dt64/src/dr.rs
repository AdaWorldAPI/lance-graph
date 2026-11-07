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
use std::sync::atomic::{AtomicU64, Ordering};
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

    /// The axis a selection of `survivors` positions needs (D-DYNBAG §4.1):
    /// up to 64 fit one DR64 tile; 65 to 256 one DR256 tile; more take
    /// `ceil(n / 256)` DR256 tiles; none takes one DR64 tile. The width is a
    /// cost decision: a DR64 register is 512 B and a DR256 one 8 KiB, so the
    /// narrow tile is used whenever the selection fits it. `tiles` saturates
    /// at `u32::MAX` rather than overflow.
    pub const fn for_survivors(survivors: usize) -> Axis {
        if survivors <= 64 {
            Axis::new(Width::W64, 1)
        } else if survivors <= 256 {
            Axis::new(Width::W256, 1)
        } else {
            let tiles = survivors.div_ceil(256);
            let tiles = if tiles > u32::MAX as usize {
                u32::MAX
            } else {
                tiles as u32
            };
            Axis::new(Width::W256, tiles)
        }
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

/// The one counter behind [`RouteVersion`]. A [`RouteLog`], a [`FrameStore`]
/// and a [`MapTable`] built from clones of the same clock advance ONE
/// sequence: a version then names a single point of the DR log, and a frame's
/// version answers `RouteLog::routed_at` as well as `MapTable::map_at`.
///
/// `clone` SHARES the counter (that is how two structures are joined);
/// [`VersionClock::fork`] copies its present value into an independent one.
/// The counter only ever increases; nothing rewinds it.
#[derive(Debug, Default, Clone)]
pub struct VersionClock(Arc<AtomicU64>);

impl VersionClock {
    pub fn new() -> VersionClock {
        VersionClock::default()
    }

    /// The last version issued (`RouteVersion(0)` before any).
    pub fn current(&self) -> RouteVersion {
        RouteVersion(self.0.load(Ordering::SeqCst))
    }

    /// Issue the next version.
    pub fn tick(&self) -> RouteVersion {
        RouteVersion(self.0.fetch_add(1, Ordering::SeqCst) + 1)
    }

    /// An independent clock starting at this one's present value.
    pub fn fork(&self) -> VersionClock {
        VersionClock(Arc::new(AtomicU64::new(self.0.load(Ordering::SeqCst))))
    }
}

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
#[derive(Debug, Default)]
pub struct RouteLog {
    /// Per position, the route events in version order: `(version, class)`,
    /// `None` for a retract.
    routes: BTreeMap<u16, Vec<(RouteVersion, Option<u32>)>>,
    clock: VersionClock,
}

/// A cloned log gets a forked clock: copying a log must not make the copy's
/// routes advance the original's version sequence.
impl Clone for RouteLog {
    fn clone(&self) -> RouteLog {
        RouteLog {
            routes: self.routes.clone(),
            clock: self.clock.fork(),
        }
    }
}

impl RouteLog {
    pub fn new() -> RouteLog {
        RouteLog::default()
    }

    /// A log that advances `clock`, shared with whoever else holds a clone of
    /// it (typically a [`MapTable`]).
    pub fn with_clock(clock: VersionClock) -> RouteLog {
        RouteLog {
            routes: BTreeMap::new(),
            clock,
        }
    }

    /// The clock this log advances (a clone that shares the counter).
    pub fn clock(&self) -> VersionClock {
        self.clock.clone()
    }

    /// The current end of the (possibly shared) version sequence.
    pub fn current_version(&self) -> RouteVersion {
        self.clock.current()
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
        self.check(position, class, identity)?;
        Ok(self.push(position, Some(class)))
    }

    /// Whether [`RouteLog::route`] would accept routing `position` to
    /// `class`, without routing it: the same refusal, no event, no version.
    /// `route` runs this exact check, so the two cannot disagree.
    pub fn check(&self, position: u16, class: u32, identity: Axis) -> Result<(), RouteRefusal> {
        let reserved = identity.reserved();
        if usize::from(position) >= reserved {
            return Err(RouteRefusal::OutsideReservation {
                class,
                position,
                reserved,
            });
        }
        Ok(())
    }

    /// Remove `position`'s route; returns the new version.
    pub fn retract(&mut self, position: u16) -> RouteVersion {
        self.push(position, None)
    }

    fn push(&mut self, position: u16, class: Option<u32>) -> RouteVersion {
        let version = self.clock.tick();
        self.routes
            .entry(position)
            .or_default()
            .push((version, class));
        version
    }

    /// The classid `position` routed to at `version`, `None` if it was
    /// unrouted then.
    pub fn routed_at(
        &self,
        version: RouteVersion,
        position: u16,
    ) -> Result<Option<u32>, RouteRefusal> {
        let current = self.clock.current();
        if version > current {
            return Err(RouteRefusal::FutureVersion {
                asked: version,
                current,
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
    clock: VersionClock,
    /// The version the current seal was issued at.
    version: RouteVersion,
}

impl FrameStore {
    pub fn new(revisions: FrameRevisions) -> Self {
        Self::with_clock(revisions, VersionClock::new())
    }

    /// A store whose advances tick `clock`.
    pub fn with_clock(revisions: FrameRevisions, clock: VersionClock) -> Self {
        let version = clock.current();
        Self {
            identity: Arc::new(()),
            current: Arc::new(revisions),
            clock,
            version,
        }
    }

    /// Seal the current frame, stamped with the version it was issued at.
    pub fn seal(&self) -> Frame {
        Frame {
            store: Arc::downgrade(&self.identity),
            seal: Arc::downgrade(&self.current),
            at: *self.current,
            version: self.version,
        }
    }

    /// Move to new revisions and a new version; `check` refuses every frame
    /// sealed before (a version-keyed reader such as [`MapTable::map`] still
    /// answers them, at their own version). Returns the new version.
    pub fn advance(&mut self, revisions: FrameRevisions) -> RouteVersion {
        let version = self.clock.tick();
        self.advance_at(revisions, version);
        version
    }

    fn advance_at(&mut self, revisions: FrameRevisions, version: RouteVersion) {
        self.current = Arc::new(revisions);
        self.version = version;
    }

    /// The version `frame` was issued at, if this store issued it. A frame of
    /// another store is `ForeignFrame` even at equal revisions and versions.
    /// Unlike [`FrameStore::check`] this does not require the frame to be
    /// current: it is the key for reading history.
    pub fn issued(&self, frame: &Frame) -> Result<RouteVersion, FrameRefusal> {
        if !Weak::ptr_eq(&frame.store, &Arc::downgrade(&self.identity)) {
            return Err(FrameRefusal::ForeignFrame);
        }
        Ok(frame.version)
    }

    pub fn revisions(&self) -> FrameRevisions {
        *self.current
    }

    /// Whether `other` is this same store (identity, never value): the
    /// lane-socket gate (D-LANESOCK) uses it to refuse a caller-chosen store
    /// that is not the one a source's data lives with.
    pub(crate) fn same_store(&self, other: &FrameStore) -> bool {
        Arc::ptr_eq(&self.identity, &other.identity)
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
    version: RouteVersion,
}

impl Frame {
    /// The version this frame was issued at: the key it reads history under.
    pub fn version(&self) -> RouteVersion {
        self.version
    }

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

    /// This map, whole, as a [`TileMap`]: what a table would read for it
    /// once it is appended. Lets a caller validate a map it has not yet
    /// handed to [`MapTable::advance`] (an extension check, a lookup)
    /// without advancing the table or issuing a version.
    pub fn view(&self) -> TileMap<'_> {
        TileMap {
            map: self,
            len: self.globals.len(),
        }
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
        fnv_ids(&self.globals)
    }
}

/// FNV-1a over a local → global table, in local order.
fn fnv_ids(globals: &[u64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &g in globals {
        for b in g.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// A tile's name in a [`MapTable`], chosen by the caller.
pub type TileId = u32;

/// Why a [`MapTable`] read was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapRead {
    /// The frame was not issued by this table's store (`ForeignFrame`). A
    /// frame of THIS table is never stale: it reads the layout it was sealed
    /// under.
    Stale(FrameRefusal),
    /// The table holds no map for this tile at the frame's (or asked) version.
    NoSuchTile(TileId),
    /// A numeric version past the end of the shared version sequence.
    FutureVersion {
        asked: RouteVersion,
        current: RouteVersion,
    },
}

/// A tile's history: append-only. Nothing here is updated in place or
/// compacted, so any past version stays answerable.
///
/// - an EXTENSION (the new map keeps every old position and only appends) is a
///   longer length at a new version over the SAME generation: the generation
///   is replaced by its superset (the old prefix is bit-identical inside it,
///   so no information is lost) and no copy is kept;
/// - a REPLACEMENT (reorder, removal of an id, width change, or a re-add after
///   a removal) is a new generation;
/// - a REMOVAL is an event to `None`.
#[derive(Debug, Default)]
struct TileLog {
    gens: Vec<LocalMap>,
    /// `(version, Some((generation, len)))`, `None` = removed; version order.
    events: Vec<(RouteVersion, Option<(usize, usize)>)>,
}

impl TileLog {
    fn append(&mut self, version: RouteVersion, m: LocalMap) {
        let len = m.len();
        let live = match self.events.last() {
            Some((_, Some((g, _)))) => Some(*g),
            _ => None,
        };
        if let Some(g) = live {
            if let Some(slot) = self.gens.get_mut(g) {
                if m.extends(slot) {
                    m.lookups.set(slot.lookups.get());
                    *slot = m;
                    self.events.push((version, Some((g, len))));
                    return;
                }
            }
        }
        let g = self.gens.len();
        self.gens.push(m);
        self.events.push((version, Some((g, len))));
    }

    fn remove(&mut self, version: RouteVersion) {
        if matches!(self.events.last(), Some((_, Some(_)))) {
            self.events.push((version, None));
        }
    }

    /// The layout in force at `version`: the last event at or before it.
    fn at(&self, version: RouteVersion) -> Option<TileMap<'_>> {
        let seen = self.events.partition_point(|(v, _)| *v <= version);
        let (g, len) = self.events.get(seen.checked_sub(1)?)?.1?;
        Some(TileMap {
            map: self.gens.get(g)?,
            len,
        })
    }
}

/// A borrowed view of one tile's map at one version: a generation plus a
/// length. It owns no ids and allocates nothing. `local` and `global` respect
/// `len`, so an id appended after the version is absent at it.
#[derive(Clone, Copy, Debug)]
pub struct TileMap<'t> {
    map: &'t LocalMap,
    len: usize,
}

impl<'t> TileMap<'t> {
    pub fn width(&self) -> Width {
        self.map.width
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// global → local. Counted (on the underlying generation). A position at
    /// or past `len` was appended later, so it is `Unmapped` here.
    pub fn local(&self, global: u64) -> Result<u8, MapRefusal> {
        let local = self.map.local(global)?;
        if usize::from(local) < self.len {
            Ok(local)
        } else {
            Err(MapRefusal::Unmapped)
        }
    }

    /// local → global, `None` at or past `len`.
    pub fn global(&self, local: u8) -> Option<u64> {
        if usize::from(local) < self.len {
            self.map.global(local)
        } else {
            None
        }
    }

    /// The local → global table at this version, in local order.
    pub fn globals(&self) -> &'t [u64] {
        self.map.globals.get(..self.len).unwrap_or(&[])
    }

    /// How many lookups the underlying generation has served.
    pub fn lookups(&self) -> u64 {
        self.map.lookups()
    }

    /// FNV-1a over the local → global table at this version.
    pub fn digest(&self) -> u64 {
        fnv_ids(self.globals())
    }

    /// True iff `older` is a pure prefix of `self`: same width, and every
    /// position of `older` keeps its id here, with `self` only appending.
    pub fn extends(&self, older: TileMap<'_>) -> bool {
        self.map.width == older.map.width
            && self.len >= older.len
            && self.globals().get(..older.len) == Some(older.globals())
    }
}

/// The versioned global → local table (D-DT64-7 O5): every tile's history of
/// [`LocalMap`]s under one [`FrameStore`]. Maps change only through
/// [`MapTable::advance`], which appends: an extension is a length at a
/// version, a replacement a new generation, a removal an event; nothing is
/// updated in place or compacted. A frame carries the version it was sealed
/// at ([`RouteVersion`], one per `advance`, shared with a [`RouteLog`] when
/// they share a [`VersionClock`]) and reads the layout in force then, so an
/// old frame replays the old layout instead of being refused. The
/// `coordinates` revision is still set from the CURRENT state, for reporting
/// and for [`FrameStore::check`]'s currency test; it cannot key history (a
/// digest collides and is not monotone), the version does.
#[derive(Debug)]
pub struct MapTable {
    frames: FrameStore,
    tiles: BTreeMap<TileId, TileLog>,
}

impl MapTable {
    /// An empty table at the given source and relation-map revisions, with
    /// its own version clock.
    pub fn new(source: u64, relation_map: u64) -> Self {
        Self::with_clock(source, relation_map, VersionClock::new())
    }

    /// An empty table whose advances tick `clock` (shared with a
    /// [`RouteLog`] to make both one DR log).
    pub fn with_clock(source: u64, relation_map: u64, clock: VersionClock) -> Self {
        let tiles = BTreeMap::new();
        let coordinates = Self::coordinates(&tiles);
        Self {
            frames: FrameStore::with_clock(
                FrameRevisions {
                    source,
                    relation_map,
                    coordinates,
                },
                clock,
            ),
            tiles,
        }
    }

    /// The clock this table advances (a clone that shares the counter).
    pub fn clock(&self) -> VersionClock {
        self.frames.clock.clone()
    }

    /// The end of the (possibly shared) version sequence.
    pub fn current_version(&self) -> RouteVersion {
        self.frames.clock.current()
    }

    /// FNV-1a over `(tile, map digest)` in tile order, over the CURRENT state
    /// (removed tiles are absent).
    fn coordinates(tiles: &BTreeMap<TileId, TileLog>) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for (tile, log) in tiles {
            let Some(view) = log.at(RouteVersion(u64::MAX)) else {
                continue;
            };
            let bytes = tile
                .to_le_bytes()
                .into_iter()
                .chain(view.digest().to_le_bytes());
            for b in bytes {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }

    /// Insert or replace (`Some`) or remove (`None`) tile maps, then re-seal
    /// at a new version. Past versions are untouched: `check` still calls
    /// every earlier frame non-current, but `map` reads it at its own
    /// version. An advance that changes nothing adds no event and no
    /// generation, yet still takes a new version (a new, distinct frame).
    /// Returns that version.
    pub fn advance(
        &mut self,
        source: u64,
        relation_map: u64,
        changes: Vec<(TileId, Option<LocalMap>)>,
    ) -> RouteVersion {
        let version = self.frames.clock.tick();
        for (tile, map) in changes {
            match map {
                Some(m) => self.tiles.entry(tile).or_default().append(version, m),
                None => {
                    if let Some(log) = self.tiles.get_mut(&tile) {
                        log.remove(version);
                    }
                }
            }
        }
        let coordinates = Self::coordinates(&self.tiles);
        self.frames.advance_at(
            FrameRevisions {
                source,
                relation_map,
                coordinates,
            },
            version,
        );
        version
    }

    /// Seal the current frame.
    pub fn seal(&self) -> Frame {
        self.frames.seal()
    }

    pub fn revisions(&self) -> FrameRevisions {
        self.frames.revisions()
    }

    /// The map of `tile` as of `frame`'s version. A frame of this table
    /// always resolves (old frames replay their layout); a frame of another
    /// table is `Stale(ForeignFrame)`; a tile absent then is `NoSuchTile`.
    /// Borrows; allocates nothing.
    pub fn map(&self, frame: &Frame, tile: TileId) -> Result<TileMap<'_>, MapRead> {
        let version = self.frames.issued(frame).map_err(MapRead::Stale)?;
        self.view(version, tile)
    }

    /// The map of `tile` as of a numeric `version` (for replay where only the
    /// number survives). A version past the end of the sequence is
    /// `FutureVersion`.
    pub fn map_at(&self, version: RouteVersion, tile: TileId) -> Result<TileMap<'_>, MapRead> {
        let current = self.frames.clock.current();
        if version > current {
            return Err(MapRead::FutureVersion {
                asked: version,
                current,
            });
        }
        self.view(version, tile)
    }

    fn view(&self, version: RouteVersion, tile: TileId) -> Result<TileMap<'_>, MapRead> {
        self.tiles
            .get(&tile)
            .and_then(|log| log.at(version))
            .ok_or(MapRead::NoSuchTile(tile))
    }

    /// How many generations `tile` has stored (diagnostic: an extension does
    /// not add one).
    pub fn generation_count(&self, tile: TileId) -> usize {
        self.tiles.get(&tile).map_or(0, |l| l.gens.len())
    }

    /// How many events `tile` has logged (diagnostic).
    pub fn event_count(&self, tile: TileId) -> usize {
        self.tiles.get(&tile).map_or(0, |l| l.events.len())
    }

    /// Find which of the listed `tiles` holds `global` under `frame`.
    /// Returns `(ordinal, local)`: the index of the holding tile in `tiles`
    /// and the id's local position in that tile's map. Every listed tile is
    /// read, so an id held by two of them is refused rather than resolved to
    /// the first.
    pub fn locate(
        &self,
        frame: &Frame,
        tiles: &[TileId],
        global: u64,
    ) -> Result<(u32, u8), LocateRefusal> {
        let mut found: Option<(u32, u8)> = None;
        for (ordinal, &tile) in tiles.iter().enumerate() {
            let map = self.map(frame, tile).map_err(LocateRefusal::Read)?;
            // A lookup that finds nothing is not an error here; only the
            // overall absence is.
            if let Ok(local) = map.local(global) {
                if found.is_some() {
                    return Err(LocateRefusal::Ambiguous);
                }
                let ordinal = u32::try_from(ordinal).map_err(|_| LocateRefusal::Ambiguous)?;
                found = Some((ordinal, local));
            }
        }
        found.ok_or(LocateRefusal::Unmapped)
    }
}

/// Why [`MapTable::locate`] refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocateRefusal {
    /// A listed tile could not be read under the frame (stale frame, or no
    /// map for that tile).
    Read(MapRead),
    /// The id is held by more than one listed tile.
    Ambiguous,
    /// No listed tile holds the id.
    Unmapped,
}

impl LocalMap {
    /// True iff `older` is a pure prefix of `self`: same width, and every
    /// position of `older` keeps its global id in `self`, with `self` only
    /// appending. [`LocalMap::prepare`] does not sort the local order (only
    /// its lookup index is sorted), so an appended id never renumbers an
    /// existing position, whatever its value. A reorder, a removal or a
    /// width change makes this false.
    pub fn extends(&self, older: &LocalMap) -> bool {
        self.width == older.width
            && self.globals.len() >= older.globals.len()
            && self.globals[..older.globals.len()] == older.globals[..]
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
