//! D-DT64-G2L-1: one witnessed global → local translation, bound once and
//! reused across repeated GEMMs.
//!
//! Two parties hold the same population in different row orders. The
//! consumer names its selection in **global ids**. The source holds its u8
//! lanes in **its own order**, and DR's [`MapTable`] maps global id →
//! (tile, local position) for that order. Reading the source's lanes in the
//! consumer's order directly would be a gather, and is refused
//! ([`lance_graph_dt64::LaneRefusal::CoordinateSpace`]).
//!
//! ADMISSION B (#1640): preparation may establish coordinates; execution
//! operates on co-addressed sets; neither may hide an unaccounted gather.
//! So:
//!
//! 1. **Prepare, once, counted.** [`translate`] locates every selected global
//!    id under one frame of the source's map table (every lookup is counted
//!    by the table) and expresses the selection in the **source's** row order
//!    ([`Selection`] in [`SPACE_B`]). It refuses a map whose coordinate
//!    revision is not the source's current one (a stale mapping).
//! 2. **Bind as usual.** The object is bound on the source in the source's
//!    own coordinate space, so DT64's ordinary checks apply and `borrow_all`
//!    resolves it with one check.
//! 3. **Prepare B once.** The consumer's B is keyed by identity. It is read
//!    through the translation ([`TranslatedB`]: position → global through the
//!    map's pointer table, uncounted) only while packing, into the source's
//!    row order. Its operand identity names the translation, so a prepared B
//!    never outlives the translation it was built for.
//! 4. **Execute, many times, with zero lookups.** Every run reads the
//!    source's lanes in place, in the source's order, against the prepared
//!    B. The sum over K does not depend on K's order, so the result is the
//!    consumer's answer.
//!
//! A source reorder advances its frames and its map table; the old binding
//! and everything prepared under it are then refused, and a fresh
//! translation is exact again.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

use lance_graph_dt64::dr::{
    Frame, FrameRevisions, LocalMap, LocateRefusal, MapRead, MapTable, TileId, Width,
};
use lance_graph_dt64::CoordSpace;

use crate::bind::READER;
use crate::fixture::Fixture;
use crate::int8::{a_u8, U8Lanes, U8Socket};
use crate::shape::{BValues, InnerSource, OperandId, OperandSource, Selection};

/// The source's own row order of the population.
pub const SPACE_B: CoordSpace = CoordSpace {
    population: 1,
    coordinates: 3,
};

/// Positions per DR256 tile.
const TILE_POSITIONS: usize = 256;

/// The global id of canonical identity `i`: neither its index nor any
/// source's position.
pub fn global_of(i: usize) -> u64 {
    1_000 + 7 * i as u64
}

/// The canonical identity of a global id.
pub fn identity_of(g: u64) -> usize {
    ((g - 1_000) / 7) as usize
}

/// A source holding the population's u8 lanes in its own row order, with
/// the DR map table for that order. Position `l` holds identity `order[l]`.
pub struct OrderedSource {
    lanes: U8Lanes,
    map: MapTable,
    tiles: Vec<TileId>,
    order: Vec<usize>,
}

impl OrderedSource {
    /// `props` lanes over `fx.n` identities in `order` (a permutation).
    pub fn new(fx: &Fixture, props: usize, order: Vec<usize>) -> OrderedSource {
        assert_eq!(order.len(), fx.n);
        let lanes = U8Lanes::from_fn(fx.n, props, |l, p| a_u8(fx, order[l], p));
        let tiles: Vec<TileId> = (0..fx.n.div_ceil(TILE_POSITIONS) as u32).collect();
        let mut src = OrderedSource {
            lanes,
            map: MapTable::new(1, 1),
            tiles,
            order,
        };
        src.publish();
        src
    }

    /// Reorder the source in place: rewrite its lanes, advance its map table
    /// and then its frames to the map's new coordinate revision.
    pub fn reorder(&mut self, fx: &Fixture, order: Vec<usize>) {
        assert_eq!(order.len(), fx.n);
        self.order = order;
        let order = &self.order;
        self.lanes.rewrite(|l, p| a_u8(fx, order[l], p));
        self.publish();
    }

    /// Reorder the lanes but NOT the map table: the stale-mapping case.
    pub fn reorder_lanes_only(&mut self, fx: &Fixture, order: Vec<usize>) {
        self.order = order;
        let order = &self.order;
        self.lanes.rewrite(|l, p| a_u8(fx, order[l], p));
        let rev = self.lanes.frames_mut().revisions();
        self.lanes.frames_mut().advance(FrameRevisions {
            coordinates: rev.coordinates.wrapping_add(1),
            ..rev
        });
    }

    /// Replace every tile's map with the current order, then advance the
    /// source's frames to the map's coordinate revision.
    fn publish(&mut self) {
        let changes = self
            .tiles
            .iter()
            .map(|&t| {
                let lo = t as usize * TILE_POSITIONS;
                let hi = (lo + TILE_POSITIONS).min(self.order.len());
                let globals: Vec<u64> = self.order[lo..hi].iter().map(|&i| global_of(i)).collect();
                (
                    t,
                    Some(LocalMap::prepare(Width::W256, &globals).expect("a tile holds 256 ids")),
                )
            })
            .collect();
        self.map.advance(1, 1, changes);
        let coordinates = self.map.revisions().coordinates;
        let rev = self.lanes.frames_mut().revisions();
        self.lanes
            .frames_mut()
            .advance(FrameRevisions { coordinates, ..rev });
    }

    /// The DT64 socket over this source's lanes, in [`SPACE_B`].
    pub fn socket(&self, props: usize) -> U8Socket<'_> {
        U8Socket {
            lanes: &self.lanes,
            rows: (0..props).collect(),
            space: SPACE_B,
            scope: READER,
            borrows: Cell::new(0),
        }
    }

    /// Global → local lookups the map table has served, over every tile.
    pub fn lookups(&self) -> u64 {
        let frame = self.map.seal();
        self.tiles
            .iter()
            .filter_map(|&t| self.map.map(&frame, t).ok())
            .map(|m| m.lookups())
            .sum()
    }

    /// The source's coordinate revision: the digest of its current order.
    pub fn coordinates(&self) -> u64 {
        self.lanes.frames().revisions().coordinates
    }

    /// Identities held by the source.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// True if the source holds no identity.
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
}

/// Why a translation was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum G2lRefusal {
    /// The map table's coordinate revision is not the source's: the source's
    /// order moved without its map (a stale mapping).
    MapNotSource { map: u64, source: u64 },
    /// A selected global id is not held by the source, or is held twice.
    Locate { global: u64, refusal: LocateRefusal },
    /// A global id is selected twice.
    Duplicate { global: u64 },
    /// The translation was made under another order of the source.
    Stale { made: u64, now: u64 },
    /// The translation was made on another source: its frame was issued by
    /// another map table, even if that table holds the same order.
    Foreign,
}

static NEXT_TRANSLATION: AtomicU64 = AtomicU64::new(1);

/// A witnessed translation of a global-id selection into the source's row
/// order, made once under one frame of the source's map table.
#[derive(Debug, Clone)]
pub struct Translation {
    id: u64,
    /// The source's coordinate revision the translation was made under.
    coordinates: u64,
    frame: Frame,
    tiles: Vec<TileId>,
    selection: Selection,
    lookups: u64,
}

impl Translation {
    /// The selection in the source's row order ([`SPACE_B`]).
    pub fn selection(&self) -> &Selection {
        &self.selection
    }

    /// Lookups this translation cost (once, at preparation).
    pub fn lookups(&self) -> u64 {
        self.lookups
    }

    /// Refuses a translation made on another source (its frame was issued by
    /// another map table), and one made under another order of this source.
    pub fn check(&self, source: &OrderedSource) -> Result<(), G2lRefusal> {
        if let Some(&tile) = self.tiles.first() {
            if let Err(MapRead::Stale(_)) = source.map.map(&self.frame, tile) {
                return Err(G2lRefusal::Foreign);
            }
        }
        let now = source.lanes.frames().revisions().coordinates;
        if now != self.coordinates {
            return Err(G2lRefusal::Stale {
                made: self.coordinates,
                now,
            });
        }
        Ok(())
    }
}

/// Prepare: locate every global id under one frame of the source's map
/// table and express the selection in the source's own row order. Counted;
/// done once.
pub fn translate(source: &OrderedSource, globals: &[u64]) -> Result<Translation, G2lRefusal> {
    let map = source.map.revisions().coordinates;
    let src = source.lanes.frames().revisions().coordinates;
    if map != src {
        return Err(G2lRefusal::MapNotSource { map, source: src });
    }
    let before = source.lookups();
    let frame = source.map.seal();
    let mut positions = Vec::with_capacity(globals.len());
    for &g in globals {
        let (ordinal, local) = source
            .map
            .locate(&frame, &source.tiles, g)
            .map_err(|refusal| G2lRefusal::Locate { global: g, refusal })?;
        positions.push(ordinal as usize * TILE_POSITIONS + usize::from(local));
    }
    let mut sorted = positions.clone();
    sorted.sort_unstable();
    if let Some(w) = sorted.windows(2).find(|w| w[0] == w[1]) {
        let local = w[0];
        let global = globals[positions.iter().position(|&p| p == local).unwrap_or(0)];
        return Err(G2lRefusal::Duplicate { global });
    }
    let selection = Selection::mask(SPACE_B, source.len(), positions);
    Ok(Translation {
        id: NEXT_TRANSLATION.fetch_add(1, Ordering::Relaxed),
        coordinates: src,
        frame,
        tiles: source.tiles.clone(),
        selection,
        lookups: source.lookups() - before,
    })
}

/// The consumer's B (keyed by canonical identity) read at the source's row
/// positions through a translation. Used only to prepare B; counts its reads
/// so a test can show execution never comes back to it.
pub struct TranslatedB<'a, T> {
    inner: &'a T,
    inner_source: InnerSource,
    source: &'a OrderedSource,
    translation: &'a Translation,
    reads: Cell<u64>,
    admission_reads: Cell<u64>,
    /// The last position resolved and its identity. Packing reads every
    /// column of one row before the next, so one map read per row suffices.
    last: Cell<Option<(usize, usize)>>,
}

impl<'a, T: BValues> TranslatedB<'a, T> {
    /// Refuses a translation made under another order of the source.
    pub fn new(
        inner: &'a T,
        source: &'a OrderedSource,
        translation: &'a Translation,
    ) -> Result<Self, G2lRefusal> {
        translation.check(source)?;
        let inner_source = match inner.identity().source {
            OperandSource::Function { seed } => InnerSource::Function { seed },
            OperandSource::Table { instance } => InnerSource::Table { instance },
            OperandSource::Translated { .. } => panic!("a translation of a translation"),
        };
        Ok(TranslatedB {
            inner,
            inner_source,
            source,
            translation,
            reads: Cell::new(0),
            admission_reads: Cell::new(0),
            last: Cell::new(None),
        })
    }

    /// Map reads made through the translation so far (one per row packed).
    pub fn reads(&self) -> u64 {
        self.reads.get()
    }

    /// Map reads made while admitting the operand (one per selected row,
    /// to find the highest inner row it reads), counted apart from
    /// [`Self::reads`].
    pub fn admission_reads(&self) -> u64 {
        self.admission_reads.get()
    }

    /// Source position → canonical identity, through the map's pointer table
    /// at the translation's frame (the inverse read). A position outside the
    /// translation's tiles, or unmapped, is `None`.
    fn resolve(&self, position: usize) -> Option<usize> {
        let tile = *self.translation.tiles.get(position / TILE_POSITIONS)?;
        let local = u8::try_from(position % TILE_POSITIONS).ok()?;
        let map = self.source.map.map(&self.translation.frame, tile).ok()?;
        Some(identity_of(map.global(local)?))
    }

    /// [`Self::resolve`] for packing, cached per row. Packing runs only after
    /// [`crate::shape::admit_operand`] resolved every selected position.
    fn identity_at(&self, position: usize) -> usize {
        if let Some((p, identity)) = self.last.get() {
            if p == position {
                return identity;
            }
        }
        self.reads.set(self.reads.get() + 1);
        let identity = self
            .resolve(position)
            .expect("admission resolved every selected position");
        self.last.set(Some((position, identity)));
        identity
    }
}

impl<T: BValues> BValues for TranslatedB<'_, T> {
    fn identity(&self) -> OperandId {
        OperandId {
            source: OperandSource::Translated {
                translation: self.translation.id,
                inner: self.inner_source,
            },
            rev: self.inner.identity().rev,
        }
    }
    fn bf16_at(&self, i: usize, j: usize) -> u16 {
        self.inner.bf16_at(self.identity_at(i), j)
    }
    fn i8_at(&self, i: usize, j: usize) -> i8 {
        self.inner.i8_at(self.identity_at(i), j)
    }
    /// The inner operand's own extent: its rows are identities, not source
    /// positions, so a translated identity past the inner table is refused
    /// through [`Self::highest_row`] (#1990's OperandShape contract).
    fn extent(&self) -> Option<(usize, usize)> {
        self.inner.extent()
    }
    /// The highest identity any selected position reads. An unmapped
    /// position reads as `usize::MAX`, so admission refuses it too.
    fn highest_row(&self, selection: &Selection) -> Option<usize> {
        let mut highest = None;
        selection.for_each(|_, position| {
            self.admission_reads.set(self.admission_reads.get() + 1);
            let identity = self.resolve(position).unwrap_or(usize::MAX);
            highest = highest.max(Some(identity));
        });
        highest
    }
}
