//! LAB: shared library of the D-GLOBAL-WHERE P1 probe (falsify_p1, sweep, big).
//!
//! Lab only. No production change; identities live ONLY in
//! `lance_graph_dt64::dr::{MapTable, LocalMap}`. Everything here is either an
//! instrument (allocator, rng, timers), a positional structure over DR
//! coordinates (masks, coordinate CSR, tile matrix), a derived index that is
//! built FROM `MapTable` views under one frame (arm C), or a LAB type the spec
//! names (`SharedSelection`, EXO-46).
//!
//! # Model
//!
//! * `WorkingSet`: ONE `MapTable` (own frame store and clock) holding up to 256
//!   DR256 tiles (TileIds are ordinals 0..tiles for a freshly built set). Its
//!   members are chosen by the QUERY (not contiguous, not by id bits).
//!   Coordinate of an identity = `tile_ordinal * tw + local`, `tw` = tile width
//!   (64 or 256). For a set built by `WorkingSet::build(ids)` the coordinate
//!   of `ids[i]` is `i`, and `extend` keeps that.
//! * Lanes are read through the pointer table at PREPARATION only
//!   (`gather_lane`); folds read coordinate-indexed arrays (len `n_coords`,
//!   padding = `PAD`). A lane array is therefore the source lane re-laid in
//!   coordinate order; a fold never looks an identity up.
//! * Node reference (`noderef`): `(set << 16) | coord` (coord < 65536, set <
//!   32768). Coordinate-CSR targets store `coord` for a same-set target and
//!   `CROSS | noderef` for a target in another working set.
//!
//! # API index (for sweep.rs / big.rs)
//!
//! * instruments: `CountingAlloc`, `alloc_bytes/alloc_calls/live_bytes/
//!   peak_bytes/reset_peak`, `Rng`, `SEED`, `time_runs`, `stats`, `Stage`,
//!   `StageTimes`, `fmt_us`, `gate_label`
//! * bits: `bit`, `set_bit`, `clear_bit`, `popcount`, `for_each_bit`, `words_for`
//! * identity: `WorkingSet::{build, from_tiles, extend, replace_tile,
//!   remove_tile, seal, gather_lane, valid_mask, coord_a, n_coords, tw}`
//! * routing: `route_batch_a_in`, `route_batch_b_in`, `route_batch_c`,
//!   `CoordIndex::{build, build_in, build_with, lookup}`, `RouteStats`,
//!   `COORD_UNMAPPED`, `COORD_AMBIGUOUS`, `Arm`
//! * relation view: `CoordCsr` (+ `delete`), `Binding::{prepare, rewhere_b,
//!   guard, delete}`, `TileMatrix`
//! * where: `virtual_where`, `Prepared` (Quack WHERE over a corpus lane)
//! * folds: `w2`, `w2_and`, `w2_xor`, `w3_csr`, `w3_front`, `w4_csr`,
//!   `w4_front`, `TileMatrix::fold` (W4 tile), `w5`, `w6` (+ `FoldScratch`,
//!   `semijoin`, `step`)
//! * oracle: `oracle()` over a global edge list (HashSet/HashMap allowed here)
//! * EXO-46: `SharedSelection`, `ShareRefusal`
//!
//! Workload semantics (A, B are masks over the working-set coordinates):
//! W2 = |{x in A : exists x->y, y in B}|; W3 = |{x in A : exists x->y->z, z in B}|;
//! W4 = |{z in B : z reached from A in exactly 3 hops}| (distinct);
//! W5 = |{y in B : exists x in A, x->y}| (count distinct, 1 hop, as SQL G5);
//! W6 = sum over x in A, x->y of cnt(y), cnt(y) = outdeg(y), or with `Some(B)`
//! the out-edges of y into B (bag count, as SQL G6) = sum_y indegA[y]*cnt(y).
//! Only VALID (not deleted) edges take part in every fold.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use lance_graph_dt64::dr::{
    Frame, LocalMap, LocateRefusal, MapRead, MapRefusal, MapTable, RouteVersion, Scope, TileId,
};
use lance_graph_dt64::{Axis, BindStore, MethodRef, Refusal, Width};
use lance_graph_mask_risc::{execute_into, Foreign, LaneRef, Out, Planes, Program, Scratch, Value};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, Query};

// ---------------------------------------------------------------------------
// counting allocator (each binary declares the #[global_allocator] static)
// ---------------------------------------------------------------------------

/// A pass-through allocator that counts bytes, calls, live bytes and the peak.
pub struct CountingAlloc;

static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: a pure pass-through to `System`; the atomics are the only addition.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_BYTES.fetch_add(layout.size(), Relaxed);
        ALLOC_CALLS.fetch_add(1, Relaxed);
        let live = LIVE_BYTES.fetch_add(layout.size(), Relaxed) + layout.size();
        PEAK_BYTES.fetch_max(live, Relaxed);
        // SAFETY: same layout, same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), Relaxed);
        // SAFETY: `ptr` came from `alloc` above with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

pub fn alloc_bytes() -> usize {
    ALLOC_BYTES.load(Relaxed)
}
pub fn alloc_calls() -> usize {
    ALLOC_CALLS.load(Relaxed)
}
pub fn live_bytes() -> usize {
    LIVE_BYTES.load(Relaxed)
}
pub fn peak_bytes() -> usize {
    PEAK_BYTES.load(Relaxed)
}
pub fn reset_peak() {
    PEAK_BYTES.store(LIVE_BYTES.load(Relaxed), Relaxed);
}

// ---------------------------------------------------------------------------
// deterministic data
// ---------------------------------------------------------------------------

pub const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64.
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------------------
// bit helpers
// ---------------------------------------------------------------------------

#[inline]
pub fn words_for(n: usize) -> usize {
    n.div_ceil(64)
}
#[inline]
pub fn bit(m: &[u64], i: usize) -> bool {
    (m[i >> 6] >> (i & 63)) & 1 == 1
}
#[inline]
pub fn set_bit(m: &mut [u64], i: usize) {
    m[i >> 6] |= 1u64 << (i & 63);
}
#[inline]
pub fn clear_bit(m: &mut [u64], i: usize) {
    m[i >> 6] &= !(1u64 << (i & 63));
}
/// Set bits of a mask (`ndarray::simd::popcount_batch_u64`).
#[inline]
pub fn popcount(m: &[u64]) -> u64 {
    ndarray::simd::popcount_batch_u64(m)
}
/// Call `f` with the index of every set bit, ascending. No allocation.
#[inline]
pub fn for_each_bit(m: &[u64], mut f: impl FnMut(usize)) {
    for (w, &word) in m.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            f(w * 64 + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
    }
}

// ---------------------------------------------------------------------------
// timing and stage helpers
// ---------------------------------------------------------------------------

/// Run `f` `runs` times, returning each wall time in microseconds.
pub fn time_runs(runs: usize, mut f: impl FnMut()) -> Vec<f64> {
    let mut out = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        f();
        out.push(t.elapsed().as_nanos() as f64 / 1000.0);
    }
    out
}

/// `(median, p10, p90)` of a sample (sorts a copy).
pub fn stats(samples: &[f64]) -> (f64, f64, f64) {
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |p: f64| s[(((s.len() - 1) as f64) * p).round() as usize];
    (at(0.5), at(0.1), at(0.9))
}

/// The measured stages of a query over an unbounded corpus (SPEC correction).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Global WHERE -> candidate ids (scan arm or attribute-index arm).
    Discover,
    /// Reachable demanded ids via the global relation index, bounded at 65,536.
    Expand,
    /// DR coordinate binding of exactly those ids (arms A/B/C).
    Bind,
    /// Coordinate CSR + presence bits restricted to the working set.
    Project,
    /// The fold over the working set.
    Fold,
}

impl Stage {
    pub const ALL: [Stage; 5] = [Stage::Discover, Stage::Expand, Stage::Bind, Stage::Project, Stage::Fold];
    pub fn name(self) -> &'static str {
        match self {
            Stage::Discover => "discover",
            Stage::Expand => "expand",
            Stage::Bind => "bind",
            Stage::Project => "project",
            Stage::Fold => "fold",
        }
    }
    fn idx(self) -> usize {
        self as usize
    }
}

/// Accumulated nanoseconds per stage. `time` wraps one stage execution.
#[derive(Clone, Debug, Default)]
pub struct StageTimes {
    ns: [u128; 5],
    calls: [u64; 5],
}

impl StageTimes {
    pub fn new() -> StageTimes {
        StageTimes::default()
    }
    /// Run `f`, charge its wall time to `stage`.
    pub fn time<R>(&mut self, stage: Stage, f: impl FnOnce() -> R) -> R {
        let t = Instant::now();
        let r = f();
        self.ns[stage.idx()] += t.elapsed().as_nanos();
        self.calls[stage.idx()] += 1;
        r
    }
    /// Charge an externally measured duration (nanoseconds).
    pub fn add_ns(&mut self, stage: Stage, ns: u128) {
        self.ns[stage.idx()] += ns;
        self.calls[stage.idx()] += 1;
    }
    pub fn us(&self, stage: Stage) -> f64 {
        self.ns[stage.idx()] as f64 / 1000.0
    }
    pub fn calls(&self, stage: Stage) -> u64 {
        self.calls[stage.idx()]
    }
    pub fn total_us(&self) -> f64 {
        self.ns.iter().sum::<u128>() as f64 / 1000.0
    }
    pub fn reset(&mut self) {
        *self = StageTimes::default();
    }
    /// One line per stage. `cold` stages (preparation) are labelled, not gated.
    pub fn report(&self, cold: bool) -> String {
        Stage::ALL
            .iter()
            .map(|&s| format!("{}={}", s.name(), fmt_us(self.us(s), cold)))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Gate label for a microsecond value: warm values >= 1000 us are a failed
/// gate; cold/preparation values >= 1000 us are reported, not gated.
pub fn gate_label(us: f64, cold: bool) -> &'static str {
    if us < 1000.0 {
        ""
    } else if cold {
        " COLD (reported, not gate)"
    } else {
        " FAILED-GATE"
    }
}

pub fn fmt_us(us: f64, cold: bool) -> String {
    format!("{us:.3}us{}", gate_label(us, cold))
}

// ---------------------------------------------------------------------------
// working set
// ---------------------------------------------------------------------------

/// Tiles a working set may hold: the `Extent::DR256_GLOBAL_X64` reservation.
pub const MAX_TILES: usize = 256;
/// Lane value at coordinates that hold no identity.
pub const PAD: i32 = i32::MAX;

#[derive(Debug)]
pub enum WsError {
    Map(MapRefusal),
    Read(MapRead),
}
impl From<MapRefusal> for WsError {
    fn from(e: MapRefusal) -> WsError {
        WsError::Map(e)
    }
}
impl From<MapRead> for WsError {
    fn from(e: MapRead) -> WsError {
        WsError::Read(e)
    }
}

/// One `MapTable`, up to 256 tiles, <= 65,536 identities chosen by the query.
#[derive(Debug)]
pub struct WorkingSet {
    pub table: MapTable,
    /// The tile list `locate` reads, in ordinal order.
    pub tile_ids: Vec<TileId>,
    pub width: Width,
    source: u64,
    relmap: u64,
}

impl WorkingSet {
    /// Explicit tile contents (a tile may even repeat another tile's id: that
    /// is the Ambiguous fixture). Refuses duplicates inside one tile.
    pub fn from_tiles(width: Width, tiles: Vec<Vec<u64>>, source: u64) -> Result<WorkingSet, WsError> {
        if tiles.len() > MAX_TILES {
            return Err(MapRefusal::TooMany.into());
        }
        let mut table = MapTable::new(source, 0);
        let mut changes = Vec::with_capacity(tiles.len());
        let mut tile_ids = Vec::with_capacity(tiles.len());
        for (t, g) in tiles.iter().enumerate() {
            changes.push((t as TileId, Some(LocalMap::prepare(width, g)?)));
            tile_ids.push(t as TileId);
        }
        table.advance(source, 0, changes);
        Ok(WorkingSet {
            table,
            tile_ids,
            width,
            source,
            relmap: 0,
        })
    }

    /// Chunk `ids` (in the given order) into tiles of the width
    /// `Axis::for_survivors` picks. The coordinate of `ids[i]` is `i`.
    pub fn build(ids: &[u64], source: u64) -> Result<WorkingSet, WsError> {
        if ids.len() > MAX_TILES * 256 {
            return Err(MapRefusal::TooMany.into());
        }
        let width = Axis::for_survivors(ids.len()).tile;
        let tw = width.positions();
        let tiles: Vec<Vec<u64>> = ids.chunks(tw).map(|c| c.to_vec()).collect();
        WorkingSet::from_tiles(width, tiles, source)
    }

    pub fn tw(&self) -> usize {
        self.width.positions()
    }
    pub fn n_coords(&self) -> usize {
        self.tile_ids.len() * self.tw()
    }
    pub fn seal(&self) -> Frame {
        self.table.seal()
    }
    /// Identities held under `frame` (sum of tile lengths).
    pub fn n_ids(&self, frame: &Frame) -> usize {
        self.tile_ids
            .iter()
            .filter_map(|&t| self.table.map(frame, t).ok())
            .map(|m| m.len())
            .sum()
    }

    /// Arm A for one id: `MapTable::locate` over this set's tile list.
    pub fn coord_a(&self, frame: &Frame, g: u64) -> Result<usize, LocateRefusal> {
        coord_a_in(&self.table, frame, &self.tile_ids, self.tw(), g)
    }

    /// Bit per coordinate that holds an identity under `frame`.
    pub fn valid_mask(&self, frame: &Frame) -> Vec<u64> {
        let tw = self.tw();
        let mut m = vec![0u64; words_for(self.n_coords())];
        for (o, &t) in self.tile_ids.iter().enumerate() {
            if let Ok(view) = self.table.map(frame, t) {
                for l in 0..view.len() {
                    set_bit(&mut m, o * tw + l);
                }
            }
        }
        m
    }

    /// Re-lay a source lane in coordinate order (preparation only: the one
    /// place identity is read to fetch a value). Length `n_coords`, `PAD`
    /// where no identity.
    pub fn gather_lane(&self, frame: &Frame, source: impl Fn(u64) -> i32) -> Vec<i32> {
        let tw = self.tw();
        let mut lane = vec![PAD; self.n_coords()];
        for (o, &t) in self.tile_ids.iter().enumerate() {
            if let Ok(view) = self.table.map(frame, t) {
                for (l, &g) in view.globals().iter().enumerate() {
                    lane[o * tw + l] = source(g);
                }
            }
        }
        lane
    }

    /// Append identities: fill the last non-full tile, then new tiles, in ONE
    /// `MapTable::advance`. An existing tile only ever grows by appending, so
    /// `TileMap::extends` holds and every old coordinate is kept. Returns the
    /// number of tiles changed.
    pub fn extend(&mut self, new_globals: &[u64]) -> Result<usize, WsError> {
        if new_globals.is_empty() {
            return Ok(0);
        }
        let frame = self.table.seal();
        let tw = self.tw();
        let mut changes: Vec<(TileId, Option<LocalMap>)> = Vec::new();
        let mut rest = new_globals;
        if let Some(&last) = self.tile_ids.last() {
            let view = self.table.map(&frame, last)?;
            let free = tw - view.len();
            if free > 0 {
                let take = free.min(rest.len());
                let mut v = view.globals().to_vec();
                v.extend_from_slice(&rest[..take]);
                changes.push((last, Some(LocalMap::prepare(self.width, &v)?)));
                rest = &rest[take..];
            }
        }
        let mut next_ids = self.tile_ids.clone();
        while !rest.is_empty() {
            if next_ids.len() >= MAX_TILES {
                return Err(MapRefusal::TooMany.into());
            }
            let take = tw.min(rest.len());
            let t = next_ids.last().map_or(0, |l| l + 1);
            changes.push((t, Some(LocalMap::prepare(self.width, &rest[..take])?)));
            next_ids.push(t);
            rest = &rest[take..];
        }
        let n = changes.len();
        self.source += 1;
        self.table.advance(self.source, self.relmap, changes);
        self.tile_ids = next_ids;
        Ok(n)
    }

    /// Replace one tile's map (a reorder / removal of an id is a replacement).
    pub fn replace_tile(&mut self, tile: TileId, globals: &[u64]) -> Result<RouteVersion, WsError> {
        let m = LocalMap::prepare(self.width, globals)?;
        self.source += 1;
        Ok(self.table.advance(self.source, self.relmap, vec![(tile, Some(m))]))
    }

    /// Remove a tile (an event to `None`). `tile_ids` is NOT edited: callers
    /// that read the new frame pass the reduced list to the `_in` functions.
    pub fn remove_tile(&mut self, tile: TileId) -> RouteVersion {
        self.source += 1;
        self.table.advance(self.source, self.relmap, vec![(tile, None)])
    }
}

// ---------------------------------------------------------------------------
// routing arms (identical answers, including refusals)
// ---------------------------------------------------------------------------

pub const COORD_UNMAPPED: u32 = u32::MAX;
pub const COORD_AMBIGUOUS: u32 = u32::MAX - 1;

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteStats {
    /// `MapTable::locate` calls made (0 for arm C queries).
    pub locate_calls: u64,
    pub distinct: u64,
    pub unmapped: u64,
    pub ambiguous: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    A,
    B,
    C,
}
impl Arm {
    pub const ALL: [Arm; 3] = [Arm::A, Arm::B, Arm::C];
    pub fn name(self) -> &'static str {
        match self {
            Arm::A => "A",
            Arm::B => "B",
            Arm::C => "C",
        }
    }
}

/// Arm A for one id over an explicit tile list.
pub fn coord_a_in(
    table: &MapTable,
    frame: &Frame,
    tiles: &[TileId],
    tw: usize,
    g: u64,
) -> Result<usize, LocateRefusal> {
    let (ordinal, local) = table.locate(frame, tiles, g)?;
    Ok(ordinal as usize * tw + usize::from(local))
}

fn code_of(r: Result<usize, LocateRefusal>) -> Result<u32, LocateRefusal> {
    match r {
        Ok(c) => Ok(c as u32),
        Err(LocateRefusal::Unmapped) => Ok(COORD_UNMAPPED),
        Err(LocateRefusal::Ambiguous) => Ok(COORD_AMBIGUOUS),
        Err(e) => Err(e),
    }
}

/// Arm A over a batch: one `locate` per endpoint. `out[i]` is the coordinate
/// or `COORD_UNMAPPED` / `COORD_AMBIGUOUS`. A frame-read refusal is `Err`
/// (it hits every id alike).
pub fn route_batch_a_in(
    table: &MapTable,
    frame: &Frame,
    tiles: &[TileId],
    tw: usize,
    endpoints: &[u64],
    out: &mut [u32],
) -> Result<RouteStats, LocateRefusal> {
    let mut st = RouteStats::default();
    for (i, &g) in endpoints.iter().enumerate() {
        st.locate_calls += 1;
        let c = code_of(coord_a_in(table, frame, tiles, tw, g))?;
        tally(&mut st, c);
        out[i] = c;
    }
    Ok(st)
}

/// Arm B over a batch: distinct sorted ids, `locate` ONCE each, then resolve
/// endpoints by binary search into the transient sorted array (dropped on
/// return: not an index).
pub fn route_batch_b_in(
    table: &MapTable,
    frame: &Frame,
    tiles: &[TileId],
    tw: usize,
    endpoints: &[u64],
    out: &mut [u32],
) -> Result<RouteStats, LocateRefusal> {
    let mut d: Vec<u64> = endpoints.to_vec();
    d.sort_unstable();
    d.dedup();
    let mut codes: Vec<u32> = Vec::with_capacity(d.len());
    let mut st = RouteStats {
        distinct: d.len() as u64,
        ..RouteStats::default()
    };
    for &g in &d {
        st.locate_calls += 1;
        codes.push(code_of(coord_a_in(table, frame, tiles, tw, g))?);
    }
    for (i, &g) in endpoints.iter().enumerate() {
        let at = d.binary_search(&g).expect("endpoint is in the distinct set");
        out[i] = codes[at];
        tally(&mut st, codes[at]);
    }
    Ok(st)
}

fn tally(st: &mut RouteStats, c: u32) {
    if c == COORD_UNMAPPED {
        st.unmapped += 1;
    } else if c == COORD_AMBIGUOUS {
        st.ambiguous += 1;
    }
}

/// Switches for the falsifier disable runs.
#[derive(Clone, Copy, Debug, Default)]
pub struct CFlags {
    /// Skip marking equal adjacent globals Ambiguous (first coordinate wins).
    pub disable_dupcheck: bool,
    /// Skip the `same_issue` frame check in `lookup`.
    pub disable_framecheck: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CRefusal {
    StaleIndex,
    Ambiguous,
    Unmapped,
}

/// Arm C (LAB, optional acceleration): a sorted `(global, coord)` array built
/// FROM `table.map(frame, t).globals()` of every tile under ONE frame. It
/// carries that frame and refuses any other. Never an identity authority.
#[derive(Clone, Debug)]
pub struct CoordIndex {
    built: Frame,
    entries: Vec<(u64, u32)>,
    flags: CFlags,
}

impl CoordIndex {
    pub fn build(ws: &WorkingSet, frame: &Frame) -> Result<CoordIndex, LocateRefusal> {
        CoordIndex::build_with(ws, frame, CFlags::default())
    }
    pub fn build_with(ws: &WorkingSet, frame: &Frame, flags: CFlags) -> Result<CoordIndex, LocateRefusal> {
        CoordIndex::build_in(&ws.table, &ws.tile_ids, ws.tw(), frame, flags)
    }
    pub fn build_in(
        table: &MapTable,
        tiles: &[TileId],
        tw: usize,
        frame: &Frame,
        flags: CFlags,
    ) -> Result<CoordIndex, LocateRefusal> {
        let mut raw: Vec<(u64, u32)> = Vec::new();
        for (o, &t) in tiles.iter().enumerate() {
            let view = table.map(frame, t).map_err(LocateRefusal::Read)?;
            for (l, &g) in view.globals().iter().enumerate() {
                raw.push((g, (o * tw + l) as u32));
            }
        }
        raw.sort_unstable();
        let mut entries: Vec<(u64, u32)> = Vec::with_capacity(raw.len());
        for (g, c) in raw {
            match entries.last_mut() {
                Some(last) if last.0 == g => {
                    if !flags.disable_dupcheck {
                        last.1 = COORD_AMBIGUOUS;
                    }
                }
                _ => entries.push((g, c)),
            }
        }
        Ok(CoordIndex {
            built: frame.clone(),
            entries,
            flags,
        })
    }
    pub fn frame(&self) -> &Frame {
        &self.built
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// One binary search; refuses a frame this index was not built for.
    pub fn lookup(&self, frame: &Frame, g: u64) -> Result<u32, CRefusal> {
        if !self.flags.disable_framecheck && !frame.same_issue(&self.built) {
            return Err(CRefusal::StaleIndex);
        }
        match self.entries.binary_search_by_key(&g, |e| e.0) {
            Ok(at) if self.entries[at].1 == COORD_AMBIGUOUS => Err(CRefusal::Ambiguous),
            Ok(at) => Ok(self.entries[at].1),
            Err(_) => Err(CRefusal::Unmapped),
        }
    }
}

/// Arm C over a batch: 0 `locate` calls. `StaleIndex` is `Err`.
pub fn route_batch_c(
    idx: &CoordIndex,
    frame: &Frame,
    endpoints: &[u64],
    out: &mut [u32],
) -> Result<RouteStats, CRefusal> {
    let mut st = RouteStats::default();
    for (i, &g) in endpoints.iter().enumerate() {
        let c = match idx.lookup(frame, g) {
            Ok(c) => c,
            Err(CRefusal::Unmapped) => COORD_UNMAPPED,
            Err(CRefusal::Ambiguous) => COORD_AMBIGUOUS,
            Err(CRefusal::StaleIndex) => return Err(CRefusal::StaleIndex),
        };
        tally(&mut st, c);
        out[i] = c;
    }
    Ok(st)
}

// ---------------------------------------------------------------------------
// node references and the coordinate CSR (the project stage)
// ---------------------------------------------------------------------------

/// Marks a CSR target that lives in another working set.
pub const CROSS: u32 = 1 << 31;
pub const UNROUTED: u32 = u32::MAX;

#[inline]
pub fn noderef(set: usize, coord: usize) -> u32 {
    ((set as u32) << 16) | coord as u32
}
#[inline]
pub fn nr_set(n: u32) -> usize {
    ((n >> 16) & 0x7FFF) as usize
}
#[inline]
pub fn nr_coord(n: u32) -> usize {
    (n & 0xFFFF) as usize
}
/// CSR target encoding: local coord, or `CROSS | noderef` for another set.
#[inline]
pub fn tgt_pack(src_set: usize, dst: u32) -> u32 {
    if nr_set(dst) == src_set {
        nr_coord(dst) as u32
    } else {
        CROSS | dst
    }
}
#[inline]
pub fn tgt_unpack(src_set: usize, t: u32) -> u32 {
    if t & CROSS != 0 {
        t & !CROSS
    } else {
        noderef(src_set, t as usize)
    }
}

/// Per working set: edges by source coordinate. `valid` is the presence
/// plane over edge slots; a deletion clears a bit and moves no coordinate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoordCsr {
    pub n: usize,
    pub off: Vec<u32>,
    pub tgt: Vec<u32>,
    pub valid: Vec<u64>,
}

impl CoordCsr {
    /// Counting-sort the routed edges whose source is in set `si`.
    pub fn from_routed(si: usize, n_coords: usize, edges: &[(u32, u32)]) -> CoordCsr {
        let mut off = vec![0u32; n_coords + 1];
        for &(s, _) in edges {
            if nr_set(s) == si {
                off[nr_coord(s) + 1] += 1;
            }
        }
        for i in 0..n_coords {
            off[i + 1] += off[i];
        }
        let total = off[n_coords] as usize;
        let mut tgt = vec![0u32; total];
        let mut cur: Vec<u32> = off[..n_coords].to_vec();
        for &(s, d) in edges {
            if nr_set(s) == si {
                let x = nr_coord(s);
                tgt[cur[x] as usize] = tgt_pack(si, d);
                cur[x] += 1;
            }
        }
        let mut valid = vec![0u64; words_for(total)];
        for e in 0..total {
            set_bit(&mut valid, e);
        }
        CoordCsr {
            n: n_coords,
            off,
            tgt,
            valid,
        }
    }
    pub fn edges(&self) -> usize {
        self.tgt.len()
    }
    pub fn valid_edges(&self) -> usize {
        popcount(&self.valid) as usize
    }
    /// Clear the presence bit of one valid edge `x -> t` (packed target).
    pub fn delete(&mut self, x: usize, t: u32) -> bool {
        for e in self.off[x] as usize..self.off[x + 1] as usize {
            if self.tgt[e] == t && bit(&self.valid, e) {
                clear_bit(&mut self.valid, e);
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// the binding: routed endpoints, coordinate CSR, selections (bind + project)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum BindError {
    Locate(LocateRefusal),
    C(CRefusal),
    /// An endpoint id is held by more than one tile of a set.
    Ambiguous(u64),
    /// The binding's frame of set `i` is no longer that set's current seal.
    Stale(usize),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BindStats {
    pub locate_calls: u64,
    pub endpoints: usize,
    /// Edges with an endpoint outside every set (dropped, not routed).
    pub dropped_edges: usize,
}

/// A reusable binding over one or more working sets: the frames it was built
/// under, the coordinate CSR per set, the coordinate-ordered lanes and the
/// A/B masks. Reuse is legal only while `guard` says every frame is current.
#[derive(Clone, Debug)]
pub struct Binding {
    pub frames: Vec<Frame>,
    pub graphs: Vec<CoordCsr>,
    pub valid: Vec<Vec<u64>>,
    pub lane_a: Vec<Vec<i32>>,
    pub lane_b: Vec<Vec<i32>>,
    pub a: Vec<Vec<u64>>,
    pub b: Vec<Vec<u64>>,
    pub stats: BindStats,
}

impl Binding {
    /// Route every endpoint of `edges` (global ids) with `arm` into the sets,
    /// build the coordinate CSR per set, gather the lanes and evaluate the
    /// virtual WHERE (`a < a_lt`, `b < b_lt`). The first set that maps an id
    /// owns it.
    pub fn prepare(
        sets: &[&WorkingSet],
        edges: &[(u64, u64)],
        arm: Arm,
        a_of: &dyn Fn(u64) -> i32,
        b_of: &dyn Fn(u64) -> i32,
        a_lt: i32,
        b_lt: i32,
    ) -> Result<Binding, BindError> {
        let frames: Vec<Frame> = sets.iter().map(|s| s.seal()).collect();
        let mut endpoints: Vec<u64> = Vec::with_capacity(edges.len() * 2);
        for &(s, d) in edges {
            endpoints.push(s);
            endpoints.push(d);
        }
        let mut routed = vec![UNROUTED; endpoints.len()];
        let mut out = vec![0u32; endpoints.len()];
        let mut stats = BindStats {
            endpoints: endpoints.len(),
            ..BindStats::default()
        };
        for (si, ws) in sets.iter().enumerate() {
            let st = match arm {
                Arm::A => route_batch_a_in(&ws.table, &frames[si], &ws.tile_ids, ws.tw(), &endpoints, &mut out)
                    .map_err(BindError::Locate)?,
                Arm::B => route_batch_b_in(&ws.table, &frames[si], &ws.tile_ids, ws.tw(), &endpoints, &mut out)
                    .map_err(BindError::Locate)?,
                Arm::C => {
                    let idx = CoordIndex::build(ws, &frames[si]).map_err(BindError::Locate)?;
                    route_batch_c(&idx, &frames[si], &endpoints, &mut out).map_err(BindError::C)?
                }
            };
            stats.locate_calls += st.locate_calls;
            for i in 0..endpoints.len() {
                if routed[i] != UNROUTED || out[i] == COORD_UNMAPPED {
                    continue;
                }
                if out[i] == COORD_AMBIGUOUS {
                    return Err(BindError::Ambiguous(endpoints[i]));
                }
                routed[i] = noderef(si, out[i] as usize);
            }
        }
        let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(edges.len());
        for e in 0..edges.len() {
            let (s, d) = (routed[2 * e], routed[2 * e + 1]);
            if s == UNROUTED || d == UNROUTED {
                stats.dropped_edges += 1;
            } else {
                pairs.push((s, d));
            }
        }
        let mut graphs = Vec::new();
        let mut valid = Vec::new();
        let mut lane_a = Vec::new();
        let mut lane_b = Vec::new();
        let mut a = Vec::new();
        let mut b = Vec::new();
        for (si, ws) in sets.iter().enumerate() {
            let n = ws.n_coords();
            graphs.push(CoordCsr::from_routed(si, n, &pairs));
            let v = ws.valid_mask(&frames[si]);
            let la = ws.gather_lane(&frames[si], a_of);
            let lb = ws.gather_lane(&frames[si], b_of);
            let mut ma = vec![0u64; words_for(n)];
            let mut mb = vec![0u64; words_for(n)];
            virtual_where(&la, a_lt, &v, &mut ma);
            virtual_where(&lb, b_lt, &v, &mut mb);
            valid.push(v);
            lane_a.push(la);
            lane_b.push(lb);
            a.push(ma);
            b.push(mb);
        }
        Ok(Binding {
            frames,
            graphs,
            valid,
            lane_a,
            lane_b,
            a,
            b,
            stats,
        })
    }

    /// A B-side parameter change: re-run only the virtual WHERE.
    pub fn rewhere_b(&mut self, b_lt: i32) {
        for si in 0..self.b.len() {
            virtual_where(&self.lane_b[si], b_lt, &self.valid[si], &mut self.b[si]);
        }
    }

    /// The version guard: every frame the binding was built under must still
    /// be the current seal of its set. `enabled = false` is the `bindguard`
    /// disable run (the guard never refuses).
    pub fn guard(&self, sets: &[&WorkingSet], enabled: bool) -> Result<(), BindError> {
        if !enabled {
            return Ok(());
        }
        for (si, ws) in sets.iter().enumerate() {
            if !self.frames[si].same_issue(&ws.seal()) {
                return Err(BindError::Stale(si));
            }
        }
        Ok(())
    }

    /// Presence-only update: clear one edge `(set si, src coord x) -> dst`.
    pub fn delete(&mut self, si: usize, x: usize, dst: u32) -> bool {
        let t = tgt_pack(si, dst);
        self.graphs[si].delete(x, t)
    }
}

/// Arm C routing of one global id across sets (index per set): the noderef.
pub fn route_global_c(idx: &[CoordIndex], frames: &[Frame], g: u64) -> Option<u32> {
    for (si, ix) in idx.iter().enumerate() {
        if let Ok(c) = ix.lookup(&frames[si], g) {
            return Some(noderef(si, c as usize));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// virtual WHERE
// ---------------------------------------------------------------------------

/// The virtual WHERE over a working set: `lane[c] < lt` at every coordinate
/// that holds an identity (`valid`), as a coordinate mask. `out` has
/// `words_for(lane.len())` words.
pub fn virtual_where(lane: &[i32], lt: i32, valid: &[u64], out: &mut [u64]) {
    ndarray::simd::lt_i32_to_mask(lane, lt, out);
    ndarray::simd::mask_and_assign(out, valid);
}

/// One Quack query over one `i32` lane of `n_rows` rows, lowered once (the
/// global scan arm of `discover`, and `Agg::Count` for G1).
pub struct Prepared {
    program: Program,
    scratch: Scratch<'static>,
    n_rows: usize,
}

impl Prepared {
    pub fn new(cmp: Cmp, agg: Agg, n_rows: usize) -> Prepared {
        let q = Query {
            filter: Filter::cmp(Col(0), cmp),
            agg,
        };
        let program = lower(&q).expect("quack lowers");
        let scratch = Scratch::for_program(&program, n_rows).expect("scratch carves");
        Prepared {
            program,
            scratch,
            n_rows,
        }
    }
    /// `Agg::Rows`: the kept mask lands in `out` (`words_for(n_rows)` long).
    pub fn keep(&mut self, lane: &[i32], live: &[u64], out: &mut [u64]) {
        let masks: [&[u64]; 1] = [live];
        let lanes = [LaneRef::I32(lane)];
        let planes = Planes {
            n_rows: self.n_rows,
            masks: &masks,
            lanes: &lanes,
        };
        execute_into(&self.program, &planes, &Foreign::NONE, &mut self.scratch, Out::Mask(out))
            .expect("mask-risc runs");
    }
    /// `Agg::Count`.
    pub fn count(&mut self, lane: &[i32], live: &[u64]) -> usize {
        let masks: [&[u64]; 1] = [live];
        let lanes = [LaneRef::I32(lane)];
        let planes = Planes {
            n_rows: self.n_rows,
            masks: &masks,
            lanes: &lanes,
        };
        match execute_into(&self.program, &planes, &Foreign::NONE, &mut self.scratch, Out::None)
            .expect("mask-risc runs")
        {
            Value::Count(c) => c,
            other => panic!("expected Value::Count, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// folds over the coordinate CSR (no allocation; scratch is passed in)
// ---------------------------------------------------------------------------

/// Reusable masks and queues for the folds, sized once per binding.
pub struct FoldScratch {
    pub m1: Vec<Vec<u64>>,
    pub m2: Vec<Vec<u64>>,
    pub m3: Vec<Vec<u64>>,
    pub seen: Vec<Vec<u64>>,
    pub q_a: Vec<u32>,
    pub q_b: Vec<u32>,
}

impl FoldScratch {
    /// `sizes[i]` = coordinates of set `i`.
    pub fn new(sizes: &[usize]) -> FoldScratch {
        let mk = || sizes.iter().map(|&n| vec![0u64; words_for(n)]).collect::<Vec<_>>();
        let total: usize = sizes.iter().sum();
        FoldScratch {
            m1: mk(),
            m2: mk(),
            m3: mk(),
            seen: mk(),
            q_a: Vec::with_capacity(total),
            q_b: Vec::with_capacity(total),
        }
    }
}

fn pop_all(m: &[Vec<u64>]) -> i64 {
    m.iter().map(|w| popcount(w) as i64).sum()
}

/// `|a AND m|` summed over sets, using `tmp` (distinct from `a` and `m`).
pub fn and_count(a: &[Vec<u64>], m: &[Vec<u64>], tmp: &mut [Vec<u64>]) -> i64 {
    let mut total = 0i64;
    for si in 0..a.len() {
        ndarray::simd::mask_and(&a[si], &m[si], &mut tmp[si]);
        total += popcount(&tmp[si]) as i64;
    }
    total
}

/// `out[x]` = x has a valid edge into `target` (row pass over every row).
pub fn semijoin(g: &[CoordCsr], target: &[Vec<u64>], out: &mut [Vec<u64>]) {
    for (si, c) in g.iter().enumerate() {
        out[si].fill(0);
        for x in 0..c.n {
            for e in c.off[x] as usize..c.off[x + 1] as usize {
                if bit(&c.valid, e) {
                    let n = tgt_unpack(si, c.tgt[e]);
                    if bit(&target[nr_set(n)], nr_coord(n)) {
                        set_bit(&mut out[si], x);
                        break;
                    }
                }
            }
        }
    }
}

/// `dst` = the 1-hop neighbour set of `src` over valid edges (scatter OR).
pub fn step(g: &[CoordCsr], src: &[Vec<u64>], dst: &mut [Vec<u64>]) {
    for d in dst.iter_mut() {
        d.fill(0);
    }
    for (si, c) in g.iter().enumerate() {
        for_each_bit(&src[si], |x| {
            for e in c.off[x] as usize..c.off[x + 1] as usize {
                if bit(&c.valid, e) {
                    let n = tgt_unpack(si, c.tgt[e]);
                    set_bit(&mut dst[nr_set(n)], nr_coord(n));
                }
            }
        });
    }
}

fn hits_b(g: &[CoordCsr], si: usize, x: usize, b: &[Vec<u64>]) -> bool {
    let c = &g[si];
    for e in c.off[x] as usize..c.off[x + 1] as usize {
        if bit(&c.valid, e) {
            let n = tgt_unpack(si, c.tgt[e]);
            if bit(&b[nr_set(n)], nr_coord(n)) {
                return true;
            }
        }
    }
    false
}

fn hits_b2(g: &[CoordCsr], si: usize, x: usize, b: &[Vec<u64>]) -> bool {
    let c = &g[si];
    for e in c.off[x] as usize..c.off[x + 1] as usize {
        if bit(&c.valid, e) {
            let n = tgt_unpack(si, c.tgt[e]);
            if hits_b(g, nr_set(n), nr_coord(n), b) {
                return true;
            }
        }
    }
    false
}

/// W2, direct: for each x in A, early-exit scan of its edges into B.
pub fn w2(g: &[CoordCsr], a: &[Vec<u64>], b: &[Vec<u64>]) -> i64 {
    let mut count = 0i64;
    for si in 0..g.len() {
        for_each_bit(&a[si], |x| {
            if hits_b(g, si, x, b) {
                count += 1;
            }
        });
    }
    count
}

/// W2 over ONE set (set 0) with a BORROWED A mask: the fold borrows the
/// selection's mask (EXO-46), nothing is copied or passed between programs.
pub fn w2_borrowed(g: &[CoordCsr], a: &[u64], b: &[Vec<u64>]) -> i64 {
    let mut count = 0i64;
    for_each_bit(a, |x| {
        if hits_b(g, 0, x, b) {
            count += 1;
        }
    });
    count
}

/// W2 as mask algebra: `popcount(A AND semijoin(B))`.
pub fn w2_and(g: &[CoordCsr], a: &[Vec<u64>], b: &[Vec<u64>], sc: &mut FoldScratch) -> i64 {
    semijoin(g, b, &mut sc.m1);
    and_count(a, &sc.m1, &mut sc.m2)
}

/// W2 through XOR: `|A AND S| = (|A| + |S| - |A XOR S|) / 2`.
pub fn w2_xor(g: &[CoordCsr], a: &[Vec<u64>], b: &[Vec<u64>], sc: &mut FoldScratch) -> i64 {
    semijoin(g, b, &mut sc.m1);
    let mut x = 0i64;
    for si in 0..a.len() {
        ndarray::simd::mask_xor(&a[si], &sc.m1[si], &mut sc.m2[si]);
        x += popcount(&sc.m2[si]) as i64;
    }
    (pop_all(a) + pop_all(&sc.m1) - x) / 2
}

/// W3, DFS with early exit per source.
pub fn w3_csr(g: &[CoordCsr], a: &[Vec<u64>], b: &[Vec<u64>]) -> i64 {
    let mut count = 0i64;
    for si in 0..g.len() {
        for_each_bit(&a[si], |x| {
            if hits_b2(g, si, x, b) {
                count += 1;
            }
        });
    }
    count
}

/// W3 by frontier masks: `popcount(A AND semijoin(semijoin(B)))`.
pub fn w3_front(g: &[CoordCsr], a: &[Vec<u64>], b: &[Vec<u64>], sc: &mut FoldScratch) -> i64 {
    semijoin(g, b, &mut sc.m1);
    semijoin(g, &sc.m1, &mut sc.m2);
    and_count(a, &sc.m2, &mut sc.m3)
}

/// W4, BFS per level with a per-level visited bitmap; the answer counts the
/// distinct nodes at exactly 3 hops that are in B.
pub fn w4_csr(g: &[CoordCsr], a: &[Vec<u64>], b: &[Vec<u64>], sc: &mut FoldScratch) -> i64 {
    let FoldScratch { q_a, q_b, seen, .. } = sc;
    q_a.clear();
    for si in 0..g.len() {
        for_each_bit(&a[si], |x| q_a.push(noderef(si, x)));
    }
    for _ in 0..3 {
        for s in seen.iter_mut() {
            s.fill(0);
        }
        q_b.clear();
        for &n in q_a.iter() {
            let (si, x) = (nr_set(n), nr_coord(n));
            let c = &g[si];
            for e in c.off[x] as usize..c.off[x + 1] as usize {
                if bit(&c.valid, e) {
                    let m = tgt_unpack(si, c.tgt[e]);
                    let (ms, mc) = (nr_set(m), nr_coord(m));
                    if !bit(&seen[ms], mc) {
                        set_bit(&mut seen[ms], mc);
                        q_b.push(m);
                    }
                }
            }
        }
        std::mem::swap(q_a, q_b);
    }
    q_a.iter().filter(|&&n| bit(&b[nr_set(n)], nr_coord(n))).count() as i64
}

/// W4 by frontier masks: three scatter-OR steps, then `popcount(F3 AND B)`.
pub fn w4_front(g: &[CoordCsr], a: &[Vec<u64>], b: &[Vec<u64>], sc: &mut FoldScratch) -> i64 {
    for si in 0..a.len() {
        sc.m1[si].copy_from_slice(&a[si]);
    }
    step(g, &sc.m1, &mut sc.m2);
    step(g, &sc.m2, &mut sc.m3);
    step(g, &sc.m3, &mut sc.m1);
    and_count(&sc.m1, b, &mut sc.m2)
}

/// W5: distinct y in B with an edge from A (1 hop).
pub fn w5(g: &[CoordCsr], a: &[Vec<u64>], b: &[Vec<u64>], sc: &mut FoldScratch) -> i64 {
    step(g, a, &mut sc.m1);
    and_count(&sc.m1, b, &mut sc.m2)
}

/// W6: bag 2-path count, checked i64 (`None` = overflow).
pub fn w6(g: &[CoordCsr], a: &[Vec<u64>], b: Option<&[Vec<u64>]>) -> Option<i64> {
    let mut total = 0i64;
    let mut ovf = false;
    for si in 0..g.len() {
        let c = &g[si];
        for_each_bit(&a[si], |x| {
            for e in c.off[x] as usize..c.off[x + 1] as usize {
                if !bit(&c.valid, e) {
                    continue;
                }
                let y = tgt_unpack(si, c.tgt[e]);
                let (ys, yc) = (nr_set(y), nr_coord(y));
                let c1 = &g[ys];
                let mut cnt = 0i64;
                for e2 in c1.off[yc] as usize..c1.off[yc + 1] as usize {
                    if !bit(&c1.valid, e2) {
                        continue;
                    }
                    match b {
                        None => cnt += 1,
                        Some(bm) => {
                            let z = tgt_unpack(ys, c1.tgt[e2]);
                            if bit(&bm[nr_set(z)], nr_coord(z)) {
                                cnt += 1;
                            }
                        }
                    }
                }
                match total.checked_add(cnt) {
                    Some(t) => total = t,
                    None => ovf = true,
                }
            }
        });
    }
    if ovf {
        None
    } else {
        Some(total)
    }
}

// ---------------------------------------------------------------------------
// W4 tile: dense row masks, only when the total active coordinates <= 4096
// ---------------------------------------------------------------------------

/// Dense adjacency rows over all sets' coordinates (`off[si] + coord`).
pub struct TileMatrix {
    pub total: usize,
    wpr: usize,
    off: Vec<usize>,
    rows: Vec<u64>,
    f1: Vec<u64>,
    f2: Vec<u64>,
    a: Vec<u64>,
    b: Vec<u64>,
}

impl TileMatrix {
    /// `None` when the total coordinate count exceeds 4096.
    pub fn build(g: &[CoordCsr]) -> Option<TileMatrix> {
        let total: usize = g.iter().map(|c| c.n).sum();
        if total > 4096 {
            return None;
        }
        let wpr = words_for(total.max(1));
        let mut off = Vec::with_capacity(g.len());
        let mut acc = 0usize;
        for c in g {
            off.push(acc);
            acc += c.n;
        }
        let mut rows = vec![0u64; total.max(1) * wpr];
        for (si, c) in g.iter().enumerate() {
            for x in 0..c.n {
                for e in c.off[x] as usize..c.off[x + 1] as usize {
                    if bit(&c.valid, e) {
                        let n = tgt_unpack(si, c.tgt[e]);
                        let d = off[nr_set(n)] + nr_coord(n);
                        let r = off[si] + x;
                        set_bit(&mut rows[r * wpr..(r + 1) * wpr], d);
                    }
                }
            }
        }
        Some(TileMatrix {
            total,
            wpr,
            off,
            rows,
            f1: vec![0; wpr],
            f2: vec![0; wpr],
            a: vec![0; wpr],
            b: vec![0; wpr],
        })
    }
    /// Presence-only update on the dense rows.
    pub fn clear_edge(&mut self, si: usize, x: usize, dst: u32) {
        let r = self.off[si] + x;
        let d = self.off[nr_set(dst)] + nr_coord(dst);
        let wpr = self.wpr;
        clear_bit(&mut self.rows[r * wpr..(r + 1) * wpr], d);
    }
    /// Scatter the per-set A and B masks into the dense space.
    pub fn load(&mut self, a: &[Vec<u64>], b: &[Vec<u64>]) {
        self.a.fill(0);
        self.b.fill(0);
        for si in 0..a.len() {
            let o = self.off[si];
            let (da, db) = (&mut self.a, &mut self.b);
            for_each_bit(&a[si], |x| set_bit(da, o + x));
            for_each_bit(&b[si], |x| set_bit(db, o + x));
        }
    }
    /// W4: three dense OR-of-rows steps, then `popcount(F3 AND B)`.
    pub fn fold(&mut self) -> i64 {
        let wpr = self.wpr;
        let TileMatrix { rows, f1, f2, a, b, .. } = self;
        f1.copy_from_slice(a);
        for step_no in 0..3 {
            let (src, dst) = if step_no % 2 == 0 { (&*f1, &mut *f2) } else { (&*f2, &mut *f1) };
            dst.fill(0);
            for_each_bit(src, |x| ndarray::simd::mask_or_assign(dst, &rows[x * wpr..(x + 1) * wpr]));
        }
        // three steps: f1 -> f2 -> f1 -> f2; the result is f2.
        ndarray::simd::mask_and(f2, b, f1);
        popcount(f1) as i64
    }
}

// ---------------------------------------------------------------------------
// scalar oracle over the global edge list (HashSet/HashMap allowed: oracle)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Oracle {
    pub w2: i64,
    pub w3: i64,
    pub w4: i64,
    pub w5: i64,
    /// Bag 2-path count without a B filter.
    pub w6: i64,
    /// Bag 2-path count with z in B (SQL G6).
    pub w6b: i64,
}

pub fn oracle(
    edges: &[(u64, u64)],
    in_a: &dyn Fn(u64) -> bool,
    in_b: &dyn Fn(u64) -> bool,
    deleted: &[(u64, u64)],
) -> Oracle {
    let del: HashSet<(u64, u64)> = deleted.iter().copied().collect();
    let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
    for &(s, d) in edges {
        if !del.contains(&(s, d)) {
            adj.entry(s).or_default().push(d);
        }
    }
    let a_nodes: Vec<u64> = adj.keys().copied().filter(|&x| in_a(x)).collect();
    let nbrs = |x: u64| adj.get(&x).map(|v| v.as_slice()).unwrap_or(&[]);
    let w2 = a_nodes.iter().filter(|&&x| nbrs(x).iter().any(|&y| in_b(y))).count() as i64;
    let w3 = a_nodes
        .iter()
        .filter(|&&x| nbrs(x).iter().any(|&y| nbrs(y).iter().any(|&z| in_b(z))))
        .count() as i64;
    let mut front: HashSet<u64> = a_nodes.iter().copied().collect();
    let mut reach1: HashSet<u64> = HashSet::new();
    for hop in 0..3 {
        let mut next: HashSet<u64> = HashSet::new();
        for &x in &front {
            next.extend(nbrs(x).iter().copied());
        }
        if hop == 0 {
            reach1 = next.clone();
        }
        front = next;
    }
    let w4 = front.iter().filter(|&&z| in_b(z)).count() as i64;
    let w5 = reach1.iter().filter(|&&y| in_b(y)).count() as i64;
    let mut w6 = 0i64;
    let mut w6b = 0i64;
    for &x in &a_nodes {
        for &y in nbrs(x) {
            w6 += nbrs(y).len() as i64;
            w6b += nbrs(y).iter().filter(|&&z| in_b(z)).count() as i64;
        }
    }
    Oracle { w2, w3, w4, w5, w6, w6b }
}

// ---------------------------------------------------------------------------
// EXO-46 shared selection (LAB)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum ShareRefusal {
    /// dt64's `BindStore::admit` refused the method reference.
    Bind(Refusal),
    /// The method reference is for another class than the selection's.
    ClassMismatch { selection: u32, asked: u32 },
    /// The method presented a permit that does not hold its reference
    /// (a permit is per method; it is never inherited).
    NotPermitted,
    Scope,
    /// The selection's frame is not the table's current seal.
    StaleFrame,
}

/// A sealed selection one working set owns: the frame it was evaluated under,
/// the scope it was authorized for, a criteria tag and the mask. Methods
/// borrow the mask through `admit`; no mask is passed between programs.
#[derive(Debug, Clone)]
pub struct SharedSelection {
    pub class: u32,
    pub frame: Frame,
    pub scope: Scope,
    pub criteria: u64,
    pub mask: Vec<u64>,
}

impl SharedSelection {
    /// Frame currency is `frame.same_issue(&current)`, `current` being
    /// `table.seal()`: the seal is replaced by every `advance`, so a stale
    /// frame fails even when numbers are equal (an identity compare of issues,
    /// never a digest compare). Order: dt64 admit (class bound, method known,
    /// sig rev, guard), class match, the method's own permit, scope, frame.
    /// Allocation-free on the success path (unguarded methods).
    pub fn admit(
        &self,
        store: &BindStore,
        mref: &MethodRef,
        permit: &[MethodRef],
        current: &Frame,
        scope_now: Scope,
    ) -> Result<&[u64], ShareRefusal> {
        store.admit(mref).map_err(ShareRefusal::Bind)?;
        if mref.class != self.class {
            return Err(ShareRefusal::ClassMismatch {
                selection: self.class,
                asked: mref.class,
            });
        }
        if !permit.contains(mref) {
            return Err(ShareRefusal::NotPermitted);
        }
        if self.scope != scope_now {
            return Err(ShareRefusal::Scope);
        }
        if !self.frame.same_issue(current) {
            return Err(ShareRefusal::StaleFrame);
        }
        Ok(&self.mask)
    }
}

// ---------------------------------------------------------------------------
// multi-domain helpers
// ---------------------------------------------------------------------------

/// The global-id union check for domains (working sets) that should be
/// disjoint: `Ok(total ids)` or `Err(first id held by two members)`. It reads
/// the pointer tables, never a second identity table.
pub fn union_check(sets: &[&WorkingSet]) -> Result<usize, u64> {
    let mut all: Vec<u64> = Vec::new();
    for ws in sets {
        let f = ws.seal();
        for &t in &ws.tile_ids {
            if let Ok(v) = ws.table.map(&f, t) {
                all.extend_from_slice(v.globals());
            }
        }
    }
    all.sort_unstable();
    for w in all.windows(2) {
        if w[0] == w[1] {
            return Err(w[0]);
        }
    }
    Ok(all.len())
}

/// Checked sum of per-domain contributions (`None` on overflow).
pub fn checked_sum(parts: &[i64]) -> Option<i64> {
    parts.iter().try_fold(0i64, |acc, &p| acc.checked_add(p))
}
