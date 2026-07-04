//! Lane J — the parameterized batch pipeline: lane I's shape with the
//! three knobs the operator's follow-up questions name (2026-07-02):
//!
//! > *"but with Orchestrator it was 39.4? / is the batch writer with
//! > different cache? / does it need 8, or 64 lanes? / or is
//! > Orchestration with 400 the sweet spot / or should we assign 8x8 or
//! > 64x64 gridlake soa / what if we match the soa into a grid
//! > 64x64 = 4096 xBF16 = 16kb?"*
//!
//! One lane, three knobs, so every question is a measured cell of the
//! same matrix rather than a new lane:
//!
//! 1. **`grid`** — the SoA address-space size: 65536 (lane I's 256×256)
//!    or **4096 (the 64×64 gridlake tile)**. The gridlake batch table
//!    is 4096 cells × (i32 min + i32 max + i64 sum + u32 count) = 80 KB
//!    integer-exact — the cache-matched unit (vs lane I's 1.25 MB
//!    L2-busting table + a 64K-slot memo per worker). The literal
//!    "4096 × BF16 = 16 KB" plane pair is ndarray #227's PROVEN tier
//!    (`bf16_tile_gemm` VDPBF16PS ladder; its `onebrc_cascade_probe`
//!    measured the 64×64 Z-order grid at ~448 Mrows/s single-thread,
//!    bit-exact via the hi/lo split) — this probe keeps integer tenths
//!    for exactness-without-tile-GEMM and cites that example as the
//!    BF16 continuation.
//! 2. **`sink_lanes`** — 1 / 8 / 64 ownership+lance lane PAIRS, each
//!    owning a contiguous row-range slice of the grid; every frozen
//!    batch `Arc` is cast to ALL lanes (messages = batches × 2·lanes,
//!    still ∝ batches); each lane merges only its range's dirty rows.
//!    Answers "does the batch writer need 8, or 64 lanes?".
//! 3. **`registry`** — spawn the full upfront `RowOwner` mailbox
//!    registry (one per grid cell) or skip it. The pipeline is
//!    byte-identical either way, so `registry off − on` ISOLATES the
//!    residency cost that t6 could only flag as CONJECTURE.
//!
//! Everything else is lane I verbatim: codebook-minted direct CAM
//! addressing (mint once, memoized, no probe in the hot loop),
//! whole-table `Arc` double-casts, refcount-gated flush cache
//! (flush/refill interleave), per-batch witness on every sink lane
//! (`Σ ownership journals == Σ lance journals == batches × lanes`
//! asserted), one `DatasetVersion`-shaped tick per batch per lance
//! lane.

use crate::lane_f::{fnv1a64, morton_slot};
use crate::lane_i::RowOwner;
use crate::{chunk_bounds, merge_maps, parse_temp_tenths, Stats};
use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
use ndarray::simd::MultiLaneColumn;
use ractor::{Actor, ActorProcessingErr, ActorRef, RpcReplyPort};
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ─── Grid-sized codebook + memo (runtime capacity, else lane I's) ───────

/// Mint registry over a runtime-sized grid (power of two): identity →
/// unique cell, collisions resolved once at mint.
struct GridCodebook {
    mask: usize,
    tags: Vec<u64>,
    names: Vec<Vec<u8>>,
    used: Vec<bool>,
    len: usize,
}

impl GridCodebook {
    fn new(grid: usize) -> Self {
        assert!(grid.is_power_of_two(), "grid must be a power of two");
        Self {
            mask: grid - 1,
            tags: vec![0; grid],
            names: vec![Vec::new(); grid],
            used: vec![false; grid],
            len: 0,
        }
    }

    fn mint(&mut self, h: u64, name: &[u8]) -> u16 {
        let mut s = morton_slot(h) as usize & self.mask;
        loop {
            if !self.used[s] {
                self.used[s] = true;
                self.tags[s] = h;
                self.names[s] = name.to_vec();
                self.len += 1;
                return s as u16;
            }
            if self.tags[s] == h && self.names[s] == name {
                return s as u16;
            }
            s = (s + 1) & self.mask;
        }
    }
}

/// Worker-local `hash → cell` memo over the same grid.
struct GridMemo {
    mask: usize,
    tags: Vec<u64>,
    names: Vec<Vec<u8>>,
    slots: Vec<u16>,
    used: Vec<bool>,
}

impl GridMemo {
    fn new(grid: usize) -> Self {
        Self {
            mask: grid - 1,
            tags: vec![0; grid],
            names: vec![Vec::new(); grid],
            slots: vec![0; grid],
            used: vec![false; grid],
        }
    }

    #[inline(always)]
    fn resolve(&mut self, h: u64, name: &[u8], global: &Mutex<GridCodebook>) -> u16 {
        let mut s = morton_slot(h) as usize & self.mask;
        loop {
            if !self.used[s] {
                let slot = global.lock().expect("codebook lock").mint(h, name);
                self.used[s] = true;
                self.tags[s] = h;
                self.names[s] = name.to_vec();
                self.slots[s] = slot;
                return slot;
            }
            if self.tags[s] == h && self.names[s] == name {
                return self.slots[s];
            }
            s = (s + 1) & self.mask;
        }
    }
}

// ─── Grid batch table (the gridlake SoA unit at grid=4096: ~80 KB) ──────

pub struct GridBatch {
    mins: Vec<i32>,
    maxs: Vec<i32>,
    sums: Vec<i64>,
    counts: Vec<u32>,
    dirty: Vec<u16>,
}

impl GridBatch {
    fn new(grid: usize) -> Self {
        Self {
            mins: vec![i32::MAX; grid],
            maxs: vec![i32::MIN; grid],
            sums: vec![0; grid],
            counts: vec![0; grid],
            dirty: Vec::with_capacity(1024),
        }
    }

    #[inline(always)]
    fn observe(&mut self, slot: u16, tenths: i32) {
        let s = slot as usize;
        if self.counts[s] == 0 {
            self.dirty.push(slot);
        }
        if tenths < self.mins[s] {
            self.mins[s] = tenths;
        }
        if tenths > self.maxs[s] {
            self.maxs[s] = tenths;
        }
        self.sums[s] += tenths as i64;
        self.counts[s] += 1;
    }

    fn reset(&mut self) {
        for &slot in &self.dirty {
            let s = slot as usize;
            self.mins[s] = i32::MAX;
            self.maxs[s] = i32::MIN;
            self.sums[s] = 0;
            self.counts[s] = 0;
        }
        self.dirty.clear();
    }
}

// ─── Gridlake carrier: the batch table AS ndarray MultiLaneColumns ──────

/// The lane-J `GridBatch` accumulators rendered as `ndarray::simd`
/// [`MultiLaneColumn`] gridlake carriers — the SoA-contract carrier the
/// proven 64×64 gridlake tile rides (`E-1BRC-GRIDLAKE-SWEETSPOT-1`), and the
/// **same** `MultiLaneColumn` the COCA cognitive `Cell`
/// (helix48/campq48/count/truth, `crates/deepnsm/examples/gridlake_coca_wire.rs`)
/// composes from. This is the DeepNSM→V3 D-DNV-1 recognition
/// (`.claude/plans/deepnsm-v3-convergence-v1.md`,
/// `E-V3-DEEPNSM-IS-THE-ENCODER-NOT-A-MIGRATION-1`): the batch table is not a
/// bespoke struct, it is typed lanes over one carrier — "wire, don't invent."
///
/// Lane widths follow the integer lanes ndarray added for exactly this
/// (`iter_i32x16` "min/max tile columns", `iter_i64x8` "running sums"):
/// min/max ride `I32x16`, sum rides `I64x8`, count (a non-negative
/// accumulator) rides `U64x8`. Each column's backing buffer is a 64-byte
/// multiple whenever `grid` is a multiple of 16 (i32·16 = i64·8 = u64·8 =
/// 64 B), which the gridlake `grid = 4096` satisfies.
pub struct GridlakeColumns {
    pub mins: MultiLaneColumn,
    pub maxs: MultiLaneColumn,
    pub sums: MultiLaneColumn,
    pub counts: MultiLaneColumn,
}

/// Failure rendering a [`GridBatch`] as gridlake [`MultiLaneColumn`]s.
///
/// A plain enum rather than a `snafu` type on purpose: `onebrc-probe` is a
/// workspace-excluded standalone probe whose lanes A/C are dependency-free by
/// design (see its `Cargo.toml`) — it carries no `snafu`. This surfaces the
/// same alignment failure `MultiLaneColumn::new` reports, but with the
/// offending `grid` for diagnostics instead of a bare `()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridlakeCarrierError {
    /// A column buffer was not 64-byte aligned because `grid` is not a
    /// multiple of 16 (i32·16 = i64·8 = u64·8 = 64 B), so
    /// `MultiLaneColumn::new` rejected it.
    UnalignedGrid { grid: usize },
}

impl std::fmt::Display for GridlakeCarrierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnalignedGrid { grid } => write!(
                f,
                "gridlake carrier: grid {grid} is not a multiple of 16, so a \
                 column buffer is not 64-byte aligned for MultiLaneColumn"
            ),
        }
    }
}

impl std::error::Error for GridlakeCarrierError {}

impl GridBatch {
    /// Render the four accumulator columns as [`MultiLaneColumn`] gridlake
    /// carriers (little-endian bytes, zero semantic change — a *reading*, not
    /// a re-layout). `count` is widened `u32 → u64` to ride the unsigned
    /// 64-bit accumulator lane. Returns
    /// [`GridlakeCarrierError::UnalignedGrid`] if a column buffer is not
    /// 64-byte aligned (i.e. `grid % 16 != 0`), surfacing the alignment
    /// contract `MultiLaneColumn::new` enforces with the offending grid size.
    pub fn as_gridlake_columns(&self) -> Result<GridlakeColumns, GridlakeCarrierError> {
        let grid = self.mins.len();
        let unaligned = |_: ()| GridlakeCarrierError::UnalignedGrid { grid };
        fn col_i32(v: &[i32]) -> Result<MultiLaneColumn, ()> {
            let mut b = Vec::with_capacity(v.len() * 4);
            for &x in v {
                b.extend_from_slice(&x.to_le_bytes());
            }
            MultiLaneColumn::new(Arc::from(b))
        }
        fn col_i64(v: &[i64]) -> Result<MultiLaneColumn, ()> {
            let mut b = Vec::with_capacity(v.len() * 8);
            for &x in v {
                b.extend_from_slice(&x.to_le_bytes());
            }
            MultiLaneColumn::new(Arc::from(b))
        }
        fn col_u64_from_u32(v: &[u32]) -> Result<MultiLaneColumn, ()> {
            let mut b = Vec::with_capacity(v.len() * 8);
            for &x in v {
                b.extend_from_slice(&(x as u64).to_le_bytes());
            }
            MultiLaneColumn::new(Arc::from(b))
        }
        Ok(GridlakeColumns {
            mins: col_i32(&self.mins).map_err(unaligned)?,
            maxs: col_i32(&self.maxs).map_err(unaligned)?,
            sums: col_i64(&self.sums).map_err(unaligned)?,
            counts: col_u64_from_u32(&self.counts).map_err(unaligned)?,
        })
    }
}

// ─── Laned sinks: each lane owns a contiguous row-range slice ───────────

enum LaneMsg {
    Batch {
        table: Arc<GridBatch>,
    },
    /// Ownership lanes reply `(partial map, journal_len)`; lance lanes
    /// reply an empty map + journal_len (address table summarized via
    /// journal/version equality asserts).
    Finish {
        reply: RpcReplyPort<(BTreeMap<String, Stats>, usize)>,
    },
}

/// One ownership-side lane: the canonical SoA slice for rows
/// `[lo, hi)`, merged O(dirty-in-range) per batch, witnessed per batch.
struct OwnershipLaneState {
    lo: usize,
    hi: usize,
    mins: Vec<i32>,
    maxs: Vec<i32>,
    sums: Vec<i64>,
    counts: Vec<u32>,
    journal: Vec<KanbanMove>,
    codebook: Arc<Mutex<GridCodebook>>,
}

struct OwnershipLane;

impl Actor for OwnershipLane {
    type Msg = LaneMsg;
    type State = OwnershipLaneState;
    /// `(lo, hi, grid, codebook)`.
    type Arguments = (usize, usize, usize, Arc<Mutex<GridCodebook>>);

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        (lo, hi, grid, codebook): Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        let _ = grid;
        Ok(OwnershipLaneState {
            lo,
            hi,
            mins: vec![i32::MAX; hi - lo],
            maxs: vec![i32::MIN; hi - lo],
            sums: vec![0; hi - lo],
            counts: vec![0; hi - lo],
            journal: Vec::new(),
            codebook,
        })
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        msg: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match msg {
            LaneMsg::Batch { table } => {
                for &slot in &table.dirty {
                    let s = slot as usize;
                    if s < state.lo || s >= state.hi {
                        continue; // another lane's row range
                    }
                    let l = s - state.lo;
                    if table.mins[s] < state.mins[l] {
                        state.mins[l] = table.mins[s];
                    }
                    if table.maxs[s] > state.maxs[l] {
                        state.maxs[l] = table.maxs[s];
                    }
                    state.sums[l] += table.sums[s];
                    state.counts[l] += table.counts[s];
                }
                let pos = state.journal.len() as u32 + 1;
                state.journal.push(KanbanMove {
                    mailbox: state.lo as u32,
                    from: KanbanColumn::CognitiveWork,
                    to: KanbanColumn::Evaluation,
                    witness_chain_position: pos,
                    libet_offset_us: 0,
                    exec: ExecTarget::Native,
                });
            }
            LaneMsg::Finish { reply } => {
                let cb = state.codebook.lock().expect("codebook lock");
                let mut out = BTreeMap::new();
                for l in 0..(state.hi - state.lo) {
                    if state.counts[l] > 0 {
                        let name = String::from_utf8(cb.names[state.lo + l].clone())
                            .expect("station name utf8");
                        out.insert(
                            name,
                            Stats {
                                min: state.mins[l],
                                max: state.maxs[l],
                                sum: state.sums[l],
                                count: state.counts[l],
                            },
                        );
                    }
                }
                let _ = reply.send((out, state.journal.len()));
            }
        }
        Ok(())
    }
}

/// One lance-side lane: `row → latest version` for its range + one
/// version tick per batch.
struct LanceLaneState {
    lo: usize,
    hi: usize,
    latest_version: Vec<u32>,
    version: u32,
    journal: Vec<KanbanMove>,
}

struct LanceLane;

impl Actor for LanceLane {
    type Msg = LaneMsg;
    type State = LanceLaneState;
    /// `(lo, hi)`.
    type Arguments = (usize, usize);

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        (lo, hi): Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(LanceLaneState {
            lo,
            hi,
            latest_version: vec![0; hi - lo],
            version: 0,
            journal: Vec::new(),
        })
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        msg: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match msg {
            LaneMsg::Batch { table } => {
                state.version += 1;
                for &slot in &table.dirty {
                    let s = slot as usize;
                    if s >= state.lo && s < state.hi {
                        state.latest_version[s - state.lo] = state.version;
                    }
                }
                let pos = state.journal.len() as u32 + 1;
                state.journal.push(KanbanMove {
                    mailbox: state.lo as u32,
                    from: KanbanColumn::CognitiveWork,
                    to: KanbanColumn::Evaluation,
                    witness_chain_position: pos,
                    libet_offset_us: 0,
                    exec: ExecTarget::Native,
                });
            }
            LaneMsg::Finish { reply } => {
                let _ = reply.send((BTreeMap::new(), state.journal.len()));
            }
        }
        Ok(())
    }
}

// ─── Flush cache (identical mechanism to lane I, grid-sized tables) ─────

struct GridFlushCache {
    grid: usize,
    pool: VecDeque<Arc<GridBatch>>,
    peak: usize,
    allocated: usize,
}

impl GridFlushCache {
    fn new(grid: usize) -> Self {
        Self {
            grid,
            pool: VecDeque::new(),
            peak: 0,
            allocated: 0,
        }
    }

    fn next_table(&mut self) -> GridBatch {
        if let Some(front) = self.pool.pop_front() {
            match Arc::try_unwrap(front) {
                Ok(mut table) => {
                    table.reset();
                    return table;
                }
                Err(still_in_flight) => {
                    self.pool.push_front(still_in_flight);
                }
            }
        }
        self.allocated += 1;
        self.peak = self.peak.max(self.allocated);
        GridBatch::new(self.grid)
    }

    fn park(&mut self, table: Arc<GridBatch>) {
        self.pool.push_back(table);
    }
}

// ─── The lane ────────────────────────────────────────────────────────────

/// Lane J: the parameterized batch pipeline. `grid` = SoA cell count
/// (4096 = the 64×64 gridlake tile; 65536 = lane I's full space);
/// `sink_lanes` = ownership+lance lane pairs; `registry` = spawn the
/// full upfront per-cell mailbox registry or skip it (isolates the
/// residency cost); `batch_rows` = rows per frozen batch.
pub fn lane_j_grid_pipeline_with(
    data: &[u8],
    workers: usize,
    grid: usize,
    sink_lanes: usize,
    registry: bool,
    batch_rows: usize,
) -> BTreeMap<String, Stats> {
    let workers = workers.max(1);
    let grid = grid.clamp(64, 1 << 16).next_power_of_two();
    let sink_lanes = sink_lanes.clamp(1, grid);
    let batch_rows = batch_rows.max(1);
    let bounds = chunk_bounds(data, workers);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .build()
        .expect("build tokio runtime for lane J");

    runtime.block_on(async move {
        let shared: Arc<Vec<u8>> = Arc::new(data.to_vec());
        let codebook = Arc::new(Mutex::new(GridCodebook::new(grid)));
        let batches = Arc::new(AtomicUsize::new(0));

        // Knob 3: the standing per-cell mailbox registry, or none.
        let spawn_t0 = Instant::now();
        let mut owners = Vec::new();
        if registry {
            owners.reserve(grid);
            for _ in 0..grid {
                let (actor, _handle) = Actor::spawn(None, RowOwner, ())
                    .await
                    .expect("spawn row-owner mailbox");
                owners.push(actor);
            }
        }
        let spawn_ms = spawn_t0.elapsed().as_secs_f64() * 1000.0;

        // Knob 2: laned sinks — each pair owns rows [lo, hi).
        let span = grid.div_ceil(sink_lanes);
        let mut own_lanes = Vec::with_capacity(sink_lanes);
        let mut lance_lanes = Vec::with_capacity(sink_lanes);
        let mut lane_handles = Vec::with_capacity(sink_lanes * 2);
        for l in 0..sink_lanes {
            let lo = l * span;
            let hi = ((l + 1) * span).min(grid);
            let (o, oh) = Actor::spawn(None, OwnershipLane, (lo, hi, grid, Arc::clone(&codebook)))
                .await
                .expect("spawn ownership lane");
            let (z, zh) = Actor::spawn(None, LanceLane, (lo, hi))
                .await
                .expect("spawn lance lane");
            own_lanes.push(o);
            lance_lanes.push(z);
            lane_handles.push(oh);
            lane_handles.push(zh);
        }
        let own_lanes = Arc::new(own_lanes);
        let lance_lanes = Arc::new(lance_lanes);

        let mut worker_handles = Vec::with_capacity(bounds.len());
        for &(start, end) in &bounds {
            let shared = Arc::clone(&shared);
            let codebook = Arc::clone(&codebook);
            let own_lanes = Arc::clone(&own_lanes);
            let lance_lanes = Arc::clone(&lance_lanes);
            let batches = Arc::clone(&batches);
            worker_handles.push(tokio::task::spawn_blocking(move || {
                let mut memo = GridMemo::new(grid);
                let mut cache = GridFlushCache::new(grid);
                let mut table = cache.next_table();
                let mut rows_in_batch = 0usize;

                let double_cast = |cache: &mut GridFlushCache, full: GridBatch| {
                    if full.dirty.is_empty() {
                        cache.park(Arc::new(full));
                        return;
                    }
                    let frozen = Arc::new(full);
                    for lane in own_lanes.iter() {
                        lane.cast(LaneMsg::Batch {
                            table: Arc::clone(&frozen),
                        })
                        .expect("cast to ownership lane");
                    }
                    for lane in lance_lanes.iter() {
                        lane.cast(LaneMsg::Batch {
                            table: Arc::clone(&frozen),
                        })
                        .expect("cast to lance lane");
                    }
                    batches.fetch_add(1, Ordering::Relaxed);
                    cache.park(frozen);
                };

                let data = &shared[..];
                let mut i = start;
                while i < end {
                    let name_start = i;
                    while data[i] != b';' {
                        i += 1;
                    }
                    let name = &data[name_start..i];
                    i += 1;
                    let temp_start = i;
                    while data[i] != b'\n' {
                        i += 1;
                    }
                    let tenths = parse_temp_tenths(&data[temp_start..i]);
                    i += 1;

                    let h = fnv1a64(name);
                    let slot = memo.resolve(h, name, &codebook);
                    table.observe(slot, tenths);
                    rows_in_batch += 1;
                    if rows_in_batch == batch_rows {
                        let full = std::mem::replace(&mut table, cache.next_table());
                        double_cast(&mut cache, full);
                        rows_in_batch = 0;
                    }
                }
                double_cast(&mut cache, table);
                cache.peak
            }));
        }
        let mut flush_cache_peak = 0usize;
        for h in worker_handles {
            flush_cache_peak = flush_cache_peak.max(h.await.expect("lane J worker join"));
        }

        let batches_total = batches.load(Ordering::Relaxed);
        let mut maps = Vec::with_capacity(sink_lanes);
        let mut own_journal = 0usize;
        for lane in own_lanes.iter() {
            let (map, j) = ractor::call!(lane, |reply| LaneMsg::Finish { reply })
                .expect("ownership lane finish");
            maps.push(map);
            own_journal += j;
        }
        let mut lance_journal = 0usize;
        for lane in lance_lanes.iter() {
            let (_, j) =
                ractor::call!(lane, |reply| LaneMsg::Finish { reply }).expect("lance lane finish");
            lance_journal += j;
        }
        assert_eq!(
            own_journal,
            batches_total * sink_lanes,
            "every batch witnessed on every ownership lane"
        );
        assert_eq!(
            lance_journal,
            batches_total * sink_lanes,
            "every batch witnessed on every lance lane"
        );

        eprintln!(
            "lane_j: grid={grid} sink_lanes={sink_lanes} registry={registry} \
             mailbox_spawn_ms={spawn_ms:.1} batches={batches_total} \
             flush_cache_peak={flush_cache_peak} codebook_len={}",
            codebook.lock().expect("codebook lock").len
        );

        for lane in own_lanes.iter() {
            lane.stop(None);
        }
        for lane in lance_lanes.iter() {
            lane.stop(None);
        }
        for h in lane_handles {
            h.await.expect("sink lane join");
        }
        for owner in &owners {
            owner.stop(None);
        }

        merge_maps(maps)
    })
}

/// Lane J at the gridlake defaults: 64×64 grid (4096 cells), 1 sink
/// lane pair, no standing registry, 64K-row batches.
pub fn lane_j_grid_pipeline(data: &[u8], workers: usize) -> BTreeMap<String, Stats> {
    lane_j_grid_pipeline_with(data, workers, 4096, 1, false, 1 << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// D-DNV-1 carrier foundation: the gridlake `GridBatch` renders
    /// losslessly through `ndarray::simd::MultiLaneColumn` — the LE bytes
    /// roundtrip cell-for-cell against the source accumulators, and the typed
    /// integer lanes (i32 min/max, i64 sum, u64 count) each yield the right
    /// window count over the carrier. This is the "wire, don't invent" proof
    /// that the batch table IS a MultiLaneColumn composition (the carrier the
    /// COCA cognitive Cell also rides).
    #[test]
    fn gridlake_batch_rides_multilane_column_losslessly() {
        let grid = 4096usize;
        let mut batch = GridBatch::new(grid);
        // A spread of cells with distinct signed extremes + running sums,
        // incl. the i32x16 / lane boundaries (15|16) and the tile edge (4095).
        for (k, &slot) in [0u16, 1, 15, 16, 255, 256, 4095].iter().enumerate() {
            let k = k as i32;
            batch.observe(slot, -(k * 10) - 3);
            batch.observe(slot, k * 7 + 11);
            batch.observe(slot, 5);
        }
        let cols = batch
            .as_gridlake_columns()
            .expect("grid=4096 columns are 64-byte aligned");

        // Typed-lane views are wired: 4096 i32 = 256 × i32x16;
        // 4096 i64/u64 = 512 × i64x8/u64x8 (the lanes ndarray added for this).
        assert_eq!(cols.mins.len_i32x16(), grid / 16);
        assert_eq!(cols.mins.iter_i32x16().count(), grid / 16);
        assert_eq!(cols.maxs.len_i32x16(), grid / 16);
        assert_eq!(cols.sums.len_i64x8(), grid / 8);
        assert_eq!(cols.sums.iter_i64x8().count(), grid / 8);
        assert_eq!(cols.counts.len_u64x8(), grid / 8);
        assert_eq!(cols.counts.iter_u64x8().count(), grid / 8);

        // LE roundtrip is cell-for-cell exact against the source accumulators.
        let dec_i32 = |c: &MultiLaneColumn| -> Vec<i32> {
            c.as_bytes()
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().expect("4-byte i32 chunk")))
                .collect()
        };
        let dec_i64 = |c: &MultiLaneColumn| -> Vec<i64> {
            c.as_bytes()
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes(b.try_into().expect("8-byte i64 chunk")))
                .collect()
        };
        let dec_u64 = |c: &MultiLaneColumn| -> Vec<u64> {
            c.as_bytes()
                .chunks_exact(8)
                .map(|b| u64::from_le_bytes(b.try_into().expect("8-byte u64 chunk")))
                .collect()
        };
        assert_eq!(dec_i32(&cols.mins), batch.mins);
        assert_eq!(dec_i32(&cols.maxs), batch.maxs);
        assert_eq!(dec_i64(&cols.sums), batch.sums);
        let counts_u64: Vec<u64> = batch.counts.iter().map(|&c| c as u64).collect();
        assert_eq!(dec_u64(&cols.counts), counts_u64);
    }

    /// The carrier refuses a mis-aligned grid (not a multiple of 16) rather
    /// than silently producing a non-64-byte column — the `MultiLaneColumn`
    /// contract surfaced at the batch boundary.
    #[test]
    fn gridlake_carrier_rejects_unaligned_grid() {
        let batch = GridBatch::new(72); // 72 % 16 != 0 → i32 col = 288 B, not 64-mult
        assert!(matches!(
            batch.as_gridlake_columns(),
            Err(GridlakeCarrierError::UnalignedGrid { grid: 72 })
        ));
    }

    /// Parity across the knob matrix corners: gridlake (4096) and full
    /// (65536) grids × 1 and 8 sink lanes × registry on/off, all with a
    /// small batch to force multi-batch flush-cache recycling.
    #[test]
    fn lane_j_agrees_with_lane_a_across_knob_corners() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("onebrc_probe_test_j_{}.txt", std::process::id()));
        let result = crate::gen::gen(&path, 50_000, 123).expect("gen");
        assert_eq!(result.rows, 50_000);

        let data = std::fs::read(&path).expect("read generated corpus");
        std::fs::remove_file(&path).ok();

        let a = crate::lane_a_scalar(&data);
        for (grid, lanes, registry) in [
            (4096usize, 1usize, false),
            (4096, 8, false),
            (65536, 1, false),
            (4096, 8, true),
        ] {
            let j = lane_j_grid_pipeline_with(&data, 3, grid, lanes, registry, 1000);
            assert_eq!(
                a, j,
                "lane J (grid={grid}, lanes={lanes}, registry={registry}) must match lane A"
            );
        }
        assert!(!a.is_empty());
    }
}
