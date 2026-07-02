//! Lane I — the substrate-native batch pipeline, per the operator's
//! spec (2026-07-02): all 65536 mailboxes spawned UPFRONT, two fixed
//! aligned indices (mailbox index == SoA row index), codebook-minted
//! identity for direct CAM addressing, whole-table double-casts into
//! the mailbox-ownership-guarantee table AND the Lance row-address
//! table, and a flush cache so flushing and reindexing interleave.
//!
//! ## The spec, mechanism by mechanism
//!
//! 1. **All 65536 mailboxes upfront.** [`RowOwner`] actors are standing
//!    infrastructure — the ownership REGISTRY — spawned eagerly before
//!    any data moves (spawn wall-time measured and reported separately
//!    on stderr). Unlike lanes G/H they are NOT a data path: in steady
//!    state no record, morsel, or batch is ever addressed to an
//!    individual mailbox — that fan-out was t4a's measured 20×
//!    anti-pattern.
//! 2. **Two fixed indices, aligned.** Index one: the mailbox index
//!    (0..65536, fixed at spawn). Index two: the row index of every
//!    65536-row SoA table (batch tables, the ownership table, the
//!    Lance row-address table). They are the SAME space: mailbox `i`
//!    owns row `i` in every table by index correspondence — the
//!    ownership guarantee is the ownership sink's `row_owner[i] == i`
//!    binding plus every batch applied ON BEHALF of the row owners
//!    (the write-on-behalf iron rule), not a message path.
//! 3. **Codebook index.** Station identity is MINTED once into a
//!    unique row slot ([`Codebook`]: Morton placement + linear probe +
//!    full `(h, name)` verification, at mint only). Workers memoize
//!    `hash → slot` locally, so the global mint mutex is touched only
//!    on first sight of a station (~stations per worker, total). After
//!    mint, the hot loop is a direct CAM index — no probe, no name
//!    compare, no per-record hash-table walk.
//! 4. **Whole-table double-cast.** Workers accumulate into full
//!    65536-row SoA [`BatchTable`]s (dirty-list tracked). A full table
//!    freezes into an `Arc` and is cast ONCE, WHOLE, to both ends —
//!    the ownership-guarantee sink and the Lance row-address sink; one
//!    allocation travels to both (the double cast), nothing is
//!    repacked into per-owner entry lists (lanes G/H's shape). Each
//!    sink journals one [`KanbanMove`] per applied batch; the Lance
//!    side additionally ticks a per-batch version (the `DatasetVersion`
//!    shape) and stamps `row → latest version` in its address table.
//! 5. **Flush cache.** Each worker cycles a pool of batch tables:
//!    freeze + double-cast batch `n`, then immediately continue
//!    filling batch `n+1` from the pool. A pooled table is reused only
//!    once BOTH sinks dropped their `Arc` (refcount back to 1 — flush
//!    complete); otherwise the pool grows. Flushing (at the sinks) and
//!    reindexing-next (at the worker) therefore INTERLEAVE — the
//!    worker never waits for a flush.
//!
//! ## Invariants asserted
//!
//! - `ownership journal == lance journal == total batches` — every
//!   batch witnessed on BOTH ends (double-cast completeness).
//! - Output map (rendered from the ownership table + codebook) equals
//!   lane A byte-for-byte.

use crate::lane_f::{fnv1a64, morton_slot, SLOTS};
use crate::{chunk_bounds, parse_temp_tenths, Stats};
use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
use ractor::{Actor, ActorProcessingErr, ActorRef, RpcReplyPort};
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Rows accumulated per batch table before freeze + double-cast — the
/// "65536-sized CAM-addressed batches" of the spec.
pub const BATCH_ROWS: usize = SLOTS;

// ─── Codebook: identity minted once, direct CAM addressing after ────────

/// The mint registry: station identity → unique row slot in the fixed
/// 0..65536 space. Collisions are resolved HERE, once, at mint (Morton
/// placement + linear probe + full name verification); every consumer
/// afterwards addresses rows directly by the minted slot.
struct Codebook {
    tags: Vec<u64>,
    names: Vec<Vec<u8>>,
    used: Vec<bool>,
    len: usize,
}

impl Codebook {
    fn new() -> Self {
        Self {
            tags: vec![0; SLOTS],
            names: vec![Vec::new(); SLOTS],
            used: vec![false; SLOTS],
            len: 0,
        }
    }

    /// Mint (or find) the unique slot for `(h, name)`.
    fn mint(&mut self, h: u64, name: &[u8]) -> u16 {
        let mut s = morton_slot(h) as usize;
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
            s = (s + 1) & (SLOTS - 1);
        }
    }
}

/// Worker-local memo: `hash → slot`, open-addressed on the same 64K
/// space with full name verification, so the global mint mutex is
/// touched only on first local sight of a station.
struct SlotMemo {
    tags: Vec<u64>,
    names: Vec<Vec<u8>>,
    slots: Vec<u16>,
    used: Vec<bool>,
}

impl SlotMemo {
    fn new() -> Self {
        Self {
            tags: vec![0; SLOTS],
            names: vec![Vec::new(); SLOTS],
            slots: vec![0; SLOTS],
            used: vec![false; SLOTS],
        }
    }

    #[inline(always)]
    fn resolve(&mut self, h: u64, name: &[u8], global: &Mutex<Codebook>) -> u16 {
        let mut s = morton_slot(h) as usize;
        loop {
            if !self.used[s] {
                // First local sight: consult the global mint (the only
                // lock on the whole hot path; ~stations per worker).
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
            s = (s + 1) & (SLOTS - 1);
        }
    }
}

// ─── BatchTable: the 65536-row CAM-addressed accumulation unit ──────────

/// One full-address-space SoA batch: 65536 rows, direct-indexed by the
/// codebook slot (no probe in the hot loop), dirty-list tracked so the
/// sinks' merges are O(dirty), and clear-by-undo recyclable through the
/// flush cache.
pub(crate) struct BatchTable {
    mins: Vec<i32>,
    maxs: Vec<i32>,
    sums: Vec<i64>,
    counts: Vec<u32>,
    dirty: Vec<u16>,
}

impl BatchTable {
    fn new() -> Self {
        Self {
            mins: vec![i32::MAX; SLOTS],
            maxs: vec![i32::MIN; SLOTS],
            sums: vec![0; SLOTS],
            counts: vec![0; SLOTS],
            dirty: Vec::with_capacity(1024),
        }
    }

    /// Direct CAM write — the codebook guarantees slot uniqueness, so
    /// this is a pure indexed fold: no probe, no compare.
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

    /// Clear-by-undo: reset exactly the dirty rows (O(dirty), never
    /// O(SLOTS)) — the recycle step of the flush cache.
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

// ─── RowOwner: the 65536 standing mailboxes (ownership registry) ────────

/// A standing row-owner mailbox. All 65536 are spawned upfront; mailbox
/// `i` owns row `i` of every 65536-row table by index correspondence.
/// Deliberately message-free in steady state: the ownership guarantee
/// is structural (aligned fixed indices + the sinks applying writes on
/// behalf), so the data path never fans out to 64K actors.
struct RowOwner;

impl Actor for RowOwner {
    type Msg = ();
    type State = ();
    type Arguments = ();

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        _args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(())
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        _msg: Self::Msg,
        _state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        Ok(())
    }
}

// ─── Ownership sink: the mailbox-SoA ownership-guarantee table ──────────

/// Messages the ownership sink accepts.
enum OwnershipMsg {
    /// A frozen whole batch (the `Arc` is SHARED with the Lance sink —
    /// one allocation, double-cast).
    Batch { table: Arc<BatchTable> },
    /// Drain: render the canonical table through the codebook into the
    /// common output map shape; reply `(map, journal_len)`.
    Finish {
        reply: RpcReplyPort<(BTreeMap<String, Stats>, usize)>,
    },
}

/// Ownership sink state: the canonical 65536-row SoA whose row `i` is
/// owned by mailbox `i` (`row_owner[i] == i as MailboxId` — the
/// guarantee table itself), every batch applied on behalf of the row
/// owners and witnessed with one [`KanbanMove`].
struct OwnershipState {
    row_owner: Vec<MailboxId>,
    mins: Vec<i32>,
    maxs: Vec<i32>,
    sums: Vec<i64>,
    counts: Vec<u32>,
    journal: Vec<KanbanMove>,
    codebook: Arc<Mutex<Codebook>>,
}

struct OwnershipSink;

impl Actor for OwnershipSink {
    type Msg = OwnershipMsg;
    type State = OwnershipState;
    type Arguments = Arc<Mutex<Codebook>>;

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        codebook: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(OwnershipState {
            // The two fixed indices, aligned: row i is owned by mailbox i.
            row_owner: (0..SLOTS).map(|i| i as MailboxId).collect(),
            mins: vec![i32::MAX; SLOTS],
            maxs: vec![i32::MIN; SLOTS],
            sums: vec![0; SLOTS],
            counts: vec![0; SLOTS],
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
            OwnershipMsg::Batch { table } => {
                // Merge O(dirty): fold each dirty row of the frozen
                // batch into the canonical row — applied ON BEHALF of
                // `row_owner[s]` (the write-on-behalf discipline; the
                // debug_assert pins the aligned-indices guarantee).
                for &slot in &table.dirty {
                    let s = slot as usize;
                    debug_assert_eq!(state.row_owner[s], s as MailboxId);
                    if table.mins[s] < state.mins[s] {
                        state.mins[s] = table.mins[s];
                    }
                    if table.maxs[s] > state.maxs[s] {
                        state.maxs[s] = table.maxs[s];
                    }
                    state.sums[s] += table.sums[s];
                    state.counts[s] += table.counts[s];
                }
                let pos = state.journal.len() as u32 + 1;
                state.journal.push(KanbanMove {
                    mailbox: 0, // the sink writes on behalf; witness position orders batches
                    from: KanbanColumn::CognitiveWork,
                    to: KanbanColumn::Evaluation,
                    witness_chain_position: pos,
                    libet_offset_us: 0,
                    exec: ExecTarget::Native,
                });
            }
            OwnershipMsg::Finish { reply } => {
                let cb = state.codebook.lock().expect("codebook lock");
                let mut out = BTreeMap::new();
                for s in 0..SLOTS {
                    if state.counts[s] > 0 {
                        let name =
                            String::from_utf8(cb.names[s].clone()).expect("station name utf8");
                        out.insert(
                            name,
                            Stats {
                                min: state.mins[s],
                                max: state.maxs[s],
                                sum: state.sums[s],
                                count: state.counts[s],
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

// ─── Lance sink: the row-address table (persistence half) ───────────────

/// Messages the Lance-side sink accepts.
enum LanceMsg {
    Batch {
        table: Arc<BatchTable>,
    },
    /// Drain: reply `(rows_addressed_total, versions_ticked, journal_len)`.
    Finish {
        reply: RpcReplyPort<(usize, u32, usize)>,
    },
}

/// Lance row-address table state: `slot → latest batch version` (the
/// row address the columnar writer would consume), one version tick
/// per applied batch (the `DatasetVersion` shape), same per-batch
/// witness journal as the ownership side.
struct LanceState {
    latest_version: Vec<u32>,
    rows_addressed: usize,
    version: u32,
    journal: Vec<KanbanMove>,
}

struct LanceSink;

impl Actor for LanceSink {
    type Msg = LanceMsg;
    type State = LanceState;
    type Arguments = ();

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        _args: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(LanceState {
            latest_version: vec![0; SLOTS],
            rows_addressed: 0,
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
            LanceMsg::Batch { table } => {
                // One version tick per batch; every dirty row's address
                // is stamped with the version that carries it.
                state.version += 1;
                for &slot in &table.dirty {
                    state.latest_version[slot as usize] = state.version;
                    state.rows_addressed += 1;
                }
                let pos = state.journal.len() as u32 + 1;
                state.journal.push(KanbanMove {
                    mailbox: 0,
                    from: KanbanColumn::CognitiveWork,
                    to: KanbanColumn::Evaluation,
                    witness_chain_position: pos,
                    libet_offset_us: 0,
                    exec: ExecTarget::Native,
                });
            }
            LanceMsg::Finish { reply } => {
                let _ = reply.send((state.rows_addressed, state.version, state.journal.len()));
            }
        }
        Ok(())
    }
}

// ─── The worker: fill → freeze → double-cast → recycle (flush cache) ────

/// The flush cache: previously double-cast batches, recycled once BOTH
/// sinks have dropped their `Arc` (refcount 1). If the head is still in
/// flight, a fresh table is allocated instead — the pool grows to
/// whatever depth the flush/refill interleave needs.
struct FlushCache {
    pool: VecDeque<Arc<BatchTable>>,
    peak: usize,
    allocated: usize,
}

impl FlushCache {
    fn new() -> Self {
        Self {
            pool: VecDeque::new(),
            peak: 0,
            allocated: 0,
        }
    }

    fn next_table(&mut self) -> BatchTable {
        if let Some(front) = self.pool.pop_front() {
            match Arc::try_unwrap(front) {
                Ok(mut table) => {
                    // Both sinks are done with it: recycle by undo.
                    table.reset();
                    return table;
                }
                Err(still_in_flight) => {
                    // Flush not complete yet — keep it queued, grow.
                    self.pool.push_front(still_in_flight);
                }
            }
        }
        self.allocated += 1;
        self.peak = self.peak.max(self.allocated);
        BatchTable::new()
    }

    fn park(&mut self, table: Arc<BatchTable>) {
        self.pool.push_back(table);
    }
}

/// One worker: scan its chunk, resolve each record's slot through the
/// memoized codebook (direct CAM index afterwards), fill the current
/// batch table; at `batch_rows` records freeze the table into an `Arc`,
/// DOUBLE-CAST it (ownership + lance) and continue on the next table
/// from the flush cache.
#[allow(clippy::too_many_arguments)]
fn worker_fill(
    data: &[u8],
    start: usize,
    end: usize,
    batch_rows: usize,
    codebook: &Mutex<Codebook>,
    ownership: &ActorRef<OwnershipMsg>,
    lance: &ActorRef<LanceMsg>,
    batches: &AtomicUsize,
) -> usize {
    let mut memo = SlotMemo::new();
    let mut cache = FlushCache::new();
    let mut table = cache.next_table();
    let mut rows_in_batch = 0usize;

    let double_cast = |cache: &mut FlushCache, full: BatchTable| {
        if full.dirty.is_empty() {
            // Nothing accumulated (empty tail) — nothing to cast.
            cache.park(Arc::new(full));
            return;
        }
        let frozen = Arc::new(full);
        ownership
            .cast(OwnershipMsg::Batch {
                table: Arc::clone(&frozen),
            })
            .expect("double-cast: ownership side");
        lance
            .cast(LanceMsg::Batch {
                table: Arc::clone(&frozen),
            })
            .expect("double-cast: lance side");
        batches.fetch_add(1, Ordering::Relaxed);
        cache.park(frozen);
    };

    let mut i = start;
    while i < end {
        let name_start = i;
        while data[i] != b';' {
            i += 1;
        }
        let name = &data[name_start..i];
        i += 1; // skip ';'
        let temp_start = i;
        while data[i] != b'\n' {
            i += 1;
        }
        let tenths = parse_temp_tenths(&data[temp_start..i]);
        i += 1; // skip '\n'

        let h = fnv1a64(name);
        let slot = memo.resolve(h, name, codebook);
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
}

/// Lane I with an explicit batch size (tests use a small batch to force
/// multiple double-casts + flush-cache recycling on small corpora).
pub fn lane_i_batch_pipeline_with(
    data: &[u8],
    workers: usize,
    batch_rows: usize,
) -> BTreeMap<String, Stats> {
    let workers = workers.max(1);
    let batch_rows = batch_rows.max(1);
    let bounds = chunk_bounds(data, workers);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .build()
        .expect("build tokio runtime for lane I");

    runtime.block_on(async move {
        let shared: Arc<Vec<u8>> = Arc::new(data.to_vec());
        let codebook = Arc::new(Mutex::new(Codebook::new()));
        let batches = Arc::new(AtomicUsize::new(0));

        // 1. ALL 65536 mailboxes upfront — the standing ownership
        //    registry. Spawn wall-time reported separately (stderr):
        //    it is infrastructure setup, not steady-state data path.
        let spawn_t0 = Instant::now();
        let mut owners = Vec::with_capacity(SLOTS);
        for _ in 0..SLOTS {
            let (actor, _handle) = Actor::spawn(None, RowOwner, ())
                .await
                .expect("spawn row-owner mailbox");
            owners.push(actor);
        }
        let spawn_ms = spawn_t0.elapsed().as_secs_f64() * 1000.0;

        // 2. The two ends of the double cast.
        let (ownership, ownership_handle) =
            Actor::spawn(None, OwnershipSink, Arc::clone(&codebook))
                .await
                .expect("spawn ownership sink");
        let (lance, lance_handle) = Actor::spawn(None, LanceSink, ())
            .await
            .expect("spawn lance sink");

        // 3. Workers: fill → freeze → double-cast → recycle.
        let mut worker_handles = Vec::with_capacity(bounds.len());
        for &(start, end) in &bounds {
            let shared = Arc::clone(&shared);
            let codebook = Arc::clone(&codebook);
            let ownership = ownership.clone();
            let lance = lance.clone();
            let batches = Arc::clone(&batches);
            worker_handles.push(tokio::task::spawn_blocking(move || {
                worker_fill(
                    &shared, start, end, batch_rows, &codebook, &ownership, &lance, &batches,
                )
            }));
        }
        let mut flush_cache_peak = 0usize;
        for h in worker_handles {
            flush_cache_peak = flush_cache_peak.max(h.await.expect("lane I worker join"));
        }

        // 4. Drain both ends; assert double-cast completeness.
        let (map, ownership_journal) =
            ractor::call!(ownership, |reply| OwnershipMsg::Finish { reply })
                .expect("ownership finish rpc");
        let (rows_addressed, versions, lance_journal) =
            ractor::call!(lance, |reply| LanceMsg::Finish { reply }).expect("lance finish rpc");

        let batches_total = batches.load(Ordering::Relaxed);
        assert_eq!(
            ownership_journal, batches_total,
            "every batch must be witnessed on the ownership end"
        );
        assert_eq!(
            lance_journal, batches_total,
            "every batch must be witnessed on the lance end"
        );
        assert_eq!(
            versions as usize, batches_total,
            "one DatasetVersion tick per batch"
        );

        ownership.stop(None);
        lance.stop(None);
        ownership_handle.await.expect("ownership sink join");
        lance_handle.await.expect("lance sink join");
        let stop_t0 = Instant::now();
        for owner in &owners {
            owner.stop(None);
        }
        let stop_ms = stop_t0.elapsed().as_secs_f64() * 1000.0;

        eprintln!(
            "lane_i: mailboxes={SLOTS} mailbox_spawn_ms={spawn_ms:.1} mailbox_stop_ms={stop_ms:.1} \
             batches={batches_total} versions={versions} rows_addressed={rows_addressed} \
             flush_cache_peak_tables_per_worker={flush_cache_peak} codebook_len={}",
            codebook.lock().expect("codebook lock").len
        );

        map
    })
}

/// Lane I — the substrate batch pipeline at the spec'd batch size
/// (65536-row CAM-addressed batches).
pub fn lane_i_batch_pipeline(data: &[u8], workers: usize) -> BTreeMap<String, Stats> {
    lane_i_batch_pipeline_with(data, workers, BATCH_ROWS)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lane I must agree byte-for-byte with lane A — with a SMALL batch
    /// size (1000 rows) so many batches double-cast and the flush cache
    /// actually recycles tables mid-stream (Arc refcount path), across
    /// an odd worker count.
    #[test]
    fn lane_i_agrees_with_lane_a_with_recycled_flush_cache() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("onebrc_probe_test_i_{}.txt", std::process::id()));
        let result = crate::gen::gen(&path, 50_000, 91).expect("gen");
        assert_eq!(result.rows, 50_000);

        let data = std::fs::read(&path).expect("read generated corpus");
        std::fs::remove_file(&path).ok();

        let a = crate::lane_a_scalar(&data);
        let i = lane_i_batch_pipeline_with(&data, 3, 1000);
        assert_eq!(a, i, "lane I must produce identical aggregates to lane A");
        assert!(!a.is_empty());
    }

    /// The codebook mints stable, unique slots: same identity always
    /// resolves to the same slot; distinct identities never share one.
    #[test]
    fn codebook_mints_unique_stable_slots() {
        let mut cb = Codebook::new();
        let a1 = cb.mint(fnv1a64(b"alpha"), b"alpha");
        let b1 = cb.mint(fnv1a64(b"beta"), b"beta");
        let a2 = cb.mint(fnv1a64(b"alpha"), b"alpha");
        assert_eq!(a1, a2, "mint must be idempotent per identity");
        assert_ne!(a1, b1, "distinct identities get distinct slots");
        assert_eq!(cb.len, 2);
    }
}
