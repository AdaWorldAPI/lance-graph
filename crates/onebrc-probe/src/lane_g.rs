//! Lane G — the kanban-update write path: the 64K SoA as OWNED state
//! behind mailbox actors, with every write witnessed on the board.
//!
//! The operator's question (2026-07-02): *"compare morton and the kanban
//! vs without — if 64k concurrent SoA vs Morton tile can help us
//! understand the pros and cons of our architecture when using kanban
//! update."* Lane F answered the ADDRESS question (Morton vs radix,
//! ~10% tax) with per-worker PRIVATE tables merged once at the end —
//! fast, but the accumulator state is invisible until the merge and no
//! write is witnessed. Lane G answers the OWNERSHIP question: the same
//! 64K-slot Morton-tile SoA, but held as **owned state by shard mailbox
//! actors** (mailbox-as-owner), updated by **streamed morsel casts**
//! (the kanban-update path), each applied batch journaled as a
//! [`KanbanMove`] — the write-ahead witness trail.
//!
//! ## Topology
//!
//! ```text
//! workers (spawn_blocking, scalar scan)          shard owners (ractor actors)
//! ┌─────────────────────────────────┐            ┌──────────────────────────┐
//! │ chunk → 64K-row morsels          │  cast      │ shard 0: Morton tiles    │
//! │ per-morsel local SoA pre-reduce  │──Apply───► │  [0, 64K/S)   SoA + WAL  │
//! │ dirty-slot extract, clear-by-undo│  (by tile  │ shard 1: tiles ...       │
//! │ route entries by Morton PREFIX   │   prefix)  │ shard S-1: tiles ...     │
//! └─────────────────────────────────┘            └──────────────────────────┘
//! ```
//!
//! - **One mailbox per SoA (the canon, verbatim):** each owner actor's
//!   `State` IS its own complete [`OwnerSoa`], sized to its tile span —
//!   there is NO shared table and no "one SoA sharded across owners".
//!   `shards` only chooses how many (mailbox, SoA) pairs partition the
//!   tile space: 1 = one mailbox owning one SoA covering all tiles …
//!   65536 = one mailbox per tile, each owning its own tiny SoA — the
//!   literal "64K concurrent SoAs" end of the operator's question.
//! - **Route by prefix:** an entry's owner is the top bits of its Morton
//!   slot (`slot * shards >> 16`) — contiguous tile ranges, the HHTL
//!   prefix route. A station's hash always lands with the same owner, so
//!   the owners' SoAs are disjoint by construction.
//! - **Mailbox-as-owner:** the serialized message loop is the single
//!   writer of the owner's SoA (the same compile-time no-aliasing
//!   argument as `KanbanActor`, E-CE64-MB-4). No lock, no shared `&mut`.
//! - **Kanban update = witnessed write:** every applied morsel batch
//!   appends one `KanbanMove` (`CognitiveWork → Evaluation`, a legal
//!   Rubicon forward edge) to the owner's journal — recorded directly to
//!   the WAL trail (the `storage_kanban` journaling precedent: the FSM
//!   gates live advancement; post-hoc witness records write the fields).
//!   The lane asserts `Σ journal lengths == total casts` — nothing
//!   applied unwitnessed, nothing witnessed unapplied.
//! - **Morsels** (64K rows, `#227`'s morsel size): workers pre-reduce
//!   each morsel in a private table (identical hot loop to lane F),
//!   extract only the dirty slots (≤ ~station-count entries), reset them
//!   by undo, and cast the pre-reduced entries to the owning shards.
//!   Streaming morsels — not one merge at the end — is what makes the
//!   owner state LIVE: at any instant the shard mailboxes hold a bounded-
//!   staleness view of the whole aggregation (queryable mid-flight, the
//!   substrate claim lane F cannot make).
//!
//! ## What G vs F measures
//!
//! Same address (Morton tiles), same scan, same pre-reduction. The delta
//! is pure architecture: streamed witnessed writes through owner
//! mailboxes (+ the actor-boundary `Arc` corpus copy, as lanes D/E)
//! versus private-merge-at-end. `shards` sweeps the GRANULARITY of
//! ownership — 1 mailbox (every update serializes through one queue),
//! through Morton-tile-range groupings (4/16/256/…), out to 65536 (one
//! mailbox per tile, 64K concurrent SoAs, spawn cost included in the
//! measurement). That curve is the pros-and-cons ledger of "64K
//! concurrent SoA vs Morton tile grouping" under kanban update.

use crate::lane_f::{fnv1a64, morton_slot, SoaTable, SLOTS};
use crate::{chunk_bounds, merge_maps, parse_temp_tenths, Stats};
use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::{ExecTarget, KanbanColumn, KanbanMove};
use ractor::{Actor, ActorProcessingErr, ActorRef, RpcReplyPort};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Rows per worker morsel — `#227`'s morsel size (64K rows ≈ L1/L2-resident
/// pre-reduction working set).
pub const DEFAULT_MORSEL_ROWS: usize = 1 << 16;

/// One pre-reduced station aggregate extracted from a worker morsel.
struct MorselEntry {
    h: u64,
    name: Vec<u8>,
    stats: Stats,
}

/// Messages a shard owner accepts.
enum ShardMsg {
    /// One morsel's pre-reduced entries for THIS shard's tile range —
    /// fire-and-forget (`cast`): workers never wait on owners, owners
    /// drain their mailbox in arrival order (single-writer serialization).
    Apply { entries: Vec<MorselEntry> },
    /// Drain point: reply with the shard's aggregate map + journal length.
    /// Sent only after every worker has joined, so mailbox FIFO guarantees
    /// all `Apply`s are applied first.
    Finish {
        reply: RpcReplyPort<(BTreeMap<String, Stats>, usize)>,
    },
}

/// A mailbox's OWN SoA — one per owner, sized to its tile range (the
/// canon: one ractor mailbox per SoA; there is no shared table and no
/// "one SoA sharded across owners"). Same parallel-array shape as lane
/// F's `SoaTable`, capacity ∝ the owner's tile span (clamped: at least
/// 64 slots so tiny per-tile owners can still absorb hash-collided
/// stations; at most 4096 — ~10× the whole corpus's station count — so
/// a 1-owner topology doesn't allocate 64K slots it can never fill).
/// Local placement = `morton_slot & (capacity-1)` + linear probe with
/// full `(h, name)` verification; the PREFIX routing to this mailbox
/// already happened at the worker.
struct OwnerSoa {
    capacity: usize, // power of two
    tags: Vec<u64>,
    names: Vec<Vec<u8>>,
    mins: Vec<i32>,
    maxs: Vec<i32>,
    sums: Vec<i64>,
    counts: Vec<u32>,
}

impl OwnerSoa {
    fn for_span(span: usize) -> Self {
        let capacity = span.clamp(64, 4096).next_power_of_two();
        Self {
            capacity,
            tags: vec![0; capacity],
            names: vec![Vec::new(); capacity],
            mins: vec![i32::MAX; capacity],
            maxs: vec![i32::MIN; capacity],
            sums: vec![0; capacity],
            counts: vec![0; capacity],
        }
    }

    /// Fold one pre-reduced entry (the commutative BUNDLE merge) into
    /// this mailbox's own SoA. Panics if the owner's table is full —
    /// a probe-sizing bug, never silent corruption.
    fn merge_entry(&mut self, e: &MorselEntry) {
        let mut s = morton_slot(e.h) as usize & (self.capacity - 1);
        for _ in 0..self.capacity {
            if self.counts[s] == 0 {
                self.tags[s] = e.h;
                self.names[s] = e.name.clone();
                self.mins[s] = e.stats.min;
                self.maxs[s] = e.stats.max;
                self.sums[s] = e.stats.sum;
                self.counts[s] = e.stats.count;
                return;
            }
            if self.tags[s] == e.h && self.names[s] == e.name {
                if e.stats.min < self.mins[s] {
                    self.mins[s] = e.stats.min;
                }
                if e.stats.max > self.maxs[s] {
                    self.maxs[s] = e.stats.max;
                }
                self.sums[s] += e.stats.sum;
                self.counts[s] += e.stats.count;
                return;
            }
            s = (s + 1) & (self.capacity - 1);
        }
        panic!("OwnerSoa full: more stations routed to one mailbox than its capacity");
    }

    /// Sweep occupied slots into the common output map shape.
    fn into_map(self) -> BTreeMap<String, Stats> {
        let mut out = BTreeMap::new();
        for s in 0..self.capacity {
            if self.counts[s] > 0 {
                let name = String::from_utf8(self.names[s].clone()).expect("station name utf8");
                out.insert(
                    name,
                    Stats {
                        min: self.mins[s],
                        max: self.maxs[s],
                        sum: self.sums[s],
                        count: self.counts[s],
                    },
                );
            }
        }
        out
    }
}

/// Owner state: THIS mailbox's own SoA + its kanban WAL.
struct ShardState {
    id: MailboxId,
    table: OwnerSoa,
    journal: Vec<KanbanMove>,
}

/// The shard-owner actor — mailbox-as-owner: the actor and its SoA are
/// one thing; `shards` is simply HOW MANY (mailbox, SoA) pairs the tile
/// space is partitioned into, from 1 (one mailbox owns all tiles) to
/// 65536 (one mailbox per tile — the literal "64K concurrent SoAs").
struct ShardOwner;

impl Actor for ShardOwner {
    type Msg = ShardMsg;
    type State = ShardState;
    /// `(mailbox id, tile span)` — span sizes this owner's own SoA.
    type Arguments = (MailboxId, usize);

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        (id, span): Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(ShardState {
            id,
            table: OwnerSoa::for_span(span),
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
            ShardMsg::Apply { entries } => {
                for e in &entries {
                    state.table.merge_entry(e);
                }
                // The witnessed write: one legal Rubicon forward move per
                // applied morsel batch, recorded to the WAL trail (see
                // module doc — journaling, not live FSM advancement).
                let pos = state.journal.len() as u32 + 1;
                state.journal.push(KanbanMove {
                    mailbox: state.id,
                    from: KanbanColumn::CognitiveWork,
                    to: KanbanColumn::Evaluation,
                    witness_chain_position: pos,
                    libet_offset_us: 0,
                    exec: ExecTarget::Native,
                });
            }
            ShardMsg::Finish { reply } => {
                let journal_len = state.journal.len();
                let table = std::mem::replace(&mut state.table, OwnerSoa::for_span(1));
                let _ = reply.send((table.into_map(), journal_len));
            }
        }
        Ok(())
    }
}

/// A station's owning shard: the top bits of its Morton slot — contiguous
/// tile ranges (the prefix route). Computed from the hash's PRE-probe slot
/// so it is deterministic per station regardless of worker-local probing.
#[inline(always)]
fn shard_of(h: u64, shards: usize) -> usize {
    (morton_slot(h) as usize * shards) >> 16
}

/// Extract the dirty slots of a worker's morsel table into per-shard entry
/// batches, resetting each extracted slot by undo (`#227`'s clear-by-undo:
/// O(dirty), never O(SLOTS)), and cast every non-empty batch to its owner.
/// Grouping is sort-based (O(dirty log dirty) on ≤ ~station-count entries),
/// NOT a dense `Vec` per shard — a 64K-owner topology must not allocate
/// 64K empty vecs per morsel flush.
fn flush_morsel(
    table: &mut SoaTable,
    dirty: &mut Vec<usize>,
    shards: usize,
    owners: &[ActorRef<ShardMsg>],
    casts: &AtomicUsize,
) {
    if dirty.is_empty() {
        return;
    }
    let mut tagged: Vec<(usize, MorselEntry)> = Vec::with_capacity(dirty.len());
    for &s in dirty.iter() {
        let h = table.tags[s];
        tagged.push((
            shard_of(h, shards),
            MorselEntry {
                h,
                name: std::mem::take(&mut table.names[s]),
                stats: Stats {
                    min: table.mins[s],
                    max: table.maxs[s],
                    sum: table.sums[s],
                    count: table.counts[s],
                },
            },
        ));
        // Clear-by-undo: reset exactly the slots this morsel touched.
        table.tags[s] = 0;
        table.mins[s] = i32::MAX;
        table.maxs[s] = i32::MIN;
        table.sums[s] = 0;
        table.counts[s] = 0;
    }
    dirty.clear();
    tagged.sort_unstable_by_key(|(shard, _)| *shard);
    let mut it = tagged.into_iter().peekable();
    while let Some((shard, first)) = it.next() {
        let mut entries = vec![first];
        while it.peek().is_some_and(|(s, _)| *s == shard) {
            entries.push(it.next().expect("peeked").1);
        }
        owners[shard]
            .cast(ShardMsg::Apply { entries })
            .expect("cast morsel batch to shard owner");
        casts.fetch_add(1, Ordering::Relaxed);
    }
}

/// One worker's scan: identical per-record hot loop to lane F (same hash,
/// same Morton slot, same probe), pre-reducing into a private table, with
/// a morsel-boundary flush that streams the dirty entries to the owners.
fn worker_scan(
    data: &[u8],
    start: usize,
    end: usize,
    morsel_rows: usize,
    shards: usize,
    owners: &[ActorRef<ShardMsg>],
    casts: &AtomicUsize,
) {
    let mut table = SoaTable::new();
    let mut dirty: Vec<usize> = Vec::with_capacity(1024);
    let mut rows_in_morsel = 0usize;
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
        let s = table.observe(morton_slot(h), h, name, tenths);
        if table.counts[s] == 1 {
            dirty.push(s); // first observation of this station this morsel
        }
        rows_in_morsel += 1;
        if rows_in_morsel == morsel_rows {
            flush_morsel(&mut table, &mut dirty, shards, owners, casts);
            rows_in_morsel = 0;
        }
    }
    flush_morsel(&mut table, &mut dirty, shards, owners, casts);
}

/// Lane G with an explicit morsel size (tests use a tiny morsel to force
/// multi-morsel flushes + clear-by-undo on small corpora).
pub fn lane_g_kanban_soa_with_morsel(
    data: &[u8],
    workers: usize,
    shards: usize,
    morsel_rows: usize,
) -> BTreeMap<String, Stats> {
    let workers = workers.max(1);
    let shards = shards.clamp(1, SLOTS); // 1 mailbox … 1 mailbox per tile
    let morsel_rows = morsel_rows.max(1);
    let bounds = chunk_bounds(data, workers);

    // Async threads host the shard owners; the scan workers run on the
    // blocking pool so a CPU-bound scan cannot starve the owners' mailbox
    // processing (real overlap between producing and applying — the
    // streamed-write architecture under measurement, not an artifact of
    // task scheduling).
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .build()
        .expect("build tokio runtime for lane G");

    runtime.block_on(async move {
        // Actor-model boundary cost: one upfront corpus copy, as lanes D/E.
        let shared: Arc<Vec<u8>> = Arc::new(data.to_vec());
        let casts = Arc::new(AtomicUsize::new(0));

        let mut owners = Vec::with_capacity(shards);
        let mut owner_handles = Vec::with_capacity(shards);
        // One (mailbox, SoA) pair per shard — each owner's OWN SoA is
        // sized to its tile span (SLOTS/shards, i.e. the whole tile space
        // at shards=1 down to one tile per mailbox at shards=65536).
        let span = SLOTS.div_ceil(shards);
        for sid in 0..shards {
            let (actor, handle) = Actor::spawn(None, ShardOwner, (sid as MailboxId, span))
                .await
                .expect("spawn shard owner");
            owners.push(actor);
            owner_handles.push(handle);
        }
        let owners = Arc::new(owners);

        let mut worker_handles = Vec::with_capacity(bounds.len());
        for &(start, end) in &bounds {
            let shared = Arc::clone(&shared);
            let owners = Arc::clone(&owners);
            let casts = Arc::clone(&casts);
            worker_handles.push(tokio::task::spawn_blocking(move || {
                worker_scan(&shared, start, end, morsel_rows, shards, &owners, &casts);
            }));
        }
        for h in worker_handles {
            h.await.expect("lane G worker join");
        }

        // All workers joined ⇒ every Apply is already enqueued; mailbox
        // FIFO applies them before Finish is handled.
        let mut maps = Vec::with_capacity(shards);
        let mut journal_total = 0usize;
        for actor in owners.iter() {
            let (map, journal_len) = ractor::call!(actor, |reply| ShardMsg::Finish { reply })
                .expect("lane G shard finish rpc");
            maps.push(map);
            journal_total += journal_len;
        }
        assert_eq!(
            journal_total,
            casts.load(Ordering::Relaxed),
            "every applied morsel batch must be witnessed (journal == casts)"
        );

        for actor in owners.iter() {
            actor.stop(None);
        }
        for h in owner_handles {
            h.await.expect("shard owner join");
        }

        merge_maps(maps)
    })
}

/// Lane G — kanban-update write path at the default (64K-row) morsel size.
pub fn lane_g_kanban_soa(data: &[u8], workers: usize, shards: usize) -> BTreeMap<String, Stats> {
    lane_g_kanban_soa_with_morsel(data, workers, shards, DEFAULT_MORSEL_ROWS)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lane G must agree byte-for-byte with lane A — across a TINY morsel
    /// size (1000 rows) so the multi-morsel flush + clear-by-undo path and
    /// the cross-morsel owner merge are exercised, for both the
    /// single-mailbox (shards=1) and tile-sharded (shards=4) topologies.
    #[test]
    fn lane_g_agrees_with_lane_a_across_morsels_and_shard_counts() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("onebrc_probe_test_g_{}.txt", std::process::id()));
        let result = crate::gen::gen(&path, 50_000, 55).expect("gen");
        assert_eq!(result.rows, 50_000);

        let data = std::fs::read(&path).expect("read generated corpus");
        std::fs::remove_file(&path).ok();

        let a = crate::lane_a_scalar(&data);
        let g1 = lane_g_kanban_soa_with_morsel(&data, 3, 1, 1000);
        let g4 = lane_g_kanban_soa_with_morsel(&data, 3, 4, 1000);
        // Fine-grained ownership: 4096 mailboxes, each owning its own
        // 16-tile-span SoA (span < 64 → minimum 64-slot capacity per
        // owner) — exercises the per-owner sizing + local probing.
        let g4096 = lane_g_kanban_soa_with_morsel(&data, 3, 4096, 1000);
        assert_eq!(a, g1, "lane G (1 mailbox) must match lane A");
        assert_eq!(a, g4, "lane G (4 mailboxes) must match lane A");
        assert_eq!(a, g4096, "lane G (4096 mailboxes) must match lane A");
        assert!(!a.is_empty());
    }

    /// Prefix routing is total and deterministic: every possible hash maps
    /// to a valid shard, and equal hashes always land in the same shard.
    #[test]
    fn shard_routing_is_total_and_stable() {
        for shards in [1usize, 3, 4, 16] {
            for h in [0u64, 1, 0xFFFF, 0xDEAD_BEEF, u64::MAX] {
                let s = shard_of(h, shards);
                assert!(s < shards, "shard {s} out of range for {shards}");
                assert_eq!(s, shard_of(h, shards), "routing must be deterministic");
            }
        }
    }
}
