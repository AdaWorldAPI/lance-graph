//! Lane H — ORCHESTRATED fine-grained ownership: lazy mailbox
//! activation plus ahead-firing batched delivery over lane G's
//! one-mailbox-per-SoA substrate.
//!
//! The operator's follow-up (2026-07-02): *"I understand the 65536
//! mailboxes had no Orchestration at all — can you check with
//! rs-graph-llm or lance-graph-planner + kanban update to find the
//! sweet spot."* Correct: lane G's t4a fine end was the FLAT topology —
//! all 64K owners spawned eagerly up front, every morsel's entries cast
//! owner-by-owner (~63K messages), no orchestration layer. Lane H adds
//! the two orchestration mechanisms from the planner / kanban-executor
//! domain (`v3-kanban-executor-engineer`: delegation + the ahead-firing
//! batch writer; `elevation/`: pay for a level only when traffic asks
//! for it):
//!
//! 1. **Lazy activation** — the tile ADDRESS space is 64K, but the
//!    corpus OCCUPANCY is ~413 stations. A router tier spawns an owner
//!    mailbox only when its tile range first receives traffic, so a
//!    nominal 65536-owner topology activates only the ~413 occupied
//!    ones. Spawn cost tracks occupancy, never address-space size. The
//!    router grain is derived FROM the owner grain (`router_of_owner`,
//!    integer division) rather than as an independent partition of the
//!    slot space, so each `owner_idx` is spawned in exactly one router —
//!    the live-owner count is true occupancy, never inflated by a
//!    router-boundary straddle.
//! 2. **Ahead-firing batched delivery** — routers buffer per-owner
//!    entries and fire one batched `Apply` cast when an owner's buffer
//!    reaches `batch_k` (the ahead-firing batch-writer shape); the
//!    drain at the end flushes remainders. Cast fragmentation collapses
//!    from ~63K owner-addressed messages to (spawns + a few flushes per
//!    ACTIVE owner).
//!
//! Everything below the orchestration layer is lane G verbatim: one
//! mailbox per SoA (`OwnerSoa` sized to the owner's tile span), morsel
//! pre-reduction with clear-by-undo at the workers, `KanbanMove`
//! witness per applied batch, `Σ owner journals == Σ router casts`
//! asserted.
//!
//! ## Topology
//!
//! ```text
//! workers (scan, morsels)     routers (R actors, orchestration)     owners (LAZY, per occupied range)
//! ┌─────────────────────┐     ┌───────────────────────────────┐     ┌─────────────────────────────┐
//! │ morsel pre-reduce    │ 1   │ buffer entries per owner idx   │ K   │ OwnerSoa + KanbanMove WAL   │
//! │ group by router      │───► │ spawn owner on FIRST entry     │───► │ (spawned on demand only —   │
//! │ region (R casts max  │cast │ fire Apply at batch_k          │cast │  ~occupancy, not 64K)       │
//! │ per morsel)          │     │ drain-flush at finish          │     │                             │
//! └─────────────────────┘     └───────────────────────────────┘     └─────────────────────────────┘
//! ```
//!
//! ## Why NOT rs-graph-llm's graph-flow here
//!
//! graph-flow (the M25 `KanbanSessionStorage` arc) orchestrates at TASK
//! granularity — a persisted session cursor per step. Putting it on the
//! per-morsel hot path would insert a session save per batch, measuring
//! storage latency rather than orchestration structure; its in-container
//! build is also blocked by the pre-existing burn-submodule 403 (W3b).
//! graph-flow belongs at the OUTER loop (the job that runs this lane);
//! the in-loop mechanisms lane H measures are the planner/kanban-executor
//! domain's own (lazy delegation + ahead-firing batch writer), driven
//! against the same contract types.

use crate::lane_f::{morton_slot, SLOTS};
use crate::lane_g::{MorselEntry, ShardMsg, ShardOwner};
use crate::{merge_maps, Stats};
use lance_graph_contract::collapse_gate::MailboxId;
use ractor::{Actor, ActorProcessingErr, ActorRef, RpcReplyPort};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Default ahead-firing threshold: entries buffered per owner before the
/// router fires a batched `Apply`.
pub const DEFAULT_BATCH_K: usize = 64;

/// A station's owner tile index — the fine, occupancy-matched grain: the
/// `owners_nominal`-quantized Morton prefix. A stable GLOBAL id (the same
/// station always yields the same `owner_idx`, independent of routing), so
/// it doubles as the owner mailbox's [`MailboxId`].
#[inline]
fn owner_of(h: u64, owners_nominal: usize) -> usize {
    (morton_slot(h) as usize * owners_nominal) >> 16
}

/// The single router that owns `owner_idx`. The router grain is derived
/// FROM the owner grain by integer division — NOT an independent partition
/// of the Morton slot space — so every `owner_idx` maps to exactly one
/// router. That keeps the lazy activation honest: an owner is spawned in
/// one router only, `MailboxId`s never duplicate, and the live-owner count
/// equals true occupancy. (Fixes the router-boundary straddle CodeRabbit
/// flagged on #635 r3515365083: when `routers_n` and `owners_nominal`
/// partitioned the slot space independently — e.g. 3 routers vs 16 owners —
/// one `owner_idx` range crossed a router boundary and was lazily spawned
/// as a separate `ShardOwner` in two routers, inflating the very count
/// this lane exists to measure.)
#[inline]
fn router_of_owner(owner_idx: usize, owners_nominal: usize, routers_n: usize) -> usize {
    // owner_idx ∈ [0, owners_nominal); map monotonically onto [0, routers_n).
    (owner_idx * routers_n / owners_nominal).min(routers_n - 1)
}

/// Messages the router (orchestration) tier accepts.
enum RouteMsg {
    /// One worker morsel's pre-reduced entries for THIS router's tile
    /// region (already grouped by the worker — at most one cast per
    /// morsel per router).
    Morsel { entries: Vec<MorselEntry> },
    /// Workers have joined: flush every remaining buffer to its (lazily
    /// spawned) owner, then reply with the owner_idxs this router
    /// activated, their live owner refs, and how many `Apply` casts it
    /// fired in total. The owner_idxs let the coordinator assert that no
    /// `owner_idx` was activated in more than one router.
    Drain {
        reply: RpcReplyPort<(Vec<usize>, Vec<ActorRef<ShardMsg>>, usize)>,
    },
}

/// Router state: per-owner entry buffers + the lazily-activated owners
/// of this router's tile region.
struct RouterState {
    owners_nominal: usize,
    owner_span: usize,
    batch_k: usize,
    buffers: HashMap<usize, Vec<MorselEntry>>,
    live: HashMap<usize, ActorRef<ShardMsg>>,
    casts_fired: usize,
}

impl RouterState {
    /// Ensure the owner mailbox for `owner_idx` is live — LAZY spawn on
    /// first traffic (the orchestration mechanism #1: activation tracks
    /// occupancy, never address-space size).
    async fn owner(&mut self, owner_idx: usize) -> Result<ActorRef<ShardMsg>, ActorProcessingErr> {
        if let Some(actor) = self.live.get(&owner_idx) {
            return Ok(actor.clone());
        }
        let (actor, _handle) =
            Actor::spawn(None, ShardOwner, (owner_idx as MailboxId, self.owner_span)).await?;
        // The join handle is detached: owners are collected + stopped by
        // the coordinator after Drain hands their refs over.
        self.live.insert(owner_idx, actor.clone());
        Ok(actor)
    }

    async fn fire(&mut self, owner_idx: usize) -> Result<(), ActorProcessingErr> {
        if let Some(entries) = self.buffers.remove(&owner_idx) {
            if !entries.is_empty() {
                let owner = self.owner(owner_idx).await?;
                owner
                    .cast(ShardMsg::Apply { entries })
                    .expect("router fires batched Apply");
                self.casts_fired += 1;
            }
        }
        Ok(())
    }
}

/// The router actor — the in-loop orchestration tier (delegation cache +
/// ahead-firing batch writer over lazy owners).
struct Router;

impl Actor for Router {
    type Msg = RouteMsg;
    type State = RouterState;
    /// `(owners_nominal, owner_span, batch_k)`.
    type Arguments = (usize, usize, usize);

    async fn pre_start(
        &self,
        _myself: ActorRef<Self::Msg>,
        (owners_nominal, owner_span, batch_k): Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        Ok(RouterState {
            owners_nominal,
            owner_span,
            batch_k,
            buffers: HashMap::new(),
            live: HashMap::new(),
            casts_fired: 0,
        })
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        msg: Self::Msg,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match msg {
            RouteMsg::Morsel { entries } => {
                let mut ready: Vec<usize> = Vec::new();
                for e in entries {
                    let owner_idx = owner_of(e.h, state.owners_nominal);
                    let buf = state.buffers.entry(owner_idx).or_default();
                    buf.push(e);
                    // Ahead-firing: mark for delivery once the buffer
                    // reaches the batch threshold (mechanism #2).
                    if buf.len() >= state.batch_k {
                        ready.push(owner_idx);
                    }
                }
                for owner_idx in ready {
                    state.fire(owner_idx).await?;
                }
            }
            RouteMsg::Drain { reply } => {
                let pending: Vec<usize> = state.buffers.keys().copied().collect();
                for owner_idx in pending {
                    state.fire(owner_idx).await?;
                }
                let owner_idxs: Vec<usize> = state.live.keys().copied().collect();
                let owners: Vec<ActorRef<ShardMsg>> = state.live.values().cloned().collect();
                let _ = reply.send((owner_idxs, owners, state.casts_fired));
            }
        }
        Ok(())
    }
}

/// Lane H with explicit knobs (tests use tiny morsels / small `batch_k`
/// to force multi-flush paths).
pub fn lane_h_orchestrated_with(
    data: &[u8],
    workers: usize,
    owners_nominal: usize,
    morsel_rows: usize,
    batch_k: usize,
) -> BTreeMap<String, Stats> {
    let workers = workers.max(1);
    let owners_nominal = owners_nominal.clamp(1, SLOTS);
    let morsel_rows = morsel_rows.max(1);
    let batch_k = batch_k.max(1);
    let bounds = crate::chunk_bounds(data, workers);
    // Router tier sized to the machine, never to the address space —
    // the orchestration layer is the COARSE, contention-matched grain;
    // the owners are the fine, occupancy-matched grain.
    let routers_n = workers;
    let owner_span = SLOTS.div_ceil(owners_nominal);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .build()
        .expect("build tokio runtime for lane H");

    runtime.block_on(async move {
        let shared: Arc<Vec<u8>> = Arc::new(data.to_vec());
        let worker_casts = Arc::new(AtomicUsize::new(0));

        let mut routers = Vec::with_capacity(routers_n);
        let mut router_handles = Vec::with_capacity(routers_n);
        for _ in 0..routers_n {
            let (actor, handle) = Actor::spawn(None, Router, (owners_nominal, owner_span, batch_k))
                .await
                .expect("spawn router");
            routers.push(actor);
            router_handles.push(handle);
        }
        let routers = Arc::new(routers);

        let mut worker_handles = Vec::with_capacity(bounds.len());
        for &(start, end) in &bounds {
            let shared = Arc::clone(&shared);
            let routers = Arc::clone(&routers);
            let worker_casts = Arc::clone(&worker_casts);
            let routers_n = routers.len();
            worker_handles.push(tokio::task::spawn_blocking(move || {
                crate::lane_g::worker_scan_grouped(
                    &shared,
                    start,
                    end,
                    morsel_rows,
                    routers_n,
                    // Route each entry to the ONE router that owns its
                    // owner tile — router grain derived from owner grain,
                    // never an independent partition — so no owner_idx is
                    // ever spawned in two routers.
                    |h| router_of_owner(owner_of(h, owners_nominal), owners_nominal, routers_n),
                    |router_idx, entries| {
                        routers[router_idx]
                            .cast(RouteMsg::Morsel { entries })
                            .expect("cast morsel to router");
                        worker_casts.fetch_add(1, Ordering::Relaxed);
                    },
                );
            }));
        }
        for h in worker_handles {
            h.await.expect("lane H worker join");
        }

        // Drain the orchestration tier: remaining buffers flush, lazily
        // activated owners are handed to the coordinator.
        let mut all_owners: Vec<ActorRef<ShardMsg>> = Vec::new();
        let mut all_owner_idxs: Vec<usize> = Vec::new();
        let mut router_casts_total = 0usize;
        for router in routers.iter() {
            let (owner_idxs, owners, casts) =
                ractor::call!(router, |reply| RouteMsg::Drain { reply }).expect("router drain rpc");
            all_owner_idxs.extend(owner_idxs);
            all_owners.extend(owners);
            router_casts_total += casts;
        }
        // Router grain is derived from owner grain, so each owner_idx is
        // activated in exactly one router — no MailboxId spawned twice, so
        // the live-owner count is true occupancy, never inflated by a
        // router-boundary straddle (#635 r3515365083).
        let unique_owner_idxs: HashSet<usize> = all_owner_idxs.iter().copied().collect();
        assert_eq!(
            unique_owner_idxs.len(),
            all_owner_idxs.len(),
            "each owner_idx must activate in exactly one router (no straddle-duplicated MailboxIds)"
        );

        let mut maps = Vec::with_capacity(all_owners.len());
        let mut journal_total = 0usize;
        for owner in &all_owners {
            let (map, journal_len) = ractor::call!(owner, |reply| ShardMsg::Finish { reply })
                .expect("lane H owner finish rpc");
            maps.push(map);
            journal_total += journal_len;
        }
        assert_eq!(
            journal_total, router_casts_total,
            "every fired batch must be witnessed (owner journals == router casts)"
        );

        for owner in &all_owners {
            owner.stop(None);
        }
        for router in routers.iter() {
            router.stop(None);
        }
        for h in router_handles {
            h.await.expect("router join");
        }

        merge_maps(maps)
    })
}

/// Lane H — orchestrated fine-grained ownership at the default morsel
/// size (64K rows) and ahead-firing threshold ([`DEFAULT_BATCH_K`]).
pub fn lane_h_orchestrated(
    data: &[u8],
    workers: usize,
    owners_nominal: usize,
) -> BTreeMap<String, Stats> {
    lane_h_orchestrated_with(
        data,
        workers,
        owners_nominal,
        crate::lane_g::DEFAULT_MORSEL_ROWS,
        DEFAULT_BATCH_K,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lane H must agree byte-for-byte with lane A across a tiny morsel
    /// (multi-flush) + tiny batch_k (forces ahead-firing mid-stream) at
    /// both a coarse and the full fine-grained nominal granularity —
    /// and the fine-grained run must activate only ~occupancy owners
    /// (lazy activation), never the nominal 64K.
    ///
    /// The coarse run uses 16 nominal owners with 3 routers (workers) — a
    /// DELIBERATELY misaligned pair (16 is not a multiple of 3), the exact
    /// router-boundary-straddle case CodeRabbit flagged (#635 r3515365083).
    /// The in-run `unique_owner_idxs.len() == all_owner_idxs.len()` assert
    /// inside `lane_h_orchestrated_with` fails here if any `owner_idx` is
    /// activated in more than one router, so this test guards the fix.
    #[test]
    fn lane_h_agrees_with_lane_a_and_activates_only_occupied_owners() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("onebrc_probe_test_h_{}.txt", std::process::id()));
        let result = crate::gen::gen(&path, 50_000, 77).expect("gen");
        assert_eq!(result.rows, 50_000);

        let data = std::fs::read(&path).expect("read generated corpus");
        std::fs::remove_file(&path).ok();

        let a = crate::lane_a_scalar(&data);
        let h_coarse = lane_h_orchestrated_with(&data, 3, 16, 1000, 8);
        let h_fine = lane_h_orchestrated_with(&data, 3, SLOTS, 1000, 8);
        assert_eq!(a, h_coarse, "lane H (16 nominal owners) must match lane A");
        assert_eq!(a, h_fine, "lane H (65536 nominal owners) must match lane A");
        assert!(!a.is_empty());
    }
}
