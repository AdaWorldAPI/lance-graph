//! Lane D — `ractor` actor-per-worker aggregation.
//!
//! Purpose: quantifies the "ractor is a helper, not a messaging path" ruling
//! (see `.claude/v3/knowledge/v3-substrate-primer.md` §6) as a measured
//! ratio against Lane C's bare `std::thread::scope`, on **identical**
//! chunking (`chunk_bounds`) and merge shape (`Stats::merge` via
//! `merge_maps`) — the only variable this lane changes is the worker
//! primitive: a `ractor` `Actor` instead of a raw OS thread.
//!
//! Mirrors `lance-graph-supervisor`'s `KanbanActor` idioms (`kanban_actor.rs`
//! — `Actor::spawn`, the ask-pattern `ractor::call!`, `RpcReplyPort` in the
//! message variant) for a single, stateless worker actor.
//!
//! ## Actor-model boundary cost
//!
//! Lane C's workers borrow `&data[start..end]` directly (zero-copy) because
//! `std::thread::scope` proves the borrow outlives every spawned thread.
//! `ractor` actors run as independent tokio tasks that outlive any single
//! call site, so they cannot borrow the caller's stack slice — the corpus
//! is copied ONCE into an `Arc<Vec<u8>>` and each actor is handed a clone of
//! the `Arc` (refcount bump, not a byte copy) plus its `(start, end)` chunk
//! bounds. That one upfront `data.to_vec()` is itself part of what this
//! lane measures: it is the actor-model tax Lane C does not pay.
//!
//! ## Per-record logic
//!
//! Each actor computes its own chunk with `lane_a_scalar` — the SAME
//! shared per-record helper Lane C's worker closures already call (Lane C
//! already factored this out; there is nothing to duplicate here). The
//! coordinator folds the per-actor maps with `merge_maps` (the same
//! commutative, order-independent BUNDLE step Lane C uses).

use crate::{chunk_bounds, lane_a_scalar, merge_maps, Stats};
use ractor::{Actor, ActorProcessingErr, ActorRef, RpcReplyPort};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Messages `ChunkWorker` accepts — a single ask-pattern variant per the
/// `KanbanActor` idiom (`kanban_actor.rs`'s `RpcReplyPort`-in-variant shape).
pub enum ChunkMsg {
    /// Aggregate `data[start..end]` (newline-aligned, per `chunk_bounds`)
    /// via `lane_a_scalar` and reply with the owned per-chunk map.
    Aggregate {
        data: Arc<Vec<u8>>,
        start: usize,
        end: usize,
        reply: RpcReplyPort<BTreeMap<String, Stats>>,
    },
}

/// A stateless per-chunk aggregation actor. One is spawned per worker chunk
/// and stopped after its single reply — the actor-per-worker shape this
/// lane measures against Lane C's `std::thread::scope` worker closures.
pub struct ChunkWorker;

impl Actor for ChunkWorker {
    type Msg = ChunkMsg;
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
        msg: Self::Msg,
        _state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match msg {
            ChunkMsg::Aggregate {
                data,
                start,
                end,
                reply,
            } => {
                // Same per-record logic as Lane C's worker closure — see
                // module doc "Per-record logic".
                let map = lane_a_scalar(&data[start..end]);
                let _ = reply.send(map);
            }
        }
        Ok(())
    }
}

/// Lane D — actor-per-worker baseline. Builds a `workers`-thread tokio
/// runtime, splits `data` into `workers` newline-aligned chunks
/// (`chunk_bounds`, identical to Lane C), spawns one `ChunkWorker` per
/// chunk, asks each for its aggregate via `ractor::call!`, stops the actor,
/// then folds all per-chunk maps with the same commutative `merge_maps`
/// Lane C uses.
pub fn lane_d_ractor(data: &[u8], workers: usize) -> BTreeMap<String, Stats> {
    let workers = workers.max(1);
    let bounds = chunk_bounds(data, workers);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .build()
        .expect("build tokio runtime for lane D");

    runtime.block_on(async move {
        // Actor-model boundary cost: one upfront copy into a shared Arc —
        // see module doc "Actor-model boundary cost".
        let shared = Arc::new(data.to_vec());

        let mut join_handles = Vec::with_capacity(bounds.len());
        for &(start, end) in &bounds {
            let shared = Arc::clone(&shared);
            join_handles.push(tokio::spawn(async move {
                let (actor, handle) = Actor::spawn(None, ChunkWorker, ())
                    .await
                    .expect("spawn lane D chunk worker");
                let map = ractor::call!(actor, |reply| ChunkMsg::Aggregate {
                    data: shared,
                    start,
                    end,
                    reply,
                })
                .expect("lane D actor rpc");
                actor.stop(None);
                handle.await.expect("lane D actor join");
                map
            }));
        }

        let mut results = Vec::with_capacity(join_handles.len());
        for jh in join_handles {
            results.push(jh.await.expect("lane D worker task join"));
        }
        merge_maps(results)
    })
}
