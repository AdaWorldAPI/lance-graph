# D-MBX-A6 owner-consume + the fire-and-forget persistence sink

> READ BY: kanban-executor-engineer, mailbox-warden, any Sonnet-5 worker
> picking up the D-MBX-A6 persistence sink or the `owner_adapter` module.
> Companion to `mailbox-kanban-model.md` (read that first) and
> `write-on-behalf.md`.

## Status

- **SHIPPED (pre-write half):** `lance_graph_planner::owner_adapter` — the
  `Outcome → KanbanMove` bootstrap-rebind + ahead-cast adapter. Lance-free,
  5 falsifiable probes green. Completes the D-MBX-A6-P3b "owner-consume"
  deferral (STATUS_BOARD `D-MBX-A6-P3c`).
- **VERIFIED-BUT-GATED (post-write half):** the drain→Lance persistence sink.
  The API it must wire is confirmed to exist (below). It is **not buildable in
  the medcare-session container** — `protoc` is missing and the
  lance+datafusion+arrow compile would exhaust disk. It is the offline /
  next-environment slice.

## The corrected causal model (operator-ruled 2026-08-01 — earned via 5 corrections)

**A `KanbanMove` is the destination written on the parcel BEFORE dispatch.** It
is the transition the completed thought *intends* the mailbox to become. It is
cast **ahead** of persistence and travels WITH the write descriptor. This is
NOT the tugboat — casting a move does not advance any lifecycle.

**A KanbanStep is the delivery scan AFTER Lance accepts the write.** The
lifecycle mutation (`try_advance_phase`) happens post-persistence, on the
successful `LanceVersion`. **No successful write ⇒ no applied step.**

```
SoA thinks
  → StrategyOutcome carries its intended KanbanMove (bootstrap sentinel: owner 0, cycle 0)
  → owner_adapter rebinds the sentinel to the live owner + casts on_behalf   ← THIS MODULE (pre-write)
  → thinker CONTINUES IMMEDIATELY (fire-and-forget; write latency is masked)
  ── independent persistence path ──
  → BatchWriter drains casts, coalesces stacked intents, reads the LIVE SoA backing state
  → Lance MemWAL/ShardWriter durably appends the write                       ← the OFFLINE sink
  → the LSM view immediately treats WAL/memtable state as the SoA's latest thinking state
  → the PAIRED move is applied (try_advance_phase) — the KanbanStep
  → later flush/compaction calcifies into the base Lance dataset
```

The separation of these two paths is what masks thinking-time against
write-time and permits ~64k trajectories at ~2M SoA-evals/s while persistence
is amortized across batch flushes.

## Hard rules — DO NOT ADD (operator, 2026-08-01)

The thinker never waits for the WAL, version compaction, or step
acknowledgement. Do **not** add any of:

- a confirmation ledger / per-thought acknowledgement (`E-ACK-ELIMINATED-1`);
- a replay queue;
- a custom WAL (Lance 7 already ships one — below);
- ownership/version arbitration (the ractor single-owner guarantee IS the
  version-ordering guarantee; there is no competing writer to arbitrate);
- a synchronous callback into the thinker.

Ownership travels in the cast's `on_behalf` envelope. The move payload does not
need a synchronous ownership-adoption ceremony before it can be reported.

## The post-write invariant (for whoever builds the sink)

The version-completion path must apply the move **paired with that write** —
NOT manufacture a generic `next_phases().first()` transition merely because
some Lance version appeared. `NextPhaseScheduler`'s mechanical forward-arc march
is the wrong driver for the result-derived step. The specific result written on
behalf of the SoA determines what became durable; the paired move is the
witness of that.

The falsifier must prove **ordering**, not sentinel substitution:
- no successful write → no KanbanStep;
- successful write → exactly one *corresponding* post-write KanbanStep;
- the new version is visible as the SoA's latest state before its next
  evaluation.

## Verified Lance 7.0.0 MemWAL surface (wire this — invent nothing)

Read from the real crate source (`lance-7.0.0/src/dataset/mem_wal/`):

- `mem_wal/wal.rs` — `WalAppender::append(batches: Vec<RecordBatch>) -> Result<WalAppendResult>` (async); `flush(...)` (async).
- `mem_wal/memtable/batch_store.rs` — `BatchStore::append(batch: RecordBatch) -> Result<(usize, u64, usize), StoreFull>`; `append_batches(...)`.
- `mem_wal/memtable/flush.rs` — the memtable→base calcification.
- `dataset.rs` — `merge_insert` (the upsert routing).

The WAL append is the durable handoff that makes fire-and-forget safe; base-
dataset compaction is NOT the semantic synchronization boundary. Before
implementing, re-verify the exact checked-out signatures — if the API requires
an `LsmWriteSpec` or `merge_insert` routing, wire that existing surface.

## Environment gate (why the sink is offline)

The mandatory stack for a lance build: **lance 7.0.0 / lancedb 0.30 / arrow 58 /
datafusion 53 + `protoc` + ndarray**. In the private medcare session container
`protoc` is absent and disk headroom is ~8 G against a multi-GB datafusion/lance
build. `lance-graph-planner` (BatchWriter's crate) has no lance dep today — that
is the gap the sink must close (add the mandatory stack to its Cargo), done in
an environment that has protoc and disk.

## Where the pieces live

- Pre-write cast adapter: `crates/lance-graph-planner/src/owner_adapter.rs`.
- Intent staging: `crates/lance-graph-planner/src/batch_writer.rs` (`cast`,
  `drain_pending_payloads`, delegation cache; zero production callers today).
- The owner + lifecycle mutation: `MailboxSoaOwner::try_advance_phase`
  (`lance-graph-contract::soa_view`); real production advance paths already exist
  in `cognitive-shader-driver::mailbox_soa` and `symbiont::kanban_loop`.
- Durability read side: `crate::temporal` (`QueryReference::at` + deinterlace).
