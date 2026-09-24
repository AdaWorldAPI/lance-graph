# 2026-09-24 — three clocks that never meet implicitly; state folds, events witness

**Status:** DECISION (clock separation, below) · MEASURED (history depth; NULL width) · OPEN (see below)
**D-ids:** none new — frames D-HWV-1 / D-HWV-1a (#1277) and the storage recalibration after the
Lance 12 probes. Companion to `2026-09-24-cost-scales-with-dirty-rows-replay-budget-before-materialization.md`.

## DECISION — three clocks, one coupling number

| clock | carries | unit | when paid |
|---|---|---|---|
| **LIVE** | resident 64k field + coordinate folds | ns | always; **never waits implicitly** |
| **PERSISTENCE** | elected dirty state + append-only witnesses | ms | trails live state |
| **RECOVERY** | checkpoint + persisted tail | fragments since checkpoint | only on restart / cold rebuild |

Coupling number: **BACKLOG = irreducible event bytes + current coalesced state frontier** — not
"cycles waiting". The steady-state condition is
`drain capacity ≥ rate of non-coalescible persistence information`, not write throughput ≥
fold throughput: 131M folds that elect 50 changed rows cost the writer 50 rows.

**The clocks never meet implicitly.** Exactly three rendezvous, each explicit:

1. capacity exhausted **after every permitted merge / spill / buffer step** → backpressure policy;
2. explicit durability demand via `wait_durable()`;
3. resident state lost → recovery.

SCOPE: live cognition loop, cycle persistence, recovery. BASIS: dirty sweep (companion entry),
history depth (below), #1277 submit 1.9 µs vs inline seal 72.2 ms. REVISIT WHEN: a measured
sustained-dirty run shows coalescing cannot keep backlog bounded.

"Dirty" = elected by semantic execution (a real move), **not** `final_bytes != initial_bytes`.
`emit_bootstrap_intent` (`owner_adapter.rs:99-100`) casts on the move, never on a memcmp.

**#1277 is the first explicit live ↔ persistence clock-separation membrane**, not merely a
background seal optimisation (`try_submit` 1.9 µs vs inline seal 72.2 ms at 64k).

## State folds, events witness (direction — see OPEN; nothing implemented)

Current code, as written: `cycle_sink.rs` writes image rows (`:524-535`, 512-B NodeRow state) and
landing rows (`:521`, one per KanbanMove, NULL payload) per cycle; #1277 queues one ticket per
cycle and nothing coalesces. The state/event split and its consequences are listed under OPEN.

## Measured — history depth (recovery-clock evidence)

64k initial rows, then 1 dirty row per cycle, lance 11, release. Scratch:
`/tmp/claude-0/cyclesink-probe/src/bin/historydepth.rs`.

| cycles | fragments | commit p50 / max | reopen | latest-state scan | fold | store |
|---|---|---|---|---|---|---|
| 10 | 11 | 1.63 / 2.79 ms | 7.5 ms | 37 ms | 4.5 ms | 68 MB |
| 100 | 101 | 2.47 / 3.74 ms | 24 ms | 57 ms | 4.7 ms | 69 MB |
| 1,000 | 1,001 | 3.28 / 13 ms | 209 ms | 334 ms | 5.3 ms | 125 MB |
| 10,000 | 10,001 | 16 / 179 ms | 1,954 ms | 3,428 ms | 6.9 ms | 5.3 GB |

The scan grows ~90× over 10→10k fragments; the fold ~1.5×. This prices **recovery / cold / as-of
reconstruction**, never the live loop (the resident field is already latest state). Checkpoint
cadence is a restart-latency target. Metadata growth (~5.3 GB for ~128 MB of data) is inferred from
store bytes; the manifest-size column was misread and dropped.

## Measured — the witness channel pays state-channel width

NULL `FixedSizeBinary(512)` mixed with set values stores 512 zero bytes per NULL (64k set + 64k NULL
= 1,030 B per set row; all-NULL ≈ 1.9 B/row). Not a complaint about the 64k reservation: the
non-coalescible, supposedly tiny witness channel inherits the 512-B shape of the state channel.

## OPEN (none of these is an implementation claim)

- Merge-before-block when `try_submit` returns `Full` (today `Full` hands the casts back; nothing merges).
- State channel coalesces by NodeGuid; event / landing channel remains append-only.
- Landing / event rows currently inherit the 512-B nullable payload column despite carrying no payload
  (measured cost above).
- Merged persistence batches imply a coarser `durable_head`.
- Reconciliation identity must evolve from `(cycle, hash)` toward a range-aware form such as
  `(start_cycle, end_cycle, commitment)`; what the commitment covers is undecided.
- Reconciliation remains a persistence-clock concern; checkpoint + tail replay defines the future
  recovery-clock mechanism (unbuilt).
