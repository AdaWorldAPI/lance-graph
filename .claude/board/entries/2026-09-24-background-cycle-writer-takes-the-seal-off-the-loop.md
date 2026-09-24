# 2026-09-24 — the background cycle writer takes the seal off the thought loop

**Status:** MEASURED + TEST-PINNED (`crates/lance-graph-supervisor/src/cycle_writer.rs`) · OPEN (the rest of D-HWV-1)
**D-ids:** D-HWV-1a (new, In PR) — the first built piece of D-HWV-1 (hot version window).

## What it is

`cycle_writer(sink, head, depth)` returns a writer that owns the WAL sink and a handle for the
thought loop. `try_submit(cycle, casts)` moves the cast vector into a bounded queue and returns a
`SealTicket` at once; the writer runs the existing `seal_cycle` — sort, fold copy, hash, the
retry-cache clone and the Lance commit — in submission order. Still exactly one writer: the task
owns the sink.

- **The writer frames each cycle.** `base_version` = the head the previous cycle published
  (unchanged on `NoChange`, the reconciliation head on `Reconciled`), so cycle N+1 can be queued
  before N has landed.
- **Durability barrier.** `durable()` / `wait_durable(cycle)` report the last landed cycle; that
  progress is always a gap-free prefix of the submitted cycles.
- **Failure poisons, nothing is lost.** The failing ticket gets the unchanged `SealFailure`
  (its `recovery()` still applies); every queued cycle gets `Poisoned` with its casts back; new
  submissions get `Closed` with their casts back. Recovery = resolve, then a fresh writer on the
  durable head.
- **Steps unchanged.** Transitions arrive on the ticket; apply only after it resolves `Ok` — the
  existing "no sealed version, no applied step" rule. Running steps ahead of durability is NOT
  decided here.

## Measured (release, 4-core host, median of 7; no Lance I/O — the fake sink commits instantly)

| per cycle, 64k casts × 512 B | loop pays |
|---|---|
| `seal_cycle` inline (freeze + retry clone) | **72.2 ms** |
| `try_submit` to the writer | **1.9 µs** |

The Lance write itself (not measured here) moves off the loop too.

## Pinned by tests, each disable-verified

| assertion | disable that turns it red |
|---|---|
| cycle 2 is framed on cycle 1's published version | writer keeps its starting head |
| a failed seal poisons the queue; queued casts come back intact | never set the poison |
| an out-of-order cycle id is refused, the writer keeps going | drop the order guard |
| a writer seal is byte-identical to an inline seal (same outcome, hash, commits) | control |
| a full queue returns the casts (`Full`) | control |

## OPEN

- The rest of D-HWV-1: RAM publication (`published_head`) visible to readers before durability,
  batched durability (K commits + one sync), and whether steps may apply before durability.
- The fold still copies every payload (on the writer now). Moving payloads into the image instead
  needs the Lance writer's 512-byte ABI check (`cycle_sink.rs`, reads landing payload lengths) to
  move to the image — a separate change.
