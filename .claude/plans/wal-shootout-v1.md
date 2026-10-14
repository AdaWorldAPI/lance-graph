# WAL shootout v1 — `LanceCycleWriter` vs the lancedb 0.40 MemWAL write path

**Status:** PRE-REGISTERED (2026-10-14), nothing run yet. Feeds D-LNC-7 (the one-WAL
decision, `lance-convergence-staged-migration-v1.md` §4C); this plan does not decide it.

## 1. What is compared

| arm | write | durable unit | read |
|---|---|---|---|
| **L** — ours | `LanceCycleWriter::commit_cycle(DetachedCycleBatch)` (`crates/lance-graph/src/graph/cycle_sink.rs`) | one Lance append = one `DatasetVersion` per cycle; frame + landing + image rows in one `RecordBatch` | plain Lance scan at the head version |
| **M** — lancedb 0.40 | `Table::set_lsm_write_spec(LsmWriteSpec::unsharded())`, then `merge_insert(["pk"])` upsert, which `table/merge.rs:320` dispatches to `execute_lsm_merge_insert` → MemWAL `ShardWriter::put` | one WAL entry per `put` (`durable_write = true` by default, flush ticker 100 ms) | LSM scan (`table/query.rs:178`: memtables + flushed generations); `use_lsm(false)` reads the base table only |

"Our hand-rolled batch writer" is the pair `BatchWriter` (staging, no I/O,
`batch_writer.rs`) + `LanceCycleWriter` (the durable half). `BatchWriter` does no I/O, so
only `LanceCycleWriter` has a MemWAL counterpart (`.claude/temporal/08-memwal-vs-batchwriter.md`).

Both arms store the same logical cycle: the rows `LanceCycleWriter::build_batch` writes
(1 frame row, N landing rows, N image rows with a 512-byte payload). M adds one column,
`pk: Int64` (`cycle << 24 | ordinal`; lancedb rejects `UInt64` as a primary key), carrying the unenforced-primary-key marker MemWAL
requires. Same N, same payload bytes.

Prior evidence, not re-run here: D-MW-P2 (lance 11, raw `ShardWriter::put`, 5,000 rows)
found MemWAL puts crash-atomic.

## 2. Validation (D-WAL-1), each test on both arms

- **V1 atomicity under SIGKILL.** A child process writes one composite cycle of
  N = 4,096 + 4,096 + 1 rows and is killed at a sweep of delays. Recovered rows per cycle
  ∈ {0, all}. The sweep must land on both sides of the durability boundary or the test
  reports INCONCLUSIVE (the D-MW-P2 anti-vacuity rule). For M, "recovered" is read twice:
  (a) by a fresh process that only reads, (b) after that process opens the shard writer
  (replay).
- **V2 acknowledged ⇒ durable.** The child writes cycles in a loop and prints `ACK k` after
  each call returns; the parent SIGKILLs right after reading an ACK. Every acknowledged
  cycle must be readable after restart. M is read the same two ways as V1. A cycle that is
  durable but invisible to a reader until a writer replays is reported as such, not as a
  pass or a failure.
- **V3 idempotent resubmit.** Submitting the same frozen cycle twice: L returns
  `Reconciled` and the row count is unchanged; M's upsert by `pk` leaves the count
  unchanged.
- **V4 read horizon.** While cycle n+1 is acknowledged but not flushed: what does a second
  reader see? L: a reader at the head it opened sees cycle n only until it re-opens. M: the
  LSM read in the writing process, an LSM read from a second handle, and a base-only read.
  Recorded as observed behaviour; the sealed-horizon property in
  `08-memwal-vs-batchwriter.md` §"three incompatibilities" item 2 is what this checks.

Scope: process crash on a local filesystem. Not power loss (no claim about fsync), not
object storage, not concurrent writers or fencing.

## 3. Cost (D-WAL-2)

Per arm, N ∈ {64, 1,024, 8,192} image rows per cycle, 200 cycles each, release build,
program-printed tier:

- commit latency per cycle: p50 / p99 (µs), and ns per row;
- after the 200 cycles: bytes on disk, file count, Lance versions (L) or WAL entries and
  flushed generations (M), and the latency of one full scan of all rows;
- M twice: default flush interval (100 ms) and 10 ms, because `durable_write` waits for
  the WAL flush and the ticker can dominate a small cycle.

Direction only, no threshold before the run: M's per-call latency should track the flush
interval for small N; L's per-cycle cost should be flat in history length while its read
cost grows with the number of versions/fragments (never compacted).

## 4. Soak (D-WAL-3) — "monitor both for a while"

One background process per arm, own directory, same cadence: one cycle of 256 image rows
every second for 60 minutes (about 3,600 cycles, about 0.5 GB per arm at 512 B per row).
Every 30 s it appends a CSV row: cycles committed, commit p50/p99 over the window, bytes
on disk, file count, versions (L) / generations (M), full-scan latency, RSS. Questions:
does commit latency drift with history; does read latency grow without compaction; does
disk grow linearly; does RSS stay flat (M holds memtables).

Disk budget is checked before the start (this host has about 5 GB free); the soak stops
early if free space falls under 1 GB, and says so.

## 5. Deliverables

| D-id | scope | gate |
|---|---|---|
| D-WAL-1 | V1–V4 on both arms (`crates/lance-graph/tests/wal_shootout_probe.rs`, behind `lancedb-sdk`) | V1: no partial cycle on either arm, sweep brackets the boundary; V2: no acknowledged cycle lost after the replay read; V3/V4: observed behaviour recorded |
| D-WAL-2 | cost table (`crates/lance-graph/examples/wal_shootout.rs latency`) | numbers reported with tier; no threshold |
| D-WAL-3 | 60-minute soak, both arms, CSV + summary | drift, growth and RSS reported per arm |

## 6. Fences

- No production wiring: `LanceCycleWriter` keeps zero callers outside tests; nothing here
  enables MemWAL anywhere. D-LNC-7 stays a decision for the operator.
- lance and lancedb come from crates.io (the upstream carve-out); no new dependency beyond
  the existing optional `lancedb-sdk` feature.
