# WAL shootout v1 — `LanceCycleWriter` vs the lancedb 0.40 MemWAL write path

**Status:** D-WAL-1 MEASURED (§2a), D-WAL-2 MEASURED (§3a), D-WAL-3 running. Feeds D-LNC-7 (the one-WAL
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

## 2a. D-WAL-1 results (2026-10-14)

`crates/lance-graph/tests/wal_shootout_probe.rs`, lance 13.0.0 / lancedb 0.40.0, debug build
(correctness only; no timing is read from it), local filesystem, M flush interval 10 ms.

**V1 SIGKILL atomicity — holds on both arms.** N = 4,096 casts (8,193 rows) per cycle:

| kill after | 5 | 30 | 60 | 120 | 250 | 500 | 1,000 | 2,000 ms |
|---|---|---|---|---|---|---|---|---|
| L, fresh reader | 0 | 0 | 0 | 8,193 | 8,193 | 8,193 | 8,193 | 8,193 |
| M, reader that has not written | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| M, same reader after its first write (replay) | 0 | 0 | 0 | 0 | 8,193 | 8,193 | 8,193 | 8,193 |

Both sweeps land on both sides of the durability boundary; no partial cycle anywhere.

**V2 acknowledged ⇒ durable — holds on both arms, with a visibility difference.** Kill right
after reading ACK 2, 5 and 9 (N = 1,024): L — every acknowledged cycle is readable by a fresh
reader. M — a reader that has not written sees **no** cycle at all; after its first write
(which opens the shard writer and replays the WAL) every acknowledged cycle is present.
Disable run (after commit): `durable_write = false` for M → acknowledged cycle 2 lost after
replay, V2 red; restored.

**V3 idempotent resubmit — holds on both arms.** L: `Committed` then `Reconciled`, 129 rows.
M: two upserts of one cycle, 129 rows (dedup by `pk`).

**V4 read horizon.** L: a reader opened at cycle 1 keeps seeing cycle 1 after cycle 2 commits;
a fresh reader sees both. M, after two acknowledged cycles: the writing handle's LSM read sees
both; a second handle in the same process (its own writer cache) sees **neither**; the
base-only read sees neither. After `close_lsm_writers` the second handle's LSM read sees both;
the base-only read still sees neither (no compaction ran).

**What this means.** Both paths are crash-atomic per cycle and lose nothing they
acknowledged. They differ in who can read an acknowledged write: L publishes a version every
reader can open; M's acknowledged writes live in the WAL and the writer's memtable, readable
by the writing handle, by any process only after a flush into a generation (here:
`close_lsm_writers`) or after that process replays the WAL by opening a writer, and by the base
table only after compaction. The sealed read horizon (`08-memwal-vs-batchwriter.md`) is
therefore not lost by MemWAL for OTHER readers — they see less, not more — but a reader
sharing the writer's handle sees in-flight state, which is the property item 2 of that doc
warns about.

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

## 3a. D-WAL-2 results (2026-10-14)

`crates/lance-graph/examples/wal_shootout.rs latency`, release, `-Ctarget-cpu=native`, the
program prints `avx512f=true`; three runs, ranges span the three. Local filesystem; the cost
is I/O and encoding, so it is reported per cycle (ms) and per stored row (µs). A cycle of N
casts stores 2N + 1 rows (N × 512 payload bytes).

| arm | N | commit p50 ms | commit p99 ms | µs per row (p50) |
|---|---|---|---|---|
| L | 64 | 3.5–4.8 | 5.9–8.3 | 27–37 |
| M, flush 10 ms | 64 | 7.0–8.1 | 10.5–14.0 | 55–63 |
| M, flush 100 ms | 64 | 96.9–97.4 | 101–103 | 751–755 |
| L | 1,024 | 7.9–9.0 | 12.5–13.9 | 3.9–4.4 |
| M, flush 10 ms | 1,024 | 5.3–5.6 | 15.8–17.2 | 2.6–2.7 |
| M, flush 100 ms | 1,024 | 93.5–94.3 | 101–102 | 45.6–46.0 |
| L | 8,192 | 72–80 | 90–106 | 4.4–4.9 |
| M, flush 10 ms | 8,192 | 34–35 | 82–90 | 2.1–2.2 |
| M, flush 100 ms | 8,192 | 89–91 | 172–179 | 5.4–5.6 |

After the run (200 cycles at N = 64 and 1,024, 40 at N = 8,192):

| N | disk L / M (MiB) | files L / M | versions L / M base | full scan L / M ms |
|---|---|---|---|---|
| 64 | 15.6 / 15.0 | 601 / 209 | 200 / 3 | 32–36 / 9–12 |
| 1,024 | 205.7 / 427–429 | 601 / 234–241 | 200 / 3 | 33–43 / 91–147 |
| 8,192 | 323.6 / 663.8 | 121 / 97 | 40 / 3 | 21–35 / 211–298 |

**Against the pre-registration.**
- "M's per-call latency tracks the flush interval for small N": confirmed and stronger than
  expected. With the default 100 ms ticker every acknowledged write waits about 94–97 ms at
  any N; at 10 ms, M is slower than L at N = 64 (7–8 ms vs 3.5–4.8) and faster from N = 1,024
  (5.4 vs 8.5 ms; 34 vs 76 ms at 8,192).
- "L's per-cycle cost flat in history": consistent over 200 cycles (checked further in the
  soak).
- Disk: M holds about 2× L's bytes at N ≥ 1,024. Hypothesis, not checked here: the WAL
  entries stay on disk after their memtable is flushed into a generation, so the payload is
  stored twice until something removes the WAL.
- Read: L's full scan stays at 21–43 ms; M's LSM scan grows with the data (9 → 91–147 →
  211–298 ms), because nothing compacts the flushed generations into the base table (the
  base table stays at version 3 throughout).

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
