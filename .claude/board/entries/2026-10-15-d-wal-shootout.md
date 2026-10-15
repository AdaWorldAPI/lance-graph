# 2026-10-15 — D-WAL — `LanceCycleWriter` vs the lancedb 0.40 MemWAL write path (measured)

**Status:** MEASURED. Plan `.claude/plans/wal-shootout-v1.md` §2a (D-WAL-1), §3a (D-WAL-2),
§3b (spec cross-check against lance `v13.0.0` `docs/src/format/table/mem_wal.md`), §4a
(D-WAL-3, truncated). Probe `crates/lance-graph/tests/wal_shootout_probe.rs`, harness
`tests/wal_common/`, example `examples/wal_shootout.rs`, all behind `lancedb-sdk`.

**Outcome:**
- Both paths are crash-atomic per cycle and lose nothing they acknowledged (SIGKILL sweep
  brackets the boundary on both). MemWAL needs `durable_write = true` for that; with
  `false` an acknowledged cycle is lost (disable run).
- MemWAL's acknowledged writes are visible to the writing handle only; another handle or
  process sees them after a flush or after it replays the WAL by opening a writer, and the
  base table only after compaction. The spec states this as its consistency model
  (strong only with live memtable access, otherwise eventual).
- Cost: with the default 100 ms flush every MemWAL write waits about 95 ms; at 10 ms it is
  slower than `LanceCycleWriter` at 64 casts per cycle and faster from 1,024. In the soak
  (256 casts, 10 ms) MemWAL's commit stayed flat at 2.0–2.8 ms while L's grew from 5.8 to
  about 10.5 ms over 1,649 versions.
- MemWAL keeps the WAL after flushing: disk grows by a WAL increment per cycle plus a
  second copy at each memtable flush (every ~195 cycles = the 100k-row default), 1.57× L at
  1,649 cycles. The spec describes no component that removes WAL files.
- Neither path compacts; both full scans grow linearly in rows.

- L's RSS growth is Lance's session metadata cache, not our writer (D-WAL-4,
  `examples/wal_rss_probe.rs`): each commit inserts a clone of the new manifest keyed by
  version, and in an append-only store manifest v lists v fragments, so the cache holds
  about V²/2 fragment entries up to its 1 GiB default. At 1,600 versions: default session
  628 MiB cache, 570 MiB RSS; session with an 8 MiB metadata cache 7.8 MiB, 52 MiB RSS.
  Capping the metadata cache, or compacting so fragment count stops growing, bounds it.

- Follow-up done the same day (D-WAL-5/6/7): the writer's session cap, a version clock
  (`contract::temporal_pov::VersionClock`, 4.2 ns per point-of-view read on v4), and index
  forms of the `temporal.rs` projections. D-LNC-7: L stays default, MemWAL an option.

**Open:** a full-length soak (deferred; this one stopped at 27.5 min when the container restarted); the effect of compaction on both; D-LNC-7
(which WAL to keep) stays a decision for the user.
