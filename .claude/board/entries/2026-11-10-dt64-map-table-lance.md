# 2026-11-10 — DR map table made durable through the fire-and-forget cycle writer

**Status:** VERIFIED-IN-CODE (12 tests; the round trip runs on a real
`LanceCycleWriter`). Answers the storage half of D-DT64-7 O5 / D10'
(D-LITHO Q5: "which table"). WAL choice: D-DR64-WAL-1 (EXO-225).

> ⊘ **SUPERSEDED the same day:** the first version of this crate wrote its
> own Lance dataset synchronously, compared dataset versions to refuse a
> stale writer, and reported a post-commit `Race` / `ClockMoved`. That is a
> custom WAL with arbitration, which the persistence doctrine forbids
> (`.claude/v3/knowledge/d-mbx-a6-owner-consume-and-persistence.md`). It was
> rebuilt on the cycle writer.

**What.** `crates/lance-graph-dt64-lance` (excluded, own workspace) makes
`dr::MapTable` durable. dt64 stays free of lance and arrow.

- `MapLog::advance` claims the route version (`MapTable::claim`), turns the
  advance into casts on behalf of the owning mailbox, submits them with
  `CycleWriterHandle::try_submit` (cycle id = route version) and returns at
  once. One background task owns the `LanceCycleWriter`.
- `MapLog::settle` applies an advance with `MapTable::advance_at` only when
  its seal ticket is `Ok`. A failed seal applies nothing and its version is a
  gap; every later advance is refused (`Poisoned`), matching the writer,
  which seals nothing after a failure.
- One 512-byte cast per 64 ids: a W64 tile is one cast, a W256 tile four, a
  removal one; each advance adds one marker cast (version, source,
  relation-map revisions).
- `restore` reads every sealed cycle (`scan_sealed` + `scan_image`) and
  re-applies each advance at its own route version. It refuses a landing on
  behalf of another mailbox and any image that is not a valid log.
- There is no arbitration: one owner holds the `MapLog`, which owns its
  table mutably. Two writers on one dataset are outside the contract
  (`LanceCycleWriter` fences on its own cached head).

**Tests.** 4 unit + 8 integration. Disable runs, each restored:
- applying without a sealed ticket → `a_failed_seal_applies_nothing_and_poisons_what_follows`;
- no owner check on restore → `every_landing_is_on_behalf_of_the_owner`;
- applying at the clock's current version instead of the claimed one → 4 tests,
  including `a_clock_tick_between_claim_and_seal_lands_at_the_claimed_version`;
- no poison gate → the failed-seal test;
- no reserved-id check → `a_reserved_id_or_a_repeated_tile_is_refused_before_a_version_is_claimed`.

**Open.**
- Frames are not restored: a frame is one table instance's witness, so a
  frame from before a restore is `ForeignFrame` (pinned by a test). Issuing a
  frame at a past version after a restore is a DR owner decision.
- A table whose clock is shared with a `RouteLog` needs the route log
  durable too; only the map is stored here.
- The map table's compaction schedule (L's files and scan time grow with
  advances; EXO-225).
- lance-graph does not build without `ndarray-hpc`; this crate enables it.
