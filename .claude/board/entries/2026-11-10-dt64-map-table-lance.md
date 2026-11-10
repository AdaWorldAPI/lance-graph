# 2026-11-10 — DR map table stored as a separate Lance-versioned dataset

**Status:** VERIFIED-IN-CODE (tests on a real Lance 13 dataset). Answers the
storage half of D-DT64-7 O5 / D10' (D-LITHO Q5: "which table").

**What.** `crates/lance-graph-dt64-lance` (excluded, own workspace, like
`lance-graph-dir-sim-lance`) stores `dr::MapTable` in Lance. dt64 stays free of
lance and arrow.

- One `MapTable::advance` is one Lance commit. Rows are written first, then
  the same advance is applied to the table; a refused or failed commit leaves
  the table unchanged.
- One row per local position (D10'), plus genesis, advance and removal
  markers. Ids only.
- Two clocks, never mixed: the DR route version is in the rows; the Lance
  version is the physical commit counter (stale writer, race, and
  `restore_at(k)` via `checkout_version`).
- `restore` replays every advance at its own route version, so an extension
  stays an extension (generation and event counts equal), a replacement a new
  generation, a removal an event. Route-version gaps from a shared clock
  replay exactly. Rows that are not a log are refused.

**Tests.** 4 unit + 6 integration. Disable runs, each restored:
stale-writer check removed → `a_stale_writer_writes_nothing` and
`create_refuses_an_existing_dataset` fail; gap ticks removed →
`route_versions_with_gaps_replay_as_route_versions` fails.

**Open.**
- Frames are not restored: a frame is one table instance's witness, so a
  frame from before a restore is `ForeignFrame` (pinned by a test). Executing
  at a past version after a restore needs a frame issued at that version —
  a DR owner decision.
- A table whose clock is shared with a `RouteLog` needs the route log
  stored too; only the map is stored here.
