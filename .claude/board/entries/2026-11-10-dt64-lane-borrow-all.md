# 2026-11-10 — `LaneBinding::borrow_all`: one check, N borrows

**Status:** SHIPPED in this PR (`crates/lance-graph-dt64/src/lane.rs`).

**Why.** `LaneBinding::lane` re-runs `check` on every call, and `check`
walks every bound lane. Reading all N bound lanes therefore cost N² source
lookups. D-DT64-GEMM-SHAPE-2 (#1987) measured this at 256 fields: 155 µs
per resolve through `lane`, against 14 µs for a single check followed by the
borrows.

**What.** `borrow_all` runs `check` once and returns `BoundLanes`, an
`ExactSizeIterator` that borrows each bound lane in ascending field order.
- A refusal is returned before any lane is borrowed.
- A lane the source no longer holds yields `Err(Vanished { field })` at its
  position, and the remaining lanes are still handed out.
- Nothing is allocated.

**Tests** (`tests/lane_socket.rs`):
- lanes are borrowed in place and in field order, and match `lane`;
- N lookups against N² for `lane` per field;
- a refused check borrows nothing;
- a vanished lane is reported in place.

**Open:** #1987's lab still uses `check` + `addr` + `borrow` by hand and can
switch to `borrow_all` once this lands.
