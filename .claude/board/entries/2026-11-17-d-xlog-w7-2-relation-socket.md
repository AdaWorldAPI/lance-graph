# D-XLOG-W7-2 step 2: the relation socket (2026-11-17)

**Status:** MEASURED (tests); no implementer yet. Step 2 of the order in
`2026-11-17-d-xlog-w7-2-admission.md`; step 1 (`dr::PairPlane`) merged in
#2126; steps 3-5 not started.

`crates/lance-graph-dt64/src/relation.rs`, re-exported at the crate root:

- `RelationSource::borrow_relation(rel) -> Option<&PairPlane>`: an owner
  lends the plane it maintains, never a copy.
- `relation_rows(source, rel, frame, dir) -> Result<RelationRows, RelationRefusal>`:
  the one read path. A relation the owner does not hold is `NoRelation`,
  never read as empty; a frame other than the plane's issue is
  `Pair(ForeignFrame)`.
- `RelationRows`: the lent words with their row width and count; `row(at)`
  is `None` past the axis.
- Standalone, not a `LaneSource` extension: the cube, its first implementer,
  is not a lane source.

Tests (`tests/lane_socket.rs`, module `relation_socket`, 2): the lent words
are the owner's own (same address) and every row equals the plane's in both
directions; missing relation, foreign table frame and newer frame of the same
table refused. Disables, each red: rows copied, a missing relation answered
with another plane, row bound off by one (the first attempt at this disable
was equivalent to the guard and stayed green; re-run as a plain `<=`).

Next: step 3, the cube maintains a plane per relation with its delta log and
`body.rs` / `reach` read lent rows.
