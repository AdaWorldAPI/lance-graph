# D-XLOG-CUBE-SAFE-0: cube safety repair (P0, follow-up to D-XLOG-COV-0)

STATUS: fixed | OUTCOME: the two defects pinned in
`crates/xlog-coverage-probe/tests/safety.rs` (A: `refresh_closure` accepted a
closure of a stale base; B: `commit` adopted a wrong frame in release builds)
are typed refusals in every build, and a refusal leaves the cube unchanged.

- `BaseClosure` (minted only by `Cube::base_closure`) carries cube id,
  relation, base generation and frame. `refresh_closure` and `stage_closure`
  refuse `ForeignClosure`, `ClosureRelation`, `StaleBase`, `ClosureFrame`.
  `refold` advances the generation only when it folds a non-empty log.
- `Cube::commit` returns `Result<(), CubeRefusal>`; before any mutation it
  refuses `ForeignBatch`, `StaleBatch` and `FrameMismatch` (the grown tile's
  map is compared by width and global ids, not length). `Cube::check_commit`
  runs the same checks before a caller advances its `MapTable`.
- `lance-graph-codegraph` caches a `BaseClosure` while the log is empty,
  refreshes only that kind, and pre-checks the commit before mutating.

Evidence: `crates/lance-graph-xlog/tests/cube_safety.rs` (11 tests, debug and
release). Nine disable runs, one per guard, each failed exactly its named
test. lance-graph-xlog 67/67, lance-graph-codegraph 145/145 (debug and
release), xlog-coverage-probe 31/31 (debug and release, compiling the XLOG
fork's `xlog-cube` against this checkout).

OPEN: P1 (end-to-end `add_pm` over triangle pairing) unchanged.
