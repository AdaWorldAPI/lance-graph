# D-REGC-1 / -2 / -8: the read by reference, and the cube over three tiles (2026-11-21)

**Status:** MEASURED (test-pinned), in PR.

**Basis:** owner, 2026-11-21. A `ClassView` changes only while the class
proposer stages classes that do not exist yet. A registered class changes
only by a new versioned reference. Data is replayable relative to data
version × DR reference × BindStore version.

**D-REGC-1 / -2** (`lance-graph-dt64`, `tests/bound_read.rs`, 4 tests):
- **What it does:** `check_bound` and the `*_bound` borrows take no view.
  They check frame, route, the store's revision and digest (O(1)), each
  lane's mapping, scope and row order. They lend the same lanes, planes and
  positions as the view read, as the same pointers.
- **Refusals that fire:** a different-basis store (`StoreBasisMismatch`), a
  stale frame, a foreign frame store, a route moved, a remapped lane, a
  selection in another row order.
- **View changes:** a view whose fields moved after bind is refused by
  `check(view)` and changes nothing the bound read returns.
- **Disable:** `disable-bound-digest` fails the different-store test
  (`Ok(())` instead of `StoreBasisMismatch`).
- **pg-row gate:** with `Reader` on the bound read, it still equals
  PostgreSQL 16.15 on 4 of 4 cases.

**D-REGC-8** (`lance-graph-xlog/tests/cube_tiled_bijection.rs`):
- 600 scattered ids in 3 shuffled W256 tiles.
- Every id has one wide coordinate. None is shared, and each decodes back
  through its tile's pointer table.
- An unmapped id has no coordinate.
- Logged edges, a retire and a refold leave every coordinate unchanged.
- Placing a key costs one counted lookup per listed tile: 1800 for 600 keys
  over 3 tiles. That is D-REGC-7's baseline.

**Open:**
- Measuring the basis share of the census.
- D-REGC-7 (resolve once per distinct key), which needs D-REGC-5 / EXO-329.
- D-REGC-3, -4 and -13, which need decisions.
- D-REGC-6, which needs the predicate count first.
