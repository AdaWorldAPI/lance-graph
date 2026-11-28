# 2026-11-28 — D-TGB-7 S2: the resident thought spaces move into `lance-graph-thought-git`

**Status:** TEST-PINNED (`lance-graph-thought-git/src/spaces.rs` unit tests, 19, moved unchanged; 2 disables red after the move: merge on read returning the base → 10 red, absent lanes written over known values → 4 red). Board: EXO-379, D-HL-5.

`ThoughtTables`, `RowDelta`, `Placement`, `LaneBinding` and `Written` moved from `cognitive-shader-driver/examples/thought_layers_probe.rs` into `lance_graph_thought_git::spaces` with their tests; a diff against main shows only the header, one import and the test fixture `base()` moving into the test module. The example keeps its demo `main` and runs the crate. The crate now depends on the contract and the sibling ndarray (`codec9::EpiLane9` decides a never-touched 9-bit lane); the manual workflow checks ndarray out beside lance-graph for both jobs.

Not changed in S2: `ThoughtTables` still names thoughts by `ThoughtId(u8)` and tables by `TableId`, not by generation-qualified `slots::Slot`. Binding the two (a reused slot must not read a former thought's deltas) is the next change, separate from the move.

OPEN: as in S1 (one `(slot, generation)` type; EXO-148; where the receipt row lives).
