# 2026-11-28 — D-TGB-7 S2c: thought spaces keyed by the generation-qualified slot; `discard`

**Status:** TEST-PINNED (`lance-graph-thought-git/src/spaces.rs`: `discard_leaves_the_base_and_the_neighbours`, `a_reused_slot_does_not_read_the_former_thought`; 21 unit tests; 2 disables red: discard keeping the deltas, discard keeping the placement). Board: EXO-379, D-TGA-1.

`spaces::ThoughtId` is now `slots::Slot` (index + generation), not a bare `u8`, so a slot index reused under a new generation is a different thought. `ThoughtTables::discard` removes the thought's placement, saccades and deltas and frees its table or slot; the borrowed base (W0) is untouched and neighbours in a shared table keep their slots. No new type: the probe's `Slot` gains `Hash`/`Ord` derives to key the maps.

OPEN: the pool and the tables are still two objects; a caller must discard in the tables when it frees in the pool. The planner's `nars::warrant::ThoughtRef` is the other `(slot, generation)` type (owner decision, EXO-379).
