# EXO-328: tiles and codes under one frame store (`MapTable::advance_codes`)

**Date:** 2026-11-18 · **Status:** MEASURED (test-pinned) · **Scope:** lance-graph-dt64

A source that owns row identities and coded text columns could not put a `MapTable` and a `CodeTable` under one frame. `MapTable` hands its `FrameStore` out read-only (`frames()`), and `CodeTable::advance` needs that store mutably and refuses any other (`ForeignStore`). The choices left were two stores, with a lane binding checked against only one (`LaneSource::frames` returns one), or a code table that never changes. Found by zammad-rs while moving its id map onto `MapTable` (EXO-328, EXO-329).

Shipped: `MapTable::advance_codes(&mut self, codes, changes)`. It calls `codes.advance(&mut self.frames, changes)`: one advance of the table's own store, made by the code table. No tile and no `coordinates` revision moves. A frame sealed before the call is no longer current and still reads its own tile layout and its own code generation.

Not chosen: a `MapTable` over a caller-owned store (the caller could advance it apart from the tiles, and `coordinates` would stop matching the map), and two stores (a code change would leave every lane binding current).

Evidence: `tests/dr.rs`, five tests: a code change retires the frame the tiles were read under; no tile and no coordinate moves; tiles and codes share one version sequence across interleaved advances; a code table bound to another store is refused and nothing changes; a change set that is not a bijection is refused and nothing changes. Disable (the method made a no-op): four of the five red; the one that stays green is the test that nothing moves.

OPEN: none.
