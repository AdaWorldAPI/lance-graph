# D-DT64-7-5: `dr::MapTable`, the versioned global → local table

**Date 2026-11-01 · ID D-DT64-7-5 · Status IN PR · Plan `.claude/plans/d-dt64-7-behaviour-addressing-registration-v1.md` §1e**

`LocalMap` and `FrameStore` existed, and `LocalMap::digest` was documented as the frame's `coordinates` revision, but nothing tied a map to its frame. `MapTable` holds every tile's `LocalMap` under one `FrameStore`:

- `advance` derives `coordinates` from the held `(tile id, map digest)` pairs.
- `map(&Frame, tile)` refuses a stale or foreign frame (`MapRead::Stale`), and a missing tile (`NoSuchTile`).
- Existing `LocalMap` and `FrameStore` callers are unchanged: `tests/dr.rs` and `xlog-relcomp-probe` stay green.

**Disable runs:**
- Dropping the frame check turns 3 tests red.
- Dropping the tile id from the digest turns `tile renamed` red. The first version of the test used only a swap case, which stayed green without the tile id because order alone changes the digest.

**OPEN:** stored selections (D7'), which need O4: does one `0xF###` value name one stored selection, or a position in a wider basis?
