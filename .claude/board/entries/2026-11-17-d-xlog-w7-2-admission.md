# D-XLOG-W7-2 admission analysis: lifting the #1943 views (2026-11-17)

**Status:** MEASURED, read-only (no code). The W5-A shape: which parts of the
#1943 lab (D-GLOBAL-WHERE P0) production already has, which are missing, and
which wait on a decision.

Direction (operator, 2026-11-17): relation presence is a DR and DT64 concern
("both"). DR owns the pair plane, DT64 is the socket, and the cube implements
the socket. The cube does not own the plane's type.

## The three parts of a #1943 view

The lab (`.claude/probes/2026-11-03-d-global-where/probe/src/lib.rs`) answers
W4/W6 with three things:

1. **`Plane`** (lib.rs:281): a square bit matrix over DR coordinates. Row `x`
   holds the `y` with an edge `x → y`, and the wide coordinate is
   `ordinal * tile_width + local`. `falsify.rs:257-263` builds `fwd` and `rev`
   from an edge list through `coord_of`.
2. **`coord_of`** (lib.rs, after `Plane`): `MapTable::locate`, then
   `ordinal * tw + local`.
3. **`route_selection`**: for every local of a selection tile, local → global
   (the pointer table), then `locate` into the relation's tiles. It writes a
   domain mask.

## What production has

| part | production | where |
|---|---|---|
| global → (tile, local) | `MapTable::locate` reads every listed tile and refuses an id held by two of them (`Ambiguous`) | `lance-graph-dt64/src/dr.rs:1123` |
| wide coordinates | `Cube::coord`: `locate` for a tiled cube (`Cube::new_tiled`), `local` for one tile; counted | `lance-graph-xlog/src/cube.rs:768` |
| relation lanes in coordinates | `Rel { cols: [Vec<u32>; 2], live }`: both columns already as positions, with a delta log and `refold` | `cube.rs:357` |
| pair plane | **none owned.** It is rebuilt per call from the lanes: in `body.rs` (`Plan::new`, the `rows` arena, one 4-word DR256 row per position per direction) and as the edge rows of `Cube::reach` / `closure` | `cube/body.rs`, `cube.rs` |
| a socket for it | **none.** DT64's `PresenceSource::borrow_presence(&LaneAddr)` is one bit per row of a lane (absent vs zero, EXO-294); a relation needs one bit per coordinate pair | `lance-graph-dt64/src/lane.rs:137` |
| selection routing | **none in production**; the lab's `route_selection` | lab only |

## Findings

1. **"A global → tile index" is not missing.** `locate` exists and the cube
   pays it once per id at ingest, so its lanes already hold wide coordinates.
   Its cost is O(tiles × log 256) per id, because it reads every listed tile;
   that is an ingest cost, not a per-query one. It becomes a question only if
   the tile count grows large, and nothing here measures that yet.
2. **The pair plane is the missing piece, and it is a materialization today.**
   Both production readers (bodies, reach) rebuild it from the lanes on every
   call. An owned plane, maintained with the relation's delta log (set a bit on
   add, clear it on retire, rebuilt by `refold`), removes that per-call build.
   Its size per relation per direction is `coords × ⌈coords/64⌉` words: 8 KiB
   for one DR256 tile, 128 KiB for four tiles (1024 × 16 words).
3. **EXO-46 gates the selection half, not the plane half.** `route_selection`
   is the counted global → focus lookup EXO-46 asks to admit (local → global →
   `locate` per selection position). The plane is built from lanes that are
   already in coordinates, so it needs no new lookup.
4. **A body over a tiled cube needs only a wider row.** `body.rs` fixes
   `ROW_WORDS = 4` (one DR256 tile). Over `T` tiles a row is `4T` words; the
   walk is unchanged. `U64x4` would become a loop over `T` of them.

> **Finding 1 corrected the same day, after measuring.**
> `cargo run --release -p lance-graph-dt64 --example locate_cost`
> (native AVX-512; tiles of 256 ids each; every located id checked, an
> unmapped id refused):
>
> | tiles | ns per `locate` | per tile visited | one `LocalMap::local` |
> |---|---|---|---|
> | 1 | 18 | 18 | 8 |
> | 4 | 74 | 19 | 10 |
> | 16 | 327 | 20 | 8 |
> | 64 | 1 819 | 28 | 8 |
> | 256 | 8 535 | 33 | 8 |
>
> The cost is linear in the tile count. Each visited tile spends 10-25 ns
> reading its map under the frame, on top of the 8 ns lookup. At 256 tiles,
> placing 65 536 ids costs about 0.56 s. So the lookup exists, but an index
> from global id to tile is needed past a handful of tiles: it is an ingest
> cost, and it grows with the cube. Still not a blocker for steps 1-3, which
> stay on one tile or a few.

## Proposed order (each its own PR, none started)

1. DR type: a pair plane over two axes of one frame (DR64 = 1 word per row,
   DR256 = 4), with its frame and route version, and refusing a read under a
   foreign frame as `PresenceSource` does.
2. DT64 socket: `borrow_relation_rows(rel, frame, direction) -> &[u64]`,
   handing out the owner's plane with no copy.
3. Cube implements it: the plane maintained with the delta log; `body.rs` and
   `reach` read lent rows instead of building their own. Falsifier: the lent
   rows equal the rebuilt ones after every add / retire / refold, and the body
   answers stay equal to the nested-loop oracle.
4. Bodies over a tiled cube (`4T`-word rows).
5. Selection routing, after EXO-46 is decided.

## Open

- EXO-46 (operator): admitting the counted global → focus membrane, which
  `route_selection` needs.
- A global id -> tile index for cubes past a handful of tiles (measured
  above); its design is not proposed here.
