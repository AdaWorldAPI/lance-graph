# D-XLOG-W5-D: tiled `reach`, and the W7 column decision (2026-11-16)

**Status:** MEASURED (W5-D, lab); DECISION (W7 parked).

## W5-D: `reach` on a tiled cube

A tiled cube (`Cube::new_tiled`) already holds its relation lanes in wide
coordinates (`tile_ordinal * 256 + local`), and `reach_from` works on
positions only. Three `Tiled` guards were the whole block, plus `add_set`
placing ids through one tile's map. Now `add_set` places each id with
`coord()` (wide coordinate on a tiled cube, an unmapped id refused as
`Locate`), and `set_mask` / `reach` run unchanged over `256 * tiles`
positions. The answer mask is indexed by wide coordinate.

Cost note: `reach_from` builds a dense edge matrix of `n * n / 64` words,
`8192 * tiles²` bytes (2 MiB at 16 tiles). Correct, not a performance claim.

`tests/reach_tiled.rs`: three `W256` tiles holding 0..694 (wide coordinate =
global id); three graphs (a ring through every tile plus chords, a dense and a
sparse random graph), random seed and target sets; `Cube::reach` and
`set_mask(SetExpr::Reach)` both equal an independent BFS position for
position, the answer adds positions beyond the seeds, and with the ring it
spans all three tiles.

Disables, each red, each restored from the commit:
- ids placed through the first tile's map only: both tests fail.
- `reach` truncated to one block's words: the BFS test fails, by a slice
  length panic in `eval` rather than by its assertion.

Open: the xlog fork's one-column route builds single-tile cubes, so no census
row exercises the tiled path yet.

## W7: column construction for n-ary heads — PARKED

Measured over `census-007bb19.tsv` (refused + refused-on-data = 128 rules),
grouped by head arity from the program's `pred` declarations: 46 have
heads of 3 or more columns (`tri` 3, `cyc` 4, `clique5` 5, `support_1` 7,
`support_2` 10, `usable` 3, …). Not in the 46: 19 query rules and 8 whose
arity the count could not read (XLOG-word and GroupBy heads). The W7 targets `support_1`, `support_2` and
`usable` are all among them.

DECISION: do not build n-ary columns in the cube in this arc.
SCOPE: `lance-graph-xlog::cube` and the xlog fork's cube route.
BASIS: the cube stores relations as pair masks; holding a 3+ column answer as
cube columns is storing tuples as owned rows, which CLAUDE.md's P0 zero-copy /
no-tuple rule forbids at any speed. These rules already fail closed (refused,
reader refused), so nothing is answered wrongly today.
REVISIT WHEN: an n-ary answer representation that is lanes plus a pointer
table (or factorized) is measured in nanoseconds per query terminal against
the naive oracle on these programs.

The arity ≤ 2 refusals stay cube-addressable work: `filter other than
col == const` (16, all one-column heads), `Join body` (5 one-column, 1 binary),
`computed or out-of-range output column` on binary heads (24).
