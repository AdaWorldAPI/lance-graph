# D-XLOG-W7-2-STEP-3: the cube owns and lends its pair planes (2026-11-18)

**Status:** MEASURED. Bodies read lent rows; `reach` and `closure` are
unchanged and still build their own edge rows (not done here). Steps 1 and 2
merged in #2126 and #2127; steps 4 and 5 not started.

Rule applied (operator, 2026-11-17): store the cross-reference, not the
materialized answer; a derived answer is a presence mask over coordinates that
already exist.

`crates/lance-graph-xlog/src/cube.rs`:

- `Cube.planes`: one `dr::PairPlane` per relation, holding the relation's
  EFFECTIVE pairs (base folded with its delta log) in both directions, under
  the cube's frame, over `256 * tiles` coordinates per axis.
- Built on first read, from the lanes. Dropped wherever the effective lanes
  are dropped: an edge logged, a refold, a commit that grew the map, a rebase.
  It is rebuilt from the lanes, never patched, so a retire cannot leave a bit
  behind.
- `impl RelationSource for Cube`: the plane is lent through dt64's relation
  socket; an unknown relation is `NoRelation`, a frame the cube has left is
  `Pair(ForeignFrame)`.

`cube/body.rs`: `Plan` no longer allocates and fills a row arena per call. It
borrows each atom's two row blocks from the cube's plane.

Tests (`tests/body.rs`, module `lent_planes`, 4; the 7 earlier body tests
unchanged and green): the lent plane equals the effective edges in both
directions after add, retire, refold and add-after-refold; a triangle body
agrees with the oracle before and after a retire and a re-add; two reads with
no change lend the same words (same address), with a body call in between;
unknown relation and left frame refused, the rebased cube lends the same
pairs under its new frame.

Disables, each red: plane not dropped on a logged edge; planes kept across a
rebase; plane built from the base lanes instead of the effective ones; plane
rebuilt on every read.

`body_cost` (release, native AVX-512, 256 nodes, counts asserted against the
edge-walk oracle), µs, before → after: triangle 22 / 45 / 109 → 18 / 38 / 72
at 512 / 2048 / 8192 edges; four-cycle 45 / 293 / 532 → 39 / 292 / 566. The
example calls one cube 31 times, so "after" is the cost with the plane already
built; the first call also pays the build.

Open:
- `reach` / `closure` still build edge rows per call.
- A plane on a tiled cube is built (`256 * tiles` per axis) but no test reads
  one, and bodies still refuse a tiled cube (step 4).
- Presence-writer lifetime (#1941 D3): the plane is dropped and rebuilt under
  the same frame when the owner logs an edge. Whether an owner's write must
  instead issue a new frame is the operator's to rule.
