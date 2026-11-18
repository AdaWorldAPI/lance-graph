# D-XLOG-W7-2-STEP-4A: a cube relation's two lanes are lent through the lane socket (2026-11-18)

**Status:** MEASURED (test-pinned). No timing taken, so no cost is claimed.

Shape (owner, 2026-11-18): "borrow_all + xlog::cube.rs is the proposed shape
(identity[index].key :identity[index].value = X;Y". Both halves existed:
`LaneBinding::borrow_all` / `borrow_all_present` in DT64, and the cube's
`Rel { n, cols: [Vec<u32>; 2], live }`. Nothing joined them: the cube
implemented `RelationSource` only.

`crates/lance-graph-xlog/src/cube/lend.rs`, `CubeLanes`:

- Lends relation `rel`'s two effective lanes as the two fields of a class,
  by the predicate the class view resolved, and the relation's live plane as
  each lane's presence (`impl LaneSource`, `impl PresenceSource`).
- Owns nothing. The lanes are `Cube::lanes` (the base itself when the delta
  log is empty, else the cached fold). The frames are the map table's
  (`MapTable::frames`), the store the cube's frame was sealed from. That
  settles the open point of who owns the frame store: the socket borrows the
  cube and its table together, as `AbiLanes` pairs a batch with its store.
- Its `CoordSpace` is the relation's row order: base generation and delta
  length. A logged edge does not advance the cube's frame (#1941 D3, still the
  operator's to rule), but it changes the row order, so a binding made before
  it is refused after it.

Tests (`tests/lend.rs`, 5; whole crate 97 passed):

- the lent lanes, decoded through the tile's map, are the relation's tuples
  in order; a second borrow lends the same words (pointer-equal); the view
  lists `y` before `x`, so a socket that mapped positions would swap them;
- one `GroupReduce Count` over the lent lanes and live plane equals
  `Cube::column_counts` for both columns (40 rows, repeats in both);
- a binding goes stale when an edge is logged and when the table advances;
- another class, another scope, an unknown predicate and an unknown relation
  are refused.

- a binding made over one cube is refused over another cube built from the
  same map table whose relation is in the same state (review finding on
  #2136: the coordinate space named the relation and its state but not the
  cube, so such a binding read the other cube's tuples; the space now carries
  the cube's identity).

Disables, each red: the row order ignoring the delta log; columns handed out
by position; any class served; the coordinate space without the cube's
identity.

Open:
- Bodies, `reach` and `closure` do not read through the socket; this lends,
  it does not replace a terminal.
- A tiled cube's lanes hold wide coordinates; the socket lends them as they
  are and no test reads a tiled one.
- No production caller: a method executed over these lanes
  (`foundry-parity-lab::method` is the existing shape) is the next step.
