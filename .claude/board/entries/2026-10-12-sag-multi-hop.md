# 2026-10-12 — D-SAG-1-4 — multi-hop questions over the software graph

**Status:** In PR (stacked on #1519).

`SoftwareGraph::hops(start, path)` walks `(relation, direction)` hops with set
semantics per hop. Each hop is one Quack program over the resident edge lanes,
`rel = R AND from IN (S)`, keeping its rows. The previous hop's survivors arrive
as a value set the size of those survivors; no hop reads a plane another
program produced. This is the operator's reading for chained hops: the masks
of one query fold into one materialisation, each hop filtered at the source.

MEASURED (tests/awareness.rs): co-readers, field → claim, a four-hop claim
co-dependency and a four-hop shared-primitive path equal a row loop over the
edges; a hop path follows a reprojection; four disable runs red (relation
filter, direction, per-hop dedup, unknown start).

OPEN: a value set still reads every edge row. Skipping straight to the
survivors' rows needs a Lance B-tree filter (D-TELEPORT-2 arm Db) or address
ranges over a key-ordered projection, which waits on
ISS-WITNESSED-RANGE-DOES-NOT-ATTEST-PLANE-ORDER.
