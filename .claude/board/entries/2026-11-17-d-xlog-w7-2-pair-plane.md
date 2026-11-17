# D-XLOG-W7-2 step 1: `dr::PairPlane` (2026-11-17)

**Status:** MEASURED (tests); no reader yet. Step 1 of the order in
`2026-11-17-d-xlog-w7-2-admission.md`; steps 2-5 not started.

`crates/lance-graph-dt64/src/dr/pair.rs`, re-exported as
`lance_graph_dt64::dr::{PairPlane, PairDir, PairRefusal}`:

- One bit per coordinate pair of a binary relation, over a source `Axis` and
  a target `Axis` (DR64 or DR256 tiles, any count; the axes may differ).
- Both directions kept as row masks, LSB-first `u64` words, tail bits zero:
  `row(frame, Forward, x)` is every `y` with `(x, y)`, `Backward` every `x`.
  `insert` / `remove` update both and report whether the pair changed.
- Bound to the frame issue it was built under (`Frame::same_issue`): a read
  under another table's frame, or under a newer frame of the same table, is
  `ForeignFrame`. A re-seal of an unchanged table reads.
- Coordinates past an axis are `OutOfAxis`, naming the axis, and change
  nothing.
- Contents are the owner's: DR places ids (`MapTable::locate`), the owner
  inserts pairs. This type does not read lanes.

Tests (`tests/dr.rs`, module `pair_plane`, 3): 4000 random inserts / removes
on DR64×2 by DR256×1 against a set oracle, both directions and `len` checked;
frame-issue refusals; axis refusals. Disables, each red: backward row not
cleared on remove, not set on insert, frame check always passing, axis check
removed, pair count not incremented.

Next: step 2, the DT64 socket that lends a plane by `(relation, frame,
direction)`.
