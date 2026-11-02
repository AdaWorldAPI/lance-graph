# D-DR-VMAP — the DR global→local index, versioned, immutable, replayable

STATUS: PLAN. Ruling: the versioned mapping IS `dt64::dr`'s index (old version =
same lookup, version in the key); D-DYNBAG §4.3 (plan ll.120-123).

## 1. Findings (read, file:line)

- `RouteLog` already IS the pattern (dr.rs:177-182): `routes: BTreeMap<u16,
  Vec<(RouteVersion, Option<u32>)>>`, append-only `push` (dr.rs:219-226),
  `routed_at` = `partition_point(v <= version)` (dr.rs:245), `FutureVersion`
  the only refusal (dr.rs:235). Nothing overwritten except the counter.
- `RouteVersion(pub u64)` (dr.rs:155): append-only log position, +1 per event; Ord, Copy.
- `FrameRevisions` (dr.rs:290-295) = (source, relation_map, coordinates).
  `coordinates` is an FNV digest (dr.rs:519-529, 567-581): NOT monotone, two
  equal assignments collide, and `FrameResealed` (dr.rs:305) says equal
  revisions are still a different frame. => unfit as a version key.
- `FrameStore` keeps only `current: Arc<FrameRevisions>`; `advance` drops it
  (dr.rs:335-337). `Frame` = Weak store + Weak seal + copy `at` (dr.rs:370-374);
  `check` = current-seal-only (dr.rs:346-363).
- HISTORY OVERWRITTEN HERE: `MapTable.tiles: BTreeMap<TileId, LocalMap>`
  (dr.rs:550); `advance` `insert` replaces / `remove` drops (dr.rs:593-601);
  `map` refuses any non-current frame via `frames.check` (dr.rs:620-623).
- `LocalMap::prepare` keeps supplied order, sorts only the lookup index
  (dr.rs:463-481); `extends` = same width + older globals a prefix (dr.rs:668-678).
  So a prefix of a map is a valid map: `local(g)` on the full index, accepted iff
  the returned position < prefix length. `lookups: Cell<u64>` (dr.rs:459).
- Consumer workaround: xlog `Cube.snapshots: Vec<LocalMap>` (cube.rs:194-198)
  exists only because "a MapTable drops a map when it advances"; built by
  `snapshot()` which COPIES every id + re-prepares (cube.rs:830-836), at
  cube.rs:232, 272-280, 752; consumed by `rebase` (cube.rs:745-767).
- Cube reads always use its own frame: cube.rs:216, 270, 410, 420, 717; xlog
  lib.rs:280, 293. Today any table advance makes every such read `Stale`, which
  is what forces `rebase`.
- codegraph `refresh_crate`: `advance(rev.source+1, …, [(last, Some(map))])`
  then `seal` then `cube.rebase` (graph.rs:438-446); build at graph.rs:308-310.
- ClassView/WideFieldMask in dt64: only on the binding side — `resolve_at`
  (lib.rs:795-806) = `routed_at(version)` then `resolve` (ClassView basis check);
  `Resolved::field_view(&WideFieldMask)` (lib.rs:460). dr.rs imports neither
  (dr.rs:26-27 import ClassId, Morton8x8 only).

## 2. Design — `MapTable` becomes a log, shaped exactly like `RouteLog`

**Version key: `RouteVersion`, reused, not invented.** It is already the type of
"a position in an append-only DR log, +1 per event". `FrameRevisions` stays on the
frame for REPORTING only (as its own doc says, dr.rs:377); it cannot key history
(digest, collisions, `FrameResealed`). Each `MapTable::advance` bumps the table's
`RouteVersion` by one, including an advance that changes nothing (keeps
dr.rs:584-586's "a new frame even when nothing changed": new version, same maps).

**Frame carries its version.** `Frame` gains a private `version: RouteVersion`;
`FrameStore` gains `version: RouteVersion`, bumped in `advance`, stamped in
`seal`. Still no public constructor (dr.rs:366-368 holds): a version is only ever
received from a store, never typed in. New `FrameStore::issued(&Frame) ->
Result<RouteVersion, FrameRefusal>`: `ForeignFrame` unless the store Weak matches,
else `Ok(frame.version)`. `FrameStore::check` (current-only) is UNCHANGED —
`Guards::check` (dr.rs:421) keeps its currency semantics.

**Append-only per-tile layout** (replaces dr.rs:550):
```
struct TileLog {
    gens:   Vec<LocalMap>,                          // one per generation
    events: Vec<(RouteVersion, Option<(u32, u16)>)>, // (gen, len); None = removed
}
tiles: BTreeMap<TileId, TileLog>,   version: RouteVersion (in FrameStore)
```
`advance(changes)`, per `(tile, Some(m))`:
- **extension** (`m.extends(gens[last])`, i.e. same gen still live): replace
  `gens[last]` with `m` (it is a superset; the old prefix is bit-identical inside
  it, so no information is lost) and push `(v, Some((last, m.len())))`. An
  extension is a LENGTH at a version.
- **replacement** (anything else): push `m` as a new gen, push `(v, Some((g, len)))`.
- **remove** `(tile, None)`: push `(v, None)`. Nothing is ever deleted.
- unchanged tiles get no event; their last event still answers.
Never update a past event, never compact (split-tunnel rule 3, ll.71-77).

**Read** — `map(&Frame, TileId) -> Result<TileMap<'_>, MapRead>`:
`issued(frame)?` → `tiles.get(tile)` → `events.partition_point(v <= frame.version)`
→ last event → `TileMap { map: &gens[g], len }`; `None`/no event → `NoSuchTile`.
`TileMap<'t>` is `Copy`, borrows, holds no ids: `local(g)` = `map.local(g)` then
`Unmapped` if `>= len`; `global(l)` = `None` if `l >= len`; `len()`, `width()`,
`digest()` over the prefix, `extends(other: TileMap)` (same gen ⇒ `len >=`;
else prefix slice compare). `locate` (dr.rs:630) reads through `TileMap` unchanged.

**What "stale" means afterwards.** A frame of THIS table always resolves — at its
own version, to the layout it was sealed under (replay). Refused: a frame of
another table (`ForeignFrame`, dr.rs:300), a tile absent at that version
(`NoSuchTile`). `MapRead::Stale` keeps the name but carries only `ForeignFrame`;
`SourceChanged/MapChanged/CoordinatesChanged/FrameResealed` stay for
`FrameStore::check`/`Guards`. `FutureVersion` is unreachable (store-issued
frames are ≤ current) — keep a `debug_assert`, not a refusal.

**Cost (to be MEASURED, ns, release):** `map` = 1 `Weak::ptr_eq` + BTreeMap get
+ `partition_point` over that tile's events (log₂ k) ⇒ expected tens of ns;
`TileMap::local` = existing ≤8-step binary search + one compare. Zero allocation
per lookup. `advance` = per change one `extends` (≤256 compares) + amortized
`Vec::push`, plus the existing digest. Memory: one ≤256×17 B LocalMap per
GENERATION (extensions do not add one) + 16 B per event.
(Extension carries the old gen's `lookups` count over; re-add after remove = new gen.)

## 3. ClassView + WideFieldMask — where they enter (and where not)

The TILE-MAP half does not need them, and they are not forced in: a tile map is
identity → position, no fields. They enter one layer up, already in code:
- version → classid: `BindStore::routed_at` / `resolve_at(view, version, pos,
  schema_rev)` (lib.rs:768, 795-806), ClassView basis check inside `resolve`
  (lib.rs:773-791; trait at contract class_view.rs:1057, `field_count` :1097).
- what a version CARRIES = the projection `Resolved::field_view(&WideFieldMask)`
  (lib.rs:460; `WideFieldMask` class_view.rs:243, `from_positions` :310,
  `full_for` :360). Same version key (RouteVersion) on both halves makes
  "the classid at v, read through its ClassView, masked by a WideFieldMask, over
  the tile layout at v" one coordinate — IF both logs share one counter (Q1).

## 4. Consumer changes

- **xlog `Cube`:** delete `snapshots` (cube.rs:194-198), `snapshot()`
  (cube.rs:830-836) and its three call sites (cube.rs:232, 272-280, 752).
  `rebase(table, frame)` becomes: for each tile, `old = table.map(&self.frame,
  t)?`, `new = table.map(frame, t)?`, `new.extends(old)` else `Renumbered(t)`;
  then adopt `frame` and grow `n`. Reads at `self.frame` (cube.rs:216, 270, 410,
  420, 717; lib.rs:280, 293) now SUCCEED after a table advance — rebase becomes
  "adopt a newer layout", not "unblock reads".
- **xlog signatures:** `&LocalMap` → `TileMap<'_>` in `rel` (cube.rs:838) and
  lib.rs:121, 139, 219, 244, 364, 373 (`ids`, `nonzero`). Mechanical.
- **codegraph:** graph.rs:308-310, 438-446 unchanged (`advance`/`seal`/`rebase` keep signatures).

**Tests whose semantics change (re-pin, never delete; old assertion as comment):**
- dt64 tests/dr.rs:172 `a_map_is_read_only_under_the_frame_it_was_sealed_under`
  (asserts `Stale(CoordinatesChanged)` at :180) → old frame reads OLD map.
- dt64 tests/dr.rs:188 `an_advance_that_changes_nothing_still_reseals`
  (`Stale(FrameResealed)` :197) → old frame still reads; frames are distinct
  issues (`same_issue` false) — "reseals" survives as identity, not refusal.
- UNCHANGED: dt64 tests/dr.rs:231 (`ForeignFrame` :244).
- xlog tests/cube.rs:236 `a_stale_frame_…_are_refused` (:257, :261) and
  tests/cube.rs:284 `an_added_relation_joins_like_a_base_one` (:312) — the
  `Stale` arms after `advance(1,1,vec![])` flip to success.
- xlog tests/wide_closure.rs:456 `rebase_accepts_prefix_extension_and_refuses_reorder`
  (:499 `Renumbered(20)`) — UNCHANGED (must stay green; it now proves the index).

## 5. Falsifiers (dt64 tests/dr.rs) + disable-run table

| # | test | disable (commit first, then break, then checkout) |
|---|---|---|
| F1 | replay: build v1 `[10,20,30]`, replace at v2 `[30,20,10]`; for EVERY id, `map(f1).local == v1 local` and `map(f2).local == v2 local`, and they differ for 10/30 | `map` ignores `frame.version` (reads last event) |
| F2 | extension: v1 `[10,20]`, v2 `[10,20,30]`; `map(f1).local(30) == Unmapped`, `.len()==2`, `.global(2)==None`; `map(f2).local(30)==2`; gens.len()==1 (anti-vacuity: the extension did not mint a gen) | drop the `>= len` guard in `TileMap::local` |
| F3 | replacement keeps both: after F1 plus a remove at v3, f1/f2 still read, f3 → `NoSuchTile` | `remove` deletes `tiles[t]` instead of pushing `None` |
| F4 | foreign: frame of table B on table A → `Stale(ForeignFrame)` even at equal revisions and equal version | `issued` skips the store `Weak::ptr_eq` |
| F5 | no allocation: counting `#[global_allocator]` in a dedicated test binary; 10⁴ `map`+`local` calls allocate 0 bytes | make `map` return an owned prefix copy |
| F6 | silence twin: an `advance` with no changes adds NO event and no gen, yet bumps the version (f_old and f_new read identical maps, `same_issue` false) | push an event per tile on every advance (gens/events counts move) |
| F7 | xlog: cube built at f1 still `closure`/`materialize_rows` correctly after an advance, with no rebase | re-add the current-only check in `map` |

Release bench: `map`+`local` at k = 1/64/4096 events per tile, in ns (ms = failed gate).

## 6. Steps

1. HAIKU `quote-harvester`: every `table.map(` / `&LocalMap` / `.snapshots`
   site in xlog + codegraph src AND tests, verbatim with lines (closed space:
   the three crates' `src/` + `tests/`). Orchestrator re-reads two.
2. SONNET-A (owns dt64 `src/dr.rs` + `tests/dr.rs` only): §2 verbatim, re-pin
   the two dt64 tests of §4, add F1–F6. No cargo.
3. SONNET-B (owns `lance-graph-xlog/src/*` + `tests/*` only), after A lands:
   delete snapshots, rewrite `rebase`, `&LocalMap`→`TileMap`, re-pin the two
   cube tests, add F7. No cargo.
4. ORCHESTRATOR: codegraph import touch-up (if any); then ONE gate:
   `CARGO_PROFILE_DEV_DEBUG=0 cargo test -p lance-graph-dt64 -p lance-graph-xlog
   -p lance-graph-codegraph` + clippy `-D warnings` + fmt; run every disable in
   §5 (commit → break → read `test result:` line → checkout); bench in release;
   board: STATUS_BOARD row + entry file + `entries_index.py --write`; amend
   D-DYNBAG §4.2/§4.3 to cite `TileLog`; commit.

## 7. Operator questions (≤2)

Q1. One counter or two? Should `MapTable` advances and `BindStore`'s `RouteLog`
    routes share ONE `RouteVersion` sequence (one DR log: a frame's version then
    also answers `routed_at`/`resolve_at`, §3), or keep separate counters of the
    same type (plan default: separate, because no current owner holds both)?
Q2. Read by number? Plan exposes old versions ONLY through store-issued frames
    (no `map_at(RouteVersion, tile)`), so a version cannot be forged. Is a
    numeric read wanted for replay-from-Lance (where only the number survives
    a restart), or does replay re-issue frames by re-running `advance`?
