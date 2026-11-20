# BindStore addressing: what Raft / TiKV already solved, and how DR does it

> READ BY: any session that keys a cache, a kept result or a witness on a
> binding; any session that proposes a "persistent register", a KV store, TiKV
> or Raft beside DR; any session building Quack or the PostgreSQL-shaped
> polyfill over the cube; every reviewer of such a PR.
>
> Status: the DR / DT / cube column is VERIFIED-IN-CODE at the cited files on
> `main` (2026-11-20) unless marked IN-PR. The Raft column is the published
> protocol (Ongaro and Ousterhout, "In Search of an Understandable Consensus
> Algorithm"). The TiKV column is TiKV's documented design (Multi-Raft regions,
> PD routing, region epoch, Percolator MVCC); it is not checked against a
> pinned TiKV commit. The parallels are this file's READING, marked as such.
> Nothing here decides the open writes / isolation question; that is the
> operator's.
>
> Born 2026-11-20 from the operator's remark "raft might even use similar
> techniques like DR", lance-graph #2146 to #2150, zammad-rs#95 and the
> EXO-367 / EXO-368 / EXO-369 thread. Companion, storage side:
> `docs/APPEND_ONLY_RAFT_DOVETAIL.md` (Lance fragments as Raft log payloads).
> This file is the addressing side.

## 0. Thirty-second version

- **Schemas are DT's**, enforced through classes (`ClassView`, `BindStore`
  registration, `SchemaRev`, the basis witness). **The mapping is DR's**
  (`MapTable`, `LocalMap`, `Frame`, `RouteVersion`). **Relations are the
  cube's**, as ordinal lanes (`xlog::cube`). A `BindStore` binding is their
  product: `cube relation × DT[ClassView] × DR[WideFieldMask]`.
- **Ordinal lanes are versioned, replayable and immutable.** A binding is kept
  under the version it was made at and replayed by reference (S7, IN-PR #2150).
- **Raft already solved the witness problem** that XVERWEIS hit: an entry is
  named by `(term, index)`, and equal handles prove equal prefixes. A witness
  must be a version handle of the address space, never a summary of contents.
- **Coordinates are bijective; relations are bags.** Quack and the
  PostgreSQL-shaped polyfill need the bijection in both directions: inbound
  `local(global)`, outbound `global(local)`.

## 1. Who owns what

| Concern | Owner | Carrier | Where |
|---|---|---|---|
| Schema: which fields a class has, at which revision, with which predicates | DT | `ClassView`, `BindStore::register`, `SchemaRev`, basis witness | `lance-graph-dt64/src/lib.rs` |
| Mapping: identity ↔ position, per tile, per version | DR | `MapTable`, `LocalMap`, `Frame`, `RouteVersion` | `lance-graph-dt64/src/dr.rs` |
| Relations: `x[i] : y[i]`, a bag, with a live plane | cube | `Cube`, `CubeLanes` | `lance-graph-xlog/src/cube.rs`, `cube/lend.rs` |
| The binding: class × field mask × lanes × frame × row order × scope | DT over DR | `LaneBinding` | `lance-graph-dt64/src/lane.rs` |
| The kept binding, reusable and replayable | DR log, opaque to DR | `MapTable::keep / kept / kept_under`, `LaneBinding::keep_in` | IN-PR #2150 |

"DR routes, DT interprets" is enforced in code: `dr.rs` may not name the
binding side (`tests/dr.rs`, `dr_does_not_depend_on_the_binding_side`). The
S7 log therefore stores mappings opaquely; DT gives them meaning.

## 2. The comparison

| Raft / TiKV | DR / DT / cube | Grade of the parallel |
|---|---|---|
| A log entry is named by `(term, index)`: a generation plus an ordinal | `CubeLanes::space()` = (cube, relation, base generation, log length) (`cube/lend.rs:60`); DR's `Frame` carries its `RouteVersion` (`dr.rs`, `Frame::version`) | READING, both sides verified |
| Log matching: two logs holding an entry with the same `(term, index)` agree on every entry before it | A binding is admitted only under its own frame and coordinate space (`LaneBinding::check`); equal handles, equal assignment. A kept mapping is handed out only under its own version (`KeepRefusal::NotWitnessed`, IN-PR #2150) | READING |
| Snapshot / log compaction: state up to `lastIncludedIndex / Term`, then the log tail | `Cube::refold`: base plus delta log become the new base, the generation is bumped (`cube.rs`, `refold`); the cached fold is taken, not recomputed (#2146) | READING |
| Followers apply committed entries; reads at the committed index (read-index / lease read) | Lanes admit the current frame only (`FrameStore::check`); history is read as the mapping of a version (`FrameStore::issued`, `MapTable::map(frame, tile)`) | READING |
| TiKV Multi-Raft: key ranges ("regions"), each its own Raft group; PD routes a key to its region | `MapTable::locate`: global id → (tile, local); one `LocalMap` per tile | READING |
| Region epoch `(conf_ver, version)` bumped on split or merge; a request with an old epoch is refused (`EpochNotMatch`) | `FrameRefusal::{SourceChanged, MapChanged, CoordinatesChanged, FrameResealed}`; a stale binding is refused before any lane is read | READING |
| Region split | Tile growth / `Cube::rebase`: allowed only as a pure extension, refused if positions would renumber (`CubeRefusal::Renumbered`) | READING |
| MVCC: every write versioned by a timestamp; a snapshot read at a timestamp | The mapping is versioned, the contents are not (plan `dr-widefieldmask-dt-classview-v1.md` §1, EXO-255) | partial: see §4 |

## 3. What carries over: the witness

A cached answer must be tied to the address space it was read under. Raft's
answer is `(term, index)`. DR's is the frame plus the coordinate space, and for
a kept mapping its `KeptId` under its version.

Counterexample to doing otherwise, measured 2026-11-20 on report-rs `main`
`e56b971` (EXO-369): XVERWEIS's `witness()` is
`(x digest, y digest, support, codebook generation, codebook digest)`. The two
relations `[(a,K1),(a,K2),(b,K2)]` and `[(a,K1),(b,K1),(b,K2)]` get the same
witness and different answers (`Ambiguous` against `Bound`): the axis digests
cover first-seen order and `support` is only a count. A content summary is not
a version handle.

## 4. Where the parallel breaks

- **Truncation.** Raft's index → entry map is a bijection only over the
  committed prefix: a new leader may truncate a follower's uncommitted tail and
  reuse those indexes. DR's identity → position map is append-only and never
  truncates (`LocalMap::prepare` refuses `Duplicate`; `rebase` refuses
  renumbering). If DR positions are ever replicated through Raft, only
  committed positions may be lent.
- **What is indexed.** Raft indexes operations (writes); DR indexes identities
  (placement). A Raft index names a change; a DR position names a thing.
- **History.** TiKV garbage-collects MVCC versions past its GC safe point. DR
  never compacts versions (EXO-256): a mapping kept at `v` stays the mapping
  of `v`. Contents are not versioned by DR at all; they stay with their owner.
- **Cost.** A TiKV read is a network round trip. The register read is in
  process: `borrow_all` lends lanes in place, one check, no allocation (#2147).

## 5. The bijection Quack and the PostgreSQL polyfill need

| Direction | Call | Cost, measured | Where |
|---|---|---|---|
| Outbound, `identity[index].key` | `global(local)`: a pointer-table read | 0 counted lookups | `dr.rs` `LocalMap::global`; IN-PR #2149 |
| Inbound, `identity[key]` | `local(global)`: a binary search over at most 256 ids | 1 counted lookup per cell (334 for 334) | `dr.rs` `LocalMap::local`; IN-PR #2149 |
| Value, `identity[index].value` | the lane at the position, in place | one check per projection, not per field (3 lookups against 9) | #2147 |

Rules the gates pin:

- Coordinates round-trip both ways: `global(local(g)) == g`,
  `local(global(l)) == l` (IN-PR #2149, against PostgreSQL 16.15).
- Relations stay bags, never bijections: PostgreSQL's join semantics are bag
  semantics (report-rs#175 `a_repeated_binding_is_a_bag_not_a_set`; the
  "relations read as sets" disable in #2149).
- A key with no live row has no position; a dead position is refused
  (zammad-rs#95 `f1e12c8`, `Refusal::DeadPosition`).

## 6. If TiKV is ever used (WORKING-MODEL, not decided)

TiKV fits as the durable source the reference contract (identity → place in
the append-only order) is reloaded from, at a snapshot timestamp. The
in-process projection (`MapTable`, the lanes) stays where every read goes;
`borrow_all` never touches TiKV. Its transactions would be one answer to the
writes and isolation that are scoped out today; that choice is the
operator's. No TiKV client crate has been checked.

## 7. Evidence

- lance-graph: #2146 (refold takes the cached fold), #2147
  (`borrow_all_selected(_present)`), #2148 (VROW lab gate), #2149 (cube
  against PostgreSQL, inbound and outbound), #2150 (S7 kept mappings).
- zammad-rs#95: the VROW gate on the real store; dead positions refused.
- report-rs#175 (P66 pins); EXO-369 (the witness counterexample).
- Linear: EXO-332 (the three constructors), EXO-329 (DR owns positions),
  EXO-255, EXO-256, EXO-367, EXO-368, EXO-369.
