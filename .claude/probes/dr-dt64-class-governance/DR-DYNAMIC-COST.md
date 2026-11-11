# D-DRDT-CLASS-GOVERNANCE-1 — DR64 vs DR256 and the cost of execution forms

Status: census, docs only. Read at main 238b6d1e4 (2026-11-11).

Question: what does the code say about choosing DR64 vs DR256, and about the cost of physical execution forms.

Paths are under `crates/lance-graph-dt64/src/` unless stated.

## 1. What the code says

### Widths and reservations

- `Width` is `W64` or `W256`: 64 or 256 positions per axis of a tile (`dr.rs:30-44`).
- `Axis { tile: Width, tiles: u32 }`. `reserved()` is `tile.positions() * tiles` (`dr.rs:46-61`).
- `Extent` is Identity x Property x Predicate, each an `Axis` (`dr.rs:91-102`). A reservation costs nothing. Only identity tiles that a route touches are claimed (`dr.rs:91-95`).
- Three named extents (`dr.rs:104-125`):
  - `DR64_CUBE`: 64 x 64 x 64, one tile per axis.
  - `DR256_SQ_X64`: identity 256, property 256, predicate 64.
  - `DR256_GLOBAL_X64`: identity 256 tiles of 256 (65,536 positions), property 256, predicate 64.
- `Extent::tile_bytes` is `identity.tile.positions() * property.reserved() * predicate.reserved() / 8`: a dense Boolean volume per claimed identity tile, "a statement of cost, not an allocation" (`dr.rs:132-137`). The doc gives 32 KiB for DR64 cubed.
- By that formula, computed here and not measured: `DR64_CUBE` is 64*64*64/8 = 32,768 B; `DR256_SQ_X64` is 256*256*64/8 = 524,288 B per claimed identity tile.
- `Claim { identity_tiles, bytes }` is what a class claimed at a version (`dr.rs:140-148`). `RouteLog::claimed` computes it (`dr.rs:319-335`).

### Choosing a width: `Axis::for_survivors`

- `for_survivors(n)` (`dr.rs:63-83`):
  - `n <= 64` gives one `W64` tile.
  - `65..=256` gives one `W256` tile.
  - more gives `ceil(n / 256)` `W256` tiles, saturating at `u32::MAX`.
  - zero survivors gives one `W64` tile.
- The doc states the rule: the width is a cost decision. "A DR64 register is 512 B and a DR256 one 8 KiB, so the narrow tile is used whenever the selection fits it" (`dr.rs:66-68`).
- The rule is by survivor count only. It does not consider operation shape, bound field count or measured time.
- grep for `for_survivors` in `crates/`: one caller outside its definition, `lance-graph-codegraph/src/graph.rs:565` (a doc mention at `625`). Other callers: not found by that search.

### Limits

- `MAX_RESERVED = 1 << 16` because route positions are `u16` (`dr.rs:150-151`). So one route log addresses at most 65,536 positions per axis.
- `RouteLog::route` takes `position: u16` and refuses `position >= identity.reserved()` (`dr.rs:255-278`).
- `BindStore` validation: identity and predicate reservations up to `MAX_RESERVED`, property up to `MAX_FIELDS = 256`. Anything wider is `Refusal::ReservationTooWide { class, axis, reserved }` (`lib.rs:269-270`, `1318-1330`; variant at `lib.rs:504-511`).
- `FieldsExceedExtent` and `MethodsExceedExtent` refuse a basis or method list wider than the reserved axis (`lib.rs:1331-1344`).
- A `LocalMap` for one tile holds at most `Width::positions()` ids (`MapRefusal::TooMany`, `dr.rs:555-566`; `LocalMap::prepare` at `586`).
- In `dt64-lance`, a 64-wide tile map is one 512-byte cast and a 256-wide tile is four (`lib.rs:28-31`, `77-82`).

### Cost statements already in the repo

- `dr-dt64-locality-execution-contract.md` (`.claude/knowledge/`, status CURRENT-CONTRACT) states the invariant: resolve identity and location once per valid binding, then execute against source-owned planes. Its measured items are on one 4-vCPU Xeon with AMX-BF16, medians of 41 warm runs, run-to-run variation up to about 35% (`dr-dt64-locality-execution-contract.md:14-19`, `116-130`).
- That document's resolve and G2L figures are microseconds. It records G2L translation: warm run 20-24 microseconds vs 157-186 microseconds re-translating per operation (line `125`).

## 2. Measured numbers supplied for this census

Source: supplied in the task brief. One machine, 4 cores. Criterion medians, 2026-11-11. Branch `ccr-ca0562e7-hudvlz-bench-cypher-quack`, PR 2010, NOT on main. The board entry `.claude/board/entries/2026-11-11-cypher-df-vs-quack-bench.md` is not present in this tree (checked by glob), so these figures were NOT re-read or re-run here.

At 1,000,000 rows.

| question | DataFusion | Quack execute | DR256 prepared planes | order-index range | unit of the figure |
|---|---|---|---|---|---|
| filter count | 2.17 ms | 268 us | 188 us (ternlog fold over prepared planes) | 1.76 us (key range over an order index) | ms / us |
| one-hop count | 26.8 ms | 3.45 ms | 4.86 us (prepared planes) | not measured | ms / us |
| hop with a filter on the source | 28.6 ms | refuses | not measured | 2.17 us (order-index range) | ms / us |

Preparation cost at 1,000,000 rows.

| prepared form | time |
|---|---|
| node planes | 3.60 ms |
| edge planes (fk read once) | 4.55 ms |
| order index | 41.3 ms |

Lance intake at 1,000,000 rows.

| intake | time |
|---|---|
| all columns | 29.0 ms |
| masked columns | 9.01 ms |
| masked columns plus a row allow-list of 3 in 4 rows | 68.8 ms |

Rows in milliseconds (marked under the workspace rule, `CLAUDE.md` "no millisecond results": a millisecond result on a per-query path is a failed gate):

- Filter count: DataFusion 2.17 ms.
- One-hop count: DataFusion 26.8 ms; Quack 3.45 ms.
- Hop with a filter on the source: DataFusion 28.6 ms.
- Preparation: node planes 3.60 ms; edge planes 4.55 ms; order index 41.3 ms. These are one-time build costs, not per-query paths. Whether the rule exempts them: UNKNOWN. The rule's carve-out in `CLAUDE.md` covers cold compile and oracle costs, reported in microseconds and labelled "not gates".
- Lance intake: 29.0 ms, 9.01 ms, 68.8 ms. A per-query reading of these is also a millisecond figure.

## 3. What these numbers do NOT show

- No `VirtualCsr` arm. PR 1984's `vcsr.rs` is not on main. It was merged into a side branch. A glob for `crates/**/vcsr.rs` in this tree finds nothing.
- No sparse reference.
- No prepared degree lane.
- No destination filter and no both-endpoint filter.
- They show one machine, one row count (1,000,000) and one benchmark branch.
- Quack's "refuses" in the third row is not a time. The DR256 and order-index arms were not run on every question, so the table has gaps ("not measured").
- They do not choose between DR64 and DR256. At 1,000,000 rows the figures are for the "DR256 prepared planes" arm, whose tile width in that bench is not stated in the brief: UNKNOWN.

## 4. The comparison still owed

Rows: OrderIndex, VirtualCsr, sparse reference, prepared degree lane. Columns: filter on the source, filter on the destination, filter on both endpoints. Twelve cells. None is filled by the numbers above except OrderIndex x source filter (2.17 us, one question shape).

| arm | source filter | destination filter | both-endpoint filter |
|---|---|---|---|
| OrderIndex | 2.17 us (supplied) | owed | owed |
| VirtualCsr | owed | owed | owed |
| sparse reference | owed | owed | owed |
| prepared degree lane | owed | owed | owed |

## 5. Not checked

- The 188 us, 4.86 us and 2.17 us figures were not reproduced.
- Which tile width the "DR256" bench arms used.
- Whether `Axis::for_survivors` has callers other than the one found.
- Whether any production path picks DR64 vs DR256 by measured time. The code picks by survivor count only (`dr.rs:63-83`).
