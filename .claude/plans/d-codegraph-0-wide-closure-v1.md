# Wide call-chain closure: multi-word rows, block presence, merge on read

D-CODEGRAPH-0 slice C follow-up. Operator decision 2026-11-02: "Wider call
chain. Presence bit for thin provisioning and sparse write merge on read."
Planning only; nothing edited, nothing compiled.

## 1. Findings from the source

- F1 `xlog/src/cube.rs:417-419` refuses `self.n > 64` with
  `ClosureWidth(n)`. The kernel holds one `u64` per row (`edge`, `full`,
  `delta`, `next`, :420-426), and `edge[a] |= 1 << b` (:422) is the only
  reason for the limit. The semi-naive loop itself (:429-446) is already
  width-agnostic except for the per-row word.
- F2 `Closure { rows: Vec<u64> }` (:118-121); `contains(x: u8, y: u8)` hard-
  codes `y < 64` (:133). Both are part of the 64-only shape.
- F3 `ClosureStats::rows_composed == |edge⁺|` (:142-145) is a measurable
  invariant of semi-naive evaluation. It is a free falsifier for the wide
  kernel.
- F4 The `Cube` is single-tile: one `tile: TileId` (:172), built from
  `table.map(frame, tile)` (:188). `Rel.cols` are already `Vec<u32>` (:151),
  so a lane can hold a wide position without a format change.
- F5 `LocalMap::prepare` refuses `> width.positions()` (dt64 `dr.rs:443`)
  and stores `u8` locals (:446-447). `MapTable` already keeps many tiles in
  a `BTreeMap<TileId, LocalMap>` (:528). Multi-tile coordinates exist; only
  the Cube cannot span them.
- F6 `MapTable::advance` makes every earlier frame stale "even when nothing
  changed" (:562-564). So appending a tile for new symbols stales the cube,
  and an incremental refresh needs a defined way to move to the new frame.
- F7 Thin provisioning is already doctrine in dt64: "a reservation costs
  nothing; only the identity tiles a route actually touches are claimed …
  nothing is ever allocated here" (`dr.rs:70-72`). The contract's presence
  plane says the same at lane level: "A sparse write leaves an absent lane
  unwritten, so it keeps the value read before" (`lane_codec.rs:65-66`).
  HHTL: a child mask is "the parent's OR its own delta" (`hhtl.rs:34-37`).
- F8 Split tunnel (`split-tunnel-sparse-append-resident-fold.md`): rule 1
  sparse records `(version, owner, op, …)` with `op ∈ {BORN, SET, DELETE}`
  (:51-55); rule 3 never update in place, never compact the PERSISTED log
  (:73-79); rule 5 committed records stay resident (:85-88); rule 7 derived
  reads write nothing and are recomputed by folding (:95-99). The capstone's
  thought layers are "sparse write, merge on read", higher layer wins a
  shared lane (capstone :32-33, :49-50).
- F9 Codegraph refuses `n > 256` as `TooMany` (`graph.rs:162-166`),
  `Pos = u8` (:14), and builds a dense identity map `ids = 0..n` (:222).
  Symbol ids are first-seen intern order, so global id → tile is
  `g / 256` by construction.
- F10 BUG today, hidden by F1: `impact`'s `row` closure copies ONLY the
  first word of a closure row into a `w`-word mask (`query.rs:241-247`).
  With a 64-only closure that is correct; with any wider closure it would
  silently drop every position ≥ 64. It must change in the same PR.
- F11 Existing pins that move: `codegraph/tests/guards.rs:50`
  (`ClosureWidth(70)`), `:64` (`TooMany(300)`), and `xlog/tests/
  staunen5.rs:174` (`ClosureWidth(65)`). They are re-pinned, not deleted.
- F12 Measured store (coordinator, 2026-11-02; not re-derived):
  `scratchpad/mrstore`, ruff_charon rows v2 from `repro/mask_risc.llbc`,
  receipt 17154 conserved. DEFINES 459, CALLS_RESOLVED 15684, IMPLEMENTS
  278, REFERENCES 484, PUBLIC 185, SYNTHETIC 58, MAY_DISPATCH 2,
  UNRESOLVED_CALL 4. Interned symbols 694 → 3 tiles of 256. Callees by
  owner: mask_risc 194, core 132, ndarray 68, alloc 16.

## 2. Design

**Interpretation, checked against the source.** (1) Wider chain = closure
over N > 64 positions AND across tiles: confirmed necessary, F1 + F4 + F9.
(2) Presence bits = per closure row, one bit per 256-position BLOCK (one
dt64 tile); an absent block reads as zero and is never allocated: matches
F7. (3) Sparse write, merge on read = refresh appends relation deltas and a
read folds base ∪ deltas: matches F8, with one correction below — the
CLOSURE is a derived read (rule 7), so it is never persisted as a delta; it
is re-derived for the affected rows only.

**Challenge, stated plainly.** At mask-risc's 694 positions a dense row is
11 words (88 B; 694 rows ≈ 61 KB). Block presence buys almost nothing there.
It pays at workspace scale (tens of thousands of symbols, most callers
reaching few tiles). So the plan makes dense the all-present special case of
the block layout, not a separate path, and measures both on mask-risc.

### 2a. Data layout — lands in `lance-graph-xlog` (new `src/wide.rs`)

xlog owns relations and their closure; nothing here is a graph store.

```
WideRows  (one per closure / per edge matrix)
  n:        u32                 positions (wide coordinate = tile_ord*256 + local)
  blocks:   u32                 ceil(n / 256)
  presence: Vec<u64>            per row: ceil(blocks/64) words; bit b = block b allocated
  offset:   Vec<u32>            per row: start of its packed blocks in `words`
  words:    Vec<u64>            present blocks only, 4 words each, block order
```

- A block is 4 words = 256 bits = one tile, so presence aligns with dt64's
  "claimed tile" (F7). Reading bit y of row x: test presence bit y/256, then
  popcount(presence below it) → block slot; no search.
- Writes during the fixpoint use a per-row scratch of `blocks*4` words
  (dense, reused across rows), then pack: a block is allocated iff any word
  is nonzero. Counter `ClosureStats::blocks_allocated` (and
  `blocks_reserved = n_rows * blocks`) is what the presence falsifier reads.
- **64-position fast path is kept verbatim.** `Cube::closure` keeps the
  existing code for `n <= 64` and returns the narrow `Closure` unchanged
  (rows(), contains). The wide kernel is used for `n > 64` and can be forced
  on narrow input in tests to prove bit identity.

**Algorithm:** the same semi-naive loop, multi-word: `delta' = (delta ∘ edge)
andnot full; full |= delta'`, per row, with `mask_or_assign` /
`mask_andnot` / `mask_any` from `ndarray::simd` over the row's 4-word blocks.
Composition ORs edge row y into the scratch only over y's PRESENT blocks.
Cost: `rows_composed == |edge⁺|` still holds (each pair enters a delta once,
F3); each composition costs 4·popcount(presence(edge[y])) words, so total
≈ `4 · Σ_{(x,y)∈edge⁺} |blocks(edge[y])|` word ORs plus one andnot pass over
present blocks per row per round. Worst case dense is `O(N³/64)`; the round
bound stays `rounds <= n + 1` (`FixpointBound`).

### 2b. Cross-tile — block matrix, one fixpoint (not per-tile + frontier)

A position is `(tile_ord, local)` folded into one wide coordinate
`tile_ord * 256 + local` (u32, fits the existing `Rel.cols`, F4). The cube
holds an ordered `tiles: Vec<TileId>`; `tile_ord` is the index in that list.

Chosen: ONE semi-naive fixpoint over the block matrix. Rejected: per-tile
closure plus a boundary frontier — cross-tile cycles force iterating between
tiles until a global fixpoint anyway, so it is the same computation with a
second loop and a second termination argument. The one case where per-tile
wins (tiles nearly disconnected) is exactly what block presence already makes
free: an unreached tile is an absent block and costs nothing.

Lookup at build (cold, once): new dt64 `MapTable::locate(frame, tiles, g)`
tries each listed tile's `LocalMap::local` and refuses an id found in two
tiles (`MapRefusal::Duplicate`) or none. Cost O(T·log 256) per id, build
only. Codegraph chunks its dense ids `0..n` into tiles of 256 (F9), so its
tile is `g/256`; `locate` stays general for callers whose ids are not dense.

**Ordering note (cheap, measurable):** intern crate-local symbols first and
external owners (core/ndarray/alloc, F12) after, grouped by owner, so tiles
align with owners and a crate-local reach touches fewer blocks. Measured, not
assumed: report blocks_allocated for both intern orders on mask-risc.

### 2c. Sparse write, merge on read

- **Delta = relation edges, keyed `(rel, src_pos, dst_pos)`, op ∈ {ADD,
  RETIRE}, stamped `version`.** One crate re-import = RETIRE every edge whose
  provenance row belongs to that crate's old rows + ADD the new rows' edges.
  Symbols: new symbols get NEW positions appended to the last tile or a newly
  claimed tile; a vanished symbol keeps its position and is masked out by a
  `retired` symbol mask. Positions are never renumbered (base ordinal, never
  claim order; capstone §2), which is what keeps old lanes valid.
- **Where:** resident in xlog, `Cube`'s `DeltaLog` (append-only per-relation
  u32 lanes `src, dst, version, op`), next to the base lanes. Persistence of
  crate rows stays ruff_charon's store (it already writes per-crate
  `*.rows.tsv`); this plan adds no Lance write and no persisted closure.
- **Read fold for an edge row:** `edge'(x) = base(x) ∪ adds(x) andnot
  retires(x)`, deltas applied in append (= version) order, newest wins per
  key, no sort (split-tunnel rule 3).
- **Closure under insertions only** is exactly incremental: seed
  `delta(x) = (new targets of a)` for every x with `a ∈ R⁺(x) ∪ {x}` and run
  the same fixpoint; monotone, exact.
- **Closure under deletions cannot be merged incrementally** (a removed edge
  may or may not have been the only path). So: affected rows A = { x : x = a
  or bit a set in R⁺(x) } for every source `a` of a RETIRED edge — the union
  of column a over the closure, read as rows of the REVERSE closure when it
  is held (codegraph holds RCONF/RPOSS and their forwards are cheap) else a
  word test per row. Rows outside A keep their row bit-for-bit. Rows in A are
  recomputed by the same fixpoint over A, using `R⁺(y)` directly for any
  `y ∉ A` (their closure is unchanged), so cost scales with |A|, not N.
- **Compaction/checkpoint:** the RESIDENT base may be re-folded (base ∪
  deltas → new base, delta log emptied) at any time: it is a derived read
  (rule 7), nothing persistent changes. The persisted rows are never
  compacted (rule 3). Trigger: delta log longer than base edges / 8 — a
  policy pin, measured, not a law.
- **Frame:** `MapTable::advance` stales the cube (F6). New dt64
  `LocalMap::extends(&self, older: &LocalMap) -> bool` (older's `globals` is
  a prefix) lets `Cube::rebase(table, new_frame)` accept a new frame iff every
  tile it holds is unchanged or prefix-extended, and refuse otherwise. That
  is the dt64-legal statement of "positions never renumber".

Ownership: dt64 = coordinates/frames; xlog = relations, closure, deltas; codegraph = intake, symbols, crate refresh.

## 3. API changes

- dt64: `MapTable::locate(&self, frame: &Frame, tiles: &[TileId], global: u64) -> Result<(u32, u8), MapRefusal>` (tile ordinal, local); `LocalMap::extends(&self, older: &LocalMap) -> bool`. Additive.
- xlog: `Cube::new_tiled(table, frame, tiles: &[TileId], relations) -> Result<Cube, CubeRefusal>` (`Cube::new` = `new_tiled(.., &[tile], ..)`, unchanged); `Cube::closure_wide(&self, table, rel) -> Result<(WideClosure, ClosureStats), CubeRefusal>`; `WideClosure::{contains(x: u32, y: u32) -> bool, row_into(x: u32, out: &mut [u64]), presence(x) -> &[u64]}`; `Cube::apply_delta(&mut self, rel, adds: &[(u32,u32)], retires: &[(u32,u32)], version: u64)`; `Cube::refresh_closure(&self, table, rel, prev: &WideClosure) -> Result<(WideClosure, ClosureStats), CubeRefusal>`; `Cube::rebase(&mut self, table, frame) -> Result<(), CubeRefusal>`. `ClosureStats` gains `blocks_allocated`, `blocks_reserved`, `rows_recomputed` (additive fields; struct is `Default`).
- Kept: `Cube::closure` and `Closure` for `n <= 64`, bit-for-bit. `ClosureWidth` now refused only by `closure` (narrow), never by `closure_wide`; new `CubeRefusal::TileCount(usize)` above 64 tiles (presence word cap, 16,384 positions) — a policy pin.
- codegraph (BREAKING, named): `Pos = u32`; `IntakeRefusal::TooMany` only above the tile cap; `closure()` returns the wide form; `impact` reads whole rows via `row_into` (fixes F10); new `Codegraph::refresh_crate(&mut self, dir: &Path, krate: &str)`.

## 4. Steps

1. HAIKU `census-counter`: every caller of `Closure::{rows,contains}`, `Pos`, `ClosureWidth`, `TooMany` across `crates/` (with command). HAIKU `quote-harvester`: ruff_charon `rows` invocation + crate-name arg from `ruff/crates/ruff_charon/src/main.rs:1-90`.
2. SONNET W1, file `lance-graph-dt64/src/dr.rs` only: `locate` + `extends` + unit tests (two-tile duplicate refused; absent refused; prefix vs reorder for `extends`).
3. SONNET W2, file `lance-graph-xlog/src/wide.rs` (new) only: `WideRows`, pack/scratch, the multi-word semi-naive kernel as a free fn over `(n, &[u32] src, &[u32] dst)`, stats counters, insertion-only and affected-rows recompute. Spec = §2a/§2c verbatim. No cube.rs edits.
4. SONNET W3 (after W2 lands), `lance-graph-xlog/src/cube.rs` + `lib.rs`: `new_tiled`, `closure_wide`, `DeltaLog`, `apply_delta`, `refresh_closure`, `rebase`; narrow path untouched.
5. SONNET W4 (after W3), `lance-graph-codegraph/src/{graph,query}.rs`: tile chunking, `Pos` widen, impact via `row_into`, `refresh_crate`, intern order option.
6. SONNET W5, test files only: `xlog/tests/wide_closure.rs`, `codegraph/tests/wide.rs` (synthetic fixtures generated in-test, no /tmp).
7. ORCHESTRATOR: `CARGO_PROFILE_DEV_DEBUG=0` compile/test per crate once, re-pin F11 tests, disable runs (§5), mask-risc measurement (§6), board entry + commit. Workers never run cargo or git.

## 5. Falsifiers and disable runs

| falsifier | disable that must turn it red |
|---|---|
| narrow vs wide bit-identical on random graphs n ≤ 64 (wide forced) | wide kernel drops the andnot (`full` not subtracted) |
| 694-position synthetic graph (3 tiles, cross-tile cycles): `closure_wide` == per-node BFS over an adjacency list built IN THE TEST | pack only block 0 of each row |
| `rows_composed == |edge⁺|` from the BFS reference | count every OR including absent blocks |
| absent blocks allocate nothing: two disconnected tiles → `blocks_allocated` == present count, cross blocks 0 (anti-vacuity: the connected twin allocates them) | allocate every block |
| merge-on-read == full rebuild after EVERY step of a refresh sequence (add-only, retire-only, mixed, symbol appended in new tile) | skip the affected-rows recompute on retire |
| rows outside A unchanged and `rows_recomputed == |A|` (can-stay-silent: an add-only step recomputes 0 rows) | treat every row as affected |
| `rebase` refuses a reordered tile, accepts a prefix extension | `extends` returns true |
| impact over > 64 positions sees a caller at position ≥ 256 | restore the first-word-only `row` (F10) |

A disable is evidence only if its test binary built and printed `test result` red.

## 6. Timing gates and the mask-risc measurement

Store exists: `scratchpad/mrstore` (F12; 694 symbols, 3 tiles). Re-producible by `ruff_charon rows repro/mask_risc.llbc <crate>` per §4 step 1 quote; the store and LLBC are session scratch, so committed tests use synthetic fixtures only. Extend `codegraph/examples/timing.rs` with `CODEGRAPH_DIR`.

- Gated, ns: `WideClosure::contains`, `impact` per call (3 masks, 11 words/row) — gate ≤ 1 µs per call, release.
- Gated, µs: `closure_wide` for RCONF and RPOSS on mask-risc, and `refresh_crate` for one crate — must be µs; any ms is a failed gate (P0). Refresh must beat full rebuild.
- Reported, not gated: open (cold intake), `blocks_allocated / blocks_reserved` for both intern orders, `rounds`, `|edge⁺|`.

## 7. Open questions for the operator

1. Tile cap for the presence word: 64 tiles (16,384 positions) as a pin, or a multi-word presence from the start?
2. Should a refreshed resident base (re-fold) be allowed to run implicitly by the delta-length trigger, or only on an explicit call?
