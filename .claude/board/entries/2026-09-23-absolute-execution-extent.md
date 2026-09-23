# 2026-09-23 — Absolute execution extent for mask-risc

**Status:** MEASURED (extent seam shipped in PR) · OPEN (scheduler, OQ-5, the `_sym` / group merge laws)
**D-ids:** D-WFL-EXTENT (new, In PR). Corrects the wording of
`entries/2026-09-23-cubecl-llvm-boundary-and-audit-regrade.md` lesson 1 (below).
OQ-5 is untouched and still open.

## API shape
`execute_extent(program, planes, foreign, scratch, out, lo..hi)` in
`crates/lance-graph-mask-risc/src/exec.rs`. `execute_into` is now that call with
`0..n_rows`, so existing callers are unchanged and the whole extent accepts every terminal.
`extent_tiles` (the tile plan the executor iterates) is **crate-private**: tile width and
edge representation are executor detail, not API. The structural claim is pinned by the
in-crate test `exec::extent_tile_tests`; the benchmark reports touched words from the
semantic span (`touched_words`), which that test proves the plan covers exactly.

## The absolute-coordinate law (CURRENT-CONTRACT, TEST-PINNED)
The extent is an OUTER restriction in the same row coordinates as `Planes`. A program
`Range [1000, 2000)` executed over the extent `[1500, 1700)` means
`[1000, 2000) ∩ [1500, 1700)`. Lane element `r` is row `r` whatever the extent, and tile word
`w` is word `w` of every resident mask. Nothing is rebased, copied or renumbered.

**Implementation:**
- **Word bounds:** only `span_words(lo, hi)` is walked, the same law `touched_words`
  now delegates to.
- **Tile start/stop:** a word the extent cuts (an unaligned `lo`, or an unaligned `hi` short
  of `n_rows`) is its own one-word tile.
- **Edge masks:** on that tile the terminal's mask is ANDed with the in-extent bits in a
  one-word register temporary. For `All`, the out-of-extent bits are ORed in instead, since
  set bits are its identity.
- **No lane offsets:** lanes are read at `w * 64` exactly as before.
- **No copies:** no lane or mask copy was required.

The #1268 fused fold composes by intersection
(`program_range ∩ extent ∩ resident_plane`) inside the same seam. There is no second
evaluator.

## What execution now skips / what still materializes
- **Skipped:** every tile outside the extent. A 1-row extent over 1M rows visits one word.
- **Still materializes, deliberately:**
  - `Keep` writes its population-addressed `Out::Mask`, but only the in-extent bits: edge
    words are merged bit-exactly, and every other bit stays as the caller holds it. Disjoint
    extents therefore compose into one absolute buffer in any **sequential** order.
  - **Not a concurrency licence (CURRENT-CONTRACT).** An unaligned split such as
    `[0, 65)` + `[65, N)` puts both extents in the same physical `u64`, and the edge merge
    is a read-modify-write. Partial `Keep` sinks compose in any sequential order; concurrent
    execution requires word-disjoint sink ownership (boundaries on multiples of 64),
    separate partial sinks plus a merge, or another explicitly synchronized strategy. A
    future scheduler must not infer two simultaneous writers on one `Out::Mask` from this.
  - The tiled (non-fused) path still writes tile-local scratch, but only for touched tiles.
- **Refused on a partial extent**, as `ExtentUnsupported`, because there is no shipped merge
  law or disjoint sink here: `BlendI32`, `ScatterOrU32`, `ScatterCountU32`,
  `CountKeyRunsU32`, `GroupSumI32`, `GroupSumViaI32`, `GroupReduce`. `_sym` stays a recorded
  law only (`TD-SYM-SUM-MERGE-IS-NOT-ADDITION-1`).
- **Foreign planes and lanes** (`Gather`, `GroupKey::Via`) are addressed by key and are never
  sliced. That is pinned by a test whose extent rows all name foreign keys below the extent's
  own row numbers.

## Split/merge (TEST-PINNED, `tests/extent.rs`)
- **Terminals:** Count (+), Any (∨), All (∧), MaskedSumI32 (+), MaskedMin/Max (min/max),
  each over the fused and tiled shapes.
- **Partitions:**
  - two-way at k ∈ {0, 1, 63, 64, 65, 127, 128, 129, N/2, N−1, N};
  - 40 random three-way partitions;
  - N ∈ {1317, 4133}, neither a multiple of 64.
- **Orders:** every partition is merged in forward, reverse and rotated order, and each equals
  whole-population execution. `Keep` partials written in forward and reverse order into one
  buffer equal the whole `Keep`.

## Non-rebasing falsifier
Distinct lane values sit at rows 63, 64, 65, 127, 128 and 129.
- `MaskedSumI32` over extents starting at 65, 129, 63 and 127 must equal the absolute-row sum.
- The test also asserts that the worker-local sum differs, so a rebasing executor cannot pass.
- A second test pins program `Range [1000,2000)` over extent `[1500,1700)` as exactly 200
  rows, and `Keep` writes exactly bits 1500..1700.

## Disable runs (committed first, then each restored)
| disable | tests red |
|---|---|
| tiles walk from row 0 instead of the extent | 7 of 8, incl. the structural tile gate (since moved in-crate as `exec::extent_tile_tests`) |
| edge word not restricted | 6 |
| lanes read worker-local (`r0` relative to the extent) | 5, incl. the absolute-rows falsifier |
| fused fold ignores the extent | 4 |
| `Keep` overwrites the whole edge word | 3 |

## Benchmarks
Run with `cargo run --release -p lance-graph-mask-risc --example extent_probe`: N = 1,048,576
rows, a 2/3-dense plane and unaligned extents. Every result is checked against a scalar
oracle over the same absolute rows.

| shape | extent | median ns | population words read | lane elements | derived words written |
|---|---|---|---|---|---|
| fused Range∩plane→Count | 1 row | 62 | 1 | 0 | 0 |
| fused | 512 rows | 70 | 9 | 0 | 0 |
| fused | 25% | 1,172 | 4,097 | 0 | 0 |
| fused | whole | 4,102 | 16,354 | 0 | 0 |
| tiled Range→And→Count | 1 row | 161 | 1 | 0 | 2 |
| tiled | 1% | 2,820 | 165 | 0 | 330 |
| tiled | whole | 158,051 | 16,384 | 0 | 32,768 |
| lane EqU32 under plane→MaskedSumI32 | 1 row | 172 | 1 | 128 | 1 |
| lane | 25% | 379,509 | 4,097 | 524,416 | 4,097 |
| lane | whole | 1,828,878 | 16,384 | 2,097,152 | 16,384 |

- Cost follows extent width, not `n_rows`: a 1-row extent is 62–172 ns on every shape.
- The word columns come from the extent's semantic span (`touched_words`), which the in-crate
  tile-plan test pins the executor to cover exactly; not from instrumentation.

## Correction of #1267 (append-only)
`2026-09-23-cubecl-llvm-boundary-and-audit-regrade.md` lesson 1 says the ranged entry point
needs "rebasing every mask and lane for ranges that do not start on a word boundary".
**The required property is correct absolute execution over a non-zero, non-word-aligned
extent, not physical rebasing.** It is met with word bounds and register edge masks, and
rebasing would have broken the absolute-coordinate law above.

## What remains before scheduling
- OQ-5 (thread/rayon vendor) is untouched and not resolved.
- No scheduler, queue, thread or affinity exists, and no scheduler work has started.
- Merge laws are still missing for the group terminals and the `_sym` SUM, which are refused on
  partial extents.
- The next expression→terminal folds (`plane ∩ plane → Count`, lane predicate → Count,
  ternlog → Count) would make an extent-dispatched unit a small algebraic program rather than a
  tile-writing bitmap machine. They are not started.
