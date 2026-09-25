# 2026-09-25 — One shader dispatch materialized O(rows²) to return at most 8 hits

**Status:** MEASURED · DONE (two collections removed, output digest unchanged) · OPEN (the prefilter row list; the stage-by-stage trace of the rest of the chain)

## The measurement (`crates/cognitive-shader-driver/tests/dispatch_trace.rs`)
A counting global allocator measures one `ShaderDriver::dispatch`, after a warm-up, at 16..256 rows. The driver's module doc claims "no allocations beyond top-k + edges".

| rows | before | after |
|---|---|---|
| 16 | 23 KB, 44 allocs | 1.2 KB, 5 allocs |
| 256 | 5.3 MB, 530 allocs | 2.2 KB, 5 allocs |

Before, bytes grew with the square of the population and allocations grew linearly. The answer is at most 8 hits either way (`hit_count` is taken after the cut to 8).

## What was materialized and discarded
- **Content pre-pass.** It pushed two hits per resonant pair into a `Vec` (O(rows²)), then stable-sorted the Vec and cut it to 8. `TopHits` now keeps the best 8 as they arrive. A new hit goes after every kept hit whose resonance is not lower, so ties keep arrival order exactly as the stable sort did.
- **`p64-bridge` `cascade`.** It returned every candidate (up to 256 per row) in a fresh `Vec`, sorted, of which the driver read 4. `cascade_nearest::<K>` keeps the K nearest, allocation-free. Both methods share one private candidate walk.

## Evidence it is the same computation
- **Digest.** A hash over every field of the crystal (60 configurations × 3 dispatches: rows 16..300, radii 50..MAX, `Auto` + `Ordinal` 0/5/11) is identical before and after (`355d5fccbc424763`).
- **Equivalence tests.** `TopHits` and `cascade_nearest` each have a test against the collect-sort-truncate they replace, on tie-heavy inputs.
- **Disable runs.** Restoring either collection breaks the pinned constant allocation count (37 → 69, 12 → 14).

## Open
- **The row list.** `passed_rows` (4 bytes per surviving row) is the prefilter's population materialized as a list of row ids. Replacing it with a mask is the natural next step, but it changes `BackingStore::prefilter`'s shape.
- **The rest of the trace.** Stage [4] rotates each hit's row by its `cycle_index` before the XOR, so each hit contributes in its own basis. That is the relative-coordinate step of the end-to-end chain, and it has not yet been traced as such.
