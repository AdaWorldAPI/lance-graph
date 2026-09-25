# 2026-09-25 — One shader dispatch materialized O(rows²) to return at most 8 hits

**Status:** MEASURED · DONE (two collections removed, output digest unchanged; duplicate candidate slots replaced by one SPOFC candidate per row, #1293) · OPEN (the prefilter row list; writing SPOFC into CE64)

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

## The `cycle_index` trace found a cancellation (#1293)
Stage [4] builds `cycle_fp = XOR over hits of R^{cycle_index} · row`. Stage [3] offered every supporting relationship (each content pair, each P64 cascade hit) as its own slot, and a row's `cycle_index` is its position, so one row supported N ways gave N identical terms. XOR cancels pairs: on empty predicate planes, 36 of 60 dispatches had a repeated row in top-k, and in some the strongest row vanished from `cycle_fp` entirely.

**Fix.** One candidate per surviving row. Its relationships aggregate into a SPOFC record (driver-private; the only code precedent is arm-discovery's `CandidateTriple`/`TruthU8`): predicate union, best-resonance partner as object (`Row` or P64 `Palette` target), support count `m`, and `TruthU8 { f = best resonance, c = m·255/(m+k) }` via `evidence_confidence_u8`, extracted from `arm_to_truth_u8` so both share one convention.

**Effect, 60 dispatches each, old vs new.** Non-empty planes: `cycle_fp` changed in 12, edges and top-k in 60. Empty planes: 48, 48, 48; repeated rows 36 → 0. Disabling only the predicate union drops the non-empty edge change to 12, so 48 of those edge changes are the union's `CausalMask` bits. The digest above is therefore intentionally not preserved by this change.

**Allocation.** One more allocation (the candidate table, 6 total), still constant in rows; bytes grow 28 per surviving row (4 row id + 24 SPOFC). `dispatch_trace` pins both.

**Falsifiers, each disable-verified:** re-offering each relationship → the braid test fails; not counting support → both SPOFC evidence tests fail; dropping or overwriting the union → the union test fails (it had no falsifier before; all 111 lib tests stayed green without it).

**Not done here.** The SPOFC record ends at a `debug_assert`: the emitted CE64 still packs `s = row%256, p = 0, o = (row/4)%256`, `f = c = resonance`. CE64's 24-bit S/P/O could hold the target (palette256³); the driver does not write it. That, and carrying the evidence across cycles without recounting it, is the follow-up.
