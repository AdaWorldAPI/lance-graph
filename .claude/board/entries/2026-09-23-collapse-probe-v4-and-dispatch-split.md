# 2026-09-23 — The tiled path is dispatch-bound, not write-bound; AVX-512 flattens the fold

**Status:** MEASURED · OPEN (per-tile call overhead; the fold's fixed per-call cost)
**D-ids:** D-WFL-FUSE (measurement follow-up to #1272). Corrects the attribution in `entries/2026-09-23-program-collapse-boolean-chains.md`, which says the tiled gap is "not only write elimination" but does not split it.

## What was added
`examples/program_collapse_probe.rs` has a fourth arm, **BULK**. It evaluates the same chain once per op over the extent's whole touched span, into whole-population buffers preallocated outside the timing. It uses the same `ndarray::simd` kernels the executor uses per tile (`mask_and/or/xor/andnot/not`, and `ternlog_dispatch` for runtime immediates). So BULK writes every derived word the tiled path writes, but with one facade call per op instead of one per op per tile.

Two derived columns come from it:
- `wr_ns = bulk − fold`: the cost of the writes themselves.
- `disp_ns = tiled − bulk`: the per-tile overhead on top of the same writes.

This is a first-order split: BULK still pays one call per op, so `disp_ns` is per-TILE overhead, not all interpretation cost. Every arm is asserted equal to the bit-serial oracle at both tiers.

Command: `cargo run --release -p lance-graph-mask-risc --example program_collapse_probe`. v4 was built with `RUSTFLAGS="-C target-cpu=x86-64-v4"` into a throwaway target dir. N = 1 048 576 rows (16 384 words), median ns.

## Whole population, Count

| chain | tier | fold | bulk | tiled | wr (bulk−fold) | disp (tiled−bulk) |
|---|---|---|---|---|---|---|
| `(a&b)\|!c` | v3 | 8.6 µs | 17.5 µs | 220 µs | 8.8 µs | 202 µs |
| `(a&b)\|!c` | v4 | 6.2 µs | 19.6 µs | 280 µs | 13.4 µs | 260 µs |
| `((a^b)&!c)\|(a&c)` | v3 | 15.5 µs | 20.6 µs | 362 µs | 5.1 µs | 341 µs |
| `((a^b)&!c)\|(a&c)` | v4 | 6.2 µs | 25.5 µs | 354 µs | 19.3 µs | 329 µs |
| `maj(a,b,c)^a` | v3 | 17.3 µs | 16.1 µs | 167 µs | −1.2 µs | 151 µs |
| `maj(a,b,c)^a` | v4 | 6.2 µs | 16.5 µs | 183 µs | 10.3 µs | 167 µs |
| `a&d` (empty by data) | v3 | 6.5 µs | 8.3 µs | 113 µs | 1.8 µs | 105 µs |
| `a&d` (empty by data) | v4 | 4.7 µs | 9.8 µs | 120 µs | 5.1 µs | 110 µs |
| 4-plane (not collapsible) | v3 | — | 28.2 µs | 296 µs | — | 267 µs |
| 4-plane (not collapsible) | v4 | — | 23.8 µs | 274 µs | — | 251 µs |

## Readings
- **The tiled path's cost is ~90 % per-tile overhead.** `disp_ns` accounts for 90–94 % of `tiled_ns` on every whole-population row, at both tiers. `TILE_WORDS = 8` (one 512-bit vector per facade call, `exec.rs`) means 2 048 tiles per op, and the gap works out to roughly 30 ns per op per tile. The derived-word writes cost at most ~19 µs. **So most of #1272's 10–26× comes from skipping the tile interpreter, not from eliminating writes.** Write elimination is real, but it is the smaller share.
- **AVX-512 flattens the fold.** At v4 every collapsible 3-plane Count costs ~6.2 µs whatever its truth table: native VPTERNLOG is one instruction per word for any immediate. At v3 the same folds cost 8.6–17.3 µs, because AVX2 composes each table from its own instruction sequence. The tiled path does not move between tiers, since it is dispatch-bound. At v4 the fold beats BULK by 2.1–4.1×. At v3 it ranges from 0.9× (`maj^a`, where the fold is slower) to 2.0×.
- **At small extents BULK beats the fold.** At 1 % of the population (165–660 words), BULK takes 95–280 ns against the fold's 190–285 ns. The fold pays a fixed per-call cost: `validate` plus re-running the symbolic recognizer (`fused_ternlog`) on every execute. That cost is only amortized at large extents.
- `wr_ns` is negative on several 1 % rows. That is per-call overhead dominating a few hundred words of work, not a negative write cost.

## Open
- Per-tile call overhead on the tiled path (~30 ns per op per tile). Tile size is bound to the scratch-size contract (`slots × TILE_WORDS`, independent of `n_rows`), so it is **not** changed here. A larger tile trades scratch footprint for fewer calls. That is a decision for whoever owns the scratch contract, informed by these numbers. OPEN; nothing built.
- The fold's fixed per-call cost. Caching the recognized `FusedTernlog` on `Program` would remove the re-interpretation per execute. Not measured separately; OPEN.
- Only x86-64 was measured. NEON was not.
