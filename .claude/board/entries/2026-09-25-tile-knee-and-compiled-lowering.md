# 2026-09-25 — The per-tile gap was control, not physics: 256-word tile + compile-once lowering

**Status:** MEASURED · SHIPPED (this PR) · OPEN (per-call validation is now the largest small-extent cost)
**D-ids:** D-WFL-FUSE follow-up. Closes the two Open lines of `entries/2026-09-23-collapse-probe-v4-and-dispatch-split.md` (the per-tile gap; the fold's fixed per-call cost).

## What the probe asked
`examples/tile_sweep_probe.rs` classifies two bills before any architecture changed.

- **A.** Is the 8-word tile a hardware constraint, or a scheduling choice? The SIMD microtile is one 512-bit vector. The scheduling tile is how many words one pass of the op loop covers, and the executor already takes it from the caller's `Scratch` width, so it could be swept without touching the executor.
- **B.** At small extents, does the fold pay for folding, or for re-deriving the fold on every call?

Command: `cargo run --release -p lance-graph-mask-risc --example tile_sweep_probe`. N = 1 048 576 rows, median ns, every arm checked against a bit-serial oracle. Numbers vary about ±10–30 % run to run on this machine; the ratios below hold across the three runs taken.

## A — scheduling-tile sweep (whole population, Count, tiled path)

| tile (words) | `(a&b)\|!c` | `((a^b)&!c)\|(a&c)` |
|---|---|---|
| 1 | 1 484 µs | 2 719 µs |
| 8 (old default) | 140 µs | 223 µs |
| 32 | 43 µs | 64 µs |
| 128 | 33 µs | 27 µs |
| **256 (new default)** | **24 µs** | **20 µs** |
| 1 024 | 22 µs | 16 µs |
| 16 384 (whole) | 20 µs | 21 µs |

Time falls roughly in proportion to the number of passes until about 256 words, then flattens. At 256 words the tiled path matches the earlier BULK arm (one facade call per op over the whole span), so **the tiled−bulk gap in the 2026-09-23 entry was per-pass control overhead, not cache or batch-size effects.** The fix is batching, not a new mechanism.

Shipped: `TILE_WORDS` 8 → 256. The contract is unchanged (execution state `slots × TILE_WORDS` words, independent of `n_rows`). The worst case for a program declaring all 65 536 addressable slots grows from 4 MiB to 128 MiB of scratch; ordinary programs declare a handful.

## B — small-extent fused call, split (1 000 calls averaged per sample)

| rows | whole call | fold recognition | range recognition | kernel | remainder |
|---|---|---|---|---|---|
| 64 | 105–111 ns | 16–20 ns | 0 | ~37 ns | ~52 ns |
| 512 | 79–90 ns | 16–20 ns | 0 | ~7 ns | 55–62 ns |
| 4 096 | 95–148 ns | 16–20 ns | 0 | 26–58 ns | 52–70 ns |

Recognition is 16–23 % of a small call. The remainder (validation plus call plumbing) is the larger share, about 50–60 ns. So the small-extent crossover in the 2026-09-23 entry was **repeated interpretation, mostly validation**, not a cost of folding.

Shipped: `Program::compile()` returns a `Compiled` that holds the recognised `Lowering` (range fold / ternlog fold / tiled). It borrows the program, so the cached lowering cannot go stale. `execute_compiled()` skips recognition; `execute_extent` is now `compile` + `execute_compiled`. Measured saving is at the noise floor for a single program (a few ns to ~40 ns); the bigger win needs validation split too.

## Gates
- Every existing mask-risc, quack and report test passes, plus the excluded r2il-mask-abi-probe crate.
- Tests no longer let the default tile decide cross-tile coverage. `foreign.rs` runs under an explicit 8-word test tile. `differential.rs` sweeps 1, 3 and 8-word tiles beside the default. The report and r2il zero-copy tests state their populations against `TILE_WORDS`.
- New: `compile_records_the_lowering_the_executor_takes` (all three lowerings, compiled vs uncompiled over every extent), and a compiled arm in the collapse `check`.
- **Disable runs** (committed first, each red, then restored):
  - `lowering()` forced to `Tiled`.
  - The compiled ternlog path given the complemented immediate.
  - The test tile raised to 256, which the multi-tile guard refuses at compile time.

## Open
- **Per-call validation** is now the largest fixed cost of a small fused call (~50–60 ns). Its checks mix program-only facts (slot bounds, read-before-write) with per-call facts (plane and lane indices, `Out` shape). Moving the program-only half into `compile` is the next step; it has not been measured separately.
- **lance-graph-java's `lgj-abi` does not compile against current lance-graph `main`**. `ExecError::ExtentOutOfRange` / `ExtentUnsupported` are unhandled in `exports.rs`. This predates this PR and blocks running its tests against the new tile.
- The tile was swept on x86-64 only, at two chain shapes (3 and 4 ops, up to 33 declared slots).
