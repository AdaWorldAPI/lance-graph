# 2026-09-23 — Boolean membership → Count/Any without a mask (ternlog fold)

**Status:** MEASURED (fold shipped in PR, ndarray #322 + this lance-graph PR) · OPEN (multi-op trees, compare-count)
**D-ids:** D-WFL-T1-FUSED, D-WFL-T1-FUSED′, D-WFL-FUSE (single-op case). OQ-5 is untouched and still open.

## The question, and its answer
The question was the smallest T1 operation that lets a 2/3-input Boolean membership end in Count/Any without writing a mask, and whether the existing `U64x8` composition already does it without writing to memory.

**Answer (MEASURED):**
- **No new ISA primitive is needed.** `U64x8::{ternlog::<IMM>, popcnt, +, |, reduce_sum}` exists on every realization: avx512, avx2-polyfill, scalar, neon and wasm.
- **Only the slice loop was missing.** It belongs in T1, because mask-risc must not host a SIMD loop.
- **What landed in ndarray #322** is one family, not one function per op: `mask_ternlog_popcount::<IMM>` and `mask_ternlog_any::<IMM>`. A 2-input op is a table that ignores `c`.
- **Effect on the D-WFL-T1-FUSED rows:**
  - D-WFL-T1-FUSED's own falsifier held ("falsified if composition already achieves it without a buffer") at the ISA level.
  - Its differential gate (`mask_and` + `popcount_batch_u64`, identical answer, zero intermediate bytes) is met by the generalized function for all 256 tables.
  - D-WFL-T1-FUSED′'s open "plane∩plane `and_popcount`" is closed as the `AND2` table of this family.

Probe: ndarray `examples/ternlog_fold_probe.rs` compares materialize-then-reduce against the register fold.

| backend | words | Count gain | Any gain (worst case, no hit anywhere) |
|---|---|---|---|
| avx2 | 16 384 | 1.27× | 3.24× |
| avx2 | 262 144 | 1.50× | 6.18× |
| avx512 | 16 384 | 1.90× | 2.59× |
| avx512 | 262 144 | 1.94× | 4.39× |

- The first version of Any tested the accumulator after every chunk and lost to materializing on avx2 at 16 384 words (0.91×). The shipped form tests once per block of 8 chunks.
- A plain scalar fused loop about ties the register fold on avx2. **The win comes from not writing the mask, not from SIMD.**

## The mask-risc fold (CURRENT-CONTRACT, TEST-PINNED)
- **Shape recognised:** `Program::fused_ternlog()` matches a single `And`, `Or`, `Xor`, `AndNot` or `Ternlog` whose operands are all `Operand::Plane`, folded by `Count` or `Any` of its own `dst`. Like `fused_terminal`, this is a derived predicate. `requires_scratch()` now accounts for both.
- **Execution:**
  - Whole words go through `ternlog_{popcount,any}_dispatch`, a generated 256-arm table checked by `--check` and by the arm-count test.
  - Each word the extent cuts (at most two) is combined in a one-word register and ANDed with `edge_mask`.
- **Odd tables:**
  - The population's last word counts as a cut word, so an odd table (true on all-zero inputs) never counts dead tail bits.
  - The ndarray function counts at word level; the tail is the caller's to mask, and it is masked here.
- **Not fused** (these keep the tiled path): `Keep`, `All`, `Not`, scratch operands, and multi-op programs.

## Evidence
- **`tests/fused_ternlog.rs`:** the fold equals the Keep arm and a bit-serial oracle:
  - the four 2-input ops across 7 populations (1 … 4133) × 4 plane shapes × absolute extents cut at 63/64/65/127/128/129;
  - all 256 tables at 65 and 133 rows;
  - the odd-table tail;
  - zero allocation and an untouched poisoned arena, plus the paired half proving the probe can see a `Keep` carve;
  - the recogniser's admit and refuse halves.
- **Disable runs,** made after committing and each failing as intended:
  - recogniser off: 5 of 5 red;
  - tail cut ignored: 3 red;
  - head cut ignored: 2 red;
  - wrong OR table: 1 red;
  - edge word not masked: 3 red.
- **Probe:** `examples/ternlog_fused_probe.rs` at N = 1M rows, fold vs `Keep` + `popcount_batch_u64`. Count: 1.25–1.44× at 1 row, 3.4–7.4× at 1 %, 5.3–20× whole. The whole-population Any ratios (600–960×) are almost all early exit on dense planes, so they are **not** a like-for-like cost comparison; the worst-case figure for Any is the ndarray row above.

## What remains
- **Multi-op Boolean trees** (e.g. `(a & b) | !c` spread across ops) still materialize scratch. The next step is `fuse.rs` collapsing such a tree into ONE ternlog and handing that to this fold. Not started.
- **Lane predicate → Count/Any** (compare-count / compare-fold) is the next PR. Not started.
- **OQ-5** (thread/rayon vendor) is untouched. No scheduler work has started.
