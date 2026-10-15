# 2026-10-14 — Delta granularity: 16-byte slot deltas vs 512-byte row images

**Status:** MEASURED. Decision OPEN (operator).

**Question (operator, 2026-10-14):** store a thought's delta as the whole
512-byte row or only the changed 16-byte slots, when up to 10^9 folds land in
up to 64k rows. Pre-registered in `.claude/plans/ce64-hypothesis-layers-v1.md`
§ "Delta granularity" before the probe was written.

**Probe:** `crates/cognitive-shader-driver/examples/delta_granularity_probe.rs`.
Built with `rustc -O -C target-cpu=native`; the probe printed
`tier: v4 (avx512f)`. 4-core VM, two full runs, median of 5 per cell. 5 tests,
5 disable runs red.

| measure | row image (R) | slot delta (S) |
|---|---|---|
| bytes per dirty row | 516 | 12 + 16k (k = slots written) |
| 64 tables, every row dirty | 2.016 GiB | 0.109 GiB (k = 1) … 1.984 GiB (k = 31) |
| seal per dirty row, k = 1 | 64 … 406 ns | 17 … 56 ns |
| one-slot read | 8.7 … 13.4 ns | 8.5 … 20.5 ns |
| full-row read | 70 … 148 ns | 94 … 594 ns (1.1 to 4.3 times R) |

Corrected 2026-10-15 (review): the slot store keeps three `u32` per dirty
row (row, mask, start), not two; the byte rows above use 12 + 16k. The rule
at k = 1 still holds (0.054 <= 0.0625).

**Folds (same for both arms):** 2.93 ns/fold over 1 MiB of accumulators,
13.0 to 15.2 ns/fold over 31 MiB in random order, 4.64 ns/fold over the same
31 MiB sorted. 10^9 folds take 3 to 15 s; sealing 64k rows takes 3 to 35 ms.
Fold cost depends on accumulator footprint and order, not on the arm.

**Pre-registered rule:** S is the default at k = 1: met in both runs. The
crossover k* (seal + full-row read) was not stable: 31/31/16 in run 1 and
16/8/16 in run 2 for 1/10/100 % dirty rows.

**Caveats:** M3 at 1 % dirty includes a 256 KiB index allocation per seal,
so it is not a per-row cost. k* adds one seal to one read. Lance append bytes
and time are not measured.

**Recommendation:** slot deltas by default. Row images only for a thought
that writes at least half the slots per row, and only after k* is re-measured
on a quiet host. The larger lever is fold locality: keep a table's
accumulators within L2 or sort folds by row (3 to 5 times on fold cost).

**OPEN:** operator decision on the default; k* on a quiet host; Lance append
cost per arm (D-HL-7).
