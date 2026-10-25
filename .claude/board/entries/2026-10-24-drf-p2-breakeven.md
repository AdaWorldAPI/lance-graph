# D-DRF-P2 — focus vs direct break-even (lab, reported, not gated)

**Status:** MEASURED (lab). **Probe:** `crates/drf-focus-probe/examples/p2_breakeven.rs`.
**Command:** `cargo run --release --example p2_breakeven` (native AVX-512) and with
`RUSTFLAGS="-C target-cpu=x86-64-v3"` (AVX2). Medians, ns per call, 4096-slot focus,
512 B rows, i32 facet, direct lookahead 16, ndarray #376 fold body. Both paths asserted equal first.
Host: Xeon 2.1 GHz, L2 8 MiB (4 instances), L3 260 MiB.

**Warm vs cold.** A focus touches one cache line per distinct ordinal: 248–256 KiB here,
whatever the row count. So repeated reps are **warm** even for 65 536 rows (32 MiB of rows).
The **cold** arm streams a 512 MiB buffer (twice the L3) outside the timed region before
each rep (21 reps). The first version of this entry called the 65 536-row warm numbers
"DRAM"; that was wrong (codex review on #1763).

| rows | map | dens | build warm | fold warm | direct warm | build cold | direct cold | N* warm | N* cold |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | random | 1.00 | 3 804 | 690 | 8 239 | 33 623 | 35 972 | 1 | 1 |
| 1024 | random | 0.25 | 3 821 | 429 | 5 271 | 34 666 | 28 748 | 1 | 3 |
| 65536 | random | 1.00 | 23 256 | 670 | 20 982 | 137 961 | 147 286 | 2 | 1 |
| 65536 | random | 0.25 | 23 638 | 430 | 20 611 | 143 249 | 141 826 | 2 | 2 |
| 65536 | clustered | 1.00 | 15 572 | 690 | 15 020 | 79 300 | 80 321 | 2 | 1 |

AVX2 rows agree within ~15%. N* cold = smallest N with `build_cold + N·fold < direct_cold +
(N−1)·direct_warm`: the first touch is cold on both paths, later reads are warm.

- **Break-even:** 1–3 folds of one focus in every cell, warm or cold. The cold first touch
  (≈30 µs at 1024 rows, 80–150 µs at 65 536) costs both paths about the same. Cold, the
  full-mask cells already win at one fold (e.g. 65 536 random: 137 961 + 670 = 138 631 ns
  vs 147 286 ns direct); the 25 % cells need 2–3, because a sparse direct read is cheaper.
- **A2** at cold N*: 0.19–0.67 — below the pre-registered 70–90 % band because N* is so small;
  A2 = N·pop / (N·pop + 4096) rises with N (0.98 at N = 64, full mask).
- **A1, A3:** NOT RUN (no writeback, no learning in this probe).
- **P2 falsifier:** predicted-unused focus + 63 abandoned switches all reclaimed by
  `evict_unused`; held foci stay correct. Disable (keep all entries) → red.

Fold cost: before ndarray #376, `masked_sum_i32` walked set bits one by one; full-mask fold
2 341 → 650 ns (AVX-512), 2 306 → 559 ns (AVX2); 25 % masks unchanged (~470 ns).

OPEN: the warm 65 536-row build (23 µs vs 3.8 µs at 1024 rows, same cache-line footprint)
suggests TLB misses across 32 MiB of 4 KiB pages; not measured. Also open: focus churn under
a memory budget, and dirty-row writeback (A1, W1–W8).

Correction (2026-10-25, see `2026-10-25-drf-tlb-probe.md`): the "same cache-line footprint" and
"whatever the row count" statements above are wrong for the 1 024-row random case (about 1 005
distinct lines against about 3 970 at 65 536 rows). The 3.8 vs 23 µs gap is not attributable to
TLB alone.
