# D-DRF-P2 — focus vs direct break-even (lab, reported, not gated)

**Status:** MEASURED (lab). **Probe:** `crates/drf-focus-probe/examples/p2_breakeven.rs`.
**Command:** `cargo run --release --example p2_breakeven` (native AVX-512) and with
`RUSTFLAGS="-C target-cpu=x86-64-v3"` (AVX2). Medians, ns per call, 4096-slot focus,
512 B rows, i32 facet, direct lookahead 16. Both paths asserted equal first.

| rows (MiB) | map | density | build | fold | direct | N* |
|---|---|---|---|---|---|---|
| 1024 (0.5) | random | 1.00 | 3 827 | 2 341 | 8 198 | 1 |
| 1024 (0.5) | random | 0.25 | 3 824 | 475 | 5 502 | 1 |
| 65536 (32) | random | 1.00 | 20 733 | 2 323 | 20 227 | 2 |
| 65536 (32) | random | 0.25 | 19 924 | 516 | 17 514 | 2 |
| 65536 (32) | clustered | 1.00 | 16 642 | 2 312 | 15 457 | 2 |

AVX2 rows agree within ~10%. Clustered 1024-row cases match random 1024 (same N*).

- **Break-even:** the focus pays off from the 1st (L2-resident) or 2nd (32 MiB, DRAM)
  repeated fold of the same focus. No win when a focus is folded only once on DRAM.
- **A2** at N*: 0.20–0.67 — below the pre-registered 70–90% band, because N* is so small;
  A2 rises with N (N·pop / (N·pop + 4096)). At N=64 full mask, A2 = 0.98.
- **A1, A3:** NOT RUN (no writeback, no learning in this probe).
- **P2 falsifier:** predicted-unused focus + 63 abandoned switches all reclaimed by
  `evict_unused`; held foci stay correct. Disable (keep all entries) → red.

OPEN: the fold cost is `ndarray::simd::masked_sum_i32`, which walks set bits one by one
(`src/simd_masking_ops.rs:814`), ≈0.57 ns per selected slot. A dense SIMD masked sum would
lower `fold` and move N* only further toward the focus. Not changed here.
OPEN: cold page faults, focus churn under a memory budget, and dirty-row writeback (A1, W1–W8).
