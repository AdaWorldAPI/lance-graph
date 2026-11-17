# EXO-290 Ask 2: parallel tiles in mask-risc, crossover measured first

**Date:** 2026-11-17 · **Status:** MEASURED · **Scope:** `lance-graph-mask-risc/examples/par_tile_probe.rs` only; no driver built

EXO-290 Ask 2 asked for a parallel tile driver, but required the crossover to be measured first. The probe splits `[0, n)` into chunks of whole tiles (`TILE_WORDS * 64` = 16,384 rows), so no two chunks share a word, the split `execute_extent` already licenses. Each worker runs `execute_compiled` with its own scratch and sink, all allocated up front. The parts merge by each terminal's law. Every parallel answer is checked against the single-threaded one; all were equal.

The three program shapes:

- `count`: `a<500 AND NOT b<200`, then `Count`;
- `sum`: the same mask, then `MaskedSumI32`;
- `groupsum`: the same mask, then `GroupSumI32` over 64 groups.

Workers came from two arms, both std-only:

- spawned per call (`std::thread::scope`);
- a pool parked on a `std::sync::Barrier`.

Measurements: 4-vCPU container, release, two runs; times are medians in µs.

| | 2 threads | 4 threads |
|---|---|---|
| coordination floor, spawn per call | ~42 | ~82 |
| coordination floor, pool wake | ~24 | ~90 |
| speedup below 64K rows | < 1 (0.1–0.95×) | < 1 (0.1–0.95×) |
| speedup at 256K rows, pool | 0.7–1.65× | 0.76–1.72× |
| speedup at 512K rows, pool | 1.5–2.3× | 1.5–3.05× |

At 64K rows the single-threaded run takes 12–45 µs, less than the coordination floor.

The window that pays is narrow. Threads win only from about 256K–512K rows (16–32 tiles), while the house rule against millisecond results caps one batch at about 512K rows for these shapes; the single-threaded group sum at 1M rows was 0.9–1.3 ms.

**Decision taken:** no driver now. The single-threaded tile loop stays the default at every size this repository measures.

**Why:** the coordination floor is the deciding number, and this container's futex wake (~24 µs per extra thread) is a VM figure. On bare metal it is typically several times lower, which would move the crossover down by about as much. A threshold pinned from this host would be wrong elsewhere.

REVISIT WHEN: the probe is run on a non-VM host, or a consumer batches more than 32 tiles per call within budget. A driver should then take the worker count and the minimum tiles per worker from the caller, not a constant. It should use the probe's split and merge, with parity per terminal (ArgMin/ArgMax ties keep the lowest row: merge in chunk order).

OPEN: no measurement on a second host.
