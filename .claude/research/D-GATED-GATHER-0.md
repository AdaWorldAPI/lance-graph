# D-GATED-GATHER-0 — the semijoin's cost is a branch and dead rows, not the carrier

> **READ BY:** `fold-carrier-scientist`, `simd-savant`, anyone touching
> `ndarray::simd::mask_gather_u32`, mask-risc `MaskOp::Gather`, or proposing a
> sparse carrier.
> **Status:** MEASURED 2026-10-07. Instrument:
> `crates/lance-graph-benches/examples/gated_gather_probe.rs` (1M rows, release,
> debug=0, median of 15). Every route's count is asserted equal to an
> engine-free oracle before its time is printed.

## Setup

u32 fk lane into a 1 024-row foreign table, half its rows marked by a hash (so
the per-row bit is unpredictable). A gate mask of the stated density (the
first conjunct's survivors) is built once, outside the timed region
(`gate_us` ≈ 200–300 µs, the same for every route). Answer =
`popcount(gate ∧ gather(fk, foreign))`.

Every hand route uses a branchless bit test **with the kernel's
out-of-range-is-false contract kept**, so no route gets a cheaper test than
production.

## Result (µs)

| layout | density | prod | full branchless | word-gated | bit-gated | ordinal Vec<u32> |
|---|---|---|---|---|---|---|
| uniform | 100 % | 5295 | 1409 | 1386 | **1150** | 2013 |
| uniform | 50 % | 5311 | 1413 | 1398 | **747** | 1191 |
| uniform | 10 % | 5297 | 1386 | 1397 | **254** | 415 |
| uniform | 1 % | 5298 | 1395 | 751 | **65** | 80 |
| uniform | 0.1 % | 5292 | 1438 | 102 | 13.7 | **12.5** |
| uniform | 0.01 % | 5491 | 1404 | 16.6 | **5.3** | 5.6 |
| clustered | 100 % | 5247 | 1431 | 1402 | **1250** | 1952 |
| clustered | 50 % | 5263 | 1434 | 692 | **584** | 983 |
| clustered | 10 % | 5323 | 1439 | 137 | **108** | 158 |
| clustered | 1 % | 5316 | 1424 | 19.1 | **14.4** | 19.4 |
| clustered | 0.1 % | 5319 | 1453 | 7.5 | **5.1** | 6.3 |
| clustered | 0.01 % | 5297 | 1430 | 6.3 | **4.2** | 4.4 |

Exact work counters (from the probe): foreign loads issued = N for prod and
full; 64 × live words for word-gated; survivors for bit-gated and ordinal.
Ordinal additionally materialises 4 bytes per survivor (4 MB at 100 %).

## What it shows

1. **Kernel defect, independent of gating: 3.8×.** Production
   (`ndarray::simd::mask_gather_u32`, reached via mask-risc `Gather`) does
   exactly the same loads as `full branchless` and takes 5.3 ms vs 1.4 ms. The
   kernel's `if idx < rows && bit == 1 { acc |= … }` is a data-dependent branch
   on an unpredictable bit. Making it branchless is a pure kernel change with
   the same contract.
2. **Visiting live bits is the schedule.** Bit-gated (skip zero gate words;
   inside a live word visit only set bits) is fastest or within noise of
   fastest at every density and both layouts, and ties the full scan at 100 %.
3. **Word granularity is not enough for scattered survivors.** Word-gated
   gathers 64 lanes per live word; at uniform 1 % that is 47× the loads
   bit-gated issues (751 vs 65 µs). It approaches bit-gated only when survivors
   fill their words (clustered).
4. **The selection vector never wins meaningfully.** Ordinal ≥ bit-gated
   everywhere except one 0.1 % cell (12.5 vs 13.7 µs, within noise), and it
   costs 1.6–1.7× at high density. **Quack matrix R1 ("no selection vector")
   holds.** The mask IS the execution schedule; no carrier conversion needed.

## Consequences

- **BUILD (ndarray):** make `mask_gather_u32` branchless (same contract), and
  add a gated form that reads `index[i]` only for set bits of a gate mask.
  Both are scalar by nature (no bit-gather exists on any backend); the gated
  one is an iteration-order change, not a new semantic.
- **BUILD (mask-risc):** `MaskOp::Gather` gains the `under` operand `Pred`
  already has, lowered by `quack::lower` exactly like a gated `Pred`.
- **No new carrier, no new V4 opcode.** KEEP stays mask geometry; SparseOrdinal
  becomes at most an earned *materialisation* for a consumer that demands
  ordinals, never an intermediate.

Predicted effect on `bundle_probe`'s query (cmp ∧ semijoin): today ~5.5 ms at
every density; with both fixes ≈ gate cost (~0.2 ms) + bit-gated gather
(0.06–1.2 ms by density).
