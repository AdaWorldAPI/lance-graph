# 2026-10-07 — The semijoin's cost is a branch and dead rows; the mask is the schedule

## MEASURED

`crates/lance-graph-benches/examples/gated_gather_probe.rs`, 1M rows, 6
densities × 2 layouts × 5 routes, all asserted equal to an oracle.

- Production gather (`ndarray::simd::mask_gather_u32` via mask-risc `Gather`)
  takes 5.3 ms at every density. A branchless gather doing identical loads with
  the same out-of-range contract takes 1.4 ms: **3.8× from a per-row branch on an
  unpredictable bit.**
- Visiting only the set bits of the gate (bit-gated) is fastest or within noise
  at every density, both layouts. It is 65 µs at uniform 1 %, against 751 µs
  for word-gated and 80 µs for an ordinal `Vec<u32>`.
- The ordinal selection vector never wins outside noise and costs 1.6–1.7× at
  high density. Quack matrix R1 holds.

## CONSEQUENCE

- ndarray: a branchless `mask_gather_u32` (same contract), plus a gated form
  that reads `index[i]` only for set gate bits.
- mask-risc: `Gather { under }`.
- No new carrier and no new V4 opcode. The mask is the execution schedule.

## OPEN

- Neither fix is built.
- The branchless gain was measured with a 50 % random foreign plane; the gain
  shrinks for highly predictable planes.

Report: `.claude/research/D-GATED-GATHER-0.md`.
