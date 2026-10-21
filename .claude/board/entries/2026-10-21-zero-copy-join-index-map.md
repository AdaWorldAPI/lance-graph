# 2026-10-21 — Zero-copy joins: the index map `$i` over lanes

**Status:** OPEN (one failed gate, below). Lab only (mork-pathmap-probe, crepe-plan-probe).

## What changed

- A relation is two lanes (`ScalarBuffer<u32>`, shared with Lance's column buffers, no copy). A join side is an index map only: presence plane, per-slot offsets, and row positions into the lanes. Values are read in place (`lane[pos[k]]`), with a one-slot prefetch lookahead.
- Answers are folds: counts read offsets, a pinned pattern inverts to one slot, enumeration reads `JoinRow` in place. The upstream MORK/PathMap row sets are only outside comparators.
- The domain (global id → local slot) is a presence mask with a running count per word when the ids are dense (slot = rank + popcount, value order kept); a sparse id set keeps the sorted list.
- A restriction is one `eq_u32_to_mask` over the lane; semijoin centers are a domain probed per row. The index map walks the kept bits. Nothing trims or copies rows.

## Measured (crepe-plan-probe `examples/measure.rs`, release, loaded 4-core host)

- With pushdown and semijoin (36 rows prepared), prepare is µs at every n up to 50 000; mask and terminal stay ns / low µs. Parity ok against every oracle.

## Failed gate (OPEN)

- The ablation arms without pruning (`no-semijoin`, `none`) at n = 50 000 index 56 k and 119 k rows and leave the µs budget. Cause: building `$i` from unordered lanes is a counting sort whose count and scatter passes cost about 10 ns per row here. A rank-free dense counting sort was tried and did not help (bound by the random writes).
- Candidate fix, not decided: lanes sealed in key order at write time (the `OrderedLaneWitness` path `addressed.rs` already attests), so `$i` is run offsets and the scatter disappears. A relation keyed by two columns needs two orders or keeps a permutation. Write-path decision; the operator chooses.
