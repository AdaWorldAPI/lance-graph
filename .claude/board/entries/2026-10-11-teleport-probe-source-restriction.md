# 2026-10-11 — Source-restricted folds: eliminated work vs accelerated work (D-TELEPORT-0)

**Status:** MEASURED. Probe only, no production primitive.
`crates/lance-graph-quack/examples/teleport_probe.rs`.

```text
CARGO_PROFILE_RELEASE_DEBUG=0 cargo run --release -p lance-graph-quack --example teleport_probe
```

The question: for `R(s,o) = K(s,o) AND B(o)`, does addressing only the rows
that can survive ELIMINATE work, or does each arm do the same work faster?
Every arm runs the shipped mask-risc executor; every count is asserted
against a per-row oracle. Work counters come from the masks and program
shape, never from a timer.

Setup: 1,048,576 edges (`s: i32`, `o: u32` = node row ordinal), 65,536 nodes,
`B` a resident node bitmap, `K = s < t`. Host: 4 shared vCPUs, load ~1.4;
timings vary up to ~1.5× between runs, so only order-of-magnitude
differences are read. No cache-miss counter (`perf` unavailable).

## Arms

- **A** `Pred K`, ungated `Gather`, `And`, `Count` (scan, join, filter).
- **B** `Pred K`, `Gather under K`, `Count` (fused, gated gather).
- **C** clustered layout only: `hi = partition_point(s < t)`, then `Gather`,
  `Count` over `execute_extent(0..hi)`. The search is timed with the arm.

## Results (sel_B 0.5; sel_B 0.01 matches within noise)

| sel_K | A words / fk-reads / ns | B words / fk-reads / ns | C words / fk-reads / ns |
|---|---|---|---|
| 0.5 | 16384 / 1048576 / ~1.8 ms | 16384 / 523683 / ~1.1 ms | 8183 / 523683 / ~0.8 ms |
| 0.01 | 16384 / 1048576 / ~1.7 ms | 16384 / 10459 / ~0.19 ms | 164 / 10459 / ~14.5 µs |
| 0.0001 | 16384 / 1048576 / ~1.8 ms | 16384 / 126 / ~0.17 ms | 2 / 126 / ~0.48 µs |

Scattered layout: A ~1.8 ms flat; B falls to a floor of ~0.17–0.29 ms; C
does not exist.

## Reading

- **B eliminates foreign reads, not source reads.** Its fk-reads track
  `|K|`, but it still reads every `s` and writes every tile. Its floor
  (~0.17 ms) is the K predicate scan.
- **C eliminates source reads.** Words visited and fk-reads both track `|K|`;
  the `s` lane is read ~20 times (the binary search). At 0.01% C is ~370×
  faster than B, search included. At 50% it is ~1.4×: little is left to eliminate.
- **The scattered control behaves as predicted.** With no interval, the
  restriction collapses to B, so C's gain is the restriction itself, not a
  faster kernel.

## Schnittpunkt (intersection)

- Two row ranges ANDed lower today to `Range`, `Range`, `And`, `Count`:
  3 passes, 49,152 scratch words, ~10 µs. The intersection
  `[max(lo), min(hi))` computed before execution is one `Range` that hits
  `Lowering::Range`: no slot, two words touched, ~28 ns. Same answer at
  every width. **quack does not fold two ranges into one today**; the fused
  path requires exactly one `Range` op (`ir.rs` `fused_terminal`).
- **What the ~28 ns is, and is not.** It is COUNT's closed form: the fused
  `Range` terminal (`Lowering::Range`, `Count`/`Any` only) answers from the
  two boundaries, `hi - lo`, and visits no row. A fold that needs the rows
  (`SUM`, `GROUP_SUM`, `KEEP`, a gated gather) still visits every survivor of
  the intersected range; there the rewrite removes two passes and two slots
  but the cost stays proportional to the range. The finding is the
  reduction of a three-pass computation to a property of two boundaries,
  not the speed ratio.
- A monotone predicate on a sorted lane (`s < t`) scanned: ~150–190 µs.
  The crossing point by `partition_point` (~20 reads) plus one fused
  `Range`: ~0.1 µs. Only `prefix_facet`/`aperture_facet` under a witness
  make this reduction today; no lowering does it for an `i32` compare on a
  lane known to be sorted.

## Fact change vs rule change (arm B shape, scattered, sel_K 0.01)

Flip one `B` bit and re-run: ~0.40 ms. New program, compile, scratch, run:
~0.35 ms. Both are the run; neither change costs anything the run does not.

## OPEN

- An intersection rewrite (`Range ∧ Range → Range`) and a sorted-lane
  monotone-compare → `Range` lowering are the two measured gains here. Both
  need an order witness for any lane that is not a sealed facet lane; that
  witness does not exist for `i32` lanes.
- Not measured: multi-range extents, cache misses, 16M rows.
