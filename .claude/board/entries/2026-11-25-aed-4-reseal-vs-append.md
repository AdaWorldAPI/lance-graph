# 2026-11-25 — D-AED-4: reseal against append

**Status:** MEASURED (one synthetic workload). Plan: `.claude/plans/adaptive-exact-decimal-binding-v1.md` D-AED-4.

`SliceLog` (`crates/exact-decimal-probe`): `Reseal` rewrites the tail slice
at the common exponent and width; `Append` seals each batch as its own slice.
A slice a reader holds (an `Arc` clone) is never rewritten: `Reseal` falls
back to a new slice, `compact` and `add_into` refuse. `add_into` adds a PM9
delta with Abakus `add_pm` and promotes on `PmSum::carry`, taking a carry
lane's exact value through the wider path. `compact` demotes to the coarsest
exponent and narrowest width. Exact against direct reads after every event,
both policies (tests in `tests/probe.rs`; disable runs on the witness
fallback and on the carry value, each red).

Command: `cargo run --release --example slice_events`. 256 batches of 256
scale-2 amounts, every 9th absent, a scale-3 value in 16 batches, a value
past PM17 in 8:

| policy | slices | encoded B/value | values rewritten | write ns/value | aggregate: per slice | per-value rescale |
|---|---|---|---|---|---|---|
| Reseal | 1 | 4.125 | 8 421 120 | 1 929 | 44 ns (0 alignments) | 200 µs |
| Append | 256 | 1.281 | 0 | 131 | 2.1 µs (243 alignments) | 217 µs |

- Reseal rewrites quadratically (128× the 65 536 values written here), and
  one wide value widens the whole column to `i32`.
- Append keeps each slice at its own width (most stay PM9), rewrites
  nothing, and aligns aggregates per slice: 2.1 µs for 256 slices, against
  ~200 µs to rescale every value.
- The HYPOTHESIS holds on this workload; it is one synthetic write pattern.

OPEN: real write patterns (D-AED-6, blocked on samples); when compaction
should run; an append-then-compact policy is not measured.
