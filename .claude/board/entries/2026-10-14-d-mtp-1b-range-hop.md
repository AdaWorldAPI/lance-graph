# 2026-10-14 — D-MTP-1b — the hop as a range (measured)

**Status:** MEASURED (v4, `avx512f=true`, release, best of 5, three runs). Plan
`.claude/plans/metta-polyfill-v1.md` §6c (pre-registration) and §6d (results). Probe
`crates/lance-graph-metta/examples/join_range_probe.rs`.

**Outcome:**
- With `dst` non-decreasing over rows (checked once at ingest), the two-hop join's kept rows
  are `valid`'s runs intersected with their preimages under `dst` (`partition_point`).
  Pairs == datafrog on a BFS tree (F3), the same tree with every third level removed (F4)
  and both under a random address permutation; the range path refuses the permuted lane.
- At 10⁶ the count costs 0.0001–0.0004 ns/row with runs recorded at ingest (O(runs·log n),
  0.1–0.4 µs total) and 0.04–0.05 ns/row when the runs are scanned from the plane, against
  1.3–2.4 ns/row for the gather program on the same data. Pair emission is 1.2–2.0 ns/pair.
- Clustering alone (same gather, local reads) saves only 11–36 %; the win comes from the
  hop being a range, not from locality.

**Open:** the order witness is probe-local (the contract's `OrderedLaneWitness` covers sealed
facet lanes only); a chained range for the second hop; scattered layouts keep the gather.
