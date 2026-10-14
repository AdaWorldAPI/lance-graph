# 2026-10-14 — D-HOLO falsifiers H1, H5, H6 (measured)

**Status:** MEASURED (v4, avx512f). Plan `.claude/plans/d-holo-address-indexed-algebra-v1.md`,
§ Results (2026-10-14). Probes: `crates/lance-graph-mask-risc/examples/holo_{k_branch,coord,interval}_probe.rs`.

**Outcome:**
- **H1 (K branches): MIXED.** The bit-parallel arm over per-row words is flat in K (0.84–1.47×
  from K = 1 to 64). With 100-query amortization at s = 10 % it beats gate-once-then-AND only on
  the 4,096-row aperture (4.8 vs 31.0 µs); on the full tile gate-once is as fast (71.2 vs 71.7 µs).
  The verdict depends on storing coverage by row: with per-thought planes stored, gate-once wins
  every reference cell.
- **H5 (native vs crossing): CONFIRMED.** A prefix `Range` reads exactly the prefix's occupancy
  words and is 292–573× faster than a key sweep; a permuted key lane makes it disagree with the
  oracle, so it needs a layout-order witness. A row popcount answers all 64 perspectives in 1 ns.
  Reverse `memberOf` costs the aperture's rows without a seal-time inverse. The per-group plane
  inverse does not pay back on prefix-restricted queries within 1,000 queries (1,114–1,740); CSR
  does (≤ 546), at 6–13 µs per inserted edge.
- **H6 (closure): HOLDS.** Formulas over Morton boxes and prefixes stay interval sets under ∩, ∪
  and ¬, match the oracle, cost the same as box area grows 256× at a fixed interval count, and were
  cheaper than full-tile masks and value sweeps in all 24 cells. Quack's `fold_ranges` already
  covers the ∩-of-ranges case.

**OPEN:** the stored orientation of thought coverage; a vectorized positional popcount; a single
multi-interval `Range` write (not built, no new op added); external baselines (DuckDB,
differential dataflow, TACO).
