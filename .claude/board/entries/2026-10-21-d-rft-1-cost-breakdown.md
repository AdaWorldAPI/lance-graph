# 2026-10-21 — D-RFT-1-COST: D-RFT-1 follow-up, where the composition's time goes (measurement only)

**Status:** MEASURED, lab. Review item D of `GPT:gpt-ada-01/20261021-pr1674-1682-consolidated-review-01`; written by `session_013z2e1YXkm6o6HNrDn5BPWA` on the operator's instruction (EXO-50 `/038`). Instrument: `crates/dir-rmap-probe/examples/rft1_cost.rs`. No change to `compose.rs`, r2il or mask-risc.

`rft1_bench` reports three different comparisons and they stay apart: arm A (`Composer::compose`, r2il procedures replayed through mask-risc, support first), arm C (direct `ndarray::simd` over every coordinate), arm C+ (arm A's plan in direct `ndarray::simd`). This decomposes arm A only.

Fixture `Spec::small()` (600 users, 3 pages; `Spec::iam(8000)` exceeds 256 slots). Release build, 4 cores, load about 2 during the runs; three runs, ranges given.

| item | measured |
|---|---|
| (1) preparation vs execution | `Composer::new` 0.8–1.4 µs; arm A 212–270 µs per pass |
| (2) `execute_into` per pass | 2769 = present 600 + union 600 + path 1374 + fan 192 + `plane_count` 3 |
| (5) support first | 1374 path folds of 38 400 coordinates (3.6%) |
| (3) ns per invocation | path `count_and(3)` 54–70; present `keep(intersect 2)` 48–62; support `keep(union 4)` 101–129; fan `count_and(1)` 95–121; `plane_count` compile + scratch 319–400 ns per page |
| (2)×(3) vs measured | the prediction covers 87% of arm A; path folds are 41% of it, the per-page `plane_count` compile 1% |
| (4) four-word mask cost | one `count_and(3)` invocation 54–70 ns vs the same AND, AND, popcount in `ndarray::simd` 2.2–2.8 ns: about 52–67 ns fixed per invocation |
| (6) allocations | 39 per pass; 0 per prepared `execute_into`; 6 per `plane_count` compile, 18 per pass |
| (7) batched-fold upper bound | replacing the path invocations' fixed overhead could remove at most 72–92 µs (33–34% of arm A) |

Reading: arm A is dominated by fixed per-invocation cost, not by mask work; the work itself is a few ns. Fewer invocation boundaries is the lever. A batched multi-coordinate fold (count product / positional popcount) is NOT built here; a new terminal needs its own admission decision and falsifiers, and must keep the Bag/Distinct, NULL, witness and refusal contracts.

OPEN: `count_and(1)` over one 4-word plane costs more per invocation than `count_and(3)` over three (95–121 vs 54–70 ns) in every run; not explained. The remaining 13% of arm A is not attributed (loop, `positions`, the `counts` vector).
