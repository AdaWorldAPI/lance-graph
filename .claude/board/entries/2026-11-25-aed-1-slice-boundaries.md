# 2026-11-25 — D-AED-1: slice statistics by sealed boundaries

**Status:** MEASURED. Plan: `.claude/plans/adaptive-exact-decimal-binding-v1.md` D-AED-1.

Command: `cargo run --release --example slice_bounds` in `crates/exact-decimal-probe`
(1 048 576 rows, boundaries 7 rows past a word, best of 50, 4 cores, idle).
Second of two runs; the first ran 15–25 % slower on the grouped rows.

| path | ns / 64 lanes (1 / 16 / 256 slices) | bytes per call |
|---|---|---|
| grouped, slice-id lane built per call | 148 / 183 / 182 | 4.2 MB |
| grouped, slice-id lane held | 141 / 166 / 167 | 2–10 KB |
| bounds, N S Q (`GroupPowerSumsI32`, zero lane held) | 142 / 167 / 154 | 2–10 KB |
| bounds, grouped S only (`GroupSumI32`, zero lane held) | 130 / 146 / 148 | 2–4 KB |
| bounds, N S only (`Count` + `MaskedSumI32`, no lane) | 30 / 35 / 46 | 2–6 KB |

- `execute_extent` takes each sealed slice in place at unaligned boundaries, so
  the slice-id lane is not needed. Values equal the #2182 pass
  (`slice_boundaries_equal_the_slice_id_lane`; a one-row leak turns it red).
- Candidate (a) removes the per-call 4 MB lane, not the grouped cost: Σx² still
  needs a key lane, and the one-key grouped pass costs the same as the keyed one.
- The grouped cost is the grouping, not the square: grouped S alone is within
  10 % of N S Q, and 4–5× the two ungrouped passes. That is the
  execution-level insufficiency candidate (b) needs: an ungrouped
  `masked_power_sums_i32` in ndarray, then a mask-risc terminal.

OPEN: candidate (b) is an ndarray kernel plus a terminal (isa-anti-lasagne-warden
review); ndarray is a separate repository and PR.
