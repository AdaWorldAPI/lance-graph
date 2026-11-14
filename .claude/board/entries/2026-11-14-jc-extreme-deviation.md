# 2026-11-14 — D-JC-EXTREMA-1: range and largest standardized deviation in `jc::stats`

**Kind:** two functions in `crates/jc/src/stats.rs`, one test file.
**Grade:** MEASURED against an independent scalar oracle. Descriptive only.
**Depends on:** `ndarray::simd::masked_argminmax_i32` and `PowerSums`, both on
ndarray `master`. Independent of the open ndarray kernel PR.

## What

- `range_from_extrema(min, max) -> Option<i64>`: `max − min`, subtraction in
  `i64`.
- `extreme_deviation_from_power_sums(&PowerSums, min, max) ->
  Option<ExtremeDeviation>`: range, `G = max(x̄ − min, max − x̄) / s` with the
  sample standard deviation, the row and value `G` is taken at, mean, `s`.

Inputs are the summaries the substrate already folds over one population mask.
No row is read, nothing is traversed twice, no SIMD is written here.

## Rules fixed in code

| rule | how |
|---|---|
| no inference | no critical value, no p-value, no outlier verdict; the section note says why |
| the row stays in the caller's frame | returned as passed in; never interpreted |
| equal tails | exact integer comparison `n·(x̄ − min)` vs `n·(max − x̄)`; lower row wins |
| widening | `i64` range; `i128` for `n·Σx² − (Σx)²` and both tail numerators, all checked |
| degenerate input | `None` for `n < 2`, zero variance, `min > max`, mean outside `[min, max]`, both extrema at one row with different values, `G > (n − 1)/√n`, moments past the `i128` forms |

## Tests

`crates/jc/tests/extreme_deviation.rs`, 11 tests: 300 random populations
against a scalar oracle that computes `s` from centred rows; masked-out rows;
each degenerate and inconsistent input; both tie orientations; the bound
attained for `n` in 2, 3, 10, 100; the full `i32` span; three tiles merged;
rows offset by a base.

Disable runs, anchor asserted, exit 101: tie rule, tail choice, the
`(n − 1)/√n` bound, `i64` widening, mean-inside check, sample divisor,
same-row check (7 red).

`sums.n < 2` removed: tests stay green. The guard is redundant: for `n ≤ 1`
the variance term is never positive, or the deviation is not finite and
positive, so a later check returns `None`. It is kept as the stated rule.

`cargo test --manifest-path crates/jc/Cargo.toml`: all suites pass.
`cargo clippy --all-targets -- -D warnings`: clean.

## Not done

- No studentized range.
- `i32` lanes, one population per call; no grouped form.
- Nothing in JC calls these yet.
