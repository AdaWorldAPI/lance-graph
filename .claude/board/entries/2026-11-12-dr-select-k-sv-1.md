# 2026-11-12 — D-DR-SELECT-K-SV-1: a position selection admitted for COUNT and SUM

**Kind:** admission of one operand form in mask-RISC and Quack. No production caller.
**Grade:** MEASURED (`cargo test -p lance-graph-mask-risc -p lance-graph-quack`: all
suites pass; new: `tests/sel.rs` 6, `tests/sel_alloc.rs` 1, Quack `lower_sel` 1;
7 of 7 disable runs red; `cargo clippy --all-targets -- -D warnings` clean on both crates).
**Plan:** `.claude/plans/dr-select-k-v1.md` §14. **Linear (existing):** EXO-253.
**Depends on:** ndarray PR 423 (`sum_i32_at`, `count_gt_i32_at`) being in ndarray `master`.

- `execute_sel(fold, planes, positions)` beside `execute_into`: a borrowed bag
  of rows of the frame, one `i32` lane, one delegation to ndarray per fold,
  no allocation. `Program` and `Planes` are unchanged.
- Refusals shared with the oracle `reference_execute_sel`: missing lane,
  wrong lane kind, lane not spanning the frame, list over the sum bound,
  position outside the frame (first offender named).
- The sum is exact or refused, like `MaskedSumI32`; the bound is on the list.
- Quack `lower_sel`: same query as `lower`, selection plane supplied as
  positions; two shapes, the rest refused.
- mask-RISC law A1's text and Quack's matrix row R1 are amended in place.

**Open:** a witnessed binding outside the lab crates (the version check is the
caller's); the DuckDB adapter; R4 and R7.
