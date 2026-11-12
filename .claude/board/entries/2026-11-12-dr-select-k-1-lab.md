# 2026-11-12 — D-DR-SELECT-K-1: the lab ran; more than one view wins

**Kind:** measurement, lab only. **Grade:** MEASURED on one VM, in-memory
source; not certified on dedicated hardware and not a storage measurement.
**Lab:** `crates/dr-select-k-lab`. **Results and command:**
`.claude/probes/2026-11-12-dr-select-k-1/REPORT.md`.
**Plan:** `.claude/plans/dr-select-k-v1.md` §4. **Linear (existing):** EXO-253.

N = 1,000,000, medians of 41 runs, two full runs committed.

- **Parity.** Every admissible view equals an independent oracle (12
  tests, 186 timed cells checked, 10 of 10 disable runs red). After the
  bind the storage counter did not move in any cell.
- **Sparse residual (about 1%).** Positions fold 9 to 23 times faster than
  the mask at K ≥ 65,536: 0.48 against 5.26 µs for count at K = 65,536.
- **Dense residual (about 50%).** The mask is about 2 times faster for
  count (10.3 against 21.9 µs at K = 65,536) and equal for sum.
- **Building the view.** Mask 15 µs at K = 65,536; positions 43 to 351 µs.
- **Order.** Top-10 costs under 0.1 µs for every view when the binding
  witnesses key order, and is refused for every view when it does not.
- **`ALL(K)`.** mask-RISC counts without a plane but has no masked sum
  without a mask operand; the lab holds one constant plane per binding.
- **Against the fold over N.** K = 65,536: 5.5 against 201 µs per fold
  (random addresses), bind 676 µs, paid back after 4 folds. The 75%
  control does not pay at R = 1 and is about even at R = 100.

**Open:** where the density threshold between positions and mask lies (two
density points only); any storage-backed source; gates 1 and 2 of plan §5.
