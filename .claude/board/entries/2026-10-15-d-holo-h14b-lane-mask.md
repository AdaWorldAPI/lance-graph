# D-HOLO-H14b — inverted fold plus a positional lane mask per row (v4, 2026-10-15)

**Status:** MEASURED. Plan: `.claude/plans/d-holo-address-indexed-algebra-v1.md` (H14b, pre-registered before the run, results).
**Probe:** `crates/lance-graph-mask-risc/examples/holo_lane_storage_probe.rs` (arm M); `tier: v4 (avx512f)`; A, B, C and M timed in one run on one host; one class.

- Prediction (M's full-row and session read <= 1.25× B at k <= 4):
  FALSIFIED in 9 of 18 cells. At 1 % dirty all 6 cells pass (full-row
  0.94–1.14× B, session 0.93–1.13×); failures at 10 % dirty (up to 1.40 /
  1.87×) and 100 % dirty (up to 1.63 / 5.85×). M's bytes to the writer are
  at most B's in every cell, by construction rather than measurement (B was
  charged a `starts` offset per dirty row that M was not; under the same
  accounting they are equal). M default not met; returns to the operator.
- The mask fixed most of C's merge cost: M's full-row read is faster than C's
  in 35 of 36 cells (C/M 0.98–3.69×).
- F3 (fold, then build the named rows): M beats B in all 12 cells at 1 %
  dirty and in 18 of 36 overall; M beats C in 33 of 36.

**OPEN:** why M loses at higher density (HYPOTHESIS: a row's k lanes sit in k
tables, B's in one record); C's and M's seal at 1 % dirty is page-fault
bound; 16 classes untimed.

**Withdrawal note (2026-10-15, before merge).** M is withdrawn as a candidate
default layout for thought-first C, and the density-based A/B/C/M switch is
withdrawn. H14 and H14b mostly measured whole-row reconstruction, which is not
C's primary operation; `Mask[i]` transposes what C's per-lane presence planes
already hold into a row-oriented index. The numbers above stand; their reading
as a verdict on C does not. M remains an experimental row-oriented acceleration
structure. Follow-up: D-HOLO-H14c. Full note: the plan's "H14b withdrawal note".
