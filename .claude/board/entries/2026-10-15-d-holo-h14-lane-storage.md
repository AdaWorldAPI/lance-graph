# D-HOLO-H14 — full row vs connected delta vs inverted fold (v4, 2026-10-15)

**Status:** MEASURED. Plan: `.claude/plans/d-holo-address-indexed-algebra-v1.md` (H14, two amendments, results).
**Probe:** `crates/lance-graph-mask-risc/examples/holo_lane_storage_probe.rs`; `tier: v4 (avx512f)`; one class timed.

- C (per-lane tables indexed by the row address, presence masks folded) wins
  every counting query: F1 10–24× faster than B, F2 39–256× (0.017–0.59 ns
  per aperture row against B's 0.86–109).
- C loses every merge-on-read: full-row read 259–1,529 ns against B's
  51–1,005 and A's 38–152; the merge probes every touched lane table.
- Bytes handed to the writer: C ≈ B (0.7–1.2×), both far below A at low k.
- Rule: C default not met (reads), B default over A not met (B's reads at
  100 % dirty up to 4.6× A at k = 4). Returns to the operator; crossover stated in the
  plan.

**OPEN:** H14b — C plus a positional per-row lane mask, so merge-on-read
reads only the k changed tables; 16 classes untimed; C's seal at low dirt is
page-fault bound.
