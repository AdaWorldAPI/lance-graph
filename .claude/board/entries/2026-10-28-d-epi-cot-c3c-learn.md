# D-EPI-COT-C3C: learned ordering of scent leads over Codec9/lut9

**Status:** TEST-PINNED (fixture worlds). Code: `crates/lance-graph-planner/src/nars/scent_learn.rs`. Doc: `docs/research/epistemic-cot/c3c-learning.md`.

## Decision

- `ScentLearner` reorders C3b leads within a status group, learning from investigation outcomes. Its output is an order, with no truth, Epi5, CE64 or ledger write.
- It consumes merged ndarray `codec9::EpiLane9`, `Codec9<TiesAway, _>` and `simd::lut9` and adds no kernel.
- The rate table follows TFLM `LUTPopulateInt8` and matches a golden from tflite-micro `5f64e4a` on all 256 int8 codes for 4 rates.
- Only odd rules are prepared. A table for another spec refuses. One source trains once.

## Evidence

- 14 tests, plus an ignored cost probe. 9 disables, each red.
- Held-out PZ5 (separable fixture): learned 1.00 vs scent 4.00, round-robin 4.25, random 4.73. Uninformative twin: learned 4.70 vs random 4.73.
- rank 736 ns, learn 404 ns, scent 4.5 µs (8 leads, one host).

## Open

- The label should become "the ablation discriminated", using the ledger's `sensitivity`.
- Overlapping-evidence fixtures and real HIRO/Odoo outcomes.
- The spec binding is transient. Production generation binding stays with EXO-152 PZL-4/G1.
