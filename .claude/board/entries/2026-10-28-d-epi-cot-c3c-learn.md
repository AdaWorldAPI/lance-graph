# D-EPI-COT-C3C: learned ordering of scent leads over Codec9/lut9

**Status:** TEST-PINNED (fixture worlds). Code: `crates/lance-graph-planner/src/nars/scent_learn.rs`. Doc: `docs/research/epistemic-cot/c3c-learning.md`.

## Decision

- `ScentLearner` reorders C3b leads within a status group, learning from investigation outcomes. Its output is an order, with no truth, Epi5, CE64 or ledger write.
- It consumes merged ndarray `codec9::EpiLane9` and `simd::lut9` and adds no kernel. The runtime has no float roundtrip and no rounding: deltas truncate in code space, and the zero rule is `P = present AND (positive OR M == 0)`, so −0.49 → +0. The rate table is the only rounding, done offline.
- The rate table follows TFLM `LUTPopulateInt8` and matches a golden from tflite-micro `5f64e4a` on all 256 int8 codes for 4 rates.
- Only odd rules are prepared. A table for another spec refuses. One source trains once.

## Evidence

- 14 tests, plus an ignored cost probe. 10 disables, each red.
- Held-out PZ5, 600 worlds, separable fixture: learned 1.00 vs scent 4.51, round-robin 4.55, random 4.47. Uninformative twin: learned 4.59 vs random 4.47.
- rank 706 ns, learn 425 ns, scent 3.0 µs (8 leads, one host).

## Open

- The label should become "the ablation discriminated", using the ledger's `sensitivity`.
- Overlapping-evidence fixtures and real HIRO and ERP-consumer outcomes.
- `lance-graph-nsl` is gated on the NSL H7 oracle and the H1 G01–G12 fixtures; it is not built here.
- The spec binding is transient. Production generation binding stays with EXO-152 PZL-4/G1.
