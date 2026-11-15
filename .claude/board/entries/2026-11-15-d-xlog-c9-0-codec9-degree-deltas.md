# 2026-11-15 — D-XLOG-C9-0: Codec9 lanes for degree deltas (lab only)

**Kind:** one arm appended to `crates/xlog-relcomp-probe/examples/triangle_pairing_baselines.rs`
(`codec9_delta_arm`). No production code; nothing wired into W4.
**Grade:** MEASURED, correctness only; one disable control, red.
**Sources:** lance-graph `fe5308f`; `ndarray::hpc::codec9` (`Codec9Runtime`,
`EpiLane9`) on ndarray `master`; the baseline's six families × 12 seeds.

## Question

Can per-key degree DELTAS travel as Codec9 lanes (−255..=255, the −0 codeword
= never touched) and still give the exact degrees W4's planner needs?

## Method

Each trial streams inserts (shuffled), then retract-and-reinsert of a quarter
of the facts, then retraction of another quarter, in batches of 256, 4096 and
65536 events. Per batch, every touched (relation, column, value) key gets one
lane: `encode_touched(Δ)` at scale 1, a touched net zero is +0, untouched keys
stay `NEVER_TOUCHED`. Merge on read adds every lane with a value. Two write
policies:

- **escape:** a key with |Δ| > 255 is not encoded; its exact Δ is applied by a
  side path and counted.
- **saturating:** the codec as shipped. `encode_touched` clamps the magnitude
  at 255 and does not refuse.

Checked after every batch against exact `i64` degrees, at the end against the
independently built final multiset, and by the exact-fold pairing choice
(the baseline's arm c, regret = chosen inner size / best).

## Result (`cargo run --release --example triangle_pairing_baselines`)

| batch | batches | overflow lanes | escape: batches wrong | saturating: batches wrong | saturating: max regret | saturating: pairing changed |
|---|---|---|---|---|---|---|
| 256 | 21,384 | 0 | 0 | 0 | 1.000 | 0 / 72 |
| 4,096 | 1,362 | 87 | 0 | 97 | 1.000 | 0 / 72 |
| 65,536 | 108 | 728 | 0 | 72 | **378.386** (star_hub) | 7 / 72 |

- The escape arm is exact in every batch at every size; regret 1.000.
- The saturating arm is exact only while every per-batch delta fits. Its
  failure is silent: no error, wrong degrees, and at batch 65,536 a pairing
  378× worse than optimal on a hub-skewed relation. Largest delta seen: 3,010.
- Overflow is a property of the batch size and the skew, not of the codec:
  zero at 256 events, and only skewed families overflow at 4,096.

**Disable control:** removing the escape (encode every delta, merge every lane)
fails the end-of-stream assertion at batch 4,096. Run, red, reverted.

## What the never-touched codeword buys here

Skipping, not correctness. Merge-on-read reads the −0 codeword to skip a lane
without a second mask, and touched-but-net-zero lanes (+0) exist in every
configuration (110 to 42,124 per family), so the distinction carries
information. But adding a zero delta is harmless, so no result above depends on
it. Its value would be in a merge that visits only touched keys or reports
"unchanged since the last fold", which is the information defect A of
D-XLOG-COV-0 lacks. That is not measured here.

## Conclusion

Codec9 can carry W4's degree deltas exactly **only with a caller-side
overflow escape**, which Codec9 does not have. Without one it is a silent,
planner-visible error at batch sizes the cube can produce. Not measured: memory
or time against plain `i32` deltas, a packed 9-bit store, or the incremental
fold cost.

## Next falsifiers

- Bound batch size by a skew statistic (`heaviest`) so overflow cannot occur,
  and confirm the saturating path is then exact.
- Visit only touched lanes and measure merge cost against the full scan.
