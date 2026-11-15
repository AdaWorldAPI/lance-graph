# 2026-11-15 — D-XLOG-PM-1: add_pm degree ledger over the triangle-pairing stream (lab)

**Kind:** lab only. `crates/xlog-relcomp-probe/src/pm_ledger.rs` (`PmLedger`,
with a lab copy of ndarray `examples/add_pm/pm.rs` at `acaa120`), tests in
`tests/pm_ledger.rs`, and `add_pm_arm` in `examples/triangle_pairing_baselines.rs`.
Nothing wired into W4 or the cube.
**Grade:** MEASURED, correctness and counts; four disable controls, each red.

## Question

Can W4's degrees be held as Codec9 `P × M` lanes and maintained with `add_pm`
over the real delta stream, exactly, with an escape for sums past ±255 and a
way back to a lane when cancellation returns them to range?

## Method

`PmLedger`: a key's value is either a lane or an exact wide `i64`. An
addition whose exact sum leaves ±255 escalates the key (a **carry**); a wide
value back within ±255 is demoted to a lane (a **demotion**). Per batch,
events go into a delta ledger; the batch delta is absorbed into the degree
ledger 64 keys per `add_pm`. Degrees are checked against exact `i64` after
every batch, against the independently built final multiset at the end, and
by the exact-fold pairing choice. Six families × 12 seeds × 3 batch sizes on
two streams: `churn` (D-XLOG-C9-0's: insert, retract-and-reinsert a quarter,
retract a quarter) and `drain` (insert all, retract 90 %).

The three counts the coverage review asked to keep apart:
**carries** (an addition left ±255; `+255 +255 −255` carries once),
**batch overflows** (keys whose net batch delta is wide), and
**final escalations** (degrees wide at the end).

## Result (`cargo run --release --example triangle_pairing_baselines`)

Every batch exact (0 of 47,646 wrong), every final degree vector exact,
pairing unchanged in all 432 trials, regret 1.000.

| stream | batch | delta carries | batch overflow keys | degree carries | degree demotions | final wide |
|---|---|---|---|---|---|---|
| churn | 256 | 0 | 0 | 1,001 | 411 | 590 |
| churn | 4,096 | 87 | 87 | 934 | 344 | 590 |
| churn | 65,536 | 994 | 728 | 668 | 78 | 590 |
| drain | 256 | 0 | 0 | 790 | 763 | 27 |
| drain | 4,096 | 120 | 120 | 769 | 742 | 27 |
| drain | 65,536 | 1,167 | 856 | 464 | 437 | 27 |

- Batch overflow keys reproduce D-XLOG-C9-0's escape lanes exactly (87 at
  4,096, 728 at 65,536 on churn): the same quantity, computed independently.
- Delta carries exceed batch overflows where a batch delta passed ±255 and
  came back: churn 65,536 has 994 carries, 266 delta demotions, 728 overflows
  (994 − 266 = 728).
- Degrees themselves pass 255 regardless of batch size (hub values): 590 keys
  end wide on churn even at batch 256, where no batch delta overflows.

**Demotion is economy, not correctness.** With demotion disabled the arm is
still exact everywhere, but drain ends with 790 / 769 / 464 wide keys instead
of 27. Without hysteresis demotion also costs re-carries near the boundary:
churn at batch 256 has 1,001 degree carries with demotion and 790 without.

**Disable controls** (`tests/pm_ledger.rs`, each red): no escalation on an
absorb carry; no demotion; self-wide lanes not masked out of the vector sum;
an event carry ignored.

## Not measured

Time or memory against plain `i64` degrees; a demotion threshold below 255
(hysteresis); subtraction (P2).
