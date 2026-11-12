# 2026-11-12 — D-DR-SELECT-K-1-CORRECTION: byte accounting and the tie rule in the selection lab

**Kind:** correction to a merged lab (#2035). Lab crate only.
**Grade:** MEASURED (`cargo test --tests` in `crates/dr-select-k-lab`: 17 passed;
`python3 .claude/probes/2026-11-12-dr-select-k-1/disable_runs.py`: 13 of 13 red;
`cargo run --release --example report`, twice).

- `bytes_copied` omitted the 4-byte-per-row position vector both producers
  copy. It is now counted; old report rows are short by exactly `4 × K`.
  Fold times did not move (median new/old 1.000 over 186 cells).
- The lab ordered ties by guid while `SealedOrder` orders them by frame row
  position. Both are now declared contracts (`Order::KeyThenRow`,
  `Order::KeyThenGuid`); a request is admitted only for the witnessed order.
- The report labels the executor of each cell (`fold_path`). The positions
  arm is a scalar loop; no admitted consumer folds over a position list.
- Original report files kept; corrected runs are `report-v2-run{1,2}.tsv`.

**Open:** whether a position-list consumer should exist at all is a decision,
recorded as a gap in `.claude/plans/dr-select-k-v1.md` §11.
