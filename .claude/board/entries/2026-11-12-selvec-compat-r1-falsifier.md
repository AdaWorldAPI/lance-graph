# 2026-11-12 — D-SELVEC-COMPAT-0: R1's falsifier run; SelectionVector compatibility raised

**Kind:** one pre-registered falsifier run, plus a plan section recording a requested decision. No production code.
**Grade:** the falsifier result is MEASURED (`cargo run --release --example r1_falsifier` in
`crates/dr-select-k-lab`, twice; files in `.claude/probes/2026-11-12-r1-falsifier/`). The
compatibility requirement is PROPOSED.
**Plan:** `.claude/plans/dr-select-k-v1.md` §13; `.claude/plans/duckdb-to-v3-translation-matrix-v1.md` §8b.
**Linear (existing):** EXO-253.

- Matrix row R1 said: if `masked_sum_i32` loses to an index-list sum at any of
  s = 1/64, 1/256, 1/1024, the row moves from ELIMINATE to ADAPT. The mask
  kernel lost in all 12 cells in both runs (2.2x to 66x). A bit walk that
  skips zero words also lost in all 12 (1.1x to 16x). R1 is ADAPT.
- No sparse threshold was located: the mask already loses at 1/64.
- Consumption only. Building the selection is not in these cells.
- A requirement was raised that a DuckDB-compatible `SelectionVector` be
  representable at the API boundary while native execution stays free to use
  masks, order indexes or `ALL(K)`. Recorded as plan §13 with its rules, the
  decisions it touches (R4 and R7 under review; Quack R1 and mask-RISC A1
  unchanged in code) and a nine-item falsifier suite.

**Open:** whether Quack and mask-RISC admit an explicit selection operand;
where the sparse threshold lies; R4 and R7.
