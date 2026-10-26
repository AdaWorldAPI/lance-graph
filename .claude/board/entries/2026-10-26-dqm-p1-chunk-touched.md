# 2026-10-26 — D-DQM P1: chunk-touched predictor vs measured Lance 13 bytes — FAIL

**Status:** MEASURED (offline, over committed data). Verdict **FAIL**. Authorises nothing.
Pre-registration: `.claude/plans/duckdb-dr-quack-machine-room-reuse-v1.md` § A.6 (P1), pinned at commit
`62523c0e4` (2026-10-26 03:34 UTC, merged to `main` via #1788). This probe ran at 06:28 UTC the same day. The § A.6
text is byte-identical from `62523c0e4` to `main` at #1788's merge. Audit it with
`git show 62523c0e4:.claude/plans/duckdb-dr-quack-machine-room-reuse-v1.md`. The P1 bullet, verbatim:

> Measure the predicted ratio (non-zero 64-word groups / total groups, assuming 4,096-value chunks) against the
> measured masked/unmasked byte ratio per cell. Kill: |predicted − measured| > 0.10 in more than 10 % of cells,
> or rank correlation < 0.9; then test the 64 KiB-merge hypothesis once, no third tuning. Can fire: uniform 1 %
> ≈ 1.00, clustered 1 % ≈ 0.2. Can stay silent: masks offset by 2,048 rows (every run crosses a chunk boundary)
> must raise the clustered prediction.
Code and per-cell output: `.claude/probes/p1-chunk-touched-v1/{p1.py,results.csv}`.
Run: `python3 -I .claude/probes/p1-chunk-touched-v1/p1.py .` (about 2 s, standard library only).

**Question.** Does the fraction of 4,096-row groups a mask touches (non-zero 64-word groups / 16) predict
masked/unmasked Lance bytes?

## Inputs (read, not modified; no Lance run)

- H14c Lance CSV: `entries/2026-10-15-d-holo-h14c-thought-first.md` (last edit `da3f20672`). Generator:
  `crates/holo-lance-masked-io-probe/src/main.rs` (last edit `a79050ac5`, before the entry). Lance 13.0.0.
- H14d rows `2-4`: `entries/2026-10-16-d-holo-h14d-logic-mask-read.md` (`175f43742`), measured commit `802db7df`
  (= last edit of `crates/holo-h14d-probe/src/main.rs`). Lance 13.0.0.

## Checks before computing

- **Geometry and coordinates (from probe source).** Both probes write coordinates 0..65,535 in order, one batch,
  one fragment (H14c `max_rows_per_file = 1 << 20`, H14d `= N`), without stable row ids. So `_rowid` = row
  address = coordinate, and slot c of a regenerated mask is row offset c. Single version per dataset; the H14c
  map is version-checked in the probe, and H14d reads the dataset it just wrote.
- **Mask reconstruction.** The generators were transcribed into Python. All 136 regenerated masks match the probes'
  recorded counts (H14c `rows_emitted`; H14d `survivors` and `lance_rows_returned`). Disable runs that change the
  generator's input set (H14d key bit 16, H14c uniform base bit 16, H14c clustered base, H14d uniform threshold
  scale) produce 37, 11, 8 and 29 mismatches. Limit: the count check cannot see an XOR of low seed bits (it only
  permutes positions inside the input range). That permutation keeps every row in its 4,096-row group, so it
  cannot change a prediction either. Found because the first disable run (low-bit XOR) came back green.
- **Cells (fixed before computing).** H14c: `combined` vs `unmasked`, one cell per (pattern, density, k). Bytes
  were asserted identical across the three materialization styles. `independent` is excluded because it reads
  `coord` once per lane, a different projection from the baseline. H14d: one cell per (pattern, density, width, k),
  form `and`. Total 136 cells (40 + 96), all included.

## Results

| test | cells | error > 0.10 | Spearman | outcome |
|---|---|---|---|---|
| baseline, registered population (all cells) | 136 | 12 (8.8 %) | 0.988 | thresholds pass |
| baseline, cells with a selection (sensitivity) | 73 | 12 (16.4 %) | 0.930 | error threshold fires |
| baseline, H14c only | 40 | 7 (17.5 %) | 0.948 | error threshold fires |
| baseline, H14d only | 96 | 5 (5.2 %) | 0.994 | pass |

Controls (1 %, k = 1, full tile):

| fixture | pattern | selected | groups touched | predicted | measured | |
|---|---|---|---|---|---|---|
| H14c | uniform | 692 | 16/16 | 1.000 | 1.000 | ok |
| H14d | uniform | 692 | 16/16 | 1.000 | 0.999 | ok |
| H14c | clustered | 704 | 7/16 | 0.438 | 0.242 | **miss** |
| H14d | clustered | 640 | 7/16 | 0.438 | 0.191 | **miss** |

**What was registered and what was added.** Registered: the population (every cell), the two kill thresholds,
the can-fire expectations (uniform 1 % ≈ 1.00, clustered 1 % ≈ 0.2), the can-stay-silent shift, and one 64 KiB
follow-up. Added here, not registered: the "cells with a selection" and per-fixture rows of the results table
(sensitivity readings, not scored), and treating a missed can-fire expectation as failing. The registration lists
that expectation as a control, not as a kill condition. Without that step the registered kill thresholds pass.

**Verdict: FAIL**, on the registered can-fire control, not on the thresholds. The A.6 thresholds pass on the registered population only because 63 of 136 cells select no
row and are trivially exact (predicted 0, measured 0 bytes). The A.6 can-fire control fails: clustered 1 % is
predicted at 0.44 against 0.24 and 0.19 measured, a miss of 0.20 and 0.25. The miss exceeds 0.19, so any control
tolerance below that gives the same answer. On cells with a selection the error threshold fires, and the
predictor over-predicts in 59 of 73. Largest misses: H14c clustered 50 % k = 4 (predicted 1.00, measured 0.30);
H14c clustered 50 % k = 2 (1.00 vs 0.70); uniform 0.1 % k = 1 (46 rows: 1.00 vs 0.77 in H14c, 0.65 in H14d).

**Reading, not a tested claim.** Lance reads at a finer grain than 4,096 rows for these columns. A.3 already notes
mini-block chunks of at most 8 KiB, which holds about 512 values for H14c's 16-byte lanes and about 2,048 for
4-byte columns. Testing a smaller chunk would be a third tuning attempt, which A.6 forbids.

**The one registered follow-up (64 KiB range merge, reported separately): FAIL.** Registered as following a kill;
run here because the verdict is FAIL by the control, so it is reported, not used to decide. 16/136 (11.8 %) cells over 0.10
(21.9 % of cells with a selection), Spearman 0.988. The clustered controls get worse (0.56 and 0.63), because
merging only adds bytes and the baseline already over-predicts. Note: both probes wrote to a local temp
directory, where lance-io's merge block is `DEFAULT_LOCAL_BLOCK_SIZE` = 4 KiB, not the 64 KiB cloud default
(lance-io 13.0.0 `object_store.rs:87,98`). The hypothesis was tested as registered, not re-parameterised.

**Can-stay-silent control (reported, not scored).** A.6 says shifting a clustered mask by 2,048 rows must raise
its prediction, because every run then crosses a chunk boundary. These runs are 64 rows at 64-row alignment, so a
2,048 shift moves no run across a 4,096 boundary. Result: raised in 6, equal in 21, lowered in 2 of 29. The
control's premise does not hold for these fixtures, and no measured shifted read exists.

## Consequences

- No estimator of this form is proposed for use before a sealed read; no runtime or production code changed.
- A.6 forbids tuning, so the correct grain remains open: no smaller chunk size or per-column width was fitted
  here. A grain-correct predictor would need a new pre-registration, and would require reading Lance's actual
  chunk layout rather than assuming one.
- Unchanged: S-INGRESS, the joint coordinate witness, ALLOW and BLOCK stay separate operator decisions; DR64,
  `MailboxSoA`, `BatchWriter`, `CycleWriter`, the 512-byte node and MemWAL are untouched.

OPEN: the actual mini-block chunk boundaries for the H14c/H14d columns (not read from the written files); whether
a grain-correct predictor is worth a pre-registration at all, since a uniform mask already reads every chunk.
