# D-DR-SELECT-K-1-INPLACE: the arm without a staged copy (2026-11-18)

**Status:** MEASURED, lab only. Adds the arm the D-DR-SELECT-K-1 lab did not
have. Basis: the review of #2032–#2037 posted on EXO-253, and EXO-332
(`identity[index].value` is read at the position, not staged).

`crates/dr-select-k-lab/src/inplace.rs`: `InPlace` binds a selection to its
source positions and copies no lane value. It keeps the position table, one
bit per source row, and the bounds of the run when the positions are one
ascending run. Folds read the source's own lanes:

- `plane-over-n`: mask-RISC over the N source rows, gated by the plane;
- `run-slice`: mask-RISC over the K rows of the run, a sub-slice of each lane;
- `positions`: a scalar loop through the position table (no admitted executor
  folds over a position list, plan §11).

Every fold is handed the source and refuses another version (`Stale`). The
staged `Bound` has no such check (a fold on a binding older than the source
answers from its copy).

Tests (`tests/integration/inplace.rs`, 4; the lab has 21): every in-place fold
equals the row-at-a-time oracle over 72 cells; a clustered filter selection is
a run and a scattered one is not; binding copies `N / 8` bytes and nothing per
selected row, and no fold moves the storage or byte counters; every fold and a
bind refuse a source that advanced. Disables, 6 of 6 red.

Measured (`--example inplace_report`, N = 1,000,000, release, one VM, medians
of 41, two runs in `.claude/probes/2026-11-18-dr-select-k-inplace/`; µs, run 1
/ run 2; filter selection, no residual, `count(b > 500)`):

| K | addresses | arm | bind | fold |
|---|---|---|---|---|
| 65,536 | clustered | staged | 184 / 178 | 4.9 / 4.8 |
| 65,536 | clustered | run-slice | 61 / 45 | 6.6 / 5.4 |
| 750,000 | clustered | staged | 11,203 / 10,368 | 79 / 94 |
| 750,000 | clustered | run-slice | 493 / 498 | 83 / 97 |
| 65,536 | random | staged | 696 / 651 | 5.0 / 5.0 |
| 65,536 | random | positions | 51 / 48 | 156 / 119 |
| 65,536 | random | plane-over-n | 51 / 48 | 179 / 176 |
| 750,000 | random | staged | 2,389 / 2,390 | 94 / 109 |
| 750,000 | random | plane-over-n | 916 / 919 | 172 / 178 |

Bind terms are once per version, reported and not gates. Two of them exceed a
millisecond (staged at K = 750,000).

Findings:

1. **A contiguous selection needs no copy.** The run-slice fold costs the same
   as the staged fold (within run-to-run spread) and its bind is 3 to 4 times
   cheaper at K = 65,536 and about 20 times cheaper at K = 750,000.
2. **On scattered addresses the staged copy is faster after about 5 folds**
   at K = 65,536 (bind 650 µs dearer, fold about 125 µs cheaper), and after
   about 20 at K = 750,000. This corrects finding 2 of the EXO-253 review,
   which said the payback had not been shown against an arm without staging:
   it is shown now, and it holds for scattered addresses.
3. So the copy is a cost question only for scattered selections. Whether a
   staged copy is admitted at all there is the zero-copy rule's question, not
   this lab's.

Open:
- The positions arm is a scalar loop and the plane arm folds N rows; a
  per-word sparse walk for the gated predicate (plan §11) is not measured.
- Residual predicates are not in this arm.
- No Lance cost; the source is the in-memory stand-in.
