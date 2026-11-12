# D-DR-SELECT-K-1: which view of one bound selection is cheaper

Lab: `crates/dr-select-k-lab`. Plan: `.claude/plans/dr-select-k-v1.md` §4.
Run 2026-11-12 on one VM (Intel Xeon 2.10 GHz, 4 vCPU), release build,
single thread, N = 1,000,000, medians of 41 runs after 3 warm-ups. Two full
runs are committed (`report-run1.tsv`, `report-run2.tsv`); across the 186
timed cells the fold time of run 2 over run 1 has median 1.00, range
0.59 to 1.87. Not certified on dedicated hardware.

Command:
`cargo run --release --manifest-path crates/dr-select-k-lab/Cargo.toml --example report`

## What was checked

- 12 tests; every admissible view equals an independent row-at-a-time
  oracle, including the ordered guid sequence with ties broken by guid.
- The report has 216 cells: 186 timed, each checked against the oracle
  first, and 30 printed as not admissible (top-10 on a binding without a
  witnessed order).
- `disable_runs.py`: 10 guards broken one at a time, 10 red.
- After the bind the storage counter did not move in any cell
  (`storage_after_bind` = 0 in all 186 timed cells). Produce plus bind is exactly two
  storage accesses.

## Scope limits

- The source is an in-memory stand-in for storage. No Lance cost is
  measured, and which Lance 14 call returns an exact survivor set is still
  not verified (plan §7).
- `u128` stands in for `NodeGuid`.
- Residual density has two points (about 1% and about 50%), so the lab
  shows that a crossover exists, not where it is.
- The positions residual is built with a plain push loop, and the resident
  baseline builds its N-bit plane with a scalar loop and allocates its
  scratch per call. Those three terms are upper bounds.

## Fold time per view (µs, run 1, ordered selection)

Random addresses. Clustered addresses give the same fold times within the
run-to-run range, because folds read the staged K-local lanes.

| K | residual | count, positions | count, mask | sum, positions | sum, mask |
|---|---|---|---|---|---|
| 256 | about 1% | 0.034 | 0.130 | 0.033 | 0.085 |
| 256 | about 50% | 0.119 | 0.145 | 0.075 | 0.127 |
| 65,536 | about 1% | 0.48 | 5.26 | 0.28 | 2.15 |
| 65,536 | about 50% | 21.9 | 10.3 | 11.6 | 12.2 |
| 750,000 | about 1% | 5.28 | 121.7 | 5.83 | 57.1 |
| 750,000 | about 50% | 260.2 | 139.4 | 177.0 | 135.0 |

With no residual (`ALL(K)`): count 0.135 / 5.5 / 109.8 µs and sum
0.127 / 13.5 / 139.8 µs at K = 256 / 65,536 / 750,000.

Ordered top-10: 0.035 to 0.062 µs for every view at every K, when the
binding witnesses key order. Without the witness all three views are
refused; nothing sorts at query time.

## Once-per-version terms (µs, reported, not gates)

| K | addresses | produce (ordered) | bind and stage | to positions 1% / 50% | to mask |
|---|---|---|---|---|---|
| 256 | random | 0.16 | 1.1 | 0.18 / 0.46 | 0.11 |
| 65,536 | random | 7.4 | 676 | 43 / 351 | 15 |
| 65,536 | clustered | 7.0 | 182 | 42 / 342 | 15 |
| 750,000 | random | 230 | 8,241 | 683 / 4,253 | 181 |

Bind is almost entirely staging: 28 bytes per row copied out of the source.

## Against the resident fold over N (µs, random addresses, no residual)

| K | fold over K | fold over N | bind | bind paid back after |
|---|---|---|---|---|
| 256 | 0.135 | 23.6 | 1.1 | the first fold |
| 65,536 | 5.5 | 201 | 676 | 4 folds |
| 750,000 (75% control) | 110 | 199 | 8,241 | about 93 folds |

For the 75% control the totals at R = 1 are 8,583 against 3,660 µs, and at
R = 100 they are 19,454 against 23,400 µs for count and 22,448 against
22,198 µs for sum. The control behaves as a negative control: binding
three quarters of the table does not pay.

With clustered addresses the resident fold over N is much cheaper (30 to
37 µs at K = 65,536) because the N-bit plane is one contiguous run, and
the bind is paid back after about 7 folds.

## Findings

1. **More than one view wins.** Sparse positions win when the residual is
   sparse (9 to 23 times faster than the mask at about 1%, K ≥ 65,536). The
   mask wins or ties when it is dense (count about 2 times faster at about
   50%; sum equal within noise). At K = 256 the mask-RISC call has a floor
   near 0.1 µs, so positions win or tie at both densities.
2. **The mask is the cheaper form to build**: 15 µs at K = 65,536 against
   43 to 351 µs for positions. For a single use of a dense residual the
   mask also wins on the total.
3. **Order is a property of the binding, not of the view.** With a
   witnessed key order every view answers top-10 in under 0.1 µs; without
   it none may.
4. **mask-RISC has no implicit `ALL(K)` for a sum.** `MaskedSumI32` takes a
   mask operand. Count needs no plane (`under: None`). The lab holds one
   constant plane of K/8 bytes per binding for the sum.
5. **Binding K pays quickly when K is a small share of N**, and not when it
   is most of N.

## What this does and does not support

It supports a deterministic handoff rule keyed on residual density and the
next operation (plan §3: "it becomes finer only if the lab shows different
views winning"). It does not locate the density threshold, and it says
nothing about a storage-backed source. Gates 1 and 2 of plan §5 are
untouched.

## Correction, 2026-11-12 (after #2035 merged)

Three things in the lab changed. The original files `report-run1.tsv` and
`report-run2.tsv` are kept as measured; the corrected runs are
`report-v2-run1.tsv` and `report-v2-run2.tsv` (same VM, same build flags).

1. **Byte accounting.** `select_filter` and `select_ordered` each copy a
   position vector of 4 bytes per selected row, and the first version did not
   count it. `bytes_copied` is now 32 per selected row: 4 for the position
   vector, 28 for the staged lanes. Every row of the old files is short by
   exactly `4 × K` (1,024 / 262,144 / 3,000,000 bytes at K = 256 / 65,536 /
   750,000; checked on all 150 rows that carry the column). Retained bytes,
   storage accesses and operand reads were already separate columns and did
   not change.
2. **Fold latency did not change.** Over the 186 timed cells present in both
   versions, the median of new fold time over old fold time is 1.000. Single
   cells differ by up to about 2x in either direction, which is the same
   spread as between the two v2 runs (0.40 to 1.91), so no cell is evidence
   of a change.
3. **Tie-breaking.** The first version ordered by key, then guid, and called
   that the order. `SealedOrder` already defines key, then frame-relative row
   position. The lab now carries both as separately witnessed contracts,
   `Order::KeyThenRow` (the existing rule) and `Order::KeyThenGuid`. A top-T
   request names its order and is admitted only on a binding witnessed for
   exactly that order; a view change does not change the result. The v2
   files add 30 `ordered_guid` rows at K = 65,536 (246 rows in total).

The v2 files also add a `fold_path` column. The positions arm of count and
sum is a plain Rust loop over the local row list (`scalar-loop`); the mask
and `ALL(K)` arms go through mask-RISC (`mask-risc`); top-T is a plain walk
on every view (`scalar-walk`). The positions-against-mask numbers above
therefore compare two different executors as well as two representations.
No admitted consumer in mask-RISC, Quack or ndarray folds over a position
list (mask-RISC `lib.rs` A1 and Quack `lib.rs` R1 say so in terms), so the
scalar arm stays a lab measurement and is recorded as a gap in
`.claude/plans/dr-select-k-v1.md` §11, not as an executor.

Checks at this state: 17 tests; `disable_runs.py` 13 guards, 13 red.
