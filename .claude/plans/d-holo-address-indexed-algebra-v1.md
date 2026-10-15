# D-HOLO — address-indexed algebra: falsifiers H1, H5, H6 (v1)

**Status:** MEASURED 2026-10-14 (H1, H5, H6, H6b, H6c); H8, H9, H11, H12, H14
PRE-REGISTERED, not run. The pre-registration below is unchanged; results are in § Results
(2026-10-14), the next falsifiers in § Next falsifiers.

## Overview

The D-HOLO-0 research verdict (session scratchpad, 2026-10-14) classified the
address-indexed algebra (`query[$address]`) as a DISTINCT COMPOSITION with one
CANDIDATE CONTRIBUTION:

> bounded parallel evaluation of coordinate-aligned relational constraints,
> independent of the number of selected thought branches up to the physical mask
> width, with no query-time intermediate relation materialization.

It named three classes of computation, coordinate-native, coordinate-crossing and
semantic, and three kinds of materialization:
- representation maintenance (allowed, charged at seal and update);
- intermediate relational materialization (forbidden except at a named terminal);
- semantic reconciliation.

This plan pre-registers the three falsifiers the operator asked to run first.

DECISION (operator, 2026-10-14): run H1, H5, H6. SCOPE: one dense class tile,
in-process baselines only. BASIS: the D-HOLO-0 verdict. REVISIT WHEN: a
falsifier fails, or an external baseline (DuckDB, differential dataflow, TACO)
is run.

## Checklist

- [x] D-HOLO-0b: this pre-registration and its board rows, committed before any probe exists
- [x] D-HOLO-H1: K-branch probe, oracle tests, disable runs, v4 run, result
- [x] D-HOLO-H5: coordinate-native vs coordinate-crossing probe, same
- [x] D-HOLO-H6: closure of address restrictions probe, same
- [x] Results section, board entry, status rows
- [x] D-HOLO-H6b: normal forms on the Context × Operation tile (with the connectedness column)
- [x] D-HOLO-H6c: group-count switch, the 2–60 group crossover, declared construction
- [ ] D-HOLO-H8: semantic diversity does not add instruction families
- [ ] D-HOLO-H9: relational emptiness without a fold
- [ ] D-HOLO-H11: stacked logic (pre-registered after H6b; runs after it)
- [ ] D-HOLO-H12: semantic map of inference functions across NARS systems
- [ ] D-HOLO-H14: full-row writes vs lane deltas that stay in their row (pre-registered 2026-10-15); amended 2026-10-15: arm C is the inverted fold and a candidate

## Accounting rules (apply to every falsifier)

1. **Every resident mask a program reads is charged.** Its construction cost (ns
   per row, bytes) is reported beside the query cost, and the decision uses the
   total = construction amortized over the stated number of queries + query. A
   program that only wins because an input mask was assembled outside the timed
   region has not won.
2. **Work avoided is measured, not inferred.** Each arm reports the coordinates
   it examined (64-bit words read, rows evaluated), next to its time. Cost should
   follow admitted coordinates, not the population.
3. **A `Range` over row indexes requires a layout-order witness.** On the dense
   tile the order holds by construction (row = Morton HEEL code). This is a
   correctness obligation, so H5 includes one adversarial run on a permuted tile,
   where `Range` must disagree with the sweep and the oracle.
4. **Value access is mathematical, not semantic.** Reading a value lane (`v < t`,
   a foreign key through `Gather`) is part of the substrate; only interpretation
   is optional.
5. Every arm is checked against a row-at-a-time oracle on every cell. Timings:
   v4 only, the tier printed by the program, median of 11 runs, ns per query and
   ns per element. Allocations per query come from a counting allocator; the
   mask-risc arms must allocate 0 after scratch setup.

## Common fixture

- One dense class tile: 65,536 rows; row r has Morton HEEL code r
  (x = even bits, y = odd bits); key lane u64 = (classid << 32) | (r << 16).
- Generated deterministically by the probe from a fixed seed; no fixture files.
- mask-risc arms run through `exec::execute_into` on the real executor.
- Out-of-contract arms are plain Rust in the same probe and are labelled as such.

## H1 — K-branch evaluation of a coordinate-aligned Boolean constraint

**Data.**
- Value lane v: i32, uniform in [0, 1000).
- Base predicate Q = v < t, with t chosen for selectivity s in {1 %, 10 %, 50 %}.
- Aperture P: the whole tile, or one HEEL nibble prefix (4,096 rows).
- K in {1, 8, 64} branches. Branch k covers each row independently with
  probability 10 %. The same coverage is held as per-thought planes M_k
  (1,024 words each) and as per-row written-by words w.

**Query.** For each k in the selection T: count(P ∧ Q ∧ M_k). The answer is K
counts.

**Arms.**
- **A, mask-risc per branch.** K programs, each `Range` → `LtI32` under it →
  `And` with `Plane(k)` → `Count`. P ∧ Q is recomputed in every program.
- **B, mask-risc, gate once.** One program keeps G = P ∧ Q into a caller mask
  (`Out::Mask`), then K programs `And(Plane(G), Plane(k))` → `Count`. G is the
  cross-program population mask from the verdict's materialization audit; its
  cost is charged.
- **C, bit-parallel over row words (outside the contract).** One pass over the
  rows set in G, adding w[r] & T into 64 position counters. G comes from B's
  first program, charged.
- **D, snapshot baseline.** Per branch, a materialized list of the branch's row
  ids. The count evaluates P ∧ Q per listed row; K passes. The build cost of the
  lists is reported and charged.

**Charged inputs.** Building M_k (per-thought planes) and w (per-row words) from
the same coverage, ns per row, reported once.

**Decision.** Reference cells: s = 10 %, both apertures.
- The candidate survives at a cell if C's total cost at K = 64 is at most 2× its
  cost at K = 1 and at most 0.5× the best of A, B, D at K = 64.
- It fails if C grows linearly in K, or if A, B or D comes within 2× of C at
  K = 64.

**Not run in this pass:** DuckDB on materialized snapshots, differential dataflow
with a 64-element partial order, TACO. D is the in-process snapshot stand-in.

## H5 — coordinate-native vs coordinate-crossing (F2c)

**Q1: `identity[$prefix].count()`.**
- Prefix of p in {1, 2, 3} HEEL nibbles (4,096 / 256 / 16 rows).
- Occupancy plane O, density in {100 %, 30 %}.
- Arms:
  - **R:** `Range` under `Plane(O)` → `Count`, fused, reading only O's words in
    the range.
  - **S:** `MatchU64` sweep on the key lane under `Plane(O)` → `Count`.
  - **closed form:** 16^(4-p) when occupancy is 100 %, no read.
- Adversarial: R on a tile whose key lane is permuted must disagree with S and
  the oracle.

**Q2: `identity[$node].thoughts($all).count()`.** 64 thoughts at 10 % coverage.
- **W:** popcount(w[row]), one register operation.
- **P:** the contract form, 64 mask-risc programs `Range{row, row+1}` under
  `Plane(t)` → `Count`, summed.
- **B:** 64 bit reads from the per-thought planes, plain Rust.

**Q3: reverse `memberOf`.**
- Each identity has m = 4 membership lanes (u32 group id, 0 = none), over 1,024
  groups.
- Two distributions: uniform, and skewed (Zipf exponent 1).
- Queries, over 256 sampled groups:
  - Q3a `reverse(g).count()`;
  - Q3b `reverse(g) ∧ prefix(p = 1)`, then count;
  - Q3c `reverse(g) ∧ (v < t)`, then count. The inverse orders neither the
    value lane nor this predicate.
- Arms:
  - **S, scan in mask-risc:** 4 `EqU32` predicates ORed, under `Range` for Q3b,
    `And LtI32` for Q3c, then `Count`.
  - **I, CSR inverse built at seal:** offsets per group plus sorted member rows.
    Q3a is an offset difference, Q3b a binary search, Q3c per-member evaluation
    of `v < t`.
  - **IM, inverse as per-group mask planes (1,024 × 8 KiB):** mask-risc `And` /
    `Count`.
- Maintenance metrics:
  - build time (ns per membership edge) and bytes held, for I and IM;
  - cost of inserting one edge (CSR shift vs one bit set);
  - break-even query count N = build cost ÷ (scan cost − inverse cost), per query
    shape.

**Decision.**
- Coordinate-native advantage confirmed if:
  - Q1-R reads exactly the occupancy words in the prefix;
  - Q1-R is at least 5× faster than Q1-S at every cell;
  - Q2-W is under 5 ns.
- Coordinate-crossing classification confirmed if:
  - Q3-S's cost is proportional to the aperture's rows;
  - every arm that beats the scan by more than 5× needs a seal-time structure.
- "A maintained inverse is always better" is refuted if, after build cost:
  - the break-even N exceeds 1,000 queries for a shape; or
  - Q3c is slower on I than on S.
- The adversarial run must disagree; if it agrees, the fixture cannot detect an
  order violation and the run is invalid.

## H6 — closure of address restrictions

On the dense tile (row = Morton code), restrictions are:
- prefix intervals;
- Morton boxes [x0, x1) × [y0, y1), decomposed into maximal aligned quadtree
  intervals;
- their unions and complements.

**Formulas.**
- F1 = B1 ∩ P
- F2 = B1 ∪ B2
- F3 = ¬B1 ∩ P
- F4 = (B1 ∪ B2) ∩ ¬B3

**Box shapes.**
- aligned squares of side 16, 64 and 256 (one interval each);
- unaligned squares;
- thin boxes 256 × 1 and 1 × 256 (many intervals).

**Arms.**
- **S, symbolic:** interval sets as sorted (lo, hi) vectors, ∩ / ∪ / ¬ by merge,
  count = Σ length. No row is touched.
- **M, masks in mask-risc:** each leaf as `Range` ops ORed into a slot, then the
  formula's `And` / `Or` / `Not`, then `Count`, in one program per formula.
- **V, value sweep:** x and y as u32 lanes, `GeI32` / `LtI32` per box bound,
  `And` / `Or` / `Not` → `Count`. No intervals at all.

**Decision.**
- Closure holds if:
  - every S result equals the oracle and is again an interval set;
  - at a fixed interval count, S's cost stays within 2× as box area grows 256×
    (aligned side 16 → 256);
  - S's cost grows with the interval count, not the population.
- The interval count at which M or V becomes cheaper than S is reported. It is
  not a pass/fail criterion.

## Amendment before any run (2026-10-14)

H1 and H5 name no query count for amortizing construction. Fixed here, before
any probe exists: construction is amortized over N = 1 and N = 100 queries, and
each decision is stated at both. H5's break-even N stays as defined.

## Boundaries

- No change to mask-risc, Quack, the contract or ndarray. The probes are examples.
- No new opcode or terminal. Arm C (H1) and the inverse arms (H5) are labelled
  out-of-contract, because no existing terminal expresses them.
- No external dependency.

## Results (2026-10-14)

All runs: tier printed by the program, `v4 (avx512f)`, 4-core VM, release,
debug 0, median of 11. Every arm matched the row oracle in every cell (asserted
by the probes and by their `cargo test` arms). Every mask-risc arm allocated 0
bytes per query after scratch setup.

Probes and disable runs:
- `examples/holo_k_branch_probe.rs` (H1). Five disables, all red: C without its
  gate, A without both extent and `Range`, D without the aperture lower bound,
  B reading the wrong plane, the gate without the value predicate. A without
  only the extent stays green, because `Range` alone enforces the aperture.
- `examples/holo_coord_probe.rs` (H5). Four disables, all red: the sweep without
  HEEL care bits, CSR without the aperture lower bound, the scan with 3 of 4
  lanes, the permuted test without its permutation.
- `examples/holo_interval_probe.rs` (H6). Six disables, all red: complement
  dropping the trailing interval, intersection always advancing one side, box
  children in the wrong Morton order, `push` never merging adjacent intervals,
  V without the y0 bound, M emitting only the first interval.

### H1 — K-branch evaluation: MIXED; depends on the stored orientation

Construction: thought planes 29.27 ns/row (512 KiB, 1.92 ms per tile);
snapshot lists 55.16 ns/row (1,636 KiB, 3.62 ms). The probe takes per-row words
w as the stored coverage, so A and B are charged the plane build and C is not.

s = 10 %, ns per query:

| aperture | K | A | B | C | D |
|---|---|---|---|---|---|
| 65,536 | 1 | 23,025 | 23,630 | 73,415 | 9,694 |
| 65,536 | 64 | 1,528,007 | 51,971 | 71,729 | 587,145 |
| 4,096 | 1 | 1,708 | 2,022 | 4,783 | 699 |
| 4,096 | 64 | 103,100 | 11,814 | 4,814 | 42,863 |

- C is flat in K: C(64)/C(1) is 0.84–1.47 over all 18 cells. The K-independence
  half of the candidate holds.
- Rule applied at the reference cells (C ≤ 2× its K = 1 cost, and ≤ 0.5× the
  best of A, B, D at K = 64, construction amortized):

| aperture | N | best other (total) | C | verdict |
|---|---|---|---|---|
| 65,536 | 1 | B 1,970 µs | 72 µs | PASS |
| 65,536 | 100 | B 71.2 µs | 71.7 µs | FAIL (B within 2×, in fact faster) |
| 4,096 | 1 | B 1,930 µs | 4.8 µs | PASS |
| 4,096 | 100 | B 31.0 µs | 4.8 µs | PASS |

- The verdict depends on which orientation is stored. If the planes M_k are the
  stored form, C must build w instead (same pass shape), and B wins every
  reference cell at N = 100 and at N = 1. The candidate's advantage is
  therefore not a property of the algebra; it is a property of storing coverage
  by row.
- B is already sub-linear in K: the gate dominates, and each further branch
  costs about 0.45 µs (one AND + popcount over 1,024 words, about 0.44 ns per
  word). C pays about 11 ns per gated row in a scalar 64-counter loop, so it
  loses at high gate density (s = 50 %, full tile: C 252 µs vs B 52 µs).
- Not measured: a vectorized positional popcount for C; external baselines
  (DuckDB, differential dataflow, TACO).
- Run-to-run noise: a second run (after a lint-only change to C's loop) moved
  absolute times by up to about 40 % on this 4-core VM (full tile, K = 64: B
  52.0 → 36.4 µs, C 71.7 → 60.7 µs; 4,096 rows, K = 64: B 11.8 → 6.6 µs, C 4.8 →
  3.6 µs). Both verdicts are the same in both runs (full tile N = 100: B 55.6 vs C
  60.7 µs, FAIL; 4,096 rows N = 100: B 25.8 vs C 3.6 µs, PASS). Differences under
  2× in this table are not evidence.

### H5 — coordinate-native vs coordinate-crossing: CONFIRMED, one refutation

Q1 `identity[$prefix].count()`:

| occupancy | p | R ns | S ns | R words read | S rows read | R / S |
|---|---|---|---|---|---|---|
| 100 % | 1 | 154.8 | 62,076 | 64 | 65,536 | 401× |
| 100 % | 2 | 124.1 | 61,628 | 4 | 65,536 | 497× |
| 100 % | 3 | 118.5 | 59,642 | 1 | 65,536 | 503× |
| 30 % | 1 | 125.0 | 36,497 | 64 | 65,536 | 292× |
| 30 % | 2 | 65.3 | 35,267 | 4 | 65,536 | 540× |
| 30 % | 3 | 61.5 | 35,248 | 1 | 65,536 | 573× |

- R reads exactly the occupancy words of the prefix; at p = 3 the 60–120 ns is
  call overhead, not work.
- Adversarial permuted key lane: R 1,212, sweep 1,214, oracle 1,214. R
  disagrees, so the fixture detects an order violation and the run is valid.
  The layout-order witness is a correctness obligation, as registered.
- Q2 per identity: W (row word popcount) 1.00 ns, B (64 plane bit reads)
  55.08 ns, P (64 mask-risc programs) 4,841 ns. W under 5 ns: PASS. The 1.00 ns
  is at the timer floor.
- Coordinate-native advantage: CONFIRMED.

Q3 reverse `memberOf` (1,024 groups, 4 lanes):

| dist | query | S ns | I (CSR) ns | IM (planes) ns | break-even N CSR / planes |
|---|---|---|---|---|---|
| uniform | Q3a count | 59,451 | 42.7 | 393 | 36 / 115 |
| uniform | Q3b ∧ prefix | 3,971 | 23.6 | 84 | 546 / 1,740 |
| uniform | Q3c ∧ v < t | 81,220 | 234.0 | 6,980 | 27 / 91 |
| zipf | Q3a count | 57,909 | 22.8 | 500 | 34 / 73 |
| zipf | Q3b ∧ prefix | 3,861 | 18.5 | 101 | 509 / 1,114 |
| zipf | Q3c ∧ v < t | 77,398 | 465.7 | 5,388 | 25 / 58 |

Maintenance: CSR builds at 10.25–10.95 ns/edge (749–772 KiB) and costs
6,386–13,225 ns per inserted edge; the plane inverse builds at 21.95–34.40
ns/edge (8,200 KiB) and costs 1.8–2.2 ns per inserted edge.

- The scan's cost follows the aperture: Q3a 65,536 rows vs Q3b 4,096 rows, time
  ratio 15.0 for a row ratio of 16.
- Every arm beating the scan by more than 5× is a seal-time structure. This
  holds only over the arms tried; the probe contains no structure-free
  alternative to S, so it is a non-finding rather than a proof.
- Coordinate-crossing classification: CONFIRMED in this sense.
- "A maintained inverse is always better": REFUTED for the plane inverse on
  Q3b (break-even 1,740 and 1,114 queries, over the 1,000 bound). Not refuted
  for CSR on these shapes (worst 546). Q3c is not slower on I than on S.

### H6 — closure of address restrictions: HOLDS

Leaves are three boxes and one prefix; `intervals_in` counts their intervals.
S_eval evaluates the formula over decomposed leaves; S_total also decomposes the
boxes. M and V run prepared programs.

| shape | intervals in | S_eval ns | S_total ns | M ns | V ns |
|---|---|---|---|---|---|
| aligned 16 | 4 | 45–81 | 264–333 | 722–1,450 | 27,662–104,522 |
| aligned 64 | 4 | 47–79 | 190–240 | 678–1,223 | 31,399–87,844 |
| aligned 256 | 4 | 46–84 | 131–196 | 816–1,283 | 49,036–134,829 |
| unaligned | 269 | 722–2,872 | 7,187–11,613 | 77,189–117,139 | 48,019–93,460 |
| thin 256×1 | 321 | 125–2,259 | 9,814–21,487 | 52,027–154,810 | 53,188–170,007 |
| thin 1×256 | 641 | 105–7,440 | 9,117–25,420 | 92,573–384,066 | 29,252–140,791 |

(Ranges are over F1–F4; per-cell values in the probe output.)

- Every S result equals the oracle and is again a sorted, disjoint,
  non-adjacent interval set (asserted on every cell).
- At 4 intervals, S_eval is flat as box area grows 256× (side 16 → 256): 45–81
  vs 46–84 ns. PASS.
- S grows with the interval count: about 7–20 ns per input interval for F4.
  The population was not varied; S reads no row, so its independence of the
  population is structural, not measured.
- Crossover: none against S. S was the cheapest arm in all 24 cells, decomposition
  included. Between the two row-level arms, M (Range per interval) beats V
  (value sweep) at few intervals (aligned: about 1 µs vs 28–135 µs), and V beats
  M in 8 of the 12 many-interval cells. M costs about 0.24–0.30 µs per op,
  because each `Range` writes the whole tile; V costs about 7 µs per
  predicate over 65,536 rows.
- Two F3 cells (aligned 64 and 256) and one F1 cell (thin 1×256) answer 0; they
  exercise only the empty-result path.
- Prior art: quack's `fold_ranges` already intersects ranges on one column
  symbolically before lowering, and `photo_probe`'s `support` takes the hull
  for OR and gives up on NOT. S is the exact closure under ∩, ∪ and ¬ of that
  same idea.

### What this changes for the D-HOLO-0 candidate

- Kept: coordinate-native operations (prefix count, row perspective) cost
  what the admitted coordinates cost, by two to three orders of magnitude over
  a sweep, given a layout-order witness.
- Kept: address restrictions are closed as interval sets and can be answered
  symbolically before any row is read.
- Narrowed: K-branch independence exists (C is flat in K), but its advantage
  over gating once and ANDing per branch holds only when coverage is stored by
  row and the gated population is small.
- Narrowed: a maintained inverse is a cost, not a default; the per-group plane
  inverse does not pay back on aperture-restricted queries within 1,000 queries.

### Open

- Whether coverage is stored by row or by thought. H1's verdict turns on it.
- A vectorized positional popcount for C (not tested).
- One `Range`-set write per program instead of one full-tile write per
  interval. Not built: no new op was added for this pass.
- External baselines (DuckDB, differential dataflow, TACO).

## Results, corrected after review (2026-10-14)

The Codex review of #1550 found two accounting errors in the H1 and H5 probes.
Both are fixed in `30fb38dd`; the results above are left as first measured.

- **Lowering recognised inside the timed path.** Every timed call ran
  `Program::compile()`, so arms A and B paid K recognitions per query, and H5's
  Q3 also rebuilt its scan program for every group. The probes now store
  `CompiledProgram` and time execution only.
- **H5 charged no input construction**, against accounting rule 1. Q1 now builds
  its occupancy plane from a presence lane and charges it to both arms; Q2
  measures both coverage orientations. H1 now also measures row words built from
  planes.

Re-run, v4 (avx512f), same fixtures.

### H1: unchanged verdict, B faster

Construction: planes from row words 20.56 ns/row (1.35 ms per tile); row words
from planes 17.17 ns/row (1.13 ms); snapshot lists 37.67 ns/row.

s = 10 %, K = 64: A 924,205, B 33,762, C 54,862, D 394,664 ns. Without the
repeated `compile()`, B dropped from 51,971 to 33,762 ns; each further branch
now costs about 0.31 µs.

| stored form | aperture | N | B total | C total | verdict |
|---|---|---|---|---|---|
| row words | 65,536 | 1 | 1,381 µs | 55 µs | PASS |
| row words | 65,536 | 100 | 47.2 µs | 54.9 µs | FAIL |
| row words | 4,096 | 1 | 1,354 µs | 3.6 µs | PASS |
| row words | 4,096 | 100 | 20.2 µs | 3.6 µs | PASS |
| planes | 65,536 | 100 | 33.8 µs | 66.1 µs | FAIL |
| planes | 4,096 | 100 | 6.7 µs | 14.8 µs | FAIL |

C stays flat in K (0.58–1.08 across all cells). The verdict still turns on
which orientation is stored.

### H5: Q1 fails at N = 1, passes at N = 100

Building the occupancy plane costs 109–113 µs (1.66–1.73 ns/row). Both arms
read it, so both are charged:

| occupancy | p | R total, N = 1 | S total, N = 1 | ratio | R total, N = 100 | S total, N = 100 | ratio |
|---|---|---|---|---|---|---|---|
| 100 % | 1 | 109.1 µs | 144.3 µs | 1.32× | 1.17 µs | 36.4 µs | 31× |
| 100 % | 2 | 109.1 µs | 150.3 µs | 1.38× | 1.21 µs | 42.4 µs | 35× |
| 100 % | 3 | 109.1 µs | 150.3 µs | 1.38× | 1.15 µs | 42.3 µs | 37× |
| 30 % | 1 | 113.2 µs | 152.8 µs | 1.35× | 1.22 µs | 40.8 µs | 33× |
| 30 % | 2 | 113.2 µs | 151.4 µs | 1.34× | 1.19 µs | 39.4 µs | 33× |
| 30 % | 3 | 113.2 µs | 151.9 µs | 1.34× | 1.20 µs | 39.9 µs | 33× |

- At N = 1 the ≥ 5× rule FAILS in every cell: building the plane is the query.
  At N = 100 it PASSES in every cell. The coordinate-native advantage is real
  only once the occupancy plane is maintained, not rebuilt per query.
- At 100 % occupancy neither arm needs the plane (the closed form answers from
  `16^(4-p)`). The probe charges it anyway, because it reads it.
- The adversarial permuted run still disagrees (Range 1,212, sweep and oracle
  1,214).

Q2 per identity:

| stored form | N | W | B | P |
|---|---|---|---|---|
| row words | 1 | 1.0 ns | 1.18 ms | 1.18 ms |
| row words | 100 | 1.0 ns | 11.9 µs | 15.2 µs |
| planes | 1 | 1.01 ms | 54.3 ns | 3.38 µs |
| planes | 100 | 10.1 µs | 54.3 ns | 3.38 µs |

The "W under 5 ns" rule passes only when row words are the stored form, the same
dependence as H1.

Q3, after compiling the scan programs once:

| dist | query | S ns | I ns | IM ns | break-even N CSR / planes |
|---|---|---|---|---|---|
| uniform | Q3a count | 99,578 | 42.9 | 469 | 22 / 71 |
| uniform | Q3b ∧ prefix | 6,636 | 23.4 | 50 | 339 / 1,073 |
| uniform | Q3c ∧ v < t | 101,843 | 243.2 | 7,514 | 22 / 75 |
| zipf | Q3a count | 62,141 | 21.0 | 361 | 56 / 126 |
| zipf | Q3b ∧ prefix | 3,621 | 17.7 | 52 | 958 / 2,175 |
| zipf | Q3c ∧ v < t | 100,599 | 355.9 | 5,416 | 34 / 82 |

- The plane inverse is still refuted on Q3b (break-even 1,073 and 2,175, over
  1,000). CSR now comes close (958 on zipf Q3b).
- This run was noisy: the uniform scan took 99.6 µs against 59.5 µs before, and
  the CSR insert 40.5 µs against 13.2 µs. The scan's cost still follows the
  aperture (time ratio 15.0 and 17.2 for a row ratio of 16).

### Corrected verdicts

| falsifier | first run | after review |
|---|---|---|
| H1 | MIXED, depends on stored orientation | unchanged |
| H5 Q1 (coordinate-native) | CONFIRMED | CONFIRMED at N = 100; FAILS at N = 1 |
| H5 Q2 | PASS | PASS only with row words stored |
| H5 Q3 (crossing; plane inverse refuted on Q3b) | as before | unchanged |
| H6 | HOLDS | not affected (it already used `CompiledProgram`) |

## Next falsifiers: H6b, H8, H9 (PRE-REGISTERED 2026-10-14, NOT RUN)

DECISION (operator, 2026-10-14): H6b approved as a probe; the terminal
representation selector is approved as a concept and held until H6b reports.
SCOPE: examples only, no new opcode or executor, no production selector.
BASIS: H6 above. REVISIT WHEN: H6b reports.

**Research objective.** Whether symbolic normalization lowers the total cost of
composing a solution (construction, normalization, reuse across queries, and
the terminal), not whether individual Boolean kernels get faster.

Separate from D-ORACLE-0 (#1549, epistemic oracle fidelity). That work asks
whether an answer is faithful to its oracle; this one asks which representation
computes it most cheaply. Cross-referenced, not combined.

### H6b — symbolic normal forms on the Context × Operation tile

**Universe.** The 256 × 256 tile read as Context × Operation. This is one
contextual reading of the existing `u8:u8` rail, not a new ISA schema: what a
context and an operation mean stays bound by Class / Schema / Datalog, and the
256 operations are references to (possibly composed) methods, not hardware
opcodes.

**Representations.**
- **P, canonical product form.** Contexts grouped by identical operation mask:
  a list of (context mask, operation mask), each 256 bits.
- **I, normalized Morton interval unions** (as in H6).
- **D, dense mask:** 8 KiB, operated on with existing ndarray SIMD.
- **H, hybrid:** P while it is smaller than D, else D.
- **O, row oracle** on every cell.

**Correction to the earlier claim.** P is NOT bounded by D's footprint.
256 distinct operation masks, each paired with a context mask, take 256 × 64 B
= 16 KiB, twice the dense 8 KiB. Canonicalization and comparison also cost
time. H6b measures this; neither P nor H is assumed to be optimal.

**Patterns.** Row-oriented (one context, many operations), column-oriented,
rectangles, prefixes, diagonal, checkerboard, random at 1 % and 50 %, sparse,
dense, complementary pairs, disjoint pairs, and high fragmentation (one
distinct mask per context).

**Measured per representation and pattern.** Construction; normalization;
∩, ∪, ¬; emptiness; equality; bytes held; total cost amortized over N = 1, 10,
100 and 1,000 queries. Tier printed, v4 only, median of 11, ns per operation.

**Decision.**
- P (or H) is worth a selector if, on the structured patterns (row, column,
  rectangle, prefix), its total cost at N = 100 is at most 0.5× D's.
- H is safe as a default only if, on the adversarial patterns (checkerboard,
  random 50 %, high fragmentation), it is at most 1.25× D's total cost.
- Differences under 2× are reported as undecided: H1 measured about 40 %
  run-to-run noise on this VM.
- Thresholds (group count, interval count, density) where each representation
  stops being the cheapest are reported, with every counterexample.

**Amendment before any run (2026-10-14): connectedness as a measured column.**
DECISION (operator, 2026-10-14). BASIS: Haspelmath 2003, "The Geometry of
Grammatical Meaning": a gram covers a connected region of a semantic map; a
1-D map makes every region an interval (Fig. 8.19), an implicational
hierarchy makes it a prefix (Fig. 8.20), and the map can have cycles
(Fig. 8.12) and non-connected "doughnuts" (Turkish present, p. 236).
- For every pattern, also report its 4-connected component count on the
  256 × 256 grid and its Morton interval count.
- Add one fixture: a doughnut (a filled rectangle with an interior hole).
- Report whether component count predicts the cheapest representation better
  than density does (rank agreement of each predictor with the winner, over
  all patterns).
- Connectedness is a selector input to be tested, never a constraint: no
  representation may assume it, and the doughnut must still match the oracle.

### H8 — semantic diversity does not add instruction families

- One fixed formula (a restriction ∧ a forward hop ∧ a negation, then count),
  written in three unrelated domains through the existing frontends (Quack,
  R2IL): dir-sim `memberOf`, thought coverage, planet → location.
- Compared, as three separate dimensions:
  - the canonical logical algebra (the normalized formula, before any physical
    choice);
  - the physical realization (op families, value families, terminals, carriers);
  - semantic leakage (a meaning that reaches the algebra or the physical layer).
- Value access is mathematical, not semantic: reading a lane (`v < t`, a key
  through `Gather`) is substrate; only interpretation is optional.
- Accounting: the work avoided through gating (coordinates examined), and the
  construction of every resident input mask, charged.
- Passes: no semantic leakage. Instruction families may grow with
  representation shape and value algebra, never with meaning, and only where an
  existing generic op cannot support the new shape efficiently or correctly.
  Answers match the row oracle.
- Fails: a domain needs its own predicate, op, terminal or lowering layer
  because of what it means rather than how its values are represented.

### H9 — relational emptiness without a fold

Hop and Join stay symbolic terms until an evaluation is required. Some
simplifications need no fold:
- an empty operand;
- known disjoint domains (classes, apertures, value intervals);
- contradictory algebraic constraints.

General relational emptiness and equality may still need evaluation.

**Measured.** Over a generated set of composed relational queries: the fraction
decided symbolically, the fraction that needs the fold, and the cost of each
route.

**Decision.** Soundness is one-sided and absolute: no query may be declared
empty symbolically while the fold finds a row (zero false empties). A
"cannot decide" answer is always allowed; it only costs a fold.

### H11 — stacked logic answers more without inflating confidence

PRE-REGISTERED 2026-10-14, NOT RUN; runs after H6b. DECISION (operator,
2026-10-14). SCOPE: examples only, existing ops; no new opcode, executor or
NARS-specific path (the bet is that stacking needs none). BASIS: the operator's
bet that "thinking becomes smarter through logic stacked", read as reusable,
replayable arithmetic over deterministic addresses.

**Setup.** `$var` composed algebraically (a chain of bricks at depth
d = 1, 2, 4, 8) before one evaluation, `$var.think()`. Bricks are the existing
NAL polyfill methods (D-NALP, `.claude/plans/nal-polyfill-v1.md`) lowered to
mask-risc / Quack; H6b's measured normal-form costs choose the representation
at the terminal.

**Three criteria, each with its own falsifier.**
- **Smarter:** a depth-d chain answers a query set that no single brick
  answers, and every answer matches the oracle (row oracle; for truth values,
  the pinned OpenNARS for Research / ONA fixtures of D-NAL-AUDIT-0). Fails if
  the chain adds no answer a single brick lacks, or any answer differs from the
  oracle.
- **Cheap stacking:** total cost (construction of every input charged,
  composition, the one terminal) at depth d is below d independent evaluations
  of the bricks, and cost per brick falls as d grows. Fails if the composed
  form costs at least as much as running the bricks separately at d = 4.
  Same accounting rules as H1/H5/H6: v4 only, tier printed, median of 11, ns.
- **No confidence inflation:** stacking adds structure, not evidence. The
  confidence of a stacked conclusion never exceeds what the truth functions
  give for its premises, and two derivation paths sharing an evidential base
  are not revised together (NAL-1 stamp rule, `nal_inventory.tsv`
  `NAL1.evidential_base_stamp`). The Gadamer gate (G1–G7) must be
  load-bearing: a disable run that removes the gate must produce an inflated
  confidence on the fixture, or the test is vacuous.

**Not claimed.** That deeper chains are always better: a depth where the
chain stops adding answers is a result, not a failure.

### H12 — a semantic map of inference functions across NARS systems

PRE-REGISTERED 2026-10-14, NOT RUN. DECISION (operator, 2026-10-14). SCOPE:
external systems run outside the build at pinned commits, as oracles only (the
D-NAL-AUDIT-0 fences: no NARS clone, no CE64 migration, no schema migration).
BASIS: Haspelmath's selection and stability rules, applied to reasoning
systems instead of languages.

**Method.**
- Systems: OpenNARS for Research and ONA (pinned in D-NAL-AUDIT-0), our NAL
  polyfill, and a MeTTa NAL implementation if one can be pinned (otherwise the
  map is built from three systems and says so).
- Corpus: the D-NAL-AUDIT-0 fixtures plus `nal_inventory.tsv`'s falsifier
  column, run on every system.
- Selection rule: an inference function becomes a node if at least two systems
  give different results on it (a different truth value beyond the u8
  allowance, a different derived term, or one derives and the other does not).
  Functions on which all systems agree are recorded but do not split a node.
- Each system's coverage is a region of the map. Report whether each region is
  connected.

**Measured.** Nodes added by each additional system, in a fixed order and in
reverse order; the nodes our polyfill does not cover; disconnected regions.

**Decision.**
- Assimilation is complete when the last system added contributes zero new
  nodes, in both orders, and our polyfill covers every node or lists the
  missing ones as named gaps.
- A difference is classified before it counts as a node: spec-conformant
  disagreement (a real distinction), or a defect in one system (reported, not a
  node). D-NAL-AUDIT-0 already found the engines disagree with each other and
  with Wang's 2009 spec on compound-term confidence; that is the first case to
  classify.
- Fails, as a method: if most nodes are introduced by only one system, the map
  measures implementation noise, not a shared space of functions.

### Open beyond these

H13 (attention-limited expansion vs full expansion, "saccades") and an
inertness test for every proposed meta-awareness level stay unregistered: H13
needs a named focus signal, and the levels need a stated meaning.

## Results: H6b (2026-10-14)

Probe `crates/lance-graph-mask-risc/examples/holo_tile_forms_probe.rs`, v4
(avx512f), three runs. Every arm matches the row oracle cell by cell on all 14
patterns, including the doughnut. Disable runs, each red: P without merging
equal operation masks (`product_form_is_canonical_across_routes` and the oracle
test fail); binary ops without the implicit empty group (both fail); the hybrid
ignoring its threshold (`hybrid_switches_both_ways` fails).

Deviation from the pre-registration: normalization is folded into construction
and into each operation, not timed as its own step. Every arm builds by sweeping
the pattern's membership function over all 65,536 cells.

**Verdicts against the pre-registered rules.**

| Rule | Result, three runs | Verdict |
|---|---|---|
| Structured (row, column, rectangle, prefix): P or H total at N = 100 ≤ 0.5× D | P 0.64–1.28, H 0.71–1.01 | NOT MET; undecided (under 2×) |
| Adversarial: H total at N = 100 ≤ 1.25× D | checkerboard 0.60–0.85, random 50 % 1.04–1.17, fragmented 1.19–1.33 | checkerboard and random 50 % pass; fragmented passes in one run, undecided in two |
| Component count predicts the winner better than density (Spearman with ln(best symbolic / D)) | components +0.34 / +0.55 / +0.52, density −0.06 / −0.07 / +0.12 | components ahead, weak at 14 patterns |

**Why the structured rule fails: construction dominates.** The 65,536-cell
build costs about 0.3 ms in every arm, against a query of 0.6–2.6 µs, so at
N = 100 the totals are mostly build. At N = 1,000 P reaches 0.29–0.77× D on the
structured patterns. Query-only (post hoc, not pre-registered), P is 0.21–0.73×
D on those four patterns, and 0.25–0.62× on checkerboard, complementary,
disjoint and doughnut: about 2–4× per query, at the edge of the 2× noise rule.
A symbolic construction (from a declared rectangle or class, O(groups) instead of
O(cells)) was not measured; the rule cannot be judged without it.

**Counterexample: the hybrid's byte switch is wrong.** H stays in P while P is
smaller than 8 KiB. On sparse (60 groups, 3.8 KiB) and dense (61 groups) that
keeps P, and H's total is 1.8–4.8× D's (query 6.8–14.6× D's). H as specified is
not safe as a default: it is more than 2× worse than D on two patterns outside
the adversarial set. P's cost follows the group count (a binary op builds the
g_a × g_b refinement), not its bytes. Post hoc, P's group count predicts
ln(P query / D query) with Spearman +0.81 and +0.88; components +0.37 / +0.45,
density −0.17 / −0.12.

**Thresholds and counterexamples.**
- P is the cheapest per query at 1–2 groups (row, column, rectangle, prefix,
  complementary, disjoint, doughnut, checkerboard) and loses at 60 or more
  (sparse, dense: 7–11× D; diagonal, random, fragmented: 59–188× D). No fixture
  lies between 2 and 60 groups, so the crossover is not located.
- I is cheapest only on the prefix (one interval, 0.04–0.05× D per query), as
  in H6; it loses heavily wherever intervals multiply (checkerboard 120×, random
  50 % 400–474× D).
- Density predicts nothing: checkerboard and random 50 % share density 0.5 and
  sit at opposite ends (P 0.25–0.30× vs 96–188× D per query).
- Connectedness alone is not enough: the staircase ("fragmented") is one
  component with 255 groups and loses as P by 60–77×; the doughnut is one
  component with 2 groups and wins.

**What this changes.** The terminal representation selector stays held. The
evidence points at a switch on group count, not bytes, and at symbolic
construction as the term that decides whether P pays at small N. Neither is
measured. Open: a fixture family between 2 and 60 groups to locate the
crossover; P built from a declaration instead of a cell sweep.

## Next falsifier: H6c (PRE-REGISTERED 2026-10-14, NOT RUN)

DECISION (operator, 2026-10-14): pre-register H6c after H6b. SCOPE: examples
only, no new opcode or executor, no production selector. BASIS: H6b's three
open terms. REVISIT WHEN: H6c reports.

Same universe, arms, oracle, tier, timing and noise rule as H6b (v4 only, tier
printed, median of 11, three runs, min–max reported, differences under 2×
undecided). Every arm still matches the row oracle on every cell first.

### Q1 — the crossover in group count

**Fixture family.** For g ∈ {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64}: g
distinct operation masks (seeded, density 0.5), assigned to contexts in two
layouts, so that group count is separated from contiguity:
- *blocked:* contiguous context ranges of equal size;
- *interleaved:* context c gets mask c mod g.

A and B are two such patterns with the same g and layout and independent seeds.

**Measured.** P and D per query (the H6b query), ∩, ∪ and ¬, per g and layout.

**Prediction (falsifiable).** P's query cost grows with g_A × g_B (the
refinement), D's stays flat, and the per-query ratio P / D crosses 1 between
g = 4 and g = 16 in both layouts. It is falsified if the crossover lies outside
that range, or if the two layouts disagree on it by more than one step of the
g grid (then contiguity, not group count, is the driver).

### Q2 — a group-count switch for the hybrid

H' uses P while P's group count is at most G, else D; G ∈ {2, 4, 8, 16, 32}.
The group count is known after construction, so the switch adds no pass.

**Decision.** H' with the G closest to Q1's crossover is safe as a default if,
on all 14 H6b patterns and the Q1 family, its total at N = 100 and its query
cost are both at most 1.25× D, and it is never 2× or more above D. The byte
switch (H) is run beside it; H6b showed it fails on sparse and dense, and H6c
must reproduce that or say why not.

### Q3 — construction from a declaration

Each arm is also built from a declaration instead of a cell sweep: a list of
(context range, operation range) rectangles for row, column, rectangle,
complementary, disjoint and doughnut, and a Morton prefix for prefix.
- P: one group per distinct operation mask, built from the ranges.
- D: rows filled word by word from the ranges.
- I: H6's `box_set` decomposition, or the prefix interval.

**Decision.** Rerun H6b's structured rule with every arm built from the
declaration: P or H at most 0.5× D's total at N = 100. Report build cost per
arm and the N at which P's total falls below D's.

**Fairness rule.** D gets the same declaration as P. A rule passed only
because D is swept while P is declared does not count.

### Tests and disable runs (before any timing)

- The declared build of every arm equals its swept build (P: equal group list; D
  and I: equal words or intervals). Disable: an off-by-one in the declared
  range end must fail it.
- H' switches both ways on the Q1 family (P at g = 1, D at g = 64). Disable:
  ignoring G must fail it.
- Every Q1 fixture really has g groups (P's group count equals g) and is neither
  empty nor full. Disable: assigning every context the first mask must fail it.

## Results: H6c (2026-10-14)

Probe `crates/lance-graph-mask-risc/examples/holo_tile_crossover_probe.rs`, v4
(avx512f), three runs. The arms, query and patterns are copied from the H6b
probe; the hybrid gains a switch policy. Five tests (declared == swept build in
P, D and I; Q1 fixtures have exactly g groups; the group switch goes both ways;
every arm and switch matches the row oracle on the 14 H6b patterns and 24 Q1
fixtures; a rectangle's Morton decomposition is exact). The three pre-registered
disables each turned a test red: an off-by-one in the declared range end, the
switch ignoring G, every context given the first mask.

| Question | Result, three runs | Verdict |
|---|---|---|
| Q1: P/D per query crosses 1 between g = 4 and 16 in both layouts | blocked: 8 → 12 in all runs; interleaved: 8 → 12 twice, 12 → 16 once | HOLDS; layouts agree within one step |
| Q2: a group switch ≤ 1.25× D everywhere, never ≥ 2× (G nearest the crossover: 8) | G = 8: max query 1.68 / 2.28 / 1.21; max total 1.88 / 1.72 / 1.20 | NOT shown safe (undecided, fails, passes) |
| Q2: the byte switch | max query 9.8–15.1× D | NOT safe in every run |
| Q3: declared build, P or H ≤ 0.5× D at N = 100 on row, column, rectangle, prefix | P 0.32–0.76, H 0.38–0.83; the better of the two 0.32–0.42 | PASS in all runs |

**Q1.** P's query cost grows with the group count; D's does not. P is 0.14–0.49×
D up to g = 4, 0.40–0.63× at g = 6–8, and 0.92–1.86× at g = 12–16. At g = 64 it is
9.4–12.4× D. Single operations sit above the full query: at g = 12, ∩ and ∪ are
1.29–1.94× D, and ¬ already crosses 1 at g = 6–8 in the interleaved layout
(1.03–1.38×).

**Q2.** The group switches at G = 2–16 beat the byte switch by a wide margin
(worst query 1.21–2.88× against 9.8–15.1×); G = 32 keeps P past the crossover
and reaches 3.1–7.3×. But the maximum over 38 fixtures moves from run to
run (worst fixture: doughnut, fragmented, g = 12 blocked, g = 48 interleaved),
which is what a maximum over noisy ratios does at about 40 % noise. The rule as
registered cannot certify any switch here. One effect repeats in two runs: at
G = 2 the doughnut (2 groups) costs 1.8–1.9× D per query, because its
intermediate results exceed 2 groups and are converted to D mid-query.

**Q3.** Built from a declaration, P costs about 0.3–0.7 µs to build, against
about 0.3 ms for the 65,536-cell sweep (H6b), and P's total is below D's from
the first query (N ≤ 1) on every declared pattern. I's declared build (Morton
decomposition of the rectangles, 2–10 µs) makes I the most expensive arm
except on the prefix. The "swept D" column in the probe's output builds through
the declaration's membership function, which re-collects the rectangles on
every cell; it is not comparable with H6b's sweep and is not used in any ratio.

**What this changes.** The H6b structured rule fails only because of the sweep:
with declared construction it passes in every run. P is the cheaper form up to
about 8 groups and loses beyond about 12. A group-count switch near G = 8 fixes
H6b's byte-switch failure, but the registered max-over-fixtures criterion is
too noise-sensitive to certify it at 14 + 24 fixtures. The selector stays held.
Open: a noise-robust criterion (per-fixture median over runs, or a repeated
maximum) and whether switching intermediate results mid-query should be
allowed at all.


## Next falsifier: H14 — full-row writes vs lane deltas that stay in their row (PRE-REGISTERED 2026-10-15, NOT RUN)

**Question (operator, 2026-10-15).** For one graph (classid) x thought (rung) x
64k-row tile, is a dirty row written as (a) the full 512-byte SoA row, or (b) a
32-lane index plus the changed 16-byte lanes? D-HL-5 measured the in-memory
tile (slot delta 12 + 16k bytes per dirty row, 0.054 of a row at k = 1); this
falsifier owns the persistence choice. Written before the probe exists; a
change after the run is a new dated section.

**DECISION (operator, 2026-10-15).** Two levels of awareness stay distinct and
are never folded into each other:
- a **session** holds several thoughts: one thought mask per session;
- a **thought** holds several lanes: one lane mask per (thought, row).
A lane is never stored apart from its row. Either the whole row is written, or
the delta carries its row key and lane mask in the same record, so a read never
has to re-join lanes to rows. Storage that splits lanes from their rows creates
a dissociation that a later read must re-merge, and that re-merge is a
materialization cost. SCOPE: H14. REVISIT WHEN: an arm without the connection
wins on read and bytes together.

**Arms.**
- A (full row): per dirty (thought, row), the 512-byte row image.
- B (connected delta): per dirty (thought, row), one record = row key, lane
  mask (u32), the changed lanes packed in lane order. Lanes exist only inside
  their record.
- C (dissociated, control): one table per (thought, lane), keyed by row. It is
  kept only to measure the re-merge it forces; it cannot become the default
  under the decision above.
All three hold the same session thought masks; the session level is not part
of any arm's lane storage.

**Grid.** Classes 1 and 16; thoughts per session 1, 8, 64; 65,536 rows per
tile; dirty fraction 1 %, 10 %, 100 %; changed lanes per dirty row k = 1, 4,
16, 31. Fixtures deterministic (seeded), identical for every arm.

**Metrics.** Bytes stored and bytes one append would write (Arrow batch size,
computed exactly); seal ns per dirty row; full-row read ns per row (base merged
with the thought's record); session read ns per row (every thought of one
session over a 4,096-row aperture); for C, the re-merge ns per row reported on
its own. v4 only, tier printed by the program; ns per row or per record, never
whole-call milliseconds. The Lance write itself is not timed here (D-WAL owns
it); this falsifier reports what each arm would hand to it.

**Rule.** B is the default if, at k <= 4 on every (dirty, thoughts) cell, B's
full-row read is <= 1.25x A's, its session read <= 1.25x A's, and its bytes <=
0.5x A's. Otherwise A is the default. C is reported, never chosen. If B passes
only at some k, the crossover k* is reported and the decision returns to the
operator.

**Tests and disable runs (before any timing).**
- Every arm reads the same bytes as a dense reference tile, row by row
  (disable: drop the merge in B's read).
- The session mask names exactly the thoughts that wrote in the session
  (disable: derive it from the lane masks of one thought).
- A thought's lane mask equals the lanes that differ from the base at that row
  (disable: store every lane).
- B never yields a lane without its row key; C's re-merge returns the same rows
  as A (disable: skip a lane table in the re-merge).

### H14 amendment before any run (2026-10-15): arm C is the inverted fold, and a candidate

**Correction (operator, 2026-10-15).** "If you use c) with the inverted
holographic addresses you just make the fold the default and computation
happens inside the inverted fold, only pay for materialisierung." The text
above treated C as a dissociated control that "cannot become the default" and
is "reported, never chosen". Both statements are SUPERSEDED. Nothing has been
run, so this is a pre-registration change, not a result change.

**Why C does not dissociate.** The decision above forbids storage that a read
must re-join to its rows. Per-lane planes over the tile's own row addresses
need no re-join: every plane lives in the same 65,536-row coordinate space, so
combining lanes, thoughts and the aperture is an address-aligned mask fold
(`And` / `Or` / `Count` in mask-risc), not a join. The row connection is
carried by the address instead of by co-locating lanes in one record. Lanes
are assembled into a row only when a terminal asks for a row, and that
assembly is the one materialization C pays.

**C (inverted fold), replacing the control.** Per (thought, lane) one plane:
- the set of rows whose lane differs from the base, as an address mask
  (carrier chosen by density: dense bitmap, sparse list or runs; the probe
  prints which carrier each cell used, since a dense 8 KiB plane per
  (thought, lane) is 64 x 31 x 8 KiB = 15.9 MB per tile and would lose on
  bytes at 1 % dirty);
- the changed 16-byte values for those rows, in address order.
Queries stay inside the fold until a named terminal. The session thought mask
is unchanged and still not part of any arm's lane storage.

**Relation to H5.** H5 refuted a maintained inverse built in addition to the
primary store (the plane inverse did not pay back its build within 1,000
queries). Here the inverse planes are the primary store: there is no second
structure to build, only the seal cost of writing into planes, which is
measured below. If C's seal cost per dirty row exceeds B's by more than the
H5 plane build ratio, that is reported as the same failure, not hidden.

**Added queries (all arms, same fixtures).** Each answered without building a
row, except where a terminal is named:
- F1: rows thought t changed in any lane of a lane set L, inside an aperture,
  as a count;
- F2: rows changed by any thought of the session in lane l, inside an
  aperture, as a count;
- F3: F1's rows, materialized as full 512-byte rows (terminal).
A and B answer F1 and F2 by scanning their records; C by folding planes.

**Metrics added.** For C, fold ns per row of the aperture (F1, F2) and
terminal materialization ns per emitted row (F3, and the full-row and session
reads), reported separately so the materialization cost is visible on its
own. Bytes as before, per carrier.

**Revised rule (replaces "C is reported, never chosen").**
- C is the default if, at k <= 4 on every (dirty, thoughts) cell: F1 and F2
  on C are at least 2x faster than on B; the full-row read and session read on
  C, including terminal materialization, are <= 1.25x B's; and C's bytes are
  <= B's.
- Otherwise the A-vs-B rule above decides between A and B.
- If C wins the fold queries but loses the materializing reads, both numbers
  are reported, the crossover (how many rows a query materializes before C
  stops winning) is stated, and the decision returns to the operator.

**Tests and disable runs added (before any timing).**
- F1, F2 and F3 return the same rows and bytes on every arm as on the dense
  reference (disable: drop one lane plane from C's fold).
- C's materialization counter is zero after F1 and F2 and equals the emitted
  row count after F3 (disable: assemble rows inside the fold).
- C's planes name exactly the rows whose lane differs from the base
  (disable: store every row of a touched lane).

### H14 correction to the amendment (2026-10-15): the address is the register, nothing is traversed

**Correction (operator, 2026-10-15).** "You don't traverse, the address is
already the loci register. Instead of `Table1[index][a][b]`,
`Table1[index][a] Table2[index][b]`." The amendment above stored C's changed
values "in address order", compacted. Finding a row's value in a compacted run
needs a rank or a search over the plane, which is a traversal. That part is
SUPERSEDED.

**C, as corrected.** One table per (thought, lane), indexed by the row address
itself: `Lane_l[index]` is the 16-byte value of lane l at that row. The
address is the position; reading lanes a and b of row i is two positional
reads at the same i, with no search, no rank and no join. The presence mask
per (thought, lane) stays, and it is what F1 and F2 fold over; it is never
used to locate a value.

Contrast with B: B's record is `Table[index][a][b]`, one row key with its lanes
packed behind a lane mask, so reading lane b means walking the record (a
popcount rank over the lane mask). The probe reports that rank as part of B's
read cost.

**Bytes.** Logically every (thought, lane) table spans the tile, which is
65,536 x 16 B = 1 MiB. Two rules keep that from becoming the stored size:
- only (thought, lane) pairs that changed at least one row get a table;
  an absent table falls through to the base row (the zero-fallback);
- the bytes handed to the writer are the dirty positions only (Lance sparse
  write, merge on read), reported per cell next to the resident size.
The resident size is reported on its own so the cost of positional tables in
memory is visible, not hidden behind the persisted size.

**Test added.** Reading lane l of row i on C touches exactly position i of one
table: an access counter equals the number of lanes read, independent of how
many rows the table holds (disable: store C's values compacted and look them
up by rank, which makes the counter grow with the plane).

**Addendum (operator, 2026-10-15): "it's merge on read but for the join."**
Combining lane tables is not a separate join step. It is the same
merge-on-read that overlays dirty bytes on the base: for row i, start from the
base row and overlay position i of each touched (thought, lane) table. Base
over delta and lane next to lane are one operation at one index. So:
- the original metric "for C, the re-merge ns per row reported on its own" is
  SUPERSEDED; C's full-row read is measured as one merge-on-read and reported
  with the same metric as A and B;
- "terminal materialization" for C means exactly this merge-on-read, run only
  for the rows a terminal asks for; F1 and F2 never run it;
- the original DECISION's concern (a dissociation that a later read must
  re-merge at extra cost) does not apply to positional tables, because the
  merge is the read itself. It still applies to any layout that would need a
  key lookup or a sort to line lanes up, which is why compacted runs stay
  superseded.

### H14 results (v4, 2026-10-15): C wins every fold, loses every merge-on-read; returns to the operator

Run: `cargo run --release -p lance-graph-mask-risc --example
holo_lane_storage_probe` on an AVX-512 host (the program printed
`tier: v4 (avx512f)`). One class per run; 16 classes were not timed (the
64-thought cells need about 8 GB for one class on a 15 GB host). Bytes scale
linearly with classes; the timing effect of more classes is untested.
Seven oracle tests pass against a dense reference; five disable runs each
turned their test red (drop B's merge, store every lane in B, skip a lane
plane in C's fold, build rows inside C's fold, session mask from one thought).

Measured, ns per row unless noted (full table in the board entry):

| quantity | A full row | B connected delta | C inverted fold |
|---|---|---|---|
| F1 (one thought, 4 lanes, per aperture row) | 0.5–38.8 | 0.43–1.6 | 0.027–0.075 |
| F2 (all thoughts, one lane, per aperture row) | 1.2–272 | 0.86–109 | 0.017–0.59 |
| full-row read (merge-on-read for C) | 38–152 | 51–1,005 | 259–1,529 |
| session read (per thought-row) | 6.7–61 | 11.5–173 | 57–539 |
| seal per dirty row, 1 % dirty | 31–292 | 13–30 | 2,454–21,106 |
| seal per dirty row, 100 % dirty | 271–389 | 5–59 | 32–222 |
| bytes handed to the writer | 516 B per dirty row | 12 + 16k | ≈ B (0.7–1.2× B) |
| resident | as written | as written | 31 MiB per thought, logical |

Rule at k <= 4 on every (dirty, thoughts) cell:
- C's folds are at least 2× faster than B's on F1 and F2: **yes**, in every
  cell (F1 10–24×, F2 39–256×). F2 is the session-wide count and
  grows with the number of thoughts on all three arms: C ORs one presence
  window per thought, 0.017–0.023 ns per aperture row at 1 thought,
  0.058–0.072 at 8, 0.38–0.59 at 64.
- C's reads with merge-on-read within 1.25× of B's: **no**, in every cell
  (2.8–6.1× slower).
- C default: not met. B default over A: not met (B's full-row read is up to
  4.6× A's at 100 % dirty with k = 4, and its session read 1.4–2.2× A's at
  1 % dirty).

Per the pre-registered rule this returns to the operator with both numbers.
The crossover, from F3 (fold, then materialize the rows the fold names):
C beats B on F3 at 1 % dirty in every cell (C 59–136 vs B 87–690 ns per
emitted row) and at 10 % dirty when k = 1; it loses in most other cells. A
query on C pays about 0.4 ns per aperture row less than B for F1 and about
200–450 ns more per materialized row, so C wins a mixed query while it
materializes fewer than about 1 row in 500–1,000 of the aperture for a
single-thought count, and between about 1 in 8 and 1 in 3 for a
session-wide count at 64 thoughts.

What the numbers point at (HYPOTHESIS, not measured here):
- C's read cost is the merge itself, not the positional access: the merge
  probes every touched lane table of the thought (31 tables at every k in
  this fixture, because the changed lanes vary by row), and each probe is a
  separate 1 MiB table, so a row read touches up to 31 cache lines in 31
  tables. A per-row lane mask held positionally beside the tables
  (`Mask[i]`, 4 bytes per row) would let the merge read only the k tables
  that changed, keep the fold, and keep the address as the register. That is
  B's lane mask without B's packed record, and it is the next arm to try
  (H14b).
- C's seal at 1 % dirty is page faults: the positional tables are zeroed
  lazily, so each scattered dirty row touches a fresh page. At 100 % dirty
  the same seal costs 32–222 ns. Persisted bytes are unaffected.

## Next falsifier: H14b — C with a positional lane mask per row (PRE-REGISTERED 2026-10-15, NOT RUN)

Approved by the operator 2026-10-15 ("go ahead with H14b").

**Arm M.** C's per-(thought, lane) tables and presence masks, unchanged, plus
one `u32` per row per thought held positionally: `Mask[i]` names the lanes row
i changed in that thought (0 for a clean row). The address stays the register:
no index, no traversal. Folds are C's folds (the same presence masks).
Merge-on-read reads `Mask[i]` and then position i of only the tables it names,
instead of probing every touched table.

Resident cost added per thought: 4 bytes per row (256 KiB per 64k tile).
Bytes handed to the writer: per dirty row its key, its lane mask and the
changed lanes (8 + 16k); per-lane positions follow from the masks.

**Prediction.** At k <= 4, on every (dirty, thoughts) cell of the H14 grid,
M's full-row read and M's session read are each at most 1.25× B's. Falsified
if any such cell exceeds 1.25×.

**Decision rule.** M becomes the default for a dirty tile if, at k <= 4 on
every cell: the prediction holds; M's bytes to the writer are at most B's;
and C's fold advantage (at least 2× B on F1 and F2) carries over, which holds
by construction because M folds C's masks (checked by test, not re-timed). If
the prediction fails, the result returns to the operator with the cells that
failed and the F3 crossover between M and B.

Also reported, not part of the rule: seal per dirty row, F3 per emitted row
for M against A, B and C.

**Comparisons are within one run.** A, B, C and M are timed in the same run on
the same host with the tier printed. The H14 numbers (another host) are not
compared against.

**Tests before any timing.**
- M reads the dense reference on every row (k = 1, 4, 31).
- M's `Mask[r]` equals B's lane mask for every row r.
- M's merge probes exactly `popcount(Mask[r])` tables for row r, and C's
  merge probes every touched table; the fixture must have more touched tables
  than k, or the test proves nothing.
- M's F1, F2 and F3 equal C's.

**Disable runs.** M's merge iterating the touched tables instead of `Mask[r]`
must turn the probe test red. A mask that drops one lane must turn the
reference test red.

### H14b results (v4, 2026-10-15): the prediction fails above 1 % dirty; M is C's read fixed, not B's

Run: `cargo run --release -p lance-graph-mask-risc --example
holo_lane_storage_probe`; the program printed `tier: v4 (avx512f)`. A, B, C
and M were timed in the same run on the same host (a different host from the
H14 run, so the two runs are not compared). One class. 11 tests pass; the two
pre-registered disable runs turned their tests red (M's merge over every
touched table; a mask that drops a lane).

**Rule at k <= 4 (18 cells).**
- Prediction (M's full-row and session read each <= 1.25× B): **FALSIFIED**
  in 9 of 18 cells. All 6 cells at 1 % dirty pass (full-row 0.94–1.14× B,
  session 0.93–1.13×). The failures are at 10 % dirty (T = 8 k = 4: 1.40 /
  1.34; T = 64: session 1.34–1.87) and at 100 % dirty (full-row up to 1.63×,
  session up to 5.85×).
- M's bytes to the writer <= B's: yes, in every cell (one u32 per dirty row
  less than B).
- Folds: M uses C's presence masks; tested equal, not re-timed.
- M default: **not met.** Returns to the operator.

**What M did change.** Against C in the same run, M's full-row read is
0.98–3.69× as fast (C/M): faster in 35 of 36 cells, from 297.4 vs 209.8 ns
(T = 1, 1 %, k = 1) to 1,129 vs 306 ns (T = 64, 1 %, k = 1); the one cell
where C is faster is T = 1, 100 %, k = 31 (1,299 vs 1,320 ns). On F3 (fold,
then build the rows the fold names), M beats C in 33 of 36 cells and B in
18 of 36. At 1 % dirty M is the cheapest arm in 10 of 12 cells (M 84–184 ns
per emitted row, B 175–1,260, C 123–241); C is cheaper at k = 31 for 1 and 8
thoughts.

**Where M loses (HYPOTHESIS, not measured).** B keeps a row's changed lanes
next to each other in one record, so a row read is one index lookup and one
or two cache lines. M keeps lane l of row i in table l at position i, so a
row with k changed lanes is k cache lines in k tables. That costs little when
few rows are dirty and grows with density; the session read at 8–64 thoughts
also walks up to T × k tables at once. A test would be to time the read
against k at fixed density and count cache misses per row.

**Crossover.** At 1 % dirty, M matches B on reads (within 1.25× in all 6
cells at k <= 4) and beats B on F3 in all 12 cells; at 10 % and above, B's
reads are faster. M's seal at 1 % dirty is still the
page-fault cost of C's positional tables (6.5–15 µs per dirty row at one
thought).
