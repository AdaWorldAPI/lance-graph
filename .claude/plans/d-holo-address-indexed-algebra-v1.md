# D-HOLO — address-indexed algebra: falsifiers H1, H5, H6 (v1)

**Status:** MEASURED 2026-10-14 (H1, H5, H6). The pre-registration below is
unchanged; results are in § Results (2026-10-14).

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
