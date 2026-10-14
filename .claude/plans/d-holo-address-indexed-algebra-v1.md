# D-HOLO — address-indexed algebra: falsifiers H1, H5, H6 (v1)

**Status:** PRE-REGISTERED 2026-10-14. Nothing below is measured yet; results go
in a new dated section after the run, never into this one.

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

- [ ] D-HOLO-0b: this pre-registration and its board rows, committed before any probe exists
- [ ] D-HOLO-H1: K-branch probe, oracle tests, disable runs, v4 run, result
- [ ] D-HOLO-H5: coordinate-native vs coordinate-crossing probe, same
- [ ] D-HOLO-H6: closure of address restrictions probe, same
- [ ] Results section, board entry, status rows

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

## Boundaries

- No change to mask-risc, Quack, the contract or ndarray. The probes are examples.
- No new opcode or terminal. Arm C (H1) and the inverse arms (H5) are labelled
  out-of-contract, because no existing terminal expresses them.
- No external dependency.
