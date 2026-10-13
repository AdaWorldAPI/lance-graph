# relalg-v1 — relational algebra over coordinate populations, rewritten before it runs

**Status:** D-RA-1 MEASURED (prototype crate `crates/lance-graph-relalg`, own
workspace root, excluded). D-RA-2..4 PROPOSAL.
**Date:** 2026-10-13. **Repo:** lance-graph.
**Relation to D-CMR:** a prototype for
`d-cmr-concept-chain-normalization-v1.md`. It tests the rule catalogue (D-CMR-6)
on residual comparisons and a lowering through R2IL (D-CMR-2's job, via the
route NAL uses). It is not a third frontend syntax and not the home of the set
algebra; D-CMR puts that in `lance-graph-contract` (D-CMR-1).

## 1. What D-RA-1 built

- `Rel`: a set of rows of one table, described (`All`, `Empty`, a resident
  plane, one comparison on one `I32` or `U32` lane, `And` / `Or` / `Not`), and
  a `Measure` (`Count`, `Sum`, `Min`, `Max`, `Keep`).
- `normalize`: to a fixpoint — negation to the leaves; flattening, unit and
  zero; duplicates; `p ∧ ¬p = ∅`, `p ∨ ¬p = U`; per `I32` lane in a
  conjunction, one value interval plus the excluded values inside it, `∅` when
  empty; per lane in a disjunction, overlapping or adjacent intervals merged,
  `U` when they cover the lane; `U32` equalities agree or contradict;
  absorption. Intervals are emitted as `<` / `>` / `==`, the forms
  `IntSLess` / `IntEqual` lower without a negation pass.
- `symbolic`: `∅` answers `Count 0`, `Sum 0`, `Min/Max None`; `U` answers
  `Count n_rows`. No program runs.
- `run`: R2IL calls over `FoldDialect` (`LOAD`, `IntSLess`, `IntEqual`,
  `IntAnd`, `IntOr`, `IntNot`, `RANGE`, then `PopCount` / `SUM` / `MIN` /
  `MAX` / `KEEP`). mask-risc executes; the crate has no row loop.

In D-CMR terms every comparison here is a residual: the lanes are not a proven
row ordering, so no row interval (`Runs`) is formed. The interval work is
constraint propagation over value intervals of one lane (§3.5's first
mechanism), not geometry.

## 2. Measured (`cargo test --manifest-path crates/lance-graph-relalg/Cargo.toml`)

| test | result |
|---|---|
| 400 generated descriptions × 5 measures: rewritten == as written == row oracle, idempotent | 273/400 rewritten; 70 of 2,000 as-written runs refused (below) |
| four comparisons on one lane | rewritten 2 mask-risc ops, as written 8 |
| `x=3 ∨ x=4 ∨ x=5` | rewritten 2 ops, as written 5 |
| contradictions (`x<5 ∧ x>10`, `k=1 ∧ k=2`, `k=1 ∧ k≠1`, `p ∧ ¬p`, `x<MIN`, `x>MAX`) | answered with 0 programs, all four scalar measures |
| tautologies (`x<5 ∨ x≥5`, `p ∨ ¬p`, `x≤MAX`, `x≥MIN`) | `Count = n_rows`, 0 programs |
| ten comparisons on one lane | as written refused (`ScratchBufferTooSmall`), rewritten runs |

Eight disable runs, each red: fusing across lanes, no symbolic answer, no
absorption, union ignoring adjacency, no end tightening, `≤` negated to `≥`,
no complement check, `<` off by one.

**Finding:** normalization decides whether a description fits one fold, not
only how fast it runs. `FoldDialect` gives one fold 8 scratch slots and every
negation as written takes a fresh one, so 70 of 2,000 small descriptions (at
most three leaves) do not lower as written at all. After the rewrite every one
of them runs.

## 3. Path

| D-id | step | gate / falsifier |
|---|---|---|
| D-RA-1 | Prototype: `Rel`, `normalize`, `symbolic`, R2IL lowering | == row oracle on generated descriptions; each guard removal fails |
| D-RA-2 | Move the rules into the contract set IR (D-CMR-1/-6) with a universe per set; relalg keeps only the R2IL lowering, or is deleted | relalg's tests pass unchanged against the contract IR |
| D-RA-3 | Thin Datalog front end: rule text → `Rel` (non-recursive rules, constants bound to lanes) | parsed rules == hand-written `Rel` on the D-RA-1 suite |
| D-RA-4 | Bound recursion: `ancestor(c, Y)` as a frontier fixpoint in an external driver (option d′ of `entries/2026-10-03-coresearch-chained-hop.md`), each step a D-CMR-5 preimage; unbound closure refused | == row oracle closure; a step that grows the frontier past its bound is refused, never truncated |

D-RA-2 is a decision for the maintainer (see the PR). D-RA-3 and D-RA-4 wait
for it.

## 4. Not built

Joins (a hop is a preimage over coordinates, D-CMR-5, never a foreign key),
recursion, row intervals, `Sum(A) − Sum(A∧B)`, Datalog text.
