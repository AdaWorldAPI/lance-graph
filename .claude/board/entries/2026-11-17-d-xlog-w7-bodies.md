# D-XLOG-W7 items 1 and 3: conjunctive bodies over the cube (2026-11-17)

**Status:** MEASURED. Cyclic bodies (`tri`, `cyc`, `clique5`) are answered over
the cube as a count, a support mask per variable, and, at the edge only, named
witness rows. Item 2 (lifting the #1943 virtual SoA views) is not in this entry.

## What exists

`crates/lance-graph-xlog/src/cube/body.rs`:

- `Cube::body_answer(table, atoms, k) -> BodyAnswer { count, support }`. A body
  is up to 8 binary atoms over variables `0..k`, refused when malformed
  (`BodyRefusal`). One tile only; a tiled cube is refused.
- The count builds no row. It walks row masks (one 4-word row per position per
  atom and direction). Independent components of the unbound variables multiply.
  A bind whose remainder is one variable, or a split into single variables, costs
  one AND and one popcount per leaf per candidate (`U64x4`).
- The variable order is a DP over `(unbound, bound)` states, planned once per
  call. A candidate set is estimated as `|proj| × Π selectivity`, where the
  selectivity is the mean nonzero row popcount over `n` for each atom to a bound
  variable.
- `Cube::materialize_witnesses` is the one place rows are built (global ids, `k`
  per witness), and it counts them in `stats().rows_materialized`.

## Falsifiers

`crates/lance-graph-xlog/tests/body.rs`, 6 tests, all against a nested-loop
oracle that shares no code with the cube: XLOG's `triangle_inputs` (5 witnesses),
a four-cycle (4), `clique5` over ten relations (1), 60 random seeds × 6 shapes
(tri, tri with reversed atoms, cyc, chain, star, two disconnected parts; at least
200 of 360 cases have witnesses), malformed-body refusals, and "the count builds
no rows".

Disable runs (each anchor asserted, file restored from a backup):

| disable | result |
|---|---|
| split multiplies → adds | red (count) |
| split keeps support when a component is empty | red (support) |
| leaf kernel keeps support when a leaf is empty | red (support) |
| bind marks support for a candidate with no witness | red (support, 3 tests) |
| `leaves()` replaced by `unreachable!()` | green → removed as dead code |
| optimizer replaced by declaration order | green (affects speed only); cyc at 8192 edges 532 → 1706 µs |
| selectivity dropped from the estimate | green; cyc at 512 / 2048 edges 70 → 100 / 293 → 432 µs |

## Cost

`cargo run --release -p lance-graph-xlog --example body_cost` (native AVX-512,
one tile of 256 nodes, 4 random relations, median of 31; every count asserted
against an edge-walk oracle):

| edges per relation | tri µs | cyc µs | cyc witnesses |
|---|---|---|---|
| 512 | 22 | 45 | 20 |
| 2048 | 45 | 293 | 3 490 |
| 8192 | 109 | 532 | 634 378 |

Before factorization the four-cycle took 1 086 µs at 2048 edges and 15 744 µs
at 8192, both failed gates under the no-millisecond rule.

## Open

- Item 2: the #1943 virtual SoA views are still in the lab.
- Bodies over more than one tile are refused; a tiled body needs the
  global-to-tile index that item 2 also needs.
- The selectivity estimate assumes the atoms are independent. A correlated
  relation can make the chosen order slower, never wrong.
