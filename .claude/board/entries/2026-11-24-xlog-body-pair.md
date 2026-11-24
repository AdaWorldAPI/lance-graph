# 2026-11-24 — EXO-396 reworked: a body's demanded projection as presence

**Status:** MEASURED. Supersedes the row-store approach on xlog branch
`ccr-85acaff0-oszq40-wide-rows`, which is not merged (EXO-330 / EXO-332).

**Insufficiency of what existed.** `Cube::body_answer` always enters the walk
with nothing bound (`count(all, 0, …)`), and `Atom` has no constant, so it
answers only the witness count and independent per-variable supports.
Those cannot express a correlated two-column head: on the two-triangle twin
the support product is 4 pairs where the projection has 2. The only
constraint a plan could apply was a lent plane row, so `X != Z` had no form.

**Extension (reuse of the walk over lent `PairPlane` rows; no rows built):**
- `Cube::body_pair(table, atoms, k, distinct, [u, v])`: for each candidate
  position `x` of `u`, the existing walk runs with `u` bound at `x`; row `x`
  is the support of `v`. Other variables are existential. The result is a
  presence plane over existing coordinates, like `Closure` (EXO-330 §3).
  The pinned state `(all − u, {u})` is planned explicitly: when the top
  state splits into components the first plan never reaches it (the disable
  run overflows the stack).
- Distinctness pairs: once the partner is bound at `p`, bit `p` is cleared
  from the candidates. The diagonal of the two axes is implicit; nothing is
  stored. A pair also couples its variables for component splitting.

`sg_witness` stays a recipe: `sg_result` is the OR of rule 1's (X, Y) plane
and the unfolded rule 2's (W, Z) plane. `fof` is one plane with `[0, 2]`
distinct.

**Tests:** `crates/lance-graph-xlog/tests/body_pair.rs`, 10 tests. Every
demand of the triangle, the four-cycle (with and without distinctness) and
a chain matches a nested-loop oracle. The EXO-396 hand oracles (`sg_result`,
`fof`) are kept. There is also a cross-component product, the twin, a test
that no rows are built, refusals, and an empty body. Each of five disable
runs fails: no clear (2), support of `u` (6), leaves ignore distinctness
(1), split ignores distinctness (1), no pinned plan (stack overflow).

**Cost** (`examples/body_cost.rs`, release, one tile of 256, median of 31,
µs):

| edges/rel | tri | tri→XZ | cyc | cyc→WY | sg2→WZ | fof→XZ |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 19.9 | 20.5 | 41.7 | 41.9 | 42.1 | 19.8 |
| 2048 | 39.9 | 39.0 | 315.5 | 319.9 | 315.7 | 34.5 |
| 8192 | 86.0 | 96.1 | 724.1 | 722.2 | 743.0 | 80.3 |

The projection costs about the same as the count. The densest four-cycle
is 743 µs, the largest number in the table.

`materialize_distinct_witnesses` serves heads that keep every variable
(`mutual(X, Y, M) :- friend(X, M), friend(Y, M), X != Y`): the same Plan,
the pairs cleared in the emit path. `materialize_witnesses` delegates to it
with no pairs. Disable run (pairs ignored): 2 tests red.

OPEN:
- Wiring `body_pair` into xlog-cube for `sg_result` and `fof`.
- The result reaches later readers as a cube relation today only through
  `add_relation` (pairs). A presence plane handed over without decoding is
  not built.
- One tile only, as `body_answer`.
- A distinctness pair is the identity law's image: within a tile, row x of
  the identity map is bit x, which is what `clear_distinct` clears. A body
  needing `X != f(Z)` for an affine `f` would read a law lane (#2160,
  `contract::lane_law`) and clear the bit at `f(bound)`. No such body exists
  yet, so no law-lane reader is built.
