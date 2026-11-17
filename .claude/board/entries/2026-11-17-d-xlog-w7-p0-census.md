# D-XLOG-W7-P0 step 1: census of the wide heads (2026-11-17)

**Status:** MEASURED (read only; no cube, no evaluator).

`crates/xlog-coverage-probe/examples/w7_census.rs` reads `census-007bb19.tsv`
and the xlog sources at `007bb19` (`git show 007bb19:<path>`, never a
checkout), and classifies every refused head of 3+ columns by body shape and by
consumer. Output: `crates/xlog-coverage-probe/w7-census-007bb19.tsv`.

Command:

    cargo run --release --manifest-path crates/xlog-coverage-probe/Cargo.toml \
      --example w7_census -- census-007bb19.tsv <root holding the 007bb19 files>

## Counts

- 128 refused census rows are 104 distinct heads (one row per rule).
- 52 heads have 3+ columns: 13 are query heads (`__xlog_query_N`) and 39 are
  rule heads. The 39 cover 47 census rule rows; the park entry counted 46 from
  `pred` declarations, while this census reads the head atom. The off-by-one
  is that difference in method, not reconciled further.
- 5 heads are unread: the ILP `reach` rules in `ilp_integration_tests.rs`,
  whose heads come from learnable templates.

| rule heads (39) | shape | consumer | heads |
|---|---|---|---|
| 22 | join | rule 15, query 2, query+rule 2, none 3 | `tri` ×9, `cyc` ×5, `clique5`, `support_1` ×2, `support_2` ×2, `mutual`, `user_perm`, `local_order` |
| 11 | one atom | rule 8, query 1, query+rule 1, none 1 | `echo` ×6, `tri_echo`, `usable` ×3, `sales_by_cat_region` |
| 6 | computed columns | query | `add_result`, `div_result`, `mod_result`, `mul_result`, `sub_result`, `sum_squares` |

Checked against the source: `test_sort_labels.rs` #0 (`support_1` a 3-atom
join, `usable` a projection of it) and `test_wcoj_recursive_dispatch.rs` #0
(`tri` a 3-atom join, `echo` and `tri` defining each other).

## What it shows

1. **No join head is consumed by a fold.** No aggregate reads `tri`, `cyc`,
   `clique5` or `support_*`. They are read as rows, by queries or by other
   rules. So a factorized count terminal (for example the triangle count as
   `popcount(adj[x] & adj[y])`) has no consumer in this corpus.
2. **10 of the 11 one-atom heads read a join head,** not a stored relation:
   `usable` projects `support_1`, and `echo` / `tri_echo` re-read `tri` (they
   define each other recursively). They follow whatever representation the
   join head gets, and as a projection they need no new lanes.
3. **One head reads a stored relation:** `sales_by_cat_region(Cat, Region,
   sum(Amount)) :- sales(Cat, Region, Amount)`, a GROUP BY sum over one base
   relation. That fits the existing path: the lanes through one
   `borrow_all`, and mask-risc's `GroupSumI32` terminal.
4. **The 6 computed-column heads** compute values with `is` expressions. They
   are out of W7's scope, as computing constructs are for VRL.

## Decision

W7 stays parked for the 22 join heads. Their consumers want rows, so the only
representation that serves them is a named sink holding one position table
per head column into the cube's coordinate space, and none is measured.
REVISIT WHEN unchanged.

Next, small: `sales_by_cat_region` through `borrow_all` + `GroupSumI32`, with
parity against the naive oracle.
