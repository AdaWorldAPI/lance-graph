# 2026-11-24 — EXO-394: stratified negation over one-column sets

**Status:** MEASURED (census run against the unmerged branches, see below).

- lance-graph: `SetExpr::Column { rel, col }` makes one column of a stored
  relation a set leaf, evaluated in the same terminal as the rest of the set
  expression. No mask crosses between terminals. 9 hand-oracle tests in
  `crates/lance-graph-xlog/tests/set_column.rs`. A disable run (column read
  forced to 0) turned 5 of them red.
- xlog (`ccr-85acaff0-oszq40-negation-sets`): xlog-cube routes `Diff`, a
  semi-join on one column, and a `Project[Column(c)]` over a binary scan to
  that set expression. 6 tests in `crates/xlog-cube/tests/negation.rs`. A
  disable run (`Diff` routed as `and`) turned 5 red.

Census (`xlog-coverage-probe --example reconcile`, xlog crates patched to the
branch, lance-graph at this branch), against `census-097c351.tsv`:

| rung | before | after |
|---|---:|---:|
| oracle-agrees | 102 | 109 |
| refused | 79 | 72 |
| refused-on-data | 49 | 49 |
| disagreements | 0 | 0 |

- Newly oracle-agreeing rules, all with Datafrog agreeing: `isolated` ×2,
  `no_in` ×2, `leaf_part`, and the two `__xlog_query_0` rules that read `no_in`.
- 9 rules that already agreed moved to the set route: `has_in` ×2, `part` ×2,
  `has_children`, `dummy_e1`/`e2`/`e3` (4 rows). Set routes went from 13 to 29.
- No other row changed rung or route.

OPEN:
- `keep` and `dry` stay refused (negation over a join-shaped body).
- The census TSV is not committed: it pins unmerged revisions. Regenerate it
  once both PRs merge and the probe's xlog rev is bumped.

**Update 2026-11-24:** lance-graph#2171 and xlog#11 merged. The census was regenerated at
xlog `1435111` against lance-graph `2075f7b` and committed as
`crates/xlog-coverage-probe/census-1435111.tsv`. Once the revision label is removed it is
identical to the branch run above.
