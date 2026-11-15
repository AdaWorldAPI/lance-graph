# 2026-11-15 — D-XLOG-COV-1: XLOG coverage parity (lab only)

**Kind:** lab probe change (`crates/xlog-coverage-probe`) plus two fixes in the
XLOG fork's cube executor (AdaWorldAPI/xlog#8). Nothing in production calls
`xlog-cube`; nothing is admitted.
**Grade:** MEASURED. Census over 230 compiled rules from
`xlog-integration/tests` (47 files), on the XLOG fork at `656def9`.
Disable runs: 6 of 6 probe controls red, 3 of 3 cube controls red, each on
exactly its own test.

## Result

Every compiled rule now has a verdict.

| rung | D-XLOG-COV-0 | now |
|---|---|---|
| refused (empty bases) | 90 | 94 |
| refused-on-data | 49 | 49 |
| lowered-unverified | 20 | **0** |
| oracle-disagrees | 0 | **0** |
| oracle-agrees | 71 | 87 |
| native-tested (agrees, no host loop) | 32 | 43 |
| production-admitted | 0 | 0 |

The 4 extra refusals are not regressions. They are rules that used to run
over an empty stand-in for a refused relation, and are now refused for it.

## What the 20 were

| cause | rules | outcome |
|---|---|---|
| one non-binary base made the whole program unjudged | 10 | judged per dependency: agree, or refused because they read that base |
| generated query heads, with no AST rule | 7 | judged through the rule the compiler generates |
| a seed over the naive oracle's bound | 2 | that seed skipped; the small dense seed (0..4) judges `clique7` and `clique8`; without it both fall back to agreement on empty answers |
| facts on a derived predicate | 1 | judged with the facts: exposed cube defect 2 |

## Two cube defects, fixed in xlog#8

1. **A reader of a refused relation ran.** Every arity-2 relation gets an
   empty stand-in, so a rule reading a relation whose SCC was refused ran over
   the stand-in and reported a wrong answer as `ran`. Example:
   `executor_config_tests.rs` literal 5, where the cube returned `{}` and the
   right answer is `{(1,20), (2,10)}`. Now refused with
   `reads refused relation `keep``.
2. **Rows put for a derived predicate were dropped.** Example:
   `test_same_head_union_batching.rs` literal 2, where the cube returned `{}`
   for `reach` and XLOG's own test expects `{(1,2), (1,3)}`. These rows now
   seed the SCC.

Both were hidden by the same thing: the probe marked these rules unverified
instead of judging them. The admission census (`coverage.rs`) cannot see
either one.

## Merge order

The probe depends on XLOG `main`. Merge xlog#8 first, then this lance-graph PR.
Merged the other way round, the probe's re-pinned tests fail against the old
cube. They fail with the two disagreements above, so the failure is loud, not
silent.

## Open, unchanged

The three refusal clusters from D-XLOG-COV-0 (n-ary relations, column
construction and projection, predicates and join shapes) account for the 143
rules refused, on empty bases or on data. W5 (`Fixpoint`) and W7 (F100) remain
undecided. Bag counts are still not checked, and correctness is still shown
only over the 8 seeds and this corpus.
