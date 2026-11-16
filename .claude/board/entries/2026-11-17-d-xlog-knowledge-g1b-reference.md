# D-XLOG-KNOWLEDGE-G1b: expected answers for refused rules (2026-11-17)

**Status:** MEASURED, lab only. Plan `.claude/plans/xlog-knowledge-spine-v1.md` §4 G1.

## Why

The Datafrog column only exists for rules the cube ran, so after G1 it could
only ever report on the 87 oracle-agrees rows. Widening Datafrog's subset
would have changed no number. The 143 refused rules had no answer from either
side.

## What

- A reference stage runs the naive oracle and Datafrog over the same seeds for
  every compiled rule, with no cube. Columns `reference`, `reference_nonempty`,
  `reference_reason`. It never decides a rung.
- The census then showed 7 rules the naive oracle answers and Datafrog could
  not express. Datafrog gains comparisons (filtered join against a one-row
  relation), anonymous variables in negated atoms (anti-join on the named
  columns) and rules with no positive atom (start from one empty row).

## Result

Census at lance-graph this branch, xlog `beac406` (tree of `656def9`):

| rung | reference |
|---|---|
| oracle-agrees 87 | agree 87 |
| refused-on-data 49 | agree 49 |
| refused 94 | agree 35 (28 before the wider subset), agree on empty answers only 15, unavailable 44 |

0 disagreements. Rungs, reasons and the datafrog columns are byte-identical to
the committed census. Unavailable: floats 16, `is` expressions 15, ILP
`learnable` rules with no ordinary AST rule 5, bases with non-integer facts 5,
aggregates 3.

## Falsifiers

- Datafrog join reads the right row from the wrong end: 114 reference
  disagreements, 43 on refused rules; rungs and reasons unchanged.
- Comparison filter ignored, wildcard anti-join over the full row, no unit row:
  each turns `comparisons_negated_wildcards_and_ground_rules_agree_with_naive`
  red.
- Reference stage skipping refused rules: `a_refused_rule_has_a_reference_answer`
  and `an_empty_reference_agreement_is_counted_as_empty` red.

## Open

- The 44 unavailable refused rules need floats, `is` and aggregates in both
  evaluators; the 15 empty-only agreements need seeds that make them produce
  rows.
- The reference answer is what the cube should produce once it admits a rule;
  admitting the three refusal clusters is still W5/W7.
