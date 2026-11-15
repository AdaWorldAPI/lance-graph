# D-XLOG-KNOWLEDGE-G1: census reproduced, Datafrog as an independent comparator (2026-11-15)

**Status:** MEASURED, lab only. Plan `.claude/plans/xlog-knowledge-spine-v1.md` §4 G1.

## Reproduction

`cargo run --release --manifest-path crates/xlog-coverage-probe/Cargo.toml --example reconcile` over
`xlog/crates/xlog-integration/tests/*.rs`, lance-graph `bdee050`, xlog `beac406`
(merge of xlog#8; tree identical to `656def9`). After normalising the path
prefix the output equals the committed `census-656def9.tsv` line for line:
230 compiled rules = 87 oracle-agrees (43 without a host loop), 94 refused,
49 refused-on-data, 0 disagree; 208 literals never compile, 202 at the EIR
boundary.

## Datafrog arm

`crates/xlog-coverage-probe/src/datafrog_arm.rs` evaluates each component with
Datafrog (`AdaWorldAPI/datafrog` `edd7cfc`, v2.0.1): every body join becomes a
keyed `from_join`, every negated atom a `from_antijoin` against a relation of a
lower component. Subset: positive and negated atoms over distinct variables and
`_`; no constants, comparisons, aggregates or other literals; negation only
outside the component. Anything else is `not-applicable` with a reason. It
shares no code with the naive oracle, XLOG's reference evaluator or the cube,
and never decides a rung. Two TSV columns: `datafrog`, `datafrog_reason`.

| oracle-agrees rows (87) | count |
|---|---|
| datafrog agree | 85 |
| datafrog not-applicable (integer constant in a body atom: `neighbor`, `from_one`) | 2 |
| datafrog disagree | 0 |

Rungs: byte-identical to the committed census over the original 9 columns.

## Falsifiers (disable runs, each red, anchor asserted, restored from commit)

- D1 join takes the wrong right-hand column: 7 of 8 arm tests red; in the
  census 68 of 87 rows `disagree`, 0 rung changes.
- D2 anti-join replaced by a map (negation ignored): stratified-negation test red.
- D3 own-component negation guard removed: out-of-subset test red.
- D4 `not-applicable` labelled `agree`: the `reconcile` label test red.

Tests: 8 in `tests/datafrog_arm.rs` (closure, stratified negation, cross
product, triangle, four out-of-subset cases with a silence twin, corrupted rule
body, swapped base columns, dropped seed fact, labels through `reconcile`).

## Open

The 2 not-applicable rules need constants (a filter) in the arm; G2..G8 untouched.
