# 2026-11-25 — D-AED-8: VRL decimal literals over decimal fields

**Status:** TEST-PINNED (probe). Plan: `.claude/plans/adaptive-exact-decimal-binding-v1.md` D-AED-8.

- `DecimalRecipe::compile_vrl` (`crates/exact-decimal-probe`) takes plain VRL
  source. `if <path> <op> <literal> { abort }`, where the path is a field
  (`.properties.<label>.value`) or a name assigned from one, becomes an exact
  decimal filter when the owner declares the field's exponent; its literal is
  resolved by `threshold` at that exponent, never through `f64`. Every other
  statement is plain VRL; a decimal literal on a field with no declared
  exponent is still refused by VRL.
- `DecimalRecipe::sum` adds `SUM` under the same exponent, width and frame
  checks as `count`.
- Gate (`a_vrl_decimal_literal_equals_the_oracle_and_is_witnessed`): counts
  and sums equal the exact per-value oracle for `>=`, `<`, `==`, `!=` on the
  #2182 fixture and equal the structured `DecimalFilter` path; a moved
  exponent is refused. Disable run (literal passed to VRL as is) is red.

OPEN: Vector JSON ingress (D-VRL-RECIPE-0) is the other value source; this
probe reads values through intake only. Lowering the rewrite into
`lance-graph-vrl` itself is not done: production decimal admission stays
closed (D-LANESOCK).
