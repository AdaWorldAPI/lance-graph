# 2026-11-25 — D-RSP-3: measure law, declaration half

**Status:** TEST-PINNED (declaration); enforcement OPEN. Plan: `.claude/plans/registered-source-profiles-v1.md` D-RSP-3.

- Read first (census OPEN 7): OGAR has no measure law. `unit_of_measure` in
  `ogar-class-view` is an odoo field, not a law; `AlgebraLaw` and
  `set_rules::Semantics` describe merges, not measures.
- Declaration: `PropertySpec.measure: Measure` beside `semantic_type`
  (`Undeclared` default, `Additive`, `Snapshot`, `Identifier`), built with
  `const fn with_measure`; adding the field breaks no caller (no struct
  literal of `PropertySpec` exists in any repo, only a doc comment).
- Law: `Measure::admits_sum(Spread)`. A snapshot summed across instants is
  `AcrossInstants`; seats prepaid summed across SKUs at one instant is
  admitted; an identifier has no sum; an undeclared law admits nothing.
  Gate test `a_sum_is_lawful_only_under_its_declared_measure`; disable run
  (snapshot admitted across instants) is red.
- Premise correction to candidate (b): `MethodBinding` is DT64's ON / WHEN /
  DO trigger method (`on`, `when`, `reads`, `effects`); it has no aggregate
  admission. Registering lawful aggregates as `MethodBinding`s would give that
  type a second meaning, so the law is a method on `Measure` instead.

OPEN: enforcement in the frontends. A sum in VRL or Quack must ask the
field's `Measure` and the population's `Spread`, which needs the field seam
(D-RSP-5, `predicate_iri → column, code space, exponent, validity`) to also
carry the measure, and a population's `Spread` from its time-field terms.
Intake attaches no `PropertySpec` to its columns yet.
