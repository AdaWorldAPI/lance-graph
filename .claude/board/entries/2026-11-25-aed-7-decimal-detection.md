# 2026-11-25 — D-AED-7: schema-on-write exact-decimal detection

**Status:** TEST-PINNED (probe). Plan: `.claude/plans/adaptive-exact-decimal-binding-v1.md` D-AED-7.

`detect_decimal` / `merge_decimal` in `crates/exact-decimal-probe`:

- A text column reads as an exact decimal when every present value parses
  exactly at one exponent within `i32` and some value has a decimal point.
  Recorded: the coarsest common exponent, the coefficient range, the width,
  provenance `Lexical`, the support.
- Refused, with the first row: exponent notation or stray characters
  (`Malformed`), a coefficient past `i32` (`Overflow`), and a binary-float
  artefact (`FloatArtefact`: more than 15 significant digits that are
  exactly the shortest `f64` rendering, as `0.30000000000000004`).
- A declaration wins (text, a decimal exponent, or the explicit `f64`
  method), but a declared exponent coarser than the data is refused
  (`NotExact`), never rounded. Integer-only columns stay with intake
  (`NotDecimal`), and intake still detects decimal text as text.
- Gate samples: log durations, JSON CPU percent, Graph-report storage —
  detection agrees with the declared exponent on all three.

Disable run: with the artefact rule off the artefact column is still refused,
as `Overflow` at row 0, not `FloatArtefact` at row 1. In the `i32` carrier a
17-digit artefact cannot be read as a decimal anyway; the rule names the
reason, which matters once a wider carrier exists.

OPEN: production admission stays closed (D-LANESOCK); intake's `PgType` has
no numeric type, so the reading lives in the probe.
