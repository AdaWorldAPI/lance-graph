# S2 — upstream admission gate (`vrl-vector-openobserve-census-v1`)

**Status:** In PR.

`Recipe::admit(src, view, class, schema)` runs behind the existing
`upstream-vector` / `upstream-o2` features. It compiles `src` with upstream VRL
against the declared schema, then lowers it with `Recipe::compile`. A program
either side refuses never runs:
- An upstream refusal is `Refusal::Upstream { codes }`, upstream's diagnostic
  codes, sorted and deduplicated.
- The lowering's refusals are unchanged.

The schema is a list of `Declared { label, kind, optional }`. `key` sits at
the top level; every other field is at `.properties.<label>.value`. An
optional field is `null | undefined`.

`Refusal::Upstream` is added unconditionally, so the enum has one shape under
every feature set. No crate in the workspace matches `Refusal` exhaustively.

## Measured (both pins, `tests/admission.rs`)

- `if .properties.cpu.value > 3 { abort }`:
  - over an optional integer: refused, `codes == [100]`;
  - over a non-optional integer: admitted, lowering to the same outputs and
    field mask as `compile`.
- `.k = .key + 1`: upstream compiles it, but the lowering refuses
  (`Computes`). `.k = no_such_fn(.key)`: upstream refuses with `[105]`.
- With the presence lanes optional, the gate admits exactly `presence.rs`
  programs 0 and 2. The other three are refused with `[100]`.

## Bearing on S1

S1's fail-fast class (Drop → Unknown, 79 rows) occurs only in program 4,
which the gate refuses. On admitted programs, the only divergence left is
`==`/`!=` over an absent value (program 2, Unknown → Drop, 60 rows). That is
exactly Q1.

## Ablation

With the gate bypassed (`admit` returns `compile`), all 3 admission tests
fail.

## Also in this PR

The suites' program lists move to `tests/common/programs.rs`, carried from
CodeRabbit's nitpick on #2200. A program added to `recipe.rs` or
`presence.rs` now reaches the differential; the copy guard is gone.

OPEN:
- Q1 only: refuse `==`/`!=` over optional fields in the lowering, or adopt
  null semantics in a `VrlCompat` mode.
