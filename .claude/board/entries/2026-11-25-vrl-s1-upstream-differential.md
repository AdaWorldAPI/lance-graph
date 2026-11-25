# S1 — upstream VRL differential (`vrl-vector-openobserve-census-v1`)

**Status:** In PR.

`crates/lance-graph-vrl/tests/upstream.rs` runs every program of
`recipe.rs` (10) and `presence.rs` (5) through upstream VRL. Each is compiled
against a typed `ExternalEnv` (every lane a non-optional integer) and run one
event per row. An absent value (presence 0 and value 0, EXO-294) becomes a
missing `value` key. Each row's KEEP/DROP/UNKNOWN is compared with upstream's
ok/abort/error, and on kept rows each output is compared value by value.

Pins are optional features on the AdaWorldAPI fork, so a default build never
compiles VRL:
- `upstream-vector` is vrl 0.39.0 at 7738f74, Vector's lock at c879bd6.
- `upstream-o2` is vrl 0.31.0 at 4d61e53, OpenObserve's.

The workflow `vrl-upstream-diff.yml` runs both pins. It is manual-only, like
the other workflows here.

## Measured

Fully present lanes, at 1, 64, 65 and 256 rows: every row agrees on both
pins, outcomes and outputs alike.

Presence suite (300 rows × 5 programs): 210 rows diverge. The counts are
identical on both pins, so 0.31 → 0.39 shows no drift for these programs
(data for Q2). Every divergent row has an absent value; this is asserted.

| program | recipe → upstream | rows | why |
|---|---|---|---|
| p2 `if .status != 2 { abort }` | Unknown → Drop | 60 | upstream compares absent as null: `null != 2` is true |
| p3 `if .region == 0 { abort }` | Unknown → Keep | 40 | `null == 0` is false |
| p4 `… .status > 2 … .a <= 10 … .r != 1` | Drop → Unknown | 79 | upstream stops at the first erroring statement (an absent value under an order compare); the recipe's three-valued rule lets a later decided `abort` win |
| p4 `if .r != 1 { abort }` | Unknown → Drop | 31 | as p2 |

The third row is a class the plan did not list: it is not about `==`/`!=`.
Upstream is sequential and fails fast; the recipe is order-free. It bears on
Q1 and on S2's admission gate. A program with an order compare over a
possibly absent field before an `abort` decides rows differently.

## Ablations

Both were run red, then restored:
- Gt lowered as `GeI32`: 2 of 3 tests fail.
- An untyped `ExternalEnv`: upstream refuses the order compares with E100
  ("unhandled fallible").

OPEN:
- Q1: a `VrlCompat` mode, or refusal. The Drop → Unknown class means
  refusing `==`/`!=` alone would not close the gap.
- Absent rendered as JSON `null` rather than a missing key is not run.
