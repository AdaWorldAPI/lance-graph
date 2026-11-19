# D-GPW, first step: a `LIKE` pattern is resolved at Quack's bind (2026-11-19)

**Status:** MEASURED (test-pinned). No timing taken: binding is cold, once
per binding, and nothing in `lower` or the executor changed.

Decision (owner, 2026-11-19, verbatim): "Or just inside quack Easy and
simple", then "Go ahead inside quack". Plan:
`.claude/plans/global-pattern-words-v1.md` §5b.

`crates/lance-graph-quack/src/bind.rs`:

- `Pattern`, with `Pattern::like` (SQL's spelling: `%` any run, `_` one
  character, `\` escapes, whole value, case-sensitive) and `matches`.
- `Op::Like`, `Draft::where_like`, `FieldRef::like`.
- `Binder::codes_matching(table, field, &Pattern) -> Option<Vec<u32>>`,
  default `None`. The binder owns the codebook, so it walks its own values;
  nothing is minted and no text leaves it.
- The bound `Query` holds the covered identities as `Filter::in_u32`, the
  membership filter that already existed. No new executable leaf, no new
  opcode; `tests/string_fence.rs` still passes (text only in `bind.rs`).

The absent case, stated in the type's doc and pinned by a test: a pattern
that covers no value binds to a filter that keeps no row; `field = 'x'` for an
`x` the domain has never seen is still refused (`UnknownValue`). A pattern
states a shape, not a member.

Tests (crate: 170 passed, 4 new; clippy `-D warnings` and fmt clean):

- five patterns over the test table each keep the rows a scalar count gives
  (67, 199, 66, 132, 66 of 199 live rows), and one bound query equals the
  membership filter written by hand;
- an empty cover counts 0 and returns no row; the equality twin is refused;
- 21 `LIKE` cases against SQL's semantics, including escapes and non-ASCII;
- a bad pattern, a pattern on an integer field and a binder that cannot
  enumerate are each refused by name; the same binder still binds `=`.

Disables, each red: `_` matches any run; an empty cover keeps every live row;
the escape is ignored; the test binder returns every code.

Open:
- Only the within dimension, and only SQL's spelling. The across dimension
  (a hole bound through a relation) is not touched; it lowers to
  `Semijoin` / `EqU32Via`, which are gated by EXO-140.
- The word has no position in `POLYFILL` (decision 1 of the plan).
- A pattern covering most of a domain binds to a long `OR` of equalities.
  Not measured.
- The 21 `LIKE` cases are a hand-written table, not a differential run
  against PostgreSQL. zammad-rs has that oracle and is the place to run it.
