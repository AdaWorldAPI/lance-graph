# D-GPW: `lance-graph-spacy`, spaCy's Matcher token-pattern format as a front end (2026-11-19)

**Status:** TEST-PINNED against a table read from spaCy's source. **NOT run
against spaCy.** Scope is a pending decision (EXO-345); this is its scope 1.

Decision (owner, 2026-11-19, verbatim): "feel free to add Haiku agents for
harvesting spaCy and Sonnet agents for drafting Lance-graph-spaCy as
polyfill".

`crates/lance-graph-spacy` (its own workspace root, like `lance-graph-metta`;
dependency `serde_json` only):

- `TokenPattern::parse` reads a pattern as spaCy writes it, a JSON list of
  token-spec objects. Supported: `ORTH` / `TEXT`, `LOWER`, `LENGTH`, `IN` /
  `NOT_IN`, the `LENGTH` comparisons, and every `OP` of `_get_operators`
  (`!`, `?`, `+`, `*`, `{n}`, `{n,m}`, `{n,}`, `{,m}`).
- `spans` is spaCy's result without `greedy`, minus the match id; `covers`
  and `matches_value` ask whether the whole value matches.
- A token is one part of a value under a declared `Split` (one separator,
  empty parts kept). No document, tokenizer rule, vocabulary or model.
- Refused by name: 32 token-spec keys spaCy's schema has (the annotation
  attributes, the `IS_*` and `LIKE_*` flags, `_`) and 14 predicates (`REGEX`,
  `FUZZY`, `FUZZY1`..`9`, `IS_SUBSET`, `IS_SUPERSET`, `INTERSECTS`).

How it was made: three Haiku mappers read spaCy 3.8.18 (MIT) into fact maps
(matcher grammar; phrase and dependency matchers; rulers, tokenizer and
attribute table), a Sonnet worker drafted the crate from a brief that named
every type, and the orchestrator compiled and tested. The orchestrator
spot-checked the keys and the 20 dependency operators against the source.

Tests (`cargo test --manifest-path crates/lance-graph-spacy/Cargo.toml`: 6
passed; clippy `-D warnings` and fmt clean):

- the operators expand as `_get_operators` does, with its refusals;
- `spans` over `["a","b","b","c"]` equals the listed spans for `+`, `*`,
  `{}`, a bare `+`, `?` and `!`;
- the conditions, including a non-ASCII token for `LENGTH`;
- what spaCy has and this does not is refused by name;
- under a `.` split, `[{"ORTH":"admin"},{"OP":"+"}]` agrees with Quack's
  `Pattern::like("admin.%")` on nine values, both true and false occurring.

Disables, each red: `!` read as an ordinary token; `?` must consume a token;
`*` stops after one token; the split drops empty parts.

What it does NOT show:

- **No differential oracle.** Installing and running spaCy was denied by the
  session's permission system, and that was not worked around. Every
  expectation is a table derived from reading `matcher.pyx`, cited by line in
  the code. `!` in particular rests on one reading of `get_action`
  (matcher.pyx:690-692).
- `LOWER` uses Rust's `to_lowercase`; equal to Python's `str.lower()` for
  ASCII, not verified beyond. The `IS_*` flags were left out for the same
  reason.
- Stricter than spaCy in named places (a falsy non-object spec, an unknown
  predicate, a negative or float `LENGTH`, a repeat count above 4,096).
- Not connected to Quack's bind: `Binder::codes_matching` takes Quack's own
  `Pattern`. A binder can call `TokenPattern::matches_value` itself; a common
  type for the two spellings is undecided.
