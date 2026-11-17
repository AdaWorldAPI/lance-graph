# D-VRL-RECIPE-0: a stored late-materialization recipe from a VRL subset (2026-11-17)

**Status:** MEASURED, lab only (`crates/vrl-recipe-probe`, own workspace).
Follows the Vector/VRL adapter contract in
`2026-11-16-g2-dr-virtual-projection-reconcile.md` §D.

## Claim tested

Whatever the size (64, 256, 512, dynamic), an answer has one shape:
`identity | select {name = identity.key; identity.properties.f.value}`. So
the stored answer can be a recipe, never rows: the store version it was
compiled against, the population terms, the participating fields as a
`WideFieldMask`, and the output aliases. Width only decides how many mask
words a terminal allocates.

## What

- VRL subset, compiled once against the store schema: `.out = .key`,
  `.out = .other`, `.out = .properties.f.value`, `del(.out)`,
  `if .p == v { abort }`, `if .p != v { abort }`. Paths are resolved at
  compile time with VRL's in-place mutation semantics (a reassigned `.key`
  is what later reads see).
- Output semantics, stated: the names assigned and not deleted afterwards.
- Refused by name: function calls other than `del`, arithmetic, string and
  integer literals as values, comparisons other than `==`/`!=`, an `if`
  body other than `abort`, assignment into `.properties`, unknown paths.
- Terminals: `count` (popcount), `sum(output)` (lane read in place at set
  bits), `materialize_rows` (the only row writer, O(rows kept)). Every
  terminal refuses a recipe compiled against an older store version.
- Oracle: a naive VRL interpreter that materializes every row as an event,
  runs the statements, and projects. It shares the parser only.

## Result

- 7 programs × widths 0, 1, 63, 64, 65, 256, 512, 1000: rows (in row order),
  count and the sum of every output equal the interpreter.
- Each filtered program keeps strictly fewer of 512 rows than its unfiltered
  form.
- `count` and `sum` make exactly one allocation, `8 · ⌈n/64⌉` bytes (the
  mask), at n = 64, 512, 4096, 65536; `materialize_rows` makes more
  allocations than rows kept. These are allocation counts, not timings.
- 12 refusal cases return the named refusal.

## Falsifiers

Each patch anchored (`count == 1` asserted), run, restored from the commit:

| disable | red |
|---|---|
| `count` by `materialize_rows().len()` | `folds_allocate_only_the_mask` |
| version check dropped | `a_recipe_is_refused_after_the_store_moves` |
| `== v { abort }` terms ignored | `recipe_equals_the_interpreter_at_every_width`, `every_filter_drops_rows` |
| `.key` always reads the source lane | `recipe_equals_the_interpreter_at_every_width` |

Two first attempts (count, rebinding) did not apply because `cargo fmt` had
moved the anchor text; their green runs were discarded, re-anchored, and the
recorded rows are the applied runs.

## Open

- Population terms are the probe's own `== v` list; the cube's `SetExpr` is
  the native population and would replace them.
- Ingress from real Vector events (JSON) is not built; this probe starts
  from columnar lanes.
