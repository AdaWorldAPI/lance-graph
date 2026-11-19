# N-1 first receipt: source rows written by ruff open in the Codegraph reader (2026-11-19)

**Status:** MEASURED (test-pinned). Stage N-1 of `codegraph-behaviour-v1.md`
(source facts as graded store rows), one corpus shape, synthetic.

ruff `crates/ruff_codegraph` (ruff branch `claude/great-bohr-gwiktx`,
`1778245`, draft PR) projects a Rails `ModelGraph` into `<ns>.rows.tsv` v2,
`.files.tsv` and `.residuals.tsv`. Its rev1 fixture is copied to
`crates/lance-graph-codegraph/tests/data/ruff_codegraph/`.

`tests/ruff_codegraph_store.rs` (4 tests, `cargo test --test ruff_codegraph_store`):

- `Codegraph::open` accepts the store unchanged: 73 rows, crate `shop`, `Partial`.
- Predicates outside the reader's nine are kept verbatim and countable
  (`CALLBACK` 5, `VALIDATES` 4, `ASSOCIATION` 4, `IMPORTS` 4, `CALLS_NAMED` 1).
- The name-matched call is a row only: `called()` is empty, `CALLS_RESOLVED` 0.
- Residuals: `CALLEE_NOT_NAMED` 2, `REFERENCE_NOT_NAMED` 2.
- The store opens beside a `ruff_charon` store (`fnref`) in one directory.

No reader code changed. ruff does not depend on lance-graph; the boundary is
the file format.

OPEN:
- The writer's ordered lane (`events.tsv`) and `receipt.json` have no reader
  here. Whether the order lane is read by this crate or stays with crai-rs is
  undecided.
- Whether name-matched calls should enter the graph as a possible relation
  (`MAY_DISPATCH` carries " via "; reusing it would change its meaning) needs
  a decision before any impact query uses them.
- Spans are `-` for every source row: the Ruby frontend reports none.
- Not run on a real corpus.
