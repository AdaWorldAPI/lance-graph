# 2026-11-28 — VersionedAppend: the stale-checked Lance commit over any caller's records

**Status:** TEST-PINNED (`crates/lance-graph-dir-sim-lance/tests/versioned_append.rs`, 4 tests; disable run of the stale check: 2 red). Board: EXO-483, EXO-367.

`LanceLog`'s commit discipline (one batch = one Lance commit, a stale writer refused before it
writes, a race reported after the fact, the logical version kept in the records and never in the
Lance version) is lifted into `lance_graph_dir_sim_lance::append::VersionedAppend`, with nothing
of the dir-sim log in it. `LanceLog` now holds a `VersionedAppend` and keeps its own law (batch
number) on top; its 6 tests are unchanged and green.

The tests use a document version log (doc, n, sha, parent), the shape report-rs's versioned
documents need: two writers at one version, the second writes nothing; only one writer creates
the dataset; two documents at logical version 1 are commits 1 and 2; a batch of another schema is
refused by Lance and nothing lands.

Not added: an idempotence rule, a receipt type or a log format. Whether a record may be appended
(its parent is the head) stays the caller's own check.

OPEN: the stale check and the commit are two steps; a writer landing between them is reported as
`Race` after its batch is stored, not prevented. No report-rs consumer is wired yet.
