# Frontend parity witness: Gremlin and SQL meet at the same Quack `Query` (2026-10-03)

**Status:** MEASURED (SQL ↔ Gremlin); ANALYSIS (Cypher against the #1306 plan,
SurrealQL against its AST). Plan: `.claude/plans/frontend-parity-witness-v1.md`.
Code: `crates/lance-graph-quack/tests/gremlin_parity.rs` (test-only, 13 tests).

- A typed Gremlin step vocabulary lowers onto `lance_graph_quack::Query` through a
  test-only adapter. For the shapes that lower, the Gremlin `Query` is `==` the
  SQL-shaped `Query`, and execution equals DuckDB's committed answers
  (`join_sum_country` 1237848, `join_group_count_country`,
  `join_count_docs_with_posted` 511 via the oracle) and an independent
  row-at-a-time bulk oracle.
- **Anchor rule:** a functional hop keeps the anchor (fk reads); one fan-out
  re-anchors on the table whose rows are the paths (child or edge table). One M:N
  hop is therefore one `Program` over the edge population, with exact bag counts.
- **Input for #1306 D-CML-2:** RF-BAG refuses any non-DISTINCT count after a hop.
  After a functional hop, or one hop over an edge population, the bag count is
  exact (anchor rows = paths). RF-BAG is broader than the semantics require.
- **Gaps passing the two-witness rule** (none built): composed functional read
  (= #1308 (a)); barrier + workspace after a fan-out (= #1308 (b)/(d), #1310);
  ordered compare through an fk; sum of a foreign value; ordered compare on `u32`.
- **SurrealQL:** a hop is `Part::Graph(Lookup)`, a per-hop mini-SELECT over a
  first-class edge table (`in`/`out`), bag-flattened. Structurally the same as the
  edge-population re-anchor. Not reusable as a dependency: `sql::*` is
  `pub(crate)`, the public arena AST is unpublished, license is BSL 1.1.
- DuckGQL / DuckPG: no code or plan by those names in the tree.
