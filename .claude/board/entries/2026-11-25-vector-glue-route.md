# D-VEC-GLUE-1 — `lance-graph-vector`: lg-vrl as Vector glue, the NDJSON / typed route

**Status:** In PR. **Plan:** `vrl-vector-openobserve-census-v1` (S4 receiver).

One VRL program, two engines. In-process, `lance-graph-vrl` runs it in-fold
over intake lanes (the polyfill). When Vector runs it, lg-vrl is the glue:
`output_profile(recipe, source, class)` declares the profile the output lands
in, with one column per recipe output. Each column carries the source field's
IRI (a rename keeps identity) and a fully declared `ColumnConfig`, followed by
the reserved `lg.event_id` and the key column `lg.seq`. An output over an
undeclared source field is refused, so the receiver checks the shape and never
detects it.

`VectorRoute` lands into a `Table::from_rows_in` table under that profile.
There are two front ends:
- `receive(ndjson)`, for Vector's http sink, where numbers keep their raw text
  and are never parsed as f64;
- `receive_events(&[Event])`, for a sink inside Vector, with no JSON.

Both go through one all-or-nothing admit:
- a `lg.schema_rev` mismatch, an unknown field, a nested value, a missing event
  id or a value that does not fit its column → 400, nothing appended;
- dedup through the table's code table (`code_for(frame, "lg:event_id", id)`)
  plus an in-batch set, so a retried batch commits 0 new rows → 200.

The path is VRL normalize → DR frame / CodeTable → BindStore (via the shared
`SourceProfile`) → Quack (the landed table binds `draft(..).where_eq / where_cmp`).

Tests (`crates/lance-graph-vector/tests/route.rs`, 5):
- profile IRIs and the undeclared refusal;
- a hand-written Vector output lands equal to the polyfill over the same raw
  events;
- retry, in-batch duplicate, schema_rev, unknown, misfit and nested refusals;
- typed equals NDJSON, including an absent cell;
- Quack counts over the landed table.

Disable run: with dedup off, the retry assertion fails.

OPEN:
- How upstream VRL treats `null < 0` (the polyfill drops the row). This is the
  S1 differential's question, so the test keeps absent amounts out of the
  convergence case.
- The in-Vector sink crate (`vrl::value::Value` → `Cell`): there is no push
  access to the Vector fork from this session.
