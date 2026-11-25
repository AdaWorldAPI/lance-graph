# D-VEC-GLUE-1 rework — in-process handoff, nothing serialized between Vector and lance-graph

**Status:** In PR.

DECISION (operator, 2026-11-25): Vector and lance-graph are compiled into one binary, and nothing is serialized between them. Vector keeps its own external intake (`datadog_agent`, OpenTelemetry sources); that is Vector's job and it is unchanged.

## Convergence crate (measured)

Dependency counts, `cargo tree -e normal`, deduplicated:
- `vrl` with the `value` feature: 107 crates.
- `vrl` with `compiler` and `stdlib-base`: 223.
- `vector-core`: 385.

`vector-core`'s `LogEvent::value()` returns `&vrl::value::Value`, so lance-graph converges on `vrl` (`value` feature only). `vector-core` stays in the glue on Vector's side.

A binary linking `vector-core` and `lance-graph-vrl` held two `vrl` crates: `vectordotdev` at `branch=main`, and the AdaWorldAPI fork pin. That means two distinct `Value` types. A `[patch."https://github.com/vectordotdev/vrl.git"]` entry pointing at the fork at Vector's locked rev 7738f74 collapses them to one (verified with `cargo tree`).

## What changed

- `lance-graph-vector`:
  - **Removed:** the NDJSON receiver, `serde_json`, `Cell::Number` and `Ack::status`.
  - **Added:** `receive_values(&[&Value])`, which reads Vector's values in place; `receive_events` keeps the typed cells.
  - **Event id:** a 128-bit FNV-1a hash in four `Int8`/unsigned lanes. Dedup is a hash set on the route, with no unique-string dictionary.
  - **Refused:** a float, which is an `f64` inside Vector already.
- `lance-graph-intake`:
  - **`Table::append_values(&[Vec<Option<Datum>>])`**, with `Datum` being `Int`, `Bool` or `Text`. Values are placed from what they are, never rendered to text and parsed back. It shares placement, keys, tiles and codes with `append`.
  - **`Table::check_values`**, which checks fit without appending.

## Tests and disable runs

- `lance-graph-vector`: 6 route tests, including polyfill convergence, retry, in-batch repeat, misfit replay, schema rev, unknown field, nested value, float, typed-equals-values, and a Quack bind.
- `lance-graph-intake`:
  - `typed_append_equals_text_append`: same PostgreSQL-shaped answers as the text append.
  - `typed_append_refuses_what_does_not_fit`.
- Both disable runs went red: dedup off, and the duplicate fit check off.

## Measured

Floor, reading `&Value` in place into lanes: 30 ns per event (123 µs per 4,096-event batch).

The route itself does not meet the gate yet. Intake's append leaves the fold at the per-key tile scan; see ISS-INTAKE-APPEND-LEAVES-THE-FOLD.

OPEN:
- ISS-INTAKE-APPEND-LEAVES-THE-FOLD.
- The Vector-side sink (`lib/lance-graph-sink` in the Vector fork) now reduces to `route.receive_values(log.value())` per batch. There is no push access to the fork from this session.
