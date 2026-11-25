# D-VEC-ENRICH-1 — Vector's enrichment tables answered by an intake table through Quack

**Status:** In PR.

Vector's internal store that VRL reads is its enrichment-table registry. `get_enrichment_table_record` and `find_enrichment_table_records` call `enrichment::Table` (`lib/vector-vrl/enrichment`, whose only dependencies are `vrl`, `chrono`, `dyn-clone`, `snafu` and `arc-swap`). `crates/lance-graph-enrichment` implements that trait as `QuackTable` over an intake `Table`, in the same process as Vector:

- **Conditions:** bound once per lookup through Quack's binder (`draft(..).where_eq`). The positions they hold come from the new `Table::positions_bound(&self, &BoundQuery)`, the read half of `select_bound` with no kept selection.
- **Rows:** read from the lanes at those positions by a projection recipe (`lance-graph-vrl`, compiled once per table). Each value is handed to VRL as its own `vrl::value::Value`, chosen by lane encoding: text code → text, signed or unsigned → integer, bool → boolean, epoch → timestamp, absent → null. Never through text egress (`PgRows`).
- **Refused, with the enrichment error that names it:** a date range (`OnlyEqualityConditionAllowed`), an unknown field (`MissingDatasetFields`), and a case-insensitive lookup, a wildcard, a null or a float condition (`FailedToEncodeValue`).
- **A text value the codebook does not hold** answers no rows, not an error.

The root `Cargo.toml` patches `vectordotdev/vrl` to the AdaWorldAPI fork at Vector's locked rev, so a crate that links `enrichment` holds one `vrl`.

## Tests (`tests/lookup.rs`, 3)

- Typed rows for equality on text, integer and bool, a `select` subset, and an absent cell as null.
- The refusals above, plus none versus several rows for a single-row lookup.
- VRL's own `get_enrichment_table_record!`, compiled against Vector's `TableRegistry` holding the intake table, returns the same record.

Disable run: with the conditions dropped, 3 of 3 fail.

## Not done

A Lance-backed adapter: intake tables are not persisted to Lance today (see ISS-DR-FRAME-TO-LANCE-VERSION-HAS-NO-WITNESS). Vector's disk buffers stay as they are; they are a store of serialized events.

OPEN:
- Each lookup binds its literal again (per event). No reuse of a bound query across events has been measured.
