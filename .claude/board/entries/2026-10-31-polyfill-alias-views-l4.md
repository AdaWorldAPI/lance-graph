# D-DT64-7 L4: production alias views for the polyfill lens — 2026-10-31

**Status:** MEASURED (`crates/lance-graph-r2il/tests/integration/polyfill_views.rs`). First step of the execution order in plan §1d.

- Each `POLYFILL` word records which of its existing aliases is its SQL name and which its Quack name (`sql_at`, `quack_at`); no name is spelled twice and no spelling rule picks the language. `FrozenWord::{name_of, name_in, is_alias}`.
- `PolyfillView::new(AliasLanguage)` answers `0xFFFF` with one label per position. The `predicate_iri` names the word (`lg:polyfill:f000/<R2IL name>`), the label the language; a position with no name in that language carries its R2IL name (SQL: `SUM`, `MIN`, `MAX`, `ALL`, where SQL's empty-selection `NULL` differs).
- `WithPolyfill::new(&consumer, &view)` answers `0xFFFF` from the polyfill view and delegates every other concept, template and DOLCE category to the consumer's view. It checks no activation (L3).
- Measured: R2IL, SQL and Quack views over the same consumer view bind the same steps and plan id; the consumer's own stale `0xFFFF` labels are refused alone and hidden behind the adapter.
- Disable runs, all red: SQL reading the Quack index (2 fail), one wrong index (2), no R2IL fallback (3), the polyfill view answering every concept (1), the adapter not routing `0xFFFF` (2), a per-language IRI (1), the adapter dropping the consumer's DOLCE category (1).
- Unchanged: `binding.rs`, the `0xC401` reading, the 9 existing polyfill tests.
