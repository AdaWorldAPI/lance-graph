# D-DT64-7-L1: behaviour lenses read their basis by the full classid — 2026-10-31

**Status:** MEASURED (`crates/lance-graph-dt64/tests/bind_store.rs`, `crates/lance-graph-r2il/tests/integration/polyfill*.rs`). Implements decision O9 (plan §1d), step 2 of the execution order.

## Contract change and its compatibility impact
- `lance_graph_contract::class_view::ClassView` gains `fn lens_fields(&self, lens: u32) -> Option<&[FieldRef]>` with a default that returns `None`. **Source-compatible:** every existing `impl ClassView` compiles unchanged (a test pins this with a view that implements only the required methods).
- New `class_view::is_behaviour_lens(classid) -> bool`: right (custom) half in `0xF000..=0xFFFF`.
- `lance_graph_dt64::Refusal` gains `UnknownLensBasis(u32)`. No crate matches `Refusal` exhaustively (searched across `crates/`), so no match breaks.

## Behaviour change
- `BindStore::resolve_algebra` reads a behaviour lens's basis through `lens_fields(lens)`. `None` refuses with `UnknownLensBasis`; the left half's basis is never inherited, even when its field count matches.
- Unchanged: every lens with a right half below `0xF000` (the `0xC401` R2IL reading, the lab lenses `0x7F0A_0001` and friends) and every data class (`resolve`) keep the left-half lookup.
- The one lens on `main` this moves is the polyfill lens `0xFFFF_F000` (#1899). A caller that passed a view answering it only through `fields(0xFFFF)` now gets `UnknownLensBasis`, which is O9's intent. `PolyfillView` answers it through `lens_fields` and no longer holds a concept basis; `WithPolyfill` routes `lens_fields` for lenses under `0xFFFF`. The existing polyfill test view gained `lens_fields`; its assertions are unchanged.

## Measured
- Red first: on the old `resolve_algebra`, 4 of the 5 new DT64 tests failed; the compatibility twin passed.
- Two lenses under `0xFFFF` with 17 and 5 words resolve independently; a data class `0x7F0B_0001` and a behaviour lens `0x7F0B_F001` do not cross-read; an unknown behaviour lens refuses even when the concept's basis has the right count.
- Disable runs, all red: behaviour branch off (4 fail), inherit the concept basis on an unknown lens (2), every lens treated as a behaviour lens (2), `WithPolyfill` not routing `lens_fields` (2).
- Dependents checked: `lance-graph-contract` (1562 + others), `lance-graph-dt64`, `lance-graph-r2il`, `xlog-relcomp-probe`, `dt64-admission-probe` green; their lenses use right halves `0x0000`/`0x0001`.

Not widened: `AlgebraRef.word`, `WideFieldMask`, the 256-position limit.
