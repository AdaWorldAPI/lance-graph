# D-DRF-G-FACET — the focus cache key now includes the facet (lab fix)

**Status:** FIXED, TEST-PINNED (lab). **Code:** `crates/drf-focus-probe/src/lib.rs` `FocusKey`.
**Test:** `g_two_facets_of_one_map_are_two_foci_not_one_cached_column`.

`FocusKey` was `(source, map, scope)`, but a focus caches one gathered facet column. Loading
the same map with a second facet returned the first facet's column: measured 36 595 where the
second facet holds -5. The facet is now a key field. Two facets of one map build two foci;
the same facet again is a cache hit (silence twin).

Disable (key on a constant facet in `load`): the new test and `l2_retire_*` go red.

Not affected: W1–W7 writeback and handoff results. Writeback stages by an explicit facet and
never reads the cached column.
