# Thought layers: a NaN lane is not written, merge on read keeps the prior

**Status:** TEST-PINNED, lab only. Code: `crates/cognitive-shader-driver/examples/thought_layers_probe.rs` (`EpiLanes`, `ThoughtTables::write_epi`). Read at lance-graph `main` `ea709af`, ndarray `master` `2cbe131` (provenance receipt, not a pin).

## Rule (DECISION)

- Never touched (NaN, `$null`, UNKNOWN) is presence 0 **and** value 0, at U8 and at U9 alike. A sparse write does not write it; the lane keeps what the thought read before, so UNKNOWN never overwrites a known value.
- U9 / lut9 / palette512: the presence bit is also the sign, so presence 0 with magnitude > 0 is a negative number and is written. Presence alone is not a skip test.
- A touched value that rounds to zero is +0 (presence 1, magnitude 0), decided in the same step (`codec9::encode_touched`, `simd_stride::lut9`).
- U8 presence 0 with a nonzero value cannot arise except by reading U9 bytes through a U8 layout. That is excluded by the lane width type and CI, not checked per lane at runtime.

## Before

`RowDelta` compares opaque bytes: a NaN lane differed from a known base value, was stored, and read back as NaN. `thought-git-basics-v1.md` item 5 (presence alone decides the write) holds for its lane-presence-mask + payload-at-rank layout, not for U9 lanes.

## Tests (each disable-verified)

`nan_over_a_known_value_is_not_written` (U8, U9), `nan_after_an_own_write_keeps_the_own_value`, `a_u9_negative_is_written`, `a_computed_negative_zero_is_written_as_plus_zero`, `nan_beside_a_change_stores_only_the_change`, `layouts_over_the_key_or_overlapping_are_refused`. Disables: skip the keep step; take the prior from the base instead of the thought's read; presence-only test at U9; magnitude-only test at U9.

## Open

The lane layout is a fixed lab layout; production resolves it, with its width, from the class's ClassView. No ClassView lane-width descriptor exists today.
