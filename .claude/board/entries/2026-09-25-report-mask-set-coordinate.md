# 2026-09-25 — A SET coordinate for the report substrate: `CoordSpec::MaskSet`

**Status:** TEST-PINNED · OPEN (one pass per member, not one keyed pass)

## What landed
- `CoordSpec::MaskSet { base, count }`: member `m` is the resident mask `base + m`. It is the coordinate of a many-to-many axis — paperless-ngx's tags, the gap `tesseract-paperless::axes` named ("count per tag is one scalar fold per tag, not one pivot").
- A row lands in every member whose mask holds it (zero, one or several), so the dimension's cells do not sum to the selected population. The per-row test oracle (`tests/common`) now places a row in the cartesian product of its per-dimension memberships.
- Planned as `Provider::MaskPlanes`: each member is one mask plane read in place, always a partition, never the fold key. A missing member mask is `UnknownMask`; `count == 0` is the new `EmptyMaskSet`.
- `CoordSpec::field()` now returns `Option<FieldId>` (a mask set reads no lane).

## Evidence
- `tests/mask_set.rs` (6): alone, crossed with an ordinal fold key (dense and sparse), under a selection — all against the oracle; per-tag counts equal the (row, tag) membership count and exceed the tagged rows; missing mask refused; empty set refused; explain names the set. The fixture asserts it contains untagged rows and rows with ≥ 2 tags.
- Disable runs, each red then restored: member filter reading the validity plane (3 tests fail); empty-set guard removed; missing mask read as the validity plane.

## Open
- Cost is one population pass per member (× the other partitions), exactly as before — the win is one plan, one result space and one render, not fewer passes. A keyed multi-membership fold (one pass for all members) would need a mask-RISC aggregation that does not exist.
