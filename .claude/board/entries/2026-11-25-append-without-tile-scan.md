# Append without a global tile scan (execution order, step 3)

**Status:** In PR. Monotone keys are closed. Overlapping key ranges are OPEN, pending a decision.

DECISION applied (operator, 2026-11-25, #3): DT64 tile bounds are allowed only as a derived description of the DR tiles, never as a second key registry. Removing the global tile scan is the goal; a cheaper scan is not closure.

## What was measured

All numbers are for one 4,096-row batch through `Table::append_values`, release build, best of 5, plotted against rows already held. They come from two sources:

1. **The key check.** `MapTable::locate` fetched every tile's map for every key, at about 54 ns per key per tile: O(keys × tiles).
2. **DT64's coordinates revision.** `MapTable::apply` rehashed every tile's map digest on every advance (`fnv_ids` over all 256 globals per tile): O(rows) per commit. It was found by timing a single append into a table built in one shot. Pure key lookups were already flat at that point.

## What changed

- **intake (`Owner`):**
  - Each tile's `(min, max)` key, aligned with the tile list, is recomputed from each changed tile's map at commit.
  - An `ordered` flag records whether those ranges are disjoint and ascending.
  - `locate` reads only the tiles whose range holds the key: one binary search while ordered.
  - `new_keys`, when ranges overlap, reads the sorted batch per tile only within that tile's range.
  - A bound can be wider than the live keys (deletes never shrink it). It filters tiles; it never answers a lookup by itself.
- **DT64 (`MapTable::apply`):** `coordinates` is now `COORDINATES_BASE` plus the wrapping sum of one term per tile, FNV over the tile id and its map digest. Each advance updates it for the tiles it changes. In debug builds, a `debug_assert` checks the running value against the full sum on every advance. The existing DT64 tests compare `coordinates` only for equality, and they pass (158).

## Results (ns per row)

| rows held | before | monotone keys | interleaved keys |
|---|---|---|---|
| 0 | 48 | 69 | 54 |
| 4,096 | 506 | 82 | 118 |
| 16,384 | 2,618 | 83 | 298 |
| 65,536 | 11,984 | 99 | 1,038 |
| 262,144 | 55,943 | 124 | 3,991 |

- A pure key lookup (`live_positions_for_keys`) is flat at 86–93 ns per key, from 4,096 to 262,144 rows.
- Monotone keys come to 507 µs per batch at 262,144 rows held, within the microsecond batch gate. The remaining slope is lane and plane growth on commit.
- Interleaved keys fail the gate at large sizes. Every tile's range covers the batch, so each key is checked in each tile.

## Tests and disable runs

- **New intake test,** `keys_are_located_through_tile_bounds_ordered_or_overlapping`, in both layouts:
  - every held key is found;
  - every held key is refused when appended again;
  - new keys are admitted.
- **Disable runs,** both red, then restored:
  - the ordered search excludes a tile's largest key;
  - the overlap path excludes a tile's upper bound.
- **Suites,** all passing:

  | crate | tests |
  |---|---|
  | dt64 | 158 |
  | intake | 57 |
  | vector | 6 |
  | enrichment | 5 |
  | vrl | 40 |
  | quack | 170 |
  | pg-row | 9 |
  | r2il | 108 |
  | rbac | 68 |
  | report | 66 |
  | xlog | 144 |

- **Not this PR's:** `lance-graph-mailbox-lanes` `a_read_only_act_allocates_what_was_measured` fails identically on main (43 allocations against 41).

OPEN (operator decision): overlapping key ranges. Three ways forward:
- **(a)** Require append-ordered keys for append-heavy tables. The route already uses a monotone `lg.seq`.
- **(b)** Accept O(tiles) per batch for interleaved keys.
- **(c)** A key-ordered structure, which must be weighed against D-REGC-7.
