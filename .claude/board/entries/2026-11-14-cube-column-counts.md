# 2026-11-14 — D-CUBE-COUNTS-1: `Cube::column_counts` and `Cube::join_size`

**Kind:** two public methods in `crates/lance-graph-xlog/src/cube.rs`, one test file.
**Grade:** MEASURED against plain enumeration. No timing claimed.

## What

- `column_counts(table, side, col) -> Vec<i64>`: rows per position of one
  (relation, column) side; a scan or a bind. It is the `GroupReduce Count`
  program `bag_count` already ran internally, now public.
- `join_size(table, join) -> i128`: the sum of `bag_count`, i.e.
  `Σ_p left(p) · right(p)` over two count vectors. The number of tuples
  `materialize_rows` would build, without building one.
- `self_join_size(table, side, col) -> i128`: `Σ c²`, the self-join's
  derivation count from one count program.
- `heaviest(table, side, col) -> Option<(u32, i64)>`: the coordinate with the
  most rows and its count; ties to the lower coordinate; `None` for an empty
  side. The coordinate stays in the cube's frame; the caller decodes it.

Every relation and both columns are counted over the same positions of one
map, so any two count vectors line up position by position. That is what
makes a join size one dot product.

## The DR question this answers

On a DR identity axis the order of positions is the map's, not a value
order. The neighbour-addition chain behind `code_power_sums_u8` (ndarray #428)
needs an ordered numeric code axis, so it does not apply here. What applies
on identity coordinates is the count step and moments over the counts:
`Σc` (rows), `Σ c_L·c_R` (join size), `Σ c²` (self-join size), and the
largest count with its id.

## Tests (`tests/cube_counts.rs`, 9)

- all 6 count vectors equal enumeration, id by id, and sum to the row count;
- all 36 (relation, column) joins: `join_size` and the dot product of the two
  count vectors equal the enumerated pair count (fixture has non-joining pairs);
- `join_size` builds no tuple (`rows_materialized` unchanged, 2 programs) and
  equals `materialize_rows(..).len()`;
- a bound side counts only its rows; counting the bound column is refused;
  a value outside the map counts nothing;
- the heaviest id does not change when the map moves positions;
- a join as a side, column 2, a non-join for `join_size`, and a table that
  never issued the frame are refused; advancing the table is NOT a refusal.

- `self_join_size` equals the self-join's `join_size` and enumeration for all
  6 (relation, column) pairs; at least 2 have repeated keys;
- `heaviest` decodes to id 1 with 4 rows in `r.X`; an empty side gives `None`;
  a tie at 2 rows goes to the lower coordinate.

Disable runs, all red: `join_size` returning 0 (2 tests), skipping the frame
check (1), counting the other column (4), summing `c` instead of `c²` (1),
ties to the later coordinate (1), an empty side reporting coordinate 0 (1).

## Open

- Not measured: time. Both methods run the programs `bag_count` runs.
- No grouped or tiled-specific test beyond what `degrees` already covers.
