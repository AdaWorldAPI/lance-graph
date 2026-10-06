# 2026-10-06 — Morton8x8: Moore neighbours and trie ascent from the code (D-MORTON-0)

## MEASURED — all eight Moore offsets over a full 256 × 256 tile

`lance-graph-contract` test `morton8x8::tests::nibble_climb_matches_the_trie_and_the_counts_are_pinned`.

| nibble climb | visits | share |
|---|---|---|
| 1 | 344,064 | 66.0 % |
| 2 | 132,096 | 25.3 % |
| 3 | 35,904 | 6.9 % |
| 4 | 9,156 | 1.8 % |
| total | 521,220 | |

91.35 % of directed Moore visits stay within two HHTL nibbles. A Morton nibble is two `x` bits and two `y` bits (fan-out 16), so one nibble climb is two quadtree levels.

## FINDING — what the code carries

- The neighbour comes from `checked_offset` on the code; leaving `0..=255` is read from the carry or borrow. No `(x, y)` is decoded.
- The trie ascent is `ceil(bit_length(a XOR b) / 4)`, checked against `NiblePath::common_prefix_depth` for every visit.
- For a power-of-two subgrid of side `2^k`, in-grid is `code < 4^k`: the 4 × 4 `Register128` is the code prefix below 16 (84 visits).
- The #1346 Moore step run on Morton lanes, re-bound to row-major lanes, equals the row-major step for three gates.

## OPEN

- `helix/examples/morton_shift_motion_probe.rs` keeps its own copy of the arithmetic (helix does not depend on the contract).
- No 64k-field consumer yet.
