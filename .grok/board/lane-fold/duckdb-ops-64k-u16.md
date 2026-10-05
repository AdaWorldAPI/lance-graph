# DuckDB operations on a 64k lane, index stays in 16 bit

The address space is `u16`. `65536 = 2^16`, so a row index never needs a wider integer, and a plan that widens it has already left the lane. One lane is 64k ordered 128-bit SoA slots, 1 MB. A mask over it is 1024 `u64`s, 8 KB. A pair list of `u16` indexes is the materialization the planner refuses.

Cost unit for a resident word-fold: 0.6 ns. A full-plane pipeline in the count probe was 605 ns. A tile is 256 words, 16,384 rows. Alignment is paid once per stable aperture. The terminal is the only write.

## Address

| object | width | bytes |
|---|---|---|
| row index | `u16` | 2 |
| full index, if you ever built one | 64k × 2 | 128 KB |
| mask | 64k bits | 8 KB |
| 128-bit SoA lane | 64k × 16 | 1 MB |
| tile mask | 256 words | 2 KB |

A `u32` fk that addresses this lane is truncated to the low 16 bits or it is not an address in this lane. A join key that is not a `u16` rail offset is an alignment event, once, producing a mask. It is not a second index type.

## Scan

Read the lane. No copy. A scan is a bound: `Planes { n_rows: 65536, lanes, masks }`.

Prove: heap 0, checksum of a reduction matches a scalar walk.

Caveat: a scan that allocates a `Vec` of rows has left the technique.

## Filter

Comparisons `= <> < <= > >=` on `i32` / `u32` / the low or high half of the 128-bit slot, against a constant. `BETWEEN` is two comparisons and an AND. `IN` of a small list is K equalities, or one ternary match if the list is a mask of the value domain.

Boolean `AND` / `OR` / `NOT` are mask algebra. `AND` gates the next predicate (`under`). `OR` does not shrink the aperture; it widens it. `NOT` cannot be dropped from a gate.

`CASE` is `Terminal::Blend` through the mask. No branch per row.

Prove: count matches the row oracle and the DuckDB fixture. Gated arm faster than pred-then-AND when whole words are dead.

Caveat: a word with one live row is live. `LIKE`, regex, and arbitrary scalar functions are not mask ops. They run only on the survivors, and only if the survivor count is the terminal's business. Running them on all 64k is a scan, not a fold.

NULL is a resident validity plane, 8 KB, not a side bitmap with its own index. Three-valued `AND` / `OR` / `NOT` follow that plane. A NULL literal in an expression is out.

## Project

Projection is a lane number. The slot stays where it is. A computed column that fits in 128 bits is a new lane, written once, still `u16`-addressed. A computed column that does not fit is a refusal.

## Aggregate

`COUNT`, `EXISTS`, `MIN`, `MAX`, `SUM` are terminals over the final mask. `AVG` is sum and count, divided at the terminal. `SUM` widens to `i64` and rejects past the carry bound rather than wrapping.

`COUNT DISTINCT` is not a fold unless the domain fits in a mask (a `u16` domain is 8 KB, so a distinct over the rail itself is a bitset). Distinct over a wide value is a sort of the lane or a refusal.

Prove: DuckDB differential, `alloc_bytes_exec = 0`.

## Group by

K groups are K masks over the kept filter. One-terminal group-sum when the key is on this lane. Having is a predicate on the K aggregates, not on the rows.

The key has to be a small categorical, or a `u16` domain you bin into K. Past a measured K the K passes lose to a hash aggregate, and the hash aggregate is the refusal path: it leaves the fold.

Prove: same sums as DuckDB, K printed.

## Join

Only while both sides are 64k ordered lanes and the planner does not build pairs.

Rail join. The key is the `u16` address. Alignment is a shift `d`. A fold reads `lane_b[i+d]` under the aperture. No zipper, no pair list.

Ordered-but-not-address join. One forward pass writes a mask of matching ticks. Then 100 to 1,000,000 folds reuse it. A million word-folds at 0.6 ns is 0.6 ms. The pass is noise after that.

Fk fold. `EqU32Via` reads the foreign predicate through the `u16` fk. `Gather` is a semijoin against a resident foreign mask. `ScatterOrU32` is the hop back, OR-ing into a caller-owned mask. `GroupSumVia` is a sum grouped by the foreign lane, one program.

Inner is the aperture. Semi is the aperture without the foreign payload. Anti is the complement of the aperture. Left is the aperture plus the unmatched bit in the validity plane of the payload lane. A mark join is the aperture itself, stored as the 8 KB mask.

Asof and range joins are a shift window: for each tick, the aperture is `[t, t+w]` on the other ruler. Still a mask, still no pairs, as long as `w` is applied as a gate and not as a generated row range.

Caveat: a many-to-many that the user asked to see as rows is a terminal that emits `u16` pairs. That terminal is materialization. Keep it at the end, cap it, or refuse. A cyclic n-hop is not this join. It needs a binding-order intersection the IR does not have.

## Set operations

`UNION ALL` of two 64k lanes does not fit in one lane. Refuse, or write a second lane. `UNION` is the bit-or of two masks if both are already the same rail; otherwise an alignment, then the or. `INTERSECT` is the and. `EXCEPT` is the and-not. All three return a mask, not a row list.

## Distinct

Distinct on the rail is the mask. Distinct on a value lane is a sort of that lane or a bitset if the domain is `u16`. A sort permutes a `u16` index of 128 KB. That index is the one allowed materialization, and it is still 16 bit. Do not sort the 1 MB lane if the index will do.

## Order by and top-n

Ordering does not reorder the lane. It writes a `u16` permutation, 128 KB, or it writes nothing and the terminal walks the lane in key order because the lane is already ordered. Top-n on an ordered lane is a prefix. Top-n on a value is a heap of `u16` indexes, n wide, never a sorted copy of the rows.

`LIMIT` / `OFFSET` on an ordered rail are a range on the `u16` address. On a mask they are the first n set bits. Finding them is a popcount scan of 8 KB, not a row copy.

## Window

A window over an ordered rail is a prefix fold: running sum, rank, lag. Lag is a shift. Rank on the rail is the address. Rank on a value needs the `u16` permutation. A frame of width `w` is the same shift window as a range join. The output is a new lane, written once, or a terminal. It is not a copy of the input per row.

## Pivot

A pivot is K masks and K terminals. The coordinates move. The lane does not. This is the Excel case, and it is in technique only while K is the group cardinality you already accepted.

## Sample

A sample is a mask built from a `u16` hash of the address. Reservoir sampling that stores rows is a refusal. Store the `u16`s, and only if the consumer asked to see them.

## Unnest

Unnest of a nested value leaves the 128-bit slot and the 64k bound. Refuse, or write a second lane and say so. Do not grow the index past 16 bit to hold child rows.

## Subquery

`EXISTS` is a semijoin aperture. `IN` is the same. A scalar subquery is a terminal that must return one value; zero or many is an error, not a row. A correlated subquery is a fold per outer tick only if the inner side is a shift against the outer `u16`. Otherwise it is a nested loop, and a nested loop over 64k × 64k is the materialization in disguise. Refuse it.

## What the planner emits

A plan is a list of mask ops, a shift or a once-only alignment, and one terminal. Intermediates are masks of 8 KB or a `u16` permutation of 128 KB. Nothing else is allocated.

Refuse, in the planner, before execution:

- a pair list
- an index wider than 16 bit
- a string, a regex, a dictionary
- a hash table
- a group cardinality past the measured K knee
- a nested loop
- an unnest that grows the lane
- a second evaluator inside the lowering

## Proof, per operation

Checksum against DuckDB on the fixture. `alloc_bytes_exec = 0`. `pair_relation_bytes = 0`. `population_state_bytes = 0`. Any `u16` permutation printed as its own line, 128 KB, and not called zero-copy. Tile ns and plane ns on separate lines. Dead-word fraction next to any skip claim.
