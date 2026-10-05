# Stay on the lane

Guardrails, and the moves that solve the next caveats, for a planner that only emits folds over 64k ordered 128-bit SoA lanes, index kept in `u16`.

## Do not fall astray

1. One number, one object. 0.6 ns is a word-fold. 4–12 ns is a tile aperture. 605 ns is the plane pipeline from the count probe. 0.6 ms is a million word-folds. 0.6 s is a million plane executes. Citing one of these as another is how the claim dies.

2. The planner is the product. The executor already matches the oracle and allocates nothing. A new operator that builds a pair list, a hash table, or a `u32` index is not a feature. It is the technique ending in the middle of a plan.

3. Quack lowers. It does not execute. A hill-climb, a timer, or a second match inside the lowering is the duplicate evaluator the crate exists to avoid. The controller lives in the consumer that already calls `execute`.

4. Fused and gated are different plans. `lower_fused` cannot skip, because ternlog has no `under`. Shipping the fused form and claiming the skip is a false result.

5. A resident plane makes `and_by_skip` inert. The plane is the gate. Reordering conjuncts under it does not move a word. Do not port DuckDB's reorder there.

6. A caller-owned bitmap is still materialization. `fixture_view_bytes` and `population_state_bytes` sit next to every zero. A zero beside a non-zero view-build is not zero-copy.

7. K is printed. Past the knee, K masks lose to a hash aggregate. That loss is a refusal, not a silent hash table inside the fold.

8. The boundary is the invariant. One unordered batch, one fk that is not a `u16` rail offset, one string. Time the stacking at ingest, or the join line is no longer a reading.

9. A million apertures are not a million folds. A million masks at 8 KB are 8 GB. That is the route, stored as coordinates.

10. Do not send the 6,000-commit tree anywhere as the proof. The proof is a checksum, a heap counter, and two timings on one page.

## Epiphanies

The missing optimizer is a popcount. Selectivity of a 64k mask is `popcount / 65536`, and the mask is 8 KB. DuckDB builds a catalog and a histogram because it does not already hold the aperture. You do. `and_by_skip` does not need statistics propagation. It needs the popcounts of the candidate gates, cheapest dead-word fraction first. That is the static `REORDER_FILTER`, and it is a thousand instructions.

Alignment is a value, not a pass you redo. A shift `d: u16` or a mask is a first-class plan input. Fold 2 through fold 1,000,000 receive it by borrow. If a fold would write a new mask, the planner names it a new aperture and counts it against the 8 GB bound. Most folds will not write one.

A million linear folds of one aperture are one fold. Sum of sums, count of counts, min of mins collapse before execution. The terminal is associative. The planner that notices this turns 0.6 ms into one plane pass. The planner that does not has done a million times the work for the same answer.

The word aperture is the wrong grain only when survivors are scattered. Then the allowed move is still 16 bit: extract the set bits into a `u16` list, at most 128 KB, and fold the value lane through that list. You have not built pairs. You have narrowed the address. Do this only when the dead-word fraction is low and the popcount is small. When the dead-word fraction is high, the word gate is already the win, and the list is a regression.

Left, anti, and mark are the same mask. Left is the aperture plus the unmatched bit in a validity plane. Anti is the complement. Mark is the aperture stored. Three operators, one object. A plan that lowers them to three joins has left the lane.

The ingest type is the proof of order. An `OrderedLane<u16>` witness that will not construct from an unsorted buffer makes the zipper impossible to invoke by accident. The contract crate already has a witness type. Use it at the boundary, not as a comment.

## Next caveats, and the move

Scattered survivors. Word skip does nothing. Move: popcount the gate; if dead words are rare and live rows are few, extract `u16`s and fold those. Publish both fractions.

Group cardinality. K passes fall off a cliff. Move: measure the knee once, put it in the planner, refuse past it. Do not hide a hash table under the same terminal name.

Foreign plane not resident. The fk fold is a gather into a plane someone else built. Move: the build is an ingest line, checksummed, never added to the fold ns. If the fk is not a `u16` into this lane, refuse or align once into a mask.

Asof and range. A window `w` generated as rows is a pair list. Move: the window is a shift range on the other ruler, applied as a gate. The terminal reduces through it. Emit rows only if the consumer asked, capped, at the end.

Distinct on a wide value. Move: if the domain fits in 16 bit, a bitset of 8 KB. If not, a `u16` permutation sort of the lane, 128 KB, named as the one allowed index. Never sort the 1 MB body.

Nested loop correlation. 64k × 64k is the materialization in disguise. Move: a correlated predicate must lower to a shift against the outer `u16`, or the planner rejects the query.

Unnest and strings. They do not fit the slot. Move: a second lane, explicitly, with its own `u16` space, or a refusal. Do not widen the index.

Adaptive order at runtime. The static popcount handles the stable case. The unstable case is a consumer-side swap: run the gated plan, keep the swap only if ns/exec fell, never inside quack.

Algebraic blowup. A Boolean tree of depth d can be a d-pass fold or one ternlog. Move: fuse when the gate is dense, gate when it is not, and collapse associative terminals before either runs.

Proof rot. A fixture `expected` column edited by hand will confirm whatever you last believed. Move: regenerate from DuckDB in the test, fail the build on a non-zero `alloc_bytes_exec` for a fold plan, fail the build on a plan node whose output is a pair list.

## Generous order of work

1. Pin the three numbers on the proof page, with the object each one measures.
2. Make the aperture a borrowed plan value. Count how many distinct masks a query allocates. Cap it.
3. Collapse associative folds before execution. This is the largest free win in the million-fold story.
4. Feed `and_by_skip` from popcount. That is the optimizer you do not have to port.
5. Extract `u16` survivors only under the scattered-survivor test.
6. Put the K knee and the refusal list in the planner. A refusal is a result.
7. Keep the DuckDB oracle, and add the timed column on the same cases. Checksum first.
