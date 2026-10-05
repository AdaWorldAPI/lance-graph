# Claude capstone — lane fold

Read this before editing `lance-graph-mask-risc` or `lance-graph-quack`. It is the heads-up. The depth is linked. Do not re-derive it.

Invariant: `board/LANE_FOLD.md`. Collapse: `05_query_languages/associative_collapse.md`. Integration: `board/LANE_FOLD_INTEGRATION.md`. Rooms: `board/LANE_FOLD_SEVENTEEN_ROOMS.md`. Literature: `05_query_languages/lane_fold_deforestation.md`. Sketch check: `board/lane-fold/lane_guard.rs`.

## Settled, do not reopen

- Address is `u16`. Lane is 64k ordered 128-bit SoA, 1 MB. Mask is 8 KB. A permutation is 128 KB and is the only allowed index. A `u32` fk that does not fit is a refusal, not a cast.
- `count_probe`, release, this class of host: count 4855 on all four arms, heap 0, gate ok. Handwritten 605 ns, interpreted 626 ns, fused 636 ns. Interpreter tax is about 20 ns. Sparse gate 2722 ns against 6614 ns.
- 0.6 ns is a word-fold. 4–12 ns is a claimed tile aperture, not what that run measured. 605 ns is the plane. A million word-folds is 0.6 ms. A million plane executes is 0.6 s. Citing one as another is a defect.
- Quack lowers and must not evaluate. `lower_fused` cannot skip: `MaskOp::Ternlog` has no `under`. `and_by_skip` is inert under a resident plane. The plane is the gate.
- `pair_relation_bytes = 0` is the join claim. A caller-owned bitmap is still materialization. `fixture_view_bytes` sits beside every zero.
- Do not route this through cognitive layers, the shader driver, or ontology bridges.

## Points that need addressing

1. Collapse is not implemented. A million identical sums must lower to one terminal. Partition by `(aperture, operation, lane)`. `AVG` collapses as `(sum, count)`, divided once. Repeat-the-value and scale-the-value are different requests. Print `collapsed_terminals`. Carry bound stays. Empty `MIN` stays the seed.
2. `lane_guard.rs` is a sketch in `.grok`. It is not on the build. `Plan::check` belongs before `lower`. `Metrics::fold_ok` belongs in the differential. A fused plan that claims the skip must fail.
3. The tile loop does not skip a 256-word zone of zeros. Word skip exists (`pack_under`). Tile skip does not. Print dead-tile fraction beside dead-word fraction.
4. `Pred::Range` and the strided predicates have no `_under` twin. They scan, then gate. On an ordered rail a range is a `u16` span.
5. No timed DuckDB column. The differential is semantic. A latency claim needs the same cases, checksum first.
6. The 4–12 ns tile band has no probe. Four tiles times three passes at that rate would be 50–150 ns. The measured pipeline was 605 ns. Do not cite the band until a single-tile probe exists.
7. The 1.7 ns register ABI is not in these crates. It is event R0 in `board/lane-fold/graph-bench-challenge.md`. Citing it as a join is a DQ.
8. K past the knee is a refusal, not a silent hash table under `GroupSumI32`. The knee is unmeasured. Pin it with a number, do not invent 64 and ship it as physics. The sketch uses 64 as a placeholder.
9. Scattered survivors do not skip. A word with one live bit is live. Extract `u16`s only when the dead-word fraction is low. Cap 128 KB.
10. Aperture cap is unenforced. A million masks at 8 KB are 8 GB. Cap distinct masks. The sketch uses 8.

## Best version, so the edits aim at it

One borrowed aperture. One terminal per homomorphism class. Executor unchanged except the tile-zero continue. Checksum is a fold, not a row vector that is then hashed. Residuals named and refused: a real zip, a non-associative consumer, a string, a scattered extract the caller asked to see, a cyclic n-hop.

## Order of work

Collapse rewriter and its 1,000-sum test. Repeat versus scale. Tile skip. Fuse-or-gate switch. Range-as-span. Timed DuckDB column. Guardrail on the build.

## Do not

Send the 6,000-commit tree as the proof. Edit the differential `expected` column by hand. Add a second evaluator inside quack. Call a hash aggregate a fold. Widen the index.
