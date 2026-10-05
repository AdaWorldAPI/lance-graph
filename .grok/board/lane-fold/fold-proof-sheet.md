# Fold proof sheet

Techniques already named in `lance-graph-mask-risc` and `lance-graph-quack`, plus the ones the DuckDB comparison showed are missing. Each row is something to implement or finish, the measurement that proves it, the way it fails, and the fix.

Machine for every timed row: the same box that ran `count_probe` (605 ns handwritten, 626 ns interpreted, 636 ns fused, heap 0, count 4855, gate ok). A number from another machine does not extend that run.

## 1. Mask fold over a resident plane

What it is. A Boolean program over borrowed `u64` bit-planes, one word per 64 rows. The plan describes, the executor borrows, `ndarray::simd` computes, the caller owns the bytes. Tiled at `TILE_WORDS = 256` (16,384 rows, 2 KB).

Prove. `cargo run --release -p lance-graph-mask-risc --example count_probe`. Four arms agree with the row oracle. Heap bytes after warmup are 0, and the canary shows the counter can move.

Criteria. Count match. `gate: ok`. Hot pipeline on 65,536 rows in the same band as 605–636 ns on this host, not a copied number.

Caveat. 605 ns is the whole execute, about 50 ns per tile-pass. The 4–12 ns band is one aperture over one tile, and this run did not hit it.

Solve. Time one tile and the full plane on separate lines. Never divide by rows and call the result a join.

## 2. Survivor skip (the lithography aperture)

What it is. `MaskOp::Pred { under }` skips a word whose gate is zero. `pack_under` does not load it.

Prove. The F-X1 half of `count_probe`: sparse gate, 1 word in 8 nonzero. This run: gated 2722 ns, pred+and 6614 ns, same count 1741, heap 0.

Criteria. Same answer. Gated arm strictly faster on a gate with whole dead words. Heap 0.

Caveat. A word with one survivor is live. A 256-row block with one live word is live. Scattered predicates do not skip. This is coarser than DuckDB's selection vector, which drops the 63 dead rows inside a live word.

Solve. Report dead-word fraction and dead-block fraction, as `adaptive_order_probe` already does. If they disagree, publish both. Do not claim row-level skip.

## 3. Predicate order as an input (`and_by_skip`)

What it is. The execute side of DuckDB's `REORDER_FILTER`. Term k is gated on the accumulator of 1..k−1, so a selective term first shrinks the aperture. Order is an argument, not a search.

Prove. `cargo run --release -p lance-graph-quack --example adaptive_order_probe`. It counts skipped words, it does not time them. Recorded spread: up to 99.61 percentage points of skipped words on a clustered conjunction, about nothing on a permissive one.

Criteria. Best and worst order both printed, exhaustive over the 4-term fixture. Conjunction selects a non-empty proper subset (the disjoint fixture was degenerate and is not a result).

Caveat. Worth zero under `lower_fused`, because `MaskOp::Ternlog` has no `under`. Worth zero under a resident plane (`ISS-QUACK-AND-BY-SKIP-IS-INERT-UNDER-A-PLANE`): the plane is already the gate. Quack does not execute, so it cannot see min, max, distinct, or expression cost.

Solve. Keep fused and gated as separate events. The caller is the optimizer. Feed it a selectivity estimate, or stop claiming adaptive order.

## 4. Runtime hill-climb (not ported)

What it is. DuckDB's `AdaptiveFilter`: adjacent swap, observe 10 iterations, execute 20, keep the swap only if mean runtime fell, else reverse and halve swap-likeliness. Exists because the static order is often wrong.

Prove. Not implemented. Do not cite `and_by_skip` as this.

Criteria. A swap is kept only when the timed gated lowering got faster, on the same checksum. Likeliness decays when it did not.

Caveat. The crate's own rule is that quack must never evaluate. A hill-climb inside quack would be the second evaluator the design forbids.

Solve. Put the controller in the consumer that already calls `execute`. Quack stays a lowering. The signal is the ns/exec the probe already prints, not a selectivity guess.

## 5. Fuser versus gated lowering

What it is. `lower` folds in place and can skip. `lower_fused` gives every predicate a slot and hands the Boolean tree to the fuser. Fewer passes, no progressive narrowing.

Prove. `count_probe` fused arm: 2 passes against 3, 636 ns against 605 ns, same count.

Criteria. Bit-identical with the oracle. State which configuration was timed.

Caveat. On a hot 8 KB plane the extra ternlog dispatch cost more than the pass it saved. Fusing is not a win at this size.

Solve. Pick fused when scratch-bound and the gate is dense. Pick gated when the dead-word fraction is high. Publish the switch, do not pick one and call it the technique.

## 6. Zero-copy on the hot path

What it is. Law A1 / L1. No per-row object, no selection vector, scratch is caller-owned, tiled state is `slots × TILE_WORDS`.

Prove. `tests/no_alloc.rs` plus the counting allocator in `count_probe` and `duckdb_differential.rs`. This run: heap 0 on every arm, canary proved the counter moves.

Criteria. `alloc_bytes_exec = 0` after warmup. Setup, index build, and fixture views are separate lines.

Caveat. A caller-owned bitmap is still materialisation. `fixture_view_bytes` is the harness building a view the terminal assumes is resident. `population_state_bytes` is the one fold that carries a population-sized accumulator. Either non-zero line voids a zero-copy claim for that case.

Solve. Print all three, as the differential already does: `alloc_bytes_exec`, `population_state_bytes`, `fixture_view_bytes`. Zero-copy applies to a case only when the first two are 0 and the third is either 0 or labeled as harness cost.

## 7. DuckDB semantic differential

What it is. Quack lowers, mask-risc executes, DuckDB is the oracle via `tests/duckdb/oracle.py` and `cases.tsv`. Expected values are not hand-edited.

Prove. `cargo test -p lance-graph-quack --test duckdb_differential` with the oracle regenerated, not a stale `expected` column.

Criteria. Every case id matches. METRIC lines printed. `pair_relation_bytes = 0`.

Caveat. This is correctness on a fixture, not a timing. It does not support "faster than DuckDB". Coverage is comparisons, Boolean algebra, count, exists, min, max, sum, blend, a categorical group-sum, and three fk shapes. No strings, no `ORDER BY`, no NULL literal, no bag join.

Solve. Keep the oracle. Add a second binary that times the same cases against DuckDB on the same fixture. Checksum match is the entry ticket. Latency is a different column.

## 8. Factored foreign-key fold

What it is. `Filter::EqU32Via` reads the foreign predicate through the fk in one pass. `Filter::Semijoin` is `MaskOp::Gather` over a caller-supplied resident foreign plane. `Agg::ScatterOrU32` is the one-to-many hop back. `GroupSumViaI32` is `SUM(l.amount) GROUP BY p.country` as one program.

Prove. The three join cases in the differential: `join_sum_country`, `join_count_docs_with_posted`, `join_group_sum_country`. Checksum against DuckDB. `pair_relation_bytes = 0`.

Criteria. Same aggregate as DuckDB. No N×M pair list.

Caveat. The foreign plane has to already be resident. Building it is `fixture_view_bytes`, and that build is not a fold. This is not a hash join and not a worst-case optimal join.

Solve. Report build of the foreign plane on its own line. For a key that is not resident, this technique does not apply. That case is event J1, a real hash probe, and citing the fk fold there is a DQ.

## 9. Group-by as K masks

What it is. K gated equalities over the kept filter. No hash table. One-terminal `GroupSumI32` when the key is on this table.

Prove. `group_sum_cc` in the differential, both the K-program spelling and the one-terminal spelling, METRIC lines for both.

Criteria. Same sums as DuckDB. K is the group cardinality, printed.

Caveat. Only a win for a small categorical key. At large K this is K passes over the plane, which loses to a hash aggregate.

Solve. Publish K. Switch to a hash aggregate past a measured knee. Do not call the mask spelling a general group-by.

## 10. Register ABI (the 1.7 ns claim)

What it is. Equality and mask of two values already in registers. Claimed 1.7 ns hot, 4.3 ns from L1. Masks 4–40 ns.

Prove. Not in these crates. Needs `perf stat` and `objdump -d` of the hot block, plus the spill curve from 4 keys to 2^20.

Criteria. Hot ≤ 2 ns, L1 ≤ 5 ns, L1-misses ≈ 0, instruction count ≤ 12. A knee where the register file ends.

Caveat. 1.7 ns is about 7–9 cycles. It is a compare, not a join. The count probe did not measure it.

Solve. Isolate it as event R0. Never cite it next to a Cypher expand or a DuckDB join.

## 11. Tile latency versus plane latency

What it is. 4–12 ns hot, 25–44 ns cold, for one mask of one fold.

Prove. A probe that times a single 256-word tile, hot and after a cache flush, separate from the 1,024-word pipeline.

Criteria. Hot in 4–12 ns with the basic block shown. Cold in 25–44 ns with L2 hits, not a DRAM miss, confirmed in `perf stat`.

Caveat. Four tiles times three passes at that rate would be 50–150 ns. The measured pipeline was 605 ns. Until the single-tile probe exists, the band is a claim, not a result.

Solve. Print tile ns and plane ns on the same line. The operator pays the plane number.

## 12. What is out of the technique

Do not implement these inside the fold and then call them folds.

- Bag-semantics join. Needs a hash table or WCOJ. Ladybug wins n-hop on that, not on mask algebra.
- `ORDER BY`. No representation in the IR.
- Strings, dictionaries, NULL literals, `COALESCE`.
- A selection vector. The vocabulary cannot express one, on purpose.
- Runtime reorder inside quack. Belongs in the consumer.

## Advice, in order

1. Lead with the run you have. 605 ns, heap 0, count match, gated 2.4× on a sparse word-gate. That is a result.
2. Split every number into tile and plane, gated and fused, hot and cold. The technique is the split.
3. Treat DuckDB as the oracle you already wired, then add a timed column on the same cases. Semantic match first.
4. Put the hill-climb in the caller. Do not break the "quack only lowers" rule to get it.
5. Print `fixture_view_bytes` next to every zero-copy line. The teleport is the terminal. The view is not free.
6. Stop at the fk fold for joins. Past a non-resident key, measure a hash probe or do not claim the join.
7. Publish K for group-by and the dead-word fraction for skip. Both are the conditions under which the technique wins.
8. Do not cite R0, the 4–12 ns band, or the 605 ns pipeline as the same event. Reviewers will.
