# Lane fold, deforested

Depth note for the 64k ordered lane. The invariant page is `board/LANE_FOLD.md`. The integration plan is `board/LANE_FOLD_INTEGRATION.md`. The seventeen rooms ahead are `board/LANE_FOLD_SEVENTEEN_ROOMS.md`. This file is the literature and the residual.

## What the name means

Deforestation, Wadler 1990, is a program transformation that deletes an intermediate tree because the consumer can fold the producer directly. Shortcut deforestation, Gill, Launchbury and Peyton Jones, FPCA 1993, reduces that to one equation:

```text
foldr k z (build g) = g k z
```

The producer is written as `build`. The consumer is `foldr`. The rewrite substitutes the consumer's step for the list constructor. GHC still ships this as a `RULES` pragma. The paper states the failure: the rule does not fuse `zip`, and it does not fuse `foldl`. A zipper is the combinator the shortcut cannot eat.

Stream fusion (Coutts, Leshchinskiy, Stewart, ICFP 2007) represents the list as a step function so `zip` and `filter` can fuse. Kiselyov, Biboudis, Palladinos and Smaragdakis, arXiv:1612.06668, push that to a semantic model and name the residual: both push and pull streams delete the list and still leave the function passed into `map`. Chen and Parreaux, arXiv:2410.02232, ICFP 2024, technical report revised October 2025, say shortcut fusion is still the only industrial system, and that it fires only on `build` and `foldr`.

The database name for the same deletion is late materialization. Abadi, Myers, DeWitt and Madden, ICDE 2007, scan a column, emit positions as a range, a list, or a bitmap, intersect the bitmaps, and only then stitch tuples. They also measure the reversal: on some workloads the position list costs more than the tuple it replaced. Ultra-late materialization keeps the position set longer. Hybrid materialization, arXiv:2304.08532, exists because that set is sometimes the new tree.

Indexed stream fusion, arXiv:2507.06456, July 2025, is the paper that reaches a join. It folds an indexed stream so two collections allocate no pair list, with a mechanized proof in Lean and a Rust port. It does not use the word deforestation. It handles the shortcut's failure case by making the index the structure.

## Three layers, each deleting what the previous layer emitted

1. Wadler deletes the tree. Gill makes it a rule. The residual is anything that is not `foldr/build`, notably `zip`.
2. Abadi deletes the tuple and leaves the bitmap. The residual is the position set. Hybrid materialization is the admission that the residual sometimes costs more than the tree.
3. Aggregate fusion deletes the bitmap when every consumer is the same homomorphism. Sum of sums is one sum. The residual is a non-associative fold, a scattered survivor list, or a real `zip`.

The lane fold sits on layer 2 and has layer 3 available. The 8 KB mask is Abadi's bitmap and Gill's `build`. It is the intermediate the shortcut was willing to keep. A million linear folds of one aperture becoming one fold is the equation applied to the output of the first fusion. Shortcut fusion cannot see it, because the mask was the object the first rule produced and the second `foldr` is in another plan node.

A shift on a `u16` rail is not a `zip`. Tick `i` is tick `i+d`. That is why an indexed stream can join without allocating and the 1993 rule cannot. The moment the key is ordered but is not the address, the alignment mask is the intermediate that may be built once. Building it per fold is the deforestation failing at the combinator it has always failed at.

## What this tree already is

`lance-graph-mask-risc` is the consumer. A `Program` is a list of mask ops and one terminal. Execution is tiled at `TILE_WORDS = 256`. `pack_under` skips a word whose gate is zero. The row oracle in `reference.rs` never touches `ndarray`. `tests/no_alloc.rs` pins the zero-allocation law.

`lance-graph-quack` is the producer. It lowers and must not evaluate. `lower` gates. `lower_fused` cannot skip, because `MaskOp::Ternlog` has no `under`. `and_by_skip` takes order as an input. `EqU32Via`, `Gather`, and `ScatterOrU32` are the factored fk forms. `pair_relation_bytes` is printed as 0 in the DuckDB differential so the absence is a fact.

The count probe on 65,536 rows returned 4855 on every arm, heap 0, gate ok. Handwritten 605 ns, interpreted 626 ns, fused 636 ns. The interpreter is 20 ns. The rest is passes over 1,024 words. Sparse gate: 2722 ns against 6614 ns. A word-fold on that run is 0.6 ns. A million of those is 0.6 ms. A million plane executes is 0.6 s. Those are different objects.

## What the latest research does not do

Lumberhack's average on 38 programs is 8.2 percent, code growth about 1.79×, two programs slower. It would have to see the mask as a producer and the sum as its consumer across a plan node. It does not. Indexed stream fusion stops once the join allocates nothing. It does not collapse a million identical sums into one pass. Neither paper deletes the bitmap the first deletion left behind. That step is the open one, and it is a planner step, not an executor step.

## Residuals that must stay visible

- `fixture_view_bytes`: the harness built a view the terminal assumed was resident. A zero next to it is not a proof.
- A word with one survivor is live. Scattered predicates do not skip. The allowed narrowing is a `u16` list, 128 KB at most, only when the dead-word fraction is low.
- K masks lose past the knee. That loss is a refusal, not a silent hash table.
- A pair list, a string, a nested loop, an index wider than 16 bit. Refuse before execution.
