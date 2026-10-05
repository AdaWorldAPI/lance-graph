# Lane fold

Invariant page for the 64k ordered lane. Claude heads-up: [`CLAUDE_LANE_FOLD_CAPSTONE.md`](CLAUDE_LANE_FOLD_CAPSTONE.md). Depth, literature, and the residual: [`../05_query_languages/lane_fold_deforestation.md`](../05_query_languages/lane_fold_deforestation.md). Collapse rewrite: [`../05_query_languages/associative_collapse.md`](../05_query_languages/associative_collapse.md). Integration: [`LANE_FOLD_INTEGRATION.md`](LANE_FOLD_INTEGRATION.md). Seventeen rooms ahead: [`LANE_FOLD_SEVENTEEN_ROOMS.md`](LANE_FOLD_SEVENTEEN_ROOMS.md). Derivations live in [`lane-fold/`](lane-fold/). Scaffold: [`lane-fold/scaffold/`](lane-fold/scaffold/). The check that fails a bad plan is [`lane-fold/lane_guard.rs`](lane-fold/lane_guard.rs). It belongs next to `lance-graph-quack` once it is wired in front of `lower`. This page does not execute anything.

## Address

| object | width | bytes |
|---|---|---|
| row index | `u16` | 2 |
| mask | 64k bits | 8 KB |
| permutation, the one allowed index | 64k × `u16` | 128 KB |
| 128-bit SoA lane | 64k × 16 | 1 MB |
| tile | 256 words, 16,384 rows | 2 KB |

A `u32` foreign key that does not fit in `u16` is a refusal, not a cast. An unsorted buffer does not construct an ordered lane.

## Numbers, one object each

Measured by `cargo run --release -p lance-graph-mask-risc --example count_probe` on 65,536 rows. Count 4855 on every arm. Heap 0. Gate ok.

| object | figure | what it is not |
|---|---|---|
| word-fold | 0.6 ns | not a join |
| tile aperture | 4–12 ns hot, 25–44 ns cold | not what that run measured |
| plane pipeline | 605 ns handwritten, 626 interpreted, 636 fused | not 0.6 ns |
| million word-folds | 0.6 ms | not 0.6 s |
| sparse-gate skip | 2722 ns gated, 6614 ns pred+and | not row-level skip |

## Planner

A plan is mask ops, one shift or one alignment, and one terminal. Intermediates are 8 KB masks or one 128 KB `u16` permutation.

Refuse before execution: a pair list, an index wider than 16 bit, a hash table, a string or regex, a nested loop, an unnest that grows the lane, a group past the K knee, a fused plan that claims the skip, a reorder under a resident plane.

`lower_fused` cannot skip. `MaskOp::Ternlog` has no `under`. `and_by_skip` is inert under a resident plane. The plane is already the gate.

Selectivity is `popcount / 65536` of a mask the plan already holds. That is the static reorder. The runtime hill-climb stays in the consumer that calls `execute`. Quack only lowers.

A million linear folds of one aperture are one fold. Sum of sums collapses before execution.

## Proof line

Checksum against DuckDB. `alloc_bytes_exec = 0`. `pair_relation_bytes = 0`. `population_state_bytes = 0`. A permutation printed as its own 128 KB, not called zero-copy.
