# Lane fold integration plan

Wire the deforestation into the crates that already exist. Do not open a new zone. Quack lowers. Mask-risc executes. The planner between them is the missing producer/consumer pair the literature cannot see across a crate boundary.

## Where it lands

| piece | crate | status |
|---|---|---|
| mask executor, tile loop, `under` gate | `lance-graph-mask-risc` | landed |
| lowering, `and_by_skip`, fk forms | `lance-graph-quack` | landed |
| DuckDB oracle | `lance-graph-quack/tests/duckdb_differential.rs` | landed, semantic only |
| refusal check | `crates/lance-graph-lane-fold` | on the build, not yet called by quack |
| aperture as a borrowed plan value | neither crate | missing |
| associative collapse | neither crate | missing |
| tile-level zone skip | `exec.rs` tile loop | missing |
| range as a `u16` span | `Pred::Range` has no `_under` twin | missing |

## Steps

0. Land the associative collapse rewriter in quack before `lower` returns. Depth: `../05_query_languages/associative_collapse.md`. A partition is `(aperture, operation, lane)`. One terminal per partition. Distinguish repeat-the-value from scale-the-value. Print `collapsed_terminals`. A test with 1,000 identical sums asserts one terminal. No kernel change in this step.
1. Move `lane_guard.rs` next to quack as a dev dependency of the lowering, not as a note. `Plan::check` runs before `lower`. `Metrics::fold_ok` runs in the differential. A non-zero `pair_relation_bytes` fails the test.
2. Give the aperture a type the plan borrows. A shift `d: u16` or a mask of 1,024 words. Cap distinct masks per query at 8. The ninth is `Refuse::ApertureCap`.
3. Collapse associative terminals before `lower` returns. Sum of sums, count of counts, min of mins. A repeated terminal over the same aperture becomes one terminal. This is the third layer. It is a rewrite, not a kernel.
4. In `execute`, test a tile's 256 words before the word loop. All zero is a continue. This is zone-map pruning at the tile the executor already has. Publish dead-tile fraction next to dead-word fraction.
5. Switch fused versus gated from that fraction. Fuse when the gate is dense. Gate when it is not. Do not claim the skip on a fused plan. `MaskOp::Ternlog` has no `under`.
6. Lower `Pred::Range` on an ordered rail to a `u16` interval. A bit-span, one pass, no compare per row. Strided predicates get the same treatment once the stride is the rail. Until then they are scans.
7. Feed `and_by_skip` from `popcount / 65536` of the candidate gates. Do not run it under a resident plane. The plane is the gate. The runtime hill-climb stays in the consumer that calls `execute`, never inside quack.
8. Add a timed column to the DuckDB differential on the same cases. Checksum first. Latency second. A number with no reference column is not a result.

## What this must not integrate

Cognitive layers, shader driver, ontology bridges, OGIT. The fold couples to the SoA surface and to the mask executor. A plan that routes it through `CausalEdge64` has opened a zone the deletion does not need.

A hash aggregate past the K knee is a refusal path with its own name, not a faster `GroupSumI32`. A many-to-many the caller asked to see as rows is a terminal, capped, at the end, and printed as materialization.

## Done when

`cargo test -p lance-graph-quack --test duckdb_differential` still matches the oracle. `alloc_bytes_exec = 0` on fold plans. A fused plan that claims the skip does not compile past `Plan::check`. A million identical sums lower to one terminal. The count probe's three numbers stay on separate lines.
