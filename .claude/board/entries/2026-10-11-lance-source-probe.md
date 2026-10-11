# 2026-10-11 — Lance 13 source pruning before the fold (D-TELEPORT-2)

**Status:** MEASURED. Probe only, no production change.
`crates/lance-graph-benches/examples/lance_source_probe.rs`.

```text
CARGO_PROFILE_RELEASE_DEBUG=0 cargo run --release -p lance-graph-benches --example lance_source_probe
```

The question: can Lance 13's existing scanner and index paths prune source
reads before the Quack/mask-risc fold, without a row-sized intermediate
population, and with identical answers? Every arm is asserted against the
oracle COUNT and SUM. No new executor, opcode, IR or OGAR code.

Setup: 1,048,576 rows, fragments of 131,072, columns `x`, `xz`, `xb`
(identical i32 values) and `v`; zone map on `xz`, B-tree on `xb`. Cold =
fresh `Dataset::open` after `drop_caches` (works in this container); warm =
best of 3. Bytes and iops come from Lance's `ExecutionSummaryCounts`.
Host: 4 shared vCPUs; only order-of-magnitude differences are read.

## Arms

- **A** full scan of `x`, `v`; mask-risc `x < t` under the validity plane, then COUNT and masked SUM over the borrowed Arrow buffers.
- **C** sorted, no NULLs only: `with_row_addr_prefilter` over the prefix `[0, k)`, then a Range-mask SUM. Uses the fixture's known order.
- **Dz** `filter("xz < t")`, project `v`; no recheck in mask-risc.
- **Db** `filter("xb < t")`, project `v`.
- **N** `filter("xb < t")`, empty projection, COUNT only.

## Results (bytes read / warm ms)

| layout | sel | A | C | Dz | Db | N |
|---|---|---|---|---|---|---|
| sorted | 0.5 | 3.87 MB / 7.5 | 664 KB / 2.1 | 1.85 MB / 5.5 | 664 KB / 8.6 | 0 / 5.7 |
| sorted | 0.01 | 3.83 MB / 5.9 | 14 KB / 0.8 | 48 KB / 2.1 | 14 KB / 1.1 | 0 / 0.7 |
| sorted | 0.0001 | 3.83 MB / 6.5 | 1.3 KB / 0.7 | 23 KB / 1.7 | 1.3 KB / 1.1 | 0 / 0.6 |
| partial (shuffled within 64 Ki) | 0.5 | 3.84 MB / 8.1 | — | 1.86 MB / 4.1 | 664 KB / 22.6 | 0 / 14.2 |
| periodic (16 ramps) | 0.01 | 3.83 MB / 5.6 | — | 432 KB / 3.4 | 21 KB / 3.0 | 0 / 1.8 |
| shuffled | 0.5 | 3.96 MB / 9.1 | — | 3.96 MB / 9.8 | 1.33 MB / 80.4 | 0 / 35.4 |
| shuffled | 0.0001 | 3.96 MB / 5.3 | — | 2.17 MB / 5.9 | 214 KB / 2.6 | 0 / 0.9 |

NULLs (every 17th row) change none of the conclusions. After deleting
`x % 10 = 0`, N's COUNT stays exact (471,856 / 9,439 / 93) with 0 data bytes.

## Readings

1. **Pruning happens before the fold, and it is selectivity-proportional on
   ordered data.** Sorted: Db and C read the same bytes, 664 KB at 50% down
   to 1.3 KB at 0.01%, against a constant ~3.8 MB full scan. Nothing
   row-sized is materialised on the Java/Quack side: the scanner's survivors
   ARE the fold input.
2. **The zone map prunes only what clustering lets it.** Sorted it reads
   ~2–20× the B-tree's bytes (zone granularity 8,192 rows); fully shuffled
   it reads the whole column at every selectivity above 0.01%. Shuffled
   layouts are the control for lost pruning, not a timing target.
3. **Bytes and latency diverge.** On shuffled data at 50%, Db reads a third
   of A's bytes and is ~9× slower warm (80 ms vs 9 ms): a scattered
   row-address take is worse than a sequential scan. The B-tree pays off
   only at low selectivity or on ordered data.
4. **COUNT needs no data column.** N reads 0 data bytes and 0 data iops,
   stays exact after deletions, and is the fastest arm below ~1% on every
   layout. Caveat: `ExecutionSummaryCounts.bytes_read` is recorded by
   `FilteredReadExec`, `TakeExec` and KNN; no scalar-index exec records it,
   so B-tree page I/O is NOT in the 0. Cold N at 50% sorted (21.8 ms vs
   5.7 warm) shows that index I/O exists. "0 bytes" means "no data
   columns", never "no I/O".
5. **Index build cost:** zone map 22–117 ms, B-tree 136–229 ms per million
   rows here; amortised over queries, not per query.

## Smallest remaining seam

None is required for this to work: Lance's own `filter()` + index paths do
the pruning, and the fold consumes the scanner's output directly. The one
missing piece is a **cost choice between full scan and index take**, driven
by selectivity and clustering (reading 3 above). That is a planner
decision over existing scanner calls, not a new primitive; it is OPEN until
a real workload measures where the crossover sits. `with_row_addr_prefilter`
(arm C) is only correct when the order is witnessed; nothing in lance-graph
carries that witness yet (`ISS-WITNESSED-RANGE-DOES-NOT-ATTEST-PLANE-ORDER`),
so C stays a probe arm.
